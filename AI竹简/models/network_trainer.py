import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.twin_network import TwinNetwork, DistanceLoss
from preprocess.pair_generator import build_positive_pairs, build_negative_pairs
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import numpy as np
import pickle  # 添加pickle模块用于保存分割数据
import re  # 添加正则表达式模块用于解析竹简编号

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 早停机制类
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_loss = float('inf')
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False


# 图像对数据集
class ImagePairDataset(Dataset):
    def __init__(self, image_pairs, transform_fn):
        self.image_pairs = image_pairs
        self.transform_fn = transform_fn

    def __getitem__(self, index):
        img1_path, img2_path, label = self.image_pairs[index]
        
        # 读取图像
        image1 = cv2.imread(img1_path, 0)
        image2 = cv2.imread(img2_path, 0)
        
        # 应用变换
        image1 = self.transform_fn(image1)
        image2 = self.transform_fn(image2)
        
        return image1, image2, torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return len(self.image_pairs)


# 创建数据变换（训练时使用数据增强，测试时不使用）
def create_data_transform(is_training=True):
    if is_training:
        # 训练时使用数据增强
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((240, 240)),  # 稍大一些再裁剪
            transforms.RandomRotation(degrees=3),  # 轻微旋转
            transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),  # 随机裁剪
            transforms.ColorJitter(brightness=0.1, contrast=0.1),  # 亮度和对比度调整
            transforms.RandomHorizontalFlip(p=0.2),  # 低概率水平翻转
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化到[-1,1]
        ])
    else:
        # 测试时不使用数据增强
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  # 直接调整到目标尺寸
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])


# 从文件路径中提取缀合组（文件夹名）
def extract_group_from_path(file_path):
    # 获取路径中倒数第二级目录名（即缀合组文件夹名）
    path_parts = file_path.replace('\\', '/').split('/')
    
    # 寻找yizhuihe或weizhuihe之后的文件夹名
    for i, part in enumerate(path_parts):
        if part in ['yizhuihe', 'weizhuihe'] and i + 1 < len(path_parts):
            return path_parts[i + 1]
    
    # 如果没有找到，返回文件所在的直接父目录名
    return os.path.basename(os.path.dirname(file_path))


# 根据缀合组过滤样本对
def filter_pairs_by_groups(pairs, group_set):
    filtered_pairs = []
    
    for img1_path, img2_path, label in pairs:
        group1 = extract_group_from_path(img1_path)
        group2 = extract_group_from_path(img2_path)
        
        # 只保留两张图片都属于指定缀合组集合的样本对
        if group1 in group_set and group2 in group_set:
            filtered_pairs.append((img1_path, img2_path, label))
    
    return filtered_pairs


# 从文件路径中提取竹简编号
def extract_bamboo_id(file_path):
    # 提取文件名部分
    filename = os.path.basename(file_path)
    
    # 尝试从文件名中提取编号，例如 0001_a.jpeg 或 0001_1.jpeg 中提取 0001
    match = re.search(r'(\d+)[_-]', filename)
    if match:
        return match.group(1)
    
    # 如果上面的模式没有匹配到，尝试从父目录名提取编号
    parent_dir = os.path.basename(os.path.dirname(file_path))
    match = re.search(r'(\d+)', parent_dir)
    if match:
        return match.group(1)
    
    # 如果都没有匹配到，返回文件名作为标识符
    return filename




# 按缀合组分割数据集策略V3 - 保持相邻关系完整性
def smart_split_dataset_v3_group_based(data_root, test_ratio=0.2):
    
    # 获取所有缀合组（文件夹）
    group_folders = []
    group_images = {}
    
    for group_dir in os.listdir(data_root):
        group_path = os.path.join(data_root, group_dir)
        if os.path.isdir(group_path):
            # 收集该组的所有图片
            group_imgs = []
            for img_file in os.listdir(group_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    group_imgs.append(os.path.join(group_path, img_file))
            
            if group_imgs:  # 只保留有图片的组
                group_folders.append(group_dir)
                group_images[group_dir] = group_imgs
    
    # 随机分割缀合组
    random.shuffle(group_folders)
    test_size = max(1, int(len(group_folders) * test_ratio))
    
    test_groups = set(group_folders[:test_size])
    train_groups = set(group_folders[test_size:])
    
    # 收集训练和测试图片
    train_images = []
    test_images = []
    
    for group in train_groups:
        train_images.extend(group_images[group])
    
    for group in test_groups:
        test_images.extend(group_images[group])
    
    print(f"按缀合组分割完成:")
    print(f"  训练集: {len(train_groups)} 个缀合组, {len(train_images)} 张图片")
    print(f"  测试集: {len(test_groups)} 个缀合组, {len(test_images)} 张图片")
    print(f"  训练集缀合组示例: {sorted(list(train_groups))[:5]}...")
    print(f"  测试集缀合组示例: {sorted(list(test_groups))[:5]}...")
    
    return train_groups, test_groups, train_images, test_images

# 训练孪生网络主函数
def train_twin_network():
    print("开始训练孪生网络...")
    
    # 数据路径配置 - 只使用已缀合数据进行训练
    data_root = "yizhuihe/"
    weizhuihe_dir = "weizhuihe/"  # 保留用于后续推理，但不参与训练
    
    # 只扫描已缀合数据目录用于训练
    yizhuihe_image_paths = []
    print(f"正在扫描已缀合数据目录: {data_root}")
    
    for group_dir in os.listdir(data_root):
        group_path = os.path.join(data_root, group_dir)
        if os.path.isdir(group_path):
            for img_file in os.listdir(group_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    yizhuihe_image_paths.append(os.path.join(group_path, img_file))
    
    print(f"找到已缀合图片总数: {len(yizhuihe_image_paths)}")
    
    # 使用新的按缀合组分割策略
    train_groups, test_groups, train_images, test_images = smart_split_dataset_v3_group_based(data_root, test_ratio=0.2)
    
    print(f"数据集分割完成: 训练集包含 {len(train_groups)} 个缀合组, 测试集包含 {len(test_groups)} 个缀合组")
    print(f"训练集图片总数: {len(train_images)}, 测试集图片总数: {len(test_images)}")
    print(f"注意: 未缀合数据(weizhuihe)将保留用于最终推理评估，不参与训练过程")
    
    # 保存分割信息
    dataset_split = {
        'train_groups': list(train_groups),
        'test_groups': list(test_groups),
        'data_source': 'yizhuihe_group_based'  # 标记数据来源
    }
    
    with open('dataset_split.pkl', 'wb') as f:
        pickle.dump(dataset_split, f)
    
    print(f"数据集分割信息已保存至 dataset_split.pkl")

    # 【恢复原逻辑】先全体生成样本对，再过滤分配到训练集和测试集
    print("正在生成所有样本对...")
    
    # 1. 先生成所有正样本对（基于Excel中的相邻关系）
    all_positive_pairs = build_positive_pairs(data_root)
    print(f"生成的全部正样本对数量: {len(all_positive_pairs)}")
    
    # 2. 创建Excel读取器用于负样本生成时排除实际可拼接的对
    from preprocess.zhuihe_order_reader import ZhuiheOrderReader
    excel_reader = ZhuiheOrderReader()
    
    # 3. 生成更多负样本对（负样本数量为正样本的5倍，排除实际可拼接的对）
    negative_sample_multiplier = 5  # 负样本倍数
    all_negative_pairs = build_negative_pairs(yizhuihe_image_paths, len(all_positive_pairs) * negative_sample_multiplier, excel_reader)
    print(f"生成的全部负样本对数量: {len(all_negative_pairs)} (正样本的{negative_sample_multiplier}倍)")
    
    # 4. 过滤分配到训练集和测试集（基于缀合组）
    train_positive_pairs = filter_pairs_by_groups(all_positive_pairs, train_groups)
    train_negative_pairs = filter_pairs_by_groups(all_negative_pairs, train_groups)
    
    test_positive_pairs = filter_pairs_by_groups(all_positive_pairs, test_groups)
    test_negative_pairs = filter_pairs_by_groups(all_negative_pairs, test_groups)
    
    # 合并并打乱训练集和测试集
    all_training_pairs = train_positive_pairs + train_negative_pairs
    all_test_pairs = test_positive_pairs + test_negative_pairs
    random.shuffle(all_training_pairs)
    random.shuffle(all_test_pairs)
    
    print(f"训练集: 正样本对 {len(train_positive_pairs)}, 负样本对 {len(train_negative_pairs)}, 总计 {len(all_training_pairs)}")
    print(f"测试集: 正样本对 {len(test_positive_pairs)}, 负样本对 {len(test_negative_pairs)}, 总计 {len(all_test_pairs)}")
    print(f"过滤统计: 从全部 {len(all_positive_pairs)} 正样本对和 {len(all_negative_pairs)} 负样本对中分配")
    过滤掉的正样本对 = len(all_positive_pairs) - len(train_positive_pairs) - len(test_positive_pairs)
    过滤掉的负样本对 = len(all_negative_pairs) - len(train_negative_pairs) - len(test_negative_pairs)
    print(f"过滤掉的样本对: 正样本对 {过滤掉的正样本对}, 负样本对 {过滤掉的负样本对}")

    # 创建数据集和加载器（使用不同的变换）
    train_transform = create_data_transform(is_training=True)
    test_transform = create_data_transform(is_training=False)
    
    training_dataset = ImagePairDataset(all_training_pairs, train_transform)
    test_dataset = ImagePairDataset(all_test_pairs, test_transform)
    
    # 设置合适的批次大小，确保BatchNorm正常工作
    batch_size = 16  # 降低批次大小以减少过拟合风险
    # 使用drop_last=True防止最后一个batch只有1个样本导致BatchNorm错误
    train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=4, pin_memory=True, drop_last=False)

    # 初始化模型和训练组件
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    model = TwinNetwork().to(device)
    loss_function = DistanceLoss()
    # 添加L2正则化（权重衰减）
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # 优化学习率调度器 - 组合使用多种调度策略
    # 1. 基于验证损失的自动调整
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7
    )
    
    # 2. 余弦退火调度器（可选，用于更平滑的学习率变化）
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=10, eta_min=1e-6
    )
    
    # 初始化早停机制
    early_stopping = EarlyStopping(patience=6, min_delta=0.001, restore_best_weights=True)

    # 训练参数（增加训练轮数）
    num_epochs = 20  # 增加到20轮，配合早停机制
    train_losses_history = []
    train_accuracies_history = []
    test_losses_history = []
    test_accuracies_history = []
    learning_rates_history = []
    
    print("开始训练...")
    
    for epoch_idx in range(num_epochs):
        # ===== 训练阶段 =====
        model.train()
        epoch_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc=f"训练轮次 {epoch_idx+1}/{num_epochs}")
        
        for batch_img1, batch_img2, batch_labels in progress_bar:
            batch_img1 = batch_img1.to(device)
            batch_img2 = batch_img2.to(device)
            batch_labels = batch_labels.to(device)
            
            # 前向传播
            features1, features2 = model(batch_img1, batch_img2)
            loss = loss_function(features1, features2, batch_labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # 计算训练准确率
            with torch.no_grad():
                distances = torch.pairwise_distance(features1, features2)
                predictions = (distances < 1.0).float()
                correct_predictions += (predictions == batch_labels).sum().item()
                total_samples += batch_labels.size(0)
            
            # 更新进度条显示
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{correct_predictions/total_samples:.4f}'
            })
        
        # 计算训练指标
        avg_train_loss = epoch_loss / len(train_loader)
        train_accuracy = correct_predictions / total_samples
        
        # ===== 测试集验证阶段 =====
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        print(f"  验证轮次 {epoch_idx+1}...")
        with torch.no_grad():
            for test_img1, test_img2, test_labels in tqdm(test_loader, desc="验证中", leave=False):
                test_img1 = test_img1.to(device)
                test_img2 = test_img2.to(device)
                test_labels = test_labels.to(device)
                
                # 前向传播
                test_features1, test_features2 = model(test_img1, test_img2)
                t_loss = loss_function(test_features1, test_features2, test_labels)
                
                test_loss += t_loss.item()
                
                # 计算测试准确率
                test_distances = torch.pairwise_distance(test_features1, test_features2)
                test_predictions = (test_distances < 1.0).float()
                test_correct += (test_predictions == test_labels).sum().item()
                test_total += test_labels.size(0)
        
        # 计算测试指标
        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = test_correct / test_total
        
        # 记录指标
        train_losses_history.append(avg_train_loss)
        train_accuracies_history.append(train_accuracy)
        test_losses_history.append(avg_test_loss)
        test_accuracies_history.append(test_accuracy)
        learning_rates_history.append(optimizer.param_groups[0]['lr'])
        
        # 打印详细信息
        print(f"轮次 {epoch_idx+1}/{num_epochs}:")
        print(f"  训练 - 损失: {avg_train_loss:.4f}, 准确率: {train_accuracy:.4f} ({correct_predictions}/{total_samples})")
        print(f"  测试 - 损失: {avg_test_loss:.4f}, 准确率: {test_accuracy:.4f} ({test_correct}/{test_total})")
        print(f"  学习率: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 学习率调度器更新
        plateau_scheduler.step(avg_test_loss)
        
        # 早停检查
        if early_stopping(avg_test_loss, model):
            print(f"早停触发！在第 {epoch_idx+1} 轮停止训练")
            print(f"最佳验证损失: {early_stopping.best_loss:.4f}")
            break
            
        print("-" * 60)

    # 保存模型
    torch.save(model.state_dict(), "twin_model.pth")
    print("模型训练完成，已保存为 twin_model.pth")
    
    # 实际训练轮数（可能因早停而少于预设轮数）
    actual_epochs = len(train_losses_history)
    print(f"实际训练轮数: {actual_epochs}")

    # 绘制详细的训练曲线
    plot_detailed_training_curves(
        train_losses_history, train_accuracies_history,
        test_losses_history, test_accuracies_history,
        learning_rates_history, actual_epochs
    )
    print("详细训练曲线已保存为 training_metrics.png")
    
    return train_groups, test_groups


# 绘制详细的训练曲线
def plot_detailed_training_curves(train_losses, train_accuracies, test_losses, test_accuracies, learning_rates, epochs):
    plt.figure(figsize=(20, 12))
    
    # 设置整体标题
    plt.suptitle('竹简孪生网络训练详细监控 (增强版)', fontsize=16, fontweight='bold')
    
    # 1. 损失对比曲线
    plt.subplot(2, 3, 1)
    plt.plot(range(1, epochs + 1), train_losses, marker='o', color='blue', label='训练损失', linewidth=2)
    plt.plot(range(1, epochs + 1), test_losses, marker='s', color='red', label='测试损失', linewidth=2)
    plt.title("损失变化对比", fontsize=14, fontweight='bold')
    plt.xlabel("训练轮次")
    plt.ylabel("损失值")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 准确率对比曲线
    plt.subplot(2, 3, 2)
    plt.plot(range(1, epochs + 1), train_accuracies, marker='o', color='green', label='训练准确率', linewidth=2)
    plt.plot(range(1, epochs + 1), test_accuracies, marker='s', color='orange', label='测试准确率', linewidth=2)
    plt.title("准确率变化对比", fontsize=14, fontweight='bold')
    plt.xlabel("训练轮次")
    plt.ylabel("准确率")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. 学习率变化
    plt.subplot(2, 3, 3)
    plt.plot(range(1, epochs + 1), learning_rates, marker='d', color='purple', linewidth=2)
    plt.title("学习率变化", fontsize=14, fontweight='bold')
    plt.xlabel("训练轮次")
    plt.ylabel("学习率")
    plt.yscale('log')  # 使用对数坐标
    plt.grid(True, alpha=0.3)
    
    # 4. 训练vs测试损失散点图
    plt.subplot(2, 3, 4)
    plt.scatter(train_losses, test_losses, c=range(epochs), cmap='viridis', s=80, alpha=0.7)
    plt.colorbar(label='训练轮次')
    for i in range(epochs):
        plt.annotate(f'{i+1}', (train_losses[i], test_losses[i]), fontsize=9)
    plt.plot([min(train_losses), max(train_losses)], [min(train_losses), max(train_losses)], 'r--', alpha=0.5)
    plt.title("训练损失 vs 测试损失", fontsize=14, fontweight='bold')
    plt.xlabel("训练损失")
    plt.ylabel("测试损失")
    plt.grid(True, alpha=0.3)
    
    # 5. 过拟合检测图
    plt.subplot(2, 3, 5)
    # 计算损失差异
    loss_gap = [test_losses[i] - train_losses[i] for i in range(epochs)]
    acc_gap = [train_accuracies[i] - test_accuracies[i] for i in range(epochs)]
    
    plt.plot(range(1, epochs + 1), loss_gap, marker='o', color='red', label='测试损失 - 训练损失', linewidth=2)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.title("过拟合检测 (损失差异)", fontsize=14, fontweight='bold')
    plt.xlabel("训练轮次")
    plt.ylabel("损失差异")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. 性能总结表格
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    # 创建性能总结数据
    summary_data = [
        ['指标', '最佳值', '最终值', '轮次'],
        ['训练损失', f'{min(train_losses):.4f}', f'{train_losses[-1]:.4f}', f'{train_losses.index(min(train_losses))+1}'],
        ['测试损失', f'{min(test_losses):.4f}', f'{test_losses[-1]:.4f}', f'{test_losses.index(min(test_losses))+1}'],
        ['训练准确率', f'{max(train_accuracies):.4f}', f'{train_accuracies[-1]:.4f}', f'{train_accuracies.index(max(train_accuracies))+1}'],
        ['测试准确率', f'{max(test_accuracies):.4f}', f'{test_accuracies[-1]:.4f}', f'{test_accuracies.index(max(test_accuracies))+1}'],
        ['', '', '', ''],
        ['训练状态', '', '', ''],
        ['实际轮数', f'{epochs}', '', ''],
        ['数据增强', '已启用', '', ''],
        ['早停机制', '已启用', '', ''],
        ['收敛状态', '已收敛' if epochs > 5 and abs(train_losses[-1] - train_losses[-2]) < 0.01 else '训练中', '', '']
    ]
    
    # 创建表格
    table = plt.table(cellText=summary_data[1:], colLabels=summary_data[0], 
                     cellLoc='center', loc='center',
                     colWidths=[0.3, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    plt.title("训练总结", fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig("training_metrics.png", dpi=300, bbox_inches='tight')
    plt.show()


# 简单的训练曲线（保持向后兼容）
def plot_training_curves(losses, accuracies, epochs):
    plt.figure(figsize=(12, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), losses, marker='o', color='blue')
    plt.title("训练损失变化")
    plt.xlabel("训练轮次")
    plt.ylabel("损失值")
    plt.grid(True)
    
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), accuracies, marker='s', color='red')
    plt.title("训练准确率变化")
    plt.xlabel("训练轮次")
    plt.ylabel("准确率")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=300, bbox_inches='tight')
    plt.show()


# 从文件路径中提取缀合组（文件夹名）
def extract_group_from_path(file_path):
    # 获取路径中倒数第二级目录名（即缀合组文件夹名）
    path_parts = file_path.replace('\\', '/').split('/')
    
    # 寻找yizhuihe或weizhuihe之后的文件夹名
    for i, part in enumerate(path_parts):
        if part in ['yizhuihe', 'weizhuihe'] and i + 1 < len(path_parts):
            return path_parts[i + 1]
    
    # 如果没有找到，返回文件所在的直接父目录名
    return os.path.basename(os.path.dirname(file_path))


# 根据缀合组过滤样本对
def filter_pairs_by_groups(pairs, group_set):
    filtered_pairs = []
    
    for img1_path, img2_path, label in pairs:
        group1 = extract_group_from_path(img1_path)
        group2 = extract_group_from_path(img2_path)
        
        # 只保留两张图片都属于指定缀合组集合的样本对
        if group1 in group_set and group2 in group_set:
            filtered_pairs.append((img1_path, img2_path, label))
    
    return filtered_pairs


# 根据竹简编号过滤样本对
def filter_pairs_by_bamboo_ids(pairs, bamboo_ids_set):
    filtered_pairs = []
    
    for img1_path, img2_path, label in pairs:
        bamboo_id1 = extract_bamboo_id(img1_path)
        bamboo_id2 = extract_bamboo_id(img2_path)
        
        # 只保留两张图片都属于指定竹简编号集合的样本对
        if bamboo_id1 in bamboo_ids_set and bamboo_id2 in bamboo_ids_set:
            filtered_pairs.append((img1_path, img2_path, label))
    
    return filtered_pairs


if __name__ == "__main__":
    train_twin_network()
