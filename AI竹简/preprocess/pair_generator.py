import os
import random
from pathlib import Path
from .zhuihe_order_reader import ZhuiheOrderReader


# 构建正样本对（基于Excel文件中的相邻片段）
def build_positive_pairs(data_dir):
    positive_pairs = []
    
    # 创建顺序读取器
    reader = ZhuiheOrderReader()
    
    if len(reader.sheets_data) == 0:
        print("警告：无法读取Excel文件，回退到旧的配对方式")
        return build_positive_pairs_fallback(data_dir)
    
    # 获取Excel中定义的相邻图片文件对
    training_pairs = reader.generate_adjacent_pairs_for_training()
    
    print(f"从Excel中解析出 {len(training_pairs)} 个相邻图片对")
    
    valid_pairs = 0
    for i, (img_path1, img_path2) in enumerate(training_pairs):
        # 检查文件是否存在
        if os.path.exists(img_path1) and os.path.exists(img_path2):
            positive_pairs.append((img_path1, img_path2, 1))
            valid_pairs += 1
            
            # 打印前5个样例
            if valid_pairs <= 5:
                print(f"  正样本对 {valid_pairs}: {os.path.basename(img_path1)} <-> {os.path.basename(img_path2)}")
        else:
            if not os.path.exists(img_path1):
                print(f"警告：图片文件不存在: {img_path1}")
            if not os.path.exists(img_path2):
                print(f"警告：图片文件不存在: {img_path2}")
    
    print(f"成功生成 {valid_pairs} 个有效的正样本对")
    
    if valid_pairs == 0:
        print("警告：没有找到有效的相邻片段对，回退到旧的配对方式")
        return build_positive_pairs_fallback(data_dir)
    
    return positive_pairs


# 在实际文件中找到匹配的文件路径
def find_matching_file_path(filename, actual_files_dict):
    # 在实际文件中找到匹配的文件路径
    # 直接匹配
    if filename in actual_files_dict:
        return actual_files_dict[filename]
    
    # 不区分大小写匹配
    filename_lower = filename.lower()
    for key, path in actual_files_dict.items():
        if key.lower() == filename_lower:
            return path
    
    # 去掉扩展名匹配
    filename_base = os.path.splitext(filename)[0]
    if filename_base in actual_files_dict:
        return actual_files_dict[filename_base]
    
    # 模糊匹配
    for key, path in actual_files_dict.items():
        key_base = os.path.splitext(key)[0]
        if key_base.lower() == filename_base.lower():
            return path
    
    return None


# 回退方案：使用旧的配对方式（同目录内任意配对）
def build_positive_pairs_fallback(data_dir):
    # 回退方案：使用旧的配对方式（同目录内任意配对）
    print("使用回退方案：同目录内任意配对")
    positive_pairs = []
    
    for group_name in os.listdir(data_dir):
        group_path = os.path.join(data_dir, group_name)
        if not os.path.isdir(group_path):
            continue
            
        # 收集组内所有图片
        group_images = []
        for filename in os.listdir(group_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                group_images.append(os.path.join(group_path, filename))
        
        # 生成组内所有图片对
        for i in range(len(group_images)):
            for j in range(i + 1, len(group_images)):
                positive_pairs.append((group_images[i], group_images[j], 1))
    
    return positive_pairs


# 构建负样本对（不同组的图片对，排除实际可拼接的对）
def build_negative_pairs(all_images, num_pairs, excel_reader=None):
    # 构建负样本对（不同组的图片对，排除实际可拼接的对）
    negative_pairs = []
    
    # 按目录分组
    groups = {}
    for img_path in all_images:
        group_key = Path(img_path).parent.name
        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(img_path)
    
    group_list = list(groups.keys())
    
    # 获取所有实际可拼接的图片对（如果提供了Excel读取器）
    excluded_pairs = set()
    if excel_reader:
        print("获取Excel中定义的可拼接对，避免生成错误的负样本...")
        training_pairs = excel_reader.generate_adjacent_pairs_for_training()
        for img1, img2 in training_pairs:
            # 标准化路径用于比较
            normalized_pair1 = (os.path.normpath(img1), os.path.normpath(img2))
            normalized_pair2 = (os.path.normpath(img2), os.path.normpath(img1))
            excluded_pairs.add(normalized_pair1)
            excluded_pairs.add(normalized_pair2)
        print(f"排除 {len(excluded_pairs)} 个实际可拼接的图片对")
    
    attempts = 0
    max_attempts = num_pairs * 10  # 防止无限循环
    
    while len(negative_pairs) < num_pairs and attempts < max_attempts:
        attempts += 1
        
        # 随机选择两个不同的组
        group1, group2 = random.sample(group_list, 2)
        
        # 从两个组中各选一张图片
        img1 = random.choice(groups[group1])
        img2 = random.choice(groups[group2])
        
        # 标准化路径
        norm_img1 = os.path.normpath(img1)
        norm_img2 = os.path.normpath(img2)
        
        # 检查是否为实际可拼接的对
        if excel_reader and ((norm_img1, norm_img2) in excluded_pairs or (norm_img2, norm_img1) in excluded_pairs):
            continue  # 跳过实际可拼接的对
        
        # 避免重复
        pair_tuple = (img1, img2, 0)
        reverse_pair_tuple = (img2, img1, 0)
        
        if pair_tuple not in negative_pairs and reverse_pair_tuple not in negative_pairs:
            negative_pairs.append(pair_tuple)
    
    if attempts >= max_attempts:
        print(f"警告：在 {max_attempts} 次尝试后只生成了 {len(negative_pairs)} 个负样本对")
    
    return negative_pairs
