#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 竹简缀合推理模块 - 全新设计
# 基于yizhuihe测试集内部进行竹简缀合推理

import os
import pickle
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import cv2
import re
from tqdm import tqdm
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from models.twin_network import TwinNetwork
from preprocess.zhuihe_order_reader import ZhuiheOrderReader


class BambooSlipInferenceEngine:
    # 竹简缀合推理引擎 - 新版本
    def __init__(self, model_path, device='cpu'):
        # 初始化推理引擎
        self.device = device
        self.model_path = model_path
        self.model = None
        self.excel_reader = ZhuiheOrderReader()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])
        self.test_images = []
        self.test_ids = set()
        self.processed_images = set()
        self._load_model()
        self._load_test_data()
        print(f"推理引擎初始化完成")
        print(f"   模型: {model_path}")
        print(f"   设备: {device}")
        print(f"   测试图片: {len(self.test_images)} 张")

    def _load_model(self):
        # 加载孪生网络模型
        try:
            self.model = TwinNetwork()
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print(f"模型加载成功: {self.model_path}")
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise

    def _load_test_data(self):
        # 加载测试集数据
        try:
            if os.path.exists('dataset_split.pkl'):
                with open('dataset_split.pkl', 'rb') as f:
                    dataset_split = pickle.load(f)
                    if 'test_groups' in dataset_split:
                        self.test_ids = set(dataset_split['test_groups'])
                    elif 'test_ids' in dataset_split:
                        self.test_ids = set(dataset_split['test_ids'])
                    else:
                        print("数据集分割文件格式错误")
                        return
                print(f"测试集包含 {len(self.test_ids)} 个缀合组")
            else:
                print("未找到dataset_split.pkl文件")
                return
            yizhuihe_path = "yizhuihe"
            if not os.path.exists(yizhuihe_path):
                print(f"目录不存在: {yizhuihe_path}")
                return
            for group_folder in os.listdir(yizhuihe_path):
                if group_folder in self.test_ids:
                    group_path = os.path.join(yizhuihe_path, group_folder)
                    if os.path.isdir(group_path):
                        for filename in os.listdir(group_path):
                            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                                full_path = f"yizhuihe/{group_folder}/{filename}"
                                self.test_images.append(full_path)
            print(f"收集到 {len(self.test_images)} 张测试图片")
        except Exception as e:
            print(f"测试数据加载失败: {e}")

    def _is_complete_slip(self, image_path):
        # 判断是否为完整简
        try:
            image = cv2.imread(image_path)
            if image is None:
                return False
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return False
            main_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(main_contour)
            if area < 100:
                return False
            x, y, w, h = cv2.boundingRect(main_contour)
            rect_area = w * h
            extent = float(area) / rect_area
            hull = cv2.convexHull(main_contour)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                return False
            solidity = float(area) / hull_area
            if extent > 0.9 and solidity > 0.95:
                return True
            else:
                return False
        except Exception as e:
            print(f"完整性检查失败 {os.path.basename(image_path)}: {e}")
            return False

    def _extract_slip_features(self, image_path):
        # 提取竹简的详细特征（简化版）
        features = {
            'width': 100,
            'break_position': 'unknown',
            'break_shape': [1.0],
            'texture': [0.5],
            'ink_traces': [0.3]
        }
        try:
            image = cv2.imread(image_path)
            if image is not None:
                height, width, _ = image.shape
                features['width'] = width
                features['break_position'] = self._detect_break_position(image_path)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / edges.size
                features['break_shape'] = [float(edge_density)]
                texture_variance = np.var(gray)
                features['texture'] = [float(texture_variance) / 10000.0]
                dark_pixel_ratio = np.sum(gray < 100) / gray.size
                features['ink_traces'] = [float(dark_pixel_ratio)]
        except Exception as e:
            print(f"特征提取失败 {os.path.basename(image_path)}: {e}")
        return features

    def _detect_break_position(self, image_path):
        # 检测断口位置
        try:
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            height, width = gray.shape
            top_edge = edges[0:height//10, :]
            bottom_edge = edges[height*9//10:height, :]
            left_edge = edges[:, 0:width//10]
            right_edge = edges[:, width*9//10:width]
            top_complexity = np.sum(top_edge) / top_edge.size
            bottom_complexity = np.sum(bottom_edge) / bottom_edge.size
            left_complexity = np.sum(left_edge) / left_edge.size
            right_complexity = np.sum(right_edge) / right_edge.size
            complexities = {
                'top': top_complexity,
                'bottom': bottom_complexity,
                'left': left_complexity,
                'right': right_complexity
            }
            break_position = max(complexities, key=complexities.get)
            return break_position
        except Exception as e:
            print(f"断口位置检测失败: {e}")
            return 'unknown'

    def _extract_break_shape(self, image_path):
        # 提取断口形状特征
        try:
            if not os.path.exists(image_path):
                print(f"断口形状提取失败: 图片不存在 {image_path}")
                return []
            image = cv2.imread(image_path)
            if image is None:
                print(f"断口形状提取失败: 无法读取图片 {image_path}")
                return []
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                print(f"断口形状提取: 未找到轮廓 {os.path.basename(image_path)}")
                return []
            max_contour = max(contours, key=cv2.contourArea)
            if len(max_contour) < 4:
                print(f"断口形状提取: 轮廓点太少 {os.path.basename(image_path)}")
                return []
            epsilon = 0.02 * cv2.arcLength(max_contour, True)
            simplified_contour = cv2.approxPolyDP(max_contour, epsilon, True)
            try:
                if simplified_contour.ndim != 3 or simplified_contour.shape[2] != 2:
                    print(f"断口形状提取: 轮廓格式错误 {os.path.basename(image_path)}")
                    return []
                if len(simplified_contour) < 4:
                    print(f"断口形状提取: 简化轮廓点太少 {os.path.basename(image_path)}")
                    return []
                hull = cv2.convexHull(simplified_contour, returnPoints=False)
                if hull is None or len(hull) < 3:
                    print(f"断口形状提取: 凸包计算失败 {os.path.basename(image_path)}")
                    perimeter = cv2.arcLength(simplified_contour, True)
                    area = cv2.contourArea(simplified_contour)
                    if area > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        return [float(circularity)]
                    return [0.0]
                defects = cv2.convexityDefects(simplified_contour, hull)
                if defects is not None and len(defects) > 0:
                    depths = []
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        if d > 0:
                            depths.append(float(d))
                    if depths:
                        depths = np.array(depths, dtype=np.float32)
                        features = [
                            float(np.mean(depths)),
                            float(np.std(depths)),
                            float(np.max(depths)),
                            float(len(depths))
                        ]
                        return features
                perimeter = cv2.arcLength(simplified_contour, True)
                area = cv2.contourArea(simplified_contour)
                if area > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    return [float(circularity)]
                return [0.0]
            except cv2.error as e:
                perimeter = cv2.arcLength(max_contour, True)
                area = cv2.contourArea(max_contour)
                if area > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    return [float(circularity)]
                else:
                    return [0.0]
        except Exception as e:
            print(f"断口形状提取失败 {os.path.basename(image_path)}: {str(e)}")
            return []

    def _compute_siamese_similarity(self, img1_path, img2_path):
        # 使用孪生网络计算相似度
        try:
            if not os.path.exists(img1_path):
                print(f"孪生网络计算失败: 图片1不存在 {img1_path}")
                return 0.0
            if not os.path.exists(img2_path):
                print(f"孪生网络计算失败: 图片2不存在 {img2_path}")
                return 0.0
            try:
                img1 = Image.open(img1_path).convert('L')
                img2 = Image.open(img2_path).convert('L')
            except Exception as e:
                print(f"孪生网络计算失败: 图像加载错误 {str(e)}")
                return 0.0
            try:
                img1_tensor = self.transform(img1).unsqueeze(0).to(self.device)
                img2_tensor = self.transform(img2).unsqueeze(0).to(self.device)
            except Exception as e:
                print(f"孪生网络计算失败: 图像预处理错误 {str(e)}")
                return 0.0
            try:
                with torch.no_grad():
                    features1, features2 = self.model(img1_tensor, img2_tensor)
                    distance = torch.pairwise_distance(features1, features2).item()
                    if distance <= 1.0:
                        similarity = 1.0 - distance * 0.5
                    else:
                        similarity = 0.5 / (1.0 + (distance - 1.0))
                return similarity
            except Exception as e:
                print(f"孪生网络计算失败: 模型推理错误 {str(e)}")
                return 0.0
        except Exception as e:
            print(f"孪生网络相似度计算失败: {str(e)}")
            import traceback
            print(f"详细错误: {traceback.format_exc()}")
            return 0.0

    def _compute_traditional_similarity(self, features1, features2):
        # 基于传统特征计算相似度（简化版）
        try:
            similarity_scores = []
            width1 = features1.get('width', 100)
            width2 = features2.get('width', 100)
            try:
                width1 = float(width1) if width1 else 100.0
                width2 = float(width2) if width2 else 100.0
                if width1 > 0 and width2 > 0:
                    width_diff = abs(width1 - width2)
                    width_sim = max(0, 1 - width_diff / max(width1, width2))
                    similarity_scores.append(width_sim)
            except (TypeError, ValueError):
                similarity_scores.append(0.5)
            break_compatibility = self._check_break_compatibility(
                features1.get('break_position', 'unknown'),
                features2.get('break_position', 'unknown')
            )
            similarity_scores.append(break_compatibility)
            texture1 = features1.get('texture', [0.5])
            texture2 = features2.get('texture', [0.5])
            try:
                if texture1 and texture2:
                    t1_val = float(texture1[0]) if texture1 else 0.5
                    t2_val = float(texture2[0]) if texture2 else 0.5
                    texture_sim = 1.0 - abs(t1_val - t2_val)
                    similarity_scores.append(max(0, texture_sim))
            except (TypeError, ValueError, IndexError):
                similarity_scores.append(0.5)
            ink1 = features1.get('ink_traces', [0.3])
            ink2 = features2.get('ink_traces', [0.3])
            try:
                if ink1 and ink2:
                    i1_val = float(ink1[0]) if ink1 else 0.3
                    i2_val = float(ink2[0]) if ink2 else 0.3
                    ink_sim = 1.0 - abs(i1_val - i2_val)
                    similarity_scores.append(max(0, ink_sim))
            except (TypeError, ValueError, IndexError):
                similarity_scores.append(0.5)
            if similarity_scores:
                return float(np.mean(similarity_scores))
            else:
                return 0.0
        except Exception as e:
            print(f"传统特征相似度计算失败: {e}")
            return 0.0

    def _check_break_compatibility(self, pos1, pos2):
        # 检查断口位置兼容性
        compatible_pairs = {
            ('top', 'bottom'), ('bottom', 'top'),
            ('left', 'right'), ('right', 'left')
        }
        if (pos1, pos2) in compatible_pairs:
            return 1.0
        elif pos1 == pos2:
            return 0.5
        else:
            return 0.0

    def _compute_texture_similarity(self, texture1, texture2):
        # 计算纹理相似性
        try:
            if not texture1 or not texture2:
                return 0.0
            t1 = np.array(texture1)
            t2 = np.array(texture2)
            if len(t1) != len(t2):
                min_len = min(len(t1), len(t2))
                t1 = t1[:min_len]
                t2 = t2[:min_len]
            dot_product = np.dot(t1, t2)
            norm1 = np.linalg.norm(t1)
            norm2 = np.linalg.norm(t2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            similarity = dot_product / (norm1 * norm2)
            return max(0, similarity)
        except Exception as e:
            return 0.0

    def _compute_ink_compatibility(self, ink1, ink2):
        # 计算墨迹兼容性
        try:
            if not ink1 or not ink2:
                return 0.5
            i1 = np.array(ink1) if isinstance(ink1, list) else np.array([ink1])
            i2 = np.array(ink2) if isinstance(ink2, list) else np.array([ink2])
            if len(i1) != len(i2):
                min_len = min(len(i1), len(i2))
                i1 = i1[:min_len]
                i2 = i2[:min_len]
            diff = np.abs(i1 - i2)
            max_val = max(np.max(i1), np.max(i2))
            if max_val == 0:
                return 1.0
            similarity = 1 - np.mean(diff) / max_val
            return max(0, similarity)
        except Exception as e:
            return 0.5

    def _get_bamboo_id_from_path(self, image_path):
        # 从图片路径提取竹简编号
        try:
            filename = os.path.basename(image_path)
            numbers = re.findall(r'\d+', filename)
            if numbers:
                return str(int(numbers[0]))
            return ""
        except:
            return ""

    def _check_ground_truth(self, img1_path, img2_path):
        # 检查两张图片是否应该相邻（根据Excel数据）
        try:
            id1 = self._get_bamboo_id_from_path(img1_path)
            id2 = self._get_bamboo_id_from_path(img2_path)
            if not id1 or not id2:
                return False
            possible_ids1 = [f"9-{id1}", f"8-{id1}", id1]
            possible_ids2 = [f"9-{id2}", f"8-{id2}", id2]
            adjacent_pairs = self.excel_reader.get_adjacent_pairs()
            for pid1 in possible_ids1:
                for pid2 in possible_ids2:
                    if (pid1, pid2) in adjacent_pairs or (pid2, pid1) in adjacent_pairs:
                        return True
            return False
        except Exception as e:
            print(f"真值检查失败: {e}")
            return False

    def predict_matches(self, query_image, top_k=5):
        # 为指定图片预测最匹配的候选
        print(f"开始推理: {os.path.basename(query_image)}")
        is_complete = self._is_complete_slip(query_image)
        if is_complete:
            print(f"   跳过完整简: {os.path.basename(query_image)}")
            return []
        print(f"   提取特征...")
        query_features = self._extract_slip_features(query_image)
        candidates = []
        candidate_images = [img for img in self.test_images if img != query_image]
        print(f"   评估 {len(candidate_images)} 个候选...")
        for candidate_img in tqdm(candidate_images, desc="匹配计算"):
            candidate_features = self._extract_slip_features(candidate_img)
            siamese_score = self._compute_siamese_similarity(query_image, candidate_img)
            traditional_score = self._compute_traditional_similarity(query_features, candidate_features)
            final_score = 0.7 * siamese_score + 0.3 * traditional_score
            is_ground_truth = self._check_ground_truth(query_image, candidate_img)
            candidates.append({
                'image_path': candidate_img,
                'siamese_score': siamese_score,
                'traditional_score': traditional_score,
                'final_score': final_score,
                'is_ground_truth': is_ground_truth,
                'candidate_features': candidate_features
            })
        candidates.sort(key=lambda x: x['final_score'], reverse=True)
        first_correct_index = None
        for i, candidate in enumerate(candidates):
            if candidate['is_ground_truth']:
                first_correct_index = i
                break
        if first_correct_index is not None:
            print_count = max(top_k, first_correct_index + 1)
        else:
            print_count = top_k
        print_count = min(print_count, 50)
        top_results = candidates[:top_k]
        print(f"\n匹配结果 (显示前{print_count}个):")
        for i, result in enumerate(candidates[:print_count], 1):
            gt_mark = "正确" if result['is_ground_truth'] else "错误"
            print(f"   {i}. {os.path.basename(result['image_path'])} "
                  f"(融合: {result['final_score']:.3f}, "
                  f"孪生: {result['siamese_score']:.3f}, "
                  f"传统: {result['traditional_score']:.3f}) {gt_mark}")
            if result['is_ground_truth'] and i == first_correct_index + 1:
                print(f"      ↑ 首个正确答案位于第 {i} 位")
        if first_correct_index is not None:
            print(f"\n正确答案排名: 第 {first_correct_index + 1} 位")
        else:
            print(f"\n在前 {len(candidates)} 个候选中未找到正确答案")
        return top_results

    def batch_evaluate(self, max_queries=None):
        # 批量评估所有测试图片
        print(f"开始批量评估...")
        query_images = self.test_images.copy()
        if max_queries:
            query_images = query_images[:max_queries]
        print(f"   查询图片: {len(query_images)}")
        print(f"   候选池大小: {len(self.test_images)}")
        total_queries = 0
        top1_correct = 0
        top3_correct = 0
        top5_correct = 0
        rank_sum = 0
        for query_img in tqdm(query_images, desc="批量评估"):
            results = self.predict_matches(query_img, top_k=10)
            if not results:
                continue
            total_queries += 1
            correct_rank = None
            for i, result in enumerate(results, 1):
                if result['is_ground_truth']:
                    correct_rank = i
                    break
            if correct_rank:
                rank_sum += correct_rank
                if correct_rank == 1:
                    top1_correct += 1
                if correct_rank <= 3:
                    top3_correct += 1
                if correct_rank <= 5:
                    top5_correct += 1
        if total_queries > 0:
            metrics = {
                'total_queries': total_queries,
                'top1_accuracy': top1_correct / total_queries,
                'top3_accuracy': top3_correct / total_queries,
                'top5_accuracy': top5_correct / total_queries,
                'average_rank': rank_sum / total_queries if rank_sum > 0 else 0,
                'found_correct': rank_sum > 0
            }
        else:
            metrics = {
                'total_queries': 0,
                'top1_accuracy': 0,
                'top3_accuracy': 0,
                'top5_accuracy': 0,
                'average_rank': 0,
                'found_correct': False
            }
        print(f"\n批量评估结果:")
        print(f"   总查询数: {metrics['total_queries']}")
        print(f"   Top-1 准确率: {metrics['top1_accuracy']:.3f}")
        print(f"   Top-3 准确率: {metrics['top3_accuracy']:.3f}")
        print(f"   Top-5 准确率: {metrics['top5_accuracy']:.3f}")
        print(f"   平均排名: {metrics['average_rank']:.2f}")
        return metrics

    def predict_matches_in_weizhuihe(self, query_image, top_k=5):
        # 在 weizhuihe 目录下全库匹配，返回 top-k 结果
        print(f"\n[全库匹配] 开始推理: {os.path.basename(query_image)}")
        is_complete = self._is_complete_slip(query_image)
        if is_complete:
            print(f"该图片为完整竹简，跳过匹配。")
            return []
        print(f"该图片为残简，开始匹配...\n")
        print(f"   提取特征...")
        query_features = self._extract_slip_features(query_image)
        weizhuihe_path = "weizhuihe"
        all_images = []
        for fname in os.listdir(weizhuihe_path):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                full_path = os.path.join(weizhuihe_path, fname)
                if os.path.abspath(full_path) != os.path.abspath(query_image):
                    all_images.append(full_path)
        print(f"   候选图片数: {len(all_images)}")
        candidates = []
        for candidate_img in tqdm(all_images, desc="全库匹配计算"):
            candidate_features = self._extract_slip_features(candidate_img)
            siamese_score = self._compute_siamese_similarity(query_image, candidate_img)
            traditional_score = self._compute_traditional_similarity(query_features, candidate_features)
            final_score = 0.7 * siamese_score + 0.3 * traditional_score
            candidates.append({
                'image_path': candidate_img,
                'siamese_score': siamese_score,
                'traditional_score': traditional_score,
                'final_score': final_score
            })
        candidates.sort(key=lambda x: x['final_score'], reverse=True)
        top_results = candidates[:top_k]
        print(f"\nTop-{top_k} 匹配结果：")
        for i, result in enumerate(top_results, 1):
            print(f"   {i}. {os.path.basename(result['image_path'])} (融合: {result['final_score']:.3f}, 孪生: {result['siamese_score']:.3f}, 传统: {result['traditional_score']:.3f})")
        return top_results


if __name__ == '__main__':
    # 测试推理引擎
    print("测试竹简缀合推理引擎...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    engine = BambooSlipInferenceEngine(
        model_path='twin_model.pth',
        device=device
    )
    def collect_all_weizhuihe_images():
        # 递归收集weizhuihe目录下所有图片路径
        image_list = []
        for root, dirs, files in os.walk('weizhuihe'):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_list.append(os.path.join(root, file))
        return image_list
    print("\n请选择推理模式：")
    print("1. 测试集单图推理（默认）")
    print("2. 从weizhuihe任意选图并全库匹配")
    mode = input("输入模式编号（1/2，回车默认1）：").strip()
    if mode == '2':
        # 模式2：weizhuihe全库匹配
        all_images = collect_all_weizhuihe_images()
        if not all_images:
            print("weizhuihe目录下未找到图片！")
        else:
            print(f"共找到 {len(all_images)} 张图片。")
            print("请输入要推理的图片路径（可直接粘贴weizhuihe/xxx/xxx.jpg），或回车随机选一张：")
            img_path = input().strip()
            if not img_path:
                import random
                img_path = random.choice(all_images)
                print(f"随机选择图片：{img_path}")
            elif not os.path.exists(img_path):
                print(f"图片不存在：{img_path}")
                exit(1)
            # 直接调用引擎方法，避免重复代码
            engine.predict_matches_in_weizhuihe(img_path, top_k=5)
            print("\n全库匹配完成！")
    else:
        # 模式1：测试集单图推理
        if engine.test_images:
            test_image = engine.test_images[0]
            print(f"\n 单图推理测试: {os.path.basename(test_image)}")
            results = engine.predict_matches(test_image, top_k=5)
            print(f"\n 单图推理完成!")
            # 快速批量评估（限制5张图片）
            print(f"\n 快速批量评估...")
            metrics = engine.batch_evaluate(max_queries=5)
            print(f"\n测试完成!")
        else:
            print("错误：没有找到测试图片")
