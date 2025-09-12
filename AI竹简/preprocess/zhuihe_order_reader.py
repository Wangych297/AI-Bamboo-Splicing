# 竹简缀合统计Excel文件读取器
# 功能：解析zhuihetongji.xlsx获取正确的竹简片段拼接顺序
# 支持多个分表：上下拼、左右拼、上下+左右拼、遥缀、特殊等

import pandas as pd
import os
import re
from typing import Dict, List, Tuple, Optional, Set


class ZhuiheOrderReader:
    # 竹简缀合顺序读取器
    def __init__(self, excel_path: str = "zhuihetongji.xlsx"):
        # 初始化读取器
        self.excel_path = excel_path
        self.sheets_data = {}  # 存储各分表数据
        self.adjacency_pairs = set()  # 存储所有相邻片段对
        self.load_data()
    def load_data(self):
        # 加载Excel数据
        try:
            if not os.path.exists(self.excel_path):
                print(f"警告：Excel文件不存在：{self.excel_path}")
                return
            # 读取所有分表
            xl = pd.ExcelFile(self.excel_path)
            print(f"发现分表：{xl.sheet_names}")
            # 读取主要的拼接分表
            target_sheets = ['上下拼', '左右拼', '上下+左右拼']
            for sheet_name in target_sheets:
                if sheet_name in xl.sheet_names:
                    try:
                        df = pd.read_excel(self.excel_path, sheet_name=sheet_name)
                        self.sheets_data[sheet_name] = df
                        print(f"成功读取分表：{sheet_name}，数据形状：{df.shape}")
                    except Exception as e:
                        print(f"读取分表{sheet_name}失败：{e}")
            # 解析各分表的相邻关系
            self._parse_adjacency_relationships()
        except Exception as e:
            print(f"读取Excel文件失败: {e}")
    def _parse_adjacency_relationships(self):
        # 解析各分表中的相邻关系
        print("开始解析相邻关系...")
        # 解析上下拼
        if '上下拼' in self.sheets_data:
            self._parse_vertical_pairs('上下拼')
        # 解析左右拼
        if '左右拼' in self.sheets_data:
            self._parse_horizontal_pairs('左右拼')
        # 解析复合拼接
        if '上下+左右拼' in self.sheets_data:
            self._parse_composite_pairs('上下+左右拼')
        print(f"总共解析到 {len(self.adjacency_pairs)} 对相邻片段")
    def _parse_vertical_pairs(self, sheet_name: str):
        # 解析上下拼分表
        df = self.sheets_data[sheet_name]
        count = 0
        # 跳过说明行（第一行通常是说明）
        for idx, row in df.iterrows():
            if idx == 0:  # 跳过表头说明行
                continue
            # 获取该行的所有简号
            bamboo_ids = []
            for col in df.columns:
                if '简号' in col or col.startswith('简号'):
                    val = row[col]
                    if pd.notna(val) and str(val).strip():
                        cleaned_id = self._clean_bamboo_id(str(val))
                        if cleaned_id:
                            bamboo_ids.append(cleaned_id)
            # 生成相邻对（上下关系）
            for i in range(len(bamboo_ids) - 1):
                pair = (bamboo_ids[i], bamboo_ids[i + 1])
                self.adjacency_pairs.add(pair)
                # 也添加反向对
                self.adjacency_pairs.add((bamboo_ids[i + 1], bamboo_ids[i]))
                count += 1
        print(f"从{sheet_name}解析到 {count} 对上下相邻关系")
    def _parse_horizontal_pairs(self, sheet_name: str):
        # 解析左右拼分表
        df = self.sheets_data[sheet_name]
        count = 0
        # 跳过说明行
        for idx, row in df.iterrows():
            if idx == 0:  # 跳过表头说明行
                continue
            # 获取左、中、右的简号
            bamboo_ids = []
            for col in ['简号（左）', '简号（中）', '简号（右）']:
                if col in df.columns:
                    val = row[col]
                    if pd.notna(val) and str(val).strip():
                        cleaned_id = self._clean_bamboo_id(str(val))
                        if cleaned_id:
                            bamboo_ids.append(cleaned_id)
            # 生成相邻对（左右关系）
            for i in range(len(bamboo_ids) - 1):
                pair = (bamboo_ids[i], bamboo_ids[i + 1])
                self.adjacency_pairs.add(pair)
                # 也添加反向对
                self.adjacency_pairs.add((bamboo_ids[i + 1], bamboo_ids[i]))
                count += 1
        print(f"从{sheet_name}解析到 {count} 对左右相邻关系")
    def _parse_composite_pairs(self, sheet_name: str):
        # 解析上下+左右拼分表
        df = self.sheets_data[sheet_name]
        count = 0
        # 这个分表比较复杂，需要特殊处理
        for idx, row in df.iterrows():
            if idx == 0:  # 跳过表头说明行
                continue
            # 提取各种简号
            bamboo_ids = []
            for col in df.columns:
                if '简号' in col:
                    val = row[col]
                    if pd.notna(val) and str(val).strip():
                        # 处理复合表示，如 [9-543]+[9-835]
                        composite_ids = self._parse_composite_bamboo_id(str(val))
                        bamboo_ids.extend(composite_ids)
            # 根据备注信息确定拼接关系
            remark = str(row.get('缀合情况', '')) if '缀合情况' in df.columns else ''
            # 这里需要根据具体的缀合情况描述来解析
            # 暂时简单处理为相邻关系
            for i in range(len(bamboo_ids) - 1):
                pair = (bamboo_ids[i], bamboo_ids[i + 1])
                self.adjacency_pairs.add(pair)
                self.adjacency_pairs.add((bamboo_ids[i + 1], bamboo_ids[i]))
                count += 1
        print(f"从{sheet_name}解析到 {count} 对复合相邻关系")
    def _clean_bamboo_id(self, bamboo_id: str) -> str:
        # 清理竹简编号
        if not bamboo_id or bamboo_id == 'nan':
            return ''
        
        # 移除方括号
        bamboo_id = bamboo_id.strip('[]')
        
        # 移除多余的空格
        bamboo_id = bamboo_id.strip()
        
        # 基本格式验证（如8-199, 9-1758等）
        if re.match(r'^\d+-\d+$', bamboo_id):
            return bamboo_id
        
        return ''
    def _parse_composite_bamboo_id(self, text: str) -> List[str]:
        # 解析复合竹简编号，如 [9-543]+[9-835]
        ids = []
        
        # 查找所有形如[数字-数字]的模式
        pattern = r'\[(\d+-\d+)\]'
        matches = re.findall(pattern, text)
        
        if matches:
            ids.extend(matches)
        else:
            # 如果没有方括号，直接清理
            cleaned = self._clean_bamboo_id(text)
            if cleaned:
                ids.append(cleaned)
        
        return ids
    def get_adjacent_pairs(self) -> Set[Tuple[str, str]]:
        # 获取所有相邻的竹简片段对
        
        return self.adjacency_pairs.copy()
    
    def are_adjacent(self, id1: str, id2: str) -> bool:
        # 判断两个竹简片段是否相邻（可以拼接）
        
        return (id1, id2) in self.adjacency_pairs
    
    def get_bamboo_sequence(self, bamboo_id: str) -> List[str]:
        # 获取特定竹简编号的完整序列（已废弃，请使用get_adjacent_pairs）
        print("警告：get_bamboo_sequence已废弃，请使用get_adjacent_pairs获取相邻对")
        return []
    
    def get_all_sequences(self) -> Dict[str, List[str]]:
        # 获取所有竹简的片段序列（已废弃）
        print("警告：get_all_sequences已废弃")
        return {}
    
    def generate_adjacent_pairs_for_training(self) -> List[Tuple[str, str]]:
        # 为训练生成相邻片段对，返回实际的图片文件名对

        image_pairs = []
        
        for (id1, id2) in self.adjacency_pairs:
            # 查找对应的图片文件
            img1_files = self._find_image_files(id1)
            img2_files = self._find_image_files(id2)
            
            # 生成所有可能的组合
            for img1 in img1_files:
                for img2 in img2_files:
                    image_pairs.append((img1, img2))
        
        print(f"生成 {len(image_pairs)} 个图片文件相邻对")
        return image_pairs
    def _find_image_files(self, bamboo_id: str, data_dirs: List[str] = None) -> List[str]:
        # 查找指定竹简编号对应的图片文件
        
        # Args:
        #     bamboo_id: 竹简编号，如 "9-1897"
        #     data_dirs: 图片目录列表
            
        # Returns:
        #     List[str]: 找到的图片文件路径列表
        if data_dirs is None:
            data_dirs = ["yizhuihe/", "weizhuihe/"]
        
        found_files = []
        
        # 提取简号中的数字部分
        # 如 "9-1897" -> "1897"
        number_part = self._extract_number_from_bamboo_id(bamboo_id)
        if not number_part:
            return found_files
        
        # 在yizhuihe目录中搜索（按缀合组搜索）
        yizhuihe_path = "yizhuihe/"
        if os.path.exists(yizhuihe_path):
            # 遍历所有缀合组文件夹
            for group_folder in os.listdir(yizhuihe_path):
                group_path = os.path.join(yizhuihe_path, group_folder)
                if not os.path.isdir(group_path):
                    continue
                
                # 在该缀合组文件夹中查找匹配的图片文件
                for filename in os.listdir(group_path):
                    if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        continue
                    
                    if self._is_filename_match_bamboo_number(filename, number_part):
                        full_path = os.path.join(group_path, filename)
                        found_files.append(full_path)
        
        # 在weizhuihe目录中搜索
        weizhuihe_path = "weizhuihe/"
        if os.path.exists(weizhuihe_path):
            for filename in os.listdir(weizhuihe_path):
                if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    continue
                
                if self._is_filename_match_bamboo_number(filename, number_part):
                    full_path = os.path.join(weizhuihe_path, filename)
                    found_files.append(full_path)
        
        return found_files
    def _extract_number_from_bamboo_id(self, bamboo_id: str) -> str:
        # 从竹简编号中提取数字部分
        # 如 "9-1897" -> "1897"
        import re
        # 查找 "-" 后面的数字
        match = re.search(r'-(\d+)', bamboo_id)
        if match:
            return match.group(1)
        
        # 如果没有找到"-"，尝试提取最后一组数字
        numbers = re.findall(r'\d+', bamboo_id)
        if numbers:
            return numbers[-1]
        
        return ''
    def _is_filename_match_bamboo_number(self, filename: str, number: str) -> bool:
        # 检查文件名是否匹配竹简数字编号
        
        # 移除文件扩展名
        base_name = os.path.splitext(filename)[0]
        
        # 提取文件名中的第一组数字
        import re
        file_numbers = re.findall(r'\d+', base_name)
        if not file_numbers:
            return False
        
        first_number = file_numbers[0]
        
        # 标准化比较：移除前导零
        file_num_clean = first_number.lstrip('0') or '0'
        number_clean = number.lstrip('0') or '0'
        
        # 精确匹配（避免前缀匹配问题）
        return file_num_clean == number_clean
    def validate_image_files(self, data_dirs: List[str] = None) -> Dict[str, Dict]:
        # 验证Excel中的竹简编号是否与实际图片文件匹配

        if data_dirs is None:
            data_dirs = ["yizhuihe/", "weizhuihe/"]
        
        validation_results = {}
        
        # 统计所有相邻对中涉及的竹简编号
        all_bamboo_ids = set()
        for (id1, id2) in self.adjacency_pairs:
            all_bamboo_ids.add(id1)
            all_bamboo_ids.add(id2)
        
        print(f"需要验证 {len(all_bamboo_ids)} 个竹简编号")
        
        for bamboo_id in all_bamboo_ids:
            found_files = self._find_image_files(bamboo_id, data_dirs)
            
            validation_results[bamboo_id] = {
                'found_files': found_files,
                'file_count': len(found_files),
                'status': 'found' if found_files else 'missing'
            }
        
        found_count = sum(1 for v in validation_results.values() if v['status'] == 'found')
        print(f"找到图片文件的竹简: {found_count}/{len(all_bamboo_ids)}")
        
        return validation_results


if __name__ == '__main__':
    # 测试Excel读取功能
    print("测试竹简缀合顺序读取器...")
    
    reader = ZhuiheOrderReader()
    
    print(f"\n=== 相邻关系统计 ===")
    adjacent_pairs = reader.get_adjacent_pairs()
    print(f"总共 {len(adjacent_pairs)} 对相邻关系")
    
    # 显示前10个相邻对作为示例
    print("前10个相邻对:")
    for i, (id1, id2) in enumerate(list(adjacent_pairs)[:10]):
        print(f"  {i+1}. {id1} <-> {id2}")
    
    print(f"\n=== 图片文件验证 ===")
    validation = reader.validate_image_files()
    
    found_bamboos = [k for k, v in validation.items() if v['status'] == 'found']
    missing_bamboos = [k for k, v in validation.items() if v['status'] == 'missing']
    
    print(f"找到图片的竹简编号 ({len(found_bamboos)}):")
    for bamboo_id in found_bamboos[:10]:  # 显示前10个
        files = validation[bamboo_id]['found_files']
        print(f"  {bamboo_id}: {len(files)} 个文件")
    
    if missing_bamboos:
        print(f"\n缺失图片的竹简编号 ({len(missing_bamboos)}):")
        for bamboo_id in missing_bamboos[:10]:  # 显示前10个
            print(f"  {bamboo_id}")
    
    print(f"\n=== 生成训练用相邻对 ===")
    training_pairs = reader.generate_adjacent_pairs_for_training()
    print(f"生成 {len(training_pairs)} 个图片文件相邻对")
    
    # 显示前5个作为示例
    print("前5个训练对:")
    for i, (img1, img2) in enumerate(training_pairs[:5]):
        print(f"  {i+1}. {os.path.basename(img1)} <-> {os.path.basename(img2)}")
    
    print("\n测试完成！")
