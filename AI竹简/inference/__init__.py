#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference包初始化文件
包含推理和匹配预测相关模块
"""

# 导入当前可用的模块
# from .slip_matcher import SlipMatcher, predict_matches

# try:
#     from .feature_comparator import BambooSlipFeatureComparator, compare_bamboo_slip_features
# except ImportError:
#     pass

try:
    from .visualizer import MatchingVisualizer, visualize_matching_results
except ImportError:
    pass

__all__ = [
    # 'SlipMatcher',
    # 'predict_matches',
    # 'BambooSlipFeatureComparator',
    # 'compare_bamboo_slip_features',
    'MatchingVisualizer',
    'visualize_matching_results'
]

__version__ = '1.0.0'
