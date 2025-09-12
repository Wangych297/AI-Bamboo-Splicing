#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess包初始化文件
包含数据预处理相关模块
"""

from .pair_generator import build_positive_pairs, build_negative_pairs

__all__ = [
    'build_positive_pairs',
    'build_negative_pairs'
]

__version__ = '1.0.0'
