# -*- coding: utf-8 -*-
"""
算法通用性与数据完整性验证
"""

import unittest
import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Framework import run_simulation
from algorithm_base_config import ALGORITHM_LIBRARY
from Tool import SimulationAnalytics


TEST_TARGETS = ['EAQ_CBB'] 

EXPECTED_METRICS = [
    'total_protocol_time_ms',
    'throughput_tags_per_sec',
    'system_efficiency',
    'collision_slots',
    'avg_query_efficiency',
    'idle_slots',
    'total_energy_uj',
    'avg_tag_bits',
    'avg_reader_bits',
    'avg_tag_responses',
    'peak_stack_depth'  # 重点检查项
]

class TestAlgorithmDataQuality(unittest.TestCase):
    
    def setUp(self):
        """初始化测试环境"""
        self.base_scenario = {
            'TOTAL_TAGS': 1000,             # 使用小规模标签数以加快测试速度
            'BINARY_LENGTH': 256,
            'id_distribution': 'random'  
        }
        
        # 动态加载测试目标
        if TEST_TARGETS:
            self.targets = TEST_TARGETS
        else:
            self.targets = list(ALGORITHM_LIBRARY.keys())
            
        print(f"\n[Setup] 待测算法列表 ({len(self.targets)}个): {self.targets}")

    def test_algorithms_metrics_integrity(self):
        """对列表中的算法进行全指标完整性检查"""
        for algo_name in self.targets:
            with self.subTest(algorithm=algo_name):
                self._verify_single_algo_metrics(algo_name)

    def _verify_single_algo_metrics(self, algo_name):
        """核心验证逻辑"""
        print(f"\n{'-'*60}")
        print(f"Testing Algorithm: [ {algo_name} ]")
        
        # 1. 准备算法配置 (强制理想信道，开启资源监控)
        if algo_name not in ALGORITHM_LIBRARY:
            print(f"Skipping {algo_name}: Not found in library.")
            return

        algo_info = ALGORITHM_LIBRARY[algo_name]
        algo_class = algo_info["class"]
        
        # 复制配置并开启 resource_monitoring
        test_config = {
            **algo_info.get("config", {}),
            'enable_resource_monitoring': True, # 必须开启，否则无法捕获 stack_depth
            'ber': 0.0,
            'dropout_rate': 0.0,
            'collision_miss_rate': 0.0
        }
        
        # 2. [运行] 仿真
        print("  -> Running Simulation...")
        try:
            raw_result = run_simulation(self.base_scenario, algo_class, test_config)
        except Exception as e:
            self.fail(f"CRITICAL: 算法 {algo_name} 运行崩溃! Error: {str(e)}")

        # 3. [摄入] Tool 数据处理
        print("  -> Ingesting data into Tool...")
        analytics = SimulationAnalytics()
        try:
            analytics.add_run_result(
                result_dict=raw_result, 
                scenario_config=self.base_scenario, 
                algorithm_name=algo_name, 
                run_id=1, 
                algorithm_config=test_config
            )
            df_raw = analytics.get_results_dataframe()
        except Exception as e:
            self.fail(f"FAIL: Tool hhh无法接收数据。Error: {e}")

        # 4. [计算] 衍生指标
        print("  -> Calculating derived metrics...")
        try:
            df_final = analytics._calculate_derived_metrics(df_raw)
        except Exception as e:
            self.fail(f"FAIL: Tool 计算衍生指标失败。Error: {e}")

        if df_final.empty:
            self.fail("FAIL: 生成的 DataFrame 为空")
        
        record = df_final.iloc[0]
        
        # 5. [验证] 全面指标检查
        print("  -> Verifying Metrics Existence and Rationality...")
        
        for metric in EXPECTED_METRICS:
            # 5.1 检查指标是否存在
            if metric not in record:
                self.fail(f"FAIL: 缺失关键指标 '{metric}'。检查算法是否输出了必要的基础数据。")
            
            val = record[metric]
            print(f"     [{metric}]: {val:.4f}")
            
            # 5.2 检查数值合理性
            if metric == 'system_efficiency':
                self.assertGreaterEqual(val, 0.0)
                self.assertLessEqual(val, 1.0, "FAIL: 系统效率不能超过 1.0")
                
            elif metric == 'peak_stack_depth':
                self.assertGreater(val, 0, "FAIL: 峰值栈深度必须 > 0")
                if 'NLHQT' in algo_name and self.base_scenario['TOTAL_TAGS'] > 10:
                    self.assertGreater(val, 1, "FAIL: 对于 50 个标签，NLHQT 栈深度理应显著增加")
                    
            elif metric == 'avg_tag_responses':
                self.assertGreaterEqual(val, 1.0, "FAIL: 平均响应次数物理极限不能小于 1.0")
                
            elif metric == 'total_energy_uj':
                self.assertGreater(val, 0, "FAIL: 总能耗必须为正数")
                
            elif metric == 'collision_slots':
                self.assertGreaterEqual(val, 0)
                
        print(f"[PASS] {algo_name} passed all metrics integrity checks.")

if __name__ == '__main__':
    unittest.main()