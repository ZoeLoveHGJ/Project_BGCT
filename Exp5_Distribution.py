# -*- coding: utf-8 -*-
"""
Exp5_Distribution.py: 标签分布适应性测试 (对应论文 Table 5)

==============================================================================
 实验说明 (Table 5 - Distribution Adaptability)
==============================================================================
实验配置:
1.  **变量**: ID分布模式 (id_distribution)。
    - **Random**: 标准随机分布（基准）。
    - **Sequential**: 连续ID（如 00...01, 00...02），模拟流水线产品。
    - **Prefixed**: 具有相同长前缀（如 厂家代码），模拟同一批次密集堆叠。
    - **Dispersed**: 非连续、稀疏碰撞位，模拟最坏情况下的树分裂复杂度。

2.  **固定参数**:
    - **标签数量**: 固定为 2000 (中等规模)，控制变量以聚焦分布特征。
    - **信道环境**: 理想信道 (BER=0)，排除物理层噪声干扰，专注于逻辑效率。
==============================================================================
"""

import os
import time
import multiprocessing
import numpy as np
from tqdm import tqdm

from Framework import run_simulation

from algorithm_base_config import ALGORITHM_LIBRARY, ALGORITHMS_TO_TEST

from Tool import SimulationAnalytics

ALGORITHMS_TO_TEST = ['BGCT']
# 结果保存根目录
RESULTS_BASE_DIR = "results"

# 定义实验列表
EXPERIMENTS_TO_RUN = [
    {
        "name": "Exp5_Distribution_Adaptability",
        "description": "评估算法在随机、连续、前缀密集和分散分布下的性能一致性",
        
        # --- 变量配置 (X轴: 分布模式) ---
        "varying_param_key": "id_distribution",
        # 测试四种典型的分布模式
        "varying_param_values": ["random", "sequential", "prefixed", "dispersed"],
        
        # --- 场景配置 (固定参数) ---
        "scenario_config": {
            'TOTAL_TAGS': 5000,            # 固定 2000 个标签
            'BINARY_LENGTH': 96,           # 标准 EPC 长度
            'prefix_length': 48,           # 仅对 'prefixed' 模式有效：前48位相同
        },
        
        # --- 算法通用配置 ---
        "algorithm_specific_config": {
            'ber': 0.0,                    # 理想信道
            'dropout_rate': 0.0,
            'collision_miss_rate': 0.0,
            'enable_refined_energy_model': True,
            'enable_resource_monitoring': True,
        }
    },
]


NUM_RUNS_PER_POINT = 50


def run_single_task(task_params: tuple):
    """
    执行单次仿真任务的包装器。
    
    Args:
        task_params (tuple): (算法名, 场景配置, 算法配置, 运行ID)
        
    Returns:
        tuple: (仿真结果, 完整配置日志, 算法名, 运行ID)
    """
    algo_name, scenario_config, algo_specific_config, run_id = task_params
    
    algo_info = ALGORITHM_LIBRARY[algo_name]
    algo_class = algo_info["class"]
    default_config = algo_info["config"]
    
    final_algo_config = {**default_config, **algo_specific_config}

    result_dict = run_simulation(
        scenario_config=scenario_config,
        algorithm_class=algo_class,
        algorithm_specific_config=final_algo_config
    )
    
    full_config_log = {**scenario_config, **final_algo_config}
    
    return (result_dict, full_config_log, algo_name, run_id)


def main():
    print(f"=== RFID防碰撞算法：分布适应性测试 (Table 5) ===")
    print(f"适配框架: Framework | 工具库: Tool")
    print(f"对比算法: {ALGORITHMS_TO_TEST}")
    print(f"保存目录: {RESULTS_BASE_DIR}\n")
    
    for experiment in EXPERIMENTS_TO_RUN:
        exp_name = experiment["name"]
        output_dir = os.path.join(RESULTS_BASE_DIR, exp_name)
        os.makedirs(output_dir, exist_ok=True)

        print("-" * 60)
        print(f"正在执行: {exp_name}")
        print(f"变量 (X轴): {experiment['varying_param_key']}")
        print(f"变量值: {experiment['varying_param_values']}")
        print("-" * 60)

        tasks = []
        varying_key = experiment['varying_param_key'] # 这里是 'id_distribution'
        
        for dist_mode in experiment['varying_param_values']:
            current_scenario = experiment['scenario_config'].copy()
            current_algo_config = experiment['algorithm_specific_config'].copy()
            
            current_scenario[varying_key] = dist_mode

            for algo_name in ALGORITHMS_TO_TEST:
                for r in range(NUM_RUNS_PER_POINT):
                    tasks.append((algo_name, current_scenario, current_algo_config, r))
        
        print(f"任务总数: {len(tasks)}")
        num_cores = max(1, multiprocessing.cpu_count() - 2)
        print(f"使用 {num_cores} 核并行计算...")
        
        analytics = SimulationAnalytics()
        start_time = time.time()
        
        with multiprocessing.Pool(processes=num_cores) as pool:
            iterator = pool.imap_unordered(run_single_task, tasks)
            for res in tqdm(iterator, total=len(tasks), unit="job", desc="Simulating"):
                result_dict, config_log, algo_name, run_id = res
                
                analytics.add_run_result(
                    result_dict=result_dict, 
                    scenario_config=config_log, # 包含当前的 id_distribution
                    algorithm_name=algo_name, 
                    run_id=run_id,
                    algorithm_config=config_log 
                )
                
        duration = time.time() - start_time
        print(f"\n耗时: {duration:.2f} 秒")
        
        print(f"正在生成统计表格和图表 (X轴: {varying_key})...")
        
        analytics.save_to_csv(x_axis_key=varying_key, output_dir=output_dir)
        

        analytics.plot_results(
            x_axis_key=varying_key,
            algorithm_library=ALGORITHM_LIBRARY,
            save_path=os.path.join(output_dir, "Table5_Distribution_Adaptability.png")
        )
        
        print(f"实验完成，结果已保存至: {output_dir}\n")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()