
"""
Exp2_Communication.py: ID长度对性能影响实验 (对应论文图6)
==============================================================================
本脚本用于生成论文中的 [Figure 6]，重点评估标签ID长度 (BINARY_LENGTH) 
变化时，不同防碰撞算法的性能适应性。

实验设计:
1.  **变量**: ID长度 (BINARY_LENGTH)，范围 [64, 96, 128, 160, 192, 256]。
2.  **固定**: 标签总数 (TOTAL_TAGS = 10000)，理想信道 (BER=0)。
3.  **目标**: 验证算法在处理长ID时的通信开销(Communication Cost)和
    吞吐率(Throughput)变化趋势。

==============================================================================
"""

import os
import time
import multiprocessing
import numpy as np
import pandas as pd
from tqdm import tqdm


from Framework import run_simulation
from Tool import SimulationAnalytics

from algorithm_base_config import ALGORITHM_LIBRARY, ALGORITHMS_TO_TEST


RESULTS_BASE_DIR = "results"
EXPERIMENT_NAME = "Exp8_Peak_Stack_Depth_ID"
OUTPUT_DIR = os.path.join(RESULTS_BASE_DIR, EXPERIMENT_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)


ID_LENGTH_VALUES = [64, 96, 128, 160, 192, 256]


SCENARIO_CONFIG_TEMPLATE = {
    'TOTAL_TAGS': 10000,
    'id_distribution': 'random',

}


ALGO_BASE_CONFIG = {
    'ber': 0.0,
    'guard_interval_us': 0.0,
    'collision_miss_rate': 0.0,
    'dropout_rate': 0.0,

    'enable_refined_energy_model': True,
    'enable_resource_monitoring': True
}


NUM_RUNS = 50


def run_id_length_task(task_args):
    """
    执行单次仿真任务，测试特定ID长度下的性能。

    Args:
        task_args: (algo_name, id_length, run_id)

    Returns:
        tuple: (result, scenario_config, algo_name, run_id, algo_config)
    """
    algo_name, id_length, run_id = task_args

    if algo_name not in ALGORITHM_LIBRARY:
        raise ValueError(f"算法 '{algo_name}' 未定义！")

    algo_info = ALGORITHM_LIBRARY[algo_name]
    algo_class = algo_info["class"]

    final_algo_config = {**algo_info.get("config", {}), **ALGO_BASE_CONFIG}

    current_scenario = SCENARIO_CONFIG_TEMPLATE.copy()
    current_scenario['BINARY_LENGTH'] = int(id_length)

    result_dict = run_simulation(
        scenario_config=current_scenario,
        algorithm_class=algo_class,
        algorithm_specific_config=final_algo_config
    )

    return (result_dict, current_scenario, algo_name, run_id, final_algo_config)


def main():
    print(f"\n{'='*80}")
    print(f"实验: {EXPERIMENT_NAME} (对应论文 Figure 6)")
    print(f"目标: 测试 ID 长度 {ID_LENGTH_VALUES} 对协议性能的影响")
    print(f"固定标签数: {SCENARIO_CONFIG_TEMPLATE['TOTAL_TAGS']}")
    print(f"对比算法: {ALGORITHMS_TO_TEST}")
    print(f"{'='*80}\n")

    tasks = []
    for length in ID_LENGTH_VALUES:
        for algo_name in ALGORITHMS_TO_TEST:
            for r in range(NUM_RUNS):
                tasks.append((algo_name, length, r))

    total_tasks = len(tasks)
    print(
        f"任务准备就绪: {total_tasks} 个 (长度点 {len(ID_LENGTH_VALUES)} * 算法 {len(ALGORITHMS_TO_TEST)} * 重复 {NUM_RUNS})")

    analytics = SimulationAnalytics()
    num_cores = max(1, multiprocessing.cpu_count() - 2)

    print(f"开始并行仿真 (Cores: {num_cores})...")
    start_time = time.time()

    with multiprocessing.Pool(num_cores) as pool:
        results_iter = pool.imap_unordered(run_id_length_task, tasks)

        for res_tuple in tqdm(results_iter, total=total_tasks, desc="Running ID Length Exp"):
            analytics.add_run_result(*res_tuple)

    duration = time.time() - start_time
    print(f"\n仿真完成。总耗时: {duration:.2f} 秒")

    print(f"正在处理数据...")

    x_axis = "BINARY_LENGTH"

    analytics.save_to_csv(x_axis_key=x_axis, output_dir=OUTPUT_DIR)

    plot_filename = f"{EXPERIMENT_NAME}_Figure6.png"
    analytics.plot_results(
        x_axis_key=x_axis,
        algorithm_library=ALGORITHM_LIBRARY,
        save_path=os.path.join(OUTPUT_DIR, plot_filename)
    )

    print(f"\n实验结束。")
    print(f" -> CSV 数据: {OUTPUT_DIR}/")
    print(f" -> 趋势图: {os.path.join(OUTPUT_DIR, plot_filename)}")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
