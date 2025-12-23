

import os
import time
import multiprocessing
import numpy as np
import pandas as pd
from tqdm import tqdm


from Framework import run_simulation
from Tool import SimulationAnalytics

from BGCT import BGCT

from algorithm_base_config import ALGORITHM_LIBRARY


RESULTS_BASE_DIR = "results"
EXPERIMENT_NAME = "Exp0_d_target_Sensitivity"
OUTPUT_DIR = os.path.join(RESULTS_BASE_DIR, EXPERIMENT_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)


DTARGET_VALUES = np.arange(2, 17, 2, dtype=int)


FIXED_SCENARIO = {
    'TOTAL_TAGS': 4000,
    'BINARY_LENGTH': 96,
    'id_distribution': 'random',
}


BASE_ALGO_CONFIG = {

    'ber': 0.0,
    'guard_interval_us': 0.0,
    'collision_miss_rate': 0.0,
    'dropout_rate': 0.0,


    'enable_refined_energy_model': True,
    'enable_resource_monitoring': True
}


NUM_RUNS_PER_POINT = 50


def run_tuning_task(task_params: tuple):
    """
    执行单次仿真任务。

    Args:
        task_params: (algo_name, scenario_conf, algo_conf, run_id)

    Returns:
        tuple: (result_dict, scenario_config_for_log, algo_name, run_id, full_algo_config)
        注意：返回 5 个元素以匹配 SimulationAnalytics.add_run_result
    """
    algo_name, scenario_conf, algo_conf, run_id = task_params

    result_dict = run_simulation(
        scenario_config=scenario_conf,
        algorithm_class=BGCT,
        algorithm_specific_config=algo_conf
    )

    log_scenario_config = scenario_conf.copy()
    if 'd_target' in algo_conf:
        log_scenario_config['d_target'] = algo_conf['d_target']

    return (result_dict, log_scenario_config, algo_name, run_id, algo_conf)


def main():
    print(f"\n{'='*80}")
    print(f"实验: {EXPERIMENT_NAME}")
    print(f"目标: 寻找 BGCT 在 {FIXED_SCENARIO['TOTAL_TAGS']} 标签下的最优 d_target")
    print(f"参数范围: {DTARGET_VALUES}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"{'='*80}\n")

    tasks = []

    algo_display_name = "BGCT_Param_Sweep"

    for d_val in DTARGET_VALUES:

        current_algo_conf = BASE_ALGO_CONFIG.copy()
        current_algo_conf['d_target'] = int(d_val)

        for i in range(NUM_RUNS_PER_POINT):
            tasks.append((
                algo_display_name,
                FIXED_SCENARIO.copy(),
                current_algo_conf,
                i
            ))

    print(
        f"生成任务总数: {len(tasks)} (参数点 {len(DTARGET_VALUES)} * 重复 {NUM_RUNS_PER_POINT})")

    analytics = SimulationAnalytics()
    num_cores = max(1, multiprocessing.cpu_count() - 2)

    print(f"开始并行计算 (使用 {num_cores} 核心)...")
    start_time = time.time()

    with multiprocessing.Pool(num_cores) as pool:
        results_iter = pool.imap_unordered(run_tuning_task, tasks)

        for res_tuple in tqdm(results_iter, total=len(tasks), desc="Running Simulation"):

            analytics.add_run_result(*res_tuple)

    elapsed = time.time() - start_time
    print(f"\n所有任务完成。总耗时: {elapsed:.2f} 秒")

    print(f"正在处理数据并绘图...")

    x_axis = 'd_target'

    analytics.save_to_csv(x_axis_key=x_axis, output_dir=OUTPUT_DIR)

    tuning_lib = {
        algo_display_name: {
            "style": {"color": "red", "marker": "o", "linestyle": "-", "linewidth": 2},
            "year": 25
        }
    }

    plot_path = os.path.join(OUTPUT_DIR, f"{EXPERIMENT_NAME}_Curve.png")
    analytics.plot_results(
        x_axis_key=x_axis,
        algorithm_library=tuning_lib,
        save_path=plot_path
    )

    print(f"\n结果摘要:")
    print(f"1. CSV 数据已保存至 {OUTPUT_DIR}/")
    print(f"2. 性能趋势图已保存为 {plot_path}")
    print(f"请检查图表以确定使得 System Efficiency 最高或 Time 最低的 d_target 值。")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
