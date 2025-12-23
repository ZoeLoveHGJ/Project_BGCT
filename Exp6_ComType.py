
"""
Exp6_ComType.py: 算法可扩展性对比实验 (对应论文图5)
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


ALGORITHMS_TO_TEST = ['BGCT', 'HT_EEAC', 'FHS_RAC']

RESULTS_BASE_DIR = "results"
EXPERIMENT_NAME = "Exp6_Compare_Type"
OUTPUT_DIR = os.path.join(RESULTS_BASE_DIR, EXPERIMENT_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)


TAG_COUNTS = np.linspace(1000, 10000, 10, dtype=int)


SCENARIO_CONFIG_TEMPLATE = {
    'BINARY_LENGTH': 96,
    'id_distribution': 'random',

}


ALGO_BASE_CONFIG = {
    'ber': 0.0,
    'guard_interval_us': 0.0,
    'collision_miss_rate': 0.0,
    'dropout_rate': 0.0,

    'enable_refined_energy_model': True,
    'enable_resource_monitoring': False
}


NUM_RUNS = 50


def run_scalability_task(task_args):
    """
    并行执行单个仿真任务。

    Args:
        task_args: (algo_name, n_tags, run_id)

    Returns:
        tuple: (result, scenario_config, algo_name, run_id, algo_config)
        注: 返回5个元素，以适配 SimulationAnalytics.add_run_result
    """
    algo_name, n_tags, run_id = task_args

    if algo_name not in ALGORITHM_LIBRARY:
        raise ValueError(f"算法 '{algo_name}' 未在 algorithm_base_config.py 中定义！")

    algo_info = ALGORITHM_LIBRARY[algo_name]
    algo_class = algo_info["class"]

    final_algo_config = {**algo_info.get("config", {}), **ALGO_BASE_CONFIG}

    current_scenario = SCENARIO_CONFIG_TEMPLATE.copy()
    current_scenario['TOTAL_TAGS'] = int(n_tags)

    result_dict = run_simulation(
        scenario_config=current_scenario,
        algorithm_class=algo_class,
        algorithm_specific_config=final_algo_config
    )

    return (result_dict, current_scenario, algo_name, run_id, final_algo_config)


def main():
    print(f"\n{'='*80}")
    print(f"实验: {EXPERIMENT_NAME} (对应论文 Figure 5)")
    print(f"目标: 测试算法在标签数量 [{TAG_COUNTS[0]} - {TAG_COUNTS[-1]}] 范围内的可扩展性")
    print(f"对比算法: {ALGORITHMS_TO_TEST}")
    print(f"{'='*80}\n")

    tasks = []
    for n_tags in TAG_COUNTS:
        for algo_name in ALGORITHMS_TO_TEST:
            for r in range(NUM_RUNS):
                tasks.append((algo_name, n_tags, r))

    total_tasks = len(tasks)
    print(
        f"任务生成完毕: 共 {total_tasks} 个仿真任务 (标签点 {len(TAG_COUNTS)} * 算法 {len(ALGORITHMS_TO_TEST)} * 重复 {NUM_RUNS})")

    analytics = SimulationAnalytics()

    num_cores = max(1, multiprocessing.cpu_count() - 2)

    print(f"启动并行池 (Cores: {num_cores})...")
    start_time = time.time()

    with multiprocessing.Pool(num_cores) as pool:

        results_iter = pool.imap_unordered(run_scalability_task, tasks)

        for res_tuple in tqdm(results_iter, total=total_tasks, desc="Running Scalability Exp"):

            analytics.add_run_result(*res_tuple)

    duration = time.time() - start_time
    print(f"\n所有任务完成。总耗时: {duration:.2f} 秒")

    print(f"正在分析数据并生成图表...")

    x_axis = "TOTAL_TAGS"

    analytics.save_to_csv(x_axis_key=x_axis, output_dir=OUTPUT_DIR)

    plot_filename = f"{EXPERIMENT_NAME}_Figure5.png"
    analytics.plot_results(
        x_axis_key=x_axis,
        algorithm_library=ALGORITHM_LIBRARY,
        save_path=os.path.join(OUTPUT_DIR, plot_filename)
    )

    print(f"\n实验结束。请查看结果:")
    print(f" -> 数据目录: {OUTPUT_DIR}/")
    print(f" -> 最终图表: {os.path.join(OUTPUT_DIR, plot_filename)}")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
