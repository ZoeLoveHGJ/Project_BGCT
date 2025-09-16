

import os
import time
import multiprocessing
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.cm as cm
import matplotlib.colors as mcolors


from Framework import run_simulation

from BGCT import BGCT
from Tool import SimulationAnalytics


RESULTS_BASE_DIR = "results"
EXPERIMENT_NAME = "exp0_dtarget_tuning"
OUTPUT_DIR = os.path.join(RESULTS_BASE_DIR, EXPERIMENT_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)


VARYING_DTARGET_VALUES = []
index = 1
while index <= 16:
    VARYING_DTARGET_VALUES.append(index)
    index += 1


SCENARIO_CONFIG = {
    'TOTAL_TAGS': 4000,
    'BINARY_LENGTH': 96,
    'id_distribution': 'random',
}


ALGORITHM_CONFIG = {
    'ber': 0.0,
    'enable_refined_energy_model': True,
    'enable_resource_monitoring': True,
}


NUM_RUNS_PER_POINT = 5


def run_single_task(task_params: tuple):
    algo_name, scenario_config, algo_specific_config, run_id, algo_class = task_params

    result_dict = run_simulation(
        scenario_config=scenario_config,
        algorithm_class=algo_class,
        algorithm_specific_config=algo_specific_config
    )

    full_config_log = {**scenario_config, **algo_specific_config}

    return (result_dict, full_config_log, algo_name, run_id)


def main():

    print("\n" + "="*80)
    print(f"开始实验: {EXPERIMENT_NAME}")
    print(f"实验目的: 寻找 BGCT 算法的最优 d_target 参数")
    print(
        f"固定场景: {SCENARIO_CONFIG['TOTAL_TAGS']} 个标签, {SCENARIO_CONFIG['BINARY_LENGTH']}位 ID")
    print(f"测试的 d_target 范围: {VARYING_DTARGET_VALUES}")
    print("="*80)

    tasks = []
    dynamic_algorithm_library = {}

    for d_target in VARYING_DTARGET_VALUES:

        algo_name = f'BGCT(d_target={d_target})'

        algo_conf = ALGORITHM_CONFIG.copy()
        algo_conf['d_target'] = d_target

        dynamic_algorithm_library[algo_name] = {
            "class": BGCT,
            "config": algo_conf,
            "style": {
                "color": cm.get_cmap('viridis')(VARYING_DTARGET_VALUES.index(d_target) / (len(VARYING_DTARGET_VALUES)-1)),
                "marker": ['o', 's', '^', 'D', 'v', 'p', '*', 'X'][VARYING_DTARGET_VALUES.index(d_target) % 8]
            }
        }

        for i in range(NUM_RUNS_PER_POINT):

            tasks.append((algo_name, SCENARIO_CONFIG.copy(),
                         algo_conf.copy(), i, BGCT))

    print(f"\n任务总数: {len(tasks)}")
    num_processes = max(1, multiprocessing.cpu_count() - 1)
    print(f"将使用 {num_processes} 个CPU核心并行执行...")

    analytics = SimulationAnalytics()
    start_time = time.time()

    with multiprocessing.Pool(processes=num_processes) as pool:
        results_iterator = pool.imap_unordered(run_single_task, tasks)
        for result_tuple in tqdm(results_iterator, total=len(tasks), desc=f"执行 [{EXPERIMENT_NAME}]"):
            analytics.add_run_result(*result_tuple)

    end_time = time.time()
    print(f"\n所有仿真任务执行完毕。总耗时: {end_time - start_time:.2f} 秒")

    print("\n正在处理和分析数据...")

    x_axis_key = 'd_target'

    analytics.save_to_csv(x_axis_key=x_axis_key, output_dir=OUTPUT_DIR)

    analytics.plot_results(
        x_axis_key=x_axis_key,
        algorithm_library=dynamic_algorithm_library,
        save_path=os.path.join(OUTPUT_DIR, f"{EXPERIMENT_NAME}_plot.png")
    )

    print("\n实验已全部完成。")
    print(f"所有结果（CSV文件和图表）已保存至目录: {OUTPUT_DIR}")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
