
import os
import time
import multiprocessing
import numpy as np
import pandas as pd
from tqdm import tqdm

from Framework import run_simulation
from algorithm_base_config import ALGORITHM_LIBRARY, ALGORITHMS_TO_TEST
from Tool import SimulationAnalytics

RESULTS_BASE_DIR = "results"

EXPERIMENTS_TO_RUN = [
    {
        "name": "exp_dispersed_collision_vs_performance",
        "description": "测试在非连续、分散碰撞场景下的算法性能",
        "varying_param_key": "TOTAL_TAGS",
        "varying_param_values": np.linspace(1000, 5000, 10, dtype=int),
        "scenario_config": {
            'BINARY_LENGTH': 96,
            'id_distribution': 'dispersed',
        },
        "algorithm_specific_config": {
            'ber': 0.0,
            'enable_refined_energy_model': True, 
            'enable_resource_monitoring': True,
        }
    },
]


NUM_RUNS_PER_POINT = 5


def run_single_task(task_params: tuple):
    algo_name, scenario_config, algo_specific_config, run_id = task_params

    algo_info = ALGORITHM_LIBRARY[algo_name]
    algo_class = algo_info["class"]

    final_algo_config = {**algo_info["config"], **algo_specific_config}

    result_dict = run_simulation(
        scenario_config=scenario_config,
        algorithm_class=algo_class,
        algorithm_specific_config=final_algo_config
    )

    full_config_log = {**scenario_config, **algo_specific_config}

    return (result_dict, full_config_log, algo_name, run_id)


def main():

    for experiment in EXPERIMENTS_TO_RUN:
        exp_name = experiment["name"]
        output_dir = os.path.join(RESULTS_BASE_DIR, exp_name)
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "="*80)
        print(f"开始执行实验: {exp_name}")
        print(f"描述: {experiment['description']}")
        print(f"对比算法: {', '.join(ALGORITHMS_TO_TEST)}")
        print(f"可变参数: '{experiment['varying_param_key']}'")
        print(f"参数范围: {str(experiment['varying_param_values'])}")
        print(f"每个数据点重复运行: {NUM_RUNS_PER_POINT} 次")
        print("="*80)

        tasks = []
        varying_key = experiment['varying_param_key']

        for value in experiment['varying_param_values']:
            if varying_key in experiment['scenario_config'] or varying_key in ['TOTAL_TAGS', 'BINARY_LENGTH', 'id_distribution']:
                scenario_conf = experiment['scenario_config'].copy()
                scenario_conf[varying_key] = value
                algo_conf = experiment['algorithm_specific_config'].copy()
            else:
                scenario_conf = experiment['scenario_config'].copy()
                algo_conf = experiment['algorithm_specific_config'].copy()
                algo_conf[varying_key] = value

            for algo_name in ALGORITHMS_TO_TEST:
                for i in range(NUM_RUNS_PER_POINT):
                    tasks.append(
                        (algo_name, scenario_conf.copy(), algo_conf.copy(), i))

        print(f"\n任务总数: {len(tasks)}")
        num_processes = max(1, multiprocessing.cpu_count() - 1)
        print(f"将使用 {num_processes} 个CPU核心并行执行...")

        analytics = SimulationAnalytics()
        start_time = time.time()

        with multiprocessing.Pool(processes=num_processes) as pool:
            results_iterator = pool.imap_unordered(run_single_task, tasks)
            for result_tuple in tqdm(results_iterator, total=len(tasks), desc=f"执行 [{exp_name}]"):
                analytics.add_run_result(*result_tuple)

        end_time = time.time()
        print(f"\n实验 [{exp_name}] 执行完毕。总耗时: {end_time - start_time:.2f} 秒")

        print("\n正在处理和分析数据...")

        analytics.save_to_csv(x_axis_key=varying_key, output_dir=output_dir)
        analytics.plot_results(
            x_axis_key=varying_key,
            algorithm_library=ALGORITHM_LIBRARY,
            save_path=os.path.join(output_dir, f"{exp_name}_plot.png")
        )

        print(f"\n实验 [{exp_name}] 已全部完成。")
        print(f"所有结果已保存至目录: {output_dir}")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
