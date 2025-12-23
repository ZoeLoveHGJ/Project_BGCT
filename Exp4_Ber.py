
"""
Exp4_Ber.py: 误码率(BER)鲁棒性测试 (对应论文 Figure 8)

==============================================================================
 实验说明 (Figure 8 - BER Robustness)
==============================================================================
本实验旨在评估各算法在非理想信道下的鲁棒性，重点考察**误码率 (Bit Error Rate, BER)** 对系统识别效率、吞吐率和能耗的影响。

实验配置:
1.  **场景**: 标准随机分布 (id_distribution='random')。
    - 固定标签数量 (TOTAL_TAGS = 1000)，以控制变量，专注于信道噪声的影响。
2.  **变量**: 误码率 (ber)，范围 [0.0, 0.01]。
    - 模拟从理想信道 (0%) 到 高噪声信道 (1%) 的环境变化。
    - 注意：1% 的 BER 在 RFID 系统中已属于极其恶劣的工况。
3.  **信道**: 激活 NoisyChannel，配置 BER 变量，保持 Dropout 和 MissRate 为 0。
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


RESULTS_BASE_DIR = "results"

EXPERIMENTS_TO_RUN = [
    {
        "name": "Exp4_BER_Robustness",
        "description": "测试算法在不同误码率(BER)环境下的性能稳定性",

        "varying_param_key": "ber",
        "varying_param_values": [0, 0.00001, 0.00005, 0.0001, 0.0005, 0.0008, 0.001],

        "scenario_config": {
            'TOTAL_TAGS': 5000,
            'BINARY_LENGTH': 96,
            'id_distribution': 'random',
        },

        "algorithm_specific_config": {
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
    print(f"=== RFID防碰撞算法：BER 鲁棒性实验 (Figure 8) ===")
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
        print(f"变量范围: {experiment['varying_param_values']}")
        print("-" * 60)

        tasks = []
        varying_key = experiment['varying_param_key']

        for value in experiment['varying_param_values']:
            current_scenario = experiment['scenario_config'].copy()
            current_algo_config = experiment['algorithm_specific_config'].copy(
            )

            if varying_key in current_scenario:
                current_scenario[varying_key] = value
            else:
                current_algo_config[varying_key] = value

            for algo_name in ALGORITHMS_TO_TEST:
                for r in range(NUM_RUNS_PER_POINT):
                    tasks.append(
                        (algo_name, current_scenario, current_algo_config, r))

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
                    scenario_config=config_log,
                    algorithm_name=algo_name,
                    run_id=run_id,
                    algorithm_config=config_log
                )

        duration = time.time() - start_time
        print(f"\n耗时: {duration:.2f} 秒")

        print(f"正在生成图表 (X轴: {varying_key})...")

        analytics.save_to_csv(x_axis_key=varying_key, output_dir=output_dir)

        analytics.plot_results(
            x_axis_key=varying_key,
            algorithm_library=ALGORITHM_LIBRARY,
            save_path=os.path.join(output_dir, "Figure8_BER_Robustness.png")
        )

        print(f"实验完成，结果已保存至: {output_dir}\n")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
