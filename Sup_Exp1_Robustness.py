
"""
Sup_Exp1_Robustness.py: 针对审稿人意见的物理层鲁棒性验证实验

==============================================================================
 实验目标
==============================================================================
本脚本专门用于生成 "Major Revision" 回复信中所需的抗干扰性能数据。
它不再改变标签数量，而是改变物理层信道参数，以验证算法的鲁棒性。

包含三个核心子实验：
1.  **Robustness-Jitter**: 测试时间抖动 (Guard Interval) 对识别时间的影响。
    - 对应 Reviewer 1 关于 "Propagation delays" 的质疑。
    
2.  **Robustness-MissDetection**: 测试碰撞漏检 (Miss Rate) 对系统效率的影响。
    - 对应 Reviewer 3 关于 "Reliability of collision detection" 的质疑。

3.  **Robustness-Dropout**: 测试标签掉线 (Dropout Rate) 对吞吐率的影响。
    - 对应 Reviewer 1 关于 "Realistic operational conditions" 的要求。
==============================================================================
"""

import os
import time
import multiprocessing
import numpy as np
from tqdm import tqdm


from Framework import run_simulation
from Tool import SimulationAnalytics


from algorithm_base_config import ALGORITHM_LIBRARY


TARGET_ALGORITHMS = [
    'BGCT',
    'DQTA(k_max=3)',
    'EMDT',
    'LAPCT',
    'NLHQT(n=2)',
    'NLHQT(n=1)',
    'EAQ_CBB'
]


BASE_SCENARIO = {
    'TOTAL_TAGS': 5000,
    'BINARY_LENGTH': 96,
    'id_distribution': 'random'
}


RESULTS_DIR = "results/sup_robustness"


ROBUSTNESS_EXPERIMENTS = [

    {
        "name": "Exp_R1_Timing_Jitter",
        "description": "分析引入物理层保护间隔(Guard Interval)后的性能衰减",
        "varying_param_key": "guard_interval_us",

        "varying_param_values": np.linspace(0, 20, 5),
        "base_algo_config": {
            'ber': 0.0,
            'collision_miss_rate': 0.0,
            'dropout_rate': 0.0
        }
    },

    {
        "name": "Exp_R2_Sensing_Error",
        "description": "分析阅读器未能检测到碰撞(Missed Detection)时的系统稳定性",
        "varying_param_key": "collision_miss_rate",

        "varying_param_values": np.linspace(0, 0.20, 5),
        "base_algo_config": {
            'guard_interval_us': 0.0,
            'ber': 0.0,
            'dropout_rate': 0.0
        }
    },

    {
        "name": "Exp_R3_Tag_Dropout",
        "description": "分析信道不稳定导致标签概率性不响应(Dropout)时的吞吐率",
        "varying_param_key": "dropout_rate",

        "varying_param_values": np.linspace(0, 0.1, 5),
        "base_algo_config": {
            'guard_interval_us': 0.0,
            'collision_miss_rate': 0.0,
            'ber': 0.0
        }
    }
]


NUM_RUNS = 5


def run_robustness_task(task_args):
    """单次仿真任务封装"""
    algo_name, scenario_conf, algo_conf, run_id = task_args

    algo_info = ALGORITHM_LIBRARY[algo_name]
    algo_class = algo_info["class"]

    final_config = {**algo_info["config"], **algo_conf}

    result = run_simulation(scenario_conf, algo_class, final_config)

    full_config_log = {**scenario_conf, **final_config}

    return (result, full_config_log, algo_name, run_id)


def main():
    print(f"=== 开始执行鲁棒性验证实验 ")
    print(f"对比算法: {TARGET_ALGORITHMS}")
    print(f"基础场景: {BASE_SCENARIO}")
    print(f"保存目录: {RESULTS_DIR}\n")

    for exp in ROBUSTNESS_EXPERIMENTS:
        exp_name = exp["name"]
        var_key = exp["varying_param_key"]
        var_values = exp["varying_param_values"]
        base_algo_conf = exp["base_algo_config"]

        output_path = os.path.join(RESULTS_DIR, exp_name)
        os.makedirs(output_path, exist_ok=True)

        print(f"\n>>> 正在运行实验: {exp_name}")
        print(f"    变量: {var_key} -> {var_values}")

        tasks = []
        for val in var_values:

            current_algo_conf = base_algo_conf.copy()
            current_algo_conf[var_key] = val

            for algo in TARGET_ALGORITHMS:
                for r in range(NUM_RUNS):
                    tasks.append(
                        (algo, BASE_SCENARIO.copy(), current_algo_conf, r))

        analytics = SimulationAnalytics()
        num_cores = max(1, multiprocessing.cpu_count() - 2)

        with multiprocessing.Pool(num_cores) as pool:

            for res_tuple in tqdm(pool.imap_unordered(run_robustness_task, tasks),
                                  total=len(tasks),
                                  desc=f"Computing {exp_name}"):

                result_dict, config_log, algo_name, run_id = res_tuple

                analytics.add_run_result(
                    result_dict, BASE_SCENARIO, algo_name, run_id, algorithm_config=config_log)

        print(f"    正在生成图表...")

        analytics.save_to_csv(x_axis_key=var_key, output_dir=output_path)

        analytics.plot_results(
            x_axis_key=var_key,
            algorithm_library=ALGORITHM_LIBRARY,
            save_path=os.path.join(
                output_path, f"{exp_name}_robustness_plot.png")
        )
        print(f"    {exp_name} 完成。")

    print("\n所有鲁棒性实验执行完毕。")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
