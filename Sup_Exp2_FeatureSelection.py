
"""
Sup_Exp2_FeatureSelection.py: 验证特征选择策略的有效性

==============================================================================
 实验目标: 回应 Reviewer 3 Point 5
==============================================================================
对比 "Earliest-Position-First" (BGCT标准版) 与 "Random-Selection" (BGCT随机版)
的性能差异，证明现有设计的合理性。
"""

import os
import multiprocessing
import numpy as np
from tqdm import tqdm
from Framework import run_simulation
from Tool import SimulationAnalytics
from algorithm_base_config import ALGORITHM_LIBRARY


TARGET_ALGORITHMS = [
    'BGCT',
    'BGCT_Random'
]


SCENARIO_CONFIG = {
    'BINARY_LENGTH': 96,
    'id_distribution': 'random'
}


TAG_COUNTS = np.linspace(1000, 10000, 10, dtype=int)

RESULTS_DIR = "results/sup_feature_selection"
NUM_RUNS = 50


def run_task(args):
    algo_name, n_tags, run_id = args
    scenario = SCENARIO_CONFIG.copy()
    scenario['TOTAL_TAGS'] = int(n_tags)

    algo_info = ALGORITHM_LIBRARY[algo_name]

    algo_config = {**algo_info["config"], 'ber': 0.0, 'dropout_rate': 0.0}

    result = run_simulation(scenario, algo_info["class"], algo_config)

    return (result, {**scenario, **algo_config}, algo_name, run_id)


def main():
    print("=== 执行特征选择策略验证实验 ===")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    tasks = []
    for n in TAG_COUNTS:
        for algo in TARGET_ALGORITHMS:
            for r in range(NUM_RUNS):
                tasks.append((algo, n, r))

    analytics = SimulationAnalytics()

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()-1) as pool:
        for res in tqdm(pool.imap_unordered(run_task, tasks), total=len(tasks)):
            result_dict, config, algo_name, run_id = res
            analytics.add_run_result(result_dict, config, algo_name, run_id)

    print("生成图表...")
    analytics.save_to_csv(x_axis_key="TOTAL_TAGS", output_dir=RESULTS_DIR)
    analytics.plot_results(x_axis_key="TOTAL_TAGS", algorithm_library=ALGORITHM_LIBRARY,
                           save_path=os.path.join(RESULTS_DIR, "Feature_Selection_Validation.png"))


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
