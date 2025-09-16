

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
import os
import math


plt.style.use('seaborn-v0_8-whitegrid')


class SimulationAnalytics:
    def __init__(self):
        self.results_data = []

    def add_run_result(self, result_dict: Dict[str, Any], scenario_config: Dict, algorithm_name: str, run_id: int):

        record = {**scenario_config, **result_dict,
                  'algorithm_name': algorithm_name, 'run_id': run_id}
        if 'peak_metrics' in record and isinstance(record['peak_metrics'], dict):
            for key, value in record['peak_metrics'].items():

                record[f"peak_{key}"] = value
            del record['peak_metrics']
        self.results_data.append(record)

    def get_results_dataframe(self) -> pd.DataFrame:
        """从收集的所有运行结果中创建一个Pandas DataFrame。"""
        return pd.DataFrame(self.results_data) if self.results_data else pd.DataFrame()

    def save_to_csv(self, x_axis_key: str = 'TOTAL_TAGS', output_dir: str = "simulation_results"):
        df = self.get_results_dataframe()
        if df.empty:
            print("警告: 没有仿真结果可供保存。")
            return

        os.makedirs(output_dir, exist_ok=True)
        df = self._calculate_derived_metrics(df)

        kpi_map = {
            'total_protocol_time_ms': 'Total Time (ms)',
            'throughput_tags_per_sec': 'Throughput (tags/sec)',
            'system_efficiency': 'System Efficiency',
            'collision_slots': 'Collision Slots',
            'avg_query_efficiency': 'Average Query Efficiency',
            'idle_slots': 'Idle Slots',
            'total_energy_uj': 'Total Energy (uJ)',
            'avg_tag_bits': 'Avg Tag Bits Sent',
            'avg_reader_bits': 'Avg Reader Bits Sent',
            'avg_tag_responses': 'Avg Tag Responses',

            'peak_stack_depth': 'Peak Stack Depth',
        }

        print(f"\n正在将结果保存到目录 '{output_dir}' 中...")
        for key, title in kpi_map.items():
            if key in df.columns:
                try:

                    pivot_df = df.pivot_table(
                        index=x_axis_key, columns='algorithm_name', values=key, aggfunc='mean').reset_index()
                    filename = os.path.join(output_dir, f"{key}.csv")
                    pivot_df.to_csv(filename, index=False, float_format='%.4f')
                    print(f" -> 已保存 {filename}")
                except Exception as e:
                    print(f"错误: 无法为 '{key}' 保存CSV。原因: {e}")

    def _calculate_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算框架未直接提供的衍生指标。
        V2 修改: 移除了重复的能耗计算。
        """

        if all(k in df.columns for k in ['success_slots', 'idle_slots', 'collision_slots']):
            df['total_slots'] = df['success_slots'] + \
                df['idle_slots'] + df['collision_slots']
            df['system_efficiency'] = df.apply(
                lambda r: r['success_slots'] / r['total_slots'] if r['total_slots'] > 0 else 0, axis=1)

        for col in ['total_tag_bits', 'total_reader_bits']:
            avg_col = f"avg_{col.split('_')[1]}_bits"
            if 'TOTAL_TAGS' in df.columns and col in df.columns:
                df[avg_col] = df.apply(
                    lambda r: r[col] / r['TOTAL_TAGS'] if r['TOTAL_TAGS'] > 0 else 0, axis=1)

        if 'TOTAL_TAGS' in df.columns and 'collision_slots' in df.columns:
            df['avg_query_efficiency'] = df.apply(
                lambda r: r['TOTAL_TAGS'] / r['collision_slots'] if r['collision_slots'] > 0 else r['TOTAL_TAGS'], axis=1)

        if 'total_protocol_time_us' in df.columns:
            df['total_protocol_time_ms'] = df['total_protocol_time_us'] / 1000.0
        return df

    def plot_results(self, x_axis_key: str = 'TOTAL_TAGS', algorithm_library: Dict = None, save_path: str = None):
        df = self.get_results_dataframe()
        if df.empty:
            print("警告: 没有仿真结果可供绘图。")
            return

        df = self._calculate_derived_metrics(df)
        algorithms = sorted(df['algorithm_name'].unique())

        kpi_config = {
            'Total Time (ms)': {'key': 'total_protocol_time_ms', 'ylabel': 'Time (ms)'},
            'Throughput (tags/sec)': {'key': 'throughput_tags_per_sec', 'ylabel': 'Tags / Second'},
            'Responses per Tag (avg)': {'key': 'avg_tag_responses', 'ylabel': 'Count'},
            'Collision Slots': {'key': 'collision_slots', 'ylabel': 'Slots'},
            'Average Query Efficiency': {'key': 'avg_query_efficiency', 'ylabel': 'Efficiency (Tags/Collision)'},
            'Identification Efficiency': {'key': 'system_efficiency', 'ylabel': 'Efficiency'},
            'Avg Reader Bits Sent': {'key': 'avg_reader_bits', 'ylabel': 'Bits'},
            'Avg Tag Bits Sent': {'key': 'avg_tag_bits', 'ylabel': 'Bits'},
            'Total Energy (uJ)': {'key': 'total_energy_uj', 'ylabel': 'Energy (uJ)'},

            'Peak Stack Depth': {'key': 'peak_stack_depth', 'ylabel': 'Stack Depth'},
        }

        available_kpis = {title: config for title,
                          config in kpi_config.items() if config['key'] in df.columns}
        num_kpis = len(available_kpis)
        nrows = math.ceil(num_kpis / 3)
        fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(26, 8 * nrows))
        axes = axes.flatten()

        for i, (title, config) in enumerate(available_kpis.items()):
            ax = axes[i]
            kpi_key = config['key']

            for algo_name in algorithms:
                algo_df = df[df['algorithm_name'] == algo_name]
                grouped = algo_df.groupby(x_axis_key)[kpi_key].mean()

                plot_kwargs = {'marker': 'o', 'linestyle': '-',
                               'linewidth': 2.5, 'markersize': 8, 'alpha': 0.8}
                legend_label = algo_name
                if algorithm_library and algo_name in algorithm_library:
                    algo_info = algorithm_library[algo_name]
                    if "style" in algo_info:
                        plot_kwargs.update(algo_info["style"])
                    if "year" in algo_info:
                        legend_label = f"{algo_name} ({algo_info['year']})"

                ax.plot(grouped.index, grouped,
                        label=legend_label, **plot_kwargs)

            ax.set_title(title, fontsize=20, fontweight='bold')
            ax.set_xlabel(x_axis_key.replace('_', ' ').title(), fontsize=16)
            ax.set_ylabel(config['ylabel'], fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.grid(True, which='both', linestyle='--', alpha=0.7)
            ax.legend(fontsize=12, frameon=True, framealpha=0.8, loc='best')

        for j in range(num_kpis, len(axes)):
            fig.delaxes(axes[j])

        fig.tight_layout(pad=4.0)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n图表已保存到 {save_path}")

        plt.show()


class RfidUtils:
    @staticmethod
    def get_collision_info(tag_ids: List[str]) -> Tuple[str, List[int]]:
        if not tag_ids:
            return '', []
        if len(tag_ids) == 1:
            return tag_ids[0], []

        min_len = min(len(tid) for tid in tag_ids)

        prefix = ""
        for i in range(min_len):
            first_bit = tag_ids[0][i]
            if all(tid[i] == first_bit for tid in tag_ids):
                prefix += first_bit
            else:
                break

        collision_positions = []
        for i in range(len(prefix), min_len):
            bits_at_pos = {tid[i] for tid in tag_ids}
            if len(bits_at_pos) > 1:
                collision_positions.append(i)

        return prefix, collision_positions
