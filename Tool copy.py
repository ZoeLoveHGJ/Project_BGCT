# -*- coding: utf-8 -*-
"""
RFID仿真分析与工具类 - V3.4 (Fix Metrics Calculation)

更新说明:
- V3.4: 修复 _calculate_derived_metrics 中缺失 'avg_tag_responses' 计算逻辑的问题。
        现在只要算法输出 'total_tag_responses'，Tool 就会自动计算平均值。
- V3.3: 适配物理层增强版，支持记录物理层干扰参数(X轴)。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Optional
import os
import math

# 使用清晰的科研绘图风格
plt.style.use('seaborn-v0_8-whitegrid')

class SimulationAnalytics:
    """
    仿真结果分析器。
    负责收集单次运行数据、聚合统计、导出CSV以及生成可视化图表。
    """
    def __init__(self):
        self.results_data = []

    def add_run_result(self, 
                       result_dict: Dict[str, Any], 
                       scenario_config: Dict, 
                       algorithm_name: str, 
                       run_id: int,
                       algorithm_config: Optional[Dict] = None):
        """
        收集单次仿真的运行结果。
        
        V3.3 更新:
        - 新增 `algorithm_config` 参数，用于捕获物理层配置 (如 guard_interval_us)。
        - 自动展平 'peak_metrics' 以支持资源监控绘图。
        """
        # 1. 基础信息合并
        record = {
            **scenario_config, 
            **result_dict, 
            'algorithm_name': algorithm_name, 
            'run_id': run_id
        }
        
        # 2. 【V3.3 新增】 提取物理层/算法特定配置作为潜在的分析维度 (X轴)
        # 这允许我们画出 "Time vs Guard Interval" 或 "Throughput vs Miss Rate" 的图
        if algorithm_config:
            # 只提取关键的物理层参数，避免污染数据表
            noise_keys = ['guard_interval_us', 'collision_miss_rate', 'dropout_rate', 'ber']
            for k in noise_keys:
                if k in algorithm_config:
                    record[k] = algorithm_config[k]

        # 3. 处理资源监控指标 (扁平化 peak_metrics)
        if 'peak_metrics' in record and isinstance(record['peak_metrics'], dict):
            for key, value in record['peak_metrics'].items():
                # 例如将 {'stack_depth': 10} 转换为 'peak_stack_depth' = 10
                record[f"peak_{key}"] = value
            del record['peak_metrics'] # 删除原始字典，避免 DataFrame 转换报错

        self.results_data.append(record)

    def get_results_dataframe(self) -> pd.DataFrame:
        """从收集的所有运行结果中创建一个 Pandas DataFrame。"""
        return pd.DataFrame(self.results_data) if self.results_data else pd.DataFrame()

    def save_to_csv(self, x_axis_key: str = 'TOTAL_TAGS', output_dir: str = "simulation_results"):
        """
        将聚合后的平均结果保存为多个 CSV 文件。
        支持按任意维度 (x_axis_key) 进行聚合，例如按 'collision_miss_rate' 聚合。
        """
        df = self.get_results_dataframe()
        if df.empty:
            print("警告: 没有仿真结果可供保存。")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        df = self._calculate_derived_metrics(df)
        
        # 定义需要导出的 KPI 及其 CSV 文件名映射
        kpi_map = {
            'total_protocol_time_ms': 'Total Time (ms)', 
            'throughput_tags_per_sec': 'Throughput (tags/sec)',
            'system_efficiency': 'System Efficiency', 
            'collision_slots': 'Collision Slots',
            'avg_query_efficiency': 'Average Query Efficiency', 
            'idle_slots': 'Idle Slots',
            'total_energy_uj': 'Total Energy (uJ)', # Framework V3.0 直接计算的精确能耗
            'avg_tag_bits': 'Avg Tag Bits Sent',
            'avg_reader_bits': 'Avg Reader Bits Sent', 
            'avg_tag_responses': 'Avg Tag Responses', # V3.4: 确保此映射存在
            'peak_stack_depth': 'Peak Stack Depth', # 资源监控指标
        }
        
        print(f"\n正在将结果保存到目录 '{output_dir}' 中...")
        
        # 检查 x_axis_key 是否存在 (防止用户指定了新的噪声参数但没在某些跑分中设置)
        if x_axis_key not in df.columns:
            print(f"警告: 指定的聚合键 '{x_axis_key}' 不在数据列中。跳过 CSV 保存。")
            print(f"可用列: {df.columns.tolist()}")
            return

        for key, title in kpi_map.items():
            if key in df.columns:
                try:
                    # 按 x轴 和 算法名称 分组，计算平均值
                    pivot_df = df.pivot_table(index=x_axis_key, columns='algorithm_name', values=key, aggfunc='mean').reset_index()
                    filename = os.path.join(output_dir, f"{key}.csv")
                    pivot_df.to_csv(filename, index=False, float_format='%.4f')
                    print(f" -> 已保存 {filename}")
                except Exception as e:
                    print(f"错误: 无法为 '{key}' 保存CSV。原因: {e}")

    def _calculate_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算框架未直接提供的衍生指标。
        """
        # 计算总时隙和系统效率
        if all(k in df.columns for k in ['success_slots', 'idle_slots', 'collision_slots']):
            df['total_slots'] = df['success_slots'] + df['idle_slots'] + df['collision_slots']
            df['system_efficiency'] = df.apply(lambda r: r['success_slots'] / r['total_slots'] if r['total_slots'] > 0 else 0, axis=1)
        
        # 计算平均比特数
        for col in ['total_tag_bits', 'total_reader_bits']:
            avg_col = f"avg_{col.split('_')[1]}_bits"
            if 'TOTAL_TAGS' in df.columns and col in df.columns:
                # 注意：如果是按 Miss Rate 分组，TOTAL_TAGS 可能是一个固定值
                df[avg_col] = df.apply(lambda r: r[col] / r['TOTAL_TAGS'] if r['TOTAL_TAGS'] > 0 else 0, axis=1)
        
        # [V3.4 新增] 计算平均响应次数 (avg_tag_responses)
        # 前提: 算法必须输出 'total_tag_responses' (由算法逻辑自行统计)
        if 'total_tag_responses' in df.columns and 'TOTAL_TAGS' in df.columns:
            df['avg_tag_responses'] = df.apply(
                lambda r: r['total_tag_responses'] / r['TOTAL_TAGS'] if r['TOTAL_TAGS'] > 0 else 0, 
                axis=1
            )
        
        # 计算平均查询效率
        if 'TOTAL_TAGS' in df.columns and 'collision_slots' in df.columns:
            df['avg_query_efficiency'] = df.apply(lambda r: r['TOTAL_TAGS'] / r['collision_slots'] if r['collision_slots'] > 0 else r['TOTAL_TAGS'], axis=1)
        
        # 转换时间单位 us -> ms
        if 'total_protocol_time_us' in df.columns:
            df['total_protocol_time_ms'] = df['total_protocol_time_us'] / 1000.0
            
        return df

    def plot_results(self, x_axis_key: str = 'TOTAL_TAGS', algorithm_library: Dict = None, save_path: str = None):
        """
        生成并保存包含所有性能指标的九宫格(或更多)图表。
        
        V3.3 提示:
        - 要画鲁棒性分析图 (如 Time vs Jitter)，请将 `x_axis_key` 设置为 
          'guard_interval_us', 'collision_miss_rate' 或 'dropout_rate'。
        """
        df = self.get_results_dataframe()
        if df.empty:
            print("警告: 没有仿真结果可供绘图。")
            return
        
        if x_axis_key not in df.columns:
            print(f"错误: 数据中不存在指定的X轴键值 '{x_axis_key}'。")
            return

        df = self._calculate_derived_metrics(df)
        algorithms = sorted(df['algorithm_name'].unique())
        
        # 定义绘图配置: Key -> {DataFrame列名, Y轴标签}
        kpi_config = {
            'Total Time (ms)': {'key': 'total_protocol_time_ms', 'ylabel': 'Time (ms)'},
            'Throughput (tags/sec)': {'key': 'throughput_tags_per_sec', 'ylabel': 'Tags / Second'},
            'Responses per Tag (avg)': {'key': 'avg_tag_responses', 'ylabel': 'Count'}, # V3.4: 确保能绘图
            'Collision Slots': {'key': 'collision_slots', 'ylabel': 'Slots'},
            'Average Query Efficiency': {'key': 'avg_query_efficiency', 'ylabel': 'Efficiency (Tags/Collision)'},
            'Identification Efficiency': {'key': 'system_efficiency', 'ylabel': 'Efficiency'},
            'Avg Reader Bits Sent': {'key': 'avg_reader_bits', 'ylabel': 'Bits'},
            'Avg Tag Bits Sent': {'key': 'avg_tag_bits', 'ylabel': 'Bits'},
            'Total Energy (uJ)': {'key': 'total_energy_uj', 'ylabel': 'Energy (uJ)'},
            'Peak Stack Depth': {'key': 'peak_stack_depth', 'ylabel': 'Stack Depth'},
        }

        # 筛选出当前数据中存在的 KPI
        available_kpis = {title: config for title, config in kpi_config.items() if config['key'] in df.columns}
        num_kpis = len(available_kpis)
        if num_kpis == 0:
            print("警告: 没有可绘制的KPI指标。")
            return

        nrows = math.ceil(num_kpis / 3)
        fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(26, 8 * nrows))
        # 确保 axes 是一个一维数组，即使只有一个子图
        if num_kpis == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for i, (title, config) in enumerate(available_kpis.items()):
            ax = axes[i]
            kpi_key = config['key']
            
            for algo_name in algorithms:
                algo_df = df[df['algorithm_name'] == algo_name]
                # 按 x_axis_key 分组计算平均值，确保曲线平滑
                grouped = algo_df.groupby(x_axis_key)[kpi_key].mean()
                
                plot_kwargs = {'marker': 'o', 'linestyle': '-', 'linewidth': 2.5, 'markersize': 8, 'alpha': 0.8}
                legend_label = algo_name
                
                # 应用自定义样式
                if algorithm_library and algo_name in algorithm_library:
                    algo_info = algorithm_library[algo_name]
                    if "style" in algo_info:
                        plot_kwargs.update(algo_info["style"])
                    if "year" in algo_info:
                        legend_label = f"{algo_name} ({algo_info['year']})"
                    # 如果配置文件中有 style_id，也可以在这里处理逻辑（这里假设 style 字典已经包含了绘图参数）
                
                ax.plot(grouped.index, grouped, label=legend_label, **plot_kwargs)

            ax.set_title(title, fontsize=20, fontweight='bold')
            
            # 设置 X 轴标签格式
            xlabel = x_axis_key.replace('_', ' ').title()
            # 为特定的物理层参数添加单位说明
            if x_axis_key == 'guard_interval_us': xlabel += ' (us)'
            if x_axis_key == 'collision_miss_rate': xlabel += ' (Rate)'
            
            ax.set_xlabel(xlabel, fontsize=16)
            ax.set_ylabel(config['ylabel'], fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.grid(True, which='both', linestyle='--', alpha=0.7)
            ax.legend(fontsize=12, frameon=True, framealpha=0.8, loc='best')
        
        # 移除多余的空子图
        for j in range(num_kpis, len(axes)):
            fig.delaxes(axes[j])
        
        fig.tight_layout(pad=4.0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n图表已保存到 {save_path}")
        
        # 注意: 在批量运行时可能不需要 plt.show()，可根据需要注释
        plt.show()

# ==============================================================================
# 4. 核心仿真工具类 (理想参考实现)
# ==============================================================================
class RfidUtils:
    """
    RFID仿真辅助工具类。
    
    【架构说明】: 
    此类实现了“理想信道”下的逻辑。
    在 V3.0 架构中，它仅作为静态参考、理论计算或单元测试使用。
    真实的仿真循环应使用 Framework 中注入的 `self.channel` 进行交互，
    以支持 Dropout、误检等非理想物理层特性。
    """
    @staticmethod
    def get_collision_info(tag_ids: List[str]) -> Tuple[str, List[int]]:
        """
        计算一组标签ID的公共前缀和所有碰撞位的位置 (理想逻辑)。

        Args:
            tag_ids (List[str]): 待分析的标签ID列表。

        Returns:
            Tuple[str, List[int]]: (公共前缀, 碰撞位索引列表)。
        """
        if not tag_ids:
            return '', []
        if len(tag_ids) == 1:
            return tag_ids[0], []
        
        # 确保所有ID长度相同，取最短的作为安全边界
        min_len = min(len(tid) for tid in tag_ids)
        
        # 计算公共前缀
        prefix = ""
        for i in range(min_len):
            first_bit = tag_ids[0][i]
            if all(tid[i] == first_bit for tid in tag_ids):
                prefix += first_bit
            else:
                break
        
        # 从公共前缀之后开始，查找所有发生碰撞的比特位
        collision_positions = []
        for i in range(len(prefix), min_len):
            bits_at_pos = {tid[i] for tid in tag_ids}
            if len(bits_at_pos) > 1:
                collision_positions.append(i)
                
        return prefix, collision_positions