# -*- coding: utf-8 -*-
"""
DL_PCT_Improve_Algorithm.py: DL-PCT 的增强版实现 (带位图预测)

本算法在 DL-PCT 的基础上，引入了“位图预测”机制，旨在消除大量的空闲时隙。

核心机制:
1.  三阶段状态机: PLANNING -> PREDICTION -> EXECUTING。
2.  PREDICTION 阶段: 读写器请求一个长度为 2^d 的复合位图，以预知哪些子时隙会有响应。
3.  EXECUTING 阶段: 只针对位图中标记为非空的子时隙进行查询，从而跳过所有空闲时隙。
4.  新的成本模型: 精确计算了位图预测本身带来的时间和能耗开销。
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set

# 假设这些类从您的框架文件中被正确导入
from Framework import TraditionalAlgorithmInterface, AlgorithmStepResult, Tag, CONSTANTS
from Tool import RfidUtils

@dataclass
class DLPCTImproveState:
    """
    用于在主堆栈中存储待处理的标签集合。
    """
    tags_to_identify: List[Tag] = field(default_factory=list)

class DL_PCT_Improve_Algorithm(TraditionalAlgorithmInterface):
    """
    DL-PCT 的增强版实现，通过位图预测消除空闲时隙。
    """
    def __init__(self, tags_in_field: List[Tag], **kwargs):
        super().__init__(tags_in_field)
        self.d_max = kwargs.get('d_max', 8)
        initial_state = DLPCTImproveState(tags_to_identify=self.tags_in_field)
        self.query_stack: List[DLPCTImproveState] = [initial_state]
        self.id_length = len(tags_in_field[0].id) if tags_in_field else 0
        self.tag_response_counts: Dict[str, int] = {t.id: 0 for t in tags_in_field}

        # --- 状态机核心变量 ---
        self.current_mode = 'PLANNING_PHASE'
        
        # --- 用于在各阶段传递任务信息的变量 ---
        self.planned_groups: List[List[Tag]] = []
        self.non_idle_groups: List[List[Tag]] = [] # 只存储非空闲的分组
        self.sub_slot_cursor: int = 0
        self.current_split_prefix_len: int = 0
        self.current_d_to_use: int = 0
        
        self.pending_cmd_reader_bits: float = 0.0

    def is_finished(self) -> bool:
        finished = len(self.identified_tags) == len(self.tags_in_field)
        if finished and 'avg_tag_responses' not in self.metrics:
            counts = list(self.tag_response_counts.values())
            if counts:
                self.metrics['avg_tag_responses'] = np.mean(counts)
                self.metrics['min_tag_responses'] = np.min(counts)
                self.metrics['max_tag_responses'] = np.max(counts)
            else:
                self.metrics.update({'avg_tag_responses': 0, 'min_tag_responses': 0, 'max_tag_responses': 0})
        return finished

    def perform_step(self) -> AlgorithmStepResult:
        # ======================================================================
        # 状态一: 规划阶段 - 分析碰撞，准备发起预测请求
        # ======================================================================
        if self.current_mode == 'PLANNING_PHASE':
            if not self.query_stack:
                return AlgorithmStepResult('internal_op', operation_description="Finished")

            current_state = self.query_stack.pop(0)
            tags_in_this_state = [t for t in current_state.tags_to_identify if t.id not in self.identified_tags]
            num_tags = len(tags_in_this_state)

            if num_tags <= 1:
                if num_tags == 1:
                    tag = tags_in_this_state[0]
                    self.identified_tags.add(tag.id)
                    self.metrics['success_slots'] += 1
                    self.tag_response_counts[tag.id] += 1
                    reader_bits = CONSTANTS.READER_CMD_BASE_BITS
                    return AlgorithmStepResult('success_slot', reader_bits, self.id_length, self.id_length)
                else:
                    return AlgorithmStepResult('internal_op', operation_description="Empty state popped")

            # --- 复杂碰撞，进入规划 ---
            tag_ids = [t.id for t in tags_in_this_state]
            common_prefix, collision_positions = RfidUtils.get_collision_info(tag_ids)
            d_detected = len(collision_positions)
            d_to_use = min(d_detected, self.d_max)
            if d_to_use == 0:
                collision_positions = [len(common_prefix)]
                d_to_use = 1
            positions_to_use = collision_positions[:d_to_use]

            num_sub_slots = 2**d_to_use
            groups = [[] for _ in range(num_sub_slots)]
            for tag in tags_in_this_state:
                feature_vector = "".join([tag.id[i] for i in positions_to_use])
                group_index = int(feature_vector, 2)
                groups[group_index].append(tag)

            # 保存规划结果
            self.planned_groups = groups
            self.current_d_to_use = d_to_use
            self.current_split_prefix_len = len(common_prefix)
            
            # 计算并暂存 QueryDL-Predict 指令的开销
            bits_for_d = 5
            bits_per_pos = 7
            self.pending_cmd_reader_bits = CONSTANTS.READER_CMD_BASE_BITS + len(common_prefix) + bits_for_d + d_to_use * bits_per_pos

            # 切换到预测阶段
            self.current_mode = 'PREDICTION_PHASE'
            
            for tag in tags_in_this_state:
                self.tag_response_counts[tag.id] += 1

            return AlgorithmStepResult('internal_op', 
                                       operation_description=f"Planned QueryDL-Predict, d={d_to_use}")

        # ======================================================================
        # 状态二: 预测阶段 - 接收复合位图，筛选非空时隙
        # ======================================================================
        elif self.current_mode == 'PREDICTION_PHASE':
            # 筛选出所有非空闲的分组
            self.non_idle_groups = [group for group in self.planned_groups if group]
            
            # 如果预测后发现所有时隙都是空的（理论上不太可能，但作为保护），则直接结束这一轮
            if not self.non_idle_groups:
                self.current_mode = 'PLANNING_PHASE'
                return AlgorithmStepResult('internal_op', operation_description="Prediction resulted in all idle slots")

            # 切换到执行阶段
            self.current_mode = 'EXECUTING_PHASE'
            self.sub_slot_cursor = 0
            
            # --- 返回位图传输本身的开销 ---
            # 这是增强方案引入的核心成本
            bitmap_len = 2**self.current_d_to_use
            
            # 附加之前暂存的指令开销
            reader_bits_cost = self.pending_cmd_reader_bits
            self.pending_cmd_reader_bits = 0.0

            # 我们将这个事件模拟成一个特殊的“碰撞时隙”，因为有多个标签在贡献位图
            # 它的时间由位图的长度决定
            return AlgorithmStepResult('collision_slot',
                                       reader_bits=reader_bits_cost,
                                       tag_bits=bitmap_len, # 所有标签逻辑或后的位图长度
                                       expected_max_tag_bits=bitmap_len,
                                       operation_description=f"Received prediction bitmap (len={bitmap_len})")

        # ======================================================================
        # 状态三: 执行阶段 - 只处理非空时隙
        # ======================================================================
        elif self.current_mode == 'EXECUTING_PHASE':
            # 从【非空闲】分组列表中取出当前要处理的
            current_group = self.non_idle_groups[self.sub_slot_cursor]
            num_tags_in_group = len(current_group)
            self.sub_slot_cursor += 1

            expected_max_tag_bits = self.id_length - self.current_split_prefix_len
            result = None

            # 在这里，num_tags_in_group 不可能为 0
            if num_tags_in_group == 1:
                tag = current_group[0]
                self.identified_tags.add(tag.id)
                self.metrics['success_slots'] += 1
                actual_tag_bits = expected_max_tag_bits
                result = AlgorithmStepResult('success_slot',
                                             tag_bits=actual_tag_bits,
                                             expected_max_tag_bits=expected_max_tag_bits)
            else: # > 1, 残余碰撞
                self.query_stack.insert(0, DLPCTImproveState(tags_to_identify=current_group))
                self.metrics['collision_slots'] += 1
                total_collision_bits = num_tags_in_group * expected_max_tag_bits
                result = AlgorithmStepResult('collision_slot',
                                             tag_bits=total_collision_bits,
                                             expected_max_tag_bits=expected_max_tag_bits)

            # 如果这是最后一个非空闲时隙，则结束本轮分裂
            if self.sub_slot_cursor == len(self.non_idle_groups):
                self.current_mode = 'PLANNING_PHASE'
                self.non_idle_groups = []

            return result

        # 兜底
        return AlgorithmStepResult('internal_op', operation_description="Error: Unknown state")
