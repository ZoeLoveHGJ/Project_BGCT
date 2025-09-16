# -*- coding: utf-8 -*-
"""
DL-PCT (Deep-Locking Parallel Collision Tree) 算法的实现。 最开始完成的版本存档

*** V1.3 - 实现标签响应次数指标 ***

该版本核心升级:
1.  增加了对每个标签响应次数的精确追踪。
2.  在仿真结束时，会自动计算响应次数的平均、最小和最大值，并作为metrics返回。
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set

# 导入框架和工具类
from Framework import TraditionalAlgorithmInterface, AlgorithmStepResult, Tag, CONSTANTS
from Tool import RfidUtils

@dataclass
class DLPCTState:
    tags_to_identify: List[Tag] = field(default_factory=list)

class DL_PCTAlgorithm(TraditionalAlgorithmInterface):
    def __init__(self, tags_in_field: List[Tag], **kwargs):
        super().__init__(tags_in_field)
        self.d_max = kwargs.get('d_max', 8)
        initial_state = DLPCTState(tags_to_identify=self.tags_in_field)
        self.query_stack: List[DLPCTState] = [initial_state]
        self.id_length = len(tags_in_field[0].id) if tags_in_field else 0
        # (*** 已新增 ***) 初始化标签响应计数器
        self.tag_response_counts: Dict[str, int] = {t.id: 0 for t in tags_in_field}

    def is_finished(self) -> bool:
        finished = len(self.identified_tags) == len(self.tags_in_field)
        # (*** 已新增 ***) 在仿真结束前，计算并记录最终指标
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
        if not self.query_stack:
            return AlgorithmStepResult(operation_type='internal_op')

        current_state = self.query_stack.pop(0)
        tags_in_this_state = [t for t in current_state.tags_to_identify if t.id not in self.identified_tags]
        num_tags = len(tags_in_this_state)

        if num_tags == 0:
            return AlgorithmStepResult(operation_type='internal_op')

        # (*** 已新增 ***) 所有参与本轮查询的标签，其响应次数都+1
        for tag in tags_in_this_state:
            self.tag_response_counts[tag.id] += 1

        current_prefix, _ = RfidUtils.get_collision_info([t.id for t in tags_in_this_state])

        if num_tags == 1:
            tag = tags_in_this_state[0]
            self.identified_tags.add(tag.id)
            self.metrics['success_slots'] += 1
            reader_bits = CONSTANTS.READER_CMD_BASE_BITS + len(current_prefix)
            actual_tag_bits = self.id_length - len(current_prefix)
            return AlgorithmStepResult('success_slot', reader_bits, actual_tag_bits, actual_tag_bits)
        
        self.metrics['collision_slots'] += 1
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

        total_bits_sent_by_all_responding_tags = 0
        for group_tags in groups:
            bits_per_tag_response = self.id_length - len(common_prefix)
            if not group_tags:
                self.metrics['idle_slots'] += 1
            elif len(group_tags) == 1:
                self.metrics['success_slots'] += 1
                successful_tag = group_tags[0]
                self.identified_tags.add(successful_tag.id)
                total_bits_sent_by_all_responding_tags += bits_per_tag_response
            else:
                self.query_stack.insert(0, DLPCTState(tags_to_identify=group_tags))
                total_bits_sent_by_all_responding_tags += len(group_tags) * bits_per_tag_response
        
        bits_for_d = 5
        bits_per_pos = 7
        reader_bits = CONSTANTS.READER_CMD_BASE_BITS + len(common_prefix) + bits_for_d + d_to_use * bits_per_pos
        expected_max_tag_bits = self.id_length - len(common_prefix)
        
        return AlgorithmStepResult(
            operation_type='collision_slot',
            reader_bits=reader_bits,
            tag_bits=total_bits_sent_by_all_responding_tags,
            expected_max_tag_bits=expected_max_tag_bits,
            operation_description=f"DL-PCT-{num_sub_slots}-way split"
        )
