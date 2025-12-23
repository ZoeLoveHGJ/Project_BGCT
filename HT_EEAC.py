import random
import math
from typing import List, Dict, Optional


try:
    from scipy.special import lambertw
except ImportError:
    raise ImportError("HT-EEAC error")


from Framework import TraditionalAlgorithmInterface, AlgorithmStepResult, Tag, CONSTANTS


class HT_EEAC(TraditionalAlgorithmInterface):
    def __init__(self, tags_in_field: List[Tag], **kwargs):
        super().__init__(tags_in_field, **kwargs)

        self.q = 4
        self.frame_size = 2**self.q
        self.slot_counter = 0
        self.unidentified_tags = list(self.tags_in_field)

        self.frame_ni = 0
        self.frame_ns = 0
        self.frame_nk = 0

        self.tags_in_slot: Dict[int, List[Tag]] = {}

        self.energy_threshold = 17.5
        self.checkpoint_divisor = 16

        self.r_factor = self._calculate_r()

        self.metrics['total_tag_responses'] = 0

        self._start_new_frame()

    def is_finished(self) -> bool:
        """当所有标签都已被识别时，仿真结束。"""
        return len(self.unidentified_tags) == 0

    def perform_step(self) -> AlgorithmStepResult:
        """
        执行一个时隙的仿真。
        """
        if self.is_finished():
            return AlgorithmStepResult(operation_type='internal_op')

        candidates = self.tags_in_slot.get(self.slot_counter, [])

        active_tags = self.channel.filter_active_tags(candidates)

        self.metrics['total_tag_responses'] += len(active_tags)

        current_internal_metrics = {
            'unidentified_tags_count': len(self.unidentified_tags)}

        signal_str, collision_indices = self.channel.resolve_collision(
            active_tags)

        step_result = None

        if not active_tags:
            self.metrics['idle_slots'] += 1
            self.frame_ni += 1
            step_result = AlgorithmStepResult(
                operation_type='idle_slot',
                reader_bits=CONSTANTS.READER_CMD_BASE_BITS,
                operation_description=f"Frame 2^{self.q}, Slot {self.slot_counter}: Idle",
                internal_metrics=current_internal_metrics
            )

        elif len(collision_indices) == 0:

            if active_tags:
                identified_tag = active_tags[0]

                if identified_tag in self.unidentified_tags:
                    self.identified_tags.add(identified_tag.id)
                    self.unidentified_tags.remove(identified_tag)

                    current_internal_metrics['unidentified_tags_count'] = len(
                        self.unidentified_tags)

                self.metrics['success_slots'] += 1
                self.frame_ns += 1

                tag_response_bits = CONSTANTS.RN16_RESPONSE_BITS + \
                    len(signal_str)

                step_result = AlgorithmStepResult(
                    operation_type='success_slot',
                    reader_bits=CONSTANTS.READER_CMD_BASE_BITS,
                    tag_bits=tag_response_bits,
                    expected_max_tag_bits=tag_response_bits,
                    operation_description=f"Frame 2^{self.q}, Slot {self.slot_counter}: Success",
                    internal_metrics=current_internal_metrics
                )
            else:

                step_result = AlgorithmStepResult(operation_type='internal_op')

        else:
            self.metrics['collision_slots'] += 1
            self.frame_nk += 1
            step_result = AlgorithmStepResult(
                operation_type='collision_slot',
                reader_bits=CONSTANTS.READER_CMD_BASE_BITS,
                tag_bits=CONSTANTS.RN16_RESPONSE_BITS,
                expected_max_tag_bits=CONSTANTS.RN16_RESPONSE_BITS,
                operation_description=f"Frame 2^{self.q}, Slot {self.slot_counter}: Collision",
                internal_metrics=current_internal_metrics
            )

        self.slot_counter += 1

        checkpoint_slot = self.frame_size / self.checkpoint_divisor
        is_checkpoint = (checkpoint_slot >=
                         1 and self.slot_counter == int(checkpoint_slot))
        is_frame_end = (self.slot_counter >= self.frame_size)

        if (is_checkpoint or is_frame_end) and not self.is_finished():

            e_per_ns = self._calculate_e_per_ns()

            should_terminate = False
            if is_frame_end:
                should_terminate = True
            elif e_per_ns is not None and e_per_ns > self.energy_threshold:

                should_terminate = True

            if should_terminate:

                estimated_total = self._mmse_estimate(
                    self.frame_ni, self.frame_ns, self.frame_nk, self.frame_size)

                n_next = max(0, estimated_total - self.frame_ns)

                if n_next > 0 and self.r_factor > 0:
                    log_arg = n_next / self.r_factor

                    next_q = round(math.log2(log_arg)) if log_arg > 0 else 0
                    self.q = max(0, min(15, next_q))
                else:

                    self.q = 2 if self.unidentified_tags else 0

                self.frame_size = 2**self.q
                self._start_new_frame()

        return step_result

    def _start_new_frame(self):
        """初始化一个新帧，重置计数器并让未识别标签随机选择时隙。"""
        self.slot_counter = 0
        self.frame_ni, self.frame_ns, self.frame_nk = 0, 0, 0
        self.tags_in_slot.clear()

        if self.is_finished():
            return

        if self.frame_size == 0:

            self.frame_size = 1

        for tag in self.unidentified_tags:
            chosen_slot = random.randint(0, self.frame_size - 1)
            if chosen_slot not in self.tags_in_slot:
                self.tags_in_slot[chosen_slot] = []
            self.tags_in_slot[chosen_slot].append(tag)

    def _get_slot_durations_us(self):
        """使用框架常量计算HT-EEAC论文中定义的时隙时长"""

        ti_us = CONSTANTS.T1_US * 2

        ts_us = (CONSTANTS.READER_CMD_BASE_BITS / CONSTANTS.READER_BITS_PER_US +
                 CONSTANTS.T1_US +
                 (CONSTANTS.RN16_RESPONSE_BITS + CONSTANTS.EPC_CODE_BITS) / CONSTANTS.TAG_BITS_PER_US +
                 CONSTANTS.T2_MIN_US)

        tk_us = (CONSTANTS.READER_CMD_BASE_BITS / CONSTANTS.READER_BITS_PER_US +
                 CONSTANTS.T1_US +
                 CONSTANTS.RN16_RESPONSE_BITS / CONSTANTS.TAG_BITS_PER_US +
                 CONSTANTS.T2_MIN_US)
        return ti_us, ts_us, tk_us

    def _calculate_r(self) -> float:
        """根据论文 Eq. (12) 和框架常量计算 r 因子。"""
        ti_us, ts_us, tk_us = self._get_slot_durations_us()
        tepc_us = CONSTANTS.EPC_CODE_BITS / CONSTANTS.TAG_BITS_PER_US

        pr_s_mw = 125.0
        ptx_mw = 825.0

        numerator = ptx_mw * (ti_us - tk_us) - pr_s_mw * tepc_us
        denominator = math.e * (ptx_mw * tk_us + pr_s_mw * tepc_us)

        if denominator == 0:
            return 1.0

        w_val = lambertw(numerator / denominator).real
        return 1 + w_val

    def _calculate_e_per_ns(self) -> Optional[float]:
        """根据论文 Eq. (6-8) 和框架常量计算 E/Ns 指标。"""
        if self.frame_ns == 0:
            return None

        ti_us, ts_us, tk_us = self._get_slot_durations_us()
        trn16_us = CONSTANTS.RN16_RESPONSE_BITS / CONSTANTS.TAG_BITS_PER_US
        tepc_us = CONSTANTS.EPC_CODE_BITS / CONSTANTS.TAG_BITS_PER_US

        pr_s_mw = 125.0
        ptx_mw = 825.0

        ns, nk, ni = self.frame_ns, self.frame_nk, self.frame_ni

        et = ptx_mw * (ns * ts_us + nk * tk_us + ni * ti_us) / 1000.0
        er = pr_s_mw * (ns * (trn16_us + tepc_us) + nk * trn16_us) / 1000.0

        total_energy = et + er
        return total_energy / ns

    def _mmse_estimate(self, ni: int, ns: int, nk: int, L: int) -> int:
        """根据论文 Eq. (14) 实现 MMSE 标签数量估算。"""
        min_error = float('inf')

        best_n = ns + 2 * nk
        search_range = 100

        search_start = max(0, best_n - search_range)
        search_end = best_n + search_range

        for n_hat in range(search_start, search_end + 1):
            if L <= 1:
                continue

            try:

                p_idle = (1 - 1/L)**n_hat if n_hat > 0 else 1.0
                p_success = n_hat * (1/L) * ((1 - 1/L) **
                                             (n_hat - 1)) if n_hat > 0 else 0.0

                exp_ni = L * p_idle
                exp_ns = L * p_success
                exp_nk = L - exp_ni - exp_ns
            except (ValueError, OverflowError):
                continue

            error = (exp_ni - ni)**2 + (exp_ns - ns)**2 + (exp_nk - nk)**2

            if error < min_error:
                min_error = error
                best_n = n_hat

        return best_n
