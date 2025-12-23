# -*- coding: utf-8 -*-

# 1. 导入所有需要配置的算法实现类
from NLHQT import NLHQTAlgorithm
from BGVT import BGVT_Algorithm
from BGCT import BGCT
from BGVT_Final import BGVT_Final_Algorithm
from LAPCT import LAPCTAlgorithm
from DQTA import DQTAAlgorithm
# from HT_EEAC import HTEEABGVT_AlgorithmCAlgorithm # 代码有问题
from EMDT import EMDTAlgorithm
from EAQ_CBB import EAQCBBAlgorithm
from ICT import ICT_Algorithm
from SD_CGQT import SDCGQTAlgorithm
from SUBF_CGDFSA import SUBF_CGDFSA_Algorithm
from HT_EEAC import HT_EEAC
from FHS_RAC import FHS_RAC
from BGCT_Random import BGCT_RandomSelection
# 2. (新增) 定义一个全局的绘图样式库 (Plot Style Palette)
# 预定义10种不同的绘图风格，新增算法可直接通过索引引用。
PLOT_STYLE_PALETTE = [

    # Style 0: 专为 DL-PCT (主) 设计，最突出
    {"color": "purple", "linestyle": "-", "marker": "*", "linewidth": 2.5, "markersize": 10, "zorder": 10},
    # Style 1: 专为 DL-PCT (次) 设计
    {"color": "deeppink", "linestyle": "--", "marker": "p", "linewidth": 2.0},
    # Style 2:
    {"color": "red", "linestyle": "-", "marker": "o"},
    # Style 3:
    {"color": "green", "linestyle": "-.", "marker": "s"},
    # Style 4:
    {"color": "blue", "linestyle": ":", "marker": "x"},
    # Style 5:
    {"color": "darkorange", "linestyle": "--", "marker": "^"},
    # Style 6:
    {"color": "brown", "linestyle": "-", "marker": "d"},
    # Style 7:
    {"color": "cyan", "linestyle": ":", "marker": "+"},
    # Style 8:
    {"color": "olive", "linestyle": "-.", "marker": "v"},
    # Style 9:
    {"color": "gray", "linestyle": "--", "marker": "."},
]
"""
    'SD-CGQT': {
        "class": SDCGQTAlgorithm,
        "config": {},
        "year": 23,
        "style_id": 7,
    },
    'SUBF_CGDFSA': {
        "class": SUBF_CGDFSA_Algorithm,
        "config": {},
        "year": 24,
        "style_id": 7,
    },
    'ICT': {
        "class": ICT_Algorithm,
        "config": {},
        "year": 24,
        "style_id": 7,
    },
"""
# 3. 定义算法库 (ALGORITHM_LIBRARY)
#    现在每个算法通过 "style_id" 引用 PLOT_STYLE_PALETTE 中的样式。 3,4取值的效果最好
ALGORITHMS_TO_TEST = [
    'NLHQT(n=2)', # 要测试的版本
    'NLHQT(n=1)', # 要测试的版本
    'LAPCT', # 要测试的版本
    'DQTA(k_max=3)', # 成功复现但不测试的版本,太旧了
    'EMDT', # 要测试的版本
    'EAQ_CBB', # 要测试的版本
    'HT_EEAC',
    'FHS_RAC',
    # 'BGCT_Random',
    'BGCT', 

]
ALGORITHM_LIBRARY = {
    'BGCT': { # 性能非常好，但是没有比得上SD-CGQT
    "class": BGCT,
    "config": {}, # <--- 在这里配置
    "year": 25,
    "style_id": 0,
    },
    'LAPCT': {
        "class": LAPCTAlgorithm,
        "config": {'k_threshold_divisor': 3.0},
        "year": 24,
        "style_id": 1  # 引用3号风格
    },
    'NLHQT(n=2)': {
        "class": NLHQTAlgorithm, 
        "config": {'n_way': 2},
        "year": 23,
        "style_id": 2  # 引用2号风格
    },
    'NLHQT(n=1)': {
        "class": NLHQTAlgorithm, 
        "config": {'n_way': 1},
        "year": 23,
        "style_id": 3  # 引用2号风格
    },
    'DQTA(k_max=3)': {
        "class": DQTAAlgorithm,
        "config": {'k_max': 3},
        "year": 19,
        "style_id": 4,
    },
    'EMDT': {
        "class": EMDTAlgorithm,
        "config": {},
        "year": 24,
        "style_id": 5,
    },
    'EAQ_CBB': {
        "class": EAQCBBAlgorithm,
        "config": {'block_len': 8, 'map_len': 256},
        "year": 23,
        "style_id": 6,
    },
    'HT_EEAC': {
        "class": HT_EEAC,
        "config": {},
        "year": 24,
        "style_id": 7,
    },
    'FHS_RAC': {
        "class": FHS_RAC,
        "config": {},
        "year": 21,
        "style_id": 8,
    },
    'BGCT_Random': {
        "class": BGCT_RandomSelection,
        "config": {},
        "year": 25,
        "style_id": 9,
    },    
}


"""
    'BGVT(d=8)': {
    "class": BGVT_Algorithm,
    "config": {"d_max": 8}, # <--- 在这里配置
    "year": 25,
    "style_id": 0,
    },
    'BGVT_Final': { # 性能非常好，但是没有比得上SD-CGQT
    "class": BGVT_Final_Algorithm,
    "config": {}, # <--- 在这里配置
    "year": 25,
    "style_id": 9,
    },
"""

"""
    'DL-PCT-Final([2,2])': { 
        "class": DL_PCT_Final_Algorithm,
        # 核心配置在这里
        "config": {"prediction_level_bits": [2, 2, 2]},
        # 为其定义一个独特的样式，方便在图表中识别
        "style_id": 7,
    },
    'DL-PCT(d_max=5)': {
        "class": DL_PCTAlgorithm,
        "config": {'d_max': 5},
        "year": 25,
        "style_id": 5  # 引用1号风格
    },
    'DL-PCT(d_max=2)': {
        "class": DL_PCTAlgorithm,
        "config": {'d_max': 2},
        "year": 25,
        "style_id": 5  # 引用1号风格
    },
    'DL-PCT(d_max=6)': {
        "class": DL_PCTAlgorithm,
        "config": {'d_max': 6},
        "year": 25,
        "style_id": 7  # 引用1号风格
    },
    'DL-PCT(d_max=1)': {
        "class": DL_PCTAlgorithm,
        "config": {'d_max': 1},
        "year": 25,
        "style_id": 0  # 引用0号风格 (最突出)
    },
    'HT_EEAC': {
        "class": HTEEACAlgorithm,
        "config": {},
        "year": 24,
        "style_id": 5,
    },
    'DL-PCT(d_max=4)': {
        "class": DL_PCTAlgorithm,
        "config": {'d_max': 4},
        "year": 25,
        "style_id": 0  # 引用1号风格
    },
    'DL-PCT(d_max=3)': {
        "class": DL_PCTAlgorithm,
        "config": {'d_max': 3},
        "year": 25,
        "style_id": 1  # 引用1号风格
    },
    'DL-PCT-Improve(d_max=4)': {
        "class": DL_PCT_Improve_Algorithm,
        "config": {'d_max': 3},
        "year": 25,
        "style_id": 5  # 引用1号风格
    },
    'DL-PCT-Improve(d_max=4)': {
        "class": DL_PCT_Improve_Algorithm,
        "config": {'d_max': 4},
        "year": 25,
        "style_id": 6  # 引用1号风格
    },
    'DL-PCT-Final(d_max=12)': {
    "class": DL_PCT_Final_Algorithm,
    "config": {"d_max": 12}, # <--- 在这里配置
    "style_id": 8,
    },
"""
