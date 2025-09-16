# -*- coding: utf-8 -*-
from NLHQT import NLHQTAlgorithm
from BGVT import BGVT_Algorithm
from BGCT import BGCT
from BGVT_Final import BGVT_Final_Algorithm
from LAPCT import LAPCTAlgorithm
from DQTA import DQTAAlgorithm
from EMDT import EMDTAlgorithm
from EAQ_CBB import EAQCBBAlgorithm
from ICT import ICT_Algorithm
from SD_CGQT import SDCGQTAlgorithm
from SUBF_CGDFSA import SUBF_CGDFSA_Algorithm
from HT_EEAC import HT_EEAC
from FHS_RAC import FHS_RAC
from BGCT_Random import BGCT_RandomSelection
PLOT_STYLE_PALETTE = [


    {"color": "purple", "linestyle": "-", "marker": "*", "linewidth": 2.5, "markersize": 10, "zorder": 10},

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
ALGORITHMS_TO_TEST = [
    'NLHQT(n=2)', # 要测试的版本
    'NLHQT(n=1)', # 要测试的版本
    'LAPCT', # 要测试的版本
    'DQTA(k_max=3)', # 成功复现但不测试的版本,太旧了
    'EMDT', # 要测试的版本
    'EAQ_CBB', # 要测试的版本
    'BGCT', 
    'HT_EEAC',
    'FHS_RAC',
    # 'BGCT_Random',
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
        "config": {},
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