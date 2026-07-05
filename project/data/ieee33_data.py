"""IEEE 33 节点 12.66 kV 辐射型配电网与设备接入数据。

本文件只定义静态数据，不运行潮流。pandapower 建网逻辑在
``project.networks.power_network`` 中读取这些表。节点编号采用 Python
习惯的 0-based 编号，因此经典 IEEE33 的 1 号母线在代码里是 bus=0。
"""

from __future__ import annotations

from typing import Tuple

from project.config import ESSConfig, GFGConfig, P2GConfig, RenewableConfig


# from_bus, to_bus, r_ohm, x_ohm。线路长度在建模时取 1 km，
# 因此 r/x 直接作为 ohm_per_km 写入 pandapower。
IEEE33_LINE_DATA: Tuple[Tuple[int, int, float, float], ...] = (
    (0, 1, 0.0922, 0.0470),
    (1, 2, 0.4930, 0.2511),
    (2, 3, 0.3660, 0.1864),
    (3, 4, 0.3811, 0.1941),
    (4, 5, 0.8190, 0.7070),
    (5, 6, 0.1872, 0.6188),
    (6, 7, 0.7114, 0.2351),
    (7, 8, 1.0300, 0.7400),
    (8, 9, 1.0440, 0.7400),
    (9, 10, 0.1966, 0.0650),
    (10, 11, 0.3744, 0.1238),
    (11, 12, 1.4680, 1.1550),
    (12, 13, 0.5416, 0.7129),
    (13, 14, 0.5910, 0.5260),
    (14, 15, 0.7463, 0.5450),
    (15, 16, 1.2890, 1.7210),
    (16, 17, 0.7320, 0.5740),
    (1, 18, 0.1640, 0.1565),
    (18, 19, 1.5042, 1.3554),
    (19, 20, 0.4095, 0.4784),
    (20, 21, 0.7089, 0.9373),
    (2, 22, 0.4512, 0.3083),
    (22, 23, 0.8980, 0.7091),
    (23, 24, 0.8960, 0.7011),
    (5, 25, 0.2030, 0.1034),
    (25, 26, 0.2842, 0.1447),
    (26, 27, 1.0590, 0.9337),
    (27, 28, 0.8042, 0.7006),
    (28, 29, 0.5075, 0.2585),
    (29, 30, 0.9744, 0.9630),
    (30, 31, 0.3105, 0.3619),
    (31, 32, 0.3410, 0.5302),
)


# bus, p_mw, q_mvar
IEEE33_LOAD_DATA: Tuple[Tuple[int, float, float], ...] = (
    (1, 0.100, 0.060),
    (2, 0.090, 0.040),
    (3, 0.120, 0.080),
    (4, 0.060, 0.030),
    (5, 0.060, 0.020),
    (6, 0.200, 0.100),
    (7, 0.200, 0.100),
    (8, 0.060, 0.020),
    (9, 0.060, 0.020),
    (10, 0.045, 0.030),
    (11, 0.060, 0.035),
    (12, 0.060, 0.035),
    (13, 0.120, 0.080),
    (14, 0.060, 0.010),
    (15, 0.060, 0.020),
    (16, 0.060, 0.020),
    (17, 0.090, 0.040),
    (18, 0.090, 0.040),
    (19, 0.090, 0.040),
    (20, 0.090, 0.040),
    (21, 0.090, 0.040),
    (22, 0.090, 0.050),
    (23, 0.420, 0.200),
    (24, 0.420, 0.200),
    (25, 0.060, 0.025),
    (26, 0.060, 0.025),
    (27, 0.060, 0.020),
    (28, 0.120, 0.070),
    (29, 0.200, 0.600),
    (30, 0.150, 0.070),
    (31, 0.210, 0.100),
    (32, 0.060, 0.040),
)


ESS_CONFIGS: Tuple[ESSConfig, ...] = (
    # p_mw>0 表示充电，p_mw<0 表示放电；环境会根据 SOC 做安全投影。
    ESSConfig("ESS_0", bus=6, max_p_mw=1.0, capacity_mwh=4.0, eta_charge=0.92, eta_discharge=0.92),
    ESSConfig("ESS_1", bus=15, max_p_mw=1.0, capacity_mwh=4.0, eta_charge=0.92, eta_discharge=0.92),
    ESSConfig("ESS_2", bus=29, max_p_mw=0.5, capacity_mwh=2.0, eta_charge=0.90, eta_discharge=0.90),
)


RENEWABLE_CONFIGS: Tuple[RenewableConfig, ...] = (
    # 每台新能源设备的 Actor 动作有两维：无功 Q 和有功削减比例。
    RenewableConfig("PV_0", bus=9, kind="pv", capacity_mw=1.0, s_rated_mva=1.08),
    RenewableConfig("PV_1", bus=13, kind="pv", capacity_mw=1.0, s_rated_mva=1.08),
    RenewableConfig("WT_0", bus=17, kind="wind", capacity_mw=1.5, s_rated_mva=1.60),
    RenewableConfig("WT_1", bus=20, kind="wind", capacity_mw=1.5, s_rated_mva=1.60),
    RenewableConfig("PV_2", bus=23, kind="pv", capacity_mw=1.0, s_rated_mva=1.08),
    RenewableConfig("PV_3", bus=24, kind="pv", capacity_mw=1.0, s_rated_mva=1.08),
    RenewableConfig("WT_2", bus=31, kind="wind", capacity_mw=1.0, s_rated_mva=1.05),
    RenewableConfig("PV_4", bus=32, kind="pv", capacity_mw=1.0, s_rated_mva=1.08),
)


# 这些电侧接入点沿用原 IEEE 33 配网，不使用 Excel 中 IEEE 39 的电网参数。
GFG_CONFIGS: Tuple[GFGConfig, ...] = (
    # GFG 在电侧表现为发电机，在气侧表现为天然气 sink。
    GFGConfig("GFG_0", power_bus=18, gas_junction=4, max_p_mw=2.0, efficiency=0.38),
    GFGConfig("GFG_1", power_bus=22, gas_junction=5, max_p_mw=2.0, efficiency=0.38),
    GFGConfig("GFG_2", power_bus=32, gas_junction=18, max_p_mw=1.5, efficiency=0.36),
)


P2G_CONFIGS: Tuple[P2GConfig, ...] = (
    # P2G 在电侧表现为负荷，在气侧表现为天然气 source。
    P2GConfig("P2G_0", power_bus=7, gas_junction=7, max_p_mw=1.5, efficiency=0.70),
    P2GConfig("P2G_1", power_bus=24, gas_junction=14, max_p_mw=1.5, efficiency=0.70),
    P2GConfig("P2G_2", power_bus=30, gas_junction=19, max_p_mw=1.0, efficiency=0.65),
)


N_POWER_BUSES = 33
