import os
import random
import time
from typing import Dict, List, Tuple, Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# =============================================================================
# 文件: car_logistics_env.py
# 作者: [Neo]
# 日期: 2023-10-27
# 描述: 基于Gymnasium的汽车物流模拟环境，用于强化学习决策。
#       模拟汽车从工厂、仓库到客户和经销商的调度与运输过程，目标是最小化总成本。
# =============================================================================

# --- 常量定义 ---
NUM_WEEKS_PER_YEAR: int = 52  # 每年模拟的周数

# 车型定义
CAR_TYPES: List[str] = ['High', 'Mid', 'Low']
NUM_CAR_TYPES: int = len(CAR_TYPES)

# 地点定义
FACTORY: str = "Factory"  # 工厂
NUM_TEMP_WAREHOUSES: int = 6  # 临时仓库数量
WAREHOUSES: List[str] = [f"WH_{i}" for i in range(NUM_TEMP_WAREHOUSES)]

NUM_LARGE_CUSTOMERS: int = 3  # 大客户数量
LARGE_CUSTOMERS: List[str] = [f"LC_{i}" for i in range(NUM_LARGE_CUSTOMERS)]

NUM_4S_STORES: int = 5  # 4S店数量 (通常只销售高端车)
S4_STORES: List[str] = [f"4S_{i}" for i in range(NUM_4S_STORES)]

NUM_DEALERS: int = 8  # 经销商数量 (通常销售中低端车)
DEALERS: List[str] = [f"Dealer_{i}" for i in range(NUM_DEALERS)]

ALL_LOCATIONS: List[str] = [FACTORY] + WAREHOUSES + LARGE_CUSTOMERS + S4_STORES + DEALERS

# 成本与容量
TRANSPORT_COST_PER_KM: Dict[str, float] = {  # 每公里每辆车的运输成本
    'High': 3.0,
    'Mid': 2.0,
    'Low': 1.5
}

FACTORY_PROD_RANGES: Dict[str, Tuple[int, int]] = {  # 工厂各车型周生产范围
    'High': (30, 80),
    'Mid': (60, 150),
    'Low': (70, 180)
}

LC_DEMAND_RANGES: Dict[str, Tuple[int, int]] = {ct: (1, 6) for ct in CAR_TYPES}  # 大客户需求范围
S4_DEMAND_RANGES: Dict[str, Tuple[int, int]] = {'High': (2, 10)}  # 4S店需求范围 (只销售High型车)
DEALER_DEMAND_RANGES: Dict[str, Tuple[int, int]] = {'Mid': (3, 15), 'Low': (3, 15)}  # 经销商需求范围 (销售Mid, Low型车)

MAX_WAREHOUSE_CAPACITY_PER_TYPE: int = 500  # 单个仓库单车型的最大库存容量（用于观测值归一化）
MAX_PRODUCTION_PER_TYPE: int = max(val[1] for val in FACTORY_PROD_RANGES.values()) + 20  # 最大生产量（用于观测值归一化）
MAX_DEMAND_PER_ENTITY_PER_TYPE: int = 20  # 单个客户或经销商单车型的最大需求量（用于观测值归一化）

UNMET_DEMAND_PENALTY: int = 100000  # 未满足需求惩罚


class CarLogisticsEnv(gym.Env):
    """
    汽车物流模拟环境，用于强化学习决策。

    环境模拟了汽车从工厂、仓库到不同类型客户（大客户、4S店、经销商）的
    生产、库存和调度过程。代理的目标是最小化总运输成本和未满足需求惩罚。
    """
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(self, seed: Optional[int] = None):
        """
        初始化汽车物流环境。

        Args:
            seed (int, optional): 随机种子，用于复现环境状态。
        """
        super().__init__()
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)

        self.current_week: int = 0
        self.distances: Dict[Tuple[str, str], int] = self._generate_distances()
        self.car_type_to_idx: Dict[str, int] = {name: i for i, name in enumerate(CAR_TYPES)}

        # 观测空间定义
        # 组成：
        # - 工厂当前生产量 (3: High, Mid, Low)
        # - 各仓库库存量 (6个仓库 * 3种车型 = 18)
        # - 大客户需求量 (3个客户 * 3种车型 = 9)
        # - 4S店需求量 (5个店 * 1种车型 = 5)
        # - 经销商需求量 (8个经销商 * 2种车型 = 16)
        # 总计: 3 + 18 + 9 + 5 + 16 = 51
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(51,), dtype=np.float32
        )

        # 动作空间定义
        # 每个需求点（大客户、4S店、经销商）的每种车型，代理选择来源（0: 工厂, 1-6: WH_0到WH_5）
        # - 大客户 (3 * 3 = 9个决策)
        # - 4S店 (5 * 1 = 5个决策)
        # - 经销商 (8 * 2 = 16个决策)
        # 生产过剩的决策：工厂生产过剩的每种车型，代理选择一个仓库存储 (6个仓库选项)
        # - 生产过剩 (3个决策)
        # 总计：9 + 5 + 16 + 3 = 33 个决策
        # 每个决策的可能值范围是0到 NUM_TEMP_WAREHOUSES (0代表工厂, 1-6代表仓库)
        # 生产过剩决策的可能值范围是0到 NUM_TEMP_WAREHOUSES-1 (0-5代表仓库)

        num_demand_points_decisions = (NUM_LARGE_CUSTOMERS * NUM_CAR_TYPES) + \
                                      (NUM_4S_STORES * 1) + \
                                      (NUM_DEALERS * 2)

        num_source_options = 1 + NUM_TEMP_WAREHOUSES  # 0: Factory, 1-6: WH_0 to WH_5

        demand_action_dims = [num_source_options] * num_demand_points_decisions

        num_excess_prod_decisions = NUM_CAR_TYPES  # 每个车型一个决策
        excess_prod_action_dims = [NUM_TEMP_WAREHOUSES] * num_excess_prod_decisions  # 0-5 for warehouses

        self.action_space = spaces.MultiDiscrete(demand_action_dims + excess_prod_action_dims)

        # 状态变量初始化
        self.warehouse_inventory: np.ndarray = np.zeros((NUM_TEMP_WAREHOUSES, NUM_CAR_TYPES), dtype=int)
        self.current_production: Dict[str, int] = {}  # 工厂当周生产量
        self.current_demands: Dict[str, Any] = {}  # 当周客户需求

    def _generate_distances(self) -> Dict[Tuple[str, str], int]:
        """
        生成所有地点之间的随机距离。
        工厂到仓库距离较近 (50-200km)，其他地点之间距离较远 (100-1000km)。

        Returns:
            Dict[Tuple[str, str], int]: 地点对到距离的映射。
        """
        distances: Dict[Tuple[str, str], int] = {}
        for i in range(len(ALL_LOCATIONS)):
            for j in range(i + 1, len(ALL_LOCATIONS)):
                loc1, loc2 = ALL_LOCATIONS[i], ALL_LOCATIONS[j]
                # 特殊处理工厂到仓库的距离
                is_fac_wh = (loc1 == FACTORY and loc2.startswith("WH")) or \
                            (loc2 == FACTORY and loc1.startswith("WH"))
                dist_val = random.randint(50, 200) if is_fac_wh else random.randint(100, 1000)
                distances[(loc1, loc2)] = dist_val
                distances[(loc2, loc1)] = dist_val
            # 同一地点距离为0
            distances[(ALL_LOCATIONS[i], ALL_LOCATIONS[i])] = 0
        return distances

    def get_distance(self, loc1_name: str, loc2_name: str) -> int:
        """
        获取两个地点之间的距离。

        Args:
            loc1_name (str): 地点1的名称。
            loc2_name (str): 地点2的名称。

        Returns:
            int: 两个地点之间的距离（公里）。如果距离不存在，返回无穷大。
        """
        return self.distances.get((loc1_name, loc2_name), float('inf'))

    def _generate_production(self) -> Dict[str, int]:
        """
        生成当周工厂生产量。

        Returns:
            Dict[str, int]: 各车型当周生产量。
        """
        production: Dict[str, int] = {}
        for car_type in CAR_TYPES:
            min_p, max_p = FACTORY_PROD_RANGES[car_type]
            production[car_type] = random.randint(min_p, max_p)
        return production

    def _generate_demands(self) -> Dict[str, Any]:
        """
        生成当周各类型客户的随机需求。

        Returns:
            Dict[str, Any]: 包含LC, 4S, Dealer客户需求的字典。
        """
        demands: Dict[str, Any] = {
            'LC': [{} for _ in range(NUM_LARGE_CUSTOMERS)],
            '4S': [{} for _ in range(NUM_4S_STORES)],
            'Dealer': [{} for _ in range(NUM_DEALERS)]
        }

        # 大客户需求 (所有车型)
        for i in range(NUM_LARGE_CUSTOMERS):
            for car_type in CAR_TYPES:
                min_d, max_d = LC_DEMAND_RANGES[car_type]
                demands['LC'][i][car_type] = random.randint(min_d, max_d)

        # 4S店需求 (只针对'High'型车)
        for i in range(NUM_4S_STORES):
            min_d, max_d = S4_DEMAND_RANGES['High']
            demands['4S'][i]['High'] = random.randint(min_d, max_d)

        # 经销商需求 (只针对'Mid', 'Low'型车)
        for i in range(NUM_DEALERS):
            for car_type in ['Mid', 'Low']:
                min_d, max_d = DEALER_DEMAND_RANGES[car_type]
                demands['Dealer'][i][car_type] = random.randint(min_d, max_d)
        return demands

    def _get_obs(self) -> np.ndarray:
        """
        根据当前环境状态生成观测值。
        所有观测值会被归一化到 [0, 1] 范围。

        Returns:
            np.ndarray: 归一化后的观测值数组。
        """
        obs: List[float] = []

        # 1. 工厂当前生产量
        for car_type in CAR_TYPES:
            obs.append(self.current_production.get(car_type, 0) / MAX_PRODUCTION_PER_TYPE)

        # 2. 各仓库库存量
        for i in range(NUM_TEMP_WAREHOUSES):
            for j in range(NUM_CAR_TYPES):
                obs.append(self.warehouse_inventory[i, j] / MAX_WAREHOUSE_CAPACITY_PER_TYPE)

        # 3. 大客户需求量
        for i in range(NUM_LARGE_CUSTOMERS):
            for car_type in CAR_TYPES:
                obs.append(self.current_demands['LC'][i].get(car_type, 0) / MAX_DEMAND_PER_ENTITY_PER_TYPE)

        # 4. 4S店需求量 (仅'High'型车)
        for i in range(NUM_4S_STORES):
            obs.append(self.current_demands['4S'][i].get('High', 0) / MAX_DEMAND_PER_ENTITY_PER_TYPE)

        # 5. 经销商需求量 (仅'Mid', 'Low'型车)
        for i in range(NUM_DEALERS):
            for car_type in ['Mid', 'Low']:
                obs.append(self.current_demands['Dealer'][i].get(car_type, 0) / MAX_DEMAND_PER_ENTITY_PER_TYPE)

        # 确保所有值都在 [0, 1] 范围内
        return np.array(obs, dtype=np.float32).clip(0, 1)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        重置环境到初始状态。

        Args:
            seed (int, optional): 重置时的随机种子。
            options (Dict, optional): 额外选项。

        Returns:
            Tuple[np.ndarray, Dict]: 初始观测值和信息字典。
        """
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.current_week = 0
        self.warehouse_inventory = np.zeros((NUM_TEMP_WAREHOUSES, NUM_CAR_TYPES), dtype=int)
        self.current_production = self._generate_production()
        self.current_demands = self._generate_demands()

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def _get_info(self) -> Dict[str, Any]:
        """
        获取当前环境的辅助信息。

        Returns:
            Dict[str, Any]: 包含当前周、库存、生产和需求的字典。
        """
        return {
            "current_week": self.current_week + 1,  # 周从1开始计数
            "warehouse_inventory": self.warehouse_inventory.copy(),
            "factory_production": self.current_production.copy(),
            "demands": self.current_demands.copy(),
        }

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行一个动作，推进环境一个时间步。

        Args:
            action (np.ndarray): 代理的动作数组。

        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict]:
                - observation (np.ndarray): 新的观测值。
                - reward (float): 奖励值。
                - terminated (bool): 是否达到终止状态。
                - truncated (bool): 是否因时间限制而终止（未完全完成）。
                - info (Dict): 包含额外信息的字典。
        """
        week_for_log = self.current_week + 1
        total_transport_cost = 0.0
        unmet_demand_this_step = 0

        # 在本周内用于计算的库存和生产量，防止直接修改self.变量
        current_factory_stock_for_step = self.current_production.copy()
        current_wh_stock_for_step = self.warehouse_inventory.copy()

        weekly_dispatch_records: List[Dict[str, Any]] = []

        # --- 1. 解析动作并构建需求履行任务列表 ---
        demand_fulfillment_tasks: List[Dict[str, Any]] = []
        action_idx = 0

        # Large Customers (LC)
        for lc_idx in range(NUM_LARGE_CUSTOMERS):
            for car_type in CAR_TYPES:
                demand_qty = self.current_demands['LC'][lc_idx].get(car_type, 0)
                if demand_qty > 0:
                    source_choice = action[action_idx]
                    demand_fulfillment_tasks.append({
                        'qty': demand_qty,
                        'car_type': car_type,
                        'ct_idx': self.car_type_to_idx[car_type],
                        'source_choice': source_choice,
                        'dest_name': LARGE_CUSTOMERS[lc_idx],
                        'original_demand_qty': demand_qty  # 用于后续判断是否需回退
                    })
                action_idx += 1

        # 4S Stores (High Car Type Only)
        for s4_idx in range(NUM_4S_STORES):
            demand_qty = self.current_demands['4S'][s4_idx].get('High', 0)
            if demand_qty > 0:
                source_choice = action[action_idx]
                demand_fulfillment_tasks.append({
                    'qty': demand_qty,
                    'car_type': 'High',
                    'ct_idx': self.car_type_to_idx['High'],
                    'source_choice': source_choice,
                    'dest_name': S4_STORES[s4_idx],
                    'original_demand_qty': demand_qty
                })
            action_idx += 1

        # Dealers (Mid and Low Car Types Only)
        for dlr_idx in range(NUM_DEALERS):
            for car_type in ['Mid', 'Low']:
                demand_qty = self.current_demands['Dealer'][dlr_idx].get(car_type, 0)
                if demand_qty > 0:
                    source_choice = action[action_idx]
                    demand_fulfillment_tasks.append({
                        'qty': demand_qty,
                        'car_type': car_type,
                        'ct_idx': self.car_type_to_idx[car_type],
                        'source_choice': source_choice,
                        'dest_name': DEALERS[dlr_idx],
                        'original_demand_qty': demand_qty
                    })
                action_idx += 1

        # --- 2. 尝试满足每个需求任务 (首次调度和回退逻辑) ---
        for task in demand_fulfillment_tasks:
            qty_to_fulfill = task['qty']
            car_type = task['car_type']
            ct_idx = task['ct_idx']
            source_choice_agent = task['source_choice']
            dest_name = task['dest_name']
            fulfilled_this_task = 0

            # 定义一个内部函数，用于从特定来源尝试发货
            def _try_dispatch(source_name: str, current_stock: Dict[str, int] | np.ndarray,
                              target_qty: int, is_factory: bool, wh_idx: Optional[int] = None) -> Tuple[int, float]:
                """尝试从给定来源发货，并返回实际发货量和运输成本。"""
                actual_shipped = 0
                cost_incurred = 0.0

                if is_factory:
                    available = current_stock[car_type]  # type: ignore
                else:
                    available = current_stock[wh_idx, ct_idx]  # type: ignore

                can_ship = min(target_qty, available)
                if can_ship > 0:
                    if is_factory:
                        current_stock[car_type] -= can_ship  # type: ignore
                    else:
                        current_stock[wh_idx, ct_idx] -= can_ship  # type: ignore

                    cost_incurred = can_ship * self.get_distance(source_name, dest_name) * TRANSPORT_COST_PER_KM[
                        car_type]
                    actual_shipped = can_ship
                return actual_shipped, cost_incurred

            # 首次尝试从代理选择的来源发货
            source_name_primary: str
            if source_choice_agent == 0:  # 工厂
                source_name_primary = FACTORY
                shipped, cost = _try_dispatch(source_name_primary, current_factory_stock_for_step, qty_to_fulfill, True)
                reason_prefix = "Demand (Agent: Factory)"
            else:  # 仓库
                wh_idx_primary = source_choice_agent - 1
                source_name_primary = WAREHOUSES[wh_idx_primary]
                shipped, cost = _try_dispatch(source_name_primary, current_wh_stock_for_step, qty_to_fulfill, False,
                                              wh_idx_primary)
                reason_prefix = f"Demand (Agent: WH_{wh_idx_primary})"

            if shipped > 0:
                total_transport_cost += cost
                fulfilled_this_task += shipped
                weekly_dispatch_records.append({
                    'week': week_for_log, 'type': 'demand_fulfillment', 'car_type': car_type,
                    'quantity': shipped, 'source': source_name_primary, 'destination': dest_name,
                    'reason': reason_prefix, 'cost': round(cost, 2)
                })

            remaining_to_fulfill = qty_to_fulfill - fulfilled_this_task

            # 回退逻辑：如果代理选择的来源不足，尝试从其他来源补足
            if remaining_to_fulfill > 0:
                # 尝试从工厂补足 (如果代理选择的不是工厂，或工厂还有剩余)
                if current_factory_stock_for_step[car_type] > 0:
                    shipped_fallback_fac, cost_fallback_fac = _try_dispatch(FACTORY, current_factory_stock_for_step,
                                                                            remaining_to_fulfill, True)
                    if shipped_fallback_fac > 0:
                        total_transport_cost += cost_fallback_fac
                        fulfilled_this_task += shipped_fallback_fac
                        remaining_to_fulfill -= shipped_fallback_fac
                        weekly_dispatch_records.append({
                            'week': week_for_log, 'type': 'demand_fallback', 'car_type': car_type,
                            'quantity': shipped_fallback_fac, 'source': FACTORY, 'destination': dest_name,
                            'reason': "Demand (Fallback: Factory)", 'cost': round(cost_fallback_fac, 2)
                        })

            if remaining_to_fulfill > 0:
                # 尝试从其他仓库补足 (按成本排序，优先选择成本低的仓库)
                candidate_whs = []
                for wh_alt_idx_cand in range(NUM_TEMP_WAREHOUSES):
                    if current_wh_stock_for_step[wh_alt_idx_cand, ct_idx] > 0:
                        cost = self.get_distance(WAREHOUSES[wh_alt_idx_cand], dest_name) * TRANSPORT_COST_PER_KM[
                            car_type]
                        candidate_whs.append({'idx': wh_alt_idx_cand, 'cost': cost,
                                              'stock': current_wh_stock_for_step[wh_alt_idx_cand, ct_idx]})
                candidate_whs.sort(key=lambda x: x['cost'])  # 按运输成本升序排序

                for wh_cand in candidate_whs:
                    if remaining_to_fulfill == 0:
                        break  # 已满足所有需求

                    wh_alt_idx = wh_cand['idx']
                    source_name_alt = WAREHOUSES[wh_alt_idx]

                    shipped_fallback_wh, cost_fallback_wh = _try_dispatch(source_name_alt, current_wh_stock_for_step,
                                                                          remaining_to_fulfill, False, wh_alt_idx)

                    if shipped_fallback_wh > 0:
                        total_transport_cost += cost_fallback_wh
                        fulfilled_this_task += shipped_fallback_wh
                        remaining_to_fulfill -= shipped_fallback_wh
                        weekly_dispatch_records.append({
                            'week': week_for_log, 'type': 'demand_fallback', 'car_type': car_type,
                            'quantity': shipped_fallback_wh, 'source': source_name_alt, 'destination': dest_name,
                            'reason': f"Demand (Fallback: WH_{wh_alt_idx})", 'cost': round(cost_fallback_wh, 2)
                        })

            # 记录未满足的需求
            if remaining_to_fulfill > 0:
                unmet_demand_this_step += remaining_to_fulfill
                weekly_dispatch_records.append({
                    'week': week_for_log, 'type': 'unmet_demand', 'car_type': car_type,
                    'quantity': remaining_to_fulfill, 'source': 'N/A', 'destination': dest_name,
                    'reason': "Unmet Demand", 'cost': 0.0  # 未满足需求无运输成本
                })

        # 更新环境的仓库库存（已考虑本周调度消耗）
        self.warehouse_inventory = current_wh_stock_for_step.copy()

        # --- 3. 处理工厂生产过剩的库存调度 ---
        excess_prod_actions = action[action_idx:]  # 动作数组的剩余部分用于处理生产过剩

        for i, car_type in enumerate(CAR_TYPES):
            ct_idx = self.car_type_to_idx[car_type]
            qty_excess = current_factory_stock_for_step[car_type]  # 工厂剩余库存（未满足需求）

            if qty_excess > 0:
                # 代理选择目标仓库 (0-5)
                target_wh_idx_agent = excess_prod_actions[i]

                # 确保目标仓库索引有效，如果无效，默认存入第一个仓库WH_0
                if not (0 <= target_wh_idx_agent < NUM_TEMP_WAREHOUSES):
                    target_wh_idx_agent = 0

                source_name = FACTORY
                dest_name = WAREHOUSES[target_wh_idx_agent]

                cost_this_shipment = qty_excess * self.get_distance(source_name, dest_name) * TRANSPORT_COST_PER_KM[
                    car_type]
                total_transport_cost += cost_this_shipment

                # 将工厂剩余库存转移到目标仓库
                self.warehouse_inventory[target_wh_idx_agent, ct_idx] += qty_excess
                current_factory_stock_for_step[car_type] = 0  # 工厂清零

                weekly_dispatch_records.append({
                    'week': week_for_log, 'type': 'factory_to_wh', 'car_type': car_type,
                    'quantity': qty_excess, 'source': source_name, 'destination': dest_name,
                    'reason': f"Excess Prod. to WH_{target_wh_idx_agent}", 'cost': round(cost_this_shipment, 2)
                })

        # --- 4. 计算奖励 ---
        reward: float = -total_transport_cost  # 运输成本是负奖励
        if unmet_demand_this_step > 0:
            reward -= unmet_demand_this_step * UNMET_DEMAND_PENALTY  # 未满足需求惩罚

        # --- 5. 推进时间步和更新环境状态 ---
        self.current_week += 1
        terminated = (self.current_week >= NUM_WEEKS_PER_YEAR)
        truncated = False  # 如果有TimeLimitWrapper会设置这个

        if not terminated:
            # 如果没有终止，生成下一周的生产和需求
            self.current_production = self._generate_production()
            self.current_demands = self._generate_demands()

        observation = self._get_obs()
        info = self._get_info()
        info['unmet_this_step'] = unmet_demand_this_step
        info['dispatch_records'] = weekly_dispatch_records

        return observation, reward, terminated, truncated, info

    def render(self):
        """
        环境渲染方法 (当前未实现)。
        """
        pass

    def close(self):
        """
        关闭环境资源 (当前无特定资源需关闭)。
        """
        pass


def make_env(seed: int = 42, rank: int = 0, log_dir: Optional[str] = None) -> callable:
    """
    创建并返回一个用于Stable-Baselines3的Gymnasium环境初始化函数。
    可以指定随机种子，并可选地配置Monitor来记录环境信息。

    Args:
        seed (int): 基础随机种子。
        rank (int): 环境的排名（用于多环境并行训练）。
        log_dir (Optional[str]): Monitor日志文件保存目录。

    Returns:
        callable: 一个返回 Gymnasium 环境实例的函数。
    """

    def _init() -> gym.Env:
        env_raw = CarLogisticsEnv(seed=seed + rank)
        if log_dir:
            monitor_path = os.path.join(log_dir, str(rank))
            os.makedirs(monitor_path, exist_ok=True)
            # Monitor用于记录训练过程中的奖励和额外信息
            env_monitored = Monitor(env_raw, filename=monitor_path,
                                    info_keywords=("unmet_this_step", "dispatch_records", "demands"))
        else:
            env_monitored = Monitor(env_raw, info_keywords=("unmet_this_step", "dispatch_records", "demands"))
        return env_monitored

    return _init


def train_model(model: PPO, env: VecNormalize, total_timesteps: int,
                model_path: str, stats_path: str) -> None:
    """
    训练强化学习模型并保存。

    Args:
        model (PPO): PPO模型实例。
        env (VecNormalize): 归一化后的环境。
        total_timesteps (int): 总训练步数。
        model_path (str): 模型保存路径。
        stats_path (str): 环境统计数据保存路径。
    """
    print(f"开始训练，目标步数: {total_timesteps}...")
    start_time = time.time()
    try:
        model.learn(total_timesteps=total_timesteps, progress_bar=True)
        model.save(model_path)
        env.save(stats_path)
        print("训练完成。模型和环境统计数据已保存。")
    except KeyboardInterrupt:
        model.save(model_path + "_interrupted")
        env.save(stats_path + "_interrupted")
        print("训练被中断。模型和环境统计数据已保存为中断版本。")
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        model.save(model_path + "_error")
        env.save(stats_path + "_error")
    end_time = time.time()
    print(f"训练耗时 {end_time - start_time:.2f} 秒.")


def evaluate_model(model_path: str, stats_path: str, num_eval_episodes: int = 1,
                   monitor_log_root: Optional[str] = None) -> None:
    """
    评估训练好的模型，并生成详细的报告Excel文件。

    Args:
        model_path (str): 模型文件路径。
        stats_path (str): 环境统计数据文件路径。
        num_eval_episodes (int): 评估回合数。
        monitor_log_root (Optional[str]): 评估日志保存目录 (可选，用于make_env)。
    """
    if not (os.path.exists(model_path) and os.path.exists(stats_path)):
        print(f"在 {model_path} 或 {stats_path} 未找到模型/统计文件用于评估。")
        return

    print(f"\n--- 正在评估来自 {model_path} 的训练模型 ---")

    # 创建评估环境，确保不记录Monitor日志以避免干扰训练日志
    eval_env_callable = make_env(seed=123, rank=0, log_dir=monitor_log_root)
    eval_env_vec = DummyVecEnv([eval_env_callable])

    # 加载环境统计数据，并设置评估模式
    eval_env = VecNormalize.load(stats_path, eval_env_vec)
    eval_env.training = False  # 关闭训练模式
    eval_env.norm_reward = False  # 评估时不归一化奖励

    # 加载模型
    model_to_eval = PPO.load(model_path, env=eval_env)

    total_reward_eval = 0.0

    for episode in range(num_eval_episodes):
        obs, _ = eval_env.reset()
        terminated = np.array([False])
        truncated = np.array([False])
        episode_reward = 0.0
        unmet_total_year = 0
        current_week_in_year = 0
        yearly_dispatch_log: List[Dict] = []
        yearly_demand_log: List[Dict] = []

        print(f"\n--- 评估年 {episode + 1} ---")
        while not (terminated[0] or truncated[0]):
            action, _states = model_to_eval.predict(obs, deterministic=True)
            obs, reward, dones, infos = eval_env.step(action)
            terminated = dones
            # 兼容不同版本的gymnasium TimeLimitWrapper
            truncated_info = infos[0].get("TimeLimit.truncated", False)
            truncated[0] = truncated_info if isinstance(truncated_info, bool) else truncated_info.item()

            episode_reward += reward[0]
            unmet_total_year += infos[0].get('unmet_this_step', 0)
            current_week_in_year = infos[0].get('current_week', 0)
            yearly_dispatch_log.extend(infos[0].get('dispatch_records', []))

            # 收集当周需求
            current_week_demands_info = infos[0].get('demands', {})
            if current_week_demands_info:
                # 遍历大客户需求
                for lc_idx, lc_demand in enumerate(current_week_demands_info.get('LC', [])):
                    for car_type, qty in lc_demand.items():
                        yearly_demand_log.append({
                            'week': current_week_in_year,
                            'customer_type': '大客户',
                            'customer_id': LARGE_CUSTOMERS[lc_idx],
                            'car_type': car_type,
                            'demand_qty': qty
                        })
                # 遍历4S店需求
                for s4_idx, s4_demand in enumerate(current_week_demands_info.get('4S', [])):
                    for car_type, qty in s4_demand.items():
                        yearly_demand_log.append({
                            'week': current_week_in_year,
                            'customer_type': '4S店',
                            'customer_id': S4_STORES[s4_idx],
                            'car_type': car_type,
                            'demand_qty': qty
                        })
                # 遍历经销商需求
                for dlr_idx, dlr_demand in enumerate(current_week_demands_info.get('Dealer', [])):
                    for car_type, qty in dlr_demand.items():
                        yearly_demand_log.append({
                            'week': current_week_in_year,
                            'customer_type': '经销商',
                            'customer_id': DEALERS[dlr_idx],
                            'car_type': car_type,
                            'demand_qty': qty
                        })

            # 每10周或回合结束时打印摘要
            if (current_week_in_year % 10 == 0 and current_week_in_year > 0) or terminated[0] or truncated[0]:
                print(
                    f"  周: {current_week_in_year:<3}, 当周奖励: {reward[0]:<8.2f}, 年累计奖励: {episode_reward:<10.2f}, 当周未满足: {infos[0].get('unmet_this_step', 0)}"
                )
                monitor_episode_info = infos[0].get('episode')
                if (terminated[0] or truncated[0]) and monitor_episode_info:
                    print(
                        f"    Monitor回合信息 - 奖励: {monitor_episode_info['r']:.2f}, 长度: {monitor_episode_info['l']}, 时间: {monitor_episode_info['t']:.2f}s"
                    )

        print(f"评估年 {episode + 1} 结束. 总奖励: {episode_reward:.2f}, 全年未满足需求总量: {unmet_total_year}")

        # 导出评估报告到Excel
        excel_filename_summary = f"评估报告_第{episode + 1}年.xlsx"
        try:
            with pd.ExcelWriter(excel_filename_summary, engine='openpyxl') as writer:
                # 调度日志
                if yearly_dispatch_log:
                    df_year_dispatch = pd.DataFrame(yearly_dispatch_log)
                    column_name_mapping_cn_dispatch = {
                        'week': '周次', 'type': '调度类型', 'car_type': '车型', 'quantity': '数量',
                        'source': '来源地', 'destination': '目的地', 'reason': '原因/备注', 'cost': '运输成本'
                    }
                    df_year_dispatch_cn = df_year_dispatch.rename(columns=column_name_mapping_cn_dispatch)
                    df_year_dispatch_cn.to_excel(writer, sheet_name='调度日志', index=False)
                    print("  调度日志已写入Excel。")

                # 每周需求
                if yearly_demand_log:
                    df_year_demand = pd.DataFrame(yearly_demand_log)
                    demand_column_mapping_cn = {
                        'week': '周次', 'customer_type': '客户类型', 'customer_id': '客户ID',
                        'car_type': '车型', 'demand_qty': '需求量'
                    }
                    df_year_demand_cn = df_year_demand.rename(columns=demand_column_mapping_cn)
                    df_year_demand_cn.to_excel(writer, sheet_name='每周需求', index=False)
                    print("  每周需求已写入Excel。")

                # 获取原始环境实例以访问距离数据
                original_env_instance = eval_env.envs[0].env
                if hasattr(original_env_instance, 'distances') and original_env_instance.distances:
                    distances_list = []
                    for (loc1, loc2), dist_val in original_env_instance.distances.items():
                        # 只添加一次 (loc1, loc2) 和 (loc2, loc1) 的配对，避免重复
                        if loc1 < loc2:  # 确保顺序，避免重复，例如(A,B)和(B,A)只记录一次
                            distances_list.append({'地点1': loc1, '地点2': loc2, '距离_km': dist_val})
                    if distances_list:
                        df_distances = pd.DataFrame(distances_list)
                        df_distances.to_excel(writer, sheet_name='地点距离', index=False)
                        print("  地点距离已写入Excel。")

                # 单位运输成本
                df_transport_costs = pd.DataFrame(list(TRANSPORT_COST_PER_KM.items()),
                                                  columns=['车型', '单位运输成本_每公里'])
                df_transport_costs.to_excel(writer, sheet_name='单位运输成本', index=False)
                print("  单位运输成本已写入Excel。")

            print(f"评估报告已导出到: {excel_filename_summary}")
        except ImportError:
            print("无法导出到Excel，请安装 'openpyxl' 库。")
        except Exception as e:
            print(f"导出到Excel时发生错误: {e}")

        total_reward_eval += episode_reward

    print(f"\n{num_eval_episodes} 个评估年的平均奖励: {total_reward_eval / num_eval_episodes:.2f}")
    eval_env.close()


if __name__ == "__main__":
    # 配置路径
    MONITOR_LOG_ROOT = "./monitor_logs_full_report/"
    MODEL_PATH = "ppo_car_logistics_full_report.zip"
    STATS_PATH = "vec_normalize_stats_full_report.pkl"
    TENSORBOARD_LOG_PATH = "./ppo_logistics_tensorboard_full_report/"
    TOTAL_TIMESTEPS = 100000  # 训练总步数

    os.makedirs(MONITOR_LOG_ROOT, exist_ok=True)
    os.makedirs(TENSORBOARD_LOG_PATH, exist_ok=True)  # 确保TensorBoard日志目录存在

    # 1. 环境初始化
    env_callable = make_env(seed=42, rank=0, log_dir=MONITOR_LOG_ROOT)
    env_vec = DummyVecEnv([env_callable])
    # VecNormalize 用于观测值和奖励的归一化，有助于RL模型训练
    env = VecNormalize(env_vec, norm_obs=True, norm_reward=True, clip_reward=10.0, gamma=0.99)

    # 2. 模型加载或新建
    model: PPO
    if os.path.exists(MODEL_PATH) and os.path.exists(STATS_PATH):
        print("加载预训练模型和统计数据...")
        # 为了加载VecNormalize的统计数据，需要一个空的DummyVecEnv
        env_to_load_stats_callable = make_env(seed=42, rank=0, log_dir=MONITOR_LOG_ROOT)
        env_to_load_stats_vec = DummyVecEnv([env_to_load_stats_callable])
        env = VecNormalize.load(STATS_PATH, env_to_load_stats_vec)
        env.training = True  # 确保加载后设置为训练模式
        model = PPO.load(MODEL_PATH, env=env)
    else:
        print("未找到预训练模型。正在训练新模型...")
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
                    gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0,
                    tensorboard_log=TENSORBOARD_LOG_PATH)

    # 3. 训练模型
    train_model(model, env, TOTAL_TIMESTEPS, MODEL_PATH, STATS_PATH)
    env.close()  # 训练环境关闭

    # 4. 评估模型
    evaluate_model(MODEL_PATH, STATS_PATH, num_eval_episodes=2)  # 评估2年
