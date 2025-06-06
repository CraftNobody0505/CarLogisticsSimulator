import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import os
import time
import pandas as pd
import multiprocessing  # 用于获取CPU核心数

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

# --- 常量定义 ---
NUM_WEEKS_PER_YEAR = 52
CAR_TYPES = ['High', 'Mid', 'Low']
NUM_CAR_TYPES = len(CAR_TYPES)
FACTORY = "Factory"
NUM_TEMP_WAREHOUSES = 6
WAREHOUSES = [f"WH_{i}" for i in range(NUM_TEMP_WAREHOUSES)]
NUM_LARGE_CUSTOMERS = 3
LARGE_CUSTOMERS = [f"LC_{i}" for i in range(NUM_LARGE_CUSTOMERS)]
NUM_4S_STORES = 5
S4_STORES = [f"4S_{i}" for i in range(NUM_4S_STORES)]
NUM_DEALERS = 8
DEALERS = [f"Dealer_{i}" for i in range(NUM_DEALERS)]
ALL_LOCATIONS = [FACTORY] + WAREHOUSES + LARGE_CUSTOMERS + S4_STORES + DEALERS
TRANSPORT_COST_PER_KM = {'High': 3.0, 'Mid': 2.0, 'Low': 1.5}
FACTORY_PROD_RANGES = {'High': (30, 80), 'Mid': (60, 150), 'Low': (70, 180)}
LC_DEMAND_RANGES = {ct: (1, 6) for ct in CAR_TYPES}
S4_DEMAND_RANGES = {'High': (2, 10)}
DEALER_DEMAND_RANGES = {'Mid': (3, 15), 'Low': (3, 15)}
MAX_WAREHOUSE_CAPACITY_PER_TYPE = 500
MAX_PRODUCTION_PER_TYPE = max(val[1] for val in FACTORY_PROD_RANGES.values()) + 20
MAX_DEMAND_PER_ENTITY_PER_TYPE = 20
UNMET_DEMAND_PENALTY_ENV = 2000000


class CarLogisticsEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(self, seed=None):
        super().__init__()
        self.np_random, self.seed = gym.utils.seeding.np_random(seed)
        if self.seed is not None: random.seed(int(self.seed))
        self.current_week = 0
        self.distances = self._generate_distances()
        self.car_type_to_idx = {name: i for i, name in enumerate(CAR_TYPES)}
        self.observation_space = spaces.Box(low=0, high=1.0, shape=(51,), dtype=np.float32)
        num_demand_points = (NUM_LARGE_CUSTOMERS * NUM_CAR_TYPES) + (NUM_4S_STORES * 1) + (NUM_DEALERS * 2)
        num_source_options = 1 + NUM_TEMP_WAREHOUSES
        num_excess_prod_decisions = NUM_CAR_TYPES
        num_excess_dest_options = NUM_TEMP_WAREHOUSES
        action_dims = [num_source_options] * num_demand_points + [num_excess_dest_options] * num_excess_prod_decisions
        self.action_space = spaces.MultiDiscrete(action_dims)
        self.warehouse_inventory = np.zeros((NUM_TEMP_WAREHOUSES, NUM_CAR_TYPES), dtype=int)
        self.current_production = {ct: 0 for ct in CAR_TYPES}
        self.current_demands = {}
        self.initial_stock_per_type_per_wh = 15

    def _generate_distances(self):
        distances = {}
        for i in range(len(ALL_LOCATIONS)):
            for j in range(i + 1, len(ALL_LOCATIONS)):
                loc1, loc2 = ALL_LOCATIONS[i], ALL_LOCATIONS[j]
                is_fac_wh = (loc1 == FACTORY and loc2.startswith("WH")) or (loc2 == FACTORY and loc1.startswith("WH"))
                dist_val = self.np_random.integers(50, 200, endpoint=True) if is_fac_wh else self.np_random.integers(
                    100, 1000, endpoint=True)
                distances[(loc1, loc2)] = dist_val
                distances[(loc2, loc1)] = dist_val
            distances[(ALL_LOCATIONS[i], ALL_LOCATIONS[i])] = 0
        return distances

    def get_distance(self, loc1_name, loc2_name):
        return self.distances.get((loc1_name, loc2_name), float('inf'))

    def _generate_production(self):
        production = {}
        for car_type in CAR_TYPES: min_p, max_p = FACTORY_PROD_RANGES[car_type]; production[
            car_type] = self.np_random.integers(min_p, max_p, endpoint=True)
        return production

    def _generate_demands(self):
        demands = {'LC': [{} for _ in range(NUM_LARGE_CUSTOMERS)], '4S': [{} for _ in range(NUM_4S_STORES)],
                   'Dealer': [{} for _ in range(NUM_DEALERS)]}
        for i in range(NUM_LARGE_CUSTOMERS):
            for car_type in CAR_TYPES: min_d, max_d = LC_DEMAND_RANGES[car_type]; demands['LC'][i][
                car_type] = self.np_random.integers(min_d, max_d, endpoint=True)
        for i in range(NUM_4S_STORES): min_d, max_d = S4_DEMAND_RANGES['High']; demands['4S'][i][
            'High'] = self.np_random.integers(min_d, max_d, endpoint=True)
        for i in range(NUM_DEALERS):
            for car_type in ['Mid', 'Low']: min_d, max_d = DEALER_DEMAND_RANGES[car_type]; demands['Dealer'][i][
                car_type] = self.np_random.integers(min_d, max_d, endpoint=True)
        return demands

    def _get_obs(self):
        obs = []
        for car_type in CAR_TYPES: obs.append(self.current_production[car_type] / MAX_PRODUCTION_PER_TYPE)
        for i in range(NUM_TEMP_WAREHOUSES):
            for j in range(NUM_CAR_TYPES): obs.append(self.warehouse_inventory[i, j] / MAX_WAREHOUSE_CAPACITY_PER_TYPE)
        for i in range(NUM_LARGE_CUSTOMERS):
            for car_type in CAR_TYPES: obs.append(
                self.current_demands['LC'][i].get(car_type, 0) / MAX_DEMAND_PER_ENTITY_PER_TYPE)
        for i in range(NUM_4S_STORES): obs.append(
            self.current_demands['4S'][i].get('High', 0) / MAX_DEMAND_PER_ENTITY_PER_TYPE)
        for i in range(NUM_DEALERS):
            for car_type in ['Mid', 'Low']: obs.append(
                self.current_demands['Dealer'][i].get(car_type, 0) / MAX_DEMAND_PER_ENTITY_PER_TYPE)
        return np.array(obs, dtype=np.float32).clip(0, 1)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None: self.np_random, self.seed = gym.utils.seeding.np_random(seed); random.seed(int(self.seed))
        self.current_week = 0
        self.warehouse_inventory = np.full((NUM_TEMP_WAREHOUSES, NUM_CAR_TYPES), self.initial_stock_per_type_per_wh,
                                           dtype=int)
        self.current_production = self._generate_production()
        self.current_demands = self._generate_demands()
        return self._get_obs(), self._get_info()

    def _get_info(self):
        return {"current_week": self.current_week + 1, "warehouse_inventory": self.warehouse_inventory.copy(),
                "factory_production": self.current_production.copy(), "demands": self.current_demands.copy(),
                "distances": self.distances.copy()}

    def step(self, action):
        week_for_log = self.current_week + 1
        total_transport_cost_this_step = 0.0
        unmet_demand_units_this_step = 0
        current_factory_stock_for_step = self.current_production.copy()
        current_wh_stock_for_step = self.warehouse_inventory.copy()
        weekly_dispatch_records = []
        action_idx = 0
        demand_fulfillment_tasks = []
        for lc_idx in range(NUM_LARGE_CUSTOMERS):
            for car_type in CAR_TYPES:
                ct_idx = self.car_type_to_idx[car_type]
                demand_qty = self.current_demands['LC'][lc_idx].get(car_type, 0)
                if demand_qty > 0: source_choice = action[action_idx]; demand_fulfillment_tasks.append(
                    {'qty': demand_qty, 'car_type': car_type, 'ct_idx': ct_idx, 'source_choice': source_choice,
                     'dest_name': LARGE_CUSTOMERS[lc_idx], 'original_demand_qty': demand_qty})
                action_idx += 1
        ht_idx = self.car_type_to_idx['High']
        for s4_idx in range(NUM_4S_STORES):
            demand_qty = self.current_demands['4S'][s4_idx].get('High', 0)
            if demand_qty > 0: source_choice = action[action_idx]; demand_fulfillment_tasks.append(
                {'qty': demand_qty, 'car_type': 'High', 'ct_idx': ht_idx, 'source_choice': source_choice,
                 'dest_name': S4_STORES[s4_idx], 'original_demand_qty': demand_qty})
            action_idx += 1
        for dlr_idx in range(NUM_DEALERS):
            for car_type in ['Mid', 'Low']:
                ct_idx = self.car_type_to_idx[car_type]
                demand_qty = self.current_demands['Dealer'][dlr_idx].get(car_type, 0)
                if demand_qty > 0: source_choice = action[action_idx]; demand_fulfillment_tasks.append(
                    {'qty': demand_qty, 'car_type': car_type, 'ct_idx': ct_idx, 'source_choice': source_choice,
                     'dest_name': DEALERS[dlr_idx], 'original_demand_qty': demand_qty})
                action_idx += 1
        for task in demand_fulfillment_tasks:
            car_type, ct_idx, source_choice_agent, dest_name, original_demand_qty_for_task = task['car_type'], task[
                'ct_idx'], task['source_choice'], task['dest_name'], task['original_demand_qty']
            fulfilled_this_task_total = 0
            if source_choice_agent == 0:
                source_name = FACTORY;
                available = current_factory_stock_for_step[car_type];
                can_ship = min(original_demand_qty_for_task, available)
                if can_ship > 0: current_factory_stock_for_step[
                    car_type] -= can_ship; cost_this_shipment = can_ship * self.get_distance(source_name, dest_name) * \
                                                                TRANSPORT_COST_PER_KM[
                                                                    car_type]; total_transport_cost_this_step += cost_this_shipment; fulfilled_this_task_total += can_ship; weekly_dispatch_records.append(
                    {'week': week_for_log, 'type': 'demand_fulfillment', 'car_type': car_type, 'quantity': can_ship,
                     'source': source_name, 'destination': dest_name, 'reason': f"Demand (Agent: Factory)",
                     'cost': round(cost_this_shipment, 2)})
            else:
                wh_idx = source_choice_agent - 1
                if 0 <= wh_idx < NUM_TEMP_WAREHOUSES:
                    source_name = WAREHOUSES[wh_idx];
                    available = current_wh_stock_for_step[wh_idx, ct_idx];
                    can_ship = min(original_demand_qty_for_task, available)
                    if can_ship > 0: current_wh_stock_for_step[
                        wh_idx, ct_idx] -= can_ship; cost_this_shipment = can_ship * self.get_distance(source_name,
                                                                                                       dest_name) * \
                                                                          TRANSPORT_COST_PER_KM[
                                                                              car_type]; total_transport_cost_this_step += cost_this_shipment; fulfilled_this_task_total += can_ship; weekly_dispatch_records.append(
                        {'week': week_for_log, 'type': 'demand_fulfillment', 'car_type': car_type, 'quantity': can_ship,
                         'source': source_name, 'destination': dest_name, 'reason': f"Demand (Agent: {source_name})",
                         'cost': round(cost_this_shipment, 2)})
            remaining_to_fulfill_for_task = original_demand_qty_for_task - fulfilled_this_task_total
            if remaining_to_fulfill_for_task > 0 and source_choice_agent != 0:
                source_name_alt = FACTORY;
                available_alt = current_factory_stock_for_step[car_type];
                can_ship_alt = min(remaining_to_fulfill_for_task, available_alt)
                if can_ship_alt > 0: current_factory_stock_for_step[
                    car_type] -= can_ship_alt; cost_this_shipment = can_ship_alt * self.get_distance(source_name_alt,
                                                                                                     dest_name) * \
                                                                    TRANSPORT_COST_PER_KM[
                                                                        car_type]; total_transport_cost_this_step += cost_this_shipment; fulfilled_this_task_total += can_ship_alt; remaining_to_fulfill_for_task -= can_ship_alt; weekly_dispatch_records.append(
                    {'week': week_for_log, 'type': 'demand_fallback', 'car_type': car_type, 'quantity': can_ship_alt,
                     'source': source_name_alt, 'destination': dest_name, 'reason': f"Demand (Fallback: Factory)",
                     'cost': round(cost_this_shipment, 2)})
            if remaining_to_fulfill_for_task > 0:
                candidate_whs = []
                for wh_alt_idx_cand in range(NUM_TEMP_WAREHOUSES):
                    if source_choice_agent == (wh_alt_idx_cand + 1): continue
                    if current_wh_stock_for_step[wh_alt_idx_cand, ct_idx] > 0: cost = self.get_distance(
                        WAREHOUSES[wh_alt_idx_cand], dest_name) * TRANSPORT_COST_PER_KM[car_type]; candidate_whs.append(
                        {'idx': wh_alt_idx_cand, 'cost': cost,
                         'stock': current_wh_stock_for_step[wh_alt_idx_cand, ct_idx]})
                candidate_whs.sort(key=lambda x: x['cost'])
                for wh_cand in candidate_whs:
                    if remaining_to_fulfill_for_task == 0: break
                    wh_alt_idx, source_name_alt = wh_cand['idx'], WAREHOUSES[wh_cand['idx']];
                    available_alt = current_wh_stock_for_step[wh_alt_idx, ct_idx];
                    can_ship_alt = min(remaining_to_fulfill_for_task, available_alt)
                    if can_ship_alt > 0: current_wh_stock_for_step[
                        wh_alt_idx, ct_idx] -= can_ship_alt; cost_this_shipment = can_ship_alt * self.get_distance(
                        source_name_alt, dest_name) * TRANSPORT_COST_PER_KM[
                                                                                      car_type]; total_transport_cost_this_step += cost_this_shipment; fulfilled_this_task_total += can_ship_alt; remaining_to_fulfill_for_task -= can_ship_alt; weekly_dispatch_records.append(
                        {'week': week_for_log, 'type': 'demand_fallback', 'car_type': car_type,
                         'quantity': can_ship_alt, 'source': source_name_alt, 'destination': dest_name,
                         'reason': f"Demand (Fallback: {source_name_alt})", 'cost': round(cost_this_shipment, 2)})
            if remaining_to_fulfill_for_task > 0: unmet_demand_units_this_step += remaining_to_fulfill_for_task
        self.warehouse_inventory = current_wh_stock_for_step.copy()
        excess_prod_actions = action[action_idx:]
        for i, car_type in enumerate(CAR_TYPES):
            ct_idx = self.car_type_to_idx[car_type];
            qty_excess = current_factory_stock_for_step[car_type]
            if qty_excess > 0:
                target_wh_idx_agent = excess_prod_actions[i]
                if not (0 <= target_wh_idx_agent < NUM_TEMP_WAREHOUSES): target_wh_idx_agent = 0
                source_name, dest_name = FACTORY, WAREHOUSES[target_wh_idx_agent];
                cost_this_shipment = qty_excess * self.get_distance(source_name, dest_name) * TRANSPORT_COST_PER_KM[
                    car_type];
                total_transport_cost_this_step += cost_this_shipment;
                self.warehouse_inventory[target_wh_idx_agent, ct_idx] += qty_excess;
                weekly_dispatch_records.append(
                    {'week': week_for_log, 'type': 'factory_to_wh', 'car_type': car_type, 'quantity': qty_excess,
                     'source': source_name, 'destination': dest_name, 'reason': f"Excess Prod. to {dest_name}",
                     'cost': round(cost_this_shipment, 2)})
        unmet_demand_penalty_this_step = unmet_demand_units_this_step * UNMET_DEMAND_PENALTY_ENV
        reward = -(total_transport_cost_this_step + unmet_demand_penalty_this_step)
        self.current_week += 1;
        terminated = (self.current_week >= NUM_WEEKS_PER_YEAR);
        truncated = False
        if not terminated: self.current_production = self._generate_production(); self.current_demands = self._generate_demands()
        observation = self._get_obs();
        info = self._get_info()
        info['unmet_this_step'] = unmet_demand_units_this_step;
        info['transport_cost_this_step'] = total_transport_cost_this_step;
        info['dispatch_records'] = weekly_dispatch_records
        return observation, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass


def make_ppo_env(seed=42, rank=0, log_dir_for_monitor_csv=None, version_tag="ppo_model", enable_monitor_logging=True):
    def _init():
        current_seed = seed + rank;
        set_random_seed(current_seed)
        env_raw = CarLogisticsEnv(seed=current_seed)
        info_keywords_for_monitor = ("unmet_this_step", "dispatch_records", "demands")
        monitor_filename = None
        if enable_monitor_logging and log_dir_for_monitor_csv and version_tag:  # Ensure version_tag is also provided for path construction
            monitor_path_base = os.path.join(log_dir_for_monitor_csv, version_tag, "monitor_logs_csv")
            os.makedirs(monitor_path_base, exist_ok=True)
            monitor_filename = os.path.join(monitor_path_base, f"monitor_rank_{rank}.csv")
        env_monitored = Monitor(env_raw, filename=monitor_filename, info_keywords=info_keywords_for_monitor)
        return env_monitored

    return _init


def run_ppo_experiment(
        model_version_tag,  # Still used for organizing TensorBoard/Monitor logs if training occurs
        training_seed,
        eval_seed_start,
        num_train_timesteps,
        ppo_hyperparams,
        log_root_dir_for_training_artifacts,  # For TB/Monitor logs if training
        num_eval_episodes=100,
        force_retrain=False,
        num_parallel_envs=1
):
    print(f"\n--- 开始PPO模型实验 (日志版本标签: {model_version_tag}) ---")
    print(f"PPO超参数: {ppo_hyperparams}")
    print(f"训练步数: {num_train_timesteps}, 并行环境数: {num_parallel_envs}")
    print(f"评估回合数: {num_eval_episodes}, 评估起始种子: {eval_seed_start}")

    # MODIFIED: Define fixed model paths
    FIXED_MODEL_DIR = "./model/"
    FIXED_MODEL_FILENAME = os.path.join(FIXED_MODEL_DIR, "model.zip")
    FIXED_STATS_FILENAME = os.path.join(FIXED_MODEL_DIR, "vecnormalize.pkl")

    # Paths for TensorBoard and Monitor logs (still versioned if training occurs)
    # These are only created if training_occurred is True.
    current_experiment_training_artifacts_path = os.path.join(log_root_dir_for_training_artifacts, model_version_tag)
    # current_experiment_tensorboard_log_dir # Defined if training_occurred
    # Monitor log path will be constructed inside make_ppo_env if needed

    train_env = None
    training_occurred = False

    # Determine if training is needed
    if not force_retrain and os.path.exists(FIXED_MODEL_FILENAME) and os.path.exists(FIXED_STATS_FILENAME):
        print(f"\n找到固定的预训练模型文件:\n  Model: {FIXED_MODEL_FILENAME}\n  Stats: {FIXED_STATS_FILENAME}")
        print("将加载此模型并跳过训练。")
        model_to_load_for_eval = FIXED_MODEL_FILENAME
        stats_to_load_for_eval = FIXED_STATS_FILENAME
    else:
        training_occurred = True
        if force_retrain:
            print(f"\n强制重新训练模型...")
        elif not (os.path.exists(FIXED_MODEL_FILENAME) and os.path.exists(FIXED_STATS_FILENAME)):
            print(f"\n未在固定路径 {FIXED_MODEL_DIR} 找到模型，开始训练新模型...")
        else:  # Should not happen if logic is correct, but as a fallback
            print(f"\n开始训练新模型 (可能是因为一个文件丢失或未知原因)...")

        # Create directories for training artifacts (logs, etc.) and the fixed model path
        print(f"为训练日志创建根目录 (如果尚不存在): {log_root_dir_for_training_artifacts}")
        os.makedirs(log_root_dir_for_training_artifacts, exist_ok=True)

        print(f"为当前实验的训练日志创建目录: {current_experiment_training_artifacts_path}")
        # current_experiment_model_dir is now FIXED_MODEL_DIR for model saving
        os.makedirs(FIXED_MODEL_DIR, exist_ok=True)  # Ensure ./model directory exists for saving

        current_experiment_tensorboard_log_dir = os.path.join(current_experiment_training_artifacts_path,
                                                              "tensorboard_logs")
        os.makedirs(current_experiment_tensorboard_log_dir, exist_ok=True)

        # Monitor logs will be under current_experiment_training_artifacts_path/model_version_tag/monitor_logs_csv
        # log_dir_for_monitor_csv passed to make_ppo_env will be current_experiment_training_artifacts_path

        if num_parallel_envs > 1:
            env_fns = [make_ppo_env(seed=training_seed, rank=i,
                                    log_dir_for_monitor_csv=current_experiment_training_artifacts_path,
                                    version_tag=model_version_tag, enable_monitor_logging=True) for i in
                       range(num_parallel_envs)]
            train_env_vec = SubprocVecEnv(env_fns, start_method='spawn')
        else:
            train_env_vec = DummyVecEnv([make_ppo_env(seed=training_seed, rank=0,
                                                      log_dir_for_monitor_csv=current_experiment_training_artifacts_path,
                                                      version_tag=model_version_tag, enable_monitor_logging=True)])

        train_env = VecNormalize(train_env_vec, norm_obs=False, norm_reward=True, clip_reward=10.0,
                                 gamma=ppo_hyperparams.get('gamma', 0.99))
        model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log=current_experiment_tensorboard_log_dir,
                    seed=training_seed, **ppo_hyperparams)
        start_train_time = time.time()
        try:
            model.learn(total_timesteps=num_train_timesteps, progress_bar=True)
            # MODIFIED: Save to fixed model paths
            model.save(FIXED_MODEL_FILENAME)
            train_env.save(FIXED_STATS_FILENAME)
            print(f"PPO模型训练完成并保存到 {FIXED_MODEL_DIR}。耗时: {(time.time() - start_train_time) / 60:.2f} 分钟。")
            model_to_load_for_eval = FIXED_MODEL_FILENAME  # For subsequent evaluation
            stats_to_load_for_eval = FIXED_STATS_FILENAME
        except Exception as e:
            print(f"PPO模型训练过程中发生错误: {e}")
            # Attempt to save to fixed path even on error, if model object exists
            if 'model' in locals() and model is not None: model.save(FIXED_MODEL_FILENAME + "_error")
            if train_env is not None: train_env.save(FIXED_STATS_FILENAME + "_error")
            import traceback;
            traceback.print_exc()
            if train_env is not None: train_env.close()
            return
        finally:
            if train_env is not None:  # Close train_env whether training succeeded or failed
                train_env.close();
                print("训练环境已关闭。")

    if training_occurred:
        print(f"训练日志 (TensorBoard, Monitor CSV) 保存在: {current_experiment_training_artifacts_path}")
    else:
        print(f"由于加载了现有模型 {FIXED_MODEL_FILENAME}，未生成新的训练日志目录。")

    print(f"\n--- 正在评估模型 (从 {model_to_load_for_eval} 加载) ---")
    if not (os.path.exists(model_to_load_for_eval) and os.path.exists(stats_to_load_for_eval)):
        print(f"错误：未能找到模型 '{model_to_load_for_eval}' 或统计 '{stats_to_load_for_eval}' 进行评估。");
        return

    eval_env_vec_for_loading_stats = None
    eval_env = None
    all_episodes_summary_data = []
    master_dispatch_log_records = []
    master_demand_log_records = []
    first_episode_distances_for_report = None

    try:
        eval_env_vec_for_loading_stats = DummyVecEnv(
            [make_ppo_env(seed=training_seed, rank=0, log_dir_for_monitor_csv=None,
                          version_tag=model_version_tag + "_eval_temp_load", enable_monitor_logging=False)])
        eval_env = VecNormalize.load(stats_to_load_for_eval, eval_env_vec_for_loading_stats)
        eval_env.training = False;
        eval_env.norm_reward = False
        model_to_eval = PPO.load(model_to_load_for_eval, env=eval_env)

        for episode in range(num_eval_episodes):
            current_eval_seed = eval_seed_start + episode;
            eval_env.seed(current_eval_seed)
            obs = eval_env.reset()
            terminated, truncated = np.array([False]), np.array([False])
            episode_reward_sum_raw, episode_unmet_total = 0.0, 0
            is_first_step_of_episode = True;
            current_week_in_year = 0
            print(
                f"\n--- 评估年 {episode + 1}/{num_eval_episodes} (种子: {current_eval_seed}) ---")  # Removed model_version_tag from here for brevity
            while not (terminated[0] or truncated[0]):
                action, _ = model_to_eval.predict(obs, deterministic=True)
                obs, reward, dones, infos = eval_env.step(action)
                terminated, truncated = dones, infos[0].get("TimeLimit.truncated", np.array([False])) if isinstance(
                    infos[0].get("TimeLimit.truncated", False), np.ndarray) else np.array(
                    [infos[0].get("TimeLimit.truncated", False)])
                episode_reward_sum_raw += reward[0];
                episode_unmet_total += infos[0].get('unmet_this_step', 0)
                current_week_in_year = infos[0].get('current_week', 0)
                if is_first_step_of_episode:
                    if first_episode_distances_for_report is None: first_episode_distances_for_report = infos[0].get(
                        'distances', {})
                    is_first_step_of_episode = False
                current_episode_dispatch_records = infos[0].get('dispatch_records', [])
                for record_d in current_episode_dispatch_records: record_d[
                    '评估年份'] = episode + 1; master_dispatch_log_records.append(record_d)
                demands_info = infos[0].get('demands', {})
                if demands_info:
                    temp_demand_records_this_step = []
                    for lc_idx, lc_demand in enumerate(demands_info.get('LC', [])):
                        for ct_key, qty in lc_demand.items(): temp_demand_records_this_step.append(
                            {'评估年份': episode + 1, 'week': current_week_in_year, 'customer_type': '大客户',
                             'customer_id': LARGE_CUSTOMERS[lc_idx], 'car_type': ct_key, 'demand_qty': qty})
                    for s4_idx, s4_demand in enumerate(demands_info.get('4S', [])):
                        for ct_key, qty in s4_demand.items(): temp_demand_records_this_step.append(
                            {'评估年份': episode + 1, 'week': current_week_in_year, 'customer_type': '4S店',
                             'customer_id': S4_STORES[s4_idx], 'car_type': ct_key, 'demand_qty': qty})
                    for dlr_idx, dlr_demand in enumerate(demands_info.get('Dealer', [])):
                        for ct_key, qty in dlr_demand.items(): temp_demand_records_this_step.append(
                            {'评估年份': episode + 1, 'week': current_week_in_year, 'customer_type': '经销商',
                             'customer_id': DEALERS[dlr_idx], 'car_type': ct_key, 'demand_qty': qty})
                    master_demand_log_records.extend(temp_demand_records_this_step)
                if (
                        current_week_in_year % 26 == 0 and current_week_in_year > 0 and current_week_in_year <= NUM_WEEKS_PER_YEAR) or \
                        terminated[0] or truncated[0]:
                    print(
                        f"  年 {episode + 1}, 周: {current_week_in_year:<3}, 当周奖励: {reward[0]:<8.2f}, 年累计奖励: {episode_reward_sum_raw:<10.2f}, 当周未满足: {infos[0].get('unmet_this_step', 0)}")
            episode_total_transport_cost_from_log = sum(
                d['cost'] for d in master_dispatch_log_records if d.get('评估年份') == episode + 1 and 'cost' in d)
            all_episodes_summary_data.append(
                {'评估年份': episode + 1, '随机种子': current_eval_seed, '年度总奖励': round(episode_reward_sum_raw, 2),
                 '全年未满足需求总量': episode_unmet_total,
                 '总运输成本(调度日志)': round(episode_total_transport_cost_from_log, 2)})
            print(
                f"评估年 {episode + 1} 结束。奖励: {episode_reward_sum_raw:.2f}, 未满足: {episode_unmet_total}, 运输成本: {episode_total_transport_cost_from_log:.2f}")

        # MODIFIED: Excel filename is fixed
        excel_filename_summary = f"./评估报告100年汇总.xlsx"  # Removed model_version_tag and num_eval_episodes from here
        print(f"\n正在生成汇总Excel报告: {excel_filename_summary}")
        with pd.ExcelWriter(excel_filename_summary, engine='openpyxl') as writer:
            df_yearly_summary = pd.DataFrame(all_episodes_summary_data)
            df_yearly_summary.to_excel(writer, sheet_name=f'{num_eval_episodes}年汇总统计', index=False);
            print(f"  {num_eval_episodes}年汇总统计已写入。")
            if not df_yearly_summary.empty:
                avg_data = {'指标': ['平均年度总奖励', '平均全年未满足需求总量', '平均总运输成本(调度日志)'],
                            '平均值': [round(df_yearly_summary['年度总奖励'].mean(), 2),
                                       round(df_yearly_summary['全年未满足需求总量'].mean(), 2),
                                       round(df_yearly_summary['总运输成本(调度日志)'].mean(), 2)]}
                pd.DataFrame(avg_data).to_excel(writer, sheet_name='平均统计', index=False);
                print(f"  平均统计已写入。")
            if master_dispatch_log_records:
                df_master_dispatch = pd.DataFrame(master_dispatch_log_records);
                cols = ['评估年份'] + [col for col in df_master_dispatch.columns if col != '评估年份'];
                df_master_dispatch = df_master_dispatch[cols]
                col_map_dispatch = {'week': '周次', 'type': '调度类型', 'car_type': '车型', 'quantity': '数量',
                                    'source': '来源地', 'destination': '目的地', 'reason': '原因/备注',
                                    'cost': '运输成本'}
                df_master_dispatch.rename(columns=col_map_dispatch).to_excel(writer, sheet_name='调度日志汇总',
                                                                             index=False);
                print(f"  调度日志汇总已写入。")
            if master_demand_log_records:
                df_master_demand = pd.DataFrame(master_demand_log_records);
                cols = ['评估年份'] + [col for col in df_master_demand.columns if col != '评估年份'];
                df_master_demand = df_master_demand[cols]
                col_map_demand = {'week': '周次', 'customer_type': '客户类型', 'customer_id': '客户ID',
                                  'car_type': '车型', 'demand_qty': '需求量'}
                df_master_demand.rename(columns=col_map_demand).to_excel(writer, sheet_name='每周需求汇总',
                                                                         index=False);
                print(f"  每周需求汇总已写入。")
            if first_episode_distances_for_report:
                distances_list = [];
                processed_pairs = set();
                sorted_dist_items = sorted(first_episode_distances_for_report.items())
                for (loc1, loc2), dist_val in sorted_dist_items:
                    pair = tuple(sorted((loc1, loc2)))
                    if loc1 != loc2 and pair not in processed_pairs: distances_list.append(
                        {'地点1': loc1, '地点2': loc2, '距离_km': dist_val}); processed_pairs.add(pair)
                if distances_list: pd.DataFrame(distances_list).to_excel(writer, sheet_name='地点距离',
                                                                         index=False); print(f"  地点距离已写入。")
            pd.DataFrame(list(TRANSPORT_COST_PER_KM.items()), columns=['车型', '单位运输成本_每公里']).to_excel(writer,
                                                                                                                sheet_name='单位运输成本',
                                                                                                                index=False);
            print(f"  单位运输成本已写入。")
        print(f"评估报告已导出到: {excel_filename_summary}")
    except Exception as e_eval:
        print(f"评估或生成报告时发生错误: {e_eval}"); import \
            traceback; traceback.print_exc()  # Removed model_version_tag from error msg
    finally:
        if eval_env is not None: eval_env.close(); print("评估环境(VecNormalize)已关闭。")
        if eval_env_vec_for_loading_stats is not None and eval_env_vec_for_loading_stats != eval_env: eval_env_vec_for_loading_stats.close(); print(
            "用于加载统计的临时评估环境(DummyVecEnv)已关闭。")
    print(f"\n--- 实验执行完毕 (日志版本标签: {model_version_tag}) ---")


if __name__ == "__main__":
    LOG_ROOT_FOR_TRAINING_ARTIFACTS = "./ppo_logistics_training_artifacts/"

    try:
        num_cpus = multiprocessing.cpu_count()
        num_parallel_training_envs = min(num_cpus, 8) if num_cpus > 1 else 1
        print(f"检测到 {num_cpus} CPU核心。将用 {num_parallel_training_envs} 个并行环境训练。")
    except NotImplementedError:
        print("无法检测CPU核心数。默认用1个并行环境。")
        num_parallel_training_envs = 1

    # model_version_tag is now primarily for versioning training logs (TB, Monitor CSV)
    # if training occurs. The main model and Excel report have fixed names/paths.
    CURRENT_MODEL_VERSION_TAG = "PPO_Logistics_Run_Default"
    ORIGINAL_TRAINING_SEED = 789
    ORIGINAL_EVAL_SEED_START = 2000
    ORIGINAL_TRAIN_STEPS = 150000

    ORIGINAL_PPO_HYPERPARAMS = {
        'learning_rate': 1e-3, 'n_steps': 512, 'batch_size': 32, 'n_epochs': 5,
        'gamma': 0.98, 'gae_lambda': 0.92, 'clip_range': 0.3, 'vf_coef': 0.6,
        'ent_coef': 0.01, 'policy_kwargs': dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))
    }

    run_ppo_experiment(
        model_version_tag=CURRENT_MODEL_VERSION_TAG,  # Used for log subdirs if training
        training_seed=ORIGINAL_TRAINING_SEED,
        eval_seed_start=ORIGINAL_EVAL_SEED_START,
        num_train_timesteps=ORIGINAL_TRAIN_STEPS,
        ppo_hyperparams=ORIGINAL_PPO_HYPERPARAMS,
        log_root_dir_for_training_artifacts=LOG_ROOT_FOR_TRAINING_ARTIFACTS,
        num_eval_episodes=100,
        force_retrain=False,
        num_parallel_envs=num_parallel_training_envs
    )

    print(f"\n--- 所有实验执行完毕 ---")
    print(f"训练模型保存在 './model/' 目录下。")
    print(f"TensorBoard和Monitor日志保存在 '{LOG_ROOT_FOR_TRAINING_ARTIFACTS}' 下的对应版本子目录中。")
    print(f"最终的Excel汇总报告 '{'评估报告100年汇总.xlsx'}' 直接保存在脚本根目录下。")
