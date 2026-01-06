"""项目文件：Fixedwing-Waypoints-v3 PPO 训练脚本

说明:
    使用 Stable-Baselines3 的 PPO 算法训练 PyFlyt/Fixedwing-Waypoints-v3 环境。
    该环境的目标是通过控制 roll, pitch, yaw, thrust 来追踪一系列航点。
"""

import os
import sys
import argparse
import gymnasium as gym
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from PyFlyt.gym_envs.fixedwing_envs.fixedwing_waypoints_env import FixedwingWaypointsEnv

# 确保能导入 PyFlyt
import PyFlyt.gym_envs
from PyFlyt.gym_envs import FlattenWaypointEnv

# 训练配置
TRAIN_CONFIG = {
    "total_timesteps": 4_000_000,
    "num_envs": 32,
    "num_targets": 8,
    "goal_reach_distance": 4,
    "sparse_reward": True,
    "n_eval_episodes": 20,
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 128,
    "n_epochs": 20,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.001,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "seed": 42,
    "log_dir": "logs/waypoints_ppo_v3.6",
    "model_dir": "models/waypoints_ppo_v3.6",
    "flight_dome_size": 100.0,
    "max_duration_seconds": 120.0,
    "context_length": 2,  # 观测中包含当前目标点和下一个目标点
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", type=str, default='models/waypoints_ppo_v3.5/best_model.zip')
    parser.add_argument("--vecnorm_path", type=str, default=None)
    parser.add_argument("--total_timesteps", type=int, default=None)
    return parser.parse_args()

def _infer_vecnorm_path(pretrained_model: str | None, vecnorm_path: str | None) -> str | None:
    if vecnorm_path:
        return vecnorm_path
    if not pretrained_model:
        return None

    candidate_dirs = [
        os.path.dirname(pretrained_model),
        TRAIN_CONFIG["model_dir"],
    ]
    for d in candidate_dirs:
        if not d:
            continue
        p = os.path.join(d, "vecnorm.pkl")
        if os.path.exists(p):
            return p
    return None

def make_env(rank: int, seed: int = 0):
    """
    创建环境的工厂函数
    Args:
        sparse_reward (bool): whether to use sparse rewards or not.
        num_targets (int): number of waypoints in the environment.
        goal_reach_distance (float): distance to the waypoints for it to be considered reached.
        flight_mode (int): The flight mode of the UAV.
        flight_dome_size (float): size of the allowable flying area.
        max_duration_seconds (float): maximum simulation time of the environment.
        angle_representation (Literal["euler", "quaternion"]): can be "euler" or "quaternion".
        agent_hz (int): looprate of the agent to environment interaction.
        render_mode (None | Literal["human", "rgb_array"]): render_mode
        render_resolution (tuple[int, int]): render_resolution
    """
    def _init():
        # 创建基础环境
        # Actions: [roll, pitch, yaw, thrust]
        env = gym.make(
            "PyFlyt/Fixedwing-Waypoints-v3",
            sparse_reward=TRAIN_CONFIG["sparse_reward"],
            num_targets=TRAIN_CONFIG["num_targets"],
            goal_reach_distance=TRAIN_CONFIG["goal_reach_distance"],
            render_mode=None,
            angle_representation="euler", # 使用欧拉角更直观
            flight_dome_size=TRAIN_CONFIG["flight_dome_size"],
            max_duration_seconds=TRAIN_CONFIG["max_duration_seconds"],
            agent_hz=30,
        )
        
        # 扁平化观测空间，以便 MLP 网络处理
        # FlattenWaypointEnv 会将环境的 Dict 观测转换为 Box 观测
        env = FlattenWaypointEnv(env, context_length=TRAIN_CONFIG["context_length"])
        
        env.reset(seed=seed + rank)
        return env
    return _init


class WaypointEvalCallback(EvalCallback):
    def __init__(self, *args, num_targets_total: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_targets_total = int(num_targets_total)
        self._num_targets_reached_buffer: list[int] = []

    def _log_success_callback(self, locals_: dict, globals_: dict) -> None:
        info = locals_.get("info")
        if isinstance(info, (list, tuple)) and len(info) > 0:
            info = info[0]

        if locals_.get("done") and isinstance(info, dict):
            num_targets_reached = info.get("num_targets_reached")
            if num_targets_reached is not None:
                self._num_targets_reached_buffer.append(int(num_targets_reached))

        super()._log_success_callback(locals_, globals_)

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self.model.get_vec_normalize_env() is not None:
                from stable_baselines3.common.vec_env import sync_envs_normalization

                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            self._is_success_buffer = []
            self._num_targets_reached_buffer = []

            from stable_baselines3.common.evaluation import evaluate_policy

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                assert isinstance(episode_rewards, list)
                assert isinstance(episode_lengths, list)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = float(mean_reward)

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")

            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._num_targets_reached_buffer) > 0 and self.num_targets_total > 0:
                reached = np.asarray(self._num_targets_reached_buffer, dtype=np.float64)
                for i in range(1, self.num_targets_total + 1):
                    self.logger.record(f"eval/wp{i}_reach_rate", float(np.mean(reached >= i)))

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = float(mean_reward)
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

def train(args):
    """
    训练主函数
    """
    # 设置随机种子
    set_random_seed(TRAIN_CONFIG["seed"])
    
    # 创建目录
    os.makedirs(TRAIN_CONFIG["log_dir"], exist_ok=True)
    os.makedirs(TRAIN_CONFIG["model_dir"], exist_ok=True)

    # 检测设备
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    print(f"Using device: {device}")

    # 创建并行训练环境
    print("Creating training environments...")
    env = SubprocVecEnv([make_env(i, TRAIN_CONFIG["seed"]) for i in range(TRAIN_CONFIG["num_envs"])])
    
    # 观测和奖励归一化
    vecnorm_path = _infer_vecnorm_path(args.pretrained_model, args.vecnorm_path)
    if vecnorm_path:
        env = VecNormalize.load(vecnorm_path, env)
        env.training = True
        env.norm_reward = True
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # 创建评估环境（独立于训练环境）
    print("Creating evaluation environments...")
    eval_env = SubprocVecEnv([make_env(i + TRAIN_CONFIG["num_envs"], TRAIN_CONFIG["seed"]) for i in range(4)])
    if vecnorm_path:
        eval_env = VecNormalize.load(vecnorm_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
    else:
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    
    # 设置回调函数
    eval_callback = WaypointEvalCallback(
        eval_env,
        best_model_save_path=TRAIN_CONFIG["model_dir"],
        log_path=TRAIN_CONFIG["log_dir"],
        n_eval_episodes=TRAIN_CONFIG["n_eval_episodes"],
        eval_freq=10000 // TRAIN_CONFIG["num_envs"], # 每 10000 步评估一次
        deterministic=True,
        render=False,
        num_targets_total=TRAIN_CONFIG["num_targets"],
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000 // TRAIN_CONFIG["num_envs"],
        save_path=TRAIN_CONFIG["model_dir"],
        name_prefix="waypoints_ppo"
    )

    # 初始化 PPO 模型
    # 无论是否加载预训练模型，都使用当前的配置初始化 PPO
    # 这样可以确保使用新的训练策略（学习率、batch_size 等）
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=TRAIN_CONFIG["learning_rate"],
        n_steps=TRAIN_CONFIG["n_steps"],
        batch_size=TRAIN_CONFIG["batch_size"],
        n_epochs=TRAIN_CONFIG["n_epochs"],
        gamma=TRAIN_CONFIG["gamma"],
        gae_lambda=TRAIN_CONFIG["gae_lambda"],
        clip_range=TRAIN_CONFIG["clip_range"],
        ent_coef=TRAIN_CONFIG["ent_coef"],
        vf_coef=TRAIN_CONFIG["vf_coef"],
        max_grad_norm=TRAIN_CONFIG["max_grad_norm"],
        verbose=1,
        tensorboard_log=TRAIN_CONFIG["log_dir"],
        seed=TRAIN_CONFIG["seed"],
        device=device
    )

    # 如果指定了预训练模型，仅加载其参数
    if args.pretrained_model:
        print(f"Loading pretrained parameters from: {args.pretrained_model}")
        # 加载预训练模型到 CPU（避免设备不匹配）
        pretrained_model = PPO.load(args.pretrained_model, device="cpu")
        # 将参数复制到新模型
        model.set_parameters(pretrained_model.get_parameters())
        # 清理内存
        del pretrained_model
        print("Pretrained parameters loaded successfully.")

    configured_timesteps = args.total_timesteps if args.total_timesteps is not None else TRAIN_CONFIG["total_timesteps"]
    
    # 始终重置时间步计数，因为这是一个新的训练会话（即使加载了权重）
    # 用户要求"步数...重新定义"，意味着从 0 开始计数到 configured_timesteps
    reset_num_timesteps = True

    print(f"Starting training for {configured_timesteps} timesteps...")
    
    try:
        model.learn(
            total_timesteps=configured_timesteps,
            reset_num_timesteps=reset_num_timesteps,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        # 保存最终模型
        print("Saving final model...")
        model.save(os.path.join(TRAIN_CONFIG["model_dir"], "final_model"))
        env.save(os.path.join(TRAIN_CONFIG["model_dir"], "vecnorm.pkl"))
        
        env.close()
        eval_env.close()
        print("Done.")

if __name__ == "__main__":
    train(parse_args())
