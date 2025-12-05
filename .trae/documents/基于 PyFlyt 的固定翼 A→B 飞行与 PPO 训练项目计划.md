## 项目概述
- 目标：用 PyFlyt 固定翼环境实现稳定起飞、A→B 飞行与可视化；随后接入 PPO 并并行训练，产出可靠策略。
- 环境来源：`PyFlyt/Fixedwing-Waypoints-v3`（官方用法与说明已提供，参考：https://taijunjet.com/PyFlyt/documentation/gym_envs/fixedwing_waypoints_env.html）。
- 兼容性：此环境观测为 Dict/Sequence，使用官方 `FlattenWaypointEnv(context_length=1)` 扁平化（参考：https://taijunjet.com/PyFlyt/_sources/documentation/core/aviary.md.txt）。

## 观测空间设计
- 原始信息：位置、速度、姿态（`angle_representation` 可选 `euler`/`quaternion`）、航点序列、时间等（源自 Waypoints 系列环境的 Dict/Sequence）。
- 扁平化方案（训练用的 Box 向量）：
  - 机体状态：`pos_xyz`（世界系）、`vel_xyz`（世界系或机体系）、`attitude_quat`（或 `euler`）、`ang_vel_xyz`。
  - 任务相关：到目标 B 的向量 `goal_vec = B - pos`，距离 `goal_dist`，航向误差 `heading_err`（速度方向与 `goal_vec` 的夹角）。
  - 飞行安全：`altitude`、`airspeed`（若可得）、风向风速（若启用风）、上一步动作 `prev_action`（提升稳定性）。
  - 归一化：对位置/速度/角速度/距离进行尺度归一（基于 `flight_dome_size`、最大期望速度/角速度、`goal_reach_distance`）。
- 实施：在自定义环境包装中读取原观测，构造上述特征并返回扁平化向量；保持与 SB3 兼容。

## 价值函数设计
- 基础：采用 PPO 的 critic（值函数）作为状态价值近似，网络为 MLP（与策略共享干路或分头）。
- 结构建议：
  - 干路：2–3 层 MLP（64–256 宽度），`tanh` 或 `relu`；正交初始化。
  - 分头：policy_head 与 value_head 分离，避免相互干扰。
- 训练细节：
  - 使用 GAE（`gae_lambda≈0.95`）与优势归一化；
  - 值函数损失系数 `vf_coef≈0.5`；可启用值函数裁剪减少过估（`clip_range_vf`）。
  - 观测与回报归一化（`VecNormalize`），减少尺度漂移对价值学习的影响。

## 目标/奖励函数设计
- 总体设计原则：以“向 B 前进并安全稳定”为核心；采用潜在式（potential-based）塑形，避免改变最优策略集合。
- 建议项：
  - 进度奖励（主）：`r_progress = k1 * (d_prev - d_curr)`，鼓励距离递减；到达阈值（`goal_reach_distance`）给终端奖励 `R_goal`。
  - 航向对齐：`r_heading = k2 * cos(heading_err)`，鼓励速度方向对准 `goal_vec`。
  - 高度/空速约束：偏离安全包线小惩罚，极端越界大惩罚并截断（俯仰/横滚超限、最低高度、最大过载）。
  - 控制平滑：`r_action_smooth = -k3 * ||a_t - a_{t-1}||`，减小瞬态震荡。
  - 时间惩罚：每步微弱负奖励，促使尽快到达。
- 终止条件：
  - 成功到达 B；
  - 仿真超时（`max_duration_seconds`）；
  - 越界（`flight_dome_size`）；
  - 姿态/空速严重异常（安全中止）。
- 参数初值：`R_goal=+5.0`，`k1=1.0`，`k2=0.5`，`k3=0.01`（后续网格/贝叶斯优化）。

## 并行训练设计
- 向量化：`SubprocVecEnv(num_envs=8–32)`（WSL2/Ubuntu 下获得更好 CPU 利用）；
- 归一化：`VecNormalize`（obs+reward）；
- 采样：`n_steps≈1024`、`batch_size≈256`、`num_envs` 与 `n_steps` 乘积控制每轮样本量；
- 日志：TensorBoard/CSV；评估环境独立、同构参数但固定 A/B；
- 随机化：起点轻微扰动、风弱随机（后续逐步增强），提升泛化。

## 环境与包装实现
- 创建基于 `Fixedwing-Waypoints-v3` 的 A→B 包装：`num_targets=1`，将唯一航点固定为 B。
- `FlattenWaypointEnv(context_length=1)` 扁平化后，再进行自定义特征拼接与归一化。
- 可视化：`render_mode="human"`；记录轨迹用于二维/三维绘图；输出到 `utils/vis.py`。

## PPO 训练管线
- 模型：`PPO(MlpPolicy, vec_env, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, vf_coef=0.5, n_steps=1024, batch_size=256)`；
- 回调：定期评估成功率/到达时间，保存最优模型，必要时早停；
- 评估：单独 `eval_env` 固定 A/B，统计 100 次运行的成功率与平均耗时。

## 评估与可视化
- 指标：成功率、平均到达时间、越界率、最小高度、最大姿态偏差；
- 产出：轨迹图、训练曲线、渲染视频/gif。

## 摄像头与目标锁定（后续阶段）
- 固定翼类具备相机参数能力（文档片段显示支持相机配置），在完成 A→B 后接入相机帧并做简化目标检测（颜色/形状）；
- 二阶段控制：阶段一导航到 B；阶段二视觉闭环指向目标并撞击（可用分层策略或统一奖励）。

## 代码风格与约定
- 语言：中文；作者：`wdblink`；
- 风格：Google 风格，文件/类/函数级注释齐备；
- 结构：配置与代码分离，模块职责清晰，便于扩展维护。

## 里程碑与交付
- 里程碑 1：A→B 环境包装 + 渲染演示；
- 里程碑 2：PPO 并行训练最小闭环；
- 里程碑 3：评估与可视化报告；
- 里程碑 4（后续）：相机与目标锁定集成。

## 下一步
- 若认可本计划，我将初始化项目骨架并交付里程碑 1 与 2 的实现（含并行训练、观测/价值/奖励设计落地）。