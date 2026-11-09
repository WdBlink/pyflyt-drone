# 共享控制执行器：开发资格工作单

2026-11-05。角色：plan 阶段的前置工程验证，不是正式 benchmark、test、pilot 或论文结果。控制器状态仍由 `.paper/` 唯一维护。预算问答已完成，计划尚未获得科学准入。

## 本工作单的可检验结果

先验证最不确定的物理执行能力：单机控制方向 → 有界姿态/高度/速度 → 航线与圆周 → 四机同一 PyBullet world。每步保留配置、轨迹、失败原因及版本，上一单元通过后再推进。只用舵面与电机施力；除初态重置外不得改写位置/速度，不以运动学轨迹替代 PyFlyt。

候选任务条件沿用原计划：20 m/s、AGL 60–100 m、四机、600 s、有界转弯与分开的空中退出圆。先用加速物理仿真排除控制问题；这不证明 1× 时钟或实际推理期间世界持续推进。小规模开发烟测不满足原计划候选的每控制任务族 ≥20 dev roots / ≥100 episodes / ≥95% 安全完成门槛，也不冻结这些门槛。

后续真实参考链的独立条件：只按公开航线搜索 → 因果传感包到达 → 证据确认与报告 → 持续飞行安全退出；隐藏真值只交给传感器与 grader。输出必须分开标注 physics、permission、grading、clock 资格；不能用一条轨迹代替全部 G1–G4。

## 复用与缺口

| 需求 | 组件/固定版本 | 已检查证据与决策 | 未验证部分 |
| --- | --- | --- | --- |
| 共享真实固定翼动力学与同步多机 | PyFlyt 0.29.0 `core.Aviary`、`core.drones.Fixedwing`，PyBullet 3.2.7 | 复用原包。源码 `Aviary.step` 每控制步推进两次 240 Hz 物理步；所有机体在同一 world。`Fixedwing.update_state` 提供本体角速率、欧拉角、本体速度、世界位置 | 控制性能、RTF、风、边界、碰撞、稳定性 |
| 航线执行而非 Agent 直接控舵 | 同包 `Fixedwing.update_control` | 需补共享执行器。模式0只映射舵面；其索引 `[0,0,1,1,2,3]` 使水平/垂直尾舵共用一个输入。使用文档支持的模式−1明确控制六个舵面/油门，避免改变上游包 | 各轴方向及姿态/速度闭环实测 |
| 既有任务/裁判能否原样复用 | MultiUAV-Plat `1794e45e421fb5de03094f0b63f9ca95f86ab42f` | 阅读真实 `server/models/{drone,task}.py`、`server/controllers/drone_controller.py`、`server/main.py`、Area Search 样例和 path partial 测试。`_validate_and_execute_path` 结束时直接 `update_position(final_waypoint)`；任务模型区分 done/passed，提供既有机制先例 | 未复跑服务/评分入口，不能声称整体不支持连续执行；当前明确位置更新路径不能替代本研究物理执行器。GPL 组件未复制到项目 |
| 因果观测、限额报告、连续时钟与配对统计 | 原研究计划与现有项目 | 只补已确认缺口，保留原需求。旧 RL/RTA 不作新基准数据 | 均未验收；强经典基线、八配置、root bootstrap、失败分母待实现 |

检索范围、邻近论文与未知机制另见 `literature/nearest-work-matrix.v1.md` 和 `venue/RAL_CALIBRATION.md`。上表是实际源码补核，不代表穷尽现有 benchmark。

来源：https://github.com/zhangsheng93/MultiUAV-Plat/tree/1794e45e421fb5de03094f0b63f9ca95f86ab42f 。`drone_controller.py` 721–939行，尤以926–929行；`main.py` 157–171行另有 target/environment 后台更新，因此不把整个世界笼统描述为暂停。

## 本地环境修复

Python 3.11，独立 `.venv`。首次 `uv pip install numpy==1.26.4 PyFlyt==0.29.0 pytest` 失败：PyBullet 的旧 zlib `zutil.h:128` 定义 `fdopen(fd, mode) NULL` 与当前 macOS SDK 冲突。失败日志保留，不当作模型失败。

按实际预处理条件，使用 `CFLAGS='-Dfdopen=fdopen'` 阻止该 fallback 宏，重新从同一 PyPI 3.2.7 源码编译成功；未修改空气动力学/物理源码，也未换成未核验的第三方 fork。完整安装命令与版本在资格记录中保留；单纯可导入不证明物理正确。

```sh
uv venv --python /opt/homebrew/bin/python3.11 .venv
CFLAGS='-Dfdopen=fdopen' uv pip install --python .venv/bin/python numpy==1.26.4 PyFlyt==0.29.0 pytest
```

本阶段不训练 PPO、不调用付费模型、不改源项目、不使用 test 调参。人工领域复核状态：`human_review_pending`。
