# 平台前置开发验证：2026-11-05

后续参考任务与G1完整资格进展见 [PROGRESS.md](PROGRESS.md)。本页保留早期探针的历史范围。

本轮实际完成了运行环境修复、共享固定翼控制器、控制开发飞行、短时1×隔离探针和第6.2节部分评分语义。**没有完成 G1–G4、合法任务参考 episode、正式八配置实验或论文初稿。** 这些记录不能进入主结果表。

用户已实际确认 RA-L 与默认 `budget={}`；无隐含任务总量上限，没有付费 API、额外算力或投稿授权。本轮没有模型/API科研调用。

## 实测开发结果

| 记录 | 条件与实际运行 | 检查结果 | 结论边界 |
| --- | --- | --- | --- |
| `surfaces.v1.json` | 中立、左副翼、水平尾舵、垂直尾舵，分别0.2 s；固定初态、seed=0 | 测得舵面控制方向，使用模式−1独立控舵 | 不是闭环或任务资格 |
| `straight.v1.json` | 单机30 s、无风、加速物理 | 高度79.196–80.000 m，速度19.974–20.279 m/s，包线检查通过 | 无航线转弯或任务工具 |
| `orbit.v1.json` | 单机120 s、半径90 m、无风、加速物理 | 包线通过；10 Hz trace最大径向误差约3.016 m | 径向误差是诊断，没有事后将其设成资格阈值 |
| `four-orbits.v1.json` | 同一world四机600 s；共144,000物理步，每机独立圆 | 高度75.318–80.000 m，速度19.725–20.829 m/s；控制步检查未见碰撞或包线违规，最小机间距279.941 m | 这四个圆用于控制探针，未受800×800 m ROI约束，不是搜索或退出任务 |
| `clock.v1.json` | 四机20 s、地面5 s阻塞替代等待；未预热 | **失败**：5.005 s墙钟期间仿真推进5.917 s；最大滞后0.914 s | 启动积压后追赶，不能当作合格1×；JIT归因与预热结果相容，但尚无独立性能剖析证明 |
| `clock.v2.json` | 单独保留1 s预热记录，重建相同seed的全新world后开始20 s时钟 | **开发阈值通过**：5.005 s等待期间推进5.017 s、位移96.524 m；最大滞后0.010748 s，p95约0.003797 s | 阈值0.1 s仅用于本探针；不是正式冻结阈值，也不是实际模型时延或完整G3 |

全部飞行保持上游 PyFlyt 0.29.0 / PyBullet 3.2.7 动力学，通过舵面与电机施力。没有以传送位置、改写速度或运动学圆周代替飞行。均在平坦零高程地面进行；不将 world-z 在一般地形中等同于 AGL。包线摘要按120 Hz控制步检查，保存轨迹10 Hz；不是逐240 Hz物理步的完整硬安全审计。加速RTF只表示本地开发步进，不能推导实际模型主评测吞吐。

## 实现与验证

- `envs/inspection/control.py`：共享PI/PD执行器和圆周向量场，仅使用本机导航状态；当前只验证无风条件，尚未给出风下相对空速控制资格。
- `envs/inspection/scoring.py`：单个已固定关联实体的静态最终版本质量、动态完整60格、6 s宽限、先合法性筛选再按最新测量挑选、同观测时刻真值、格起点权重、observed独立挑样。它依赖 evaluator-only 的有效性标志，**尚未实现这些标志的实际传感/权限判定、报告名额与身份匹配，也未实现全实体聚合**。
- 新增32项工程测试，含实际PyBullet重置重放、1/4机首步异常与非有限状态、严格JSON保存，以及20项手算评分fixture。全仓库现有测试一次整体验证为57项通过，`compileall envs train eval scripts utils`通过。工程fixture不是正式任务样本。
- 独立上下文审查发现失败未落盘、终止现场缺失、四机首步失败含inf及摘要来源不清；对应修复与复验见 `review/control-review.v1.md`。未冒称跨模型或人类专家审查。`human_review_pending`。

所有已跑结果版本均保留，未用成功重跑覆盖失败。三条v1控制飞行的执行器源码归档于 `source/v1/`，其SHA与原飞行绑定一致；后续runner会在运行开始捕获源码字节并检查期间是否变化，`clock.v2.world.json`包含当时源码快照。历史 `clock.v1` 只保存了当时probe/runner哈希，未嵌入那一中间版本的源码，故可核对失败结果与worker文件绑定，不能声称可从当前源码精确重建其字节。运行环境、原始安装失败、成功安装、测试与编译日志见 `runtime/`。

## 复跑入口

在仓库根目录，先按 `planning/control-qualification.v1.md` 建立 `.venv`。输出路径必须是新版本，以免覆盖证据。

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q envs train eval scripts utils
.venv/bin/python eval/probe_inspection_surfaces.py --out /tmp/inspection-surfaces-new.json
.venv/bin/python eval/qualify_inspection_control.py --duration 600 --task orbit --aircraft 4 --out /tmp/inspection-control-new.json
.venv/bin/python eval/probe_inspection_clock.py --out /tmp/inspection-clock-new.json
```

当前代码有审查修复，重跑不应期待生成与v1历史runner完全相同的记录格式。PyFlyt seed=0使用显式 `np_random`，避免上游真值判断将0当作未提供seed。

## 下一项可检验工作

继续公开航线/退出圆切换与风包线验证，然后把实际视锥、遮挡、噪声与因果包到达接入；落实job后验完成、报告限额和固定身份关联及其正反例。只有非LLM参考任务在同信息下闭合，G1–G4与dev冻结通过后，才核定完整八配置及真实模型资源并进入正式实验。没有把旧RTA、工程fixtures或这些开发探针当成论文结果。
