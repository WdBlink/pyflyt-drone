# 有限固定弧执行器 F4 修复后有界复核 v3

证据冻结：2026-11-05 10:29:34 UTC（工具时钟）。审查者为独立上下文 `/root/finite_arc_review`。保留 v1/v2；本次只写 v3，没有修改 production、契约、旧结果或研究状态。

**F4 的两类确定初始化反例在本次有界输入中已修复。九个代表风向的36个初始 program与18个提前 exit/survey 请求均通过时间、有限性、速度斜率和全 fleet 几何检查。没有发现本轮新的几何阻断；这不是完整连续风域、G1 或 G3 的放行。**

当前 follower 还新增 requested airspeed 上限18.5 m/s候选。它改变闭环行为，旧24 m/s上限的短探针和 pilot 不能自动作为新候选的物理资格。原6 m/2 m及 hard 安全要求继续保留。

## 版本绑定与改动核验

| 文件 | 当前复核 SHA-256 |
| --- | --- |
| `envs/inspection/planning.py` | `fdc2a43ec6772b5657f8b2c2a96470a15084d03eca4753b7c44ad2f0eb9400e8` |
| `envs/inspection/paths.py` | `f4c20fb4b8952856cd90a07b065add4ba5ee089bcb120940781c19b2df650ca2` |
| `envs/inspection/execution.py` | `72611b32fb01e3679fa616f2fd51969508ab4d263718a5810f70ea37ab4294e9` |
| `envs/inspection/mission.py` | `491fe4c8123241e61a3d6392bfeea618604f9bb2717df035d095e0506ca550da` |

对比 `qualification/source/finite-arcs-v13/`：planning 唯一增量是在 `smooth_own_joins` 的窗口内点使用 `np.arange(0.,seconds-1e-8,.05)`，随后仍追加精确 endpoint。该修复消除端点旁的近重复内点，未绕开严格时间检查、未更改几何端点/PID或放宽27 m/s前提。paths 增量为 follower airspeed clamp及说明。F1 bank逆解、F2 course门、seam日志与v2所复核实现保持一致。

## 实际执行的 F4 复验

只运行纯几何，无物理步。风向为 calm、±X/±Y各2 m/s，以及四个对角方向各2 m/s。每个方向构造全部四个home中心（−300/−100/100/300），并构造 drone 1 在 `now=29.5,commit_at=30` 的 exit 和 survey 请求。

原始失败输入 `initial_program(-300.,(0.,-2.))` 与 `initial_program(-100.,(-sqrt(2),-sqrt(2)))` 均完成构造；九方向共36个初始program、18个完整请求均成功，18个请求均被准入。

| 检查 | 观测结果 |
| --- | --- |
| 所有 prefix 数值 | 全部有限，样本数≤16000 |
| 所有 prefix 时间 | 严格递增；最小间隔`0.0005444425413259069 s` |
| 所有初始/请求 prefix 插值斜率 | 最大`20.051827914883646 m/s`，≤27 m/s |
| nominal bank | 最大`28.672481417375792°`，≤30° |
| prefix→tail 接缝 | XY/speed/bank匹配；course按wrap匹配 |
| 同一风向四机 tail 周期 | 最大差<1e-9 s |
| tail 重复一周期 XY 误差 | 最大`1.0859919924850373e-12 m` |
| `xy_at` 对标量 `at(t)[:2]` | 全部通过`allclose(atol=1e-10)` |
| 四机完整初始 fleet | continuous lower≥27 m且fence upper≤544 m |
| 18个请求的完整 fleet | 由同一真实`plan/fleet_clearance`入口检查并准入 |
| exit gate bank | 新旧差<1e-10 rad |

此前两个失败方向不再产生零时间间隔，也未通过删除几何或吞掉异常来规避。上述检查覆盖九个代表方向，不是对所有可能实数风向或所有未来姿态的穷尽证明。四个主要文件另经内存AST解析通过。主 Agent 告知的136项全套通过未在本次重跑。

## 当前物理事实与新增 follower 候选

只解析已完成 `qualification/executor-pilot.v8` 的两个结果，没有重跑或改写。root100 delivered D=`0.8555555556`、XY peak=`3.291724 m`、零XY越管；root101 D=`0.8166666667`虽然超过.8，XY peak=`6.689173 m`、955个物理步越6 m管，仍失败。两条`seam_capture_failures=0`。后者说明入口点通过不能替代整个闭环过程保持管内。

root101峰值witness为`t=278.4916666667,drone=0`。本次仅核对保存字段，没有独立重建该时刻的完整动力学并证明因果根因。主 Agent 对phase追赶与bank饱和的解释仍是待否证假设。

当前 `paths.py:252` 将 requested airspeed clamp从16..24改为16..18.5，nominal仍18、实际hard仍15..25。它可能改变相位收敛、转弯与持续观察，必须作为新profile取得独立绑定的物理结果。`cell-recovery.v2`使用旧paths正文；v2中对其36000行与入口状态的独立复算继续有效，但适用版本不迁移。本轮没有新物理/1×/API/600 s或全套测试证据。

## Attention

reviewed by GPT-6 family（同家族独立上下文）

- F4在本次确定反例及代表方向中关闭；保留v2原失败记录。
- v8 root101仍失败，不能因D过线或seam计数为零而判通过。
- 新18.5候选未在本次获得物理资格；G1/G3及完整风域/任务资格仍待执行相应证据。
