# Frozen evaluator v2 定向修复复验

日期：2026-11-06。结论：**v1 的 P1 同名 package / 未列入 manifest 的 bytecode 遮蔽反例，在下述固定 v2 上均被拒绝；原 7 条 completed flight 的完整 score 仍全部重算一致。** 新增 case semantic digest 和 ROI 网格绑定也通过对应反例。此为有界修复复验，不授予真实 model/trace/episode 或 G4 admission。

## 版本绑定与执行范围

| 对象（研究根 `paper/research/20261104-inspection/`） | SHA-256 |
|---|---|
| `experiment/evaluator.v2.py` | `de826d2488b5a8ad366ccec9ead8b81bebf6b794a533a38ac55338916848611b` |
| `experiment/evaluator-source-manifest.v2.json` | `ecc2069fc8795adb9096173ab2d859b6540a94309ecf99310755cb7b3177bf80` |
| `qualification/evaluator-actual-recount.v2.json` | `b822934f04fad9371e1f9ab23a8337215fa78237da9516b863c91c470d71ffe4` |
| 保留的 `review/frozen-evaluator-review.v1.md` | `b99c37e742086fc75acd2e85d9466e1b93da754dc50b1c66062a313c6af43102` |

v2 manifest 与 v1 manifest 字节相同，17 个列出源文件的哈希约束相同；评分本体未改。解释器为仓库 `.venv/bin/python`，Python 3.11.15 / NumPy 1.26.4。审查前后 entry/manifest 哈希如上；实际 v2 闭包文件集合仍精确等于 manifest 的 17 项，`.pyc` 为 0。

本次只新增本报告。原 v1 报告、冻结源码、manifest、flight/world/terminal、live logs、ownerstate 和 controller 未改。反例全部在临时目录中复制的 v2 上运行并清理。没有 physics、模型生成、付费 API 或新 flight。

## P1 的定向关闭证据

v2 在任何闭包 import 前检查递归文件集合精确等于 manifest，并拒绝后代 symlink；随后验证每个源 SHA。`sys.dont_write_bytecode=True` 位于闭包 import 前。未仅依赖禁写 bytecode 来防读取已有缓存。

实际 fresh subprocess argv：`.venv/bin/python <v2 entry 绝对路径> --world <world 绝对路径> --expected-world-sha256 <文件实际SHA>`，cwd `/tmp`，`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`。**子进程环境显式删除 `PYTHONDONTWRITEBYTECODE`**，以核验入口自身禁写；外层审查脚本仍设置它以避免自身生成无关缓存。正常重复使用同一临时闭包运行 3 次，全部 exit 0、同一完整 score；之后仍只有 17 个列出的源文件。

下表除另述外，使用实际 L3-B2 world 的临时副本，并传其真实 SHA。遮蔽 payload 与 v1 一致：`_score` 直接返回 `world["score"]`；stored Y 改为 `0.637176850843642`。

| 隔离反例 | v2 实际结果 |
|---|---|
| 新增 `eval/inspection_common_calendar/__init__.py` 同名 package | exit 1，`unlisted, missing or symlinked evaluator file` |
| 新增 UNCHECKED_HASH `__pycache__/inspection_common_calendar.cpython-311.pyc` | exit 1，同上 |
| 新增 legacy `inspection_common_calendar.pyc` | exit 1，同上 |
| 新增无关 `README.txt` | exit 1，同上 |
| 删除 manifest 中的评分源文件 | exit 1，同上 |
| 将列出的评分源文件换成指向同字节外部文件的 symlink | exit 1，同上 |
| 评分 `.py` 追加注释，文件集合不变 | exit 1，`frozen evaluator import differs: eval/inspection_common_calendar.py` |
| manifest 追加换行 | exit 1，`frozen evaluator manifest differs` |
| 无遮蔽，只改 stored Y | exit 1，`original full score recount differs` |

原 P1 的两种实际错误代码执行路径被文件集合 gate 阻止，拒绝发生在 import 前。本结论针对这些固定反例及审查时不可变文件，不扩大为对恶意解释器、预加载模块或任意并发目录替换的保证。

## 新 case / ROI 绑定

v2 对 private case 按 `json.dumps(private, sort_keys=True, allow_nan=False)` 计算 SHA，并与 `world.case_sha256` 比较；按冻结 `coverage_cells` 从 ROI 重建完整 20m 网格映射，比较全部 cell 键及坐标值。之后仍比较 private/hidden 两份 entity 数组，并完整重算 score。

| 隔离反例 | 实际结果 |
|---|---|
| case_sha 改成 64 个 0 | exit 1，`worker semantic case digest differs` |
| 改 private entity 的 radius，保留旧 case_sha | exit 1，同上 |
| 改 private entity 的 radius，并重算 case_sha，但不改 hidden copy | exit 1，`private geometry and original world differ` |
| ROI 换小三角形，并重算 case_sha，保留原 required_cells | exit 1，`required cells differ from the frozen ROI/lattice rule` |
| 只改 required_cells 坐标 | exit 1，同上 |
| 只新增 required_cells 键 | exit 1，同上 |
| world 内容改变，仍提供原 world SHA | exit 1，`world differs from the caller's admitted receipt` |

这关闭了 v1 报告列出的 stale digest / ROI 与 cell 映射脱节情况。它没有执行完整的 `CaseSpec` 解析及 `validate_case`，也不从可信 receipt 自行取得 expected world SHA；semantic digest 与给定内容自洽，不等于外部身份已被准入。若调用者同时替换内容、自洽 digest、derived score 并提供新 expected world SHA，来源保证仍须由上游 admission 提供。

## 原 7 flight 复算

对 `qualification/evaluator-actual-recount.v2.json` 中全部 7 条记录实际启动原 v2 entry：

| completed episode | exit | Y | 与 v1 同一 terminal / world SHA | 完整 score 等于 v2记录及world |
|---|---:|---:|---|---|
| dev-L1-00-B0-0 / attempt-01 | 0 | 0.9970187393526406 | 是 | 是 |
| dev-L2-00-B0-0 / attempt-00 | 0 | 0.7845597455424045 | 是 | 是 |
| dev-L2-00-B1-0 / attempt-00 | 0 | 0.5789292595650977 | 是 | 是 |
| dev-L2-00-B2-0 / attempt-00 | 0 | 0.5582428561233956 | 是 | 是 |
| dev-L3-00-B0-0 / attempt-00 | 0 | 0.8245752535103837 | 是 | 是 |
| dev-L3-00-B1-0 / attempt-00 | 0 | 0.65594260176561 | 是 | 是 |
| dev-L3-00-B2-0 / attempt-00 | 0 | 0.6271768508436419 | 是 | 是 |

逐项核对 terminal 文件实际 SHA = recount terminal SHA；world 实际 SHA = recount world SHA = terminal record_hashes 中的 world SHA；stdout 的 evaluator SHA = 记录 SHA = 上述 v2 SHA。没有将重算计为新 flight。17 源哈希与完整 score 未变，因此 v1 对这些相同 world 的独立实体 key / 60 个固定动态 bin / L1分母4/0、L2分母5/180、L3分母9/331的算术复核仍适用，未重复扩成物理验证。

## 保留的准确边界

1. **已检查文件集合和后代 symlink，不是所有路径/目录形态均禁止。** 临时闭包根目录本身替换成 symlink，目标仍是那 17 个原源时，exit 0；新增空 `__pycache__` 目录也 exit 0。原因是 `rglob` 没有检查 closure 根本身，且集合只收集 `is_file()`。这两个探针未导入额外载荷或接受错误 score，不是原 P1 的复现；若合同要字面禁止根 symlink/任何额外目录，需再加相应 guard。当前可声称的是 17 项实际源文件集合及后代 symlink 校验。后续新增文件仍须在执行前重验。
2. **remaining range 是继承输入。** 首架 stored remaining `7895.256409585785` 增 1m，并传临时 world 新 SHA，仍 exit 0、Y 不变；与 v1 相同，不是实际飞行里程积分重放。H/E 继续依赖记录中的 safety/exit/remaining primitives。原始 trace、传感器生成、物理因果链及 model/actor 行为未在此入口重验。
3. **运行环境保证有边界。** NumPy 检查版本 1.26.4；manifest 的 Python 3.11 字段未显式执行版本检查，也未绑定整个解释器/NumPy 二进制或启动环境。要求 fresh process；任意恶意预加载或检查后并发修改不在本次定向复验保证内。

判定：固定 v2 上原 P1 定向关闭；新增 case/网格 guard 与原 7 条 score 复算通过。原 v1 报告保持原判定。没有发现需要把这次数值重算撤回的残留错误代码执行反例，也没有据此批准 G4、正式八配置成绩、actual model/trace 或后续源码版本。

Attention / reviewed by inherited-parent model：同模型家族独立上下文的有界代码与实际文件探针复验，非跨模型盲审或人类专家认证；只覆盖上述固定 v2 和 7 个已完成的原 world，不认证未来运行或其他 upstream admission。
