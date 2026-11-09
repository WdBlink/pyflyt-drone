# Frozen evaluator v1 独立审查

审查日期：2026-11-06。结论：**7 条已完成 classic 飞行的 score 可由冻结的 17 个源文件重算一致；v1 的导入闭包冻结声明存在 P1 阻断，不能据此声称实际执行代码已被完整绑定。** 已复现同名 package 和未列入 manifest 的 bytecode 两种绕过。此结论只覆盖离线评分一致性，不授予 model、trace、episode 或 G4 admission。

## 范围与不可变绑定

本次仅新增本报告，所有反例在 `tempfile.TemporaryDirectory(prefix="frozen-evaluator-…")` 中运行并清理；未修改源码、manifest、world、terminal、ownerstate、live logs、controller，未启动 physics、LLM 或付费调用。工作区原有修改保留。只读取 7 个 `status=completed` terminal 对应的 `flight/world.json`；未把 pilot 的其他槽或部分执行算作完成。采用 good-benchmarks 的评分正确性及来源检查，不重做 G1–G3。

| 绑定对象（研究根 `paper/research/20261104-inspection/`） | SHA-256 |
|---|---|
| `experiment/evaluator.v1.py` | `1716592abd6a7ea317aebb90e41367f338e7f32ac7ab3f0a69a9d2bd4ff02a3d` |
| `experiment/evaluator-source-manifest.v1.json` | `ecc2069fc8795adb9096173ab2d859b6540a94309ecf99310755cb7b3177bf80` |
| `qualification/evaluator-actual-recount.v1.json` | `b53fbd3b4459200788ea87b486e8d29d4e18a31ae2227d18981db216e43c2d3b` |
| 闭包 `eval/inspection_common_calendar.py` | `c5246ab15bbeaf5ccb9baf6d9037e74a830072723b6f6421f2288ea8d50240ba` |
| 闭包 `eval/inspection_cases.py` | `7537b17aa0cbcfd4cffdb00c1d35e45e56217f477e055db77ba80724c0868cef` |
| 闭包 `eval/inspection_scenarios.py` | `d2b0fbfc77ee40e06115393624a3ac2f36d11fa21981590c71f3bc0beea5dcec` |
| 闭包 `envs/inspection/mission.py` | `715fce4a99325b6d26fbb1e9ffc20896603bfbd4d48396f0513a46f5ad14aec7` |
| 闭包 `envs/inspection/scoring.py` | `bc8a21947d340fce28c83c3d2313ad9a2e7711e6af07c1bd275f774009b5631a` |

manifest 的全部 17 个 `.py` 文件逐个复核 SHA，均匹配；审查前后 entry/manifest SHA 均如上。解释器为仓库 `.venv/bin/python`，Python **3.11.15**、NumPy **1.26.4**。manifest 写有 Python `3.11`，入口只显式验证 NumPy 版本，不验证 Python 版本或二进制依赖哈希；不把它称为完整运行环境冻结。

## P1：验证源文件哈希没有绑定实际导入模块

位置：`evaluator.v1.py:30–38`。入口验证 manifest 中列出的 `.py`，随后 `sys.path.insert(0, closure)` 并按常规模块名 import。它既不拒绝额外文件，也不验证所导入模块的实际来源/执行载荷。

**反例 A：同名 package 遮蔽。** 在隔离副本中保持 entry、manifest、17 个源文件全部字节不变，只新增：

```text
evaluator-source.v1/eval/inspection_common_calendar/__init__.py
```

内容：

```python
def _score(world, geometry, demand_calendar):
    return world["score"]
```

Python 将这个 package 作为 `eval.inspection_common_calendar`，而非已校验的同名 `.py`。使用实际 `dev-L3-00-B2-0` world 的临时副本，把 stored `score.Y` 从 `0.6271768508436419` 改为 `0.637176850843642`，给 CLI 传该临时文件的实际 SHA。正常干净副本：exit **1**，`ValueError: original full score recount differs`；新增 package 后：exit **0**，`score_equal: true`，输出错误 Y，同时仍报告原 entry SHA `1716592…` / manifest SHA `ecc2069…`。因此不是 source/manifest 被替换导致的预期哈希拒绝。

**反例 B：未绑定 bytecode。** 在另一份干净副本中，不新增 package；用以下参数将相同 payload 编译成评分模块的 `__pycache__/inspection_common_calendar.cpython-311.pyc`：

```python
py_compile.compile(
    str(payload), cfile=importlib.util.cache_from_source(str(checked_source)),
    dfile=str(checked_source), doraise=True,
    invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
)
```

17 个源 SHA、entry SHA 和 manifest SHA 仍全部匹配。fresh CLI 在 **`PYTHONDONTWRITEBYTECODE=1`** 下仍读取该缓存，exit **0**，再次接受错误 Y `0.637176850843642`。禁止写入 bytecode 不等于禁止读取已有 bytecode。

原冻结目录只读盘点得到 **33 个文件 = 17 个 manifest 源 + 16 个 `__pycache__/*.cpython-311.pyc`**。这不证明原缓存遭篡改；它证明原 manifest 没有覆盖全部可能执行的闭包载荷。本审查没有写入或删除这些缓存。为消除原缓存对正例核验的影响，另仅从 17 个校验后的源文件构建干净临时闭包，再对全部 7 world 重跑：**7/7 全 score 一致，且 7 次之后文件集合仍精确等于这 17 个源。**

修复验收应另开版本：执行 import 前检查闭包文件集合精确等于 manifest，拒绝额外 package、legacy `.pyc` 和 `__pycache__`，并防止正常运行自行增加缓存；继续验证文件字节。运行目录仍需保持不可变，或另行约束检查与导入之间的并发替换窗口。此修复建议不等于已经验证后续版本。

反例中重新提供临时 world SHA，只用于隔离测试评分器是否实际重算；若可信外部 receipt 仍绑定原 world SHA，原入口的 hash gate 会拒绝改过的 world。P1 影响的是**冻结代码身份**，不构成对原 7 个实际 world 或 receipt 被篡改的指控。

## 实际 7 条重算及固定分母

实际执行方式为 fresh subprocess，argv 为仓库 `.venv/bin/python`、entry 绝对路径、`--world`、completed world 绝对路径、`--expected-world-sha256`、recount 记录 SHA；环境设置 `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`，工作目录 `/tmp`，capture stdout/stderr。示例参数对应：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python paper/research/20261104-inspection/experiment/evaluator.v1.py \
  --world logs/inspection-classic-dev-flights.v1/dev-L3-00-B2-0/attempt-00/flight/world.json \
  --expected-world-sha256 6db6b46c31c8295ebedaec55548f566f97847d7a09c9fe41444b1c6377c66b4c
```

原入口 7 次、仅 17 源干净副本 7 次，全部 exit **0**。每次 parsed JSON 的**完整 score dict** 均等于 world 内 score，也等于 recount 的完整 score；不只比较 Y。7 个 terminal 文件 SHA 均等于 recount 记录，world SHA 同时等于 recount 的 world SHA 和对应 terminal `record_hashes["flight/world.json"]`。这只是只读绑定核对，没有重新执行完整 receipt cache admission。

| completed episode | attempt | Y | static/dynamic 目标数 | static / dynamic 权重分母 | dynamic 零质量 bin（仍入分母） |
|---|---|---:|---:|---:|---:|
| dev-L1-00-B0-0 | 01 | 0.9970187393526406 | 4 / 0 | 4 / 0 | 0 |
| dev-L2-00-B0-0 | 00 | 0.7845597455424045 | 5 / 3 | 5 / 180 | 115 |
| dev-L2-00-B1-0 | 00 | 0.5789292595650977 | 5 / 3 | 5 / 180 | 153 |
| dev-L2-00-B2-0 | 00 | 0.5582428561233956 | 5 / 3 | 5 / 180 | 157 |
| dev-L3-00-B0-0 | 00 | 0.8245752535103837 | 5 / 3 | 9 / 331 | 91 |
| dev-L3-00-B1-0 | 00 | 0.65594260176561 | 5 / 3 | 9 / 331 | 161 |
| dev-L3-00-B2-0 | 00 | 0.6271768508436419 | 5 / 3 | 9 / 331 | 178 |

独立算术脚本未 import 评分闭包：从 private case 的全部 `rule_id` 非空实体生成 static/dynamic key 列表，与 score 中列表逐项对应；对每个动态实体独立生成 `appeared_at + 3*i`、`i=0..59` 的 60 个 bin，核对 index、起止时刻和优先级权重。每域本例各 flight 的实体及全部 bin 都相符，未以有上报实体或有成功测量 bin 缩分母。L3 `priority_at=196`；动态出现时刻为 `240/121/186`，分别贡献 `120/95/116`，合计 **331**；static-0 在终点权重为 3，其余目标权重为 1，合计 **9**。

再独立按实体/bin 表求加权 S/D、按所有按时到达覆盖 packet 的 cell 并集求 C、按冻结公式求 Q/H/E/Y/SR，7 个 flight 的每个标量及两个分母均精确相等。L1 的 D 为 `None`、Q=(C+S)/2、SR 不含动态门限；L2/L3 的 Q=(C+S+D)/3。H 来自 violations 为空，E 要求四机退出停留至少 `20-1e-6` 秒且余航程至少 1000m。

代码路径也核对到：`_score` 重建 ReportLedger，按到达时刻注册 delivered evidence，逐条提交 report 并比较 sealed response；静态选择最终合法版本，动态选择 bin 内合法及时的最新证据之后再判断 identity/class/quality/localization，缺失记零，重算 delivered 与 observed quality、实体表、分母及 association ambiguity。构造使用 `ScenarioMission.__new__`，没有调用 mission/aircraft 的 `__init__`，这次执行没有新物理 episode。

## 隔离反例结果与准确边界

除 P1 外，以下均在干净、未遮蔽的副本/原入口上执行，world 内容探针默认传临时文件重新计算的 SHA，以区分 SHA gate 与业务重算 gate。

| 探针 | 实际结果 |
|---|---|
| listed `.py` 追加注释 | exit 1，`frozen evaluator import differs: eval/inspection_common_calendar.py` |
| manifest 追加换行 | exit 1，`frozen evaluator manifest differs` |
| 修改 world，仍传原 SHA | exit 1，`world differs from the caller's admitted receipt` |
| 仅改 private entities 中一份 true_class | exit 1，`private geometry and original world differ` |
| 各自改 stored C/S/D/Q/H/E/Y/SR/recall/ambiguities | 各 exit 1，`original full score recount differs` |
| 改 static quality、删除 static 实体、改两个分母 | 各 exit 1，同上 |
| 改 dynamic quality/observed、删除 bin、改 bin weight | 各 exit 1，同上 |
| 新增一个未覆盖 required cell 键而不修 score | exit 1，同上 |
| coverage 指向 required_cells 外的键 | exit 1，`coverage contains cells outside the frozen demand` |
| delivered evidence 重复 ID | exit 1，`duplicate delivered evidence identity` |
| 修改 sealed report response | exit 1，`sealed report reconstruction differs from original response` |

两次初版探针生成器误把 `required_cells` 的 dict / `association_ambiguities` 的 int 当 list，中止于探针构造，尚未启动相应 evaluator 子进程；纠正类型后分别重跑遗漏项目。未把这两次 harness 错误计作评分器的拒绝能力。

**以下是 score-only 范围，不是已实现的 provenance 拒绝：**

- `_score` 的 remaining range 直接读取 `world["score"]["remaining_range_m"]`，经 `18000 - remaining` 再求回；不是从实际轨迹独立积分。临时 world 将首架值 `7895.256409585785` 增 1m，exit 0，score 输出变为 `7896.256409585785`，Y 不变。故“全 score 相等”不意味着这个测量独立复算。
- H/E 使用已记录的 `world.safety.violations`、`exit_dwell_seconds` 与上述余航程，未重放安全/退出/240Hz trace；coverage 使用记录的 packets 和 required cell **键**集合，未从相机轨迹重建检测覆盖。
- 入口只用 SimpleNamespace 的 level/priority_at/entities，没有执行完整 `CaseSpec.from_dict`、`validate_case`、digest 比较或由 ROI 重建网格。将 world 的 `case_sha256` 改为 64 个 0、将 private ROI 改为小三角形、只改 required_cells 坐标但保留键、同时将两份非目标 static entity 的未使用 radius 改为 12345，分别传新 world SHA后均 exit 0、原 Y 不变。因此它验证两份 entity 记录彼此相等及其参与评分的结果，不能声称拒绝所有一致篡改或不相关字段变更。
- 外部可信 admitted receipt 的 world SHA、case 来源及 worker/flight/trace 的独立 admission 仍是必要前提。CLI 的 `--expected-world-sha256` 是调用者提供的字符串；入口本身不读取 receipt/模型来源来证明它可信。NumPy 只验版本，Python环境、外部依赖载荷和潜在恶意解释器不在本次闭包保护保证内。

这些边界与入口明示 `offline recount, not new rollout or provenance admission` 一致；如未来要声称 ROI 网格、canonical case、余航程或 safety trace 已独立审计，需独立接线和证据，不能由本次数值相等推导。

## 交付判定

1. **原 7 个 completed world 的评分数值与实体/固定 bin 分母：通过此有界复算。** 干净源副本也得到完全相同结果，未发现这 7 条 score 算术错误或漏失败 bin。
2. **v1 的冻结执行闭包防篡改：不通过，P1 未在 v1 关闭。** 原版本保留，后续版本需要独立定向复验。原 7world 的一致性结果不因该实现缺陷自动被判假，也不能反过来关闭该缺陷。
3. **不作 G4、真实 model/actor/trace、SDK 全矩阵、费用、正式统计或方法优劣认证。** 本报告没有创造新 flight，也不扩大真实飞行样本数。

Attention / reviewed by inherited-parent model：独立上下文、同模型家族的代码与可执行探针审查，不是跨模型盲审或人类专家认证；覆盖边界为上述固定 v1 及 7 个完成 world，未覆盖后续 evaluator 修订、其他运行槽或 G1–G3 重验。
