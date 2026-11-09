# 有界数值推断 v2 独立复验 v1

日期：2026-11-05。结论：**有界公式、两类区间、固定global6家庭及严格 `RootTable.parse → analyze_roots` 主路径得到支持；旧bootstrap已被隔离为诊断。发现独立helper遗漏拒绝 `np.bool_` 的窄输入缺陷，待小版本修补复验。此结论不授予G4。**

本次复用前次独立审查上下文，与主Agent分离但属于同模型家族，不是跨模型家族审查。只新增本报告，不改production、规划、旧记录或审查脚本。执行均为只读/内存中的纯CPU数值复算，没有物理、模型/API、正式方法、test或真实holdout测量。

## 1. 冻结对象与证据

| 文件 | SHA256 |
| --- | --- |
| `eval/inspection_statistics.py` | `4a6610e45a23ceed2e4e27ed2b96162837ae3c45bc223555ac70b430a1ed8c44` |
| `eval/preflight_inspection_statistics.py` | `be41dff720aeb237d1f26c7401befe946f680ea0cc5a3c700236f49a6980b52d` |
| `tests/test_inspection_statistics.py` | `3e0b5dd18a2b4fc2fed692f5022fb23b183f5ca1807caa4b5b2409ebda206377` |
| `planning/statistical-protocol.ral.v1.md` | `b18123390584f3f260d9bcc80b0455402ccc68f606ad428fe06804e04d2e795f` |

已读取 `qualification/statistics-preflight.v2/report.json`、`analysis.json` 与 `source-snapshot.json`，核对其3份来源的快照SHA、报告SHA和当前文件SHA一致。本审查不重写该旧预检。原800组合成效应输入来自前次 `review/statistics-implementation-v1/results.v1/stress-replicates.jsonl`，先验证其SHA仍等于原 `review-freeze.json` 所绑定值。

本次实际复验环境为项目 `.venv/bin/python`，Python3.11.15 / NumPy1.26.4；检查前后统计源码SHA未变。主Agent已获知helper问题，并承诺待本报告冻结后另作小版本修补；此报告不提前批准该修补。

## 2. 公式与区间的独立核对

域内K等级、每层n_l个root，配对差 `d_lr∈[-1,1]`、权重 `w_lr=1/(K n_l)`，因此 `W=Σw²=Σ_l1/(K²n_l)`。每个独立加权项的范围宽度为2w。Hoeffding两侧界为 `P(|delta−Edelta|≥t)≤min(1,2exp(−t²/(2W)))`；检验弱宏均值零假设的保守基础p就是 `min(1,2exp(−delta²/(2W)))`。

`bounded_reference:112-118` 与该式一致；`:152`按实际各层root数计算W。pointwise95半宽为 `sqrt(2W log40)`，六项同时95半宽为 `sqrt(2W log240)`，后者来自Bonferroni union bound，**不是Holm反演**。裁剪到[-1,1]保持对同一有界参数的覆盖性质。

| 域 | W | pointwise95半宽 | Bonferroni-global6同时95半宽 |
| --- | ---: | ---: | ---: |
| ID，16/16/16 | 1/48 | .392050137849843 | .477870228345782 |
| OOD，12/12 | 1/24 | .554442622077489 | .675810557980933 |

独立复算28个标量组合：delta为−1/−.9/−.5/0/.5/.9/1，配以16/16/16、12/12、2/3/5、2/5四种层数配置；p及两类区间与单独 `math.fsum/log/exp` 算式一致（容差1e-15）。δ=1、W=1/48时，pointwise区间[.607949862150157,1]，simultaneous区间[.522129771654218,1]；δ=−1镜像夹边正确。δ=0时p=1且区间对称。

有限样本保守控制仍以独立root差、固定已冻结权重、完整分母与无可选停止/挑选为前提，不要求方法标签交换、正态、相同repeat或六假设间独立。严格保证来自该不等式及Holm/union bound，不来自几个合成场景通过。[Hoeffding原论文](https://www.cs.rpi.edu/academics/courses/spring06/random/hoefding.pdf)，[R官方Holm说明](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html)

## 3. 主字段与bootstrap角色隔离

源码`:155-157`将bound的p送入主 `p_values`，bootstrap p另存；`:166-172`分别调整两套家庭。确认性输出只为顶层 `p_raw`、`p_holm_global6`、`reject_primary_alpha05`。bootstrap p仅位于 `bootstrap_reference_diagnostic`，其校正字段为 `p_holm_global6_not_confirmatory`；区间改名 `ci95_bootstrap_pointwise_nominal`。旧 `reject_nominal_alpha05` 和含糊的 `ci95_pointwise` 在v2主输出中不存在。

独立遍历v2预检六项输出，将每个effect/W重新代入有界式并手写Holm排序/累积max，主p、主Holm及主拒绝逐项一致，固定家庭大小6。`:130-131`还要求ID/OOD各自全部预声明等级，避免缺层悄悄改宏平均。

协议`:15-27`明确有限界前提、区间角色、bootstrap降为诊断以及正式数据前根据189/200公开反例修订；`:7-11`保留8配置、1/3重复、72主root、两域等级宏平均及3×2家庭；没有据正式方法结果变更N、选择方向或缩小家庭。效应正负仍须由下游单独解释，负效应拒绝不是改善。

## 4. 零、常量、非平衡及非法输入

完整72root纯常量数据（31draws，仅用于检查诊断独立性）得到：

| 每root六项差值 | ID主p / Holm | OOD主p / Holm | bootstrap诊断p |
| --- | ---: | ---: | ---: |
| 0 | 1 / 1，均不拒绝 | 1 / 1，均不拒绝 | 1 |
| +.5 | .00495750 / .02974503，拒绝 | .09957414 / .29872241，不拒绝 | 1 |
| +1或−1 | 7.55027e−11 / 4.53016e−10，拒绝 | 1.22884e−5 / 3.68653e−5，拒绝 | 1 |

这确认主有界p不受bootstrap零SE规则污染。非平衡小夹具（仅 `require_main_shape=False` 测试）ID层数2/3/5、各层差.1/.2/.3，得到宏差.2、W=31/270，区别于pool差.23；OOD层数2/5、差.2/.3，得到宏差.25、W=.175。正式输入仍要求原16/12形状。

14个Python标量非法边界均拒绝：delta为True/False/NaN/±inf/越过±1，或W为True/False/0/负数/>1/NaN/inf。

**剩余缺陷 B1（窄helper输入边界）：** `bounded_reference:108-110` 只用 `isinstance(x,bool)` 拒绝布尔，遗漏NumPy布尔。独立复现：

```python
bounded_reference(np.bool_(True), 1/48)["p_raw"]  # 7.550269088558195e-11，被接受
bounded_reference(.5, np.bool_(True))["p_raw"]   # 1.0，被接受
```

`RootTable.parse:43`已拒绝Python/NumPy布尔，`analyze_roots:155`明确把effect转Python float，W由层数/计数算出，因此**严格默认parse→analyze路径不受B1影响**。这不是有界数学否证，但helper是可独立调用的数值入口，接受非法布尔效果可制造看似很小的p，应在下一小版本将两个检查扩为 `(bool,np.bool_)`，补delta/W两类回归例。未修的4a6610版本不应宣称其独立helper已满足全部非法输入拒绝契约。

## 5. 原四场景与rare典型的复核

未重复800组bootstrap。只对原冻结800组的六项effect，以新有界主公式、独立W和手写Holm复算；每个场景200组均为：

| 场景 | 新主Holm6全家庭误报 | pointwise区间覆盖真0 | 同时六区间全部覆盖真0 |
| --- | ---: | ---: | ---: |
| symmetric_uniform_null | 0/200 | 1,200/1,200项 | 200/200组 |
| skew_episode_null | 0/200 | 1,200/1,200项 | 200/200组 |
| rare_root_loss_null | 0/200 | 1,200/1,200项 | 200/200组 |
| shared_zero_gate_null | 0/200 | 1,200/1,200项 | 200/200组 |

1,200项互相关联，不当作1,200个独立覆盖试验；单场景零误报200次的Wilson95%上界约1.88%，不能据此宣称真实误报率为0。原bootstrap的189/200结果及源快照保持原样，未被新表覆盖。

另外重新生成固定rare rep0/1，各只跑999draws核查真实输出角色。OOD-M1分别effect=.01956243/.01940474，bootstrap诊断p=.001、诊断Holm=.006，nominal区间仍在0以上；但两例主p=1、主Holm=1、`reject_primary_alpha05=false`，有限界同时区间覆盖0。原小p得以保留供诊断，同时不再成为主结论。

## 6. 实际检查、验收范围与决策轨迹

已实际执行：项目环境 `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_inspection_statistics.py`，**7 passed in 0.11s**；一次内存独立审查脚本完成上述28个算例、14个非法边界、v2快照/输出核对、800组旧effect重算、常量/非平衡和两个rare小复跑；源码前后SHA一致。没有新增脚本/结果目录，复算摘要保存在本报告，原输入可沿原冻结清单复核。

有界修订的数学与严格主路径可在其前提下作为预注册确认性候选；仍应先完成B1的独立helper小修补与短复验，再宣称该数值入口完整验收。低功效、宽区间、未拒绝不等效、不得见结果后扩N的限制保留。

真实generator/lineage/split/episode/provider/费用、经典dev选择、硬中断resume、SR、事件/权重日历及恢复删失均不在本次数值验收范围，未获认证。协议自身也明确这些尚待完成，因此本报告不授予G4或正式运行授权。

| 决策/检查点 | 依据 | 结果 |
| --- | --- | --- |
| 复核新主p而不重复原800组bootstrap | 原冻结effect及独立有界算式 | 保留反例，独立核对新输出 |
| 用常量正/负差分离主检验与诊断保护 | SE0但主p按固定界计算 | 主/诊断角色正确 |
| 不将np.bool_遗漏扩大为严格主路径故障 | helper复现及parse/float调用方 | B1窄缺口已通知主Agent |
| 冻结4a6610版本，不提前批准后续修补 | 本报告第1节SHA及前后检查 | 待小版本guard复验 |

Attention

reviewed by inherited-parent model（同模型家族独立上下文，执行环境未提供可核验的具体模型ID）

- B1的NumPy布尔拒绝仍待修；其余结论只绑定本报告来源SHA。
- 合成复核支持实现，有限样本有效性依赖真实独立有界前提；本报告没有证明真实数据满足这些前提，也没有跨模型家族二次审查。
