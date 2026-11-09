# G4回执guard与SR适配独立复核 v1

日期：2026-11-06（Asia/Shanghai）。**原P2的两个missing-run反例均关闭；SR数值适配在其显式合成/描述性契约下通过复核；现存L3-B0回执与缓存记录一致。** 本报告不授予G4，不把一条dev回执或合成SR矩阵当正式八配置结果。

与主Agent分离、同模型家族独立上下文。只写本报告及`review/g4-receipt-sr-guard-v1/`证据，不改实现/旧报告/真实回执。测试仅为轻量文件、合成JSON子进程及小规模CPU统计，没有新物理、模型/API或重负载。真实回执核验完全只读，没有调用可启动物理的执行入口。

## 1. 冻结对象与实际检查

| 文件 | SHA256 |
| --- | --- |
| `eval/inspection_episode_receipts.py` | `59f88d5a3ec90497e19ec91ac9b914f09fe6a2303d029e6aecbb6fb2f958387b` |
| `tests/test_inspection_episode_receipts.py` | `447cb8c08cb592a562cf88528c3c7c7f39897182cc9ded83827e4127b3d1a71a` |
| `eval/inspection_secondary_statistics.py` | `b056b9d42223651c9d467ba332ee7803505bc3b7a382f69e3c83bdd197c68ab1` |
| `tests/test_inspection_secondary_statistics.py` | `5711b3099d9c4e7ba31262a931f44aa197003fbe7c8a1dac0aa05ff911760252` |
| 依赖`eval/inspection_statistics.py` | `70c8783be6f6665b81b1190d6b905f4a22938445cc4c641458202bab5c710545` |
| 依赖`eval/inspection_recording.py` | `9613a8e800d470110dbf86890837ba853c3f50834ae29f0722994927d27cb352` |

六份源快照、前后SHA、独立探针及结果存于`review/g4-receipt-sr-guard-v1/`。检查前后来源未变。实际命令：

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider --basetemp=paper/research/20261104-inspection/review/g4-receipt-sr-guard-v1/pytest-temp tests/test_inspection_episode_receipts.py tests/test_inspection_secondary_statistics.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python paper/research/20261104-inspection/review/g4-receipt-sr-guard-v1/probe.py
```

**24 passed in 0.52s**（receipt16项、SR8项）。独立探针另完成两类旧反例、真实回执只读核对、完整合成SR分母和标量分位数复算。运行环境为项目Python3.11 / NumPy1.26.4。

## 2. P2共享根因关闭

`validate_flight:87-99`现在分别判断现存复制case的字节SHA、现存world声明的worker case SHA。二者不依赖run存在；只有run→world_sha关联仍需两份文件同时存在，符合各证据可用性的边界。

独立重放原两类组合故障：

| 轻量合成反例 | 当前结果 |
| --- | --- |
| run缺失，复制case存在但字节错误 | 1次worker调用，保留worker_operational_failure及input_or_condition_mismatch，不允许retry |
| run缺失，复制case正确但world.worker case SHA错误 | 1次worker调用，保留worker_operational_failure及world_binding_mismatch，不允许retry |

均抛`source/input/score defect blocks the episode`，未补第二次、未产生合法Y。原P2由同一门控根因修复，无需修改旧反例记录。此前R1/R2/R4的结论继续限于已测缓存链、进程清理与实际fsync/process-recovery范围，不外推物理掉电认证。

## 3. 现存真实L3回执的缓存链

只读核对`qualification/dev-receipts.v2/dev-L3-00-B0-0/`及`/tmp/inspection-L3-cache-reuse.v2.log`，结果：

- 仅有attempt-00，terminal为completed、returncode=0；intent/started/terminal/log的plan SHA均为`a8ec212bdaa5f61fdc2fb88baa143729e016c579540abee473d986b53ead20a5`。
- terminal SHA为`6d3bdd5e4abbaedf298f9f6eab9d68636043dae32b86397155191e55a1bfdbd6`，与cache日志一致。日志为cache_reused=true、actual_attempts=1，没有第二个physical attempt的证据。
- intent/start/stdout/run/world/private-case六文件哈希均匹配terminal.record_hashes，stdout独立字段一致。run内冻结source文本逐份重算SHA与run/plan source_hashes一致；依赖哈希、线程环境、case绑定一致。
- 现存world记录wallclock_1x、600s、144000物理步、240Hz物理/120Hz控制，max lag约.030503s。这里只核对现存证据，未重新物理执行或独立认证控制/计分器。
- 原冻结validator独立读回该产物，重算Y=`.7541293810345713`、evidence与terminal及cache日志一致；当前guard对同一完整产物也返回相同校验结果。这一Y只用于单条回执一致性，不是方法比较成绩。

**历史版本边界明确保留。** 此次实际飞行plan的receipt_adapter_sha256是`bedc309eba9276a0846d928ab81616f8c51a8db727b4767c8f5d2b2ef239c543`，不是新59f88d guard。`qualification/source/scenario-and-receipts-v2/`绑定文件全部SHA核对通过，旧adapter与supervisor均匹配该plan。其binding明示replayer是在飞行之后加入，不能将其当作当时执行的flight代码。新guard没有倒改旧receipt或被描述为当时已运行版本。

## 4. SR数值口径与独立复算

`sr_root_table:15-29`建立全计划身份集合，classic三配置各1次，LLM五配置各3次，即每root18槽；每行必须planned、唯一、status=completed、Python int的二元SR。缺行、重复、操作失败、bool、非binary及非法repeat均阻断，不将失败/未启动填0。

它先对每root/方法的规定repeat求SR均值，再复用严格RootTable身份/形状校验。SR=0且测量completed仍保留，不能按成功过滤；它接收上游已判定的episode SR，不从平均C/S/D再过阈值，也不自行认证receipt或计算业务成功条件。

独立检查覆盖：

- 完整合成72root、1296episode槽，得到72×8表；逐root按原始1/3个二元值另用标量求和复算，全部一致。
- ID/OOD分开，每域各等级等权，层内root等权。非平衡小夹具ID层数2/3/5、OOD层数2/5也与独立宏平均一致，未按pool数量给等级加权。
- 在每层抽一个root索引，携带完整8方法均值向量。独立用固定`SeedSequence([domain_seed,level,1])`重现97draws，再以Python标量求和/手写线性分位数复算，两域16个区间最大绝对误差`1.11e-16`。
- metadata/episode输入倒序及batch7→31改变不影响完整结果；repeat没有变成额外root，未采用flattened-repeat Wilson。
- 独立6类坏行探针全部拒绝；现有8项SR测试另覆盖未知root等边界。
- 全0/全1合成数据分别给经验点区间[0,0]/[1,1]。这是退化经验bootstrap，不能解释为总体概率已确定；沿输出的全局nominal/descriptive角色理解。

输出角色明确为`secondary_descriptive_SR_not_new_confirmatory_tests`，区间名为`ci95_root_cluster_percentile_nominal`，没有新增p、拒绝判定或显著性家庭。该区间是每方法/每域的pointwise名义描述，不能称为有限样本保证或16项同时区间。上次稀有损失反例对经验bootstrap的局限仍适用，未因SR是二元量而自动消失。

## 5. 完成范围与开放事项

在上述SHA和接口契约下，没有发现需要再改的guard或SR算术阻断；原P2关闭。SR仍仅用显式合成完整矩阵验证，真实八配置/全部root的SR输入尚未齐。不能用这1次dev-L3-B0和synthetic补齐主表，也不能把dev root强行标为ID/OOD以满足主shape。

真实root/lineage独立性、完整正式矩阵、LLM/provider/费用、SDK真实measure/可移植打包、显式unknown reconcile、事件兄弟/同权重日历及恢复删失仍需各自完成。receipt/SR模块的数值或文件校验不认证这些前提；本报告不授G4或新运行授权。

决策轨迹：重放两个已知P2并确认只运行一次；对既有真实回执只读审计并保留旧版本；对SR另写标量实现核对分母/宏平均/分位数；以明确nominal输出限制结论，没有重跑物理或扩大确认性分析。

Attention

reviewed by inherited-parent model（同模型家族独立上下文，非跨模型家族审查）

“已有1条可一致验证的实际dev回执”“新guard通过合成反例”“SR通过合成数值复核”是三个不同结论，不合并成正式方法、总体成功率或G4验收。
