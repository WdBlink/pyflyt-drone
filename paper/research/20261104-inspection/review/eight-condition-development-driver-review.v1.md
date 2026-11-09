# 八条件开发 driver 独立工程审查

审查者 `/root/control_review`；同模型家族、独立上下文。范围是外围 `run_inspection_eight_condition_pilot.py`、实际 prepare 产物及其与 classic/LLM episode adapter、既有 recording SDK 的接口。未运行 MLX forward、PyFlyt、真实 flight、native reserve/settle 或 owner 命令；未修改生产源、既有结果或 `.paper`。

## 原版结论

**原版存在两项 P2，建议真实 dispatch 前修复。** 六根/108内槽/48外层均值的排程及失败分母检查通过，不能据此授予 G4 或正式运行 GO。原版精确 SHA 为 `d81f808e24bbdb6729ead51e0e927d6122ce0cd7035567dc37fa65e655407369`；已冻结原字节于 [run_inspection_eight_condition_pilot.d81f.py](eight-condition-development-driver-v1-evidence/run_inspection_eight_condition_pilot.d81f.py)。以下行号均属于该归档版本，后续源码修复不回填本次原版结论。

### P2-1：evaluator 在每槽重新取当前 SHA，未参与 pilot 的预冻结和漂移拒绝

归档源 **81–88、106–108、147–158行**：44份 source 和9份 input不含 evaluator；`check()`只检查这些既存表。每个 LLM 槽又用 `digest(evaluator)` 建立新 plan 基准。这样 pilot 声明的“同一 grader”没有贯穿整个108槽。

隔离复现只复制必要 owner/registry/evaluator 到 `/tmp`，第一条 fake LLM callback 后给临时 evaluator 加一条注释；第二条 plan 的 evaluator SHA 从 `de826d24…` 变为 `1d853e75…`，完整108 callback/48 means仍完成、未报漂移。这个反例证明冻结缺口，未声称注释改变实际分数。实际内层只比对本槽 plan 的 evaluator SHA，不能补足跨槽同一基准。

最小修复：从 accepted design 的 evaluator 身份预冻结 entry、closure manifest及closure完整文件集合，开跑前检查，纳入每槽前后漂移检查；各槽使用同一已冻结 SHA，不能当场把新字节升级为基准。证据：[check_evaluator_drift.py](eight-condition-development-driver-v1-evidence/check_evaluator_drift.py)、[evaluator-drift-compact.json](eight-condition-development-driver-v1-evidence/evaluator-drift-compact.json)。

### P2-2：classic 外层超时/中断缺少 descendant cleanup

归档源 **119–123行**使用 `subprocess.run(...timeout=1700)`，无外层树清理 finally。被调用的 classic adapter 又用 `Popen(...start_new_session=True)` 启动 flight。直接 supervisor 被 timeout 杀死时，它的 finally 不能可靠执行，脱离进程组的 flight 可残留；超时也跳过当前 invocation JSON 的保存。正常两次780s内层等待低于1700s，只能降低常态触发概率，不能覆盖卡住或外层中断。

复现仅用 Python fake supervisor 和独立 session 的 sleep worker，将同种 `subprocess.run` 等待缩为0.15s；收到 `TimeoutExpired` 后 worker仍存活。审查脚本已自行终止该假 worker，没有运行物理。最小修复：classic 分支也使用已有 birth-bound `SubprocessTree`，在 finally 处理直接进程及所记录子孙；unknown cleanup阻断后续执行，同时持久保存失败 invocation，不将其当作 policy Y=0。证据：[check_classic_timeout.py](eight-condition-development-driver-v1-evidence/check_classic_timeout.py)、[classic-timeout-result.json](eight-condition-development-driver-v1-evidence/classic-timeout-result.json)。

## 已验证范围

实际成功 prepare 位于 repo 的 `logs/inspection-eight-condition-pilot-preparation.v1`，不是研究根下的 logs。独立核对44份 source文本逐一重算匹配其SHA、61份依赖和9份输入文件逐一匹配；原18槽结果也确实包含同六个根、全部completed，selection绑定其结果SHA，冻结选择仍为B1。source snapshot与其hash内部完整，并不代表当前全部源仍是该版本：检查时 `run_inspection_llm_episode.py` 和 `inspection_llm_episode_receipts.py` 已因独立审查修复而变化。原 prepare 保留旧身份；实际执行须使用新的完整 envelope。

历史首 prepare **确实失败**：`inspection-all8-pilot-preparation.v1.log` 显示 accepted plan/active round guard拒绝；根 agent说明根因为 episode spaced canonical digest 与 owner compact digest不同，归档 d81f 源已改用 `inspection_model_service._digest`。`v2.log`为0字节成功输出，配套实际七份 prepare产物已逐字节核验。没有把该过程写成首次成功，也没有将无stdout单独当作执行成功证据。

实际 schedule seed `2026110611`，sampler master `2026110612`。六个 case严格为 `dev-L{1,2,3}-{00,01}`，lineage六个不同；独立展开 expected Cartesian identities 与 schedule恰好一致：

| 记录 | 分母与重复 |
| --- | --- |
| classic 内层 | B0/B1/B2各6，合18；每根/条件1次 |
| LLM 内层 | B3-M1/M2/M3、B4-M1/M2各18，合90；每根/条件3次 |
| 全内层 | 108唯一identity；每根完整18槽，order_index为0..107 |
| 外层 | 六根×八条件=48均值；没有把outer算成额外flight |

按 schedule 的LLM顺序重新实例化 NumPy PCG64，90个uint32 sampler seed逐一完全相同且本轮无重复。这些是 provider sampler输入，不是90个独立科学根，也不保证 provider确定性。

纯CPU fake检查使用真实 argparse/main、实际 SDK `ExperimentRecorder` 和 `record_matrix`：

- `--prepare-only`：subprocess/Popen、classic与LLM adapter全部拦截；实际callback=0、process请求=0，只有临时输出/receipts目录和意图文件，没有SDK results。审查audit hook禁止repo写入，真实control/checkpoint SHA前后相同，没有执行owner预算动作。
- 完整fake矩阵：18次 classic invocation被截获后调用真实 `run_inspection_dev.main` 构造plan，再在其receipt执行边界替换为fake；driver随后18次post-return cache检查均吻合。90次LLM接口捕获plan/command，23项 worker source集合与当前runner源列表吻合；case、condition、repeat、sampler、profile/calls/tokens reservation、native frozen-plan及解释器参数一致。没有调用真实LLM episode reserve/dispatch。
- 真实SDK写出18/90/48 completed行；fake输入 classic=.25、LLM=.5，outer分别18行=.25、30行=.5，均值与分母正确。`primary_inference_ready=false`，completion文案保留非G4边界。
- 第一条LLM callback抛 `RuntimeError`：18个classic为stopped；LLM为1 failed+89 stopped；48个outer全stopped。reported_denominator仍18/90/48，attempted分别0/1/0；没有丢失槽、静默完成outer或生成成功completion，也没有把基础设施异常当真实Y=0。

复现脚本和完整/压缩结果见 [check_driver.py](eight-condition-development-driver-v1-evidence/check_driver.py)、[result-compact.json](eight-condition-development-driver-v1-evidence/result-compact.json)、[sdk-status-counts.json](eight-condition-development-driver-v1-evidence/sdk-status-counts.json)。全部证据SHA见 [binding.v1.json](eight-condition-development-driver-v1-evidence/binding.v1.json)，binding SHA为 `005887f5b085cae9a1cb9b836a40de91c20f225ff87894104a51986875924f8b`。

## 限制与运行边界

本报告不取代另一独立 agent 对 LLM episode重试、外部结算、causal admission与清理的审查；其在修的源不能因本外围fake通过而获准。fake Y/evidence只用于验证SDK形状和交接，不是模型、飞行或评分正确性的实际结果；本报告未重新核算原18次飞行的全部物理trace，也未读取模型权重或验证新600s运行。

修复并有界复验上述两项、冻结新的当前完整 source/input/dependency/evaluator envelope，且内层独立审查闭合之后，才具备开始该开发pilot的工程前提；这不等于完整G4、main/holdout、正式性能或人类领域认证。旧six-root选择不因新classic重跑而重选，旧源码attempt和原Y不得被新framing或新prepare追改。
