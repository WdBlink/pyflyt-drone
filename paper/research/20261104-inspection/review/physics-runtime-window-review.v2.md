# Physics runtime window v2：有限修复与adapter复验

结论：**原v1 post-start constructor rollback P2已关闭，本次有限修复/adapter及指定单次载荷准备未发现新的明确阻断。** 可以按这套新来源为一次独立、无分数工程载荷走原生预约并执行；不是已通过600s、G4、硬实时或旧pilot槽的第三retry。原v1快照、反例及报告保持不变，不能据修复重分类旧失败。

subject为immutable `planning/physics-runtime-window.v1/freeze.v2/binding.json`，SHA `8c08adb80bba6179b78401e9374c9beb3abeb1546a7208dbd07198403117bfac`。四源码：physics_runtime `bf4e0b38…`（与v1相同），timed_profile `e4afc007…`，timed_worker `3fe5cdf4…`，probe `712aab6d…`；完整SHA见独立binding。9个冻结manifest成员全部匹配，4source+3tests的working字节与freeze一致。因worker AST test的PROJECT定位针对working目录，复跑使用经SHA核实的working tests，未把freeze层级的路径差异当实现缺陷。

原独立反例适配成新脚本，仅换v2强SHA guard与预期恢复结果，v1脚本未改：实际PhysicsRuntimeWindow配Fake native/collector，在window.start后让get_count抛原错误。v2仍抛同一错误，但 **window.closed=true、native RT=false、collector enabled=true、base callback已恢复原集合、audit.errors=[]**。一行constructor except从super().stop改为self.stop确实走到已有window.close；没有仅修测试期望。没有真实修改host调度或GC的负例外溢，反例finally仍有独立兜底。

`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider <working>/tests` 实际 **13 passed in .04s**；包括短self-thread native读回/恢复、GC初态开/关、startup set失败、demotion、restore失败、重复close、postloop collector异常后GC恢复、单行metadata跨SHA、post-start metadata故障，以及Fake worker final errors传播。没有MLX、aircraft或正式RNG。集中检查和fresh解释器导入探针均已结束，并在root预约前明确通知。

adapter具体边界：

- timed_worker全文与已审 `eval/inspection_diagnostic_worker.py` 比较，只有保存metadata后新增 `runtime_policy.errors` 非空→operational_failure 两行（`:131–133`）；原while-loop AST完全相同，不改变RPC drain、120Hz sleep/epoch、物理step、publication/logging或lag计算。元数据save异常原路径也operational_failure。Fake的空errors和非空errors分别得到completed/operational_failure，实际保存world status与metadata一致。
- `timed_worker.py:22–38` 先构造warmup、再构造fresh mission/private case，之后才调用profile_factory，再取原epoch；`:99,120–136` 在loop结束及finally stop，mission cleanup与profile save在control外。TimedProfile.stop先撤base callback再close窗口；save内部重复stop也不重做恢复/collect。constructor自身修复覆盖profile尚未返回、worker变量仍为None的清理缺口。
- probe相对已审M2/L3 harness，独立AST归一后仅ROLE、八variant来源、local import path、worker和factory发生声明的变化。B3/M2、原dev-L3-00、sampler1096691202、600s1×、.1s门、32calls180000tokens、soft60s、actor/service、caffeinate及cleanup/supervision逻辑不变。父子admission仍先验live development reservation和精确variant SHA，再启动；没有SDK callback、自动retry或正式namespace入口。
- 用新fresh解释器实际unpickle `partial(timed_worker.worker,profile_factory=TimedClockProfile)`，精确导入working `source/timed_worker.py` 和 `source/timed_profile.py`；仅导入，未调用worker或构造factory。这验证顶层module名与sys.path在新解释器可解析，不冒称已经验证真实multiprocessing/physics进程整合。
- 成功单行metadata再次核runtime audit、GC状态、clock→rusage→world指针SHA；physics_runtime ABI/tick/soft语义沿用v1已审未变源。本次没有另一个已复现的metadata或恢复缺陷。Mach首尾读回不证明整个窗口从未demote，GC deferral是physics整个进程的循环收集，reference counting和其他进程默认GC仍保留；实际memory/fault/cleanup均待载荷证据。

载荷末核独立绑定：`qualification/clock-runtime-M2-L3.v1/freeze.v1/binding.json` SHA `0ed2471d31e2814e487b62a4b3fc3544130da2ec16ed32ca651f889b6bdfb3ce`；`candidate-load-intent.v1.json` SHA `9dddd9c5bc1bcf195dc7360549a732c38906425bb4ea5cf5d1f9b1800043e751`。113冻结文件、109当前来源、61依赖、11输入全部匹配，109精确为原pilot103与runtime八variant之并集（净增6）。八项SHA与本v2 source/已有四telemetry-cleanup source一致；原103 production字节未变，没有生产安装。case、既存sampler、M2 registry/download receipt及generation profile、reference diagnostic与runtime subject指针均匹配，未draw任何新科学种子/实例。

审查时新output `logs/inspection-clock-runtime-M2-L3.v1`、reservation `dev-clock-runtime-M2-L3-v1-20261106`、对应service operation/reserve packet及ledger项均尚不存在。下一步仍须由sole native reserve精确development ROLE/case/sampler/八variant和32calls180000tokens后启动canonical working probe；本审查不写owner，不提供unreserved旁路。未知计量/清理不得零化或自动retry。root应在本轮检查完成后才运行，并在途停止自身集中tests/checkpoints/消耗源改写/并行物理或模型；这只是避免可控干扰，不保证host绝对无其他负载。

允许的后续仅这一次工程验证：完整clock+rusage72000与GC数组、实际active_end/standard restore、GC disabled-at-epoch/restored-after、runtime audit.errors空、真实RAM/page pressure、清理和usage需另审；maxlag<=.1才能说明这一个candidate load的clock门通过。scheduler/GC两项及仪器、caffeinate条件使其不是严格单因素A/B，也不能消除旧v3失败或授G4/main/holdout/人类领域认证。本次是同模型家族独立有限工程review，未运行新模型、physics、正式RNG或nativeowner写动作。

独立结果/脚本及绑定见 `review/evidence/physics-runtime-window-v2/binding.json`。报告冻结后停止R写，不再执行集中检查，等待后续独立实际证据任务。
