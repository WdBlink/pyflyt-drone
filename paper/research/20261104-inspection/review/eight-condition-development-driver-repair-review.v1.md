# 八条件开发 driver 修复复验

审查者 `/root/control_review`，同模型家族、独立工程上下文。只对[原审查](eight-condition-development-driver-review.v1.md)两项P2及修复中发现的固定manifest锚遗漏做有界复验；原d81f报告/证据未改写。本轮不重跑真实108飞行、不运行MLX forward、GPU或PyFlyt，不调用owner reserve/settle，不修改生产源或真实state。

## 结论

**两项外围P2在最终候选 `0922f87f728fa383ec269759a29f8dea70aadcaa37d1138b41c93a3b50361586` 关闭，本轮未发现仍阻止外围新prepare的缺陷。** 共享cleanup实际SHA为 `361ba11b8ea088627e4c0bd832168bb26821297e5d8f54bf6d43c2ba327a9c70`。这允许重新冻结当前开发pilot envelope；真实dispatch仍须内层LLM episode独立审查闭合、全部实际来源稳定及原生资源预约满足。不是G4、正式main/holdout或性能批准。

## 版本与证据

中间候选b978实现了预冻结与tree finally，但遗漏初始manifest的固定锚。隔离临时run只给manifest增一个JSON字段，文件集合/原closure文件字节不变：b978的prepare成功；同一实际evaluator离线启动则立即报 `frozen evaluator manifest differs`。因此它仍会让可预先拒绝的错误输入进入真实flight。本反例已保留 [repair-results.json](eight-condition-development-driver-repair-v1-evidence/repair-results.json) 与 [b978原源](eight-condition-development-driver-repair-v1-evidence/run_inspection_eight_condition_pilot.b978.py)。没有把中间修复写成最终通过。

最终0922候选较b978只增加该固定manifest SHA检查。生产源与共享cleanup均保存原字节于本报告独占evidence目录；全部证据binding为 [binding.v1.json](eight-condition-development-driver-repair-v1-evidence/binding.v1.json)，SHA `b452ec7cbc5fda9508a8f00fc70d454f167f64863c5d97690a84bd6ee5b78a27`。

## P2-1：grader冻结关闭

最终归档源72–88行先核accepted design本体SHA，再按其中evaluator path/SHA核入口；81–82行将manifest绑定于该固定入口实际内嵌的 `ecc2069fc8795adb9096173ab2d859b6540a94309ecf99310755cb7b3177bf80`。83–88行核完整closure文件集合/无symlink/每文件字节。99–107行将入口、manifest、17份closure文件和accepted design纳入来源/输入表；125–129行在槽前后检查字节与closure完整文件集合；191行使用同一预冻结evaluator SHA建plan。

最终8个短fixture中相关结果为：

| 变异/路径 | 实际结果 |
| --- | --- |
| 正常临时prepare | 成功，callback0，无真实reserve/模型/物理 |
| 开跑前入口改字节 | callback0即SystemExit2，未写pilot成功 |
| 开跑前manifest加字段（原b978反例） | callback0即SystemExit2 |
| 首fake callback后入口改字节 | 当槽后check抛ValueError，不开始第二槽 |
| 首fake callback后manifest改字节 | 当槽后check抛ValueError |
| 首fake callback后增加closure文件 | exact file-set check抛ValueError |

后3项均保留真实SDK的classic18/LLM90/outer48完整分母，没有pilot completion。新prepare source文本与hash独立重算全部匹配，共63份源（原44+入口/manifest/17closure）、61依赖、10输入。这里“17 closure文件”只表示该冻结evaluator的实际集合，不增加研究根或实验分母。

## P2-2：classic外层清理关闭

最终源144–159行已使用 `SubprocessTree`：Popen后attach、持续wait；finally调用close，外层finally保存invocation；只有code0且cleanup=`closed`才进入后续receipt解析。

原反例改为经过实际driver classic分支，仅把command替换为Python假supervisor及独立session sleep worker，将tree wait缩至0.35s：实际捕获两个born身份，超时后close为closed、remaining_owned为空；独立进程存活检查确认child不再活跃。失败invocation JSON/stdout/stderr实际保存，原18/90/48分母保留，没有pilot completion。不是仅凭cleanup布尔自报已清理。

另在真实清理完成后由fixture强制返回cleanup=`unknown`、同时模拟code0；driver仍抛 `classical operational failure or unknown blocks the pilot`，没有把未知清理当成可入账结果。此项只验证caller拒绝分支，不声称人为造出的unknown来自真实遗留进程。

共8个fixture完整输出见 [repair-final-results.json](eight-condition-development-driver-repair-v1-evidence/repair-final-results.json)，复现代码见 [check_driver_repair_final.py](eight-condition-development-driver-repair-v1-evidence/check_driver_repair_final.py)。真实control/checkpoint SHA在fixture前后相同。早期b978输出的 `evaluator_frozen=false` 是fixture将`/tmp`与实际`/private/tmp`路径作未normalize比较导致；独立resolved路径核对全部为true，说明见 [repair-path-normalization-check.json](eight-condition-development-driver-repair-v1-evidence/repair-path-normalization-check.json)。最终fixture已修正此比较，无产品问题被隐藏。

## 有效边界

本次未重审通用cleanup的所有进程/host-loss边界，也不将periodic born-identity inventory说成kernel subreaper。它能清理本次观察并绑定的子孙；未知新同UID Python orphan保守阻断，不授予杀任意进程的权限。此共享模块及内层episode更多异常路径仍由另一独立审查负责。

原 `logs/inspection-eight-condition-pilot-preparation.v1` 继续绑定旧44-source组合，不能替代本候选及正在修复的内层模块的新freeze。原d81f全fake108/48形状测试与本次短修复guard共同提供外围工程证据，不是新实际飞行；旧B1选择、旧Y、旧失败与首次prepare拒绝历史均保留。后续新prepare和真实执行的记录必须另建，不复用本报告fake输出充当真实成绩。
