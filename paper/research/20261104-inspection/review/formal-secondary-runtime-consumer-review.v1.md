# Formal secondary/runtime consumers：独立限定审查

结论：**在本次冻结源码与有界软件复验范围内，未发现新的实质阻断。** 216 clean、108 ablation、180 latency 的追加flight形状、原parent/seed身份、参数到worker的链与runtime失败守卫相符。该结论不是G4、真实正式flight或全graph/human认证；全部formal仍未安装，没有实际main/holdout namespace、seed/order或dataset被创建或使用。

审查者 `/root/control_review`，同模型家族独立工程上下文。输入仅 `planning/formal-admission-candidate.v3/secondary-consumer-runtime-freeze.v1/binding.json` SHA `d0ce314b471a85438493d4399665df598ceb8bc7219e747aaac202d65dd06ef0` 与对应源码/测试/日志，复用既有primary-v3/G4/namespace/dataset/worker证明的边界，不重审整个架构。source-only期间只轻量读取，没有在在途pilot上做集中CPU检查、批量hash或新probe。root确认UTC2026-11-06T15:38:14.546728停在下一slot预约/spawn前后，才执行下面的软件复验；未改或批量审production104来源、未读取pilot成绩。

63个冻结manifest成员（37source/21tests/5logs）逐SHA验证，37source+21tests的working副本也与freeze一致。root保存的最终全套日志实际为393 passed/63.95s；早期386总数中的2个class-identity fixture失败、secondary runner的4fail/3pass及修复后的7pass/8.94s都保留。它们是root的软件证据，不说成本审查者重跑或真实main结果。

本人在已确认barrier只复跑4个working测试模块（先核SHA）：test_formal_secondary、test_formal_secondary_runner、test_secondary_recording、test_formal_common_calendar，实际 **29 passed in20.74s**。比先前3–12s粗估更久，但全过程处于barrier，未启动模型/physics/native操作。另执行独立实际CLI+Fake dispatch探针8例：4个ablation/delay参数错配在registry、mkdir、模型client或physics context之前以intervention错误拒绝；4个匹配输入通过这一比较后，在刻意缺失的fixture registry处停止。两类都没有输出目录，model_client/physics_context调用均0。正例仅说明该CLI门未误拒合法参数，不冒称其通过真实G4/owner或完整dispatch。所有fixtures只使用合成native-shaped状态、literal order/seed及已有dev body的内存序列化副本；没有actual formal RNG、model、physics或真实native state写动作。

主要输入源精确SHA：

| 冻结 `source/eval` 文件 | SHA256 |
| --- | --- |
| `inspection_formal_secondary.py` | `4eaaff4106706ce244633d88d2023407b4f8cbc9e960c8cbc8ad7ba57320da26` |
| `inspection_formal_intervention.py` | `d30db3aaf1df79e5e76d934b48ea48535265e257dd7a3d4365459ba5b9a097cc` |
| `run_inspection_formal_secondary.py` | `440f17d3a98b0504b0d50c8acd005a1ad6284589d9c5b46118c21396bb57a4f5` |
| `inspection_secondary_recording.py` | `9af7274d93abc2018592dd7c134ace946e2b7f57c5b86dce14d7090201bf274f` |
| `inspection_common_calendar.py` | `b42f8944079adbf23ec236c20de8de51b8697a12d7d94e40275fbdc65ff9a364` |
| `inspection_formal_case.py` | `d625e0c0ba75a2bb86ef2d98974043d08a5d9017fdedf22723a918e852ee4a59` |
| `inspection_formal_dispatch.py` | `a54eb94ce214a2e4782017121024a754a869d328fe89538775d8554636ba9e58` |
| `run_inspection_formal.py` | `0ea0b2882ab095aede6a1aa41c3880dd81da9538aa16036e36cd680f83518f48` |
| `inspection_episode_receipts.py` | `11339aaffe69a6ceed537057364b5bc0de5a84672f7285e03e56e733221f9ec7` |
| `inspection_worker_runtime.py` | `bf4e0b38f1d7da98ccd559091b2b722c74ed0e6fdfd0088336ccaa441bdfc21c` |
| `inspection_formal_admission.py` | `b4d90554192068a846e0886efe4b64fdaf328cb85fb3308ea099035bfb302bc0` |
| `run_inspection_llm_development.py` | `4d854e2bf8d1b641724d7f64d737ad2216f1abae37e384a09d1db71fe3142794` |
| `inspection_measurement_loader.py` | `4e5677a60166e92031aca2bc48d4944954ecaeb9c20cbc7c79edd7b014c0bf52` |

具体审查结论与证据范围：

1. **完整追加分母与原身份**。inspection_formal_secondary.py:14–58、:89–93、:101–142固定main-ID/OOD各6个L3 parent；clean复用全部8条件与原repeats，12×18=216、96root means；ablation仅B4-M1×3arms×3repeats=108、36means；latency仅B4-M1×5arms×3repeats=180、60means。Slot直接复用对应primary case/condition/repeat/sampler与parent_order_index，按primary root顺序及预声明SHA order排序，不新draw、不按结果挑选。latency zero也用fresh额外flight，未启用可选primary-trace复用，仍在声明180内；这些是同12 parent的追加flight，不增加独立root。reader校typed完整分母、canonical artifact/order、exact内容/native先后与fresh primary/source，不允许holdout secondary或覆盖partial plan。测试实际覆盖全grid与rehash错arm拒绝，不只是计数常量。

2. **same-level clean是受限变换**。inspection_formal_intervention.py:39–61先对完整main L3 parent做正式内容验证，保留level/domain/geometry/target motion/seed/lineage，严格仅将EVENT_FIELDS中四类事件的全部字段清空，另绑定完整parent_case和parent semantic SHA。不是降成L2或重标一个新root；不放宽原population范围/geometry guard。:64–71要求effective clean world的case/hidden entities吻合parent+变换。提供的正反test覆盖level、seed、wind、event、parentSHA和错误root改变。

3. **packet→plan/argv→worker-v2闭环**。inspection_formal_dispatch.py:127–168在G4/context/fresh secondary proof下选择唯一track/arm slot，并生成显式worker-v2+dispatch；episode_plan和admit_episode_plan核case原bytes/semantic/config、condition/repeat/seed、ablation或delay、profile、.1门、2attempt与4thread1。run_inspection_formal.py:88–101把参数送CLI；LLM runner:70–78又将实际--ablation/--injected-delay与admitted slot比对，零延迟/clean不能静默拿非零wait。independent Fake CLI探针直接执行这段当前冻结main前门。worker loader:47–75先独立重建context/admit exact dispatch、核payload equality，验证同一bounded private buffer与fresh dataset后，仅clean分支转换case；最终fresh dispatch与effective binding再比对，然后才warmup/mission构造。保留parent原private bytes与effective clean semantic identity的区别，由正式receipt的clean分支再检查，未把nested truth送给actor/service。

4. **actual consumer与SDK失败分母**。run_inspection_formal_secondary.py:36–104共享唯一matrix lock、固定next-slot，classic/LLM都经过既有真实episode/receipt/measurement入口，而非callback自己提供任意Y。clean走未改record_matrix完整8条件；ablation/latency走inspection_secondary_recording.py:15–62，在flight前打开episode和root-means两个完整SDK envelope，异常保留未启动行，只有全部实际内层值后才聚合同parent repeat mean。原B1 selector从primary proof继承，没有重选。软件测试用Fake effects/values配真实SDK核216/108/180与96/36/60、顺序、raw mean provenance及infra failure完整分母；没有验证真实formal飞行、资源操作或正式统计结论。SDK内部额外descriptive contrasts不替代计划中与primary B4-M1的单独分析join，不产生第七个confirmatory test。

5. **common-calendar是离线受限adapter**。inspection_common_calendar.py:84–111要求来自fresh上游dataset/dispatch的原L3 parent digest；event world或clean world必须保留同level/geometry/motion/noise身份。先用原geometry重建sealed report历史并exact重算原score，再按parent原demand calendar计价；不新rollout、不改原score/world、不把truth返给actor，也不用10Hz轨迹伪造里程。保留原240Hz-hook的四机remaining_range和exit测量。test涵盖event/clean两正例、level/seed/identity/hidden/parent/score反例与world不变。该函数明确依赖上游实际receipt/source/loader admission；它自身不是任意raw JSON的真实性认证、完整formal事件分析或primary-analysis lock。真实动态60格/事件response-recovery和完整原始trace的正式分析仍需实际数据及对应分析入口，不能由这几个literal fixtures代替。

6. **已审runtime传播未另改控制门**。inspection_worker_runtime.py的完整SHA与已审runtime helper `bf4e0b38…`相同，仍为physics-process循环GC deferral与self-thread Mach soft constraint；非QoS调高、非硬实时。shared worker在formal内容前门/暖机/mission构造后、原epoch前start，在control结束及finally关闭，恢复错误进入world operational_failure；LLM inventory含helper，formal REQUIRED_SOURCES也明确列入runtime、新secondary/intervention/recording/runner/common-calendar等实际消费者，不能仅在声明中漏依赖。inspection_episode_receipts.py:70–90、:135–137按source vintage要求runtime audit开始/结束/恢复和GC/错误状态；formal全闭包必含该helper。这个shared guard不重新解释旧source records，也不把一次runtime candidate通过升级为所有conditions合格。新增.1s/600s/144000/2attempt/profile门没有放宽；现有primary/runtime guard软件项沿用root393证据，本轮只读其实现，不重跑已关闭全套。

7. **共享预约前checkpoint**。run_inspection_formal.py:104–117由primary与secondary共用：ack append+flush/fsync后在现有request上sleep .2，移除后fresh revalidate；两measure入口均在write_dispatch/episode_plan、reserve/spawn之前调用，后面再次经过typed fresh dispatch/admission。它不会暂停已在途physics/model，也不把ack当slot完成。改变G4/source/namespace/native状态时由原fresh门拒绝；本轮源码定位与已有primary guard边界一致，没有新增平行scheduler或owner写路径。

未证事项必须保留：393与本人的29+8都是软件/合成fixture检查；没有真实formal root、namespace/order seed draw、实际secondary packet/owner reservation、600s main/holdout/secondary载荷、完整新G4/final source inventory或领域human审核。原primary/namespace/dataset及G4Proof的权限边界不因本component审查扩大。实际部署时仍由既有G4→namespace/dataset→execution/dispatch→service/worker→receipt/measurement门提供权限，不能只换dev标签或拿这个review当GO。

独立日志、CLI脚本/结果与完整subject SHA在 `review/evidence/formal-secondary-runtime-consumer-v1/binding.json`。本轮限定结论已完成并冻结；集中checks已结束，停止写R，不修改任何实现、旧证据或在途pilot来源。
