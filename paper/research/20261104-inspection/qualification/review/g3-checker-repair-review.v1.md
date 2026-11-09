# G3检查入口硬化独立复验 v1

日期：2026-11-05。独立上下文、同模型家族工程Verify。仅写本报告，不回填g3-v19-control-sensor-review.v1.md，不修改production、原run/world、旧receipt或其他Agent产物。没有重跑长physics、600秒重放或1×；本次只验证已执行receipt、源码及短guards。

## 结论

**F1依赖闭包、F2未emit分母完整性锚已关闭；原F3“无关observe句柄也通过”的反例已被拒绝，但F3 helper仍有需要补齐的因果seal。** 当前20项guards通过，原3个变异副本均被新qualify拒绝；另分别测试dependency/replay/chain helper，避免仅由同一个receipt SHA失败掩盖缺少join。

新实际receipt确实绑定原run/world、当前replayer、完整22字段组和116 RPC，包含原始execution_tracking_trace；qualification.v2的真实chain witness也经原measurement独立核对成立。下述新增反例只表明helper的独立因果校验尚不完整，不表示原实际链或新重放有误；绑定外部真实receipt的整体流程会拒绝这些变异，不能因此省略helper本身的封口。

## 绑定身份与历史保留

路径相对paper/research/20261104-inspection；分析源码相对repo根。

| 对象 | SHA256 |
| --- | --- |
| eval/qualify_inspection_reference.py（本轮受审v2） | 70dff34389bb02460635b24f63f24e57af7d66ab6d8e7362cf3ececfb92c2ec8 |
| eval/replay_inspection_reference.py | 1c04a1340eca069398ca5b2e121e37df3701fd68b63fdad7413ea0e633159480 |
| qualification/reference/executor-realtime.v2.replay.v2.json | 899bfadc4a16c8104fcd50cb393a83649f0339f8a94480ffec051d93d1f0afeb |
| qualification/reference/executor-realtime.v2.qualification.v2.json | a8602536dfad2a8045889c612ad75ff6275fc9283a5eff37c8e259bffe040b88 |
| qualification/runtime/reference-admission.v2.log | 22d16909bb9cb74a42c226188c6992b148a70f2b1c5b6b545b394e2266eedfcb |
| 原run.json | 122e9f47ae63080ef0dac318a669e85b7e60a616d0e2f4ce3510fa4626c5f871 |
| 原world.json | 8d60ea4bff2b4624835b4d9bd260ff72d644e76c85a873c6ad4b387f554ad3c6 |

source/reference-analysis-v1内5份旧分析源码实际SHA与binding一致；其中旧checker为09b19a6a…cbc21，旧replayer为93dd2d2b…5ba69c。binding所指原资格v1 SHA80785057…f58f67及旧21组replay.v1 SHA badd7268…b0e53仍匹配。新旧版本明确保留，不将v2重放升级成原v1本来含22组。

新replayer相对旧source的唯一改动是对比keys增加execution_tracking_trace；其余纯规划重算、记录tick释放响应、RPC重放和物理推进流程未变。新receipt记录加速600秒、约97.47秒wall、22组全等、116/116 RPC且response_discrepancies=[]，输入run/world SHA和replayer SHA均匹配。它是原执行日志的加速确定性重放，不是新的策略、统计样本、模型时延或第二次真实1×运行。

## F1关闭：完整实际依赖集合及字节

受审v2:35–51从实际PyFlyt安装根枚举指定.py/.yaml/.urdf/.obj/.stl/.dae及PyBullet二进制，要求normalized path集合与记录完全相等且无重复别名，再逐文件重hash。实际集合为61项，原run全部满足。

独立直接测试verify_dependencies：完整61项true；仅保留1项false；保持集合完整但改一个digest为全0 false。原F1变异再送qualify也false。该helper不依赖replay receipt即可拒绝原闭包子集问题，不是碰巧被run SHA不匹配挡住。

## F2关闭：强制外部完整重放receipt

受审v2:54–64、182–186及CLI:235要求外部receipt绑定实际run字节、world字节、当前replayer SHA、精确22字段组及每项literal true、600秒、RPC replayed=expected=实际tool_calls数、response_discrepancies=[]。缺receipt的函数调用也返回passed=false，CLI不可省略--replay。

独立单测verify_replay：真实receipt true；空receipt、错run SHA、错world SHA、错replayer SHA、仅少execution_tracking_trace组、sensor_opportunities组false、RPC115、非空response差异，逐项均false。少组测试保留其他正确hash，证明完整字段组本身受检查；未仅测试一个无关SHA错误。

原F2副本删除全部未emit机会及背景trial后，即使为副本重算world/run绑定，qualify仍false，外部实际receipt不再匹配。这是正确的独立完整性锚。recount_sensor_opportunities自身仍只负责给定行的对账与计数，不能从行表自行发现被删掉的不可见行；完整性责任由强制、实际执行并绑定的22组重放承担。

## F3：原断链反例拒绝，仍有P2残余

受审v2:67–104确实已连接observe(candidate/report/drone/service)、accepted dynamic report及成功inspect的同candidate两视图。直接把原world全部observe改为从未inspect/report的句柄，causal_chain_witness返回None，qualify false；此原F3反例已修复，不仅由receipt gate拒绝。

但report证据当前只检查candidate、quality、arrival≤report及evidence.measured_at≤request.measured_at（:90–99），没有以下独立约束：claim不能晚于report、至少一份引用必须与claim同刻、report引用应连接本次inspect的实际有效视图。使用新增chain_fixture（inspect1…3秒；a@1.5、b@2.5，相隔25m；report@3.5引用b；observe创建4、local@4.5）单独调用helper，三种不合法/未闭合输入仍得到同一完整witness：

| 独立变异 | v2 helper实际结果 | 缺口 |
| --- | --- | --- |
| request.measured_at=999，report.t=3.5 | 仍返回witness | 未来claim未拒绝 |
| request.measured_at=3，引用证据只有2.5 | 仍返回witness | 没有与claim同刻的新measurement |
| report改引用inspect开始前的survey旧包@.5 | 仍返回witness | report evidence未连接物理inspect有效视图 |

前两项连ReportLedger的causal provenance条件都不满足；accepted标志不能代替独立验证。第三项虽可有同candidate的合法旧包，但不能单凭它声称report由上述两视图确认链支持。

最小必要修复：验证report claim为合法有限测量时间且≤report.t；所有引用与request的candidate/type/time等封存条件一致，至少一份measurement与claim同刻；require该新同刻证据属于所选成功inspect的实际qualified view集合（或另行给出显式合法跨机/交接关联）。若声明精确measurement封存，还应核对request estimate/class与该证据一致；避免accepted布尔值替代真实provenance。补充独立helper负例，不能只断开world SHA。

残余反例与受审v2源已冻结在`/tmp/g3-checker-repair-v2-control-evidence/`。chain_extra_faults.py SHA `21b0d89b4a3193e3da5b74b4c5dde33d2a05411dca09afd13a53ce9327800f9c`，含完整输入/输出的chain-extra-inputs-and-results.json SHA `4d7b53026c2dc258c2e14a94ac2cd917001d1698052250cd1b6aa583c81e8617`。即使随后production继续修复，此报告仍绑定70dff版本，不回写历史发现。

## 原真实witness的独立可信度核对

qualification.v2选中candidate-0-00005：inspect job-0037、report-candidate-0-00005、observe job-0040、first local obs-0-289-0。原数据不包含上述反例：

- inspect创建140.508333333、结束142.5秒；同机0的qualified inspect实际视图在141、141.5、142、142.5秒，最大机位基线25.495652m≥20。
- accepted report在143.008333333秒，claim142.5秒，引用obs-0-285-0；该包确为最后一个inspect view，measured142.5、arrived143，request estimate/class与原包一致。
- observe创建144.008333333秒，同candidate/report；新local obs-0-289-0在144.5秒采集、145秒到地面，机0、inspect模式、quality=.95、dynamic，144.5也为实际service_started_at。

因此真实链在加强封存条件下仍应成立，F3残余是检查器泛化拒绝能力，不是需要“修复”或重跑原物理数据。全部11项v2 checks在正例调用中true，但本报告不由这些bool独立证明完整业务评分或所有链条；完整业务由另一报告负责。

## 实际短检查与限制

本次命令：`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q tests/test_inspection_reference_admission.py tests/test_inspection_sensor_recount.py tests/test_inspection_qualification_analysis.py -p no:cacheprovider --basetemp /tmp/g3-checker-repair-pytest`，20 passed in0.09s，与保存日志20passed一致。

独立recheck.py在上述/tmp目录，SHA `26710886dcd03ee7221403f0a00aa286cc709e2d00630006ff27f9e6d3315f14`，退出0；results.json SHA `1db9188cd571f03533e0f20b3f5f887e49c12d1095c8fd339d5593d7a4723dd7`，记录各helper、原3变异、原正例及无receipt结果。没有重跑物理或replayer；新实际receipt的执行事实由保存产物及可审源码支持。

原g3-v19-control-sensor-review.v1.md的物理/时钟/传感统计继续有效。22组加入raw error trace后，可说该完整字段也被加速重放精确比较；仍不能称为原完整physical-state/墙钟重放。原72,000 control lag数组未保存，不能离线独立重求全量max/p95；原240Hz实际位移未保存，10Hz不能精确重建里程。新重放不是新1×测量，也不消除这些原始记录范围限制。

reviewed by inherited model configuration, independent context, same model family

- F1/F2关闭，原F3断链反例拒绝；F3独立helper仍需上述测量封存负例修复，当前不能把三项全部标为关闭。
- 原真实witness、22组/116 RPC绑定、新正例检查一致；未改任何原数据、旧报告或旧receipt。
- 不授予完整G2/G3/G4、正式模型性能、真实vision或领域人工认证。human_review_pending。

本版冻结上述受审版本；后续F3修复仅另行有界复验，不删旧发现。
