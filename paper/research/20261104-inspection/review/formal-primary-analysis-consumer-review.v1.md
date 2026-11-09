# Formal primary analysis consumer v2：独立有限审查

结论：**存在1项P2 receipt闭包绑定阻断，另有P3 CLI格式问题；freeze v2不宜作为可接受分析消费者。** 根Agent新focused7和全suite400均通过，但未覆盖该验证/绑定时序缺口。subject不是lock或独立accept，这个边界声明正确，却不能替代对subject自身数据证据的绑定。后续修复必须另freeze复验，本报告不认证working新版本。

审查者 `/root/control_review`，同模型家族独立工程上下文。责任只覆盖新analysis entry、REQUIRED_SOURCE三项增量与两份原数值模块，复用primary/G4/namespace/dataset/worker/secondary已审边界，不重审全部架构。正式namespace/root/seed/order/data仍不存在；未运行模型、physics、真实formal RNG或native写动作。定向源码/5个成员小hash检查在pilot在途期间完成；收到root确认UTC2026-11-06T16:47:33.153405Z的预约前barrier后，才执行三个有界合成反例。没有读取pilot成绩或改动production来源。

subject binding：`planning/formal-admission-candidate.v3/primary-analysis-freeze.v2/binding.json` SHA `61f4f22c821201f422721fae4289f660c9adbc4fb6702a5b6fb0852a2a43f7ff`。五个明确成员逐SHA/bytes匹配，反例启动时working五成员同SHA：analyze_inspection_formal `22db10209ef94caa1a7f3d9b49329b7bb5306ba3873fd1634f69a50847db0b03`；formal_admission `cc363a9fd99d432afbe8c2e532621f27dd0bf403018462345a05028630aeb16c`；statistics `70c8783be6f6665b81b1190d6b905f4a22938445cc4c641458202bab5c710545`；secondary_statistics `b056b9d42223651c9d467ba332ee7803505bc3b7a382f69e3c83bdd197c68ab1`；test_formal_analysis `55687d7461854d4e7ca9ca771081492faa67d225f89c7c2c3986dc3d9a07b46b`。两数值模块与原production数值模块精确相同；formal_admission相对已审secondary baseline只增加新analysis和两数值模块三条REQUIRED_SOURCE，未放宽原门。

P2：`source/eval/analyze_inspection_formal.py:140–178,195–200,282–284` 在receipt验证后重新采集可变文件的hash，并且发布前未闭合全部验证依据。

- 先read plan、admit、measure(:141–149)，随后才_ref(plan/terminal)(:165–166)；这些新hash没有与刚验证过的同一buffer/原快照比较。因此验证返回后变化的新bytes可能被当作trusted receipt_refs，后面的checked_path只是认可这份新hash。
- 发布前只fresh request与重新核plan/terminal(:282–284)。world/run/privatecopy/actor/实际model-operation以及settlement闭包未纳入同一最终原快照复查；G4/dataset/execution重验不能替代这些episode文件的验证。
- `.matrix.lock`能防止遵守该锁的正常producer并行运行，但不能把“验证后hash新bytes”变成正确绑定，也不能证明所有闭合receipt文件持续未变。本finding是函数实际可观测窗口，不声称提供无限返回后/全host原子性保证。

三个独立边界反例均实际写出了analysis subject：

| 注入点 | 变化 | 实际错误接纳 |
| --- | --- | --- |
| 首slot的trusted measure返回后、首次_ref前 | plan.case_sha256改为全0 | 输出subject，recording-index采信变化后的plan SHA |
| 同一验证返回边界 | terminal.status改unknown_model_usage | 输出subject并采信新terminal SHA，未再次拒绝unknown终态 |
| collect完成、statistics入口 | 已有synthetic world文件改字节 | 输出subject，world本来不在最终receipt_refs集合 |

反例采用原new-test的real SDK完整1296 synthetic grid、合成native-shaped fixture与Fake physical validated-return API；用固定统计替身避开99999重采样，只隔离receipt信任边界，未捏造真实formal数据或原生资源操作。前两例直接展示新hash重绑定，第三展示发布前全闭包漏验；不是对真实硬件receiptvalidator本身的通审。必要修复是在验证前固定完整episode+闭合model-operation/canonical resource/reserve/settle文件集合和hash，验证后/发布前仅对照原快照拒绝新增、缺失或变化，不重新将变化后的hash采为可信。大集合可以分episode压成metadata manifest并绑定索引，但不能用metadata声称已保存/认证未读取的raw bytes。新增修复未在本v2报告内测试。

P3：`analyze_inspection_formal.py:324–330` 的CLI最终 `print(result)` 输出Python dict repr，包含单引号等，不是标准JSON。函数/subject文件本身仍结构化，当前读取范围未见依赖stdout的已安装caller，故此项低于P2；机器CLI应以json.dumps输出，并测试stdout能strict JSON解码且不把subject误包装为lock/accept。本项是源码可证格式问题，未启动真实CLI/native流程。

已验证的正向设计与限制：

1. read_request→fresh admitted_request承接G4/namespace/dataset/execution和phase；collect只接受main或once-opened holdout。三SDK envelope的原design、原order、完成status/outcome、raw canonical身份、SDK native _experiment验证、evaluator/closure bytes、completion/index和每slot order-receipt都有检查。每slot走read-only measure_completed_episode及已有真实receipt/native settlement验证，拒绝secondary混入primary，维持72/1296/576与12/216/96分母及原repeat聚合。P2不表示这些前门不存在，而是其结果没有被完整固定到最终输出。
2. copied SDK捕获bytes按原路径写入新目录，main outer design保留accepted原字节；outer projection只改raw相对路径（holdout另将design引用改到其已复制modified design），不改value/evidence或重新编码源SDK raw。输出前路径经过safe/new canonical目录，既存分析不覆盖；_ref在hash前拒symlink。坏SDK/重算错/失败分母的普通输入应在输出前拒绝，已有7测试覆盖这些静态坏输入；其绿灯不覆盖上述时间窗口。
3. RootTable/primary analyze_roots和SR模块是精确原副本，main仍固定99999、原equal-level/paired-root语义及global6 family，SR仍descriptive cluster；新producer没有增加第七个confirmatory test。component_summary先repeat→root→stratum，并将L1 D标not_applicable/不投影为伪零。未重做原数值方法评审，也未据Fake值作科学推断。
4. v2 holdout_directions(:221–248)只接受ID L2/L3各6个的12-root表；对locked main的完整48 ID root_differences按L2/L3各16取等等级均值，保留all-level full-ID effect单列，并仅描述direction/size、p_values=not_computed。未再将holdout L2/L3与混入L1的main宏均值直接判方向；v1草稿保留，未偷改既定主统计或population。真实main-stat文件仍须来自已锁主分析的独立证据链，不能拿任意raw JSON当完成证明。
5. subject明确needs_independent_review_not_a_lock_or_G4_decision；新入口不生成namespace/data、nativefinding、main lock或holdout opening，新增消费者在源码required闭包中。数值统计的固定分析重采样不是formal场景/seed生成。现有调用方边界以既有runner/measurement/sourceproof复用，未发现另一个本次限定范围内的权限升级；没有因此完成全graph或实际formal运行资格。

实际检查：root focused日志 **7 passed15.96s**（SHA `0da0e66a5101891b52f58d7388491e06766fef9a515bb80aa24d2944e4523ade`）；全suite **400 passed90.30s**（SHA `89f5654546a744176a8b5e1d6bb85331c10ca581a4c6b871bd8934c270c3fdef`）。compileall root报告exit0，保存日志为空（SHA e3b0c442…），空文件单独不能证明exit状态。本人只读核对这些日志并归档，未重复跑7/400套件；本人实际执行的是上述3项独立边界反例，三项均误收而触发finding，不写成“3pass表示产品通过”。393和secondary29+8的既有边界未重复通审。

counter脚本/三项前后SHA结果、actual logs和完整绑定见 `review/evidence/formal-primary-analysis-consumer-v1/binding.json`。脚本强guard要求v2 working五成员；root之后有意修复时拒绝运行是正确版本保护。需要重现历史可从freeze v2与已绑定baseline fixture在隔离目录恢复，不得回退共享working或用旧hash审新代码。本轮v2报告/证据冻结，停止R写与集中checks；新修复应另版有限复验。本报告不授G4、formal execution、锁主分析、开holdout或human认证。
