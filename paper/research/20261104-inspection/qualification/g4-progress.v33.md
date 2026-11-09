# G4 开发继续点 v33

G4仍needs_work。v4现18/108个有效callback，首个dev-L1-01完整物理root block已完成（八配置、经典各1次、LLM各3次，共18槽）；其余90槽仍按原计划保留。provided SDK的最终完整outer recording尚未产出，不把本地首root描述摘要冒称正式SDK结果。仍无实际main/holdout namespace、seed/order、dataset、主分析锁或论文结果。原19义务、RA-L、budget={}、600s/144000步、100ms门、两attempt上限、B1原选择、17源grader与global6均不变。

UTC2026-11-06T20:44:18.795973Z，同一supervisor exec4162确认order18（dev-L3-01/B4-M1）预约/spawn前barrier。首root18槽均attempt00、completed/return0、600s/144000步、0超100ms、physics_runtime.errors=[]。整个block maxlag为0.08571254166625408s；新增order15/16/17为B4-M2 repeat2、B3-M1 repeat1、B4-M1 repeat2，max分别0.0118115420009417、0.012330584002484102、0.013777959000435658s，P95分别0.0035612916677791873、0.002634958333260329、0.003005667000252288s。原world/plan/terminal/SDK raw/callback/order/settlement及当时stdout/pause/status字节已绑定于 `pilot-v4-first-root.v1/binding.json`（SHA256 34ea02afc84e4cdff84adcf420c2f10ad9cb06bc39fc4c286ba9b51341d627f0）。这是普通时钟摘要/最大见证，不是完整72000-phase向量。

新增三模型槽实际用量依次27calls177098tokens、26calls173671tokens、24calls172076tokens。唯一owner实际status为known883calls5634314tokens、reserved0、unknown/unresolved=[]、stale=[]；money/host/Codex/capital/energy未计量。容量模板中的合成条目不能并入这些实际用量。

首root描述性结果显示经典三策略S=1且H/E=1；LLM各配置的S均较低（B3-M1/B3-M2/B4-M2为0，B3-M3均值1/36，B4-M1均值1/12），仍保留固定C/S/D/Q/H/E/Y/SR与全部原重复。L1的D=None；未重新挑B1、改评分或把格式失败转成基础设施重试。单个L1开发root不能支持总体方法优劣或模型能力结论。

M2闭合输出诊断发现明显架构依赖：B3三重复83条可解码/5条格式失败/1条deadline后忽略；B4三重复2条可解码/70条格式失败/3条非完成或过晚忽略（包含未启动请求，不全是生成）。B4第一原始service响应就有重复与非法编号，短JSON smoke此前正确输出1call/51tokens，故既不能归为解析器制造错误，也不能称整个后端失去生成能力。MLX上游历史4-bit循环问题只作线索，当前没有确定、可复现的实现修复证据。保持registry、权重、temperature/top_p/thinking、prompt与解析器原冻结组合，继续pilot。见 `planning/local-model-output-diagnostic.v1.md` 及 `local-model-output-diagnostic.v1/binding.json`（e9ed4c9015839399cc6171263598176510004e77ba63143631fb567b1e71d50f）；没有新增诊断推理或模型资源动作。

独立source-only整图预审未见确定的schema/path/producer死路；actual runtime必须为d809 owner包，SDK保持原稳定包，不能以旧直接Controller取代现行owner。44source/27tests映射不等于最终全source闭包。合法one-off writer/native API可以完成已存在typed协议下的G4 subject/review/acceptance、dataset review、execution与主分析lock；这些产物必须基于真实证据，源码可连接并不等于已产生接受。报告 `review/formal-wholegraph-preflight-review.v1.md` SHA c9702559ebce86dc3fd6269330555785edb122b381334a16fa65f7715cad6657；binding SHA f8ba074465bc782fe7fb58e5d8e69e60dbf7e7f820b2482a05efd45ee21a6ad5。未运行整图新测试/模型/物理探针，未授G4/main/human接受。

独立容量预检设计限定2016=1728LLM+288classic、两attempt至多3456LLM reservations/6912 reserve-settle operations，另加语义round及其他操作；单个请求不是新reservation。source-only报告 SHA b448c5e58ad5288b27b277274195611c5bca26aa5209f3bfafd196231874e46e，binding SHA2a677490b5d7db0fe2d03a8021d66d5accc81bc3f8387f5421418476950241c7。root仅在actual pause后做只读事实与内存表示投影，无init/migrate/reserve/settle或新ledger。

fresh原control SHA2284d2759f78ef0f8aa29abf9d8ca3389ef14d5338cd1e9746805a15673c1206、compact62542961bytes；原pair另存offline ZIP，CRC及解压SHA吻合，不是active owner。复用旧probe的13000未来ops/5000attempts/2000round模板投影control128056961、checkpoint128057047，低于134217728bytes，余量6160681bytes；当前round-envelope模板18469bytes、最大usage-entry模板3386bytes，不能承诺未来更大载荷仍通过。另一3456entry形状投影的compact result167885、pretty wire251916、CLI trace174970bytes，未到16MiB单对象门。原费用缺失仍保留；没有伪造未来usage receipts或实际owner operation。只读完整status实测exit0约0.86s；库存/shape probe1.23s，进程峰值RSS505577472bytes。未冒称已测保存/fsync/恢复/导出或120s整链峰值。

closed L1观测最大episode原件27819190、world26167915、model原件3098662、complete-attempt ZIP10567091bytes。按观测最大样本复制到2016/4032次attempt的原件+model+ZIP估算分别83633645088/167267290176bytes；它不是L2/L3/失败的worst-case界。额外peak wire/trace与unique-result对象模板估算分别3804408032/1496191120bytes；其2000额外wire调用只是披露的形状假设，不是任务预算或无限增长保证。实测同文件系统free187914485760bytes；最终round-doc/分析/导出并存峰值与余量尚未闭合，不能授storage/capacity通过。证据 `formal-capacity-first-root.v1/binding.json` SHA edd4e53deecc01e4b8c5a99152a4b11f8c043be94a66d283d702acf67d645a6d。pilot结束、L3/失败证据及final freeze后必须fresh重算。

生产104source/61deps/10inputs未改；formal候选仍隔离未安装，既有522软件检查不扩成真实正式数据。两source-only审查者已完成并停止R写入。本轮记录后恢复同一pilot order18，继续固定108分母；actual clean/20s-delay集成、最终source/model/prompt/generator/split/retry/hardware冻结、最终容量/完整图预检和独立G4决定仍待。G1–G3仿真资格与human_review_pending边界保持，G4通过后才进入正式实验、图表与论文。
