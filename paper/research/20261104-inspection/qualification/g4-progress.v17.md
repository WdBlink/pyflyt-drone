# G4开发进度v17

进程监督修复已经实际安装，尚未完成载荷下时钟复核。独立v1发现foreign UID节点没有按声明回退的P2，v2已以foreign root/child实证关闭（41合并检查+2独立反例）；当前两个production文件与已审freeze.v2逐字节一致。原global PS baseline/attach、unbound-root恢复、shutdown及unknown Python orphan检查、signal/birth逻辑和.1s wait cadence保持。只有已绑定root的等待期观测改用libproc递归当前已观测树，native不确定或foreign节点回原fullPS。初始全表native权限失败及中间方案保留；最终4进程20pair空闲微测fullPS75.394ms/native.315ms，不作为实际clock通过或唯一根因证明。见 `review/process-inventory-repair-review.v2.md`、`process-inventory-repair-installation.v1/`。

安装前确认无活model/flight/driver及held/unknown资源，保留原源码和native status；这属于普通研究代码amendment，不修改controller或迁移owner。安装后正常import路径46项process/receipt/clock/admission检查通过。为覆盖新依赖，parent诊断purpose/envelope纳入cleanup与native provider，共6variant源，并保存最终inventory counters；未来pilot源包补齐provider、现owner37源+旧CLI compat与bound transport。实际prepare-only核得103源、108slots（18classic+90LLM）/48outer及固定B1选择；0model calls、0flights、无SDK callback。仅旧63源中的cleanup和driver两文件改变，旧actor、service、physics、grader、prompt和模型profile不变。

下一条单次post-repair dev-L2-00/B4-M1无分数资格保持原case bytes/sampler2026110616、600s/144000步/72000clock+rusage、.1s门槛、GC/epoch/pacer/4thread1和32calls/180000tokens上限。新107源/111文件意图已冻结于 `clock-localization-adaptive-L2-postinventory.v1/`，尚未reserve或运行；source/provenance glue有限独立复核已冻结，4项Fake通过且未发现新增阻断（`review/postinventory-clock-qualification-source-review.v1.md`）；真实预算动作仍未发生。任何clock invalid仍保留，pass也不能填旧SDK或单独授G4。新的完整pilot准备不等于已获执行/科学接受；最终接线/冻结/集成仍待。

历史catalog R1已有限关闭：新只读archive identity路径严格schema/types/finite/dev/native seeds，复用原dataclass/digest并仅把现行mission validation记作元数据，不以它筛掉历史。独立16项测试及12个额外类型/结构/parent负例通过；56个版本/别名保留原semantic，55个现行accepted输入config一致；旧dev-L2-03仍保留原file/semantic/config与ground-proxy rejection，原loader继续拒绝，不放宽实际mission。见 `review/development-catalog-component-review.v2.md`。完整private-origin/使用史对账及最终catalog尚未完成，没有正式样本生成。

实际资源仍为19操作allknown，391calls/2509976tokens，无held/unknown/unresolved；cost/host/资本/能耗未计量，不写0。L2旧诊断1.839878s/17020超界步和旧v2 SDK全部失败/未启动分母完整保留。G1–G3与同级非LLM参考对保持原声明范围，G4仍needs_work。原19项义务、budget={}、正式实验分母不变；actual main/holdout种子/顺序/实例及论文结果正文均未开始。
