# GC localization（一次诊断，非配置比较）

对原始clock-invalid M3尝试的固定执行journal作instrumented non-realtime replay；原数据不改，临时classic-replayer字段仅从已核private input派生。72000 control steps、82RPC、25字段全部exact；不是另一策略或模型样本。最大step=0.8949315419886261s发生t=0，包含该replayer的冷启动，不能当作原worker在役1x时钟指标。in-step GC最大=0.01577433303464204s；本次未复现同等118msGC停顿。预先冻结原JSON堆与新增callbacks改变profiling条件，不能排除或确认原worker的GC/其他根因。

结论inconclusive；不禁用GC、不选快配置、不热改63源，继续原阈值与有限infra retry。profiling后不追加无依据的重复对照；若将来实际新故障提供线索，再作针对性验证。此前host load约3.8/18cores；本次于fresh order2 gate、无活model/physics时运行。所有原模型用量仍由原owner管理，不把本诊断写入generation calls。
