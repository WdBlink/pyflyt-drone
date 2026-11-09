# SDK 记录尺寸的定向检查

这是首root之后、第二root在飞时的只读源码检查；没有导入生产模块/NumPy、运行probe、读大world或改生产源。它补充容量计划，不授最终容量/G4通过。

提供的 `runtime/autoresearch-state-reader-v1/lib/experiments.py` 中，`ExperimentRecorder._record` 把完整evidence只写入逐slot raw文件；`self.rows`仅保留case_id/seed/condition/status/value/id/raw及可选reason/failure_kind/attempted。`_finish`的results.json汇总的是这些短行。因此不能把1296份完整world、service计量或score展开乘到同一个SDK results.json里。当前closed `logs/inspection-eight-condition-pilot.v4/llm/raw/dev-L1-01-B4-M1-2.json` 为983bytes，其evidence保存回执路径/SHA等引用；该尺寸不是future formal raw的上界。

候选formal `run_inspection_formal.run_primary` 的measure调用 `measure_completed_episode`。后者将完整score放入单个raw的evidence，同时原model计量与世界仍由闭合episode回执引用；经典/LLM的reader分别重计完整score。分析器 `analyze_inspection_formal.collect_recording` 对每份SDK JSON单独应用16MiB界，逐份same-buffer SHA/JSON捕获后才交provided SDK检查，并不先把全部raw evidence串成一份大JSON。因而现阶段未发现“main1080个LLM slot使results必然超过16MiB”的确定故障；逐raw最大尺寸、完整completion/index、分析摘要仍须用最终L3/失败与正式布局做最终容量检查。

实际round47 baseline documents_ref对象为5068311bytes/27926条，尚小于16MiB。future资源request/receipt、正式dispatch、SDK原件与分析副本都会增加文档map；不能沿用旧projection里的30000 count字段当最终文档数上界，也不能以本次单map尺寸推出全部未来map累计存储已合格。以上结论仅涉及所读数据结构；没有新建正式case/seed/order、SDK envelope或native操作。
