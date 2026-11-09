# G4 开发继续点 v71

原108飞行与SDK完整闭合已由native round85记录，active round86进入最终开发集成。实际安装44个审阅源（20新建、11相同、13替换）与27个候选测试；替换前13个源已保存在 `qualification/formal-installation.v1/before/`，原v4与staged证据保持不动。安装回执SHA 9ae381e69cae2a253422ffdc0c76541db488570c620b6b464afabc6f4007bf7e。

安装后完整 `pytest -q tests` 970 passed/187.85s、exit0；compileall exit0，44个运行源逐个与审阅字节相同。测试按实际路径迁移，旧AST和生成器比较从保存的安装前源码读取；避免在测试收集时替换共享population模块。普通不产生孙进程的合成回执fixture局部隔离PPID1宿主清理噪声，真实孤儿进程测试仍用全扫描。原收集28错、共享模块2失败、随后原测试1次实际清理unknown及其原证据均保留；没有归因该宿主PID或放宽生产清理。定向60/31项通过，整体970通过。完整安装验证 `qualification/formal-installation.v1/verified-installed.v1.json` SHA 26d051085a5de6cd344090a34e9331b729705a39df099eea2766c13e9bbeb096。

下一条是新安装源码下的B0清洁开发飞行，复用既定同级对的devext-OOD-L3-00 clean case，仍是600s/144000步、100ms与最多两次的基础设施合同，配对事件侧使用既有实际飞行并声明源码时代差别。先决意图 `qualification/final-development-clean.v1/intent.json` SHA a326be1dc4b7a78cb9690a0a1aa6c7be6776f978fd97e3d36b5f9fc930bc92d0。还须完成真实B4-M1 20秒臂、最终容量/冻结与独立G4。当前无新本地LLM资源操作、无正式数据或论文结果。
