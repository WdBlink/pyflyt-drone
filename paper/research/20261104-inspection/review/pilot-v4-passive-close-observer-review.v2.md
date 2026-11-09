# v4 passive close observer 限定复验 v2

结论：**accept_passive_metadata_scope**。v1的sidecar引用未重核P2已关闭；本轮无剩余阻断。接受仅限精确冻结observer的被动元数据功能，**不接受030 late-cleanup，不认证caller接线、实际飞行/时钟或G4**。原v1 request_changes与失败证据保留不改。

## 版本绑定

审查人为 `/root/control_review`，同模型家族独立工程上下文。freeze.v2 binding SHA=`f2e539dcd6a5bd6ac369e03527962c2a7f5519efb184a9fef1dac8a966c4e55f`；observer源SHA=`83ffb02d0582c243e33421ba0f831d754010c1cb99c70b3022b5a743d96c646c`；测试SHA=`8a6d0f8128e59b9d93029cc573c012837dc3f97ac27a0fd70f652079a0e310d8`。baseline与本轮真实import的cleanup均为 `ce210650067d2b4e6f49c47845dea36e034d2c7c2e1187899023ddf22f792a9d`。原v1报告SHA `deee31342af84ab3a062dc122399b5bc7b75db2079c875ade17d7446f2d14703`。本轮复制冻结成员、baseline、精确source delta、独立脚本/日志/结果；作者25 passed/0.04s日志另绑定，没有机械复跑整套。

## 原反例与相邻边界实证

独立 `probe.py` 以同一真实原 `SubprocessTree.close`、合成库存和fake wait重放v1攻击；只作source/binding路径迁移及按预期拒绝判断，文件I/O均在 `/tmp`。库存只含新同uid Python orphan 99，已登记root不存活；信号函数设为一旦调用就失败。没有实际worker、模型、ps/libproc扫描、native owner或信号动作。

| 检查 | v2实际结果 |
| --- | --- |
| 原sidecar删除 | check拒绝 |
| 合法JSON中把original_result.status由unknown改closed | check拒绝 |
| 换成指向project外同字节文件的symlink | check拒绝 |
| 增加同文件hardlink | check拒绝 |
| 哈希读窗口将OUTPUT父目录移出再设软链 | check拒绝 |
| 第二次close的_write不生成新record | 原unknown返回保留，check拒绝 |
| 修改上次check返回ref的SHA | 新check仍返回原私有SHA，不被重绑定 |

各负例使用独立正常context或在正常侧录后变异，初始check均通过，未用先前sticky error掩盖新的边界。完整结果在 `results.json`。

正常原close和观察后原close的unknown决定逐值相同；本探针原库存调用3次，观察后每次close仍3次，原表及baseline/unknown row/birth字符串保持原值。该次数对应本探针fallback式observe，不外推为真实固定扫描数。写sidecar前已断言 `_inventory` 恢复原函数，context退出恢复原close。

真实I/O负例把OUTPUT变成普通文件，mkdir失败，原unknown返回仍保留，check拒绝观察完整性。原库存抛出的**同一个**OSError对象仍原对象抛出，hook恢复，侧录保存原异常类型。正常文件字节的SHA与预先保存ref相同。未因修复改动原return/error或增加库存调用。

## 源码修复检查

139–143行要求每次本次close新增恰一条记录，不能借用上次record；147–165行从将写出的固定payload bytes预先计算SHA，以该SHA核写后文件并保存私有`_records`，不把后来读到的新字节采作可信基准。167–182行限定原目录/项目路径、拒绝symlink、要求bounded regular/nlink1；读后复核父路径及文件stamp，再比较原SHA。184–196行逐条核私有初始ref，失败标incomplete并拒绝，返回dict拷贝。

原close与_inventory委托/恢复主体、扫描元数据字段及timeout仍可能包含活worker的limits未变。本轮复用前次source审查的线程/重入及原扫描能力边界，没有新增线程/全套运行来扩大结论。检查封闭本次可观测读窗口，不声称返回后永久不可变、内核级进程归属保证或文件系统原子事务。

## 接受限制

独立 `review.json` 可作为该精确observer版本的有限元数据审查证据。caller仍须真正调用check、绑定每次实际close的records并保留检查结果；缓存未发生close时的空records不能被解释为已观察新一集。caller接线、sidecar归档/引用、030精确处置与native应用须另审。

该observer仍不能补回扫描间丢失的祖先历史、认领或排除86715，原030 unknown和原失败分母继续保留。没有运行continuation、新资源、MLX、PyFlyt、正式RNG或native写动作，没有改变104生产源码。没有实际载荷开销或时钟资格证据。报告/JSON/binding完成后停止本轮CPU检查和R写入。
