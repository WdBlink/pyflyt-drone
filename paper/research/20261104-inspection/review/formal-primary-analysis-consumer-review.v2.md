# Formal primary analysis consumer v3：有限修复复验

结论：**原三项具体时序反例已关闭，P3 CLI JSON已关闭；但同一receipt闭包仍有一项P2递归目录漏验，v3暂不能作为无阻断分析消费者。** root13/406软件通过不覆盖新发现的空call目录语义。root后续working v4修补不属于本报告，原v2/v1报告与失败证据不覆盖或改写。

subject：`planning/formal-admission-candidate.v3/primary-analysis-freeze.v3/binding.json` SHA `e24c021d1ddf297712c756bab2a06d2fe9c0eb3a4d67bb2c8f76e5b6d276186e`；entry SHA `f08b6d6218bfe8b6441eed7ad6df7d1a32d6b7603777e131d5f503c1a83c4d2a`，新test SHA `5fce217fa494e3817b8fd995393eda1a479c0bd51177fc02fc1a8104ac0ae4aa`。五明确成员的冻结SHA/bytes均核对，独立probe启动时working五成员也同SHA。另三成员（REQUIRED_SOURCE增量与两个数值模块）与前v2完全相同，未改主数值规则/分母/权限。此后root已获知checks结束并开始有意working v4编辑，本报告仅绑定immutable v3与probe当时guard，不将后续差异算作v3漂移。

同模型家族独立工程上下文。全程在root明确保持的order6预约前barrier做限定Fake/CPU检查，无模型、physics、真实formal RNG或nativeowner写动作，不重跑13/406或整个架构。独立脚本/结果在 `review/evidence/formal-primary-analysis-consumer-v2/`；原脚本仍强guard旧subject，本轮适配另文件。首次适配运行因新fixture已创建world目录而FileExistsError，改本probe的mkdir为exist_ok后重放完成；不是产品失败或源修复。

已关闭的具体路径：

| 独立重放原反例 | v3实际结果 |
| --- | --- |
| trusted measure返回后把plan.case_sha256改全0 | `closed receipt input changed after validation`，拒绝 |
| 同边界把terminal.status改unknown_model_usage | 同上，拒绝 |
| collect后、statistics入口改变world字节 | 同上，拒绝 |

三例都实际检查 `analysis/main.v1`、`experiment/main-recording.v1`、`experiment/results.main.v1.json` 全不存在。它们使用real SDK完整1296 synthetic grid、Fake physical validated-return API、固定统计替身，只验证绑定窗口，未声称真实formal receipt或科学数据。v3 `_receipt_snapshot`(:40–63)在验证前固定episode/按原operation_directory求得的operation/canonical resource/reserve/settle；plan通过预快照Artifact同buffer解析(:186–196)，postvalidation、collect-end及publish前只比较原快照(:197–199,244–246,329–330)。独立新增operation文件与canonical settlement字节变更均被同一守卫正确拒绝。

P3关闭：对冻结 `main()` 使用无效果分析stub实际捕获stdout，含Unicode与false布尔的结果可strict JSON解码且等于原对象。`:385`为json.dumps(...allow_nan=False)，不再输出Python repr；这个正例不运行真实G4/owner/collector，也不让subject变成lock或accept。

**仍开的P2：有语义的空目录变化绕过完整闭包。** `analyze_inspection_formal.py:54–63` rglob只把is_file节点加入files，返回directories仅列base根目录，没有记录其后代目录。新增空call-NNNN不会改变files或bases，`:66–69`因而判“未变化”。model meter的call枚举本身依赖目录结构：原 `inspection_model_receipts.py:69–78`枚举call-*并要求每个call有完整三文件树，所以这不只是无关的空目录装饰。

独立短单元证据先手工构造**一个有效的合成one-call journal**，用原meter实际验为1call/4tokens；这些是literal软件计量字段，不是真实模型生成/资源操作。正常snapshot和守卫均通过。然后仅新增空 `call-0002/`：

- `_receipt_snapshot(...) == before` 仍为true，`_check_snapshots`错误通过；
- 同一个原meter立即拒绝 `incomplete or unexpected call record tree`；
- 删除该目录恢复正常；普通新增文件与settlement变更仍拒绝，说明不是仅被某一总SHA gate掩盖。

这是最终发布守卫实际调用的helper的可复现误收；本轮没有为了这一相邻点再跑第四个全1296输出反例。它可使验证后出现的部分新call/未知操作目录不进入最终metadata manifest与复查，而真正重读meter已不合格。最小修复是把所有递归目录（以及声明要拒绝的非regular entry类型）纳入同一个预快照/后验，保留symlink拒绝；新空call目录或新增未知entry应在输出前拒绝。不要重新采新目录集作trusted baseline。此finding限本函数可观测验证/发布窗口，不要求无限返回后原子性。root承诺的v4目录修复尚未在本报告复验。

其余修复源码范围：provided SDK _experiment现在从captured bytes构造私有TemporaryDirectory再读取(:163–173)，避免重新打开原SDK不同版本；收集结束仍核原SDK bytes，复制保留captured原字节。原三项独立重放均经过此新收集路径，但本轮未另声称做过完整新的SDK竞态探针。raw-path-only projection、main/holdout既定repeat/statistics及matched-level描述不变，原数值模块SHA和三项REQUIRED_SOURCE没有再变。

新 `receipt-closures.zip`按slot保存原快照metadata，索引绑定ZIP与各member SHA(:348–362)，比把全部路径塞入单个JSON更符合16MiB读取边界；metadata并不声称自己是原始flight bytes。该设计避免体积问题，但目录漏验意味着v3 manifest的完整性声明仍须上述P2修复。root13正例核了新ZIP/SDK投影；本轮对其源码/已有日志取证，不再重复整套正例或认证实际raw真实性。

root实际日志已只读核对归档：focused13 passed45.24s（SHA `004e9a8e2b6c3c30f76b7e7595b5455e23fabaef83464174302e9f7209d68054`）；全suite406 passed120.07s（SHA `32f58c1fb0fee65861ccaf1c438b415e75e30c88034a9bc540d7f34b32e99724`）。compileall exit0由root报告，保存日志为空，单独空日志不证明exit状态。这些不关闭独立目录反例，本人未重跑13/406。

结论只覆盖v3限定返工：保留原三闭合和CLI修复，同时保留新递归目录P2；不认证未来working v4、全graph、G4、实际main/holdout/正式flight或human review。所有正式namespace/data仍未生成。报告/脚本/结果及绑定冻结后停止R写与集中checks，供checkpoint继续；下一版应只复验目录/entry这一小delta。
