# v4 精确030恢复与双处置caller实现审查 v1

结论：**request_changes**。原030事实、024既有接受链与联合闭包重核通过；原四窗口保护仍有效。但新caller在delegate执行后会接受重新取SHA的observer review，未维持初始独立review引用，存在1项P2。未签发030 `accept_late_cleanup_only`。此结论不改变两个原unknown、原SDK分母，不授continuation/G4/formal资格。

## 对象与检查方式

审查人为 `/root/control_review`，同模型家族独立工程上下文。helper源SHA=`722eedd58a0ed0f2f47e7db916047ef6391e24677f527fe06d9d0a8be305c098`；proposal.v2 SHA=`a4307b9ade56c82ddd60dc3af6d116be8b4525f8146545f43b17ce862ee09ced`；freeze binding SHA=`2e2053d17db9073ee65e62ce2cf071a9723dbec5d1a898f035910764a05a23cc`；测试SHA=`179e89b6998375ba9ccdbe7c2a12808f0b64cd770cc08b864c0a2414bb0444bd`。冻结五成员复制到本轮evidence。作者52 passed/0.80s日志单独保留；早稿/fixture失败不属于实际通过证据，也未被本轮覆盖。

独立检查仅包括：只读actual inspect/旧024verify/原ref和闭包核对；隔离native-shaped owner的verify正反例；原样AST抽取nested `recovered`函数、真实已接受observer及真实 `/tmp` sidecar I/O、Fake delegate。没有调用outer `run_continuation`、SDK main、实际worker或产生资源effect。原源码、记录、.paper均未修改。

## P2：caller把observer review的新字节重采为基线

`observer_refs()` 53–63行固定observer83ff源SHA，但review只在54行对**当时**文件取新SHA，再按有限语义检查。`recovered` 358、389行都重新调用这个函数，没有将结果与context在353行持有的初始 `passive_observer` refs比较。365行仅在delegate之前比较companion语义，delegate/两个精确reconciled返回前未按最初companion SHA终验。

实际最小反例：先以原33ed review正常运行Fake delegate和真实observer侧录，正控通过。第二次Fake delegate在close正常完成后、返回callback前，将fixture中的observer review改为仍满足字段约束、但reviewer=`unaccepted-new-reviewer`的新JSON。原函数成功返回Y=0.1及一条侧录；这只是Fake值，**不是原030 Y或新飞行**。companion仍持原review SHA=`33ed4010f841d2f935639b854e77c435aa9f36b206baf1a6e052230bcf341f23`，当前review已变为 `820dc7f8d3aaef338da40966286a157974527d91dd22eebe233bb20de88ad8a1`。新字节没有取得原独立接受，却被返回前动态重采掩盖。原件未改，所有变异均在/tmp。证据：`probe_caller.py`和`caller-results.json`。

最小修复边界：显式pin当前实际独立observer review的33ed SHA；每个callback在delegate/old verify前持有observer固定refs及companion初读path/SHA，语义核对后，在新delegate和两个精确reconciled分支返回前都对照初始refs终验。不要在验证后重新取当前SHA作可信基线。新delegate应在全部外部引用检查之后最后再执行一次 `observation.check()`。companion终验缺失是相邻source-only发现，本轮没有将它冒称第二个已实证I/O反例。上述修复不需改变原104、原unknown或科学阈值。

## 030原事实与024链的实际只读重核

独立实际调用 `inspect_original()`、旧reader `verify(OLDER_ACCEPTANCE)`，再核联合closed_inputs及全部proposal.files，完成用时2.2909s：

- 030 Y=`0.3436623024638006`，24calls/173756tokens、24causal joins；原evidence canonical SHA=`bb7a0d2b5aa731d1d03a39fd744884420e5f8d4eb05b2944b08cb9ad71ce6337`，与proposal及原terminal一致。实际旧冻结source/deps/inputs和native known用途检查通过。
- 024 Y=`0.368353660488492`，原bb196 reader与e4cd acceptance通过既有native applied finding读取；返回proof与proposal.prior_cleanup_evidence精确相同。
- proposal全部104个ref逐SHA匹配；联合闭包61目录/189文件与原snapshot相同。该闭包同时覆盖两套episode/model/resource树。
- 原030仍 `unknown_process_cleanup`，原terminal Y仍null，当前没有030实际acceptance；本轮只读结果不能代替新finding实际应用或caller修复。

`actual-results.json`保存结果。没有复制真实owner全文或手写native状态，没有因024已接受而代授030。

## 原窗口和caller范围的有限检查

隔离verify fixture使用真实 `require_native_binding`、原030 plan/terminal/tree/receipt字节拷贝、合成review/native-shaped owner；payload集合stub仅用于隔离返回窗口，旧024 proof/closure在该fixture中是可信stub，另由上面的实际024verify补证。正常正控、恢复后的正常正控通过；以下均实际拒绝：archive在owner窗口换字节、acceptance换字节、新空call目录、新FIFO、错误repeat、缺applied finding、把finding ID换成旧024 ID，以及prior024 proof变化。没有伪称这些合成row是真实native应用。脚本与完整输出在 `probe_windows.py`、`window-results.json`。

source tracing显示：两精确分支分别核024/030 plan/prefix并调用各自reader（366–379）；030返回前写下一新effect前checkpoint marker。其他调用委托原API，正常返回后核新attempt侧录数量和observer check（380–393），旧cache要求零新侧录。source/observer指针与新SDK目录、原order/sampler约束也保留。完整29 strict+2 reconciled=31-cache的providedSDK重建、marker实际停在order31前，**本轮未运行，仍待实际prefix核对**；没有把作者Fake接线当作真实31cache证据。

未扩大到旧失败部分重启等未实际存在的slot状态；本轮发现足以阻断当前caller接受，修复后应另版有限复验。原52软件绿不关闭该新窗口。报告、request_changes JSON及绑定完成后停止所有CPU检查与R写入；无模型、物理、新resource、正式RNG、信号或native写动作。
