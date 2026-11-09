# Controller state-reader迁移独立工程审查 v1

日期：2026-11-05。实际审查Agent：`/root/control_review`，同模型家族独立上下文。范围只含runtime读取上限、显式迁移、版本绑定与状态保全，不是paper内容、研究结果或科学审批审查。只写本报告和配套真实review JSON；真实ledger始终只读，测试仅在/tmp fixture中运行，未修改全局skill、production runtime或真实状态。

## 结论

**批准指定source的state-reader迁移。** decision=`approve_state_reader_migration`，preserves_contract=true，open_findings=[]。未发现阻止此限定迁移的必要修复项；批准不等于真实迁移已执行，不授予用户付费、预算调整、科学门、G4、paper交付或完成。

从旧实现`bfd85afe5d6745b4111d67f3c7d79bef4b0b3e15894b356b1468c180e48a9df2`迁往候选`a112e7aaff3ab6c99f983a2030fc190d7ccd1af940a943604d94ad318e8fef20`。只提高controller-owned control/checkpoint读限至128MiB；外部packet/review默认16MiB、重复key及literal nonfinite保护保留。迁移保留研究/资源状态、旧历史及旧operations，追加迁移事件与operation，失效旧rules下的pending request并要求重新prepare。

## 实际绑定

R为paper/research/20261104-inspection，下列路径相对R。

| 对象 | SHA256 |
| --- | --- |
| recovery/state-reader-v1/candidate-subject.v1.json | 1f287403b5381e1ac6bd284ff05229947d4487c853d426e2bea4a30ceb5a5842 |
| runtime/autoresearch-state-reader-v1/lib/runtime.py | 5f887d7b188203a3beda96127b055d4e5009017eced0533f55afacc2a3c19f52 |
| runtime/autoresearch-state-reader-v1/scripts/research.py | 5ca105c6a26cd631ca6c5cd70a1698677b23b924a0f26a0579a349607121d7a0 |
| runtime/autoresearch-state-reader-v1/tests/test_state_reader_migration.py | a92f4ca659c6733d020f62c18b3f17a0d08787732fe8eede4478023ea44bcbb9 |
| .paper/control.json（审查时） | 22b2b35b5c5a979b0da0e018e40f7aff106a0763355d5a39f95ba431ff67a1e0 |
| .paper/checkpoint.json（审查时） | dfa9f69a649450052fc8d6c0b1277e4098feb9ee46725373fe68aa0398df8321 |

原global `/Users/wdblink/.codex/skills/autoresearch/paper`与R/runtime/autoresearch-original计算的implementation均为上述bfd85afe；原lib/runtime.py及scripts/research.py字节也分别相同。候选与original的完整文件树（忽略__pycache__）仅两份既有文件变化：lib/runtime.py和scripts/research.py；仅新增test_state_reader_migration.py，无删除。

实际Controller.implementation只覆盖既有lib/reference/vendor指纹范围，CLI entrypoint不在该既有aggregate中，因此本审查另外逐SHA绑定上述entrypoint与subject；批准只适用于这些未再改动的指定字节，不推广到同library fingerprint下任意另改CLI。配套review JSON包括subject及source哈希，迁移packet必须绑定该真实JSON哈希。独立身份的真实性由本次实际Agent审查事实提供，非把非空reviewer字符串当作身份认证系统。

## 真实故障与pair只读核对

control为16,754,137B，尚小于16MiB；checkpoint为17,326,544B，超过16MiB。旧CLI在apply round15期间返回`JSON exceeds the 16 MiB input limit`与此一致。/tmp/inspection-round15-applied.json为refused，不是一次完成的round apply。

两个真实文件分别与recovery/state-reader-v1的.before逐字节相同。canonical state digest为`3ebd2c22bf02b642e6fdb6bf1d002d5144112477f1e02f32c7cf7f354b863397`，等于checkpoint.sha256及checkpoint.state重算值。schema2、Brief hash、旧implementation绑定完整；history36、operations19、已录rounds14、active round-15保留。pending request为`d8ce9f78d858c9026192127a4eec4247a6bbc6e44513b7154fb3bce5e8152eb9`，其原control/round-decisions请求文件存在、与pending对象相等（3,579,739B），不会因清理pending指针而丢失原请求artifact。

没有调用真实Controller的load/lock/save或migrate。真实pair仅用有界read_json、digest和文件hash读取；审查完成前再次与.before核对。

## 源码机制与保全

- runtime.py:48–61保留同一个JSON parser；默认16MiB，仅显式max_bytes可提高且必须为native正整数≤128MiB。扩大读取的调用点仅load:208/216、recover:722/727及migration:743/747的controller-owned pair；review:766及CLI input_json:16–24仍走默认16MiB。
- load仍核schema、Brief、pair digest及implementation；新runtime不能直接接受旧implementation，须显式迁移。recover仍要求最新durable checkpoint、当前pinned implementation和非完整live pair，未改成任意旧export覆盖。
- migrate:737–771要求精确packet形状、完整pair、schema2及原Brief hash，旧snapshot实施指纹等于state/from，新implementation等于to且不同from；实际source-bound review须匹配hash/from/to/decision、preserves_contract=true、无open findings，且不接受codex-primary自称审查。
- :772–785仅调整implementation_sha256和pending_round；其余研究/资源字段整体digest前后一致。历史先追加含previous_history_sha256、旧pending SHA及真实review绑定的迁移事件；record:889–902只增加新operation和对应history，不覆盖旧operation。active_round、rounds、Brief、approval、budget确认/usage、attempts、receipts、bindings等均保留。
- operation按request digest幂等；同operation同参数返回已存结果，不再次写pair，变参数拒绝；换operation也不能在已迁入新implementation后重复执行from→to。旧pending是旧state/rules下的prepare产物，清除后不能直接apply，原request artifact保留，需新rules重新prepare。

迁移没有调用科学完成、预算确认或资源执行逻辑，也没有清除uncertain attempts或task_stop。之后研究行动仍走既有ready、预算/人类回答及stage校验。这里的preserves_contract是对状态与代码差异的判断，不是重新批准原科学内容。

## 实际有界验证

候选现有3项迁移测试独立运行，3 passed in1.336s：真实旧CLI拒绝17MiB fixture、迁移字段/历史/active round保全、精确重放、外部16MiB、重复key/literal NaN及129MiB越界拒绝、错误target/review/prior拒绝。全局skill未被用于写入，旧CLI只作用于临时root。

另用17MiB临时ledger独立检查：未review的候选load拒绝旧版本；缺review、错review target、自评codex-primary、有open finding、否决review、破损pair、已改Brief、未知schema、改过的旧snapshot、linked checkpoint均拒绝，拒绝前后该fixture pair字节不变。正例保留所有非4类迁移字段、全部旧history与operations，恰增2项history/1项operation；原历史digest和失效pending SHA对应正确。同operation重放不写入，变参数或另一operation的再次迁移拒绝。

独立加入uncertain attempt及reserved usage-shaped字段，迁移原样保留；后续ready仍拒绝`Uncertain effects require reconciliation`，没有把读取修复变成执行授权。脚本`/tmp/audit_state_reader_migration.py` SHA `f44d013f274b1dca0d67413f235b9680808889e3faccf36bd180a9872eba01a1`，结果`/tmp/audit_state_reader_migration_results.json` SHA `a9fd9667b8886859c169b6912702c2160b2f2fe3bd4ae42a90cf4aebc9cd0c09`。

另注入checkpoint已durable、写live control前抛OSError：load拒绝mixed pair；新pinned runtime的recover从最新checkpoint恢复完整迁移operation、原history/Brief/active round/rounds，pending为None。只在临时fixture验证，未模拟真实进程/OS掉电。日志为/tmp/audit_state_reader_migration_crash.log。

主Agent提供的66项运行日志/tmp/inspection-state-reader-tests.v5.log已读取，66 tests OK、skipped1、26.729s，SHA `63d7bc6e636e682ab07e9be1cc92d7742629a64c61943ea83b02c45510e3abf6`；本审查未机械重跑全部66项，不把其自检当独立批准。此前import/路径配置失败历史未被删除，也未解释为当前源自动通过。

## 适用限制与Attention

reviewed by /root/control_review, inherited model configuration, independent context

- 无open findings，批准指定bfd→a112状态reader迁移。真实ledger仍未迁移；执行方须使用未再改动的candidate、真实review JSON及新operation ID，随后重新prepare round15，不重用旧request apply。
- 128MiB仍是有限上限，不是无限历史容量或长期增长问题的完整解决；实测大fixture为17MiB，129MiB只验证边界提前拒绝，没有做128MiB满容量性能/掉电测试。
- 保留既有审批/预算/研究约束，未增加付费/论文/科学审批、不授予G4或完成，不把独立工程review JSON伪称用户或人类领域批准。

本报告与配套JSON冻结指定source和只读状态观察；之后任何source/target变化须新审查。
