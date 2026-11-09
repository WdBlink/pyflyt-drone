# Primary consumer：PC5-b 最小独立复验 v3

结论：**PC5-b（P2）在此次限定 delta 中关闭，未发现本范围新缺陷。** 原 nested false/0 反例及 meter 的 order/repeat bool、sampler float/int 反例，均在旧冻结 v2 被接受而在两份新 source 被拒绝；合法 typed binding 保持通过。仅复用 v2 已完成的 PC1–PC4/C1 有限结论，不批准 working、secondary 全图或 G4。

## 输入与隔离

reviewer `/root/statistics_design_review`：独立上下文、同模型家族，非人工/跨模型复核；遵循 autoresearch 证据边界。

- 资格 delta binding：`qualification/primary-nested-binding-repair.v1/binding.json` SHA256 `0c2e61c08a174ed165c179e1ba56e889f9b04d2dbfcf56773d47d0066f7b037a`。
- service 副本 SHA256 `22c35639ecd5f0d79127c706f7c5b23ec421bdf7afd1a2c5a2480621d154e2dc`。
- receipts 副本 SHA256 `f68f149960254ef9e50a47354722d7f64d83417adda26a845fb076847dab6a49`。
- base 为已审 immutable consumer-repair v2（binding SHA `4ea0511c325f47a15c5463a7ad2b137d48864ddf60b46a443b2f7442f5616ef0`）；原报告 v2 SHA `15405679e6082c710b86845d675d13e62c27ca17e094fe1b55e3d918577554e7` 保持不变。

此前 `/tmp` overlay 已不可用，因此从同一 immutable v2 subject 逐字重建 `/tmp/formal-primary-nested-binding-review-v3/project`，保持 HERE/project/runtime 层级及优先 copied eval imports；仅覆盖资格副本的2 source与对应2 tests。其余34 source与base完全一致。旧 meter literal fixture只作复制/记SHA的测试资源，不是后来 working 输入。未读取或 import 后续 working candidate。

source diff 只有两处比较：service 将 nested binding 普通 dict equality 改为已有 `_digest` 的 canonical 比较；receipts 将普通 equality 改为 sort_keys / allow_nan=False 的 JSON 字符串比较。未发现相关 caller 或其他流程变动，故未重跑 PC1–PC4/C1、SDK矩阵或旧完整组件。

## 实际验证

```sh
PYTHONPATH=/tmp/formal-primary-nested-binding-review-v3/project PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -p no:cacheprovider <overlay>/tests/test_formal_model_service.py <overlay>/tests/test_formal_meter_types.py --basetemp=<home>/provided-fixtures -q
PYTHONPATH=/tmp/formal-primary-nested-binding-review-v3/project PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -p no:cacheprovider /tmp/formal-primary-nested-binding-review-v3/test_independent_nested.py --basetemp=<home>/independent-fixtures -q
```

其中 `<overlay>` 是该 project 内原六层 tests 路径，`<home>` 为上述 `/tmp` home；解释器实际使用 repo `.venv/bin/python`。**附测18 passed in0.26s；独立8 passed in0.14s。** 资格日志中的作者18项声明没有被直接当作本人结果；这18项本次实际重跑。Python3.11/NumPy1.26.4环境，仅软件/Fake/IO，无实际formal RNG、MLX、模型、物理、native写或child/service spawn。

独立8项覆盖：

| 数据 | 新值对 typed expected | 原冻结v2 | 新delta |
|---|---|---|---|
| service nested repeat_ordinal | false 对 int0 | 接受 | 拒绝 |
| service nested order_index | false 对 int0 | 接受 | 拒绝 |
| service nested sampler_seed | float70.0 对 int70 | 接受 | 拒绝 |
| service合法typed binding（key重排） | native ints不变 | — | 接受 |
| meter nested repeat_ordinal | false 对 int0 | 接受 | 拒绝 |
| meter nested order_index | false 对 int0 | 接受 | 拒绝 |
| meter nested sampler_seed | float70.0 对 int70 | 接受 | 拒绝 |
| meter合法typed binding（key重排） | native ints不变 | — | 接受 |

service每项负例除直接调用 `admitted_limits` 外，还实际调用其 main()/argparse入口；均抛原 `primary slot` ValueError，未触 `verify_model_files` / operation-directory hooks。这个入口在load前，per-call使用同一admit函数；没有运行真正model load或生成循环来声称硬件证明。meter正控重算为literal calls1/tokens4，负例以 `formal dispatch differs` 拒绝，读取前后所有fixture bytes不变；不是实际owner计量或真实请求成绩。

canonical序列化区分 JSON false/0 和70.0/70，并对object key重排保持不敏感。本结论以 expected 来自已审 typed dispatch为前提，不把此比较单独升级成任意不可信 expected 的schema/native/G4验证。

## 测试停止与结论限制

所有测试进程均exit0，无pending exec session；本轮从未spawn child worker/service。日志UTC元数据窗口（shell-open birthtime→最后输出mtime）：

- 附测：2026-11-06 **13:59:51.847328–13:59:52.248084Z**；
- 独立8项：**13:59:52.275871–13:59:52.511853Z**。

14:00:42.418537Z记录并及时告知主Agent集中测试已结束；之后只做报告归档，不再测试。元数据窗口不是内部精确采样点，不对后续单次telemetry作原因推断。

v2其余有限修复结论沿用，原证据/红测试不改。本报告仅关闭PC5-b及所列meter类型差异，未认证full primary/secondary graph、最终source inventory、真实formal receipts或G4/main；不以两行修补批准实际模型/飞行。
