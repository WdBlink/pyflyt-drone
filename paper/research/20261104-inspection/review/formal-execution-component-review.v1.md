# Primary formal execution plan/slot：独立有限组件审查 v1

结论：**本次限定组件审查未发现实质缺陷。** primary plan 保留 main/holdout 的完整 root/18-slot 矩阵，sampler 与原有算法兼容；仅重算 hash 不能改变已 native 绑定的 sampler/order。末 native snapshot 的 source、dataset、revocation 和前后顺序反例均拒绝。此结论仅适用于未安装 plan/slot 与 guard 抽取组件，不授 G4、实际 primary dispatch 或 runner/service/receipt 全图准入。

## 来源及实际执行

reviewer `/root/statistics_design_review`：独立上下文、同模型家族，不是人工或跨模型审查。按 autoresearch 保留正负证据，所有新测试均明确为 `/tmp` literal/Fake。

freeze `planning/formal-admission-candidate.v3/execution-component-freeze.v1/binding.json` SHA256 `4e4a1369ac81d3f61d7eda7a155b11ac40dac88d4c545efd119d749447ee5918`。全部 frozen files 与 15 sources / 11 tests 的 working SHA 独立核对一致。

| source | SHA256 |
|---|---|
| `inspection_formal_execution.py` | `9cfd8fcf18de029ed975e07f7391ddf5381c84819aff538a319f3129dc87e290` |
| `prepare_inspection_formal_execution.py` | `b7fc1391f6b43de6e5bf809122436d6858a7c3a5103f2601bd9919507d4ccd55` |
| staged `inspection_recording.py` | `115ecd64a3e451e7b3ad1990215f780f6fc4939bd1c5a24eb0a642f12ea59ce9` |
| 原 production recording（未变） | `a2fbdddcf7546b3f69f1cc8d2fc1d565620be3ce6582c6f61f8e93923decffcb` |

已读 172 行 reader、63 行 producer、recording 全文/diff、所附测试，以及 original pilot sampler 赋值表达式。G4 REQUIRED_SOURCES 的差异是增加 formal-case/development-helper 与两份 execution source 的显式库存；未改其 predicate 正文，也不等于实际完整闭包已安装/获准。

实际命令：

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -p no:cacheprovider paper/research/20261104-inspection/planning/formal-admission-candidate.v3/tests/test_formal_execution.py --basetemp=/tmp/formal-execution-review-provided-v1 -q
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -p no:cacheprovider /tmp/test_formal_execution_independent_v1.py --basetemp=/tmp/formal-execution-independent-v1 -q
```

结果：**26 passed in 2.74s；新增 independent 17 passed in 3.18s**。26 已包含 1 项 AST extraction，不重复记为 27。作者 combined282 / 单独 AST / 原 unanchored-sampler 红日志仅核绑定并定向阅读，没有把其余 stage 测试计为本人独立重跑。Python 3.11.15 / NumPy 1.26.4；所有实际 PCG64/Generator/SeedSequence/default_rng/randbits 在独立测试中禁止，producer 正例只用显式 Fake 替换。

## 规律、分母与 sampler

所附 AST 检查本轮实际执行：`episode_schedule` 和 `designs` 与原 production AST 相同；helper 的原 guard 正文相同，将 `record_matrix` 的 helper 调用 inline 后，其正文与原版本 AST 一致。没有增加新 scheduler，也没有改 root 内均值/统计设计。helper 只抽取既有完整矩阵与 18/root block 检查。

reader 在 DatasetProof 后验证 typed metadata/header/slot、native int 索引/level/repeat、固定 condition/repeat 集、registered schedule master、registered sampler master、环境标签及每个 LLM order-index 的唯一 sampler key。sampler 数值必须 native uint32；经典 slot 为 None，不把 repeat ordinal 当 provider seed。独立全向量对照：

| phase | roots | slots | sampler entries | outer root×condition |
|---|---:|---:|---:|---:|
| main | 72 | 1296 | 1080 | 576 |
| holdout | 12 | 216 | 180 | 96 |

每 root 恰 18 slot，三经典各 1、五 LLM 条件各 3；case/condition/repeat 三元组完整唯一，18/root 同 root block，metadata 与 DatasetProof 精确一致。这是计划分母，不是已执行 flight/方法结果。

producer 在 DatasetProof、current G4 / `_require_unissued` 后才使用原 `episode_schedule(metadata, registered_schedule_seed)` 和 `Generator(PCG64(registered_sampler_master))`；只对按 order 顺序遇到的 LLM slot 调用 `integers(0, 2**32)`。独立在 main/holdout 两个 Fake producer 输出中，执行**原 pilot 的 sampler 赋值 AST**（同样仅 Fake RNG），逐 key/value/迭代顺序与新文件一致；int key→JSON string key 的变化不改变 map。实际 NumPy formal stream 未运行。

reader 不重新生成 PCG stream，也不凭 master header 证明任意 map 的数学来源；它认证真实独立审查后所 native 登记的精确 artifacts。源码/producer 的复用加上实际 native registration，是该组件的声明边界。所附“改合法 sampler 值并重 hash”的反例现被原 native evidence 拒绝；原红日志保留，不把过去仅 hash 自洽当成有效批准。

## G4 subjects、native binding 与末快照

`classic_selection` 必须来自 G4 subject 并在 qualification_evidence 中；要求 native int 18 completed slots / 6 roots，classic_best 仅 B1/B2。model registry、dataset/review、namespace、source 均逐 Artifact 匹配已获准 subjects。此处未重新计算历史 18-slot 的 equal-level selector；未来真实 G4 必须冻结原已审 selection，不能用测试中的 synthetic selection 替换。B0 仍是完整正式条件，不进入 B1/B2 selector 候选。

reader 末尾取得新的 sole native owner snapshot，在同一 snapshot 上 fresh `read_dataset_proof`，再验证固定 phase execution finding 精确绑定 **manifest+order+sampler+selection**，并要求 dataset native application 先于 execution。最后重 hash execution、order、sampler、selection、registry，并核 canonical execution tree 的三文件集合和无 symlink。

所附 main 末快照五种 execution 文件/额外文件反例均本轮独立重跑通过；新增测试又在 main 第 5 / holdout 第 9 个 owner read 点分别注入以下变化，全部拒绝：

- qualified execution source 改字节；
- admitted dataset 的 case bytes 改字节；
- dataset finding 撤销；
- execution finding 撤销；
- dataset/execution native application 顺序反转。

正常 main/holdout 在相同 5/9 次 native-read 对照中接受。没有用旧 snapshot 跳过 fresh DatasetProof；也不宣称这一有界重核是原子文件系统 capability 或任意宿主篡改证明。

## Producer 与未集成范围

独立原生历史对照：current/revoked execution plan 移到保留目录后，producer 在 schedule/sampler 前拒绝，不重建 canonical plan。独立 sampler 文件 IO 失败：保留 order 与 preparation-failure，无成功 manifest；再次调用不产生新的 Fake schedule/sampler 调用。正常 producer 输出在 native binding 前不能成为 execution proof，不自签 independent-review 或 owner finding。

`admit_primary_episode` 是完整静态 plan 中的精确 slot membership gate，不是进度 ledger、互斥执行 lease 或按时间顺序派发器。`FormalEpisode.worker_input` 当前只构造 case-loading schema/refs；condition/repeat/sampler、order_index 的实际 CLI/actor/service/receipt 绑定和顺序执行仍由未来 runner 接线并审查。本报告不以该方法证明实际 episode 已执行或配置已传入。

本 schema 仅 primary 八条件；clean/ablation/delay 不在本轮计划里。未来 runner/service/receipt、worker effect boundary、真实 registry/model/selection/正式 dataset 完整闭包仍须单独验收。DatasetProof v2、writer W1 修复及 shared-law 沿用其既有有界报告，相关 source/hash 已绑定，未重新审查整个依赖图。

实际 production recording 字节未改，候选未安装，R/datasets 开始和结束均不存在。临时 `.paper` 仅测试 setup 的 synthetic owner；未运行或修改真实 owner，未产生真实 formal seed/order/数据、模型、物理或方法成绩。结论不授 G4/main/novelty，不替代后续实际 fresh gate/native approval。完成后停止 R 写入。
