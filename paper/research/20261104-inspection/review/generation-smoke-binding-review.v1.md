# Generation smoke：窄绑定独立复核 v1

结论：本次相对f5版新增的 reservation purpose 与唯一 durable request 绑定，在14个隔离fake分支中符合预期；未发现本次小差异的实证阻断。三份准备好的resource request与当前只读来源匹配。本报告**不是实际smoke、模型资格、预约批准或G4放行**。

审查为同模型家族独立上下文工程复核。没有真实MLX import/load/forward、真实tokenizer、PyFlyt物理、owner/live修改、预算预留/结算/派发。fake client只有纯CPU函数；其合成token数不是实际模型输出。checkpoint要求R静止期间，全部新文件仅写 `/tmp/inspection-generation-smoke-binding-review-v1/`，收到恢复通知后才复制新报告/证据到R；旧来源未修改。

## 精确差异与来源

工作树 `/Users/wdblink/.codex/worktrees/538e/pyflyt-drone`，HEAD `270c6f0946b3be59527b2b77982f75a70da7ef77`。本次实际probe SHA-256：

```text
53dff69f11ff8065ffabf28ed301940b0e91ecc9e665b085b0cfb1a609ca8190
```

对已审 `f5f6e2193d65c599857274f6ac6c9650e590218e44f6883e2f1e49fde54f0c1f` 的完整diff仅两段：probe49–56在创建client前核对purpose的kind/phase/model alias/registry SHA/model receipt SHA/owner frozen plan/service SHA及实际request budget；91–100把唯一durable request的SHA、messages与native-int seed join回本probe。完整diff和源码已保存，不重做全G4。

相关未变文件SHA：service `64e8307ece55b1d6928e91607a269dfb082926265f1d98a2bde90a5931eb2e8f`；meter `922372548ceb972e9f95d1c52923cfe7d9dd6ce12dba2feec3b2d44a99167683`；client `40ccd5f5f4100e4ed42a6ac22ecf35f3bc24a9db4e2b57f1826625b78afcb158`；wire `c323f708fd2c2eb54a440758bd162785f3ee2fcebd95b9bca21eaf7ed6ef44d4`。本轮不重复其旧测试资格。

## 实际fake结果

| 分支 | 调度/结果与计费 |
|---|---|
| 正常purpose与request | fake init=1、submit=1；exit0，合成1call/4tokens，且唯一request/response与meter绑定一致。仅证明代码路径，不是一次真实生成。 |
| wrong kind、phase、alias、registry SHA、model SHA、frozen-plan值、service SHA | 各在FakeClient创建前ValueError；init=submit=0、无operation，不占用一个已派发请求后再偷换身份。 |
| wrong actual output config，或max_calls=1.0而预期为native-int1 | 各在FakeClient创建前ValueError；init=submit=0。对actual_request_budget排序JSON表示的比较能区分这些值；未声称做过无限格式/恶意宿主认证。 |
| durable messages与本probe不同 | exit1/operational_failure_or_unknown；保留重算1call/4tokens，无第二次init/submit。 |
| durable seed变为2，或bool True冒充uint32 seed1 | 同上；native-int seed检查拒绝bool等值混淆。 |
| meter正确重算后request文件字节变化 | SHA join失败，exit1；保留已重算用量，无自动重试。该变体仅在隔离fixture的validator返回后改变request，不改真实文件。 |

共14分支：1正常、9个purpose/config调度前拒绝、4个request/seed/SHA调度后拒绝。早期拒绝发生在probe.json intent保存前，harness以明确自定义标签记录raised和无receipt；这不是伪造的生产终态。request/seed是**派发后**与durable文件核对，不能称未生成前已知其错误；它们仍被计费且不能免费重试。拒绝只阻止错误的成功判定，不会追溯抹去可能已经发生的真实资源使用。

fixture沿上一复审fake main，加入新的purpose字段；使用真实纯CPU admitted_limits和完整小blob核验，所有owner/blob/operation均在 `/tmp`。没有安装或调用真实MLX。没有重新运行68项全套，因为仅probe小差异变化，当前14条检查覆盖其新增条件，原精确版本的结论按原范围保留。

## 准备好的request输入只读核对

实际registry SHA为 `6d8547cb0597113d5411a771e1c75f807e43d769ecdfe4d92d32d41633e0fda6`；当前只读owner frozen-plan值为 `f17d41111458e5f2b2529e94499c9f372a826d959c4cc5a28c8f6af2b6deff01`。三份请求的usage均为1call/16384tokens，purpose实际配置均为1call/16384total/8192input/64output/60seconds、temp.2、top_p.95、thinkingFalse；其余身份/SHA字段与对应alias、当前service、registry、小receipt文件SHA及owner frozen值一致。

| 准备文件 | 实际 SHA-256 |
|---|---|
| `resource-requests/model-smoke-M1.v1.json` | `5af88ce870ea11960f09842171fd833b308bbbe5efbd5f2a513af5a0f67eb0ee` |
| `resource-requests/model-smoke-M2.v1.json` | `bfbc3edffcf182f2e5eddb93b656ac095010bb204ddef3d3ac002a694765d9d4` |
| `resource-requests/model-smoke-M3.v1.json` | `3497a3a50c97ab8a2d7afdb4dcdaf2d8e4e83caba6d12f662d75563f113b624b` |

这里只SHA核对小receipt JSON文件，没有重哈希大权重。准备请求是输入文件，**不是reservation、dispatch、使用或结算证据**；本报告没有从文件存在推导授权已获得。

## 执行与边界

实际从工作树运行：

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B /tmp/inspection-generation-smoke-binding-review-v1/probe.py
```

另以纯CPU inline脚本只读计算三份resource request、registry和小receipt的SHA，并核对当前owner frozen值，结果保存为prepared-inputs-check.json。probe源/JSON/log、完整diff、源码及输入快照保存于 `review/evidence/generation-smoke-binding-v1/`，`/tmp`原件保留。当前变化未引入新的model生成、服务operation或物理进程，未改旧报告和结果。

本次只证明新增身份/配置/request来源检查的已列边界。真实smoke是否有输出、实际计费与时延、完整episode/SDK/G4、GPU隐藏工作/资本/能耗和外部unknown恢复仍须对应独立证据，不能从本轮fake或准备请求推导。本报告不预留、派发、结算、免费重试或代替owner决策。
