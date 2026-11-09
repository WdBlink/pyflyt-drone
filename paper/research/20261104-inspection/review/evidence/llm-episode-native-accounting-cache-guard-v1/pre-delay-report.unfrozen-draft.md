# LLM episode cache guard 窄复验 v1

日期：2026-11-06。**固定 37a40 小版在本次定向反例中关闭了 506f16 复验报告的 K1/K2；正常 Y=0/cache 路径保留。** 还核对了删除 intent/owner log 并同步缩减 payload map 的变体，均被拒。这是缓存 guard 的小范围判定，不回改原报告，不认证 G4、真实模型/物理或新增 injected-delay 功能。

## 固定版本与证据

只写本报告及独占证据目录 `E/`：

`/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/review/evidence/llm-episode-native-accounting-cache-guard-v1/`

实际执行 E/source 的源快照：

| 模块 | SHA-256 |
|---|---|
| `run_inspection_llm_episode.py` | `37a40dcc989059a4a6abb43a6fc7b741153a7291fc075225bd06046a78928364` |
| `inspection_llm_episode_receipts.py` | `a0e4193b78769e3b8e8fd7383bc55de80d1ee00151539ab3337c1301a25adff9` |
| `inspection_process_cleanup.py` | `361ba11b8ea088627e4c0bd832168bb26821297e5d8f54bf6d43c2ba327a9c70` |

共同依赖和 prior synthetic fixture 从已冻结的 repair/source 复制，仅替换 supervisor 为上述小版；fixture tests SHA 仍为 `907ba637358ba02483e784619b5ea98988001e6422d1af4df72af197505cde2c`。未混入后续注入延迟 validator。source-binding SHA `e3cf6833b80c368b35377b6f29541970d11ea5f6d57d8da119a3c258bcfa9558`。

`probe-results.json` SHA：`19ee62ac67197f48bdd9877d3e57b2a2346ed22ded34067b276c6c52221b359d`。
`evidence-index.json` SHA：`7e0d1319507bf5eaa080eff52323b24f02850da540f9838f00650c373afd8a57`，绑定 151 个源/脚本/日志/fixture 文件。

原报告 SHA `90a241…` 和 506f16 修复复验报告 SHA `7db3b1291a82589e9da0dfd2623030cfee1360646a32c2372dc6b846e23e3c8a` 已重核未变。

## 实际反例与正例

全部是 CPU 真实 subprocess 产生的**合成**模型/flight/owner 记录；假 evaluator 只输出 Y=0，不是实际物理或 MLX。每场只有 1 个 synthetic worker dispatch，始终只有原 reserve/settle 两个假 owner effects，后续 cache/拒绝不产生新 flight/生成或 owner 操作。

| 场景 | 固定小版实际结果 |
|---|---|
| 正常首次 Y=0 后请求 cache（传 always-clock，只有误重跑才会生效） | 返回 Y=0、cache_reused=true，counter仍1 |
| K1：删除 archive，同时从 terminal settlement_file_hashes 删除 zip 键 | 拒为 `settled archive/usage/packet changed or missing` |
| 删除 intent.json，并同步从 terminal payload_hashes 删除其键 | 拒为 `LLM attempt dispatch/return proof is missing or changed` |
| 删除 reserve-owner.json，并同步缩减 payload_hashes | 同上拒绝 |
| K2：初次因真实 fast 独立 PG orphan 产生 unknown，再只改 terminal status/Y 为 completed/0 | 仍拒为 dispatch/return proof missing/changed；没有返回 cached Y |

K2 使用了同类实际快速 orphan 时序，没有 mock monitor：首次 process-tree.status=unknown，terminal=unknown_process_cleanup/Y=None；子代 PID **90418** 的 `ps` 为 `90418 1 Ss Fri Nov 6 06:22:50 2026`。terminal 两字段改动后，独立读取的 tree proof 仍 unknown，因此本版拒绝升级。未知归属子代没有被 module 擅杀；审查脚本最后只对自己创建的 PG 发 SIGKILL，并确认 ps 为空。记录、child、终点改动后的 terminal、原未知 process-tree、资源文件均留在 E/probes。

## 实现核对与判定范围

小版从 run/reservation 独立导出 archive、usage receipt、settle packet 三路径，要求 terminal settlement map **键集合精确相等**，再检查 run containment 和 SHA。未知/不可重试 status 先阻断；可复用路径还必须读取不可缺的 intent/start/process-return/process-tree，比较 plan/reservation/ordinal、root PID、returncode、stdout hash、tree closed 及两个 owner log 存在，之后才比较完整 payload/model map 和重跑 admission。

这些检查分别覆盖本次“缩减自报集合”和“只改 leaf terminal 状态”的漏洞，不能写成对任意所有证据一致重写、恶意 host 或外部 terminal 身份的通用密码学保证。缓存仍依赖 caller 的完整预冻结 plan/source/evaluator 绑定及相应环境假设；本次没有实测 live native owner ledger，也未重新执行整套真实 flight、模型、费用或 scorer。

本次不重跑原 18 项 episode suite；上版已实测该 suite，当前只针对 cache 改动运行上述五场。真实执行命令（工作目录 E/source，仓库 `.venv/bin/python` 3.11.15）：

```text
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B ../probe_cache_guard.py
```

日志为 probe.log，脚本使用 exist_ok=False 保留单次证据。未更改 production/owner/live 数据，未创建 `.paper`；没有 GPU/MLX/physics/付费调用。完成该组后按主 Agent 要求留出完整 30s 不创建反例子进程的窗口；之后仅做报告与哈希 IO，未与其 positive 组继续并发 orphan 测试。

结论：**K1/K2 在固定 37a40/a0e419/361ba 组合上的定向关闭通过；不扩展为新增 injected-delay 接线或 G4 认证。** 后续 validator/runner 的新 SHA 与字段绑定必须另作窄复验，不能用本报告支持该新增语义。

Attention / reviewed by inherited-parent model：同模型家族独立上下文的有界合成 CPU/subprocess 审查，非跨模型盲审或人类专家认证；实际模型、物理、live native owner 和新增延迟不在覆盖范围。
