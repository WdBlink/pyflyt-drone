# LLM episode cache / injected-delay guard 独立复验 v1

日期：2026-11-06。**固定 37a40 supervisor 的 K1/K2 缓存反例已关闭；最终 624504 validator 的延迟字段检查大部分通过，但新增 D1：public source clock 可晚于 worker 请求前采样，甚至负的 sim age 仍被 admission 接受。** 因此本版不能无条件宣布延迟元数据完整准入通过。全部是合成 CPU/subprocess 或独立计时器测试，未运行 MLX、GPU、真实 physics/LLM、paid API 或 live owner mutation，不授予 G4。

本报告保留原版与 506f16 修复复验报告，不回填其结论。此前同目录 a0e419 预检记录也保留；本文件在最终冻结前按用户补充的最终 validator/runner 范围重写。该预检的未冻结草稿存为 E/pre-delay-report.unfrozen-draft.md，不能用它替代本报告。

## 固定最终版本与证据

证据根 E/：

`/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/review/evidence/llm-episode-native-accounting-cache-guard-v1/`

最终测试使用 E/source-final/，共 30 个逐字节快照文件；早期 cache-only 记录使用 E/source/（a0e419），二者没有覆盖或混写。

| 最终对象 | SHA-256 |
|---|---|
| validator `inspection_llm_episode_receipts.py` | `62450413d92897e359de529cf372e27a7a4d48b78848573d8794dcf57ac6cf83` |
| supervisor `run_inspection_llm_episode.py` | `37a40dcc989059a4a6abb43a6fc7b741153a7291fc075225bd06046a78928364` |
| cleanup `inspection_process_cleanup.py` | `361ba11b8ea088627e4c0bd832168bb26821297e5d8f54bf6d43c2ba327a9c70` |
| actual producer source `run_inspection_llm_development.py`（仅静态接线与 helper 测试） | `c3bed0403ca4a8e7f5d79c0db3df5da0aeb27e46e37423e94474447fbd4da552` |
| final episode test source | `f8dbd74ed9ef0297c42ac9098932699dad835e3d224a8141faebac790e3baf35` |

`source-final-binding.json` SHA：`d3532193df11037e1e39871c76c9b3b81635c521006ed1828ed0d9e42ba180eb`。

| 实测证据 | SHA-256 |
|---|---|
| `pytest-final.log` | `e612c18401536b8a12d7c8358680fd674c5463038f87a4abae468f11ece894dc` |
| `probe-results-final.json`：最终组合 cache 反例 | `5ef9f820dc20e90185361f749bb03f30f2e5058627234bd8cbe0e7ee95ada1cd` |
| `delay-metadata-results.json`：24 个正负元数据例 | `8c6553b79ee35da48c97b78d31aac0c7941a25b732c89f8afe8bcbb33af6a2fb` |
| `evidence-index.final.json`：1200 个源/脚本/日志/fixture 文件 | `5ecf36c9f83756968e5ddf9763640992f189e63e2ab0131cd67c6118abfa4594` |

原报告 SHA `90a2415883e924a1a070e4e9326ced2b7247ce89c0b148ddf9b8da11a7aafd16` 与 506f16 报告 SHA `7db3b1291a82589e9da0dfd2623030cfee1360646a32c2372dc6b846e23e3c8a` 已重核未变。

## 最终组合的 cache 复验

在 624504+37a40+361ba 上实际重放五个场景，每场仅 1 个 synthetic worker dispatch、原 reserve/settle 两个假 effects；再次 cache 或拒绝没有重飞/生成/新 owner 操作。独立早期 37a40+a0e419 预检也得到相同结果，最终结论以下表为准。

| 场景 | 最终小版结果 |
|---|---|
| 正常首次 Y=0 后请求 cache，传 always-clock 以检测误重跑 | Y=0、cache_reused=true，counter仍1 |
| K1：删 archive 并删 terminal settlement map 的 zip 键 | 拒为 `settled archive/usage/packet changed or missing` |
| 删 intent.json，同时从 payload_hashes 删其键 | 拒为 `LLM attempt dispatch/return proof is missing or changed` |
| 删 reserve-owner.json，同时缩减 payload_hashes | 同上拒绝 |
| K2：初次因真实 CPU fast orphan 为 unknown，再仅改 terminal status/Y 为 completed/0 | 仍因 unknown process-tree proof 拒绝，不返回 cached Y |

最终 K2 的自建孤儿 PID 204 在探针期间确为 PPID1/Ss 活进程；module 未擅杀未登记的子代，而是 unknown 阻断。审查脚本最后只清理自己创建的 PG，ps 为空。早期预检 PID90418 也已清理。全部 fixture、原 process-tree、终点修改记录、源和资源文件留存；没有把 unknown 当 known/held 的同义词。

静态对照确认：37a40 从 run/reservation 独立导出三个固定 settlement 路径，键集合要求精确相等，再查 containment/SHA；未知/不可重试 status 先阻断。可缓存路径独立读取不可缺的 intent/start/process-return/process-tree，核 plan/reservation/ordinal、root PID、returncode、stdout hash、closed 状态及 owner log 存在，再比较完整 payload/model maps、重做 admission。这里关闭的是原 K1/K2，不声称对任意所有证据一致重写、恶意 host、任意 fork/setsid 或断电恢复作密码学/内核级保证。

## 最终 29 项 suite 的实际执行

仓库 `.venv/bin/python` 3.11.15，E/source-final/ 下一次串行执行：

```text
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B -m pytest tests/test_inspection_llm_episode_receipts.py tests/test_inspection_process_cleanup.py tests/test_inspection_injected_delay.py -q -p no:cacheprovider --basetemp=../pytest-final-tmp
```

结果 **29 passed in 14.66s**。提供的 owner 成功日志为 29 passed in14.75s，单独保留，未冒充本次执行。既有 owner 失败日志未抹除；本次没有为 PASS 修改 orphan guard 或自动反复重跑 suite。

suite 中的注入 helper 正例实际 sleep 2s，同时一个独立 CPU 计时器进程继续更新 shared progress，验证 helper 不暂停该计时器且不改 response 原 service/parent latency；其余 skip 分支覆盖 not_started/runtime_failure/final/t600 不等待。它不是 PyFlyt/MLX 物理并发证明。5/10/20s 档位在本次是合成元数据与静态 generic helper 检查，没有假称实际等待这些时长的真实 flight。

## 0/2/5/10/20 单因素与延迟年龄正负例

实际运行 E/probe_delay_metadata.py。6 个 cold synthetic episode（五档位与 not_started）写原 request/response/token/flight/actor，假 owner 效果如实记录；其余为同一保存 fixture 的 counterfactual 只读 admission 检查。后者刻意绕开 cache 的旧文件哈希，以测试语义；改动文件保存到 counterfactuals/，baseline 字节每例恢复。总计24例：8个有效例接受、14个无效例拒绝、D1两个无效例被接受。

五档均为 B4-M1、ablation=None。原 service wall=1、parent compute wall=59：

| requested / run / actual synthetic injection | reply sim | after sim | dispatch sim age | dispatch wall age | 结果 |
|---:|---:|---:|---:|---:|---|
| 0 | 1 | 1 | 1 | 59 | 接受 |
| 2 | 1 | 3 | 3 | 61 | 接受 |
| 5 | 1 | 6 | 6 | 64 | 接受 |
| 10 | 1 | 11 | 11 | 69 | 接受 |
| 20 | 1 | 21 | 21 | 79 | 接受 |

这些数字是明确 synthetic 的记录，不是实测模型/物理延迟。它们证明本 validator 的分类不把注入部分加进 service/parent compute 的 soft60。额外 parent compute=60、injection=20、wall age=80 仍允许 decoded；parent compute=61 而原 response 未超时会被拒为 late/noncompleted，符合所宣告分离。

已实测拒绝：

- plan省略delay（默认0）但run=20；plan20但run2。
- 非零 delay 配 B3-M1、B4-M2，或混合 periodic ablation；bool/float/非枚举3。
- response requested 值不匹配、actual injection19.5却声明20、注入期间无模拟推进。
- wall age遗漏注入，或 sim age不等于after−source。

已实测接受：not_started在20s arm仍actual injection0/model usage0；final/t600合成记录actual injection0且无入队命令。真实producer c3bed0 的静态接线也对照到：CLI仅允许0/2/5/10/20，非零仅B4-M1且无ablation；真实reply后协调器sleep，服务/原parent latency保持独立，after sample记录注入后progress；final/600不入队，dispatch sim/wall age分别按差值/parent+consume wall记录。没有启动该 producer main、模型service或物理worker。

## D1 [P2] source clock 未与请求前 worker witness 约束，允许年龄遗漏/负值

624504 验证 public source_sim_time 在[0,590)，验证before≤reply≤after及注入推进，也验证 age_sim==after−source；但没有要求 **source≤worker_before_request[0]** 或 age_sim为非负数。仅自洽地重算消息hash和age差值不能证明公共输入已经发生。

在20s fixture的counterfactual中，保留合成基线中的请求前 witness=.1、reply=1、after=21、actual injection20、wall age79，只同步修改：public source clock及public_state.sim_time、消息canonical SHA、原 call/request.messages、decoded audit.request、dispatch sim age。原 model usage/profile、token journals、source maps、service/parent latency不变。

| public source | worker before | reply | after | sim age | readonly admission |
|---:|---:|---:|---:|---:|---|
| 5 | .1 | 1 | 21 | 16 | 接受Y0，年龄短于已经记录的20s注入 |
| 580 | .1 | 1 | 21 | -559 | 接受Y0，负年龄 |

证据：delay-metadata-results.json 的 future-public-clock-5/580，以及 counterfactuals/ 同名目录。不是实际runner发生未来消息的指控，也不是新模型/flight；是该只读checker未拒绝内部时间关系不可能的合成输入。若走原cache且仍信任旧terminal SHA，篡改可被更早hash gate拦截；首次admission与独立语义审计仍不应因此依赖这个偶然条件。

最小修复建议：在request/response join处要求 `0 <= source_sim_time <= before[0] <= reply[0] <= after[0] <= 600`，并对sim age作严格finite numeric/nonnegative检查后再核差值；不得把录入时间关系的缺失解释为一般物理/trace认证。修复后至少重放上述两例、一个正常有stale public frame例与各档正例，另版绑定新SHA，保留本报告不回填。

## 结论与未执行事项

K1/K2 缓存 guard 在最终624504+37a40+361ba组合上的定向关闭通过；0/2/5/10/20的字段绑定、单因素限制、分开soft60及大部分wait/age规则得到合成支持。**D1仍阻止把本版宣称为完整因果时间元数据准入通过。** 后续补丁未被本报告预批。

没有MLX/physics/GPU/live owner或.paper操作（证据目录无真实paper state），没有新增实际样本或正式推断。原meter/held含义不改变：已known settle后丢terminal只阻断episode，不伪称usage仍held；unknown meter与未知进程归属不同。出生身份追踪/未登记Python孤儿的保守阻断范围沿用前版，不是kernel subreaper或host-loss恢复保证。

不认证G4、真实模型/权重、正式八配置/main/holdout、真实generator/trace、费用或所有外层driver接线。资本/能源/隐藏work没有独立计量；fake evaluator只输出Y0，不能替代真实frozen score对完整world的重算。所有自建反例子进程均已结束或由审查脚本清理；没有给不明归属进程发信号。按主Agent要求曾留出完整30s不创建反例的窗口，最后组合测试串行执行，未改guard以消除falsepositive。

Attention / reviewed by inherited-parent model：同模型家族独立上下文的有界代码/合成CPU subprocess审查，非跨模型盲审或人类专家认证；真实模型与物理并发未在本次执行。
