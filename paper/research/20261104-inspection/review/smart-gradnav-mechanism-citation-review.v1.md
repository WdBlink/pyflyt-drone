# SMART / GRaD-Nav++ 机制 citation 独立追加审查 v1

结论：新增两条记录的标题、作者顺序、arXiv 固定版本及首版/更新日期与 primary metadata 和官方页面一致；既有两份 PDF 与对应官方版本 PDF 的 SHA256 逐字节一致。指定方法段支持当前限定的机制主张，未发现身份错误或应阻断这两条 citation records 的 unsupported extrapolation。两篇均有相关先行机制，不能以此授予本研究 novelty、G4、main 或文献覆盖完整性批准。

审查日期：2026-11-06（Asia/Shanghai）。审查者与主 Agent 分离上下文，但属同模型家族，不声称跨模型或人类专家审查。本次仅追加 SMART/GRaD-Nav++；前七条未重新进行内容审查。共享 source、records、owner、controller 及既有报告均未修改。无物理、模型、paid API 或本研究实际数据生成。

## 身份与版本

| 来源 | 独立核对结果 |
|---|---|
| [SMART, arXiv:2503.04798v3](https://arxiv.org/abs/2503.04798v3) | 完整题名为 *Advancing MAPF Toward the Real World: A Scalable Multi-Agent Realistic Testbed (SMART)*；10 名作者与记录逐项同序。首版 2025-03-03，本次 v3 更新于 2026-04-19。记录 `year=2025` 表示 arXiv 首版年份，与 v3 的 2026 年更新时间不冲突。 |
| [GRaD-Nav++, arXiv:2506.14009v2](https://arxiv.org/abs/2506.14009v2) | 完整题名为 *GRaD-Nav++: Vision-Language Model Enabled Visual Drone Navigation with Gaussian Radiance Fields and Differentiable Dynamics*；7 名作者与记录逐项同序。首版 2025-06-16，本次 v2 更新于 2026-05-15；`year=2025` 同样按 arXiv 首版处理。 |

官方固定版本 PDF 的在线字节哈希分别为：

- [SMART v3 PDF](https://arxiv.org/pdf/2503.04798v3)：9,784,894 bytes，SHA256 `aecbfbbadec1541634f03f5a95b8515d97ae0e524413bedc31102b8cda12ae2b`，与 `venue/samples/S04-smart.pdf` 相同。
- [GRaD-Nav++ v2 PDF](https://arxiv.org/pdf/2506.14009v2)：5,811,491 bytes，SHA256 `46161e3ef6f41b91cccfa89093db57a4780e6fe7bbb967f0497cfa5f6ddaa2ba`，与 `venue/samples/S08-gradnav.pdf` 相同。

独立重跑 `/opt/homebrew/bin/pdftotext -layout` 到临时目录，两个产物均与已登记 text 字节完全相同，SHA256 分别为 `58c4a0ea051a7db61d8f3bfbddad43aef9cec4fc77ce2037f0c68471e95869fa`、`89b112e5c05c7c0117b8ee395fb55b9ccc5ae9c5869ac3ed0dfaf339103eb67e`。两份 PDF 首页也带有相应 arXiv 版本和更新日期。

这里只引用固定 arXiv 版本。arXiv metadata 中的 related DOI（SMART `10.1109/LRA.2026.3688062`、GRaD-Nav++ `10.1109/LRA.2025.3643290`）只作旁证，不据此替换为已独立核验的正式期刊版本。SMART 的 2021 模板页眉、GRaD-Nav++ 的 accepted November 2025 页眉均不能替代 arXiv 版本历史或正式出版核验。

## 机制主张与阅读范围

SMART：定向读取任务定义及 III-B/C。III-B 描述 ADG 的 staged/enqueued/finished 状态、依赖约束和经 RPC 发送动作；只有 executor 确认 associated action 完成才标 finished。III-C 描述每台机器人自己的 RPC client、FIFO queue、controller 以及完成回执，并明确独立操作与并行执行。因此当前“ADG execution monitor + RPC completion confirmations + independently operating executors；任务为 MAPF 路径执行”的有限 claim 受支持。它是监测执行机制的先行工作；此处 physical execution 包括物理引擎仿真执行，不能改称本次已核验真实硬件部署，也不能误叫 LLM 多机任务论文。[SMART 固定版本](https://arxiv.org/pdf/2503.04798v3)

GRaD-Nav++：定向读取 III-B1 与 III-D。III-B1 明确说 CLIP 与 policy 异步、每 10 步更新特征；III-D 明确把 onboard inference profile 归于实际飞行采集，并给出机载设备与 profile 的描述。因此当前“异步 CLIP/learned policy + 文中呈现飞行中采集的 onboard profile”的 claim 受支持。这是对论文报告内容的核验，不是审查者重新测量硬件；不能由此进一步保证 policy 在任何 CLIP 调用期间绝不等待、任意 hard real-time 期限，或推导本研究多机工具/证据交付对比。III-D 同时报告周期性延迟峰，故不能把“异步”简写成“没有推理开销”。[GRaD-Nav++ 固定版本](https://arxiv.org/pdf/2506.14009v2)

确切阅读定位（1-based LF 行号）与哈希见 `reading-log.json`。未做新 visual figure audit、implementation reproduction、规模/成功率数字复核，亦未验证外部研究日志和飞行原始数据。记录中更广的 IV-C task switch 阅读声明属于原记录作者；本次不把它算作重新审查过的独立机制结论。

## 跨记录一致性与 bibliography repair

机械 `citation-check.mechanism-neighbors.v1.json` 的 `pass` 只负责 record integrity。本次另外逐项检查新增 plan、单独 verification、合并 verification 与 bibliography native packet 的身份字段及 claim 一致性；结果均一致。

新 plan 与合并 verification 的前七条内容和旧版本逐对象相等，且新版 bibliography 的前七条是旧 v5 的精确字节前缀。此为不改变继承内容的检查，不是重审前七项科学主张。

独立调用原 `lib.production.bibliography` 纯输出函数，在临时目录重现 native v1 bibliography，字节完全一致。v1→v2 恰好新增 9 条 versioned eprint，无其他改变；去掉 9 行逐字节恢复 native v1，且九个 identifier 与 plan 同序一致。native v1 SHA256 `b9738b645879653669fdfe6030111b465222f6b4ecbb62b670eebd6fa5b4b5bb`，修补 v2 SHA256 `2a08bcf123aa79bcbea0cc7f846a13d1fe2f78ec1fc8026f48efea646b10e4f0`。没有调用 stage/controller 或修改 formatter。

`mechanism-boundaries.v2.md` 仍说后两项 citation 待补全，是写入时历史状态；当前新增 verification/plan 已具备上述可核验来源，不把该历史措辞误判为版本身份失败，也未覆盖原文。新 catalog 保留 retrospective 及 adoption timestamp 解释，不追认成问题形成前的灵感来源。

## 证据与边界

独立证据目录：`review/evidence/smart-gradnav-mechanism-citation-v1/`。其中 `audit.py` 为实际执行脚本（纯文件、临时 bibliography/text 产物及两次官方 PDF 读取），`identity-and-conversion-checks.json`、`official-pdf-byte-checks.json`、`bibliography-and-inheritance-checks.json`、`cross-record-checks.json` 分别保存身份/转换、官方字节、修补/继承及跨记录断言；`input-bindings.json` 保存全部输入 SHA256，`freeze.json` 记录最终产物与未变化输入核验。

本报告仅认可这两条固定版本 citation records 的身份与列明机制支持。不是正式期刊版本升级、全文复现、完整近邻覆盖、统计审查、实验实际验证或研究放行。
