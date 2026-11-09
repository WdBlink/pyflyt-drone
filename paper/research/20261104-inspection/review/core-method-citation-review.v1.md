# PyFlyt / ReAct 核心 citation 独立追加审查 v1

结论：新增两条记录的标题、作者列表、固定 arXiv 版本及日期与 primary metadata、官方页面和 PDF 一致；两项限定主张均有指定段落支持。未发现阻断这两条 citation records 的身份错误或 unsupported extrapolation。PyFlyt 只支持软件基础归属，ReAct 只支持方法来源；不能据此认证当前包版本、控制包线、硬件、B3 完整复现或研究成绩。

审查日期：2026-11-06（Asia/Shanghai）。审查者与主 Agent 分离上下文、同模型家族，不声称跨模型或人类专家独立性。仅新增本报告及独立证据；未改既有 source/records、owner、live/controller、论文或前版审查。未运行物理、模型、付费 API 或 formal 数据生成。前九条只查继承内容一致性，未重新进行科学内容审查。

## 两项身份与来源

| 记录 | 独立核对 |
|---|---|
| [PyFlyt, arXiv:2304.01305v1](https://arxiv.org/abs/2304.01305v1) | *PyFlyt -- UAV Simulation Environments for Reinforcement Learning Research*；6 作者与 metadata 同序；首版及 v1 日期均为 2023-04-03。PDF 标题使用长破折号，与 metadata 的 `--` 是题名排印差别。 |
| [ReAct, arXiv:2210.03629v3](https://arxiv.org/abs/2210.03629v3) | *ReAct: Synergizing Reasoning and Acting in Language Models*；7 作者同序；首版 2022-10-06，v3 更新 2023-03-10。记录 `year=2022` 按 arXiv 首版，不能把 v3 更新误写成首次发表。 |

PyFlyt 的第三作者在 arXiv metadata 为 **Mauro Innocente**，PDF 为 **Mauro S. Innocente**；此为同一固定版本中同位置署名的来源变体。记录采用 metadata 字符串，证据保留 PDF 变体；未新增作者、推断中间名或把两个字符串暗称字面完全相等。PDF 另有 IEEE membership 标签，不是新增作者。所有作者完整顺序与 metadata 断言见 `identity-and-conversion-checks.json`。

独立在线读取两个固定版本 PDF，并将字节 SHA256 与既有文件对比：

- [PyFlyt v1 PDF](https://arxiv.org/pdf/2304.01305v1)：854,799 bytes，SHA256 `fb6a9a04c8d96082515274edf2cf735df1680af0f76b2ad3921046381d826b46`，与 `literature/primary/pyflyt-v1.pdf` 相同。
- [ReAct v3 PDF](https://arxiv.org/pdf/2210.03629v3)：633,805 bytes，SHA256 `f285b0971ae4a790e402fb93966bed3adde2cf0a04977d08b2b40d6ab0cace69`，与 `literature/primary/react-v3.pdf` 相同。

独立重跑 `/opt/homebrew/bin/pdftotext -layout` 到临时目录，两份派生 text 均与登记文本字节一致，SHA256 分别为 `15a8222d3119de5c5d59f1450e445780af80dcd8b25efd99bb21bb5595167132`、`0bdbef2564106889c48ecff9f3ce107cdd4dd225064d3dcd4bee0acd56be72d9`。临时产物已删除，没有覆盖来源文件。

## 有限机制支持与适配边界

PyFlyt：定向读摘要、功能与 III-B/D 组件段，原文直接归属 Bullet、可配置模块化组件、fixed-wing 及多个回路/step scheduling。当前 claim 用于软件基础归属，受这些段落支持。原论文不能验证本项目当前 PyFlyt 0.29.0 包线、飞机参数、17m/s 控制范围、合成感知或真实硬件。本次也没有核验已安装包、动力学有效性或 G1–G3 实际结果。[PyFlyt 固定版本](https://arxiv.org/pdf/2304.01305v1)

ReAct：定向读 §2 及相邻方法上下文，原设置将语言 thought/reasoning 加入动作空间，与领域 action 和 observation 交互；描述的主要 frozen PaLM-540B 设置使用人工编写的 few-shot 轨迹，决策任务的 thought 可以稀疏出现。因此记录中“描述的 prompts 包含 few-shot”准确，不等于所有任务必须每一步固定 thought/action 交替。原文所说 thoughts/actions 的异步出现也不能直接变成飞行物理并行运行保证。[ReAct 固定版本](https://arxiv.org/pdf/2210.03629v3)

对 `core-method-adaptation-notes.v1.md` 做了限定静态核对：当前 `eval/inspection_llm_actor.py` SHA256 为 `cd1bf019c637d447946cba30e1e2d8572c4ff90594de4b5b2d90f49cd65ccc7b`。相关代码确有单在途请求、最多32尝试、B3 在相隔至少20个 simulation seconds 后才 due、最多最近两次 exchange 与公共帧、统一 JSON 回复的边界。两次 exchange 还可因字节预算继续缩减，due 也不是恰好每20秒必然 dispatch。该 prompt/历史/资源设置与原文设置不同，称 **bounded ReAct-style 适配** 合适；本次不判断该 baseline 是否足以复现原文效果。未导入或执行 actor，未认证 runner 接线、实际 messages/trace 或模型效果。原论文分数不能作为本项目分数或跨任务直接对比。

## OpenReview、记录整理与出版边界

已读 `core-methods-partial-fetch-and-conversion.v1.json`，并实际核验其指向的已有 fetch log 内容为 `status=refused`、`HTTP Error 403: Forbidden`；证据目录保留该 log 的原样副本。四份成功 arXiv metadata/PDF 的 provenance 与文件 hash 一致。此前 browser challenge 是该记录保存的历史观察，本次未重新重放挑战，也未取得成功的官方 decision artifact。

ReAct v3 自述 camera-ready、PDF 页眉写有会议信息，是固定 arXiv 文件的内容；本轮 citation 仍明确为 arXiv preprint，不用该自述替代独立取得的 conference decision。403 不表示未录用，本报告也不作这种推断。PyFlyt 页眉的 2021 模板年份不替代 2023 arXiv 日期。当前 PyFlyt verification 的 `acceptance` 段混入了 ReAct/OpenReview 说明，是无身份影响的整理瑕疵；本报告未将它解读为 PyFlyt 的额外来源或出版证据。

适配 notes 中“PyFlyt 尚未进入最终 citation packet”保留了写入时历史状态；当前 packet 已包含它，不能用历史措辞否定本次明确身份，也不能据此追认更早的阅读/采用时间。当前仍按 retrospective/adoption timestamp 的既定边界解释。

## 继承与11行输出修补

新增两条的 plan、单独 verification、合并 verification 与 bibliography native packet 的身份/claim 字段一致；机械 pass 只是 record integrity，本次另做上述外部来源核验。

新 plan 与合并 verification 的前九条和 mechanism-neighbors v1 逐对象相等；新修补 bibliography 的前九条也是旧 v2 的精确字节前缀。没有重新认可前九项的全部内容。

独立执行原 `lib.production.bibliography` 纯输出函数，在临时目录重现 native core v1，字节完全一致。v1→v2 恰好新增11条与 plan 同序的 versioned eprint 行；删除这些行逐字节还原 v1，没有其他改动。native v1 SHA256 `5b5f73096ceed66cbfa4977e2ffbbedbe711bd3e1f452074cb260996697bf9cd`；修补 v2 SHA256 `bd05fa60cbdd2c0557828e237aa6c1a31d3ca77d6674e0b5de56358cabc35069`。本次未调用 native stage/controller 或修改 helper。

## 可审计证据与最终范围

独立证据目录为 `review/evidence/core-method-citation-v1/`。实际执行命令：

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B paper/research/20261104-inspection/review/evidence/core-method-citation-v1/audit.py
```

该脚本仅作静态文件检查、临时文本/bibliography 纯输出和两个官方版本 PDF 的只读字节核验。`identity-and-conversion-checks.json`、`official-pdf-byte-checks.json`、`bibliography-and-inheritance-checks.json`、`scope-and-source-variants.json` 保存实际断言、作者变体及读取行范围；`input-bindings.json` 绑定31项输入，`freeze.json` 绑定本次产物并复查输入未变。

本次是两条核心来源的定向阅读和归属审查，不是全文复现、当前软件资格、方法效果、正式出版版本升级、文献完整性、novelty、G4 或 main approval。
