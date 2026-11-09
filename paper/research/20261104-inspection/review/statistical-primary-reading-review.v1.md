# 统计一手来源定向阅读独立审查 v1

结论：新增两篇的身份、已声明阅读范围、有限句子支持及所列公式未发现实质阻断。该结论只针对来源与推导；不认证实际 root 独立性、实现完整性、正式数据、literature 完整性、新颖性、human review、G4 或 main 准入。审查者 `/root/statistics_design_review` 为独立上下文、同模型家族，无实现职责。

Subject：`literature/packet.sixteen.v1.json`，SHA256 `45ddced9f0ea7be022861e9adecec091084305d0e2d4175f27d2d8220d39e3b3`；`literature/statistical-primary-reading.v1.md`，SHA256 `fc824ed58bff64829d5a010fbd487f4a06b902007ac3d94f63235ba266812fe0`。29 个明确输入文件的实际 SHA/大小、检查代码、结果和实际阅读范围保存在同目录 evidence 的 binding；没有全树扫描。

## 实际阅读与身份

已通过 `view_image` 实际查看保存的 `hoeffding-03.png`、`-04.png`、`-05.png`，对应印刷页14、15、16。页14明确使用独立性，页15给出共同区间的变换，页16 Theorem 2 / Eq.(2.6) 允许各变量区间宽度不同。视觉公式与所保存的 directed ASCII transcription 一致；后者是有标识的人工转写/数学归一化，不是原论文全文 OCR。页16紧随的两独立样本均值差推论没有用于这里的配对 root。没有重新渲染扫描，也没有声称视觉读完19页。三图的实际 SHA 分别为 `f9e5d2737a91b56ce51d071fb401fbfe5a313ce02d43c499d1c7e8ac47f6cb5f`、`1dcab1f7ff777c53c943b821712b611e5aea2befbf048f9f69d8bb65ecc03c29`、`c63901fe44e311c3eb538c3e36a7e76ff5d935e4fbaa66cb146925553286498a`，与保存的 render binding 相符。扫描 PDF SHA 为 `3021bcc097ef23a99a84d0eef8dcce2fd835c71df79a105727d3e03f8f0b8930`。

保存的 Crossref JSON 明确给出单作者 `given=Wassily, family=Hoeffding`、原出版/印刷时间1963年3月、JASA 58(301):13–30 及 DOI [10.1080/01621459.1963.10500830](https://doi.org/10.1080/01621459.1963.10500830)。2012-04-10 是 online 日期，不能替换1963首发表年。标题、作者、日期与卷期页支持当前 verification/plan/bib 身份。

Holm 身份依据保存的官方 JSTOR issue 目录与 Wiley 周年目录观察：Sture Holm，1979，*A Simple Sequentially Rejective Multiple Test Procedure*，Scandinavian Journal of Statistics 6(2):65–70，[稳定条目4615733](https://www.jstor.org/stable/4615733)。Wiley 列表使用 Holm, S.，JSTOR issue 给出全名，两者相容。本轮没有核得 DOI，bib 也未填写 DOI；不由 lookup 404 推断该文必然没有 DOI。

实际定向阅读 Holm 原文镜像 OCR 的 §2、印刷页66–67、Scheme I、Theorem 1 和 Boole 证明。OCR 两栏串行交错且公式有损；可读文字与流程支持升序逐步 Bonferroni、首次不通过即停止及对任意真零假设子集的控制。记录中的 adjusted-p 公式是对此过程的现代数学表达，不冒称 OCR 中逐字清晰出现该公式。本轮没有本地原始 Holm PDF 视觉阅读或全文阅读。官方目录身份与镜像承载的原文内容分开核对，不把镜像平台自身当作权威方法解释。

## 配对 root 推导与假设

先在 root 内对各方法的预声明 repeats 求均值，再得到配对差 (R_i\in[-1,1])。固定非负权重满足 \(\sum_iw_i=1\)，令 \(d=\sum_iw_iR_i\)、\(\theta=E[d]\)、\(W=\sum_iw_i^2>0\)。取 Theorem 2 的变量 \(X_i=w_iR_i\)，其均值变量为 \(\bar X=d/n\)，均值为 \(\theta/n\)，阈值为 \(t/n\)，而区间宽度平方和为 \(4W\)。代入 Eq.(2.6) 后 n 消去，合并对 \(R_i\) 和 \(-R_i\) 的上下两尾，得到

\[
P(|d-\theta|\ge t)\le2\exp[-t^2/(2W)].
\]

所以在 \(H_0:\theta=0\) 下，\(p=\min\{1,2\exp[-d^2/(2W)]\}\) 是保守的边际 p 值：对 \(0<u<1\)，事件 \(p\le u\) 对应 \(|d|\ge\sqrt{2W\ln(2/u)}\)，概率至多 u。这个弱均值零假设不要求每个 root 的均值分别为零，也不要求 root 同分布。

pointwise95 半宽为 \(\sqrt{2W\ln40}\)，六项 Bonferroni 同时95半宽为 \(\sqrt{2W\ln240}\)；将 \([d-h,d+h]\) 与 \([-1,1]\) 相交保留覆盖意义。当前等等级宏平均给出 ID 的 \(W=1/48\)、OOD 的 \(W=1/24\)。标准库独立复算如下：

| 域 | pointwise95 半宽 | 六项同时95半宽 |
|---|---:|---:|
| ID | 0.39205013785 | 0.47787022835 |
| OOD | 0.55444262208 | 0.67581055798 |

宽区间意味着对小效应的信息有限；不拒绝不能解释为等效，也不能据结果追加或筛选 roots。这里的同时区间来自 Bonferroni union bound，不是 Holm 的检验反演。

概率保证要求参与某个对比的不同 root 随机变量独立、权重及方法/设计预先固定、完整合法输入和无结果驱动停采。root 内各方法和 repeats 可依赖，classic 确定性或模型 repeats 不等本身不妨碍先求 root 均值后的有界性。独立种子、物理时钟门或独立生成场景不等于已证明共享主机上所有 root 输出独立：运行顺序、共享状态、温度/负载、未重置状态等仍可能形成跨 root 依赖。本轮未证明或测量该独立性，不能把公式写成允许任意跨 root 相关的无条件保证。

对已预声明三对比×两域的六项 family，Holm 按升序 p 与 \(\alpha/(6-j+1)\) 比较，并用 \(\min\{1,\max_{k\le j}[(6-k+1)p_{(k)}]\}\) 表示调整 p。强 FWER 控制需每个真零假设下边际 p 有效；六个对比的 p 彼此可依赖，域间 p 也不需独立。它不修复无效的单项 p，也不补救数据驱动 family/选样。bootstrap 的 nominal/descriptive 身份与既有 rare-loss 反例保持原样，holdout 不额外加入第七个确认检验。

## 引用完整性与检查范围

独立标准库字段比对确认：16 条 bib/plan key 唯一且一致；新两条 verification、aggregate verification、plan 与 bib 身份相符；新 bib 前14条所有字段（包括 entry type）与 `references.core-and-backward.v2.bib` 精确相同。本轮没有重审原14篇的外部真实性或机制。native formatter v1→v2 的唯一字段修补是原14条 versioned `eprint`，删除这14字段可逐字段恢复 v1；新增两篇期刊条目未变。repair 前/后实际 SHA 与记录一致，其中最终 bib SHA 为 `b58d7988c22884a04d0b993e29a811f2586b886c0515c81b11dd5eb76fb00449`。

保存的 source-fetch v1/v2 原始输出分别为404/403拒绝。成功保存的 Hoeffding 源与官方目录网页观察不被概括成所有 source-fetch 均成功；没有为失败后的请求捏造已下载文件。native16 citation-check 尚待下一 barrier，本审查没有调用或宣称其通过，也不以已存 plan 的 verified 字段代替本次独立证据。

23 项微小标准库字段/公式检查于 `2026-11-06T16:13:56.619713Z`–`16:13:56.631851Z` 完成，脚本内部耗时0.0119秒，全部通过；包括不等权下的 Theorem2 缩放抵消、两域半宽反算、零差 p=1、边界夹区间，以及六项 Holm 累计最大值手例。无 pytest、生产 import、原生 CLI、RNG、模型、物理、全树扫描或额外持久子进程。全部检查结束；本报告与 binding 归档后停止写 R。
