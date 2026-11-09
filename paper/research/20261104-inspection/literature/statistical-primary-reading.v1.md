# 统计一手来源定向阅读 v1

本记录是研究依据与公式核对，不是论文正文、正式数据分析或新的统计协议。冻结的 `eval/inspection_statistics.py` 与已接受设计不变；本轮没有用 pilot 成绩改变样本量、选择方法或检验。

Hoeffding（1963），*Probability Inequalities for Sums of Bounded Random Variables*，JASA 58(301):13–30，DOI `10.1080/01621459.1963.10500830`。已取得 RPI 课程站镜像的原始期刊扫描与 Crossref 元数据，分别为 `primary/hoeffding-1963-original-scan.pdf`（SHA256 `3021bcc097ef23a99a84d0eef8dcce2fd835c71df79a105727d3e03f8f0b8930`）及 `primary/hoeffding-1963-crossref.json`（`6a0e700d79e8e120de6d470dc505721c6bacef41035bc4d97b4566e285e230d9`）。原始出版年为 1963；元数据中的 2012 是 online 日期。实际用 Poppler 渲染并通过 view_image 阅读印刷页14、15、16，重点是独立性假设、Theorem 1 的范围变换及 Theorem 2 / Eq.(2.6) 的不同有界区间；没有宣称读完19页扫描。印刷页16后面的两独立样本组推论不用于本研究的配对差。

以下是根据 Theorem 2 对当前设计的自行推导。每个独立 root 先把同一方法的预声明 repeats 求均值，再形成配对差 R_i∈[-1,1]。设预先固定的非负归一化权重为 w_i，估计量 d=Σw_iR_i，W=Σw_i²，目标为 θ=E[d]。对变量 w_iR_i 使用区间宽度2w_i，再合并上下两尾，得到 P(|d−θ|≥t)≤2exp(−t²/(2W))。因此零均值原假设的保守双侧 p 上界是 min(1,2exp(−d²/(2W)))；pointwise95 半宽为 √(2W ln40)，六项同时95的 Bonferroni 半宽为 √(2W ln240)，区间裁到[-1,1]。这是当前实现的公式，不是从两组独立样本推论套来的公式。

ID 的三等级各16 roots、OOD 的两等级各12 roots，先等等级宏平均，故当前两个域恰好分别 W=1/48 和1/24。自行算得 pointwise95 半宽约0.39205、0.55444，同时六项半宽约0.47787、0.67581。这些宽区间说明既定设计对较小效应的信息有限；不能把不拒绝解释为等效，也不据此追加或筛选 roots。

上述概率保证以独立 root 随机变量、预先固定权重与方法、完整合法数据和无选择性停采为条件。root 内方法/repeats不需要相互独立；不同对比的 p 值之间也不需要独立。生成独立种子或通过时钟门，均不能单独证明不同 root 的全部输出独立；共享主机状态、顺序效应与未重置状态仍是应报告的假设/限制。不得将本式描述成允许任意跨root相关的无条件保证。bootstrap仍只作预注册的 nominal/descriptive 诊断，既有 rare-loss 反例不被删除。

Holm（1979），*A Simple Sequentially Rejective Multiple Test Procedure*，Scandinavian Journal of Statistics 6(2):65–70，作者 Sture Holm，稳定身份 `https://www.jstor.org/stable/4615733`。出版社周年目录和 JSTOR issue 目录核过身份；原文定向阅读来自 Academia 上的期刊原文转写镜像 `https://www.academia.edu/77036515/A_Simple_Sequentially_Rejective_Multiple_Test_Procedure`，web 返回的原始观察保存在 `searches/holm-original-mirror-reading.20261106.v1.txt`。实际阅读范围为印刷页66–67的 §2、Scheme 1、Theorem 1 与 Boole inequality 证明；镜像的 OCR/公式排版有损，未取得本地原始PDF，不能声称完成原PDF视觉审阅或全文阅读。读取的是原论文内容，不借镜像平台的二手解释作依据。

其逐步检验把升序 p_(j) 与 α/(m−j+1) 比较，遇到首个不通过即停止；六项 family 使用 m=6。当前实现以升序累计最大值计算 adjusted p：min(1,max_{k≤j}[(m−k+1)p_(k)])。强 FWER 控制需要各真原假设下边际 p 有效，Boole inequality 不要求六个 p 独立；它不能修复无效的单项 p、未登记新增 family 或结果驱动选样。当前同时区间来自 Bonferroni union bound，不是 Holm 检验的反演；holdout仍只作已登记的一次描述性检查，不增第七个确认检验。

访问限制完整保留：provided source-fetch v1 已成功保存 Hoeffding 两项，随后第三项 Crossref/Holm 请求404而停止；v2 的首个 Wiley 请求403，未据此称后续 JSTOR文件已获取。相关 input packets 和 `/tmp/inspection-literature-statistics-source-fetch.v{1,2}.stdout` 仍在；网页可读目录不等同于这些 source-fetch 请求成功。此记录尚未独立审查，也未自动进入已核验14条引用或接受 literature stage。
