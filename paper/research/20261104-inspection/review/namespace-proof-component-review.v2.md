# namespace / phase 组件 v2 修复独立复核

审查者 `/root/control_review`，同模型家族、独立工程上下文。仅复验原两项P2、同一父链的新增顺序检查；未扩展writer、真实dataset/raw/statistics完成认证，不改候选、生产63源或实际owner，不运行模型、物理、科学RNG或正式geometry。所有seed为隔离fixture的literal，非实际main/holdout输入。

**原两项P2关闭；新增父链中仍有一项P2：main namespace被superseded后，holdout仍接受。** 当前组件未安装，没有真实准入或资源动作；结论不授完整G4/formal GO。

## 绑定与执行

冻结binding SHA实核 `5bb32118eccdba44ce19e176ffbbfc2052eb1fee31700f52e66f8e523a571a8e`。namespace源 `74aa46e67e77d591beb2b6fbe61895742ff8d70b9c8050bcd9fe32bb5944cf5c`，phase源 `44abef28366a3104bdc9c653f056c1596e3b7a9cfd23501343eaabbb4434f321`。七项freeze文件匹配，working四份源/测试先核为相同字节，G4Proof仍为未更改的v3 `c9814494…`。

按要求分进程独立复跑：**namespace33 passed/0.69s；phase23 passed/0.01s**。原v1报告/证据保留，不回写为“当时已通过”。

## 原两项修复关闭

1. **最新snapshot的完整G4谓词**：`current_g4_binding`现以`replace(verification, owner_reader=lambda _: owner)`复用原read_g4_proof，输入是刚由原生reader核过的同一snapshot；fresh proof还须与初始G4Proof相等。没有复制一套近似算法或增加原始owner读取。

   独立重放原反例，在第2次owner_reader调用改变临时accepted plan packet并保持一致pair，或改变临时qualified mission source bytes：两者都在第2次owner读取后拒绝，**intent bytes尚未读取，owner_reader总调用仍为2**。具体报错分别为accepted plan/owner要求及artifact changed。原P2-1关闭。

2. **实际outer grid**：phase现在同时核top-level conditions确为原八项的顺序列表，seeds为严格int `[0]`。独立测试少一arm、`[0,1]`、`[False]`、`[0.0]`均拒绝；原P2-2关闭，没有改变任何实验数或方法。

正常main/holdout合成正例仍分别接受72/12 roots。新holdout fixture仅增加了main namespace与canonical main dataset的native-shaped父记录，使用分开的literal seed，不是实际生成过两套数据。

## 新增temporal检查通过，但不代表producer完成

独立逐项交换五对native父/子位置：

`G4 < main namespace < main dataset < primary lock < once opening < holdout namespace`

**五项全部以strictly before拒绝**。canonical `datasets/main.v1/manifest.json`及其native binding也已进入lock父链。这比v1仅验证读取依赖进一步约束了应用索引；本轮未把这种synthetic round-list检查夸大为实际writer/RNG发生时刻证明。

main dataset目前仍只有规范路径、Artifact SHA与native绑定下限，未验证实际case/SDK/raw统计完成；独立review/lock producer今后仍须做真实完整验证。此未完成范围已由contract声明，本轮不报为组件缺陷，也不据合成positive宣称main已完成。

## P2：main namespace祖先撤回未传递到holdout校验

在`inspection_namespace_admission.py`的`read_primary_analysis_lock`中，dataset、lock和opening均调用require_native_binding，包含既有supersession检查；main namespace只在require_precedes中按ID定位索引，没有复核该祖先记录仍有效。

独立反例先建立完整holdout正例，再只追加一条native-shaped finding，`revises=inspection-main-namespace-v1`，control/checkpoint保持一致，G4/source/数据/lock/opening文件均不改。结果仍返回12-root holdout NamespaceProof。

同一脚本仅将revise目标分别换成main dataset、primary lock或opening时，三者全部正确报`qualification finding has been superseded`。因此不是要求raw真实性校验或广泛撤销推理，而是**同一已采用native父链里，一个已失效父节点只查位置而漏了有效性**。main namespace自身的独立路径已有supersession语义，作为holdout祖先时也应保持。

最小修复：在同一个已验证owner snapshot上，对main namespace父记录复用现有native绑定/撤回检查，必要时绑定其规范intent Artifact或复用已有main namespace reader，再保留严格顺序比较。无需新增writer、读取geometry或解析真实main raw/statistics。

## 证据与边界

精简[binding](namespace-proof-component-review.v2.binding.json)保存冻结/working/helper及原报告SHA、独立33/23日志和探针结果，SHA `f74978015049a36a323d1f3a9a6bdc79fdbe3d6e0f380881b646ab08628db0ca`。复现脚本/结果在`/tmp/inspection-namespace-review-v2/probe_v2.py`及`results.json`，关键键为`ancestor-revisions`；所有owner、native rounds和锁subject均为synthetic fixture，未读写真实owner，也不涉及其迁移审查。

本轮没有判断实际dataset完成、SeedSequence rederivation、catalog producer、跨main/holdout真实碰撞、最终source inventory或实际执行接线。返回对象仍不是跨后续变化有效的永久权限；后续consumer仍须在副作用边界重新校验。当前新增P2未闭合，不能据本组件宣布完整formal/G4通过。
