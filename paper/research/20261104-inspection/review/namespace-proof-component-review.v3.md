# namespace proof v3：main 祖先撤销修复独立复验

结论：v2新增P2“main namespace已被superseded，holdout仍可接受”已关闭。正常合成holdout仍接受12 roots；初始祖先撤销、祖先artifact/内容错误及最后snapshot新撤销均拒绝。本次限定差异范围未发现其他阻断项。组件仍未安装，只读artifact/owner/phase/seed-intent链；不授G4、main/holdout执行或完整formal GO。

审查者 `/root/control_review`，同模型家族独立工程上下文。R=`paper/research/20261104-inspection`。只审原P2及相邻的canonical祖先artifact、native顺序、最新snapshot和额外reader边界；不改候选、实际owner、生产源或旧报告，不执行模型、物理、RNG或正式geometry。全部seed为隔离fixture literal；fixture复制公开root名字不等于实际生成正式实例。

冻结输入 `planning/formal-admission-candidate.v3/namespace-proof-freeze.v3/binding.json` SHA256 `513f0c3cab0d124ae03528e5a27c8172de7a40abbccd6c4b11738cf12251e332`，其6项文件逐一核SHA。namespace源 `2ebd0e4b710457691a4d93f69ef9e204e30cd1a7aa3f0cd5a3856f28e229b7f9`、测试 `3eda5705abb1febc716c68afdf1e9a034787a671023f0dcd4df36f7b8d654ee7`。working源/测试先核与freeze相同；G4Proof仍为已审v3 `c981449425a835fad40eda9bbdc47dc76d2aa0e473a26bde8312743739a1779d`，fixture helper `191d7af54bee6062053897d40fbd5f6e81f2c639a1f455004951a073ee94dc53` 未变。

**修复位置和因果边界。** 以下行号指冻结的 `inspection_namespace_admission.py`：

- `:83–98` 新 `read_main_namespace_ancestor` 先复用原 `require_native_binding(owner, fixed_main_finding_id, [])`，核单一、当前、有效、未撤销的原生finding，之后才从该finding的request.evidence取固定 `datasets/main.v1/namespace-intent.json` 的SHA。没有另造alias或让caller选择其他main路径。缺证据或改字节都会在Artifact校验拒绝。
- `:93–95` 对此canonical main intent复用完整原 `read_namespace_proof`，而非仅查finding存在/索引。通过 `replace(verification, owner_reader=lambda _: owner)` 复用同一个已经原生核验的snapshot，不增加未验证raw reader。`:96–97`还比较返回的native request SHA与本次祖先finding。内部phase由canonical main路径决定，不走holdout-parent分支（`:182–196`），没有holdout递归。
- 初次holdout通过 `:194 → :169 → :145` 在`:197`读取holdout intent之前复核祖先。最后的原生snapshot在`:241–242`取得并核完整G4后，`:252`再次复核相同父链；因此第二次之后新发生的main撤销也不能沿用旧缓存。
- `:147–149`的 `G4 < main namespace < main dataset < main lock` 严格native索引，以及 `:172–173,253–257`的opening/holdout子链仍保留。复用完整main proof也保留原G4/accepted design/source/pop/catalog、literal seed范围及dev collision等谓词，没有降低分母或允许新namespace。

**独立执行与原反例闭合。** 核字节后，纯fixture执行namespace测试37 passed/0.83s。另在 `/tmp/inspection-namespace-review-v3/test_repair.py` 运行7个独立fixture检查，7 passed/0.22s：

1. 完整正例返回holdout12 roots；原始 `owner_reader` 恰好3次。通过对象identity探针确认两次main祖先复验分别使用第2/第3次已经读取的同一owner对象，没有额外原始reader。
2. 原v2撤销反例：只新增 `revises=inspection-main-namespace-v1` 的native-shaped finding，其他父artifact不变，现在在holdout intent bytes读取前拒绝，原始owner读取2次。
3. 修改main intent字节但不改SHA，或移除main finding的canonical evidence并重算native request/decision digest，均在holdout intent读取前拒绝。
4. 不止验证一个外层SHA gate：把main `master_seed`改为已在qualified dev catalog中的literal7，更新main artifact SHA、main native evidence、request/decision digest和paired synthetic owner，仍由完整main谓词拒绝，holdout intent尚未读取。
5. 交换main namespace/main dataset的applied位置，保持其他材料不变，仍在holdout intent读取前拒绝。第3次原始owner读取才新增main撤销的独立反例也在final check以superseded拒绝；它不是声称该后发变化能倒退阻止已发生的只读，而是拒绝最终返回proof。

上述测试全部为临时目录中的合成owner/checkpoint和native-shaped记录，无真实owner写入。新四个所附反例与旧33均实际执行；phase源 `44abef28366a3104bdc9c653f056c1596e3b7a9cfd23501343eaabbb4434f321`、phase测试 `194275f8e42a4a73a09e9058cae86077c6b3199b970003f48f567396eff843eb` 与v2逐字节不变，本轮没有重跑phase23，其历史23通过只作条件复用，不说新运行。

独立小binding为 [namespace-proof-component-review.v3.binding.json](namespace-proof-component-review.v3.binding.json)，SHA `c41296b3b9fc499325cb283b95394c0bfe87055055d5712ba75240449dc26c6e`，保存freeze/working/helper、旧报告和两组日志/独立脚本SHA。v2原报告及其当时的开放P2保留，不回写成“当时已关闭”。

未验证范围保持contract原声明：真实namespace writer、catalog来源完成、SeedSequence重推导、dataset/SDK/raw/statistics真实性、跨main/holdout真实碰撞以及最终G4 source inventory/实际consumer接线均不在本次组件修复验收内。返回NamespaceProof不是跨后续状态变更永久有效的权限，consumer必须在其实际副作用边界重新复核。本报告/绑定冻结后停止写R，不扩大审查。
