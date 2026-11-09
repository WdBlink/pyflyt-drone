# DatasetProof / namespace seed guard：未安装组件独立审查

结论：发现一项P2，当前候选应保持未安装，先闭合最终native snapshot之后的dataset字节/tree复核。所附dataset29、namespace41及pre-draw Fake7项共77项独立通过，不能覆盖已复现的窗口。其余本次限定的结构、碰撞和native负例未发现新增阻断；这不是真实formal dataset、G4或完整执行graph验收。

审查者 `/root/control_review`，同模型家族独立工程上下文。R=`paper/research/20261104-inspection`。输入 `planning/formal-admission-candidate.v3/dataset-component-freeze.v1/binding.json` SHA `409dc5addc3d214f8f463ee472da3141ecf4e13a8fc7c24c74c9d8fc16e3a138`；18项冻结文件及7源/7测试working字节全部核对。Dataset源 SHA `efbcc5e3c2e734ed7d4134cbb8889f8c4c10fb7e35b9934d5bc41d8c2d3404fa`，namespace源 `0a8ab3868d0c5b1970a8f777d843b1977072a7080c47e65632639d71833aee37`，G4Proof源 `b6643f98f7a719cb7017eeca3d5fc91f92c91afd1ec97c9bfe1022e1d83d0ec1`。

## P2：最终owner读取后，dataset closure变化仍被接受

冻结 `inspection_dataset_admission.py:70`读取manifest/review，`:105`核report，`:155`核case/audit，`:159–162`核tree；然后`:163`才读取最后owner snapshot。后面的`:165–168`只fresh namespace/source/liveness，`:169–170`只比较native evidence中的Artifact引用和应用顺序，没有再次验证dataset实际字节和tree。

独立main六反例：正常72-root合成fixture中，在第4次原生reader调用期间，分别仅向manifest、independent-review、report、一个case或audit追加空白，或增加一个unbound文件。owner/pair、G4、源、namespace及native finding均不变。**六例全部返回72-record DatasetProof**；前五例紧接着用其原Artifact.checked_path读取当前文件，均明确报SHA变化。不是缺少mission反序列化，也不是返回后无限期权限问题，而是本函数可观测的最后snapshot→返回窗口没有闭锁此前已声明的字节/tree条件。

holdout依赖同样受影响：完整结构正例有8次原始reader；在第8次才修改locked-main case或main independent-review，或新增holdout目录文件，仍返回12-record DatasetProof。相同位置的task_stop、main dataset finding修订和holdout dataset finding修订三项控制例全部正确拒绝，正常holdout正例仍接受12。说明最新native谓词本身工作，遗漏的是当前dataset及递归locked-main的实际文件闭包。

最小必要修复：在最后snapshot/fresh namespace之后、返回之前，对**同一已绑定版本**的manifest/review/report、所有case/audit及精确tree重新闭合，并涵盖holdout所依赖的locked-main DatasetProof闭包。继续保持hash-only私有字节边界；不以替换为新SHA、新manifest或新review“接受”变化。未来worker在真正读取/副作用边界仍须单buffer哈希/内容身份检查；本修复也不是原子文件系统或永久返回后保证。

## 已核正向与必要负例

- 72/12固定manifest、独立review/report、source/pop/catalog/namespace Artifact、root/episode/mean分母、metadata顺序、严格native整数、canonical case/audit路径及初始字节/tree都实际检查。scope内的源变化、stop及namespace/dataset撤销负例能拒绝。所附完整重哈希/native-bound错误manifest并非只靠外层SHA被拒绝。
- within/dev/main↔holdout的sensor/root/master集合及semantic/config标签碰撞检查存在；holdout DatasetProof取primary-lock所绑定main manifest并递归验证其结构review，然后加入该main的sensor/semantic/config集合。所有这类结果仍只是被真实独立review绑定的标签/字节 provenance，不是本组件自己计算私有mission语义真值。
- 新namespace guard在合法opening/父链后、读取holdout intent并取得literal seed后，拒绝与main master/root相交；四个组合负例已在独立41项namespace测试中执行。pre-draw七项使用Fake generator，证明namespace失败/无效root/错误NumPy环境不调用真实生成器，合法场景只把已有literal seed传给Fake；本轮没有正式RNG或geometry draw。
- formal_admission变化仅为REQ增加五个已staged consumer；原G4谓词无其他diff。shared law、phase及catalog不在本轮重做分布/历史覆盖审计；最终source inventory和writer/worker/service全graph接线仍是后续职责。

## 证据、执行范围与未证项

独立实际执行：dataset **29 passed/1.89s**；namespace **41 passed/1.37s**；pre-draw Fake **7 passed/0.95s**。root的173 combined日志是输入证据，本轮未独立重跑全部173或shared-law的5个实际dev种子回放。main/holdout结构正例均为synthetic non-mission blobs及literal seeds，不能称真实qualified数据、真实独立review通过或formal实例已生成。

[证据目录](evidence/dataset-admission-component-v1/)保存两个独立脚本、main六反例/holdout七个结果与三份测试日志。[binding](evidence/dataset-admission-component-v1/binding.v1.json) SHA `6ccfd05d8d569f814c8ca26b53b7bd865070fe12a534aaa15d8b860017e5395f`。脚本先核working14文件必须与冻结v1同SHA，未来v2工作文件变更时会拒绝误测；需要在隔离的对应布局重现旧版，不改回共享工作树。所有owner/checkpoint/native rounds只在/tmp合成fixture，不复制/修改真实owner。

本轮没有审新writer、实际内容decoder、真实raw/statistics完成、完整catalog覆盖或整个未实现graph；这些明确待办不被另报为本组件已承诺功能的缺陷。未运行模型、物理、正式种子/实例或资源动作，不改production/冻结源/旧证据。原P2和六反例保持v1历史状态，后续修复另版复验；报告/绑定冻结后立即停止写R。
