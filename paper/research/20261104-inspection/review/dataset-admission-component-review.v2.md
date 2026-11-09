# DatasetProof v2：最终闭包 P2 有界复验

结论：v1 P2已关闭，本次限定修复未发现新增阻断。原九个最终窗口反例现在全部拒绝，正常结构正例及原stop/撤销拒绝仍正确。候选仍是未安装的字节/provenance组件；不是实际formal数据、内容资格、writer/fullgraph或G4许可。旧v1报告与反例不改写。

审查者 `/root/control_review`，同模型家族独立工程上下文。R=`paper/research/20261104-inspection`。freeze.v2 binding SHA `821ec3e9979b37423fec762fdd954769a514dbc4b64eb818659c6e932d424937`，全部17项冻结文件及working14文件SHA相同。Dataset源 SHA `890ae2432ac6453c3f8c13ff95112703f858c43e35a50d24a6ab6abe65de1f93`。其余6源（namespace、G4 REQUIRE5、phase、formal population、shared law、catalog）与v1字节不变，本轮未扩大重审或重复其全套测试。

**修复覆盖原窗口。** 冻结 `inspection_dataset_admission.py:37–45` 的frozen DatasetProof新增同版本report Artifact及可选main DatasetProof；records仍为tuple，main parent由已通过递归读取的结果构造。`:186–193`完成最后native owner、fresh namespace/source及finding/order校验后，`:194–196`构造该版本并调用closure检查，之后才返回。

`:60–70`先递归locked-main，再对保留的manifest/review/report及每一个case/audit调用原Artifact.checked_path，最后重核精确文件tree及symlink条件。没有重新算一个“当前SHA”来换掉已绑定版本，也没有deserialize私有case/audit。`:48–57`与初次tree检查共享同一规则，避免最后窗口只检查了部分文件。

**独立复用原反例。** 只在/tmp适配到v2，保留原v1强hash guard脚本不动：

- main第4次原始owner读取期间，改manifest/review/report/case/audit字节或增加额外文件的6例，现在全部拒绝；前五例报artifact changed，最后报fixed tree。均仍恰4次原始reader。
- holdout第8次读取期间，改locked-main case、locked-main review或新增当前holdout文件的3例，现在全部拒绝；均仍恰8次reader，证明递归依赖也闭合。
- 正常holdout仍返回12 records并携main72 records、同版本report；immutable parent phase为main且其parent为None。最后task_stop、main dataset撤销、holdout dataset撤销三项控制例继续拒绝，没有因新hash阶段吞掉native失败。

所附 **47 dataset tests passed/4.17s**，包含原29正负例和新增18窗口回归；私有内容不可反序列化的原检查也执行。root报告“新增18在旧source17 failed/1 passed”和208 combined作为输入记录，本轮未独立重跑旧版18、writer或208。v1已实际执行的41 namespace/7 pre-draw结果只按未变字节条件复用，不宣称本轮新运行。

[evidence](evidence/dataset-admission-component-v2/)保存两份适配脚本、13个场景结果和日志，[binding](evidence/dataset-admission-component-v2/binding.v1.json) SHA `6b1106e0149f43608c660b0b2eeab847bcdcb9d4cb0db38d2104fd2483ec85c6`。脚本先拒绝与冻结v2不一致的working输入，防止未来误把新实现测成旧版。

本结论只关闭函数内已复现的最后snapshot→返回窗口，不保证文件系统原子性、任意并发变化或返回后永久权限。实际consumer仍须在其副作用/内容读取边界验证同一字节和语义身份；synthetic non-mission blobs不能据此变成真实qualified dataset或真实独立内容判断。本轮没有正式RNG、模型、物理、实际owner写入或安装，不审statistics agent负责的新writer。报告/绑定冻结后立即停止写R。
