# G1–G4 仿真资格与正式主实验登记

唯一科研 owner 已应用 round88，request SHA `a039c6f760326ec0de87348019b9d758c318356b777131881f33e3e4655d68cf`，无 stale 或计划变更。原生 finding `inspection-g4-simulation-qualified-v1` 绑定：

- 冻结 subject：`qualification/formal-freeze-final.v1/subject.json`，SHA `123fecfb9a283874e6d3064cff4dc702cf86fdfcfa525cb5ede3ce703111bac5`。
- 独立审查：`qualification/formal-freeze-final.v1/independent-review.json`，SHA `15693d6620917969fe8b07927afb8b272d4c57d1f14e1a3f8374706b24373290`，`/root/control_review`，`accept_declared_simulation_scope`，无未解决阻断项。
- 资格回执：`qualification/formal-freeze-final.v1/acceptance.json`，SHA `52f795fc918b619ed89fe3ddc9e3c18e0c4320540d5b31959d9098f5bceb8d92`。原生应用后，生产 `read_g4_proof` 在 active round89 重新验证通过。

通过的是当前版本、当前主机、单机串行仿真的 G1–G4：非 LLM 参考已完成搜索、观测、确认/报告、安全退出；开发 pilot 108/108、安装测试970项、真实 clean/20秒时延、开发排除与容量均随 subject 绑定。实机与领域人工审查仍待。正式阶段须按关闭的 SDK 对原字节做无损压缩并复测空间；157,206,686,420 B 是含预留的实测样本形状压力投影。首次正式种子效果前实测可用 164,460,761,088 B。

在资格后，一次性正式主 namespace intent 写入 `datasets/main.v1/namespace-intent.json`，SHA `ce25fda5bbb58b6462d632b633c001e9ded354f7fedd5a198d7e3ab2dd882a6a`；72根（ID L1/L2/L3 各16，OOD L2/L3 各12），在正式生成前须独立于本回执做原生登记。没有在 G4 前抽种子，没有开启 holdout。本点无正式几何、飞行、得分、统计或论文正文。
