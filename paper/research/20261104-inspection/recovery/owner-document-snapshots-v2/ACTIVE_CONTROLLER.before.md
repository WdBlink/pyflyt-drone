# 当前唯一控制器入口

当前入口是本run专用副本：

`runtime/autoresearch-state-reader-v1/scripts/research.py`

完整路径：`/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/runtime/autoresearch-state-reader-v1/scripts/research.py`。

它继续拥有原`.paper/control.json`与`checkpoint.json`，不是新研究run或第二套scheduler。全局安装的Skill未修改；`runtime/autoresearch-original/`仅作原实现只读快照。原默认16MiB reader拒绝了自己生成的17,326,544B checkpoint（control为16,754,137B），因此原round15 apply未完成，未丢数据。

新副本提高controller自有pair读取上限至128MiB，外部packet默认16MiB及严格JSON检查保留。通过专用副本新增、经实际独立审查的迁移API，implementation从`bfd85afe5d6745b4111d67f3c7d79bef4b0b3e15894b356b1468c180e48a9df2`迁往`a112e7aaff3ab6c99f983a2030fc190d7ccd1af940a943604d94ad318e8fef20`；全部研究/预算字段、旧history与operations、active round保持，过期pending request只在保留来源后清除，须按新rules重新prepare。

证据见`recovery/state-reader-v1/`的原pair备份/binding、candidate subject、independent review、实际migration result和post-migration validation。66项软件测试（1跳过）及独立反例检查仅验证此运行时迁移，不授予任何额外科学、付费、投稿或G4许可。后续CLI操作都使用当前入口，不能把旧安装入口的implementation pin当作现owner。
