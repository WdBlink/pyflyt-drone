# 精确037追加处置 v1 独立审查

结论为 `request_changes`。v1在读取native owner后的返回窗口丢失固定facts和helper源码引用的约束。原037归属证据的独立结论保持有效。

审查人为 `/root/control_review`。本报告绑定proposal SHA `4d63026f0ba4e1e0d3cbaa22942c78aa60c4325b0b5bceff366eeeed1c7f1c2b` 和disposition源SHA `b672966121e7789f7541a37c75f3cbfad3863e70969cf45b73c45606b6c48e07`。原工作源已切换至v2时，首次脚本的SHA guard拒绝执行。该日志保留。后续从SHA相同的 `rejected.v1/disposition037.py` 原样重放，未把v2当成v1。

源77和78行先保存facts/source引用，86行读取owner。96行却重新从当前FACTS读取成员列表，返回前也未核FACTS自身及source_ref的初始SHA。独立隔离fixture使用真实 `verify` 和真实 `require_native_binding`，原plan/terminal/tree/sidecar的字节拷贝，合成native applied row，冻结原recount输出替身。正常对照通过。以下两项实际文件变异均被接受，仍返回0.3314065510597302。

- 在owner读取窗口把事实文件的成员列表清空并改写一个原成员。返回的external_attribution仍指旧SHA，实际FACTS和原成员均已变。
- 单独在相同窗口给helper源追加字节。当前源码SHA不再是独立review/native引用的原值，仍返回结果。

缺native applied finding和新增空目录均被拒，恢复正常字节后正控通过。完整运行和变异SHA在 `window-results.json`。修复需保留初始解析的事实成员和引用，在闭合树重验之后逐一检查固定成员、FACTS本身、helper本身以及proposal/review/acceptance。prepare应同样在发布前终验最初选定的ref。只在/tmp执行反例，原记录、native、进程及正式数据未改。037原unknown和旧SDK分母继续保留。
