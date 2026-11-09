# G4 开发进度 v2

G1–G3 的限定资格继续有效，G4 仍为 needs_work；不授予正式主评测、holdout 或结果写作。原六图四表、八条件和19项义务均保留。主评测+holdout候选1512个600s飞行/252串行飞行小时不含开发、clean siblings、消融、延迟或重试。

初始研究计划已通过正常唯一owner API接受：`packets/plan.ral.v3.json`、`experiment/design.ral.v3.json`，初始plan不是G4。实际绑定和两次CLI（旧schema拒绝/正常接受）见`initial-plan-admission.v1.json`及相邻native回执。Brief原38处字符串改为原文单元素列表，精确可逆；正常set-contract/既有用户指令approve、当前预算问题重绑定原真实默认预算答案后接受。不是用户新作答、预算增加或scope缩减。

冻结评分器v2将整个17源文件集合与manifest绑定，拒绝未列包/pyc/后代symlink，禁止生成pyc；独立复验关闭旧v1两个导入遮蔽P1。原7条实际飞行全score精确复算，新增private semantic digest与ROI完整格映射校验。它仍继承实际range/safety审计输入，不能称完整物理重放或model真实性认证。旧v1源码/反例/报告全部保留。

SDK canonical bridge以预先存在的原始design字节初始化outer recorder，不再重建另一份design。1296个合成inner callbacks、576个root mean记录及native science validator通过；不是1296条飞行，也未生成真实main实例。开发经典selector必须等固定18slot全部完成；本次prefix为10个完成slot，完整回执与评分重计均通过。一次时钟lag=.239176s的已知基础设施失败被保留，按原一次retry规则后完成，不计策略零分，不取最好分。live staging仍在`logs/inspection-classic-dev-*`，任何最终选择须以完整同版本批次为准。

LLM wire/service五项正路径缺陷经fake runtime定向复验关闭。B3/B4 actor、异步pipe client与开发runner现为未完成独立集成验收的原型；43项边界测试及compileall通过。tokenizer-only对三个固定模型、五个synthetic公共context变体得到4830–6480input tokens，低于8192，实际模型forward/generation仍0；这不证明实际所有场景context或GPU竞争已合格。尚未进行资源reserve或付费调用。

G4剩余：完成18slot强经典开发比较；精确实现并在dev验证固定ID/OOD population（含45m身份guard，不能把旧dev重标为已通过新guard）；模型runner/操作receipt和未知结果恢复、同预算/提示/消融资格、真实三家模型与全八配置六根pilot、实际GPU/物理1×竞争、clean/event兄弟和响应/恢复/删失loader、最终源/模型/提示/分流/重试manifest及独立资格复核。全部通过后才生成main/holdout。领域人工审阅仍pending；结构化合成感知、有限平地包线与空中退出不等于实机能力或安全认证。
