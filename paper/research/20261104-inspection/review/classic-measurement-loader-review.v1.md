# 经典已完成回执measurement loader独立审查 v1

日期：2026-11-06。审查者 `/root/mission_review`，同模型家族独立上下文，不是人工领域或跨模型审查。仅审 `eval/inspection_measurement_loader.py` 的非LLM已完成回执读取与相关调用边界。只新增本报告及附属 `classic-measurement-loader-review.v1.evidence/`；所有反例修改均在/tmp副本，未修改logs、frozen classic/core/runner、真实.paper或已有结果。没有启动physics、模型生成、GPU或重放，没有把进行中cohort当完成证据。

## 结论

**基础拒绝与重计分路径有效，但仍有三项具体一致性缺口，不宜仅据当前loader结果宣称回执链完整。** 缺失/未知/非连续/追加attempt、未重绑的intent/start/stdout破坏、冻结expected不符、source声明/case文件改动、连同本地hash更新的错误score均被拒；只有operational failure的目录抛异常，没有被转换成Y=0。

三个/tmp反例仍被接受：把本来有效的首飞改标可重试失败后选择后一attempt；嵌入run.sources字节不符声明SHA；已完成记录的stdout缺失但hash字段全为null。前两类含重新计算本地回执hash的步骤，检验的是语义/完整性联接，不是声称普通位翻转能绕过原hash。原真实输入的嵌入source正确、文件字节与基线copy一致，本报告没有发现它们已经受损。

LLM当前明确拒绝；非development的G4 receipt只从调用者expected引用读取，缺gate会拒绝。本轮不认证调用者的外部冻结过程、完整SDK接线、任何正式G4或物理资格。human_review_pending。

## 输入与隔离方式

只选择已保存completed terminal的 `logs/inspection-classic-dev-flights.v1/dev-L3-00-B0-0` 作为单attempt基线，读取Y=.8245752535103837；另只读 `dev-L1-00-B0-0` 的合法失败后完成链，读取Y=.9970187393526406、retained_attempts=2。这些是既有记录的读取结果，不是新flight、额外统计样本或方法比较。

基线先真正复制到 `/tmp/classic-measurement-review/baseline/`，各反例只在这份**已隔离副本**下创建，写入使用替换新文件，未让硬链接写回原日志。最终核对原plan、intent、start、stdout、terminal、run、world、private-case字节均与未改基线一致。未读取缺terminal的L1-B2等进行中episode作为结果。

探针expected从已复制plan取值并加phase=development，仅用于固定比较分支。它不证明调用者已独立冻结design/dataset/method；生产调用者不能从待测episode临时自签expected，否则失去该参数的外部约束意义。

## F1 [P1] 先前失败/可重试资格只信terminal声明

位置：`inspection_measurement_loader.py:63–84`，特别81–82行。completed分支调用validate_flight并比对evidence，known_operational_failure分支却只看status和retry_allowed truthiness，没有核对其实际失败、原因或是否符合有限重试白名单。

原始反例：复制已完成基线为attempt-00和attempt-01，更新第二份intent.ordinal及文件hash；把第一份terminal改为known_operational_failure/retry_allowed=true。第一份仍是validate_flight可接受的有效完成，loader却接受第二份并返回retained_attempts=2。

增强反例进一步把第一份Y设null、伪写failure_reasons=[clock_lag_exceeded]与terminal.evidence的lag=.2，不依赖“失败terminal还留着正Y”这个明显格式矛盾。结果仍接受：

```text
claimed attempt-00 = known_operational_failure / Y:null / clock_lag_exceeded
actual validate_flight(attempt-00) = Y:.8245752535103837 / failure_reasons:[]
loader returns attempt-01 Y:.8245752535103837 / retained_attempts:2
```

这说明“第一个合法完成之前只有符合政策的操作失败”没有被读取器自己验证。当前反例第二份是隔离复制，没有实际执行第二次；没有把它说成两次真实飞行，也没有观察到原cohort按成绩重试。该漏洞允许接纳内部矛盾的重试记录，足以削弱“不挑后一次”的读取保证。

应在loader中对已dispatch的每个失败attempt重新计算validate_flight/evidence，要求确有操作失败且属于同一冻结重试政策，而非把policy Y=0或任何自报失败变成免费重试；strict bool校验retry_allowed等标志。合法not_started/spawn failure须单独核对intent、未start/未flight及具体错误，不能用它跳过一个实际存在的完成flight。不要为本局部修复修改已冻结classic/core/runner。

## F2 [P2] source声明匹配，但嵌入source内容没有被验算

位置：`inspection_measurement_loader.py:52–56,77–79`，调用现有validate_flight比较run.source_hashes与plan/expected声明，但没有验算run.sources字符串。

隔离反例只改run.sources某一源码字符串，保留source_hashes声明不变；再更新本地run/terminal record hash及validate_flight evidence。loader仍接受Y=.8245752535103837，而 `sha256(run.sources[key]) != run.source_hashes[key]`。

普通文件改动不重绑回执会被拒；这里证明即使本地hash自洽，loader仍可输出“source_bound”而源码包自身不符冻结声明。它不等于能认证任意有权重签整包的恶意文件写者，也不证明运行时执行了篡改代码；缺口是应能检查、却未检查的归档source完整性。

应核对sources/source_hashes/外部expected的完整key集合及每份字符串SHA。应按归档/记录源验证，不要求历史结果对应当前不断变化的live文件，也不能用重写旧source_hashes来“修复”原结果。合法空__init__.py字符串仍应接受。

## F3 [P2] completed可以缺stdout，null等于null仍被当成绑定

位置：`inspection_measurement_loader.py:66–73`。_bindings允许缺文件返回None，这对部分未启动失败有用途；loader没有为completed强制stdout必须存在。若terminal.record_hashes.stdout.log与stdout_sha256同时为None，比较仍相等。

隔离反例删除completed的stdout.log，将上述字段按实际缺失状态重新计算为None；其余flight和score不变，loader仍接受同一Y。未重绑的单纯删除会被拒，说明现有hash防线生效但没有补齐mandatory evidence要求。

应要求completed所需intent/start/stdout/run/world/case文件存在且hash为有效值，并保留逐项相符检查。不要用“文件非空”替代“文件存在”：合法捕获到的空stdout有真实SHA，和缺文件的null不同。

## 已验证的拒绝与正确读取

主探针共19个分支，结果为2个既有完成正例、14个正确拒绝、3个上述误接受；增强重试反例另做一次确认。没有运行pytest全库或新增实际episode。

| 检查 | 实测 |
| --- | --- |
| 缺terminal | FileNotFoundError；未知不归零。 |
| 缺intent/start/stdout，原terminal未重绑 | record chain differs，拒绝。 |
| status=unknown或只有known operational failure、没有completed | 抛异常，不返回0。 |
| 完成后追加未知attempt或attempt非连续 | 拒绝，已有completed必须是最后attempt。 |
| source_hashes声明不符冻结expected | physical validation拒绝。 |
| copied case文件字节变化 | physical validation拒绝。 |
| 修改S/Q/Y使内部聚合表面自洽，再更新world/run/terminal hash/evidence | 完整report/coverage/denominator重算不符，拒绝；不是只信保存总Y。 |
| frozen expected.case_id不同 | 读取前置比较拒绝。 |
| expected condition=B3-M1 | 明确报model receipt adapter未qualified；不能拿physics trace冒充模型证据。 |
| phase=main且expected缺g4_receipt | 拒绝；未读取某个硬编码本地“通过”文件。 |
| 实际单attempt与实际合法失败后完成链 | 返回原Y及真实retained attempt数。 |

代码用`value is None`区分操作失败与合法数值0，不是用`if not value`拒绝/替换0。本轮没有为制造“真实Y0正例”伪造一份flight；能确认的结论是上述操作失败/未知路径在loader中抛错、没有被impute为Y0。SDK上层必须传播或按预先定义缺失规则处理异常，本局部审查不证明上层从未吞异常。

## 外部冻结与资格边界

expected承载case、condition/repeat、suite、source/dependency、supervisor/receipt-adapter、thread环境和clock阈值的外部冻结值。loader比较这些字段，并执行两次以内的连续attempt结构、case canonical身份及完整score重建。它没有独立认证expected来自哪个批准者；这与文件头声明一致，不能将自签expected当外部设计约束。

对于main/holdout，代码从expected.g4_receipt取得path/SHA并检查status=qualified。本轮仅确认缺gate拒绝及该读取来源，没有制作一个假qualified文件来宣布正式通过，也没有审阅完整G4 receipt与设计/方法/数据集的资格关系。实际本次输入全为development，局部loader检查不是正式G4。

该函数是已完成文件读取器，不运行实际world、不认证任意并发重签文件者、不重新证明完整240Hz控制/传感器资格。完整score重算仍使用原记录的safety及240Hz-hook余量输入，不是新odometry或完整物理重放。caller/owner应保障completed记录不可变及外部manifest/后续SDK记录绑定。

## 实际命令与证据

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python -B /tmp/classic-measurement-review/probe.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python -B /tmp/classic-measurement-review/strong-retry.py
```

两者退出0，意味着成功捕获并断言这些行为，**不表示三个误接受行为正确**。小反例副本留在/tmp；长期证据只保存脚本/结果、受审loader及原完成输入hash，未把大world克隆加入仓库或计入样本。

附属 `classic-measurement-loader-review.v1.evidence/binding.json` SHA=`efe2060764f5f47d6c20448755d0e639929ffeb6ece154b9d4d71e863be5997e`，保存六证据文件及调用上下文source SHA。baseline-bindings保存原plan/intent/start/stdout/terminal/run/world/case的逐文件hash。原选择输入的plan SHA=`ea9f9294ae92ab41e3ee865397d9a76749e160c88b9a0b5a1115143409e53a54`，terminal SHA=`35ed4a32fff671d3812d8aaffabd6eb8f6185cfca1619e0d6c7f4978f0703ac7`，world SHA=`142a752289d501d58bbf74eb4054fe40bcc18fe6b0ad2b7a6ec03606c87776a4`。

| 源文件 | 本次SHA-256 |
| --- | --- |
| eval/inspection_measurement_loader.py | `f358bb7f32ecb254f6cba804353a6909b26e66b29044abf6d7b30b03f407537c` |
| eval/inspection_episode_receipts.py，仅调用上下文 | `59f88d5a3ec90497e19ec91ac9b914f09fe6a2303d029e6aecbb6fb2f958387b` |
| eval/inspection_common_calendar.py，仅重计分上下文 | `c5246ab15bbeaf5ccb9baf6d9037e74a830072723b6f6421f2288ea8d50240ba` |
| eval/inspection_recording.py，仅读取条件网格常量 | `9613a8e800d470110dbf86890837ba853c3f50834ae29f0722994927d27cb352` |

HEAD=`270c6f0946b3be59527b2b77982f75a70da7ef77`。结论仅绑定本轮loader与临时反例，不修改/否定此前已冻结真实记录，也不为后续修复或SDK正式接入预先背书。
