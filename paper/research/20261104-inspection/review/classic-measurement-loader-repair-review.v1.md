# 经典measurement loader三项修复定向复审 v1

日期：2026-11-06。审查者 `/root/mission_review`，同模型家族独立上下文，不是人工领域或跨模型审查。只新增本报告和附属证据；未改生产、真实logs、frozen classic/core/runner、真实.paper或旧报告。没有新physics、模型生成、GPU、重放或SDK记录写入。

## 结论

**原审查的三项误接受在新loader中均已闭合。** 四个原/强化反例全部拒绝；正常完成、实际合法重试链以及三种隔离known failure正常链均获准，没有在这些分支中发现误拒。失败attempt没有被当成Y=0，返回值仍来自唯一有效完成。

本轮只做该修复范围的9分支定向检查，不授予SDK整体、任意回执认证、LLM adapter、物理或正式G4资格。旧 `classic-measurement-loader-review.v1.md` 与f358源码/反例保留，不回填为当时已经通过。human_review_pending。

## 原反例关闭

当前source为 `eval/inspection_measurement_loader.py` SHA=`fa93cdc64a12271908bafeeea7fd099a5d9c70d0d7bab57492d0db26daec109b`。本次直接重新读取上一报告在/tmp构建的隔离反例，不改这些对应的真实原文件。

| 原反例 | 新loader实测 |
| --- | --- |
| completed缺stdout，terminal两处hash重绑为null | ValueError，attempt intent or output binding differs。 |
| run.sources字符串改动、声明SHA不改、本地回执hash/evidence重绑 | ValueError，embedded worker source bytes differ from the frozen manifest。 |
| 有效首飞被改标known failure/retry_allowed，复制完成结果为下一attempt | ValueError，prior failure is not an independently recomputed retryable infrastructure outcome。 |
| 强化首飞反例：Y=null，伪称clock_lag_exceeded/.2，但实际validate_flight仍通过 | 同样拒绝，不仅靠“失败还留正Y”这一个格式矛盾拒绝。 |

对应实现：70–71行拒绝stdout缺失的None绑定；75–83行对已有run的完整嵌入source字节映射重算SHA并与冻结plan一致；93–99行重新计算每个prior failure，要求failed_value=None、terminal Y=None、非空原因且仅属于冻结infra集合，并在已dispatch情况下比对evidence。合法空__init__.py源码仍可通过真实完成正例，未误用字符串truthiness拒绝它。

## 正常known failure未误拒

下列全为0物理步文件分支fixture。第一attempt在/tmp副本中按现有receipt adapter语义构造，第二attempt使用复制的既有完成记录并更新intent ordinal与必要hash；不是新执行了两个flight，也不是向真实cohort追加attempt。

| 正常分支 | 结果 |
| --- | --- |
| known spawn failure | intent存在，stdout为合法空文件，not_started=true，spawn_error明确，returncode=None，无start/flight；随后完成被接受，retained_attempts=2。 |
| known worker在world前失败 | 有start和捕获stdout，known returncode=1，无flight，validate_flight给非空worker_operational_failure，terminal evidence匹配；随后完成被接受。 |
| known worker在audit后失败 | 完整绑定输入/源/score元数据，run/world标操作失败及受控cleanup错误，returncode=1，重算仅worker_operational_failure；随后完成被接受。 |

三个fixture均返回完成attempt原Y=.8245752535103837，失败部分没有贡献一个0或额外统计样本。它们检验的是当前冻结receipt adapter允许的已知失败语义，不扩展其retry白名单，也不推断所有部分损坏/任意schema的失败都应重试。

另读取两个原始已完成正例：

- 单attempt `logs/inspection-classic-dev-flights.v1/dev-L3-00-B0-0`，Y=.8245752535103837、retained_attempts=1。
- 实际合法失败后完成链 `logs/inspection-classic-dev-flights.v1/dev-L1-00-B0-0`，Y=.9970187393526406、retained_attempts=2。

两者原Y保持不变。没有读正在进行的episode作为结果，没有改其任何文件或将读取动作算作新measurement/flight。

## 范围及外部冻结责任

本次沿用上一报告的expected分支fixture（原plan字段加development），它不是独立认证caller已经冻结design/dataset/method。非development G4 gate仍由caller frozen expected提供；LLM显式拒绝及原有错误不impute为0的边界未被本次修复更改。本轮没有重新制作正式G4正例，也没有审阅SDK外层是否始终传播异常。

源码/回执内部一致性验证不等于认证任意有权重写整包/返回码的人；completed记录不可变、外部expected与后续SDK绑定仍由caller/owner保证。新loader对字段的增量检查也不是全world物理资格或完整240Hz重放。

## 实际命令与证据

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python -B /tmp/classic-measurement-repair-review/probe.py
```

退出0，9分支断言全部通过：4个原/强化拒绝、2个真实完成读取、3个正常失败fixture。没有追加pytest全库、长物理或模型运行。脚本沿用上一报告已准备的/tmp反例；长期复现应先按旧报告在新的临时目录构造相同反例，不能在真实logs上运行变更步骤。

附属 `classic-measurement-loader-repair-review.v1.evidence/` 保存probe/JSON及受审loader，binding记录它们、上下文source及父报告/父evidence SHA，binding自身SHA为 `94db5a27b6deb246b6f315e9be97bd10b3535b42d58caaf23ef4970761de9b14`。父报告SHA保持 `0ad7bd519f84650f2a5924b58718f13047b2a3546f7fe9237b535f880369abae`，父evidence binding保持 `efe2060764f5f47d6c20448755d0e639929ffeb6ece154b9d4d71e863be5997e`。

| 源文件 | 本轮SHA-256 |
| --- | --- |
| eval/inspection_measurement_loader.py | `fa93cdc64a12271908bafeeea7fd099a5d9c70d0d7bab57492d0db26daec109b` |
| eval/inspection_episode_receipts.py，仅调用上下文 | `59f88d5a3ec90497e19ec91ac9b914f09fe6a2303d029e6aecbb6fb2f958387b` |
| eval/inspection_common_calendar.py，仅重计分上下文 | `c5246ab15bbeaf5ccb9baf6d9037e74a830072723b6f6421f2288ea8d50240ba` |
| eval/inspection_recording.py，仅读取条件网格常量 | `a2fbdddcf7546b3f69f1cc8d2fc1d565620be3ce6582c6f61f8e93923decffcb` |

recording.py上下文在两轮间更新，本轮仅使用其条件网格常量，未将上下文hash记录扩为对新SDK实现的整体审查。HEAD仍为 `270c6f0946b3be59527b2b77982f75a70da7ef77`，结论截止于上述版本和9个有界分支。
