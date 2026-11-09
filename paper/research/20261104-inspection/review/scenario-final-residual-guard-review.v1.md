# 两项最终P2闭合复验 v1

日期：2026-11-06。审查者 `/root/mission_review`，同模型家族独立上下文，不是人工领域或跨模型审查。仅新增本报告和附属 `scenario-final-residual-guard-review.v1.evidence/`，未修改源码、历史输入、实际结果、旧报告或旧manifest。未执行physics、API、长episode或重放；实际L1/L3仅作为已存数据读取。

## 结论

**上一份 `scenario-final-guard-and-dev-review.v1.md` 的两个P2均可按当前SHA关闭。** 原copy+world组合缺失反例现在保存已知operational_failure并退出1；common-calendar拒绝priority_at=0和非法schema。正常终态/合法日历没有被一并拒绝，原实际L1/L3的整个score字典仍精确复现。

本轮只证明这两项修复闭合，不重审新场景平台、B1实际运行、统计比较或G4。原报告的缺陷实测及旧source SHA继续保留，不能改写成当时已通过。clean需求日历重计分仍是同一trace离线换权重，不能称clean兄弟已飞或事件因果实验完成。human_review_pending。

## P2终态guard：关闭

当前 `eval/run_inspection_reference.py:251–268` 统一try读取/解析world，缺文件、坏JSON、无效顶层/状态/score容器落operational_failure并保存world_verification_error；private_case_unchanged=false在world条件之外强制失败。原来的“world不存在则跳过失败赋值”路径已移除。

本次直接复用前报告附属原guard-combined.py，在新临时目录使用真实main/worker入口和全flight/process双替身。所有检查为0实际物理步。补充必要的坏world与正常/已知失败分支，结果如下：

| 输入/终态情况 | 当前实测 |
| --- | --- |
| 正常copy及最小合法worker结果 | main退出0，manifest=completed，copy unchanged=true。 |
| worker读入前更改copy字节 | worker=operational_failure，ScenarioMission未创建。 |
| 结束改字节或仅删除copy | main退出1，manifest=operational_failure，unchanged=false。 |
| **原copy+world同时缺失** | **退出1，无未捕获异常，manifest=operational_failure；copy/world verification error均保存。** |
| 仅缺world | 退出1，保存operational_failure及缺文件错误。 |
| world坏JSON、list顶层、非法status、score为list | 均退出1，保存operational_failure和解析/schema错误。 |
| worker已报告operational_failure | 保留operational_failure并退出1，不错误转为completed。 |

这些是CLI故障容错/终态语义检查，不是完整业务记录的资格验收。双替身的0s world与空score不能作为真实flight或科学样本。本次未把容器检查当成完整评分/物理schema验证；外层receipt/资格校验职责继续保留。

## P2需求日历入口：关闭

当前 `eval/inspection_common_calendar.py:63–65` 在任何重建/重计分前调用validate_case(demand_calendar)。独立用原实际L3的同一CaseSpec派生原反例，得到：

- priority_at=0：ValueError `invalid event calendar`。
- schema=`unsupported-calendar-schema`：ValueError `unsupported development case identity`。
- 不同wind几何：原同root身份门仍拒绝，错误为event siblings must have identical underlying root geometry, motion and noise seed。
- 合法clean L2需求日历：仍接受，角色保持offline_common_demand_calendar_not_a_new_rollout_or_primary_score，离线D=.32222222222222224、Y=.7688882486635297，与修复前合法换权重行为一致。

保留的before-guard日志确有calendar/schema两项DID NOT RAISE，结果2 failed、2 passed；本次当前四项测试为4 passed in .58s。它们包含手算同日历/静态权重变化，以及非法calendar、schema、不同geometry反例，不是仅检查代码中存在函数名。

## 原实际L1/L3结果保持精确

读取原world与private_case，调用当前rescore_common_calendar(world, original_case)，两份结果均original_score_exact=true，返回整个score字典（包括完整实体/格选择、分母、余量和诊断）与原world.score完全相等：

| 既有实际记录 | C | S | D | Y | H/E | SR |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| qualification/dev-diagnostic.v1/L1-00-full | .9940374787052811 | 1 | null | .9970187393526406 | 1/1 | 1 |
| qualification/dev-receipts.v2/dev-L3-00-B0-0/attempt-00/flight | .9844425237683665 | 1 | .27794561933534745 | .7541293810345713 | 1/1 | 0 |

L3重建产物也与原 `L3-00-common-calendar.v1.json` 整体精确一致。原L1 world SHA=`7d899a029f0b18f8d3995722271c8b57e47ac489ca6fa23812030e7a47189639`，原L3 world SHA=`faaf04929e7ccd4558f96a4b346b30e241fcccd902c2224d3a0d6b2112b41bca`。未重跑它们，没有将新guard版本倒灌其原source/manifest，也没有把父任务后来完成的B1加入本次检查或样本数。

## 实际命令与证据

在仓库根执行：

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python -B /tmp/scenario-residual-review/guard.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python -B /tmp/scenario-residual-review/calendar_probe.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest tests/test_inspection_common_calendar.py -q -p no:cacheprovider --basetemp /tmp/scenario-residual-review/pytest
```

两独立脚本及测试均退出0；guard对十种main分支和读入前字节变更分别断言，calendar_probe对原实际记录与原非法/合法需求日历断言。本次只自行运行上述4项测试，没有把父任务报告的相关52项或旧全库运行计入本审查执行数。

附属证据含guard.py/JSON、calendar_probe.py/JSON、before/after测试日志及本轮两个受审源码副本。binding.json保存八个证据文件、当前source和两份实际world的SHA。复现须使用新临时目录，避免探针已存在的fake输出目录；不可覆盖真实记录。

## 当前源码绑定

以下为本次实际读取/执行SHA，不延用旧f357/852版。HEAD仍为 `270c6f0946b3be59527b2b77982f75a70da7ef77`，工作树未提交内容由SHA绑定。

| 文件 | SHA-256 |
| --- | --- |
| eval/run_inspection_reference.py | `1afeab04ba55b8311fe0c2f093c7174627fb9226918886f4066161ff62e04068` |
| eval/inspection_common_calendar.py | `c5246ab15bbeaf5ccb9baf6d9037e74a830072723b6f6421f2288ea8d50240ba` |
| tests/test_inspection_common_calendar.py | `f862c633f6228c0889794302dff163e7b9885b5e8494a6ea6ae8b9e395a57d61` |
| tests/test_inspection_cases.py | `ed0d873e823a65dd1661ee2462690dfe1e4505b290167a53981642f8e75f2d19` |

结论截止于本轮明确版本和有界检查。两项P2的关闭不自动授予G4、clean因果比较、全平台正确性或人类领域审阅资格。
