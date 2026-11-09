# G2 退休采样修复独立复核 v1

审查日期：2026-11-05。审查者为 `/root/mission_review`，是同模型家族的独立上下文，不是人类领域专家或跨模型审查。工作树为 `/Users/wdblink/.codex/worktrees/538e/pyflyt-drone`。本次只新增本报告及 `/tmp/g2-retirement-repair-review/` 下的小检查证据；没有修改生产代码、旧结果、manifest 或其他审查，没有重跑长 physics、grid、600 s 或 API。经典候选报告已经另文冻结，本报告不覆盖其后续修复。

## 结论与缺陷状态

**原退休报告的 F1 已在明确绑定的 v19 源码和 retirement.v3 证据范围内关闭。** 退役机自 360 s 事件 tick 起停止新增 target/coverage/opportunity/background mission 数据，359.5 s 已曝光的旧包仍于 360 s 保留原值交付，物理返航继续。没有发现本次修复引入的其他具体 G2 阻断缺陷。

这是对冻结记录、唯一源码差异、现有协议/评分正反例和补充 sampler 边界的独立核验，不是仅转述 18 个布尔值。G2 十项在下表注明的语义/fixture 范围内都有通过证据；未执行的综合场景与门级条件仍列为 pending。不能将此报告称为当前平台、G1、G3 或 G4 全部通过，不能将旧 v18 的缺陷回填为已修复。human_review_pending。

`retirement.v3.json` 是从 0 实际积分到 500 s 的受控物理 fixture，使用 controlled next-tick planner，不是墙钟 1×、策略比较、完整科学 episode 或新策略样本。新 600 s 真实 1× 不在本次证据内。G1 无事件组合复用由独立 control review 决定，本报告没有读取未完成组作为最终结果，也没有替它授予资格。

## 唯一修复与版本隔离

逐份比较 v2/v3 的 17 个 source snapshots，只有下列两个文件不同：

- `envs/inspection/mission.py:168`：`if self.flight.exit_active[drone_id]` 增加 `or drone_id in self.retired`。这是唯一环境差异，位于相机几何、coverage、目标 RNG、背景 RNG、tracker、pending 和 local_measurement 之前。
- `eval/probe_inspection_retirement.py:81–92,103–104,123–124`：独立统计四通道的新采集数、保留旧观测，新增两项断言及原始字段导出；原有物理执行顺序与 16 项检查保留。

没有修改 `_deliver`。`mission.py:265–285` 仍按 pending 中的原始证据与曝光路由交付，并将实际 arrival 更新为当时 t。生产 `step` 在积分后、采样前再次执行 `_events`（`mission.py:307–315`）；360 s 精确事件先设置 retired，再进入 `_sample`，所以不是在下一帧才停采。

v19 相对 v18 的 G2 checks 十项文字未变，新增 `retirement_sensing` 明确 360 s 后无新 mission acquisition、旧包不追溯删除、实际返航与独立安全盘旋继续。G1 数值阈值、G3、scorer 未改变；版本、说明和 G1 复用提案元数据变化不是新的执行资格。

17 份 v3 source snapshots 均通过 embedded SHA 校验，并与审查读取时及检查结束时的当前文件逐字节一致；sources_unchanged=true。61 项依赖与 v2 同一索引，当前逐项 hash 一致，dependencies_unchanged=true。依赖索引 SHA 为 `e341ce56ff077476e9752cf57372189de38091462697e24d048c5d9b3a823dbd`；它覆盖声明的 PyFlyt 源码/资产与 PyBullet 二进制，不等于整个 Python 环境的正式锁定。

## 从原始记录独立重计

本次使用 stdlib `base64/zlib/struct` 解码误差 trace，没有调用生产 recount 函数复算；逐行检查有限值、字节数、时间列和 envelope。v3 的 18 项 checks 均为 native bool true，但以下事实另由原始记录得出：

| 核验项 | 独立结果 |
| --- | --- |
| 连续物理范围 | initial=0、final=500；120000=500×240 个物理步、physics audits 和 raw tracking rows。没有 epoch 跳钟。 |
| raw 量 | 120000×17×8=16320000 bytes；时间逐行 `t=i/240`，全部有限。 |
| raw SHA | `f91bf44aa607cdecb7598cc60e29341dabd3cbb52ae215db5978c669f0e8b6a3`，v2/v3 完全相同。 |
| tracking | 明确 xy_tube_m=10；最大 XY=4.090187918431436 m，水平及记录的 Z 区间越界步数均 0。没有用新阈值解释旧 6 m 失败。 |
| 实际轨迹 | 两份结果的全部 5000 个 10Hz 时间戳、四机位置逐项精确相同。它不等于完整 240Hz 状态 replay。 |
| 硬安全摘要 | violations={}；最小间距38.81609667589731 m，实际 Z=95.0640173743258..97.09720555874213 m。逐步 raw 只有误差/Z，不足以独立重构全部速度、bank、碰撞。 |

原 v2 的 F1 与 v3 修复效果由同一时间过滤器重计如下。coverage 这里计实际交付 packet，opportunity 计曝光机会；没有把报告 ID 或最终得分当采集计数。

| 3 号机 measured_at/t≥360 s | 原 v2 | 新 v3 |
| --- | ---: | ---: |
| target sensor_audit | 101 | 0 |
| coverage_evidence packets | 124 | 0 |
| sensor_opportunities | 113 | 0 |
| background_trials | 124 | 0 |
| 新测量的 delivered Observation | 101 | 0 |

合法旧包 `obs-3-719-0` 的 measured_at=359.5、arrived_at=360.0；candidate、估计、类别、quality、covariance、sensor_mode、aircraft_xy 和 camera_xyz 全部与 v2 精确相同，不只是保留 ID。它的质量为0.784169413344689，本报告只证明合法到达，没有冒称它满足计分质量门槛。v3 没有通过重新时间戳将退休后新采样伪装成旧包。

360 s 前的全机 target/coverage/opportunity/background/delivered 数据也与 v2 原始列表完全相同。该修复既没有追溯修改旧采集数据，也没有影响本 fixture 的实际飞行轨迹。

## 退出链仍成立

280 s 批准的 survey job-0001 在359.9916666666667 s 仍 running。360 s 生产事件将它 cancelled、finished_at=360，并使机3版本1→2，创建 exogenous_capacity_exit 的 job-0002。事件前后原 program hash 都是 `ff51a32db2f4c6e2d68cceebb2c3d38f9aba74240c111ee5b90d2a09e68e6a31`，四机该控制步实际位移均约0.142 m；没有在逻辑取消时瞬移或抹去原物理程序。退休机新工作和旧版本 cancel 均拒绝。

新退出 program 在360.5 s future capture 后提交，revision2/hash `accf45d58379f6e31978b93872fbae4f86baedcf03d2f0ea9c54287f9bfd325f`；441.8833333333333 s 才因实际 home circle 20 s dwell 成功。四机终态 dwell=[41.33333333333926,41.33333333333926,41.33333333333926,78.11666666667779] s，剩余航程均约9494..9498 m，高于1000 m reserve；这是实际里程预算，不是电池模型。

本次另从480..500 s 的实际10Hz位置重算，每机201点；半径范围分别为79.84222..79.92438、79.84614..79.91583、79.84770..79.91883、79.89656..79.90077 m。中心差分估计的最大切线 course error 分别为0.1191°、0.1217°、0.1137°、0.0049°。这些是10Hz轨迹旁证，20 s dwell 的全240Hz判断仍由生产 safety hook 产生。原始误差 trace、离散 OwnState 和10Hz位置不能被称作完整物理状态 replay。

## G2 十项覆盖与剩余范围

“PASS（范围内）”表示这项的当前语义已有对应正反例或明确的物理证据，不能解释为穷尽所有场景。前两份冻结报告保留详细反例与旧绑定：`g2-v16-review.v1.md`、`g2-v18-retirement-review.v1.md`。协议/评分纯 core 字节不变，本次重新运行当前测试；不能把旧 sampler/planner/控制结论整体迁移到新版本。

| v19 G2 项 | 状态与当前支持证据 | 仍未由该证据证明 |
| --- | --- | --- |
| 1. 坐标与 flat terrain AGL | PASS（范围内）。已审查平地坐标/实际相机原点、真实 nadir proxy 正例；本次物理 Z 与 sampler 边界一致。 | 非平地/真实视觉传感器不在协议中。 |
| 2. 同步飞机与移动代理 | PASS（范围内）。生产物理 hook 同时刻移动代理；v16真实0→241补传及本次0→500连续步数。 | 没有覆盖所有目标几何/遮挡情形；raw误差不是完整world replay。 |
| 3. actor RPC 隔离 | PASS（范围内）。allowlist与拷贝边界、truth/seed/future/grader禁止入口的当前正反测试；只用public actor输入。 | 不声称任意外部沙箱/进程安全审计。 |
| 4. minted evidence/因果到达 | PASS（范围内）。不可变证据、未来/错误candidate/entity/低质量反例；v16曝光路由及真实240恢复，v3旧包完整值保持。 | cancel/reassign 后不同route的长物理组合未单独运行；对应延迟路由已有mock语义测试。 |
| 5. 作业/版本/取消 | PASS（范围内）。queued ACK不等于成功、stale cancel、短真实handoff接收新合格观测后才完成；v3实际running→cancel/新exit→succeeded。 | 不代表新经典策略自己的pending生命周期已正确；其P1/P2在独立冻结报告中。 |
| 6. 名额不退/固定首次关联 | PASS（范围内）。withdraw/重复ID不退、超容量拒绝、首次未匹配不能救回、20m边界/等距/类别无关及动态seal重排反例。 | 多目标交叉可辨识性不是由单目标fixture证明的性能。 |
| 7. 静态最后合法版本/全分母 | PASS（范围内）。无观测仍保留全隐藏分母，最后错误修订降低成绩、同刻接收顺序与无证据反例。 | 本次退休fixture无完整静态成绩主张。 |
| 8. 动态同刻truth/先筛后选再判 | PASS（范围内）。v16不调用scorer的60格独立重算、当前边界/错误最新/同刻ID/迟到因果反例；protocol/scoring与v11纯core相同。 | v16仅飞到241 s，不能称完整180 s需求窗口成功或强动态跟踪性能。 |
| 9. 补传/格起点priority | PASS（范围内）。v16实际34条210–240断链补传、239 pending与240恢复；当前late-route/coverage到达/priority bin反例；180 s发布边界语义测试。 | 未飞一个完整目标窗口实际跨越180 s并评估策略效应；这是综合覆盖边界，不以代码存在补成实际运行。 |
| 10. 退出/退休/自致失败不豁免 | PASS（本次F1已关闭）。v3实际完整退休/释放/保留原物理占用/返回/dwell/reserve；两sensor正反边界与四通道零；退休/非退休缺dwell、资源不足、自致硬违规的既有负例。 | controlled next-tick非墙钟1×；同版本完整事件episode与真实planner负载须由后续G3证明。 |

目前没有发现另一项具体 G2 代码缺陷需要先修再进入新1×。若门级审批还要求上述未执行的综合场景，应保持 pending 并补相应新证据；不能由本报告自行扩充或取消契约。当前 G1 composition、真实1×/内部计划事件 replay、G4冻结及人类审阅均不在本报告的通过范围内。

## 有界检查

执行命令：

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python -B -m pytest tests/test_inspection_protocol.py tests/test_inspection_scoring.py /tmp/g2-retirement-repair-review/test_sensor_boundary.py -q -p no:cacheprovider --basetemp /tmp/g2-retirement-repair-review/pytest
```

退出0，**56 passed in 11.27s**。其中52项为当前仓库协议/评分用例，4项是自有只读反例：survey/inspect × active/retired。后者在受控几何与合成359.5/360时钟下，先证明有目标样本/coverage/local callback，随后证明 active仍采样、退休且尚未exit_active时全部停止；两种模式的旧包都原样交付。每项明确断言0物理步，不将其称作360 s真实事件结果。原协议套件另有短真实handoff和短worker失败检查，也不算新科学样本。

必要证据保留为 `/tmp/g2-retirement-repair-review/recount.json`、`test_sensor_boundary.py`、`tests.log`。recount SHA=`bcdbae4ed47488115564faba19108e0fdef9b845203c8dede17155f9221cb2f3`，测试脚本 SHA=`830fd6d023a48ee907cdaf68ef0f558a3aebde1b65f9bb01197cf3be818d1713`，测试日志 SHA=`47ce7d67d3c660c4cc229aaf28f070c2a6b52e87e9b83365244a2fb8b3ae62e3`。长期证据为原始JSON和本报告；/tmp仅为本次重计的辅助输出。

## 当前源码与原始证据绑定

HEAD=`270c6f0946b3be59527b2b77982f75a70da7ef77`，inspection为工作树未提交文件，故实际内容由SHA绑定。以下路径中planning/qualification相对 `paper/research/20261104-inspection/`，其余相对仓库。后续源变化必须另版本审查，不能回填本表。

| 文件 | SHA-256 |
| --- | --- |
| planning/gate-contract.v19.json | `c0af9d0c6f49f2f879313d7bea767a917f8fa6a80e96701009d63ef83dab9b74` |
| qualification/protocol/retirement.v3.json | `73c0f9da170f3870a2ffb3a88e4daa553d5a9d9182f942ffc3b1d0936ce36452` |
| qualification/protocol/retirement.v2.json（原失败边界保留） | `99f1f0408dd48f950710d053c3f1d5cd99575dfc62400fd8f26164a90f393086` |
| envs/inspection/mission.py | `715fce4a99325b6d26fbb1e9ffc20896603bfbd4d48396f0513a46f5ad14aec7` |
| eval/probe_inspection_retirement.py | `d60efec443723c4dfa61e6e9821aae333845ffe47c7d655c99394ae980bbc61d` |
| envs/inspection/execution.py | `dc301939ea1d55f08f01c69af5561c75f762ac45bdd0d144d677514387425ea4` |
| envs/inspection/planning.py | `b87fcdedf3aa26a46560056376fb69f9ff969270149c9cae429b4a9e294df6ab` |
| envs/inspection/paths.py | `dbd31dc94a7cdcbcb17b8d9f1847fa88cf4cd12e4609ec8ba57e2233a367855b` |
| envs/inspection/flight.py | `ef9be915c92c207a3e8f43b9ddc1907010beea59be6d96136330e82a5d67cbdb` |
| envs/inspection/control.py | `c676d237efe1b6227d238b4e5948c54a5725626e9d3f69e74ed9c1629a235270` |
| envs/inspection/guidance.py | `9ab9c5290cbce60ad7f9fead635318f41561dedd1cfc2f0984b1cd45385bc0b9` |
| envs/inspection/protocol.py | `c32bdabb67bf8493ff5125d0af4fc289d1d0210db762016a9c56a96dd2f174a4` |
| envs/inspection/scoring.py | `bc8a21947d340fce28c83c3d2313ad9a2e7711e6af07c1bd275f774009b5631a` |
| envs/inspection/scenario.py | `c4f56e2783bc21c087c3abdcbfae9a7472feae2bbbe1e007d86bb39b28c40af2` |
| envs/inspection/reference.py | `85f97b6cb632c7f178c20ad9f53097510cab1dca0cd56aec15a5fe222b9365c3` |
| envs/inspection/__init__.py | `cb0d38bbca6b531b261ca5a5f2f79664e352c9ddb6c19acc1777784c3115107a` |
| envs/__init__.py | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| eval/qualify_inspection_executor.py | `b76f6745e4f1cfed8c0b4055b683a93811f23685378ef2c84c3e723d8010a2cd` |
| eval/qualify_inspection_g1.py | `852ea8c02cade269b48ea1bf99158913ddaef2038f6a368bd0987c31cc794522` |
| eval/inspection_tracking.py | `19151659b4a969c51a6cbcb94b4f10aa1cb4489bebec6bd50f28b639a07884ba` |
| tests/test_inspection_protocol.py | `c486fff43579020140e883f87b2831d3b3c9ab64257bc45144b03e61e7d2382e` |
| tests/test_inspection_scoring.py | `0f9c4f34c5c0f6a92dcaca6fe51b1c6831b826e221eae7ca710499bc41049403` |
| qualification/review/g2-v16-review.v1.md | `4f64037a1cf6e10b521c2baf72fbdcd8f3f5e4f8936d6146ab7b3fbdda0485de` |
| qualification/review/g2-v18-retirement-review.v1.md | `6c9067a926bbdb19be349874d21529d7249ca8e53092d4af3fc7564827ac2a2d` |

protocol.py/scoring.py 已独立比对，均与 `qualification/source/executor-v11/envs/inspection/` 对应纯core逐字节相同。可复用其账本/评分审查，不能复用当时旧控制资格替代当前G1/G3。首次退休v1输出序列化失败与原v18 F1历史继续保留；本报告只为新v3关闭该缺陷。
