# 六根classic pilot驱动独立审查 v1

日期：2026-11-06（Asia/Shanghai）。**旧v1驱动的全分母、正常/缓存路径和选择算式得到支持，但存在三项P1接线缺口，不能据其18/18机械完成就接受dev选择。** 三项均在隔离的提供SDK+合成JSON回执中成功复现，未启动任何真实physics CLI，未写当前live logs。

本报告绑定旧驱动快照，不审查正在修补的current文件。与主Agent分离、同模型家族独立上下文，不是跨模型家族验收。仅写本报告与`review/classic-pilot-driver-v1/`独立证据；旧输入、SDK、live logs、真实flight均未改。

## 1. 版本、现场观察与独立探针

| 绑定对象 | SHA256 |
| --- | --- |
| `qualification/source/classic-pilot-driver-v1/eval/run_inspection_classic_pilot.py` | `f03ad63c236f267ef59f6ce28d753d0ab9b178415f7193915411b11936ae0d5e` |
| 提供SDK `runtime/autoresearch-state-reader-v1/lib/experiments.py`（研究根下） | `4dffd762966ef27d78fd521341aefb2145895b26ad9d382d16f04f1bab6eec02` |
| 同SDK `lib/production.py` | `8e8b41849893980267378fa2843af69b0f65555cac5523bfc33f7a3e37275770` |

初次只读现场观察：`logs/inspection-classic-dev-recording.v1/`已有6×1×3的design及18槽schedule，首槽为dev-L3-00/B0；results.json和selection.json尚不存在。design SHA为`d70e2c1244597702f8e17a28a4dc0d6ec121eed4daf134f80195b651814b001b`，schedule SHA为`413cb00aef766135d634959a9deed7d8823e2b5d6e7433efa9d088ffbb02331b`。此观察是partial，不是任何slot/矩阵完成结论。

`qualification/classic-pilot-live.v1.json`声明native local classic dev、generation_calls=0、非G4/正式结果；初始paper plan仍未接受。已有本地dev授权与paper科学接受是不同层次，driver/SDK机械完成不能替代后者。

独立加载f03ad6快照，将fixture的ROOT指向仓库读取稳定源；未修改该快照。真实`run_inspection_dev`的subprocess.run全部被拦截，模拟其CLI响应，仅实际启动无物理/无API的短Python JSON worker。输入、receipt、SDK输出均在新review目录；源码/探针与结果位于`review/classic-pilot-driver-v1/{probe.py,results/}`。

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python paper/research/20261104-inspection/review/classic-pilot-driver-v1/probe.py
```

探针包整体明确为synthetic。其SDK raw内的`actual_source_bound_dev_flight_or_verified_cache`是被测试driver自动写的角色文本，**不能据此把这些探针产物计为真实flight或pilot成绩**。

## 2. 得到支持的SDK/选择行为

- schedule固定PCG64 seed2026110601，六个预选ID必须为L1/L2/L3各00/01，lineage声明唯一；每root随机B0/B1/B2各1次，共18槽。metadata记录倒序不改变schedule。这里验证身份约束，不证明真实root独立性。
- 直接使用提供的ExperimentRecorder。design在任何回调前落盘，18 planned身份固定，实际执行按schedule而非SDK canonical grid顺序。
- 合成正常包恰18行completed，B0合法Y=0保留在分母。全18完成后才生成selection，未过滤零分/低分。
- 对同一verified synthetic flight store建立第二个新SDK envelope，18次callback均cache_reused=true；合成worker累计仍18，新增0次。新SDK attempt表示记录/导入回调，不是重复飞行；不能因新包另增独立样本。
- 模拟首槽CLI退出7：SDK reported denominator=18、18行，attempted1、completed0、execution_status=failed、无selection。正常栈展开的KeyboardInterrupt同样保留完整18行和无selection。
- hard kill/机器崩溃不经过Python上下文展开，不能声称已经补齐SDK终态；持久化design/schedule仍保留计划分母。恢复应使用新SDK目录，从真实receipt检查cache或被unknown阻断，不覆盖旧不完整包。

选择算式为每层两root平均再三等级等权，B1/B2比较；全部18格成功后才进入算式。合成手例B1=.5、B2=.5000000000005，按≤1e−12 tie规则选择B1；差距.2时选择B2。B0始终保留在三配置design及原正式矩阵，不因未被选择而删除。这只是dev选择，不是八配置结论或显著性检验。

## 3. P1-A：最后slot的suite/case变化可进入完成与选择

旧驱动`:80-82`仅在slot调用前检查suite SHA；`:100-103`调用后只再检查source，未比对receipt plan的suite_manifest_sha256、case_file_sha256与最初冻结manifest，也未在最终selection前重新验证suite。

独立`results/suite-final-race`在第18次模拟CLI接手后，更改当前slot的隔离case文件并更新隔离manifest对应hash。模拟child读到新suite，产生自身内部一致的plan/receipt；旧driver仍得到18/18 completed并生成selection。18个receipt plan里出现**两个suite SHA**，同root最后一个条件用了不同输入字节。

这不是证明live输入已经变化，而是可复现的准入窗口。单靠下个slot前检查不能覆盖最后slot，且不能将“文件在slot前相同”当成“child实际读取的plan绑定相同”。需在slot前/后及最终检查suite，逐条比对plan中的原suite、预期case文件/canonical hash和case/condition/repeat身份；变化时保留全分母并禁止选择。

## 4. P1-B：跨slot依赖版本未冻结到同一个pilot设置

旧`:49-54`冻结项目源与SDK源，但没有运行级的依赖初始指纹；child每次自行生成plan.dependency_hashes，`:99-102`仅以该plan验证自己。不同slot各自自洽，不等于18slot共享同一个依赖包线。

独立`results/dependency-drift`只改变隔离的synthetic依赖文件，不改真实PyFlyt/pybullet。最后child plan的依赖指纹与前17个不同，旧driver仍18/18 completed且生成selection，`dependency_hash_sets=2`。

需运行级冻结完整物理/运行依赖，slot前/后及最终核对，并要求每条receipt plan与该初始依赖字典相同。source、suite、依赖及必要runtime设置均须属于统一的可审计比较条件；不能将单次run的dependencies_unchanged=true当成跨slot一致证明。

## 5. P1-C：CLI返回后的完整receipt链未重核

旧`:99`只调用validate_flight核对run/world/private-case；`:100-102`检查terminal SHA及其部分evidence，但没有重核terminal.record_hashes中intent/start/stdout等完整链，也没有调用缓存完整校验器。

独立`results/receipt-chain-race`先让模拟CLI通过真实本地receipt模块生成完整、可缓存的合成回执；在返回response后、driver准入前只改bound stdout.log。terminal文件及其response SHA不变，run/world/private-case仍一致。旧driver接受该slot，随后18/18 completed并生成selection；最终检查发现**1条record hash不匹配**。

该窗口与之前修复的缓存链问题同属完整性根因，只是发生在child已返回、SDK raw尚未完成之间。应以receipt模块的完整cache入口做post audit，禁止dispatch的回调作为断言：若必须新启动或unknown，则直接阻断；不得在审核过程中偷偷补飞。核对plan与原冻结条件后再写SDK complete，并把完整receipt SHA/身份/cache角色留在raw。

## 6. source、失败与当前live运行边界

提供SDK在操作异常下拒绝科学结果；failure_policy=penalty/failure_value0是其固定envelope约定，不授权将infra/unknown当策略Y0。probe确认操作失败行不是合法测量、未启动17格仍保留，selection不产生。

主Agent已接受上述三项并另作current修补。其说明当前旧parent将等首个child正常结束后，因旧冻结driver源已改变而source check abort，保留全18 planned failed/unstarted；真实child receipt不丢弃，新envelope随后严格cache复用。**这属于主Agent告知的执行/恢复安排，本报告没有将它写成已观察完成，也没有审查current修补。**

当前首槽真实receipt即使完成，也不能按本报告直接宣称pilot已完成；应分别检查其物理/source receipt、旧SDK失败包的分母和新SDK包的verified cache重建。不能把old parent的source不一致解释为策略0分，也不能因内部修复把同一个flight复制成两次样本。

本次没有修改live command/session66295、当前logs或任何真实源/输入。formal test/holdout、五个LLM配置、模型/API费用、正式八配置SR/事件/删失与paper plan科学接受均不在本次dev driver验收范围。

## 7. 决策轨迹与冻结结论

| 检查/决策 | 独立证据 | 结论 |
| --- | --- | --- |
| 只读观察live初始化 | design/schedule SHA，results/selection未存在 | partial，不计完成 |
| 旧快照+provided SDK跑合成正常/缓存 | baseline、cache-reuse summaries | 全18逻辑槽；重包新增0 worker |
| 模拟操作失败/栈展开中断 | operational-failure、interrupt summaries | 全分母18，attempted1，无selection |
| 最后slot更新隔离suite/case | suite-final-race summary与plans | 2个suite SHA仍完成，P1-A |
| 隔离依赖变化 | dependency-drift summary与plans | 2组依赖仍完成，P1-B |
| CLI response后破坏bound stdout | receipt-chain-race summary与terminal | 1条hash失配仍完成，P1-C |

旧v1选择不可仅凭SDK完成标记验收。current修补必须以新SHA单独复审，保留原source/反例及实际receipt来源；本报告不提前批准新实现，也不赋予G4。

Attention

reviewed by inherited-parent model（同模型家族独立上下文，非跨模型家族审查）

三项P1是隔离合成复现的可行窗口，不是live数据已损坏的指控。真实pilot尚需完整18槽与严格统一绑定后，才可冻结dev-only B_classic*选择。
