# classic pilot source-bind guard独立复审 v1

日期：2026-11-06（Asia/Shanghai）。**source ABA及source/supervisor/adapter错版准入问题在本次实测范围内关闭。正常、严格cache重包与全分母故障路径通过复验，未发现新的接线阻断。** 这不是pilot完成、B_classic*已选定或G4验收。

本次为同模型家族独立上下文，与主Agent分离；只新增本报告及`review/classic-pilot-source-guard-v1/`证据，不改源码/SDK/输入/live logs。所有行为探针使用隔离源/suite/61依赖文件及合成JSON worker，真实physics CLI全部拦截；实际V2产物只读检查，没有新飞行/API/模型调用。

## 1. 精确版本与命令

受审`eval/run_inspection_classic_pilot.py` SHA256：

`b59263b31b413caaee0924600e1e8974b104ee47a15199ca1d6e89ad36267464`

完整源码保存于`review/classic-pilot-source-guard-v1/results/driver-source-snapshot.json`。原db4bdaa驱动、ABA反例和旧报告保持不变。独立复用前次隔离框架，只在本次证据中增加source-map、supervisor、adapter三个负例注入，保留同一ABA机制与真实提供SDK。

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python paper/research/20261104-inspection/review/classic-pilot-source-guard-v1/probe.py
```

结果及合成SDK包均保留在`results/`，所有场景`physics_cli_executed=0`。这些包不得以driver写出的actual角色文本当作真实pilot数据。

## 2. guard语义与独立复验

`:62-67`从初始frozen构造预期worker源集合，B0不含classic模块，B1/B2包含；`:122-124`要求receipt plan的source_hashes精确等于该集合，并分别核对supervisor与receipt-adapter SHA。它比较的是**最初冻结字节**，没有拿当前receipt自己定义预期版本，故补上了cache自洽与运行级frozen之间的缺边。

| 独立场景 | 实际结果 |
| --- | --- |
| 原source ABA：child记录变化源、返回前恢复磁盘 | 17completed+1failed，18行/18attempted，SDK failed，无selection |
| source-map中单个指纹错误，磁盘始终正确 | 同上，无selection |
| supervisor_sha256单独错版 | 同上，无selection |
| receipt_adapter_sha256单独错版 | 同上，无selection |
| 原最后slot suite/case变化 | 同上，无selection |
| 原最后slot依赖变化 | 同上，无selection |
| 原CLI返回后bound stdout改变 | 首槽审计拒绝；18行，attempted1、completed0、余17未启动，无selection |
| CLI非零 | 18行，attempted1、completed0、余17未启动，无selection |

四类source版本故障在SDK complete前被拒绝，没有转成策略Y0，也没有因post audit补飞。原source ABA探针仍生成一条内部自洽但不属于初始源map的receipt，修补后该条不会进入已完成统计。

正常包18/18 completed、18次完整post cache审计；B0合法Y0保留。第二个新SDK envelope全18槽复用同store，再做18次完整post cache审计；合成worker累计仍18，**新增0次**。近tie手例仍按1e−12规则选择B1。选择仅出现在合成完整18槽以后，不将其作为真实B_classic*。

## 3. V2稳定失败包和实际完成回执

只读`logs/inspection-classic-dev-recording.v2/results.json`，SHA：

`cda2e3332b30909ffe55afc8ad4167c810efb5616da716ed64def5baaa359a11`

核对execution_status=failed、reported denominator=18：**1 completed（B0 cache）、1 failed（B2 parent-source变动）、16 stopped/未启动**，attempted2，无selection。该失败不代表B2策略Y0。

两条实际terminal及六文件绑定、Y/evidence、单attempt数量均核对通过：

| dev-L3-00条件 | 完成terminal SHA | actual attempts | worker源数 |
| --- | --- | ---: | ---: |
| B0 | `35ed4a32fff671d3812d8aaffabd6eb8f6185cfca1619e0d6c7f4978f0703ac7` | 1 | 16 |
| B2 | `d9d255ed9ef6279b16907f8bc09ffe817fc5f583e8721b97a979dd8794549c5c` | 1 | 17 |

其plan worker source-map、supervisor/adapter SHA、依赖字典和suite SHA与V2最初绑定一致；B2的成功CLI返回也明确actual_attempts=1，terminal SHA同上。实际物理回执完成与parent SDK拒收被正确分别保存，没有重写旧包或将回执复制为第二个样本。

这两条是未来严格cache重建的可核验输入；**本审查没有启动V3，也不报告V3已复用、已完成或已选择**。后续状态应看新SDK实际raw/results与原terminal哈希，不能将计划写成完成。

## 4. 验收边界与决策轨迹

当前source guard在所测试的本地经典pilot接口范围内通过。后续真实18槽仍须同source/suite/依赖与完整receipt闭环，全部完成后才能冻结dev-only B_classic*。model generation0和paper初始plan未接受是父任务提供的运行边界；本审查没有推进paper controller或调用模型。

硬kill/机器故障仍按unknown保守阻断，实际fsync不等于物理掉电认证。正式八配置、LLM/provider费用、SR/事件/删失及G4并未因本guard复审通过而完成。

决策轨迹：保留旧ABA机制，逐项增加source-map/监督器/adapter负例；正常与cache正例确认未封死合法路径；复查suite/依赖/回执链及非零返回的完整18分母；只读已结束V2及两条完成terminal，未执行V3或物理CLI。所有结果绑定本报告SHA版本，不批准以后同名文件的变化。

Attention

reviewed by inherited-parent model（同模型家族独立上下文，非跨模型家族审查）

source-bind P1已在已测范围关闭；真实pilot完成、dev选择与G4仍是尚需独立证据的后续结论。
