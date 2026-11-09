# staged G4 proof validator 独立复核

审查者 `/root/control_review`，同模型家族、独立工程上下文。审查对象仅为冻结的proof-validator组件，不是完整formal admission；不将缺少namespace/dataset/holdout-open功能列为未兑现承诺。没有修改候选、执行源、真实owner或当前pilot63源，没有模型/物理、正式seed/order/case生成。

## 结论与版本

**候选尚不宜作为可靠的已验证G4 proof边界接线：3项P1、1项P2及1项低优先级JSON缺口如下。** 当前只是未安装候选，没有据此发生真实formal准入。

实际源SHA `3656097e2bd431c616e07ef2e8619c7c8d30892f946972b8ba30983e338d14f5`；测试 `b7a42277841ed858ad5a99d165cf731ba958e7423bacd718bee5b793906c338b`。binding声明的四项实际文件（含contract/log）全部核验。独立复跑原测试为 **26 passed/0.07s**，另做的隔离负例说明其覆盖尚不足以支撑本组件的强绑定声明。

以下行号指独立归档 [inspection_formal_admission.365609.py](staged-g4-proof-validator-v1-evidence/inspection_formal_admission.365609.py)。所有positive/negative subjects、reviews、native rounds均为`/tmp`合成软件fixture，不能当作真实G4或权限证据。

## P1-1：JSON验SHA与解析不是同一份bytes

**位置：77–88、45–60行。** `Artifact.json`先调用checked_path，按pathname流式计算SHA，再重新打开该path解析JSON；两次打开之间可以更换内容。解析出的subject/closure因此不一定是native绑定的那个SHA对应的内容。

独立反例先改变临时mission source，使原qualified manifest正常拒绝；随后只在closure原SHA计算返回后、JSON读取前替换closure，令其认可新source SHA。结果正常路径拒绝，同一输入在此确定性check/parse间隙注入后接受：返回的G4Proof仍带旧manifest SHA `dd469ae7…`，实际已解析manifest SHA为 `19de87bf…`。这不是猜测任意后续源码变化，而是在本次proof构造过程中接受了未被所声明SHA绑定的JSON。

**最小修复**：JSON artifact从同一次打开、同一个受限byte buffer计算SHA并解析；例如最多读取maximum+1字节，核长度、SHA、编码与严格JSON后返回，不能先hash pathname再二次open。保留root/path约束。证据：[probe_json_binding.py](staged-g4-proof-validator-v1-evidence/probe_json_binding.py)、[json-binding-results.json](staged-g4-proof-validator-v1-evidence/json-binding-results.json)。最初探针因`/tmp`与`/private/tmp`比较未触发hook；原输出保留在results.json，最终探针使用resolve后实际触发1次并稳定重现误收。

## P1-2：声明closure的哈希齐全，不等于实际消费闭包齐全

**位置：15–29、172–185行。** source仅要求人工列出的31项为子集；实际Python会执行的`envs/__init__.py`、`envs/inspection/__init__.py`不在必需集合。dependencies则只要求非空字典，没有实际PyFlyt源码/资产及PyBullet等必需依赖集合比对。未列的有效消费者可在qualification后变化而不触发检查。

两个隔离事实：

- 在被接受的synthetic source tree中建立`envs/inspection/__init__.py`，随后改变它，proof前后均接受且native_request相同；该文件始终不在manifest。
- 原positive fixture只有一个任意`synthetic-dependency.bin`，仍被当作“complete qualified source/dependency closure”接受，根本不含实际物理消费依赖。

这是**完整集合校验缺失**，不是要求合成fixture运行PyFlyt，也不是说任意source hash都不检查。列在manifest内的文件变更会拒绝；漏列的import initializer、依赖或其他消费者没有保护。实际完整入口尚未接线，最终consumer闭包还需覆盖相应evaluator/SDK/新入口，不能把当前人工最小名单当作其自动完整性证明。

**最小修复**：由独立确定的实际consumer/source与环境依赖集合核对manifest，至少加入package init并核真实物理依赖集合；如允许superset，也必须有完整必需集合与新增/缺失策略，不能以“非空依赖”替代。测试使用显式fixture environment resolver供合成输入，不用生产路径或模型。证据：[probe.py](staged-g4-proof-validator-v1-evidence/probe.py)、[results.json](staged-g4-proof-validator-v1-evidence/results.json)的initializer_drift与arbitrary_one_dependency。

## P1-3：未验证native owner完整性，control内自洽一行被当作实际apply

**位置：106–125、151–161行。** request compact digest、decision回指及evidence SHA比对本身正确；但owner直接read_json，既不核Brief正文摘要，也不核control/checkpoint pair、schema或implementation pin。由此不能从一行自洽round JSON推出“actual applied native round”。

独立负例：

- control包含qualified round，但checkpoint的有效state完全没有该round，两个state digest明确不同，proof仍接受。
- 将初始Brief与brief_sha建立一致关系并绑定subject；之后只改变owner.brief，保持旧brief_sha。即使把checkpoint更新成与control一致，proof仍接受，尽管Brief digest已经不匹配。这项与pair缺口独立。

现有native运行时`lib/runtime.py:205–220`恰有schema、Brief digest、checkpoint pair与implementation校验。本probe没有调用真实CLI或写真实state；它证明当前reader未承接这些既有原生完整性条件。

**最小修复**：在read-only模式承接原生owner完整性协议，保证读取一致的control/checkpoint状态、核Brief/accepted plan身份及schema/pin；遇到原生原子保存期间的不一致可有界重读或拒绝，不能忽略pair。无需新ledger、放宽gate或在此组件执行预算动作。证据同results.json中的unapplied_control_round与brief_drift。

## P2-1：后续明确撤回/修订资格，旧finding仍被接受

**位置：106–125、192行。** 只寻找同finding_id的一条历史verified_progress记录，没有处理原生已有的`finding.revises`后续修订关系。

隔离fixture在资格之后追加另一条自洽native round，明确`revises=synthetic-finding`并声明该subject资格withdrawn/G4needs_work，附bound revocation artifact；control/checkpoint保持一致。旧acceptance/source/plan未变，旧proof仍接受。

**最小修复**：定义并验证该qualification finding的当前、typed supersession/revocation状态，或跟随现有revises链核终态；后续明确撤回应拒绝，直至新的source-bound接受记录闭合。不要靠任意自然语言字符串猜判，也不要把任何无关后续round都当作撤回。证据为results.json的superseded_finding。这是资格生命周期问题，不是要求本组件实现holdout opening。

## P3：非有限JSON guard漏掉指数溢出

**位置：45–60行。** parse_constant拒绝NaN/Infinity literal，但标准数字`1e999`会由json.loads解析为Python inf，read_json实际返回该值。已有测试只覆盖NaN。

建议parse_float或递归数值检查拒绝溢出，并保留有限native数字的原canonical行为。本轮只证实helper缺口，**未将它夸大为现有必需字符串字段的一条独立formal越门路径**；主阻塞是上述P1/P2。结果见results.json的json_overflow_helper。

## 已通过与未证范围

已实核：正确synthetic proof形状会返回G4Proof；原26项native request/evidence/科学引用/缺源/重复key等检查仍通过。当前真实run的 `qualification/g4-next-integration-intent.v1.json` 被用作acceptance输入时，正确报`a typed G4 qualification acceptance is required`；没有在真实run建立假acceptance或synthetic round。

设计上将acceptance、subject、independent review分开，避免自引用hash，且绑定当前frozen plan与design/population/registry，这个方向成立；但reviewer非空字符串只能验证声明，不能认证独立审查确实发生。qualification evidence/hardware在此仅做immutable artifact绑定，不验证实际性能或当前硬件适用性，本轮不据此授予任何科学或dispatch资格。

模块、G4Proof类及函数docstring已经明确**不是namespace/dataset/holdout permission**；本轮不将尚未实现的其余触点当作本组件违约。未来caller仍须另核namespace/dataset/holdout-open，并在副作用边界重新验证所需证明；`for_dispatch=True`目前只是owner liveness检查，离线返回的同一G4Proof类型不能被当成正式运行能力凭证。

本轮没有实现修复、审计完整formal系统或复核当前pilot成绩。全部独立证据SHA见 [binding.v1.json](staged-g4-proof-validator-v1-evidence/binding.v1.json)，binding SHA `419334924264cec0efb1f7b995793b58ac0a85a70adb1a0c892222f49e37579b`。没有G4/main GO结论。
