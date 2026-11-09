# 本地模型wire/service v2修复独立复审 v1

日期：2026-11-06。审查者 `/root/mission_review`，同模型家族独立上下文，不是人工领域或跨模型审查。只新增本报告及附属 `local-model-wire-service-repair-review.v1.evidence/`，不改生产、旧报告、真实.paper、真实resource目录、模型或实际结果。正向执行全部是隔离FakeMLX/微型blob；真实generation、真实MLX load/forward、GPU和physics均0。

## 结论与版本界限

**原五类反例在本轮最新v2b中均得到正确处理，可按下述范围关闭。** 独立Fake完整main检查及28项边界单元通过；实际无plan的main仍在模型receipt/MLX之前拒绝，真实control字节不变且没有创建operation目录。未获得正向硬件、时延、模型行为、episode或G4资格。

本轮先检查service `64d6e196...`（v2a），原五类反例已修，但实证了模型SHA验证期间owner撤销后仍到Fake load的窗口。父任务随后仅增加两次admission重读，形成 `cfbb8e90...`（v2b）；本次在全新临时路径重新运行全部Fake probe和28单元，撤销反例现在在Fake load前拒绝。两版source、各自service-intent及probe结果分开保存，不将v2b行为回填v2a或原v1。

当前没有发现本轮既定反例在最新版本仍误通过。该结论不是任意秘密识别、原子reservation lease、GPU硬取消或整个资源owner/LLM episode链正确性的证明。human_review_pending。

## 原五类反例复验

| 原问题 | 最新v2b独立实测 |
| --- | --- |
| F1，浅plan/round与非有限caps获准 | 假plan对象、错digest、空round/空stage、NaN calls/tokens、bool calls、浮点tokens全部拒绝。真实owner纯函数与独立canonical字节oracle匹配；fixture plan正确绑定时接受，wrong purpose/phase/model均拒绝。 |
| F2，空/截断模型清单绕过pin | 已绑定receipt/profile的empty files仍在Fake load前拒绝；unlisted blob、同大小改blob均拒。完整4B blob成功，`.cache`与`.DS_Store`按声明排除。profile max-output超限、漏Mistral显式flag也在Fake load前拒。 |
| F3，IO失败却称已journal token | ENOSPC发生在JSONL写入前，generated=1、journaled=0、实际journal行0、status=runtime_failure。usage为calls1/tokens3（input2+runtime-observed1），没有把已生成消耗免费归零或虚称落盘。 |
| F4，无yield慢流超时却completed | Fake时钟10s、soft cap1s，无yield结束后status=policy_timeout、deadline_exceeded=true、actual wall10s；generated/journaled=0，usage=input2。没有真实等待或GPU调用。 |
| F5，嵌套grader字段通过serializer | score/truth/ground_truth/grader_output分别嵌入合法8顶层frame的observations，均被递归拒绝；正常public fixture仍可序列化。worker public_state仍是信息流主边界，不宣称能识别任意秘密文本。 |

原v1及本轮v2a的失败/修复证据继续保留。正向Fake不是某个真实模型输出；这里的“接受”是服务谓词/文件/响应分支，不是owner真实接受研究plan或允许真实资源消耗。

## owner canonical digest独立比对

没有以service._digest作为自身oracle。读取实际owner `runtime/autoresearch-state-reader-v1/lib/runtime.py:36`，只用AST提取其digest纯函数，未import整个owner、未load/save任何state。该owner在922行使用digest(packet)设置frozen_plan，随后将同packet写入plan receipt；本次只读取这些源码。

另先写定独立canonical字节：

```text
输入对象：{"z":[1,"中文"],"a":false}
预期字节：{"a":false,"z":[1,"\u4e2d\u6587"]}
SHA-256：4020f3c434fab19a1dad8c5d230f5b5fb6d1f7870246835905ac9b6257d28fa9
```

实际owner函数与service函数均精确匹配。这个非ASCII/乱序键正例覆盖默认ensure_ascii=True、sort_keys、compact separators的关键差异；不是两个函数对纯ASCII样例恰好相同。

本轮/tmp owner fixture的frozen_plan由**owner实际纯函数**计算，服务核验该digest与receipt packet，而非用service helper同时生成预期与检查。每个fixture registry都在临时run内，plan packet绑定其file SHA，purpose JSON绑定kind、development phase、registry SHA和receipt SHA，选中唯一profile。

真实frozen_plan仍null，真实plan receipt尚缺；未来真实accepted packet的digest现在不存在，不能由fixture预先认证。本轮证明的是规范序列化相同及给定fixture联接正确，真实完整计划仍必须由唯一owner按其完整计划/预算验证流程接受。

## registry与模型pin范围

当前prospective registry为 `planning/local-model-registry.v1.json`，三候选M1/M2/M3的download receipt小文件SHA、file_count和repo/revision元数据逐项相符，file_count分别15/12/10。registry记录generation_calls_at_creation=0，三者均标prospective_development_candidate_not_G4_qualified。该文件尚未被真实plan接受，不是资源许可。

服务当前在admitted_limits中将plan digest、run内registry路径/哈希、reservation purpose与唯一profile联接；main约束CLI请求不超过profile、enable_thinking=False、fix_mistral_regex精确匹配；receipt file_count与profile一致后，verify_model_files要求非空/去重/实际可见文件全集相同，再逐blob核大小、路径和完整SHA。

Fake反例明确在身份联接成立之后测试empty manifest，而非因wrong model先失败：registry/receipt/purpose均绑定空清单，main仍在完整清单检查拒绝。unlisted与同大小改blob同样没有Fake load。正常fixture还放入被明确排除的.cache/ignored与.DS_Store，完整可见清单仍通过，证明排除规则不是口头说明。

本次**没有**重读十余GB真实权重做SHA，也没有真实MLX正向load；元数据联接及4B fixture不替代未来真实blob重核、tokenizer兼容或硬件验证。

## runtime与journaled计量、soft deadline

最新正常Fake请求input=2，runtime-observed=1，JSONL成功fsync记录1行，response及completion usage=calls1/tokens3。第二条请求因本地max-calls被not_started，未建第二call；同reservation再次启动被exclusive mkdir拒绝，不能通过重启自动清零继续消费。正常服务前后/tmp owner control也未改变。

JSONL每条写入/flush/fsync后才增加journaled。ENOSPC例有一个观察到的runtime token、零个成功journal token，response分别记录generated_tokens_observed=1与emitted_tokens_journaled=0；text为空，没有将未落盘片段当完整结果。usage保持input+runtime-observed，这与修复后的声明一致；它不是FLOPs或内部未暴露的backend工作计数。资本、加载、能耗仍未计，completion继续显式说明。

无yield慢流现在在循环结束后的elapsed检查中变为policy_timeout，保留实际10s overrun和soft deadline说明。此修复是**结果分类**，不表示1s内终止GPU。计时仍从模板/编码后进入生成侧开始，load、排队、tokenize与客户端总时延须另计；真正硬取消、撤销进程和晚响应不得当工具命令的client接线仍待验证。

## 审查期间的live-load窗口与v2b复验

v2a中在verify_model_files返回前，由fixture模拟owner将task_stop设为撤销；stdin为空，没有请求生成。服务仍到Fake load/ready，usage0。这是原v1曾静态注明窗口的0GPU实证，不是实际owner被修改或真实加载。

v2b在重哈希之后、模型load之前分别重读admission（当前service117、128行）。相同fixture在全新目录复验：ValueError、Fake load count=0、无ready/response，正常及其他故障分支保持正确。v2a/v2b的唯一差异是这两次readonly检查；旧版源码按原service-intent中的64d6 SHA核对保存。

该闭合仅覆盖本次撤销点。重复读取不等于原子lease，无法声称owner在任意瞬间撤销后已开始的MLX load/forward立即停止；未来client/owner监督仍须定义这个语义。fixture撤销case中control字节变化由测试模拟owner主动写入，**不是service写budget state**；真实control在本轮负向检查中保持不变。

## 真实负向检查与实际命令

真实main仍使用不存在的review-negative-no-reservation及不可读取的模型receipt路径；它在owner gate抛ValueError，模型receipt尚未读取、MLX import不可达，真实对应operation目录不存在，真实control前后字节相同。没有写真实.paper或造真实reservation。

实际执行：

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python -B /tmp/local-model-service-repair-review/probe.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python -B /tmp/local-model-service-repair-review-v2b/probe.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python -B /tmp/local-model-service-repair-review-v2b/negative.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest tests/test_inspection_llm_protocol.py tests/test_inspection_model_admission.py -q -p no:cacheprovider --basetemp /tmp/local-model-service-repair-review-v2b/pytest
```

两个阶段Fake probe及最终negative均退出0；最新unit为28 passed in .02s。所有MLX模块在隔离进程sys.modules中替换为标记的纯PythonFake，load只返回object，stream只yield手写记录；未真实import MLX、读取权重或执行GPU forward。源码检查、registry/receipt小文件读取和纯函数digest均低负载，没有新physics。

## 证据与版本绑定

附属 `local-model-wire-service-repair-review.v1.evidence/` 保存v2a/v2b分开的probe/JSON、各自service-intent与服务源码、negative/单元日志、owner digest摘录、registry元数据核对及protocol副本。binding.json SHA=`9cffa6c909ee1bc5609fc330a364a9f0d7eadcc26ae37d3761810ce1e2832641`。未来重现旧版须在临时模块/包中选择相应源码，不能用新服务重跑后回填旧行为，也不能覆盖真实state或模型目录。

| 文件/版本 | SHA-256 |
| --- | --- |
| eval/inspection_model_service.py，v2a已测快照 | `64d6e196fc4e4a13246fd76499cbc52d5418f701385c168657bc7121e6a4baef` |
| eval/inspection_model_service.py，最新v2b | `cfbb8e908677d439c56c31ebcc3ca196a72ccecd3d7d6829b07d6aead6c11810` |
| eval/inspection_llm_protocol.py | `c323f708fd2c2eb54a440758bd162785f3ee2fcebd95b9bca21eaf7ed6ef44d4` |
| tests/test_inspection_model_admission.py | `3552c9bec37b38f46c6ba03a8e17ab2d9b818e96e79dd8450a511d0038387233` |
| tests/test_inspection_llm_protocol.py | `20424f56767596406e0b9b2f14d027a528df7fb8af1fca8741448020004a733d` |
| runtime/autoresearch-state-reader-v1/lib/runtime.py，仅读digest函数 | `5f887d7b188203a3beda96127b055d4e5009017eced0533f55afacc2a3c19f52` |
| planning/local-model-registry.v1.json，未接受的候选 | `6d8547cb0597113d5411a771e1c75f807e43d769ecdfe4d92d32d41633e0fda6` |

runtime/planning路径相对 `paper/research/20261104-inspection/`，其余相对仓库。HEAD仍为 `270c6f0946b3be59527b2b77982f75a70da7ef77`。本轮关闭的是给定源码下原五类反例及已测live-load撤销点；真实accepted-plan正例、外层硬deadline、硬件时延、LLM episode/方法比较和G4均未因此完成。
