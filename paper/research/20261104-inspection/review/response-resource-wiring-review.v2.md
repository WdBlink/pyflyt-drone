# Response/resource wiring v2 修复有限复验

结论：v1 的 P2（进度 float 覆盖原闭合 snapshot，第二响应异常）已关闭，本轮小 delta 未发现新阻断。只关闭此具体 reader 缺陷；main/secondary 接线、计时/事件/数值、真实 receipt、whole graph、G4/main/human 的既有边界不随本结论扩大。

审查者 `/root/control_review`，同模型家族独立工程上下文。未改实现、测试、协议、旧证据、生产来源或 native state。所有执行为 `/tmp` 合成 closed 文件与 trusted-evidence stub，没有实际 formal RNG/model/physics/资源动作。

## Subject 与精确差异

Subject `planning/formal-admission-candidate.v3/response-resource-freeze.v2/binding.json`，SHA256 `8441e871ceb1c2b95602eaa45d1b430a6b0bcb5dceeec03dbb360374c13356e0`。

reader SHA256 `d4cf1af0aef3a085ceffe2014cfd088e2ba836b4d80c53b6c6c3793f622b7a8f`；测试 SHA256 `a8f7df595ca6dee0a9d5ea4df53982f9ac709f9b2b0e999d9317d23ff162331c`。七个冻结成员 SHA/size 全部核对，初始 working 对应成员与冻结字节相同。

v1→v2 精确 diff 仅将 reader 的 before/reply/after 局部进度名改为 request_progress/reply_progress/response_progress，并同步其排序比较和输出引用；另加两项 multiresponse 回归。`before` 在 `inspection_response_diagnostics.py:41` 解出后保持原 snapshot dict，`:129` 后续 `_bytes` 因此仍使用原闭合 SHA。main/secondary consumer、REQ 及其集成测试均保持 v1 原字节；计时数值表达式与输出字段名没有变化。完整七 SHA 与 patch 见 evidence。

原 v1 报告 `response-resource-wiring-review.v1.md` SHA256 `c012a12f23416cb1657a29e1cc1ba8e9e6033512a93b804e7c609c70dfb12587`、原失败 trace 和输入均保留，没有回填旧结果。

## 原反例与正反检查

自有 `probe_v2_repair.py` 最终实际 exit 0，约 0.057s；使用 `.venv/bin/python`，线程限制设 1、关闭 pyc，直接加载 immutable v2 reader。

- **原 v1 双调用原输入复放**：旧 two-call-inputs.json SHA256 `3048bc7a7b5a961ed8e1d6aee7c78bf642b66f55016cab1998f7ce2d2c899fbd`。将旧/新 PROJECT 绝对路径统一为占位符后，对 slot/evidence/actor/original calls/native usage/snapshot 文件数做递归精确比较，相同。新版本正确返回 2 calls/204 tokens、两个 valid-flight responses、service intervals [5,5]，不再出现原 trace 的 float 下标异常。
- **not_started→generated**：两个 request/join、一个实际 admitted call，正确返回 1 call/102 tokens；declined service interval 保持 None、后一个保持 5，non_generation_responses 为 1，没有填零或伪造两次模型调用。
- **原数据守恒**：双调用前后原 snapshot dict、actor bytes、evidence 深度内容相同；declined-first 的 snapshot 也不变。没有重新采纳晚读新 SHA。
- **单 call / known retry 正控**：valid usage 1/102，all-known 2/110；完整三个 repeat 的原均值和 known_attempt_count=6 正确，缺/重 repeat 仍拒绝。money/energy 的 None 沿用原语义。
- **旧 SHA 与拒绝边界**：改原 call response bytes 后 `_bytes` 仍拒绝 `closed response/resource bytes changed after admission`；missing actor、unknown terminal、wrong kind、tokens 不一致、actor SHA 改变、missing usage、missing causal join 七项仍拒绝。missing usage 的 KeyError 是 fail-closed，未补零。

这些是 synthetic closed 输入及原可信-evidence 接口 stub，不是实际 native settlement 或 actor validator 全链验收。它们精确覆盖原程序缺陷和相邻守卫，不认证计时解释，也没有实际 model/physics。

## 复用与未证范围

root 已告知全候选 522 passed / 137.58s、compileall exit 0，本审查没有重跑或用该 author 结果替代原反例复放。全 SDK、主/secondary ZIP 实际数据、事件及数值模块和其他流水线均未重复运行。未发生新的 reader 分母、source-ref/main-lock 权限或计时语义变更；这些仍限 v1 源级接线审查及其他独立评审的各自范围。

本次关闭原 P2 不等同完整 response/resource 工程资格、真实 formal 数据完整性或 G4/main/human 认证。全部 CPU checks 已结束；报告与 binding 冻结后停止 R 写入，不扩展任务。
