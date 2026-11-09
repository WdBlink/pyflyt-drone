# Response/resource wiring v1 有限独立复核

结论：有一项 P2 阻断，当前 reader 不能处理合法多响应 episode。第一次处理 actor 响应时将原快照变量 `before` 覆盖为 float，第二个 generated response 再读原 receipt 时抛 `TypeError`；它会使 main/secondary 分析在发布前失败。单响应、known retry 用量和若干拒绝边界实际通过，不能据 48 项 green 宣称新 reader 可用于多调用数据。后续 working 修补不在本 frozen v1 结论内。

审查者 `/root/control_review`，同模型家族独立工程上下文，仅写本报告及独立 evidence，未改实现/测试/协议、旧结果、生产来源或 native state。未执行实际 formal 抽样、模型、物理或 reserve/settle。

## Subject 与复用边界

Subject `planning/formal-admission-candidate.v3/response-resource-freeze.v1/binding.json`，SHA256 `2cbc108e4e7550aa0abbb4b904f07b00d98256c561f3434497e5c21aa89b31a3`。

- reader SHA256 `0ae4487c8dcf9904afbb83d6860ae9c961ea97a745d18753e7f26eeec0114ee2`。
- main entry SHA256 `aca945d84a661b07164922eca525e207074d9bdfc2a938e516c7447c9678d52f`。
- secondary entry SHA256 `480e897a8ed634b93aae67f139c18652871e3ecb4127f684470929f79e488452`。
- REQ SHA256 `3272e19e4ec4d3ad4802ff5a939feea8cf0a258f6777ee60a5b21ef2d758c5fa`。

七个冻结成员的 SHA/size 全核对，初始读取时 working 对应字节相同。随后 root 仅在 working 修 reader 名称并加回归，本审查实际探针直接加载 immutable v1，不把有意新版本当漂移或已修复依据。完整成员表与 probe 绑定在 evidence。

main `_receipt_snapshot` 与 `scalar_score` 的原函数文本分别与已审 primary-v6 逐字节相同（独立 helper 文本 SHA 见 reused-helper-check.json）。原 SDK/main-lock/secondary proof、event v5、数值模块、计时语义沿用既有或另行审查边界；本轮不接受 whole graph 或 G4。

## P2：局部进度变量覆盖原快照，第二响应必然异常

定位 `source/eval/inspection_response_diagnostics.py`：

- `:41` 将 `receipt_snapshot` 解为 `slot, directory, before`，此时 before 是原闭合 snapshot dict。
- `:129` 每个 generated join 都调用 `_bytes(..., before, project)`，依赖 `:19` 的 `snapshot["files"]`。
- `:141` 在首个 join 内执行 `before = _seconds(received["worker_before_request"][0])`，令 before 变成 float。
- 下一 generated join 再到 `:129`，实得 `TypeError: 'float' object is not subscriptable`。这不是数据未知或坏 actor，也不是计时解释分歧，而是确定的有效多响应拒绝。

独立重现使用冻结测试里的可信证据 stub/合成 closed fixture：正常一调用基线通过；增加第二个正确 path/usage 的原 call response（各 1 call/102 tokens，总 2/204），补齐 2 requests、2 replies、2 decoded entries 和 2 个因果 join，更新原 snapshot 的文件 SHA。每个必要字段与时间序列均符合 reader 的本地检查。实际直接运行 frozen reader，在第二响应上述位置失败，完整 traceback 与合成输入已归档。没有声称该 stub 通过真实 actor/native/物理 validator；此反例只用到后续任何合法第二 generated join 必经的代码。

必要最小修复：保留不可覆盖的原 snapshot dict，另用独立变量名存 worker-before-request sim progress；相关排序和输出字段同时引用新名字。新增至少两个 generated joins，以及 not_started 后再 generated 的回归，确保第二次 `_bytes` 仍用原 snapshot。不能以重采新可信 SHA、略过第二响应或把异常/资源填零绕过。root 已告知正在修 working，但本报告不审未冻结的新版本。

## 已实际检查的资源与拒绝范围

自有 `probe_response_reader.py` 最终 exit 0，约 0.022s；四线程环境设 1、禁 pyc，纯 `/tmp` 合成文件，无执行器或资源效果。

- 单 final call + 一个 known failed attempt：valid usage 为 1 call/102 tokens，all-known 为 2/110，失败 attempt 的 service call 仍留 records；context rejection 单独计数，money/energy 保持 None。
- 三个原 repeat 聚合：valid/all-known 用量分别先按原重复均值，known_attempt_count 为 6；缺一个 repeat 或重复一个 ordinal 均拒绝，不把 retry 当额外科学重复。
- 七个独立负例：missing actor、unknown terminal、wrong actor kind、token 总量不符、actor 原 SHA 改变、missing native usage、missing causal join 均拒绝。missing usage 是 KeyError 的 fail-closed 传播，不是零用量。
- 二调用阳性应通过却抛出的 P2 已作为预期缺陷捕获，probe 整体 exit 0 表示成功复现并检查反例，不表示产品通过。

首次自有直接装配未将 repo 放到 Python path，冻结 fixture 的精简 source 中没有 inspection_recording，出现 ModuleNotFoundError。仅修自有 bootstrap（加 repo import path）后上述检查完成，保留原装配错误日志；未改候选源或掩盖首次结果。

root 告知新 focused 48 passed / 70.77s，本审查未重复该套；源码中的 reader fixture 仅有一个 final admitted response，集成 fixture 用 Fake response reader，因此未覆盖此第二响应缺陷。旧 499/78 或本 48 均不能关闭这个反例。

## 接线源读结论与未证项

1. Reader `:17`–`:23` 通过原 snapshot 中的 Artifact SHA 和同 buffer 解析读取 actor/usage/service bytes，未以 actor 自报路径或未绑定同名文件替代。`:43`–`:79` 对 condition/sampler、固定 reservation kind、known status、整数 usage 和 token 总量做 fail-closed 检查。其权威前提仍是 fresh `measure_completed_episode` 已完成 native/actor 验收；本模块没有重新认证这些前提。
2. `:110`–`:136` 消费调用方刚返回的 evidence.causal_request_joins，保留完整 requests/replies/decoded 分母、admitted call ordinal、原 response 字段与 decode disposition。计时字段的采样点解释由另审负责，本报告没有因名字相似推导精确 enqueue/adoption、source wall age 或缺失延迟。
3. main 在每个 raw value/full evidence 重验之后，使用刚保存的原 receipt snapshot 和该次 evidence 追加 responses；secondary 对每个 intervention 及每个重新验收的 locked primary comparator 同样调用，method 与诊断 row 一一连接。没有另从未绑定 actor 自报构造科学分数，原 SDK value/evidence 未改。
4. main 将逐 episode detail 放 response-records.zip，recording-index 绑定 ZIP 与每成员 SHA，并在 subject 绑定 response_resource_summary。secondary 把 response_resources 放已绑定的 diagnostic-records.zip，并同样绑定摘要。source wiring 没有新增主/holdout lock/opening 或资源执行调用；REQ 仅新增 inspection_response_diagnostics.py。
5. 摘要计算在各 entry 最后的 fresh admission/原集合快照复核之前；发布仍只复制 captured SDK 原字节，原 design/raw/value/evidence 的处理没有新 delta。读取后 bytes/集合漂移的最终防线仍是既有 `_check_snapshots` 和 main/secondary fresh proof，本轮只确认调用保留，不重新跑此前所有 SDK/closure 反例。

上述 1–5 是源级接线核对，独立实际执行限 reader/summary 及其失败反例；本轮没有另跑完整 SDK 集成、ZIP 真实数据输出或新的物理/native receipt。计时语义、event/数值统计的另审结论不由本报告替代。P2 修复后尚须新版有界复验，不能先移除“实现未获独立接受”的边界。

报告/binding 冻结后停止全部 CPU checks 与 R 写入，不授 G4/main/human，也不继续扩展。
