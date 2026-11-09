# M2 输出诊断：初步证据与解释边界

本记录是开发期诊断，不改变冻结 registry、模型文件、生成参数、prompt、解析器或在途 v4 pilot；不是 G4 或正式结果接受。

已闭合的 `dev-L1-01-B4-M2-2` 实际 service 用量为27 calls/177098 tokens。其 terminal 的29次 decision attempts 中，25条 causal joins 为 `policy_format_failure`、2条为 `decoded_not_tool_acceptance`、2条为未完成或过晚而忽略。解码成功只表示线格式接受，不表示工具接受、任务完成或得分。第一条实际 service 响应 status=completed，但混有重复 JSON/fence、非法 drone/candidate/region 编号与残缺结构；原始 service 字节已经如此，不能归为下游解析器制造的内容。该请求输入2288 tokens、seed4173772581、输出上限1024，temperature0.2/top_p0.95/enable_thinking=false。

同一 pinned M2 在先前独立短文本 smoke 中正确返回 `{"commands":[],"note":"ready"}`，实际1 call/51 tokens、9个生成 tokens。该 smoke 只证明当时短提示生成链工作，不能替实际任务上下文资格。另一个已闭合 M3 槽 `dev-L1-01-B3-M3-2` 有18条可解码响应及1条 deadline 后忽略；因此不能把 M2 的异常扩为所有模型共同的格式链故障。

本地配置为 `mlx-community/GLM-4.7-Flash-4bit`，revision `1454cffb1a21737e162f508e5bc70be9def89276`，registry SHA256 `6d8547cb0597113d5411a771e1c75f807e43d769ecdfe4d92d32d41633e0fda6`，download receipt SHA256 `bdc6feb2770ca83b02426f9315e422c9a251ff378c351d3b3f29b6c77fd73ce4`。当前 MLX-LM0.32.2/MLX0.32.4。源码定向读取确认 service 使用 pinned 本地 tokenizer/template、显式关闭 thinking、直接编码 template 后送 stream_generate；generation_config 的 EOS 列表会由 MLX-LM load 读取。本轮没有导入 tokenizer/model 或新增推理，也没有做权重重 hash。

2026-11-06 UTC 读取的上游 [MLX-LM issue777](https://github.com/ml-explore/mlx-lm/issues/777) 记录了此模型 MLX 4-bit 的循环退化；维护者在2026-01-22将其部分归为8-bit修复、部分归为4-bit的预期行为。该历史讨论提供量化/实现敏感性的线索，未证明本次根因。[issue877](https://github.com/ml-explore/mlx-lm/issues/877) 的维护者说明 thinking 起始标签属于 chat template；当前 pinned template 同样在 enable_thinking=false 时以 `</think>` 开始 assistant completion，不能凭缺少可见 `<think>` 就认定模板坏了。[原模型卡](https://huggingface.co/zai-org/GLM-4.7-Flash/raw/main/README.md) 的默认采样建议不同于本研究冻结的受限部署参数；当前测量解释必须指向实际部署组合。

PR781 的完整 diff 已通过 GitHub patch endpoint只读获取并定向比较；本地实现与该历史分支结构不同，不能宣称该 PR已合入或历史修复在当前后端已被独立验证。GitHub API 请求遇到403 rate limit，未重复请求。没有将外部讨论当作新的运行回执。

首 root 的18个 callback已于 UTC2026-11-06T20:44:18.795973Z 在order18前完整闭合并确认暂停。绑定 `qualification/pilot-v4-first-root.v1/binding.json` SHA256 `34ea02afc84e4cdff84adcf420c2f10ad9cb06bc39fc4c286ba9b51341d627f0`。同一 M2 的 B3 三重复合计有83条可解码、5条格式失败、1条 deadline后忽略；B4 三重复则为2条可解码、70条格式失败、3条因deadline/非完成而忽略（最后一类包含未启动请求，不能全算实际生成）。这说明失败具有部署/任务架构依赖，不能称整个 M2 后端无法生成合法响应，也不能将 B4 失败重分类为基础设施重试。

当前未找到足以支持直接修改实现的确定错误；保留全部失败与八配置分母，继续原冻结pilot。后续如有可复现的实现修复证据，须另存旧失败、重新冻结受影响来源并补资格验证。当前证据不能区分模型本身、4-bit量化、数值实现及任务提示的各自贡献，不能写成 GLM 整体能力结论。原始closed请求/响应与短smoke的字节绑定另见 `qualification/local-model-output-diagnostic.v1/binding.json`。
