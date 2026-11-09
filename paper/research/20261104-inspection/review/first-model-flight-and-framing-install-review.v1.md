# 首次模型飞行与 framing 安装独立复核

审查者 `/root/control_review`；同模型家族、独立工程上下文。只读原诊断归档/结算及新 framing 安装；未运行 MLX forward、新物理、模型重试或 owner 操作。

## 结论

**本轮未发现原诊断归档、26 次请求/持久用量或当前 framing 安装的实质矛盾。** 原 strict c323 诊断确有完整600s/144000步记录，26条 actor请求与26份持久call一一对应；用量独立合计26 calls/162043 tokens，原 score 重算仍为 **Y=0.40539069752552903、SR=0**。原生 ledger 已知总量为29/162210、held为0、unknown为空。

当前11a4协议/4904 actor实际保留 raw类型/非空/24,000字符限制，再 whole-fence unwrap，再原 strict checker；原回复及 token journal 不被替换。19项安装解析检查、strict AST等价及raw response不变检查通过。新接口是后续 development policy 版本，**不改变原 run 的22次格式失败、晚回复或Y，不能据离线可解析数推导反事实成绩。**

这是一例旧 dev-L3-00 的 B3-M1 共存诊断，非完整G4、八条件/六根pilot、前瞻新总体/物理OOD、main/holdout、真实vision或人类领域认证。

## 原归档、来源与请求链

输入 [complete-operation.zip](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/qualification/local-model-generation/M1-B3-L3-00-diagnostic.v1/complete-operation.zip) SHA `ed97d8c1…`、[usage-receipt.json](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/qualification/local-model-generation/M1-B3-L3-00-diagnostic.v1/usage-receipt.json:1) SHA `21843dd0…` 均实核匹配。ZIP含95个payload加其archive-manifest，共96个条目；manifest集合、无重复项及全部95项SHA逐一验证。terminal四文件映射（run/world/private-case/actor）也全部匹配，没有只凭returncode=0接受诊断。

run保存的22份源文本与其source_hashes逐一重算一致；当前61份外部PyFlyt源码/资产及PyBullet依赖文件与原fingerprints一致。原 actor=`b22b93af…`、protocol=`c323f708…`、runner=`859c5eda…`、service=`64e8307e…`，不是当前新framing源。后续正常源码版本变化不被回填为原attempt启动源。

对全部26个call独立核验：

- request ordinal恰为1..26；sampler seed恰为base `2026110610` 加ordinal−1模2^32；actor messages与持久request.json逐字段相同，messages_sha256按原canonical规则匹配。
- actor response去除client追加的两项parent timing/ordinal后，与相应持久response.json完全相同；transport stdout为1个ready+26个response，26个response也与durable记录完全相同。
- 原c323 strict checker离线复验26段原文本：3段接受、23段拒绝；除最后一条按真实600s截止忽略，原decode/实际enqueued命令与旧checker逐项一致。原actor记录为22次policy_format_failure、3次decoded_not_tool_acceptance、1次ignored_after_episode_deadline。
- 共12条命令入队；world实际记录12次工具处理（10 accepted、2 rejected）。decoder接受、入队、worker ACK、作业成功及评分仍分开，不把12条入队当12项交付。

service intent绑定本次development reservation、原owner frozen plan、M1下载receipt和原service源；profile为32calls/180000tokens、input8192/output1024、soft60s、同一固定temperature/top_p。该检查核对保存关联，不重新读取数十GB权重或作硬件/进程来源认证。

## 用量及已有原生结算

全部call的request/response/tokens SHA均与receipt中的逐call指针匹配。每行token journal的runtime_prompt_tokens等于该call实际提交input，runtime_generation_tokens连续1..N；拼接journal text精确恢复response.text。独立合计：

| 项目 | 复算结果 |
| --- | ---: |
| 持久call数 / actor请求数 | 26 / 26 |
| submitted input tokens | 153503 |
| runtime-observed generated tokens | 8540 |
| durable journal token行 | 8540 |
| local metered total tokens | 162043 |
| 单call input范围 | 2032..6483 |
| 单call generated范围 | 73..1024 |

最后一条虽在flight截止后被忽略，其实际调用和生成用量仍完整计入。没有漏掉晚回包、以解析失败冲销资源或把unused reservation当已使用。

只读实际owner的四个ledger entry及对应receipt SHA均通过：本次26/162043，加三条smoke的1/62、1/51、1/54，独立加总 **29calls/162210tokens**；原生 `budget.summary` 同样给出该known usage，reserved calls/tokens均0，unresolved/unknown均空。entry中的历史reserved上限仍保留，不误算为当前held。cost在四条均unreported，不能把本机资本/能耗或Codex原生用量写成已测零；本轮没有执行或修改settlement。

## 原评分、物理计数与时钟

从归档提取原world至自有临时目录，调用已冻结evaluator.v2（`de826d24…`）作纯CPU原score重建，返回0且完整score匹配。原world SHA为 `3ba6bd03…`；复算使用其已有报告/覆盖/测量记录，未把新framing解码出的命令加入历史。

| 原分量 | 值 |
| --- | ---: |
| C / S / D | .9939498703543648 / .2222222222222222 / 0 |
| Q / Y | .40539069752552903 / .40539069752552903 |
| H / E / SR | 1 / 1 / 0 |
| static / dynamic完整权重分母 | 9 / 331 |

world实际sim600s、physics144000、flight audits144000、tracking samples144000。独立解压并核raw240 SHA、144000×17形状、全部有限值及从1/240至600的完整240Hz时格；重计maxXY=4.597510997668039m、XY>10m步数0、Z区间breach步数0，与保存tracking一致。硬事故summary为空，保存min四机间距52.539080m、四机exit dwell最小282.616667s、remaining range最小7905.579782m；这些保留其原逐物理步hook来源，不从稀疏位置重造里程/完整硬状态。

26次模型等待的before/after记录中，仿真时间与位置均实际变化：sim增量范围5.741667..45.991667s，service wall5.622995..45.885994s。最后一条等待跨过600s，其flight推进截于deadline，回复未派发命令；不能要求它的Δsim等于完整晚结束的service wall。

原world wall600.004083s；保存maxlag=.023628416766s、p95=.004664000077s、over.1 ticks=0，并有529.358333s最大见证。600条1Hz progress可独立复算的max(wall_elapsed−sim)=.011788958102s，支持连续1×推进；**完整72000个control-tick lag数组未保存，本轮不能离线精确重算全数组max/p95。** raw error trace也不是完整240Hz位姿/速度/里程重放；score的E重计仍使用已保存的range/exit审计。

world遗留 `non_llm` role字符串由run的worker_role_metadata明确解释；本次policy与26条实际模型/命令链证明这是B3-M1诊断，不因旧label把它误称无LLM，也不因模型已参与而称完整G4。

## 已安装 framing 的定向复核

安装源与 [wire-json-fence-installed.v1/binding.json](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/qualification/wire-json-fence-installed.v1/binding.json:1) 的四份snapshot/current文件SHA一致：protocol=`11a4e863…`、actor=`49047b98…`及两份tests。候选独立报告 [llm-whole-fence-normalizer-review.v1.md:34](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/review/llm-whole-fence-normalizer-review.v1.md:34) 指出的raw guard要求已实际安装。

具体 [parse_reply:55](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/eval/inspection_llm_protocol.py:55) 先检查原text的str/非空/≤24000，再在第58行unwrap。将唯一post-guard unwrap赋值从新函数AST去掉后，函数剩余AST与归档c323完全相同，原strict JSON/schema没有放宽、修参数、补字段或改工具语义。

独立19项纯解析例子符合预期：raw JSON、裸/json/CRLF整层fence、外部空白及JSON字符串内U+2028/backticks保留；prose前后缀、多块、嵌套、python标识、inline opening、duplicate key、NaN、未知tool、缺字段、多JSON均拒绝。raw24012及外部24000空格两反例都由原始长度guard拒绝，而非先缩短后接受。

[actor.finish_request:224](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/eval/inspection_llm_actor.py:224) 增加wire_framing识别审计；原response字典在实际纯CPU调用前后完全相同，reply中的U+2028/嵌入backticks也未改。该flag须与decode status合读，不表示strict通过、命令被接受或交付成功。所有B3/B4/消融共用同一个parse函数，没有按模型选择性提取。

runner原本先记录actual_model_response，再传新字典给actor；service持久raw response和token journal不经此wrapper改写。原archive/receipt的SHA在复核尾仍是ed97/2184，原strict Y不变。本轮没有用新parser重放命令、重跑模型或给旧失败补分。安装记录的87tests是主Agent执行数；本 Agent实际独立运行上述19项及AST/raw/audit检查，不冒称独立重跑87项。

## 冻结绑定与交付

证据目录 [first-model-flight-and-framing-install-review.v1.evidence](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/review/first-model-flight-and-framing-install-review.v1.evidence/binding.json:1) 包含原归档recount脚本、逐call/transport/ledger/score/clock/安装检查结果及strict/installed源码快照，逐文件SHA绑定。没有复制或覆盖原诊断world/usage。临时原件保留于 `/var/folders/n_/2cw9gf5n52z_w6kllyb44ttw0000gn/T/inspection-first-model-flight-review-2bkooj8h/`。

| 对象 | SHA-256 |
| --- | --- |
| 原完整operation ZIP | `ed97d8c1ee690cd6f0043ad607f049cd2673c4e8e0afebd53ff19cef45b95900` |
| 原usage receipt | `21843dd00b3f6d3f8d9be153f3a28dd3953b75840dd660a5cbb270f59e30f4b1` |
| 原world | `3ba6bd03fe9561adc1b4f638b9d8de3e5560796f797d95e31210a79202b7092f` |
| 原strict protocol | `c323f708fd2c2eb54a440758bd162785f3ee2fcebd95b9bca21eaf7ed6ef44d4` |
| 当前installed protocol | `11a4e86348dad6fa94849f20a1086c21c24a09307d3e9f9c4fd8d642c14bcd91` |
| 当前installed actor | `49047b98d1f6f9c4d09c28f65e66f431e86a8275195a1f017fd22ffe4cf2204d` |
| 候选独立报告 | `f8765cbbd0beb21c46626e39a2dfe46bf0959b8bdcdd27ecb811f8ba0ed3ee9d` |
| 本次evidence binding | `002ad888c2e21ffea5bfe2c56620d0bf7fb964bc5e5c59ddb746828b42f62091` |

写R前已通知根Agent，等待其round20 apply/round21恢复写入通知后才新增本报告/evidence。未修改任何源、owner、live日志、旧报告或原诊断；所有执行仅只读/纯CPU，无新forward或物理。真实新population worker接线及G4其余门仍独立pending，本报告不扩大其资格范围。
