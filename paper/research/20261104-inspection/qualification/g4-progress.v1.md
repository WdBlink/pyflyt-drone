# G4 开发进度（2026-11-06）

G4仍为needs_work。没有生成/开封真实main或holdout，没有正式八条件成绩、模型生成调用或论文结果文字。G1–G3原资格证据和失败记录保留，新增adapter不自动取得原范围以外的资格。

## 新输入与实际开发执行

`dev-cases.v1/`首先冻结24个dev root，L1/L2/L3各8，首轮pilot事先选定每级00/01。每root单独抽取ROI、实体数量/布局/运动、风和事件日历；private CaseSpec、种子、隐藏实体及未来日历不传policy。L1的D为null、Q=(C+S)/2，复用原static质量器；L2/L3沿用原完整动态格评分。

独立输入审查复现五项缺口：重叠动态代理、离散轨迹检查混叠、非原生数字/ID、共线但不相交边误拒及worker读入文件未绑定初始SHA。原版保存在`source/scenario-adapter-v1/`。修补后601采样加当前解析轨迹速度上界，分别给ROI中心边界、building2m及proxy3m的连续余量，非任意曲线证明。dev radius60–85m、heading±.12、bend0；未来OOD参数不能直接套此通过。

`dev-cases.v2/`对原24根重新验收，23个输入字节相同；`dev-L2-03`旧动态/静态proxy在445s约0.96m，仅此几何不合格输入用同seed生成器重建，原输入保留。该root此前没跑策略，重建不使用分数，原6个pilot不变。v1/v2保持同lineage，不是48个独立root。

| 实际开发任务 | 版本/物理证据 | 分量与边界 |
| --- | --- | --- |
| L1-00/B0 | `dev-diagnostic.v1/L1-00-full`，600s、144000步，source/deps unchanged | C=.994037479、S=1、D=null、Y=.997018739、H/E/SR=1；max lag=.053401750s；maxXY4.323529m、max稳态Z1.020409m；旧guard版本诊断，未倒称为新receipt flight |
| L3-00/B0 | `dev-receipts.v2/dev-L3-00-B0-0/attempt-00/flight`，600s、144000步，1attempt | C=.984442524、S=1、D=.277945619、Y=.754129381、H/E=1、SR=0；max lag=.030502833s；实际receipt terminal SHA `6d3bdd5e4abbaedf298f9f6eab9d68636043dae32b86397155191e55a1bfdbd6` |

L3源闭包另存`source/actual-dev-L3-v2/`；实际重放`dev-receipts.v2/L3-00-replay.v1.json`对105次RPC响应和25组完整业务/跟踪产物精确一致。重放不增加独立样本或策略重复，也不构成完整240Hz位姿状态重建。原max/p95与里程的保存边界仍按GATE_STATUS说明。

L3同plan重调CLI实际返回cache_reused=true、actual_attempts=1，没有再飞。随后新guard改变adapter/runner SHA，旧receipt和旧source快照保持；新CLI不能静默把旧plan身份改成新版本。B1现已在同根L3完成一次真实600s：C=.981849611、S=.888888889、D=.151057402、Y=.673931967、H/E=1、SR=0、maxlag=.050169500s，1attempt。源存actual-dev-L3-B1-v3；这不是完整开发比较，也未作独立方法效果验收。B2仍待真实运行，B_classic*未选。

## 记录与统计适配

`eval/inspection_episode_receipts.py`仅支持本地B0/B1/B2，提供exclusive/fsync intent、spawn观察、终态绑定、并发锁及最多2attempt。Y=0是正常策略成绩；unknown阻止自动重跑/补零；已知源/输入/评分缺陷不准作免费基础设施重试。根目录及attempt目录父级也显式fsync。实际process故障检查不等于物理掉电认证，不支持付费request恢复。

独立反例先发现缓存链未绑定、start写失败遗留进程、缺文件遮盖已知源错误；修补后又发现missing-run遮盖已有copy/world case SHA错误。全部原源码/反例/报告保留，最新两个guard已通过16项测试，独立最终复验已关闭原P2，见review/g4-receipt-sr-guard-review.v1.md。首次真实L3 receipt使用先前已绑定版本，并无上述错误输入；不回写历史结果。

八条件SDK矩阵仍使用`inspection_recording.py`及autoresearch提供SDK。现有真实flight receipt还未完成全部八条件SDK实际接线，合成1296+216矩阵不冒称1512条已飞数据。`inspection_secondary_statistics.py`补SR root内平均及按域/等级宏平均的root-cluster名义percentile区间；8项定向及独立完整合成分母复验通过，不将重复flatten成Wilson样本，不新增确认性检验。主Y仍是`statistical-protocol.ral.v1.md`的有界root推断/globalHolm6，bootstrap仅名义或诊断。

`inspection_common_calendar.py`对保存L1/L3报告账重新构建并逐项复现原分数，另提供纯离线同需求日历重评分；真实clean兄弟尚未运行。独立发现其demand-calendar入口校验缺口后已补validate_case；手算与非法日历4项测试通过，最新终态组合缺失guard及此入口已在review/scenario-final-residual-guard-review.v1.md独立复核关闭。不能称事件差值/恢复分析已完成。右删失/无需改派判据和实际loader仍待。

## 本机模型可行性

独立`.venv-inspection-llm`，不改已资格的physics环境。三个候选权重均按公开repo固定revision下载并核对全部Git/LFS blob：Qwen3.8-27B-4bit `10c35caafbb80f7dc6a7a432cdd11af10a6d4818`；GLM-4.7-Flash-4bit `1454cffb1a21737e162f508e5bc70be9def89276`；Mistral-Small-24B-Instruct-2501-4bit `51ed0c6d2f98a25d8a60f19994f0e86c944995e0`。证据在`local-model-preflight/`和`planning/model-sources/`；下载权重在gitignored models/。

仅load/tokenize，无forward/generation：GLM4.920s/峰值16.855GB、Qwen2.931s/15.134GB、Mistral2.821s/13.260GB。Mistral原load提示regex问题；现按官方2501固定revision9527884的TekkenV7 pattern核对，fix_mistral_regex=True后模式逐字一致、4条分词往返正确、权重/tokenizer原文件字节未变；修复后load3.313s，配置单独记录。这仍不表示模型策略已合格。加载时间与内存不代表推理延迟或策略能力。三者尚未被正式选为M1/M2/M3；还需统一配置、prompt/token预算、实际资源/时钟竞争及pilot。没有付费推理、采购算力或投稿；API调用费为0只因无API推理，资本/能耗未计量，不称计算免费。

## 下一步硬条件

先关闭上述最后guard与分析入口问题并独立复验；完成预选6根B0/B1/B2同协议开发比较/只从B1/B2选择classic；补全初始科研计划与G4执行冻结的阶段边界，再通过唯一controller登记任何模型生成资源；完成LLM适配/实际八条件记录、pilot、事件兄弟/SR/删失、模型/预算/重试、ID/OOD/holdout生成规则与版本。G4正式冻结后才建立main/holdout实例并运行正式实验。19项原义务和八配置均未减少。
