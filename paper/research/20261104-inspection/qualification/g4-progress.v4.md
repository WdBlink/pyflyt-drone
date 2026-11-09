# G4开发进度v4

记录于2026-11-05T21:31Z。G1–G3仅在已声明开发包线内通过，G4仍为needs_work。尚未生成或运行main/holdout，未开始论文结果正文。以下是开发结果，不能作为正式实验或G4通过记录。

固定六root×B0/B1/B2的18个slot全部完成；19次实际尝试含一次已知clock失败，保留失败原始轨迹后按既定规则重试一次。两份先前已完成的有效缓存按原规则复用，不增加root或实际飞行计数。B0/B1/B2宏平均Y依次为0.9060895302670936、0.7659332720006325、0.7381647224577993。既定B1/B2选择器选B1；B0得分更高，不能称B1为所有经典方法中的最强。独立审查见`review/classic-six-root-cohort-review.v1.md`，完整raw仍在logs中，研究根保存了source-bound v3选择摘要。

三个固定本地模型各完成一次有原生预算预约与回执的实际文本生成smoke，共3次调用、167tokens；这是生成能力探针，不是飞行/任务性能或公平速度比较。独立审查见`review/actual-three-model-smokes-review.v1.md`。原生Codex用量、硬件资本与能耗未计量，没有付费API。

首次M1-B3、dev-L3-00开发诊断完成600秒1×、144000物理步，最大时钟滞后0.023628416766086957秒。26次实际模型请求与actor记录逐条对齐，messages、seed、原始response及durable token journal均绑定；只读meter复算为26 calls/162043 tokens。22份源码及61份物理依赖pre/post绑定一致，冻结evaluator复算完整score相同。原strict wire下C=0.9939498703543648、S=0.2222222222222222、D=0、H=E=1、Y=0.40539069752552903、SR=0；完整95项源文件/回执/飞行/模型操作archive及独占usage receipt位于`qualification/local-model-generation/M1-B3-L3-00-diagnostic.v1/`。已由唯一owner正常结算；累计已知29 calls/162210 tokens，held与unknown均为空。不是全八配置pilot，不证明任务成功或总体时钟可靠性。

这次26份response中22份为整段Markdown围栏；原strict仅3份解码成功。离线候选检查只衡量格式接受，不执行命令、重跑模型或改原Y。两版候选、prefix与版本恢复历史均保留；独立审查发现raw长度可能绕过，安装时保留原raw类型/非空/24000字符限制，再仅去除一个完整json或无label围栏，继续复用全部JSON/schema/tool校验。原始response不修改，actor记录wire_framing；87项相关软件测试通过。新协议只适用于后续开发运行，不能把24份候选可解码响应当作旧飞行已执行行为。

segmented context的tokenizer-only v3已完成三个固定tokenizer×五种架构/消融投影，样本input为3939–5401tokens，未做forward。其actor SHA绑定本次首次诊断时的版本；后续安装仅改解码审计，不把这些合成上下文样本当作动态任务永不超限的证明。实际26次调用均受8192 input、1024 output、每次60秒软期限及总180000tokens等守卫约束。

仍待：完整LLM episode预算/未知用量恢复/有限retry与八条件SDK接线；新devext总体suite命名、独立seed复现、跨namespace去重、新schema实际worker与OOD飞行；clean/event兄弟、事件响应/恢复/右删失适配；固定六root全八条件pilot以及最终源码、prompt、模型、硬件与统计入口独立验收。旧dev不能重标为新总体合格；未完成门不能由纯软件测试、一次诊断或原生plan接受替代。
