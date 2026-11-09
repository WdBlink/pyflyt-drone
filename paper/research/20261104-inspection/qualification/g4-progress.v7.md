# G4开发进度v7

记录于2026-11-05 UTC，完整八配置固定开发pilot已实际开始，G4仍needs_work。原六root、108内层episode（18classic/90LLM）与48outer设计和预定顺序/seed不改，B1不重选。当前完成order0 B3-M3/repeat0、order1 B3-M2/repeat2，两者都在dev-L1-00；不是两个独立root。原始callback、source/input/dependency/sampler绑定与设计前缀见`qualification/eight-condition-pilot-prefix.v1/`。provided SDK在finish/exit才写results.json，当前未伪造部分SDK结果或对未开始槽补Y0。

B3-M3完成600s/144000步，max lag=0.05607066589874421、H=E=1，C=0.9940374787052811、S=0.08333333333333333、Y=0.5386854060193073、SR=0；23calls/160190tokens。B3-M2完成600s/144000步，Y=0.49701873935264057、SR=0；30calls/152462tokens。原始低分保留，均不按策略失败重试。两份完整archive/usage receipt已由唯一owner正常结算，累计已知82calls/474862tokens，held与unknown为空；资本/能耗/native host仍未计量，无付费API。

已实际收到23:27:23Z的fresh集间ack，order2 B3-M1/repeat2尚未reserve或spawn；当前只等待内部检查点，不暂停任何飞行/活模型。`checkpoint.request`将由本次请求的owner在apply/begin完成且相关检查结束后撤除。模型/worker/reader/evaluator源保持本轮冻结；`review/eight-condition-checkpoint-boundary-review.v1.md`和最终`review/llm-episode-native-accounting-clock-guard-review.v1.md`已关闭各自限定软件问题，不授予G4或main许可。

新增prospective ID-L3-B0参考资格完成，只有一次尝试，600s/144000步、max lag=0.02665958368839938、H=E=1；C=0.9701739850869926、S=1、D=0.29493087557603687、Y=0.7550349535543431、SR=0。17源/61依赖/输入绑定，冻结evaluator完整score相同，archive见`qualification/population-realtime-ID-L3-B0.v1/`。结合此前OOD-L3 trace，已有两种新场景族的完整L3参考链路；其他新strata仍仅有明确范围的生成/几何/reader检查，不能扩称均已实际飞行。

事件scope/33手算/实际n1 B0配对已完成`review/event-endpoint-scope-and-pair-review.v1.md`限定复核。重要边界保留：当前clean派生将公开level从L3改为L2；B0源码不读该标签，因此现有B0配对仅作限定开发验证，不能扩成所有LLM条件的纯事件因果资格。正式clean干预须保留父根初始公开任务层级并显式切换事件，源/reader需在G4最终清单前明确验收；现有全部pair及已知clock失败仍保留，不回改。

source-only公共任务说明充分性审查已追加，用来核对模型可见task/tool说明与经典baseline使用的公开协议一致，不依据当前得分添加优势性提示；必要公开输入/输出语义、应保留隐藏的truth/正确性反馈与实现状态字段分别核验。此检查尚无结论，执行源暂不改。

同时在literature阶段补充原始来源检索及版本化获取；Retriever v2、网络/会话UAV benchmark等限制了泛化的异步/因果接口首创性。原始全文/metadata及转换记录保存为阅读材料，不等于已读完/已核所有主张，也不是论文正文。未生成main/holdout，未作正式模型比较或开始结果写作。
