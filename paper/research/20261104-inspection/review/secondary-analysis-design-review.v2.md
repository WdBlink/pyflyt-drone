# Secondary分析设计 v2：有限复核

结论：D2、D3在设计层面已补齐；D1双路径/trace不可识别的方向也已补齐，但**同刻phase文字尚有一处P2残余，需更正后再冻结接受**。不是实现审查，不授G4/main或其他整阶段、人类认证批准。审查者 `/root/statistics_design_review`，独立上下文、同模型家族。

Subject `planning/secondary-analysis-consumer.v2.md` SHA256 `d52a326840a5ebc023629128a986555a7221816d2846cd5a024f2e9fc8ec367a`。v1正文除标题外保持原样，新增末节前瞻决定；原19义务、固定216/108/180、root配对单位、主global6均未在该delta中改变。旧报告保留，本次不重审全graph或既有统计原文。

## 已闭合的设计项

D2明确只保留原表、完整分母、eligible/各NA、identified witness、未响应/未恢复、H及U/H/indicator；pre_event_threshold_met不冒称全部未来需求已完成；NA、policy右删失、trace不可识别分开。observed-only conditional distribution不用于概括总体，不筛双方成功，不新增τ*/KM/log-rank/p。静态start/confirmed_at与动态10原格/确认见证明确，符合原有限描述角色。实际实现仍须证明这些规则没有被汇总器遗漏。

D3已具体固定NumPy1.26.4、99999、linear、PCG64与SeedSequence([domain_seed,3,salt])，ID/OOD seeds为2026110401/2026110402，clean/ablation/latency salts为101/102/103；每域canonical6parent一次生成(99999,6)int64索引矩阵，所有方法/arm/分量共享，batch仅切片，保存little-endian int64字节SHA。该规格可实施，且避免与既有SR salt1混用；不要求本轮生成矩阵。使用独立计算流不等于证明真实root独立，所有区间仍仅nominal descriptive。

8端点scalar投影+full-score SHA、完整bins留在原SDK/world并按episode/ZIP成员绑定的方案合理，可避免将明细膨胀为单个超限JSON。实施须从已admit的full score核出这8值并绑定所引用的原始文件/路径和字节，不能让自报full-score SHA取代真实score/receipt核验；原始全分母与所有bin仍需可达。这是输出布局建议，不是已经验证过的实现。

## D1-phase：精确残余与最小更正

v2写“mission._events→command/job→executor.advance…”。实际读取的worker `eval/run_inspection_reference.py:71–83` 是先drain最多16个commands并调用mission.tool，再调用mission.step。后者 `envs/inspection/mission.py:308–316` 才执行首部_events→advance/targets→flight.step→尾部_events→条件相机帧/local_measurement/deliver/refresh。不能把上一轮尾部事件与本轮首部事件压成一条错误序列，或由相同浮点t猜先后。

建议直接替换为下面的准确描述（不是新调度或新测量定义）：

> 每次worker控制迭代起点t：commands/job(t) → mission.step内head_events(t) → executor.advance/targets(t) → controller/setpoint(t) → world.step(t到t+Δcontrol，含实际physics hooks) → tail_events(t+Δcontrol) → 条件camera-frame/local_measurement/deliver/refresh(t+Δcontrol)；随后worker才可能发布状态。重复_events是实际调用顺序，不应省略成“命令之前总先调用_events”。若上一轮刚跨到事件边界，尾部事件以及该边界相机帧可以早于下一轮标为同一t的commands；此后head_events通常只是幂等重查。

因此，在边界τ才被worker处理的新job，不得被倒算成已作用于**上一迭代末、同为τ的相机帧**。仅ACK、plan_response、无后续真实step、或同advance已terminal仍不算新program采用。commit与local-acquisition两路径、不可识别不冒充删失这些v2保护应全部保留。control tick和后续physics hook时间仍是不同字段，不能用10Hz pose或墙钟替换。

还须明确三种时刻的来源：

- τ_nominal：原calendar/公开规则登记的生效时刻，保留原endpoint起点；不从分析中悄悄改成actor获知时刻。
- t_apply_tick：冻结源和完整步覆盖可以证明或推断的实际应用tick/phase。`ScenarioMission._events`记录priority `public_events.t=c.priority_at`、effective_from同样为登记时间，**不是额外记录了一次self.t应用测量**；如果τ不落control边界，不可直接把两者等同。link/camera采用按时间计算的availability/quality，不能假称它们都有priority式事件mutation记录。若无法唯一推断，保留不可识别和依据。
- t_actor_known：只据实际因果消息/actor输入见证；producer public_state或updates.put本身不是actor已收到，更不是已经在模型请求中看到。缺失则明确unknown，不拿τ或新card编号自动补齐。

同刻零响应必须有合法新card/参数和完整phase证据；上述修正既不保证当前源必然产生零响应，也不要求创造这种结果。仅当所有必要来源能关联时，才能保留v2所说的“冻结单写者调用链支持的control-tick推断”；仍不称逐行job/setpoint直接记录。无需为修这段设计文字修改在途生产来源。

当前实际读取的worker SHA256 `09189b827047bf0ecca51e044db4fb3eadeffa0dda8b3e87ade91c007be29742`；mission SHA `715fce4a99325b6d26fbb1e9ffc20896603bfbd4d48396f0513a46f5ad14aec7`；scenario SHA `9159ba44790d6d315074a824e52adeebd96468ab3032c6f0f07fdb755ce2753f`。这些仅绑定本轮阶段判断，未核所有运行来源闭包或实际flight时间线。

全部实际检查为7个明确文件的定向读取/哈希及9项标准库字段/微小字面时间格比较；最后脚本 `2026-11-06T18:06:15.090646Z`–`18:06:15.091340Z`，内部耗时0.00053秒，全部通过。“通过”表示这些有限比对成立，不表示上述phase缺陷已被修复。无pytest、集中CPU、全树hash、NumPy draws、生产import、模型、物理、native操作或遗留子进程。报告/binding归档后停止全部检查与R写入。
