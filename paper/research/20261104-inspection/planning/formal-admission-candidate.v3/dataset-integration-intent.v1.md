# 下一组件：namespace writer / dataset proof 与实际读取

这是round32的实现约束记录，不是G4接受、正式抽样或新scheduler。共享population law/预随机组件已有限独立复验（`review/population-shared-law-component-review.v1.md`），仍未安装。当前没有实际main/holdout seeds、顺序或实例。

1. Namespace writer只接受canonical phase路径；先验证当前source-bound native G4，holdout还须先验证同一次opening和main-primary-lock。之后才可构造fresh128bit master/SeedSequence和32bit schedule/sampler seeds；完整intent在geometry前exclusive+fsync持久化，并停在待native finding绑定。existing namespace/failed或不完整输出不自动换目录、reseed或覆盖。holdout目录可能已仅有合法opening，不能因此误判为新空namespace，也不能绕过固定opening。
2. 必须补齐main→holdout跨namespace碰撞检查：现有namespace v3完整保护G4/祖先撤销/当前source/dev seeds，但还未将main master/root/sensor seeds加入holdout排除。新检查在实际holdout RNG/geometry之前；旧v3范围与证据保留，不静默扩称其已经完成全部cross-namespace guard。
3. Geometry writer复用已审shared joint law，不改五子流或1000 whole-root候选/拒绝原因；使用已native绑定namespace中的精确root seeds和固定metadata顺序。每root原始audit/case只写一次；耗尽、碰撞或部分写入保留并阻断，不换root、不按方法分数筛选。实际操作前后重核source/namespace，无法确认时不交付可执行manifest。
4. Dataset manifest固定到`datasets/<phase>.v1/manifest.json`，绑定G4、namespace、population、source、prior catalog、完整72/12身份及其raw-case/generation-audit SHA、semantic/config/sensor/root seed和固定分母。核within-phase、dev、main/holdout全部配置和seed碰撞。Manifest后另有绑定其SHA的独立dataset review，native固定dataset finding同时绑定manifest+review；消除manifest↔review互相哈希循环。
5. DatasetProof每次实际effect前重新核current native/source/namespace、完整manifest结构/身份与review/native应用。允许model service检查manifest及选定raw-case字节SHA，不能反序列化私有case或把私有内容送进actor。只有worker在warmup之前用正式内容reader重核选定case的file/semantic/config/phase身份并构造mission；其他offline grader可按权限读取全部。DatasetProof本身不是已经验证完所有业务结果的凭据。
6. 新增组件须先在/tmp synthetic native fixtures/无实际formal RNG下测试正常路径与source drift、ancestor撤销、别名路径、缺root、坏phase、seed/config碰撞、坏review、raw-case替换和副作用之前拒绝。最终还须贯通classic/LLM runner、service每call、receipt/measurement、SDK完整矩阵和analysis/once-holdout入口，扩展完整source inventory。当前各组件pass不能代替全graph或实际pilot/integration。

后续仍按原计划完整开发pilot与clean+20s真实集成→最终source/model/prompt/split/retry/hardware冻结及独立G4接受→main→主分析锁定→once-only descriptive holdout。Formal source接线不改变已接受19项义务、root数、评分、阈值、B1/B2选择或budget={}。
