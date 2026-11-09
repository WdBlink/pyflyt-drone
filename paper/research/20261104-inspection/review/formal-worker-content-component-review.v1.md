# Formal worker content / pre-warmup v1：独立有界审查

结论：存在一项P2，当前应保持未安装：实际执行的configuration身份hash helper没有进入G4/worker独立来源闭包。已在/tmp合成项目复现错误configuration描述因这个未绑定依赖改变而被接受；不是缺少dispatcher/完整graph的泛化待办。其余本轮已核的单buffer、身份顺序和Fake预warmup测试通过，不据此授真实formal数据、物理或G4许可。

审查者 `/root/control_review`，同模型家族独立工程上下文。R=`paper/research/20261104-inspection`。freeze binding SHA `041e6ff0606c8dba1bc65c18ce5e5160a9cbedccdbc2cb2566b361bdb6cfddfb`，28项冻结文件及11源/10测试working字节均核对；三项production-before锚也未变。独立执行所附case/payload与worker/scenario/AST **41 passed/4.87s**。root的256 combined与历史fixture构造错误日志保留为输入，本轮未重跑256、147dev或其他pipeline。

## P2：配置身份比较依赖未冻结的实际执行文件

`inspection_formal_case.py:13,91`调用 `inspection_development_catalog.physical_configuration_digest`；冻结catalog `:17,26–30`继续调用 `eval.prepare_inspection_development_population.configuration_digest`。实际实现文件的函数在repository `eval/prepare_inspection_development_population.py:31–37`，但该文件不在本版G4 `REQUIRED_SOURCES`或11份冻结源码中，也没有被本版独立source resolver所要求。

这已经是当前worker内容身份guard的直接执行依赖，不只是未来producer。即使formal_case/catalog自身源SHA不变，修改该helper仍可改变configuration判断，而fresh DatasetProof不能报告其source drift。

独立坏config实证：只用/tmp serialization fixture和synthetic/native-shaped owner，将一个已绑定manifest的configuration描述设为0…，private copy和真实semantic身份不变。使用原helper时，完整读取明确以`private content differs from its admitted semantic/configuration/noise identity`拒绝。随后只修改tmp项目中未入REQ/verified inventory的该helper，模拟fresh worker对catalog依赖的正常导入，再运行相同内容读取：fresh DatasetProof仍通过，错误0…configuration被接受。

helper的原始实现SHA `9d564a45412165aab72795c9dca6cf61ad1d96f314066b0d0447e86947e5248a`，tmp改变后SHA `97f49e21a3610b4fa226cc0872953305681a8e24ecc0e2b2e0e15a404370cdbf`。实际configuration应为 `4790be2aa4a5a7065cc0438f48226d6b425b85a91d4b7cbaa64d7f8d3bc16634`。真实production/helper、owner和冻结输入均未改；反例没有改G4/Dataset算法或跳过其谓词。首个probe的/tmp符号别名路径已canonicalize，最终baseline断言特指身份拒绝，未把路径拒绝冒作坏config被挡。

最小修复：将实际文件加入独立 `REQUIRED_SOURCES`/context resolver/source manifest/freeze消费清单，并新增该文件字节漂移的拒绝回归；不用移动/复制hash算法或改producer语义。原P2在新冻结版本闭合前保留。

## 正向范围和身份边界

- `load_formal_worker_input`先核privileged runner payload的完整schema/字段和Artifact refs，再在本进程构造verification；不接收pickle过来的callable verifier。payload不是case JSON或actor/tool消息，本轮不按不可信模型能指定runtime路径来误报接口漏洞。
- `load_formal_case`先fresh DatasetProof，限定一个record，再核project内copy路径、符号/硬链接、expected raw SHA和16MiB bound。hash与strict JSON用同一个buffer，phase/root/domain/level/sensor/semantic/config逐项比较，然后再fresh DatasetProof并比较完整proof。除上述未绑定helper外，所附坏身份、重复key、source/stop/revocation等负例拒绝；没有把结构manifest标签直接当作私有内容真值。
- 单buffer读取后文件变化的所附例故意仍返回原已验buffer的case和digest：这是版本身份，不是永久路径能力。未来end-copy/receipt/读取边界仍需各自检查，本轮不将该设计重报为返回后无限保证缺陷。
- formal worker先核4thread1、seed0、eventsFalse、reference_layoutFalse、600s/1×及自身private-case路径，再做formal admission/content，最后才到warmup constructor。Fake事件顺序实测为admission→verifier→content-verified→第一warmup constructor；坏payload/copy/G4/参数均未到constructor。
- ScenarioMission显式formal_phase走内容law，None默认仍走dev guard；main serialization fixture在默认分支拒绝。控制while-loop和其余scenario methods AST检查通过，business audit仅显式区分adapter/结果binding。没有运行真实legacy飞行或147dev重放，不能把AST兼容扩大为完整回归资格。

这些测试在第一物理constructor处用Fake停止；未运行真实spawn、native环境完整context、MLX或物理。selected输入是旧dev数据的临时serialization/copy fixture、synthetic目录catalog/owner，不是实际main实例、重新标注科学数据或真实qualified dataset。primary base population之外的formal clean/CLI/runner/service/receipt/measurement/fullgraph都不在本轮完成承诺内，不另扩审。

## 交付与停止边界

[evidence](evidence/formal-worker-content-component-v1/)保留41项日志、独立probe及结果，[binding](evidence/formal-worker-content-component-v1/binding.v1.json) SHA `5d1666345484b7654997e2ff79c04d0088b5490a50a4f416f5e6b0ba4b767dcb`。probe有冻结working字节guard，未来版本改变时拒绝误测旧版。结论限本项P2和已有有限检查；原输入不改，后续修复另版复验。报告/绑定冻结后立即停止写R，不继续dispatcher或兼容矩阵审查。
