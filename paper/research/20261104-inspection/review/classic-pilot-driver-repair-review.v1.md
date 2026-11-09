# classic pilot驱动修复独立复审 v1

日期：2026-11-06（Asia/Shanghai）。**原三项P1在已测路径关闭，严格cache重包和源/依赖变化的全分母行为得到支持；另发现child source-map未绑定初始frozen的P1残留，已成功复现source ABA准入。** 本报告不批准后来的guard，不认定V2 pilot完成，也不授G4。

同模型家族独立上下文，与主Agent分离。只新增本报告和`review/classic-pilot-driver-repair-v1/`证据，不改生产源、SDK、输入或live logs。不调用真实physics CLI，所有模拟运行和依赖/源变化均在隔离副本；实际历史与V2已完成raw只读核对。

## 1. 精确绑定与检查方式

受审驱动为`db4bdaa73e3b091d2249b0b1ab18a9a30d5ee2f70c28687830e1df4f67ad5843`。检查启动时将其源码完整保存于`review/classic-pilot-driver-repair-v1/results/driver-source-snapshot.json`，再以相同字节在隔离根目录加载。主Agent另保留于`qualification/source/classic-pilot-driver-v2/`。报告不绑定修补中的current同名文件。

提供的ExperimentRecorder保持此前`4dffd762966ef27d78fd521341aefb2145895b26ad9d382d16f04f1bab6eec02`实现，未替换或修改。探针为隔离源/六根suite/61个轻量依赖文件及真实本地回执模块生成的**合成JSON**，通过find_spec指向隔离依赖；真实`run_inspection_dev`调用全部拦截。短Python worker只写JSON，不导入/运行物理组件。

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python paper/research/20261104-inspection/review/classic-pilot-driver-repair-v1/probe.py
```

所有场景`physics_cli_executed=0`。证据里的SDK完成/选择只是合成测试输出，不是实际pilot或方法成绩。

## 2. 原三项修复与全分母复验

| 场景 | 修复版独立结果 |
| --- | --- |
| 正常18槽 | 18/18 completed，18次post cache审计，合法B0 Y0保留；近tie按1e−12规则选B1 |
| 新SDK包复用同receipt store | 18/18 completed及18次post cache审计；合成worker总数仍18，新增0次 |
| 原P1-A：最后slot改变suite/case | 17completed+1failed，18行/18attempted，execution_status=failed，无selection |
| 原P1-B：最后slot依赖变化 | 同上，61文件初始依赖集合被前/后检查阻断，无selection |
| 持续source变化 | 同上，17completed+1failed/全18，无selection |
| 原P1-C：CLI返回后改bound stdout | 首槽完整cache审计拒绝；18行，attempted1、completed0，其余未启动，无selection |
| 首槽CLI非零 | 18行，attempted1、completed0、failed，无selection |

`:62-66`统一检查suite/source/dependency，slot前、返回后、complete前和最终分别调用。`:112-115`比对plan的初始suite/case/canonical case/dependency；`:117-121`以禁止dispatch回调调用共享cache校验器，核对完整intent/start/stdout/flight链。探针中原链篡改没有引发新飞行，已验证缓存审计职责。

正常包和缓存包都保留6预选root、B0/B1/B2各1、18逻辑槽。provided SDK的attempt代表本次回调/记录，不把cache读取增加为physical flight；操作失败保持None/operation，未变为策略Y0。B_classic*规则仍是完整18后等级宏平均只比较B1/B2、tie偏B1，B0不被删除。

## 3. 剩余P1：当前磁盘复查不能替代child源版本绑定

受审版`:112-115`没有比对`plan.source_hashes`与初始预期worker源集合，也没有核对plan的supervisor/receipt-adapter SHA与初始frozen。完整cache审计只证明flight内部与**它自己的plan**自洽；check_frozen证明检查时磁盘仍为初始字节，两者之间缺少连接。

独立`results/source-ABA`操作全部限于隔离副本：

1. 前17slot正常。
2. 最后child接手时改变一个被初始frozen覆盖的源副本；child plan/source记录变化后的SHA，run/receipt全链与该plan自洽。
3. 模拟CLI返回前把源副本恢复到原字节。
4. 驱动所有磁盘check_frozen都通过，共18次完整cache审计，最终仍18/18 completed并写selection。

结果`receipt_source_maps_differing_from_initial=1`，其余suite/case/61依赖保持一致。这证明可以接受不属于最初冻结源版本的receipt；**不是说live源发生了ABA或该合成变更造成真实Y优势**。只增加前/后/最终磁盘检查不能关闭此窗口。

最小修补应从最初frozen精确构造每个condition的预期worker source-map（B0的源集合与B1/B2的classic模块差异按已冻结规则处理），逐条比对receipt plan；同时核对supervisor与receipt-adapter SHA。禁止以刚读取的receipt plan自行定义“预期版本”。旧物理receipt能否复用仍由原source/suite/dependency/输入精确一致决定，不因parent升级而重飞。

主Agent已接受并另补该guard。其修订发生在本次探针完成后，不在本报告受审字节中；需单独复验正常/cache与同一ABA/source-bind反例，不能回填为本报告已通过。

## 4. V1稳定失败包与V2部分cache证据

只读`qualification/classic-pilot-generation1.v1/`，核对results SHA为`c92f92c5fa68dd207accd27c2f92ee9ab5a04f0a9ab90b9b477bc9e7b1605fba`：full denominator=18，1条failed/attempted/operation、17条stopped/未启动，execution_status=failed，无selection。这里的失败是parent-source变更，不是策略0分。

meaning绑定的实际terminal SHA为`35ed4a32fff671d3812d8aaffabd6eb8f6185cfca1619e0d6c7f4978f0703ac7`。只读实际store确认其completed、returncode0、仅attempt-00，六文件hash链与Y/evidence一致，Y=.8245752535103837。这只用于核对单条实际回执来源，不是方法比较成绩；本审查没有重飞。

V2现场只读取完整可解析的completed SDK raw，观察到dev-L3-00/B0的cache_reused=true、值同上、terminal SHA同35ed4a；该raw SHA为`87020d2e3a1430d2e21d47ac790f4a8883da15b50c7ad77cf017dce9cc00f641`。观察时V2 results/selection均不存在，故只证明这一槽cache接线，不证明18槽完成。未读取/判定其他进行中的native flight为完成。

主Agent后续告知新guard只改变parent driver，child physics/CLI/env未改；V2 parent将在当前child结束后因parent源变化abort，完整实际receipt保留，V3再严格cache重建。此为执行安排，不在本报告写成已发生。源改动后的SDK拒收不能丢弃原physical receipt或制造另一个独立样本。

## 5. 冻结结论与限制

原三项P1的已测修复可以确认；source ABA/source-plan绑定P1仍阻断该受审版的统一源版本选择验收。所有操作都在审查副本中，未动live logs/真实源/依赖。真实完整18slot、最终B_classic*及正式八配置/G4均未验收。

| 检查点 | 结果 |
| --- | --- |
| 正常/严格cache重包 | 18post审计，新增0合成worker |
| suite/依赖/持续source变化 | 全18分母，17完成+1操作失败，无selection |
| CLI后回执链破坏 | 首槽阻断，余17未启动 |
| source ABA | 磁盘恢复后全部通过，1条plan源map错误仍选择，P1保留 |
| 稳定实际证据 | V1完整失败分母与35ed4a实际receipt分开保留；V2只观察1条cache raw |

Attention

reviewed by inherited-parent model（同模型家族独立上下文，非跨模型家族审查）

本报告绑定db4bdaa快照，不能用于批准之后的source-map guard。SDK机械完成、单条实际receipt和pilot dev选择仍是不同结论。
