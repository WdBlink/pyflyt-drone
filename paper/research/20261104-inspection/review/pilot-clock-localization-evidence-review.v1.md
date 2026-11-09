# Fixed-load clock localization：独立证据复核 v1

结论：记录、二进制数组、固定模型请求及归档的一致性检查通过。本次工程诊断未出现>0.1s lag，**并未证明旧故障已修复**。主峰继承了上一tick中86.357ms的低CPU墙钟间隔；该tick不执行1Hz logging，也没有重叠的已记录GC。因此不能把区间名称当logging原因，也不能把394s的gen2 GC称为最大lag或旧故障原因。

2026-11-06，`/root/statistics_design_review`，分离上下文、同模型家族。采用 autoresearch 的原始证据与失败保留边界；只读原始诊断/原失败/计量回执和源码，以NumPy及JSON/ZIP作离线复算。未运行模型、物理或新的测试，未改生产、owner、失败槽、SDK结果或阈值。原两次失败14份文件与上一审查binding的SHA仍全部一致。

## 完整性与时钟重算

输入为 `logs/inspection-clock-localization-fixed-load.v1/`。按metadata声明的little-endian float64读取：

- ticks为72,000×30，17,280,000 bytes，SHA256 `8469d702d41984f2b1b1e49c81c60bcc099d725c28d8f5364b27a0f895526c25`；所有值finite，sim_after严格对应1/120至600s，每行推进1/120s，物理总步144,000。
- GC为282×11，24,816 bytes，SHA256 `6f20b10886ab3847ee184ccde0dd90dae3544db7fdc84ee08069ac051e2afa22`。回调采集窗口内start/stop按thread/generation完整配为141对，无剩余事件，gc_overflow=0。
- 独立重算max **0.08569320901295896s**，0-based row59870，sim_after=498.925；p95 **0.005028416697541616s**。这里p95使用源码声明的floor-order统计量，sorted索引68399，不是默认线性插值分位数。>0.1s为**0/72,000**；与profile/world完全一致。

归档 `qualification/clock-localization-fixed-load.v1/complete-diagnostic-and-model-operation.zip` SHA256为 `f859e75086d4d44bfa968f2884adbfb78b453faf725441c59f9305da36c9081c`；72成员无重复/路径逃逸，CRC全通过。71个diagnostic/model-operation成员逐字节hash匹配现存原始文件；另一个supervisor log由归档SHA/CRC绑定，未伪称匹配一个未定位的外部原件。主Agent完整binding `6d009178e08736ac5516c04dc7ffb40bf3bc3b6c068facdd3eb67e58bca9febc` 所列8份文件均复核一致。

## 主峰及前一tick：有长间隔，不是在执行logging

| 已保存观测 | 前一tick row59869 | 主峰 row59870 |
|---|---:|---:|
| sim_after | 498.916666667 | 498.925 |
| lag | 3.853500ms | 85.693209ms |
| RPC count / sensor flag | 0 / false | 0 / false |
| 对应physics_steps | 119740 | 119742 |
| publication `%120==0` | false | false |
| 1Hz logging `%240==0` | **false** | **false** |
| pub_end→log_end wall | **86.357333ms** | .001708ms |
| 同区间main-thread CPU | **.102458ms** | .001750ms |
| 同区间worker-process CPU | **.104000ms** | .001000ms |

源码的1Hz条件是在pub_end和log_end标记之间检查，但此处条件为false，未执行该块json.dumps/write/flush。间隔仍包含标记代码、取时及其间可能的等待/调度；不能把它标成“86ms logging开销”。全部600个真实1Hz logging tick与600行progress时刻精确一致，其pub_end→log_end墙钟最长仅**4.374166ms**。

前一log_end至当前begin仅18.959µs；主峰这一tick requested_sleep=0，mission.step对应墙钟3.634708ms、main CPU3.448042ms。相邻lag增加81.839709ms；相邻progress_end墙钟增加减去1/120s为81.838167ms（约1.5µs差来自取样位置）。这支持主峰继承上一tick尾部间隔的时间账，而非当前短step制造全部lag。main/process CPU都很低，支持进一步调查off-CPU/等待类机制；它不能区分调度饥饿、锁/GIL、页错误、内核等待，也不覆盖其他模型/规划进程或GPU的CPU/资源使用。

以下每行是该阶段**墙钟最长的同一行**及同期CPU，不混用不同位置的独立最大值；单位ms：

| 标记间隔 | sim_after | wall | thread CPU | process CPU |
|---|---:|---:|---:|---:|
| begin→wake（含有意sleep） | .008333 | 11.873083 | .017167 | .036000 |
| wake→rpc_end | 467.516667 | .537833 | .499375 | .547000 |
| rpc_end→step_end | 47.691667 | 19.199958 | 1.174625 | 1.175000 |
| step_end→progress_end | .008333 | .782417 | .079083 | .079000 |
| progress_end→stats_end | 387.416667 | .066334 | .001875 | .002000 |
| stats_end→pub_end | 394 | 28.881375 | 28.874875 | 28.880000 |
| pub_end→log_end | 498.916667 | 86.357333 | .102458 | .104000 |

标记是顺序取时的区间，细小差值含计时器/记录开销，不能把表格当函数级CPU profiler。

## GC证据与时间范围

141对GC为gen0/1/2各129/11/1；138对在physics main thread，3对在另一worker线程。GC线程CPU按实际回调线程配对，不能把另一线程CPU当main线程值。事件采集只覆盖已安装callback的窗口，不代表其他进程或旧失败的GC。

唯一gen2发生在worker epoch后394.006031–394.034356s，wall **28.324541ms**、thread CPU **28.317584ms**、process CPU **28.318000ms**；main thread，tick47279，stage_index5（stats_end后的区间），collected/uncollectable均0。其附近stats_end→pub_end为28.881ms，下一tick394.008333的lag为27.584ms。它是本次真实的局部CPU开销，却与498.925主峰相隔约105s；主峰前86ms区间没有任何配对GC重叠。

本次有worker UTC/monotonic配对anchor，主峰映射约05:07:09.02667Z。anchor取样bracket约0.667µs仅是本地取样间距，不是外部UTC准确性界，更不能回溯替代旧attempt缺失的anchor。

## 20个固定请求与原attempt00

逐call独立对比，20/20的**request文件bytes、messages、seed、输入token元数据、输出text、输出token数及token journal bytes**均与原失败attempt00相同。另逐行检查journal的generation ordinal连续，prompt-count与请求元数据一致；共136,335 input tokens、5,799 output/journal行，合142,134 tokens、20calls。这是读盘计量核对，未加载模型或重算tokenizer。

相同的是固定内容/顺序/seed，不是整个自适应运行：本次B0 ReferencePolicy控制物理，固定模型响应只作load、没有通向command queue的执行路径。实际提交最多比tape目标晚.633333sim秒；service wall总和本次477.559995s、原attempt00 590.058777s。主峰时call18在途，目标497.516667、实际提交497.516667、回复535.558333。策略、host状态、时间和实际并发时序不同；既不能把模型wall差异当性能结论，也不能把B0诊断未复现等同于adaptive B4已修复。

## 来源、退出与计量边界

独立复核24份source、61份dependency、private case、tape及tape绑定的原始来源SHA，均与记录相符；保存的四项unchanged flags也为true。diagnostic为 `completed_diagnostic_not_qualification`，worker completed、errors=[]、physics_cleanup remaining_live=[]。只读ps观察中保存的supervisor/service/physics三个PID均已不在进程表；这是点时退出佐证，不扩大为任意后代或host-loss认证。

本次usage receipt/archive绑定一致，native settlement保存为known；自身20calls/142134tokens，保存的owner summary总373calls/2382225tokens，reserved/unknown/unresolved均空。cost仍明确unreported、capital/energy及host native usage unmetered；known不意味着费用被测成0。未读取或改写真实owner状态。

## 仅一个下一诊断建议

若主Agent另行冻结新诊断，最小增量可为现有begin/wake/log_end边界加进程rusage的minor/major fault与voluntary/involuntary switch差分、调用bracket，以及该worker实际requested QoS，先核平台计数有意义并测新增开销。可证伪目标是“该长wall/低CPU间隔是否伴本进程fault或switch增量”；零/不可用计数不能误当所有等待机制已排除，不建议提高QoS或更改GC/logging/门槛。

新鲜dev-L2-00的unscored adaptive instrumentation在这个解释边界内可作现象定位，但它不是原L1单因素复现；一次pass不提供旧根因修复证据。该意图是否冻结、计量及执行由主Agent另行决定，本报告不安排新运行，不恢复失败槽、不生成SDK，也不作G4/策略结果批准。

binding为 `review/evidence/pilot-clock-localization-evidence-v1/bindings.json`；完整独立离线脚本与中间数值留在 `/tmp` 并绑定SHA。原失败及本次工程诊断均保留，完成后停止R写入。
