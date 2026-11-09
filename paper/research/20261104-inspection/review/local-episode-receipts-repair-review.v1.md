# 本地episode回执修复独立复审 v1

日期：2026-11-06（Asia/Shanghai）。**原R1、R2、R4在本次实测范围内关闭；原R3反例已修，但其一般错误分类规则仍有P2残留。完整回执集成与G4不获认证。**

本次与主Agent分离、同模型家族独立上下文。只写本报告及`review/local-episode-receipts-repair-v1/`证据，不改实现、控制器、旧报告/反例。只运行轻量JSON、sleep与进程组测试，没有物理/API/模型或重负载。审查期间准备/启动的首次真实600s回执飞行不进入本报告，旧L1运行也不倒认为回执flight。

## 1. 来源绑定与实际命令

| 文件 | SHA256 |
| --- | --- |
| `eval/inspection_episode_receipts.py` | `bedc309eba9276a0846d928ab81616f8c51a8db727b4767c8f5d2b2ef239c543` |
| `tests/test_inspection_episode_receipts.py` | `944117d48bd90a5967e7de09deab5693b10c499ea9a804557ea63d0dd01e03a9` |
| `eval/run_inspection_dev.py` | `e63d69e75251752330da2806915bae711ad4be2920e9017c2ffea726d64386dc` |
| `eval/run_inspection_reference.py` | `1f4ef28d9935a6293f9ce6ea97d3472b2781f9aee3dc166ba20b8804c2e719b1` |

四份源码全快照、SHA及探针结果保留在`review/local-episode-receipts-repair-v1/results/`，检查开始与结束SHA一致。行号以下述快照为准，不批准之后同名文件的新字节。前次v1报告/快照/全部反例保持原样。

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider --basetemp=paper/research/20261104-inspection/review/local-episode-receipts-repair-v1/pytest-temp tests/test_inspection_episode_receipts.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python paper/research/20261104-inspection/review/local-episode-receipts-repair-v1/probe.py
```

实际结果：**14 passed in 0.59s**；独立探针执行完成并保存成功/残留反例。合成JSON不是物理样本，“探针完成”不等于所有缺陷关闭。

## 2. R1–R4逐项复审

| 原问题 | 独立复验 | 结论 |
| --- | --- | --- |
| R1 缓存链缺少intent/start/stdout/Y核验 | 分别删除start、改intent.argv、改stdout、改terminal.Y，均阻断，worker计数维持1；另将completed标not_started=true也阻断 | 原反例关闭 |
| R2 start写入失败遗留进程 | 注入started.json写入ENOSPC，异常返回前子进程已reap；下一调用仍unknown阻断。正常退出的合成worker另生成sleep后代，进程组清理后后代不存在且延迟存活标记未写出 | 原反例关闭，含已测后代组清理 |
| R3 可见源缺陷被missing-case遮盖 | 原bad-source-missing-case现在只执行1次，保留source与input两个原因，retry_allowed=false | 原具体反例关闭；一般规则仍有下节P2 |
| R4 新目录父fsync缺失 | 新建两级receipt目录链与attempt，实际_sync_directory调用成功；祖先/receipt父目录及attempt父目录sync均先于Popen | 原缺口在实际fsync/process-recovery范围内关闭，不认证物理掉电 |

R1实现现在将intent、started、stdout、flight/run、flight/world及flight/private-case共6项哈希绑定到terminal。缓存重用复核全绑定、stdout字段、plan/ordinal/start及重算Y。它仍依赖本地受信写者/文件边界，不是对任意外部同时重写整条链的密码学认证。

R2的spawn后start写入与wait已处于统一finally `_cleanup`中；清理实际发送进程组终止/kill并wait。监督进程自身SIGKILL/机器崩溃不能执行finally，仍须按unknown进行显式源绑定reconcile，不能把该范围写成已自动恢复。独立terminal写ENOSPC检查也仍unknown阻断，worker只调用1次。

R4的`_mkdir`逐级创建后sync父目录，attempt.mkdir后sync episode目录，stdout创建后sync attempt，再spawn；本次轨迹覆盖此前遗漏。没有做真实断电或硬件掉电实验，普通fsync的设备/平台边界及错误返回仍需如实声明。

## 3. 剩余P2：run缺失时跳过已可见的case不匹配

位置：`inspection_episode_receipts.py:87-92`。case_hash已独立算出，但整个case/线程条件仍被`run is not None`包住。因此run缺失时，即使private-case文件已存在且其SHA明确不等于plan，也只剩普通worker故障，允许补跑。

独立`missing-run-wrong-case`探针执行一个轻量合成worker。它正常退出0，写world和case后删除run.json，再把case改为错误字节。结果：

```text
worker_calls = 2
terminal[0/1].evidence.failure_reasons = ["worker_operational_failure"]
terminal[0/1].retry_allowed = true
最终错误 = finite symmetric infrastructure retry budget exhausted
```

这没有返回合法Y或突破两次cap，影响是对已知错误输入多运行一次，并把其原因误标为可重试worker故障。它属于原R3“先收集所有可见源/输入缺陷”的未覆盖组合，不应以单个旧反例通过而宣布一般规则完整。

同一门控还影响`:94-96`：独立附加探针保留正确复制case、让world.private_case_file_sha256明确错误并删除run，仍得到2次worker调用、两个retry_allowed=true、原因仅worker_operational_failure。其证据位于`local-episode-receipts-repair-v1/extra-missing-run-world/result.json`，绑定相同bedc309e源码。

最小修法是将`case_hash is not None and case_hash != plan["case_file_sha256"]`独立记为nonretryable input mismatch，不依赖run存在；同理，现存world的worker case SHA与plan不符应独立判断，只有run→world_sha关联需要两者同时存在。run和case都缺失仍可保留正常未写结果的已知worker failure分类。主Agent已收到两个反例，并明确等当前真实600s任务完成后才改绑定模块；本报告不批准尚未实施的新guard。

## 4. worker身份/线程接线与范围

源码静态追踪确认worker读取case_bytes后先与expected_case_sha比较，再反序列化；world记录该SHA，run结束检查复制case未变，并记录线程环境。修复版validator比较worker case SHA、结束不变标记及plan线程环境。

独立注入三种合成故障：world读入case SHA错误、run.private_case_unchanged=false、runtime_thread_environment不匹配。三者均只运行1次、retry_allowed=false，分别以world_binding/input mismatch阻断。这里证明的是对应检查会拒绝故障，不证明真实generator、root谱系或真实运行环境已被认证。

`run_inspection_dev`仍只支持本地B0/B1/B2，plan来源与固定CLI闭包静态支持相同设置；当前没有本审查验收的真实receipt flight、LLM/provider/费用路径，也没有与完整8配置SDK矩阵的真实measure集成。既有unknown阻断、策略Y0与操作失败分离、两次cap的适用前提保留。SR/事件/恢复删失及导出可复现绑定不因本修复复审完成。

## 5. 决策轨迹与冻结边界

| 决策/检查 | 证据 | 结果 |
| --- | --- | --- |
| 重复全部旧故障机制并加not_started/后代组 | 独立probe的cache/start/orphan条目 | R1/R2原问题关闭 |
| 以新目录链观察真实sync顺序 | directory-sync.events | R4实际fsync范围关闭 |
| 将run缺失与错误case叠加 | missing-run-wrong-case | P2保留，已立即通知主Agent |
| 不修改正在真实运行绑定的模块 | 本报告来源SHA及主Agent回复 | 等运行完成后另版guard复验 |

Attention

reviewed by inherited-parent model（同模型家族独立上下文，非跨模型家族审查）

剩余P2、显式unknown reconcile与真实集成边界仍开放。合成测试、进程清理和fsync轨迹不构成物理飞行、掉电认证或G4放行。
