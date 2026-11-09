# Production runtime integration：有限独立审查 v1

结论：**7源最小接线范围内无明确阻断。** helper 与已审候选逐字相同；原控制loop未改，warmup后/epoch前启用、正常及异常路径关闭，恢复失败有audit并阻断receipt。集中检查已结束；该结论不授G4，不把20s烟测替代新600s资格。

reviewer `/root/statistics_design_review`，同模型家族独立上下文；依autoresearch区分源码、软件fixture、真实短烟测与正式成绩。

## 输入与差异

- freeze binding SHA256 `b57679da7beea68d611ed7eb96c8ff5b5bbc793f055cad302f959f6dd59da9b8`；7个生产源、2个tests和日志均逐SHA核对；当前7源与freeze相同。before6源备份及before-hashes（SHA `6401fa5a4388747ab9871ab48549551e5513e840a07079cf32aca60ed3a1d9a1`）匹配。
- 新helper SHA `bf4e0b38f1d7da98ccd559091b2b722c74ed0e6fdfd0088336ccaa441bdfc21c`，与已审 `physics-runtime-window.v1/freeze.v2/source/physics_runtime.py` 逐字相同；Mach ABI/soft语义沿用其有限审查，本轮不重复实际native设置。
- reference worker：fresh mission在warmup后构造，runtime在原epoch前start；loop后和finally close（幂等），mission.close在其后。start/loop/restore异常保持operational_failure；physics_runtime及errors进入world。原while-loop AST一致，未改RPC drain、120Hz sleep、step、publication/logging、lag算法、阈值或业务规则。
- reference/dev/classic/all-eight/LLM的worker与外层source inventories均加helper；all-eight LLM继承base_sources，classic expected-worker subset也一致，没有本范围helper漏绑。
- shared validate_flight仅当plan来源包含该helper时要求typed runtime audit；runtime_window_invalid不在已知clock/process可重试集合中。旧plan无helper时不补发新runtime资格，历史档案不回填。

## 实际软件检查

`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -p no:cacheprovider tests/test_inspection_worker_runtime.py tests/test_inspection_runtime_integration.py -k 'not short_native_self_thread_round_trip' --basetemp=/tmp/runtime-production-integration-review-v1-provided -q`：**17 passed,1 deselected in0.54s**。有意不重复短native调度测试；本轮未修改宿主RT/GC。

独立脚本 `/tmp/runtime-production-integration-review-v1/test_independent.py`：**5 passed in0.56s**。实际helper配Fake native/collector和Fake mission，normal接受、start/restore失败保存audit并operational_failure；accelerated不构造runtime。另2个真实synthetic subprocess只写fixture JSON：新helper restore-failure terminal非retryable，重入不派第二child；无helper旧receipt按旧scope仍可读取。它们均非物理飞行/模型结果。

作者115tests日志只核绑定及初始Fake缺属性失败记录，不冒称本人重跑115。初次inventory脚本误import未安装的formal抽取helper，ImportError日志保留；改本地纯矩阵检查后通过，不是生产缺陷。没有展开先前formal graph/架构。

集中核验最后于 **2026-11-06T14:42:27.870372Z** 完成，已先通知root。随后仅只读bound烟测/归档，无重复测试或实际模型/physics/native owner动作。具体日志UTC元数据见evidence；不作clock因果归因。

## 已有v4 prepare：只读复核

binding SHA `59122e64c8dbeb812655ed1792859ca79cd8846f10a8048b91e5365f0c771317`。原7份文件SHA、source嵌入文本与104当前source、61依赖、10输入全匹配，helper精确纳入。6root完整108槽=18classic+90LLM，48rootmeans；18/root矩阵、90个LLM sampler key/native uint32、8成员archive SHA/CRC正确。没有重draw。

schedule `2026110641`、sampler master `2026110642`；首槽 `dev-L1-01/B4-M2/repeat1`，order_index0/root_block0。原B1 selection SHA `75f57580f0f5371d91a741fd4f9573e52192d019e9730335123a7ae67897a32e` 保留，未重选。这里是prepare-only意图，不是108次实际完成或新的独立root。

旧diagnostic_worker及clock/profile派生文件保持自己的历史worker/无新runtime helper调用；已记录当前hash，但未重跑旧诊断或把其结果提升到新环境。已归档旧source证据保留原scope。

## 已发生20s真实非LLM smoke：有限核对

smoke binding SHA `211edd621ae879702d46d165f1d603cb6f37356957fd111e88359b1b9e8e37d2`。核对bound ZIP SHA、7成员/CRC和6项原成员SHA；run/world均completed，20s、4800physics steps，helper来源SHA正确，private copy/world指针一致，source/dep/private unchanged标志true。CLI exit0来自root的session46562 witness，本人未重新附着该OS session。

world报告 maxlag **8.42125ms**、p95 **3.45604ms**、0个>.1 ticks；runtime errors[]，active首尾请求读回一致，standard policy/GC恢复。普通world/1Hz progress不提供完整lag向量，本人未伪称重算所有ticks或证明窗口从未demote。root以真实run字段调用validate_flight仅因 `incomplete_or_wrong_clock` 拒绝600s成绩；源码也保留600s/144000steps门。本轮不重复调用或延长烟测。

这是生产接线短烟测，不是完整mission/root/pilot，无模型负载、不是受控A/B，不能重分类旧clock失败。新helper只作用physics当前线程调度及其进程cyclic-GC；reference counting、其他进程GC和宿主干扰仍在。无硬实时、任意恢复或600s性能保证。下一步完整108槽pilot仍是新的真实开发资格；本报告不授G4/main。报告冻结后停止R写入，烟测/launch新证据另行处理。
