# LLM episode source-clock / age guard 最终窄复验 v1

日期：2026-11-06。**固定 ceed2a validator 在本次列举反例中关闭了 624504 报告的 D1：两条未来 public clock 均被拒，正常 0/20s 元数据保留；numeric age 的 boolean 别名也被拒。** 只复验新增两处 guard，不重跑31项 suite、K1/K2、orphan 或实际模型/物理。

## 固定对象与证据

| 对象 | SHA-256 |
|---|---|
| validator | `ceed2af65c21da0901d80d996537ac6f1469a2840256989ba4dca75d158ab53f` |
| unchanged supervisor | `37a40dcc989059a4a6abb43a6fc7b741153a7291fc075225bd06046a78928364` |
| unchanged cleanup | `361ba11b8ea088627e4c0bd832168bb26821297e5d8f54bf6d43c2ba327a9c70` |
| reused synthetic fixture source | `f8dbd74ed9ef0297c42ac9098932699dad835e3d224a8141faebac790e3baf35` |

证据根 E/：

`/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/review/evidence/llm-episode-native-accounting-clock-guard-v1/`

R 静止期间只在临时目录复制源码、执行 CPU 合成检查；明确收到恢复后才将证据原样复制到 E/。历史记录保留当时的 tmp 绝对路径，没有事后重写消息或输入来伪造执行位置。`execution-origin.json` 说明来源；脚本可在新目录重新生成 fixture。

| 证据 | SHA-256 |
|---|---|
| `source-binding.json`：8项原源快照 | `92519d62682370aacc74d38d0f3b96fa63f6dd7eaf371a9e6714c52263338fcc` |
| `results.json`：5项实际检查 | `e8d0778392a1c743638340c65d61ba5eaeed7d1a98f9236051238aae7d6ddb43` |
| `evidence-index.json`：54个文件 | `a436ee3468041959f84280b76fc22016f8fd244f83ca329e70f11c75d509c3f9` |

`validator.diff` 对照已冻结624504：只有公共时钟加入before采样顺序，及sim age增加严格native数值/finite/nonnegative检查两处变化。源码同时保留原差值等式、before≤reply≤after和wall age规则。

## 实际5项检查

直接调用只读 `validate_llm_episode`。两套 baseline 由原 synthetic child 前台运行后写出全部假 flight/request/response/token/actor 数据，再调用 validator；没有调用 supervisor执行入口、fake或真实owner action，没有reserve/settle。反例同步修改 public frame、canonical message SHA、实际call/request.messages、decoded audit.request与age，避免仅因消息hash不一致而拒绝。

| 检查 | public / before / reply / after | sim age | 实际结果 |
|---|---|---:|---|
| 正常0s arm | 0 / .1 / 1 / 1 | 1 | 接受Y0 |
| 正常20s arm | 0 / .1 / 1 / 21 | 21 | 接受Y0；parent compute59+inject20，wall age79 |
| 原future public=5 | 5 / .1 / 1 / 21 | 16 | 拒为 `invalid physical concurrency witnesses` |
| 原future public=580 / negative age | 580 / .1 / 1 / 21 | -559 | 同上拒绝 |
| boolean age=True（本可与native 1相等） | 0 / .1 / 1 / 1 | true | 拒为 `dispatch age conflates or omits the injected delay` |

前两条证明新guard没有把记录中原合法的0/20 arm或注入部分误作service soft timeout。后三条分别验证原D1及新增age严格类型；finite/nonnegative与精确差值的实现也做源码核对。没有扩成全字段 hostile forgery/fuzz 或硬件attestation保证。

这些全是假计时与假模型/flight记录，Y0是fake evaluator输出，不是实际方法成绩。20s本次没有真实等待，也没有MLX、GPU、PyFlyt实例/physics或实际forward。反例测试修改的actor/request文件均在本次临时fixture内，baseline按原字节恢复；改动副本留在counterfactuals/。

## worker顺序的静态旁证

对照此前冻结的actual runner依赖 `run_inspection_reference.py`，SHA `4c280adab488f9c81d68265dd9930d9775dbb6372b36d1d776d9350f9f36f3ca`（保存为E/static/worker-source.py）：常规循环先step，再在78–80行写shared_progress，90–92行才put公共state；初始57行发t0 state时共享时钟也是0。结合该worker的单调时钟和之后读取的before sample，source≤before关系适用于这条接线。这里只核对源码顺序，没有为此启动worker或扩大为host/硬件时钟认证。

实际命令为仓库 `.venv/bin/python` 3.11.15，在临时source目录中：

```text
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B ../probe_clock_guard.py
```

script/日志/source/fixture及字节差异全部保留。两条foreground synthetic child均返回0，readonly evaluator subprocess正常结束；没有sleep/orphan反例或后台进程，也没有live owner/`.paper`操作。

## 判定与限制

**D1在固定ceed2a及本报告5个有界检查中关闭。** 原624504报告及其两条误接收证据完整保留；原版、506f16及cache/delay报告均不回改。沿用同37a supervisor已有独立K1/K2证据，本轮未重复整套cache/suite/cleanup压力测试。

本结论限于新增clock/age guard。仍要求外部在dispatch前提供完整、可信的expected source/input/profile/evaluator绑定；不声称任意全证据一致改写的防伪能力。未认证实际模型/权重/费用、真实case/trace、完整scorer或live native ledger，不授予G4、main/holdout或正式八配置运行通过。

Attention / reviewed by inherited-parent model：同模型家族独立上下文的有限代码与合成CPU只读检查，非跨模型盲审或人类专家认证；未运行真实模型或物理。
