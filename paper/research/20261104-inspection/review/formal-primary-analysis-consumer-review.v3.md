# Formal primary analysis consumer v4 有界修复复验

结论：本轮唯一 P2（递归目录漏绑）已关闭，未发现本次小 delta 的阻断项。此结论仅限未安装分析消费者的目录/条目快照修复；不是 whole graph、G4、main/holdout 执行许可或人类领域认证。

审查者为 `/root/control_review`，同模型家族独立工程上下文。仅写本报告及其独立证据目录，未修改实现、旧报告、owner 或生产来源。

## 精确版本与增量

- Subject：`planning/formal-admission-candidate.v3/primary-analysis-freeze.v4/binding.json`，SHA256 `1eb375245f88fc9c29a21c828a75a04e3112937ec02c61a1705d0d2b8e4e5216`。
- 分析 entry SHA256 `be023589f6c6c54b48ba51ba2dde51d96a373d857b6f2b4ed98114f69f182f67`；测试 SHA256 `e8a686d35d2c64646ed2ee798bb57e94a165bcf5d6b5f706750c43cb9cf519a5`。
- 五成员冻结字节与声明 SHA/size 均核对，执行自有探针前后 working 对应五成员与 freeze 相同。逐字节 v3→v4 diff 仅修改 `_receipt_snapshot` 的递归目录集合与非 regular 条目拒绝分支，以及相应测试。`inspection_formal_admission.py`、两数值模块均保持 v3 原字节；未扩权限、统计、SDK 或分母规则。五个完整 SHA 见独立 binding 与 `subject-member-check.json`。
- 复用前次 v3 报告（`formal-primary-analysis-consumer-review.v2.md`，SHA256 `63090cebad5de1feb2937b942b47dc37a832e33ab4bde5367fc16b1daaf9a405`）已关闭的三项时序反例和 CLI JSON 结论；本轮没有重复三次 1296 SDK fixture 或数值重采样。原 v3 漏洞及证据不回填为通过。

## 原反例重放及相邻条目

实际执行脚本为 `review/evidence/formal-primary-analysis-consumer-v3/probe_recursive_entries.py`，四个 thread 环境变量设为 1，`PYTHONDONTWRITEBYTECODE=1`，使用 `.venv/bin/python`；最终进程 exit 0，约 0.11s。脚本使用原实际 meter `secondary-consumer-runtime-freeze.v1/source/eval/inspection_model_receipts.py`（SHA256 `f68f149960254ef9e50a47354722d7f64d83417adda26a845fb076847dab6a49`）验证完全合成 one-call 日志，未调用模型、物理、正式 RNG 或原生 owner 操作。仅模块导入，无 physics 构造或运行。

| 检查 | 实际结果 |
| --- | --- |
| 原有效 one-call 日志及快照 | 原 meter 通过，1 call / 4 tokens；快照 guard 正常通过 |
| 验证后新增空 `call-0002/` | v4 目录集合改变，文件 SHA 集合保持原样；最终 `_check_snapshots` 拒绝 `closed receipt input changed after validation`；原 meter 同时拒绝 `incomplete or unexpected call record tree` |
| FIFO | 初次快照与最终 guard 均在读取其内容前拒绝 `must be regular files or directories`，没有阻塞读取 |
| 本地 UNIX socket | 同一非 regular 条目分支拒绝 |
| 普通文件、目录和断链 symlink | 均保留 `cannot traverse symlinks` 拒绝 |
| 移除各异常条目 | 原 one-call meter 与原快照重新通过 |

代码定位：v4 entry `analyze_inspection_formal.py:54` 创建含 base 的目录集合；`:62` 将每个递归目录加入；`:64` 保留常规文件；`:66` 拒绝其他类型；`:67` 将完整目录集合纳入快照。`:71` 的最终 guard 仍比较验证前快照，未采纳新树为可信。原 meter 的 `inspection_model_receipts.py:69` 枚举 `call-*`，`:77` 要求每个调用树完整，因此空调用目录是实际计量语义变化。

首次额外 socket fixture 使用的临时路径超出 AF_UNIX 路径长度，出现 `OSError: AF_UNIX path too long`，不是产品拒绝或通过证据。保留该日志；仅缩短自有 `/tmp` fixture 路径后重跑上述小探针成功。未改候选实现。

## 测试证据与限制

实际读取 root focused 日志：15 passed in 52.65s，SHA256 `468257bf3dfc6016a52407ceed35ab20acb4732ca128d519ebc3a385e8b91f14`；这是 root 执行结果，本审查没有重跑。root 随后明确告知 compileall exit 0、408 项全套刚启动仍 pending；本审查未重跑或读取其最终全套结果，不声称全套已完成。

本轮检查精确覆盖已发现空调用目录及相邻条目类型。快照中的 metadata ZIP 仍只是闭合输入的路径/哈希索引，不是 raw flight、真实物理状态或全 SDK 重放。没有证明对外部并发修改的原子或返回后永久不变性；本轮未增加此类承诺。原三时序窗口、SDK captured bytes、权限和统计范围沿用 v3 的已审边界。无真实 formal 数据、模型/物理性能或 G4 验收。

全部自有检查已结束；本报告与 binding 冻结后停止 R 写入，不继续扩展。
