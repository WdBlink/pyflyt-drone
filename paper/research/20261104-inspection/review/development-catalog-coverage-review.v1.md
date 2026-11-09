# Development identity catalog：持久化覆盖独立审查 v1

结论：**在下述声明目录、持久化输入类型及本次冻结快照中，未发现未映射输入或身份/seed 计数偏差。** 实际 catalog 的 56 个 artifact 版本、32 个 semantic 身份、32 个 physical configuration、120 个排除 seeds 均独立重算一致。该结论支持当前持久化碰撞库存的有限覆盖声明，不是通用历史 RNG 审计、flight/计量准入或 G4 放行。

## 来源、方法与执行

reviewer 为 `/root/statistics_design_review`，独立上下文、同模型家族，不是人工或跨模型审查。使用 autoresearch 的证据边界要求，仅复核既有工程证据；未推进模型/物理/正式生成。

- 本次输入：`qualification/development-identity-catalog.v1/results.v2/binding.json` SHA256 `f90c3d2f06f0721cc9313cdc3fa0cdb74962140a7806960550d026858f187580`。
- `catalog.json` SHA256 `d77a0a2ff297423efdc25c4fdcef16fecfc32a064b4653f4533180d8b46685cb`；`coverage.json` SHA256 `972a0872fec0ba5c292777f3cb937f4ff22c7d6a77bad0b9b9e54736028f101a`，均与 binding 相符。
- adapter `source/build_catalog.py` SHA256 `356c17b9a89237ca2b970e3d80f44f5fe57c4d1bb8bab8f4fbad3c0d326de2bc`；其调用的 producer 为已审 v2 SHA `b3a2d91f3589b0b1bc3f7da8518637b99620e0c3099f4bd8d2542b4ff0a5fe08`。coverage 列出的 6 个主 source SHA 全部与当前字节一致；本审查另记录实际导入的 scenario/receipt helper SHA 作为复算上下文，不宣称正式执行完整闭包认证。
- 已读 adapter 全部 134 行，并核对保留的 v1 source：唯一 diff 是将 `datasets_before` 采样移到 producer 调用之前；没有借此改输入/身份计算。**v1/v2 catalog 文件逐字节相同。** 原 v1 及此前 R1 报告未修改。

在 repo cwd 实际运行：

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B /tmp/development-catalog-coverage-independent-v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B /tmp/development-catalog-coverage-named-input-scan-v1.py
```

两个独立脚本均 exit 0，Python 3.11.15 / NumPy 1.26.4。脚本与结果保存于本报告的 evidence 目录。没有运行 adapter 的 writer `main()`；仅调用已审 producer 的只读函数在内存重建 catalog，再与现有文件作精确字节比较。

## 显式输入的全分母与身份

独立递归枚举 qualification 中声明 `split=dev` 的 manifest，找到以下全部 6 份，集合精确等于 catalog refs；不是从作者给出的 root 总数推断覆盖：

| 持久化来源 | artifact records |
|---|---:|
| `qualification/dev-cases.v1/manifest.json` | 24 |
| `qualification/dev-cases.v2/manifest.json` | 24 |
| `qualification/dev-population.v1/manifest.json` | 5 |
| `qualification/dev-clean-sibling.v1/manifest.json`（退役变体） | 1 |
| `qualification/dev-same-level-event-pair.v2/event/manifest.json` | 1 |
| `qualification/dev-same-level-event-pair.v2/clean/manifest.json` | 1 |
| 合计 | 56 |

逐条核对 manifest/file SHA、已知 schema、dev 标记和原始 canonical semantic SHA；从原 JSON 的 CaseSpec 字段独立删除身份字段、实体 key 并排序实体，重算 physical configuration SHA。56 条全部一致，去重后恰为 32 semantic / 32 physical。再次调用原 loader 得到 **55 accepted、1 rejected**，与 catalog metadata 一致；这些状态不取代实际 flight admission。

旧 `dev-cases.v1/dev-L2-03.json` 仍保存 file SHA `12806f5d90bde3a4b8ac99d89d91c00c5ca7dc95d6d85029dafdb83243b5bc49`、semantic `ed93254799d5b4634b9ed50c974f21f67a62b387f189f35a1d1f38c3e4d38947`、config `43404aa97ed8857bfcafd0a90330a9c78ef27560cb659ac4326164795e357a84` 及原拒绝原因 `ground proxies cannot be certified separated between samples`。原 loader 本次仍实际抛错，未放宽规则。

退役 clean（当时 public level L2）与当前同级 L3 clean 都保留；没有把它们互相替代。event 的重复源路径仍按 artifact 记录，身份集合再去重。**56、32 均不代表独立 root/flight 分母**；invalid、alias、版本和 sibling 不被重标成独立样本。

## 磁盘 private origins 与归档

独立遍历 `logs/` 和 `R/qualification/` 的全部 `private-case.json`，没有套用 adapter 的 `source`/`freeze.v1` 排除规则；实际未发现这两类被排除路径中的额外副本。枚举集合与 coverage 精确相同：

| 范围 | 实际副本 |
|---|---:|
| repo `logs/` | 45 |
| R `qualification/` | 6 |
| 合计 | 51 |

51 份均逐字节 SHA 匹配已知显式来源，且 catalog_paths 的完整别名集合正确。50 份有相邻 `run.json`，其文件 SHA 与 coverage 绑定一致，`private_case_sha256` 精确等于副本；剩下固定负载诊断 `logs/inspection-clock-localization-fixed-load.v1/private-case.json` 没有相邻 run，但本次独立读得 `diagnostic.json.case_sha256` 精确匹配。后者是诊断来源绑定，不被补称成普通 flight receipt。

另独立枚举全部 **29 个 qualification ZIP**（整包合计 416,970,121 bytes），每包完整 SHA 与 coverage 一致；读取其中全部 **36 个 private-case 成员**，解压/CRC 检查通过，成员 bytes SHA 和 semantic 均映射到 catalog，未发现重复 private 成员名。没有解压写盘，也没有把重复归档副本计为新 flight。整包 SHA 绑定不等于逐项验证 ZIP 中其他非输入成员的语义。

为核验约定命名之外的明显输入副本，额外扫描两个目录的 `*case*.json` / `dev*.json`，按 3 个已知 CaseSpec schema 筛选：除 private 副本外恰为上述 56 份源文件，全部已知。另读取 ZIP 内 326 个其他 case/dev 命名 JSON 候选，没有发现其他 CaseSpec 成员；它们不被当成新增输入。该有界补查没有宣称任意文件名、任意外部存储的完整历史。

## 保守 seed 排除与无生成边界

独立从 6 份 manifest、实际 case sensor seed、root-generation seed 及绑定的 namespace intent 重算基础 **36** 个 native unsigned128 seeds。另独立枚举 reference 的全部 10 份 run manifest，以及 qualification 中全部带 grid 的 20 份 manifest（枚举不限 `g1-` / `executor-` 前缀；实际集合与 coverage 相同），依据已持久化整数 seed / 非 unstarted 的 grid root 重算 **84** 个历史 procedural 参数 seeds。两集合无交，最终并集精确为 **120**，与 catalog 一致。

这些 procedural 参数属于旧 control/executor/reference 法则，只是额外碰撞排除；不是正式 benchmark 独立 roots，也不是 84 个实际独立 flight。重复出现的 seed 按集合保存，没有重复累加。未假定未知成本为 0，也未借本报告认证 usage、receipt 或预算。

无 RNG 调用由本次只读路径的源码检查及 `default_rng` / `SeedSequence` / `secrets.randbits` 抛错替换共同核验；未调用 generator 函数。producer 重建结果的所有字段和重新序列化 bytes 与现有 catalog 精确一致。显式输入/副本在 probe 结束时 SHA 不变，`R/datasets` 在开始和结束均不存在，未生成正式 seeds/datasets、模型、物理结果或 owner 状态。

## 结论边界与后续义务

本次未发现当前声明范围内的实际漏项、摘要失配或排除 seed 计数错误；此前 R1 的历史无效输入漏收问题保持关闭。可据此表述“本次快照中，列明的持久化 dev 来源、logs/qualification 的 private-case 副本及 qualification ZIP 输入均已对账”。

不能据此表述“所有历史输入/RNG 使用都已覆盖”：未持久化、已删除、树外 `/tmp`/其他 run/外部存储，以及不符合已核命名/schema 的未知表示均未被证明。未来新增或改变输入、archive、source 后需要重新冻结并复核覆盖。catalog 只是资格候选；实际 formal caller 的 source/namespace/admission 接线与其他 G4 条件另行验收。本审查不实现、不安装，不批准 G4/main，不恢复旧失败槽，也不将 dev/diagnostic 结果提升为正式成绩。
