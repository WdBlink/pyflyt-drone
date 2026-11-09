# Development catalog component：R1 有限修复复验 v2

结论：**关闭 v1 报告的 R1**。冻结候选已把“保存历史输入身份”与“通过现行 mission 准入”分开；本次限定复验没有发现新的实质缺陷。该结论只针对未安装 producer 组件，不批准实际 catalog、全部使用史覆盖、正式数据生成或 G4。

## 绑定与实际执行

- reviewer：`/root/statistics_design_review`，独立上下文、同模型家族；不是跨模型或人工复核。
- 输入 freeze：`planning/formal-admission-candidate.v3/catalog-component-freeze.v2/binding.json`，SHA256 `b0851db1a9e274ff39e74f1b8e29a2f4e1055853872b5fbcb201bd3c18d52aff`。其中 4 个文件与 contract 的 5 个 dependency SHA 均重新核对一致；working producer/test 与冻结副本逐字节相同。
- producer SHA256：`b3a2d91f3589b0b1bc3f7da8518637b99620e0c3099f4bd8d2542b4ff0a5fe08`；test：`98de52d0a8677891e2de5077e40f4a3d2797f91edd22b45148d923fbcac3aeca`。
- `eval/inspection_cases.py`、`eval/inspection_population.py`、`eval/prepare_inspection_development_population.py` 与 v1 独立报告绑定字节相同。现行 loader/物理检查未因本修复改变。

独立执行（repo cwd）：

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -p no:cacheprovider paper/research/20261104-inspection/planning/formal-admission-candidate.v3/tests/test_development_catalog.py --basetemp=/tmp/development-catalog-review-tests-v2 -q
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B /tmp/development-catalog-repair-review-v2.py
```

结果：**16 passed in 5.11s**；另附独立 probe 完整脚本与精简 JSON。Python 3.11.15 / NumPy 1.26.4。新增检查只读取既有输入、在内存构造 catalog 和明确合成的变体；没有落盘实际 catalog。probe 将 `default_rng` / `SeedSequence` 替换为抛错函数，未调用 RNG、模型、物理、formal generator 或 owner CLI。

## 原真实反例及身份重算

读取 v1/v2 dev suites、5-root dev population，以及退役 clean、当前 event/clean 的显式 case：共 **56 条 artifact records**，不是 56 个独立 root，也不是新增实验。每个 record 的 semantic SHA 均与原始 JSON 的 sorted/default-separator canonical 重算一致。55 条被现行 validator 接受的记录，归档解码与执行 loader 的 semantic/configuration 摘要全部一致；剩余 1 条保留为 rejected metadata。

原 `qualification/dev-cases.v1/dev-L2-03.json` 现在进入库存，同时保留：

| 字段 | 独立核对值 |
|---|---|
| 原 file SHA256 | `12806f5d90bde3a4b8ac99d89d91c00c5ca7dc95d6d85029dafdb83243b5bc49` |
| 原 semantic SHA256 | `ed93254799d5b4634b9ed50c974f21f67a62b387f189f35a1d1f38c3e4d38947` |
| configuration SHA256 | `43404aa97ed8857bfcafd0a90330a9c78ef27560cb659ac4326164795e357a84` |
| 当前 mission metadata | `rejected_by_current_mission_validator` |
| reason / 原 loader 实际抛错 | `ground proxies cannot be certified separated between samples` |

configuration 另从原 JSON 独立删除 CaseSpec 身份字段、实体 key 并排序实体后重算，与 producer 相同；未通过当前 loader 来取得这一历史身份。旧输入的失败没有被修复版 input 覆盖，也没有被 catalog 当成可执行 flight。

## 结构、clean parent 与准入边界

新增 `_archived_identity_case` 仅接受 3 个已知 schema；主 record 要 exact dataclass field set、dev split、native unsigned128 seed、明确字符串/数值/事件类型。JSON artifact 的既有有界、hash/path/symlink 防护仍由未变的 Artifact helper 执行。它不以当前几何/事件可执行性筛掉历史输入。

除 suite 已有 unknown-schema、boolean coordinate、nonfinite coordinate 外，独立 probe 的 12 个负例均拒绝：未知 schema、boolean/negative seed、boolean event time、boolean aircraft、boolean entity speed、infinite entity coordinate、额外顶层字段、嵌套 clean parent、parent 额外字段、boolean parent seed、boolean parent digest。

clean parent 只允许 population schema；后者 exact field set 不含 `parent_case`，因此该读取路径最多一层 parent，不接受递归 clean 包装。规范化仍用原 dataclass/as_dict/digest；物理 configuration 通过 CaseSpec 字段投影，不把 parent 的 ID/seed/provenance 当新物理配置。

**归档 dataclass 类型不代表通过准入。** 独立把真实 clean 的 parent SHA 改为格式合法的全零：归档身份容器可表达该历史坏关系，但原 loader 实际拒绝 `clean intervention requires an intact L3 development parent`。这符合库存用途；producer 对外只返回带 `identity_role=archived_input_not_mission_qualification` 及 accepted/rejected metadata 的字典，未提供可执行 admission。现行任务仍必须走原 loader，不能拿 catalog status 或 dataclass 构造替代它。

原 8 项输入拒绝、duplicate provenance 的 richer-merge/冲突拒绝、clean parent metadata/config 分离、错误 NumPy 版本前置拒绝及 no-RNG 测试均在本次 16 项中重跑通过。probe 所读源输入 SHA 在完成时不变，未产生 datasets/catalog。

## 保留限制

本轮只复验 R1 和上述新增解析边界；未扩展 v1 的 used-input 扫描。56 条是显式来源的 artifact 版本/别名集合，不能据此认定全部曾尝试、退役或私有副本已经覆盖。实际 run `private-case` origins 的完整逐项对账、source-bound 归档和最终 catalog 冻结仍由后续独立步骤完成；v1 的这一覆盖限制继续有效。

无实际 flight/model/trace 复验，无正式 seed/data 生成，无安装或 owner/state 更改，不恢复旧失败槽，不授 G4/main/novelty approval。结论只关闭 producer 的已复现历史无效输入漏收问题。
