# 有界helper guard独立复验 v1

日期：2026-11-05。**B1已关闭。** 本次仅复验NumPy布尔拒绝的最小修补，不改变旧审查报告，不扩大为G4或真实数据认证。

| 绑定文件 | SHA256 |
| --- | --- |
| `eval/inspection_statistics.py` | `70c8783be6f6665b81b1190d6b905f4a22938445cc4c641458202bab5c710545` |
| `tests/test_inspection_statistics.py` | `e7932fde257fbcc94b33008503090d13830a5cfcad0e6da5f36fd0ebc58bb7f9` |

将当前统计源码与`statistics-preflight.v2/source-snapshot.json`绑定的`4a6610…`版本作全文diff，只有两行变化：delta与W的`isinstance(x,bool)`均改为`isinstance(x,(bool,np.bool_))`；公式、权重、主字段、区间和bootstrap路径未改。

实际独立检查（项目Python3.11 / NumPy1.26.4）：

- `np.bool_(True/False)`分别作为delta或W的4个调用均抛`ValueError`，原B1反例不再生成p。
- 从旧快照AST单独载入旧helper，对36组合合法Python/NumPy实数输入逐项比较，新旧返回字典完全相同。
- 独立运行`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_inspection_statistics.py`：**11 passed in 0.12s**。新增参数化例核验NumPy及Python布尔的delta/W拒绝。
- `statistics-preflight.v3/report.json`绑定的3份源码、其source snapshot及当前文件SHA逐项一致；该预检仍为显式合成证据。

决策轨迹：全文diff限定修补范围；独立复现原反例确认拒绝；旧函数合法输入逐项相等；现有11测试通过，据此仅关闭B1。未重跑800组压力检查、物理或模型/API；旧bootstrap反例、有限界前提、低功效及全部真实输入/接线限制继续沿用先前报告。

Attention

reviewed by inherited-parent model（同模型家族独立上下文，非跨模型家族复核）

本结论仅绑定上述SHA。真实generator/episode/provider/费用/硬恢复/SR/事件/删失及G4均未获认证。
