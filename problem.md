# 问题复盘与功能整理（2026-04-17）

## 1. 背景
- 本轮工作目标：对 FastAPI 项目进行路由拆分、主入口收敛、冗余代码清理，并补充回归测试以降低后续重构风险。

## 2. 本轮新增/调整的功能

### 2.1 路由层结构化拆分
- 新增并启用路由模块：
	- `app/api/routers/conversation.py`
	- `app/api/routers/ingestion.py`
	- `app/api/routers/ops.py`
	- `app/api/routers/tools.py`
- `app/main.py` 改为以应用装配为主：通过 `app.include_router(...)` 统一挂载路由。

### 2.2 调试 UI 外置
- 将调试页面 HTML 从内联字符串迁移到静态文件：
	- `app/api/static/debug_ui.html`

### 2.3 共享依赖解析能力抽取
- 新增共享模块：
	- `app/api/deps.py`
- 抽取并统一复用：
	- 运行时依赖解析：`resolve_runtime_dependency`
	- add_urls 请求参数归一化：`resolve_add_urls_payload`

### 2.4 测试补强
- 新增路由拆分专项回归测试：
	- `tests/test_router_split_coverage.py`
- 补充了 tools/ops/conversation 的关键分支与兼容路径验证。

## 3. 遇到的问题、根因与解决方法

### 问题 1：清理后测试失败（`main.Qdrant` 缺失）
- 现象：`tests/test_add_urls_contract.py` 中多条用例失败，报错为 `AttributeError: app.main has no attribute 'Qdrant'`。
- 根因：为精简 `main.py` 导入时移除了 `Qdrant` 符号，但现有测试通过 `monkeypatch(main.Qdrant, ...)` 进行兼容替换。
- 解决方法：
	- 在 `app/main.py` 恢复 `Qdrant` 导出。
	- 增加回归测试，显式校验 `main` 仍导出 `Qdrant` 以保障兼容契约。

### 问题 2：WebSocket 路由存在运行时断链风险
- 现象：`conversation` 路由的 `/ws` 通过运行时依赖读取 `main.ensure_websocket_auth` 和 `main.ensure_ws_rate_limit`。
- 根因：拆分过程中 `main` 精简曾清掉相关导入，导致路由运行时可能拿到 `None`，触发服务不可用分支。
- 解决方法：
	- 在 `app/main.py` 恢复 `ensure_websocket_auth`、`ensure_ws_rate_limit` 符号导出。
	- 新增 `/ws` 成功路径、默认回退路径与服务不可用分支测试。

### 问题 3：请求参数归一化逻辑重复，存在后续分叉风险
- 现象：`_resolve_add_urls_payload` 在 `main.py` 与 `ingestion.py` 中重复实现。
- 根因：拆分过程中优先保证兼容，导致重复逻辑遗留。
- 解决方法：
	- 新增 `app/api/deps.py`，统一实现 `resolve_add_urls_payload`。
	- `ingestion.py` 改为直接依赖共享函数。
	- 删除 `main.py` 中重复版本，降低后续维护成本。

### 问题 4：运行时依赖解析函数重复
- 现象：`conversation.py`、`ops.py`、`ingestion.py` 各自维护同名 `_resolve_runtime_dependency`。
- 根因：拆分初期以“快速迁移可跑通”为优先，未及时抽公共函数。
- 解决方法：
	- 将函数统一抽取到 `app/api/deps.py`。
	- 三个路由统一引用共享实现。

## 4. 验证与结果
- 已执行：
	- `run_tests.ps1 -q tests/test_router_split_coverage.py`
	- `run_tests.ps1 -q`
- 结果：全量测试通过。

## 5. 当前状态结论
- 主入口职责更清晰：`main.py` 以装配与兼容导出为主。
- 路由拆分完成并稳定运行。
- 重复代码已进一步收敛（共享 deps 模块落地）。
- 关键兼容路径（尤其 monkeypatch 依赖的符号）已通过测试锁定。

## 6. 后续建议
- 中长期建议将“路由对 main 的运行时依赖”继续演进为显式 DI（如 `app.state` 或统一 provider），减少 `getattr(main, ...)` 反射依赖。
- 若后续继续清理 `main.py`，应保留“兼容导出清单”并同步更新对应回归测试，避免再次出现符号断链。
