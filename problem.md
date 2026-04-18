# 问题复盘与功能整理（2026-04-18）

## 1. 背景
- 本轮工作目标：在既有路由拆分基础上，系统化建设工具层治理能力，完成工具注册规范化、调用策略统一、权限分级、参数契约增强、意图映射配置化与健康聚合可观测。

## 2. 本轮新增/调整的功能

### 2.1 工具元数据与权限分级
- 在 `app/tools/registry.py` 新增并启用工具元数据模型：`ToolMeta`。
- 工具元数据扩展字段：
	- `owner`
	- `version`
	- `risk_level`
	- `idempotent`
	- `timeout_seconds`
	- `retry_count`
	- `debug_tier`（`public` / `protected` / `internal`）
	- `requires_env`
- 新增权限查询能力：
	- `get_tool_debug_access`
	- `is_tool_debug_allowed`

### 2.2 统一调用策略与错误码规范
- 在 `app/tools/invoker.py` 增加统一策略解析：
	- 元数据默认策略
	- 全局环境变量覆盖（`TOOL_DEFAULT_TIMEOUT_SECONDS`、`TOOL_DEFAULT_RETRY_COUNT`）
	- 工具级环境变量覆盖（`TOOL_<TOOL_NAME>_TIMEOUT_SECONDS`、`TOOL_<TOOL_NAME>_RETRY_COUNT`）
- 增加统一调用埋点事件：`tool.invoke`。
- 统一运行时错误码：
	- `TOOL_TIMEOUT`
	- `TOOL_EXEC_ERROR`

### 2.3 参数契约增强
- 增强工具参数校验逻辑：
	- `schema` 模式增加 unexpected field 拦截
	- `signature` 模式同时校验必填字段缺失与额外字段
- 新增契约摘要能力：`get_tool_contract_summary`，统一输出：
	- `required_fields`
	- `optional_fields`
	- `allow_additional_fields`

### 2.4 调试接口能力扩展
- `app/api/routers/tools.py` 新增并增强：
	- `GET /tools/schema/{tool_name}`：返回结构化 `data`，包含 schema、input_example、parameter_contract、metadata、debug_access、effective_policy、runtime_checks。
	- `GET /tools/catalog`：返回全部工具目录，并携带 `intent_mapping`、每个工具的 `input_example/contract/debug_access/policy`。
	- `GET /tools/health`：返回工具健康聚合（权限、配置、契约与策略维度）。
- 旧调试路由保持兼容并标记为 `[Legacy]` + `deprecated`。

### 2.5 意图映射配置化
- `app/tools/registry.py` 增加 `get_effective_intent_tool_names`：
	- 支持 `INTENT_TOOL_MAPPING_JSON` 环境变量覆盖默认意图映射
	- 无效配置自动回退默认映射
- `Master._select_tools_by_intent` 保持原行为，但底层映射来源已可配置。

## 3. 遇到的问题、根因与解决方法

### 问题 1：目录接口返回空列表（`/tools/catalog` 的 `count=0`）
- 现象：新增 `catalog` 接口初版测试失败，`count` 为 0。
- 根因：目录构建时依赖 `tool_obj.name`，而当前注册对象存在函数形态，`name` 为空导致被过滤。
- 解决方法：
	- 改为以 `TOOL_REGISTRY` 的 key 作为工具名来源。
	- 保留对象本身仅用于 schema/策略计算。

### 问题 2：统一错误结果被误包装为成功（`ok=True`）
- 现象：`TOOL_TIMEOUT`、`TOOL_EXEC_ERROR` 的回归断言失败，返回被包装成 `OK`。
- 根因：`wrap_tool_result` 初版仅对字符串做结构化解析，字典形式错误结果未走结构化分支。
- 解决方法：
	- 增加字典类型直通解析。
	- 统一保持 `ok/code/error` 语义不丢失。

### 问题 3：`app/tools/__init__.py` 导出在自动修补时被破坏
- 现象：导入段出现嵌套结构，语法不完整风险。
- 根因：补丁冲突后自动纠错合并结果异常。
- 解决方法：
	- 立即重写该文件导入与 `__all__` 清单，恢复为单一、完整、可读导出。
	- 追加静态检查与全量测试确认无残留。

### 问题 4：参数契约在 schema 模式下对额外字段拦截不足
- 现象：工具参数模型存在时，额外字段可能被放行。
- 根因：校验流程过于依赖模型层默认行为，未在工具层做统一前置约束。
- 解决方法：
	- 在 `validate_tool_payload` 的 schema 分支增加字段白名单校验。
	- 对额外字段统一返回 `REQUEST_VALIDATION_ERROR` 分支。

## 4. 验证与结果
- 已执行：
	- `run_tests.ps1 -q`
- 结果：全量测试通过。
- 新增回归覆盖点包括：
	- 权限分级拦截（`TOOL_ACCESS_DENIED`）
	- 参数契约 extra field 拒绝
	- schema/catalog 返回契约、示例、权限与意图映射
	- tools health 聚合接口返回结构
	- 统一错误码与超时行为回归

## 5. 当前状态结论
- 工具层已从“可调用”演进为“可治理”：
	- 有统一元数据与分级权限
	- 有统一调用策略和标准错误码
	- 有可配置意图映射
	- 有可观测的目录与健康聚合接口
- 路由兼容性保持稳定：旧路由仍可用，新入口能力更完整。
- 当前实现可直接支撑后续新增工具的低风险接入。

## 6. 后续建议
- 建议为 `/tools/health` 增加 `strict=true` 门禁模式：出现不健康工具时返回非 200，以便 CI/CD 直接接入发布门禁。
- 建议将 `TOOL_INPUT_EXAMPLES` 逐步迁移为按工具自动推导 + 人工覆盖模式，减少维护成本。
- 建议补充“工具元数据完整性校验测试”（注册工具必须存在 metadata 与示例），防止新增工具遗漏治理字段。
