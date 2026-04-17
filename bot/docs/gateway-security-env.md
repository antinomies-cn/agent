# 网关安全环境变量清单

## 目的
本清单用于定义生产部署时的网关安全环境变量。
默认情况下所有项均为可选；未设置时，现有行为保持不变。

## 鉴权
| 变量 | 生产建议值 | 说明 |
|---|---|---|
| GATEWAY_AUTH_ENABLED | true | 启用入口网关鉴权护栏 |
| GATEWAY_AUTH_TOKENS | 逗号分隔的高强度令牌 | 支持 Authorization Bearer 与 X-API-Key |

## 速率限制
| 变量 | 生产建议值 | 说明 |
|---|---|---|
| GATEWAY_RATE_LIMIT_ENABLED | true | 启用固定窗口限流 |
| GATEWAY_RATE_LIMIT_IP_REQUESTS | 120 | 单窗口内每个 IP 最大请求数 |
| GATEWAY_RATE_LIMIT_SESSION_REQUESTS | 60 | 单窗口内每个会话最大请求数 |
| GATEWAY_RATE_LIMIT_WINDOW_SECONDS | 60 | 固定窗口大小（秒） |

## 请求护栏
| 变量 | 生产建议值 | 说明 |
|---|---|---|
| GATEWAY_MAX_REQUEST_BODY | 2097152 | 最大请求体字节数（2 MB） |
| GATEWAY_REQUEST_TIMEOUT_SECONDS | 25 | 网关请求超时秒数，0 表示关闭 |
| GATEWAY_TRUST_PROXY_HEADERS | true | 在反向代理后信任 X-Forwarded-For 与 X-Real-IP |

## CORS
| 变量 | 生产建议值 | 说明 |
|---|---|---|
| GATEWAY_CORS_ALLOW_ORIGINS | https://your-console.example.com | 逗号分隔的来源白名单 |
| GATEWAY_CORS_ALLOW_METHODS | GET,POST,OPTIONS | 仅保留最小必要方法 |
| GATEWAY_CORS_ALLOW_HEADERS | Authorization,Content-Type,X-API-Key,X-Session-Id | 仅放行必需请求头 |
| GATEWAY_CORS_ALLOW_CREDENTIALS | false | 除非前端确实需要携带凭证，否则保持 false |

## 上线顺序
1. 先配置 CORS 白名单并验证浏览器预检请求。
2. 开启鉴权（先配置一个令牌），验证 401/200 行为。
3. 开启限流，验证 429 行为。
4. 开启超时与请求体限制，验证 504/413 行为。

## 验收项
1. 开启鉴权后，OPTIONS /chat 仍返回 CORS 头。
2. 未授权请求返回 401，且日志不泄露令牌数据。
3. 突发请求按配置触发 IP/会话级 429。
4. 超大请求返回 413。
5. 启用超时后，慢请求返回 504。
