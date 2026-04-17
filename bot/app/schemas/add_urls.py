from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class AddUrlsErrorCode(str, Enum):
    FETCH_ERROR = "FETCH_ERROR"
    BLOCKED_URL = "BLOCKED_URL"
    INVALID_URL = "INVALID_URL"
    UNSUPPORTED_SCHEME = "UNSUPPORTED_SCHEME"
    MISSING_HOST = "MISSING_HOST"
    INVALID_IP = "INVALID_IP"
    BLOCKED_PRIVATE_IP = "BLOCKED_PRIVATE_IP"
    BLOCKED_LOOPBACK = "BLOCKED_LOOPBACK"
    BLOCKED_LINK_LOCAL = "BLOCKED_LINK_LOCAL"
    BLOCKED_INTERNAL_HOST = "BLOCKED_INTERNAL_HOST"


ADD_URLS_ERROR_EXPLANATIONS = {
    AddUrlsErrorCode.FETCH_ERROR.value: "目标地址访问失败或内容解析失败，请检查URL可访问性、SSL配置和页面结构。",
    AddUrlsErrorCode.BLOCKED_URL.value: "URL被安全策略拦截，请更换为可公开访问的HTTP/HTTPS地址。",
    AddUrlsErrorCode.INVALID_URL.value: "URL格式无法解析，请检查是否为完整地址。",
    AddUrlsErrorCode.UNSUPPORTED_SCHEME.value: "仅支持HTTP/HTTPS协议，请调整URL协议。",
    AddUrlsErrorCode.MISSING_HOST.value: "URL缺少主机名，请提供包含域名或IP的完整地址。",
    AddUrlsErrorCode.INVALID_IP.value: "IP地址格式无效，请检查IP是否正确。",
    AddUrlsErrorCode.BLOCKED_PRIVATE_IP.value: "检测到私网地址，为避免内网探测风险已拒绝访问。",
    AddUrlsErrorCode.BLOCKED_LOOPBACK.value: "检测到环回地址，为避免访问本机服务已拒绝。",
    AddUrlsErrorCode.BLOCKED_LINK_LOCAL.value: "检测到链路本地地址，为避免访问局域网络设备已拒绝。",
    AddUrlsErrorCode.BLOCKED_INTERNAL_HOST.value: "检测到疑似内网主机名（无公网域名特征），已按策略拦截。",
}


class AddUrlsRequest(BaseModel):
    """/add_urls 与 /add_urls/dry_run 请求体。"""

    url: Optional[str] = Field(default=None, description="单个URL；缺失/None/空字符串视为未提供")
    urls: List[str] = Field(default_factory=list, description="批量URL列表；缺失=空列表，列表中的空白URL会被忽略")
    chunk_strategy: Literal["balanced", "faq", "article", "custom"] = Field(
        default="balanced",
        description="切块策略：balanced|faq|article|custom",
    )
    chunk_size: Optional[int] = Field(default=None, ge=100, le=4000, description="可选；缺失/None时沿用策略默认值")
    chunk_overlap: Optional[int] = Field(default=None, ge=0, le=1000, description="可选；缺失/None时沿用策略默认值")
    separators: Optional[List[str]] = Field(default=None, description="仅custom策略建议传入；缺失/None/空列表时沿用策略默认分隔符")
    preview_limit: int = Field(default=3, ge=1, le=20, description="dry_run时返回示例chunk数量")


class FailedUrlItem(BaseModel):
    """单个失败URL信息。"""

    url: str = Field(description="失败URL")
    code: AddUrlsErrorCode = Field(default=AddUrlsErrorCode.FETCH_ERROR, description="失败类型编码")
    error: str = Field(description="失败原因")
    explanation: str = Field(default="", description="错误解释与处理建议")


class ChunkConfigModel(BaseModel):
    """本次请求实际生效的切块配置。"""

    chunk_size: int = Field(description="实际chunk_size")
    chunk_overlap: int = Field(description="实际chunk_overlap")
    separators: List[str] = Field(description="实际分隔符")


class ChunkPreviewItem(BaseModel):
    """dry_run 预览项。"""

    source_url: str = Field(description="来源URL")
    chunk_index: int = Field(description="chunk序号")
    content_length: int = Field(description="chunk长度")
    content_preview: str = Field(description="chunk预览")


class AddUrlsResponse(BaseModel):
    """/add_urls 响应体。"""

    response: str
    collection: str
    mode: Literal["remote", "local"]
    source_urls: int
    chunks: int
    failed_urls: List[FailedUrlItem]
    chunk_strategy: Literal["balanced", "faq", "article", "custom"]
    chunk_config: ChunkConfigModel
    quality_report: Dict[str, Any]


class AddUrlsDryRunResponse(BaseModel):
    """/add_urls/dry_run 响应体。"""

    response: str
    source_urls: int
    chunks: int
    failed_urls: List[FailedUrlItem]
    chunk_strategy: Literal["balanced", "faq", "article", "custom"]
    chunk_config: ChunkConfigModel
    chunk_preview: List[ChunkPreviewItem]
    quality_report: Dict[str, Any]
