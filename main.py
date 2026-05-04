"""OpenAI 兼容层插件入口。"""

from __future__ import annotations

from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, register

from .compat import (
    is_deepseek_reasoning_target,
    normalize_deepseek_tool_call_history,
    normalize_tool_set_schemas,
)

PLUGIN_NAME = "astrbot_plugin_openai_compat"


@register(
    PLUGIN_NAME,
    "Codex",
    "为 AstrBot 内置 Agent 的 OpenAI 兼容接口修复工具调用与推理字段格式。",
    "0.1.0",
)
class OpenAICompatPlugin(Star):
    """为标准 LLM 流程提供 OpenAI 兼容请求修复。"""

    def __init__(self, context: Context, config: dict[str, Any] | None = None) -> None:
        super().__init__(context)
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", True))
        self.deepseek_enabled = bool(
            self.config.get("enable_deepseek_reasoning_fix", True)
        )
        self.tool_schema_enabled = bool(
            self.config.get("enable_tool_schema_fix", True)
        )
        self.deepseek_model_keywords = self._read_string_list(
            self.config.get("deepseek_model_keywords", [])
        )

    async def initialize(self) -> None:
        """记录插件启动配置，便于排查是否已启用。"""

        logger.info(
            "[OpenAICompat][startup] enabled=%s deepseek_fix=%s tool_schema_fix=%s extra_keywords=%s",
            self.enabled,
            self.deepseek_enabled,
            self.tool_schema_enabled,
            self.deepseek_model_keywords,
        )

    @filter.on_llm_request(priority=-20000)
    async def normalize_openai_compat_request(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
    ) -> None:
        """在 AstrBot 内置 Agent 调用 LLM 前统一修复 OpenAI 兼容请求。"""

        if not self.enabled:
            return

        changed_tool_schemas = 0
        if self.tool_schema_enabled and req.func_tool is not None:
            changed_tool_schemas = normalize_tool_set_schemas(req.func_tool)

        changed_reasoning_messages = 0
        model_name = self._resolve_request_model_name(event, req)
        if self.deepseek_enabled and is_deepseek_reasoning_target(
            model_name,
            self.deepseek_model_keywords,
        ):
            changed_reasoning_messages = normalize_deepseek_tool_call_history(
                req.contexts
            )

        if changed_tool_schemas or changed_reasoning_messages:
            logger.info(
                "[OpenAICompat][request] 修复完成 umo=%s model=%s tool_schema_count=%s reasoning_history_count=%s",
                event.unified_msg_origin,
                model_name or "",
                changed_tool_schemas,
                changed_reasoning_messages,
            )

    def _resolve_request_model_name(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
    ) -> str:
        """解析本轮请求实际使用的模型名。"""

        request_model = str(req.model or "").strip()
        if request_model:
            return request_model

        provider = None
        try:
            provider = self.context.get_using_provider(event.unified_msg_origin)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[OpenAICompat] 获取当前 Provider 失败: %s", exc)

        if provider is None:
            return ""

        get_model = getattr(provider, "get_model", None)
        if callable(get_model):
            try:
                return str(get_model() or "").strip()
            except Exception as exc:  # noqa: BLE001
                logger.debug("[OpenAICompat] 获取 Provider 模型名失败: %s", exc)

        provider_config = getattr(provider, "provider_config", None)
        if isinstance(provider_config, dict):
            return str(provider_config.get("model", "") or "").strip()

        return ""

    @staticmethod
    def _read_string_list(value: Any) -> list[str]:
        """从配置读取字符串列表，非法项会被忽略。"""

        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        return [
            str(item or "").strip()
            for item in value
            if str(item or "").strip()
        ]
