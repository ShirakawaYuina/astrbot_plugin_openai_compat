"""OpenAI 兼容请求的纯函数修复逻辑。"""

from __future__ import annotations

from typing import Any

DEFAULT_DEEPSEEK_REASONING_KEYWORDS = (
    "deepseek-reasoner",
    "deepseek-v4-flash",
    "deepseek-v4-pro",
)


def is_deepseek_reasoning_target(
    model_name: str | None,
    extra_keywords: list[str] | tuple[str, ...] | None = None,
) -> bool:
    """判断模型名是否需要 DeepSeek 推理工具调用历史兼容。"""

    normalized_model_name = str(model_name or "").strip().lower()
    if not normalized_model_name:
        return False

    keywords = [*DEFAULT_DEEPSEEK_REASONING_KEYWORDS]
    if extra_keywords:
        keywords.extend(
            str(keyword or "").strip().lower()
            for keyword in extra_keywords
            if str(keyword or "").strip()
        )

    return any(keyword in normalized_model_name for keyword in keywords)


def normalize_deepseek_tool_call_history(contexts: list[dict[str, Any]]) -> int:
    """为 DeepSeek 工具调用 assistant 历史补齐 reasoning_content。"""

    changed_count = 0
    for message in contexts:
        if not isinstance(message, dict):
            continue
        if message.get("role") != "assistant":
            continue
        if not message.get("tool_calls"):
            continue
        if "reasoning_content" in message:
            continue

        reasoning_content = _extract_think_parts(message.get("content"))
        message["reasoning_content"] = reasoning_content
        changed_count += 1

    return changed_count


def normalize_tool_set_schemas(tool_set: Any) -> int:
    """规范 ToolSet 内所有工具的参数 schema，避免中转站严格校验失败。"""

    tools = getattr(tool_set, "tools", None)
    if not isinstance(tools, list):
        return 0

    changed_count = 0
    for tool in tools:
        before_repr = repr(getattr(tool, "parameters", None))
        normalized_parameters = normalize_tool_parameters(
            getattr(tool, "parameters", None)
        )
        if getattr(tool, "parameters", None) != normalized_parameters:
            setattr(tool, "parameters", normalized_parameters)
            changed_count += 1
            continue
        if repr(normalized_parameters) != before_repr:
            changed_count += 1

    return changed_count


def normalize_tool_parameters(parameters: Any) -> dict[str, Any]:
    """把单个工具参数 schema 规范成 OpenAI function calling 可接受的对象。"""

    if not isinstance(parameters, dict):
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    normalized_parameters = parameters
    _normalize_schema_node(normalized_parameters)
    if normalized_parameters.get("type") != "object":
        normalized_parameters["type"] = "object"

    if not isinstance(normalized_parameters.get("properties"), dict):
        normalized_parameters["properties"] = {}

    required = normalized_parameters.get("required")
    if not isinstance(required, list):
        normalized_parameters["required"] = []

    return normalized_parameters


def _normalize_schema_node(schema_node: Any) -> None:
    """递归修复 schema 节点中常见的 OpenAI 兼容性问题。"""

    if isinstance(schema_node, list):
        for item in schema_node:
            _normalize_schema_node(item)
        return

    if not isinstance(schema_node, dict):
        return

    if "required" in schema_node and not isinstance(schema_node.get("required"), list):
        schema_node["required"] = []

    properties = schema_node.get("properties")
    if "properties" in schema_node and not isinstance(properties, dict):
        schema_node["properties"] = {}
        properties = schema_node["properties"]

    if isinstance(properties, dict):
        for property_schema in properties.values():
            _normalize_schema_node(property_schema)

    _normalize_schema_node(schema_node.get("items"))
    for union_key in ("anyOf", "oneOf", "allOf"):
        _normalize_schema_node(schema_node.get(union_key))


def _extract_think_parts(content: Any) -> str:
    """从 AstrBot 内容块中提取 think 片段，缺失时返回空字符串。"""

    if not isinstance(content, list):
        return ""

    reasoning_parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") != "think":
            continue
        reasoning_parts.append(str(part.get("think", "") or ""))

    return "".join(reasoning_parts)
