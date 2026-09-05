from __future__ import annotations

import unittest

from self_speculation import (
    is_qwen35_family,
    render_qwen_json_tool_prompt,
)


TOOLS = (
    {
        "type": "function",
        "function": {
            "name": "find_user",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
)


class FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[list[dict], dict]] = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return "rendered"


class QwenPromptTest(unittest.TestCase):
    def test_recognizes_qwen35_family_deployment_aliases(self) -> None:
        self.assertTrue(is_qwen35_family("qwen3.8-27b"))
        self.assertTrue(is_qwen35_family("qwen38-27b"))
        self.assertTrue(is_qwen35_family("Qwen3_5ForConditionalGeneration"))
        self.assertFalse(is_qwen35_family("qwen3-8b"))
        self.assertFalse(is_qwen35_family("qwen3-32b"))

    def test_renders_json_protocol_without_native_tools_argument(self) -> None:
        tokenizer = FakeTokenizer()
        messages = [{"role": "user", "content": "Find Mei"}]

        rendered = render_qwen_json_tool_prompt(
            tokenizer,
            messages,
            TOOLS,
            enable_thinking=True,
        )

        self.assertEqual(rendered, "rendered")
        rendered_messages, kwargs = tokenizer.calls[-1]
        self.assertNotIn("tools", kwargs)
        self.assertTrue(kwargs["enable_thinking"])
        self.assertIn('"name":"find_user"', rendered_messages[0]["content"])
        self.assertIn('{"name":"tool_name","arguments":', rendered_messages[0]["content"])
        self.assertNotIn("<function=", rendered_messages[0]["content"])
        self.assertEqual(messages, [{"role": "user", "content": "Find Mei"}])


if __name__ == "__main__":
    unittest.main()
