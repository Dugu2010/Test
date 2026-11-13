import argparse
import copy
import json
import os
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from conf.conf import DEFAULT_MODEL, model_supports_tools
from lib.agent import Agent
from lib.api import NvidiaAPIClient
from lib.tui import AgentTUI, _apply_settings_to_agent, _save_persisted_settings
import main as application_main
from main import _parse_args, _port
from rich.console import Console


class ApplicationTests(unittest.TestCase):
    def test_port_argument_requires_valid_range(self):
        self.assertEqual(_port("1432"), 1432)
        for value in ("0", "65536", "not-a-port"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                _port(value)

    def test_cli_options_are_parsed(self):
        args = _parse_args(["--web", "--host", "127.0.0.1", "--port", "1432"])
        self.assertTrue(args.web)
        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.port, 1432)

    def test_cli_preserves_the_invoking_workspace(self):
        fake_tui = ModuleType("lib.tui")
        fake_tui.AgentTUI = Mock(return_value=Mock())
        original_cwd = os.getcwd()

        with (
            patch.object(application_main, "_load_env"),
            patch.object(application_main, "_load_api_key_from_settings"),
            patch.object(application_main, "_ensure_dirs"),
            patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}),
            patch.dict(sys.modules, {"lib.tui": fake_tui}),
        ):
            application_main.main([])

        self.assertEqual(os.getcwd(), original_cwd)
        fake_tui.AgentTUI.return_value.run.assert_called_once_with()

    def test_default_model_sends_tool_definitions_without_network_access(self):
        client = NvidiaAPIClient()
        try:
            self.assertEqual(client.model, DEFAULT_MODEL)
            self.assertTrue(model_supports_tools(client.model))
            payload = client._build_payload(
                [{"role": "user", "content": "inspect this project"}],
                tools=True,
                stream=False,
            )
            tool_names = [tool["function"]["name"] for tool in payload["tools"]]
            self.assertIn("cmd_exec", tool_names)
            self.assertEqual(payload["tool_choice"], "auto")
        finally:
            client.session.close()

    def test_non_chat_model_is_rejected(self):
        client = NvidiaAPIClient()
        try:
            with self.assertRaises(ValueError):
                client.set_model("nvidia/nv-embed-v1")
        finally:
            client.session.close()

    def test_invalid_saved_tui_settings_are_ignored(self):
        client = SimpleNamespace(
            model=DEFAULT_MODEL,
            temperature=0.2,
            max_tokens=4096,
            top_p=0.9,
            api_key="",
            session=Mock(),
        )
        agent = SimpleNamespace(client=client)

        _apply_settings_to_agent(agent, {
            "model": [],
            "temperature": "nan",
            "max_tokens": 0,
            "top_p": 2,
            "api_key": {},
        })

        self.assertEqual(client.model, DEFAULT_MODEL)
        self.assertEqual(client.temperature, 0.2)
        self.assertEqual(client.max_tokens, 4096)
        self.assertEqual(client.top_p, 0.9)
        self.assertEqual(client.api_key, "")

    def test_settings_are_saved_atomically_and_failures_are_reported(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "settings.json"
            with patch("lib.tui.SETTINGS_FILE", str(target)):
                self.assertTrue(_save_persisted_settings({"model": DEFAULT_MODEL}))
            self.assertEqual(json.loads(target.read_text()), {"model": DEFAULT_MODEL})
            if os.name != "nt":
                self.assertEqual(target.stat().st_mode & 0o777, 0o600)

            bad_target = Path(temp_dir) / "missing" / "settings.json"
            with patch("lib.tui.SETTINGS_FILE", str(bad_target)):
                self.assertFalse(_save_persisted_settings({"model": DEFAULT_MODEL}))

    def test_tools_command_explains_model_access(self):
        output = StringIO()
        ui = AgentTUI.__new__(AgentTUI)
        ui.console = Console(file=output, width=100, force_terminal=False)
        ui.agent = SimpleNamespace(
            client=SimpleNamespace(
                tools_supported=False,
                model="google/gemma-3-27b-it",
            )
        )

        ui._cmd_tools()

        self.assertIn("disabled for the current model", output.getvalue())

    def test_tool_rejection_is_reported_and_retry_omits_tools(self):
        client = NvidiaAPIClient()
        rejected = Mock(status_code=400, text="tool schema unsupported")
        accepted = Mock(status_code=200)
        accepted.json.return_value = {"choices": [{"message": {"content": "answer"}}]}
        responses = iter([rejected, accepted])
        sent_payloads = []

        def post(_endpoint, **kwargs):
            sent_payloads.append(copy.deepcopy(kwargs["json"]))
            return next(responses)

        client.session.post = Mock(side_effect=post)
        messages = [{"role": "user", "content": "use a tool"}]

        try:
            with patch.object(client.rate_limiter, "wait_if_needed"):
                response = client.chat(messages, tools=True)

            self.assertTrue(response["_tool_fallback"])
            first_payload, second_payload = sent_payloads
            self.assertIn("tools", first_payload)
            self.assertNotIn("tools", second_payload)
        finally:
            client.session.close()

    def test_agent_displays_non_streaming_tool_fallback(self):
        agent = Agent()
        try:
            self.assertIn(os.getcwd(), agent.messages[0]["content"])
            agent.client.chat = Mock(return_value={
                "_tool_fallback": True,
                "choices": [{"message": {"content": "answer"}}],
            })
            response = agent.process("use a tool")
            self.assertIn("retried without tools", response)
            self.assertTrue(response.endswith("answer"))
        finally:
            agent.client.session.close()

    def test_streaming_tool_rejection_emits_visible_warning(self):
        client = NvidiaAPIClient()
        rejected = Mock(status_code=400, text="tool schema unsupported")
        accepted = Mock(status_code=200)
        accepted.iter_lines.return_value = iter([
            f'data: {json.dumps({"choices": [{"delta": {"content": "answer"}}]})}',
            "data: [DONE]",
        ])
        client.session.post = Mock(side_effect=[rejected, accepted])

        try:
            with patch.object(client.rate_limiter, "wait_if_needed"):
                events = list(client.chat_stream(
                    [{"role": "user", "content": "use a tool"}],
                    tools=True,
                ))

            self.assertEqual(events[0]["type"], "warning")
            self.assertTrue(any(event.get("content") == "answer" for event in events))
            second_payload = client.session.post.call_args_list[1].kwargs["json"]
            self.assertNotIn("tools", second_payload)
        finally:
            client.session.close()


if __name__ == "__main__":
    unittest.main()
