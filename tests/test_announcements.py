import unittest
import importlib
import sys
import types
from unittest.mock import Mock, patch

class AnnouncementsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Keep tests offline/zero-cost even when optional CLI dependencies are missing.
        rich_module = types.ModuleType("rich")
        rich_console_module = types.ModuleType("rich.console")
        rich_panel_module = types.ModuleType("rich.panel")

        class _Console:  # pragma: no cover - simple import stub
            pass

        class _Panel:  # pragma: no cover - simple import stub
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        rich_console_module.Console = _Console
        rich_panel_module.Panel = _Panel
        cls._modules_patch = patch.dict(
            sys.modules,
            {
                "rich": rich_module,
                "rich.console": rich_console_module,
                "rich.panel": rich_panel_module,
            },
        )
        cls._modules_patch.start()
        cls.announcements_mod = importlib.import_module("cli.announcements")
        cls.cli_config = importlib.import_module("cli.config").CLI_CONFIG

    @classmethod
    def tearDownClass(cls):
        cls._modules_patch.stop()

    @patch("cli.announcements.requests.get")
    def test_fetch_announcements_returns_api_payload(self, mock_get):
        response = Mock()
        response.json.return_value = {
            "announcements": ["Maintenance tonight"],
            "require_attention": True,
        }
        response.raise_for_status.return_value = None
        mock_get.return_value = response

        result = self.announcements_mod.fetch_announcements(
            url="https://example.test/announcements",
            timeout=2.5,
        )

        self.assertEqual(result["announcements"], ["Maintenance tonight"])
        self.assertTrue(result["require_attention"])
        response.raise_for_status.assert_called_once()
        mock_get.assert_called_once_with(
            "https://example.test/announcements",
            timeout=2.5,
        )

    @patch("cli.announcements.requests.get", side_effect=Exception("network down"))
    def test_fetch_announcements_uses_fallback_on_failure(self, _mock_get):
        result = self.announcements_mod.fetch_announcements()

        self.assertEqual(result["announcements"], [self.cli_config["announcements_fallback"]])
        self.assertFalse(result["require_attention"])

    @patch("cli.announcements.getpass.getpass")
    def test_display_announcements_prompts_when_attention_required(self, mock_getpass):
        console = Mock()

        self.announcements_mod.display_announcements(
            console,
            {"announcements": ["Read carefully"], "require_attention": True},
        )

        self.assertTrue(console.print.called)
        mock_getpass.assert_called_once_with("Press Enter to continue...")

    def test_display_announcements_skips_when_empty(self):
        console = Mock()

        self.announcements_mod.display_announcements(
            console,
            {"announcements": [], "require_attention": False},
        )

        console.print.assert_not_called()


if __name__ == "__main__":
    unittest.main()
