import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from snowball_notes.config import load_config


class ConfigEnvFileTests(unittest.TestCase):
    def test_load_config_reads_default_snowball_env_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            home = Path(temp_dir)
            project_root = home / "project"
            project_root.mkdir(parents=True, exist_ok=True)
            config_path = project_root / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "agent:",
                        "  provider: \"deepseek_v3\"",
                        "  model: \"deepseek-chat\"",
                    ]
                ),
                encoding="utf-8",
            )
            (home / ".snowball-notes.env").write_text(
                "\n".join(
                    [
                        f"export SNOWBALL_CONFIG=\"{config_path}\"",
                        "export DEEPSEEK_API_KEY=\"from-file\"",
                    ]
                ),
                encoding="utf-8",
            )
            with mock.patch("pathlib.Path.home", return_value=home), mock.patch.dict(os.environ, {}, clear=True):
                config = load_config()
                self.assertEqual(os.environ.get("DEEPSEEK_API_KEY"), "from-file")
            self.assertEqual(config.project_root, project_root.resolve())
            self.assertEqual(config.agent.provider, "deepseek_v3")

    def test_existing_environment_wins_over_snowball_env_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "config.yaml"
            config_path.write_text("agent:\n  provider: \"heuristic\"\n", encoding="utf-8")
            env_path = root / ".snowball-notes.env"
            env_path.write_text("export DEEPSEEK_API_KEY=\"from-file\"\n", encoding="utf-8")
            with mock.patch.dict(
                os.environ,
                {
                    "SNOWBALL_ENV_FILE": str(env_path),
                    "DEEPSEEK_API_KEY": "from-env",
                },
                clear=True,
            ):
                load_config(config_path)
                self.assertEqual(os.environ.get("DEEPSEEK_API_KEY"), "from-env")

    def test_blank_snowball_env_file_override_disables_autoload(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            home = Path(temp_dir)
            config_path = home / "config.yaml"
            config_path.write_text("agent:\n  provider: \"heuristic\"\n", encoding="utf-8")
            (home / ".snowball-notes.env").write_text(
                "export DEEPSEEK_API_KEY=\"from-file\"\n",
                encoding="utf-8",
            )
            with mock.patch("pathlib.Path.home", return_value=home), mock.patch.dict(
                os.environ,
                {"SNOWBALL_ENV_FILE": ""},
                clear=True,
            ):
                load_config(config_path)
                self.assertIsNone(os.environ.get("DEEPSEEK_API_KEY"))
