import runpy
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch


class CLIEntrypointTests(unittest.TestCase):
    def test_bin_launcher_delegates_to_application_main(self):
        project_root = Path(__file__).resolve().parents[1]
        launcher = project_root / "bin" / "ai.py"
        application = ModuleType("main")
        application.main = Mock()

        with (
            patch.dict(sys.modules, {"main": application}),
            patch.object(sys, "path", sys.path.copy()),
            patch.object(sys, "argv", [str(launcher)]),
        ):
            runpy.run_path(str(launcher), run_name="__main__")

        application.main.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
