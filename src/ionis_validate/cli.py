"""
cli.py — IONIS Validation CLI entry point

Dispatches subcommands to the appropriate runner module.
Works on Windows, Mac, and Linux.

Usage:
  ionis-validate test
  ionis-validate predict --tx-grid FN20 --rx-grid IO91 --band 20m ...
  ionis-validate custom my_paths.json
  ionis-validate adif my_log.adi
  ionis-validate report
  ionis-validate info
"""

import importlib
import sys


COMMANDS = {
    "test":    "ionis_validate.tests.run_all",
    "predict": "ionis_validate.tests.run_predict",
    "custom":  "ionis_validate.tests.run_custom",
    "adif":    "ionis_validate.tests.run_adif",
    "report":  "ionis_validate.tests.run_report",
    "info":    "ionis_validate.tests.run_info",
    "ui":      "ionis_validate.ui.app",
}

USAGE = """\
ionis-validate — IONIS V20 HF Propagation Model Validation Suite

Usage:
  ionis-validate <command> [options]

Commands:
  test       Run the full 62-test validation suite
  predict    Predict SNR for a single HF path
  custom     Run batch predictions from a JSON file
  adif       Validate the model against your ADIF QSO log
  report     Generate a beta test report for GitHub Issues
  info       Show model and system information
  ui         Launch browser-based validation dashboard

Examples:
  ionis-validate test
  ionis-validate predict --tx-grid FN20 --rx-grid IO91 --band 20m --sfi 150 --kp 2 --hour 14 --month 6
  ionis-validate adif my_log.adi
  ionis-validate info

https://github.com/IONIS-AI/ionis-validate
"""


def main():
    command = sys.argv[1] if len(sys.argv) > 1 else "help"

    if command in ("help", "--help", "-h"):
        print(USAGE)
        sys.exit(0)

    if command == "--version":
        from ionis_validate import __version__
        print(f"ionis-validate {__version__}")
        sys.exit(0)

    if command not in COMMANDS:
        print(f"Unknown command: {command}\n")
        print(USAGE)
        sys.exit(1)

    # Strip the subcommand from argv so argparse in the target module
    # sees only its own arguments
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    module = importlib.import_module(COMMANDS[command])
    sys.exit(module.main() or 0)
