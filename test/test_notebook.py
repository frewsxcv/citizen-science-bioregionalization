import runpy
import sys
import unittest
from unittest.mock import patch


class TestNotebook(unittest.TestCase):
    def test_notebook_runs(self):
        # Set up command line arguments using mo.cli_args() format (--key=value)
        sys.argv = [
            "notebook.py",
            "--",
            "--geocode-precision=9",
            "--min-clusters=2",
            "--max-clusters=10",
            "--log-file=run.log",
            "--min-lat=40",
            "--max-lat=50",
            "--min-lon=5",
            "--max-lon=10",
            "--parquet-source-path=test/sample-archive/",
            "--no-stop",
        ]

        # Run the notebook script
        runpy.run_path("notebook.py", run_name="__main__")
