import unittest
from unittest.mock import patch
import sys
import runpy

class TestNotebook(unittest.TestCase):
    def test_notebook_runs(self):
        # Set up command line arguments
        sys.argv = [
            'notebook.py',
            '--geocode-precision', '9',
            '--num-clusters', '10',
            '--log-file', 'run.log',
            'test/sample-archive/'
        ]

        # Run the notebook script
        runpy.run_path('notebook.py', run_name='__main__')
