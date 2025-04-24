"""
Unit and regression test for the graphbuilder package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import graphbuilder


def test_graphbuilder_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "graphbuilder" in sys.modules
