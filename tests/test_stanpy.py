#!/usr/bin/env python

"""Tests for `stanpy` package."""

import pytest
import numpy as np

from click.testing import CliRunner

from stanpy import stanpy as stp
from stanpy import cli


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string

def test_qs_rect() -> None:
    """Cross section Properties of a rectangle with b and h """
    # primitives as inputs
    b = 0.2
    h = 0.3
    qs = stp.qs(b,h,0,0)
    assert qs["a"] == b*h
    assert qs.iy == b*h**3/12
    assert qs.iz == b**3*h/12
    assert qs.iyz == 0
    assert qs.ys == b/2
    assert qs.zs == h/2
    assert qs.iy_main == qs.iy
    assert qs.iz_main == qs.iz

    # np.arrays as inputs
    b = np.array([0.2])
    h = np.array([0.3])
    qs = stp.qs(b,h,0,0)
    assert qs["a"] == b*h
    assert qs["iy"] == b*h**3/12
    assert qs["iz"] == b**3*h/12
    assert qs["iyz"] == 0
    assert qs["ys"] == b/2
    assert qs["zs"] == h/2
    assert qs["iy_main"] == b*h**3/12
    assert qs["iz_main"] == b**3*h/12

def test_cs_rect_array() -> None:
    """Cross section Properties of a Rectangle with b and h np.arrays as inputs"""
    b = np.array([0.2])
    h = np.array([0.3])
    qs = stp.qs(b,h,0,0)
    assert qs["a"] == b*h
    assert qs["iy"] == b*h**3/12
    assert qs["iz"] == b**3*h/12
    assert qs["iyz"] == 0
    assert qs["ys"] == b/2
    assert qs["zs"] == h/2
    assert qs["iy_main"] == b*h**3/12
    assert qs["iz_main"] == b**3*h/12

def test_cs_I() -> None:
    b = np.array([0.2])
    h = np.array([0.3])
    qs = stp.qs(b,h,0,0)
    assert qs["a"] == b*h
    assert qs.iy == b*h**3/12
    assert qs.iz == b**3*h/12
    assert qs.iyz == 0
    assert qs.ys == b/2
    assert qs.zs == h/2
    assert qs.iy_main == qs.iy
    assert qs.iz_main == qs.iz

def test_mat() -> None:
    """check if all combinations of input paramaters return expected dict"""

    assert 0 == stp.mat(E=3E10)

def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert 'stanpy.cli.main' in result.output
    help_result = runner.invoke(cli.main, ['--help'])
    assert help_result.exit_code == 0
    assert '--help  Show this message and exit.' in help_result.output
