import unittest

from click.testing import CliRunner

from sygnals.cli import cli


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_analyze_command(self):
        result = self.runner.invoke(cli, ['analyze', 'sample.csv', '--output', 'json'])
        self.assertEqual(result.exit_code, 0)

    def test_transform_command(self):
        result = self.runner.invoke(cli, ['transform', 'sample.csv', '--fft', '--output', 'output.csv'])
        self.assertEqual(result.exit_code, 0)

    def test_filter_command(self):
        result = self.runner.invoke(cli, ['filter', 'sample.csv', '--low-pass', '100', '--output', 'filtered.csv'])
        self.assertEqual(result.exit_code, 0)
