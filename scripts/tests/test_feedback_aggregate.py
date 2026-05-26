"""Lock in the per-key fallback contract for feedback._aggregate().

This test exists because of a 9-day silent-zero bug: claude_mode emits
per-window results in a flat shape (`results[i].{n_trades, win_rate, …}`)
while train_mode emits a nested shape (`results[i].best.{…}`). The
aggregator originally only recognised the nested shape, so 45 claude iter
rows were silently aggregated to zero.

If a future change to _aggregate breaks either known shape OR the empty
case, one of these tests will fail before the change reaches the DB.

Run:
    cd ~/projects/caffe-stocks
    venv/bin/python -m unittest scripts.tests.test_feedback_aggregate -v
"""

import unittest

from scripts.feedback import _aggregate


class TestAggregateTrainModeShape(unittest.TestCase):
    """train_mode: per-window result has a 'best' sub-dict (winning threshold
    from its sweep). gate_result also carries top-level avg_annualized_return.
    """

    def test_train_mode_nested_best(self):
        gate = {
            'gate_passed': True,
            'windows_passed': 6,
            'windows_total': 7,
            # train_mode sets avg_annualized_return at the top after its sweep.
            'avg_annualized_return': 0.27,
            'results': [
                {'best': {'n_trades': 28, 'win_rate': 0.45,
                          'annualized_return': 0.23, 'max_drawdown': 0.08}},
                {'best': {'n_trades': 30, 'win_rate': 0.50,
                          'annualized_return': 0.31, 'max_drawdown': 0.05}},
            ],
        }
        agg = _aggregate(gate)
        # avg_ann comes from the (non-zero) top-level value
        self.assertAlmostEqual(agg['avg_annualized_return'], 0.27, places=4)
        # avg_wr / avg_max_dd / total_trades are computed from the bests
        self.assertAlmostEqual(agg['avg_win_rate'], (0.45 + 0.50) / 2, places=4)
        self.assertAlmostEqual(agg['avg_max_dd'], (0.08 + 0.05) / 2, places=4)
        self.assertEqual(agg['total_trades'], 58)


class TestAggregateClaudeModeShape(unittest.TestCase):
    """claude_mode: per-window result is FLAT (no 'best' sub-dict). gate_result
    has top-level avg_win_rate and total_trades populated, but typically leaves
    avg_annualized_return at literal 0.0 — that field must be computed from
    the per-window values, not trusted from the top.
    """

    def test_claude_mode_flat_with_partial_top(self):
        gate = {
            'gate_passed': False,
            'windows_passed': 4,
            'windows_total': 7,
            'avg_annualized_return': 0.0,   # unfilled — must NOT be trusted
            'avg_win_rate': 0.428,          # real
            'avg_max_dd': 0.114,            # real
            'total_trades': 186,            # real
            'results': [
                {'n_trades': 23, 'win_rate': 0.348,
                 'annualized_return': -0.055, 'max_drawdown': 0.10},
                {'n_trades': 25, 'win_rate': 0.50,
                 'annualized_return': 0.30, 'max_drawdown': 0.08},
                {'n_trades': 30, 'win_rate': 0.40,
                 'annualized_return': 0.15, 'max_drawdown': 0.05},
            ],
        }
        agg = _aggregate(gate)
        # The zero top-level avg_ann is treated as unfilled → computed from windows
        expected_ann = (-0.055 + 0.30 + 0.15) / 3
        self.assertAlmostEqual(agg['avg_annualized_return'], expected_ann, places=4)
        # Non-zero top-level values are trusted
        self.assertAlmostEqual(agg['avg_win_rate'], 0.428, places=4)
        self.assertAlmostEqual(agg['avg_max_dd'], 0.114, places=4)
        self.assertEqual(agg['total_trades'], 186)

    def test_claude_mode_flat_no_top_level(self):
        """If a future producer omits top-level aggregates entirely, every
        field falls through to per-window computation."""
        gate = {
            'gate_passed': False,
            'windows_passed': 2,
            'windows_total': 7,
            'results': [
                {'n_trades': 25, 'win_rate': 0.28,
                 'annualized_return': -0.222, 'max_drawdown': 0.085},
                {'n_trades': 30, 'win_rate': 0.40,
                 'annualized_return': 0.15, 'max_drawdown': 0.05},
            ],
        }
        agg = _aggregate(gate)
        self.assertAlmostEqual(agg['avg_annualized_return'], (-0.222 + 0.15) / 2, places=4)
        self.assertAlmostEqual(agg['avg_win_rate'], (0.28 + 0.40) / 2, places=4)
        self.assertAlmostEqual(agg['avg_max_dd'], (0.085 + 0.05) / 2, places=4)
        self.assertEqual(agg['total_trades'], 55)


class TestAggregateEmpty(unittest.TestCase):
    """Empty results — must return all zeros without crashing."""

    def test_empty_results(self):
        agg = _aggregate({'gate_passed': False, 'results': []})
        self.assertEqual(agg['avg_annualized_return'], 0.0)
        self.assertEqual(agg['avg_win_rate'], 0.0)
        self.assertEqual(agg['avg_max_dd'], 0.0)
        self.assertEqual(agg['total_trades'], 0)

    def test_no_results_key(self):
        agg = _aggregate({'gate_passed': False})
        self.assertEqual(agg['avg_annualized_return'], 0.0)
        self.assertEqual(agg['total_trades'], 0)


if __name__ == '__main__':
    unittest.main()
