"""Offline numerical and coordinate checks for the full-study analysis."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_full_subjective_study import (alpha_interval, alpha_nominal, bh_adjust,
                                           interval, presentation_vectors)
from subjective_intervention_runner import request_for


class AnalysisTests(unittest.TestCase):
    def test_factorial_recovers_independent_position_and_label_terms(self):
        # Original A receives +.1 at position 1 and +.2 when called B.
        cells = {('model', 'task', condition): {'pA': np.array([value])}
                 for condition, value in [('baseline', .4), ('pair_exchange_labels', .6),
                                          ('pair_exchange_positions', .3), ('pair_exchange_both', .5)]}
        result = presentation_vectors(cells, 'model', 'task')
        self.assertAlmostEqual(result['First-position'][0], .1)
        self.assertAlmostEqual(result['B-label'][0], .2)
        self.assertAlmostEqual(result['Interaction'][0], 0)

    def test_one_cluster_per_row_matches_iid_standard_error(self):
        values = np.array([.1, .3, .6, -.1, .2])
        design = {'kind': 'cluster', 'labels': np.arange(len(values))}
        result = interval(values, design)
        from scipy.stats import t
        expected_halfwidth = t.ppf(.975, len(values)-1) * np.std(values, ddof=1) / np.sqrt(len(values))
        self.assertAlmostEqual(result[2] - result[0], expected_halfwidth)

    def test_disjoint_dyads_implement_declared_node_correction(self):
        values = np.array([.1, .3, .6, -.1, .2])
        nodes = [(2*i, 2*i+1) for i in range(len(values))]
        result = interval(values, {'kind': 'dyadic', 'nodes': nodes})
        from scipy.stats import t
        variance = np.sum((values-values.mean())**2) / len(values)**2 * 10/9
        self.assertAlmostEqual(result[2]-result[0], t.ppf(.975, 9)*np.sqrt(variance))

    def test_bh_known_values_and_permutation(self):
        np.testing.assert_allclose(bh_adjust([.04, .01, .03, .9]), [.0533333333, .04, .0533333333, .9])

    def test_alpha_perfect_and_disagreeing_units(self):
        self.assertAlmostEqual(alpha_nominal([['A','A'], ['B','B']]), 1)
        self.assertAlmostEqual(alpha_interval([[0,0], [2,2]]), 1)
        self.assertAlmostEqual(alpha_nominal([['A','B'], ['A','B']]), -.5)
        self.assertAlmostEqual(alpha_interval([[0,1], [0,1]]), -.5)

    def test_payload_mapping_and_score_polarity(self):
        row = {'id':'test', 'input':{'instruction':'Which is funnier?', 'context':'',
                                   'options':[{'label':'A','text':'one'}, {'label':'B','text':'two'}]}}
        expected = {'pair_exchange_labels': [('B','one','A'),('A','two','B')],
                    'pair_exchange_positions': [('B','two','B'),('A','one','A')],
                    'pair_exchange_both': [('A','two','B'),('B','one','A')]}
        for condition, values in expected.items():
            request, metadata = request_for('ukp_humor_pairwise.jsonl', row, condition, 'model')
            self.assertEqual(list(request['questions']['judgment']['criteria'].items()), [(k,v) for k,v,_ in values])
            self.assertEqual(metadata['option_to_original'], {k:v for k,_,v in values})
        _, meta = request_for('ukp_humor_pairwise.jsonl', row, 'pair_rejection', 'model')
        self.assertTrue(meta['inverse_question'])
        score_row = {'id':'score','input':{'text':'summary','context':'article'}}
        _, meta = request_for('summeval_relevance_scoring.jsonl', score_row, 'score_anchor_reverse', 'model')
        self.assertEqual(meta['quality_by_index'], [1,.75,.5,.25,0])


if __name__ == '__main__':
    unittest.main()
