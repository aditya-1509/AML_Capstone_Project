"""Read-only app checks: saved inference and Streamlit interactions, no fitting."""

import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

import app
from modeling import clean_features


def render_prediction(bundle):
    import app
    app.prediction_page(bundle)


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = app.load_artifact(app.RUN_PATH)

    def test_saved_models_reproduce_saved_probabilities_without_fitting(self):
        saved = pd.read_csv(app.RUN_PATH / 'pooled/evaluation_predictions.csv')
        data = app.load_data().set_index('id')
        for name, key in [('Random forest', 'model'), ('Bayesian network', 'bn_model')]:
            with self.subTest(model=name):
                rows = saved.loc[saved['model'] == name]
                probabilities = self.bundle[key].predict_proba(clean_features(data.loc[rows['source_id']]))
                self.assertEqual(probabilities.shape, (184, 2))
                self.assertTrue(np.isfinite(probabilities).all())
                self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
                np.testing.assert_allclose(probabilities.sum(axis=1), 1)
                np.testing.assert_allclose(probabilities[:, 1], rows['probability'], atol=1e-12, rtol=0)
                np.testing.assert_array_equal(probabilities[:, 1] >= rows['threshold'], rows['tuned_prediction'])
        state = json.loads((app.RUN_PATH / 'pooled/bn_state.json').read_text())
        self.assertEqual(self.bundle['bn_model'].graph_, state['graph'])
        self.assertFalse(state['graph']['approved'])
        for name, cpt in state['cpts'].items():
            np.testing.assert_array_equal(self.bundle['bn_model'].cpds_[name], cpt)

    def test_loading_rejects_missing_mismatched_threshold_model_and_graph(self):
        # Copies are disposable; the verified run is only ever read.
        for fault in ('missing', 'threshold', 'model', 'graph'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory(prefix='heart-app-test-') as directory:
                run = Path(directory) / 'run'
                shutil.copytree(app.RUN_PATH, run)
                if fault == 'missing':
                    (run / 'pooled/freeze.json').unlink()
                elif fault == 'threshold':
                    path = run / 'pooled/thresholds.json'
                    content = json.loads(path.read_text())
                    content['Random forest']['threshold'] = 0.5
                    path.write_text(json.dumps(content))
                elif fault == 'model':
                    (run / 'pooled/models/Random_forest.joblib').write_bytes(b'wrong model')
                else:
                    path = run / 'bn_graph.json'
                    content = json.loads(path.read_text())
                    content['approved'] = True
                    path.write_text(json.dumps(content))
                with self.assertRaises((OSError, ValueError)):
                    app.load_artifact(run)

    def test_prediction_uses_saved_cutoffs_including_equality(self):
        for probability, positive in [(0.38, True), (0.379, False)]:
            with self.subTest(probability=probability):
                bundle = dict(self.bundle)
                bundle['model'] = Mock()
                bundle['model'].predict_proba.return_value = np.array([[1 - probability, probability]])
                bundle['bn_model'] = Mock()
                bundle['bn_model'].predict_proba.return_value = np.array([[0.47, 0.53]])
                view = AppTest.from_function(render_prediction, args=(bundle,), default_timeout=30).run()
                view.button[0].click().run()
                self.assertFalse(view.exception)
                html = '\n'.join(element.value for element in view.markdown)
                self.assertIn(f'class="result {"positive" if positive else ""}"', html)
                captions = '\n'.join(element.value for element in view.caption)
                self.assertIn('Classification cutoff: 38%', captions)
                self.assertIn('Classification: no disease · cutoff: 54%', captions)
                self.assertEqual(view.session_state['prediction']['probability'], probability)

    def test_missing_inputs_work_and_invalid_input_clears_previous_result(self):
        view = AppTest.from_file(str(app.ROOT / 'app.py'), default_timeout=30).run()
        for field in view.text_input:
            field.set_value('')
        for field in view.selectbox:
            if 'Not available' in field.options:
                field.set_value('Not available')
        view.button[0].click().run()
        self.assertFalse(view.exception)
        self.assertFalse(view.error)
        prediction = view.session_state['prediction']
        row = pd.DataFrame([prediction['inputs']])
        self.assertEqual(row.isna().sum(axis=1).iloc[0], 8)
        for key, model in [('probability', 'model'), ('bn_probability', 'bn_model')]:
            self.assertAlmostEqual(prediction[key], self.bundle[model].predict_proba(clean_features(row))[0, 1])
        view.text_input[0].set_value('invalid')
        view.button[0].click().run()
        self.assertFalse(view.exception)
        self.assertIn('Please correct the profile', view.error[0].value)
        self.assertNotIn('prediction', view.session_state)

    def test_all_pages_and_displayed_results_match_selected_model(self):
        view = AppTest.from_file(str(app.ROOT / 'app.py'), default_timeout=30).run()
        self.assertFalse(view.exception)
        self.assertTrue(any('Selected model: Random forest' in item.value for item in view.caption))
        # A session left over from the legacy app must not display its old model score.
        view.session_state['prediction'] = {'probability': 0.99, 'inputs': {}}
        view.run()
        self.assertNotIn('prediction', view.session_state)
        view.radio[0].set_value('Model studio').run()
        self.assertFalse(view.exception)
        meta = self.bundle['metadata']
        selected = next(row for row in meta['comparison'] if row['model'] == meta['selected_model'])
        self.assertEqual(view.metric[0].label, 'Random forest · mean CV F1')
        self.assertEqual(view.metric[0].value, f"{selected['cv_f1']:.3f}")
        self.assertEqual(view.metric[2].value, f"{meta['test']['f1']:.3f}")
        table = view.dataframe[1].value
        expected = pd.DataFrame(meta['retrospective_comparison'])
        pd.testing.assert_frame_equal(table, expected)
        self.assertEqual(len(table), 5)
        self.assertTrue(any('Retrospective' in item.value for item in view.subheader))
        self.assertTrue(any('provisional' in item.value for item in view.caption))
        view.radio[0].set_value('Data story').run()
        self.assertFalse(view.exception)
        self.assertEqual(len(view.get('plotly_chart')), 7)
        self.assertTrue(any('did not calculate permutation importance' in item.value for item in view.markdown))


if __name__ == '__main__':
    unittest.main()
