"""Verify persisted models, decisions and metrics: python verify_run.py RUN_DIR."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from advanced_modeling import clean_features
from train import choose_threshold, scores, positive_probability


def verify_run(run):
    run = Path(run)
    root = Path(__file__).resolve().parent
    manifest = json.loads((run / 'manifest.json').read_text())
    assert manifest['status'] == 'complete', 'Run did not complete'
    source = root / 'heart_disease_uci.csv'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest['source_sha256']
    raw = pd.read_csv(source).set_index('id', drop=False)
    retrospective_ids = set(pd.read_csv(run / 'historical_split.csv').query("partition == 'T_retrospective'")['id'])
    count = 0
    for folder in sorted(run.iterdir()):
        if not (folder / 'freeze.json').exists():
            continue
        freeze = json.loads((folder / 'freeze.json').read_text())
        plan = json.loads((folder / 'partitions.json').read_text())
        metrics = json.loads((folder / 'metrics.json').read_text())
        thresholds = json.loads((folder / 'thresholds.json').read_text())
        comparison = json.loads((folder / 'comparison.json').read_text())
        prediction = pd.read_csv(folder / 'evaluation_predictions.csv')
        cutoff = pd.read_csv(folder / 'cutoff_predictions.csv')
        oof = pd.read_csv(folder / 'nested_oof_predictions.csv')
        selection_ids = set(plan['selection_source_ids'])
        cutoff_ids = set(plan['cutoff_source_ids'])
        evaluation_ids = set(prediction['source_id'])
        assert not (selection_ids & cutoff_ids or selection_ids & evaluation_ids or cutoff_ids & evaluation_ids)
        if folder.name.startswith('hospital_'):
            assert not ((selection_ids | cutoff_ids | evaluation_ids) & retrospective_ids)
            site = folder.name.removeprefix('hospital_').replace('_', ' ')
            assert set(raw.loc[list(evaluation_ids), 'dataset']) == {site}
            assert site not in set(raw.loc[list(selection_ids | cutoff_ids), 'dataset'])
        else:
            assert evaluation_ids == retrospective_ids
        for name, expected_hash in freeze['model_sha256'].items():
            path = folder / 'models' / (name.replace(' ', '_') + '.joblib')
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash
            model = joblib.load(path)
            rows = prediction[prediction['model'] == name]
            assert rows['source_id'].is_unique
            actual = positive_probability(model, clean_features(raw.loc[rows['source_id']]))
            assert_allclose(actual, rows['probability'], atol=1e-12, rtol=0)
            np.testing.assert_array_equal(rows['target'], (raw.loc[rows['source_id'], 'num'] > 0).astype(int))
            threshold = freeze['thresholds'][name]
            np.testing.assert_array_equal(rows['tuned_prediction'], actual >= threshold)
            for key, value in scores(rows['target'], actual, threshold).items():
                assert_allclose(value, metrics['families'][name]['tuned'][key], atol=1e-12, rtol=0)
            crows = cutoff[cutoff['model'] == name]
            assert set(crows['source_id']) == cutoff_ids
            cp = positive_probability(model, clean_features(raw.loc[crows['source_id']]))
            assert_allclose(cp, crows['probability'], atol=1e-12, rtol=0)
            selected = choose_threshold(crows['target'], cp)
            assert selected == thresholds[name]
            orows = oof[oof['model'] == name]
            assert orows['source_id'].is_unique and set(orows['source_id']) == selection_ids
            f1, auc = [], []
            for number, fold in enumerate(plan['outer']):
                frows = orows[orows['fold'] == number]
                assert set(frows['source_id']) == set(fold['validation_source_ids'])
                score = scores(frows['target'], frows['probability'])
                f1.append(score['f1']); auc.append(score['roc_auc'])
            saved = next(item for item in comparison if item['model'] == name)
            assert_allclose(f1, saved['fold_f1'], atol=1e-12, rtol=0)
            assert_allclose(auc, saved['fold_auc'], atol=1e-12, rtol=0)
            count += 1
        order = json.loads((run / 'effective_protocol.json').read_text())['selection']['fixed_family_tie_order']
        winner = min(comparison, key=lambda row: (-row['cv_f1'], -row['cv_auc'], order.index(row['model'])))['model']
        assert winner == freeze['selected_model'] == metrics['selected_model']
    assert count == 25, f'Expected 25 frozen models; verified {count}'
    result = {'status': 'PASS', 'models_verified': count, 'checks': 'Model hashes and prediction round trips; metric reconstruction; cutoff selection; nested fold scores/family selection; partition and hospital isolation'}
    (run / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    print(json.dumps(verify_run(parser.parse_args().run), indent=2))
