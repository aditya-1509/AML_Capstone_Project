"""Create a readable ML summary and diagnostic charts from a completed run."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


def summarize(run):
    run = Path(run)
    metrics = json.loads((run / 'pooled/metrics.json').read_text())
    comparison = json.loads((run / 'pooled/comparison.json').read_text())
    rows = pd.read_csv(run / 'pooled/evaluation_predictions.csv')
    lines = ['# Phase 1 ML results', '',
             'Generated from this run’s saved predictions. Historical holdout results are retrospective: these rows were previously inspected in the source project.', '',
             f"Selected family: **{metrics['selected_model']}**, chosen by mean nested outer-fold F1 at cutoff 0.50 before cutoff/evaluation scoring.", '',
             '| Model | Nested CV F1 | Cutoff | Accuracy | Precision | Recall | F1 | ROC-AUC |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    fig, ax = plt.subplots(figsize=(7, 6))
    for family in comparison:
        name = family['model']; result = metrics['families'][name]; score = result['tuned']
        lines.append(f"| {name} | {family['cv_f1']:.4f} | {result['threshold']:.2f} | " + ' | '.join(f'{score[k]:.4f}' for k in ['accuracy','precision','recall','f1','roc_auc']) + ' |')
        records = rows[rows['model'] == name]
        RocCurveDisplay.from_predictions(records['target'], records['probability'], name=name, ax=ax)
    ax.plot([0, 1], [0, 1], '--', color='grey'); ax.set_title('Retrospective evaluation ROC curves')
    fig.tight_layout(); fig.savefig(run / 'roc_curves.png', dpi=160); plt.close(fig)
    selected = rows[rows['model'] == metrics['selected_model']]
    ConfusionMatrixDisplay.from_predictions(selected['target'], selected['tuned_prediction'], display_labels=['Absent', 'Present'], colorbar=False)
    plt.title(f"{metrics['selected_model']} — retrospective evaluation")
    plt.tight_layout(); plt.savefig(run / 'selected_confusion_matrix.png', dpi=160); plt.close()
    lines += ['', '## D-only hospital holdouts', '',
              '| Hospital | Locally selected model | N | F1 | Always-positive F1 | Specificity | ROC-AUC |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for folder in sorted(run.glob('hospital_*')):
        result = json.loads((folder / 'metrics.json').read_text())
        score = result['families'][result['selected_model']]['tuned']
        reference = result['constant_reference']['always_positive_metrics']
        lines.append(f"| {folder.name.removeprefix('hospital_').replace('_',' ')} | {result['selected_model']} | {result['n']} | {score['f1']:.4f} | {reference['f1']:.4f} | {score['specificity']:.4f} | {score['roc_auc']:.4f} |")
    lines += ['', 'The BN graph is provisional and statistical; its edges and independence assumptions are not medically validated. Neural-network training remains Phase 2. Hospital checks exclude all historical holdout rows and repeat selection independently. Poor specificity or performance below the constant reference constrains transfer claims. Bootstrap intervals are conditional on saved fitted models; these results do not establish clinical utility.', '',
              'See `verification.json`, `effective_protocol.json`, per-partition metrics/predictions, search traces and model files for the evidence.']
    (run / 'RESULTS.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    summarize(parser.parse_args().run)
