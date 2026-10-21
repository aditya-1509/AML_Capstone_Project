from pathlib import Path
import json,hashlib,sys
import joblib
import numpy as np
import pandas as pd
sys.path.insert(0,'/Users/aanyamehrotra/Downloads/mk_sem5_project')
from train import scores,choose_threshold,positive_probability,best_configuration
from modeling import clean_features
root=Path('/Users/aanyamehrotra/Downloads/mk_sem5_project')
run=root/'artifacts/runs/20261020T193544Z_2a8a1491'
manifest=json.loads((run/'manifest.json').read_text())
assert manifest['status']=='complete',manifest['status']
assert manifest['failures']==[]
protocol=json.loads((run/'effective_protocol.json').read_text())
historical=pd.read_csv(run/'historical_split.csv')
D=set(historical.loc[historical.partition=='D_development','id']);T=set(historical.loc[historical.partition=='T_retrospective','id'])
assert len(D)==734 and len(T)==184 and not D&T
raw=pd.read_csv(root/'heart_disease_uci.csv').set_index('id',drop=False)
checks=[];fits=0
for folder in sorted(run.glob('*/partitions.json')):
 folder=folder.parent
 plan=json.loads((folder/'partitions.json').read_text())
 selection=set(plan['selection_source_ids']);cutoff=set(plan['cutoff_source_ids'])
 assert not selection&cutoff and not (selection|cutoff)&T
 evaluation=pd.read_csv(folder/'evaluation_predictions.csv')
 metrics=json.loads((folder/'metrics.json').read_text())
 freeze=json.loads((folder/'freeze.json').read_text())
 thresholds=json.loads((folder/'thresholds.json').read_text())
 comparisons=json.loads((folder/'comparison.json').read_text())
 chosen=min(comparisons,key=lambda r:(-r['cv_f1'],-r['cv_auc'],protocol['selection']['fixed_family_tie_order'].index(r['model'])))['model']
 assert chosen==freeze['selected_model']==metrics['selected_model']
 held=set(evaluation.source_id)
 assert not held&(selection|cutoff)
 if folder.name=='pooled':assert held==T
 else:
  site=folder.name.removeprefix('hospital_').replace('_',' ')
  expected={i for i in D if raw.loc[i,'dataset']==site}
  assert held==expected and (selection|cutoff)==D-expected
 oof=pd.read_csv(folder/'nested_oof_predictions.csv')
 searches=json.loads((folder/'searches.json').read_text())
 cp=pd.read_csv(folder/'cutoff_predictions.csv')
 for search in searches:
  fits+=3*len(search['results']['params'])+1
  assert set(search['train_source_ids'])==(selection if search['fold'] is None else set(plan['outer'][search['fold']]['train_source_ids']))
  assert search['elapsed_seconds']>0
  assert np.isfinite(search['results']['mean_test_f1']).all()
 for name in metrics['families']:
  output=evaluation.loc[evaluation.model==name]
  assert len(output)==len(held) and not output.source_id.duplicated().any()
  train_cp=cp.loc[cp.model==name]
  assert set(train_cp.source_id)==cutoff
  expected_threshold=choose_threshold(train_cp.target.to_numpy(),train_cp.probability.to_numpy())
  assert expected_threshold['grid_index']==thresholds[name]['grid_index']
  np.testing.assert_allclose(expected_threshold['scores'],thresholds[name]['scores'],atol=1e-14)
  subset=oof.loc[oof.model==name]
  assert len(subset)==len(selection) and set(subset.source_id)==selection
  for fold_number,fold in enumerate(plan['outer']):
   assert set(subset.loc[subset.fold==fold_number,'source_id'])==set(fold['validation_source_ids'])
  path=folder/'models'/(name.replace(' ','_')+'.joblib')
  assert hashlib.sha256(path.read_bytes()).hexdigest()==freeze['model_sha256'][name]
  model=joblib.load(path)
  x=clean_features(raw.loc[output.source_id.tolist()])
  np.testing.assert_allclose(positive_probability(model,x),output.probability,rtol=1e-12,atol=1e-14)
  for operating_point,threshold in [('default',.5),('tuned',thresholds[name]['threshold'])]:
   recalculated=scores(output.target,output.probability.to_numpy(),threshold)
   for key,value in recalculated.items():
    np.testing.assert_allclose(value,metrics['families'][name][operating_point][key],rtol=1e-12,atol=1e-14)
  assert np.array_equal(output.tuned_prediction.to_numpy(),output.probability.to_numpy()>=thresholds[name]['threshold'])
  if name=='Bayesian network':assert model.graph_['approved'] is False and model.alpha==1.0
 checks.append({'experiment':folder.name,'selection_rows':len(selection),'cutoff_rows':len(cutoff),'evaluation_rows':len(held),'families_verified':len(metrics['families'])})
assert len(checks)==5 and fits==manifest['planned_fits_per_experiment']*5
report={'status':'PASS','checks':checks,'verified_models':25,'completed_fit_count_reconstructed_from_search_records':fits,'roundtrip_predictions':'PASS','reconstructed_metrics':'PASS','threshold_selection':'PASS','partition_isolation':'PASS'}
(run/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
