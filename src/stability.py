"""Repeated, stratified holdout diagnostic for the preselected CatBoost classifier.

This is an exploratory sensitivity assessment, NOT independent external validation
or nested model selection. Repeated test folds overlap, so no confidence claims.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from .train import build_models, ROOT
from .pipeline import engineer, make_target, validate_raw

SPLIT_SEEDS = list(range(100, 120))  # predeclared 20 random splits

def run_stability(dataset=None, save_artifacts=True, split_seeds=SPLIT_SEEDS):
    data_file = Path(dataset) if dataset is not None else ROOT / 'data' / 'insurance.csv'
    if not data_file.is_file():
        raise FileNotFoundError(f'Dataset missing: {data_file}. Download from the link in data/README.md')
    raw = validate_raw(pd.read_csv(data_file), require_target=True)
    features, target = engineer(raw), make_target(raw['charges'])
    records = []
    for seed in split_seeds:
        x_tr, x_te, y_tr, y_te = train_test_split(features,target,test_size=.2,stratify=target,random_state=int(seed))
        model = clone(build_models()['CatBoost']).fit(x_tr,y_tr)
        prob = model.predict_proba(x_te)[:,1]
        prediction = (prob >= 0.5).astype(int)
        records.append({'split_seed':int(seed),'roc_auc':float(roc_auc_score(y_te,prob)),
                        'precision':float(precision_score(y_te,prediction,zero_division=0)),
                        'recall':float(recall_score(y_te,prediction,zero_division=0)),
                        'f1':float(f1_score(y_te,prediction,zero_division=0))})
    splits = pd.DataFrame(records)
    summary = {'model':'CatBoost (preselected using original training CV)',
               'method':'20 repeated random stratified 80/20 holdouts; overlapping samples; descriptive only',
               'seeds':list(map(int,split_seeds)), 'threshold':0.5, 'splits':len(splits),
               'metrics':{name:{'mean':round(float(splits[name].mean()),4),
                                'std':round(float(splits[name].std(ddof=1)),4),
                                'min':round(float(splits[name].min()),4),
                                'max':round(float(splits[name].max()),4)}
                          for name in ['roc_auc','precision','recall','f1']}}
    if save_artifacts:
        folder=ROOT/'reports';folder.mkdir(exist_ok=True)
        splits.to_csv(folder/'stability_splits.csv',index=False)
        (folder/'stability_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return splits, summary

if __name__ == '__main__':
    _, summary = run_stability()
    print(json.dumps(summary, indent=2))
