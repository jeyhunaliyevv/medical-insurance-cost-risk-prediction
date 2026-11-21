import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from src.pipeline import HIGH_COST_USD, engineer, make_target, validate_raw, RAW_FEATURES
from src.train import build_models, cross_validate_auc, evaluate

def sample():
    return {"age":35,"sex":"female","bmi":26.1,"children":2,"smoker":"no","region":"southwest"}

def test_feature_engineering_is_non_mutating():
    frame = pd.DataFrame([sample()])
    output = engineer(frame)
    assert len(frame.columns) == len(RAW_FEATURES)
    assert {"obese","smoker_obese","age_smoker"}.issubset(output.columns)
    assert output.iloc[0].age_smoker == 0

def test_reject_missing_column():
    bad = pd.DataFrame([sample()]).drop(columns=["smoker"])
    with pytest.raises(ValueError,match="Missing columns"):
        engineer(bad)

@pytest.mark.parametrize("field,value",[("age",-1),("bmi",0),("children",-3),("smoker","sometimes")])
def test_reject_invalid_input(field,value):
    row=sample();row[field]=value
    with pytest.raises(ValueError): validate_raw(pd.DataFrame([row]))

def test_fixed_target_boundary():
    y=make_target(pd.Series([HIGH_COST_USD-1,HIGH_COST_USD,HIGH_COST_USD+1]))
    assert y.tolist()==[0,1,1]

def test_training_pipeline_prediction_and_scores():
    rows=[];labels=[]
    for i in range(40):
        r=sample();r["age"]+=i%10;r["smoker"]="yes" if i%2 else "no"
        r["bmi"]+=i%4
        rows.append(r);labels.append(i%2)
    x=engineer(pd.DataFrame(rows));y=pd.Series(labels)
    model=build_models()["LogisticRegression"]
    model.fit(x,y)
    probs=model.predict_proba(x)[:,1]
    assert len(probs)==40
    assert ((probs>=0)&(probs<=1)).all()
    assert 0<=evaluate(y,probs)["roc_auc"]<=1

def test_cv_splits_are_train_only():
    rows=[];y=[]
    for i in range(40):
        r=sample();r["age"]+=i%10;r["smoker"]="yes" if i%2 else "no"
        rows.append(r);y.append(i%2)
    scores=cross_validate_auc(build_models()["LogisticRegression"],
                              engineer(pd.DataFrame(rows)),pd.Series(y),folds=4)
    assert len(scores)==4
    assert all(0<=s<=1 for s in scores)


def test_stability_diagnostics_on_tiny_two_split_sample(tmp_path):
    from src.stability import run_stability
    from src.pipeline import HIGH_COST_USD
    rows = []
    for i in range(60):
        row = sample()
        row['age'] = 20+i%30
        row['smoker'] = 'yes' if i%3==0 else 'no'
        row['charges'] = HIGH_COST_USD+1000 if i%3==0 else 5000
        rows.append(row)
    data_path = tmp_path/'synthetic.csv'
    pd.DataFrame(rows).to_csv(data_path,index=False)
    scores, summary = run_stability(data_path,save_artifacts=False,split_seeds=[101,102])
    assert len(scores)==2
    assert summary['splits']==2
    assert set(scores.columns)=={'split_seed','roc_auc','precision','recall','f1'}
