import numpy as np
import pandas as pd
from types import SimpleNamespace
from ampub.presentation import endpoint_profile

def test_profile_missing_source_cannot_create_evidence():
    data = SimpleNamespace(group_metrics=pd.DataFrame(), dimensions=[128,256,512])
    assert endpoint_profile(data).empty

def test_profile_retains_forecast_cost_and_snapshot_gain():
    rows=[]
    for kind,metric,pred,mult in [("reconstruction","Q","none",.5),
                                  ("short_forecast","Q","linear",1.5)]:
        for seed in [1729,2718,31415]:
            for group,base in [("a",1),("b",2),("c",4)]:
                for method,variant,value in [("ae","legacy",base),("rpc","direct_corrected",mult*base)]:
                    rows.append(dict(dimension=256,kind=kind,metric=metric,predictor=pred,
                         seed=seed,group=group,method=method,variant=variant,value=value,invalid_records=0))
    data=SimpleNamespace(group_metrics=pd.DataFrame(rows),dimensions=[256])
    result=endpoint_profile(data).set_index("endpoint")
    np.testing.assert_allclose(result.loc["Snapshot Q",["estimate","ci95_low","ci95_high"]].astype(float),50)
    np.testing.assert_allclose(result.loc["10-step Q",["estimate","ci95_low","ci95_high"]].astype(float),-50)
    data.group_metrics.loc[0,"invalid_records"]=1
    assert np.isnan(endpoint_profile(data).set_index("endpoint").loc["Snapshot Q","estimate"])
