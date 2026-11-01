"""Проверка из исходных JSON-данных без доступа к модели/объекту солвера."""
import numpy as np

def check(data,x,tol=1e-7):
    x=np.asarray(x,dtype=float).reshape((len(data['animals']),len(data['feeds'])))
    density=np.asarray(data['nutrient_per_kg'],float)
    cost=np.asarray(data['cost_rub_per_kg'],float)
    checks={}
    checks['nonnegative']=float(max(0,-x.min()))
    checks['mandatory_min']=float(np.maximum(0,np.asarray(data['mandatory_min_kg'])-x).max())
    checks['forbidden']=float(np.abs(x[np.asarray(data['allowed'])==0]).max(initial=0))
    checks['stock']=float(np.maximum(0,x.sum(axis=0)-np.asarray(data['availability_kg'])).max())
    obtained=x@density
    checks['minimum_nutrients']=float(np.maximum(0,np.asarray(data['min_nutrients_per_day'])-obtained).max())
    checks['maximum_energy']=float(np.maximum(0,obtained[:,0]-np.asarray(data['energy_max_MJ_per_day'])).max())
    checks['finite']=0.0 if np.all(np.isfinite(x)) else float('inf')
    return {'valid':all(v<=tol for v in checks.values()),'max_violation':max(checks.values()),
            'violations':checks,'objective_rub_per_day':float((x*cost).sum()),
            'nutrients_received':obtained.tolist()}
