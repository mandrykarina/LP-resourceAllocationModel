"""Линейная модель рациона: отдельная матричная постановка для HiGHS."""
import numpy as np
from scipy.optimize import linprog


def build(data):
    a,f=len(data['animals']),len(data['feeds'])
    coeff=np.asarray(data['nutrient_per_kg'],float)
    c=np.tile(np.asarray(data['cost_rub_per_kg'],float),a)
    rows,limits,names=[],[],[]
    def add(v,rhs,name):
        rows.append(v);limits.append(float(rhs));names.append(name)
    for j in range(f):
        r=np.zeros(a*f);r[np.arange(a)*f+j]=1
        add(r,data['availability_kg'][j],f'stock:{data["feeds"][j]}')
    for i in range(a):
        for k,nutr in enumerate(data['nutrients']):
            r=np.zeros(a*f);r[i*f:(i+1)*f]=-coeff[:,k]
            add(r,-data['min_nutrients_per_day'][i][k],f'min:{data["animals"][i]}:{nutr}')
        r=np.zeros(a*f);r[i*f:(i+1)*f]=coeff[:,0]
        add(r,data['energy_max_MJ_per_day'][i],f'energy_max:{data["animals"][i]}')
    lower=np.array(data['mandatory_min_kg'],float).reshape(-1)
    allowed=np.array(data['allowed'],int).reshape(-1)
    upper=np.where(allowed>0,np.inf,0.0)
    if np.any(lower>upper):raise ValueError('Обязательные порции нарушают запреты.')
    return dict(c=c,A=np.asarray(rows),b=np.asarray(limits),bounds=list(zip(lower,upper)),row_names=names,shape=(a,f))


def solve(m,method='highs-ds'):
    return linprog(m['c'],A_ub=m['A'],b_ub=m['b'],bounds=m['bounds'],method=method,
                   options={'presolve':True, 'dual_feasibility_tolerance':1e-8,
                            'primal_feasibility_tolerance':1e-8})


def scenario(data,kind):
    import copy
    d=copy.deepcopy(data)
    if kind=='base':return d
    if kind=='shortage':
        d['availability_kg'][0] *= .8
    elif kind=='stress':
        d['availability_kg'][0] *= .8
        d['availability_kg'][8] *= .8
        for row in d['min_nutrients_per_day']:
            for k in range(len(row)):row[k] *= 1.2
    else:raise ValueError(kind)
    return d
