"""Запуск расчёта и сохранение всех обязательных результатов ЛР №1."""
import csv,json,platform,sys
from pathlib import Path
import numpy as np
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import build,solve,scenario
from verify import check

ROOT=Path(__file__).resolve().parent

def save_json(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    data=json.loads((ROOT/'data'/'instance.json').read_text(encoding='utf-8'))
    out=ROOT/'results';out.mkdir(exist_ok=True)
    runs={};logs=[]
    for kind in ['base','shortage','stress']:
        d=scenario(data,kind);m=build(d);result=solve(m)
        name={'base':'Базовый','shortage':'Дефицит (-20% запас сена)','stress':'Стресс (-20% два запаса, +20% нормы)'}[kind]
        rec={'scenario':kind,'description':name,'status_code':int(result.status),
             'status':{0:'optimal',1:'time_limit_or_iteration_limit',2:'infeasible',3:'unbounded',4:'solver_error'}.get(result.status,'unknown'),
             'message':result.message,'objective_rub_per_day':None,'nit':int(getattr(result,'nit',0))}
        logs.append(f'{name}: HiGHS / highs-ds / presolve=True\nstatus={rec["status"]}; message={result.message}; iterations={rec["nit"]}')
        if result.status==0:
            v=check(d,result.x)
            if not v['valid'] or abs(v['objective_rub_per_day']-result.fun)>1e-6:raise RuntimeError(f'Не пройдена проверка {kind}: {v}')
            rec.update(objective_rub_per_day=float(result.fun),validation=v)
            a,f=m['shape'];x=result.x.reshape(a,f)
            with (out/f'solution_{kind}.csv').open('w',newline='',encoding='utf-8') as fh:
                w=csv.writer(fh);w.writerow(['animal','feed','kg_per_day','rub_per_day'])
                for i,animal in enumerate(d['animals']):
                    for j,feed in enumerate(d['feeds']):w.writerow([animal,feed,f'{x[i,j]:.9f}',f'{x[i,j]*d["cost_rub_per_kg"][j]:.9f}'])
            save_json(out/f'validation_{kind}.json',v)
            if kind=='base':
                shadow=np.asarray(result.ineqlin.marginals,float)
                slacks=np.asarray(result.ineqlin.residual,float)
                with (out/'shadow_prices.csv').open('w',newline='',encoding='utf-8') as fh:
                    w=csv.writer(fh);w.writerow(['constraint','slack','marginal_rub_per_rhs_unit','active'])
                    for n,s,p in zip(m['row_names'],slacks,shadow):w.writerow([n,f'{s:.10g}',f'{p:.10g}',int(abs(s)<=1e-7)])
                # Локальная проверка точности линейного прогноза теневой цены.
                j=int(np.argmax(np.abs(shadow[:len(d['feeds'])])))
                delta=d['availability_kg'][j]*.01
                import copy
                modified=copy.deepcopy(d);modified['availability_kg'][j]+=delta
                new=solve(build(modified))
                rec['shadow_check']={'resource':d['feeds'][j],'delta_kg':delta,
                  'marginal_rub_per_kg':float(shadow[j]),'predicted_delta_cost_rub':float(shadow[j]*delta),
                  'actual_delta_cost_rub':float(new.fun-result.fun) if new.success else None}
                rec['active_constraints']=[n for n,s in zip(m['row_names'],slacks) if abs(s)<=1e-7]
                fig,ax=plt.subplots(figsize=(9,4.4))
                xbins=np.arange(len(data['feeds']));bottom=np.zeros(len(data['feeds']))
                for i,animal in enumerate(d['animals']):
                    ax.bar(xbins,x[i],bottom=bottom,label=animal,alpha=.85);bottom+=x[i]
                ax.set_xticks(xbins,labels=d['feeds'],rotation=35,ha='right');ax.set_ylabel('Feed, kg/day');ax.legend(fontsize=8,ncol=3)
                fig.tight_layout();fig.savefig(out/'ration.png',dpi=180);plt.close(fig)
        else:
            save_json(out/f'validation_{kind}.json',{'valid':False,'reason':'Решение отсутствует: '+result.message})
        runs[kind]=rec
        logs.append(f'objective={rec["objective_rub_per_day"]}; independent_check={rec.get("validation",{}).get("valid","not_applicable")}\n')
    with (out/'scenarios.csv').open('w',newline='',encoding='utf-8') as fh:
        w=csv.writer(fh);w.writerow(['scenario','status','cost_rub_per_day','iterations'])
        for rec in runs.values():w.writerow([rec['scenario'],rec['status'],rec['objective_rub_per_day'],rec['nit']])
    (out/'solver_log.txt').write_text('HiGHS, API scipy.optimize.linprog\n'+f'Python {platform.python_version()}; SciPy {scipy.__version__}; NumPy {np.__version__}\n\n'+'\n'.join(logs),encoding='utf-8')
    save_json(out/'summary.json',runs)
    print('ЛР1 завершена:',[(s,r['status'],r['objective_rub_per_day']) for s,r in runs.items()])
if __name__=='__main__':main()
