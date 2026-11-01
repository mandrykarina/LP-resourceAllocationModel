"""Воспроизводимые учебные данные рациона для трёх групп животных."""
import json
from pathlib import Path
import numpy as np

SEED = 10017
NUTRIENTS = ['energy_MJ', 'protein_g', 'fiber_g', 'calcium_g', 'vitamin_A_mg']
FEEDS = ['hay', 'alfalfa', 'oats', 'barley', 'carrot', 'apple', 'beet', 'bran', 'pellets', 'greens']
ANIMALS = ['ungulates', 'rodents', 'primates']

def make_data():
    rng = np.random.default_rng(SEED)
    # Показатели в расчёте на 1 кг кормового сырья; учебные, не ветеринарные нормативы.
    nutrients = np.array([
        [7.5, 95, 280, 5.5, 0.18],
        [8.1, 170, 260, 13.0, 0.40],
        [12.3, 125, 95, 0.9, 0.08],
        [11.9, 105, 80, 0.7, 0.05],
        [3.4, 11, 32, 0.3, 0.60],
        [2.2, 3, 21, 0.1, 0.15],
        [2.8, 14, 44, 0.4, 0.23],
        [9.2, 165, 125, 1.3, 0.09],
        [10.6, 185, 115, 6.0, 0.65],
        [3.1, 23, 75, 2.5, 0.85],
    ],dtype=float)
    nutrients[:, 0] *= rng.uniform(0.97, 1.03, len(FEEDS))
    prices = np.round([1.6,2.7,3.2,2.9,2.1,2.6,1.8,2.4,4.1,2.5] * np.ones(10), 2)
    prices = np.round(prices*rng.uniform(.96,1.04,10),2)
    # У разных видов разные разрешённые рационы и обязательные минимальные доли.
    allowed = np.array([
       [1,1,1,1,1,0,1,1,1,1],
       [1,1,1,0,1,1,1,1,1,1],
       [0,0,1,1,1,1,1,1,1,1],
    ],dtype=int)
    baseline = np.array([
      [5.0,2.5,1.4,0.8,1.2,0,1.0,0.6,0.8,1.0],
      [0.9,0.5,0.35,0,0.9,0.35,0.5,0.25,0.55,0.5],
      [0,0,0.55,0.5,0.65,0.8,0.4,0.45,0.7,0.75],
    ],dtype=float)
    # В качестве гарантированно допустимой опорной точки используем baseline.
    min_requirement = np.round(baseline @ nutrients * np.array([.78,.76,.70,.70,.68]),4)
    max_energy = np.round((baseline @ nutrients[:,0])*1.38,4)
    stock = np.round(baseline.sum(axis=0)*1.3 + .15,4)
    mandatory = np.zeros_like(baseline)
    for i in range(3):
       for j in np.flatnonzero(baseline[i]>0)[:3]: mandatory[i,j] = .05
    return {
       'variant':16, 'N':17,'Q':0,'Y':26,'seed':SEED,
       'description':'Синтетические данные: суточные рационы трёх групп животных.',
       'feeds':FEEDS,'animals':ANIMALS,'nutrients':NUTRIENTS,
       'nutrient_per_kg':nutrients.tolist(),'cost_rub_per_kg':prices.tolist(),
       'allowed':allowed.tolist(),'mandatory_min_kg':mandatory.tolist(),
       'availability_kg':stock.tolist(), 'min_nutrients_per_day':min_requirement.tolist(),
       'energy_max_MJ_per_day':max_energy.tolist(),
       'reference_feasible_ration_kg':baseline.tolist(),
       'units':{'feed':'kg/day','cost':'rub/day','nutrients':NUTRIENTS},
    }

if __name__ == '__main__':
    dst = Path(__file__).resolve().parent/'data'/'instance.json'
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(make_data(),ensure_ascii=False,indent=2),encoding='utf-8')
    print(dst)
