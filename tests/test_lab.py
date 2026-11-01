import json,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from generate import make_data
from model import build,solve,scenario
from verify import check
class TestZoo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.d=json.loads((ROOT/'data'/'instance.json').read_text(encoding='utf8'))
    def test_seed_is_reproducible(self):self.assertEqual(self.d,make_data())
    def test_size(self):
        self.assertEqual((len(self.d['animals']),len(self.d['feeds']),len(self.d['nutrients'])),(3,10,5))
    def test_reference_feasible(self):
        x=np.asarray(self.d['reference_feasible_ration_kg'])
        self.assertTrue(check(self.d,x)['valid'])
    def test_base_and_scenarios(self):
        base=solve(build(self.d));self.assertTrue(base.success)
        for name in ('base','shortage','stress'):
            d=scenario(self.d,name);r=solve(build(d))
            if r.success:
                v=check(d,r.x);self.assertTrue(v['valid']);self.assertAlmostEqual(v['objective_rub_per_day'],r.fun,places=5)
    def test_shortage_cannot_improve_minimum(self):
        a=solve(build(self.d));b=solve(build(scenario(self.d,'shortage')))
        self.assertTrue(a.success and b.success);self.assertGreaterEqual(b.fun+1e-8,a.fun)
    def test_forbidden_feed_detected(self):
        x=np.asarray(self.d['reference_feasible_ration_kg'],float)
        x[0,5]=1
        self.assertFalse(check(self.d,x)['valid'])
    def test_shadow_local_prediction(self):
        import copy
        m=build(self.d);r=solve(m);j=int(np.argmax(np.abs(r.ineqlin.marginals[:10])))
        d=copy.deepcopy(self.d);delta=d['availability_kg'][j]*1e-4;d['availability_kg'][j]+=delta
        rr=solve(build(d));self.assertAlmostEqual(rr.fun-r.fun,r.ineqlin.marginals[j]*delta,delta=1e-4)
if __name__=='__main__':unittest.main()
