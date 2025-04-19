import numpy as np
from espprc.espprc import ESPPRC

def test_basic_solve():
    n, n_res = 5, 2
    r = np.random.rand(n, n, n_res + 1) * 10  # Random cost + 2 resources
    r_max = np.array([15.0] * n_res)

    # Inject dummy duals into the instance directly after creation
    esp = ESPPRC(r_max=r_max, r=r)
    esp.cg_dual = cg_dual
    esp.wh_dual = wh_dual
    esp.wh_pi = wh_pi

    esp.solve()

    assert isinstance(esp.best_path, list)
    assert all(isinstance(p, list) for p in esp.best_path)

