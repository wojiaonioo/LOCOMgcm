"""L-BFGS with Armijo halving and failed-trial recovery."""
import math
import numpy as np
from types import SimpleNamespace
class TrialFailure(Exception):
    pass
def failure_fields(failure, **where):
    return dict(where, evaluation=failure.get('evaluation'), error=failure.get('error'), background_cost=failure.get('background_cost'), wall_s=failure.get('wall_s'), message=failure.get('message', '')[:600])

def lbfgs_backtracking(fun, x0, maxiter, callback, on_failure, maxcor=10, maxls=12, retries=5, failure_iterations=3, gtol=1e-05, ftol=1e-09, c1=0.0001):

    def result(status, message):
        return SimpleNamespace(x=x, fun=f, status=status, message=message, nit=nit)
    x = np.array(x0, dtype=np.float64)
    (f, g) = fun(x)
    (pairs, nit, streak) = ([], 0, 0)
    eps = np.finfo(np.float64).eps
    while True:
        if np.max(np.abs(g)) <= gtol:
            return result(0, 'CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL')
        if nit >= maxiter:
            return result(1, 'STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT')
        (q, alphas) = (g.copy(), [])
        for (s, y, rho) in reversed(pairs):
            a = rho * np.dot(s, q)
            q -= a * y
            alphas.append(a)
        if pairs:
            (s, y, _) = pairs[-1]
            q *= np.dot(s, y) / np.dot(y, y)
        for ((s, y, rho), a) in zip(pairs, reversed(alphas)):
            q += s * (a - rho * np.dot(y, q))
        d = -q
        gd = float(np.dot(g, d))
        if not pairs or not gd < 0:
            (pairs, d) = ([], -g)
            gd = float(np.dot(g, d))
            step = 1.0 / math.sqrt(-gd)
        else:
            step = 1.0
        (fails, failed) = (0, False)
        for _ in range(maxls):
            xt = x + step * d
            try:
                (ft, gt) = fun(xt)
            except TrialFailure as failure:
                row = failure_fields(failure.args[0], iteration=nit + 1, retry=fails, step=step, step_norm=float(step * np.linalg.norm(d)))
                on_failure(row)
                (failed, last_failure) = (True, (failure.args[0], row))
                if fails >= retries:
                    raise TrialFailure(dict(failure.args[0], reason=f'all {retries} retries of the line search of iteration {nit + 1} failed', failure=row)) from failure
                fails += 1
                step *= 0.5
                continue
            if ft <= f + c1 * step * gd:
                break
            step *= 0.5
        else:
            return result(2, 'ABNORMAL_TERMINATION_IN_LNSRCH')
        (s, y) = (xt - x, gt - g)
        sy = float(np.dot(s, y))
        if sy > eps * (-step * gd):
            pairs = (pairs + [(s, y, 1.0 / sy)])[-maxcor:]
        (f_old, x, f, g, nit) = (f, xt, ft, gt, nit + 1)
        streak = streak + 1 if failed else 0
        try:
            callback(x)
        except StopIteration:
            return result(2, 'STOP: callback')
        if streak >= failure_iterations:
            raise TrialFailure(dict(last_failure[0], reason=f'failures in the line searches of {failure_iterations} consecutive accepted iterations (last {nit})', failure=last_failure[1]))
        if (f_old - f) / max(abs(f_old), abs(f), 1.0) <= ftol:
            return result(0, 'CONVERGENCE: REL_REDUCTION_OF_F <= FACTR*EPSMCH')
