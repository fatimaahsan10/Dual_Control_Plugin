"""
Box-constrained QP solve for iLQG/DDP control limits.

Ported from Todorov's CODES/TODOROV'S CODE/boxQP.m.

    minimize   0.5 x'Hx + g'x
    subject to lower <= x <= upper

Projected-Newton method: dimensions sitting on a bound with the gradient
pushing further into it are "clamped" (frozen); a Newton step is taken
on the remaining "free" dimensions, then backtracked (Armijo) until it
gives enough decrease without leaving the box.
"""

import numpy as np

RESULT_MESSAGES = {
    -1: "Hessian is not positive definite",
    0: "No descent direction found",
    1: "Maximum main iterations exceeded",
    2: "Maximum line-search iterations exceeded",
    4: "Improvement smaller than tolerance",
    5: "Gradient norm smaller than tolerance",
    6: "All dimensions are clamped",
}


def box_qp(H, g, lower, upper, x0=None, max_iter=100, min_grad=1e-8,
           min_rel_improve=1e-8, step_dec=0.6, min_step=1e-22, armijo=0.1,
           verbose=0):
    """
    Returns
    -------
    x      : (n,)      solution
    result : int        see RESULT_MESSAGES; result >= 1 means success
    Hfree  : (nf, nf)   upper-triangular Cholesky factor of H[free,free]
                        (Hfree.T @ Hfree == H[free,free]), MATLAB chol()
                        convention -- callers solve via
                        solve(Hfree, solve(Hfree.T, b))
    free   : (n,) bool  which dimensions ended up unclamped
    """
    n = H.shape[0]
    g = np.asarray(g, dtype=float).reshape(n)
    lower = np.asarray(lower, dtype=float).reshape(n)
    upper = np.asarray(upper, dtype=float).reshape(n)

    def clamp(z):
        return np.clip(z, lower, upper)

    if x0 is not None:
        x = clamp(np.asarray(x0, dtype=float).reshape(n))
    else:
        with np.errstate(invalid="ignore"):
            x = np.nanmean(np.where(np.isfinite([lower, upper]),
                                     [lower, upper], np.nan), axis=0)
    x = np.where(np.isfinite(x), x, 0.0)

    clamped = np.zeros(n, dtype=bool)
    free = ~clamped
    old_value = 0.0
    result = 0
    gnorm = 0.0
    nfactor = 0
    Hfree = np.zeros((n, n))

    value = x @ g + 0.5 * x @ H @ x

    if verbose > 0:
        print(f"Starting box-QP, dimension {n}, initial value: {value:.3f}")

    it = 0
    for it in range(1, max_iter + 1):
        if result != 0:
            break

        if it > 1 and (old_value - value) < min_rel_improve * abs(old_value):
            result = 4
            break
        old_value = value

        grad = g + H @ x

        old_clamped = clamped
        clamped = ((x == lower) & (grad > 0)) | ((x == upper) & (grad < 0))
        free = ~clamped

        if np.all(clamped):
            result = 6
            break

        factorize = (it == 1) or np.any(old_clamped != clamped)
        if factorize:
            try:
                # numpy gives lower-triangular L (L @ L.T == H_free); store
                # the upper-triangular transpose to match the MATLAB chol()
                # convention that callers (e.g. back_pass.py) expect.
                L = np.linalg.cholesky(H[np.ix_(free, free)])
            except np.linalg.LinAlgError:
                result = -1
                break
            Hfree = L.T
            nfactor += 1

        gnorm = np.linalg.norm(grad[free])
        if gnorm < min_grad:
            result = 5
            break

        grad_clamped = g + H @ (x * clamped)
        search = np.zeros(n)
        b = grad_clamped[free]
        y = np.linalg.solve(L, b)
        search[free] = -np.linalg.solve(L.T, y) - x[free]

        sdotg = np.sum(search * grad)
        if sdotg >= 0:  # should not happen
            break

        step = 1.0
        nstep = 0
        xc = clamp(x + step * search)
        vc = xc @ g + 0.5 * xc @ H @ xc
        while (vc - old_value) / (step * sdotg) < armijo:
            step *= step_dec
            nstep += 1
            xc = clamp(x + step * search)
            vc = xc @ g + 0.5 * xc @ H @ xc
            if step < min_step:
                result = 2
                break

        if verbose > 1:
            print(f"iter {it:3d}  value {vc: -9.5g}  |g| {gnorm:9.3g}  "
                  f"reduction {old_value - vc:9.3g}  n_clamped {clamped.sum()}")

        x = xc
        value = vc

    if it >= max_iter:
        result = 1

    if verbose > 0:
        print(f"RESULT: {RESULT_MESSAGES.get(result, result)}. "
              f"iterations {it}  gradient {gnorm:.6g}  final value {value:.6g}  "
              f"factorizations {nfactor}")

    return x, result, Hfree, free
