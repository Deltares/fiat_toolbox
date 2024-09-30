import math

# Instead of having fiat core as a dependency for this one function that doesnt use any other functions from fiat core,
# the function was copied here (26/9/2024, 661e0f2b2d6396346140316412c5957bc10eb03b) from https://github.com/Deltares/Delft-FIAT/blob/master/src/fiat/models/calc.py 

def calc_rp_coef(
    rp: list | tuple,
):
    """
    Calculates coefficients used to compute the EAD as a linear function of
    the known damages.

    Parameters
    ----------
    rp : list or tuple of int
        Return periods T1 … Tn for which damages are known.

    Returns
    -------
    list of float
        Coefficients a1, …, an used to compute the EAD as a linear function of the known damages.

    Notes
    -----
    In which D(f) is the damage, D, as a function of the frequency of exceedance, f.
    In order to compute this EAD, function D(f) needs to be known for the entire range of frequencies.
    Instead, D(f) is only given for the n frequencies as mentioned in the table above. 
    So, in order to compute the integral above, some assumptions need to be made for function D(f):
    (i)   For f > f1 the damage is assumed to be equal to 0.
    (ii)  For f < fn, the damage is assumed to be equal to Dn.
    (iii) For all other frequencies, the damage is estimated from log-linear interpolation 
          between the known damages and frequencies.
    """
    # Step 1: Compute frequencies associated with T-values.
    _rp = sorted(rp)
    idxs = [_rp.index(n) for n in rp]
    rp_u = sorted(rp)
    rp_l = len(rp_u)

    f = [1 / n for n in rp_u]
    lf = [math.log(1 / n) for n in rp_u]

    if rp_l == 1:
        return f

    # Step 2:
    c = [(1 / (lf[idx] - lf[idx + 1])) for idx in range(rp_l - 1)]

    # Step 3:
    G = [(f[idx] * lf[idx] - f[idx]) for idx in range(rp_l)]

    # Step 4:
    a = [
        (
            (1 + c[idx] * lf[idx + 1]) * (f[idx] - f[idx + 1])
            + c[idx] * (G[idx + 1] - G[idx])
        )
        for idx in range(rp_l - 1)
    ]
    b = [
        (c[idx] * (G[idx] - G[idx + 1] + lf[idx + 1] * (f[idx + 1] - f[idx])))
        for idx in range(rp_l - 1)
    ]

    # Step 5:
    alpha = [
        b[0]
        if idx == 0
        else f[idx] + a[idx - 1]
        if idx == rp_l - 1
        else a[idx - 1] + b[idx]
        for idx in range(rp_l)
    ]

    return [alpha[idx] for idx in idxs]

