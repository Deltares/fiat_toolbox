import numpy as np

from fiat_toolbox.well_being import methods


def test_utility_basic():
    assert methods.utility(10, 1.5) < 10
    arr = np.array([10, 20, 30])
    result = methods.utility(arr, 1.5)
    assert isinstance(result, np.ndarray)
    assert result.shape == arr.shape


def test_inverse_utility():
    u = methods.utility(10, 1.5)
    c = methods.inverse_utility(u, 1.5)
    assert np.isclose(c, 10)


def test_recovery_time_and_rate():
    rate = 0.5
    rebuilt_per = 90
    t = methods.recovery_time(rate, rebuilt_per)
    r = methods.recovery_rate(t, rebuilt_per)
    assert np.isclose(r, rate)


def test_recovery_cost_t():
    t = np.linspace(0, 1, 5)
    rec_rate, v, k_str = 0.5, 0.2, 100000
    cost = methods.recovery_cost_t(t, rec_rate, v, k_str)
    assert cost.shape == t.shape
    assert np.all(cost >= 0)


def test_income_loss_t():
    t = np.linspace(0, 1, 5)
    rec_rate, v, k_str, pi = 0.5, 0.2, 100000, 0.1
    loss = methods.income_loss_t(t, rec_rate, v, k_str, pi)
    assert loss.shape == t.shape
    assert np.all(loss >= 0)


def test_consumption_loss_t():
    t = np.linspace(0, 1, 5)
    rec_rate, v, k_str, pi = 0.5, 0.2, 100000, 0.1
    cl = methods.consumption_loss_t(t, rec_rate, v, k_str, pi, liquidity=1700)
    assert cl.shape == t.shape
    assert np.all(cl >= 0)


def test_consumption_loss_t_with_extra_losses():
    t = np.linspace(0, 2, 21)
    rec_rate, v, k_str, pi = 0.5, 0.2, 100000, 0.1
    # Two extra loss components with different decay rates
    extra = [(500.0, 0.3), (300.0, 0.8)]
    cl = methods.consumption_loss_t(
        t,
        rec_rate,
        v,
        k_str,
        pi,
        liquidity=0.0,
        extra_losses=extra,
    )
    assert isinstance(cl, np.ndarray)
    assert cl.shape == t.shape
    # Loss at t=0 should be larger than at later times due to decay
    assert cl[0] > cl[-1]


def test_consumption_loss_t_liquidity_fully_offsets():
    t = np.linspace(0, 2, 21)
    rec_rate, v, k_str, pi = 0.5, 0.2, 100000, 0.1
    # Compute a liquidity large enough to offset all losses
    alpha_base = pi * v * k_str + rec_rate * v * k_str
    extra = [(500.0, 0.5)]
    total_integral = (alpha_base / rec_rate) + sum(N0 / lam for N0, lam in extra)
    cl = methods.consumption_loss_t(
        t,
        rec_rate,
        v,
        k_str,
        pi,
        liquidity=total_integral + 1.0,  # just above threshold
        extra_losses=extra,
    )
    # All losses are fully offset
    assert np.allclose(cl, 0.0)


def test_consumption_t():
    t = np.linspace(0, 1, 5)
    rec_rate, v, k_str, pi, c0 = 0.5, 0.2, 100000, 0.1, 15000
    ct = methods.consumption_t(
        t,
        rec_rate,
        v,
        k_str,
        pi,
        c0,
        cmin=1000,
        liquidity=1700,
    )
    assert ct.shape == t.shape


def test_utility_loss_t():
    t = np.linspace(0, 1, 5)
    rec_rate, v, k_str, pi, c0, eta = 0.5, 0.2, 100000, 0.1, 15000, 1.5
    ul = methods.utility_loss_t(
        t,
        rec_rate,
        v,
        k_str,
        pi,
        c0,
        eta,
        cmin=1000,
        liquidity=1700,
    )
    assert ul.shape == t.shape


def test_utility_loss_t_with_extra_losses():
    t = np.linspace(0, 2, 21)
    rec_rate, v, k_str, pi, c0, eta = 0.5, 0.2, 100000, 0.1, 15000, 1.5
    extra = [(600.0, 0.4), (250.0, 0.7)]
    ul = methods.utility_loss_t(
        t,
        rec_rate,
        v,
        k_str,
        pi,
        c0,
        eta,
        cmin=1000,
        liquidity=0.0,
        extra_losses=extra,
    )
    assert isinstance(ul, np.ndarray)
    assert ul.shape == t.shape


def test_wellbeing_loss():
    du = 100
    c_avg = 15000
    eta = 1.5
    wl = methods.wellbeing_loss(du, c_avg, eta)
    assert isinstance(wl, float)


def test_equity_weight():
    c = 10000
    c_avg = 15000
    eta = 1.5
    ew = methods.equity_weight(c, c_avg, eta)
    assert isinstance(ew, float)


def test_opt_lambda_runs():
    result = methods.opt_lambda(
        v=0.2,
        k_str=100000,
        c0=20000,
        pi=0.1,
        eta=1.5,
        l_min=0.3,
        l_max=1.0,
        t_max=2.0,
        times=np.linspace(0, 2, 10),
        method="trapezoid",
        cmin=1000,
        eps_rel=0.01,
        liquidity=1700,
    )
    assert "l_opt" in result
    assert "loss_opt" in result


def test_utility_warns_on_nonpositive_consumption():
    # S1: the warning for c <= 0 was previously commented out. Subzero
    # consumption silently became NaN. Now it must warn so callers can see
    # when c_min + losses pushed c(t) below zero.
    import warnings as _w

    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        result = methods.utility(np.array([10.0, -1.0, 5.0]), 1.5)
    assert any("zero or negative" in str(w.message).lower() for w in caught), (
        "utility() must emit a UserWarning when consumption <= 0"
    )
    # And the result still coerces subzero values to NaN (unchanged behavior)
    assert np.isnan(result[1])


def test_utility_loss_t_baseline_is_c0_not_c0_minus_cmin():
    # Baseline utility is u(c_0), so the c_min subtraction from the old
    # Stone-Geary-style code is gone. At t=0 with no losses, utility loss
    # must be zero regardless of cmin.
    rec_rate, v, k_str, pi, c0, eta = 0.5, 0.0, 100000.0, 0.1, 20000.0, 1.5
    ul_zero_loss = np.atleast_1d(
        methods.utility_loss_t(
            t=np.array([0.0]),
            rec_rate=rec_rate,
            v=v,
            k_str=k_str,
            pi=pi,
            c0=c0,
            eta=eta,
            cmin=3000.0,
        )
    )
    assert np.isclose(float(ul_zero_loss[0]), 0.0), (
        "With v=0 no losses occur, so u(c_0) - u(c_t) must be 0 regardless of cmin"
    )
    # And the residual at t=0 with losses equals u(c_0) - u(c_0 - Δc(0)),
    # not u(c_0 - cmin) - u(c_0 - Δc(0) - cmin)
    v = 0.2
    cl0 = np.atleast_1d(
        methods.consumption_loss_t(
            t=np.array([0.0]), rec_rate=rec_rate, v=v, k_str=k_str, pi=pi
        )
    )
    expected = methods.utility(c0, eta) - methods.utility(c0 - float(cl0[0]), eta)
    got = np.atleast_1d(
        methods.utility_loss_t(
            t=np.array([0.0]),
            rec_rate=rec_rate,
            v=v,
            k_str=k_str,
            pi=pi,
            c0=c0,
            eta=eta,
            cmin=1000.0,  # irrelevant to the loss value
        )
    )
    assert np.isclose(float(got[0]), float(expected))


def test_opt_lambda_rejects_infeasible_lambda_at_cmin():
    # c(t) >= c_min is a feasibility constraint inside opt_lambda. Fast
    # lambdas generate high recovery-cost peaks that would push c(t) < c_min;
    # the objective must return +inf for those.
    times = np.linspace(0, 5, 50)
    # Tight setup: peak Δc at t=0 for λ=2 is (pi + λ)·v·k = (0.1+2)*0.2*100000 = 42000
    # c0 - cmin = 20000 - 15000 = 5000, so λ=2 is firmly infeasible.
    # Use an objective call via the internal minimizer path: request a tight
    # range where only slow λ are feasible, then confirm the solution respects
    # the floor.
    res = methods.opt_lambda(
        v=0.2,
        k_str=100000.0,
        c0=20000.0,
        pi=0.1,
        eta=1.5,
        l_min=0.05,
        l_max=2.0,
        times=times,
        method="trapezoid",
        cmin=15000.0,
        liquidity=0.0,
    )
    assert res["success"], res["message"]
    l_opt = res["l_opt"]
    # Reconstruct c(t) peak at the chosen λ and confirm it satisfies the floor.
    cl = methods.consumption_loss_t(
        t=times, rec_rate=l_opt, v=0.2, k_str=100000.0, pi=0.1, liquidity=0.0
    )
    peak = float(np.max(cl))
    assert 20000.0 - peak >= 15000.0 - 1e-6, (
        f"optimizer returned λ={l_opt} violating c(t) >= cmin (peak Δc={peak})"
    )


def test_opt_lambda_infeasible_everywhere_fails_gracefully():
    # When every λ in the search range is infeasible, the result dict must
    # report success=False and flag the cmin cause in the message.
    times = np.linspace(0, 5, 50)
    res = methods.opt_lambda(
        v=0.9,  # huge loss
        k_str=100000.0,
        c0=20000.0,
        pi=0.1,
        eta=1.5,
        l_min=1.0,  # all λ force large peak Δc
        l_max=5.0,
        times=times,
        method="trapezoid",
        cmin=15000.0,
    )
    assert not res["success"]
    # The message uses the "drops below the threshold" phrasing.
    assert "threshold" in (res["message"] or "").lower()


def test_opt_lambda_rho_threads_into_objective():
    # rho argument was previously hardcoded to 0. Test by comparing the
    # *objective value* at a fixed lambda: UtilityLoss.total(rho=...) must
    # strictly decrease as rho increases (discount reduces integrand weight
    # at later t where losses have tailed off).
    times = np.linspace(0, 10, 200)
    ut = methods.UtilityLoss(
        t=times,
        rec_rate=0.5,
        v=0.2,
        k_str=100000.0,
        pi=0.1,
        c0=20000.0,
        eta=1.5,
        cmin=0.0,
        liquidity=0.0,
    )
    loss_zero = ut.total(rho=0.0, method="trapezoid")
    loss_disc = ut.total(rho=0.06, method="trapezoid")
    assert loss_disc < loss_zero, (
        f"discounted loss ({loss_disc}) must be < undiscounted ({loss_zero})"
    )
    # And the shape must flow from the optimizer: opt_lambda's internal grid
    # objective must respond to rho at a fixed lambda too.
    grid_times = np.linspace(0, 10, 100)
    # Use a reachable but non-trivial l_max/l_min range where NM moves.
    res_zero = methods.opt_lambda(
        v=0.2,
        k_str=100000.0,
        c0=20000.0,
        pi=0.1,
        eta=1.5,
        l_min=0.05,
        l_max=2.0,
        times=grid_times,
        method="trapezoid",
        cmin=0.0,
        liquidity=0.0,
        rho=0.0,
    )
    res_disc = methods.opt_lambda(
        v=0.2,
        k_str=100000.0,
        c0=20000.0,
        pi=0.1,
        eta=1.5,
        l_min=0.05,
        l_max=2.0,
        times=grid_times,
        method="trapezoid",
        cmin=0.0,
        liquidity=0.0,
        rho=0.06,
    )
    assert res_zero["success"] and res_disc["success"]
    # At a minimum, the *loss value* at the optimum must differ between rho
    # settings (even if NM happens to land at the same boundary point).
    assert not np.isclose(res_zero["loss_opt"], res_disc["loss_opt"], rtol=1e-3), (
        "loss_opt did not respond to rho — rho not threaded through opt_lambda"
    )


def test_quad_integration_does_not_leak_warning_filter():
    # S2: previously `warnings.filterwarnings("ignore", ...)` was called
    # without a context manager, permanently silencing IntegrationWarning in
    # the caller's process. Now it must be context-managed.
    import warnings as _w

    # Run a quad integration that historically triggered the leak.
    loss = methods.RecoveryCost(
        t=np.array([0.0, 10.0]), rec_rate=0.5, v=0.2, k_str=100000.0
    )
    _ = loss.total(rho=0.0, method="quad")

    # After the quad call, emit an IntegrationWarning and confirm it is NOT
    # suppressed (i.e. the filter did not leak).
    from scipy.integrate import IntegrationWarning

    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        _w.warn("leak-check", IntegrationWarning)
    assert any("leak-check" in str(w.message) for w in caught), (
        "warnings.filterwarnings(IntegrationWarning) leaked out of Loss.total"
    )
