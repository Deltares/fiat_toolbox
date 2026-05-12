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


def test_opt_lambda_respects_loss_horizon():
    kwargs = {
        "v": 0.2,
        "k_str": 100000,
        "c0": 20000,
        "pi": 0.1,
        "eta": 1.5,
        "l_min": 0.3,
        "l_max": 1.0,
        "times": np.linspace(0, 10, 100),
        "method": "trapezoid",
        "cmin": 0,
        "liquidity": 0,
        "rho": 0.0,
    }
    full = methods.opt_lambda(**kwargs)
    short = methods.opt_lambda(**kwargs, loss_horizon=2.0)
    assert full["success"]
    assert short["success"]
    assert short["loss_opt"] < full["loss_opt"]


def test_opt_lambda_supports_candidate_recovery_time_horizon():
    kwargs = {
        "v": 0.2,
        "k_str": 100000,
        "c0": 20000,
        "pi": 0.1,
        "eta": 1.5,
        "l_min": 0.3,
        "l_max": 1.0,
        "times": np.linspace(0, 10, 100),
        "method": "trapezoid",
        "cmin": 0,
        "liquidity": 0,
        "rho": 0.0,
    }
    full = methods.opt_lambda(**kwargs)
    candidate = methods.opt_lambda(**kwargs, loss_horizon="recovery_time")

    assert full["success"]
    assert candidate["success"]
    assert not np.isclose(candidate["loss_opt"], full["loss_opt"])


def test_opt_lambda_rejects_invalid_loss_horizon():
    res = methods.opt_lambda(
        v=0.2,
        k_str=100000,
        c0=20000,
        pi=0.1,
        eta=1.5,
        l_min=0.3,
        l_max=1.0,
        times=np.linspace(0, 10, 100),
        method="trapezoid",
        loss_horizon=11.0,
    )
    assert res["success"] is False
    assert "loss_horizon" in (res["message"] or "")


def test_opt_lambda_rejects_invalid_loss_horizon_mode():
    res = methods.opt_lambda(
        v=0.2,
        k_str=100000,
        c0=20000,
        pi=0.1,
        eta=1.5,
        l_min=0.3,
        l_max=1.0,
        times=np.linspace(0, 10, 100),
        method="trapezoid",
        loss_horizon="bad-mode",
    )
    assert res["success"] is False
    assert "loss_horizon" in (res["message"] or "")


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
    # report status=INFEASIBLE, success=False, and name c_min in the message.
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
    assert res["status"] == methods.OptLambdaStatus.INFEASIBLE
    assert "c_min" in (res["message"] or "")


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


# ---------------------------------------------------------------------------
# OptLambdaStatus: six-way outcome classification (+ warning leak)
# ---------------------------------------------------------------------------


def test_opt_lambda_status_interior():
    # A normal case with a plausible interior optimum — the classification
    # must be INTERIOR, success=True, and message=None.
    times = np.linspace(0, 10, 200)
    res = methods.opt_lambda(
        v=0.3,
        k_str=80000.0,
        c0=50000.0,
        pi=0.15,
        eta=1.5,
        l_min=0.3,
        l_max=10.0,
        times=times,
        method="trapezoid",
        cmin=0.0,
        liquidity=0.0,
        rho=0.06,
    )
    assert res["success"] is True
    assert res["status"] == methods.OptLambdaStatus.INTERIOR
    # Interior means l_opt_min is strictly inside (l_min, l_max).
    bound_tol = 1e-6 * (10.0 - 0.3)
    assert res["l_opt_min"] > 0.3 + bound_tol
    assert res["l_opt_min"] < 10.0 - bound_tol
    assert res["message"] is None


def test_opt_lambda_status_flat():
    # There IS physical damage (v=0.2) but liquidity covers every cent of the
    # integrated loss for every λ, so wellbeing is λ-invariant at zero. Must
    # flag FLAT (not NO_RECOVERY_NEEDED — the household has something to
    # rebuild; wellbeing is just indifferent to speed). Pick the coarse-grid
    # argmin with ties toward the *largest* λ (fastest recovery — consistent
    # with eps_rel's convention).
    times = np.linspace(0, 10, 100)
    res = methods.opt_lambda(
        v=0.2,
        k_str=100000.0,
        c0=20000.0,
        pi=0.1,
        eta=1.5,
        l_min=0.3,
        l_max=10.0,
        times=times,
        method="trapezoid",
        liquidity=1e9,
    )
    assert res["success"] is True
    assert res["status"] == methods.OptLambdaStatus.FLAT
    # Ties toward fastest recovery → returned l_opt is at the largest grid
    # point (= l_max).
    assert res["l_opt_min"] >= 10.0 - 1e-9
    assert "flat" in (res["message"] or "").lower()
    assert "fastest" in (res["message"] or "").lower()


def test_opt_lambda_status_no_recovery_needed():
    # Owner has no physical damage → NO_RECOVERY_NEEDED, not FLAT. No λ is
    # reported; loss_opt is the extras-only constant (0.0 when no extras).
    times = np.linspace(0, 10, 100)
    res = methods.opt_lambda(
        v=0.0,
        k_str=100000.0,
        c0=20000.0,
        pi=0.1,
        eta=1.5,
        l_min=0.3,
        l_max=10.0,
        times=times,
        method="trapezoid",
    )
    assert res["status"] == methods.OptLambdaStatus.NO_RECOVERY_NEEDED
    assert res["success"] is True
    assert res["l_opt"] is None
    assert res["l_opt_min"] is None
    assert res["loss_opt"] == 0.0
    assert res["loss_opt_min"] == 0.0
    assert "no physical damage" in (res["message"] or "").lower()


def test_opt_lambda_status_no_recovery_needed_k_zero():
    # k=0 is the other disjunct of v·k == 0. Same outcome.
    times = np.linspace(0, 10, 100)
    res = methods.opt_lambda(
        v=0.5,
        k_str=0.0,
        c0=20000.0,
        pi=0.1,
        eta=1.5,
        l_min=0.3,
        l_max=10.0,
        times=times,
        method="trapezoid",
    )
    assert res["status"] == methods.OptLambdaStatus.NO_RECOVERY_NEEDED
    assert res["l_opt"] is None


def test_opt_lambda_no_recovery_needed_with_extras():
    # v=0 but rental/labour extras contribute a λ-independent constant. The
    # status is still NO_RECOVERY_NEEDED (owner has nothing to recover), but
    # loss_opt reflects the extras' integrated utility loss.
    times = np.linspace(0, 10, 100)
    res = methods.opt_lambda(
        v=0.0,
        k_str=100000.0,
        c0=20000.0,
        pi=0.1,
        eta=1.5,
        l_min=0.3,
        l_max=10.0,
        times=times,
        method="trapezoid",
        extra_losses=[(1000.0, 0.5)],
    )
    assert res["status"] == methods.OptLambdaStatus.NO_RECOVERY_NEEDED
    assert res["success"] is True
    assert res["l_opt"] is None
    assert res["loss_opt"] > 0.0  # extras contribute a nonzero constant


def test_opt_lambda_no_recovery_needed_demoted_to_infeasible():
    # v=0 but extras are catastrophic enough to push c(t) below c_min. The
    # early-return block must detect infeasibility and demote to INFEASIBLE
    # (honours priority: INFEASIBLE > FAILED > NO_RECOVERY_NEEDED > FLAT > ...).
    times = np.linspace(0, 10, 100)
    res = methods.opt_lambda(
        v=0.0,
        k_str=100000.0,
        c0=1000.0,
        pi=0.1,
        eta=1.5,
        l_min=0.3,
        l_max=10.0,
        times=times,
        method="trapezoid",
        cmin=500.0,
        extra_losses=[(900.0, 0.1)],
    )
    assert res["status"] == methods.OptLambdaStatus.INFEASIBLE
    assert res["success"] is False


def test_opt_lambda_flat_prefers_fastest_under_low_lambda_tilt(monkeypatch):
    # Regression for the FLAT tie-break: under a sub-tolerance monotone slope
    # that favours low λ (slow recovery), FLAT classification must still pick
    # the *largest* λ (fastest recovery). The prior reversed-argmin only broke
    # ties toward fast-recovery when probe losses were bit-for-bit equal; any
    # real-world noise tilted toward slow would slip through.
    base = 100.0
    tilt = 1e-6  # Δ(loss)/Δλ — tiny positive slope → low λ slightly cheaper.

    def tilted_total(self, rho=0.0, method="quad", t1=None, t2=None):
        rr = float(np.ravel(self.rec_rate)[0])
        return base + rr * tilt

    monkeypatch.setattr(methods.UtilityLoss, "total", tilted_total)

    # Tiny but nonzero v keeps owner damage real (so NO_RECOVERY_NEEDED's
    # v·k==0 early-return does NOT intercept) while still allowing the
    # feasibility guard at methods.opt_lambda's objective (`c0 - cl_peak
    # < cmin → +∞`) to accept every probe λ. That lets the patched
    # UtilityLoss.total drive the probe losses cleanly — the sub-tolerance
    # low-λ tilt is then the only signal the FLAT branch has to work with.
    l_min, l_max = 0.3, 10.0
    times = np.linspace(0, 10, 100)
    res = methods.opt_lambda(
        v=0.01,
        k_str=1000.0,
        c0=20000.0,
        pi=0.01,
        eta=1.5,
        l_min=l_min,
        l_max=l_max,
        times=times,
        method="trapezoid",
    )
    # loss_range ≈ (l_max − l_min) · tilt ≈ 9.7e-6; loss_scale ≈ base = 100.
    # eps_flat · loss_scale = 1e-3 · 100 = 0.1  ≫  loss_range, so FLAT fires.
    assert res["status"] == methods.OptLambdaStatus.FLAT, (
        f"expected FLAT classification, got {res['status']} — "
        f"tilt/base ratio may have broken the eps_flat budget"
    )
    # The bug case: true argmin is at the smallest λ (= l_min). The fix picks
    # the largest λ whose loss is within eps_flat · loss_scale of the min,
    # which — given the flat classification — is every finite point. So the
    # returned λ must be at l_max (fastest recovery), NOT at l_min.
    assert res["l_opt_min"] >= l_max - 1e-9, (
        f"FLAT branch returned λ={res['l_opt_min']:.6g}; expected ≈ {l_max} "
        "(fastest recovery). Low-λ tilt slipped through the tie-break."
    )


def test_opt_lambda_cliff_bounded_plateau_picks_fastest(monkeypatch):
    # Regression for the cliff-bounded plateau: utility loss is identically
    # zero across most of the search range (e.g. liquidity covers all losses)
    # but spikes sharply at the slowest-λ end (feasibility cliff / liquidity
    # exhaustion). Without the plateau-at-min tie-break, Nelder-Mead — started
    # at l_min on the cliff — walks down-gradient and stops at the cliff
    # edge: the *slowest* λ in the plateau, opposite of the convention.
    # The fix: detect the plateau via "loss within eps_flat · loss_scale of
    # the minimum" and re-pick l_opt to the largest λ in it. Classification
    # is BOUNDARY_UPPER (not FLAT) because the plateau touches only the
    # upper bound — the cliff at l_min keeps it from being globally flat.
    cliff_height = 3.5e-5

    def cliff_then_flat(self, rho=0.0, method="quad", t1=None, t2=None):
        rr = float(np.ravel(self.rec_rate)[0])
        # Sharp ramp on the slow-λ end (rr < 0.5), zero everywhere else.
        if rr < 0.5:
            return cliff_height * (0.5 - rr) / (0.5 - 0.3)
        return 0.0

    monkeypatch.setattr(methods.UtilityLoss, "total", cliff_then_flat)

    # Tiny v keeps owner damage real (so v·k != 0 → NO_RECOVERY_NEEDED's
    # early-return doesn't intercept) while letting the patched
    # UtilityLoss.total drive the probe losses cleanly.
    l_min, l_max = 0.3, 10.0
    times = np.linspace(0, 10, 100)
    res = methods.opt_lambda(
        v=0.01,
        k_str=1000.0,
        c0=20000.0,
        pi=0.01,
        eta=1.5,
        l_min=l_min,
        l_max=l_max,
        times=times,
        method="trapezoid",
    )
    # Plateau touches l_max but not l_min (cliff there), so classification
    # is BOUNDARY_UPPER, not FLAT. The pre-fix path stopped NM at the cliff
    # edge and reported BOUNDARY_LOWER (or an interior cliff-edge λ).
    assert res["status"] == methods.OptLambdaStatus.BOUNDARY_UPPER, (
        f"expected BOUNDARY_UPPER (plateau touches only l_max), got {res['status']}"
    )
    # The plateau hint must be appended to the BOUNDARY_UPPER message so
    # callers see that the surface is flat all the way back to ~λ=0.5.
    assert "plateau" in (res["message"] or "").lower()
    # Tie-break must pick the largest λ in the plateau (= l_max), NOT the
    # cliff-side edge near λ ≈ 0.5 where Nelder-Mead would otherwise stop.
    assert res["l_opt_min"] >= l_max - 1e-9, (
        f"plateau tie-break returned λ={res['l_opt_min']:.6g}; expected "
        f"≈ {l_max} (fastest recovery in plateau). NM stalled at the cliff edge."
    )


def test_opt_lambda_status_boundary_lower():
    # Notebook-4-like tight-feasibility config: heavy damage pushes the
    # feasible λ range to the slow end; the optimum lands near l_min.
    # The t_max diagnostic should fire because recovery is incomplete in 10y.
    times = np.linspace(0, 10, 520)
    res = methods.opt_lambda(
        v=0.7,
        k_str=120000.0,
        c0=65000.0,
        pi=0.15,
        eta=1.5,
        l_min=0.3,
        l_max=10.0,
        times=times,
        method="trapezoid",
        cmin=5000.0,
        liquidity=0.0,
        rho=0.06,
        recovery_per=95.0,
    )
    assert res["success"] is True
    assert res["status"] == methods.OptLambdaStatus.BOUNDARY_LOWER
    assert abs(res["l_opt_min"] - 0.3) < 1e-4
    msg = res["message"] or ""
    assert "lower λ bound" in msg
    # t_max hint should fire: 1 - exp(-0.3 * 10) ≈ 0.95, may or may not trip.
    # If the fraction is strictly below recovery_per, the hint appears.


def test_opt_lambda_status_boundary_upper():
    # Narrow the search range so that the natural interior optimum is above
    # l_max — NM lands at l_max → BOUNDARY_UPPER.
    times = np.linspace(0, 10, 200)
    res = methods.opt_lambda(
        v=0.1,
        k_str=100000.0,
        c0=40000.0,
        pi=0.15,
        eta=1.5,
        l_min=0.3,
        l_max=0.4,  # tight upper bound
        times=times,
        method="trapezoid",
        cmin=0.0,
        liquidity=0.0,
        rho=0.06,
    )
    assert res["success"] is True
    # Either BOUNDARY_UPPER (NM lands at l_max) or FLAT (wellbeing ≈ constant
    # over this narrow range). Both are acceptable; this test asserts it's
    # not INTERIOR and not a lower-bound hit.
    assert res["status"] in (
        methods.OptLambdaStatus.BOUNDARY_UPPER,
        methods.OptLambdaStatus.FLAT,
    )
    if res["status"] == methods.OptLambdaStatus.BOUNDARY_UPPER:
        assert abs(res["l_opt_min"] - 0.4) < 1e-4
        assert "upper λ bound" in (res["message"] or "")


def test_opt_lambda_status_infeasible():
    # Same setup as test_opt_lambda_infeasible_everywhere_fails_gracefully
    # but focused on the new status field explicitly.
    times = np.linspace(0, 5, 50)
    res = methods.opt_lambda(
        v=0.9,
        k_str=100000.0,
        c0=20000.0,
        pi=0.1,
        eta=1.5,
        l_min=1.0,
        l_max=5.0,
        times=times,
        method="trapezoid",
        cmin=15000.0,
    )
    assert res["success"] is False
    assert res["status"] == methods.OptLambdaStatus.INFEASIBLE


def test_opt_lambda_no_leaked_consumption_warning():
    # Infeasible λ candidates inside the solver emit
    # "Consumption contains zero or negative values" from utility().
    # The objective now suppresses that filter scope; the solver still
    # rejects the candidate via +∞, but no UserWarning should reach the
    # caller.
    import warnings as _w

    times = np.linspace(0, 10, 100)
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        methods.opt_lambda(
            v=0.7,
            k_str=120000.0,
            c0=65000.0,
            pi=0.15,
            eta=1.5,
            l_min=0.3,
            l_max=10.0,
            times=times,
            method="trapezoid",
            cmin=5000.0,
            liquidity=0.0,
            rho=0.06,
        )
    leaked = [
        w
        for w in caught
        if "Consumption contains zero or negative values" in str(w.message)
    ]
    assert leaked == [], (
        f"UserWarning leaked out of opt_lambda: {[str(w.message) for w in leaked]}"
    )
    # Also check the scipy RuntimeWarning doesn't leak. Nelder-Mead's
    # termination check computes fsim[0] - fsim[1:] on the simplex; when a
    # simplex vertex holds +inf from an infeasible candidate, numpy emits
    # "invalid value encountered in subtract". This is exploration chatter,
    # not a user-facing issue.
    scipy_leaks = [
        w for w in caught if "invalid value encountered in subtract" in str(w.message)
    ]
    assert scipy_leaks == [], (
        f"scipy RuntimeWarning leaked out of opt_lambda: "
        f"{[str(w.message) for w in scipy_leaks]}"
    )


def test_consumption_loss_t_liquidity_smooths_extras_when_alpha_base_zero():
    # When the owner term is zero (v·k = 0) but extras drive a non-zero loss
    # stream, liquidity must still smooth the extras. Previously the guard
    # at line 327 dropped the liquidity branch whenever rec_rate ≤ 0, and a
    # symmetric trap held for the v·k = 0 + placeholder rec_rate path inside
    # CommunityUnit. With the fix, the t̂ logic handles α_base = 0 directly.
    t = np.linspace(0, 5, 51)
    extra = [(1000.0, 1.5)]  # N0 = 1000, lam = 1.5 → lifetime ≈ 666.7
    # liquidity below lifetime so we expect a non-trivial t̂ plateau
    cl = methods.consumption_loss_t(
        t,
        rec_rate=1.0,
        v=0.0,
        k_str=100000,
        pi=0.1,
        liquidity=200.0,
        extra_losses=extra,
    )
    # Without the fix, the function falls through to the unsupported path and
    # cl[0] equals N0 = 1000. With the fix, liquidity smooths the early loss
    # so cl[0] is the plateau Δc(t̂) < 1000.
    assert cl[0] < 1000.0
    # Plateau then exponential decay → overall non-increasing.
    diffs = np.diff(cl)
    assert np.all(diffs <= 1e-9)
