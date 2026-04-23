from fiat_toolbox.well_being.loss import (
    CapitalStock,
    CommunityUnit,
    IncomeConfig,
    IncomeStream,
    Liquidity,
    LossType,
    SimulationConfig,
    WellBeingConfig,
)


def _make_config(
    v=0.2,
    k=100000,
    rec_rate=0.5,
    i0=20000,
    iavg=18000,
    pi=0.1,
    eta=1.5,
    rho=0.05,
    t_max=5,
    dt=0.1,
    currency="€",
    cmin=1000,
    recovery_per=90.0,
    savings=5000,
    insurance=2000,
    support=1000,
):
    return WellBeingConfig(
        owner_housing=CapitalStock(k=k, v=v, recovery_rate=rec_rate, pi=pi),
        income=IncomeConfig(i_0=i0, i_avg=iavg),
        liquidity=Liquidity(savings=savings, insurance=insurance, support=support),
        simulation=SimulationConfig(
            eta=eta,
            rho=rho,
            t_max=t_max,
            dt=dt,
            currency=currency,
            c_min=cmin,
            recovery_per=recovery_per,
        ),
    )


def test_household_initialization():
    # Default path: i_0 omitted → c0 derived from Σ π·k + (i_div or 0).
    # Fixture defaults: pi=0.1, k=100000, no rental/labour, no i_div → c0 = 10000.
    import pytest

    config = _make_config(i0=None)
    hh = CommunityUnit(config)
    assert hh.config.owner_housing.v == 0.2
    assert hh.config.owner_housing.k == 100000
    assert hh._c0() == 10000
    assert hh.config.simulation.currency == "€"
    assert hh._liquidity() == 5000 + 2000 + 1000

    # Override path: i_0 supplied and ≠ Σ π·k → override in effect with warning.
    override_config = _make_config(i0=20000)
    with pytest.warns(UserWarning, match="overrides the stock-derived baseline"):
        hh_override = CommunityUnit(override_config)
    assert hh_override._c0() == 20000


def test_calc_loss_reconstruction():
    config = _make_config(v=0.1, k=50000, rec_rate=0.7, i0=15000, iavg=14000)
    hh = CommunityUnit(config)
    loss = hh.calc_loss(LossType.RECOVERY_COST)
    assert loss > 0


def test_calc_loss_income():
    config = _make_config(v=0.1, k=50000, rec_rate=0.7, i0=15000, iavg=14000)
    hh = CommunityUnit(config)
    loss = hh.calc_loss(LossType.OWNER_HOUSING_LOSS)
    assert loss > 0


def test_calc_loss_utility():
    config = _make_config(v=0.1, k=50000, rec_rate=0.7, i0=15000, iavg=14000)
    hh = CommunityUnit(config)
    loss = hh.calc_loss(LossType.UTILITY_LOSS)
    assert loss >= 0


def test_get_losses():
    config = _make_config(v=0.1, k=50000, rec_rate=0.7, i0=15000, iavg=14000)
    hh = CommunityUnit(config)
    losses = hh.get_losses()
    assert "Wellbeing Loss" in losses
    assert "Asset Loss" in losses
    assert "Equity Weighted Asset Loss" in losses
    # Only configured loss types should be present
    assert LossType.RECOVERY_COST in losses
    assert LossType.OWNER_HOUSING_LOSS in losses
    assert LossType.CONSUMPTION_LOSS in losses
    assert LossType.UTILITY_LOSS in losses
    assert LossType.RENTAL_HOUSING_LOSS not in losses
    assert LossType.LABOUR_INCOME_LOSS not in losses


def test_get_losses_with_rental_and_labour():
    # Configure rental housing and two labour assets with different recovery specs
    base = _make_config(v=0.1, k=50000, rec_rate=0.7, i0=15000, iavg=14000)
    base.rental_housing = CapitalStock(k=30000, v=0.15, recovery_time=1.5)
    base.labour_assets = {
        "shop": CapitalStock(k=20000, v=0.1, recovery_rate=0.6),
        "workshop": CapitalStock(k=15000, v=0.08, recovery_time=2.0),
    }
    hh = CommunityUnit(base)
    losses = hh.get_losses()
    # Rental and aggregate labour income losses should be present
    assert LossType.RENTAL_HOUSING_LOSS in losses
    assert LossType.LABOUR_INCOME_LOSS in losses
    # Per-asset labour components should also be stored
    assert any(
        isinstance(c, str) and c.startswith(f"{LossType.LABOUR_INCOME_LOSS.value} (")
        for c in hh.time_series.columns
    )
    # Recovery time should be computed and exposed
    assert hh.recovery_time is not None


def test_achieved_recovery_percent_exponential_and_realized():
    # Keep liquidity below full-offset threshold to ensure defined realized percentage
    base = _make_config(
        v=0.1,
        k=50000,
        rec_rate=0.7,
        i0=15000,
        iavg=14000,
        savings=1000,
        insurance=500,
        support=0,
    )
    base.rental_housing = CapitalStock(k=30000, v=0.15, recovery_rate=0.5)
    hh = CommunityUnit(base)
    # Exponential model (ignores liquidity)
    perc_exp = hh.achieved_recovery_percent(t=2.0, realized=False)
    assert isinstance(perc_exp, float)
    assert 0.0 <= perc_exp <= 100.0
    # Realized model with liquidity effects
    perc_real = hh.achieved_recovery_percent(t=2.0, realized=True)
    assert isinstance(perc_real, float)
    assert 0.0 <= perc_real <= 100.0


def test_validation_and_completion_of_recovery_params():
    # Provide only recovery_time; recovery_rate should be filled
    base = _make_config(rec_rate=None)
    base.owner_housing.recovery_time = 1.0
    hh = CommunityUnit(base)
    assert hh.config.owner_housing.recovery_rate is not None
    assert hh.config.owner_housing.recovery_time is not None


def test_opt_lambda_runs():
    config = _make_config(v=0.1, k=50000, rec_rate=None, i0=15000, iavg=14000)
    hh = CommunityUnit(config)
    hh.opt_lambda(no_steps=10)
    # No assertion, just check it runs without error


def test_repr():
    config = _make_config(v=0.1, k=50000, rec_rate=None, i0=15000, iavg=14000)
    hh = CommunityUnit(config)
    s = repr(hh)
    assert "CommunityUnit(" in s
    assert "owner_housing" in s


def test_plot_loss_all_types():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = _make_config(v=0.1, k=50000, rec_rate=0.7, i0=15000, iavg=14000)
    hh = CommunityUnit(config)
    for lt in hh._loss_types_for_run():
        hh.calc_loss(lt)
        # Test with no ax provided
        fig = hh.plot_loss(lt)
        assert fig is None or hasattr(fig, "savefig")
        # Test with ax provided
        fig2, ax2 = plt.subplots()
        result = hh.plot_loss(lt, ax=ax2)
        assert result is None


def test_plot_consumption():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = _make_config(v=0.1, k=50000, rec_rate=0.7, i0=15000, iavg=14000)
    hh = CommunityUnit(config)
    hh.get_losses()
    # Test with no ax, plot_cmin False
    fig = hh.plot_consumption()
    assert fig is None or hasattr(fig, "savefig")
    # Test with no ax, plot_cmin True
    fig2 = hh.plot_consumption(plot_cmin=True)
    assert fig2 is None or hasattr(fig2, "savefig")
    # Test with ax provided
    fig3, ax3 = plt.subplots()
    result = hh.plot_consumption(ax=ax3)
    assert result is None
    # Test with ax and plot_cmin True
    fig4, ax4 = plt.subplots()
    result2 = hh.plot_consumption(ax=ax4, plot_cmin=True)
    assert result2 is None


def test_plot_opt_lambda():
    import matplotlib

    matplotlib.use("Agg")
    config = _make_config(v=0.1, k=50000, rec_rate=0.7, i0=15000, iavg=14000)
    hh = CommunityUnit(config)
    hh.opt_lambda(no_steps=10)
    # Test with default x_type ("rate")
    fig = hh.plot_opt_lambda()
    assert fig is None or hasattr(fig, "savefig")
    # Test with x_type="time"
    fig2 = hh.plot_opt_lambda(x_type="time")
    assert fig2 is None or hasattr(fig2, "savefig")


def test_opt_lambda_flags_flat_objective_on_zero_damage():
    # With v=0, utility loss is identically 0 across all lambdas; the
    # optimizer must classify status=FLAT rather than returning arbitrary
    # Nelder-Mead noise.
    from fiat_toolbox.well_being.methods import OptLambdaStatus

    config = WellBeingConfig(
        owner_housing=CapitalStock(k=100000.0, v=0.0, pi=0.1),
        income=IncomeConfig(i_0=20000.0, i_avg=20000.0),
        simulation=SimulationConfig(t_max=10, dt=0.1),
    )
    hh = CommunityUnit(config)
    opt = hh.opt_lambda(no_steps=10)
    assert opt["success"] is True
    assert opt["status"] == OptLambdaStatus.FLAT, (
        "zero-damage case should be classified as FLAT"
    )


def test_exponential_loss_components_owner_uses_pi_only():
    # The owner coefficient must be pi * v * k (NOT (pi + lambda) * v * k).
    # A previously buggy aggregator inflated the owner term by lambda, which
    # pulled the aggregate recovery time below the correct mix when lambda
    # was large.
    config = _make_config(v=0.1, k=50000, rec_rate=5.0, i0=15000, iavg=14000, pi=0.1)
    hh = CommunityUnit(config)
    comps = hh._exponential_loss_components()
    assert comps is not None
    coeffs, rates = comps
    # With only owner housing present, expect exactly one component.
    assert len(coeffs) == 1 and len(rates) == 1
    expected_c = 0.1 * 0.1 * 50000  # pi * v * k
    assert abs(coeffs[0] - expected_c) < 1e-9, (
        f"owner coefficient {coeffs[0]} != pi*v*k = {expected_c}; the +lambda "
        "asymmetry regressed"
    )
    assert abs(rates[0] - 5.0) < 1e-9


def test_recovery_times_per_component_structure():
    # With rental + labour present the per-component dict should expose
    # recovery_time, rate, coefficient and share for each.
    config = WellBeingConfig(
        owner_housing=CapitalStock(k=100000.0, v=0.2, recovery_rate=0.5, pi=0.1),
        rental_housing=CapitalStock(k=30000.0, v=0.1, recovery_time=4.0, pi=0.1),
        labour_assets={
            "Private": CapitalStock(k=50000.0, v=0.08, recovery_time=4.0, pi=0.1),
        },
        income=IncomeConfig(i_0=20000.0, i_avg=20000.0),
        simulation=SimulationConfig(t_max=10, dt=0.1, recovery_per=95.0),
    )
    hh = CommunityUnit(config)
    hh.get_losses("trapezoid")  # populates hh.recovery_time_per_component
    pc = hh.recovery_time_per_component
    assert pc is not None
    assert set(pc.keys()) == {"owner", "rental", "labour/Private"}
    share_sum = sum(d["share"] for d in pc.values())
    assert abs(share_sum - 1.0) < 1e-9, f"component shares sum to {share_sum}, not 1"
    for name, d in pc.items():
        assert d["recovery_time"] > 0
        assert d["rate"] > 0
        assert d["coefficient"] > 0
        assert 0 <= d["share"] <= 1


def test_capital_stock_per_stock_pi():
    # Each stock contributes to recovery_time_per_component using its own
    # CapitalStock.pi (no fallback / no shared default beyond the field's own
    # default). Owner and labour can carry different pi values independently.
    pi_h, pi_f = 0.08, 0.15
    config = WellBeingConfig(
        owner_housing=CapitalStock(k=100000.0, v=0.2, recovery_rate=0.5, pi=pi_h),
        labour_assets={
            "Private": CapitalStock(k=50000.0, v=0.1, recovery_time=4.0, pi=pi_f),
        },
        income=IncomeConfig(i_0=20000.0, i_avg=20000.0),
        simulation=SimulationConfig(t_max=10, dt=0.1, recovery_per=95.0),
    )
    hh = CommunityUnit(config)
    hh.get_losses("trapezoid")
    pc = hh.recovery_time_per_component
    # owner: pi_h * 0.2 * 100000 = 1600
    assert abs(pc["owner"]["coefficient"] - (pi_h * 0.2 * 100000)) < 1e-6
    # labour: pi_f * 0.1 * 50000 = 750
    assert abs(pc["labour/Private"]["coefficient"] - (pi_f * 0.1 * 50000)) < 1e-6


def test_owner_pi_propagates_to_owner_losses():
    # CapitalStock.pi on owner_housing must flow through calc_loss(INCOME),
    # get_losses()'s UtilityLoss, and achieved_recovery_percent. Previously
    # owner income loss read a different pi than the aggregator, giving an
    # inconsistent picture of owner costs.
    pi_a, pi_b = 0.10, 0.30

    def _cfg(owner_pi):
        return WellBeingConfig(
            owner_housing=CapitalStock(
                k=100000.0, v=0.5, recovery_rate=0.5, pi=owner_pi
            ),
            income=IncomeConfig(i_0=50000.0, i_avg=50000.0),
            simulation=SimulationConfig(t_max=10, dt=0.1, recovery_per=95.0),
        )

    hh_a = CommunityUnit(_cfg(pi_a))
    hh_b = CommunityUnit(_cfg(pi_b))

    # Income loss scales linearly with pi: pi * v * k * integral_exp
    ratio = hh_b.calc_loss(LossType.OWNER_HOUSING_LOSS) / hh_a.calc_loss(
        LossType.OWNER_HOUSING_LOSS
    )
    assert abs(ratio - (pi_b / pi_a)) < 1e-6, (
        f"Income loss ratio {ratio} should equal pi_b/pi_a="
        f"{pi_b / pi_a}; per-stock pi not applied to owner income path"
    )


def test_calc_loss_raises_when_owner_rate_missing():
    # B2: requesting a loss that needs owner's recovery rate before opt_lambda
    # was called must raise a clear ValueError, not a cryptic TypeError from
    # np.exp(-None * t).
    config = WellBeingConfig(
        # Neither rate nor time set - owner is in "waiting for optimization" state
        owner_housing=CapitalStock(k=100000.0, v=0.2),
        income=IncomeConfig(i_0=20000.0, i_avg=20000.0),
        simulation=SimulationConfig(t_max=10, dt=0.1, recovery_per=95.0),
    )
    hh = CommunityUnit(config)
    import pytest

    with pytest.raises(ValueError, match="opt_lambda"):
        hh.calc_loss(LossType.RECOVERY_COST)
    with pytest.raises(ValueError, match="opt_lambda"):
        hh.get_losses("trapezoid")


def test_total_losses_uses_renamed_equity_key():
    # B3: the module now writes "Equity Weighted Asset Loss" (not the misleading
    # "Equity Weighted Loss"). Guard against silent regression to the old key.
    config = _make_config(v=0.1, k=50000, rec_rate=0.7, i0=15000, iavg=14000)
    hh = CommunityUnit(config)
    losses = hh.get_losses()
    assert "Equity Weighted Asset Loss" in losses
    assert "Equity Weighted Loss" not in losses


def test_unit_recovery_time_is_household_only():
    # τ = ln(1/(1-f_r)) · λ_h⁻¹ uses the household rate alone.
    # With rental + labour present, composite_recovery_time() must differ from
    # the attribute set by get_losses (household-only).
    import math

    from fiat_toolbox.well_being.methods import recovery_time as _rt

    config = _make_config(v=0.2, k=100000, rec_rate=0.5, i0=20000, iavg=20000)
    config.rental_housing = CapitalStock(k=30000, v=0.15, recovery_rate=0.25)
    config.labour_assets = {"firm": CapitalStock(k=50000, v=0.1, recovery_time=4.0)}
    hh = CommunityUnit(config)
    hh.get_losses()

    # Household-only τ: recovery_time(λ_owner, rebuilt_per).
    expected = _rt(0.5, rebuilt_per=hh.config.simulation.recovery_per)
    assert math.isclose(hh.recovery_time, expected, rel_tol=1e-9), (
        f"recovery_time {hh.recovery_time} != expected τ {expected}"
    )

    # Composite should differ when rental/labour have slower rates.
    composite = hh.composite_recovery_time()
    assert composite is not None and composite > hh.recovery_time, (
        "composite τ should exceed household-only τ when rental/labour are slower"
    )


def test_wellbeing_loss_liquidity_taylor_term_present_and_positive():
    # ΔW = ∫(u_0 - u(c(t))) e^(-ρt) dt + du/dc|_c0 · ΔS.
    # The Taylor term must appear in total_losses and be >0 whenever S>0 and
    # there are losses to absorb.
    config = _make_config(v=0.2, k=100000, rec_rate=0.5, i0=20000, iavg=20000)
    hh = CommunityUnit(config)
    losses = hh.get_losses()
    assert "Wellbeing Loss (Integral)" in losses
    assert "Wellbeing Loss (Liquidity Term)" in losses
    assert losses["Wellbeing Loss (Liquidity Term)"] > 0, (
        "liquidity-depletion term should be >0 when S>0 and losses occur"
    )
    # Wellbeing Loss equals the sum of the two parts.
    assert (
        abs(
            losses["Wellbeing Loss"]
            - losses["Wellbeing Loss (Integral)"]
            - losses["Wellbeing Loss (Liquidity Term)"]
        )
        < 1e-6
    )


def test_wellbeing_loss_no_taylor_when_liquidity_zero():
    # With S=0 the Taylor correction is exactly zero, reproducing pre-fix ΔW.
    config = _make_config(
        v=0.2,
        k=100000,
        rec_rate=0.5,
        i0=20000,
        iavg=20000,
        savings=0,
        insurance=0,
        support=0,
    )
    hh = CommunityUnit(config)
    losses = hh.get_losses()
    assert losses["Wellbeing Loss (Liquidity Term)"] == 0.0
    assert losses["Wellbeing Loss"] == losses["Wellbeing Loss (Integral)"]


def test_liquidity_depleted_equals_min_S_and_lifetime():
    # _liquidity_depleted must return min(S, lifetime_loss_integral).
    config = _make_config(
        v=0.2,
        k=100000,
        rec_rate=0.5,
        i0=20000,
        iavg=20000,
        savings=10**9,
        insurance=0,
        support=0,  # huge S -> capped by lifetime
    )
    hh = CommunityUnit(config)
    depleted_huge_S = hh._liquidity_depleted()
    # Lifetime loss: (pi + lambda) * v * k / lambda
    lifetime = (0.1 + 0.5) * 0.2 * 100000 / 0.5
    assert abs(depleted_huge_S - lifetime) < 1e-6

    # Small S case: depleted == S
    config2 = _make_config(
        v=0.2,
        k=100000,
        rec_rate=0.5,
        i0=20000,
        iavg=20000,
        savings=1000,
        insurance=0,
        support=0,
    )
    hh2 = CommunityUnit(config2)
    assert abs(hh2._liquidity_depleted() - 1000.0) < 1e-6


def test_resilience_metric_equals_asset_over_wellbeing():
    # Socio-economic resilience: R = Δk_h / ΔC_eq.
    config = _make_config(v=0.1, k=50000, rec_rate=0.7, i0=15000, iavg=14000)
    hh = CommunityUnit(config)
    losses = hh.get_losses()
    expected = losses["Asset Loss"] / losses["Wellbeing Loss"]
    assert abs(losses["Socio-economic Resilience"] - expected) < 1e-9


def test_opt_lambda_rho_defaults_to_config_rho():
    # CommunityUnit.opt_lambda(rho=None) must use config.simulation.rho;
    # rho=0 must reproduce the pre-fix undiscounted objective. Pick a
    # moderate-damage / no-liquidity / c_min=0 config so losses actually
    # accrue over [0, t_max] (so rho has something to discount) and the
    # default lambda search bounds are feasible.
    cfg_disc = _make_config(
        v=0.2,
        k=100000,
        rec_rate=None,
        i0=20000,
        iavg=20000,
        rho=0.06,
        savings=0,
        insurance=0,
        support=0,
        cmin=0,
    )
    hh_disc = CommunityUnit(cfg_disc)
    res_default = hh_disc.opt_lambda(no_steps=50)
    assert res_default["success"]

    cfg_zero = _make_config(
        v=0.2,
        k=100000,
        rec_rate=None,
        i0=20000,
        iavg=20000,
        rho=0.06,
        savings=0,
        insurance=0,
        support=0,
        cmin=0,
    )
    hh_zero = CommunityUnit(cfg_zero)
    res_zero = hh_zero.opt_lambda(no_steps=50, rho=0.0)
    assert res_zero["success"]

    # The objective values at the optimum must differ when rho != 0 is
    # actually applied; boundary-landing Nelder-Mead can return the same
    # lambda even as rho shifts the welfare function, so compare the loss
    # value rather than the argmin.
    import math

    assert not math.isclose(
        res_default["loss_opt"], res_zero["loss_opt"], rel_tol=1e-6
    ), "rho=config vs rho=0 produced identical loss_opt — rho not threaded through"


def test_c0_default_is_sum_of_stock_pi_k_plus_i_div():
    # When i_0 is omitted, c0 must equal Σ π·k across configured stocks,
    # plus i_div when present. Covers owner-only, owner+rental+labour, and
    # owner + i_div.
    owner_pi, owner_k = 0.12, 200000.0
    # (a) owner only
    cfg_a = WellBeingConfig(
        owner_housing=CapitalStock(k=owner_k, v=0.1, recovery_rate=0.5, pi=owner_pi),
        income=IncomeConfig(i_avg=20000.0),
        simulation=SimulationConfig(t_max=5, dt=0.1),
    )
    hh_a = CommunityUnit(cfg_a)
    assert hh_a._c0() == owner_pi * owner_k

    # (b) owner + rental + one labour asset
    rental_pi, rental_k = 0.10, 80000.0
    lab_pi, lab_k = 0.08, 50000.0
    cfg_b = WellBeingConfig(
        owner_housing=CapitalStock(k=owner_k, v=0.1, recovery_rate=0.5, pi=owner_pi),
        rental_housing=CapitalStock(k=rental_k, v=0.1, recovery_time=3.0, pi=rental_pi),
        labour_assets={
            "firm": CapitalStock(k=lab_k, v=0.1, recovery_time=4.0, pi=lab_pi)
        },
        income=IncomeConfig(i_avg=20000.0),
        simulation=SimulationConfig(t_max=5, dt=0.1),
    )
    hh_b = CommunityUnit(cfg_b)
    expected_b = owner_pi * owner_k + rental_pi * rental_k + lab_pi * lab_k
    assert abs(hh_b._c0() - expected_b) < 1e-9

    # (c) owner + i_div
    i_div = 7500.0
    cfg_c = WellBeingConfig(
        owner_housing=CapitalStock(k=owner_k, v=0.1, recovery_rate=0.5, pi=owner_pi),
        income=IncomeConfig(i_avg=20000.0, i_div=i_div),
        simulation=SimulationConfig(t_max=5, dt=0.1),
    )
    hh_c = CommunityUnit(cfg_c)
    assert abs(hh_c._c0() - (owner_pi * owner_k + i_div)) < 1e-9


def test_c0_override_warns_on_mismatch():
    # When i_0 is supplied and disagrees with Σ π·k, a UserWarning fires at
    # construction and c0 == i_0 + (i_div or 0) — the override wins.
    import pytest

    pi_h, k_h = 0.10, 100000.0  # Σ π·k = 10000
    cfg = WellBeingConfig(
        owner_housing=CapitalStock(k=k_h, v=0.1, recovery_rate=0.5, pi=pi_h),
        income=IncomeConfig(i_0=30000.0, i_avg=20000.0, i_div=2000.0),
        simulation=SimulationConfig(t_max=5, dt=0.1),
    )
    with pytest.warns(UserWarning, match=r"overrides the stock-derived baseline"):
        hh = CommunityUnit(cfg)
    assert hh._c0() == 30000.0 + 2000.0


def test_c0_override_silent_when_consistent():
    # When i_0 exactly matches Σ π·k (within tolerance), no UserWarning fires.
    import warnings as _w

    pi_h, k_h = 0.10, 100000.0  # Σ π·k = 10000 exactly
    cfg = WellBeingConfig(
        owner_housing=CapitalStock(k=k_h, v=0.1, recovery_rate=0.5, pi=pi_h),
        income=IncomeConfig(i_0=pi_h * k_h, i_avg=20000.0),
        simulation=SimulationConfig(t_max=5, dt=0.1),
    )
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        hh = CommunityUnit(cfg)
    mismatch_warnings = [
        w for w in caught if "overrides the stock-derived baseline" in str(w.message)
    ]
    assert mismatch_warnings == [], (
        f"no mismatch warning expected, got: {[str(w.message) for w in mismatch_warnings]}"
    )
    assert hh._c0() == pi_h * k_h


def _analytical_income_integral(
    baseline: float, v: float, lam: float, t_max: float
) -> float:
    """∫₀^{t_max} baseline · v · exp(-λt) dt  =  baseline · v · (1 − e^{-λT}) / λ."""
    import math

    return baseline * v * (1.0 - math.exp(-lam * t_max)) / lam


def test_income_stream_rental_housing():
    # rental_housing as IncomeStream: _c0 must include the income flow, and
    # RENTAL_INCOME loss must equal the closed-form analytical integral.
    import math

    pi_h, k_h = 0.10, 100000.0  # owner: Σ π·k = 10000
    rental_income = 12000.0
    rental_v = 0.15
    rental_rt = 1.5
    cfg = WellBeingConfig(
        owner_housing=CapitalStock(k=k_h, v=0.1, recovery_rate=0.5, pi=pi_h),
        rental_housing=IncomeStream(
            income=rental_income, v=rental_v, recovery_time=rental_rt
        ),
        income=IncomeConfig(i_avg=20000.0),
        simulation=SimulationConfig(t_max=10, dt=0.1, recovery_per=95.0),
    )
    hh = CommunityUnit(cfg)
    # Baseline includes the rental income directly.
    assert abs(hh._c0() - (pi_h * k_h + rental_income)) < 1e-9

    # Rate derived from recovery_time via ln(1/(1-0.95))/T.
    lam = math.log(1 / (1 - 0.95)) / rental_rt
    total = hh.calc_loss(LossType.RENTAL_HOUSING_LOSS, method="trapezoid")
    expected = _analytical_income_integral(rental_income, rental_v, lam, hh.t[-1])
    assert abs(total - expected) / expected < 5e-3, (
        f"RENTAL_INCOME total {total} disagrees with analytical {expected}"
    )


def test_income_stream_labour_asset():
    # labour_assets with IncomeStream: aggregated LABOUR_INCOME total should
    # match the analytical integral of income·v·exp(-λt).
    income, v, lam = 8000.0, 0.4, 0.5
    cfg = WellBeingConfig(
        owner_housing=CapitalStock(k=100000.0, v=0.1, recovery_rate=0.5, pi=0.1),
        labour_assets={
            "Firms": IncomeStream(income=income, v=v, recovery_rate=lam),
        },
        income=IncomeConfig(i_avg=20000.0),
        simulation=SimulationConfig(t_max=10, dt=0.1, recovery_per=95.0),
    )
    hh = CommunityUnit(cfg)
    total = hh.calc_loss(LossType.LABOUR_INCOME_LOSS, method="trapezoid")
    expected = _analytical_income_integral(income, v, lam, hh.t[-1])
    assert abs(total - expected) / expected < 5e-3, (
        f"LABOUR_INCOME total {total} disagrees with analytical {expected}"
    )


def test_income_stream_mixed_with_capital_stock():
    # Same unit with CapitalStock + IncomeStream across labour_assets:
    # recovery_time_per_component must produce correct coefficients for each.
    cs_pi, cs_k, cs_v = 0.12, 40000.0, 0.2  # π·k·v = 960
    is_income, is_v = 7500.0, 0.3  # income·v = 2250
    cfg = WellBeingConfig(
        owner_housing=CapitalStock(k=100000.0, v=0.1, recovery_rate=0.5, pi=0.1),
        labour_assets={
            "Firms": CapitalStock(k=cs_k, v=cs_v, recovery_time=3.0, pi=cs_pi),
            "Public": IncomeStream(income=is_income, v=is_v, recovery_time=4.0),
        },
        income=IncomeConfig(i_avg=20000.0),
        simulation=SimulationConfig(t_max=10, dt=0.1, recovery_per=95.0),
    )
    hh = CommunityUnit(cfg)
    hh.get_losses("trapezoid")
    pc = hh.recovery_time_per_component
    assert pc is not None
    assert abs(pc["labour/Firms"]["coefficient"] - cs_pi * cs_k * cs_v) < 1e-9
    assert abs(pc["labour/Public"]["coefficient"] - is_income * is_v) < 1e-9


def test_income_stream_negative_income_rejected():
    # IncomeStream.income must be > 0 — pydantic-level constraint fires at
    # IncomeStream(...), before WellBeingConfig or CommunityUnit even see it.
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="greater than 0"):
        IncomeStream(income=-1.0, v=0.1, recovery_time=1.0)


def test_income_stream_requires_recovery_time_or_rate():
    # Neither recovery_time nor recovery_rate on a rental / labour stock must
    # raise at WellBeingConfig construction via the @model_validator on the
    # parent config. Owner_housing is allowed to have both None (opt_lambda
    # path), so this rule only applies to extras.
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="recovery_rate or recovery_time"):
        WellBeingConfig(
            owner_housing=CapitalStock(k=100000.0, v=0.1, recovery_rate=0.5, pi=0.1),
            rental_housing=IncomeStream(income=1000.0, v=0.1),
            income=IncomeConfig(i_avg=20000.0),
            simulation=SimulationConfig(t_max=5, dt=0.1),
        )


def test_labels_plot_only_do_not_change_total_losses_keys():
    # Custom recovery_label / income_label must not leak into total_losses or
    # time_series column names — those stay structural (LossType-driven).
    cfg = WellBeingConfig(
        owner_housing=CapitalStock(
            k=100000.0,
            v=0.1,
            recovery_rate=0.5,
            pi=0.1,
            recovery_label="Rebuild our house",
            income_label="Our housing services",
        ),
        rental_housing=IncomeStream(
            income=5000.0,
            v=0.2,
            recovery_time=3.0,
            income_label="Landlord rebuilding",
        ),
        labour_assets={
            "Public": IncomeStream(
                income=6000.0,
                v=0.3,
                recovery_time=5.0,
                income_label="Public sector wage loss",
            ),
        },
        income=IncomeConfig(i_avg=20000.0),
        simulation=SimulationConfig(t_max=5, dt=0.1),
    )
    hh = CommunityUnit(cfg)
    losses = hh.get_losses("trapezoid")
    # Structural keys remain
    assert LossType.RECOVERY_COST in losses
    assert LossType.OWNER_HOUSING_LOSS in losses
    assert LossType.RENTAL_HOUSING_LOSS in losses
    assert LossType.LABOUR_INCOME_LOSS in losses
    assert "Labour Income Loss (Public)" in hh.time_series.columns
    # Custom labels must NOT appear as dict/column keys
    for custom in (
        "Rebuild our house",
        "Our housing services",
        "Landlord rebuilding",
        "Public sector wage loss",
    ):
        assert custom not in losses.index
        assert custom not in hh.time_series.columns


def test_display_label_resolution():
    # _display_label and _label_for must follow the precedence rules:
    # override (when set) > enum default > enum default + fallback suffix.
    cfg = WellBeingConfig(
        owner_housing=CapitalStock(
            k=100000.0,
            v=0.1,
            recovery_rate=0.5,
            pi=0.1,
            recovery_label="Rebuild our house",
            # income_label intentionally left None → enum default
        ),
        rental_housing=IncomeStream(
            income=5000.0,
            v=0.2,
            recovery_time=3.0,
            income_label="Landlord rebuilding",
        ),
        labour_assets={
            "Public": IncomeStream(
                income=6000.0,
                v=0.3,
                recovery_time=5.0,
                income_label="Public sector",
            ),
            # "Private" has no income_label → falls back to "{default} ({name})"
            "Private": IncomeStream(
                income=4000.0,
                v=0.2,
                recovery_time=4.0,
            ),
        },
        income=IncomeConfig(i_avg=20000.0),
        simulation=SimulationConfig(t_max=5, dt=0.1),
    )
    hh = CommunityUnit(cfg)

    # Owner RECOVERY: override wins.
    assert hh._label_for(LossType.RECOVERY_COST) == "Rebuild our house"
    # Owner INCOME: no override → enum default.
    assert (
        hh._label_for(LossType.OWNER_HOUSING_LOSS) == LossType.OWNER_HOUSING_LOSS.value
    )
    # Rental: override on rental_housing.income_label.
    assert hh._label_for(LossType.RENTAL_HOUSING_LOSS) == "Landlord rebuilding"
    # LABOUR_INCOME aggregate: no per-stock override applies.
    assert (
        hh._label_for(LossType.LABOUR_INCOME_LOSS) == LossType.LABOUR_INCOME_LOSS.value
    )
    # Per-asset column key, with override:
    assert hh._label_for("Labour Income Loss (Public)") == "Public sector"
    # Per-asset column key, without override → default + suffix:
    assert (
        hh._label_for("Labour Income Loss (Private)")
        == f"{LossType.LABOUR_INCOME_LOSS.value} (Private)"
    )
    # Consumption / Utility stay as LossType defaults.
    assert hh._label_for(LossType.CONSUMPTION_LOSS) == LossType.CONSUMPTION_LOSS.value
    assert hh._label_for(LossType.UTILITY_LOSS) == LossType.UTILITY_LOSS.value


def test_extra_fields_forbidden_on_all_config_models():
    # extra='forbid' on every BaseModel must reject unknown fields with a
    # ValidationError — catches typos and silent field drift.
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match=r"[Ee]xtra"):
        CapitalStock(k=100.0, v=0.1, recovery_rate=0.5, bogus_field=True)
    with pytest.raises(ValidationError, match=r"[Ee]xtra"):
        IncomeStream(income=100.0, v=0.1, recovery_time=1.0, bogus=1)
    with pytest.raises(ValidationError, match=r"[Ee]xtra"):
        Liquidity(savings=100, bogus=True)
    with pytest.raises(ValidationError, match=r"[Ee]xtra"):
        IncomeConfig(i_avg=1000.0, typo_field=1)
    with pytest.raises(ValidationError, match=r"[Ee]xtra"):
        SimulationConfig(unknown=42)
    with pytest.raises(ValidationError, match=r"[Ee]xtra"):
        WellBeingConfig(
            owner_housing=CapitalStock(k=1.0, v=0.1, recovery_rate=0.5),
            income=IncomeConfig(i_avg=1.0),
            mystery_field=123,
        )


def test_numeric_field_bounds_raise_on_invalid():
    # Layer 2 bounds: negative k, v outside [0,1], recovery_per >= 100, etc.
    import pytest
    from pydantic import ValidationError

    # CapitalStock: negative k rejected.
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        CapitalStock(k=-1.0, v=0.1, recovery_rate=0.5)
    # v > 1 rejected.
    with pytest.raises(ValidationError, match="less than or equal to 1"):
        CapitalStock(k=100.0, v=1.5, recovery_rate=0.5)
    # pi must be > 0.
    with pytest.raises(ValidationError, match="greater than 0"):
        CapitalStock(k=100.0, v=0.1, recovery_rate=0.5, pi=0)

    # IncomeStream: income must be > 0.
    with pytest.raises(ValidationError, match="greater than 0"):
        IncomeStream(income=0, v=0.1, recovery_time=1.0)

    # SimulationConfig: recovery_per >= 100 rejected.
    with pytest.raises(ValidationError, match="less than 100"):
        SimulationConfig(recovery_per=100.0)
    # c_min negative rejected.
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        SimulationConfig(c_min=-1.0)
    # eta must be > 0.
    with pytest.raises(ValidationError, match="greater than 0"):
        SimulationConfig(eta=0)

    # Liquidity: negative savings rejected.
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        Liquidity(savings=-1)

    # IncomeConfig: i_avg must be > 0.
    with pytest.raises(ValidationError, match="greater than 0"):
        IncomeConfig(i_avg=0)


def test_recovery_xor_rule_fires_at_wellbeing_config():
    # Layer 3: both recovery_rate AND recovery_time set on the same stock at
    # construction → WellBeingConfig's @model_validator raises. Covers owner,
    # rental, and labour paths. Owner allowed to have both None (opt_lambda).
    import pytest
    from pydantic import ValidationError

    # Owner: both set at construction → raises on the XOR rule.
    with pytest.raises(ValidationError, match="Provide only one"):
        WellBeingConfig(
            owner_housing=CapitalStock(
                k=100.0, v=0.1, recovery_rate=0.5, recovery_time=2.0
            ),
            income=IncomeConfig(i_avg=1000.0),
            simulation=SimulationConfig(recovery_per=95.0),
        )

    # Owner: both None is allowed (opt_lambda will fill).
    WellBeingConfig(
        owner_housing=CapitalStock(k=100.0, v=0.1),
        income=IncomeConfig(i_avg=1000.0),
        simulation=SimulationConfig(recovery_per=95.0),
    )

    # Rental: both None raises (no opt_lambda relaxation).
    with pytest.raises(ValidationError, match="recovery_rate or recovery_time"):
        WellBeingConfig(
            owner_housing=CapitalStock(k=100.0, v=0.1, recovery_rate=0.5),
            rental_housing=CapitalStock(k=50.0, v=0.1),
            income=IncomeConfig(i_avg=1000.0),
            simulation=SimulationConfig(recovery_per=95.0),
        )


def test_recovery_xor_fill_is_idempotent():
    # Constructing already fills the missing field; re-running
    # normalize_recovery_params (as CommunityUnit does) must not raise on the
    # now-both-set state because the two values are consistent.
    cfg = WellBeingConfig(
        owner_housing=CapitalStock(k=100.0, v=0.1, recovery_rate=0.5, pi=0.1),
        income=IncomeConfig(i_avg=1000.0),
        simulation=SimulationConfig(recovery_per=95.0),
    )
    # After construction both fields are set (recovery_time was filled).
    assert cfg.owner_housing.recovery_rate is not None
    assert cfg.owner_housing.recovery_time is not None

    # Re-running the normalizer must be a no-op — this is what
    # CommunityUnit.__init__ does to cover post-construction nested mutations.
    cfg.normalize_recovery_params()

    # And CommunityUnit construction must succeed.
    hh = CommunityUnit(cfg)
    assert hh.config.owner_housing.recovery_rate == 0.5
