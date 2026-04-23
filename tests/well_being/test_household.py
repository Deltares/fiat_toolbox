from fiat_toolbox.well_being.loss import (
    CapitalStock,
    CommunityUnit,
    IncomeConfig,
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
    config = _make_config()
    hh = CommunityUnit(config)
    assert hh.config.owner_housing.v == 0.2
    assert hh.config.owner_housing.k == 100000
    assert hh._c0() == 20000
    assert hh.config.simulation.currency == "€"
    assert hh._liquidity() == 5000 + 2000 + 1000


def test_calc_loss_reconstruction():
    config = _make_config(v=0.1, k=50000, rec_rate=0.7, i0=15000, iavg=14000)
    hh = CommunityUnit(config)
    loss = hh.calc_loss(LossType.RECOVERY)
    assert loss > 0


def test_calc_loss_income():
    config = _make_config(v=0.1, k=50000, rec_rate=0.7, i0=15000, iavg=14000)
    hh = CommunityUnit(config)
    loss = hh.calc_loss(LossType.INCOME)
    assert loss > 0


def test_calc_loss_utility():
    config = _make_config(v=0.1, k=50000, rec_rate=0.7, i0=15000, iavg=14000)
    hh = CommunityUnit(config)
    loss = hh.calc_loss(LossType.UTILITY)
    assert loss >= 0


def test_get_losses():
    config = _make_config(v=0.1, k=50000, rec_rate=0.7, i0=15000, iavg=14000)
    hh = CommunityUnit(config)
    losses = hh.get_losses()
    assert "Wellbeing Loss" in losses
    assert "Asset Loss" in losses
    assert "Equity Weighted Asset Loss" in losses
    # Only configured loss types should be present
    assert LossType.RECOVERY in losses
    assert LossType.INCOME in losses
    assert LossType.CONSUMPTION in losses
    assert LossType.UTILITY in losses
    assert LossType.RENTAL_INCOME not in losses
    assert LossType.LABOUR_INCOME not in losses


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
    assert LossType.RENTAL_INCOME in losses
    assert LossType.LABOUR_INCOME in losses
    # Per-asset labour components should also be stored
    assert any(
        isinstance(c, str) and c.startswith(f"{LossType.LABOUR_INCOME.value} (")
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
    # optimizer must detect this and report flat_objective=True rather than
    # returning arbitrary Nelder-Mead noise.
    config = WellBeingConfig(
        owner_housing=CapitalStock(k=100000.0, v=0.0, pi=0.1),
        income=IncomeConfig(i_0=20000.0, i_avg=20000.0),
        simulation=SimulationConfig(t_max=10, dt=0.1),
    )
    hh = CommunityUnit(config)
    opt = hh.opt_lambda(no_steps=10)
    assert opt["success"] is True
    assert opt["flat_objective"] is True, (
        "zero-damage case should be flagged as flat_objective=True"
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
    ratio = hh_b.calc_loss(LossType.INCOME) / hh_a.calc_loss(LossType.INCOME)
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
        hh.calc_loss(LossType.RECOVERY)
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
