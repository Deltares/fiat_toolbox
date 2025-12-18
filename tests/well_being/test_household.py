from fiat_toolbox.well_being.loss import (
    CommunityUnit,
    LossType,
    WellBeingConfig,
    CapitalStock,
    Liquidity,
    IncomeConfig,
    SimulationConfig,
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
        owner_housing=CapitalStock(k=k, v=v, recovery_rate=rec_rate),
        income=IncomeConfig(i_0=i0, i_avg=iavg, pi=pi),
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
    assert "Equity Weighted Loss" in losses
    # Only configured loss types should be present
    assert LossType.RECOVERY in losses
    assert LossType.INCOME in losses
    assert LossType.CONSUMPTION in losses
    assert LossType.UTILITY in losses
    assert LossType.RENTAL_INCOME not in losses
    assert LossType.LABOUR_INCOME not in losses


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
