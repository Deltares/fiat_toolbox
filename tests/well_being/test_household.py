from fiat_toolbox.well_being.household import Household, LossType


def test_household_initialization():
    hh = Household(
        v=0.2,
        k_str=100000,
        c0=20000,
        c_avg=18000,
        rec_rate=0.5,
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
    )
    assert hh.v == 0.2
    assert hh.k_str == 100000
    assert hh.c0 == 20000
    assert hh.currency == "€"
    assert hh.savings == 5000
    assert hh.insurance == 2000
    assert hh.support == 1000


def test_calc_loss_reconstruction():
    hh = Household(0.1, 50000, 15000, 14000, rec_rate=0.7)
    loss = hh.calc_loss(LossType.RECONSTRUCTION)
    assert loss > 0


def test_calc_loss_income():
    hh = Household(0.1, 50000, 15000, 14000, rec_rate=0.7)
    loss = hh.calc_loss(LossType.INCOME)
    assert loss > 0


def test_calc_loss_utility():
    hh = Household(0.1, 50000, 15000, 14000, rec_rate=0.7)
    loss = hh.calc_loss(LossType.UTILITY)

    assert loss > 0


def test_get_losses():
    hh = Household(0.1, 50000, 15000, 14000, rec_rate=0.7)
    losses = hh.get_losses()
    assert "Wellbeing Loss" in losses
    assert "Asset Loss" in losses
    assert "Equity Weighted Loss" in losses
    for lt in LossType:
        assert lt in losses


def test_opt_lambda_runs():
    hh = Household(0.1, 50000, 15000, 14000)
    hh.opt_lambda(no_steps=10)
    # No assertion, just check it runs without error


def test_repr():
    hh = Household(0.1, 50000, 15000, 14000)
    s = repr(hh)
    assert "Household(" in s
    assert "v = 0.1" in s


def test_plot_loss_all_types():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hh = Household(0.1, 50000, 15000, 14000, rec_rate=0.7)
    for lt in LossType:
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

    hh = Household(0.1, 50000, 15000, 14000, rec_rate=0.7)
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
    hh = Household(0.1, 50000, 15000, 14000, rec_rate=0.7)
    hh.opt_lambda(no_steps=10)
    # Test with default x_type ("rate")
    fig = hh.plot_opt_lambda()
    assert fig is None or hasattr(fig, "savefig")
    # Test with x_type="time"
    fig2 = hh.plot_opt_lambda(x_type="time")
    assert fig2 is None or hasattr(fig2, "savefig")
