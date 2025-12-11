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


def test_reconstruction_cost_t():
    t = np.linspace(0, 1, 5)
    rec_rate, v, k_str = 0.5, 0.2, 100000
    cost = methods.reconstruction_cost_t(t, rec_rate, v, k_str)
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
    cl = methods.consumption_loss_t(
        t, rec_rate, v, k_str, pi, savings=1000, insurance=500, support=200
    )
    assert cl.shape == t.shape
    assert np.all(cl >= 0)


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
        savings=1000,
        insurance=500,
        support=200,
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
        savings=1000,
        insurance=500,
        support=200,
    )
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
        savings=1000,
        insurance=500,
        support=200,
    )
    assert "l_opt" in result
    assert "loss_opt" in result
