import warnings
from typing import Optional, Union

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.optimize import minimize


def utility(
    consumption: Union[float, np.ndarray], eta: float, normalize: bool = False
) -> Union[float, np.ndarray]:
    """
    Calculate the utility of given consumption(s) using a CRRA (Constant Relative Risk Aversion) utility function.
    Parameters
    ----------
    consumption : Union[float, np.ndarray]
        The consumption value(s) for which to calculate utility. Can be a single float or a numpy array of floats.
    eta : float, optional
        The elasticity of the marginal utility of consumption.
    normalize : bool, optional
        If True, normalize the utility values to the range [0, 1], by default False.
    Returns
    -------
    Union[float, np.ndarray]
        The calculated utility value(s). Returns a float if a single consumption value is provided, otherwise returns a numpy array.
    """
    consumption = np.array(consumption)  # Ensure input is a numpy array

    # Check for zero or negative consumption values and issue a warning
    if np.any(consumption <= 0):
        # warnings.warn("Consumption contains zero or negative values, resulting in NaN utility.", UserWarning, stacklevel=2)
        consumption = np.where(consumption <= 0, np.nan, consumption)

    if eta <= 0:
        raise ValueError("Elasticity of marginal utility of consumption must >= 0.")

    if eta == 1:
        # If eta is 1, use the natural log function instead
        warnings.warn(
            "Utility is calculated with the natural logarithm of the consumption when eta == 1.",
            UserWarning,
            stacklevel=2,
        )
        u = np.log(consumption)
    else:
        u = consumption ** (1 - eta) / (1 - eta)  # Calculate utility

    # Normalize utility values if requested
    if normalize and u.size == 1:
        raise ValueError("Utility cannot be normalized for a single consumption value.")
    elif normalize and u.size > 1:
        min_utility = np.nanmin(u)
        max_utility = np.nanmax(u)
        u = (u - min_utility) / (max_utility - min_utility)

    return u if u.size > 1 else u.item()


def recovery_time(
    rate: Union[float, np.ndarray], rebuilt_per: float = 95
) -> Union[float, np.ndarray]:
    """
    Calculate the recovery time based on the given rate and percentage rebuilt.

    Parameters
    ----------
    rate : Union[float, np.ndarray]
        The rate of recovery (per unit time). Can be a single float or a numpy array of floats.
    rebuilt_per : float, optional
        The percentage of the structure that needs to be rebuilt. Default is 95. Must be [0, 100).

    Returns
    -------
    Union[float, np.ndarray]
        The calculated recovery time (in the same time units as the rate). Returns a float if a single rate value is provided, otherwise returns a numpy array.

    Raises
    ------
    ValueError
        If any rate value is non-positive.
    ValueError
        If rebuilt_per is not between 0 and 100.
    """
    rate = np.array(rate)  # Ensure input is a numpy array

    if np.any(rate <= 0):
        raise ValueError("Rate must be positive.")
    if not (0 <= rebuilt_per < 100):
        raise ValueError(
            "rebuilt_per must be a percentage between 0 and 100 (exclusive)."
        )

    T = np.log(1 / (1 - rebuilt_per / 100)) / rate

    return T if T.size > 1 else T.item()


def recovery_rate(
    time: Union[float, np.ndarray], rebuilt_per: float = 95
) -> Union[float, np.ndarray]:
    """
    Calculate the recovery rate based on the given recovery time and percentage rebuilt.

    Parameters
    ----------
    time : Union[float, np.ndarray]
        The recovery time (in the same time units as the rate). Can be a single float or a numpy array of floats.
    rebuilt_per : float, optional
        The percentage of the structure that needs to be rebuilt. Default is 95. Must be [0, 100).

    Returns
    -------
    Union[float, np.ndarray]
        The calculated recovery rate (per unit time). Returns a float if a single time value is provided, otherwise returns a numpy array.

    Raises
    ------
    ValueError
        If any time value is non-positive.
    ValueError
        If rebuilt_per is not between 0 and 100.
    """
    time = np.array(time)  # Ensure input is a numpy array

    if np.any(time <= 0):
        raise ValueError("Time must be positive.")
    if not (0 <= rebuilt_per < 100):
        raise ValueError(
            "rebuilt_per must be a percentage between 0 and 100 (exclusive)."
        )

    rate = np.log(1 / (1 - rebuilt_per / 100)) / time

    return rate if rate.size > 1 else rate.item()


def reconstruction_cost_t(
    t: Union[float, np.ndarray], l: Union[float, np.ndarray], v: float, k_str: float
) -> np.ndarray:
    """
    Calculate the reconstruction cost over time. This represents a cost rate and not the total cost.

    Parameters
    ----------
    t : Union[float, np.ndarray]
        The time value(s) for which to calculate the reconstruction cost. Can be a single float or a numpy array of floats.
    l : Union[float, np.ndarray]
        The rate of recovery value(s). Can be a single float or a numpy array of floats.
    v : float
        The loss ratio, which is reconstruction cost divided by the total building structure value.
    k_str : float
        The total building structure value.

    Returns
    -------
    np.ndarray
        The calculated reconstruction cost(s) as an nxm matrix where n is the length of t and m is the length of l.
    """
    # Calculate the reconstruction cost
    cost = l * v * k_str * np.exp(-l * t)

    return cost


def income_loss_t(
    t: Union[float, np.ndarray],
    l: Union[float, np.ndarray],
    v: float,
    k_str: float,
    pi: float,
) -> np.ndarray:
    """
    Calculate the income loss over time. This represents a loss rate and not the total loss.

    Parameters
    ----------
    t : Union[float, np.ndarray]
        The time value(s) for which to calculate the reconstruction cost. Can be a single float or a numpy array of floats.
    l : Union[float, np.ndarray]
        The rate of recovery value(s). Can be a single float or a numpy array of floats.
    v : float
        The loss ratio, which is reconstruction cost divided by the total building structure value.
    k_str : float
        The total building structure value.
    pi : float
        Average productivity of capital. Can be derived using Penn World Tables.

    Returns
    -------
    np.ndarray
        The calculated reconstruction cost(s) as an nxm matrix where n is the length of t and m is the length of l.
    """
    # Calculate the income loss
    loss = pi * v * k_str * np.exp(-l * t)
    return loss


def consumption_loss_t(
    t: Union[float, np.ndarray],
    l: Union[float, np.ndarray],
    v: float,
    k_str: float,
    pi: float,
) -> np.ndarray:
    """
    Calculate the consumption loss over time as the sum of income loss and reconstruction cost. This represents a loss rate and not the total loss.

    Parameters
    ----------
    t : Union[float, np.ndarray]
        The time value(s) for which to calculate the consumption loss. Can be a single float or a numpy array of floats.
    l : Union[float, np.ndarray]
        The rate of recovery value(s). Can be a single float or a numpy array of floats.
    v : float
        The loss ratio, which is reconstruction cost divided by the total building structure value.
    k_str : float
        The total building structure value.
    pi : float
        Average productivity of capital. Can be derived using Penn World Tables.

    Returns
    -------
    np.ndarray
        The calculated consumption loss(es) as an nxm matrix where n is the length of t and m is the length of l.
    """
    cl_t = income_loss_t(t=t, l=l, v=v, k_str=k_str, pi=pi) + reconstruction_cost_t(
        t=t, l=l, v=v, k_str=k_str
    )
    return cl_t


def consumption_t(
    t: Union[float, np.ndarray],
    l: Union[float, np.ndarray],
    v: float,
    k_str: float,
    pi: float,
    c0: float,
) -> np.ndarray:
    """
    Calculate the consumption over time. This represents a consumption rate and not the total consumption.

    Parameters
    ----------
    t : Union[float, np.ndarray]
        The time value(s) for which to calculate the consumption. Can be a single float or a numpy array of floats.
    l : Union[float, np.ndarray]
        The rate of recovery value(s). Can be a single float or a numpy array of floats.
    v : float
        The loss ratio, which is reconstruction cost divided by the total building structure value.
    k_str : float
        The total building structure value.
    pi : float
        Average productivity of capital. Can be derived using Penn World Tables.
    c0 : float
        Initial consumption level.

    Returns
    -------
    np.ndarray
        The calculated consumption(es) as an nxm matrix where n is the length of t and m is the length of l.
    """
    cl_t = consumption_loss_t(t=t, l=l, v=v, k_str=k_str, pi=pi)
    ct = c0 - cl_t
    return ct


def utility_loss_t(
    t: Union[float, np.ndarray],
    l: Union[float, np.ndarray],
    v: float,
    k_str: float,
    pi: float,
    c0: float,
    eta: float,
) -> np.ndarray:
    """
    Calculate the utility loss over time. This represents a loss rate and not the total loss.

    Parameters
    ----------
    t : Union[float, np.ndarray]
        The time value(s) for which to calculate the utility loss. Can be a single float or a numpy array of floats.
    l : Union[float, np.ndarray]
        The rate of recovery value(s). Can be a single float or a numpy array of floats.
    v : float
        The loss ratio, which is reconstruction cost divided by the total building structure value.
    k_str : float
        The total building structure value.
    pi : float
        Average productivity of capital. Can be derived using Penn World Tables.
    c0 : float
        Initial consumption level.
    eta : float
        The elasticity of marginal utility of consumption.

    Returns
    -------
    np.ndarray
        The calculated utility loss(es) as an nxm matrix where n is the length of t and m is the length of l.
    """
    c_t = consumption_t(t=t, l=l, v=v, k_str=k_str, pi=pi, c0=c0)
    ul_t = utility(consumption=c0, eta=eta) - utility(consumption=c_t, eta=eta)
    return ul_t


def wellbeing_loss(du: Union[float, np.ndarray], c_avg: float, eta: float) -> float:
    """
    Calculate the wellbeing loss as the equivalent consumption change.
    The equivalent consumption change represents the amount by which a household earning an average income
    would have to decrease its consumption to experience the same well-being decrease as the considered household.

    Parameters
    ----------
    dw : Union[float, np.ndarray]
        The utility loss.
    c_avg : float
        The average consumption level.
    eta : float
        The elasticity of marginal utility of consumption.

    Returns
    -------
    float
        The equivalent consumption loss.
    """
    dc_eq = du / (c_avg ** (-eta))
    return dc_eq


def equity_weight(c: float, c_avg: float, eta: float) -> float:
    """
    Calculate the equity weight for a given consumption level.

    Parameters
    ----------
    c : float
        The consumption level.
    c_avg : float
        The average consumption level.
    eta : float
        The elasticity of marginal utility of consumption.

    Returns
    -------
    float
        The equity weight.
    """
    return (c / c_avg) ** (-eta)


def opt_lambda(
    v: float,
    k_str: float,
    c0: float,
    pi: float,
    eta: float,
    l_min: float = 0.3,
    l_max: float = 10,
    t_max: Optional[float] = None,
    times: Optional[np.ndarray] = None,
    method: str = "quad",
) -> float:
    """
    Optimize the recovery rate (lambda) to minimize utility loss.

    Parameters
    ----------
    v : float
        The loss ratio, which is reconstruction cost divided by the total building structure value.
    k_str : float
        The total building structure value.
    c0 : float
        Initial consumption level.
    pi : float
        Average productivity of capital. Can be derived using Penn World Tables.
    eta : float
        The elasticity of marginal utility of consumption.
    l_min : float, optional
        The minimum recovery rate to consider during optimization, by default 0.3.
    l_max : float, optional
        The maximum recovery rate to consider during optimization, by default 10.
    t_max : float, optional
        Maximum recovery time value. Required if `method` is "quad".
    times : np.ndarray, optional
        Array of time points. Required if `method` is "trapezoid".
    method : str, optional
        The method to use for integration. Can be either "quad" (default) or "trapezoid".
        "trapezoid" uses the numpy.trapz function, while "quad" uses scipy.integrate.quad.

    Returns
    -------
    float
        The optimized recovery rate (lambda).

    Raises
    ------
    ValueError
        If `t_max` is not provided when using the "quad" method.
    ValueError
        If `times` is not provided when using the "trapezoid" method.
    """
    if method == "quad":
        if t_max is None:
            raise ValueError("t_max must be provided when using the 'quad' method.")
        times = np.array([0, t_max])
    elif method == "trapezoid" and times is None:
        raise ValueError("times must be provided when using the 'trapezoid' method.")

    def objective(l: float) -> float:
        ut_t = UtilityLoss(times, l, v, k_str, pi, c0, eta)
        loss = ut_t.total(rho=0, method=method)
        return loss

    fun = lambda l: objective(l)
    res = minimize(fun, l_min, bounds=[(l_min, l_max)], method="Nelder-Mead")
    return res.x[0]


class Loss:
    """
    A base class for calculating losses over time based on recovery rates.

    Parameters
    ----------
    t : Union[float, np.ndarray], optional
        Time points provided as a single float or a numpy array. If `t` is None, `t_max`
        must be provided.
    l : Union[float, np.ndarray], optional
        Levels provided as a single float or a numpy array. Defaults to None.
    t_max : float, optional
        Maximum time value used to generate time points if `t` is not provided. Defaults to None.

    Attributes
    ----------
    t : np.ndarray
        Array of time points. If multiple levels are provided in `l`, this will be reshaped to align with `l`.
    l : np.ndarray
        Array of levels (e.g., recovery rates).
    _fun : callable
        A function that calculates the loss for given time points and levels.

    Raises
    ------
    ValueError
        If neither `t` nor `t_max` is provided.
    ValueError
        If both `t` and `t_max` are provided.

    Notes
    -----
    - If `t_max` is provided, `t` is generated as 100 evenly spaced points between 0 and `t_max`.
    - Inputs `t` and `l` are converted to at least 1D numpy arrays.
    - If `l` contains more than one element, `t` is reshaped to align with `l` for meshgrid-like behavior.
    """

    _fun: callable

    def __init__(
        self,
        t: Optional[Union[float, np.ndarray]] = None,
        l: Optional[Union[float, np.ndarray]] = None,
        t_max: Optional[float] = None,
    ):
        if t is None and t_max is None:
            raise ValueError("Either `t` or `t_max` must be provided.")
        elif t is not None and t_max is not None:
            raise ValueError("Only one of `t` or `t_max` should be provided.")

        # Generate time points if only t_max is provided
        if t is None:
            t = np.linspace(0, t_max, 100)  # Default to 100 points

        # Ensure inputs are at least 1D arrays
        t, l = np.atleast_1d(t), np.atleast_1d(l)

        # Create a meshgrid to calculate all combinations of t and l
        if l.size > 1:
            t = np.expand_dims(t, axis=0).transpose()

        self.t = t
        self.l = l

    @property
    def losses_t(self) -> np.ndarray:
        """
        Calculate the loss values for all combinations of time points and levels.

        Returns
        -------
        np.ndarray
            Array of loss values for all combinations of `t` and `l`.
        """
        f_t = self._fun(self.t, self.l)
        return f_t

    def total(
        self, rho: float = 0, method: str = "trapezoid"
    ) -> Union[float, np.ndarray]:
        """
        Calculate the total loss by integrating over time.

        Parameters
        ----------
        rho : float, optional
            Discount rate for the integration. Defaults to 0.
        method : str, optional
            Integration method to use. Can be either "trapezoid" (default) or "quad".
            "trapezoid" uses the numpy.trapz function, while "quad" uses scipy.integrate.quad.

        Returns
        -------
        Union[float, np.ndarray]
            The total loss. Returns a float if a single level is provided, otherwise returns a numpy array.

        Raises
        ------
        ValueError
            If `t` has less than 2 points when using the "trapezoid" method.
        ValueError
            If an invalid integration method is provided.
        """
        t_max = self.t[-1]
        if method == "trapezoid":
            if self.t.size == 1:
                raise ValueError(
                    "t must have at least 2 points to calculate the integral."
                )
            f_t = self._fun(self.t, self.l)
            f_t_dis = f_t * np.exp(-rho * self.t)
            integral = np.trapz(f_t_dis, x=self.t, axis=0)
        elif method == "quad":
            warnings.filterwarnings("ignore", category=IntegrationWarning)
            integral = np.array(
                [
                    quad(
                        lambda t, li=li: self._fun(t, li) * np.exp(-rho * t),
                        0,
                        t_max,
                    )[0]
                    for li in self.l
                ]
            )
        else:
            raise ValueError("method must be either 'trapezoid' or 'quad'.")

        # Ensure the result is always an ndarray or a single float
        return integral if integral.size > 1 else integral.item()


class ReconstructionCost(Loss):
    """
    A class to calculate reconstruction cost over time based on recovery rates.
    It includes the total method to calculate the total loss by integrating over time.

    Parameters
    ----------
    t : Union[float, np.ndarray]
        The time value(s) for which to calculate the reconstruction cost. Can be a single float or a numpy array of floats.
    l : Union[float, np.ndarray]
        The rate of recovery value(s). Can be a single float or a numpy array of floats.
    v : float
        The loss ratio, which is reconstruction cost divided by the total building structure value.
    k_str : float
        The total building structure value.

    Attributes
    ----------
    t : np.ndarray
        Array of time points. If multiple recovery rates are provided, this will be expanded to match the shape of `l`.
    l : np.ndarray
        Array of recovery rates.
    losses_t : np.ndarray
        Property that calculates the reconstruction cost values for all combinations of `t` and `l`.
    """

    def __init__(
        self,
        t: Union[float, np.ndarray],
        l: Union[float, np.ndarray],
        v: float,
        k_str: float,
    ):
        super().__init__(t, l)
        self._fun = lambda t, l: reconstruction_cost_t(t, l, v, k_str)


class IncomeLoss(Loss):
    """
    A class to calculate income loss over time based on recovery rates.
    It includes the total method to calculate the total loss by integrating over time.

    Parameters
    ----------
    t : Union[float, np.ndarray]
        Time points or a single time value.
    l : Union[float, np.ndarray]
        Recovery rates or a single recovery rate.
    v : float
        The loss ratio, which is reconstruction cost divided by the total building structure value.
    k_str : float
        The total building structure value.
    pi : float
        Average productivity of capital.

    Attributes
    ----------
    t : np.ndarray
        Array of time points. If multiple recovery rates are provided, this will be expanded to match the shape of `l`.
    l : np.ndarray
        Array of recovery rates.
    losses_t : np.ndarray
        Property that calculates the income loss values for all combinations of `t` and `l`.
    """

    def __init__(
        self,
        t: Union[float, np.ndarray],
        l: Union[float, np.ndarray],
        v: float,
        k_str: float,
        pi: float,
    ):
        super().__init__(t, l)
        self._fun = lambda t, l: income_loss_t(t, l, v, k_str, pi)


class ConsumptionLoss(Loss):
    """
    A class to calculate consumption loss over time based on recovery rates.
    It includes the total method to calculate the total loss by integrating over time.

    Parameters
    ----------
    t : Union[float, np.ndarray]
        Time points or a single time value.
    l : Union[float, np.ndarray]
        Recovery rates or a single recovery rate.
    v : float
        The loss ratio, which is reconstruction cost divided by the total building structure value.
    k_str : float
        The total building structure value.
    pi : float
        Average productivity of capital.

    Attributes
    ----------
    t : np.ndarray
        Array of time points. If multiple recovery rates are provided, this will be expanded to match the shape of `l`.
    l : np.ndarray
        Array of recovery rates.
    losses_t : np.ndarray
        Property that calculates the consumption loss values for all combinations of `t` and `l`.
    """

    def __init__(
        self,
        t: Union[float, np.ndarray],
        l: Union[float, np.ndarray],
        v: float,
        k_str: float,
        pi: float,
    ):
        super().__init__(t, l)
        self._fun = lambda t, l: consumption_loss_t(t, l, v, k_str, pi)


class UtilityLoss(Loss):
    """
    A class to calculate utility loss over time based on recovery rates.
    It includes the total method to calculate the total loss by integrating over time.

    Parameters
    ----------
    t : Union[float, np.ndarray]
        Time points or a single time value.
    l : Union[float, np.ndarray]
        Recovery rates or a single recovery rate.
    v : float
        The loss ratio, which is reconstruction cost divided by the total building structure value.
    k_str : float
        The total building structure value.
    pi : float
        Average productivity of capital.
    c0 : float
        Initial consumption level.
    eta : float
        The elasticity of marginal utility of consumption.

    Attributes
    ----------
    t : np.ndarray
        Array of time points. If multiple recovery rates are provided, this will be expanded to match the shape of `l`.
    l : np.ndarray
        Array of recovery rates.
    losses_t : np.ndarray
        Property that calculates the utility loss values for all combinations of `t` and `l`.
    """

    def __init__(
        self,
        t: Union[float, np.ndarray],
        l: Union[float, np.ndarray],
        v: float,
        k_str: float,
        pi: float,
        c0: float,
        eta: float,
    ):
        super().__init__(t, l)
        self._fun = lambda t, l: utility_loss_t(t, l, v, k_str, pi, c0, eta)
