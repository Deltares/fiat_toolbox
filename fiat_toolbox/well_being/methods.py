import warnings
from enum import Enum
from typing import Literal, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.optimize import brentq, minimize


class OptLambdaStatus(str, Enum):
    """Classification of `opt_lambda` outcomes (see `opt_lambda` docstring).

    Priority when multiple conditions would apply (e.g. a flat objective at
    a boundary):
    INFEASIBLE > FAILED > NO_RECOVERY_NEEDED > FLAT > BOUNDARY_* > INTERIOR.
    """

    INTERIOR = "interior"
    FLAT = "flat"
    NO_RECOVERY_NEEDED = "no_recovery_needed"
    BOUNDARY_LOWER = "boundary_lower"
    BOUNDARY_UPPER = "boundary_upper"
    INFEASIBLE = "infeasible"
    FAILED = "failed"

    def __str__(self):
        return self.value


def utility(
    consumption: Union[float, np.ndarray], eta: float, normalize: bool = False
) -> Union[float, np.ndarray]:
    """
    Calculate the utility of given consumption(s) using a CRRA (Constant Relative Risk Aversion) utility function.
    Parameters
    ----------
    consumption : Union[float, np.ndarray]
        The consumption value(s) for which to calculate utility. Can be a single float or a numpy array of floats.
    eta : float
        The elasticity of the marginal utility of consumption.
    normalize : bool, optional
        If True, normalize the utility values to the range [0, 1], by default False.
    Returns
    -------
    Union[float, np.ndarray]
        The calculated utility value(s). Returns a float if a single consumption value is provided, otherwise returns a numpy array.
    """
    consumption = np.array(consumption)  # Ensure input is a numpy array

    # Check for zero or negative consumption values and issue a warning.
    # The warning is emitted only once per process to avoid spam in loops; set
    # the simplefilter in the caller if you want repeats.
    if np.any(consumption <= 0):
        warnings.warn(
            "Consumption contains zero or negative values, resulting in NaN utility.",
            UserWarning,
            stacklevel=2,
        )
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


def inverse_utility(u: Union[float, np.ndarray], eta: float):
    """
    Compute the inverse of the utility function for given utility values and risk aversion parameter.

    Parameters
    ----------
    u : float or np.ndarray
        The utility value(s) to invert. Can be a scalar or a NumPy array.
    eta : float
        The elasticity of the marginal utility of consumption.

    Returns
    -------
    float or np.ndarray
        The value(s) corresponding to the inverse utility transformation of `u`, matching the input type.

    Notes
    -----
    - For eta == 1, the inverse utility is the exponential function.
    - For eta != 1, the inverse utility is computed using the CRRA formula.
    """
    if eta == 1:
        return np.exp(u)
    else:
        return (u * (1 - eta)) ** (1 / (1 - eta))


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


def recovery_cost_t(
    t: Union[float, np.ndarray], rec_rate: float, v: float, k_str: float
) -> np.ndarray:
    """
    Calculate the recovery cost over time. This represents a cost rate and not the total cost.

    Parameters
    ----------
    t : Union[float, np.ndarray]
        The time value(s) for which to calculate the recovery cost. Can be a single float or a numpy array of floats.
    rec_rate : Union[float, np.ndarray]
        The rate of recovery value(s). Can be a single float or a numpy array of floats.
    v : float
        The loss ratio, which is reconstruction cost divided by the total building structure value.
    k_str : float
        The total building structure value.

    Returns
    -------
    np.ndarray
        The calculated recovery cost(s) as an nxm matrix where n is the length of t and m is the length of rec_rate.
    """
    # Calculate the recovery cost
    cost = rec_rate * v * k_str * np.exp(-rec_rate * t)

    return cost


def income_loss_t(
    t: Union[float, np.ndarray],
    rec_rate: float,
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
    rec_rate : Union[float, np.ndarray]
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
        The calculated reconstruction cost(s) as an nxm matrix where n is the length of t and m is the length of rec_rate.
    """
    # Calculate the baseline income loss
    loss = pi * v * k_str * np.exp(-rec_rate * t)
    return loss


def consumption_loss_t(
    t: Union[float, np.ndarray],
    rec_rate: float,
    v: float,
    k_str: float,
    pi: float,
    liquidity: float = 0.0,
    extra_losses: Optional[Sequence[Tuple[float, float]]] = None,
) -> np.ndarray:
    """
    Calculate the consumption loss over time as the sum of income loss and reconstruction cost. This represents a loss rate and not the total loss.

    Parameters
    ----------
    t : Union[float, np.ndarray]
        The time value(s) for which to calculate the consumption loss. Can be a single float or a numpy array of floats.
    rec_rate : Union[float, np.ndarray]
        The rate of recovery value(s). Can be a single float or a numpy array of floats.
    v : float
        The loss ratio, which is reconstruction cost divided by the total building structure value.
    k_str : float
        The total building structure value.
    pi : float
        Average productivity of capital. Can be derived using Penn World Tables.
    liquidity : float, optional
        Total liquid support available (savings + insurance + external support). Default is 0.0.
    extra_losses : Optional[Sequence[Tuple[float, float]]], optional
        Optional list of (N0, lambda_N) pairs representing additional income-loss
        components, each decaying exponentially as N0 * exp(-lambda_N * t). Defaults to None.

    Returns
    -------
    np.ndarray
        The calculated consumption loss(es) as an nxm matrix where n is the length of t and m is the length of rec_rate.

        Notes
        -----
        Let the total consumption loss be a sum of exponentials:
        Δc(t) = α_base · exp(−λ t) + Σ_i N_i · exp(−μ_i t), where
        - α_base = income_loss_t(0) + recovery_cost_t(0) captures the base term at t=0,
        - λ = rec_rate is the recovery rate for the base term,
        - each (N_i, μ_i) describes an extra loss component with its own decay.

        Useful integrals:
        - ∫₀^t Δc(s) ds = α_base · (1 − exp(−λ t)) / λ + Σ_i N_i · (1 − exp(−μ_i t)) / μ_i
        - ∫₀^∞ Δc(s) ds = α_base / λ + Σ_i N_i / μ_i

        Liquidity threshold:
        - If liquidity ≥ α_base / λ + Σ_i N_i / μ_i, then savings/support fully offset losses and Δc(t) is set to 0 for all t.

        Optimal support period t̂:
        - When liquidity is insufficient to cover all losses, t̂ is defined implicitly by
            Δc(t̂) · t̂ + liquidity = ∫₀^{t̂} Δc(s) ds.
        - The piecewise loss is constructed as
            cl_t = Δc(t̂) for t ≤ t̂ (constant support level), and cl_t = Δc(t) for t > t̂,
            which ensures continuity at t̂ and reduces to the single-exponential case when no extra losses are provided.
    """

    def c_loss(t):
        base = income_loss_t(
            t=t, rec_rate=rec_rate, v=v, k_str=k_str, pi=pi
        ) + recovery_cost_t(t=t, rec_rate=rec_rate, v=v, k_str=k_str)
        if extra_losses:
            extra = 0.0
            for N0, lam in extra_losses:
                extra = extra + (N0 * np.exp(-lam * t))
            base = base + extra
        return base

    # α_base is the coefficient of the exp(-rec_rate * t) term (income + reconstruction at t=0),
    # while α_total = Δc(0) includes any extra_losses terms (sum of N0).
    alpha_base = income_loss_t(
        t=0, rec_rate=rec_rate, v=v, k_str=k_str, pi=pi
    ) + recovery_cost_t(t=0, rec_rate=rec_rate, v=v, k_str=k_str)

    has_owner = rec_rate > 0 and alpha_base > 0
    has_extras = bool(extra_losses)
    if liquidity <= 0 or not (has_owner or has_extras):
        # No liquidity to apply, or no loss stream to smooth.
        t_hat = 0
        cl_t = c_loss(t)
    else:
        # Compute the total integral of Δc(t) over [0, ∞) to check if liquidity
        # covers all losses. Only sum components that actually contribute, so
        # the extras-only case (α_base = 0) is handled correctly without a
        # spurious 0/rec_rate term.
        total_integral_inf = (alpha_base / rec_rate) if has_owner else 0.0
        if extra_losses:
            for N0, lam in extra_losses:
                total_integral_inf += N0 / lam

        if liquidity >= total_integral_inf:
            # Liquidity fully offsets all losses at all times
            t_hat = np.inf
            cl_t = np.zeros_like(t)
        else:
            # Solve for t̂ from: Δc(t̂) * t̂ + liquidity = ∫₀^{t̂} Δc(s) ds
            def integral_to_t(th: float) -> float:
                # Integral of the base component plus each extra loss component
                val = (
                    (alpha_base * (1 - np.exp(-rec_rate * th)) / rec_rate)
                    if has_owner
                    else 0.0
                )
                if extra_losses:
                    for N0, lam in extra_losses:
                        val += N0 * (1 - np.exp(-lam * th)) / lam
                return val

            def objective_t_hat(th: float) -> float:
                return c_loss(th) * th + liquidity - integral_to_t(th)

            # Find a suitable upper bracket where the function is negative.
            # Build min_rate from contributing components only — when α_base = 0
            # rec_rate doesn't drive a loss, so it shouldn't size the bracket.
            rates = []
            if has_owner:
                rates.append(rec_rate)
            if extra_losses:
                rates.extend(lam for _, lam in extra_losses)
            min_rate = min(rates)
            upper = max(10.0 / min_rate, 1.0)
            f_upper = objective_t_hat(upper)
            # Expand the bracket if needed
            while f_upper > 0 and upper < 1e6:
                upper *= 2
                f_upper = objective_t_hat(upper)

            if f_upper > 0:
                # Bracket never switched sign out to 1e6 years. Without this
                # guard brentq would raise a cryptic "f(a) and f(b) have the
                # same sign" error. Surface the domain cause instead.
                raise ValueError(
                    "consumption_loss_t: could not bracket t_hat in [0, 1e6] "
                    "for the liquidity-offset equation. Parameters leave no "
                    "crossing — typically happens when liquidity is extremely "
                    "large relative to the loss stream, or when recovery "
                    "rates are near zero. Check rec_rate, liquidity, and "
                    "extra_losses."
                )

            t_hat = brentq(objective_t_hat, 0.0, upper)
            # Set the constant reduction level equal to Δc(t̂) to ensure continuity
            const_level = c_loss(t_hat)
            t = np.array(t)
            cl_t = np.where(t <= t_hat, const_level, c_loss(t))
    return cl_t


def consumption_t(
    t: Union[float, np.ndarray],
    rec_rate: float,
    v: float,
    k_str: float,
    pi: float,
    c0: float,
    cmin: float = 0.0,
    liquidity: float = 0.0,
    extra_losses: Optional[Sequence[Tuple[float, float]]] = None,
) -> np.ndarray:
    """
    Calculate the consumption over time. This represents a consumption rate and not the total consumption.

    Parameters
    ----------
    t : Union[float, np.ndarray]
        The time value(s) for which to calculate the consumption. Can be a single float or a numpy array of floats.
    rec_rate : Union[float, np.ndarray]
        The rate of recovery value(s). Can be a single float or a numpy array of floats.
    v : float
        The loss ratio, which is reconstruction cost divided by the total building structure value.
    k_str : float
        The total building structure value.
    pi : float
        Average productivity of capital. Can be derived using Penn World Tables.
    c0 : float
        Initial consumption rate per year.
    cmin : float, optional
        Subsistence consumption threshold. Kept for signature compatibility but
        NOT subtracted here: c(t) = c0 - cl_t. The `cmin` value is enforced as
        a feasibility constraint inside `opt_lambda` (CRRA utility on c(t);
        subsistence is a constraint, not an arithmetic floor). Default is 0.0.
    liquidity : float, optional
        Total liquid support available (savings + insurance + external support). Default is 0.0.
    extra_losses : Optional[Sequence[Tuple[float, float]]], optional
        Optional list of (N0, lambda_N) pairs representing additional income-loss
        components, each decaying exponentially as N0 * exp(-lambda_N * t). Defaults to None.

    Returns
    -------
    np.ndarray
        The calculated consumption(es) as an nxm matrix where n is the length of t and m is the length of rec_rate.
    """
    cl_t = consumption_loss_t(
        t=t,
        rec_rate=rec_rate,
        v=v,
        k_str=k_str,
        pi=pi,
        liquidity=liquidity,
        extra_losses=extra_losses,
    )
    ct = c0 - cl_t
    return ct


def utility_loss_t(
    t: Union[float, np.ndarray],
    rec_rate: float,
    v: float,
    k_str: float,
    pi: float,
    c0: float,
    eta: float,
    cmin: float = 0.0,
    liquidity: float = 0.0,
    extra_losses: Optional[Sequence[Tuple[float, float]]] = None,
) -> np.ndarray:
    """
    Calculate the utility loss over time. This represents a loss rate and not the total loss.

    Parameters
    ----------
    t : Union[float, np.ndarray]
        The time value(s) for which to calculate the utility loss. Can be a single float or a numpy array of floats.
    rec_rate : Union[float, np.ndarray]
        The rate of recovery value(s). Can be a single float or a numpy array of floats.
    v : float
        The loss ratio, which is reconstruction cost divided by the total building structure value.
    k_str : float
        The total building structure value.
    pi : float
        Average productivity of capital. Can be derived using Penn World Tables.
    c0 : float
        Initial consumption rate per year.
    eta : float
        The elasticity of marginal utility of consumption.
    cmin : float, optional
        Minimum consumption rate per year. Default is 0.0.
    liquidity : float, optional
        Total liquid support available (savings + insurance + external support). Default is 0.0.
    extra_losses : Optional[Sequence[Tuple[float, float]]], optional
        Optional list of (N0, lambda_N) pairs representing additional income-loss
        components, each decaying exponentially as N0 * exp(-lambda_N * t). Defaults to None.

    Returns
    -------
    np.ndarray
        The calculated utility loss(es) as an nxm matrix where n is the length of t and m is the length of rec_rate.
    """
    c_t = consumption_t(
        t=t,
        rec_rate=rec_rate,
        v=v,
        k_str=k_str,
        pi=pi,
        c0=c0,
        cmin=cmin,
        liquidity=liquidity,
        extra_losses=extra_losses,
    )
    # Baseline utility is u(c_0); subsistence c_min is a feasibility
    # constraint handled in opt_lambda, not a baseline shift.
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
    loss_horizon: Optional[Union[float, Literal["recovery_time"]]] = None,
    method: str = "quad",
    cmin: float = 0.0,
    eps_rel: float = 0.0,
    eps_flat: float = 1e-3,
    liquidity: float = 0.0,
    extra_losses: Optional[Sequence[Tuple[float, float]]] = None,
    rho: float = 0.0,
    recovery_per: float = 95.0,
) -> dict:
    """
    Optimize the recovery rate (lambda) to minimize utility loss.

    Parameters
    ----------
    v, k_str, c0, pi, eta : float
        Loss ratio, structure value, initial consumption, capital productivity,
        CRRA elasticity. See module docstrings for details.
    l_min, l_max : float, optional
        Search bounds in λ-space. Defaults 0.3 / 10.
    t_max : float, optional
        Simulation horizon. Required when `method == "quad"`.
    times : np.ndarray, optional
        Time grid. Required when `method == "trapezoid"`.
    loss_horizon : float or {"recovery_time"}, optional
        End time through which to integrate the optimizer objective and
        feasibility check. If omitted, uses the full simulation horizon. If
        `"recovery_time"`, each candidate lambda is integrated through its own
        implied recovery time.
    method : str, optional
        Integration method, `"quad"` (default) or `"trapezoid"`.
    cmin : float, optional
        Subsistence consumption floor — feasibility constraint. Default 0.
    eps_rel : float, optional
        Relative tolerance for `l_opt` relabelling. When > 0, `l_opt` is set
        to the *largest* λ whose wellbeing loss stays within
        `loss_opt_min · (1 + eps_rel)`. The unrelaxed minimum remains on
        `l_opt_min` / `loss_opt_min`. Default 0.
    eps_flat : float, optional
        Relative tolerance for detecting a flat wellbeing surface. Default 1e-3.
    liquidity, extra_losses, rho : optional
        Forwarded to the consumption / utility-loss functions.
    recovery_per : float, optional
        Percentage of asset rebuilt considered "recovered" (0 ≤ p < 100).
        Used by the `t_max` diagnostic for `BOUNDARY_LOWER` — if at the
        optimum `1 − exp(−λ·t_max) < recovery_per/100`, the message flags
        that the simulation horizon may be the binding constraint rather
        than the wellbeing shape. Default 95.

    Returns
    -------
    dict
        - `status`: `OptLambdaStatus` — classification of the outcome:
          `INTERIOR` / `FLAT` / `NO_RECOVERY_NEEDED` / `BOUNDARY_LOWER` /
          `BOUNDARY_UPPER` / `INFEASIBLE` / `FAILED`. Priority when several
          apply:
          INFEASIBLE > FAILED > NO_RECOVERY_NEEDED > FLAT > BOUNDARY_* > INTERIOR.
        - `success`: bool — True for INTERIOR / FLAT / NO_RECOVERY_NEEDED /
          BOUNDARY_*; False for INFEASIBLE / FAILED.
        - `message`: str | None — case-specific explanation with an
          actionable hint. None only for INTERIOR (nothing to report).
        - `l_opt_min`, `loss_opt_min`: the true minimum (eps_rel-agnostic).
        - `l_opt`, `loss_opt`: `l_opt_min` by default; overwritten with the
          largest-λ-within-eps_rel when `eps_rel > 0`.
        - `eps_rel`: the requested tolerance (echoed).
        - `C_diff`, `T_diff`: equivalent-consumption and recovery-time gap
          between the true minimum and the eps_rel-relaxed point.

    Notes
    -----
    - This function does not raise on failure; it returns `success=False`
      and a descriptive message.
    - The `objective` is evaluated inside a `warnings.catch_warnings()`
      block that suppresses the `"Consumption contains zero or negative
      values"` `UserWarning`, since that firing is the normal way
      infeasible candidates are rejected.
    """
    # Initialize a single result dict and update it throughout
    result: dict = {
        "status": OptLambdaStatus.FAILED,
        "success": False,
        "message": None,
        "l_opt_min": None,
        "loss_opt_min": None,
        "eps_rel": eps_rel,
        "l_opt": None,
        "loss_opt": None,
        "C_diff": None,
        "T_diff": None,
    }

    # Validate inputs without raising
    if method == "quad":
        if t_max is None:
            result.update(
                {
                    "message": "t_max must be provided when using the 'quad' method.",
                }
            )
            return result
        times = np.array([0, t_max])
    elif method == "trapezoid" and times is None:
        result.update(
            {
                "message": "times must be provided when using the 'trapezoid' method.",
            }
        )
        return result

    available_horizon = (
        float(t_max) if method == "quad" else float(np.asarray(times, dtype=float)[-1])
    )
    candidate_recovery_horizon = loss_horizon == "recovery_time"
    if candidate_recovery_horizon:
        horizon = None
    elif isinstance(loss_horizon, str):
        result.update(
            {
                "message": (
                    "loss_horizon must be None, a positive finite value, or "
                    "'recovery_time'."
                ),
            }
        )
        return result
    elif loss_horizon is None:
        horizon = available_horizon
    else:
        horizon = float(loss_horizon)
        if not np.isfinite(horizon) or horizon <= 0:
            result.update({"message": "loss_horizon must be a positive finite value."})
            return result
        if horizon > available_horizon:
            result.update(
                {
                    "message": (
                        "loss_horizon must be less than or equal to the available "
                        "simulation horizon."
                    ),
                }
            )
            return result

    times_arr = np.asarray(times, dtype=float)

    def candidate_horizon(rec_rate: float) -> float:
        if candidate_recovery_horizon:
            return float(recovery_time(rate=rec_rate, rebuilt_per=recovery_per))
        return float(horizon)

    if candidate_recovery_horizon:
        max_candidate_horizon = candidate_horizon(l_min)
        if max_candidate_horizon > available_horizon:
            result.update(
                {
                    "message": (
                        "loss_horizon='recovery_time' requires all candidate "
                        "recovery times to be within the available simulation "
                        "horizon. Increase t_max/times or increase l_min."
                    ),
                }
            )
            return result

    def times_for_horizon(eval_horizon: float) -> np.ndarray:
        if method == "quad":
            return np.array([0.0, eval_horizon])
        mask_inside = (times_arr > times_arr[0]) & (times_arr < eval_horizon)
        return np.concatenate(
            ([float(times_arr[0])], times_arr[mask_inside], [eval_horizon])
        )

    # NO_RECOVERY_NEEDED: owner has no physical damage (v·k == 0), so owner's
    # λ does not enter the objective. Reporting a λ here would be meaningless —
    # the downstream FLAT branch would otherwise pick the largest probe point
    # arbitrarily. Evaluate the objective once at a placeholder λ (any λ>0
    # works because owner's λ-dependent terms carry a v·k=0 prefactor) so the
    # extras-only constant is reported honestly. If the feasibility constraint
    # fails at that placeholder, demote to INFEASIBLE to honour the priority
    # INFEASIBLE > FAILED > NO_RECOVERY_NEEDED > FLAT > ...
    if v * k_str == 0:
        placeholder_lambda = max(float(l_min), 1.0)
        eval_horizon = candidate_horizon(placeholder_lambda)
        times_eval = times_for_horizon(eval_horizon)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Consumption contains zero or negative values",
                category=UserWarning,
            )
            cl_grid = consumption_loss_t(
                t=times_eval,
                rec_rate=placeholder_lambda,
                v=v,
                k_str=k_str,
                pi=pi,
                liquidity=liquidity,
                extra_losses=extra_losses,
            )
            cl_peak = float(np.max(np.asarray(cl_grid)))
            if c0 - cl_peak < cmin:
                result.update(
                    {
                        "status": OptLambdaStatus.INFEASIBLE,
                        "success": False,
                        "message": (
                            f"No feasible reconstruction rate in [{l_min:.3g}, {l_max:.3g}]: "
                            f"owner has no physical damage (v·k == 0) but extras push "
                            f"c(t) below c_min={cmin:g}. Hint: add liquidity or reduce "
                            "extra loss magnitudes."
                        ),
                    }
                )
                return result
            ut_t = UtilityLoss(
                times_eval,
                placeholder_lambda,
                v,
                k_str,
                pi,
                c0,
                eta,
                cmin,
                liquidity,
                extra_losses,
            )
            loss_val = float(ut_t.total(rho=rho, method=method, t2=eval_horizon))
        result.update(
            {
                "status": OptLambdaStatus.NO_RECOVERY_NEEDED,
                "success": True,
                "message": (
                    "Owner has no physical damage (v·k == 0); λ is undefined "
                    "for owner housing. Extras (if any) contribute a "
                    f"λ-independent constant loss of {loss_val:.3g}. "
                    "Returned l_opt=None — do not report a recovery time."
                ),
                "l_opt_min": None,
                "loss_opt_min": loss_val,
                "l_opt": None,
                "loss_opt": loss_val,
            }
        )
        return result

    def objective(rec_rate: float) -> float:
        # Infeasible candidates are the normal way to reject bad λs during
        # search; utility() emits "Consumption contains zero or negative
        # values" UserWarning when that happens. Silence it within the
        # objective — callers that use methods.utility directly still see it.
        with warnings.catch_warnings():
            eval_horizon = candidate_horizon(rec_rate)
            times_eval = times_for_horizon(eval_horizon)
            warnings.filterwarnings(
                "ignore",
                message="Consumption contains zero or negative values",
                category=UserWarning,
            )
            # Feasibility: reject any lambda that would drop c(t) below c_min.
            # Peak Δc occurs either at t=0 (no-liquidity case) or at the plateau
            # γ = Δc(t̂) (liquidity case); consumption_loss_t encodes both, so a
            # max over the time grid is exact.
            cl_grid = consumption_loss_t(
                t=times_eval,
                rec_rate=rec_rate,
                v=v,
                k_str=k_str,
                pi=pi,
                liquidity=liquidity,
                extra_losses=extra_losses,
            )
            cl_peak = float(np.max(np.asarray(cl_grid)))
            if c0 - cl_peak < cmin:
                return np.inf
            ut_t = UtilityLoss(
                times_eval,
                rec_rate,
                v,
                k_str,
                pi,
                c0,
                eta,
                cmin,
                liquidity,
                extra_losses,
            )
            loss = ut_t.total(rho=rho, method=method, t2=eval_horizon)
            return loss

    def fun(rec_rate):
        return objective(rec_rate)

    # Silence Nelder-Mead's "invalid value encountered in subtract"
    # RuntimeWarning: it fires inside scipy's termination check when the
    # simplex contains +inf entries from infeasible candidate λs. The
    # objective already handles those correctly (treated as +∞ and
    # rejected); the warning is exploration chatter, not a real problem.
    # This filter sits outside the objective because scipy's check runs
    # outside it.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in subtract",
            category=RuntimeWarning,
        )
        res = minimize(fun, l_min, bounds=[(l_min, l_max)], method="Nelder-Mead")

    # Probe the full range to detect plateaus near the minimum. Nelder-Mead
    # has no gradient inside a plateau and "converges" at an arbitrary point
    # (dependent on the initial simplex and `l_min`); in particular, when the
    # plateau is bounded by a feasibility cliff at one end, NM started at
    # `l_min` walks down-gradient and stops at the cliff edge — the *slowest*
    # recovery in the plateau, opposite of the FLAT tie-break promise. A
    # cheap 21-point probe lets us identify all λs whose loss is within
    # `eps_flat · loss_scale` of the best observed minimum and pick the
    # largest one (fastest recovery), regardless of where NM landed.
    l_probe = np.linspace(l_min, l_max, 21)
    probe_losses = np.array([objective(lm) for lm in l_probe])
    finite_mask = np.isfinite(probe_losses)

    # --- Classification (priority: INFEASIBLE > FAILED > FLAT > BOUNDARY > INTERIOR) ---

    # INFEASIBLE: no feasible λ anywhere in the probe.
    if not finite_mask.any():
        result.update(
            {
                "status": OptLambdaStatus.INFEASIBLE,
                "success": False,
                "message": (
                    f"No feasible reconstruction rate in [{l_min:.3g}, {l_max:.3g}]: "
                    f"every λ pushes c(t) below c_min={cmin:g}. "
                    "Hint: reduce damage v, lower c_min, add liquidity, or "
                    "narrow the search to slower rates."
                ),
            }
        )
        return result

    # FAILED: NM did not converge and probe has finite points (so it isn't
    # global infeasibility — something else went wrong).
    if not res.success:
        result.update(
            {
                "status": OptLambdaStatus.FAILED,
                "success": False,
                "message": (
                    f"Solver did not converge on [{l_min:.3g}, {l_max:.3g}]: "
                    f"{res.message}. Hint: widen bounds or adjust tolerances."
                ),
            }
        )
        return result

    l_opt = res.x[0]
    loss_opt = res.fun

    # Plateau detection. Tolerance scales to the absolute loss magnitude
    # (`eps_flat · loss_scale`), which handles zero-valued minima cleanly —
    # `eps_rel`'s `loss_opt · (1+eps_rel)` band collapses to zero when
    # `loss_opt = 0` and would not break ties at all in that regime. Use the
    # better of NM and probe minima as the reference, since NM can stall at
    # a sub-optimum on a flat surface. The tie-break (largest λ in plateau)
    # always fires — it is conceptually the same as `eps_rel`'s relabel,
    # just with an absolute tolerance — but classification depends on where
    # the plateau and the picked l_opt land:
    #   FLAT          : plateau touches both bounds (globally flat).
    #   BOUNDARY_*    : l_opt sits at a boundary; if a plateau exists, its
    #                   extent is reported as a hint. This covers both
    #                   monotone shapes and cliff-bounded plateaus.
    #   INTERIOR      : l_opt is strictly interior and the plateau (if any)
    #                   does not touch a boundary.
    finite = probe_losses[finite_mask]
    loss_range = float(finite.max() - finite.min())
    loss_scale = float(max(abs(finite.max()), abs(finite.min()), 1e-300))
    tol = eps_flat * loss_scale
    ref_min = min(loss_opt, float(finite.min()))
    within = finite_mask & (probe_losses <= ref_min + tol)
    plateau_count = int(within.sum())
    if plateau_count >= 2:
        plateau_l_lo = float(l_probe[within].min())
        plateau_l_hi = float(l_probe[within].max())
        # Tie-break toward fastest recovery: pick the largest λ in the
        # plateau. Consistent with the eps_rel relabel convention.
        idx = int(np.flatnonzero(within).max())
        l_opt = float(l_probe[idx])
        loss_opt = float(probe_losses[idx])
    else:
        plateau_l_lo = plateau_l_hi = float(l_opt)

    bound_tol = max(1e-9, 1e-6 * (l_max - l_min))
    at_lower = abs(l_opt - l_min) <= bound_tol
    at_upper = abs(l_opt - l_max) <= bound_tol
    plateau_touches_min = plateau_count >= 2 and abs(plateau_l_lo - l_min) <= bound_tol
    plateau_touches_max = plateau_count >= 2 and abs(plateau_l_hi - l_max) <= bound_tol
    is_globally_flat = plateau_touches_min and plateau_touches_max
    plateau_hint = ""
    if plateau_count >= 2 and not is_globally_flat:
        plateau_hint = (
            f" A near-optimal plateau extends across "
            f"λ ∈ [{plateau_l_lo:.3g}, {plateau_l_hi:.3g}] "
            f"(loss within eps_flat={eps_flat:g} of the minimum)."
        )

    if is_globally_flat:
        status = OptLambdaStatus.FLAT
        message = (
            f"Wellbeing function is flat across the full search range "
            f"λ ∈ [{l_min:.3g}, {l_max:.3g}] (loss range {loss_range:.3e} "
            f"within eps_flat={eps_flat:g} of loss scale). Every λ gives "
            f"essentially the same wellbeing loss; returned λ={l_opt:.3g} "
            "is the largest λ in the plateau (fastest recovery — "
            "consistent with the eps_rel convention). The household is "
            "indifferent to reconstruction speed in this regime (e.g. "
            "liquidity covers all losses, or tilt is sub-tolerance). "
            "The owner still has physical damage to rebuild — for the "
            "no-damage case see NO_RECOVERY_NEEDED."
        )
    elif at_lower:
        status = OptLambdaStatus.BOUNDARY_LOWER
        # t_max diagnostic: at the slow-recovery end, check whether the
        # simulation horizon is the binding constraint rather than the
        # wellbeing shape.
        diagnostic_horizon = candidate_horizon(l_opt)
        fraction = 1.0 - float(np.exp(-l_opt * diagnostic_horizon))
        t_max_hint = ""
        if fraction < recovery_per / 100.0:
            t_max_hint = (
                f" Recovery is only {fraction:.0%} complete by "
                f"t_max={diagnostic_horizon:g}y, below recovery_per={recovery_per}%. "
                "The slow-λ preference may be an artefact of the "
                "simulation horizon — consider increasing t_max."
            )
        message = (
            f"Minimum at lower λ bound ({l_opt:.3g} ≈ l_min), "
            "i.e. the slowest recovery / longest reconstruction time "
            "in the search range. Wellbeing may keep improving beyond "
            "this bound. Hint: widen the search (increase rec_time_max "
            "on CommunityUnit.opt_lambda)." + t_max_hint + plateau_hint
        )
    elif at_upper:
        status = OptLambdaStatus.BOUNDARY_UPPER
        message = (
            f"Minimum at upper λ bound ({l_opt:.3g} ≈ l_max), "
            "i.e. the fastest recovery / shortest reconstruction time "
            "in the search range. Wellbeing may keep improving beyond "
            "this bound. Hint: widen the search (decrease rec_time_min "
            "on CommunityUnit.opt_lambda)." + plateau_hint
        )
    else:
        status = OptLambdaStatus.INTERIOR
        message = None

    # Populate result with successful optimum values
    result.update(
        {
            "status": status,
            "success": True,
            "message": message,
            "l_opt_min": l_opt,
            "loss_opt_min": loss_opt,
            "l_opt": l_opt,
            "loss_opt": loss_opt,
        }
    )
    # Relative-tolerance relabel: among all grid lambdas whose wellbeing loss
    # sits inside [loss_opt, loss_opt*(1+eps_rel)] (the near-optimal band),
    # pick the *largest* lambda — i.e. the fastest recovery that still meets
    # the tolerance. The true minimum is preserved on `l_opt_min` /
    # `loss_opt_min`; `l_opt` / `loss_opt` are overwritten with the relabelled
    # value. Flip to `valid_indices[0]` if the use case wants the slowest
    # near-optimal schedule instead. The status continues to reflect the
    # unrelaxed minimum (l_opt_min), not the relabelled l_opt.
    if eps_rel > 0:
        threshold = loss_opt * (1 + eps_rel)
        l_grid = np.linspace(l_min, l_max, 1000)
        losses = np.array([objective(rec_rate) for rec_rate in l_grid])
        valid_indices = np.where(losses <= threshold)[0]
        if valid_indices.size == 0:
            # Keep the original optimum, but flag as not meeting eps_rel tolerance
            result.update(
                {
                    "status": OptLambdaStatus.FAILED,
                    "success": False,
                    "message": (
                        f"No lambda found within the relative tolerance of "
                        f"{eps_rel} from the minimum loss."
                    ),
                }
            )
            return result
        ind = valid_indices[-1]
        l_opt_new = l_grid[ind]
        loss_opt_new = losses[ind]

        result.update(
            {
                "l_opt": l_opt_new,
                "loss_opt": loss_opt_new,
                "C_diff": inverse_utility(loss_opt_new, eta)
                - inverse_utility(loss_opt, eta),
                "T_diff": recovery_time(l_opt) - recovery_time(l_opt_new),
            }
        )

    return result


class Loss:
    """
    A base class for calculating losses over time based on recovery rates.

    Parameters
    ----------
    t : Union[float, np.ndarray], optional
        Time points provided as a single float or a numpy array. If `t` is None, `t_max`
        must be provided.
    rec_rate : Union[float, np.ndarray], optional
        Levels provided as a single float or a numpy array. Defaults to None.
    t_max : float, optional
        Maximum time value used to generate time points if `t` is not provided. Defaults to None.

    Attributes
    ----------
    t : np.ndarray
        Array of time points. If multiple levels are provided in `rec_rate`, this will be reshaped to align with `rec_rate`.
    rec_rate : np.ndarray
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
    - Inputs `t` and `rec_rate` are converted to at least 1D numpy arrays.
    - If `rec_rate` contains more than one element, `t` is reshaped to align with `rec_rate` for meshgrid-like behavior.
    """

    _fun: callable

    def __init__(
        self,
        t: Optional[Union[float, np.ndarray]] = None,
        rec_rate: Optional[Union[float, np.ndarray]] = None,
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
        t = np.atleast_1d(t)

        self.t = t
        self.rec_rate = rec_rate

    @property
    def losses_t(self) -> np.ndarray:
        """
        Calculate the loss values for all combinations of time points and levels.

        Returns
        -------
        np.ndarray
            Array of loss values for all combinations of `t` and `rec_rate`.
        """
        f_t = self._fun(self.t, self.rec_rate)
        return f_t

    def total(
        self,
        rho: float = 0,
        method: str = "trapezoid",
        t1: Optional[float] = None,
        t2: Optional[float] = None,
    ) -> Union[float, np.ndarray]:
        """
        Calculate the total loss by integrating over time, optionally between two bounds.

        Parameters
        ----------
        rho : float, optional
            Discount rate for the integration. Defaults to 0.
        method : str, optional
            Integration method to use. Can be either "trapezoid" (default) or "quad".
            "trapezoid" uses the numpy.trapz function, while "quad" uses scipy.integrate.quad.
        t1 : float, optional
            Start time for integration. If None, uses the first time in `t`.
        t2 : float, optional
            End time for integration. If None, uses the last time in `t`.

        Returns
        -------
        Union[float, np.ndarray]
            The total loss. Returns a float if a single level is provided, otherwise returns a numpy array.

        Raises
        ------
        ValueError
            If `t` has less than 2 points when using the "trapezoid" method.
        ValueError
            If `t2` is smaller than `t1`.
        ValueError
            If an invalid integration method is provided.
        """
        # Determine integration bounds
        t_arr = self.t
        if t_arr.size < 2:
            raise ValueError("t must have at least 2 points to calculate the integral.")
        t_start = float(t_arr[0]) if t1 is None else float(t1)
        t_end = float(t_arr[-1]) if t2 is None else float(t2)
        if t_end < t_start:
            raise ValueError("t2 must be greater than or equal to t1.")
        # Clamp to available domain
        t_start = max(t_start, float(t_arr[0]))
        t_end = min(t_end, float(t_arr[-1]))
        if t_end == t_start:
            return 0.0

        if method == "trapezoid":
            # Compute undiscounted values on original grid
            f_t = self._fun(t_arr, self.rec_rate)

            # Helper to linearly interpolate f at a time between grid points
            def interp_f_at(t_val: float):
                # If exactly on grid
                if t_val <= t_arr[0]:
                    f_val = f_t[0]
                elif t_val >= t_arr[-1]:
                    f_val = f_t[-1]
                else:
                    idx = np.searchsorted(t_arr, t_val, side="right")
                    i0 = idx - 1
                    i1 = idx
                    t0, t1i = float(t_arr[i0]), float(t_arr[i1])
                    w = (t_val - t0) / (t1i - t0)
                    f_val = f_t[i0] + w * (f_t[i1] - f_t[i0])
                # Apply discounting at t_val
                return f_val * np.exp(-rho * t_val)

            # Build subgrid within [t_start, t_end] including boundaries
            mask_inside = (t_arr > t_start) & (t_arr < t_end)
            t_sub = np.concatenate(
                ([t_start], t_arr[mask_inside].astype(float), [t_end])
            )

            # Discounted values on subgrid
            f_dis_inside = self._fun(t_arr[mask_inside], self.rec_rate) * np.exp(
                -rho * t_arr[mask_inside]
            )

            f_start = interp_f_at(t_start)
            f_end = interp_f_at(t_end)

            # Stack along time axis (axis=0)
            if f_t.ndim == 1:
                f_sub = np.concatenate(
                    (
                        np.atleast_1d(f_start),
                        f_dis_inside,
                        np.atleast_1d(f_end),
                    ),
                    axis=0,
                )
            else:
                f_start_2d = np.expand_dims(f_start, axis=0)
                f_end_2d = np.expand_dims(f_end, axis=0)
                f_sub = np.concatenate((f_start_2d, f_dis_inside, f_end_2d), axis=0)

            # Project pins numpy<2 (see root CLAUDE.md); np.trapz is valid here.
            integral = np.trapz(f_sub, x=t_sub, axis=0)  # noqa: NPY201
        elif method == "quad":
            # Note: quad integrates scalar-valued functions. For vector-valued rec_rate,
            # prefer method="trapezoid". Here, we keep previous behavior and integrate
            # the scalar result element if applicable.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=IntegrationWarning)
                integral = np.array(
                    quad(
                        lambda tt, li=self.rec_rate: self._fun(tt, li)
                        * np.exp(-rho * tt),
                        t_start,
                        t_end,
                    )[0]
                )
        else:
            raise ValueError("method must be either 'trapezoid' or 'quad'.")

        # Ensure the result is always an ndarray or a single float
        return integral if integral.size > 1 else integral.item()


class RecoveryCost(Loss):
    """
    A class to calculate recovery cost over time based on recovery rates.
    It includes the total method to calculate the total loss by integrating over time.

    Parameters
    ----------
    t : Union[float, np.ndarray]
        The time value(s) for which to calculate the recovery cost. Can be a single float or a numpy array of floats.
    rec_rate : Union[float, np.ndarray]
        The rate of recovery value(s). Can be a single float or a numpy array of floats.
    v : float
        The loss ratio, which is recovery cost divided by the total building structure value.
    k_str : float
        The total building structure value.

    Attributes
    ----------
    t : np.ndarray
        Array of time points. If multiple recovery rates are provided, this will be expanded to match the shape of `rec_rate`.
    rec_rate : np.ndarray
        Array of recovery rates.
    losses_t : np.ndarray
        Property that calculates the recovery cost values for all combinations of `t` and `rec_rate`.
    """

    def __init__(
        self,
        t: Union[float, np.ndarray],
        rec_rate: float,
        v: float,
        k_str: float,
    ):
        super().__init__(t, rec_rate)
        self._fun = lambda t, rec_rate: recovery_cost_t(t, rec_rate, v, k_str)


class IncomeLoss(Loss):
    """
    A class to calculate income loss over time based on recovery rates.
    It includes the total method to calculate the total loss by integrating over time.

    Parameters
    ----------
    t : Union[float, np.ndarray]
        Time points or a single time value.
    rec_rate : Union[float, np.ndarray]
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
        Array of time points. If multiple recovery rates are provided, this will be expanded to match the shape of `rec_rate`.
    rec_rate : np.ndarray
        Array of recovery rates.
    losses_t : np.ndarray
        Property that calculates the income loss values for all combinations of `t` and `rec_rate`.
    """

    def __init__(
        self,
        t: Union[float, np.ndarray],
        rec_rate: float,
        v: float,
        k_str: float,
        pi: float,
    ):
        super().__init__(t, rec_rate)
        self._fun = lambda t, rec_rate: income_loss_t(t, rec_rate, v, k_str, pi)


class ConsumptionLoss(Loss):
    """
    A class to calculate consumption loss over time based on recovery rates.
    It includes the total method to calculate the total loss by integrating over time.

    Parameters
    ----------
    t : Union[float, np.ndarray]
        Time points or a single time value.
    rec_rate : Union[float, np.ndarray]
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
        Array of time points. If multiple recovery rates are provided, this will be expanded to match the shape of `rec_rate`.
    rec_rate : np.ndarray
        Array of recovery rates.
    losses_t : np.ndarray
        Property that calculates the consumption loss values for all combinations of `t` and `rec_rate`.
    """

    def __init__(
        self,
        t: Union[float, np.ndarray],
        rec_rate: float,
        v: float,
        k_str: float,
        pi: float,
        liquidity: float = 0.0,
        extra_losses: Optional[Sequence[Tuple[float, float]]] = None,
    ):
        super().__init__(t, rec_rate)
        self._fun = lambda t, rec_rate: consumption_loss_t(
            t, rec_rate, v, k_str, pi, liquidity, extra_losses
        )


class UtilityLoss(Loss):
    """
    A class to calculate utility loss over time based on recovery rates.
    It includes the total method to calculate the total loss by integrating over time.

    Parameters
    ----------
    t : Union[float, np.ndarray]
        Time points or a single time value.
    rec_rate : Union[float, np.ndarray]
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
        Array of time points. If multiple recovery rates are provided, this will be expanded to match the shape of `rec_rate`.
    rec_rate : np.ndarray
        Array of recovery rates.
    losses_t : np.ndarray
        Property that calculates the utility loss values for all combinations of `t` and `rec_rate`.
    """

    def __init__(
        self,
        t: Union[float, np.ndarray],
        rec_rate: float,
        v: float,
        k_str: float,
        pi: float,
        c0: float,
        eta: float,
        cmin: float = 0.0,
        liquidity: float = 0.0,
        extra_losses: Optional[Sequence[Tuple[float, float]]] = None,
    ):
        super().__init__(t, rec_rate)
        self._fun = lambda t, rec_rate: utility_loss_t(
            t,
            rec_rate,
            v,
            k_str,
            pi,
            c0,
            eta,
            cmin,
            liquidity,
            extra_losses,
        )
