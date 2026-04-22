from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from pydantic import BaseModel, Field

from .methods import (
    ConsumptionLoss,
    IncomeLoss,
    RecoveryCost,
    UtilityLoss,
    equity_weight,
    opt_lambda,
    recovery_rate,
    recovery_time,
    wellbeing_loss,
)


class LossType(str, Enum):
    RECOVERY = "Recovery Costs"
    INCOME = "Loss of Housing Services"
    RENTAL_INCOME = "Loss of Housing Services (Rental)"
    LABOUR_INCOME = "Labour Income Loss"

    CONSUMPTION = "Consumption Loss"
    UTILITY = "Utility Loss"

    def __str__(self):
        return self.value


class CapitalStock(BaseModel):
    k: float = Field(..., description="Value of the capital stock")
    v: float = Field(..., description="Loss ratio for the capital stock")
    recovery_time: Optional[float] = Field(
        None, description="Recovery time for the capital stock"
    )
    recovery_rate: Optional[float] = Field(
        None, description="Recovery rate for the capital stock"
    )
    pi: Optional[float] = Field(
        None,
        description=(
            "Optional per-stock productivity of capital. When set, overrides "
            "IncomeConfig.pi for this stock in the income-loss, consumption-"
            "loss extras, and remaining-loss aggregation. Useful when housing "
            "and firm capital have different productivities (e.g. rent-adjusted "
            "pi for housing vs GDP-based pi for firms)."
        ),
    )


class Liquidity(BaseModel):
    savings: float = Field(0.0, description="Household savings")
    insurance: float = Field(0.0, description="Insurance payout")
    support: float = Field(0.0, description="External support")


class IncomeConfig(BaseModel):
    i_0: float = Field(..., description="Initial income rate per year")
    i_avg: float = Field(..., description="Average income rate per year")
    pi: float = Field(0.15, description="Productivity of capital")
    i_div: Optional[float] = Field(None, description="Diversified income per year")


class SimulationConfig(BaseModel):
    eta: float = Field(1.5, description="Elasticity of marginal utility of consumption")
    rho: float = Field(0.06, description="Discount rate")
    t_max: float = Field(10, description="Maximum simulation time")
    dt: float = Field(1 / 52, description="Time step")
    currency: str = Field("$", description="Currency symbol")
    c_min: float = Field(0.0, description="Minimum consumption rate per year")
    recovery_per: float = Field(
        95.0, description="Percentage of asset rebuilt to consider as recovered"
    )


class WellBeingConfig(BaseModel):
    owner_housing: CapitalStock
    rental_housing: Optional[CapitalStock] = None
    labour_assets: Optional[Dict[str, CapitalStock]] = None
    income: IncomeConfig
    liquidity: Optional[Liquidity] = Liquidity()
    simulation: Optional[SimulationConfig] = SimulationConfig()


class CommunityUnit:
    def __init__(self, config: WellBeingConfig) -> None:
        """
        Initialize the CommunityUnit class with the given configuration.

        Parameters
        ----------
        config : CommunityUnitConfig
            The configuration object containing all grouped parameters.

        Returns
        -------
        None
        """
        self.config = config
        # Time series based on simulation config
        self.t = self._get_time_array()
        self.time_series = pd.DataFrame({"time": self.t})
        # Storage for aggregate outputs
        self.total_losses = pd.Series(dtype=float)
        # Validate and complete recovery parameters for stocks
        self._validate_and_fill_recovery_params()

    def _get_time_array(self):
        t_max = self.config.simulation.t_max
        dt = self.config.simulation.dt
        dt_calc = t_max / (int(t_max / dt) + 1)
        return np.linspace(0, t_max, int(t_max / dt_calc) + 1)

    def __repr__(self):
        return (
            f"CommunityUnit(\n"
            f"  owner_housing = {self.config.owner_housing},\n"
            f"  rental_housing = {self.config.rental_housing},\n"
            f"  labour_assets = {self.config.labour_assets},\n"
            f"  income = {self.config.income},\n"
            f"  liquidity = {self.config.liquidity},\n"
            f"  simulation = {self.config.simulation}\n"
            f")"
        )

    # --- Internal helpers to keep methods config-driven ---
    def _reconstruction_rate(self) -> Optional[float]:
        """Housing reconstruction rate λ (formerly recovery rate)."""
        if self.config.owner_housing.recovery_rate is not None:
            return self.config.owner_housing.recovery_rate
        if self.config.owner_housing.recovery_time is not None:
            return recovery_rate(
                self.config.owner_housing.recovery_time,
                rebuilt_per=self.config.simulation.recovery_per,
            )
        return None

    def _reconstruction_time(self) -> Optional[float]:
        """Housing reconstruction time T (formerly recovery time)."""
        if self.config.owner_housing.recovery_time is not None:
            return self.config.owner_housing.recovery_time
        if self.config.owner_housing.recovery_rate is not None:
            return recovery_time(
                rate=self.config.owner_housing.recovery_rate,
                rebuilt_per=self.config.simulation.recovery_per,
            )
        return None

    # Backward-compatible aliases
    def _rec_rate(self) -> Optional[float]:
        return self._reconstruction_rate()

    def _recovery_time(self) -> Optional[float]:
        return self._reconstruction_time()

    def _stock_rec_rate(self, stock: CapitalStock) -> Optional[float]:
        if stock.recovery_rate is not None:
            return stock.recovery_rate
        if stock.recovery_time is not None:
            return recovery_rate(
                stock.recovery_time, rebuilt_per=self.config.simulation.recovery_per
            )
        return None

    def _stock_pi(self, stock: CapitalStock) -> float:
        """Resolve the productivity for a given stock: per-stock override or income.pi fallback."""
        if stock is not None and stock.pi is not None:
            return float(stock.pi)
        return float(self.config.income.pi)

    def _extra_losses(self):
        extra = []
        # Rental housing as a single optional stock
        if self.config.rental_housing is not None:
            rr = self._stock_rec_rate(self.config.rental_housing)
            if rr is None:
                raise ValueError(
                    "rental_housing must define either recovery_rate or recovery_time"
                )
            n0 = (
                self._stock_pi(self.config.rental_housing)
                * self.config.rental_housing.v
                * self.config.rental_housing.k
            )
            extra.append((n0, rr))

        # Labour assets as a dictionary of named CapitalStock entries
        if self.config.labour_assets:
            for name, stock in self.config.labour_assets.items():
                if stock is None:
                    # Skip empty entries; treat as absent
                    continue
                rr = self._stock_rec_rate(stock)
                if rr is None:
                    raise ValueError(
                        f"labour_assets['{name}'] must define either recovery_rate or recovery_time"
                    )
                n0 = self._stock_pi(stock) * stock.v * stock.k
                extra.append((n0, rr))

        return extra if extra else None

    def _validate_and_fill_recovery_params(self) -> None:
        # For any CapitalStock: only one of recovery_time or recovery_rate can be set
        # If one is provided, compute the other using simulation.recovery_per
        def complete_stock(stock: CapitalStock, allow_none: bool) -> None:
            if stock is None:
                if allow_none:
                    return
                raise ValueError("Missing required CapitalStock configuration")
            has_rate = stock.recovery_rate is not None
            has_time = stock.recovery_time is not None
            if has_rate and has_time:
                raise ValueError(
                    "Provide only one of recovery_rate or recovery_time for CapitalStock"
                )
            if not has_rate and not has_time:
                if allow_none:
                    # housing can be optimized later
                    return
                raise ValueError(
                    "For rental_housing and labour_assets, provide recovery_rate or recovery_time"
                )
            if has_rate and not has_time:
                stock.recovery_time = recovery_time(
                    rate=stock.recovery_rate,
                    rebuilt_per=self.config.simulation.recovery_per,
                )
            if has_time and not has_rate:
                stock.recovery_rate = recovery_rate(
                    time=stock.recovery_time,
                    rebuilt_per=self.config.simulation.recovery_per,
                )

        # owner_housing: allow none, will be set by optimization if missing
        complete_stock(self.config.owner_housing, allow_none=True)
        # rental_housing and labour_assets must have one specified if provided
        if self.config.rental_housing is not None:
            complete_stock(self.config.rental_housing, allow_none=False)
        if self.config.labour_assets is not None:
            # Expect a dictionary of labour asset stocks
            if isinstance(self.config.labour_assets, dict):
                for name, stock in self.config.labour_assets.items():
                    try:
                        complete_stock(stock, allow_none=False)
                    except ValueError as e:
                        raise ValueError(f"Invalid labour_assets['{name}']: {e}")
            else:
                # Backward compatibility: if a single CapitalStock is passed, convert to dict
                stock = self.config.labour_assets  # type: ignore[assignment]
                try:
                    complete_stock(stock, allow_none=False)
                except ValueError as e:
                    raise ValueError(f"Invalid labour_assets: {e}")
                # Promote to dict for internal consistency
                self.config.labour_assets = {"labour": stock}  # type: ignore[assignment]

    def _c0(self) -> float:
        # Include diversified income (if provided) in baseline income
        i_div = self.config.income.i_div or 0.0
        # Consumption equals income by assumption
        return self.config.income.i_0 + i_div

    def _c_avg(self) -> float:
        # Consumption equals income by assumption
        return self.config.income.i_avg

    def _has_liquidity(self) -> bool:
        return self._liquidity() != 0

    def _liquidity(self) -> float:
        liq = self.config.liquidity
        if not liq:
            return 0.0
        return (liq.savings or 0.0) + (liq.insurance or 0.0) + (liq.support or 0.0)

    def _loss_types_for_run(self):
        types = [
            LossType.RECOVERY,
            LossType.INCOME,
        ]
        if self.config.rental_housing is not None:
            types.append(LossType.RENTAL_INCOME)
        if self.config.labour_assets is not None:
            types.append(LossType.LABOUR_INCOME)
        types.extend([LossType.CONSUMPTION, LossType.UTILITY])
        return types

    def _unit_recovery_time(self) -> Optional[float]:
        """
        Recovery time defined from a sum-of-exponentials model.

        We model the remaining income-loss stream as a sum of exponentials
        L(t) = Σ_i C_i * exp(-λ_i t), where each component corresponds to the
        productivity-weighted remaining damaged capital:
        - Owner housing: C = pi * v_owner * k_owner, λ = λ_owner
        - Optional rental housing: C = pi * v_rental * k_rental, λ = λ_rental
        - Each labour asset: C = pi * v_labour * k_labour, λ = λ_labour

        We solve for the time T at which the remaining loss equals the
        target fraction of the initial loss, i.e.,
            Σ_i C_i * exp(-λ_i T) = r * Σ_i C_i,
        where r = 1 - recovery_per/100.

        Notes
        -----
        - This definition ignores liquidity-induced piecewise effects and
          uses the underlying exponential modes only.
        - The owner term uses pi * v * k (same shape as rental and labour), so
          the remaining-loss proxy is homogeneous in lambda. The recovery-cost
          stream lambda * v * k * exp(-lambda t) belongs to consumption loss,
          not to the remaining-capital proxy; including it here inflates the
          owner coefficient roughly (pi + lambda)/pi ~ 20x at lambda=pi*(~20),
          which makes the aggregate T collapse toward owner's fast mode even
          for small v_owner and breaks monotonicity of T vs damage.
        - Requires all involved rates λ_i > 0 and coefficients C_i ≥ 0.
        """
        comps = self._exponential_loss_components()
        if comps is None:
            return None
        coeffs, rates = comps
        if not coeffs:
            return 0.0
        C_sum = float(np.sum(coeffs))
        if C_sum <= 0:
            return 0.0

        remaining_fraction = 1.0 - (self.config.simulation.recovery_per / 100.0)
        if remaining_fraction <= 0.0:
            return 0.0
        if remaining_fraction >= 1.0:
            return 0.0

        target = remaining_fraction * C_sum

        def f(t: float) -> float:
            # Monotone decreasing in t for Ci>=0, λi>0
            return float(
                np.sum([c * np.exp(-lam * t) for c, lam in zip(coeffs, rates)]) - target
            )

        # Bracket: f(0) = C_sum - target = (1 - remaining_fraction) * C_sum > 0
        f0 = C_sum - target
        if f0 <= 0:
            return 0.0

        lam_min = float(np.min(rates))
        if lam_min <= 0:
            return None

        # Start with a conservative upper bound and expand if needed
        upper = max(10.0 / lam_min, 1.0)
        f_upper = f(upper)
        iters = 0
        while f_upper > 0 and upper < 1e6 and iters < 60:
            upper *= 2.0
            f_upper = f(upper)
            iters += 1

        if f_upper > 0:
            # Not crossed within reasonable horizon
            return None

        # Bisection
        low, high = 0.0, upper
        for _ in range(80):
            mid = 0.5 * (low + high)
            fm = f(mid)
            if abs(fm) <= 1e-9:
                return mid
            if fm > 0:
                low = mid
            else:
                high = mid
        return 0.5 * (low + high)

    def _exponential_loss_components(self):
        """
        Build the list of (C_i, λ_i) components for remaining-loss L(t).

        Returns
        -------
        Optional[Tuple[List[float], List[float]]]
            Two lists (coefficients, rates). Returns None if owner rate missing/invalid.
        """
        labelled = self._exponential_loss_components_labelled()
        if labelled is None:
            return None
        coeffs = [c for _, c, _ in labelled]
        rates = [r for _, _, r in labelled]
        return coeffs, rates

    def _exponential_loss_components_labelled(self):
        """Same as _exponential_loss_components but keeps a label per term.

        Returns [(name, C_i, lambda_i), ...] or None if owner rate missing.
        Names are "owner", "rental", "labour/<key>".
        """
        rr_owner = self._rec_rate()
        if rr_owner is None or rr_owner <= 0:
            return None
        v_owner = self.config.owner_housing.v
        k_owner = self.config.owner_housing.k

        labelled: List[Tuple[str, float, float]] = []

        c_base = self._stock_pi(self.config.owner_housing) * v_owner * k_owner
        if c_base < 0:
            return None
        if c_base > 0:
            labelled.append(("owner", float(c_base), float(rr_owner)))

        if self.config.rental_housing is not None:
            stock = self.config.rental_housing
            rr = self._stock_rec_rate(stock)
            if rr is None or rr <= 0:
                return None
            c_rental = self._stock_pi(stock) * stock.v * stock.k
            if c_rental < 0:
                return None
            if c_rental > 0:
                labelled.append(("rental", float(c_rental), float(rr)))

        if self.config.labour_assets:
            for name, stock in self.config.labour_assets.items():
                if stock is None:
                    continue
                rr = self._stock_rec_rate(stock)
                if rr is None or rr <= 0:
                    return None
                c_lab = self._stock_pi(stock) * stock.v * stock.k
                if c_lab < 0:
                    return None
                if c_lab > 0:
                    labelled.append((f"labour/{name}", float(c_lab), float(rr)))

        return labelled

    def recovery_times_per_component(self) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Per-component recovery diagnostics.

        Each entry is a dict with keys:
        - `recovery_time`: ln(1/(1-r)) / lambda_i, i.e. the time for this single
          exponential component alone to reach `recovery_per` percent rebuilt.
        - `rate`: lambda_i as used in the aggregate.
        - `coefficient`: C_i = pi * v * k (the weight in Sum(C_i)).
        - `share`: coefficient / Sum(C_i) (fraction of the aggregate driven by
          this component). Lets callers see which process dominates the
          single-number `recovery_time` and whether that dominance is load-
          bearing for their interpretation.

        Returns None when the aggregate cannot be built (e.g. owner rate missing).
        """
        labelled = self._exponential_loss_components_labelled()
        if labelled is None:
            return None
        if not labelled:
            return {}
        total_C = float(sum(C for _, C, _ in labelled))
        out: Dict[str, Dict[str, float]] = {}
        rec_per = self.config.simulation.recovery_per
        for name, C, lam in labelled:
            out[name] = {
                "recovery_time": float(recovery_time(rate=lam, rebuilt_per=rec_per)),
                "rate": float(lam),
                "coefficient": float(C),
                "share": float(C / total_C) if total_C > 0 else 0.0,
            }
        return out


    def achieved_recovery_percent(
        self, t: Optional[float] = None, realized: bool = False
    ) -> Optional[float]:
        """
        Compute the actually achieved recovery percentage by time t.

        Definitions
        -----------
        - Remaining loss L(t) = Σ_i C_i exp(-λ_i t).
        - Achieved recovery fraction R(t) = 1 − L(t)/L(0).

        Parameters
        ----------
        t : float, optional
            Time horizon to evaluate. Defaults to simulation t_max.
        realized : bool, default False
            If True, use realized consumption losses (with liquidity effects)
            by evaluating Δc(t) from ConsumptionLoss; otherwise use the
            exponential decomposition (ignoring liquidity piecewise effects).

        Returns
        -------
        Optional[float]
            Achieved recovery percent in [0, 100]. None if undefined.
        """
        if t is None:
            t = float(self.config.simulation.t_max)
        if t < 0:
            return None

        if realized:
            # Evaluate Δc at 0 and t using the realized model
            rr = self._rec_rate()
            if rr is None or rr <= 0:
                return None
            loss = ConsumptionLoss(
                t=np.array([0.0, float(t)], dtype=float),
                rec_rate=rr,
                v=self.config.owner_housing.v,
                k_str=self.config.owner_housing.k,
                pi=self._stock_pi(self.config.owner_housing),
                liquidity=self._liquidity(),
                extra_losses=self._extra_losses(),
            )
            vals = np.asarray(loss.losses_t, dtype=float).reshape(-1)
            if vals.size < 2:
                return None
            L0 = float(vals[0])
            Lt = float(vals[-1])
        else:
            comps = self._exponential_loss_components()
            if comps is None:
                return None
            coeffs, rates = comps
            if not coeffs:
                return 100.0
            L0 = float(np.sum(coeffs))
            Lt = float(
                np.sum([c * np.exp(-lam * float(t)) for c, lam in zip(coeffs, rates)])
            )

        if L0 <= 0:
            return None
        frac = 1.0 - (Lt / L0)
        # Clamp to [0,1]
        if frac < 0.0:
            frac = 0.0
        elif frac > 1.0:
            frac = 1.0
        return 100.0 * frac

    def calc_loss(self, loss_type: LossType, method: str = "trapezoid") -> float:
        """
        Calculate the loss based on the specified loss type and method.

        Parameters
        ----------
        loss_type : LossType
            The type of loss to calculate. Must be one of the following:
            - LossType.RECOVERY: Calculates recovery cost.
            - LossType.INCOME: Calculates income loss.
            - LossType.CONSUMPTION: Calculates consumption loss.
            - LossType.UTILITY: Calculates utility loss.
        method : str, optional
            The numerical method to use for calculating the total loss.
            Can be either "trapezoid" (default) or "quad"

        Returns
        -------
        float
            The total loss calculated for the specified loss type.

        Raises
        ------
        ValueError
            If an invalid loss type is provided, or if a loss type that depends
            on owner housing is requested while owner_housing has neither
            `recovery_rate` nor `recovery_time` set.
        """
        owner_touched = loss_type in (
            LossType.RECOVERY,
            LossType.INCOME,
            LossType.CONSUMPTION,
            LossType.UTILITY,
        )
        if owner_touched and self._rec_rate() is None:
            raise ValueError(
                f"Cannot compute {loss_type}: owner_housing has no recovery_rate "
                "or recovery_time set. Call opt_lambda() first to optimize the "
                "housing recovery rate, or specify one on CapitalStock."
            )

        if loss_type == LossType.RECOVERY:
            loss = RecoveryCost(
                self.t,
                self._rec_rate(),
                self.config.owner_housing.v,
                self.config.owner_housing.k,
            )
        elif loss_type == LossType.INCOME:
            loss = IncomeLoss(
                self.t,
                self._rec_rate(),
                self.config.owner_housing.v,
                self.config.owner_housing.k,
                self._stock_pi(self.config.owner_housing),
            )
        elif loss_type in (LossType.RENTAL_INCOME, LossType.LABOUR_INCOME):
            if loss_type == LossType.RENTAL_INCOME:
                stock = self.config.rental_housing
                if stock is None:
                    return 0.0
                loss = IncomeLoss(
                    self.t,
                    self._stock_rec_rate(stock),
                    stock.v,
                    stock.k,
                    self._stock_pi(stock),
                )
                self.time_series[loss_type] = loss.losses_t
                self.total_losses[loss_type] = loss.total(rho=0, method=method)
                return self.total_losses[loss_type]
            else:
                # Aggregate labour income losses over all provided labour assets
                if not self.config.labour_assets:
                    return 0.0
                losses_sum = np.zeros_like(self.t, dtype=float)
                total_sum = 0.0
                for name, stock in self.config.labour_assets.items():
                    if stock is None:
                        continue
                    rr = self._stock_rec_rate(stock)
                    if rr is None:
                        raise ValueError(
                            f"labour_assets['{name}'] must define either recovery_rate or recovery_time"
                        )
                    l = IncomeLoss(
                        self.t,
                        rr,
                        stock.v,
                        stock.k,
                        self._stock_pi(stock),
                    )
                    # Store per-asset component series and totals
                    asset_key = f"{LossType.LABOUR_INCOME.value} ({name})"
                    self.time_series[asset_key] = l.losses_t
                    self.total_losses[asset_key] = l.total(rho=0, method=method)
                    # Accumulate into aggregate
                    losses_sum = losses_sum + l.losses_t
                    total_sum = total_sum + self.total_losses[asset_key]
                self.time_series[loss_type] = losses_sum
                self.total_losses[loss_type] = total_sum
                return self.total_losses[loss_type]
        elif loss_type == LossType.CONSUMPTION:
            loss = ConsumptionLoss(
                self.t,
                self._rec_rate(),
                self.config.owner_housing.v,
                self.config.owner_housing.k,
                self._stock_pi(self.config.owner_housing),
                liquidity=self._liquidity(),
                extra_losses=self._extra_losses(),
            )
            if self._has_liquidity():
                loss_no_liq = ConsumptionLoss(
                    self.t,
                    self._rec_rate(),
                    self.config.owner_housing.v,
                    self.config.owner_housing.k,
                    self._stock_pi(self.config.owner_housing),
                    liquidity=0.0,
                    extra_losses=self._extra_losses(),
                )
                self.time_series[f"{loss_type} No Liquidity"] = loss_no_liq.losses_t
        elif loss_type == LossType.UTILITY:
            loss = UtilityLoss(
                self.t,
                self._rec_rate(),
                self.config.owner_housing.v,
                self.config.owner_housing.k,
                self._stock_pi(self.config.owner_housing),
                self._c0(),
                self.config.simulation.eta,
                self.config.simulation.c_min,
                liquidity=self._liquidity(),
                extra_losses=self._extra_losses(),
            )
        else:
            raise ValueError(f"Invalid loss type: {loss_type}")

        self.time_series[loss_type] = loss.losses_t
        self.total_losses[loss_type] = loss.total(rho=0, method=method)

        return self.total_losses[loss_type]

    def get_losses(self, method: str = "trapezoid") -> pd.Series:
        """
        Calculate and update various types of losses for the household.

        This method computes losses for each loss type defined in the `LossType` enumeration,
        calculates the equivalent consumption loss, and determines the equity-weighted loss.
        The results are stored in the `total_losses` attribute.

        Parameters
        ----------
        method : str, optional
            The numerical integration method to use for calculations.
            Can be either "trapezoid" (default) or "quad".

        Returns
        -------
        pd.Series
            A pandas Series containing the following keys:
            - "Wellbeing Loss": The calculated wellbeing loss.
            - "Asset Loss": The calculated asset loss.
            - "Equity Weighted Asset Loss": Owner asset loss weighted by the
              inverse-marginal-utility factor `(c0/c_avg)^(-eta)`. Note: this
              weights the raw asset loss (pre-recovery), not the welfare
              (wellbeing) loss; rename reflects that honestly. For a welfare-
              scaled equity metric, multiply `equity_weight(c0, c_avg, eta)` by
              `"Wellbeing Loss"` yourself.
            - Additional keys corresponding to each `LossType` (e.g., "Recovery Costs", "Income Loss").

        Notes
        -----
        - The `LossType` enumeration is iterated to calculate individual loss types.
        - The `UtilityLoss` class is used to compute the utility loss.
        - The `wellbeing_loss` and `equity_weight` functions are used to compute the respective metrics.
        - The results are stored in the `time_series` DataFrame and `total_losses` Series attributes.
        """
        # Calculate losses for configured loss types only
        for loss_type in self._loss_types_for_run():
            self.calc_loss(loss_type, method=method)

        # Calculate equivalent consumption loss
        ut_t = UtilityLoss(
            t=self.t,
            rec_rate=self._rec_rate(),
            v=self.config.owner_housing.v,
            k_str=self.config.owner_housing.k,
            pi=self._stock_pi(self.config.owner_housing),
            c0=self._c0(),
            eta=self.config.simulation.eta,
            cmin=self.config.simulation.c_min,
            liquidity=self._liquidity(),
            extra_losses=self._extra_losses(),
        )
        du_dis = ut_t.total(rho=self.config.simulation.rho, method=method)
        well_being_loss = wellbeing_loss(
            du=du_dis, c_avg=self._c_avg(), eta=self.config.simulation.eta
        )
        # Calculate equity weighted loss
        ew_loss = (
            equity_weight(
                c=self._c0(), c_avg=self._c_avg(), eta=self.config.simulation.eta
            )
            * self.config.owner_housing.v
            * self.config.owner_housing.k
        )

        # Update total losses with additional metrics
        self.total_losses["Wellbeing Loss"] = well_being_loss
        self.total_losses["Asset Loss"] = (
            self.config.owner_housing.v * self.config.owner_housing.k
        )
        self.total_losses["Equity Weighted Asset Loss"] = ew_loss

        # Expose as primary recovery time attribute for the unit
        self.recovery_time = self._unit_recovery_time()
        # Per-component breakdown: owner / rental / labour. The aggregate
        # `recovery_time` above collapses these onto one axis; callers should
        # prefer this dict when plotting recovery time against damage.
        self.recovery_time_per_component = self.recovery_times_per_component()

        return self.total_losses

    def plot_loss(
        self, loss_type: "LossType | str", ax: Optional[plt.Axes] = None
    ) -> Optional[plt.Figure]:
        """
        Plot the time series of losses for a given type.

        This method visualizes the losses over time for a specified loss type. It includes a shaded area under the curve
        to represent the total loss and formats the y-axis based on the type of loss being plotted.

        Parameters
        ----------
        loss_type : LossType
            The type of loss to plot. Must be one of the values in the LossType enum.
        ax : matplotlib.axes.Axes, optional
            The axes on which to plot. If None, a new figure and axes are created. Default is None.

        Returns
        -------
        Optional[plt.Figure]
            The matplotlib figure object if ax is None, otherwise None.

        Raises
        ------
        ValueError
            If the specified loss type has not been calculated.
        ValueError
            If an invalid loss type is provided.
        """
        if loss_type not in self.time_series.columns:
            raise ValueError(f"{loss_type} losses have not been calculated.")
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        sns.lineplot(x="time", y=loss_type, data=self.time_series, ax=ax)
        # Shade area under curve with consistent x-axis as in plot_consumption
        total_val = self.total_losses[loss_type]
        if isinstance(loss_type, LossType) and loss_type == LossType.UTILITY:
            label_total = f"Total {loss_type}: {total_val:.2f}"
        else:
            label_total = (
                f"Total {loss_type}: {total_val:,.0f} {self.config.simulation.currency}"
            )
        ax.fill_between(
            self.time_series["time"],
            0,
            self.time_series[loss_type],
            edgecolor="gray",
            alpha=0.3,
            label=label_total,
        )
        ax.set_xlabel("Time after disaster (years)")
        # Align y-axis formatting and units with plot_consumption
        if not (isinstance(loss_type, LossType) and loss_type == LossType.UTILITY):
            ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, pos: f"{int(x):,}")
            )
            ax.set_ylabel(f"{loss_type} ({self.config.simulation.currency}/year)")
        else:
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
            ax.set_ylabel(f"{loss_type}")
        # Make time axis start at 0 and end at simulation horizon
        try:
            t_min = float(np.nanmin(self.time_series["time"]))
            t_max = float(np.nanmax(self.time_series["time"]))
            ax.set_xlim(left=max(0.0, t_min), right=t_max)
        except Exception:
            pass
        # Add legend consistently
        ax.legend()

        if ax is None:
            return fig

    def plot_consumption(
        self, ax: Optional[plt.Axes] = None, plot_cmin=True
    ) -> Optional[plt.Figure]:
        """
        Plot the consumption losses over time, stacking loss of housing services and recovery costs with different hatches and colors.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes on which to plot. If None, a new figure and axes are created. Default is None.

        Returns
        -------
        Optional[plt.Figure]
            The matplotlib figure object if ax is None, otherwise None.

        Raises
        ------
        ValueError
            If the losses have not been calculated.
        """
        if LossType.CONSUMPTION not in self.time_series.columns:
            raise ValueError(
                "Losses have not been calculated. Run the 'get_losses' method first."
            )

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        # Prepare component series
        time = self.time_series["time"]
        inc = self.time_series[LossType.INCOME]
        recon = self.time_series[LossType.RECOVERY]
        rental = (
            self.time_series[LossType.RENTAL_INCOME]
            if LossType.RENTAL_INCOME in self.time_series.columns
            else None
        )
        # Identify per-asset labour income component columns created in calc_loss
        labour_asset_cols = [
            c
            for c in self.time_series.columns
            if isinstance(c, str) and c.startswith(f"{LossType.LABOUR_INCOME.value} (")
        ]

        # Colors and labels
        color_income = "brown"
        color_recon = "lightcoral"
        color_rental = "sienna"
        # Distinct colors for each labour asset
        labour_colors = {}
        if labour_asset_cols:
            palette = sns.color_palette("Set2", n_colors=len(labour_asset_cols))
            for col, col_color in zip(labour_asset_cols, palette):
                labour_colors[col] = col_color

        # Compute stacked areas from bottom to top
        # Start baseline at c0 - (all components)
        # We'll fill successive layers up to c0
        # First, sum labour components
        labour_sum = None
        for col in labour_asset_cols:
            labour_sum = (
                self.time_series[col].copy()
                if labour_sum is None
                else labour_sum + self.time_series[col]
            )
        if labour_sum is None:
            labour_sum = 0
        rental_series = rental if rental is not None else 0

        # Bottom layer: Recovery
        label_recon = (
            f"Total {LossType.RECOVERY}: "
            f"{self.total_losses[LossType.RECOVERY]:,.0f} "
            f"{self.config.simulation.currency}"
        )
        ax.fill_between(
            time,
            self._c0() - inc - rental_series - labour_sum - recon,
            self._c0() - inc - rental_series - labour_sum,
            facecolor=color_recon,
            alpha=0.6,
            label=label_recon,
        )

        # Next layer: Income
        label_income = (
            f"Total {LossType.INCOME}: {self.total_losses[LossType.INCOME]:,.0f} "
            f"{self.config.simulation.currency}"
        )
        ax.fill_between(
            time,
            self._c0() - inc - rental_series - labour_sum,
            self._c0() - rental_series - labour_sum,
            facecolor=color_income,
            alpha=0.6,
            label=label_income,
        )

        # Next layer: Rental (if any)
        if rental is not None:
            label_rental = (
                f"Total {LossType.RENTAL_INCOME}: "
                f"{self.total_losses[LossType.RENTAL_INCOME]:,.0f} "
                f"{self.config.simulation.currency}"
            )
            ax.fill_between(
                time,
                self._c0() - rental_series - labour_sum,
                self._c0() - labour_sum,
                facecolor=color_rental,
                alpha=0.6,
                label=label_rental,
            )

        # Top layers: each labour asset separately
        running_top = self._c0() - labour_sum
        for col in labour_asset_cols:
            series = self.time_series[col]
            prev = running_top
            running_top = running_top + series
            # The label uses the total stored for this component if present
            total_val = self.total_losses.get(col, float(series.to_numpy().sum()))
            label_lab = (
                f"Total {col}: {total_val:,.0f} {self.config.simulation.currency}"
            )
            ax.fill_between(
                time,
                prev,
                running_top,
                facecolor=labour_colors.get(col, "peru"),
                alpha=0.6,
                label=label_lab,
            )

        # If there is liquidity available, plot the consumption losses without liquidity
        if self._has_liquidity():
            label3 = (
                f"Total Liquidity: "
                f"{self._liquidity():,.0f} "
                f"{self.config.simulation.currency}"
            )
            # Add hatch by drawing again with no fill color, only hatch
            ax.fill_between(
                self.time_series["time"],
                self._c0() - self.time_series[LossType.CONSUMPTION],
                self._c0() - self.time_series[f"{LossType.CONSUMPTION} No Liquidity"],
                facecolor="none",
                edgecolor="black",
                hatch="///",
                linewidth=0.0,
                label=label3,
            )

        # Plot consumption losses with a dashed line and expand to the left by 4 months
        expanded_time = np.insert(self.time_series["time"], 0, [-1, -0.001])
        expanded_consumption_losses = np.insert(
            self._c0() - self.time_series[LossType.CONSUMPTION],
            0,
            [self._c0(), self._c0()],
        )
        ax.plot(
            expanded_time,
            expanded_consumption_losses,
            linestyle="--",
            color="red",
            label="Consumption",
        )

        # Plot general unit recovery time based on consumption losses
        urt = getattr(self, "unit_recovery_time", None)
        if urt is None:
            urt = self._unit_recovery_time()
        if urt is not None:
            ax.axvline(
                x=urt,
                color="black",
                linestyle=":",
                alpha=0.5,
                label=f"Recovery time: {urt:.2f} years",
            )

        # Plot cmin
        if plot_cmin:
            ax.axhline(
                y=self.config.simulation.c_min,
                color="black",
                linestyle="--",
                alpha=1,
                label=(
                    f"Minimum consumption: {self.config.simulation.c_min:,.0f} "
                    f"{self.config.simulation.currency}"
                ),
            )

        # Add lightning icon at t=0 years just above self.c0
        lightning_icon = "\u26a1"
        ax.text(
            0,
            self._c0() + 0.1,
            lightning_icon,
            fontsize=50,
            color="red",
            verticalalignment="bottom",
            horizontalalignment="center",
        )

        # Plot consumption losses
        ax.set_xlabel("Time after disaster (years)")
        ax.set_ylabel(f"Consumption rate ({self.config.simulation.currency}/year)")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x):,}"))
        # Add legend
        ax.legend()

        if ax is None:
            return fig

    def opt_lambda(
        self,
        rec_time_max: Optional[float] = None,
        rec_time_min: float = 0.3,
        no_steps: float = 1000,
        method: str = "trapezoid",
        eps_rel: float = 0.0,
        raise_on_fail: bool = True,
    ) -> None:
        """
        Optimize the recovery rate (lambda) to minimize the total well-being loss.

        Parameters
        ----------
        rec_time_max : float, optional
            The maximum recovery time to consider. If None, the maximum time in the simulation is used.
        rec_time_min : float, optional
            The minimum recovery time to consider. Default is 0.3 years.
        no_steps : float, optional
            The number of steps to divide the recovery time range into. Default is 1000.
        method : str, optional
            The numerical integration method to use for loss calculations. Can be either "trapezoid" (default) or "quad"
        eps_rel : float, optional
            Relative tolerance for the optimization. If greater than 0, the function will return
            the smallest lambda within the relative tolerance of the minimum loss.
            Default is 0.01.

        Returns
        -------
        Optional[float]
            The optimal reconstruction rate if found; otherwise None when raise_on_fail=False.
        """
        # Check if the maximum recovery time is provided, else use the maximum time
        if rec_time_max is None:
            rec_time_max = self.t[-1]

        # Create array of lambda values to check
        times = np.linspace(rec_time_min, rec_time_max, no_steps)
        lambdas = recovery_rate(times, rebuilt_per=self.config.simulation.recovery_per)

        # Calculate losses for each lambda value
        # Initialize arrays to store losses for each lambda
        recovery_costs = []
        income_losses = []
        consumption_losses = []
        utility_losses = []

        # Iterate through each lambda value and calculate losses
        for lmbd in lambdas:
            recovery_costs.append(
                RecoveryCost(
                    t=self.t,
                    rec_rate=lmbd,
                    v=self.config.owner_housing.v,
                    k_str=self.config.owner_housing.k,
                ).total(rho=0, method=method)
            )
            income_losses.append(
                IncomeLoss(
                    t=self.t,
                    rec_rate=lmbd,
                    v=self.config.owner_housing.v,
                    k_str=self.config.owner_housing.k,
                    pi=self._stock_pi(self.config.owner_housing),
                ).total(rho=0, method=method)
            )
            consumption_losses.append(
                ConsumptionLoss(
                    t=self.t,
                    rec_rate=lmbd,
                    v=self.config.owner_housing.v,
                    k_str=self.config.owner_housing.k,
                    pi=self._stock_pi(self.config.owner_housing),
                    liquidity=self._liquidity(),
                    extra_losses=self._extra_losses(),
                ).total(rho=0, method=method)
            )
            utility_losses.append(
                UtilityLoss(
                    t=self.t,
                    rec_rate=lmbd,
                    v=self.config.owner_housing.v,
                    k_str=self.config.owner_housing.k,
                    pi=self._stock_pi(self.config.owner_housing),
                    c0=self._c0(),
                    eta=self.config.simulation.eta,
                    cmin=self.config.simulation.c_min,
                    liquidity=self._liquidity(),
                    extra_losses=self._extra_losses(),
                ).total(rho=0, method=method)
            )

        # Convert lists to numpy arrays for further processing
        recovery_costs = np.array(recovery_costs)
        income_losses = np.array(income_losses)
        consumption_losses = np.array(consumption_losses)
        utility_losses = np.array(utility_losses)

        opt = opt_lambda(
            v=self.config.owner_housing.v,
            k_str=self.config.owner_housing.k,
            c0=self._c0(),
            pi=self._stock_pi(self.config.owner_housing),
            eta=self.config.simulation.eta,
            l_min=lambdas.min(),
            l_max=lambdas.max(),
            t_max=self.config.simulation.t_max,
            times=self.t,
            method=method,
            cmin=self.config.simulation.c_min,
            eps_rel=eps_rel,
            liquidity=self._liquidity(),
            extra_losses=self._extra_losses(),
        )
        self.lambda_opt = opt

        # Save optimization dataframe regardless of success
        df = pd.DataFrame(
            {
                "lambda": lambdas,
                "reconstruction_time": recovery_time(
                    rate=lambdas, rebuilt_per=self.config.simulation.recovery_per
                ),
                LossType.RECOVERY: recovery_costs,
                LossType.INCOME: income_losses,
                LossType.CONSUMPTION: consumption_losses,
                LossType.UTILITY: utility_losses,
            }
        )
        self.l_opt = df

        if not opt.get("success", True):
            if raise_on_fail:
                raise RuntimeError(
                    f"opt_lambda failed: {opt.get('message') or 'no message'}"
                )
            # Otherwise return without mutating config
            return opt

        optimal_lambda = opt["l_opt"]

        # Persist optimal parameters back into config for downstream use
        # Persist (config schema uses recovery_*; keep fields but rename semantics)
        self.config.owner_housing.recovery_rate = optimal_lambda
        self.config.owner_housing.recovery_time = recovery_time(
            rate=optimal_lambda, rebuilt_per=self.config.simulation.recovery_per
        )

        return opt

    def plot_opt_lambda(self, x_type: Literal["rate", "time"] = "rate") -> plt.Figure:
        """
        Plot the optimization results for the reconstruction rate (lambda) or recovery time.

        Parameters
        ----------
        x_type : Literal["rate", "time"], optional
            The type of x-axis to use for the plot. Can be "rate" for reconstruction rate or "time" for recovery time. Default is "rate".

        Returns
        -------
        plt.Figure
            The matplotlib figure object containing the plots.

        Raises
        ------
        ValueError
            If the optimal lambda has not been calculated.
        """
        if not hasattr(self, "l_opt"):
            raise ValueError(
                "Optimal lambda has not been calculated. Run the 'opt_lambda' method first."
            )

        # Create a 2x2 subplot
        fig, axs = plt.subplots(2, 1, figsize=(5, 8), sharex=True)
        # Check how x axis should be configured
        if x_type == "rate":
            x = self.l_opt["lambda"]
            val = self._reconstruction_rate()
            val_min = self.lambda_opt["l_opt_min"]
            leg = "Reconstruction-rate λ"
        elif x_type == "time":
            x = self.l_opt["reconstruction_time"]
            val = self._reconstruction_time()
            val_min = recovery_time(
                rate=self.lambda_opt["l_opt_min"],
                rebuilt_per=self.config.simulation.recovery_per,
            )
            leg = "Reconstruction time (years)"
            # axs[0].set_xscale('log')
        axs[1].set_xlabel(leg)
        # Make line plots for consumption losses
        sns.lineplot(
            x=x,
            y=self.l_opt[LossType.RECOVERY],
            color="green",
            ax=axs[0],
            label=LossType.RECOVERY,
        )
        sns.lineplot(
            x=x,
            y=self.l_opt[LossType.INCOME],
            color="blue",
            ax=axs[0],
            label=LossType.INCOME,
        )
        sns.lineplot(
            x=x,
            y=self.l_opt[LossType.CONSUMPTION],
            color="purple",
            ax=axs[0],
            label=LossType.CONSUMPTION,
        )
        # Add vertical line for optimal lambda
        axs[0].axvline(
            x=val, color="red", linestyle="--", label=f"Optimum value: {val:.2f}"
        )
        # Fill between the tested lambda values
        ylims = (axs[0].get_ylim()[0], axs[0].get_ylim()[1])
        axs[0].fill_between(
            x, ylims[0], ylims[1], color="steelblue", alpha=0.3, label="Tested range"
        )
        axs[0].yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, pos: f"{int(x):,}")
        )
        axs[0].set_ylabel(f"Total Loss ({self.config.simulation.currency})")
        axs[0].set_ylim(ylims)
        # Add well-being loss plot
        sns.lineplot(
            x=x,
            y=self.l_opt[LossType.UTILITY],
            color="black",
            ax=axs[1],
            label=LossType.UTILITY,
        )
        axs[1].axvline(
            x=val, color="red", linestyle="--", label=f"Optimum value: {val:.2f}"
        )
        if val_min != val:
            axs[1].axvline(
                x=val_min,
                color="orange",
                linestyle="--",
                label=f"Minimum value: {val_min:.2f}",
            )
            # Add text below the label for consumption diff
            # Place the text under the legend (outside the plot area)
            axs[1].text(
                1.02,
                -0.15,
                f"consumption diff: {self.lambda_opt['C_diff']:.2f} "
                f"{self.config.simulation.currency}",
                color="orange",
                fontsize=10,
                verticalalignment="top",
                horizontalalignment="left",
                rotation=0,
                transform=axs[1].transAxes,
            )

        ylims = (axs[1].get_ylim()[0], axs[1].get_ylim()[1])
        axs[1].fill_between(
            x, ylims[0], ylims[1], color="steelblue", alpha=0.3, label="Tested range"
        )
        axs[1].set_ylabel("Total Utility Loss")
        axs[1].set_ylim(ylims)
        # Move legends outside the figure
        axs[0].legend(loc="upper left", bbox_to_anchor=(1, 1))
        axs[1].legend(loc="upper left", bbox_to_anchor=(1, 1))

        return fig
