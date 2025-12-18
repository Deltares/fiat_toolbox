from enum import Enum
from typing import Literal, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

from pydantic import BaseModel, Field
from typing import Optional

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


# TODO Make class a pydantic model
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
    recovery_time: Optional[float] = Field(None, description="Recovery time for the capital stock")
    recovery_rate: Optional[float] = Field(None, description="Recovery rate for the capital stock")

class Liquidity(BaseModel):
    savings: float = Field(0.0, description="Household savings")
    insurance: float = Field(0.0, description="Insurance payout")
    support: float = Field(0.0, description="External support")

class IncomeConfig(BaseModel):
    i_0: float = Field(..., description="Initial income rate per year")
    i_avg: float = Field(..., description="Average income rate per year")
    pi: float = Field(0.15, description="Productivity of capital")
    c_L: Optional[float] = Field(None, description="Labour income share")
    c_i_ratio: float = Field(1.0, description="Consumption to income ratio")
    i_div: Optional[float] = Field(None, description="Diversified income per year")

class SimulationConfig(BaseModel):
    eta: float = Field(1.5, description="Elasticity of marginal utility of consumption")
    rho: float = Field(0.06, description="Discount rate")
    t_max: float = Field(10, description="Maximum simulation time")
    dt: float = Field(1/52, description="Time step")
    currency: str = Field("$", description="Currency symbol")
    c_min: float = Field(0.0, description="Minimum consumption rate per year")
    recovery_per: float = Field(95.0, description="Percentage of asset rebuilt to consider as recovered")

class WellBeingConfig(BaseModel):
    housing: CapitalStock
    rental_housing: Optional[CapitalStock] = None
    labour_assets: Optional[CapitalStock] = None
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
            f"  housing = {self.config.housing},\n"
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
        if self.config.housing.recovery_rate is not None:
            return self.config.housing.recovery_rate
        if self.config.housing.recovery_time is not None:
            return recovery_rate(
                self.config.housing.recovery_time,
                rebuilt_per=self.config.simulation.recovery_per,
            )
        return None

    def _reconstruction_time(self) -> Optional[float]:
        """Housing reconstruction time T (formerly recovery time)."""
        if self.config.housing.recovery_time is not None:
            return self.config.housing.recovery_time
        if self.config.housing.recovery_rate is not None:
            return recovery_time(
                rate=self.config.housing.recovery_rate,
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

    def _extra_losses(self):
        extra = []
        if self.config.rental_housing is not None:
            rr = self._stock_rec_rate(self.config.rental_housing)
            if rr is None:
                raise ValueError(
                    "rental_housing must define either recovery_rate or recovery_time"
                )
            n0 = (
                self.config.income.pi
                * self.config.rental_housing.v
                * self.config.rental_housing.k
            )
            extra.append((n0, rr))
        if self.config.labour_assets is not None:
            rr = self._stock_rec_rate(self.config.labour_assets)
            if rr is None:
                raise ValueError(
                    "labour_assets must define either recovery_rate or recovery_time"
                )
            n0 = (
                self.config.income.c_L
                * self.config.income.pi
                * self.config.labour_assets.v
                * self.config.labour_assets.k
            )
            extra.append((n0, rr))
        return extra if len(extra) > 0 else None

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

        # housing: allow none, will be set by optimization if missing
        complete_stock(self.config.housing, allow_none=True)
        # rental_housing and labour_assets must have one specified if provided
        if self.config.rental_housing is not None:
            complete_stock(self.config.rental_housing, allow_none=False)
        if self.config.labour_assets is not None:
            complete_stock(self.config.labour_assets, allow_none=False)

    def _c0(self) -> float:
        # Include diversified income (if provided) in baseline income
        i_div = self.config.income.i_div or 0.0
        return self.config.income.c_i_ratio * (self.config.income.i_0 + i_div)

    def _c_avg(self) -> float:
        return self.config.income.c_i_ratio * self.config.income.i_avg

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
        General unit recovery time based on consumption losses.

        Defined as the earliest time t where the cumulative recovered
        consumption loss reaches `simulation.recovery_per` of the total
        consumption loss over the simulation horizon.

        Uses the realized consumption time series (including liquidity effects).
        """
        if LossType.CONSUMPTION not in self.time_series.columns:
            return None
        times = self.time_series["time"].to_numpy()
        # Loss rate over time (currency/year): Δc(t)
        losses_t = self.time_series[LossType.CONSUMPTION].to_numpy()
        n = losses_t.size
        if n == 0:
            return None
        # Cumulative total loss via ConsumptionLoss.total() with [t0, t] integration
        loss_model = ConsumptionLoss(
            t=self.t,
            rec_rate=self._rec_rate(),
            v=self.config.housing.v,
            k_str=self.config.housing.k,
            pi=self.config.income.pi,
            liquidity=self._liquidity(),
            extra_losses=self._extra_losses(),
        )
        t0 = float(times[0])
        cum = np.array(
            [float(loss_model.total(rho=0, method="trapezoid", t1=t0, t2=float(t))) for t in times],
            dtype=float,
        )
        total_loss = float(cum[-1])
        if total_loss <= 0:
            return 0.0
        target_fraction = self.config.simulation.recovery_per / 100.0
        target = total_loss * target_fraction
        # If already achieved at t=0
        if cum[0] >= target:
            return 0.0
        # Find first index where cumulative loss reaches/exceeds target
        idxs = np.nonzero(cum >= target)[0]
        if idxs.size == 0:
            # Not reached within simulation horizon
            return None
        i = int(idxs[0])
        if i == 0:
            return float(times[0])
        # Linear interpolation in cumulative space between points i-1 and i
        t0, t1 = float(times[i - 1]), float(times[i])
        c_prev, c_curr = float(cum[i - 1]), float(cum[i])
        if c_curr == c_prev:
            return t1
        s = (target - c_prev) / (c_curr - c_prev)
        if s < 0.0:
            s = 0.0
        elif s > 1.0:
            s = 1.0
        return t0 + s * (t1 - t0)

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
            If an invalid loss type is provided.
        """
        if loss_type == LossType.RECOVERY:
            loss = RecoveryCost(
                self.t,
                self._rec_rate(),
                self.config.housing.v,
                self.config.housing.k,
            )
        elif loss_type == LossType.INCOME:
            loss = IncomeLoss(
                self.t,
                self._rec_rate(),
                self.config.housing.v,
                self.config.housing.k,
                self.config.income.pi,
            )
        elif loss_type == LossType.RENTAL_INCOME:
            if self.config.rental_housing is None:
                return 0.0
            loss = IncomeLoss(
                self.t,
                self._stock_rec_rate(self.config.rental_housing),
                self.config.rental_housing.v,
                self.config.rental_housing.k,
                self.config.income.pi,
            )
        elif loss_type == LossType.LABOUR_INCOME:
            if self.config.labour_assets is None:
                return 0.0
            loss = IncomeLoss(
                self.t,
                self._stock_rec_rate(self.config.labour_assets),
                self.config.labour_assets.v,
                self.config.labour_assets.k,
                self.config.income.pi * self.config.income.c_L,
            )
        elif loss_type == LossType.CONSUMPTION:
            loss = ConsumptionLoss(
                self.t,
                self._rec_rate(),
                self.config.housing.v,
                self.config.housing.k,
                self.config.income.pi,
                liquidity=self._liquidity(),
                extra_losses=self._extra_losses(),
            )
            if self._has_liquidity():
                loss_no_liq = ConsumptionLoss(
                    self.t,
                    self._rec_rate(),
                    self.config.housing.v,
                    self.config.housing.k,
                    self.config.income.pi,
                    liquidity=0.0,
                    extra_losses=self._extra_losses(),
                )
                self.time_series[f"{loss_type} No Liquidity"] = loss_no_liq.losses_t
        elif loss_type == LossType.UTILITY:
            loss = UtilityLoss(
                self.t,
                self._rec_rate(),
                self.config.housing.v,
                self.config.housing.k,
                self.config.income.pi,
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
            - "Equity Weighted Loss": The calculated equity-weighted loss.
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
            v=self.config.housing.v,
            k_str=self.config.housing.k,
            pi=self.config.income.pi,
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
        ew_loss = equity_weight(
            c=self._c0(), c_avg=self._c_avg(), eta=self.config.simulation.eta
        ) * self.config.housing.v * self.config.housing.k

        # Update total losses with additional metrics
        self.total_losses["Wellbeing Loss"] = well_being_loss
        self.total_losses["Asset Loss"] = (
            self.config.housing.v * self.config.housing.k
        )
        self.total_losses["Equity Weighted Loss"] = ew_loss

        # Compute and store general unit recovery time (consumption-based)
        self.unit_recovery_time = self._unit_recovery_time()
        # Expose as primary recovery time attribute for the unit
        self.recovery_time = self.unit_recovery_time

        return self.total_losses

    def plot_loss(
        self, loss_type: LossType, ax: Optional[plt.Axes] = None
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
        if loss_type not in LossType.__members__.values():
            valid_values = [member.value for member in LossType]
            raise ValueError(
                f"Invalid type '{loss_type}'. Must be one of {valid_values}."
            )
        if ax is None:
            ax_given = False
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            ax_given = True
        sns.lineplot(x="time", y=loss_type, data=self.time_series, ax=ax)
        ax.fill_between(
            self.time_series.index,
            0,
            self.time_series[loss_type],
            edgecolor="gray",
            alpha=0.3,
            label=(
                f"Total {loss_type}: {self.total_losses[loss_type]:.2f} "
                f"{self.config.simulation.currency}"
            ),
        )
        ax.set_xlabel("Time after disaster (years)")
        if loss_type != LossType.UTILITY:
            ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, pos: f"{int(x):,}")
            )
            ax.set_ylabel(f"{loss_type} ({self.config.simulation.currency})")
            # Add legend
            ax.legend()
        else:
            ax.set_ylabel(f"{loss_type}")

        if not ax_given:
            return fig

    def plot_consumption(
        self, ax: Optional[plt.Axes] = None, plot_cmin=False
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

        # Prepare component series (use 0 if not present)
        inc = self.time_series[LossType.INCOME]
        recon = self.time_series[LossType.RECOVERY]
        rental = (
            self.time_series[LossType.RENTAL_INCOME]
            if LossType.RENTAL_INCOME in self.time_series.columns
            else 0
        )
        labour = (
            self.time_series[LossType.LABOUR_INCOME]
            if LossType.LABOUR_INCOME in self.time_series.columns
            else 0
        )

        # Colors and labels
        color_income = "brown"
        color_recon = "lightcoral"
        color_rental = "sienna"
        color_labour = "peru"

        # Bottom: Recovery
        label_recon = (
            f"Total {LossType.RECOVERY}: "
            f"{self.total_losses[LossType.RECOVERY]:,.0f} "
            f"{self.config.simulation.currency}"
        )
        ax.fill_between(
            self.time_series["time"],
            self._c0() - inc - rental - labour - recon,
            self._c0() - inc - rental - labour,
            facecolor=color_recon,
            alpha=0.6,
            label=label_recon,
        )

        # Above: Income
        label_income = (
            f"Total {LossType.INCOME}: {self.total_losses[LossType.INCOME]:,.0f} "
            f"{self.config.simulation.currency}"
        )
        ax.fill_between(
            self.time_series["time"],
            self._c0() - inc - rental - labour,
            self._c0() - rental - labour,
            facecolor=color_income,
            alpha=0.6,
            label=label_income,
        )

        # Above: Rental Income (if any)
        if LossType.RENTAL_INCOME in self.time_series.columns:
            label_rental = (
                f"Total {LossType.RENTAL_INCOME}: "
                f"{self.total_losses[LossType.RENTAL_INCOME]:,.0f} "
                f"{self.config.simulation.currency}"
            )
            ax.fill_between(
                self.time_series["time"],
                self._c0() - rental - labour,
                self._c0() - labour,
                facecolor=color_rental,
                alpha=0.6,
                label=label_rental,
            )

        # Top: Labour Income (if any)
        if LossType.LABOUR_INCOME in self.time_series.columns:
            label_labour = (
                f"Total {LossType.LABOUR_INCOME}: "
                f"{self.total_losses[LossType.LABOUR_INCOME]:,.0f} "
                f"{self.config.simulation.currency}"
            )
            ax.fill_between(
                self.time_series["time"],
                self._c0() - labour,
                self._c0(),
                facecolor=color_labour,
                alpha=0.6,
                label=label_labour,
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
                self._c0()
                - self.time_series[f"{LossType.CONSUMPTION} No Liquidity"],
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
                label=f"Unit recovery time: {urt:.2f} years",
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

        # Add text annotation for unit recovery time
        y_point = self._c0() - self.time_series[LossType.CONSUMPTION].mean()
        if urt is not None:
            ax.text(
                urt + 0.1,
                y_point,
                f"Unit recovery time: {urt:.2f} years",
                verticalalignment="center",
                horizontalalignment="left",
                color="black",
                fontsize=10,
                alpha=0.7,
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
        ax.set_ylabel(
            f"Consumption rate ({self.config.simulation.currency}/year)"
        )
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
        None
        """
        # Check if the maximum recovery time is provided, else use the maximum time
        if rec_time_max is None:
            rec_time_max = self.t[-1]

        # Create array of lambda values to check
        times = np.linspace(rec_time_min, rec_time_max, no_steps)
        lambdas = recovery_rate(
            times, rebuilt_per=self.config.simulation.recovery_per
        )

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
                    v=self.config.housing.v,
                    k_str=self.config.housing.k,
                ).total(rho=0, method=method)
            )
            income_losses.append(
                IncomeLoss(
                    t=self.t,
                    rec_rate=lmbd,
                    v=self.config.housing.v,
                    k_str=self.config.housing.k,
                    pi=self.config.income.pi,
                ).total(rho=0, method=method)
            )
            consumption_losses.append(
                ConsumptionLoss(
                    t=self.t,
                    rec_rate=lmbd,
                    v=self.config.housing.v,
                    k_str=self.config.housing.k,
                    pi=self.config.income.pi,
                    liquidity=self._liquidity(),
                    extra_losses=self._extra_losses(),
                ).total(rho=0, method=method)
            )
            utility_losses.append(
                UtilityLoss(
                    t=self.t,
                    rec_rate=lmbd,
                    v=self.config.housing.v,
                    k_str=self.config.housing.k,
                    pi=self.config.income.pi,
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
            v=self.config.housing.v,
            k_str=self.config.housing.k,
            c0=self._c0(),
            pi=self.config.income.pi,
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

        optimal_lambda = opt["l_opt"]
        self.lambda_opt = opt

        # Save optimization dataframe
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

        # Persist optimal parameters back into config for downstream use
        # Persist (config schema uses recovery_*; keep fields but rename semantics)
        self.config.housing.recovery_rate = optimal_lambda
        self.config.housing.recovery_time = recovery_time(
            rate=optimal_lambda, rebuilt_per=self.config.simulation.recovery_per
        )

        self.l_opt = df

        return optimal_lambda

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
        axs[0].set_ylabel(
            f"Total Loss ({self.config.simulation.currency})"
        )
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
