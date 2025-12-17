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
    ReconstructionCost,
    UtilityLoss,
    equity_weight,
    opt_lambda,
    recovery_rate,
    recovery_time,
    wellbeing_loss,
)


# TODO Make class a pydantic model
class LossType(str, Enum):
    RECONSTRUCTION = "Reconstruction Costs"
    INCOME = "Income Loss"

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
    def _rec_rate(self) -> Optional[float]:
        if self.config.housing.recovery_rate is not None:
            return self.config.housing.recovery_rate
        if self.config.housing.recovery_time is not None:
            return recovery_rate(
                self.config.housing.recovery_time,
                rebuilt_per=self.config.simulation.recovery_per,
            )
        return None

    def _recovery_time(self) -> Optional[float]:
        if self.config.housing.recovery_time is not None:
            return self.config.housing.recovery_time
        if self.config.housing.recovery_rate is not None:
            return recovery_time(
                rate=self.config.housing.recovery_rate,
                rebuilt_per=self.config.simulation.recovery_per,
            )
        return None

    def _c0(self) -> float:
        return self.config.income.c_i_ratio * self.config.income.i_0

    def _c_avg(self) -> float:
        return self.config.income.c_i_ratio * self.config.income.i_avg

    def _has_liquidity(self) -> bool:
        return self._liquidity() != 0

    def _liquidity(self) -> float:
        liq = self.config.liquidity
        if not liq:
            return 0.0
        return (liq.savings or 0.0) + (liq.insurance or 0.0) + (liq.support or 0.0)

    def calc_loss(self, loss_type: LossType, method: str = "trapezoid") -> float:
        """
        Calculate the loss based on the specified loss type and method.

        Parameters
        ----------
        loss_type : LossType
            The type of loss to calculate. Must be one of the following:
            - LossType.RECONSTRUCTION: Calculates reconstruction cost.
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
        if loss_type == LossType.RECONSTRUCTION:
            loss = ReconstructionCost(
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
        elif loss_type == LossType.CONSUMPTION:
            loss = ConsumptionLoss(
                self.t,
                self._rec_rate(),
                self.config.housing.v,
                self.config.housing.k,
                self.config.income.pi,
                liquidity=self._liquidity(),
            )
            if self._has_liquidity():
                loss_no_liq = ConsumptionLoss(
                    self.t,
                    self._rec_rate(),
                    self.config.housing.v,
                    self.config.housing.k,
                    self.config.income.pi,
                    liquidity=0.0,
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
            - Additional keys corresponding to each `LossType` (e.g., "Reconstruction Costs", "Income Loss").

        Notes
        -----
        - The `LossType` enumeration is iterated to calculate individual loss types.
        - The `UtilityLoss` class is used to compute the utility loss.
        - The `wellbeing_loss` and `equity_weight` functions are used to compute the respective metrics.
        - The results are stored in the `time_series` DataFrame and `total_losses` Series attributes.
        """
        # Calculate losses for each loss type
        for loss_type in LossType:
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
        Plot the consumption losses over time, stacking income losses and reconstruction costs with different hatches and colors.

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

        # Plot income losses
        color1 = "brown"
        label1 = (
            f"Total {LossType.INCOME}: {self.total_losses[LossType.INCOME]:,.0f} "
            f"{self.config.simulation.currency}"
        )
        ax.fill_between(
            self.time_series["time"],
            self._c0() - self.time_series[LossType.INCOME],
            self._c0(),
            color=color1,
            alpha=0.6,
            label=label1,
        )

        # Plot reconstruction costs
        color2 = "lightcoral"
        label2 = (
            f"Total {LossType.RECONSTRUCTION}: "
            f"{self.total_losses[LossType.RECONSTRUCTION]:,.0f} "
            f"{self.config.simulation.currency}"
        )
        ax.fill_between(
            self.time_series["time"],
            self._c0()
            - self.time_series[LossType.INCOME]
            - self.time_series[LossType.RECONSTRUCTION],
            self._c0() - self.time_series[LossType.INCOME],
            facecolor=color2,
            alpha=0.6,
            label=label2,
            # linewidth=0.0
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

        # Plot recovery time
        ax.axvline(
            x=self._recovery_time(),
            color="black",
            linestyle="-",
            alpha=0.3,
            label=f"Reconstruction rate: {self._rec_rate():.2f}",
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

        # Add text annotation for recovery time
        y_point = self._c0() - self.time_series[LossType.CONSUMPTION].mean()
        ax.text(
            self._recovery_time() + 0.1,
            y_point,
            f"Recovery time: {self._recovery_time():.2f} years",
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
        eps_rel: float = 0.01,
    ) -> None:
        """
        Optimize the reconstruction rate (lambda) to minimize the total well-being loss.

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
        reconstruction_costs = []
        income_losses = []
        consumption_losses = []
        utility_losses = []

        # Iterate through each lambda value and calculate losses
        for lmbd in lambdas:
            reconstruction_costs.append(
                ReconstructionCost(
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
                ).total(rho=0, method=method)
            )

        # Convert lists to numpy arrays for further processing
        reconstruction_costs = np.array(reconstruction_costs)
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
        )

        optimal_lambda = opt["l_opt"]
        self.lambda_opt = opt

        # Save optimization dataframe
        df = pd.DataFrame(
            {
                "lambda": lambdas,
                "recovery_time": recovery_time(
                    rate=lambdas, rebuilt_per=self.config.simulation.recovery_per
                ),
                LossType.RECONSTRUCTION: reconstruction_costs,
                LossType.INCOME: income_losses,
                LossType.CONSUMPTION: consumption_losses,
                LossType.UTILITY: utility_losses,
            }
        )

        # Persist optimal parameters back into config for downstream use
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
            val = self._rec_rate()
            val_min = self.lambda_opt["l_opt_min"]
            leg = "Reconstruction-rate λ"
        elif x_type == "time":
            x = self.l_opt["recovery_time"]
            val = self._recovery_time()
            val_min = recovery_time(
                rate=self.lambda_opt["l_opt_min"],
                rebuilt_per=self.config.simulation.recovery_per,
            )
            leg = "Recovery time (years)"
            # axs[0].set_xscale('log')
        axs[1].set_xlabel(leg)
        # Make line plots for consumption losses
        sns.lineplot(
            x=x,
            y=self.l_opt[LossType.RECONSTRUCTION],
            color="green",
            ax=axs[0],
            label=LossType.RECONSTRUCTION,
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
