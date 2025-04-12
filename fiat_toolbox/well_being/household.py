from enum import Enum
from typing import Literal, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

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


class LossType(str, Enum):
    RECONSTRUCTION = "Reconstruction Costs"
    INCOME = "Income Loss"
    CONSUMPTION = "Consumption Loss"
    UTILITY = "Utility Loss"

    def __str__(self):
        return self.value


class Household:
    def __init__(
        self,
        v: float,
        k_str: float,
        c0: float,
        c_avg: float,
        l: Optional[float] = None,
        pi: Optional[float] = 0.15,
        eta: Optional[float] = 1.5,
        rho: Optional[float] = 0.06,
        t_max: Optional[float] = 10,
        dt: Optional[float] = 1 / 52,
        currency: Optional[str] = "$",
    ) -> None:
        """
        Initialize the WellBeing class with the given parameters.

        Parameters
        ----------
        v : float
            The loss ratio, which is reconstruction cost divided by the total building value.
        k_str : float
            The total building value.
        c0 : float
            Initial consumption level.
        c_avg : float
            Average consumption level.
        l : float, optional
            The rate of recovery (per unit time). Default is None.
        pi : float, optional
            Average productivity of capital. Default is 0.15.
        eta : float, optional
            Elasticity of marginal utility of consumption. Default is 1.5.
        rho : float, optional
            Discount rate. Default is 0.06.
        t_max : float, optional
            Maximum time for the simulation. Default is 10.
        dt : float, optional
            Time step for the simulation. Default is 1/52 (weekly).
        currency : str, optional
            Currency symbol for the plots. Default is "$".

        Returns
        -------
        None
        """
        self.v = v
        self.k_str = k_str
        self.c0 = c0
        self.c_avg = c_avg
        self.pi = pi
        self.eta = eta
        self.rho = rho
        self.t_max = t_max
        self.dt = self.t_max / (int(self.t_max / dt) + 1)
        self.t = np.linspace(0, self.t_max, int(self.t_max / self.dt) + 1)
        self.currency = currency
        self.l = l
        if l is not None:
            self.recovery_time = recovery_time(rate=self.l, rebuilt_per=95)
        self.time_series = pd.DataFrame({"time": self.t})
        self.total_losses = pd.Series()

    def __repr__(self):
        return (
            f"Household(\n"
            f"  v = {self.v} (loss ratio),\n"
            f"  k_str = {self.k_str} (total building value),\n"
            f"  c0 = {self.c0} (initial consumption level),\n"
            f"  c_avg = {self.c_avg} (average consumption level),\n"
            f"  l = {self.l} (recovery rate),\n"
            f"  pi = {self.pi} (average productivity of capital),\n"
            f"  eta = {self.eta} (elasticity of marginal utility of consumption),\n"
            f"  rho = {self.rho} (discount rate),\n"
            f"  t_max = {self.t_max} (maximum simulation time),\n"
            f"  dt = {self.dt} (time step),\n"
            f"  currency = {self.currency} (currency symbol)\n"
            f")"
        )

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
            loss = ReconstructionCost(self.t, self.l, self.v, self.k_str)
        elif loss_type == LossType.INCOME:
            loss = IncomeLoss(self.t, self.l, self.v, self.k_str, self.pi)
        elif loss_type == LossType.CONSUMPTION:
            loss = ConsumptionLoss(self.t, self.l, self.v, self.k_str, self.pi)
        elif loss_type == LossType.UTILITY:
            loss = UtilityLoss(
                self.t, self.l, self.v, self.k_str, self.pi, self.c0, self.eta
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
            l=self.l,
            v=self.v,
            k_str=self.k_str,
            pi=self.pi,
            c0=self.c0,
            eta=self.eta,
        )
        du_dis = ut_t.total(rho=self.rho, method=method)
        well_being_loss = wellbeing_loss(du=du_dis, c_avg=self.c_avg, eta=self.eta)
        # Calculate equity weighted loss
        ew_loss = (
            equity_weight(c=self.c0, c_avg=self.c_avg, eta=self.eta)
            * self.v
            * self.k_str
        )

        # Update total losses with additional metrics
        self.total_losses["Wellbeing Loss"] = well_being_loss
        self.total_losses["Asset Loss"] = self.v * self.k_str
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
        sns.lineplot(x="time", y=loss_type, data=self.time_series, ax=ax)
        ax.fill_between(
            self.time_series.index,
            0,
            self.time_series[loss_type],
            edgecolor="gray",
            alpha=0.3,
            label=f"Total {loss_type}: {self.total_losses[loss_type]:.2f} {self.currency}",
        )
        ax.set_xlabel("Time after disaster (years)")
        if loss_type != LossType.UTILITY:
            ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, pos: f"{int(x):,}")
            )
            ax.set_ylabel(f"{loss_type} ({self.currency})")
            # Add legend
            ax.legend()
        else:
            ax.set_ylabel(f"{loss_type}")

        if not ax_given:
            return fig

    def plot_consumption(self, ax: Optional[plt.Axes] = None) -> Optional[plt.Figure]:
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
        label1 = f"Total {LossType.INCOME}: {self.total_losses[LossType.INCOME]:,.0f} {self.currency}"
        ax.fill_between(
            self.time_series["time"],
            self.c0 - self.time_series[LossType.INCOME],
            self.c0,
            color=color1,
            alpha=0.6,
            label=label1,
        )

        # Plot reconstruction costs
        color2 = "lightcoral"
        label2 = f"Total {LossType.RECONSTRUCTION}: {self.total_losses[LossType.RECONSTRUCTION]:,.0f} {self.currency}"
        ax.fill_between(
            self.time_series["time"],
            self.c0 - self.time_series[LossType.CONSUMPTION],
            self.c0 - self.time_series[LossType.INCOME],
            color=color2,
            alpha=0.6,
            label=label2,
        )

        # Plot consumption losses with a dashed line and expand to the left by 4 months
        expanded_time = np.insert(self.time_series["time"], 0, [-1, -0.001])
        expanded_consumption_losses = np.insert(
            self.c0 - self.time_series[LossType.CONSUMPTION], 0, [self.c0, self.c0]
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
            x=self.recovery_time,
            color="black",
            linestyle="-",
            alpha=0.3,
            label=f"Reconstruction rate: {self.l:.2f}",
        )
        # Add text annotation for recovery time
        y_point = self.c0 - self.time_series[LossType.CONSUMPTION].mean()
        ax.text(
            self.recovery_time + 0.1,
            y_point,
            f"Recovery time: {self.recovery_time:.2f} years",
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
            self.c0 + 0.1,
            lightning_icon,
            fontsize=50,
            color="red",
            verticalalignment="bottom",
            horizontalalignment="center",
        )

        # Plot consumption losses
        ax.set_xlabel("Time after disaster (years)")
        ax.set_ylabel(f"Consumption over time ({self.currency})")
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
        method="trapezoid",
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

        Returns
        -------
        None
        """
        # Check if the maximum recovery time is provided, else use the maximum time
        if rec_time_max is None:
            rec_time_max = self.t[-1]
        # Create array of lambda values to check
        times = np.linspace(rec_time_min, rec_time_max, no_steps)
        lambdas = recovery_rate(times, rebuilt_per=95)
        # Calculate losses for each lambda value
        reconstruction_costs = ReconstructionCost(
            t=self.t, l=lambdas, v=self.v, k_str=self.k_str
        ).total(rho=0, method=method)
        income_losses = IncomeLoss(
            t=self.t, l=lambdas, v=self.v, k_str=self.k_str, pi=self.pi
        ).total(rho=0, method=method)
        consumption_losses = ConsumptionLoss(
            t=self.t, l=lambdas, v=self.v, k_str=self.k_str, pi=self.pi
        ).total(rho=0, method=method)
        utility_losses = UtilityLoss(
            t=self.t,
            l=lambdas,
            v=self.v,
            k_str=self.k_str,
            pi=self.pi,
            c0=self.c0,
            eta=self.eta,
        ).total(rho=0, method=method)
        optimal_lambda = opt_lambda(
            v=self.v,
            k_str=self.k_str,
            c0=self.c0,
            pi=self.pi,
            eta=self.eta,
            l_min=lambdas.min(),
            l_max=lambdas.max(),
            t_max=self.t_max,
            times=self.t,
            method=method,
        )

        # Save optimization dataframe
        df = pd.DataFrame(
            {
                "lambda": lambdas,
                "recovery_time": recovery_time(rate=lambdas, rebuilt_per=95),
                LossType.RECONSTRUCTION: reconstruction_costs,
                LossType.INCOME: income_losses,
                LossType.CONSUMPTION: consumption_losses,
                LossType.UTILITY: utility_losses,
            }
        )

        # Save lambda value
        self.l = optimal_lambda

        # Calculate the recovery time for the optimal lambda
        self.recovery_time = recovery_time(rate=optimal_lambda, rebuilt_per=95)

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
            val = self.l
            leg = "Reconstruction-rate λ"
        elif x_type == "time":
            x = self.l_opt["recovery_time"]
            val = self.recovery_time
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
        axs[0].set_ylabel(f"Total Loss ({self.currency})")
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
