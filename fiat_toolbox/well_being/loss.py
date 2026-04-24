import warnings
from enum import Enum
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    RECOVERY_COST = "Recovery Costs"
    OWNER_HOUSING_LOSS = "Loss of Housing Services"
    RENTAL_HOUSING_LOSS = "Loss of Housing Services (Rental)"
    LABOUR_INCOME_LOSS = "Labour Income Loss"

    CONSUMPTION_LOSS = "Consumption Loss"
    UTILITY_LOSS = "Utility Loss"

    def __str__(self):
        return self.value


class CapitalStock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    k: float = Field(..., ge=0, description="Value of the capital stock (≥ 0)")
    v: float = Field(
        ...,
        ge=0,
        le=1,
        description="Loss ratio for the capital stock (0 ≤ v ≤ 1)",
    )
    recovery_time: Optional[float] = Field(
        None, gt=0, description="Recovery time for the capital stock (> 0)"
    )
    recovery_rate: Optional[float] = Field(
        None, gt=0, description="Recovery rate for the capital stock (> 0)"
    )
    pi: float = Field(
        0.15,
        gt=0,
        description=(
            "Productivity of capital for this stock (> 0). Used wherever the "
            "stock contributes an income-loss stream"
        ),
    )
    recovery_label: Optional[str] = Field(
        None,
        description=(
            "Optional plot-legend text for the RecoveryCost stream this stock "
            "drives. When None, CommunityUnit falls back "
            "to the default LossType.RECOVERY_COST display string. Plot-only: does "
            "not affect total_losses / time_series dict keys."
        ),
    )
    income_label: Optional[str] = Field(
        None,
        description=(
            "Optional plot-legend text for the IncomeLoss stream this stock "
            "drives. When None, CommunityUnit falls back to the default "
            "LossType display string (optionally suffixed with a labour "
            "asset's dict key). Plot-only: does not affect total_losses / "
            "time_series dict keys."
        ),
    )


class IncomeStream(BaseModel):
    """Income flow specified directly (not decomposed into π·k).

    Use in `rental_housing` and `labour_assets` when you already have the
    baseline income figure from survey, tax, or Penn-World-Table data and
    the π / k split would be artificial. The household does not bear
    recovery cost on this stream — only income loss `income · v · exp(-λt)`
    is generated, so `IncomeStream` is never valid for `owner_housing`
    (which drives RecoveryCost and needs `k` directly).

    `income=0` is accepted and contributes zero to all losses — the stream
    is effectively omitted from the calculation (useful for placeholder
    slots or programmatically generated configs where a quintile share
    collapses to zero).
    """

    model_config = ConfigDict(extra="forbid")

    income: float = Field(
        ...,
        ge=0,
        description=(
            "Baseline income flow (per year) from this stream, pre-computed. "
            "Equivalent to π·k under a CapitalStock specification. Must be ≥ 0; "
            "income=0 is accepted and contributes zero to all losses (the stream "
            "is effectively omitted from the calculation)."
        ),
    )
    v: float = Field(
        ...,
        ge=0,
        le=1,
        description="Fraction of the income flow lost at t=0 (0 ≤ v ≤ 1)",
    )
    recovery_time: Optional[float] = Field(
        None,
        gt=0,
        description=(
            "Recovery time of the underlying productive capacity (years, > 0). "
            "Exactly one of recovery_time / recovery_rate must be provided."
        ),
    )
    recovery_rate: Optional[float] = Field(
        None,
        gt=0,
        description=(
            "Recovery rate λ (per year, > 0). Exactly one of recovery_time / "
            "recovery_rate must be provided."
        ),
    )
    income_label: Optional[str] = Field(
        None,
        description=(
            "Optional plot-legend text for the IncomeLoss stream. When None, "
            "CommunityUnit falls back to the default LossType display string "
            "(optionally suffixed with a labour asset's dict key). Plot-only."
        ),
    )


class Liquidity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    savings: float = Field(0.0, ge=0, description="Household savings (≥ 0)")
    insurance: float = Field(0.0, ge=0, description="Insurance payout (≥ 0)")
    support: float = Field(0.0, ge=0, description="External support (≥ 0)")


class IncomeConfig(BaseModel):
    """Baseline income and committed-payment configuration.

    `c0 = asset_income + (i_div or 0) − (payments or 0)`, where
    `asset_income` is `Σ π·k` derived from the configured stocks, or
    `i_0` when supplied as an override. `i_div` and `payments` are
    always applied separately on top of / underneath `asset_income`;
    they are never meant to be folded into `i_0`.
    """

    model_config = ConfigDict(extra="forbid")

    i_0: Optional[float] = Field(
        None,
        ge=0,
        description=(
            "Optional explicit override for the asset-income component only "
            "(≥ 0; equivalent to Σ π·k across stocks). When omitted (the "
            "default), this term is derived from the configured stocks. Do "
            "NOT include i_div or payments here — those are applied "
            "separately in c0. A UserWarning is emitted at CommunityUnit "
            "construction if i_0 differs from Σ π·k."
        ),
    )
    i_avg: float = Field(..., gt=0, description="Average income rate per year (> 0)")
    i_div: Optional[float] = Field(
        None,
        ge=0,
        description=(
            "Residual non-asset-based income (≥ 0; remittances, transfers, "
            "pensions, …) ADDED to c0 on top of the asset-income baseline. "
            "Always additive to i_0 / Σ π·k; never part of i_0."
        ),
    )
    payments: Optional[float] = Field(
        None,
        ge=0,
        description=(
            "Committed household outflows (≥ 0; mortgage, taxes, insurance "
            "premiums, …) SUBTRACTED from c0. Treated as fixed obligations "
            "that continue through recovery — i.e. they shift the pre-shock "
            "baseline down but leave the loss curve Δc(t) unchanged. Does "
            "not affect i_avg / c_avg or the i_0 consistency check. A "
            "UserWarning is emitted at CommunityUnit construction if "
            "payments would drive c0 ≤ 0."
        ),
    )


class SimulationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    eta: float = Field(
        1.5, gt=0, description="Elasticity of marginal utility of consumption (> 0)"
    )
    rho: float = Field(0.06, ge=0, description="Discount rate (≥ 0)")
    t_max: float = Field(10, gt=0, description="Maximum simulation time (> 0)")
    dt: float = Field(1 / 52, gt=0, description="Time step (> 0)")
    currency: str = Field("$", description="Currency symbol")
    currency_decimals: int = Field(
        0,
        ge=0,
        description=(
            "Number of decimal places used when formatting currency values in "
            "plots (total labels, y-axis tick formatters). Default 0 preserves "
            "historical 'round to whole units' behaviour; raise (e.g. 2) when "
            "using coarse-grained currency units such as millions IDR or "
            "billions JPY where sub-unit precision is meaningful."
        ),
    )
    c_min: float = Field(
        0.0, ge=0, description="Minimum consumption rate per year (≥ 0)"
    )
    recovery_per: float = Field(
        95.0,
        ge=0,
        lt=100,
        description=(
            "Percentage of asset rebuilt to consider as recovered "
            "(0 ≤ recovery_per < 100)"
        ),
    )


class WellBeingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    owner_housing: CapitalStock = Field(
        ...,
        description=(
            "Household's own housing stock. Drives both a RecoveryCost stream "
            "(λ·v·k for reconstruction) and an IncomeLoss stream (π·v·k for "
            "lost housing services). Must be CapitalStock — recovery cost "
            "needs k directly."
        ),
    )
    rental_housing: Optional[Union[CapitalStock, IncomeStream]] = Field(
        None,
        description=(
            "Rental housing stream. Models the landlord's structural capital "
            "affecting tenant housing services (paper's k^rent_str / "
            "Δi^rent(t) stream): the household, as tenant, loses housing "
            "services while the landlord's damaged building is reconstructed. "
            "Household does NOT bear recovery cost (landlord assumed outside "
            "the study area). Only an IncomeLoss stream is produced. "
            "Supply as CapitalStock (π·k decomposition) or IncomeStream "
            "(income directly)."
        ),
    )
    labour_assets: Optional[Dict[str, Union[CapitalStock, IncomeStream]]] = Field(
        None,
        description=(
            "Productive assets that generate household labour income (paper's "
            "k^pub, k^firm). Each entry drives an IncomeLoss stream; no "
            "household-borne recovery cost. Supply as CapitalStock or "
            "IncomeStream per entry."
        ),
    )
    income: IncomeConfig
    liquidity: Optional[Liquidity] = Liquidity()
    simulation: Optional[SimulationConfig] = SimulationConfig()

    def normalize_recovery_params(self) -> None:
        """Validate and fill `recovery_time` / `recovery_rate` across all stocks.

        Rules enforced for each configured stock (owner_housing,
        rental_housing, every labour_assets entry):

        - Both `recovery_time` and `recovery_rate` set → `ValueError`.
        - Exactly one set → fill the other via the inverse function using
          `simulation.recovery_per`.
        - Neither set → `ValueError`, **except** for `owner_housing`, which
          may be left fully unset so that `CommunityUnit.opt_lambda` can fill
          it later.

        Legacy shim: if `labour_assets` is passed as a single
        `CapitalStock` / `IncomeStream` (not a dict), it's wrapped into
        `{"labour": stock}`.

        Runs automatically via `@model_validator(mode='after')` at
        construction; `CommunityUnit.__init__` re-runs it to cover
        post-construction mutations of nested stocks.
        """
        rec_per = self.simulation.recovery_per

        def complete_stock(stock, allow_none: bool) -> None:
            if stock is None:
                if allow_none:
                    return
                raise ValueError("Missing required stock configuration")
            has_rate = stock.recovery_rate is not None
            has_time = stock.recovery_time is not None
            if has_rate and has_time:
                # Both set — verify consistency. This state legitimately arises
                # when `normalize_recovery_params` is called a second time
                # after a prior fill (e.g., CommunityUnit.__init__ re-runs it
                # to cover post-construction mutations). If the values agree
                # within tolerance, pass through as already normalised; if they
                # disagree, the user supplied contradictory numbers — raise.
                expected_time = recovery_time(
                    rate=stock.recovery_rate, rebuilt_per=rec_per
                )
                tol = 1e-6 * max(1.0, abs(expected_time))
                if abs(stock.recovery_time - expected_time) > tol:
                    raise ValueError(
                        f"recovery_rate={stock.recovery_rate!r} and "
                        f"recovery_time={stock.recovery_time!r} are both set "
                        f"but inconsistent (expected recovery_time ≈ "
                        f"{expected_time:g} given recovery_per={rec_per}). "
                        "Provide only one of them."
                    )
                return
            if not has_rate and not has_time:
                if allow_none:
                    # owner_housing may be left unset and filled by opt_lambda
                    return
                raise ValueError(
                    "For rental_housing and labour_assets, provide recovery_rate or recovery_time"
                )
            if has_rate and not has_time:
                stock.recovery_time = recovery_time(
                    rate=stock.recovery_rate, rebuilt_per=rec_per
                )
            if has_time and not has_rate:
                stock.recovery_rate = recovery_rate(
                    time=stock.recovery_time, rebuilt_per=rec_per
                )

        complete_stock(self.owner_housing, allow_none=True)
        if self.rental_housing is not None:
            complete_stock(self.rental_housing, allow_none=False)
        if self.labour_assets is not None:
            if isinstance(self.labour_assets, dict):
                for name, stock in self.labour_assets.items():
                    try:
                        complete_stock(stock, allow_none=False)
                    except ValueError as e:
                        raise ValueError(f"Invalid labour_assets['{name}']: {e}")
            else:
                # Backward compat: single CapitalStock / IncomeStream → dict
                stock = self.labour_assets  # type: ignore[assignment]
                try:
                    complete_stock(stock, allow_none=False)
                except ValueError as e:
                    raise ValueError(f"Invalid labour_assets: {e}")
                self.labour_assets = {"labour": stock}  # type: ignore[assignment]

    @model_validator(mode="after")
    def _run_normalize_recovery_params(self) -> "WellBeingConfig":
        self.normalize_recovery_params()
        return self


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
        # Re-normalize recovery params in case nested stocks were mutated
        # after config construction; the @model_validator on WellBeingConfig
        # already ran once at construction time.
        self.config.normalize_recovery_params()
        # Warn (once per instance) if an explicit i_0 override disagrees with
        # the stock-derived baseline Σ π·k; this is otherwise silent.
        self._check_i0_consistency()
        # Warn (once per instance) if committed payments would drive c0 ≤ 0.
        self._check_payments_feasibility()

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
        """Productivity of capital for this stock (see CapitalStock.pi).

        Defined for CapitalStock only. IncomeStream carries an explicit
        income flow instead — callers should go through `_baseline_income`,
        which routes correctly for either class.
        """
        if isinstance(stock, IncomeStream):
            raise TypeError(
                "_stock_pi is only defined for CapitalStock. Use "
                "_baseline_income(stock) to get the baseline income flow "
                "for either CapitalStock or IncomeStream."
            )
        return float(stock.pi)

    def _baseline_income(self, stock) -> float:
        """Baseline pre-shock income from `stock` (per year).

        - `CapitalStock` → `π · k`
        - `IncomeStream` → `income`

        Every helper that previously wrote `pi * k` should route through
        here so both stock types work transparently.
        """
        if isinstance(stock, IncomeStream):
            return float(stock.income)
        return float(self._stock_pi(stock) * stock.k)

    def _display_label(
        self,
        loss_type: "LossType",
        stock=None,
        fallback_suffix: Optional[str] = None,
    ) -> str:
        """Resolve the legend/display string for a loss stream.

        - If `stock` is provided and carries a matching `*_label` override
          (`recovery_label` for `RECOVERY`, `income_label` otherwise), use it.
        - Otherwise use the `LossType` default, optionally suffixed with
          `fallback_suffix` in parentheses (used for per-labour-asset names).

        Plot-only: does not affect `total_losses` / `time_series` dict keys.
        """
        if stock is not None:
            attr = (
                "recovery_label"
                if loss_type is LossType.RECOVERY_COST
                else "income_label"
            )
            override = getattr(stock, attr, None)
            if override:
                return str(override)
        base = loss_type.value
        return f"{base} ({fallback_suffix})" if fallback_suffix else base

    def _label_for(self, loss_type) -> str:
        """Resolve a display label for a loss stream (enum OR column key).

        Plot-only. Dispatches by `loss_type`:
        - `LossType.RECOVERY_COST` / `LossType.OWNER_HOUSING_LOSS` → owner-housing label override.
        - `LossType.RENTAL_HOUSING_LOSS` → rental-housing label override.
        - `LossType.LABOUR_INCOME_LOSS` (aggregate), `CONSUMPTION`, `UTILITY` →
          enum default (no per-stock override applies).
        - Per-asset labour column key like `"Labour Income Loss (shop)"` →
          labour_assets["shop"] label override, falling back to the current
          `"{default} ({name})"` string.
        - Any other string → returned as-is.
        """
        if isinstance(loss_type, LossType):
            if (
                loss_type is LossType.RECOVERY_COST
                or loss_type is LossType.OWNER_HOUSING_LOSS
            ):
                return self._display_label(loss_type, stock=self.config.owner_housing)
            if loss_type is LossType.RENTAL_HOUSING_LOSS:
                return self._display_label(loss_type, stock=self.config.rental_housing)
            return loss_type.value
        s = str(loss_type)
        prefix = f"{LossType.LABOUR_INCOME_LOSS.value} ("
        if s.startswith(prefix) and s.endswith(")"):
            name = s[len(prefix) : -1]
            stock = None
            if self.config.labour_assets:
                stock = self.config.labour_assets.get(name)
            return self._display_label(
                LossType.LABOUR_INCOME_LOSS, stock=stock, fallback_suffix=name
            )
        return s

    def _income_loss_for_stock(self, stock, t, rec_rate: float) -> "IncomeLoss":
        """Construct an IncomeLoss primitive abstracted over CapitalStock / IncomeStream.

        `methods.IncomeLoss` computes `pi · v · k_str · exp(-λt)`. We pass
        `(k_str=baseline_income, pi=1.0)` so the product collapses to
        `baseline_income · v · exp(-λt)` — the form shared by both stock
        types. Keeps methods.py's primitive untouched.
        """
        baseline = self._baseline_income(stock)
        return IncomeLoss(t, rec_rate, stock.v, k_str=baseline, pi=1.0)

    def _extra_losses(self):
        extra = []
        # Rental housing as a single optional stock (CapitalStock or IncomeStream)
        if self.config.rental_housing is not None:
            rr = self._stock_rec_rate(self.config.rental_housing)
            if rr is None:
                raise ValueError(
                    "rental_housing must define either recovery_rate or recovery_time"
                )
            n0 = (
                self._baseline_income(self.config.rental_housing)
                * self.config.rental_housing.v
            )
            extra.append((n0, rr))

        # Labour assets: dict of named CapitalStock / IncomeStream entries
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
                n0 = self._baseline_income(stock) * stock.v
                extra.append((n0, rr))

        return extra if extra else None

    def _stock_income_sum(self) -> float:
        """Baseline pre-shock income summed across modelled stocks.

        For each stock, contribution is `_baseline_income(stock)` which
        resolves to `π·k` for `CapitalStock` and `income` for
        `IncomeStream`. Iterates over owner housing, optional rental
        housing, and each entry in optional labour_assets (skipping `None`
        labour entries). Mirrors the shape used by
        `_exponential_loss_components_labelled` and `_extra_losses` so new
        stock types land in all three helpers together.
        """
        total = self._baseline_income(self.config.owner_housing)
        if self.config.rental_housing is not None:
            total += self._baseline_income(self.config.rental_housing)
        if self.config.labour_assets:
            for stock in self.config.labour_assets.values():
                if stock is None:
                    continue
                total += self._baseline_income(stock)
        return float(total)

    def _c0(self) -> float:
        # Baseline consumption: asset_income + i_div - payments.
        # Default: asset_income derived from the configured stocks (Σ π·k)
        # so c(t) and the pre-shock c0 live in the same units. If the user
        # supplied an explicit i_0, it replaces the asset_income term only
        # (any mismatch with Σ π·k is flagged via a UserWarning at
        # construction). i_div is always additive; payments are always
        # subtracted.
        i_div = self.config.income.i_div or 0.0
        payments = self.config.income.payments or 0.0
        if self.config.income.i_0 is not None:
            asset_income = float(self.config.income.i_0)
        else:
            asset_income = self._stock_income_sum()
        return asset_income + i_div - payments

    def _check_payments_feasibility(self) -> None:
        """Warn if `payments` drives the pre-shock baseline `c0` to ≤ 0.

        Fires once from `__init__`. A non-positive `c0` breaks CRRA utility
        (log / fractional-power of a non-positive number), so downstream
        calculations will produce NaNs — this surfaces the root cause early.
        """
        payments = self.config.income.payments
        if payments is None or payments <= 0:
            return
        c0 = self._c0()
        if c0 > 0:
            return
        warnings.warn(
            (
                f"IncomeConfig.payments={float(payments):g} drives the "
                f"baseline consumption c0 to {c0:g} (≤ 0). CRRA utility is "
                "undefined at non-positive consumption, so utility and "
                "well-being losses will be NaN. Reduce payments or increase "
                "the asset-income / i_div inputs."
            ),
            UserWarning,
            stacklevel=2,
        )

    def _check_i0_consistency(self) -> None:
        """Warn if an explicit `i_0` disagrees with the stock-derived Σ π·k.

        Fires once from `__init__`. Relative tolerance 1e-6 against
        max(|i_0|, |Σ π·k|, 1) so round-number survey inputs don't trip
        floating-point noise.
        """
        i_0 = self.config.income.i_0
        if i_0 is None:
            return
        stock_sum = self._stock_income_sum()
        scale = max(abs(float(i_0)), abs(stock_sum), 1.0)
        diff = float(i_0) - stock_sum
        if abs(diff) <= 1e-6 * scale:
            return
        rel = diff / scale
        warnings.warn(
            (
                f"IncomeConfig.i_0={float(i_0):g} overrides the stock-derived "
                f"baseline Σ π·k={stock_sum:g} (diff={diff:+.3g}, "
                f"rel={rel:.2%}). The override is in effect. Omit i_0 to use "
                "the stock-consistent value."
            ),
            UserWarning,
            stacklevel=2,
        )

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

    def _liquidity_depleted(self) -> float:
        """
        Amount of liquidity stock actually drawn down over the recovery period.
        """
        S = self._liquidity()
        if S <= 0:
            return 0.0
        rr = self._rec_rate()
        if rr is None or rr <= 0:
            return 0.0
        owner_pi = self._stock_pi(self.config.owner_housing)
        v = self.config.owner_housing.v
        k = self.config.owner_housing.k
        lifetime = ((owner_pi + rr) * v * k) / rr
        for n0, lam in self._extra_losses() or []:
            if lam > 0:
                lifetime += n0 / lam
        return float(min(S, lifetime))

    def _loss_types_for_run(self):
        types = [
            LossType.RECOVERY_COST,
            LossType.OWNER_HOUSING_LOSS,
        ]
        if self.config.rental_housing is not None:
            types.append(LossType.RENTAL_HOUSING_LOSS)
        if self.config.labour_assets is not None:
            types.append(LossType.LABOUR_INCOME_LOSS)
        types.extend([LossType.CONSUMPTION_LOSS, LossType.UTILITY_LOSS])
        return types

    def _unit_recovery_time(self) -> Optional[float]:
        """
        Uses the optimized *household* reconstruction rate alone, since only
        λ_h is a decision variable in the welfare optimization; rental and
        labour asset rates are exogenous (literature-based) and would conflate
        the decision axis with external constraints if mixed in.

        Returns None if the owner reconstruction rate is not set.
        For the aggregated multi-mode quantity (the pre-fix behaviour),
        call `composite_recovery_time`.
        """
        rr = self._rec_rate()
        if rr is None or rr <= 0:
            return None
        return float(recovery_time(rr, rebuilt_per=self.config.simulation.recovery_per))

    def composite_recovery_time(self) -> Optional[float]:
        """
        Aggregate recovery time across owner + rental + labour modes.

        it solves
            Σᵢ Cᵢ · exp(−λᵢ T) = r · Σᵢ Cᵢ
        for T, where the sum runs over all configured stocks (owner housing,
        optional rental housing, each labour asset) with Cᵢ = πᵢ · vᵢ · kᵢ and
        r = 1 − recovery_per/100. It is useful when callers want a single
        aggregate horizon that reflects how slow non-household-owned assets
        constrain the effective recovery profile.

        Notes
        -----
        - This method is kept for
          backward compatibility and per-stock diagnostics.
        - This definition ignores liquidity-induced piecewise effects.
        - Requires all involved rates λᵢ > 0 and coefficients Cᵢ ≥ 0.
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
        Names are "owner", "rental", "labour/<key>". Each term's `C_i` is
        `baseline_income · v` so both `CapitalStock` (π·k·v) and
        `IncomeStream` (income·v) contribute consistently.
        """
        rr_owner = self._rec_rate()
        if rr_owner is None or rr_owner <= 0:
            return None

        labelled: List[Tuple[str, float, float]] = []

        c_base = self._baseline_income(self.config.owner_housing) * (
            self.config.owner_housing.v
        )
        if c_base < 0:
            return None
        if c_base > 0:
            labelled.append(("owner", float(c_base), float(rr_owner)))

        if self.config.rental_housing is not None:
            stock = self.config.rental_housing
            rr = self._stock_rec_rate(stock)
            if rr is None or rr <= 0:
                return None
            c_rental = self._baseline_income(stock) * stock.v
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
                c_lab = self._baseline_income(stock) * stock.v
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
            - LossType.RECOVERY_COST: Calculates recovery cost.
            - LossType.OWNER_HOUSING_LOSS: Calculates income loss.
            - LossType.CONSUMPTION_LOSS: Calculates consumption loss.
            - LossType.UTILITY_LOSS: Calculates utility loss.
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
            LossType.RECOVERY_COST,
            LossType.OWNER_HOUSING_LOSS,
            LossType.CONSUMPTION_LOSS,
            LossType.UTILITY_LOSS,
        )
        if owner_touched and self._rec_rate() is None:
            raise ValueError(
                f"Cannot compute {loss_type}: owner_housing has no recovery_rate "
                "or recovery_time set. Call opt_lambda() first to optimize the "
                "housing recovery rate, or specify one on CapitalStock."
            )

        if loss_type == LossType.RECOVERY_COST:
            loss = RecoveryCost(
                self.t,
                self._rec_rate(),
                self.config.owner_housing.v,
                self.config.owner_housing.k,
            )
        elif loss_type == LossType.OWNER_HOUSING_LOSS:
            loss = IncomeLoss(
                self.t,
                self._rec_rate(),
                self.config.owner_housing.v,
                self.config.owner_housing.k,
                self._stock_pi(self.config.owner_housing),
            )
        elif loss_type in (LossType.RENTAL_HOUSING_LOSS, LossType.LABOUR_INCOME_LOSS):
            if loss_type == LossType.RENTAL_HOUSING_LOSS:
                stock = self.config.rental_housing
                if stock is None:
                    return 0.0
                # _income_loss_for_stock abstracts over CapitalStock /
                # IncomeStream so this branch handles either.
                loss = self._income_loss_for_stock(
                    stock, self.t, self._stock_rec_rate(stock)
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
                    il = self._income_loss_for_stock(stock, self.t, rr)
                    # Store per-asset component series and totals
                    asset_key = f"{LossType.LABOUR_INCOME_LOSS.value} ({name})"
                    self.time_series[asset_key] = il.losses_t
                    self.total_losses[asset_key] = il.total(rho=0, method=method)
                    # Accumulate into aggregate
                    losses_sum = losses_sum + il.losses_t
                    total_sum = total_sum + self.total_losses[asset_key]
                self.time_series[loss_type] = losses_sum
                self.total_losses[loss_type] = total_sum
                return self.total_losses[loss_type]
        elif loss_type == LossType.CONSUMPTION_LOSS:
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
        elif loss_type == LossType.UTILITY_LOSS:
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

        # Convention: monetary-unit integrals (RECOVERY, INCOME, CONSUMPTION)
        # report nominal sums (rho=0). UTILITY is a welfare-theoretic integral
        # and is discounted at config.simulation.rho so the
        # utility total here matches what get_losses uses for Wellbeing Loss
        # and what opt_lambda minimizes.
        loss_rho = (
            self.config.simulation.rho if loss_type == LossType.UTILITY_LOSS else 0.0
        )
        self.time_series[loss_type] = loss.losses_t
        self.total_losses[loss_type] = loss.total(rho=loss_rho, method=method)

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
              weights the raw asset loss (pre-recovery)
            - Additional keys corresponding to each `LossType` (e.g., "Recovery Costs", "Income Loss").

        Notes
        -----
        - The `LossType` enumeration is iterated to calculate individual loss types.
        - The `UtilityLoss` class is used to compute the utility loss.
        - The `wellbeing_loss` and `equity_weight` functions are used to compute the respective metrics.
        - The results are stored in the `time_series` DataFrame and `total_losses` Series attributes.
        """
        # Calculate losses for configured loss types only. calc_loss already
        # writes total_losses[UTILITY] at rho=config.simulation.rho, so reuse
        # that value rather than recomputing — avoids two "utility" numbers
        # disagreeing on which rho they used.
        for loss_type in self._loss_types_for_run():
            self.calc_loss(loss_type, method=method)

        eta = self.config.simulation.eta
        c_avg = self._c_avg()
        du_dis = float(self.total_losses[LossType.UTILITY_LOSS])

        # first-order correction for the permanent
        # welfare cost of a depleted liquidity buffer, evaluated at pre-shock
        # marginal utility du/dc|_{c0} = c0^(-eta). Missing this understates
        # ΔW for every household that draws on S.
        du_taylor = (self._c0() ** (-eta)) * self._liquidity_depleted()
        wbl_integral = wellbeing_loss(du=du_dis, c_avg=c_avg, eta=eta)
        wbl_taylor = wellbeing_loss(du=du_taylor, c_avg=c_avg, eta=eta)
        well_being_loss = wbl_integral + wbl_taylor

        # Calculate equity weighted loss
        ew_loss = (
            equity_weight(c=self._c0(), c_avg=c_avg, eta=eta)
            * self.config.owner_housing.v
            * self.config.owner_housing.k
        )

        asset_loss = self.config.owner_housing.v * self.config.owner_housing.k

        # Update total losses with additional metrics
        self.total_losses["Wellbeing Loss"] = well_being_loss
        self.total_losses["Wellbeing Loss (Integral)"] = wbl_integral
        self.total_losses["Wellbeing Loss (Liquidity Term)"] = wbl_taylor
        self.total_losses["Asset Loss"] = asset_loss
        self.total_losses["Equity Weighted Asset Loss"] = ew_loss
        # resilience R = Δk_h / ΔC_eq. Numerator is owner-housing
        # asset loss (household-borne); rental/labour are not household-owned
        self.total_losses["Socio-economic Resilience"] = (
            asset_loss / well_being_loss if well_being_loss > 0 else float("inf")
        )

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
        else:
            fig = None

        sns.lineplot(x="time", y=loss_type, data=self.time_series, ax=ax)
        # Shade area under curve with consistent x-axis as in plot_consumption
        total_val = self.total_losses[loss_type]
        display = self._label_for(loss_type)
        if isinstance(loss_type, LossType) and loss_type == LossType.UTILITY_LOSS:
            label_total = f"Total {display}: {total_val:.2f}"
        else:
            decimals = self.config.simulation.currency_decimals
            label_total = f"Total {display}: {total_val:,.{decimals}f} {self.config.simulation.currency}"
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
        if not (isinstance(loss_type, LossType) and loss_type == LossType.UTILITY_LOSS):
            decimals = self.config.simulation.currency_decimals
            ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(
                    lambda x, pos, decimals=decimals: f"{x:,.{decimals}f}"
                )
            )
            ax.set_ylabel(f"{display} ({self.config.simulation.currency}/year)")
        else:
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
            ax.set_ylabel(f"{display}")
        # Make time axis start at 0 and end at simulation horizon
        try:
            t_min = float(np.nanmin(self.time_series["time"]))
            t_max = float(np.nanmax(self.time_series["time"]))
            ax.set_xlim(left=max(0.0, t_min), right=t_max)
        except Exception:
            pass
        # Add legend consistently
        ax.legend()

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
        if LossType.CONSUMPTION_LOSS not in self.time_series.columns:
            raise ValueError(
                "Losses have not been calculated. Run the 'get_losses' method first."
            )

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = None

        # Prepare component series
        time = self.time_series["time"]
        inc = self.time_series[LossType.OWNER_HOUSING_LOSS]
        recon = self.time_series[LossType.RECOVERY_COST]
        rental = (
            self.time_series[LossType.RENTAL_HOUSING_LOSS]
            if LossType.RENTAL_HOUSING_LOSS in self.time_series.columns
            else None
        )
        # Identify per-asset labour income component columns created in calc_loss
        labour_asset_cols = [
            c
            for c in self.time_series.columns
            if isinstance(c, str)
            and c.startswith(f"{LossType.LABOUR_INCOME_LOSS.value} (")
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

        decimals = self.config.simulation.currency_decimals
        # Bottom layer: Recovery
        label_recon = (
            f"Total {self._label_for(LossType.RECOVERY_COST)}: "
            f"{self.total_losses[LossType.RECOVERY_COST]:,.{decimals}f} "
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
            f"Total {self._label_for(LossType.OWNER_HOUSING_LOSS)}: "
            f"{self.total_losses[LossType.OWNER_HOUSING_LOSS]:,.{decimals}f} "
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
                f"Total {self._label_for(LossType.RENTAL_HOUSING_LOSS)}: "
                f"{self.total_losses[LossType.RENTAL_HOUSING_LOSS]:,.{decimals}f} "
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
                f"Total {self._label_for(col)}: "
                f"{total_val:,.{decimals}f} {self.config.simulation.currency}"
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
                f"{self._liquidity():,.{decimals}f} "
                f"{self.config.simulation.currency}"
            )
            # Add hatch by drawing again with no fill color, only hatch
            ax.fill_between(
                self.time_series["time"],
                self._c0() - self.time_series[LossType.CONSUMPTION_LOSS],
                self._c0()
                - self.time_series[f"{LossType.CONSUMPTION_LOSS} No Liquidity"],
                facecolor="none",
                edgecolor="black",
                hatch="///",
                linewidth=0.0,
                label=label3,
            )

        # Plot consumption losses with a dashed line and expand to the left by 4 months
        expanded_time = np.insert(self.time_series["time"], 0, [-1, -0.001])
        expanded_consumption_losses = np.insert(
            self._c0() - self.time_series[LossType.CONSUMPTION_LOSS],
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
                    f"Minimum consumption: {self.config.simulation.c_min:,.{decimals}f} "
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
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, pos, decimals=decimals: f"{x:,.{decimals}f}")
        )
        # Add legend
        ax.legend()

        return fig

    def opt_lambda(
        self,
        rec_time_max: Optional[float] = None,
        rec_time_min: float = 0.3,
        no_steps: float = 1000,
        method: str = "trapezoid",
        eps_rel: float = 0.0,
        raise_on_fail: bool = True,
        rho: Optional[float] = None,
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
        rho : float, optional
            Utility discount rate used in the optimizer's objective. When
            `None` (default), the configured `SimulationConfig.rho` is used so
            the optimized λ is consistent with the discounted well-being loss
            reported by `get_losses`. Pass `rho=0.0` to reproduce the
            pre-fix undiscounted optimum.

        Returns
        -------
        Optional[float]
            The optimal reconstruction rate if found; otherwise None when raise_on_fail=False.
        """
        opt_rho = float(self.config.simulation.rho if rho is None else rho)
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

        # Iterate through each lambda value and calculate losses. The utility
        # grid uses the same `opt_rho` as the optimizer so that plot_opt_lambda
        # visualizes the same objective the solver actually minimized.
        # Silence the "Consumption contains zero or negative values" warning —
        # infeasible candidate λs produce it by design (treated as +∞ in the
        # solver). User-land methods.utility calls stay unaffected.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Consumption contains zero or negative values",
                category=UserWarning,
            )
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
                    ).total(rho=opt_rho, method=method)
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
            rho=opt_rho,
            recovery_per=self.config.simulation.recovery_per,
        )
        self.lambda_opt = opt

        # Save optimization dataframe regardless of success
        df = pd.DataFrame(
            {
                "lambda": lambdas,
                "reconstruction_time": recovery_time(
                    rate=lambdas, rebuilt_per=self.config.simulation.recovery_per
                ),
                LossType.RECOVERY_COST: recovery_costs,
                LossType.OWNER_HOUSING_LOSS: income_losses,
                LossType.CONSUMPTION_LOSS: consumption_losses,
                LossType.UTILITY_LOSS: utility_losses,
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

    def plot_opt_lambda(
        self,
        x_type: Literal["rate", "time"] = "rate",
        axs: Optional[Sequence[plt.Axes]] = None,
    ) -> Optional[plt.Figure]:
        """
        Plot the optimization results for the reconstruction rate (lambda) or recovery time.

        Parameters
        ----------
        x_type : Literal["rate", "time"], optional
            The type of x-axis to use for the plot. Can be "rate" for reconstruction rate or "time" for recovery time. Default is "rate".
        axs : Sequence[matplotlib.axes.Axes], optional
            Pair of axes `(top, bottom)` to draw into: top panel holds currency-valued losses (recovery / owner housing / consumption), bottom holds utility loss. If None, a new 2×1 figure is created internally. Mirrors the `ax` parameter of `plot_consumption` / `plot_loss`.

        Returns
        -------
        Optional[matplotlib.figure.Figure]
            The figure when `axs` is None (caller can save / display it); `None` when `axs` is supplied (caller manages the layout).

        Raises
        ------
        ValueError
            If the optimal lambda has not been calculated, or if `axs` is supplied with length ≠ 2.
        """
        if not hasattr(self, "l_opt"):
            raise ValueError(
                "Optimal lambda has not been calculated. Run the 'opt_lambda' method first."
            )

        if axs is None:
            # Wider than strictly needed for the axes — the extra width
            # reserves room for the external legend so its labels stay
            # inside the figure's bounding box (Jupyter inline and default
            # savefig both crop to that box).
            fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        else:
            if len(axs) != 2:
                raise ValueError(
                    f"plot_opt_lambda requires exactly 2 axes (top for currency "
                    f"losses, bottom for utility loss); got {len(axs)}."
                )
            fig = None
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
        # Use plain ax.plot instead of sns.lineplot: these are simple (x, y)
        # lines with no hue / estimator / CI, and sns.lineplot's label-kwarg
        # propagation has been inconsistent across seaborn 0.12–0.14 when x/y
        # are passed as pandas Series (the Series.name — here a LossType enum
        # — can shadow the explicit `label=`). Explicit str() casts on the
        # label ensure matplotlib receives a plain string regardless.
        axs[0].plot(
            x,
            self.l_opt[LossType.RECOVERY_COST],
            color="green",
            label=str(self._label_for(LossType.RECOVERY_COST)),
        )
        axs[0].plot(
            x,
            self.l_opt[LossType.OWNER_HOUSING_LOSS],
            color="blue",
            label=str(self._label_for(LossType.OWNER_HOUSING_LOSS)),
        )
        axs[0].plot(
            x,
            self.l_opt[LossType.CONSUMPTION_LOSS],
            color="purple",
            label=str(self._label_for(LossType.CONSUMPTION_LOSS)),
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
        decimals = self.config.simulation.currency_decimals
        axs[0].yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, pos, decimals=decimals: f"{x:,.{decimals}f}")
        )
        axs[0].set_ylabel(f"Total Loss ({self.config.simulation.currency})")
        axs[0].set_ylim(ylims)
        # Add well-being loss plot (plain ax.plot — see note above the top
        # panel's plot calls for why we bypass sns.lineplot here).
        axs[1].plot(
            x,
            self.l_opt[LossType.UTILITY_LOSS],
            color="black",
            label=str(self._label_for(LossType.UTILITY_LOSS)),
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

        # Status annotation: make non-interior outcomes visible on the plot.
        # Imported lazily to avoid adding a top-level dependency on the
        # OptLambdaStatus enum's location.
        from .methods import OptLambdaStatus

        status = self.lambda_opt.get("status") if self.lambda_opt else None
        if status == OptLambdaStatus.FLAT:
            # Flat welfare: shade the whole range in the bottom (utility) panel
            # and add a caption.
            axs[1].axhspan(
                ylims[0],
                ylims[1],
                facecolor="khaki",
                alpha=0.25,
                label="Welfare flat — any λ equivalent",
            )
        elif status in (
            OptLambdaStatus.BOUNDARY_LOWER,
            OptLambdaStatus.BOUNDARY_UPPER,
        ):
            # Optimum at search bound: place a red triangle at the true
            # minimum (val_min, already in the right x-axis units — λ for
            # rate plots, T for time plots). BOUNDARY_LOWER in λ-space =
            # slowest recovery = LONGEST time in the time plot, so we
            # cannot use `x.iloc[0/-1]` as a proxy: that would flip for
            # the time axis.
            side = "slowest" if status == OptLambdaStatus.BOUNDARY_LOWER else "fastest"
            axs[1].scatter(
                [val_min],
                [axs[1].get_ylim()[1]],
                marker="v",
                color="red",
                s=80,
                zorder=5,
                label=f"Optimum at {side}-recovery search bound",
            )

        # Legends rendered to the right of each panel. `bbox_to_anchor=(1.02, 1)`
        # with `borderaxespad=0` anchors the legend just outside the axes'
        # upper-right corner. When we created the figure ourselves, we pair
        # this with `fig.subplots_adjust(right=0.68)` below so the legend
        # stays inside the figure's bounding box — otherwise Jupyter's inline
        # backend (and savefig without `bbox_inches="tight"`) crops the label
        # text off the right edge while leaving the swatches visible, which
        # looks like "a legend with no labels".
        axs[0].legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
        axs[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
        if fig is not None:
            # Only adjust the layout when we own the figure. When the caller
            # supplied `axs`, they manage their own figure sizing — if their
            # parent figure is narrow, they may need a similar
            # `fig.subplots_adjust(right=…)` call or `bbox_inches="tight"` on
            # savefig to avoid legend clipping.
            fig.subplots_adjust(right=0.6)

        return fig
