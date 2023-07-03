import pandas as pd
import numpy as np
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
from pathlib import Path


def get_equity_input(block_groups_census, damages):
    block_groups_census= pd.read_csv(block_groups_census)
    damages  = pd.read_csv(damages)

    # Merge census block groups with fiat output
    gdf_all = pd.merge(block_groups_census, damages, on="Census_Bg", how="left")
    gdf_all = gdf_all.dropna().reset_index(drop=True)

    return gdf_all

def calculate_equity_weights(gdf_all, gamma):

    gamma = 1.2 # elasticity

    I_PC = gdf_all["PerCapitaIncomeBG"] # mean per capita income
    Pop = gdf_all["TotalPopulationBG"] # population

    # Calculate aggregated annual income
    I_AA = I_PC * Pop
    # Calculate weighted average income per capita
    I_WA = np.average(I_PC, weights=Pop)

    EW = (I_PC / I_WA) ** -gamma # Equity Weight

    gdf_all["I_AA"] = I_AA 
    gdf_all["EW"] = EW 

    return gdf_all

def calculate_ewced_per_rp(gdf_all, gamma):
    I_AA = gdf_all["I_AA"] 
    EW = gdf_all["EW"]

    RP_cols = [name for name in gdf_all.columns if "Total Damage" in name]

    for col in RP_cols:
        D = gdf_all[col] # Damage for return period
        # EAD = gdf_all["EAD"]
        RP = int(col.split(" ")[-1][2:]) # Return period
        t = 1 # period of interest in years
        P = 1 - np.exp(-t/RP) # Probability of exceedance
        z = D / I_AA # Social Vulnerability

        R = (1 - (1 + P*((1-z)**(1-gamma)-1))**(1/(1-gamma))) / (P*z) # Risk premium
        # This step is needed to avoid nan value when z is zero
        R[R.isnull()] = 0 
        gdf_all[f"R_RP{RP}"] = R

        CED = R * D  # Certainty Equivalent Damage
        # CEAD = R * D * P # Certainty Equivalent Annual Damage
        # CEAD_2 = R * EAD # second method to test

        ######## Why is next step needed??????
        # EWED = EW * D  # Equity Weighted Expected Damage
        # EWEAD = EW * D * P  # Equity Weighted Expected Annual Damage
        # EWEAD_2 = EW * EAD  # second method to test
        ############################################################

        EWCED = EW * CED  # Equity Weighted Certainty Equivalent Damage
        # EWCEAD = EW * CEAD  # Equity Weighted Certainty Equivalent Annual Damage
        # EWCEAD_2 = EW * CEAD_2  # Equity Weighted Certainty Equivalent Annual Damage

        # Save in dataframe
        # gdf_all[f"EWED_RP{RP}"] = EWED 
        gdf_all[f"EWCED_RP{RP}"] = EWCED
        # gdf_all[f"EWCEAD_RP{RP}"] = EWCEAD
        # gdf_all[f"EWCEAD_2_RP{RP}"] = EWCEAD_2

    return gdf_all, RP_cols 

# Taken from FIAT for now
def calculate_coefficients(T):
    """Calculates coefficients used to compute the EAD as a linear function of the known damages
    Args:
        T (list of ints): return periods T1 … Tn for which damages are known
    Returns:
        alpha [list of floats]: coefficients a1, …, an (used to compute the AED as a linear function of the known damages)

    In which D(f) is the damage, D, as a function of the frequency of exceedance, f. In order to compute this EAD,
    function D(f) needs to be known for the entire range of frequencies. Instead, D(f) is only given for the n
    frequencies as mentioned in the table above. So, in order to compute the integral above, some assumptions need
    to be made for function D(h):

    (i)	   For f > f1 the damage is assumed to be equal to 0
    (ii)   For f<fn, the damage is assumed to be equal to Dn
    (iii)  For all other frequencies, the damage is estimated from log-linear interpolation between the known damages and frequencies

    """
    # Step 1: Compute frequencies associated with T-values.
    f = [1 / i for i in T]
    lf = [np.log(1 / i) for i in T]
    # Step 2:
    c = [(1 / (lf[i] - lf[i + 1])) for i in range(len(T[:-1]))]
    # Step 3:
    G = [(f[i] * lf[i] - f[i]) for i in range(len(T))]
    # Step 4:
    a = [
        ((1 + c[i] * lf[i + 1]) * (f[i] - f[i + 1]) + c[i] * (G[i + 1] - G[i]))
        for i in range(len(T[:-1]))
    ]
    b = [
        (c[i] * (G[i] - G[i + 1] + lf[i + 1] * (f[i + 1] - f[i])))
        for i in range(len(T[:-1]))
    ]
    # Step 5:
    if len(T) == 1:
        alpha = f
    else:
        alpha = [
            b[0] if i == 0 else f[i] + a[i - 1] if i == len(T) - 1 else a[i - 1] + b[i]
            for i in range(len(T))
        ]
    return alpha

def calculate_ewced_total(gdf_all, RP_cols):
    layers = []
    return_periods = []
    for i in RP_cols:
        RP = int(i.split(" ")[-1][2:])
        return_periods.append(RP)
        layers.append(gdf_all.loc[:, f"EWCED_RP{RP}"].values)

    stacked_layers = np.dstack(tuple(layers)).squeeze()
    gdf_all[f"EWCEAD"] = stacked_layers @ np.array(calculate_coefficients(return_periods))[:, None]

    return gdf_all

def rank_ewced(gdf_all):
    gdf_all["rank_EAD"] = gdf_all["EAD"].rank(ascending=False)
    gdf_all["rank_EWCEAD"] = gdf_all["EWCEAD"].rank(ascending=False)
    gdf_all["rank_diff"] = gdf_all["rank_EWCEAD"] - gdf_all["rank_EAD"]
    return gdf_all

def calculate_resilience_index(gdf_all):
    gdf_all["soc_res"] =  gdf_all["EAD"]/gdf_all["EWCEAD"]
    gdf_all["soc_res"][gdf_all["soc_res"] == np.inf] = np.nan
    return gdf_all

