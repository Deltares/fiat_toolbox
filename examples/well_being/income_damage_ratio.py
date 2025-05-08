from fiat_toolbox.well_being import Household
from fiat_toolbox.well_being.methods import recovery_time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

df = pd.read_excel('./examples/well_being/alessia.xlsx')

# we assume relative damage and then get asset values
df['relativeDamage'] = [random.uniform(0, 0.7) for _ in range(len(df))]
df.loc[df['damage'] == 0, 'relativeDamage'] = 0


# get asset values
df['assets'] = df['damage'] / df['relativeDamage']

inc_avg = df['income'].mean() * 12

# yearly income
df['yearlyIncome'] = df['income'] * 12

# add id
df['id'] = range(1, len(df) + 1)

losses = {}
objects = {}

for i, household in df.iterrows():
    name = household['id']
    var = Household(
        v=household['relativeDamage'],
        k_str=household['assets'],
        c0=household['yearlyIncome'],
        c_avg=inc_avg
        )
    var.opt_lambda()
    losses[name] = var.get_losses()
    objects[name] = var
    
losses = pd.DataFrame(losses)
objects

df['lambda'] = df['id'].apply(lambda x: objects[x].l)
df['recoveryTime'] = df['lambda'].apply(recovery_time)

ymax= max(df['ratio'].max(),df['lambda'].max())

fig, ax= plt.subplots(figsize=(5,5))
sns.scatterplot(ax=ax, data=df, x='ratio', y='lambda')
ax.set_xlim(left=0, right=ymax)
ax.set_ylim(bottom=0, top=ymax)