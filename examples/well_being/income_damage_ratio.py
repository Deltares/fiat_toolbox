from fiat_toolbox.well_being import Household
from fiat_toolbox.well_being.methods import recovery_time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

df = pd.read_excel('alessia.xlsx')

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
        c_avg=inc_avg,
        # cmin=800 * 12,
        )
    var.opt_lambda(eps_rel=0.0)
    losses[name] = var.get_losses()
    objects[name] = var
    
losses = pd.DataFrame(losses)
objects

df['lambda'] = df['id'].apply(lambda x: objects[x].l)
df['recoveryTime'] = df['lambda'].apply(recovery_time)

ymax= max(df['ratio'].max(),df['lambda'].max())


from matplotlib.colors import LogNorm

# plot
plt.figure()
sc = plt.scatter(
    x=df['lambda'],
    y=df['ratio'],
    c=df['income'],
    cmap='viridis',
    norm=LogNorm(),
    s=10
    )
plt.colorbar(sc, label='Income (euros/month) (log scale)')
plt.ylabel('Income / Damage')
plt.xlabel('Recovery Rate (lambda)')
plt.ylim(0, 1)

# plot
plt.figure()
sc = plt.scatter(
    x=df['recoveryTime'],
    y=df['ratio'],
    c=df['income'],
    cmap='viridis',
    # norm=LogNorm(),
    s=10
    )
plt.colorbar(sc, label='Income (euros/month)')
plt.ylabel('Income / Damage')
plt.xlabel('Recovery Time (years)')
plt.ylim(0, 1)
plt.xlim(0, 10.5)

value_counts = df['recoveryTime'].value_counts()
df_subset = df[df['recoveryTime'] < 2]
# plot
plt.figure()
sc = plt.scatter(
    x=df_subset['recoveryTime'],
    y=df_subset['ratio'],
    c=df_subset['income'],
    cmap='viridis',
    norm=LogNorm(),
    s=10
    )
plt.colorbar(sc, label='Income (euros/month) (log scale)')
plt.ylabel('Income / Damage')
plt.xlabel('Recovery Time (years)')

