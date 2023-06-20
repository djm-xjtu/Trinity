import pandas as pd
import arviz as az
import pymc3 as pm
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("eight_players.csv")
data['STOCK'] = data['STL'] + data['BLK']
data['home_encoded'] = (data['Column1'] == '@').astype(int)
player_names = data['Name'].unique()
n_players = len(player_names)
name_to_idx = {name: idx for idx, name in enumerate(player_names)}
data['player_idx'] = data['Name'].apply(lambda x: name_to_idx[x])

with pm.Model() as model:
    mu_alpha = pm.Normal("mu_alpha", mu=0, sd=5)
    sd_alpha = pm.HalfCauchy("sd_alpha", beta=5)
    mu_beta = pm.Normal("mu_beta", mu=0, sd=5)
    sd_beta = pm.HalfCauchy("sd_beta", beta=5)

    player_stocks_alpha = pm.Normal("player_stocks_alpha", mu=mu_alpha, sd=sd_alpha, shape=n_players)
    home_advantage = pm.Normal("home_advantage", mu=mu_beta, sd=sd_beta, shape=n_players)
    expected_stocks = player_stocks_alpha[data['player_idx'].values] + home_advantage[data['player_idx'].values] * data[
        'home_encoded'].values
    obs = pm.Poisson("obs", mu=pm.math.exp(expected_stocks), observed=data['STOCK'])

    trace = pm.sample(5000, tune=1000, target_accept=0.95, cores=1)
home_advantage_summary = az.summary(trace, var_names=['home_advantage'], round_to=2)
print(home_advantage_summary)

home_advantage_summary_94 = az.summary(trace, var_names=['home_advantage'], hdi_prob=0.94, round_to=2)
print(home_advantage_summary_94)

lower_hdi = home_advantage_summary_94['hdi_3%'].values
upper_hdi = home_advantage_summary_94['hdi_97%'].values

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(n_players)
ax.errorbar(x, home_advantage_summary_94['mean'], yerr=[home_advantage_summary_94['mean'] - lower_hdi, upper_hdi - home_advantage_summary_94['mean']], fmt='o', capsize=5, capthick=1)
ax.axhline(y=0, color='gray', linestyle='--')

ax.set_xticks(x)
ax.set_xticklabels(player_names, rotation=45, ha='right')
ax.set_ylabel("Home Advantage Effect")
ax.set_title("Home Advantage Effect with 94% HDI for Each Player")
plt.tight_layout()
plt.savefig("home_advantage_effect_8_players.png", dpi=300)
plt.show()