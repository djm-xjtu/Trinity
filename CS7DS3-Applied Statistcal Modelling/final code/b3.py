import pandas as pd
import matplotlib.pyplot as plt
import pymc3 as pm
import seaborn as sns
import numpy as np
import arviz as az
data = pd.read_csv("eight_players.csv")
data['stocks'] = data['STL'] + data['BLK']
data['player_id'] = data['Name'].astype('category').cat.codes
n_players = len(data['Name'].unique())

with pm.Model() as model:
    player_stocks_alpha = pm.Normal("player_stocks_alpha", mu=0, sd=10, shape=n_players)
    obs = pm.Poisson("obs", mu=pm.math.exp(player_stocks_alpha[data['player_id']]), observed=data['stocks'])
    trace = pm.sample(2000, tune=1000, target_accept=0.95, cores=1)

az.plot_trace(trace, var_names=['player_stocks_alpha'])
plt.savefig("trace_8_players.png", dpi=300)
plt.show()
summary = az.summary(trace, var_names=['player_stocks_alpha'], round_to=2)
print(summary)

player_stocks_mean = np.mean(np.exp(trace['player_stocks_alpha']), axis=0)
player_stocks_sorted = np.argsort(player_stocks_mean)[::-1]

player_names = data['Name'].astype('category').cat.categories
sorted_names = player_names[player_stocks_sorted]

fig, ax = plt.subplots(figsize=(10, 6))
for i in range(n_players):
    sns.kdeplot(np.exp(trace['player_stocks_alpha'][:, player_stocks_sorted[i]]), ax=ax, label=sorted_names[i])
ax.set_xlabel('Stocks')
ax.set_ylabel('Density')
ax.set_title('Posterior Distributions of Stocks Performance for Each Player')
ax.legend()
plt.savefig("posterior_distribution_8_players.png", dpi=300)
plt.show()
