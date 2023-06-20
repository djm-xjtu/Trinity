import pandas as pd
import numpy as np
import pymc3 as pm
import seaborn as sns
import matplotlib.pyplot as plt
import arviz as az

data = pd.read_csv("Jaren_Jackson_jr_data.csv", usecols=['STOCK', 'Home/Away'])
data['home_encoded'] = data['Home/Away'].map({'Home': 1, 'Away': 0})

with pm.Model() as model:
    mu_home = pm.Gamma("mu_home", alpha=0.001, beta=0.001)
    mu_away = pm.Gamma("mu_away", alpha=0.001, beta=0.001)
    home_rate = pm.Gamma("home_rate", alpha=mu_home, beta=1)
    away_rate = pm.Gamma("away_rate", alpha=mu_away, beta=1)
    obs = pm.Poisson("obs",
                     mu=data['home_encoded'].values * home_rate + (1 - data['home_encoded'].values) * away_rate,
                     observed=data['STOCK'])
    trace = pm.sample(10000, tune=1000, target_accept=0.95, cores=1)

az.plot_trace(trace, var_names=['mu_home', 'mu_away', 'home_rate', 'away_rate'])
plt.savefig("trace_of_hyperpriors.png", dpi=300)
plt.show()

az.summary(trace, var_names=['mu_home', 'mu_away', 'home_rate', 'away_rate'], round_to=2)
summary = pm.summary(trace).round(2)
print(summary)
home_away_ratio = np.mean(trace['home_rate'] / trace['away_rate'])
print(f"Home/Away Ratio: {home_away_ratio:.2f}")
prob_home_greater_than_away = np.mean(trace['home_rate'] > trace['away_rate'])
print(f"Probability(home_rate > away_rate): {prob_home_greater_than_away * 100:.6f}%")

home_rate_samples = trace['home_rate']
away_rate_samples = trace['away_rate']

fig, ax = plt.subplots()
sns.kdeplot(home_rate_samples, ax=ax, label='Home')
sns.kdeplot(away_rate_samples, ax=ax, label='Away')
ax.set_xlabel('Rate')
ax.set_ylabel('Density')
ax.set_title('Posterior Distributions of Home and Away Rates')
ax.legend()
plt.savefig("home_away_posterior.png", dpi=300)
plt.show()

difference_samples = home_rate_samples - away_rate_samples
plt.figure()
sns.histplot(difference_samples, kde=True, bins=30)
plt.xlabel('Difference between Simulated Samples')
plt.ylabel('Probability Density')
plt.title('Difference between Simulated Home and Away Rates')
plt.savefig("difference_hist.png", dpi=300)
plt.show()

plt.figure()
plt.scatter(home_rate_samples, away_rate_samples, alpha=0.3)
plt.xlabel('HOME STOCK')
plt.ylabel('AWAY STOCK')
plt.title('Scatter Plot of Home and Away STOCK Samples')
plt.savefig("home_away_scatter.png", dpi=300)
plt.show()