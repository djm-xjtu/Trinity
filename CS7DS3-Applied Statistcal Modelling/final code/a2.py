import pandas as pd
from scipy.stats import ttest_ind

data = pd.read_csv("Jaren_Jackson_jr_data.csv")

home_stocks = data[data['Home/Away'] == 'Home']['STOCK']
away_stocks = data[data['Home/Away'] == 'Away']['STOCK']
t_value, p_value = ttest_ind(home_stocks, away_stocks)

print("t-value:", t_value)
print("p-value:", p_value)

