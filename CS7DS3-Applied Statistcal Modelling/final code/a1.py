import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("Jaren_Jackson_jr_data.csv", usecols=['STOCK', 'Home/Away'])
print(data.describe())

sns.boxplot(x='Home/Away', y='STOCK', data=data)
sns.stripplot(x='Home/Away', y='STOCK', data=data, jitter=True, color=".3")
plt.savefig("boxplot_a.png", dpi=300)
plt.show()
