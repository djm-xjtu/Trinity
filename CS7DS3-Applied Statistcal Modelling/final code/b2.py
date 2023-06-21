import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("eight_players.csv")
data["STOCK"] = data["STL"] + data["BLK"]

player_mean_stock = data.groupby("Name")["STOCK"].mean()
player_order = player_mean_stock.sort_values(ascending=False).index
plt.figure(figsize=(12, 6))
sns.boxplot(x="Name", y="STOCK", data=data, order=player_order)
plt.title("Box plot of STOCKs for 8 players")
plt.ylabel("STOCK")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("boxplot_stock_8_players.png", dpi=300)
plt.show()
