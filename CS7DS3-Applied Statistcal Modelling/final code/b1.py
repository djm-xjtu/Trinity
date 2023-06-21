import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("eight_players.csv")
data["STOCK"] = data["STL"] + data["BLK"]
stock_by_name = data.groupby("Name")["STOCK"]

avg_stock = stock_by_name.mean()
avg_stock = avg_stock.sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=avg_stock.index, y=avg_stock.values)
plt.title("Average STOCKs for 8 players")
plt.ylabel("Average STOCK")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("avg_stock_8_players.png", dpi=300)
plt.show()
