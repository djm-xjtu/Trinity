import pandas as pd
import random
df = pd.read_csv('gapminder.csv')
country = df.iloc[:, 0]
continent = df.iloc[:, 1]
year = df.iloc[:, 2]
df.iloc[:, 2] = year.apply(lambda x: x + (0.5 - random.random()) * 3)
lifeExp = df.iloc[:, 3]
pop = df.iloc[:, 4]
gdpPercap = df.iloc[:, 5]
old_year = year

df.to_csv('sgapminder.csv', index=False, encoding='utf-8')