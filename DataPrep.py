import pandas as pd

data = pd.read_csv("/Users/malcolmrodgers/Documents/Coding/Python/GDPpredictor/worlddata2023.csv")
colnames = list(data.columns)

print(colnames)
print(data.head())