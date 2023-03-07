import pandas as pd

df = pd.read_csv("chem.txt", sep="\t")
print(df.head())

data = df.loc[0:1500000, ['Smiles']]

print(data.head())
#1，3，5行，salary,name列

data.to_csv("chem1w.csv")
