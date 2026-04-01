import pandas as pd
import numpy as np

df = pd.read_pickle("dataset/output/cluster_1.pkl")
s = df[df["id"] == 9906537].sort_values("Fecha_Hora")["value"]

print("num rows:", len(s))
print("num nan:", s.isna().sum())
print("num inf:", np.isinf(s.to_numpy(dtype=float)).sum())
print(s.head())
print(s.tail())