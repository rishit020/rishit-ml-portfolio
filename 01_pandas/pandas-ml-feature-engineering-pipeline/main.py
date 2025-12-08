import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

housing = fetch_california_housing(as_frame=True)
df = housing.frame

df["OceanProximity"] = np.random.choice(
    ["INLAND", "NEAR BAY", "NEAR OCEAN", "ISLAND"], size=len(df)
)

print(df.head())

cols_with_missing = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population"]
for col in cols_with_missing:
    df.loc[df.sample(frac=0.05).index, col] = np.nan

print(df.isna().sum())

for col in cols_with_missing:
    df[col] = df[col].fillna(df[col].median())

df["Latitude"] = df["Latitude"].fillna(df["Latitude"].mean())
df["Longitude"] = df["Longitude"].fillna(df["Longitude"].mean())

print(df.isna().sum())

numerical_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
numerical_cols.remove("MedHouseVal")

scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

categorical_cols = ["OceanProximity"]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

df["Region"] = np.random.choice(["North", "South", "East", "West"], size=len(df))
label_encoder = LabelEncoder()
df["Region"] = label_encoder.fit_transform(df["Region"])

def remove_outliers_IQR(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

df = remove_outliers_IQR(df, numerical_cols)

X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

final_df = df.copy()

print(final_df.head())
print("Rows:", len(final_df))
print("Columns:", len(final_df.columns))
