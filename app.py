import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Configuração da página
st.set_page_config(page_title="Previsão de Vendas", layout="wide")

st.title("📊 Previsão de Vendas - Loja 50 / GROCERY I")

@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    stores = pd.read_csv("stores.csv")
    holidays = pd.read_csv("holidays_events.csv")

    df["date"] = pd.to_datetime(df["date"])
    df["sales"] = df["sales"].astype("float32")
    df = df.sort_values(["store_nbr", "family", "date"])

    holidays = holidays[holidays["locale"] == "National"].copy()
    holidays["date"] = pd.to_datetime(holidays["date"])
    holidays = holidays[["date"]].drop_duplicates()
    holidays["is_holiday"] = 1

    df = df.merge(stores[["store_nbr", "city"]], on="store_nbr", how="left")
    df = df.merge(holidays, on="date", how="left")
    df["is_holiday"] = df["is_holiday"].fillna(0).astype("int8")

    df = df[(df["store_nbr"] == 50) & (df["family"] == "GROCERY I")].copy()

    df["dayofweek"] = df["date"].dt.dayofweek
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype("int")
    df["dayofyear"] = df["date"].dt.dayofyear
    df["quarter"] = df["date"].dt.quarter
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)

    def ewm_features(dataframe, alphas, lags):
        dataframe = dataframe.copy()
        for alpha in alphas:
            for lag in lags:
                colname = f"ewm_a{str(alpha).replace('.', '')}_l{lag}"
                dataframe[colname] = (
                    dataframe["sales"].shift(lag).ewm(alpha=alpha).mean()
                )
        return dataframe

    alphas = [0.95, 0.9]
    lags = [7, 14]
    df = ewm_features(df, alphas, lags)

    for lag in [1, 7, 14]:
        df[f"lag_{lag}"] = df["sales"].shift(lag)

    for window in [7, 14]:
        df[f"rolling_mean_{window}"] = df["sales"].shift(1).rolling(window).mean()
        df[f"rolling_std_{window}"] = df["sales"].shift(1).rolling(window).std()

    df = df.dropna()

    return df

df = load_data()

# Separação treino/teste
train_df = df.iloc[:-15]
test_df = df.iloc[-15:]

features = [col for col in df.columns if col not in ["date", "sales", "store_nbr", "family", "city"]]
X_train = train_df[features]
y_train = train_df["sales"]
X_test = test_df[features]
y_test = test_df["sales"]

# Modelo final
model = GradientBoostingRegressor(
    learning_rate=0.1,
    max_depth=3,
    n_estimators=100,
    subsample=1.0
)
model.fit(X_train, y_train)

# Previsão
y_pred = model.predict(X_test)

# Métricas
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.markdown(f"**MAE:** {mae:.2f}")
st.markdown(f"**RMSE:** {rmse:.2f}")

# Gráfico
test_df = test_df.copy()
test_df["prediction"] = y_pred

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(test_df["date"], test_df["sales"], label="Vendas reais", marker="o")
ax.plot(test_df["date"], test_df["prediction"], label="Previsão (modelo)", marker="o")
ax.set_title("📈 Previsão de Vendas")
ax.set_xlabel("Data")
ax.set_ylabel("Vendas")
ax.legend()
ax.grid(True)

st.pyplot(fig)
