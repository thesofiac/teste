import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# Configuração da página
st.set_page_config(page_title="📈 Previsão de Vendas", layout="centered")
st.title("📈 Previsão de Vendas com Machine Learning")

# Carregamento dos dados
@st.cache_data
def load_data():
    df = pd.read_parquet("train.parquet")
    stores = pd.read_csv("stores.csv")
    holidays = pd.read_csv("holidays_events.csv")
    return df, stores, holidays

df, stores, holidays = load_data()

# Pré-processamento básico
df["date"] = pd.to_datetime(df["date"])
df["sales"] = df["sales"].astype("float32")
df = df.sort_values(["store_nbr", "family", "date"])

# Feriados nacionais
holidays = holidays[holidays["locale"] == "National"].copy()
holidays["date"] = pd.to_datetime(holidays["date"])
holidays = holidays[["date"]].drop_duplicates()
holidays["is_holiday"] = 1

# Merge com cidade e feriados
df = df.merge(stores[["store_nbr", "city"]], on="store_nbr", how="left")
df = df.merge(holidays, on="date", how="left")
df["is_holiday"] = df["is_holiday"].fillna(0).astype("int8")

# Interface: escolha de loja e família
store_options = sorted(df["store_nbr"].unique())
family_options = sorted(df["family"].unique())

store_selected = st.selectbox("Selecione a loja:", store_options)
family_selected = st.selectbox("Selecione a categoria:", family_options)

# Filtragem
df = df[(df["store_nbr"] == store_selected) & (df["family"] == family_selected)].copy()

# Features de tempo
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

# EWM
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

# Lags e Rolling
for lag in [1, 7, 14]:
    df[f"lag_{lag}"] = df["sales"].shift(lag)

for window in [7, 14]:
    df[f"rolling_mean_{window}"] = df["sales"].shift(1).rolling(window).mean()
    df[f"rolling_std_{window}"] = df["sales"].shift(1).rolling(window).std()

df = df.dropna()

# Divisão treino/teste
train_df = df.iloc[:-15]
test_df = df.iloc[-15:]

features = [col for col in df.columns if col not in ["date", "sales", "store_nbr", "family", "city"]]
X_train = train_df[features]
y_train = train_df["sales"]
X_test = test_df[features]
y_test = test_df["sales"]

with st.spinner("Treinando modelo..."):
    best_model = GradientBoostingRegressor(learning_rate=0.1, max_depth=3, n_estimators=100, subsample=1.0)

# Previsões
y_pred = best_model.predict(X_test)

# Corrigindo valores negativos
y_pred = np.where(y_pred < 0, 0, y_pred)

# Métricas
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.subheader("🔧 Melhores parâmetros do modelo")
st.json(grid.best_params_)

st.subheader("📏 Métricas de desempenho")
st.metric("MAE", f"{mae:.2f}")
st.metric("RMSE", f"{rmse:.2f}")

# Resultado
test_df = test_df.copy()
test_df["prediction"] = y_pred

# Gráfico
st.subheader("📊 Gráfico: Vendas Reais vs. Previsões")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(test_df["date"], test_df["sales"], label="Vendas reais", marker="o")
ax.plot(test_df["date"], test_df["prediction"], label="Previsão", marker="o")
ax.set_xlabel("Data")
ax.set_ylabel("Vendas")
ax.set_title(f"Previsão - Loja {store_selected} | {family_selected}")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Tabela
with st.expander("📋 Ver dados de teste com previsão"):
    st.dataframe(test_df[["date", "sales", "prediction"]].reset_index(drop=True))
