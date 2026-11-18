import pandas as pd

df = pd.read_csv("caso2_clientes.csv")
print(df.head())
print(df.shape)          # esperado: (185, 5)
print(df.isna().sum())   # esperado: todo en 0
print(df.describe())

columnas = ["gasto_mensual", "visitas_mes", "antiguedad_meses", "ticket_promedio"]
X = df[columnas]

import matplotlib.pyplot as plt

print(X.corr().round(2))

X.hist(bins=15, figsize=(10, 6))
plt.tight_layout()
plt.show()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_esc = scaler.fit_transform(X)

from sklearn.cluster import KMeans

modelo = KMeans(n_clusters=3, n_init=10, random_state=42)
df["grupo"] = modelo.fit_predict(X_esc)

print(df["grupo"].value_counts())

perfil = df.groupby("grupo")[columnas].mean().round(0)
perfil["clientes"] = df["grupo"].value_counts().sort_index()
print(perfil)

from sklearn.metrics import silhouette_score

print("Silueta:", round(silhouette_score(X_esc, df["grupo"]), 2))

plt.figure(figsize=(7, 5))
plt.scatter(df["visitas_mes"], df["ticket_promedio"], c=df["grupo"], cmap="viridis")
plt.xlabel("Visitas al mes")
plt.ylabel("Ticket promedio")
plt.title("Grupos de clientes (K-Means, k=3)")
plt.colorbar(label="Grupo")
plt.show()

df.to_csv("clientes_con_grupo.csv", index=False)
perfil.to_csv("perfil_grupos.csv")

perfil = df.groupby("grupo")[columnas].mean()
alto = perfil["gasto_mensual"].idxmax()
bajo = perfil["gasto_mensual"].idxmin()

nombres = {g: "Frecuentes de ticket bajo" for g in perfil.index}
nombres[alto] = "Fieles de alto valor"
nombres[bajo] = "Ocasionales"

promos = {
    "Fieles de alto valor": "Beneficio de lealtad (bebida gratis por visitas acumuladas)",
    "Frecuentes de ticket bajo": "Combo: agrega un postre por X pesos",
    "Ocasionales": "Cupón con vigencia corta para su próxima visita",
}

df["segmento"] = df["grupo"].map(nombres)
df["promocion"] = df["segmento"].map(promos)

print(df[["clientes", "segmento", "promocion"]].head(10))
df.to_csv("clientes_con_promocion.csv", index=False)