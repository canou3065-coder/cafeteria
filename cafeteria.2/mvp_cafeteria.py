from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # guarda las gráficas como imagen, sin abrir ventanas
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

CARPETA = Path(__file__).parent
ARCHIVO = CARPETA / "caso2_clientes.csv"
VARIABLES = ["gasto_mensual", "visitas_mes", "antiguedad_meses", "ticket_promedio"]
K = 3

PROMOCIONES = {
    "Fieles de alto valor": "Beneficio de lealtad (bebida gratis por visitas acumuladas)",
    "Frecuentes de ticket bajo": "Combo: agrega un postre por X pesos",
    "Ocasionales": "Cupón con vigencia corta para su próxima visita",
}


# ---------------------------------------------------------------- 1. DATOS
def cargar_datos():
    df = pd.read_csv(ARCHIVO)
    print(f"Clientes: {len(df)} | Columnas: {list(df.columns)}")
    # Validación: el modelo no acepta valores vacíos
    if df[VARIABLES].isna().any().any():
        raise ValueError("Hay valores vacíos en las variables; hay que limpiarlos.")
    print("Valores vacíos: 0\n")
    return df


# ---------------------------------------------------------- 2. ESTANDARIZAR
def estandarizar(df):
    # La columna 'clientes' es solo un ID y NO entra al modelo.
    # K-Means mide distancias, por eso todas las variables deben tener la misma escala.
    scaler = StandardScaler()
    X_esc = scaler.fit_transform(df[VARIABLES])
    return scaler, X_esc


# ------------------------------------------------------- 3. JUSTIFICAR K
def comparar_k(X_esc):
    ks = list(range(2, 7))
    inercias, siluetas = [], []
    for k in ks:
        m = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_esc)
        inercias.append(m.inertia_)
        siluetas.append(silhouette_score(X_esc, m.labels_))

    print("Comparación de k (inercia = método del codo; silueta: más alto = mejor)")
    for k, i, s in zip(ks, inercias, siluetas):
        marca = "  <-- elegido" if k == K else ""
        print(f"  k={k}  inercia={i:7.1f}  silueta={s:.3f}{marca}")
    print()

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(ks, inercias, marker="o")
    ax[0].set_title("Método del codo")
    ax[0].set_xlabel("k")
    ax[0].set_ylabel("Inercia")
    ax[1].plot(ks, siluetas, marker="o")
    ax[1].set_title("Silueta")
    ax[1].set_xlabel("k")
    plt.tight_layout()
    plt.savefig(CARPETA / "grafica_codo_silueta.png", dpi=150)
    plt.close()


# ------------------------------------------------------------ 4. ENTRENAR
def entrenar(df, X_esc):
    modelo = KMeans(n_clusters=K, n_init=10, random_state=42)
    df["grupo"] = modelo.fit_predict(X_esc)
    sil = silhouette_score(X_esc, df["grupo"])
    print(f"Silueta con k={K}: {sil:.2f}\n")
    return modelo


# ---------------------------------------------- 5. NOMBRAR Y ASIGNAR PROMO
def nombrar_grupos(df):
    # Los números 0, 1, 2 cambian entre corridas; por eso nombramos según el perfil.
    perfil = df.groupby("grupo")[VARIABLES].mean()
    alto = perfil["gasto_mensual"].idxmax()
    bajo = perfil["gasto_mensual"].idxmin()
    nombres = {g: "Frecuentes de ticket bajo" for g in perfil.index}
    nombres[alto] = "Fieles de alto valor"
    nombres[bajo] = "Ocasionales"

    df["segmento"] = df["grupo"].map(nombres)
    df["promocion"] = df["segmento"].map(PROMOCIONES)

    resumen = df.groupby("segmento")[VARIABLES].mean().round(0)
    resumen["clientes"] = df["segmento"].value_counts()
    resumen["% del gasto total"] = (
        df.groupby("segmento")["gasto_mensual"].sum() / df["gasto_mensual"].sum() * 100
    ).round(0)
    print("Perfil promedio de cada grupo:")
    print(resumen.to_string(), "\n")
    return nombres, resumen


# ------------------------------------------------------------- 6. GUARDAR
def guardar(df, resumen):
    df.to_csv(CARPETA / "clientes_con_promocion.csv", index=False)
    resumen.to_csv(CARPETA / "perfil_grupos.csv")

    plt.figure(figsize=(7, 5))
    for segmento, datos in df.groupby("segmento"):
        plt.scatter(datos["visitas_mes"], datos["ticket_promedio"], label=segmento)
    plt.xlabel("Visitas al mes")
    plt.ylabel("Ticket promedio")
    plt.title("Segmentos de clientes (K-Means, k=3)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CARPETA / "grafica_segmentos.png", dpi=150)
    plt.close()
    print("Archivos guardados: clientes_con_promocion.csv, perfil_grupos.csv,")
    print("                    grafica_segmentos.png, grafica_codo_silueta.png\n")


# ------------------------------------------- 7. LISTA DE ENVÍO POR PROMO
def lista_envio(df):
    """Qué clientes reciben cada promoción (la decisión que pide el cliente)."""
    columnas = ["clientes", "segmento", "promocion"] + VARIABLES
    lista = df[columnas].sort_values(["segmento", "clientes"])
    lista.to_csv(CARPETA / "lista_envio_promociones.csv", index=False)

    print("LISTA DE ENVÍO (clientes que recibe cada promoción):")
    for segmento, datos in lista.groupby("segmento"):
        ids = ", ".join(str(i) for i in datos["clientes"])
        print(f"\n[{segmento}] -> {PROMOCIONES[segmento]}")
        print(f"  {len(datos)} clientes: {ids}")
    print("\nGuardada en: lista_envio_promociones.csv (se abre en Excel)\n")
    return lista


def promocion_de_cliente(df, id_cliente):
    """Consulta la promoción de un cliente que ya existe en el archivo."""
    fila = df[df["clientes"] == id_cliente]
    if fila.empty:
        return None, None
    return fila["segmento"].iloc[0], fila["promocion"].iloc[0]


# ------------------------------------------------------ 8. CLIENTE NUEVO
def clasificar_cliente(modelo, scaler, nombres, gasto, visitas, antiguedad, ticket):
    """Asigna un cliente nuevo al grupo cuyo centro le queda más cerca."""
    nuevo = pd.DataFrame([[gasto, visitas, antiguedad, ticket]], columns=VARIABLES)
    grupo = int(modelo.predict(scaler.transform(nuevo))[0])
    segmento = nombres[grupo]
    return segmento, PROMOCIONES[segmento]


# ------------------------------------------------------ 9. MODO INTERACTIVO
def pedir_numero(texto, minimo=0):
    """Pide un número por teclado y repite la pregunta si el dato no es válido."""
    while True:
        valor = input(texto).strip().replace(",", "")
        try:
            numero = float(valor)
            if numero < minimo:
                print(f"  El valor debe ser mayor o igual a {minimo}.")
                continue
            return numero
        except ValueError:
            print("  Escribe solo un número (por ejemplo 3500).")


def modo_interactivo(df, modelo, scaler, nombres):
    print("=" * 60)
    print("MODO INTERACTIVO - ¿qué promoción le mando a un cliente?")
    print("=" * 60)
    while True:
        print("\n  1) Consultar un cliente que ya existe (por su número)")
        print("  2) Clasificar un cliente nuevo (escribes sus datos)")
        print("  3) Salir")
        try:
            opcion = input("Elige una opción: ").strip()

            if opcion == "1":
                id_cliente = int(pedir_numero("  Número de cliente: ", 1))
                seg, promo = promocion_de_cliente(df, id_cliente)
                if seg is None:
                    print(f"  No existe el cliente {id_cliente} en el archivo.")
                else:
                    print(f"\n  Cliente {id_cliente}")
                    print(f"  Segmento:  {seg}")
                    print(f"  Promoción: {promo}")

            elif opcion == "2":
                gasto = pedir_numero("  Gasto mensual (pesos): ")
                visitas = pedir_numero("  Visitas al mes: ")
                antiguedad = pedir_numero("  Antigüedad (meses): ")
                ticket = pedir_numero("  Ticket promedio (pesos): ")
                seg, promo = clasificar_cliente(
                    modelo, scaler, nombres, gasto, visitas, antiguedad, ticket
                )
                print("\n  Resultado del modelo")
                print(f"  Segmento:  {seg}")
                print(f"  Promoción: {promo}")

            elif opcion == "3":
                print("Fin de la demo.")
                break
            else:
                print("  Opción no válida. Escribe 1, 2 o 3.")
        except (EOFError, KeyboardInterrupt):
            print("\nFin de la demo.")
            break


if __name__ == "__main__":
    df = cargar_datos()
    scaler, X_esc = estandarizar(df)
    comparar_k(X_esc)
    modelo = entrenar(df, X_esc)
    nombres, resumen = nombrar_grupos(df)
    guardar(df, resumen)
    lista_envio(df)

    modo_interactivo(df, modelo, scaler, nombres)
