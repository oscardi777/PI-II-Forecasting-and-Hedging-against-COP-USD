"""
visualization.py
------------------
Generación del gráfico de pronóstico tipo "abanico" (fan chart):
serie histórica reciente + muestra de trayectorias Monte Carlo individuales
("rayos" que se abren desde el último dato observado) + banda(s) sombreada(s)
de intervalo de confianza + mediana proyectada.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

logger = logging.getLogger("forecasting_copusd.visualization")


def plot_forecast_fan(historical_df: pd.DataFrame, simulations: np.ndarray,
                       intervals: pd.DataFrame, config: dict,
                       date_col: str = "date", price_col: str = "COP_USD",
                       last_n_hist: int = 250, n_sample_paths: int = 150,
                       test_df: pd.DataFrame = None) -> str:
    """
    Genera y guarda el gráfico de pronóstico.

    Parameters
    ----------
    historical_df : pd.DataFrame
        Serie histórica usada como entrenamiento (ya validada), con
        columnas [date_col, price_col]. Si hay backtest activo, es el
        tramo de datos ANTERIOR a la fecha de corte.
    simulations : np.ndarray
        Matriz (n_simulations, horizon + 1) de trayectorias simuladas.
    intervals : pd.DataFrame
        Salida de evaluation.compute_intervals().
    config : dict
        Configuración del proyecto (usa config['output']['figures_dir'],
        config['model']['confidence_levels'] y config['model']['seed']).
    last_n_hist : int
        Cantidad de días históricos recientes a mostrar antes del pronóstico.
    n_sample_paths : int
        Cantidad de trayectorias individuales a dibujar como "rayos".
    test_df : pd.DataFrame, opcional
        Datos reales posteriores a la fecha de corte (solo si el backtest
        está habilitado). Si se provee, se dibujan como referencia para
        comparar visualmente contra el abanico simulado.

    Returns
    -------
    str
        Ruta del archivo de imagen guardado (PNG).
    """
    figures_dir = config["output"]["figures_dir"]
    os.makedirs(figures_dir, exist_ok=True)

    hist = historical_df.tail(last_n_hist).reset_index(drop=True)
    last_date = hist[date_col].iloc[-1]
    last_price = float(hist[price_col].iloc[-1])

    horizon = simulations.shape[1] - 1
    future_dates = pd.bdate_range(start=last_date, periods=horizon + 1)

    confidence_levels = sorted(config["model"]["confidence_levels"])
    widest_level = confidence_levels[-1]
    lower_wide = f"lower_{int(round(widest_level * 100))}"
    upper_wide = f"upper_{int(round(widest_level * 100))}"

    fig, ax = plt.subplots(figsize=(12, 6.5))

    # --- Serie histórica reciente ---
    ax.plot(hist[date_col], hist[price_col], color="#1f3b57", linewidth=1.6,
            label="Histórico")

    # --- Muestra de trayectorias individuales (rayos que se abren desde el lente) ---
    rng = np.random.default_rng(config["model"]["seed"])
    n_paths = simulations.shape[0]
    sample_idx = rng.choice(n_paths, size=min(n_sample_paths, n_paths), replace=False)
    for idx in sample_idx:
        ax.plot(future_dates, simulations[idx], color="#f4a261",
                linewidth=0.4, alpha=0.25, zorder=1)

    # --- Bandas de intervalos de confianza (de la más ancha a la más angosta) ---
    for level in reversed(confidence_levels):
        pct = int(round(level * 100))
        lc, uc = f"lower_{pct}", f"upper_{pct}"
        ax.fill_between(future_dates, intervals[lc], intervals[uc],
                         color="#2a9d8f", alpha=0.18, zorder=2,
                         label=f"IC {pct}%")

    # --- Mediana proyectada ---
    ax.plot(future_dates, intervals["median"], color="#e63946",
            linewidth=1.8, label="Mediana simulada", zorder=3)

    # --- Dato real (backtest): permite comparar visualmente el pronóstico ---
    if test_df is not None and not test_df.empty:
        ax.plot(test_df[date_col], test_df[price_col], color="#111111",
                linewidth=1.6, linestyle="-", marker="o", markersize=2.5,
                label="Real (backtest)", zorder=5)

    # --- Punto de origen ("lente" desde donde se abre el abanico) ---
    ax.scatter([last_date], [last_price], color="#1f3b57", s=40, zorder=4)
    ax.axvline(last_date, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_title("Pronóstico COP/USD — Random Walk + Monte Carlo", fontsize=13)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("COP por USD")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()

    fig_path = os.path.join(figures_dir, "forecast_fan_chart.png")
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)

    logger.info(f"Gráfico de pronóstico guardado en '{fig_path}'.")
    return fig_path