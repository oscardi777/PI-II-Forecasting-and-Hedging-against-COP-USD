"""
evaluation.py
--------------
Cálculo de estadísticas resumen e intervalos de confianza a partir de las
trayectorias simuladas, y persistencia de resultados en CSV.
"""

import os
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("forecasting_copusd.evaluation")


def compute_intervals(simulations: np.ndarray, confidence_levels: list) -> pd.DataFrame:
    """
    Calcula, para cada día del horizonte simulado, la media, mediana,
    desviación estándar y los límites inferior/superior de cada intervalo
    de confianza solicitado (basados en los cuantiles empíricos de las
    simulaciones).

    Parameters
    ----------
    simulations : np.ndarray
        Matriz (n_simulations, horizon + 1) de trayectorias simuladas.
    confidence_levels : list[float]
        Niveles de confianza, p. ej. [0.80, 0.90, 0.95].

    Returns
    -------
    pd.DataFrame
        Una fila por día del horizonte (día 0 = último dato observado),
        con columnas: day, mean, median, std, lower_XX, upper_XX por cada
        nivel de confianza.
    """
    horizon = simulations.shape[1] - 1
    result = {
        "day": np.arange(0, horizon + 1),
        "mean": simulations.mean(axis=0),
        "median": np.median(simulations, axis=0),
        "std": simulations.std(axis=0, ddof=1),
    }

    for level in confidence_levels:
        alpha = 1 - level
        lower_q, upper_q = alpha / 2, 1 - alpha / 2
        pct = int(round(level * 100))
        result[f"lower_{pct}"] = np.quantile(simulations, lower_q, axis=0)
        result[f"upper_{pct}"] = np.quantile(simulations, upper_q, axis=0)

    logger.info(f"Intervalos de confianza calculados para niveles: {confidence_levels}")
    return pd.DataFrame(result)


def save_results(simulations: np.ndarray, intervals: pd.DataFrame, config: dict) -> dict:
    """
    Guarda en disco:
    - Las trayectorias simuladas completas (results/simulations/)
    - Los intervalos de confianza por día (results/forecasts/)

    Returns
    -------
    dict
        Rutas de los archivos CSV guardados.
    """
    sims_dir = config["output"]["simulations_dir"]
    forecasts_dir = config["output"]["forecasts_dir"]
    os.makedirs(sims_dir, exist_ok=True)
    os.makedirs(forecasts_dir, exist_ok=True)

    sims_path = os.path.join(sims_dir, "monte_carlo_simulations.csv")
    intervals_path = os.path.join(forecasts_dir, "confidence_intervals.csv")

    pd.DataFrame(simulations).to_csv(sims_path, index=False)
    intervals.to_csv(intervals_path, index=False)

    logger.info(f"Resultados guardados en '{sims_path}' y '{intervals_path}'.")
    return {"simulations_path": sims_path, "intervals_path": intervals_path}
