"""
backtesting.py
----------------
Validación del pronóstico Random Walk + Monte Carlo contra datos reales ya
observados. Permite:

1. Separar el dataset en un tramo de entrenamiento (usado para estimar mu y
   sigma y como punto de partida S0 de la simulación) y un tramo de prueba
   (datos reales posteriores a la fecha de corte, "ocultos" al modelo, que
   se usan únicamente para comparar).
2. Calcular métricas de error (MAE, MSE, RMSE, MAPE) comparando el dato real
   contra distintas referencias del pronóstico:
     - "point"  : pronóstico ingenuo de Random Walk (S0 constante, sin
                  simular; es la predicción teórica de un RW puro).
     - "mean"   : media de las trayectorias Monte Carlo por día.
     - "median" : mediana de las trayectorias Monte Carlo por día.
     - "lower_XX" / "upper_XX" : límites de cada intervalo de confianza.
"""

import os
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("forecasting_copusd.backtesting")


def split_train_test(df: pd.DataFrame, date_col: str, train_end_date: str = None,
                      horizon_days: int = None) -> tuple:
    """
    Divide el dataset en entrenamiento (hasta train_end_date inclusive) y
    prueba (posterior a train_end_date), este último usado como "realidad"
    para validar el pronóstico.

    Si train_end_date es None, se infiere automáticamente dejando los
    últimos horizon_days registros como conjunto de prueba.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset completo, ya validado y ordenado cronológicamente.
    date_col : str
        Nombre de la columna de fecha.
    train_end_date : str, opcional
        Fecha de corte "YYYY-MM-DD".
    horizon_days : int, opcional
        Usado solo si train_end_date es None.

    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        (train_df, test_df)
    """
    if train_end_date is None:
        if not horizon_days:
            raise ValueError("Debes indicar 'train_end_date' o 'horizon_days' para el backtest.")
        if horizon_days >= len(df):
            raise ValueError("horizon_days es mayor o igual al número de observaciones disponibles.")
        train_df = df.iloc[:-horizon_days].reset_index(drop=True)
        test_df = df.iloc[-horizon_days:].reset_index(drop=True)
    else:
        cutoff = pd.to_datetime(train_end_date)
        train_df = df[df[date_col] <= cutoff].reset_index(drop=True)
        test_df = df[df[date_col] > cutoff].reset_index(drop=True)

    if train_df.empty:
        raise ValueError("El conjunto de entrenamiento quedó vacío. Revisa 'train_end_date'.")
    if test_df.empty:
        raise ValueError(
            "El conjunto de prueba quedó vacío: no hay datos reales posteriores a "
            "'train_end_date' para comparar. Elige una fecha de corte más antigua."
        )

    logger.info(
        f"Backtest -> train: {len(train_df)} obs (hasta {train_df[date_col].iloc[-1].date()}), "
        f"test: {len(test_df)} obs ({test_df[date_col].iloc[0].date()} a {test_df[date_col].iloc[-1].date()})"
    )
    return train_df, test_df


def attach_dates(intervals: pd.DataFrame, start_date, date_col: str = "date") -> pd.DataFrame:
    """
    Agrega al DataFrame de intervalos (indexado por 'day' = 0..horizon) una
    columna de fecha calendario, generada como días hábiles a partir de
    start_date (día 0 == start_date == último dato de entrenamiento).
    """
    intervals = intervals.copy()
    horizon = int(intervals["day"].max())
    dates = pd.bdate_range(start=start_date, periods=horizon + 1)
    intervals[date_col] = dates
    return intervals


def evaluate_backtest(intervals: pd.DataFrame, test_df: pd.DataFrame, S0: float,
                       date_col: str = "date", price_col: str = "COP_USD",
                       metrics: list = None) -> pd.DataFrame:
    """
    Compara el pronóstico contra los datos reales observados (test_df),
    calculando métricas de error para cada referencia del pronóstico.

    El cruce entre pronóstico y realidad se hace por fecha calendario
    exacta (inner join), por lo que solo se comparan días en los que
    ambos conjuntos tienen dato (esto evita descalces si hay festivos
    que no coinciden exactamente entre el calendario simulado y el real).

    Parameters
    ----------
    intervals : pd.DataFrame
        Salida de evaluation.compute_intervals() con columna 'date' ya
        adjuntada (ver attach_dates).
    test_df : pd.DataFrame
        Datos reales posteriores a la fecha de corte.
    S0 : float
        Último precio del set de entrenamiento (pronóstico ingenuo de RW).
    metrics : list[str]
        Subconjunto de {"mae", "mse", "rmse", "mape"}.

    Returns
    -------
    pd.DataFrame
        Una fila por referencia de pronóstico (point, mean, median,
        lower_XX, upper_XX) con las métricas solicitadas y 'n_obs'
        (cantidad de fechas efectivamente comparadas).
    """
    metrics = metrics or ["mae", "mse", "rmse", "mape"]

    merged = pd.merge(
        intervals,
        test_df[[date_col, price_col]].rename(columns={price_col: "actual"}),
        on=date_col, how="inner",
    )

    if merged.empty:
        raise ValueError(
            "No hay fechas en común entre el pronóstico simulado y los datos reales de "
            "prueba. Verifica que las fechas de test caigan dentro del horizonte simulado."
        )

    reference_cols = ["mean", "median"] + sorted(
        c for c in intervals.columns if c.startswith(("lower_", "upper_"))
    )
    merged["point"] = S0
    reference_cols = ["point"] + reference_cols

    rows = []
    for ref in reference_cols:
        error = merged["actual"] - merged[ref]
        row = {"reference": ref, "n_obs": int(len(merged))}
        if "mae" in metrics:
            row["mae"] = float(np.mean(np.abs(error)))
        if "mse" in metrics:
            row["mse"] = float(np.mean(error ** 2))
        if "rmse" in metrics:
            row["rmse"] = float(np.sqrt(np.mean(error ** 2)))
        if "mape" in metrics:
            row["mape"] = float(np.mean(np.abs(error / merged["actual"])) * 100)
        rows.append(row)

    logger.info(
        f"Backtest evaluado sobre {len(merged)} fechas coincidentes "
        f"({merged[date_col].iloc[0].date()} a {merged[date_col].iloc[-1].date()})."
    )
    return pd.DataFrame(rows)


def save_metrics(metrics_df: pd.DataFrame, config: dict) -> str:
    """Guarda las métricas de backtest en results/metrics/backtest_metrics.csv."""
    metrics_dir = config["output"]["metrics_dir"]
    os.makedirs(metrics_dir, exist_ok=True)
    path = os.path.join(metrics_dir, "backtest_metrics.csv")
    metrics_df.to_csv(path, index=False)
    logger.info(f"Métricas de backtest guardadas en '{path}'.")
    return path