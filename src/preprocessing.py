"""
preprocessing.py
-----------------
Validación de integridad de los datos y cálculo de retornos logarítmicos
diarios, insumo directo para la estimación del modelo Random Walk.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("forecasting_copusd.preprocessing")


def validate_data(df: pd.DataFrame, date_col: str = "date", price_col: str = "COP_USD") -> pd.DataFrame:
    """
    Valida y limpia el dataset:
    - elimina fechas duplicadas (se conserva el último registro)
    - elimina precios nulos o no positivos
    - reordena cronológicamente

    Levanta ValueError si el dataset queda vacío tras la limpieza.
    """
    n_before = len(df)

    df = df.drop_duplicates(subset=date_col, keep="last")
    df = df.dropna(subset=[price_col])
    df = df[df[price_col] > 0]
    df = df.sort_values(date_col).reset_index(drop=True)

    n_removed = n_before - len(df)
    if n_removed > 0:
        logger.warning(f"Se removieron {n_removed} registros inválidos o duplicados en la validación.")

    if df.empty:
        raise ValueError("El dataset quedó vacío después de la validación.")

    logger.info("Validación de datos completada.")
    return df


def preprocess(df: pd.DataFrame, date_col: str = "date", price_col: str = "COP_USD") -> pd.DataFrame:
    """
    Calcula los retornos logarítmicos diarios: r_t = ln(P_t / P_{t-1}).

    Returns
    -------
    pd.DataFrame
        DataFrame original + columna 'log_return' (sin la primera fila, NaN).
    """
    df = df.copy()
    df["log_return"] = np.log(df[price_col] / df[price_col].shift(1))
    df = df.dropna(subset=["log_return"]).reset_index(drop=True)

    logger.info(f"Preprocesamiento completado: {len(df)} retornos logarítmicos calculados.")
    return df


def get_returns(df: pd.DataFrame) -> np.ndarray:
    """Extrae el arreglo NumPy de retornos logarítmicos ya calculados."""
    if "log_return" not in df.columns:
        raise KeyError("El DataFrame no contiene la columna 'log_return'. Ejecuta preprocess() primero.")
    return df["log_return"].to_numpy()
