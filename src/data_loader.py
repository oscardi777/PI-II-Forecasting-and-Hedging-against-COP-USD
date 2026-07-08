"""
data_loader.py
--------------
Carga del dataset histórico de tasa de cambio COP/USD desde un archivo CSV.
"""

import logging
import pandas as pd

logger = logging.getLogger("forecasting_copusd.data_loader")


def load_dataset(path: str, date_col: str = "date", price_col: str = "COP_USD") -> pd.DataFrame:
    """
    Carga el dataset CSV de tasa de cambio y devuelve un DataFrame ordenado
    cronológicamente con únicamente las columnas relevantes.

    Parameters
    ----------
    path : str
        Ruta al archivo CSV (p. ej. data/raw/tasa_cop_usd_1993-2025.csv).
    date_col : str
        Nombre de la columna de fecha.
    price_col : str
        Nombre de la columna de precio/tasa de cambio.

    Returns
    -------
    pd.DataFrame
        DataFrame con columnas [date_col, price_col], ordenado por fecha.
    """
    logger.info(f"Cargando dataset desde '{path}'")

    try:
        df = pd.read_csv(path)
    except FileNotFoundError as e:
        logger.error(f"No se encontró el archivo de datos: {path}")
        raise e

    if date_col not in df.columns or price_col not in df.columns:
        raise ValueError(
            f"El dataset debe contener las columnas '{date_col}' y '{price_col}'. "
            f"Columnas encontradas: {list(df.columns)}"
        )

    df[date_col] = pd.to_datetime(df[date_col])
    df = df[[date_col, price_col]].sort_values(date_col).reset_index(drop=True)

    logger.info(
        f"Dataset cargado: {len(df)} observaciones "
        f"({df[date_col].iloc[0].date()} a {df[date_col].iloc[-1].date()})"
    )
    return df
