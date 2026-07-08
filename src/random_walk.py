"""
random_walk.py
---------------
Estimación de los parámetros del modelo Random Walk (con o sin drift)
aplicado al logaritmo de la tasa de cambio COP/USD, y simulación de una
trayectoria individual (útil para pruebas rápidas o depuración).

Modelo:
    log(S_t) = log(S_0) + sum_{i=1}^{t} r_i ,   r_i ~ N(mu, sigma^2)

Donde:
    mu    = drift diario (0 si el modelo es "sin drift")
    sigma = volatilidad diaria de los retornos logarítmicos
"""

import logging
import numpy as np

logger = logging.getLogger("forecasting_copusd.random_walk")


def estimate_parameters(returns: np.ndarray, with_drift: bool = True) -> dict:
    """
    Estima mu (drift diario) y sigma (volatilidad diaria) a partir de los
    retornos logarítmicos históricos.

    Parameters
    ----------
    returns : np.ndarray
        Retornos logarítmicos diarios históricos.
    with_drift : bool
        Si es False, mu se fija en 0 (Random Walk puro / martingala).

    Returns
    -------
    dict
        {"mu": float, "sigma": float, "with_drift": bool}
    """
    if returns.size < 2:
        raise ValueError("Se requieren al menos 2 retornos para estimar los parámetros.")

    mu = float(np.mean(returns)) if with_drift else 0.0
    sigma = float(np.std(returns, ddof=1))

    logger.info(f"Parámetros estimados -> mu={mu:.6f}, sigma={sigma:.6f}, with_drift={with_drift}")

    return {"mu": mu, "sigma": sigma, "with_drift": with_drift}


def simulate_path(S0: float, mu: float, sigma: float, horizon: int, seed: int = None) -> np.ndarray:
    """
    Simula una única trayectoria de Random Walk hacia adelante.

    Parameters
    ----------
    S0 : float
        Precio inicial (último precio observado).
    mu, sigma : float
        Parámetros del modelo (drift y volatilidad diaria).
    horizon : int
        Número de días a simular.
    seed : int, opcional
        Semilla para reproducibilidad.

    Returns
    -------
    np.ndarray
        Trayectoria de precios de longitud horizon + 1 (incluye S0 en la posición 0).
    """
    rng = np.random.default_rng(seed)
    shocks = rng.normal(loc=mu, scale=sigma, size=horizon)
    log_path = np.log(S0) + np.concatenate(([0.0], np.cumsum(shocks)))
    return np.exp(log_path)
