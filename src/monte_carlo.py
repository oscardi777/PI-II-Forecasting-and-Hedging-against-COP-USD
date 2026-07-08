"""
monte_carlo.py
----------------
Simulación Monte Carlo (vectorizada) de trayectorias Random Walk para la
tasa de cambio COP/USD, a partir de los parámetros estimados en random_walk.py.
"""

import logging
import numpy as np

logger = logging.getLogger("forecasting_copusd.monte_carlo")


def run_monte_carlo(S0: float, mu: float, sigma: float, horizon: int,
                     n_simulations: int, seed: int = 42) -> np.ndarray:
    """
    Ejecuta n_simulations trayectorias de un Random Walk en el espacio
    logarítmico de precios:

        log(S_t) = log(S_0) + sum_{i=1}^{t} r_i ,   r_i ~ N(mu, sigma^2)

    La generación de choques y la acumulación se hacen de forma vectorizada
    con NumPy para que el costo computacional sea bajo incluso con decenas
    de miles de simulaciones.

    Parameters
    ----------
    S0 : float
        Precio inicial (último precio observado en el histórico).
    mu : float
        Drift diario estimado (0 si el modelo es sin drift).
    sigma : float
        Volatilidad diaria estimada.
    horizon : int
        Número de días hábiles a simular hacia adelante.
    n_simulations : int
        Número de trayectorias Monte Carlo a generar.
    seed : int
        Semilla para reproducibilidad de los resultados.

    Returns
    -------
    np.ndarray
        Matriz de forma (n_simulations, horizon + 1) con las trayectorias
        simuladas de precio. La columna 0 corresponde a S0 en todas las filas.
    """
    if horizon <= 0:
        raise ValueError("El horizonte debe ser un entero positivo.")
    if n_simulations <= 0:
        raise ValueError("El número de simulaciones debe ser positivo.")

    rng = np.random.default_rng(seed)

    shocks = rng.normal(loc=mu, scale=sigma, size=(n_simulations, horizon))
    log_cum = np.cumsum(shocks, axis=1)
    log_paths = np.log(S0) + np.hstack([np.zeros((n_simulations, 1)), log_cum])
    price_paths = np.exp(log_paths)

    logger.info(f"Monte Carlo finalizado: {n_simulations} simulaciones, horizonte de {horizon} días.")
    return price_paths
