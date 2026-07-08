"""
main.py
--------
Orquestador principal del proyecto Forecasting-COPUSD.

Pipeline:
    1. Leer configuración
    2. Crear carpetas necesarias
    3. Configurar logging
    4. Cargar dataset
    5. Validar datos
    6. Preprocesar datos
    7. Estimar parámetros históricos
    8. Ejecutar Random Walk (estimación)
    9. Ejecutar Monte Carlo (simulación)
    10. Calcular intervalos de confianza
    11. Generar gráficas
    12. Guardar CSV
    13. Guardar imágenes
    14. Mostrar resumen final
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from utils import (load_config, ensure_directories, setup_logging,
                    timer, print_header, print_kv, print_step)
from data_loader import load_dataset
from preprocessing import validate_data, preprocess, get_returns
from random_walk import estimate_parameters
from monte_carlo import run_monte_carlo
from evaluation import compute_intervals, save_results
from visualization import plot_forecast_fan


def main():
    # 1. Leer configuración
    config = load_config("config/config.yaml")

    # 2. Crear carpetas necesarias
    ensure_directories(config)

    # 3. Configurar logging
    logger = setup_logging(config)

    date_col = config["data"]["date_column"]
    price_col = config["data"]["price_column"]

    print_header("Forecasting COP/USD")
    print("Random Walk + Monte Carlo")
    print_header("")

    with timer() as elapsed:
        # 4. Cargar dataset
        raw = load_dataset(config["data"]["raw_path"], date_col, price_col)

        print_kv("Dataset", os.path.basename(config["data"]["raw_path"]))
        print_kv("Observaciones", len(raw))
        print_kv("Primer dato", raw[date_col].iloc[0].date())
        print_kv("Último dato", raw[date_col].iloc[-1].date())
        print_kv("Último precio", round(float(raw[price_col].iloc[-1]), 2))

        model_label = ("Random Walk con drift" if config["model"]["with_drift"]
                        else "Random Walk sin drift")
        print_kv("Modelo", model_label)
        print_kv("Horizonte", f"{config['model']['horizon_days']} días")
        print_kv("Simulaciones", config["model"]["n_simulations"])
        print_kv("Semilla", config["model"]["seed"])
        print("Calculando...")

        # 5. Validar datos
        validated = validate_data(raw, date_col, price_col)

        # 6. Preprocesar datos (retornos logarítmicos)
        processed = preprocess(validated, date_col, price_col)
        returns = get_returns(processed)

        # 7. Estimar parámetros históricos
        params = estimate_parameters(returns, with_drift=config["model"]["with_drift"])
        print_step("Drift estimado")
        print_step("Volatilidad estimada")

        # 8. Ejecutar Random Walk / 9. Ejecutar Monte Carlo
        S0 = float(validated[price_col].iloc[-1])
        simulations = run_monte_carlo(
            S0=S0,
            mu=params["mu"],
            sigma=params["sigma"],
            horizon=config["model"]["horizon_days"],
            n_simulations=config["model"]["n_simulations"],
            seed=config["model"]["seed"],
        )
        print_step("Monte Carlo finalizado")

        # 10. Calcular intervalos
        intervals = compute_intervals(simulations, config["model"]["confidence_levels"])
        print_step("Intervalos calculados")

        # 11. Generar gráficas
        fig_path = plot_forecast_fan(validated, simulations, intervals, config,
                                      date_col=date_col, price_col=price_col)

        # 12. Guardar CSV / 13. Guardar imágenes (la imagen ya se guardó en el paso 11)
        paths = save_results(simulations, intervals, config)
        print_step("Resultados guardados")

    # 14. Mostrar resumen final
    print_kv("Tiempo total", f"{elapsed():.1f} segundos")
    print_header("")
    print_kv("Figura", fig_path)
    print_kv("Simulaciones CSV", paths["simulations_path"])
    print_kv("Intervalos CSV", paths["intervals_path"])

    logger.info("Ejecución finalizada correctamente.")


if __name__ == "__main__":
    main()
