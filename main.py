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
    5b. (Backtest opcional) separar train / test por fecha de corte
    6. Preprocesar datos + seleccionar ventana de estimación
    7. Estimar parámetros históricos
    8. Ejecutar Random Walk (estimación)
    9. Ejecutar Monte Carlo (simulación)
    10. Calcular intervalos de confianza
    11. Generar gráficas
    12. Guardar CSV
    13. Guardar imágenes
    13b. (Backtest opcional) calcular métricas MAE / MSE / RMSE / MAPE
    14. Mostrar resumen final
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from utils import (load_config, ensure_directories, setup_logging,
                    timer, print_header, print_kv, print_step)
from data_loader import load_dataset
from preprocessing import validate_data, preprocess, get_returns, select_estimation_window
from random_walk import estimate_parameters
from monte_carlo import run_monte_carlo
from evaluation import compute_intervals, save_results
from visualization import plot_forecast_fan
from backtesting import split_train_test, attach_dates, evaluate_backtest, save_metrics


def main():
    # 1. Leer configuración
    config = load_config("config/config.yaml")

    # 2. Crear carpetas necesarias
    ensure_directories(config)

    # 3. Configurar logging
    logger = setup_logging(config)

    date_col = config["data"]["date_column"]
    price_col = config["data"]["price_column"]
    model_cfg = config["model"]
    backtest_cfg = config.get("backtest", {"enabled": False})

    print_header("Forecasting COP/USD")
    print("Random Walk + Monte Carlo")
    print_header("")

    with timer() as elapsed:
        # 4. Cargar dataset
        raw = load_dataset(config["data"]["raw_path"], date_col, price_col)

        # 5. Validar datos
        validated = validate_data(raw, date_col, price_col)

        # 5b. Backtest: separar train / test si está habilitado
        test_df = None
        if backtest_cfg.get("enabled", False):
            train_df, test_df = split_train_test(
                validated, date_col,
                train_end_date=backtest_cfg.get("train_end_date"),
                horizon_days=model_cfg["horizon_days"],
            )
            if len(test_df) < model_cfg["horizon_days"]:
                logger.warning(
                    f"Solo hay {len(test_df)} datos reales disponibles después de la fecha "
                    f"de corte, pero horizon_days={model_cfg['horizon_days']}. Las métricas "
                    f"se calcularán únicamente sobre las fechas con dato real disponible."
                )
        else:
            train_df = validated

        print_kv("Dataset", os.path.basename(config["data"]["raw_path"]))
        print_kv("Observaciones (entrenamiento)", len(train_df))
        print_kv("Primer dato", train_df[date_col].iloc[0].date())
        print_kv("Fecha de corte (último dato train)", train_df[date_col].iloc[-1].date())
        print_kv("Último precio (S0)", round(float(train_df[price_col].iloc[-1]), 2))
        if test_df is not None:
            print_kv("Datos reales para validar (backtest)",
                     f"{len(test_df)} obs ({test_df[date_col].iloc[0].date()} a "
                     f"{test_df[date_col].iloc[-1].date()})")

        model_label = ("Random Walk con drift" if model_cfg["with_drift"]
                        else "Random Walk sin drift")
        print_kv("Modelo", model_label)
        print_kv("Horizonte", f"{model_cfg['horizon_days']} días")
        print_kv("Simulaciones", model_cfg["n_simulations"])
        print_kv("Semilla", model_cfg["seed"])
        window_label = (model_cfg.get("estimation_window_days") or "histórico completo")
        print_kv("Ventana de estimación", window_label)
        print("Calculando...")

        # 6. Preprocesar datos (retornos logarítmicos) sobre el tramo de entrenamiento
        processed = preprocess(train_df, date_col, price_col)
        windowed = select_estimation_window(
            processed, model_cfg.get("estimation_window_days"), date_col
        )
        returns = get_returns(windowed)

        # 7. Estimar parámetros históricos
        params = estimate_parameters(returns, with_drift=model_cfg["with_drift"])
        print_step("Drift estimado")
        print_step("Volatilidad estimada")

        # 8. Ejecutar Random Walk / 9. Ejecutar Monte Carlo
        S0 = float(train_df[price_col].iloc[-1])
        simulations = run_monte_carlo(
            S0=S0,
            mu=params["mu"],
            sigma=params["sigma"],
            horizon=model_cfg["horizon_days"],
            n_simulations=model_cfg["n_simulations"],
            seed=model_cfg["seed"],
        )
        print_step("Monte Carlo finalizado")

        # 10. Calcular intervalos
        intervals = compute_intervals(simulations, model_cfg["confidence_levels"])
        intervals = attach_dates(intervals, start_date=train_df[date_col].iloc[-1], date_col=date_col)
        print_step("Intervalos calculados")

        # 11. Generar gráficas
        fig_path = plot_forecast_fan(train_df, simulations, intervals, config,
                                      date_col=date_col, price_col=price_col,
                                      test_df=test_df)

        # 12. Guardar CSV / 13. Guardar imágenes (la imagen ya se guardó en el paso 11)
        paths = save_results(simulations, intervals, config)
        print_step("Resultados guardados")

        # 13b. Backtest: calcular métricas si hay datos reales para comparar
        metrics_df = None
        metrics_path = None
        if test_df is not None:
            metrics_df = evaluate_backtest(
                intervals, test_df, S0=S0, date_col=date_col, price_col=price_col,
                metrics=backtest_cfg.get("metrics"),
            )
            metrics_path = save_metrics(metrics_df, config)
            print_step("Métricas de backtest calculadas")

    # 14. Mostrar resumen final
    print_kv("Tiempo total", f"{elapsed():.1f} segundos")
    print_header("")
    print_kv("Figura", fig_path)
    print_kv("Simulaciones CSV", paths["simulations_path"])
    print_kv("Intervalos CSV", paths["intervals_path"])

    if metrics_df is not None:
        print_kv("Métricas de backtest CSV", metrics_path)
        print_header("Resumen de métricas (backtest)")
        print(metrics_df.to_string(index=False))

    logger.info("Ejecución finalizada correctamente.")


if __name__ == "__main__":
    main()