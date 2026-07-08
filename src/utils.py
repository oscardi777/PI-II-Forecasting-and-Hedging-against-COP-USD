"""
utils.py
--------
Funciones auxiliares transversales al proyecto:
- Carga de configuración (config.yaml)
- Configuración de logging (logging.yaml)
- Creación de carpetas de salida
- Medición de tiempo de ejecución
- Utilidades de impresión para la consola (resumen del pipeline)
"""

import os
import time
import logging
import logging.config
from contextlib import contextmanager

import yaml


def load_config(path: str = "config/config.yaml") -> dict:
    """Carga el archivo de configuración YAML del proyecto."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def ensure_directories(config: dict) -> None:
    """Crea todas las carpetas de salida declaradas en config['output']."""
    for path in config.get("output", {}).values():
        os.makedirs(path, exist_ok=True)


def setup_logging(config: dict, logging_config_path: str = "config/logging.yaml") -> logging.Logger:
    """
    Configura el logging del proyecto.

    Si existe config/logging.yaml se usa dictConfig; en caso contrario se
    aplica una configuración básica de respaldo usando los valores de
    config['logging'].
    """
    log_file = config["logging"]["log_file"]
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    if os.path.exists(logging_config_path):
        with open(logging_config_path, "r", encoding="utf-8") as f:
            log_cfg = yaml.safe_load(f)
        logging.config.dictConfig(log_cfg)
    else:
        logging.basicConfig(
            level=config["logging"].get("level", "INFO"),
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            handlers=[
                logging.FileHandler(log_file, encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )
    return logging.getLogger("forecasting_copusd")


@contextmanager
def timer():
    """
    Context manager para medir el tiempo de ejecución de un bloque.

    Uso:
        with timer() as elapsed:
            ...
        print(elapsed())   # segundos transcurridos
    """
    start = time.time()
    yield lambda: time.time() - start


# ---------------------------------------------------------------------
# Utilidades de presentación en consola
# ---------------------------------------------------------------------

def print_header(title: str, width: int = 58) -> None:
    """Imprime una línea de separación con un título opcional."""
    print("=" * width)
    if title:
        print(title)


def print_kv(key: str, value) -> None:
    """Imprime un par clave/valor en dos líneas, estilo 'ficha resumen'."""
    print(f"{key}:")
    print(f"{value}")


def print_step(msg: str) -> None:
    """Imprime un paso completado del pipeline con un check (✓)."""
    print(f"✓ {msg}")
