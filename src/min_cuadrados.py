# -*- coding: utf-8 -*-


"""
Python 3
17 / 07 / 2024
@author: z_tjona

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth
"""


# ----------------------------- logging --------------------------
import logging
from sys import stdout
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    stream=stdout,
    datefmt="%m-%d %H:%M:%S",
)
logging.info(datetime.now())

import numpy as np

from typing import Callable, List

from src import eliminacion_gaussiana


# ####################################################################
def ajustar_min_cuadrados(
    xs: list,
    ys: list,
    gradiente: list[Callable[[list[float], list[float]], list[float]]],
) -> list[float]:
    """
    Ajusta los parámetros para un modelo utilizando mínimos cuadrados.
    
    ## Modificación
    Ahora se maneja directamente una matriz `A` y un vector `b` en lugar de una matriz aumentada.
    """
    assert len(xs) == len(ys), "xs y ys deben tener la misma longitud."

    num_pars = len(gradiente)
    logging.info(f"Se ajustarán {num_pars} parámetros.")

    # Crear la matriz A y el vector b
    A = np.zeros((num_pars, num_pars), dtype=float)
    b = np.zeros(num_pars, dtype=float)

    for i, der_parcial in enumerate(gradiente):
        assert callable(der_parcial), "Cada derivada parcial debe ser una función."
        coeficientes = der_parcial(xs, ys)
        A[i, :] = coeficientes[:-1]  # Coeficientes de los parámetros
        b[i] = coeficientes[-1]  # Término independiente

    logging.info(f"Matriz A:\n{A}")
    logging.info(f"Vector b:\n{b}")

    # Resolver el sistema usando eliminación gaussiana
    Ab = np.hstack([A, b.reshape(-1, 1)])  # Crear matriz aumentada para la función
    return list(eliminacion_gaussiana(Ab))
