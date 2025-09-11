#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Promedia 10 realizaciones de macrodispersión y grafica Dx, Dy, Dz vs tiempo.
Cada macrodispersion_k.csv debe tener cabecera: t,Dx,Dy,Dz
Uso: python plot.py <varianza>
Valores válidos de varianza: 0.25, 1, 2.25, 4, 6.25
"""

import pathlib
import sys
import argparse

import pandas as pd
import math
import matplotlib.pyplot as plt


def get_lx_from_variance(variance):
    """Obtiene el valor de Lx según la varianza"""
    variance_to_lx = {0.25: 1024, 1.0: 2048, 2.25: 4096, 4.0: 8192, 6.25: 8192}
    return variance_to_lx.get(variance)


def get_experimental_data(variance):
    """Obtiene los datos experimentales según la varianza"""

    if variance == 0.25:
        bDL = [
            (1.0000e-01, 1.7175e-02),
            (1.2491e-01, 2.1198e-02),
            (1.5733e-01, 2.6158e-02),
            (1.9976e-01, 3.3266e-02),
            (2.5159e-01, 4.1058e-02),
            (3.1427e-01, 5.0664e-02),
            (3.9582e-01, 6.2532e-02),
            (4.9854e-01, 7.4886e-02),
            (6.2791e-01, 9.1033e-02),
            (7.9068e-01, 1.1066e-01),
            (1.0042e00, 1.2862e-01),
            (1.2543e00, 1.4949e-01),
            (1.5798e00, 1.7112e-01),
            (1.9893e00, 1.9011e-01),
            (2.5055e00, 2.0493e-01),
            (3.1557e00, 2.2090e-01),
            (3.9747e00, 2.2423e-01),
            (5.0061e00, 2.3812e-01),
            (6.3038e00, 2.3812e-01),
            (7.8741e00, 2.4177e-01),
            (1.0000e01, 2.4177e-01),
            (1.2595e01, 2.4541e-01),
            (1.5864e01, 2.4177e-01),
            (1.9976e01, 2.4177e-01),
            (2.4952e01, 2.4177e-01),
            (3.1688e01, 2.4541e-01),
            (3.9582e01, 2.4541e-01),
            (4.9854e01, 2.4912e-01),
        ]
        bDT = []  # No hay datos bDT para varianza 0.25 en el script original

    elif variance == 1.0:
        bDL = [
            (1.0083e-01, 9.0323e-02),
            (1.2491e-01, 1.1324e-01),
            (1.5992e-01, 1.3775e-01),
            (1.9976e-01, 1.7014e-01),
            (2.4952e-01, 2.0697e-01),
            (3.1952e-01, 2.4797e-01),
            (3.9912e-01, 2.9717e-01),
            (5.0269e-01, 3.6149e-01),
            (6.3299e-01, 4.2668e-01),
            (8.0390e-01, 4.9614e-01),
            (1.0042, 5.7677e-01),
            (1.2750, 6.5073e-01),
            (1.5926, 7.3418e-01),
            (1.9893, 8.2832e-01),
            (2.5264, 9.2045e-01),
            (3.1820, 1.0076),
            (4.0077, 1.0703),
            (5.0466, 1.1366),
            (6.3562, 1.1893),
            (8.0057, 1.2257),
            (10.167, 1.2257),
            (12.70, 1.2442),
            (15.864, 1.2633),
            (20.142, 1.2633),
            (25.369, 1.2633),
            (31.952, 1.2823),
            (40.244, 1.2633),
        ]
        bDT = [
            (1.0000e-01, 1.2682e-02),
            (1.1719e-01, 1.3970e-02),
            (1.4138e-01, 1.5733e-02),
            (1.6688e-01, 1.7073e-02),
            (1.9843e-01, 1.9226e-02),
            (2.3588e-01, 2.1493e-02),
            (2.8048e-01, 2.3845e-02),
            (3.3350e-01, 2.5876e-02),
            (3.9655e-01, 2.6853e-02),
            (4.7490e-01, 2.8927e-02),
            (5.6053e-01, 3.2337e-02),
            (6.6650e-01, 3.7239e-02),
            (7.9250e-01, 4.0411e-02),
            (9.4232e-01, 3.8362e-02),
            (1.1202, 3.6149e-02),
            (1.3320, 3.5876e-02),
            (1.5951, 3.6417e-02),
            (1.8832, 3.6149e-02),
            (2.2387, 2.9580e-02),
            (2.6810, 2.3496e-02),
            (3.1652, 1.6819e-02),
            (3.7905, 1.8252e-02),
            (4.4740, 1.1863e-02),
            (5.2820, 1.2311e-02),
            (6.3256, 1.6448e-02),
            (7.5214, 1.4713e-02),
            (8.8777, 1.0691e-02),
            (10.632, 1.0000e-02),
            (12.642, 1.2221e-02),
            (15.139, 7.8850e-03),
            (17.873, 9.9266e-03),
            (21.404, 8.4295e-03),
            (25.264, 9.2151e-03),
            (30.040, 8.8145e-03),
            (35.975, 9.3541e-03),
        ]

    elif variance == 2.25:
        bDL = [
            (0.1250, 0.3615),
            (0.1574, 0.4465),
            (0.1984, 0.5270),
            (0.2479, 0.6314),
            (0.3149, 0.7452),
            (0.3714, 0.8408),
            (0.3936, 0.8798),
            (0.4958, 1.0541),
            (0.6299, 1.2073),
            (0.7936, 1.4038),
            (0.9917, 1.6077),
            (1.2601, 1.8412),
            (1.5874, 2.0156),
            (2.0003, 2.3083),
            (2.4992, 2.5270),
            (3.1492, 2.8080),
            (3.9683, 2.9826),
            (4.9992, 3.2166),
            (6.3518, 3.4682),
            (8.0020, 3.6283),
            (9.9174, 3.8539),
            (12.7028, 4.0327),
            (15.8745, 4.0935),
            (20.0032, 4.2189),
            (25.2000, 4.3481),
            (32.0184, 4.3481),
            (40.0129, 4.4137),
            (50.8276, 4.4813),
            (63.5185, 4.4813),
        ]
        bDT = [
            (1.0072e-01, 3.5539e-02),
            (1.1719e-01, 3.7993e-02),
            (1.4031e-01, 4.3421e-02),
            (1.6444e-01, 4.5394e-02),
            (1.9688e-01, 4.9625e-02),
            (2.3405e-01, 5.3839e-02),
            (2.7823e-01, 6.1518e-02),
            (3.3083e-01, 6.4804e-02),
            (3.9328e-01, 6.4328e-02),
            (4.6752e-01, 6.5283e-02),
            (5.5578e-01, 7.5162e-02),
            (6.6069e-01, 8.2167e-02),
            (7.8542e-01, 9.0469e-02),
            (9.4059e-01, 9.9632e-02),
            (1.1102e00, 1.0188e-01),
            (1.3104e00, 1.0340e-01),
            (1.8518e00, 1.0889e-01),
            (2.2014e00, 1.0809e-01),
            (2.6170e00, 1.0188e-01),
            (3.1565e00, 9.6716e-02),
            (3.6991e00, 9.0469e-02),
            (4.4289e00, 7.7428e-02),
            (5.2650e00, 7.4611e-02),
            (6.2144e00, 7.0307e-02),
            (7.3875e00, 6.6757e-02),
            (8.8471e00, 6.5283e-02),
            (1.0517e01, 6.3372e-02),
            (1.2503e01, 6.2907e-02),
            (1.4757e01, 5.5873e-02),
            (1.7673e01, 5.0734e-02),
            (2.1009e01, 4.4730e-02),
            (2.4975e01, 4.6068e-02),
            (2.9902e01, 4.7457e-02),
            (3.4794e01, 4.4730e-02),
            (4.1957e01, 4.5394e-02),
            (4.9888e01, 4.7109e-02),
            (5.8452e01, 4.3742e-02),
            (7.0000e01, 4.6763e-02),
            (8.3811e01, 4.4066e-02),
        ]

    elif variance == 4.0:
        bDL = [
            (0.1008, 0.6411),
            (0.1249, 0.7798),
            (0.1573, 0.9343),
            (0.1982, 1.1031),
            (0.2516, 1.2633),
            (0.3143, 1.4468),
            (0.3958, 1.6823),
            (0.5027, 1.9266),
            (0.6330, 2.2398),
            (0.7907, 2.5657),
            (0.9959, 2.9383),
            (1.2647, 3.3151),
            (1.5798, 3.7966),
            (2.0059, 4.2835),
            (2.5055, 4.6881),
            (3.1820, 5.1322),
            (3.9747, 5.7903),
            (5.0061, 6.3387),
            (6.3038, 6.9375),
            (7.9396, 7.4817),
            (10.0000, 8.1903),
            (12.5951, 8.6976),
            (15.8635, 9.3799),
            (19.9756, 9.9632),
            (25.1594, 10.5803),
            (31.6884, 11.0713),
            (39.9117, 11.7598),
            (50.2690, 11.9371),
            (63.2995, 12.3027),
            (79.7260, 12.3027),
            (100.4153, 12.6794),
            (127.4970, 12.6794),
            (159.2575, 13.0677),
            (202.2553, 13.4679),
            (254.7417, 13.2648),
        ]
        bDT = [
            (9.9289e-02, 1.4266e-01),
            (1.1633e-01, 1.4481e-01),
            (1.3932e-01, 1.7426e-01),
            (1.6444e-01, 2.0059e-01),
            (1.9829e-01, 2.0356e-01),
            (2.3578e-01, 2.1918e-01),
            (2.7823e-01, 2.2746e-01),
            (3.9039e-01, 2.0811e-01),
            (4.7087e-01, 1.9907e-01),
            (5.5578e-01, 2.0207e-01),
            (7.9122e-01, 2.1918e-01),
            (1.1262e00, 2.1125e-01),
            (1.3391e00, 2.4496e-01),
            (1.5802e00, 2.4860e-01),
            (1.8789e00, 2.2085e-01),
            (2.2336e00, 2.3605e-01),
            (2.6363e00, 1.9041e-01),
            (3.1340e00, 1.8625e-01),
            (3.7256e00, 1.8218e-01),
            (4.4617e00, 1.6062e-01),
            (5.3040e00, 1.5477e-01),
            (6.2604e00, 1.3250e-01),
            (7.4422e00, 1.3954e-01),
            (1.0517e01, 1.2034e-01),
            (1.2503e01, 1.0378e-01),
            (1.4973e01, 8.8166e-02),
            (2.1009e01, 1.0849e-01),
            (2.9902e01, 1.0532e-01),
            (3.5294e01, 1.0301e-01),
            (4.1362e01, 1.0148e-01),
            (4.9888e01, 1.0378e-01),
            (5.9306e01, 1.0301e-01),
            (7.0000e01, 1.0378e-01),
            (8.3811e01, 1.0532e-01),
            (9.9632e01, 1.0849e-01),
            (1.1847e02, 1.0454e-01),
            (1.4083e02, 1.0688e-01),
            (1.6742e02, 1.1174e-01),
            (1.9902e02, 1.0770e-01),
            (2.3659e02, 1.0610e-01),
        ]

    elif variance == 6.25:
        # Para varianza 6.25, asumimos que no hay datos experimentales específicos
        bDL = []
        bDT = []
    else:
        bDL = []
        bDT = []

    return bDL, bDT


# --------------------------------------------------------------------
# Procesamiento de argumentos de línea de comandos
# --------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Plotea macrodispersión promedio para una varianza específica"
    )
    parser.add_argument(
        "variance", type=float, help="Valor de la varianza (0.25, 1, 2.25, 4, 6.25)"
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=100,
        help="Número de realizaciones a promediar (default: 100)",
    )

    args = parser.parse_args()
    variance = args.variance
    n_runs = args.n_runs

    # Validar varianza
    valid_variances = [0.25, 1.0, 2.25, 4.0, 6.25]
    if variance not in valid_variances:
        sys.exit(
            f"ERROR: Varianza {variance} no válida. Valores válidos: {valid_variances}"
        )

    # Configuración basada en la varianza
    lx = get_lx_from_variance(variance)
    if lx is None:
        sys.exit(f"ERROR: No se pudo determinar Lx para varianza {variance}")

    vm = 100 / (lx * 5)
    lamb = 50.0

    # Patrones de archivos
    var_str = f"{variance:.2f}".replace(".", "")
    if len(var_str) == 2:
        var_str = var_str + "0"  # Para casos como 1.00 -> 100

    file_pattern = (
        f"output/out_{variance:.2f}/macrodispersion_var_v9_{variance:.2f}_{{:d}}.csv"
    )
    outfig = f"output/out_{variance:.2f}/macrodispersion_avg.png"

    print(f"Configuración:")
    print(f"  Varianza: {variance}")
    print(f"  Lx: {lx}")
    print(f"  vm: {vm:.6f}")
    print(f"  Patrón de archivos: {file_pattern}")
    print(f"  Número de realizaciones: {n_runs}")
    print(f"  Archivo de salida: {outfig}")
    print()

    # --------------------------------------------------------------------
    # Carga de todas las realizaciones
    # --------------------------------------------------------------------

    dfs = []
    for k in range(n_runs):
        fname = file_pattern.format(k)
        path = pathlib.Path(fname)
        if not path.is_file():
            print(f"ADVERTENCIA: no se encontró «{fname}», saltando...")
            continue

        df = pd.read_csv(path)
        if not {"t", "Dx", "Dy", "Dz"} <= set(df.columns):
            print(
                f"ADVERTENCIA: «{fname}» no contiene las columnas esperadas, saltando..."
            )
            continue
        dfs.append(df)

    if not dfs:
        sys.exit("ERROR: No se pudo cargar ningún archivo de datos")

    print(f"Se cargaron {len(dfs)} realizaciones exitosamente")

    # --------------------------------------------------------------------
    # Cálculo del promedio ⟨Dx⟩, ⟨Dy⟩, ⟨Dz⟩ por instante
    # --------------------------------------------------------------------
    all_data = pd.concat(dfs, axis=0, ignore_index=True)
    mean_data = (
        all_data.groupby("t", as_index=False)[["Dx", "Dy", "Dz"]]
        .mean()
        .sort_values("t")
    )
    mean_data[["Dx", "Dy", "Dz"]] /= 2 * lamb * vm * (2 / math.sqrt(math.pi))
    mean_data["t"] /= lamb / vm

    # Obtener datos experimentales
    bDL, bDT = get_experimental_data(variance)

    # --------------------------------------------------------------------
    # Gráfica: curvas promedio
    # --------------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(mean_data["t"], mean_data["Dx"], label=r"$\langle D_x\rangle$")
    plt.plot(mean_data["t"], mean_data["Dy"], label=r"$\langle D_y\rangle$")
    plt.plot(mean_data["t"], mean_data["Dz"], label=r"$\langle D_z\rangle$")

    # --------------------------------------------------------------------
    # Añadimos los puntos experimentales
    # --------------------------------------------------------------------
    if bDL:
        bx, by = zip(*bDL)
        plt.scatter(bx, by, marker="o", color="black", zorder=4, label="bDL")

    if bDT:
        bx, by = zip(*bDT)
        plt.scatter(
            bx,
            by,
            marker="^",
            facecolors="none",
            edgecolors="black",
            zorder=4,
            label="bDT",
        )

    # --------------------------------------------------------------------
    # Configuración de la gráfica
    # --------------------------------------------------------------------
    plt.xlabel("Tiempo $t$")
    plt.ylabel(r"Macrodispersión $\langle D_\alpha(t)\rangle$")
    plt.title(
        f"Promedio de la macrodispersión (σ²={variance}, {len(dfs)} realizaciones)"
    )
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(outfig, dpi=300)
    plt.show()

    print(f"Figura guardada en «{outfig}»")


if __name__ == "__main__":
    main()
