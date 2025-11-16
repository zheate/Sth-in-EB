#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Efficiency curve fitting & prediction
- Input: CSV with columns:
    * I (required)
    * Either: EFF_percent (0-100)  OR  EFF_fraction (0-1)
- Optional: P, U (ignored for fitting here)
- Output:
    * params JSON (chosen model + parameters + metrics)
    * predictions CSV (grid I, predicted EFF%)
    * PNG plot (observations + fitted curve)
Usage:
    python efficiency_fit_predict.py --input data.csv --outdir ./out --model auto --imin 0 --imax 20 --istep 0.1
"""
import argparse, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import log
from scipy.optimize import curve_fit

def poly2(x, a, b, c):  # quadratic
    return a + b*x + c*x*x

def poly3(x, a, b, c, d):  # cubic
    return a + b*x + c*x*x + d*x*x*x

def rational_1_2(x, a, b, c):  # (a*x)/(1+b*x+c*x^2)
    return (a*x) / (1.0 + b*x + c*(x**2))

def hill(x, Emax, K, n):
    return Emax * (x**n) / (K**n + x**n)

def hill_droop(x, Emax, K, n, d):
    return (Emax * (x**n) / (K**n + x**n)) / (1.0 + d*x)

def exp_sat(x, Emax, k):
    return Emax * (1 - np.exp(-k*x))

CANDIDATES = {
    "hill_droop": (hill_droop, [60, 5, 2, 0.02]),
    "hill":       (hill,       [60, 5, 2]),
    "rational_1_2": (rational_1_2, [5, 0.1, 0.01]),
    "exp_sat":    (exp_sat,    [60, 0.2]),
    "poly3":      (poly3,      [10, 5, -0.1, 0.002]),
    "poly2":      (poly2,      [10, 5, -0.1]),
}

def fit_and_score(x, y, func, p0):
    popt, pcov = curve_fit(func, x, y, p0=p0, maxfev=10000, bounds=(-np.inf, np.inf))
    yhat = func(x, *popt)
    resid = y - yhat
    rss = float(np.sum(resid**2))
    n = len(y)
    k = len(popt)
    mse = rss / n
    rmse = float(np.sqrt(mse))
    tss = float(np.sum((y - np.mean(y))**2))
    r2 = float(1 - rss/tss) if tss > 0 else float("nan")
    aic = float(n*log(rss/n) + 2*k) if rss>0 else float("-inf")
    bic = float(n*log(rss/n) + k*log(n)) if rss>0 else float("-inf")
    return {"popt": [float(v) for v in popt], "rmse": rmse, "r2": r2, "aic": aic, "bic": bic}

def choose_model(I, EFF, prefer):
    results = {}
    for name, (fn, p0) in CANDIDATES.items():
        try:
            results[name] = fit_and_score(I, EFF, fn, p0)
        except Exception as e:
            results[name] = {"error": str(e)}
    # ranking by AIC then RMSE
    ok = [ (n, r) for n,r in results.items() if "error" not in r ]
    if not ok:
        raise RuntimeError("All models failed: " + str(results))
    if prefer != "auto" and prefer in CANDIDATES and "error" not in results.get(prefer, {}):
        best_name = prefer
    else:
        ok_sorted = sorted(ok, key=lambda t: (t[1]["aic"], t[1]["rmse"]))
        best_name = ok_sorted[0][0]
    return best_name, results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV input with columns I and EFF_percent or EFF_fraction")
    ap.add_argument("--outdir", default=".", help="Output directory")
    ap.add_argument("--model", default="auto", choices=["auto"] + list(CANDIDATES.keys()))
    ap.add_argument("--imin", type=float, default=None)
    ap.add_argument("--imax", type=float, default=None)
    ap.add_argument("--istep", type=float, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if "I" not in df.columns:
        raise ValueError("Missing column: I")
    if "EFF_percent" in df.columns:
        eff = df["EFF_percent"].astype(float)
    elif "EFF_fraction" in df.columns:
        eff = df["EFF_fraction"].astype(float) * 100.0
    else:
        raise ValueError("Provide EFF_percent (0-100) or EFF_fraction (0-1)")

    I = df["I"].astype(float).values
    EFF = eff.values

    # drop NaNs
    mask = np.isfinite(I) & np.isfinite(EFF)
    I = I[mask]; EFF = EFF[mask]
    if I.size < 4:
        raise ValueError("Need at least 4 data points for a stable fit")

    best_name, all_results = choose_model(I, EFF, args.model)
    fn, _ = CANDIDATES[best_name]
    popt = all_results[best_name]["popt"]

    # prediction grid
    i_min = float(np.min(I)) if args.imin is None else args.imin
    i_max = float(np.max(I)) if args.imax is None else args.imax
    i_step = (i_max - i_min)/200.0 if args.istep is None else args.istep
    grid = np.arange(i_min, i_max + 1e-9, i_step)
    pred = fn(grid, *popt)

    # outputs
    os.makedirs(args.outdir, exist_ok=True)
    params_path = os.path.join(args.outdir, "fit_params.json")
    preds_path = os.path.join(args.outdir, "predictions.csv")
    plot_path = os.path.join(args.outdir, "fit_plot.png")
    scores_path = os.path.join(args.outdir, "model_scores.json")

    with open(params_path, "w", encoding="utf-8") as f:
        json.dump({"chosen_model": best_name, "params": popt, "metrics": {k:all_results[best_name][k] for k in ["rmse","r2","aic","bic"]}}, f, ensure_ascii=False, indent=2)
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    pd.DataFrame({"I": grid, "EFF_percent_pred": pred}).to_csv(preds_path, index=False, encoding="utf-8-sig")

    # plot
    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(I, EFF, s=35, label="observed")
    plt.plot(grid, pred, label=f"{best_name} fit")
    plt.xlabel("I (A)")
    plt.ylabel("Efficiency (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)

    print("DONE:", {"params": params_path, "predictions": preds_path, "plot": plot_path, "scores": scores_path})

if __name__ == "__main__":
    main()
