from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from scipy.interpolate import interp1d

from config import APP_ROOT


def _candidate_db_roots() -> List[Path]:
    """Return possible locations of the refractiveindex.info database."""
    bases = [APP_ROOT, APP_ROOT.parent, APP_ROOT.parent.parent, APP_ROOT.parent.parent.parent]
    candidates: List[Path] = []

    # Common locations under the app folder
    candidates.append(APP_ROOT / "data" / "refractiveindex.info-database")
    candidates.append(APP_ROOT / "refractiveindex.info-database")

    # Climb up a few levels and try both direct and "refractive_index" wrapper
    for base in bases:
        candidates.append(base / "refractiveindex.info-database")
        candidates.append(base / "refractive_index" / "refractiveindex.info-database")

    # Deduplicate while preserving order
    seen = set()
    unique_candidates = []
    for path in candidates:
        if path not in seen:
            seen.add(path)
            unique_candidates.append(path)

    return unique_candidates


@lru_cache(maxsize=1)
def get_db_root() -> Optional[Path]:
    """Locate the refractiveindex.info database root folder."""
    for path in _candidate_db_roots():
        if path.exists():
            return path
    return None


@lru_cache(maxsize=1)
def load_catalog() -> Tuple[Optional[Path], List[Dict[str, str]]]:
    """Load and flatten the material catalog for searching."""
    root = get_db_root()
    if not root:
        return None, []

    catalog_path = root / "database" / "catalog-nk.yml"
    if not catalog_path.exists():
        return root, []

    try:
        library = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    except Exception:
        return root, []

    materials: List[Dict[str, str]] = []
    for shelf in library or []:
        if "SHELF" not in shelf:
            continue
        shelf_name = shelf.get("name", shelf.get("SHELF"))

        for book in shelf.get("content", []):
            if "BOOK" not in book:
                continue
            book_name = book.get("name", book.get("BOOK"))

            for page in book.get("content", []):
                if "PAGE" not in page:
                    continue
                page_name = page.get("name", page.get("PAGE"))
                data_path = page.get("data")
                if not data_path:
                    continue

                label = f"{shelf_name} > {book_name} > {page_name}"
                materials.append(
                    {
                        "label": label,
                        "shelf": shelf_name,
                        "book": book_name,
                        "page": page_name,
                        "data_path": data_path,
                    }
                )

    return root, materials


def search_materials(query: str) -> Tuple[Optional[Path], List[Dict[str, str]]]:
    """Search materials by keyword (case-insensitive)."""
    root, materials = load_catalog()
    if not materials:
        return root, []

    query_lower = query.strip().lower()
    if not query_lower:
        return root, materials

    filtered = [
        m
        for m in materials
        if query_lower in m["label"].lower()
        or query_lower in m["page"].lower()
        or query_lower in m["book"].lower()
        or query_lower in m["shelf"].lower()
    ]
    return root, filtered


@lru_cache(maxsize=256)
def load_material_data(data_path: str) -> Optional[dict]:
    """Load a material YAML data file."""
    root = get_db_root()
    if not root:
        return None

    full_path = root / "database" / "data" / data_path
    if not full_path.exists():
        return None

    try:
        return yaml.safe_load(full_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _parse_wavelength_range(data: dict) -> Optional[Tuple[float, float]]:
    if "wavelength_range" not in data:
        return None
    try:
        values = [float(x) for x in str(data["wavelength_range"]).split()]
        if len(values) >= 2:
            return values[0], values[1]
    except Exception:
        return None
    return None


def get_refractive_index(data_list: List[dict], wavelength: float) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculate n and k for a given wavelength (in micrometers).

    data_list: List of data dictionaries from the YAML file.
    wavelength: float, wavelength in micrometers.
    """
    n = None
    k = None

    for data in data_list:
        datatype_parts = str(data.get("type", "")).split()
        if not datatype_parts:
            continue
        type_code = datatype_parts[0]

        wl_range = _parse_wavelength_range(data)
        if wl_range and not (wl_range[0] <= wavelength <= wl_range[1]):
            continue

        if type_code == "tabulated":
            rows = str(data.get("data", "")).strip().split("\n")
            splitrows = [list(map(float, r.split())) for r in rows if r.strip()]
            if not splitrows:
                continue
            data_np = np.array(splitrows)

            wl_data = data_np[:, 0]
            if len(datatype_parts) > 1 and datatype_parts[1] == "n":
                n_data = data_np[:, 1]
                f = interp1d(wl_data, n_data, kind="linear", bounds_error=False, fill_value=np.nan)
                val = f(wavelength)
                if not np.isnan(val):
                    n = float(val)
            elif len(datatype_parts) > 1 and datatype_parts[1] == "k":
                k_data = data_np[:, 1]
                f = interp1d(wl_data, k_data, kind="linear", bounds_error=False, fill_value=np.nan)
                val = f(wavelength)
                if not np.isnan(val):
                    k = float(val)
            elif len(datatype_parts) > 1 and datatype_parts[1] == "nk":
                n_data = data_np[:, 1]
                k_data = data_np[:, 2]
                fn = interp1d(wl_data, n_data, kind="linear", bounds_error=False, fill_value=np.nan)
                fk = interp1d(wl_data, k_data, kind="linear", bounds_error=False, fill_value=np.nan)
                val_n = fn(wavelength)
                val_k = fk(wavelength)
                if not np.isnan(val_n):
                    n = float(val_n)
                if not np.isnan(val_k):
                    k = float(val_k)

        elif type_code == "formula":
            coefficients = np.array(str(data.get("coefficients", "")).split()).astype(float)
            C = np.pad(coefficients, (0, 17 - len(coefficients)), "constant")
            C = np.insert(C, 0, 0)

            wl = wavelength
            formula_type = datatype_parts[1] if len(datatype_parts) > 1 else ""

            try:
                if formula_type == "1":
                    n_sq = (
                        1
                        + C[1]
                        + C[2] / (1 - (C[3] / wl) ** 2)
                        + C[4] / (1 - (C[5] / wl) ** 2)
                        + C[6] / (1 - (C[7] / wl) ** 2)
                        + C[8] / (1 - (C[9] / wl) ** 2)
                        + C[10] / (1 - (C[11] / wl) ** 2)
                        + C[12] / (1 - (C[13] / wl) ** 2)
                        + C[14] / (1 - (C[15] / wl) ** 2)
                        + C[16] / (1 - (C[17] / wl) ** 2)
                    )
                    n = n_sq**0.5
                elif formula_type == "2":
                    n_sq = (
                        1
                        + C[1]
                        + C[2] / (1 - C[3] / wl**2)
                        + C[4] / (1 - C[5] / wl**2)
                        + C[6] / (1 - C[7] / wl**2)
                        + C[8] / (1 - C[9] / wl**2)
                        + C[10] / (1 - C[11] / wl**2)
                        + C[12] / (1 - C[13] / wl**2)
                        + C[14] / (1 - C[15] / wl**2)
                        + C[16] / (1 - C[17] / wl**2)
                    )
                    n = n_sq**0.5
                elif formula_type == "3":
                    n_sq = (
                        C[1]
                        + C[2] * wl**C[3]
                        + C[4] * wl**C[5]
                        + C[6] * wl**C[7]
                        + C[8] * wl**C[9]
                        + C[10] * wl**C[11]
                        + C[12] * wl**C[13]
                        + C[14] * wl**C[15]
                        + C[16] * wl**C[17]
                    )
                    n = n_sq**0.5
                elif formula_type == "4":
                    n_sq = (
                        C[1]
                        + C[2] * wl**C[3] / (wl**2 - C[4] ** C[5])
                        + C[6] * wl**C[7] / (wl**2 - C[8] ** C[9])
                        + C[10] * wl**C[11]
                        + C[12] * wl**C[13]
                        + C[14] * wl**C[15]
                        + C[16] * wl**C[17]
                    )
                    n = n_sq**0.5
                elif formula_type == "5":
                    n = (
                        C[1]
                        + C[2] * wl**C[3]
                        + C[4] * wl**C[5]
                        + C[6] * wl**C[7]
                        + C[8] * wl**C[9]
                        + C[10] * wl**C[11]
                    )
                elif formula_type == "6":
                    n = (
                        1
                        + C[1]
                        + C[2] / (C[3] - wl**-2)
                        + C[4] / (C[5] - wl**-2)
                        + C[6] / (C[7] - wl**-2)
                        + C[8] / (C[9] - wl**-2)
                        + C[10] / (C[11] - wl**-2)
                    )
                elif formula_type == "7":
                    n = (
                        C[1]
                        + C[2] / (wl**2 - 0.028)
                        + C[3] / (wl**2 - 0.028) ** 2
                        + C[4] * wl**2
                        + C[5] * wl**4
                        + C[6] * wl**6
                    )
                elif formula_type == "8":
                    tmp = C[1] + C[2] * wl**2 / (wl**2 - C[3]) + C[4] * wl**2
                    n = ((2 * tmp + 1) / (1 - tmp)) ** 0.5
                elif formula_type == "9":
                    n = (
                        C[1]
                        + C[2] / (wl**2 - C[3])
                        + C[4] * (wl - C[5]) / ((wl - C[5]) ** 2 + C[6])
                    ) ** 0.5
            except Exception:
                continue

        # If both found we can exit early
        if n is not None and k is not None:
            break

    # If n has value but k missing, treat as lossless
    if n is not None and k is None:
        k = 0.0

    return n, k


def get_wavelength_span(data_list: List[dict]) -> Optional[Tuple[float, float]]:
    """Return the min/max wavelength span found in the data list."""
    min_wl = float("inf")
    max_wl = float("-inf")

    for data in data_list:
        wl_range = _parse_wavelength_range(data)
        if wl_range:
            min_wl = min(min_wl, wl_range[0])
            max_wl = max(max_wl, wl_range[1])
            continue

        datatype_parts = str(data.get("type", "")).split()
        if datatype_parts and datatype_parts[0] == "tabulated":
            rows = str(data.get("data", "")).strip().split("\n")
            splitrows = [list(map(float, r.split())) for r in rows if r.strip()]
            if splitrows:
                data_np = np.array(splitrows)
                min_wl = min(min_wl, float(np.min(data_np[:, 0])))
                max_wl = max(max_wl, float(np.max(data_np[:, 0])))

    if min_wl == float("inf") or max_wl == float("-inf"):
        return None
    return min_wl, max_wl
