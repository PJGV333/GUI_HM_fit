from __future__ import annotations

from typing import Any

import numpy as np

from hmfit.kinetics.data.fit_dataset import KineticsFitDataset


def _series_id(prefix: str, *parts: object) -> str:
    tail = "_".join(str(p) for p in parts if p is not None)
    return f"{prefix}_{tail}" if tail else prefix


def build_kinetics_plot_sources(result: dict[str, Any]) -> dict[str, Any]:
    datasets = [ds for ds in (result.get("datasets") or []) if isinstance(ds, KineticsFitDataset)]
    species = list(result.get("dynamic_species") or [])

    conc_data = {"seriesOptions": [], "seriesData": {}, "x_label": "", "y_label": "concentration"}
    d_fit_data = {"seriesOptions": [], "seriesData": {}, "x_label": "", "y_label": "signal"}
    resid_data = {"seriesOptions": [], "seriesData": {}, "x_label": "", "y_label": "residual"}
    a_prof_data = {"seriesOptions": [], "seriesData": {}, "x_label": "", "y_label": "A"}

    for d_idx, dataset in enumerate(datasets):
        name = dataset.name or f"dataset {d_idx + 1}"
        t = np.asarray(dataset.t, dtype=float).ravel().tolist()
        time_unit = dataset.time_unit or ""
        if time_unit and not conc_data["x_label"]:
            conc_data["x_label"] = f"time [{time_unit}]"
        if time_unit and not d_fit_data["x_label"]:
            d_fit_data["x_label"] = f"time [{time_unit}]"
        if time_unit and not resid_data["x_label"]:
            resid_data["x_label"] = f"time [{time_unit}]"

        C = dataset.fit_C
        if C is not None and species:
            C_arr = np.asarray(C, dtype=float)
            for s_idx, sp in enumerate(species):
                if s_idx >= C_arr.shape[1]:
                    continue
                series_id = _series_id("c", d_idx, s_idx)
                label = f"{name}: {sp}"
                conc_data["seriesOptions"].append({"id": series_id, "label": label})
                conc_data["seriesData"][series_id] = {
                    "x": t,
                    "y": C_arr[:, s_idx].tolist(),
                    "mode": "lines",
                }

        D = dataset.D
        D_hat = dataset.fit_D_hat
        labels = list(dataset.channel_labels or [])
        if D is not None and D_hat is not None:
            D_arr = np.asarray(D, dtype=float)
            D_hat_arr = np.asarray(D_hat, dtype=float)
            for ch in range(D_arr.shape[1]):
                label = labels[ch] if ch < len(labels) else f"ch{ch}"
                base_id = _series_id("d", d_idx, ch)
                data_id = f"{base_id}_data"
                fit_id = f"{base_id}_fit"
                d_fit_data["seriesOptions"].append({"id": data_id, "label": f"{name}: {label} data"})
                d_fit_data["seriesOptions"].append({"id": fit_id, "label": f"{name}: {label} fit"})
                d_fit_data["seriesData"][data_id] = {
                    "x": t,
                    "y": D_arr[:, ch].tolist(),
                    "mode": "markers",
                }
                d_fit_data["seriesData"][fit_id] = {
                    "x": t,
                    "y": D_hat_arr[:, ch].tolist(),
                    "mode": "lines",
                    "style": {"linestyle": ":"},
                }

            resid = dataset.fit_residuals
            if resid is None:
                resid = D_arr - D_hat_arr
            resid_arr = np.asarray(resid, dtype=float)
            for ch in range(resid_arr.shape[1]):
                label = labels[ch] if ch < len(labels) else f"ch{ch}"
                series_id = _series_id("r", d_idx, ch)
                resid_data["seriesOptions"].append({"id": series_id, "label": f"{name}: {label}"})
                resid_data["seriesData"][series_id] = {
                    "x": t,
                    "y": resid_arr[:, ch].tolist(),
                    "mode": "lines",
                }

        A = dataset.fit_A
        x = dataset.x
        x_unit = dataset.x_unit or ""
        if x is not None and A is not None and species:
            x_vals = np.asarray(x, dtype=float).ravel().tolist()
            if x_unit and not a_prof_data["x_label"]:
                a_prof_data["x_label"] = f"x [{x_unit}]"
            A_arr = np.asarray(A, dtype=float)
            for s_idx, sp in enumerate(species):
                if s_idx >= A_arr.shape[0]:
                    continue
                series_id = _series_id("a", d_idx, s_idx)
                label = f"{name}: {sp}"
                a_prof_data["seriesOptions"].append({"id": series_id, "label": label})
                a_prof_data["seriesData"][series_id] = {
                    "x": x_vals,
                    "y": A_arr[s_idx, :].tolist(),
                    "mode": "lines",
                }

    if not conc_data["x_label"]:
        conc_data["x_label"] = "time"
    if not d_fit_data["x_label"]:
        d_fit_data["x_label"] = "time"
    if not resid_data["x_label"]:
        resid_data["x_label"] = "time"
    if not a_prof_data["x_label"]:
        a_prof_data["x_label"] = "channel"

    return {
        "kinetics_concentrations": conc_data,
        "kinetics_d_fit": d_fit_data,
        "kinetics_residuals": resid_data,
        "kinetics_a_profiles": a_prof_data,
    }
