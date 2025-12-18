from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def write_results_xlsx(
    output_path: str | Path,
    *,
    constants: list[dict] | None = None,
    statistics: dict | None = None,
    results_text: str = "",
    export_data: dict | None = None,
) -> None:
    """
    Write an XLSX file with results.

    This is a local (non-FastAPI) version of `backend_fastapi.main.export_results_xlsx`.
    """
    constants = constants or []
    statistics = statistics or {}
    export_data = export_data or {}

    output_path = Path(output_path)

    def as_dataframe(_name: str, data: Any, *, allow_none: bool = True) -> Optional[pd.DataFrame]:
        if data is None:
            return None if allow_none else pd.DataFrame()
        if isinstance(data, pd.DataFrame):
            return data
        if isinstance(data, pd.Series):
            return data.to_frame()
        if isinstance(data, dict):
            return pd.DataFrame(list(data.items()), columns=["key", "value"])
        return pd.DataFrame(data)

    def write_sheet(writer: pd.ExcelWriter, name: str, df: pd.DataFrame) -> None:
        df.to_excel(writer, sheet_name=name, index=True)

    is_nmr_export = any(
        key in export_data for key in ("Chemical_Shifts", "Calculated_Chemical_Shifts", "signal_names")
    )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # --- Common sheets ---
        if constants:
            pd.DataFrame(constants).to_excel(writer, sheet_name="Constants", index=False)

        if statistics:
            pd.DataFrame(list(statistics.items()), columns=["metric", "value"]).to_excel(
                writer, sheet_name="Statistics", index=False
            )

        if results_text.strip():
            lines = [ln.rstrip("\n") for ln in str(results_text).splitlines()]
            pd.DataFrame({"text": lines}).to_excel(writer, sheet_name="Report", index=False)

        # --- Export payload sheets (wx-like) ---
        if is_nmr_export:
            import numpy as np

            from backend_fastapi import nmr_processor

            modelo = as_dataframe("Model", export_data.get("modelo"), allow_none=False)
            Co = as_dataframe("Absorbent_species", export_data.get("Co"), allow_none=False)
            C = as_dataframe("All_species", export_data.get("C"), allow_none=False)
            C_T = as_dataframe("Tot_con_comp", export_data.get("C_T"), allow_none=False)
            dq = as_dataframe("Chemical_Shifts", export_data.get("Chemical_Shifts"), allow_none=False)
            dq_cal = as_dataframe(
                "Calculated_Chemical_Shifts",
                export_data.get("Calculated_Chemical_Shifts"),
                allow_none=False,
            )

            k_vals = export_data.get("k") or []
            percK = export_data.get("percK") or []
            k_names = [f"K{i+1}" for i in range(len(k_vals))]
            df_k = pd.DataFrame(
                {"Constants": list(k_vals), "Error (%)": list(percK)[: len(k_vals)]},
                index=k_names,
            )

            k_ini_vals = export_data.get("k_ini") or []
            k_ini_names = [f"K{i+1}" for i in range(len(k_ini_vals))]
            df_k_ini = pd.DataFrame({"Constants": list(k_ini_vals)}, index=k_ini_names)

            stats_table = export_data.get("stats_table")
            if stats_table:
                df_stats = pd.DataFrame(stats_table, columns=["metric", "Stats"]).set_index("metric")
            else:
                df_stats = pd.DataFrame(list(statistics.items()), columns=["metric", "Stats"]).set_index("metric")
            if "covfit" in df_stats.index:
                df_stats.loc["covfit", "Stats"] = str(df_stats.loc["covfit", "Stats"])

            coef = as_dataframe("Coefficients", export_data.get("Coefficients"), allow_none=True)
            if coef is None or coef.empty:
                try:
                    dq_arr = np.asarray(export_data.get("Chemical_Shifts"), dtype=float)
                    C_arr = np.asarray(export_data.get("C"), dtype=float)
                    C_T_arr = np.asarray(export_data.get("C_T"), dtype=float)
                    column_names = list(export_data.get("column_names") or [])
                    signal_names = list(export_data.get("signal_names") or [])

                    C_T_df = (
                        pd.DataFrame(C_T_arr, columns=column_names) if column_names else pd.DataFrame(C_T_arr)
                    )
                    D_cols, _ = nmr_processor.build_D_cols(
                        C_T_df, column_names, signal_names, default_idx=0
                    )

                    coef_mat = np.full((C_arr.shape[1], dq_arr.shape[1]), np.nan, dtype=float)
                    Xbase = np.asarray(C_arr, float)
                    finite_rows = np.isfinite(Xbase).all(axis=1)
                    nonzero_rows = np.linalg.norm(Xbase, axis=1) > 0.0
                    goodX = finite_rows & nonzero_rows
                    mask = np.isfinite(dq_arr) & np.isfinite(D_cols) & (np.abs(D_cols) > 0)

                    for j in range(dq_arr.shape[1]):
                        D = D_cols[:, j]
                        mj = mask[:, j] & goodX & np.isfinite(D) & (np.abs(D) > 0)
                        if int(mj.sum()) < 2:
                            continue
                        Xj = Xbase[mj, :] / D[mj][:, None]
                        y = dq_arr[mj, j]
                        coef_vec, *_ = np.linalg.lstsq(Xj, y, rcond=1e-10)
                        coef_mat[:, j] = coef_vec

                    coef = pd.DataFrame(coef_mat)
                except Exception:
                    coef = pd.DataFrame()

            if not isinstance(coef, pd.DataFrame):
                coef = pd.DataFrame(np.asarray(coef))
            if coef.shape[1] > 0 and all(isinstance(c, (int, np.integer)) for c in coef.columns):
                coef.columns = [f"coef_{i}" for i in range(1, coef.shape[1] + 1)]

            sheets: Dict[str, pd.DataFrame] = {
                "Model": modelo if modelo is not None else pd.DataFrame(),
                "Absorbent_species": Co if Co is not None else pd.DataFrame(),
                "All_species": C if C is not None else pd.DataFrame(),
                "Tot_con_comp": C_T if C_T is not None else pd.DataFrame(),
                "Chemical_Shifts": dq if dq is not None else pd.DataFrame(),
                "Calculated_Chemical_Shifts": dq_cal if dq_cal is not None else pd.DataFrame(),
                "Coefficients": coef if coef is not None else pd.DataFrame(),
                "K_calculated": df_k,
                "Init_guess_K": df_k_ini,
                "Stats": df_stats,
            }
            for name, df in sheets.items():
                write_sheet(writer, name, df)
        else:
            if export_data:
                modelo = as_dataframe("Model", export_data.get("modelo"))
                if modelo is not None:
                    modelo.to_excel(writer, sheet_name="Model", index=False)

                C = as_dataframe("Absorbent_species", export_data.get("C"))
                if C is not None:
                    C.to_excel(writer, sheet_name="Absorbent_species", index=False)

                Co = as_dataframe("All_species", export_data.get("Co"))
                if Co is not None:
                    Co.to_excel(writer, sheet_name="All_species", index=False)

                C_T = as_dataframe("Tot_con_comp", export_data.get("C_T"))
                if C_T is not None:
                    C_T.to_excel(writer, sheet_name="Tot_con_comp", index=False)

                A = export_data.get("A")
                nm = export_data.get("A_index") or export_data.get("nm")
                if A is not None:
                    dfA = as_dataframe("Molar_Absortivities", A, allow_none=True)
                    if dfA is not None:
                        if nm:
                            dfA.index = nm
                        dfA.to_excel(
                            writer,
                            sheet_name="Molar_Absortivities",
                            index_label="nm" if nm else None,
                        )

                Y = export_data.get("Y")
                if Y is not None:
                    dfY = as_dataframe("Y_observed", Y, allow_none=True)
                    if dfY is not None:
                        if nm:
                            dfY.index = nm
                        dfY.to_excel(writer, sheet_name="Y_observed", index_label="nm" if nm else None)

                yfit = export_data.get("yfit")
                if yfit is not None:
                    dfPhi = as_dataframe("Y_calculated", yfit, allow_none=True)
                    if dfPhi is not None:
                        if nm:
                            dfPhi.index = nm
                        dfPhi.to_excel(
                            writer,
                            sheet_name="Y_calculated",
                            index_label="nm" if nm else None,
                        )

                k_vals = export_data.get("k") or []
                percK = export_data.get("percK") or []
                if k_vals:
                    names = [f"K{i+1}" for i in range(len(k_vals))]
                    pd.DataFrame(
                        {"log10K": k_vals, "percK(%)": percK[: len(k_vals)]},
                        index=names,
                    ).to_excel(writer, sheet_name="K_calculated")

                k_ini = export_data.get("k_ini") or []
                if k_ini:
                    names_ini = [f"k{i+1}" for i in range(len(k_ini))]
                    pd.DataFrame({"init_guess": k_ini}, index=names_ini).to_excel(
                        writer, sheet_name="Init_guess_K"
                    )

