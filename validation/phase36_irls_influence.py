"""IRLS weight disparities and leave-one-signal-out literature diagnostics."""
import contextlib
import io
import json
import tempfile
from pathlib import Path

import numpy as np
from hmfit_core.api import run_nmr_fit
from hmfit_core.processors import nmr_processor
from validation import literature_benchmark as lit


def run():
    nmr_processor.generate_figure_base64 = lambda *a, **k: ''
    nmr_processor.generate_figure2_base64 = lambda *a, **k: ''
    output = {}
    with tempfile.TemporaryDirectory() as td:
        for cid in lit.case_ids():
            case = lit.load_case(cid); ok, reason = lit.supported(case)
            if not ok:
                output[cid] = {'unsupported':reason}; continue
            conc, shifts = lit.numeric_table(case)
            cfg = lit.gui_config(case, str(Path(td)/'case.xlsx'), conc, shifts)
            names = cfg['signal_names']; group = {'signal_names':names}
            choices = [None] + (list(range(len(names))) if cid in ('r09a-r1-3sp','r09a-r1-11-3n') else [])
            for excluded in choices:
                modes = ('irls', 'unit') if excluded is not None else ('irls',)
                row = {}
                for mode in modes:
                    keep = [i for i in range(len(names)) if i != excluded]
                    opt = dict(cfg, weights_mode=mode, signal_names=[names[i] for i in keep],
                               signal_data=[cfg['signal_data'][i] for i in keep],
                               stoichiometry_map=np.asarray(cfg['stoichiometry_map'])[:, keep].tolist())
                    with contextlib.redirect_stdout(io.StringIO()):
                        fit = run_nmr_fit(opt, progress_cb=lambda _: None)
                    if not fit.get('success'):
                        row[mode] = {'success':False, 'error':fit.get('error')}; continue
                    ed = fit['export_data']; w = np.asarray(ed['irls_weights_final'], float)
                    positive = w[w > 0]
                    row[mode] = {'success':fit['optimizer_result']['success'], 'log_beta':ed['k'],
                                 'se':ed['SE_log10K'], 'weights':w.tolist(),
                                 'weight_ratio':float(w.max()/positive.min()) if positive.size else None,
                                 'cycles':ed['irls_cycles']}
                group['all' if excluded is None else 'omit:'+names[excluded]] = row
            output[cid] = group
            print('completed',cid,flush=True)
    return output


if __name__ == '__main__':
    Path('validation/results/phase36_irls_influence.json').write_text(json.dumps(run(), indent=2, allow_nan=False)+'\n')
