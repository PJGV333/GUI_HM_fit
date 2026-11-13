"""Compare NMR weight modes on the literature bank with Powell."""
import argparse
import contextlib
import json
import os
import tempfile
from pathlib import Path

from hmfit_core.api import run_nmr_fit
from hmfit_core.processors import nmr_processor
from validation import literature_benchmark as lit


def run():
    nmr_processor.generate_figure_base64 = lambda *args, **kwargs: ''
    nmr_processor.generate_figure2_base64 = lambda *args, **kwargs: ''
    rows = {}
    with tempfile.TemporaryDirectory() as td:
        for cid in lit.case_ids():
            case = lit.load_case(cid)
            ok, why = lit.supported(case)
            if not ok:
                rows[cid] = {'unsupported': why}
                continue
            conc, shifts = lit.numeric_table(case)
            cfg = lit.gui_config(case, os.path.join(td, 'case.xlsx'), conc, shifts)
            modes = {}
            for mode in ('irls', 'unit'):
                with open(os.devnull, 'w') as sink, contextlib.redirect_stdout(sink):
                    result = run_nmr_fit(dict(cfg, weights_mode=mode), progress_cb=lambda _: None)
                if not result.get('success'):
                    modes[mode] = {'success': False, 'error': result.get('error')}
                    continue
                ed = result['export_data']
                names = [s['name'] for s in case['species']]
                reference = case['reference']['log_beta']
                modes[mode] = {'success': result['optimizer_result']['success'],
                               'log_beta': ed['k'], 'se': ed['SE_log10K'],
                               'abs_delta_reference': [abs(value - reference[name]['log_beta'])
                                                       for name, value in zip(names, ed['k'])],
                               'cycles': ed['irls_cycles'], 'final_weights': ed['irls_weights_final']}
            rows[cid] = modes
            print('completed', cid, flush=True)
    return {'optimizer': 'powell', 'cases': rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = run()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
