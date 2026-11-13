"""Five-run Powell/LM comparison with fixed unit NMR weights."""
import json
import tempfile
from pathlib import Path

from validation import literature_benchmark as lit
from validation.phase31_comparison import _measure, _nmr_phase1
from validation.phase36_diagnosis import _nmr
from hmfit_core.processors import nmr_processor as nmr


def run():
    nmr.generate_figure_base64 = lambda *a, **k: ''
    nmr.generate_figure2_base64 = lambda *a, **k: ''
    results = {}
    with tempfile.TemporaryDirectory() as td:
        for cid in lit.case_ids():
            case = lit.load_case(cid); ok, reason = lit.supported(case)
            if not ok:
                results[cid] = {'unsupported':reason}; continue
            conc, shifts = lit.numeric_table(case)
            cfg = lit.gui_config(case, str(Path(td)/'case.xlsx'), conc, shifts)
            cfg['weights_mode'] = 'unit'
            comparison = _measure('NMR', cfg, 5)
            starts = {'powell':comparison['powell'], 'lm':comparison['explicit_lm']}
            diagnosis = _nmr(cfg, starts)
            results[cid] = {'comparison':comparison, 'cross_start':diagnosis}
            print('completed', cid, flush=True)
        for name, model, truth, start, fixed, n in (
            ('phase1-11', [[1,0],[0,1],[1,1]], [3.], [2.5], [False], 12),
            ('phase1-12-fixed', [[1,0],[0,1],[1,1],[1,2]], [4.,7.], [3.5,7.], [False,True], 15),
        ):
            path = Path(td) / 'phase1.xlsx'
            _nmr_phase1(path, model, truth, n=n)
            cfg = {'file_path':str(path), 'nmr_sheet':'S', 'conc_sheet':'S',
                   'column_names':['H','G'], 'signal_names':['s1','s2'], 'receptor_label':'H',
                   'guest_label':'G', 'modelo':model, 'non_abs_species':[1], 'algorithm':'Newton-Raphson',
                   'model_settings':'Free', 'initial_k':start, 'bounds':[{'min':0.,'max':8.} for _ in start],
                   'k_fixed':fixed, 'weights_mode':'unit'}
            comparison = _measure('NMR', cfg, 5)
            results[name] = {'comparison':comparison,
                             'cross_start':_nmr(cfg, {'powell':comparison['powell'], 'lm':comparison['explicit_lm']})}
    return results


if __name__ == '__main__':
    result = run()
    Path('validation/results/phase36_nmr_unit.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
