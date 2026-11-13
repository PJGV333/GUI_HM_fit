"""Historical invalid HC results must not be presented as valid evidence."""
import json


def test_report_marks_historical_hc_invalid_and_excludes_it_from_plot(tmp_path, monkeypatch):
    from matplotlib.axes import Axes
    from validation import make_unc_report

    results = tmp_path / "results"
    figures = tmp_path / "figures"
    case = results / "case"
    case.mkdir(parents=True)
    data = {
        "scen": {"name": "case", "estimator": "ols"},
        "summary": {
            "n": 10,
            "theta": {"bias": 0.0, "mc_se_bias": 0.1, "sd": 0.1, "rmse": 0.1},
            "analytic": {"mean_se": 0.1, "cov95": 0.9, "mean_width95": 0.4},
            "hc2": {"mean_se": 0.2, "cov95": 0.8, "mean_width95": 0.6},
            "hc3": {"mean_se": 0.3, "cov95": 0.7, "mean_width95": 0.8},
        },
    }
    summary = case / "summary.json"
    original = json.dumps(data)
    summary.write_text(original)
    monkeypatch.setattr(make_unc_report, "RES", results)
    monkeypatch.setattr(make_unc_report, "FIGS", figures)
    labels = []
    annotate = Axes.annotate

    def record_label(self, text, *args, **kwargs):
        labels.append(text)
        return annotate(self, text, *args, **kwargs)

    monkeypatch.setattr(Axes, "annotate", record_label)
    make_unc_report.main()
    table = (results / "table.md").read_text()
    assert "hc2 (INVALIDADO)" in table
    assert "hc3 (INVALIDADO)" in table
    assert "0.2000" not in table and "0.3000" not in table
    assert labels == ["analytic"]
    assert summary.read_text() == original
    assert (figures / "cov_case.png").exists()
