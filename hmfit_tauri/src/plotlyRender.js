export function renderPlotly(div, figure, extraConfig = {}) {
  if (!div) return Promise.resolve();
  if (!window.Plotly) throw new Error("Plotly not available");

  const fig = figure || {};
  fig.data = fig.data || [];
  fig.layout = fig.layout || {};

  if (fig.layout.uirevision === undefined) fig.layout.uirevision = "hmfit-uirev";

  const config = {
    responsive: true,
    displaylogo: false,
    scrollZoom: true,
    editable: true,
    edits: {
      titleText: true,
      axisTitleText: true,
      legendText: true,
    },
    ...extraConfig,
  };

  return window.Plotly.react(div, fig.data, fig.layout, config);
}

export function applyPlotOverrides(div, overrides) {
  if (!div || !window.Plotly || !overrides) return;

  const relayout = {};
  if (overrides.titleText !== undefined) relayout["title.text"] = overrides.titleText;
  if (overrides.xLabel !== undefined) relayout["xaxis.title.text"] = overrides.xLabel;
  if (overrides.yLabel !== undefined) relayout["yaxis.title.text"] = overrides.yLabel;

  if (Object.keys(relayout).length) window.Plotly.relayout(div, relayout);

  const traceNames = overrides.traceNames || {};
  for (const [idx, name] of Object.entries(traceNames)) {
    const i = Number(idx);
    if (!Number.isFinite(i)) continue;
    window.Plotly.restyle(div, { name }, [i]);
  }
}

