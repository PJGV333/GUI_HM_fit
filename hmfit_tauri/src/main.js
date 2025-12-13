import { save, open } from '@tauri-apps/api/dialog';
import { writeBinaryFile, writeTextFile, readTextFile } from '@tauri-apps/api/fs';
import "./style.css";
import { BACKEND_BASE_URL, WS_BASE_URL, describeBackendTarget } from "./backend/config";
import Plotly from "plotly.js-dist-min";
import { renderPlotly, applyPlotOverrides } from "./plotlyRender";
window.Plotly = Plotly;

// --- State Factories ---

function makeDefaultPlotState() {
  return {
    availablePlots: [],   // [{id, title, kind}, ...]
    plotData: {},         // {spec: {...}, nmr: {...}}
    activePlotIndex: 0,
    controls: {
      distXAxisId: "titrant_total",     // species distribution X axis
      distYSelected: new Set(),         // species selected for Y axis
      nmrSignalsSelected: new Set(),    // NMR shifts fit - signals to display
      nmrResidSelected: new Set(),      // NMR residuals - signals to display
    }
  };
}

function makeDefaultModuleState() {
  return {
    file: null,
    fileName: "",
    // Available data lists for restoration
    availableSheets: [],
    availableColumns: [],
    availableSignals: [],

    sheetSpectra: "",
    sheetConc: "",
    sheetNmr: "",
    // Spectroscopy axis + channels
    axisValues: [],
    axisMin: null,
    axisMax: null,
    axisCount: null,
    channelsRaw: "All",
    channelsResolved: [],
    channelsMode: "all",
    // Input values
    efaEnabled: true,
    efaEigenvalues: 0,
    nComponents: 0,
    nSpecies: 0,
    // Grid data
    modelGrid: [], // We might need to store this if we want to persist it
    optGrid: [],
    // Selections
    selectedColumns: [],
    selectedSignals: [],
    receptor: "",
    guest: "",
    // Console & Plots
    consoleText: "",
    plotState: makeDefaultPlotState(),
    resultsText: "",
    resultsPayload: null,
    lastResponse: null,
  };
}

const state = {
  activeModule: "spectroscopy", // "spectroscopy" | "nmr"
  activeSubtab: "model",        // "model" | "optimization" | "plots"
  modules: {
    spectroscopy: makeDefaultModuleState(),
    nmr: makeDefaultModuleState(),
  },
  plotOverrides: {
    spectroscopy: {},
    nmr: {},
  },
  plotDefaults: {
    spectroscopy: {},
    nmr: {},
  },
};

// Helper to access active module state
function M() {
  return state.modules[state.activeModule];
}

// Helper functions for consistent data access
function getActivePlot() {
  const ps = M().plotState;
  return ps.availablePlots[ps.activePlotIndex] || null;
}

function getActivePlotData() {
  const plot = getActivePlot();
  if (!plot) return null;
  const modKey = state.activeModule === 'nmr' ? 'nmr' : 'spec';
  const moduleData = M().plotState.plotData?.[modKey] || {};
  return moduleData?.[plot.id] || null;
}

// WebSocket for progress streaming
let progressWs = null;
let isProcessing = false;

// --- Backend client (shared fetch + helpers) ---
async function ensureBackendReachable(baseError) {
  try {
    const healthResp = await fetch(`${BACKEND_BASE_URL}/health`);
    if (!healthResp.ok) {
      const detail = await healthResp.text();
      throw new Error(
        `/health respondió ${healthResp.status}: ${detail || healthResp.statusText}`,
      );
    }
  } catch (healthErr) {
    throw new Error(
      `Backend no accesible en ${BACKEND_BASE_URL}. Asegúrate de que el servidor FastAPI ` +
      `esté levantado en ${describeBackendTarget()} y que no haya un firewall bloqueándolo. ` +
      `Detalle: ${healthErr?.message || healthErr}`,
    );
  }

  // Health is reachable; bubble the original error for context.
  throw baseError;
}

async function assertBackendAvailable() {
  try {
    const healthResp = await fetch(`${BACKEND_BASE_URL}/health`);
    if (!healthResp.ok) {
      const detail = await healthResp.text();
      throw new Error(
        `/health respondió ${healthResp.status}: ${detail || healthResp.statusText}`,
      );
    }
  } catch (err) {
    throw new Error(
      `Backend no accesible en ${BACKEND_BASE_URL}. ` +
      `Asegúrate de que el servidor FastAPI esté levantado en ${describeBackendTarget()} ` +
      `y que no haya un firewall bloqueándolo. Detalle: ${err?.message || err}`,
    );
  }
}

async function callBackend(path, options = {}) {
  const { method = "GET", body } = options;
  const url = `${BACKEND_BASE_URL}${path}`;
  let resp;

  try {
    resp = await fetch(url, { method, body, credentials: "include", mode: "cors" });
  } catch (err) {
    console.error(`[HM Fit] Network/CORS error calling ${url}:`, err);
    await ensureBackendReachable(err);
  }

  if (!resp.ok) {
    let detail;
    try {
      const errJson = await resp.json();
      detail = errJson.detail || JSON.stringify(errJson);
    } catch (_) {
      detail = await resp.text();
    }

    const error = new Error(`HTTP ${resp.status} ${resp.statusText}: ${detail}`);
    error.status = resp.status;
    error.statusText = resp.statusText;
    error.body = detail;
    console.error(`[HM Fit] Backend error for ${url}:`, {
      status: resp.status,
      statusText: resp.statusText,
      body: detail,
    });
    throw error;
  }

  return resp.json();
}

const backendApi = {
  listSheets(file, mode = "spectroscopy") {
    const formData = new FormData();
    formData.append("file", file);
    const path = mode === "nmr" ? "/nmr/list_sheets" : "/list_sheets";
    return callBackend(path, { method: "POST", body: formData });
  },
  listColumns(file, sheetName, mode = "spectroscopy") {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("sheet_name", sheetName);
    const path = mode === "nmr" ? "/nmr/list_columns" : "/list_columns";
    return callBackend(path, { method: "POST", body: formData });
  },
  listSpectroscopyAxis(file, sheetName) {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("sheet_name", sheetName);
    return callBackend("/spectroscopy/list_axis", { method: "POST", body: formData });
  },
  processSpectroscopy(formData) {
    return callBackend("/process_spectroscopy", { method: "POST", body: formData });
  },
  processNmr(formData) {
    return callBackend("/process_nmr", { method: "POST", body: formData });
  },
};

function connectWebSocket() {
  if (progressWs && progressWs.readyState === WebSocket.OPEN) return;

  progressWs = new WebSocket(`${WS_BASE_URL}/ws/progress`);

  progressWs.onopen = () => {
    console.log("WebSocket connected");
  };

  progressWs.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === "progress") {
        appendLog(data.message);
      }
    } catch (err) {
      console.error("WebSocket message error:", err);
    }
  };

  progressWs.onerror = (error) => {
    console.error("WebSocket error:", error);
  };

  progressWs.onclose = () => {
    console.log("WebSocket disconnected");
    // Reconnect after 2 seconds
    setTimeout(connectWebSocket, 2000);
  };
}

function appendToConsole(text) {
  // Save to active module state
  const m = M();
  if (m) {
    m.consoleText += text;
  }

  const pre = document.getElementById("log-output");
  if (!pre) return;
  // Ensure we append a newline if the previous content didn't end with one
  if (pre.textContent && !pre.textContent.endsWith('\n')) {
    pre.textContent += '\n';
  }
  pre.textContent += text;
  // Auto-scroll to bottom
  pre.scrollTop = pre.scrollHeight;
}

function setConsoleText(text) {
  const pre = document.getElementById("log-output");
  if (pre) {
    pre.textContent = text;
    pre.scrollTop = pre.scrollHeight;
  }
}

function log(text) {
  appendToConsole(text + "\n");
}

function appendLog(text) {
  appendToConsole(text + "\n");
}

function scrollDiagnosticsToBottom() {
  const pre = document.getElementById("log-output");
  if (!pre) return;
  pre.scrollTop = pre.scrollHeight;
}

function setProcessing(active) {
  isProcessing = active;
  const processBtn = document.getElementById("process-btn");
  if (processBtn) {
    processBtn.disabled = active;
    processBtn.textContent = active ? "Processing..." : "Process Data";
  }
}

async function pingBackend() {
  log("Consultando /health …");
  try {
    const resp = await fetch(`${BACKEND_BASE_URL}/health`);
    const data = await resp.json();
    log(JSON.stringify(data, null, 2));
  } catch (err) {
    log(`Error al consultar /health: ${err}`);
  }
}

async function dummyFit() {
  log("Enviando ejemplo a /dummy_fit …");
  try {
    const payload = {
      x: [0, 1, 2, 3],
      y: [0.1, 0.5, 0.9, 1.2],
    };
    const resp = await fetch(`${BACKEND_BASE_URL}/dummy_fit`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await resp.json();
    log(JSON.stringify(data, null, 2));
  } catch (err) {
    log(`Error al llamar /dummy_fit: ${err}`);
  }
}

function resetCalculation() {
  log("Esperando...");
}

function updateUI() {
  // Tabs: módulo (Spectroscopy / NMR)
  document.querySelectorAll("[data-module-tab]").forEach((btn) => {
    const mod = btn.dataset.moduleTab;
    if (mod === state.activeModule) {
      btn.classList.add("tab-active");
    } else {
      btn.classList.remove("tab-active");
    }
  });

  // Subtabs: Model / Optimization
  document.querySelectorAll("[data-subtab]").forEach((btn) => {
    const sub = btn.dataset.subtab;
    if (sub === state.activeSubtab) {
      btn.classList.add("tab-active");
    } else {
      btn.classList.remove("tab-active");
    }
  });

  document.querySelectorAll("[data-subtab-panel]").forEach((panel) => {
    const sub = panel.dataset.subtabPanel;
    if (sub === state.activeSubtab) {
      panel.classList.add("subtab-visible");
    } else {
      panel.classList.remove("subtab-visible");
    }
  });

  // EFA solo en Spectroscopy
  const efaRow = document.querySelector(".efa-row");
  const spectraSheetRow = document.getElementById("spectra-sheet-select")?.closest(".field");
  const channelsRow = document.getElementById("spectroscopy-channels-row");
  const nmrSheetRow = document.getElementById("nmr-sheet-row");
  const nmrSignalsRow = document.getElementById("nmr-signals-row");

  if (state.activeModule === "nmr") {
    if (efaRow) efaRow.classList.add("hidden");
    if (spectraSheetRow) spectraSheetRow.classList.add("hidden");
    if (channelsRow) channelsRow.classList.add("hidden");
    if (nmrSheetRow) nmrSheetRow.classList.remove("hidden");
    if (nmrSignalsRow) nmrSignalsRow.classList.remove("hidden");
  } else {
    if (efaRow) efaRow.classList.remove("hidden");
    if (spectraSheetRow) spectraSheetRow.classList.remove("hidden");
    if (channelsRow) channelsRow.classList.remove("hidden");
    if (nmrSheetRow) nmrSheetRow.classList.add("hidden");
    if (nmrSignalsRow) nmrSignalsRow.classList.add("hidden");
  }
}

// --- Module Switching Logic ---

function renderModuleUI() {
  const m = M();

  // File Input
  const fileInput = document.getElementById("excel-file");
  const fileStatus = document.querySelector(".file-status");

  if (m.file) {
    if (fileStatus) fileStatus.textContent = m.fileName;
    // We cannot set fileInput.value to a file object.
    // But we can leave it alone if it matches? No, we can't read the file object from input easily to compare.
    // If we switched tabs, the input might hold the OTHER module's file.
    // So we should probably always clear the input value to avoid confusion, 
    // relying on the status text to show what's loaded in memory.
    // But if the user wants to re-select the SAME file to trigger change event, they can't if we don't clear it.
    // So clearing it is safer.
    if (fileInput) fileInput.value = "";
  } else {
    if (fileStatus) fileStatus.textContent = "No file selected";
    if (fileInput) fileInput.value = "";
  }

  // Sheet Selects - Restore Options
  const spectraSheet = document.getElementById("spectra-sheet-select");
  const concSheet = document.getElementById("conc-sheet-select");
  const nmrSheet = document.getElementById("nmr-sheet-select");

  const restoreOptions = (select, options) => {
    if (!select) return;
    select.innerHTML = "";
    if (!options || options.length === 0) {
      const opt = document.createElement("option");
      opt.text = "No sheets found"; // Or "Select a file first..."
      select.add(opt);
    } else {
      options.forEach(sheet => {
        const opt = document.createElement("option");
        opt.value = sheet;
        opt.text = sheet;
        select.add(opt);
      });
    }
  };

  restoreOptions(spectraSheet, m.availableSheets);
  restoreOptions(concSheet, m.availableSheets);
  restoreOptions(nmrSheet, m.availableSheets);

  // Restore Sheet Selections
  if (spectraSheet) spectraSheet.value = m.sheetSpectra;
  if (concSheet) concSheet.value = m.sheetConc;
  if (nmrSheet) nmrSheet.value = m.sheetNmr;

  // Spectroscopy channels
  const channelsInput = document.getElementById("spectroscopy-channels-input");
  const channelsRangeEl = document.getElementById("spectroscopy-channels-range");
  const channelsUsageEl = document.getElementById("spectroscopy-channels-usage");
  const channelsFeedbackEl = document.getElementById("spectroscopy-channels-feedback");
  if (channelsInput) channelsInput.value = m.channelsRaw ?? "All";
  if (channelsRangeEl && m.axisCount) {
    const minTxt = (m.axisMin ?? "") === "" ? "" : String(m.axisMin);
    const maxTxt = (m.axisMax ?? "") === "" ? "" : String(m.axisMax);
    channelsRangeEl.textContent = `${minTxt}–${maxTxt} (${m.axisCount})`;
  } else if (channelsRangeEl) {
    channelsRangeEl.textContent = "";
  }
  if (channelsUsageEl && m.axisCount) {
    const used = (m.channelsMode === "custom" && Array.isArray(m.channelsResolved)) ? m.channelsResolved.length : m.axisCount;
    channelsUsageEl.textContent = `Using ${used} / ${m.axisCount} channels`;
  } else if (channelsUsageEl) {
    channelsUsageEl.textContent = "";
  }
  if (channelsFeedbackEl) channelsFeedbackEl.textContent = "";

  // Columns Container - Restore Checkboxes
  const columnsContainer = document.getElementById("columns-container");
  if (columnsContainer) {
    columnsContainer.innerHTML = "";
    if (m.availableColumns.length === 0) {
      columnsContainer.textContent = "Select a concentration sheet to load columns...";
    } else {
      m.availableColumns.forEach(col => {
        const label = document.createElement("label");
        label.className = "checkbox-inline";
        label.style.marginRight = "10px";

        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.value = col;
        cb.name = "column_names";
        // Restore selection state? 
        // We need to store selected columns in state too.
        // We have m.selectedColumns (array of strings)
        // But we didn't populate it in the handler yet! 
        // Wait, we need to add a listener to columnsContainer to update m.selectedColumns.
        // I'll add that listener in wireSpectroscopyForm.
        // For now, let's assume it's populated or empty.
        // Actually, the previous code didn't have m.selectedColumns populated.
        // I need to add that listener.

        // Check if selected
        // We don't have m.selectedColumns populated yet.
        // I'll add the listener in the next step.

        const span = document.createElement("span");
        span.textContent = col;

        label.appendChild(cb);
        label.appendChild(span);
        columnsContainer.appendChild(label);
      });
    }
  }

  // NMR Signals Container - Restore Checkboxes
  const nmrSignalsContainer = document.getElementById("nmr-signals-container");
  if (nmrSignalsContainer) {
    nmrSignalsContainer.innerHTML = "";
    if (m.availableSignals.length === 0) {
      nmrSignalsContainer.textContent = "Select a chemical shift sheet to load signals...";
    } else {
      m.availableSignals.forEach(col => {
        const label = document.createElement("label");
        label.className = "checkbox-inline";
        label.style.marginRight = "10px";

        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.value = col;
        cb.name = "signal_names";

        const span = document.createElement("span");
        span.textContent = col;

        label.appendChild(cb);
        label.appendChild(span);
        nmrSignalsContainer.appendChild(label);
      });
    }
  }

  // Inputs
  const efaEigen = document.querySelector('input[type="number"]'); // First one
  const efaCheck = document.querySelector('input[type="checkbox"]');
  if (efaEigen) efaEigen.value = m.efaEigenvalues;
  if (efaCheck) efaCheck.checked = m.efaEnabled;

  // Model dims
  const nCompInput = document.querySelectorAll('input[type="number"]')[1];
  const nSpeciesInput = document.querySelectorAll('input[type="number"]')[2];
  if (nCompInput) nCompInput.value = m.nComponents;
  if (nSpeciesInput) nSpeciesInput.value = m.nSpecies;

  // Grids
  const modelGrid = document.getElementById("model-grid-container");
  const optGrid = document.getElementById("optimization-grid-container");

  if (m.nComponents === 0 && m.nSpecies === 0) {
    if (modelGrid) modelGrid.innerHTML = "";
    if (optGrid) optGrid.innerHTML = "";
  }

  // Receptor/Guest
  const receptor = document.getElementById("receptor-select");
  const guest = document.getElementById("guest-select");
  // We need to restore options for these too!
  // They depend on availableColumns.
  if (receptor && guest) {
    const restoreDropdown = (select, val) => {
      select.innerHTML = "";
      const defaultOpt = document.createElement("option");
      defaultOpt.value = "";
      defaultOpt.text = "";
      select.add(defaultOpt);

      m.availableColumns.forEach(col => {
        const opt = document.createElement("option");
        opt.value = col;
        opt.text = col;
        select.add(opt);
      });
      select.value = val;
    };
    restoreDropdown(receptor, m.receptor);
    restoreDropdown(guest, m.guest);
  }
}

function renderPlotsUI() {
  const m = M();
  const ps = m.plotState;

  // Restore Preset
  const presetSelect = document.getElementById("spectro-plot-preset-select");
  if (presetSelect) {
    presetSelect.innerHTML = '<option value="">Select a preset...</option>';
    ps.availablePlots.forEach(p => {
      const opt = document.createElement("option");
      opt.value = p.id;
      opt.text = p.title;
      presetSelect.add(opt);
    });

    const activePlot = ps.availablePlots[ps.activePlotIndex];
    if (activePlot) {
      presetSelect.value = activePlot.id;
    } else {
      presetSelect.value = "";
    }
  }

  // Sync other controls (X axis, Y series)
  // We need to ensure syncPlotControlsForActivePlot is available here.
  // It is defined later in the file, but function declarations are hoisted.
  // However, we should check if it exists just in case.
  if (typeof syncPlotControlsForActivePlot === 'function') {
    syncPlotControlsForActivePlot();
  } else {
    // Fallback: clear controls if function not found (shouldn't happen)
    const xAxis = document.getElementById("spectro-x-axis-select");
    const ySel = document.getElementById("spectro-y-series-select");
    if (xAxis) { xAxis.innerHTML = '<option value="">Select X axis...</option>'; xAxis.value = ""; }
    if (ySel) { ySel.innerHTML = ""; }
  }
}

function renderConsoleUI() {
  const m = M();
  setConsoleText(m.consoleText);
}

function switchModule(nextModule) {
  if (state.activeModule === nextModule) return;

  // 1. Save current state? (Already done via live updates in handlers)

  // 2. Switch
  state.activeModule = nextModule;

  // 3. Update UI visibility
  updateUI();

  // 4. Restore DOM state from new module
  renderModuleUI();
  renderPlotsUI();
  renderConsoleUI();

  // 5. Render Plot
  // We need to ensure Plotly is cleared or rendered
  const primaryPlot = document.querySelector('.primary-plot');
  const ps = M().plotState;
  if (ps.availablePlots.length > 0) {
    renderMainCanvasPlot();
  } else {
    if (primaryPlot) primaryPlot.innerHTML = ''; // Clear
    const counter = document.getElementById('plot-counter');
    if (counter) counter.textContent = '—';
  }
}

function initApp() {
  const app = document.querySelector("#app");
  if (!app) return;

  console.info("[HM Fit] Backend base URL:", BACKEND_BASE_URL);

  app.innerHTML = `
    <div class="root-container">
      <!-- Left Column: Header + Tabs + Controls -->
      <div class="left-column-wrapper">
        <header class="hmfit-header">
          <div>
            <h1 class="hmfit-title">HM Fit</h1>
            <p class="hmfit-subtitle">Hard Modeling · Spectroscopy &amp; NMR</p>
          </div>
        </header>

        <nav class="top-tabs">
          <button class="tab-btn" data-module-tab="spectroscopy">Spectroscopy</button>
          <button class="tab-btn" data-module-tab="nmr">NMR</button>
        </nav>

        <div class="left-panel">
          <!-- Selección de archivo -->
          <section class="panel file-panel">
            <label class="field-label">Select Excel File</label>
            <div class="file-row">
              <input type="file" id="excel-file" class="file-input" />
              <span class="file-status">No file selected</span>
            </div>

            <div class="field-grid">
              <div class="field">
                <label class="field-label">Spectra Sheet Name</label>
                <select id="spectra-sheet-select" class="field-input">
                  <option value="">Select a file first...</option>
                </select>
              </div>
              <div class="field">
                <label class="field-label">Concentration Sheet Name</label>
                <select id="conc-sheet-select" class="field-input">
                  <option value="">Select a file first...</option>
                </select>
              </div>
            </div>

            <div class="field" id="spectroscopy-channels-row">
              <label class="field-label">Channels <span id="spectroscopy-channels-range" class="hint"></span></label>
              <input id="spectroscopy-channels-input" class="field-input" value="All" />
              <div id="spectroscopy-channels-usage" class="hint"></div>
              <div id="spectroscopy-channels-feedback" class="hint"></div>
            </div>
            
            <!-- NMR Specific Field: Chemical Shift Sheet -->
            <div class="field hidden" id="nmr-sheet-row">
                <label class="field-label">Chemical Shift Sheet Name</label>
                <select id="nmr-sheet-select" class="field-input">
                  <option value="">Select a file first...</option>
                </select>
            </div>
            
            <div class="field hidden" id="nmr-signals-row">
              <label class="field-label">Chemical Shifts (Signals)</label>
              <div id="nmr-signals-container" class="checkbox-grid-placeholder">
                Select a chemical shift sheet to load signals...
              </div>
            </div>

            <div class="field">
              <label class="field-label">Column names</label>
              <div id="columns-container" class="checkbox-grid-placeholder">
                Select a concentration sheet to load columns...
              </div>
            </div>

            <div class="field-grid">
              <div class="field">
                <label class="field-label">Receptor or Ligand</label>
                <select id="receptor-select" class="field-input">
                  <option value="">Select columns first...</option>
                </select>
              </div>
              <div class="field">
                <label class="field-label">Guest, Metal or Titrant</label>
                <select id="guest-select" class="field-input">
                  <option value="">Select columns first...</option>
                </select>
              </div>
            </div>

            <div class="field efa-row">
              <label class="field-label">EFA Eigenvalues</label>
              <div class="efa-inline">
                <label class="checkbox-inline">
                  <input type="checkbox" checked />
                  <span>EFA</span>
                </label>
                <input type="number" class="field-input narrow" value="0" />
              </div>
            </div>
          </section>

          <!-- Modelo / Optimización -->
          <section class="panel model-panel">
            <nav class="sub-tabs">
              <button class="subtab-btn" data-subtab="model">Model</button>
              <button class="subtab-btn" data-subtab="optimization">Optimization</button>
              <button class="subtab-btn" data-subtab="plots">Plots</button>
            </nav>

            <div class="subtab-panel" data-subtab-panel="model">
              <button id="define-model-btn" class="btn full-width-btn">Define Model Dimensions</button>

              <div class="field-grid">
                <div class="field">
                  <label class="field-label">Number of Components</label>
                  <input type="number" class="field-input" value="0" min="0" />
                </div>
                <div class="field">
                  <label class="field-label">Number of Species</label>
                  <input type="number" class="field-input" value="0" min="0" />
                </div>
              </div>

              <div class="field">
                <label class="field-label">Select non-absorbent species</label>
                <div id="model-grid-container" class="model-grid-container">
                  <!-- Grid will be generated here -->
                </div>
              </div>
            </div>

            <div class="subtab-panel" data-subtab-panel="optimization">
              <div class="field-grid">
                <div class="field">
                  <label class="field-label">Algorithm for C</label>
                  <select id="algorithm-select" class="field-input"></select>
                </div>
                <div class="field">
                  <label class="field-label">Model settings</label>
                  <select id="model-settings-select" class="field-input"></select>
                </div>
              </div>
              
              <div class="field">
                <label class="field-label">Optimizer</label>
                <select id="optimizer-select" class="field-input"></select>
              </div>

              <div class="field">
                <label class="field-label">Parameters (Initial Estimates)</label>
                <div id="optimization-grid-container" class="model-grid-container">
                  <!-- Grid will be generated here -->
                </div>
              </div>
            </div>

            <div class="subtab-panel" data-subtab-panel="plots">
              <div class="field-grid">
                <div class="field">
                  <label class="field-label">Preset</label>
                  <select id="spectro-plot-preset-select" class="field-input">
                    <option value="">Select a preset...</option>
                  </select>
                </div>
                <div class="field">
                  <label class="field-label">X axis</label>
                  <select id="spectro-x-axis-select" class="field-input">
                    <option value="">Select X axis...</option>
                  </select>
                </div>
              </div>

              <div class="field-grid">
                <div class="field">
                  <label class="field-label">Y series</label>
                  <select id="spectro-y-series-select" class="field-input" multiple size="4">
                    <option value="">Select series...</option>
                  </select>
                </div>
                <div class="field">
                  <label class="field-label">Vary along</label>
                  <select id="spectro-vary-along-select" class="field-input">
                    <option value="">Auto</option>
                  </select>
                </div>
              </div>

              <div class="field" style="margin-top: 0.75rem;">
                <label class="field-label">Edit plot</label>
                <div class="field-grid">
                  <div class="field">
                    <label class="field-label">Title</label>
                    <input id="plot-edit-title" class="field-input" placeholder="Title" />
                  </div>
                  <div class="field">
                    <label class="field-label">X axis label</label>
                    <input id="plot-edit-xlabel" class="field-input" placeholder="X axis label" />
                  </div>
                </div>
                <div class="field-grid">
                  <div class="field">
                    <label class="field-label">Y axis label</label>
                    <input id="plot-edit-ylabel" class="field-input" placeholder="Y axis label" />
                  </div>
                  <div class="field">
                    <label class="field-label">Trace</label>
                    <select id="plot-edit-trace-select" class="field-input">
                      <option value="">Select trace...</option>
                    </select>
                  </div>
                </div>
                <div class="field-grid">
                  <div class="field">
                    <label class="field-label">New trace name</label>
                    <input id="plot-edit-trace-name" class="field-input" placeholder="New trace name" />
                  </div>
                  <div class="field" style="display:flex; gap: 0.5rem; align-items: end;">
                    <button id="plot-edit-apply" class="btn tertiary-btn" style="flex:1;">Apply</button>
                    <button id="plot-edit-reset" class="btn tertiary-btn" style="flex:1;">Reset</button>
                  </div>
                </div>
              </div>

              <div class="actions-row" style="margin-top: 0.5rem;">
                <div class="actions-left">
                  <button id="spectro-export-png" class="btn tertiary-btn">Export PNG</button>
                  <button id="spectro-export-csv" class="btn tertiary-btn">Export CSV</button>
                </div>
              </div>
            </div>

            <div class="actions-row">
              <div class="actions-left">
                <button id="backend-health" class="btn ghost-btn">Probar backend</button>
                <button id="import-config-btn" class="btn tertiary-btn">Import Config</button>
                <button id="export-config-btn" class="btn tertiary-btn">Export Config</button>
              </div>
              <div class="actions-right">
                <button id="reset-btn" class="btn secondary-btn">Reset Calculation</button>
                <button id="process-btn" class="btn primary-btn">Process Data</button>
                <button id="save-results-btn" class="btn ghost-btn">Save results</button>
              </div>
            </div>
          </section>
        </div>
      </div>

      <!-- Right Column: Full Height Plot Panel -->
      <div class="right-panel">
        <section class="panel plot-panel split-panel">
          <div class="split-top">
            <div class="plot-toolbar">
              <button id="plot-prev-btn" class="btn tertiary-btn">« Prev</button>
              <h2 class="section-title">Main spectra / titration plot</h2>
              <div id="plot-counter" class="plot-counter">—</div>
              <button id="plot-next-btn" class="btn tertiary-btn">Next »</button>
            </div>
            <div class="plot-side-nav left">
              <button id="plot-prev-side" class="plot-side-btn" title="Anterior">‹</button>
            </div>
            <div class="plot-placeholder primary-plot">
              <!-- Main plot area -->
            </div>
            <div class="plot-side-nav right">
              <button id="plot-next-side" class="plot-side-btn" title="Siguiente">›</button>
            </div>
          </div>
          <div class="split-resizer" title="Arrastra para redimensionar"></div>
          <div class="split-bottom">
            <h2 class="section-title">Residuals / component spectra / diagnostics</h2>
            <pre id="log-output" class="log-output">Esperando...</pre>
          </div>
        </section>
      </div>
    </div>
  `;

  // Tabs: módulo (Spectroscopy / NMR)
  // Tabs: módulo (Spectroscopy / NMR)
  document.querySelectorAll("[data-module-tab]").forEach((btn) => {
    const mod = btn.dataset.moduleTab;
    btn.addEventListener("click", () => {
      switchModule(mod);
    });
  });

  // Subtabs: Model / Optimization
  document.querySelectorAll("[data-subtab]").forEach((btn) => {
    const sub = btn.dataset.subtab;
    btn.addEventListener("click", () => {
      state.activeSubtab = sub;
      updateUI();
    });
  });

  // Botones
  const backendBtn = document.getElementById("backend-health");
  backendBtn?.addEventListener("click", pingBackend);

  // Note: processBtn and resetBtn listeners are also attached in wireSpectroscopyForm.
  // We can leave them here or rely on wireSpectroscopyForm. 
  // Since wireSpectroscopyForm is called below, it will attach the main logic.
  // The listeners here (dummyFit, resetCalculation) might be redundant or conflicting if not careful.
  // resetCalculation is simple logging. wireSpectroscopyForm's reset is more comprehensive.
  // I will comment out the simple ones here to avoid duplication/confusion, relying on wireSpectroscopyForm.

  // const processBtn = document.getElementById("process-btn");
  // processBtn?.addEventListener("click", dummyFit);

  // const resetBtn = document.getElementById("reset-btn");
  // resetBtn?.addEventListener("click", resetCalculation);

  // Wire up spectroscopy form
  wireSpectroscopyForm();

  // Initial UI Update
  updateUI();

  // Initialize splitter behavior
  initSplitPanel();
}

// === Helpers para localizar elementos existentes sin cambiar el HTML ===

// Split panel resizable log area
function initSplitPanel() {
  const panel = document.querySelector(".split-panel");
  const resizer = document.querySelector(".split-resizer");
  const top = document.querySelector(".split-top");
  const bottom = document.querySelector(".split-bottom");
  if (!panel || !resizer || !top || !bottom) return;

  let isDragging = false;
  resizer.addEventListener("mousedown", () => {
    isDragging = true;
    document.body.classList.add("resizing");
  });
  window.addEventListener("mouseup", () => {
    isDragging = false;
    document.body.classList.remove("resizing");
  });
  window.addEventListener("mousemove", (e) => {
    if (!isDragging) return;
    const rect = panel.getBoundingClientRect();
    const offset = e.clientY - rect.top;
    const totalHeight = rect.height;

    const minTop = 100;
    const minBottom = 50;

    // Calculate new bottom height based on mouse position
    // Mouse is at 'offset' from top, so bottom height is total - offset
    let newBottomHeight = totalHeight - offset;

    // Clamp
    const maxBottom = totalHeight - minTop;
    newBottomHeight = Math.min(Math.max(newBottomHeight, minBottom), maxBottom);

    bottom.style.height = `${newBottomHeight}px`;
    // Top automatically adjusts due to flex: 1

    // Resize Plotly if present
    requestAnimationFrame(resizeMainPlotIfNeeded);
  });

  // Also resize on window resize
  window.addEventListener("resize", () => requestAnimationFrame(resizeMainPlotIfNeeded));
}

function resizeMainPlotIfNeeded() {
  // Plotly
  const plotlyDiv = document.querySelector(".primary-plot .main-plotly");
  if (plotlyDiv && window.Plotly) {
    try { window.Plotly.Plots.resize(plotlyDiv); } catch (_) { }
  }
  // Images handle themselves via CSS object-fit: contain
}

// Encuentra un botón por su texto visible
function findButtonByLabel(label) {
  const norm = label.trim().toLowerCase();
  return Array.from(document.querySelectorAll("button")).find(
    (btn) => btn.textContent.trim().toLowerCase() === norm,
  );
}

// Devuelve el elemento que contiene el texto "Esperando..."
function findDiagnosticsElement() {
  const candidates = document.querySelectorAll("div, pre, p");
  for (const el of candidates) {
    if (el.textContent.includes("Esperando")) {
      return el;
    }
  }
  return null;
}

// Lectores simples
function readInt(value) {
  const v = String(value ?? "").trim();
  const n = parseInt(v, 10);
  return Number.isFinite(n) ? n : 0;
}

function readList(text) {
  const v = String(text ?? "").trim();
  if (!v) return [];
  return v.split(/[,\s]+/).filter(Boolean);
}

function parseChannelsInput(input) {
  const raw = String(input ?? "").trim();
  if (!raw) {
    return { mode: "custom", tokens: [], errors: ["Channels is empty. Use 'All' or a list like 450,650."] };
  }

  if (raw.toLowerCase() === "all") {
    return { mode: "all", tokens: [], errors: [] };
  }

  const parts = raw.split(",").map((s) => s.trim()).filter(Boolean);
  const tokens = [];
  const errors = [];

  for (const part of parts) {
    const m = part.match(/^(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)$/);
    if (m) {
      const a = Number(m[1]);
      const b = Number(m[2]);
      if (!Number.isFinite(a) || !Number.isFinite(b)) {
        errors.push(`Invalid range: '${part}'.`);
      } else {
        tokens.push({ type: "range", min: a, max: b, raw: part });
      }
      continue;
    }

    const v = Number(part);
    if (!Number.isFinite(v)) {
      errors.push(`Invalid channel value: '${part}'.`);
      continue;
    }
    tokens.push({ type: "value", value: v, raw: part });
  }

  if (tokens.length === 0 && errors.length === 0) {
    errors.push("No channels parsed. Use 'All' or values like 450,650.");
  }

  return { mode: "custom", tokens, errors };
}

function resolveChannels(tokens, axisValues, tol = 0.5) {
  const errors = [];
  const mappingLines = [];

  const axis = Array.isArray(axisValues) ? axisValues.filter((v) => Number.isFinite(v)) : [];
  if (axis.length === 0) {
    return { resolved: [], mappingLines, errors: ["Axis not loaded yet. Select a Spectra sheet first."] };
  }

  const resolvedSet = new Set();

  const nearest = (target) => {
    let best = null;
    let bestDiff = Infinity;
    for (const a of axis) {
      const d = Math.abs(a - target);
      if (d < bestDiff) {
        best = a;
        bestDiff = d;
      }
    }
    return { value: best, diff: bestDiff };
  };

  for (const tok of tokens) {
    if (tok.type === "value") {
      const { value: nearestVal, diff } = nearest(tok.value);
      if (nearestVal === null || !Number.isFinite(nearestVal)) {
        errors.push(`No axis values available to resolve '${tok.raw}'.`);
        continue;
      }
      if (diff > tol) {
        errors.push(`'${tok.raw}' is not within tolerance (tol=${tol}) of any axis value.`);
        continue;
      }
      resolvedSet.add(nearestVal);
      mappingLines.push(`${tok.raw} → ${nearestVal}`);
      continue;
    }

    if (tok.type === "range") {
      const lo = Math.min(tok.min, tok.max);
      const hi = Math.max(tok.min, tok.max);
      const inRange = axis.filter((v) => v >= lo && v <= hi);
      if (inRange.length === 0) {
        errors.push(`Range '${tok.raw}' matched 0 axis values.`);
        continue;
      }
      for (const v of inRange) resolvedSet.add(v);
      mappingLines.push(`${tok.raw} → ${inRange.length} channels`);
      continue;
    }

    errors.push(`Unsupported token '${tok?.raw ?? ""}'.`);
  }

  const resolved = axis.filter((v) => resolvedSet.has(v));
  return { resolved, mappingLines, errors };
}

// === Wire-up de Spectroscopy ===

function wireSpectroscopyForm() {
  // Botones
  const processBtn = findButtonByLabel("Process Data");
  const resetBtn = findButtonByLabel("Reset Calculation");
  const saveBtn = findButtonByLabel("Save results");
  const diagEl = document.getElementById("log-output");

  if (!processBtn || !resetBtn || !diagEl) {
    console.warn(
      "[HM Fit] No se pudieron localizar Process Data / Reset / panel de diagnóstico. Revisa los textos de los botones."
    );
    return;
  }

  // Campos de texto identificables por placeholder
  // NOTA: Ahora son selects, los buscamos por ID

  const spectraSheetInput = document.getElementById("spectra-sheet-select");
  const concSheetInput = document.getElementById("conc-sheet-select");
  const nmrSheetInput = document.getElementById("nmr-sheet-select");
  const columnsContainer = document.getElementById("columns-container");
  const nmrSignalsContainer = document.getElementById("nmr-signals-container");

  // Spectroscopy channels UI
  const channelsInput = document.getElementById("spectroscopy-channels-input");
  const channelsRangeEl = document.getElementById("spectroscopy-channels-range");
  const channelsUsageEl = document.getElementById("spectroscopy-channels-usage");
  const channelsFeedbackEl = document.getElementById("spectroscopy-channels-feedback");

  // Dropdowns para Receptor y Guest
  const receptorInput = document.getElementById("receptor-select");
  const guestInput = document.getElementById("guest-select");

  const numericInputs = Array.from(
    document.querySelectorAll('input[type="number"]'),
  );
  // asumiendo orden: EFA eigenvalues, #components, #species
  const efaEigenInput = numericInputs[0] || null;
  const nCompInput = numericInputs[1] || null;
  const nSpeciesInput = numericInputs[2] || null;

  // Único checkbox de EFA
  const efaCheckbox = document.querySelector('input[type="checkbox"]');
  const efaOriginalTitle = efaCheckbox?.title || "";

  // Botón para definir dimensiones
  const defineModelBtn = document.getElementById("define-model-btn");

  // Área de grid
  const modelGridContainer = document.getElementById("model-grid-container");
  const optGridContainer = document.getElementById("optimization-grid-container");

  // Dropdowns de Optimización
  const algoSelect = document.getElementById("algorithm-select");
  const modelSettingsSelect = document.getElementById("model-settings-select");
  const optimizerSelect = document.getElementById("optimizer-select");
  const plotPresetSelect = document.getElementById("spectro-plot-preset-select");
  const plotXAxisSelect = document.getElementById("spectro-x-axis-select");
  const plotYSeriesSelect = document.getElementById("spectro-y-series-select");
  const plotVarySelect = document.getElementById("spectro-vary-along-select");
  const plotExportPngBtn = document.getElementById("spectro-export-png");
  const plotExportCsvBtn = document.getElementById("spectro-export-csv");
  const plotEditTitleInput = document.getElementById("plot-edit-title");
  const plotEditXLabelInput = document.getElementById("plot-edit-xlabel");
  const plotEditYLabelInput = document.getElementById("plot-edit-ylabel");
  const plotEditTraceSelect = document.getElementById("plot-edit-trace-select");
  const plotEditTraceNameInput = document.getElementById("plot-edit-trace-name");
  const plotEditApplyBtn = document.getElementById("plot-edit-apply");
  const plotEditResetBtn = document.getElementById("plot-edit-reset");

  // Poblar dropdowns de optimización
  if (algoSelect) {
    ["Newton-Raphson", "Levenberg-Marquardt"].forEach(opt => {
      const el = document.createElement("option");
      el.value = opt;
      el.text = opt;
      algoSelect.add(el);
    });
  }
  if (modelSettingsSelect) {
    ["Free", "Step by step", "Non-cooperative"].forEach(opt => {
      const el = document.createElement("option");
      el.value = opt;
      el.text = opt;
      modelSettingsSelect.add(el);
    });
  }
  if (optimizerSelect) {
    ["powell", "nelder-mead", "trust-constr", "cg", "bfgs", "l-bfgs-b", "tnc", "cobyla", "slsqp", "differential_evolution"].forEach(opt => {
      const el = document.createElement("option");
      el.value = opt;
      el.text = opt;
      optimizerSelect.add(el);
    });
  }

  // Preset dropdown: jump to selected plot in main canvas
  plotPresetSelect?.addEventListener("change", () => {
    const selectedId = plotPresetSelect.value;
    if (!selectedId) return;

    const ps = M().plotState;
    const index = ps.availablePlots.findIndex(p => p.id === selectedId);
    if (index >= 0) {
      ps.activePlotIndex = index;
      renderMainCanvasPlot();
    }
  });

  plotEditTraceSelect?.addEventListener('change', () => {
    const plotDiv = document.querySelector('.primary-plot .main-plotly');
    const idx = Number(plotEditTraceSelect.value);
    if (!plotDiv || !plotEditTraceNameInput || !Number.isFinite(idx)) return;
    const name = plotDiv?.data?.[idx]?.name ?? '';
    plotEditTraceNameInput.value = name;
  });

  plotEditApplyBtn?.addEventListener('click', () => {
    const plot = getActivePlot();
    const plotDiv = document.querySelector('.primary-plot .main-plotly');
    if (!plot || plot.kind !== 'plotly' || !plotDiv || !window.Plotly) return;

    const modKey = state.activeModule;
    const presetId = plot.id;
    const ov = state.plotOverrides[modKey]?.[presetId] || { traceNames: {} };

    const titleText = String(plotEditTitleInput?.value ?? '').trim();
    const xLabel = String(plotEditXLabelInput?.value ?? '').trim();
    const yLabel = String(plotEditYLabelInput?.value ?? '').trim();

    ov.titleText = titleText;
    ov.xLabel = xLabel;
    ov.yLabel = yLabel;

    const relayout = {
      "title.text": titleText,
      "xaxis.title.text": xLabel,
      "yaxis.title.text": yLabel,
    };
    window.Plotly.relayout(plotDiv, relayout);

    const traceIdx = Number(plotEditTraceSelect?.value);
    const newTraceName = String(plotEditTraceNameInput?.value ?? '').trim();
    if (Number.isFinite(traceIdx) && newTraceName) {
      ov.traceNames = ov.traceNames || {};
      ov.traceNames[String(traceIdx)] = newTraceName;
      window.Plotly.restyle(plotDiv, { name: newTraceName }, [traceIdx]);
    }

    state.plotOverrides[modKey][presetId] = ov;
    syncEditPlotPanelFromDiv(plotDiv);
    appendLog(`Applied plot edits for ${presetId}.`);
  });

  plotEditResetBtn?.addEventListener('click', () => {
    const plot = getActivePlot();
    const plotDiv = document.querySelector('.primary-plot .main-plotly');
    if (!plot || plot.kind !== 'plotly' || !plotDiv || !window.Plotly) return;

    const modKey = state.activeModule;
    const presetId = plot.id;
    const defaults = state.plotDefaults?.[modKey]?.[presetId];
    if (!defaults) return;

    window.Plotly.relayout(plotDiv, {
      "title.text": defaults.titleText ?? '',
      "xaxis.title.text": defaults.xLabel ?? '',
      "yaxis.title.text": defaults.yLabel ?? '',
    });

    const names = defaults.traceNames || {};
    for (const [idx, name] of Object.entries(names)) {
      const i = Number(idx);
      if (!Number.isFinite(i)) continue;
      window.Plotly.restyle(plotDiv, { name }, [i]);
    }

    if (state.plotOverrides?.[modKey]) delete state.plotOverrides[modKey][presetId];
    syncEditPlotPanelFromDiv(plotDiv);
    appendLog(`Reset plot edits for ${presetId}.`);
  });

  // X axis dropdown: update controls and re-render
  plotXAxisSelect?.addEventListener("change", () => {
    const ps = M().plotState;
    const plot = ps.availablePlots[ps.activePlotIndex];
    if (plot?.id === 'spec_species_distribution' || plot?.id === 'nmr_species_distribution') {
      ps.controls.distXAxisId = plotXAxisSelect.value;
      renderMainCanvasPlot();
    }
  });

  // Y series dropdown: update selected species/signals and re-render
  plotYSeriesSelect?.addEventListener("change", () => {
    const ps = M().plotState;
    const plot = ps.availablePlots[ps.activePlotIndex];
    const selected = Array.from(plotYSeriesSelect.selectedOptions).map(o => o.value);

    if (plot?.id === 'spec_species_distribution' || plot?.id === 'nmr_species_distribution') {
      ps.controls.distYSelected = new Set(selected);
    } else if (plot?.id === 'nmr_shifts_fit') {
      ps.controls.nmrSignalsSelected = new Set(selected);
    } else if (plot?.id === 'nmr_residuals') {
      ps.controls.nmrResidSelected = new Set(selected);
    }

    renderMainCanvasPlot();
  });

  // Export PNG from main canvas (the current carousel page)
  plotExportPngBtn?.addEventListener("click", exportMainCanvasPNG);
  plotExportCsvBtn?.addEventListener("click", exportMainCanvasCSV);

  // Navigation buttons for main canvas
  document.getElementById('plot-prev-btn')?.addEventListener('click', () => navigateMainCanvas(-1));
  document.getElementById('plot-next-btn')?.addEventListener('click', () => navigateMainCanvas(1));
  document.getElementById('plot-prev-side')?.addEventListener('click', () => navigateMainCanvas(-1));
  document.getElementById('plot-next-side')?.addEventListener('click', () => navigateMainCanvas(1));

  // --- Handler: File Selection ---
  const fileInput = document.getElementById("excel-file");
  const fileStatus = document.querySelector(".file-status");

  fileInput?.addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) {
      fileStatus.textContent = "No file selected";
      M().file = null;
      M().fileName = "";
      return;
    }
    fileStatus.textContent = file.name;
    M().file = file;
    M().fileName = file.name;

    // Enviar al backend para obtener hojas
    try {
      diagEl.textContent = "Leyendo archivo Excel...";
      const data = await backendApi.listSheets(file, state.activeModule);
      const sheets = data?.sheets || [];
      M().availableSheets = sheets;

      // Poblar dropdowns
      [spectraSheetInput, concSheetInput, nmrSheetInput].forEach(select => {
        if (!select) return;
        select.innerHTML = ""; // Limpiar
        if (sheets.length === 0) {
          const opt = document.createElement("option");
          opt.text = "No sheets found";
          select.add(opt);
        } else {
          sheets.forEach(sheet => {
            const opt = document.createElement("option");
            opt.value = sheet;
            opt.text = sheet;
            select.add(opt);
          });
          // Trigger change event to load columns for the default selected sheet
          select.dispatchEvent(new Event('change'));
        }
      });
      diagEl.textContent = `Archivo cargado. ${sheets.length} hojas encontradas.`;

    } catch (err) {
      console.error(err);
      diagEl.textContent = `Error al leer Excel: ${err.message}`;
    }
  });

  // --- Handler: Concentration Sheet Selection ---
  concSheetInput?.addEventListener("change", async () => {
    const sheetName = concSheetInput.value;
    M().sheetConc = sheetName;
    const file = fileInput.files[0];

    if (!sheetName || !file) {
      columnsContainer.innerHTML = "Select a concentration sheet to load columns...";
      return;
    }

    try {
      diagEl.textContent = `Leyendo columnas de ${sheetName}...`;
      const data = await backendApi.listColumns(file, sheetName, state.activeModule);
      const columns = data?.columns || [];
      M().availableColumns = columns;

      columnsContainer.innerHTML = ""; // Limpiar
      if (columns.length === 0) {
        columnsContainer.textContent = "No columns found.";
      } else {
        columns.forEach(col => {
          const label = document.createElement("label");
          label.className = "checkbox-inline";
          label.style.marginRight = "10px";

          const cb = document.createElement("input");
          cb.type = "checkbox";
          cb.value = col;
          cb.name = "column_names"; // Para facilitar recolección si se desea

          const span = document.createElement("span");
          span.textContent = col;

          label.appendChild(cb);
          label.appendChild(span);
          columnsContainer.appendChild(label);
        });
      }
      diagEl.textContent = `Columnas cargadas de ${sheetName}.`;

    } catch (err) {
      console.error(err);
      diagEl.textContent = `Error al leer columnas: ${err.message}`;
    }

  });

  // --- Handler: NMR Chemical Shift Sheet Selection ---
  nmrSheetInput?.addEventListener("change", async () => {
    const sheetName = nmrSheetInput.value;
    M().sheetNmr = sheetName;
    const file = fileInput.files[0];

    if (!sheetName || !file) {
      nmrSignalsContainer.innerHTML = "Select a chemical shift sheet to load signals...";
      return;
    }

    try {
      diagEl.textContent = `Leyendo señales de ${sheetName}...`;
      // Reusing listColumns as it just reads headers
      const data = await backendApi.listColumns(file, sheetName, state.activeModule);
      const columns = data?.columns || [];
      M().availableSignals = columns;

      nmrSignalsContainer.innerHTML = "";
      if (columns.length === 0) {
        nmrSignalsContainer.textContent = "No signals found.";
      } else {
        columns.forEach(col => {
          const label = document.createElement("label");
          label.className = "checkbox-inline";
          label.style.marginRight = "10px";

          const cb = document.createElement("input");
          cb.type = "checkbox";
          cb.value = col;
          cb.name = "signal_names";

          const span = document.createElement("span");
          span.textContent = col;

          label.appendChild(cb);
          label.appendChild(span);
          nmrSignalsContainer.appendChild(label);
        });
      }
      diagEl.textContent = `Señales cargadas de ${sheetName}.`;
    } catch (err) {
      console.error(err);
      diagEl.textContent = `Error al leer señales: ${err.message}`;
    }
  });

  // --- Helper: Grid Generation ---
  function generateModelGrid(nComp, nSpecies) {
    if (!modelGridContainer) return;
    modelGridContainer.innerHTML = "";

    if (nComp <= 0 || nSpecies <= 0) return;

    const table = document.createElement("table");
    table.className = "model-grid-table";

    // Header
    const thead = document.createElement("thead");
    const headerRow = document.createElement("tr");
    headerRow.appendChild(document.createElement("th")); // Empty corner
    for (let c = 1; c <= nComp; c++) {
      const th = document.createElement("th");
      th.textContent = `C${c}`;
      headerRow.appendChild(th);
    }
    thead.appendChild(headerRow);
    table.appendChild(thead);

    // Body
    const tbody = document.createElement("tbody");
    const totalRows = nComp + nSpecies;

    for (let s = 1; s <= totalRows; s++) {
      const tr = document.createElement("tr");
      tr.dataset.species = `sp${s}`;

      const tdLabel = document.createElement("td");
      tdLabel.textContent = `sp${s}`;
      tdLabel.className = "species-label";
      tr.appendChild(tdLabel);

      for (let c = 1; c <= nComp; c++) {
        const td = document.createElement("td");
        const input = document.createElement("input");
        input.type = "number";
        input.className = "grid-input";
        // Default identity matrix for first nComp rows
        if (s <= nComp) {
          input.value = (c === s) ? "1.0" : "0.0";
        } else {
          input.value = "0";
        }
        td.appendChild(input);
        tr.appendChild(td);
      }

      tr.addEventListener("click", (e) => {
        if (e.target.tagName.toLowerCase() === "input") return;
        tr.classList.toggle("selected");
      });

      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    modelGridContainer.appendChild(table);
  }

  function generateOptGrid(nSpecies) {
    if (!optGridContainer) return;
    optGridContainer.innerHTML = "";
    const nConstants = nSpecies;

    if (nConstants > 0) {
      const optTable = document.createElement("table");
      optTable.className = "model-grid-table";

      const optThead = document.createElement("thead");
      const optHeaderRow = document.createElement("tr");
      ["Parameter", "Value", "Min", "Max"].forEach(text => {
        const th = document.createElement("th");
        th.textContent = text;
        optHeaderRow.appendChild(th);
      });
      optThead.appendChild(optHeaderRow);
      optTable.appendChild(optThead);

      const optTbody = document.createElement("tbody");
      for (let k = 1; k <= nConstants; k++) {
        const tr = document.createElement("tr");

        const tdParam = document.createElement("td");
        tdParam.textContent = `K${k}`;
        tdParam.className = "species-label";
        tr.appendChild(tdParam);

        const tdVal = document.createElement("td");
        const inputVal = document.createElement("input");
        inputVal.type = "number";
        inputVal.className = "grid-input";
        inputVal.placeholder = "Value";
        tdVal.appendChild(inputVal);
        tr.appendChild(tdVal);

        const tdMin = document.createElement("td");
        const inputMin = document.createElement("input");
        inputMin.type = "number";
        inputMin.className = "grid-input";
        inputMin.placeholder = "Min";
        tdMin.appendChild(inputMin);
        tr.appendChild(tdMin);

        const tdMax = document.createElement("td");
        const inputMax = document.createElement("input");
        inputMax.type = "number";
        inputMax.className = "grid-input";
        inputMax.placeholder = "Max";
        tdMax.appendChild(inputMax);
        tr.appendChild(tdMax);

        optTbody.appendChild(tr);
      }
      optTable.appendChild(optTbody);
      optGridContainer.appendChild(optTable);
    } else {
      optGridContainer.textContent = "No species defined.";
    }
  }

  // --- Handler: Define Model Dimensions (Grid Generation) ---
  defineModelBtn?.addEventListener("click", () => {
    const nComp = readInt(nCompInput?.value);
    const nSpecies = readInt(nSpeciesInput?.value);

    if (nComp <= 0 || nSpecies <= 0) {
      diagEl.textContent = "Please enter valid Number of Components and Species (>0).";
      return;
    }

    generateModelGrid(nComp, nSpecies);
    generateOptGrid(nSpecies);
    diagEl.textContent = `Grid generado: ${nComp} Componentes x ${nSpecies} Especies.`;

    // Save to state
    M().nComponents = nComp;
    M().nSpecies = nSpecies;
  });

  // --- Input Listeners for State Persistence ---
  const formatAxisVal = (v) => {
    if (!Number.isFinite(v)) return String(v ?? "");
    const rounded = Math.round(v);
    if (Math.abs(v - rounded) < 1e-9) return String(rounded);
    return String(v);
  };

  const CHANNEL_TOL = 0.5;
  const MIN_CUSTOM_CHANNELS = 5;

  function revalidateSpectroscopyChannels() {
    if (state.activeModule !== "spectroscopy") return { ok: true, mode: "all", resolved: [] };
    const m = M();

    const raw = String(channelsInput?.value ?? m.channelsRaw ?? "All");
    m.channelsRaw = raw;

    if (!channelsUsageEl || !channelsFeedbackEl) return { ok: true, mode: "all", resolved: [] };

    const axisCount = m.axisCount ?? (Array.isArray(m.axisValues) ? m.axisValues.length : 0);

    const parsed = parseChannelsInput(raw);
    if (parsed.mode === "all") {
      m.channelsMode = "all";
      m.channelsResolved = [];
      channelsFeedbackEl.textContent = "";
      channelsUsageEl.textContent = axisCount ? `Using ${axisCount} / ${axisCount} channels` : "";

      if (efaCheckbox) {
        efaCheckbox.disabled = false;
        efaCheckbox.title = efaOriginalTitle;
      }

      return { ok: true, mode: "all", resolved: [] };
    }

    m.channelsMode = "custom";
    const resolvedInfo = resolveChannels(parsed.tokens, m.axisValues, CHANNEL_TOL);
    const errors = [...(parsed.errors || []), ...(resolvedInfo.errors || [])];

    const resolved = resolvedInfo.resolved || [];
    m.channelsResolved = resolved;

    channelsUsageEl.textContent = axisCount ? `Using ${resolved.length} / ${axisCount} channels` : "";

    const lines = [];
    if (resolvedInfo.mappingLines?.length) lines.push(...resolvedInfo.mappingLines);
    if (errors.length) {
      lines.push("", "Errors:", ...errors.map((e) => `- ${e}`));
    }

    if (!errors.length && resolved.length > 0 && resolved.length < MIN_CUSTOM_CHANNELS) {
      lines.push("", `Warning: con pocos canales (k=${resolved.length}) puede haber mala identificabilidad / ajuste inestable.`);
    }

    channelsFeedbackEl.textContent = lines.join("\n").trim();

    if (efaCheckbox) {
      if (resolved.length > 0) {
        efaCheckbox.checked = false;
        efaCheckbox.disabled = true;
        efaCheckbox.title = "EFA requiere espectro completo (All).";
        m.efaEnabled = false;
      } else {
        efaCheckbox.disabled = false;
        efaCheckbox.title = efaOriginalTitle;
      }
    }

    return { ok: errors.length === 0 && resolved.length > 0, mode: "custom", resolved, errors };
  }

  async function loadSpectroscopyAxis() {
    if (state.activeModule !== "spectroscopy") return;
    if (!spectraSheetInput || !channelsRangeEl || !channelsUsageEl || !channelsFeedbackEl) return;

    const m = M();
    const sheetName = spectraSheetInput.value;
    m.sheetSpectra = sheetName;

    channelsFeedbackEl.textContent = "";

    if (!sheetName || !m.file) {
      m.axisValues = [];
      m.axisMin = null;
      m.axisMax = null;
      m.axisCount = null;
      channelsRangeEl.textContent = "";
      channelsUsageEl.textContent = "";
      return;
    }

    try {
      const data = await backendApi.listSpectroscopyAxis(m.file, sheetName);
      m.axisValues = data?.axis_values || [];
      m.axisMin = data?.min ?? null;
      m.axisMax = data?.max ?? null;
      m.axisCount = data?.count ?? (m.axisValues?.length || 0);

      if (m.axisCount) {
        channelsRangeEl.textContent = `${formatAxisVal(m.axisMin)}–${formatAxisVal(m.axisMax)} (${m.axisCount})`;
      } else {
        channelsRangeEl.textContent = "";
      }

      if (channelsInput) {
        channelsInput.value = "All";
        m.channelsRaw = "All";
        m.channelsMode = "all";
        m.channelsResolved = [];
      }
      revalidateSpectroscopyChannels();
    } catch (err) {
      m.axisValues = [];
      m.axisMin = null;
      m.axisMax = null;
      m.axisCount = null;
      channelsRangeEl.textContent = "";
      channelsUsageEl.textContent = "";
      channelsFeedbackEl.textContent = `Error reading axis: ${err?.message || err}`;
    }
  }

  spectraSheetInput?.addEventListener("change", () => { void loadSpectroscopyAxis(); });
  channelsInput?.addEventListener("input", () => { void revalidateSpectroscopyChannels(); });
  channelsInput?.addEventListener("change", () => { void revalidateSpectroscopyChannels(); });

  efaEigenInput?.addEventListener("change", () => { M().efaEigenvalues = readInt(efaEigenInput.value); });
  efaCheckbox?.addEventListener("change", () => { M().efaEnabled = efaCheckbox.checked; });

  receptorInput?.addEventListener("change", () => { M().receptor = receptorInput.value; });
  guestInput?.addEventListener("change", () => { M().guest = guestInput.value; });

  // Note: Grid content is not automatically saved on every keystroke here, 
  // but is collected when processing. If we wanted to persist grid state on switch,
  // we would need to serialize the grid to M().modelGrid on change or on switchModule.
  // For now, we rely on the user not switching tabs mid-edit if they want to keep the grid,
  // OR we implement a saveGridToState() in switchModule.
  // Given the complexity, we'll assume grid is transient until processed or we add explicit save.
  // However, the user asked to "preserve what was run". 
  // If they run, we can save the grid data from the payload?
  // Or we can just let the grid stay in DOM if we don't clear it?
  // Wait, switchModule calls renderModuleUI which might clear the grid if we don't restore it.
  // In renderModuleUI, I added a check: if nComp/nSpecies are 0, clear. Otherwise leave it?
  // If we leave it, it might show the WRONG grid if we switch from Spec to NMR and back.
  // Actually, NMR doesn't use the model grid. So it's fine.
  // But if we have two Spec tabs (hypothetically), it would be an issue.
  // Here we have Spec vs NMR. NMR hides the model panel?
  // Let's check updateUI.
  // updateUI hides efaRow, spectraSheetRow, nmrSheetRow.
  // It does NOT hide the Model/Optimization panels.
  // So the Model grid IS shared if we don't clear/restore it.
  // This means we MUST save/restore the grid if we want isolation.
  // But NMR doesn't use the model grid, so maybe we just clear it when switching to NMR?
  // And restore when switching to Spec?
  // Since NMR doesn't use it, maybe it's fine if it stays there, but it might be confusing.
  // The user said "NMR ↔ Spectroscopy se “contamina”".
  // If I type in the grid in Spec, switch to NMR, the grid is still there.
  // If NMR doesn't use it, it's just visual clutter.
  // But if I switch back to Spec, I want it back.
  // So I should save it.
  // I'll add a helper to save grid state in switchModule.

  // --- Helper: Update Receptor/Guest Dropdowns ---
  function updateDropdowns() {
    if (!receptorInput || !guestInput) return;

    const currentReceptor = receptorInput.value;
    const currentGuest = guestInput.value;

    const selectedCols = Array.from(columnsContainer.querySelectorAll('input[type="checkbox"]:checked'))
      .map(cb => cb.value);

    [receptorInput, guestInput].forEach(select => {
      select.innerHTML = "";
      const defaultOpt = document.createElement("option");
      defaultOpt.value = "";
      defaultOpt.text = "";
      select.add(defaultOpt);

      selectedCols.forEach(col => {
        const opt = document.createElement("option");
        opt.value = col;
        opt.text = col;
        select.add(opt);
      });
    });

    if (selectedCols.includes(currentReceptor)) receptorInput.value = currentReceptor;
    if (selectedCols.includes(currentGuest)) guestInput.value = currentGuest;
  }

  // Listen for checkbox changes
  columnsContainer.addEventListener("change", (e) => {
    if (e.target.matches('input[type="checkbox"]')) {
      updateDropdowns();
    }
  });

  // Also observe DOM changes (when checkboxes are added)
  const observer = new MutationObserver(() => {
    updateDropdowns();
  });
  observer.observe(columnsContainer, { childList: true, subtree: true });

  // --- Scoped Reset Functions ---
  // --- Scoped Reset Functions ---

  function clearPlotsControlsDOM() {
    const preset = document.getElementById("spectro-plot-preset-select");
    const xAxis = document.getElementById("spectro-x-axis-select");
    const ySel = document.getElementById("spectro-y-series-select");

    if (preset) {
      preset.innerHTML = '<option value="">Select a preset...</option>';
      preset.value = "";
    }
    if (xAxis) {
      xAxis.innerHTML = '<option value="">Select X axis...</option>';
      xAxis.value = "";
    }
    if (ySel) {
      ySel.innerHTML = "";
    }
  }

  function resetActiveModule() {
    // Reset state
    state.modules[state.activeModule] = makeDefaultModuleState();

    // Update UI
    renderModuleUI();
    clearPlotsControlsDOM();

    // Explicitly clear file input value in DOM to ensure it doesn't persist visually
    const fileInput = document.getElementById("excel-file");
    if (fileInput) fileInput.value = "";

    // Clear Canvas
    const primaryPlot = document.querySelector('.primary-plot');
    if (primaryPlot) primaryPlot.innerHTML = '';
    const counter = document.getElementById('plot-counter');
    if (counter) counter.textContent = '—';

    // Clear Console
    setConsoleText("");

    diagEl.textContent = `${state.activeModule === 'nmr' ? 'NMR' : 'Spectroscopy'} reset.`;
  }

  // --- Handler: Reset ---
  // --- Handler: Reset ---
  resetBtn.addEventListener("click", () => {
    resetActiveModule();
  });

  // --- Handler: Save Results ---
  // --- Handler: Save Results ---
  if (saveBtn) {
    saveBtn.addEventListener("click", async () => {
      const m = M();
      const payload = m.resultsPayload;
      const resultsText = (m.resultsText || diagEl.textContent || "").trim();
      if (!payload || !payload.success) {
        appendLog("No hay resultados para guardar.");
        return;
      }
      const filename = "hmfit_results.xlsx";

      const fetchXlsx = async () => {
        const resp = await fetch(`${BACKEND_BASE_URL}/export_results_xlsx`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            constants: payload.constants || [],
            statistics: payload.statistics || {},
            results_text: resultsText,
            export_data: payload.export_data || {},
          }),
        });
        if (!resp.ok) {
          const detail = await resp.text();
          throw new Error(detail || `HTTP ${resp.status}`);
        }
        const ab = await resp.arrayBuffer();
        return new Uint8Array(ab);
      };

      try {
        // Usar diálogo nativo de Tauri
        const savePath = await save({
          title: 'Guardar resultados de HM Fit',
          defaultPath: filename,
          filters: [{ name: 'Excel files', extensions: ['xlsx'] }]
        });

        if (!savePath) {
          // Usuario canceló
          return;
        }

        const data = await fetchXlsx();
        await writeBinaryFile({ path: savePath, contents: data });

        const baseText = m.resultsText || "";
        diagEl.textContent = `${baseText}\n\nResultados guardados como ${savePath}`;

      } catch (err) {
        const baseText = state.latestResultsText || "";
        diagEl.textContent = `${baseText}\n\nNo se pudieron guardar los resultados: ${err.message || err}`;
        console.error("Save error:", err);

        // Fallback: simple download anchor if native save fails
        try {
          console.log("Attempting fallback download...");
          const data = await fetchXlsx();
          const blob = new Blob([data], {
            type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
          });
          const url = URL.createObjectURL(blob);
          const link = document.createElement("a");
          link.href = url;
          link.download = filename;
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
          URL.revokeObjectURL(url);

          const baseText = state.latestResultsText || "";
          diagEl.textContent = `${baseText}\n\nResultados guardados (fallback) como ${filename}`;
        } catch (fallbackErr) {
          const baseText = state.latestResultsText || "";
          diagEl.textContent = `${baseText}\n\nNo se pudieron guardar los resultados (ni nativo ni fallback): ${err.message} / ${fallbackErr.message}`;
        }
      }
    });
  }

  // --- Handler: Process Data ---
  processBtn.addEventListener("click", async () => {
    if (isProcessing) {
      appendLog("Procesamiento en curso, espera a que termine...");
      return;
    }

    const m = M();
    if (!m.file) {
      diagEl.textContent = "Error: No file selected. Please select a file first.";
      return;
    }

    if (state.activeModule === "spectroscopy") {
      const chan = revalidateSpectroscopyChannels();
      if (chan?.mode === "custom" && !chan.ok) {
        const errs = chan?.errors || [];
        diagEl.textContent = `Channels inválido:\n${errs.join("\n") || "Revisa el campo Channels."}`;
        return;
      }
    }

    setProcessing(true);
    diagEl.textContent = "Verificando backend...\n";
    try {
      await assertBackendAvailable();
    } catch (err) {
      diagEl.textContent = err.message || String(err);
      setProcessing(false);
      console.error("[HM Fit] Backend check failed before processing:", err);
      return;
    }

    connectWebSocket();
    diagEl.textContent = `Procesando datos de ${state.activeModule === 'nmr' ? 'NMR' : 'Spectroscopy'}...\n`;

    // Recolectar columnas seleccionadas
    let selectedCols = [];
    if (state.activeModule === 'nmr') {
      // For NMR, columns are selected from the signals container if we are in NMR mode?
      // Actually, the UI has "Concentration Sheet" columns AND "Chemical Shift Sheet" signals.
      // The backend expects 'column_names' for concentration columns.
      selectedCols = Array.from(columnsContainer.querySelectorAll('input[type="checkbox"]:checked'))
        .map(cb => cb.value);
    } else {
      selectedCols = Array.from(columnsContainer.querySelectorAll('input[type="checkbox"]:checked'))
        .map(cb => cb.value);
    }

    // Recolectar datos del grid del modelo
    const gridData = [];
    if (modelGridContainer) {
      const rows = modelGridContainer.querySelectorAll("tbody tr");
      rows.forEach(row => {
        const inputs = row.querySelectorAll(".grid-input");
        const rowData = Array.from(inputs).map(inp => parseFloat(inp.value) || 0);
        gridData.push(rowData);
      });
    }

    // Extraer especies no absorbentes
    const nonAbsSpecies = modelGridContainer
      ? Array.from(modelGridContainer.querySelectorAll("tr.selected"))
        .map(tr => parseInt(tr.dataset.species.replace('sp', '')) - 1)
      : [];

    // Extraer parámetros de optimización (K values)
    const kValues = [];
    const kBounds = [];
    if (optGridContainer) {
      const rows = optGridContainer.querySelectorAll("tbody tr");
      rows.forEach(row => {
        const inputs = row.querySelectorAll(".grid-input");
        if (inputs.length >= 3) {
          const val = parseFloat(inputs[0].value);
          const min = parseFloat(inputs[1].value);
          const max = parseFloat(inputs[2].value);

          // Use Number.isNaN to check for valid numbers, allowing 0
          const finalVal = Number.isNaN(val) ? 1.0 : val;
          // Mantener la semántica wx: límites vacíos -> ±inf (se traducen a null y el backend los convierte)
          const finalMin = Number.isNaN(min) ? null : min;
          const finalMax = Number.isNaN(max) ? null : max;

          kValues.push(finalVal);
          kBounds.push([finalMin, finalMax]);
        }
      });
    }

    // Create FormData with file and parameters
    const formData = new FormData();
    formData.append("file", m.file);
    formData.append("conc_sheet", concSheetInput?.value || "");
    formData.append("column_names", JSON.stringify(selectedCols));
    formData.append("receptor_label", receptorInput?.value || "");
    formData.append("guest_label", guestInput?.value || "");

    formData.append("modelo", JSON.stringify(gridData));
    formData.append("non_abs_species", JSON.stringify(nonAbsSpecies));
    formData.append("algorithm", algoSelect?.value || "Newton-Raphson");
    formData.append("model_settings", modelSettingsSelect?.value || "Free");
    formData.append("optimizer", optimizerSelect?.value || "powell");
    formData.append("initial_k", JSON.stringify(kValues));
    formData.append("bounds", JSON.stringify(kBounds));

    try {
      let data;
      if (state.activeModule === "nmr") {
        const nmrSheet = nmrSheetInput?.value;
        if (!nmrSheet) throw new Error("NMR Chemical Shift sheet not selected.");

        const selectedSignals = Array.from(
          nmrSignalsContainer.querySelectorAll('input[type="checkbox"]:checked')
        ).map(cb => cb.value);

        if (selectedSignals.length === 0) throw new Error("No NMR signals selected.");

        formData.append("spectra_sheet", nmrSheet); // Backend expects spectra_sheet as the signal source
        formData.append("signals_sheet", nmrSheet);
        formData.append("signal_names", JSON.stringify(selectedSignals));

        data = await backendApi.processNmr(formData);
        displayNmrResults(data);
        displayGraphs(data.graphs || {}, data);

      } else {
        formData.append("spectra_sheet", spectraSheetInput?.value || "");
        formData.append("channels_raw", m.channelsRaw ?? (channelsInput?.value || "All"));
        formData.append("channels_mode", m.channelsMode || "all");
        formData.append("channels_resolved", JSON.stringify(m.channelsResolved || []));

        const efaToSend = (m.channelsMode === "all") ? (efaCheckbox?.checked ? "true" : "false") : "false";
        formData.append("efa_enabled", efaToSend);
        formData.append("efa_eigenvalues", readInt(efaEigenInput?.value).toString());

        data = await backendApi.processSpectroscopy(formData);
        displayResults(data);
        const graphs = data.legacy_graphs || data.graphs || {};
        displayGraphs(graphs, data);
      }

    } catch (err) {
      let message = `No se pudo procesar la solicitud. Detalle: ${err?.message || err}`;
      if (err?.message?.includes("Backend no accesible")) {
        message =
          `No se puede contactar al backend en ${BACKEND_BASE_URL}. ` +
          `Asegúrate de que el servidor FastAPI esté levantado y accesible.`;
      } else if (err?.status && err?.body) {
        message = `${err.body}`;
      } else if (err?.status) {
        message =
          `El backend devolvió un error (código ${err.status}). ` +
          `Consulta la consola para más detalles.`;
      }

      const m = M();
      m.resultsText = "";
      m.resultsPayload = null;
      // Reset main canvas on error
      m.plotState.availablePlots = [];
      m.plotData = {}; // Should be m.plotState.plotData? No, plotData is inside plotState.
      // Correct: m.plotState.plotData = {};
      m.plotState.plotData = {};

      diagEl.textContent = message;
      console.error(`[HM Fit] Process Data request failed (${state.activeModule}):`, err);
    } finally {
      setProcessing(false);
    }
  });

  // Helper function to display results
  function displayResults(data) {
    const m = M();
    if (!data.success) {
      const detail = data.detail || data.error || "Procesamiento falló.";
      m.resultsText = "";
      m.resultsPayload = null;
      m.plotState.availablePlots = [];
      m.plotState.plotData = {};
      appendToConsole(`\nError: ${detail}`);
      scrollDiagnosticsToBottom();
      return;
    }

    const fmt = (n, opts = {}) => {
      if (n === null || n === undefined || Number.isNaN(n)) return "—";
      if (opts.fixed) return Number(n).toFixed(opts.fixed);
      return Number(n).toExponential(opts.exp ?? 3);
    };

    let finalText;
    if (data.results_text) {
      finalText = (data.results_text ?? '').replace(/\r\n/g, '\n');
    } else {
      // Build a simple table if the backend didn't send a formatted one
      const constants = data.constants || [];
      const stats = data.statistics || {};

      const lines = [];
      lines.push("=== RESULTADOS ===", "");
      lines.push("Constantes:");
      constants.forEach((c) => {
        const name = c.name || "";
        lines.push(
          `${name}: log10(K) = ${fmt(c.log10K)} ± ${fmt(c.SE_log10K)}`
        );
        lines.push(
          `    K = ${fmt(c.K)} ± ${fmt(c.SE_K)} (${fmt(c.percent_error, { fixed: 2 })}%)`
        );
      });

      lines.push("", "Estadísticas:");
      lines.push(`RMS: ${fmt(stats.RMS)}`);
      lines.push(`Lack of fit: ${fmt(stats.lof)}%`);
      lines.push(`MAE: ${fmt(stats.MAE)}`);
      lines.push(`Optimizer: ${stats.optimizer || "—"}`);
      lines.push(`Eigenvalues: ${stats.eigenvalues ?? "—"}`);

      finalText = lines.join("\n");
    }

    const logText = (data.log_output ?? "").trim();
    if (logText) {
      // If log_output is present, it usually contains the full log including results
      // But since we are streaming logs, we might just want to append the final table if it's not in the logs
      // For now, let's trust results_text which is the formatted table
    }

    if (finalText) {
      appendToConsole(`\n${finalText}`);
    }

    const warnings = Array.isArray(data.warnings) ? data.warnings : [];
    if (warnings.length) {
      appendToConsole(`\nWarnings:\n${warnings.map((w) => `- ${w}`).join("\n")}`);
    }
    if (data.plot_mode) {
      const used = (data.channels_used ?? "—");
      const total = (data.channels_total ?? "—");
      appendToConsole(`\nPlot mode: ${data.plot_mode} (channels ${used} / ${total})`);
    }

    m.resultsText = finalText;
    m.resultsPayload = data;
    scrollDiagnosticsToBottom();
  }

  function displayNmrResults(data) {
    const m = M();
    if (!data?.success) {
      const detail = data?.detail || data?.error || "Procesamiento NMR falló.";
      appendToConsole(`\nError: ${detail}`);
      m.resultsText = "";
      m.resultsPayload = null;
      scrollDiagnosticsToBottom();
      return;
    }

    if (data.results_text) {
      const normalized = (data.results_text ?? '').replace(/\r\n/g, '\n');
      appendToConsole(`\n${normalized}`);
      m.resultsText = normalized;
      m.resultsPayload = data;
      scrollDiagnosticsToBottom();
      return;
    }

    // Fallback if no text provided
    const lines = [];
    lines.push("=== NMR RESULTS ===", "");
    lines.push("Processing complete.");
    const fallbackText = lines.join("\n");
    appendToConsole(`\n${fallbackText}`);
    m.resultsText = fallbackText;
    m.resultsPayload = data;
    scrollDiagnosticsToBottom();
  }

  // === Main Canvas Functions ===
  function setMainCanvasResults(resultPayload) {
    // Extract availablePlots and plotData from backend response
    const ps = M().plotState;
    ps.availablePlots = resultPayload.availablePlots || [];
    ps.plotData = resultPayload.plotData || {};
    ps.activePlotIndex = 0;

    // Clear plot overrides/defaults on new calculation (per module)
    const modKey = state.activeModule;
    state.plotOverrides[modKey] = {};
    state.plotDefaults[modKey] = {};

    // Spectroscopy: convert legacy image plots to Plotly + build numeric-backed data objects
    if (modKey === 'spectroscopy') {
      ps.availablePlots = ps.availablePlots.map(p => (
        p?.kind === 'image' ? { ...p, kind: 'plotly' } : p
      ));

      const specData = ps.plotData?.spec || {};
      const plotData = resultPayload?.plot_data || {};
      const numerics = plotData?.numerics || {};
      const exportData = resultPayload?.export_data || {};

      const dist = specData?.spec_species_distribution || {};
      const xTitrant = dist?.axisVectors?.titrant_total || [];
      const xLabel = (dist?.axisOptions || []).find(a => a.id === 'titrant_total')?.label || '[X]';

      const nm = numerics.nm || exportData.nm || [];
      const transpose2d = (m) => {
        if (!Array.isArray(m) || !Array.isArray(m[0])) return m;
        const rows = m.length;
        const cols = m[0].length;
        const out = Array.from({ length: cols }, () => Array(rows));
        for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) out[j][i] = m[i][j];
        return out;
      };

      let Yexp = numerics.Y_exp || exportData.Y || [];
      let Yfit = numerics.Y_fit || exportData.yfit || [];
      if (Array.isArray(nm) && nm.length && Array.isArray(Yexp) && Array.isArray(Yexp[0])) {
        if (Yexp.length !== nm.length && Yexp[0].length === nm.length) Yexp = transpose2d(Yexp);
        if (Array.isArray(Yfit) && Array.isArray(Yfit[0]) && Yfit.length !== nm.length && Yfit[0].length === nm.length) {
          Yfit = transpose2d(Yfit);
        }
      }

      let A = exportData.A || [];
      const A_nm = exportData.A_index || exportData.nm || nm;
      if (Array.isArray(A_nm) && A_nm.length && Array.isArray(A) && Array.isArray(A[0])) {
        if (A.length !== A_nm.length && A[0].length === A_nm.length) A = transpose2d(A);
      }

      ps.plotData.spec = {
        ...specData,
        spec_fit_overlay: {
          plotMode: resultPayload?.plot_mode || 'spectra',
          xTitrant,
          xLabel,
          nm,
          Yexp,
          Yfit,
        },
        spec_molar_absorptivities: {
          nm: A_nm,
          A,
          speciesOptions: dist?.speciesOptions || [],
        },
        spec_efa_eigenvalues: {
          eigenvalues: numerics.eigenvalues || [],
        },
        spec_efa_components: {
          xTitrant,
          xLabel,
          efaForward: numerics.efa_forward || [],
          efaBackward: numerics.efa_backward || [],
        },
      };
    }

    // Reset interactive controls (will be auto-populated on first render)
    ps.controls.distXAxisId = 'titrant_total';
    ps.controls.distYSelected.clear();
    ps.controls.nmrSignalsSelected.clear();
    ps.controls.nmrResidSelected.clear();

    renderMainCanvasPlot();
    updatePresetDropdownFromResults();
  }

  function renderMainCanvasPlot() {
    const container = document.querySelector('.primary-plot');
    const counterEl = document.getElementById('plot-counter');
    const prevBtn = document.getElementById('plot-prev-btn');
    const nextBtn = document.getElementById('plot-next-btn');
    const sidePrev = document.getElementById('plot-prev-side');
    const sideNext = document.getElementById('plot-next-side');

    const ps = M().plotState;
    const plots = ps.availablePlots;
    const index = ps.activePlotIndex;

    // Helper to enable/disable nav
    const setNavEnabled = (enabled) => {
      if (prevBtn) prevBtn.disabled = !enabled;
      if (nextBtn) nextBtn.disabled = !enabled;
      if (sidePrev) sidePrev.disabled = !enabled;
      if (sideNext) sideNext.disabled = !enabled;
    };

    if (!plots.length) {
      if (container) container.innerHTML = '<p style="color: #9ca3af; padding: 2rem;">No plots available</p>';
      if (counterEl) counterEl.textContent = '—';
      setNavEnabled(false);
      return;
    }

    const plot = getActivePlot();
    const data = getActivePlotData();

    if (counterEl) counterEl.textContent = `${index + 1} / ${plots.length}`;
    setNavEnabled(true);

    if (!container) return;

    // Render based on plot kind
    if (plot.kind === 'plotly') {
      container.innerHTML = `
        <div class="plot-title">${plot.title}</div>
        <div class="main-plotly"></div>
      `;
      const plotDiv = container.querySelector('.main-plotly');

      if (!window.Plotly) {
        appendLog("ERROR: Plotly not loaded");
        if (plotDiv) plotDiv.innerHTML = '<p style="color: #ef4444; padding: 2rem;">Plotly not available</p>';
        return;
      }

      if (!data) {
        if (plotDiv) plotDiv.innerHTML = '<p style="color: #9ca3af; padding: 2rem;">No data for this plot</p>';
        return;
      }

      const figure = buildPlotlyFigure(plot.id, data, plot.title);
      figure.layout = figure.layout || {};
      figure.layout.uirevision = `hmfit-uirev:${state.activeModule}:${plot.id}`;

      // Snapshot defaults (first render per preset)
      const modKey = state.activeModule;
      if (!state.plotDefaults[modKey]?.[plot.id]) {
        state.plotDefaults[modKey][plot.id] = snapshotFigureDefaults(figure);
      }

      try {
        renderPlotly(plotDiv, figure);
      } catch (err) {
        console.error(err);
        if (plotDiv) plotDiv.innerHTML = `<p style="color: #ef4444; padding: 2rem;">Plotly render failed: ${err.message}</p>`;
        return;
      }

      // Apply overrides after base render
      const ov = state.plotOverrides?.[modKey]?.[plot.id];
      if (ov) applyPlotOverrides(plotDiv, ov);

      // Resize after render for Tauri
      requestAnimationFrame(() => {
        if (window.Plotly?.Plots?.resize) {
          window.Plotly.Plots.resize(plotDiv);
        }
      });

      // Sync edit panel inputs for the active Plotly chart
      requestAnimationFrame(() => syncEditPlotPanelFromDiv(plotDiv));
    } else {
      container.innerHTML = `<p style="color: #9ca3af; padding: 2rem;">Plot data not available: ${plot.id}</p>`;
    }

    // Sync controls for this plot
    syncPlotControlsForActivePlot();
  }

  // Build Plotly traces based on plot type
  // Build Plotly traces based on plot type
  function buildPlotlyTraces(plotId, data) {
    const traces = [];
    const controls = M().plotState.controls;

    if (plotId === 'spec_species_distribution' || plotId === 'nmr_species_distribution') {
      // Species distribution plot
      const xAxisId = controls.distXAxisId || 'titrant_total';
      const x = data.axisVectors?.[xAxisId] || data.x_default || [];
      const speciesOptions = data.speciesOptions || [];
      const C_by_species = data.C_by_species || {};

      // If no species selected, select all
      if (controls.distYSelected.size === 0) {
        speciesOptions.forEach(opt => controls.distYSelected.add(opt.id));
      }

      speciesOptions.forEach(opt => {
        if (controls.distYSelected.has(opt.id)) {
          const y = C_by_species[opt.id] || [];
          if (y.length > 0) {
            traces.push({
              x: x,
              y: y,
              mode: 'lines+markers',
              name: opt.label,
              line: { width: 2 },
              marker: { size: 6 }
            });
          }
        }
      });
    } else if (plotId === 'nmr_shifts_fit') {
      // NMR chemical shifts fit
      const x = data.x || [];
      const signalOptions = data.signalOptions || [];
      const signals = data.signals || {};

      // If no signals selected, select all
      if (controls.nmrSignalsSelected.size === 0) {
        signalOptions.forEach(opt => controls.nmrSignalsSelected.add(opt.id));
      }

      signalOptions.forEach(opt => {
        if (controls.nmrSignalsSelected.has(opt.id) && signals[opt.id]) {
          // Observed points
          traces.push({
            x: x,
            y: signals[opt.id].obs,
            mode: 'markers',
            name: `${opt.label} obs`,
            marker: { size: 8 }
          });
          // Fit line
          traces.push({
            x: x,
            y: signals[opt.id].fit,
            mode: 'lines',
            name: `${opt.label} fit`,
            line: { width: 2, dash: 'dot' }
          });
        }
      });
    } else if (plotId === 'nmr_residuals') {
      // NMR residuals
      const x = data.x || [];
      const signalOptions = data.signalOptions || [];
      const signals = data.signals || {};

      // If no signals selected, select all
      if (controls.nmrResidSelected.size === 0) {
        signalOptions.forEach(opt => controls.nmrResidSelected.add(opt.id));
      }

      signalOptions.forEach(opt => {
        if (controls.nmrResidSelected.has(opt.id) && signals[opt.id]) {
          traces.push({
            x: x,
            y: signals[opt.id].resid,
            mode: 'markers',
            name: `${opt.label} resid`,
            marker: { size: 6 }
          });
        }
      });
    } else if (plotId === 'spec_fit_overlay') {
      const plotMode = data.plotMode || 'spectra';
      const nm = data.nm || [];
      const Yexp = data.Yexp || [];
      const Yfit = data.Yfit || [];
      const xT = data.xTitrant || [];

      if (plotMode === 'isotherms') {
        const k = Math.min(Array.isArray(nm) ? nm.length : 0, 10);
        for (let i = 0; i < k; i++) {
          const yObs = Yexp?.[i] || [];
          const yFit = Yfit?.[i] || [];
          const wl = nm?.[i];
          traces.push({
            x: xT,
            y: yObs,
            mode: 'markers',
            name: `${wl} obs`,
            marker: { size: 7 },
          });
          traces.push({
            x: xT,
            y: yFit,
            mode: 'lines',
            name: `${wl} fit`,
            line: { width: 2, dash: 'dot' },
          });
        }
      } else {
        const nSteps = (Array.isArray(Yexp) && Array.isArray(Yexp[0])) ? Yexp[0].length : 0;
        for (let j = 0; j < nSteps; j++) {
          const yObs = Yexp.map(row => row?.[j]).filter(v => v !== undefined);
          const yFit = Yfit.map(row => row?.[j]).filter(v => v !== undefined);
          traces.push({
            x: nm,
            y: yObs,
            mode: 'lines',
            name: `exp ${j + 1}`,
            line: { width: 1, color: 'rgba(0,0,0,0.35)' },
            legendgroup: 'exp',
            showlegend: j === 0,
            hoverinfo: 'x+y',
          });
          traces.push({
            x: nm,
            y: yFit,
            mode: 'lines',
            name: `fit ${j + 1}`,
            line: { width: 1.5, dash: 'dot', color: 'rgba(239,68,68,0.8)' },
            legendgroup: 'fit',
            showlegend: j === 0,
            hoverinfo: 'x+y',
          });
        }
      }
    } else if (plotId === 'spec_molar_absorptivities') {
      const nm = data.nm || [];
      const A = data.A || [];
      const speciesOptions = data.speciesOptions || [];
      const nSpecies = Array.isArray(A?.[0]) ? A[0].length : 0;

      for (let s = 0; s < nSpecies; s++) {
        const y = A.map(row => row?.[s]).filter(v => v !== undefined);
        const label = speciesOptions?.[s]?.label || `sp${s + 1}`;
        traces.push({
          x: nm,
          y,
          mode: 'lines+markers',
          name: label,
          line: { width: 2 },
          marker: { size: 5 },
        });
      }
    } else if (plotId === 'spec_efa_eigenvalues') {
      const ev = data.eigenvalues || [];
      if (Array.isArray(ev) && ev.length) {
        traces.push({
          x: ev.map((_, i) => i + 1),
          y: ev.map(v => (v > 0 ? Math.log10(v) : null)),
          mode: 'markers+lines',
          name: 'log10(EV)',
          marker: { size: 7 },
        });
      }
    } else if (plotId === 'spec_efa_components') {
      const x = data.xTitrant || [];
      const fwd = data.efaForward || [];
      const bwd = data.efaBackward || [];
      const nComp = Array.isArray(fwd?.[0]) ? fwd[0].length : 0;
      for (let c = 0; c < nComp; c++) {
        const yF = fwd.map(row => {
          const v = row?.[c];
          return (v > 0 ? Math.log10(v) : null);
        });
        const yB = bwd.map(row => {
          const v = row?.[c];
          return (v > 0 ? Math.log10(v) : null);
        });
        traces.push({
          x,
          y: yF,
          mode: 'lines+markers',
          name: `fwd ${c + 1}`,
          marker: { size: 6 },
        });
        traces.push({
          x,
          y: yB,
          mode: 'lines+markers',
          name: `bwd ${c + 1}`,
          marker: { size: 6 },
          line: { dash: 'dot' },
        });
      }
    }

    return traces;
  }

  // Build Plotly layout based on plot type
  function buildPlotlyLayout(plotId, data) {
    const layout = {
      margin: { t: 24, r: 24, b: 48, l: 64 },
      legend: { orientation: 'h', y: -0.2 },
      xaxis: { title: { text: '' } },
      yaxis: { title: { text: '' } }
    };

    if (plotId === 'spec_species_distribution' || plotId === 'nmr_species_distribution') {
      const xAxisId = M().plotState.controls.distXAxisId || 'titrant_total';
      const axisOption = (data.axisOptions || []).find(a => a.id === xAxisId);
      layout.xaxis.title.text = axisOption?.label || 'Concentration';
      layout.yaxis.title.text = '[Species], M';
    } else if (plotId === 'nmr_shifts_fit') {
      layout.xaxis.title.text = data.xLabel || 'Concentration';
      layout.yaxis.title.text = 'Δδ (ppm)';
    } else if (plotId === 'nmr_residuals') {
      layout.xaxis.title.text = data.xLabel || 'Concentration';
      layout.yaxis.title.text = 'Residuals (ppm)';
    } else if (plotId === 'spec_fit_overlay') {
      const plotMode = data.plotMode || 'spectra';
      layout.xaxis.title.text = plotMode === 'isotherms' ? (data.xLabel || '[X]') : 'λ (nm)';
      layout.yaxis.title.text = 'Y observed (u. a.)';
    } else if (plotId === 'spec_molar_absorptivities') {
      layout.xaxis.title.text = 'λ (nm)';
      layout.yaxis.title.text = 'Epsilon (u. a.)';
    } else if (plotId === 'spec_efa_eigenvalues') {
      layout.xaxis.title.text = '# eigenvalues';
      layout.yaxis.title.text = 'log10(EV)';
    } else if (plotId === 'spec_efa_components') {
      layout.xaxis.title.text = data.xLabel || '[X]';
      layout.yaxis.title.text = 'log10(EV)';
    }

    return layout;
  }

  function buildPlotlyFigure(plotId, data, title) {
    const traces = buildPlotlyTraces(plotId, data);
    const layout = buildPlotlyLayout(plotId, data);
    layout.title = { text: title || plotId };
    if (!traces.length) {
      return {
        data: [],
        layout: {
          ...layout,
          annotations: [{
            text: 'No data',
            xref: 'paper',
            yref: 'paper',
            x: 0.5,
            y: 0.5,
            showarrow: false,
          }],
        },
      };
    }
    return { data: traces, layout };
  }

  function snapshotFigureDefaults(figure) {
    const layout = figure?.layout || {};
    const data = figure?.data || [];
    const titleText = layout?.title?.text ?? '';
    const xLabel = layout?.xaxis?.title?.text ?? '';
    const yLabel = layout?.yaxis?.title?.text ?? '';
    const traceNames = {};
    data.forEach((t, idx) => {
      traceNames[String(idx)] = t?.name ?? '';
    });
    return { titleText, xLabel, yLabel, traceNames };
  }

  function getActivePlotDiv() {
    return document.querySelector('.primary-plot .main-plotly');
  }

  function syncEditPlotPanelFromDiv(plotDiv) {
    const titleInput = document.getElementById('plot-edit-title');
    const xInput = document.getElementById('plot-edit-xlabel');
    const yInput = document.getElementById('plot-edit-ylabel');
    const traceSelect = document.getElementById('plot-edit-trace-select');
    const traceNameInput = document.getElementById('plot-edit-trace-name');

    const plot = getActivePlot();
    if (!plot || plot.kind !== 'plotly' || !plotDiv || !plotDiv.layout) return;

    const titleText = plotDiv.layout?.title?.text ?? '';
    const xLabel = plotDiv.layout?.xaxis?.title?.text ?? '';
    const yLabel = plotDiv.layout?.yaxis?.title?.text ?? '';

    if (titleInput) titleInput.value = titleText;
    if (xInput) xInput.value = xLabel;
    if (yInput) yInput.value = yLabel;

    if (traceSelect) {
      traceSelect.innerHTML = '';
      const defaultOpt = document.createElement('option');
      defaultOpt.value = '';
      defaultOpt.text = 'Select trace...';
      traceSelect.add(defaultOpt);
      const traces = plotDiv.data || [];
      traces.forEach((t, idx) => {
        const opt = document.createElement('option');
        opt.value = String(idx);
        opt.text = `${idx}: ${t?.name ?? ''}`;
        traceSelect.add(opt);
      });
      if (traceNameInput) traceNameInput.value = '';
    }
  }

  // Sync plot controls in Plots tab based on active plot
  function syncPlotControlsForActivePlot() {
    const xAxisSelect = document.getElementById('spectro-x-axis-select');
    const ySeriesSelect = document.getElementById('spectro-y-series-select');
    const varySelect = document.getElementById('spectro-vary-along-select');

    const plot = getActivePlot();
    const data = getActivePlotData();

    // Handle no plot or no data: show placeholder and disable
    if (!plot) {
      if (xAxisSelect) {
        xAxisSelect.innerHTML = '<option value="">No data (run process first)</option>';
        xAxisSelect.disabled = true;
      }
      if (ySeriesSelect) {
        ySeriesSelect.innerHTML = '<option value="">No data (run process first)</option>';
        ySeriesSelect.disabled = true;
      }
      return;
    }

    // Only show axis/series controls for plots that support them
    const supportsControls = (
      plot.id === 'spec_species_distribution' ||
      plot.id === 'nmr_species_distribution' ||
      plot.id === 'nmr_shifts_fit' ||
      plot.id === 'nmr_residuals'
    );
    const isInteractive = plot.kind === 'plotly';
    if (xAxisSelect?.closest('.field')) xAxisSelect.closest('.field').style.display = isInteractive ? '' : 'none';
    if (ySeriesSelect?.closest('.field')) ySeriesSelect.closest('.field').style.display = isInteractive ? '' : 'none';
    if (varySelect?.closest('.field')) varySelect.closest('.field').style.display = 'none'; // Always hide vary for now

    // Hide controls for Plotly plots without controls
    if (isInteractive && !supportsControls) {
      if (xAxisSelect?.closest('.field')) xAxisSelect.closest('.field').style.display = 'none';
      if (ySeriesSelect?.closest('.field')) ySeriesSelect.closest('.field').style.display = 'none';
      return;
    }

    // Handle null data for interactive plots
    if (!isInteractive) return;

    if (!data) {
      if (xAxisSelect) {
        xAxisSelect.innerHTML = '<option value="">No data for this plot (check plotData keys)</option>';
        xAxisSelect.disabled = true;
      }
      if (ySeriesSelect) {
        ySeriesSelect.innerHTML = '<option value="">No data for this plot (check plotData keys)</option>';
        ySeriesSelect.disabled = true;
      }
      return;
    }

    // Enable selects since we have data
    if (xAxisSelect) xAxisSelect.disabled = false;
    if (ySeriesSelect) ySeriesSelect.disabled = false;

    if (plot.id === 'spec_species_distribution' || plot.id === 'nmr_species_distribution') {
      // Populate X axis dropdown
      if (xAxisSelect) {
        xAxisSelect.innerHTML = '';
        (data.axisOptions || []).forEach(opt => {
          const el = document.createElement('option');
          el.value = opt.id;
          el.text = opt.label;
          xAxisSelect.add(el);
        });
        xAxisSelect.value = M().plotState.controls.distXAxisId || 'titrant_total';
        const labelEl = xAxisSelect.closest('.field')?.querySelector('.field-label');
        if (labelEl) labelEl.textContent = 'X axis';
      }

      // Populate Y series (species) using speciesOptions
      if (ySeriesSelect) {
        ySeriesSelect.innerHTML = '';
        (data.speciesOptions || []).forEach(opt => {
          const el = document.createElement('option');
          el.value = opt.id;
          el.text = opt.label;
          el.selected = M().plotState.controls.distYSelected.has(opt.id);
          ySeriesSelect.add(el);
        });
        const labelEl = ySeriesSelect.closest('.field')?.querySelector('.field-label');
        if (labelEl) labelEl.textContent = 'Y species';
      }
    } else if (plot.id === 'nmr_shifts_fit') {
      // Hide X axis for NMR shifts
      if (xAxisSelect?.closest('.field')) xAxisSelect.closest('.field').style.display = 'none';

      // Populate signals using signalOptions
      if (ySeriesSelect) {
        ySeriesSelect.innerHTML = '';
        (data.signalOptions || []).forEach(opt => {
          const el = document.createElement('option');
          el.value = opt.id;
          el.text = opt.label;
          el.selected = M().plotState.controls.nmrSignalsSelected.has(opt.id);
          ySeriesSelect.add(el);
        });
        const labelEl = ySeriesSelect.closest('.field')?.querySelector('.field-label');
        if (labelEl) labelEl.textContent = 'Signals to display';
      }
    } else if (plot.id === 'nmr_residuals') {
      // Hide X axis for NMR residuals
      if (xAxisSelect?.closest('.field')) xAxisSelect.closest('.field').style.display = 'none';

      // Populate residual signals using signalOptions
      if (ySeriesSelect) {
        ySeriesSelect.innerHTML = '';
        (data.signalOptions || []).forEach(opt => {
          const el = document.createElement('option');
          el.value = opt.id;
          el.text = opt.label;
          el.selected = M().plotState.controls.nmrResidSelected.has(opt.id);
          ySeriesSelect.add(el);
        });
        const labelEl = ySeriesSelect.closest('.field')?.querySelector('.field-label');
        if (labelEl) labelEl.textContent = 'Residual signals';
      }
    }
  }

  function navigateMainCanvas(delta) {
    const ps = M().plotState;
    const total = ps.availablePlots.length;
    if (total === 0) return;

    ps.activePlotIndex = (ps.activePlotIndex + delta + total) % total;
    renderMainCanvasPlot();

    // Sync preset dropdown
    const presetSelect = document.getElementById('spectro-plot-preset-select');
    if (presetSelect && ps.availablePlots[ps.activePlotIndex]) {
      presetSelect.value = ps.availablePlots[ps.activePlotIndex].id;
    }
  }

  function updatePresetDropdownFromResults() {
    const presetSelect = document.getElementById('spectro-plot-preset-select');
    if (!presetSelect) return;

    presetSelect.innerHTML = '';

    const defaultOpt = document.createElement('option');
    defaultOpt.value = '';
    defaultOpt.text = 'Select a preset...';
    presetSelect.add(defaultOpt);

    const ps = M().plotState;
    ps.availablePlots.forEach(plot => {
      const opt = document.createElement('option');
      opt.value = plot.id;
      opt.text = plot.title;
      presetSelect.add(opt);
    });

    // Select first plot by default
    if (ps.availablePlots.length > 0) {
      presetSelect.value = ps.availablePlots[0].id;
    }
  }

  function exportMainCanvasPNG() {
    const ps = M().plotState;
    const currentPlot = ps.availablePlots[ps.activePlotIndex];
    const filename = `${currentPlot?.id || 'plot'}.png`;

    if (currentPlot?.kind === 'plotly') {
      // Export Plotly chart as PNG
      const plotDiv = document.querySelector('.primary-plot .main-plotly');
      if (plotDiv && window.Plotly) {
        window.Plotly.toImage(plotDiv, { format: 'png', width: 1200, height: 800 })
          .then(dataUrl => {
            const link = document.createElement('a');
            link.href = dataUrl;
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            appendLog(`Exported: ${filename}`);
          })
          .catch(err => appendLog(`Export failed: ${err.message}`));
      } else {
        appendLog('Cannot export: Plotly not available');
      }
    } else {
      // Export image plot
      const imgEl = document.querySelector('.primary-plot img');
      if (!imgEl || !imgEl.src) {
        appendLog('No image to export');
        return;
      }
      const link = document.createElement('a');
      link.href = imgEl.src;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      appendLog(`Exported: ${filename}`);
    }
  }

  function exportMainCanvasCSV() {
    const ps = M().plotState;
    const currentPlot = ps.availablePlots[ps.activePlotIndex];

    if (!currentPlot || currentPlot.kind !== 'plotly') {
      appendLog('CSV export only available for interactive plots');
      return;
    }

    const dataSource = state.activeModule === 'nmr'
      ? ps.plotData?.nmr
      : ps.plotData?.spec;
    const data = dataSource?.[currentPlot.id];

    if (!data) {
      appendLog('No data to export');
      return;
    }

    let csv = '';
    const controls = ps.controls;

    if (currentPlot.id === 'spec_species_distribution' || currentPlot.id === 'nmr_species_distribution') {
      // Species distribution CSV
      const xAxisId = controls.distXAxisId || 'titrant_total';
      const x = data.axisVectors?.[xAxisId] || data.x_default || [];
      const speciesNames = data.speciesNames || [];
      const C = data.C || [];
      const selectedSpecies = controls.distYSelected.size > 0
        ? speciesNames.filter(sp => controls.distYSelected.has(sp))
        : speciesNames;

      // Header
      const axisOption = (data.axisOptions || []).find(a => a.id === xAxisId);
      csv = `${axisOption?.label || 'X'},${selectedSpecies.join(',')}\n`;

      // Data rows
      for (let i = 0; i < x.length; i++) {
        const row = [x[i]];
        selectedSpecies.forEach(sp => {
          const spIndex = speciesNames.indexOf(sp);
          row.push(C[i]?.[spIndex] ?? '');
        });
        csv += row.join(',') + '\n';
      }
    } else if (currentPlot.id === 'nmr_shifts_fit') {
      // NMR shifts fit CSV
      const x = data.x || [];
      const signalNames = data.signalNames || [];
      const signals = data.signals || {};
      const selectedSignals = controls.nmrSignalsSelected.size > 0
        ? signalNames.filter(sig => controls.nmrSignalsSelected.has(sig))
        : signalNames;

      // Header
      const headers = ['X'];
      selectedSignals.forEach(sig => {
        headers.push(`${sig}_obs`, `${sig}_fit`);
      });
      csv = headers.join(',') + '\n';

      // Data rows
      for (let i = 0; i < x.length; i++) {
        const row = [x[i]];
        selectedSignals.forEach(sig => {
          row.push(signals[sig]?.obs?.[i] ?? '', signals[sig]?.fit?.[i] ?? '');
        });
        csv += row.join(',') + '\n';
      }
    } else if (currentPlot.id === 'nmr_residuals') {
      // NMR residuals CSV
      const x = data.x || [];
      const signalNames = data.signalNames || [];
      const signals = data.signals || {};
      const selectedSignals = controls.nmrResidSelected.size > 0
        ? signalNames.filter(sig => controls.nmrResidSelected.has(sig))
        : signalNames;

      // Header
      csv = `X,${selectedSignals.map(s => `${s}_resid`).join(',')}\n`;

      // Data rows
      for (let i = 0; i < x.length; i++) {
        const row = [x[i]];
        selectedSignals.forEach(sig => {
          row.push(signals[sig]?.resid?.[i] ?? '');
        });
        csv += row.join(',') + '\n';
      }
    }

    if (!csv) {
      appendLog('No data to export');
      return;
    }

    const filename = `${currentPlot.id}.csv`;
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    appendLog(`Exported: ${filename}`);
  }

  // Legacy displayGraphs for backward compatibility (uses old format if availablePlots not present)
  function displayGraphs(graphs, resultPayload) {
    // If we have the new format, use it
    if (resultPayload?.availablePlots && resultPayload.availablePlots.length > 0) {
      setMainCanvasResults(resultPayload);
      return;
    }

    // Fallback to legacy carousel rendering
    const plotContainers = document.querySelectorAll(".plot-placeholder");
    const prevBtn = document.getElementById("plot-prev-btn");
    const nextBtn = document.getElementById("plot-next-btn");
    const counterEl = document.getElementById("plot-counter");
    const sidePrev = document.getElementById("plot-prev-side");
    const sideNext = document.getElementById("plot-next-side");

    // Clear previous content
    plotContainers.forEach((container) => {
      container.innerHTML = "";
    });

    const disableNav = () => {
      if (prevBtn) prevBtn.disabled = true;
      if (nextBtn) nextBtn.disabled = true;
      if (sidePrev) sidePrev.disabled = true;
      if (sideNext) sideNext.disabled = true;
      if (counterEl) counterEl.textContent = "—";
    };

    const mainPlots = [];
    if (graphs.concentrations) mainPlots.push({ name: "Concentrations", data: graphs.concentrations });
    if (graphs.fit) mainPlots.push({ name: "Fit", data: graphs.fit });
    if (graphs.residuals) mainPlots.push({ name: "Residuals", data: graphs.residuals });
    if (graphs.eigenvalues) mainPlots.push({ name: "Eigenvalues", data: graphs.eigenvalues });
    if (graphs.efa) mainPlots.push({ name: "EFA", data: graphs.efa });
    if (graphs.absorptivities) mainPlots.push({ name: "Absorptivities", data: graphs.absorptivities });

    // Clear secondary container explicitly (it might have old content)
    if (plotContainers[1]) {
      plotContainers[1].innerHTML = "";
      plotContainers[1].style.display = "none"; // Hide it to be sure
    }

    let slidesRef = [];
    let currentSlideIndex = 0;
    let updateSlide = () => { };
    const updateCounter = () => {
      if (!counterEl) return;
      if (!slidesRef.length) {
        counterEl.textContent = "—";
        return;
      }
      counterEl.textContent = `${currentSlideIndex + 1} / ${slidesRef.length}`;
    };

    const attachNav = () => {
      if (!prevBtn || !nextBtn) return;
      const hasSlides = slidesRef.length > 0;
      prevBtn.disabled = !hasSlides;
      nextBtn.disabled = !hasSlides;
      if (sidePrev) sidePrev.disabled = !hasSlides;
      if (sideNext) sideNext.disabled = !hasSlides;
      updateCounter();

      if (!hasSlides) {
        prevBtn.onclick = null;
        nextBtn.onclick = null;
        if (sidePrev) sidePrev.onclick = null;
        if (sideNext) sideNext.onclick = null;
        return;
      }

      prevBtn.onclick = () => {
        const target = (currentSlideIndex - 1 + slidesRef.length) % slidesRef.length;
        updateSlide(target);
      };
      nextBtn.onclick = () => {
        const target = (currentSlideIndex + 1) % slidesRef.length;
        updateSlide(target);
      };
      if (sidePrev) sidePrev.onclick = prevBtn.onclick;
      if (sideNext) sideNext.onclick = nextBtn.onclick;
    };

    const renderCarousel = (container, plots) => {
      if (!plots.length) {
        container.innerHTML = "<p style='color: #9ca3af;'>No plots</p>";
        disableNav();
        return;
      }

      // Create carousel structure
      const carouselContainer = document.createElement("div");
      carouselContainer.className = "carousel-container";

      // Slides
      plots.forEach((plot, index) => {
        const slide = document.createElement("div");
        slide.className = `carousel-slide ${index === 0 ? "active" : ""}`;
        slide.dataset.index = index;

        const title = document.createElement("div");
        title.className = "plot-title";
        title.textContent = plot.name;

        const img = document.createElement("img");
        img.src = `data:image/png;base64,${plot.data}`;
        img.alt = plot.name;

        slide.appendChild(title);
        slide.appendChild(img);
        carouselContainer.appendChild(slide);
      });

      container.appendChild(carouselContainer);

      slidesRef = Array.from(carouselContainer.querySelectorAll(".carousel-slide"));
      currentSlideIndex = 0;

      updateSlide = (newIndex) => {
        if (!slidesRef.length) return;
        slidesRef[currentSlideIndex].classList.remove("active");
        currentSlideIndex = newIndex;
        slidesRef[currentSlideIndex].classList.add("active");
        updateCounter();
      };

      updateSlide(0);
      attachNav();
    };

    renderCarousel(plotContainers[0], mainPlots);
  }
  // === Plot builder helpers (spectroscopy) ===
  function initSpectroPlotControls(plotData) {
    const presetSelect = document.getElementById("spectro-plot-preset-select");
    const xAxisSelect = document.getElementById("spectro-x-axis-select");
    const ySeriesSelect = document.getElementById("spectro-y-series-select");
    const varySelect = document.getElementById("spectro-vary-along-select");
    const container = document.getElementById("spectro-plot-container");

    const resetSelect = (select, placeholder) => {
      if (!select) return;
      select.innerHTML = "";
      const opt = document.createElement("option");
      opt.value = "";
      opt.text = placeholder;
      select.appendChild(opt);
    };

    if (!presetSelect || !xAxisSelect || !ySeriesSelect || !varySelect || !container) return;

    if (!plotData) {
      resetSelect(presetSelect, "Select a preset...");
      resetSelect(xAxisSelect, "Select X axis...");
      resetSelect(ySeriesSelect, "Select series...");
      resetSelect(varySelect, "Auto");
      container.textContent = "No plot data available.";
      spectroCurrentSeries = [];
      return;
    }

    const meta = plotData.plot_meta || {};
    const presets = meta.presets || [];
    const axes = meta.axes || {};
    const series = meta.series || {};

    resetSelect(presetSelect, "Select a preset...");
    presets.forEach((p) => {
      const opt = document.createElement("option");
      opt.value = p.id;
      opt.text = p.name || p.id;
      presetSelect.appendChild(opt);
    });

    resetSelect(xAxisSelect, "Select X axis...");
    Object.values(axes).forEach((ax) => {
      const opt = document.createElement("option");
      opt.value = ax.id;
      opt.text = ax.label ? `${ax.label} (${ax.unit || ''})`.trim() : ax.id;
      xAxisSelect.appendChild(opt);
    });

    ySeriesSelect.innerHTML = "";
    Object.values(series).forEach((s) => {
      const opt = document.createElement("option");
      opt.value = s.id;
      opt.text = s.label || s.id;
      ySeriesSelect.appendChild(opt);
    });
    if (!ySeriesSelect.options.length) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.text = "No series";
      ySeriesSelect.appendChild(opt);
    }

    resetSelect(varySelect, "Auto");
    const varyOptions = new Set();
    Object.values(series).forEach((s) => {
      (s.dims || []).forEach((d) => varyOptions.add(d));
    });
    ["titration_step", "species"].forEach((d) => varyOptions.add(d));
    varyOptions.forEach((v) => {
      if (!v) return;
      const opt = document.createElement("option");
      opt.value = v;
      opt.text = v;
      varySelect.appendChild(opt);
    });

    if (presets.length) {
      const first = presets[0];
      presetSelect.value = first.id;
      xAxisSelect.value = first.x_axis;
      Array.from(ySeriesSelect.options).forEach((opt) => {
        opt.selected = (first.y_series || []).includes(opt.value);
      });
      varySelect.value = first.vary_along || "";
    }

    buildSpectroPlotFromSelection();
  }

  function buildSpectroPlotFromSelection() {
    const container = document.getElementById("spectro-plot-container");
    const presetSelect = document.getElementById("spectro-plot-preset-select");
    const xAxisSelect = document.getElementById("spectro-x-axis-select");
    const ySeriesSelect = document.getElementById("spectro-y-series-select");
    const varySelect = document.getElementById("spectro-vary-along-select");

    if (!container || !window.spectroPlotData) return [];

    const plotData = window.spectroPlotData;
    const meta = plotData.plot_meta || {};
    const numerics = plotData.numerics || {};

    const presetId = presetSelect?.value || "";
    let xAxisId = xAxisSelect?.value || "";
    let ySeriesIds = Array.from(ySeriesSelect?.selectedOptions || []).map((o) => o.value).filter(Boolean);
    let varyAlong = varySelect?.value || "";

    if (presetId) {
      const preset = (meta.presets || []).find((p) => p.id === presetId);
      if (preset) {
        xAxisId = preset.x_axis;
        ySeriesIds = (preset.y_series || []).slice();
        varyAlong = preset.vary_along || "";
        if (xAxisSelect) xAxisSelect.value = xAxisId;
        if (varySelect) varySelect.value = varyAlong;
        if (ySeriesSelect) {
          Array.from(ySeriesSelect.options).forEach((opt) => {
            opt.selected = ySeriesIds.includes(opt.value);
          });
        }
      }
    }

    if (!xAxisId) {
      container.textContent = "Select an X axis to render a plot.";
      spectroCurrentSeries = [];
      return [];
    }
    if (!ySeriesIds.length) {
      container.textContent = "Select at least one Y series.";
      spectroCurrentSeries = [];
      return [];
    }

    const axisMeta = (meta.axes || {})[xAxisId];
    if (!axisMeta) {
      container.textContent = "Axis metadata not found.";
      spectroCurrentSeries = [];
      return [];
    }

    let x = [];
    if (axisMeta.values_key && numerics[axisMeta.values_key] !== undefined) {
      if (axisMeta.values_key === "Ct" && typeof axisMeta.column === "number") {
        const matrix = numerics.Ct || [];
        x = matrix.map((row) => (row ? row[axisMeta.column] : null));
      } else {
        x = numerics[axisMeta.values_key] || [];
      }
    }

    if (!x || !x.length) {
      container.textContent = "X data not available for this axis.";
      spectroCurrentSeries = [];
      return [];
    }

    const seriesData = [];
    ySeriesIds.forEach((seriesId) => {
      const sMeta = (meta.series || {})[seriesId];
      if (!sMeta) return;
      const raw = numerics[sMeta.data_key];
      if (!raw) return;

      if ((sMeta.dims || []).join(",") === "wavelength,titration_step") {
        const steps = raw[0] ? raw[0].length : 0;
        const varying = varyAlong || "titration_step";
        if (varying === "titration_step") {
          for (let step = 0; step < steps; step++) {
            const y = raw.map((row) => row[step]);
            seriesData.push({
              name: `${sMeta.label || seriesId} · step ${step + 1}`,
              x,
              y,
            });
          }
        } else {
          const y = raw.map((row) => row[0]);
          seriesData.push({ name: sMeta.label || seriesId, x, y });
        }
      } else if ((sMeta.dims || []).join(",") === "titration_step,species") {
        const nSpecies = raw[0] ? raw[0].length : 0;
        const nSteps = raw.length;
        const varying = varyAlong || "species";

        if (varying === "species") {
          for (let sp = 0; sp < nSpecies; sp++) {
            const y = raw.map((row) => row[sp]);
            const name = (sMeta.categories && sMeta.categories[sp]) ? sMeta.categories[sp] : `Species ${sp + 1}`;
            seriesData.push({ name, x, y });
          }
        } else if (varying === "titration_step") {
          for (let step = 0; step < nSteps; step++) {
            const row = raw[step] || [];
            seriesData.push({
              name: `Step ${step + 1}`,
              x,
              y: row,
            });
          }
        }
      } else if ((sMeta.dims || []).includes("eigenvalue_index")) {
        const y = raw || [];
        const ex = Array.from({ length: y.length }, (_, idx) => idx + 1);
        seriesData.push({ name: sMeta.label || seriesId, x: ex, y });
      } else {
        const y = Array.isArray(raw) ? raw : [raw];
        seriesData.push({ name: sMeta.label || seriesId, x, y });
      }
    });

    spectroCurrentSeries = seriesData;
    renderSpectroPlot(seriesData, axisMeta.label || axisMeta.id || "X", (ySeriesIds || []).join(", "));
    return seriesData;
  }

  function renderSpectroPlot(seriesData, xLabel, yLabel) {
    const container = document.getElementById("spectro-plot-container");
    if (!container) return;

    if (!seriesData || !seriesData.length) {
      container.textContent = "No data to plot.";
      spectroPlotLayout = null;
      return;
    }

    if (window.Plotly) {
      const traces = seriesData.map((s) => ({
        x: s.x,
        y: s.y,
        mode: "lines",
        name: s.name,
      }));
      const layout = {
        margin: { t: 24, r: 12, b: 48, l: 64 },
        xaxis: { title: xLabel || "X" },
        yaxis: { title: yLabel || "Y" },
        legend: { orientation: "h" },
      };
      window.Plotly.newPlot(container, traces, layout, { displaylogo: false, responsive: true });
      spectroPlotLayout = { traces, layout };
    } else {
      const summary = seriesData.map((s) => `${s.name}: ${s.y.length} pts`).join("\n");
      container.innerHTML = `<pre class="log-output">Plotly no está disponible.\n\n${summary}</pre>`;
      spectroPlotLayout = null;
    }
  }

  function exportSpectroPlotPNG() {
    const container = document.getElementById("spectro-plot-container");
    if (!container) return;
    if (!spectroCurrentSeries.length) {
      appendLog("No hay datos de plot para exportar.");
      return;
    }
    if (window.Plotly && container.data) {
      window.Plotly.toImage(container, { format: "png", height: 600, width: 900 }).then((url) => {
        const link = document.createElement("a");
        link.href = url;
        link.download = "spectroscopy_plot.png";
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      }).catch((err) => appendLog(`No se pudo exportar PNG: ${err?.message || err}`));
    } else {
      appendLog("Plotly no está disponible para exportar PNG.");
    }
  }

  function exportSpectroPlotCSV() {
    if (!spectroCurrentSeries.length) {
      appendLog("No hay datos de plot para exportar.");
      return;
    }
    const baseX = spectroCurrentSeries[0].x || [];
    const headers = ["x", ...spectroCurrentSeries.map((s, idx) => s.name || `serie_${idx + 1}`)];
    const rows = [headers.join(",")];

    for (let i = 0; i < baseX.length; i++) {
      const cols = [baseX[i]];
      spectroCurrentSeries.forEach((s) => {
        cols.push((s.y && s.y[i] !== undefined) ? s.y[i] : "");
      });
      rows.push(cols.join(","));
    }

    const blob = new Blob([rows.join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "spectroscopy_plot.csv";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }
  // --- Helper: Build Config Objects ---
  function buildSpecConfigFromState() {
    const selectedCols = Array.from(columnsContainer.querySelectorAll('input[type="checkbox"]:checked'))
      .map(cb => cb.value);

    const gridData = [];
    if (modelGridContainer) {
      const rows = modelGridContainer.querySelectorAll("tbody tr");
      rows.forEach(row => {
        const inputs = row.querySelectorAll(".grid-input");
        const rowData = Array.from(inputs).map(inp => parseFloat(inp.value) || 0);
        gridData.push(rowData);
      });
    }

    const nonAbsSpecies = modelGridContainer
      ? Array.from(modelGridContainer.querySelectorAll("tr.selected"))
        .map(tr => parseInt(tr.dataset.species.replace('sp', '')) - 1)
      : [];

    const kValues = [];
    const kBounds = [];
    if (optGridContainer) {
      const rows = optGridContainer.querySelectorAll("tbody tr");
      rows.forEach(row => {
        const inputs = row.querySelectorAll(".grid-input");
        if (inputs.length >= 3) {
          const val = parseFloat(inputs[0].value);
          const min = parseFloat(inputs[1].value);
          const max = parseFloat(inputs[2].value);
          const finalVal = Number.isNaN(val) ? 1.0 : val;
          const finalMin = Number.isNaN(min) ? null : min;
          const finalMax = Number.isNaN(max) ? null : max;
          kValues.push(finalVal);
          kBounds.push([finalMin, finalMax]);
        }
      });
    }

    return {
      type: 'Spectroscopy',
      version: 1,
      plots: {
        plotOverrides: state.plotOverrides.spectroscopy || {},
      },
      model: {
        nComp: readInt(nCompInput?.value),
        nSpecies: readInt(nSpeciesInput?.value),
        stoichiometry: gridData,
        nonAbsorbingSpecies: nonAbsSpecies,
        efaEnabled: efaCheckbox?.checked || false,
        efaEigenvalues: readInt(efaEigenInput?.value)
      },
      roles: {
        receptor: receptorInput?.value || "",
        guest: guestInput?.value || ""
      },
      columns: {
        conc: selectedCols,
        spectraSheet: spectraSheetInput?.value || "",
        concSheet: concSheetInput?.value || ""
      },
      optimization: {
        algorithm: algoSelect?.value,
        modelSettings: modelSettingsSelect?.value,
        optimizer: optimizerSelect?.value,
        initialK: kValues,
        bounds: kBounds
      }
    };
  }

  function buildNmrConfigFromState() {
    // Reuse similar logic but for NMR specific fields
    const selectedSignals = Array.from(nmrSignalsContainer.querySelectorAll('input[type="checkbox"]:checked'))
      .map(cb => cb.value);

    const selectedCols = Array.from(columnsContainer.querySelectorAll('input[type="checkbox"]:checked'))
      .map(cb => cb.value);

    const gridData = [];
    if (modelGridContainer) {
      const rows = modelGridContainer.querySelectorAll("tbody tr");
      rows.forEach(row => {
        const inputs = row.querySelectorAll(".grid-input");
        const rowData = Array.from(inputs).map(inp => parseFloat(inp.value) || 0);
        gridData.push(rowData);
      });
    }

    const nonAbsSpecies = modelGridContainer
      ? Array.from(modelGridContainer.querySelectorAll("tr.selected"))
        .map(tr => parseInt(tr.dataset.species.replace('sp', '')) - 1)
      : [];

    const kValues = [];
    const kBounds = [];
    if (optGridContainer) {
      const rows = optGridContainer.querySelectorAll("tbody tr");
      rows.forEach(row => {
        const inputs = row.querySelectorAll(".grid-input");
        if (inputs.length >= 3) {
          const val = parseFloat(inputs[0].value);
          const min = parseFloat(inputs[1].value);
          const max = parseFloat(inputs[2].value);
          const finalVal = Number.isNaN(val) ? 1.0 : val;
          const finalMin = Number.isNaN(min) ? null : min;
          const finalMax = Number.isNaN(max) ? null : max;
          kValues.push(finalVal);
          kBounds.push([finalMin, finalMax]);
        }
      });
    }

    return {
      type: 'NMR',
      version: 1,
      plots: {
        plotOverrides: state.plotOverrides.nmr || {},
      },
      model: {
        nComp: readInt(nCompInput?.value),
        nSpecies: readInt(nSpeciesInput?.value),
        stoichiometry: gridData,
        nonAbsorbingSpecies: nonAbsSpecies
      },
      roles: {
        receptor: receptorInput?.value || "",
        guest: guestInput?.value || ""
      },
      columns: {
        conc: selectedCols,
        signals: selectedSignals,
        nmrSheet: nmrSheetInput?.value || "",
        concSheet: concSheetInput?.value || ""
      },
      optimization: {
        algorithm: algoSelect?.value,
        modelSettings: modelSettingsSelect?.value,
        optimizer: optimizerSelect?.value,
        initialK: kValues,
        bounds: kBounds
      }
    };
  }

  // --- Helper: Apply Config ---
  function applyConfigToGui(cfg) {
    // 1. Set Module
    if (cfg.type === 'NMR') {
      state.activeModule = 'nmr';
      document.querySelector('[data-module-tab="nmr"]')?.click();
    } else {
      state.activeModule = 'spectroscopy';
      document.querySelector('[data-module-tab="spectroscopy"]')?.click();
    }

    // 2. Set Model Inputs & Generate Grid
    if (nCompInput) nCompInput.value = cfg.model.nComp;
    if (nSpeciesInput) nSpeciesInput.value = cfg.model.nSpecies;

    // Trigger grid generation
    generateModelGrid(cfg.model.nComp, cfg.model.nSpecies);
    generateOptGrid(cfg.model.nSpecies);

    // 3. Fill Model Grid
    if (modelGridContainer && cfg.model.stoichiometry) {
      const rows = modelGridContainer.querySelectorAll("tbody tr");
      rows.forEach((row, i) => {
        if (i < cfg.model.stoichiometry.length) {
          const inputs = row.querySelectorAll(".grid-input");
          const rowData = cfg.model.stoichiometry[i];
          inputs.forEach((inp, j) => {
            if (j < rowData.length) inp.value = rowData[j];
          });
        }
        // Mark non-absorbing
        if (cfg.model.nonAbsorbingSpecies && cfg.model.nonAbsorbingSpecies.includes(i)) {
          row.classList.add("selected");
        }
      });
    }

    // 4. Fill Optimization Grid
    if (optGridContainer && cfg.optimization.initialK) {
      const rows = optGridContainer.querySelectorAll("tbody tr");
      rows.forEach((row, i) => {
        if (i < cfg.optimization.initialK.length) {
          const inputs = row.querySelectorAll(".grid-input");
          if (inputs.length >= 3) {
            inputs[0].value = cfg.optimization.initialK[i];
            if (cfg.optimization.bounds && cfg.optimization.bounds[i]) {
              inputs[1].value = cfg.optimization.bounds[i][0] ?? "";
              inputs[2].value = cfg.optimization.bounds[i][1] ?? "";
            }
          }
        }
      });
    }

    // 5. Set Dropdowns (Algorithm, etc)
    if (algoSelect) algoSelect.value = cfg.optimization.algorithm;
    if (modelSettingsSelect) modelSettingsSelect.value = cfg.optimization.modelSettings;
    if (optimizerSelect) optimizerSelect.value = cfg.optimization.optimizer;

    // 6. EFA (Spectroscopy only)
    if (cfg.type === 'Spectroscopy') {
      if (efaCheckbox) efaCheckbox.checked = cfg.model.efaEnabled;
      if (efaEigenInput) efaEigenInput.value = cfg.model.efaEigenvalues;
    }

    // 6b. Plot overrides (optional)
    if (cfg?.plots?.plotOverrides) {
      if (cfg.type === 'NMR') {
        state.plotOverrides.nmr = cfg.plots.plotOverrides || {};
      } else {
        state.plotOverrides.spectroscopy = cfg.plots.plotOverrides || {};
      }
    }

    // 7. Attempt to restore column selections IF sheets match
    // This is tricky because we might not have the file loaded or sheets might differ.
    // We will try to check boxes if they exist.

    // Helper to check boxes
    const checkBoxes = (container, values) => {
      if (!container || !values) return;
      const boxes = container.querySelectorAll('input[type="checkbox"]');
      boxes.forEach(cb => {
        if (values.includes(cb.value)) {
          cb.checked = true;
        } else {
          cb.checked = false;
        }
      });
      // Trigger change to update dropdowns
      container.dispatchEvent(new Event('change'));
    };

    if (cfg.type === 'Spectroscopy') {
      checkBoxes(columnsContainer, cfg.columns.conc);
    } else {
      checkBoxes(columnsContainer, cfg.columns.conc);
      checkBoxes(nmrSignalsContainer, cfg.columns.signals);
    }

    // 8. Restore Receptor/Guest (after dropdowns update)
    // We need to wait for dropdowns to populate (which happens on change event above)
    // But since that's synchronous for existing DOM elements, we can try setting values now.
    setTimeout(() => {
      if (receptorInput) receptorInput.value = cfg.roles.receptor;
      if (guestInput) guestInput.value = cfg.roles.guest;
    }, 50);

    diagEl.textContent = `Configuration loaded (${cfg.type}).`;
    // If plots exist, re-render active plot to apply overrides
    try { renderMainCanvasPlot(); } catch (_) { }
  }

  // --- Export Config ---
  async function exportCurrentConfig() {
    let cfg;
    if (state.activeModule === 'nmr') {
      cfg = buildNmrConfigFromState();
    } else {
      cfg = buildSpecConfigFromState();
    }

    try {
      const savePath = await save({
        filters: [{ name: 'JSON', extensions: ['json'] }],
        defaultPath: 'hmfit_config.json'
      });
      if (!savePath) return;

      await writeTextFile(savePath, JSON.stringify(cfg, null, 2));
      appendLog(`Configuration exported to ${savePath}`);
    } catch (err) {
      console.error(err);
      appendLog(`Error exporting config: ${err.message}`);
    }
  }

  // --- Import Config ---
  async function importConfig() {
    try {
      const openPath = await open({
        multiple: false,
        filters: [{ name: 'JSON', extensions: ['json'] }]
      });
      if (!openPath) return;

      const content = await readTextFile(openPath);
      const cfg = JSON.parse(content);

      applyConfigToGui(cfg);
      appendLog(`Configuration imported from ${openPath}`);
    } catch (err) {
      console.error(err);
      appendLog(`Error importing config: ${err.message}`);
    }
  }

  // --- Wire up Import/Export Buttons ---
  // We need to find or create them. The plan said to add them to HTML, 
  // but we can also inject them here if they don't exist, or assume they are added to HTML.
  // Let's look for them by ID.
  const exportBtn = document.getElementById("export-config-btn");
  const importBtn = document.getElementById("import-config-btn");

  if (exportBtn) exportBtn.addEventListener("click", exportCurrentConfig);
  if (importBtn) importBtn.addEventListener("click", importConfig);

}

// Ejecutar una vez que el HTML ya está montado
// initApp(); // Already called at bottom



// Primera renderización
// Primera renderización
initApp();

// Connect WebSocket for progress updates
connectWebSocket();
