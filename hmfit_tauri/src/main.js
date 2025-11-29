import "./style.css";
import { BACKEND_BASE_URL, WS_BASE_URL, describeBackendTarget } from "./backend/config";

const state = {
  activeModule: "spectroscopy", // "spectroscopy" | "nmr"
  activeSubtab: "model",        // "model" | "optimization"
  uploadedFile: null,            // Currently selected file
  latestResultsText: "",         // Cache último reporte para guardar
  latestResultsPayload: null,    // Última respuesta de backend para exportar XLSX
};

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

function log(text) {
  const pre = document.getElementById("log-output");
  if (!pre) return;
  pre.textContent = text;
}

function appendLog(text) {
  const pre = document.getElementById("log-output");
  if (!pre) return;
  pre.textContent += text + "\n";
  // Auto-scroll to bottom
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
  if (efaRow) {
    if (state.activeModule === "nmr") {
      efaRow.classList.add("hidden");
    } else {
      efaRow.classList.remove("hidden");
    }
  }
}

function initApp() {
  const app = document.querySelector("#app");
  if (!app) return;

  console.info("[HM Fit] Backend base URL:", BACKEND_BASE_URL);

  app.innerHTML = `
    <div class="root-container">
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

      <section class="layout">
        <!-- Panel izquierdo -->
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

            <div class="actions-row">
              <div class="actions-left">
                <button id="backend-health" class="btn ghost-btn">Probar backend</button>
              </div>
              <div class="actions-right">
                <button id="reset-btn" class="btn secondary-btn">Reset Calculation</button>
                <button id="process-btn" class="btn primary-btn">Process Data</button>
                <button id="save-results-btn" class="btn ghost-btn">Save results</button>
              </div>
            </div>
          </section>
        </div>

        <!-- Panel derecho: gráficos + log -->
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
              <div class="plot-placeholder primary-plot scrollable-plot">
                Plot placeholder (aquí irán las gráficas principales).
              </div>
              <div class="plot-side-nav right">
                <button id="plot-next-side" class="plot-side-btn" title="Siguiente">›</button>
              </div>
              <div class="plot-placeholder secondary-plot" style="margin-top: 1rem;">
                 <!-- Placeholder for secondary plots -->
              </div>
            </div>
            <div class="split-resizer" title="Arrastra para redimensionar"></div>
            <div class="split-bottom">
              <h2 class="section-title">Residuals / component spectra / diagnostics</h2>
              <pre id="log-output" class="log-output">Esperando...</pre>
            </div>
          </section>
        </div>
      </section>
    </div>
  `;

  // Tabs: módulo (Spectroscopy / NMR)
  document.querySelectorAll("[data-module-tab]").forEach((btn) => {
    const mod = btn.dataset.moduleTab;
    btn.addEventListener("click", () => {
      state.activeModule = mod;
      updateUI();
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
  });
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
  const columnsContainer = document.getElementById("columns-container");

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

  // Botón para definir dimensiones
  const defineModelBtn = document.getElementById("define-model-btn");

  // Área de grid
  const modelGridContainer = document.getElementById("model-grid-container");
  const optGridContainer = document.getElementById("optimization-grid-container");

  // Dropdowns de Optimización
  const algoSelect = document.getElementById("algorithm-select");
  const modelSettingsSelect = document.getElementById("model-settings-select");
  const optimizerSelect = document.getElementById("optimizer-select");

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

  // --- Handler: File Selection ---
  const fileInput = document.getElementById("excel-file");
  const fileStatus = document.querySelector(".file-status");

  fileInput?.addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) {
      fileStatus.textContent = "No file selected";
      state.uploadedFile = null;
      return;
    }
    fileStatus.textContent = file.name;
    state.uploadedFile = file;

    // Enviar al backend para obtener hojas
    try {
      diagEl.textContent = "Leyendo archivo Excel...";
      const data = await backendApi.listSheets(file, state.activeModule);
      const sheets = data?.sheets || [];

      // Poblar dropdowns
      [spectraSheetInput, concSheetInput].forEach(select => {
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
    const file = fileInput.files[0];

    if (!sheetName || !file) {
      columnsContainer.innerHTML = "Select a concentration sheet to load columns...";
      return;
    }

    try {
      diagEl.textContent = `Leyendo columnas de ${sheetName}...`;
      const data = await backendApi.listColumns(file, sheetName, state.activeModule);
      const columns = data?.columns || [];

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

  // --- Handler: Define Model Dimensions (Grid Generation) ---
  defineModelBtn?.addEventListener("click", () => {
    const nComp = readInt(nCompInput?.value);
    const nSpecies = readInt(nSpeciesInput?.value);

    if (nComp <= 0 || nSpecies <= 0) {
      diagEl.textContent = "Please enter valid Number of Components and Species (>0).";
      return;
    }

    // Generar tabla
    modelGridContainer.innerHTML = "";
    const table = document.createElement("table");
    table.className = "model-grid-table";

    // Header
    const thead = document.createElement("thead");
    const headerRow = document.createElement("tr");
    // Empty corner cell
    headerRow.appendChild(document.createElement("th"));
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
      tr.dataset.species = `sp${s}`; // Identificador de especie

      // Label cell
      const tdLabel = document.createElement("td");
      tdLabel.textContent = `sp${s}`;
      tdLabel.className = "species-label";
      tr.appendChild(tdLabel);

      // Input cells
      for (let c = 1; c <= nComp; c++) {
        const td = document.createElement("td");
        const input = document.createElement("input");
        input.type = "number";
        input.className = "grid-input";

        // Logic: First nComp rows are identity matrix
        if (s <= nComp) {
          input.value = (c === s) ? "1.0" : "0.0";
        } else {
          // Remaining rows are 0
          input.value = "0";
        }

        td.appendChild(input);
        tr.appendChild(td);
      }

      // Click listener for row selection
      tr.addEventListener("click", (e) => {
        // Evitar que el click en el input dispare la selección de fila si se desea
        // Pero el usuario pidió "cuando se selecciona una fila".
        // Si hacemos click en el input, queremos editar, no necesariamente seleccionar la fila entera para borrarla.
        // Vamos a permitir seleccionar haciendo click en la etiqueta o en el padding, pero no interferir con el input.
        if (e.target.tagName.toLowerCase() === "input") return;

        tr.classList.toggle("selected");
      });

      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    modelGridContainer.appendChild(table);

    // --- Generar Grid de Optimización ---
    if (optGridContainer) {
      optGridContainer.innerHTML = "";
      const nConstants = nSpecies; // K por cada especie adicional (complejo)

      if (nConstants > 0) {
        const optTable = document.createElement("table");
        optTable.className = "model-grid-table";

        // Header
        const optThead = document.createElement("thead");
        const optHeaderRow = document.createElement("tr");
        ["Parameter", "Value", "Min", "Max"].forEach(text => {
          const th = document.createElement("th");
          th.textContent = text;
          optHeaderRow.appendChild(th);
        });
        optThead.appendChild(optHeaderRow);
        optTable.appendChild(optThead);

        // Body
        const optTbody = document.createElement("tbody");
        for (let k = 1; k <= nConstants; k++) {
          const tr = document.createElement("tr");

          // Parameter Label
          const tdParam = document.createElement("td");
          tdParam.textContent = `K${k}`;
          tdParam.className = "species-label"; // Reusing style
          tr.appendChild(tdParam);

          // Value Input
          const tdVal = document.createElement("td");
          const inputVal = document.createElement("input");
          inputVal.type = "number";
          inputVal.className = "grid-input";
          inputVal.placeholder = "Value";
          tdVal.appendChild(inputVal);
          tr.appendChild(tdVal);

          // Min Input
          const tdMin = document.createElement("td");
          const inputMin = document.createElement("input");
          inputMin.type = "number";
          inputMin.className = "grid-input";
          inputMin.placeholder = "Min";
          tdMin.appendChild(inputMin);
          tr.appendChild(tdMin);

          // Max Input
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

    diagEl.textContent = `Grid generado: ${nComp} Componentes x ${nSpecies} Especies.`;
  });

  // --- Helper: Update Receptor/Guest Dropdowns ---
  function updateDropdowns() {
    if (!receptorInput || !guestInput) return;

    // Guardar selección actual para intentar mantenerla
    const currentReceptor = receptorInput.value;
    const currentGuest = guestInput.value;

    // Obtener columnas seleccionadas
    const selectedCols = Array.from(columnsContainer.querySelectorAll('input[type="checkbox"]:checked'))
      .map(cb => cb.value);

    // Limpiar y repoblar
    [receptorInput, guestInput].forEach(select => {
      select.innerHTML = "";
      // Opción vacía por defecto
      const defaultOpt = document.createElement("option");
      defaultOpt.value = "";
      defaultOpt.text = ""; // O "Select..."
      select.add(defaultOpt);

      selectedCols.forEach(col => {
        const opt = document.createElement("option");
        opt.value = col;
        opt.text = col;
        select.add(opt);
      });
    });

    // Restaurar selección si aún existe
    if (selectedCols.includes(currentReceptor)) {
      receptorInput.value = currentReceptor;
    }
    if (selectedCols.includes(currentGuest)) {
      guestInput.value = currentGuest;
    }
  }

  // Escuchar cambios en los checkboxes para actualizar dropdowns
  columnsContainer.addEventListener("change", (e) => {
    if (e.target.matches('input[type="checkbox"]')) {
      updateDropdowns();
    }
  });

  // También actualizar cuando se cargan las columnas (dentro del fetch de list_columns)
  // ...pero como list_columns es asíncrono y ya tiene su lógica, lo mejor es llamar updateDropdowns()
  // al final de la carga exitosa de columnas.
  // Modificamos el listener de concSheetInput para llamar a updateDropdowns al final?
  // Mejor usamos un MutationObserver o simplemente lo llamamos explícitamente si pudiéramos.
  // Dado que no puedo editar fácilmente el bloque anterior sin hacerlo gigante,
  // voy a usar un MutationObserver en columnsContainer para detectar cuando se añaden los checkboxes.

  const observer = new MutationObserver(() => {
    updateDropdowns();
  });
  observer.observe(columnsContainer, { childList: true, subtree: true });

  // --- Handler: Reset ---
  resetBtn.addEventListener("click", () => {
    if (spectraSheetInput) spectraSheetInput.innerHTML = '<option value="">Select a file first...</option>';
    if (concSheetInput) concSheetInput.innerHTML = '<option value="">Select a file first...</option>';
    if (fileInput) fileInput.value = "";
    if (fileStatus) fileStatus.textContent = "No file selected";

    if (columnsContainer) columnsContainer.innerHTML = "Select a concentration sheet to load columns...";
    if (receptorInput) {
      receptorInput.innerHTML = '<option value="">Select columns first...</option>';
      receptorInput.value = "";
    }
    if (guestInput) {
      guestInput.innerHTML = '<option value="">Select columns first...</option>';
      guestInput.value = "";
    }
    if (efaEigenInput) efaEigenInput.value = "0";
    if (nCompInput) nCompInput.value = "0";
    if (nSpeciesInput) nSpeciesInput.value = "0";
    if (efaCheckbox) efaCheckbox.checked = false;
    if (efaCheckbox) efaCheckbox.checked = false;
    if (modelGridContainer) modelGridContainer.innerHTML = "";
    if (optGridContainer) optGridContainer.innerHTML = "";

    diagEl.textContent = "Esperando...";
    state.latestResultsText = "";
    state.latestResultsPayload = null;
  });

  // --- Handler: Save Results ---
  if (saveBtn) {
    saveBtn.addEventListener("click", async () => {
      const payload = state.latestResultsPayload;
      const resultsText = (state.latestResultsText || diagEl.textContent || "").trim();
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
        const hasTauri = !!window.__TAURI__;
        if (hasTauri && window.__TAURI__.dialog?.save && window.__TAURI__.fs?.writeBinaryFile) {
          const savePath = await window.__TAURI__.dialog.save({
            defaultPath: filename,
            filters: [{ name: "Excel", extensions: ["xlsx"] }],
          });
          if (!savePath) return; // cancelado
          const data = await fetchXlsx();
          await window.__TAURI__.fs.writeBinaryFile({ path: savePath, contents: data });
          appendLog(`Resultados guardados en ${savePath}`);
        } else {
          // Fallback: descarga directa del browser
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
          appendLog(`Resultados descargados como ${filename}`);
        }
      } catch (err) {
        appendLog(`No se pudieron guardar los resultados: ${err.message || err}`);
      }
    });
  }

  // --- Handler: Process Data ---
  processBtn.addEventListener("click", async () => {
    if (isProcessing) {
      appendLog("Procesamiento en curso, espera a que termine...");
      return;
    }

    if (!state.uploadedFile) {
      diagEl.textContent = "Error: No file selected. Please select a file first.";
      return;
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
    diagEl.textContent = "Procesando datos de Spectroscopy...\n";

    // Recolectar columnas seleccionadas
    const selectedCols = Array.from(columnsContainer.querySelectorAll('input[type="checkbox"]:checked'))
      .map(cb => cb.value);

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
    formData.append("file", state.uploadedFile);
    formData.append("spectra_sheet", spectraSheetInput?.value || "");
    formData.append("conc_sheet", concSheetInput?.value || "");
    formData.append("column_names", JSON.stringify(selectedCols));
    formData.append("receptor_label", receptorInput?.value || "");
    formData.append("guest_label", guestInput?.value || "");
    formData.append("signals_sheet", ""); // placeholder for NMR flow
    formData.append("efa_enabled", efaCheckbox?.checked ? "true" : "false");
    formData.append("efa_eigenvalues", readInt(efaEigenInput?.value).toString());
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
        data = await backendApi.processNmr(formData);
        displayNmrResults(data);
      } else {
        data = await backendApi.processSpectroscopy(formData);
        displayResults(data);
        displayGraphs(data.graphs || {});
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

      state.latestResultsText = "";
      state.latestResultsPayload = null;
      diagEl.textContent = message;
      console.error(`[HM Fit] Process Data request failed (${state.activeModule}):`, err);
    } finally {
      setProcessing(false);
    }
  });

  // Helper function to display results
  function displayResults(data) {
    if (!data.success) {
      const detail = data.detail || data.error || "Procesamiento falló.";
      state.latestResultsText = "";
      state.latestResultsPayload = null;
      diagEl.textContent = detail;
      return;
    }

    if (data.results_text) {
      diagEl.textContent = data.results_text;
      state.latestResultsText = data.results_text;
      state.latestResultsPayload = data;
      return;
    }

    // Display constants and statistics
    const constants = data.constants || [];
    const stats = data.statistics || {};

    const fmt = (n, opts = {}) => {
      if (n === null || n === undefined || Number.isNaN(n)) return "—";
      if (opts.fixed) return Number(n).toFixed(opts.fixed);
      return Number(n).toExponential(opts.exp ?? 3);
    };

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

    diagEl.textContent = lines.join("\n");
    state.latestResultsText = lines.join("\n");
    state.latestResultsPayload = data;
  }

  function displayNmrResults(data) {
    if (!data?.success) {
      const detail = data?.detail || data?.error || "Procesamiento NMR falló.";
      diagEl.textContent = detail;
      return;
    }

    const lines = [];
    lines.push("=== NMR WORKBOOK SUMMARY ===", "");
    lines.push(`Traces (columns): ${data.n_traces}`);
    lines.push(`Points (rows): ${data.n_points}`);
    lines.push(`Concentration columns: ${data.n_concentrations}`);
    lines.push(`Selected columns: ${(data.columns || []).join(', ')}`);
    if (data.signals_sheet) {
      lines.push(`Signals sheet: ${data.signals_sheet}`);
      lines.push(`Signals detected: ${data.n_signals ?? 0}`);
    }
    if (data.receptor_label || data.guest_label) {
      lines.push(`Receptor: ${data.receptor_label || '—'}`);
      lines.push(`Guest: ${data.guest_label || '—'}`);
    }

    diagEl.textContent = lines.join("\n");
  }

  function displayGraphs(graphs) {
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
    let updateSlide = () => {};
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
    // Secondary plots are now merged into mainPlots, so we don't render them separately.
  }
}

// Ejecutar una vez que el HTML ya está montado
// Ejecutar una vez que el HTML ya está montado
// wireSpectroscopyForm(); // MOVED TO RENDER


// Primera renderización
// Primera renderización
initApp();

// Connect WebSocket for progress updates
connectWebSocket();
