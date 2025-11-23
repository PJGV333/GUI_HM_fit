import "./style.css";

const state = {
  activeModule: "spectroscopy", // "spectroscopy" | "nmr"
  activeSubtab: "model",        // "model" | "optimization"
};

function log(text) {
  const pre = document.getElementById("log-output");
  if (!pre) return;
  pre.textContent = text;
}

async function pingBackend() {
  log("Consultando /health …");
  try {
    const resp = await fetch("http://127.0.0.1:8000/health");
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
    const resp = await fetch("http://127.0.0.1:8000/dummy_fit", {
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

function render() {
  const app = document.querySelector("#app");
  if (!app) return;

  app.innerHTML = `
    <div class="root-container">
      <header class="hmfit-header">
        <div>
          <h1 class="hmfit-title">HM Fit</h1>
          <p class="hmfit-subtitle">Hard Modeling · Spectroscopy &amp; NMR</p>
        </div>
        <button id="backend-health" class="btn ghost-btn">
          Probar backend
        </button>
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
                <input type="text" class="field-input" placeholder="e.g. Spectra" />
              </div>
              <div class="field">
                <label class="field-label">Concentration Sheet Name</label>
                <input type="text" class="field-input" placeholder="e.g. Conc" />
              </div>
            </div>

            <div class="field">
              <label class="field-label">Column names</label>
              <input type="text" class="field-input" placeholder="Comma-separated" />
            </div>

            <div class="field-grid">
              <div class="field">
                <label class="field-label">Receptor or Ligand</label>
                <input type="text" class="field-input" />
              </div>
              <div class="field">
                <label class="field-label">Guest, Metal or Titrant</label>
                <input type="text" class="field-input" />
              </div>
            </div>

            <div class="field efa-row">
              <label class="field-label">EFA Eigenvalues</label>
              <div class="efa-inline">
                <label class="checkbox-inline">
                  <input type="checkbox" />
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
              <h2 class="section-title">Define Model Dimensions</h2>

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
                <div class="table-placeholder">
                  ABCD – checkbox grid placeholder
                </div>
              </div>
            </div>

            <div class="subtab-panel" data-subtab-panel="optimization">
              <h2 class="section-title">Optimization (placeholder)</h2>
              <p class="hint-text">
                Aquí irán los parámetros de optimización (algoritmo, iteraciones,
                tolerancias, etc.) igual que en la GUI de wx.
              </p>
            </div>

            <div class="actions-row">
              <button id="reset-btn" class="btn secondary-btn">Reset Calculation</button>
              <button id="process-btn" class="btn primary-btn">Process Data</button>
            </div>
          </section>
        </div>

        <!-- Panel derecho: gráficos + log -->
        <div class="right-panel">
          <section class="panel plot-panel">
            <h2 class="section-title">Main spectra / titration plot</h2>
            <div class="plot-placeholder">
              Plot placeholder (aquí irán las gráficas principales).
            </div>
          </section>

          <section class="panel plot-panel">
            <h2 class="section-title">Residuals / component spectra / diagnostics</h2>
            <pre id="log-output" class="log-output">Esperando...</pre>
          </section>
        </div>
      </section>
    </div>
  `;

  // Tabs: módulo (Spectroscopy / NMR)
  document.querySelectorAll("[data-module-tab]").forEach((btn) => {
    const mod = btn.dataset.moduleTab;
    if (mod === state.activeModule) {
      btn.classList.add("tab-active");
    } else {
      btn.classList.remove("tab-active");
    }
    btn.addEventListener("click", () => {
      state.activeModule = mod;
      render();
    });
  });

  // Subtabs: Model / Optimization
  document.querySelectorAll("[data-subtab]").forEach((btn) => {
    const sub = btn.dataset.subtab;
    if (sub === state.activeSubtab) {
      btn.classList.add("tab-active");
    } else {
      btn.classList.remove("tab-active");
    }
    btn.addEventListener("click", () => {
      state.activeSubtab = sub;
      render();
    });
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

  // Botones
  const backendBtn = document.getElementById("backend-health");
  backendBtn?.addEventListener("click", pingBackend);

  const processBtn = document.getElementById("process-btn");
  processBtn?.addEventListener("click", dummyFit);

  const resetBtn = document.getElementById("reset-btn");
  resetBtn?.addEventListener("click", resetCalculation);
}

// === Helpers para localizar elementos existentes sin cambiar el HTML ===

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
  const diagEl = findDiagnosticsElement();

  if (!processBtn || !resetBtn || !diagEl) {
    console.warn(
      "[HM Fit] No se pudieron localizar Process Data / Reset / panel de diagnóstico. Revisa los textos de los botones."
    );
    return;
  }

  // Campos de texto identificables por placeholder
  const spectraSheetInput = document.querySelector(
    'input[placeholder="e.g. Spectra"]',
  );
  const concSheetInput = document.querySelector(
    'input[placeholder="e.g. Conc"]',
  );
  const columnNamesInput = document.querySelector(
    'input[placeholder="Comma-separated"]',
  );

  // Otros campos: usamos el orden en el formulario
  const textInputs = Array.from(
    document.querySelectorAll('input[type="text"]'),
  );
  // asumiendo orden: Spectra, Conc, Column names, Receptor, Guest
  const receptorInput = textInputs[3] || null;
  const guestInput = textInputs[4] || null;

  const numericInputs = Array.from(
    document.querySelectorAll('input[type="number"]'),
  );
  // asumiendo orden: EFA eigenvalues, #components, #species
  const efaEigenInput = numericInputs[0] || null;
  const nCompInput = numericInputs[1] || null;
  const nSpeciesInput = numericInputs[2] || null;

  // Único checkbox de EFA
  const efaCheckbox = document.querySelector('input[type="checkbox"]');

  // Área de non-absorbent species (placeholder de momento)
  const nonAbsArea = document.querySelector("textarea, .nonabs-placeholder");

  // --- Handler: Reset ---
  resetBtn.addEventListener("click", () => {
    if (spectraSheetInput) spectraSheetInput.value = "";
    if (concSheetInput) concSheetInput.value = "";
    if (columnNamesInput) columnNamesInput.value = "";
    if (receptorInput) receptorInput.value = "";
    if (guestInput) guestInput.value = "";
    if (efaEigenInput) efaEigenInput.value = "0";
    if (nCompInput) nCompInput.value = "0";
    if (nSpeciesInput) nSpeciesInput.value = "0";
    if (efaCheckbox) efaCheckbox.checked = false;
    if (nonAbsArea && "value" in nonAbsArea) nonAbsArea.value = "";

    diagEl.textContent = "Esperando...";
  });

  // --- Handler: Process Data ---
  processBtn.addEventListener("click", async () => {
    diagEl.textContent = "Procesando datos de Spectroscopy...";

    const payload = {
      spectra_sheet: spectraSheetInput?.value || "",
      conc_sheet: concSheetInput?.value || "",
      column_names: readList(columnNamesInput?.value || ""),
      receptor_label: receptorInput?.value || "",
      guest_label: guestInput?.value || "",
      efa_enabled: !!efaCheckbox?.checked,
      efa_eigenvalues: readInt(efaEigenInput?.value),
      n_components: readInt(nCompInput?.value),
      n_species: readInt(nSpeciesInput?.value),
      non_abs_species: nonAbsArea && "value" in nonAbsArea
        ? readList(nonAbsArea.value)
        : [],
    };

    try {
      const resp = await fetch("http://127.0.0.1:8000/spectroscopy/preview", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!resp.ok) {
        const text = await resp.text();
        diagEl.textContent = `Error HTTP ${resp.status}: ${text}`;
        return;
      }

      const data = await resp.json();
      diagEl.textContent = JSON.stringify(data, null, 2);
    } catch (err) {
      diagEl.textContent = `Error de red: ${err}`;
    }
  });
}

// Ejecutar una vez que el HTML ya está montado
wireSpectroscopyForm();


// Primera renderización
render();
