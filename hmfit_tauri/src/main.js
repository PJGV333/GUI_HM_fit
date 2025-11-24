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
      return;
    }
    fileStatus.textContent = file.name;

    // Enviar al backend para obtener hojas
    const formData = new FormData();
    formData.append("file", file);

    try {
      diagEl.textContent = "Leyendo archivo Excel...";
      const resp = await fetch("http://127.0.0.1:8000/list_sheets", {
        method: "POST",
        body: formData,
      });

      if (!resp.ok) {
        throw new Error(`Error ${resp.status}: ${await resp.text()}`);
      }

      const data = await resp.json();
      const sheets = data.sheets || [];

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

    const formData = new FormData();
    formData.append("file", file);
    formData.append("sheet_name", sheetName);

    try {
      diagEl.textContent = `Leyendo columnas de ${sheetName}...`;
      const resp = await fetch("http://127.0.0.1:8000/list_columns", {
        method: "POST",
        body: formData,
      });

      if (!resp.ok) {
        throw new Error(`Error ${resp.status}: ${await resp.text()}`);
      }

      const data = await resp.json();
      const columns = data.columns || [];

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
  });

  // --- Handler: Process Data ---
  processBtn.addEventListener("click", async () => {
    diagEl.textContent = "Procesando datos de Spectroscopy...";

    // Recolectar columnas seleccionadas
    const selectedCols = Array.from(columnsContainer.querySelectorAll('input[type="checkbox"]:checked'))
      .map(cb => cb.value);

    const payload = {
      spectra_sheet: spectraSheetInput?.value || "",
      conc_sheet: concSheetInput?.value || "",
      column_names: selectedCols,
      receptor_label: receptorInput?.value || "",
      guest_label: guestInput?.value || "",
      efa_enabled: !!efaCheckbox?.checked,
      efa_eigenvalues: readInt(efaEigenInput?.value),
      n_components: readInt(nCompInput?.value),
      n_species: readInt(nSpeciesInput?.value),
      n_components: readInt(nCompInput?.value),
      n_species: readInt(nSpeciesInput?.value),
      non_abs_species: modelGridContainer
        ? Array.from(modelGridContainer.querySelectorAll("tr.selected"))
          .map(tr => tr.dataset.species)
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
// Ejecutar una vez que el HTML ya está montado
// wireSpectroscopyForm(); // MOVED TO RENDER


// Primera renderización
// Primera renderización
initApp();
