import wx
import wx.grid as gridlib
from wx.lib.scrolledpanel import ScrolledPanel

from .Methods import BaseTechniquePanel
from .plots_tab import PlotsTabPanel

CHANNEL_TOLERANCE = 0.5


def _grid_cell_to_bool(value: str) -> bool:
    s = str(value or "").strip().lower()
    return s in {"1", "true", "yes", "y", "t"}


def _parse_custom_channels(raw: str) -> dict:
    """
    Minimal parser for spectroscopy custom channels.

    Supported:
    - "300,310,320" -> list of floats
    - "250-450"     -> range (min,max)
    - "250-450:5"   -> range (step ignored for now)
    """
    text = str(raw or "").strip()
    if not text:
        raise ValueError("Custom channels is empty. Example: 250-450 or 300, 310, 320")

    if "," in text:
        vals = []
        for part in text.split(","):
            part = part.strip()
            if not part:
                continue
            vals.append(float(part))
        if not vals:
            raise ValueError("No channels parsed. Example: 300, 310, 320")
        return {"kind": "list", "values": vals, "custom": vals}

    if "-" in text:
        rng = text.split(":", 1)[0].strip()
        a_s, b_s = (x.strip() for x in rng.split("-", 1))
        a = float(a_s)
        b = float(b_s)
        lo = min(a, b)
        hi = max(a, b)
        return {"kind": "range", "min": lo, "max": hi, "custom": [lo, hi]}

    val = float(text)
    return {"kind": "list", "values": [val], "custom": [val]}


def _load_spectroscopy_axis_values(file_path: str, spectra_sheet: str) -> list[float]:
    import pandas as pd

    df = pd.read_excel(file_path, spectra_sheet, header=0, usecols=[0])
    axis = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    axis = axis.loc[axis.notna()].astype(float)
    return [float(x) for x in axis.to_numpy()]


def _resolve_custom_channels(parsed: dict, axis_values: list[float], tol: float = CHANNEL_TOLERANCE) -> list[float]:
    axis = [float(v) for v in axis_values if v is not None]
    if not axis:
        raise ValueError("Axis not available. Select a valid Spectra sheet first.")

    if parsed.get("kind") == "range":
        lo = float(parsed["min"])
        hi = float(parsed["max"])
        selected = [v for v in axis if lo <= v <= hi]
        if not selected:
            raise ValueError(f"Range '{lo}-{hi}' matched 0 axis values.")
        return selected

    targets = list(parsed.get("values") or [])
    if not targets:
        raise ValueError("No channels parsed. Example: 300, 310, 320")

    resolved_set: set[float] = set()
    for target in targets:
        best = None
        best_diff = float("inf")
        for v in axis:
            d = abs(v - float(target))
            if d < best_diff:
                best = v
                best_diff = d
        if best is None or best_diff > float(tol):
            raise ValueError(f"'{target}' is not within tolerance (tol={tol}) of any axis value.")
        resolved_set.add(best)

    return [v for v in axis if v in resolved_set]

# Clase para la técnica de Espectroscopia
class Spectroscopy_controlsPanel(BaseTechniquePanel):
    def __init__(self, parent, app_ref):
        super().__init__(parent, app_ref=app_ref)
        self.app_ref = app_ref

        self.panel = self
        #self.technique_panel = BaseTechniquePanel(self.panel)

        # Diseño usando Sizers
        self.main_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.left_sizer = wx.BoxSizer(wx.VERTICAL)
        self.right_sizer = wx.BoxSizer(wx.VERTICAL)
        # Añadir sizers al panel principal
        self.main_sizer.Add(self.left_sizer, 1, wx.EXPAND | wx.ALL, 5)
        self.main_sizer.Add(self.right_sizer, 2, wx.EXPAND | wx.ALL, 5)

        # Crear el botón para seleccionar el archivo
        self.btn_select_file = wx.Button(self.panel, label="Select Excel File")
        self.btn_select_file.Bind(wx.EVT_BUTTON, self.select_file)
        self.left_sizer.Add(self.btn_select_file, 0, wx.ALL | wx.EXPAND, 5)

        # Crear el ScrolledWindow
        #self.scrolled_window = wx.ScrolledWindow(self.panel, style=wx.HSCROLL)
        #self.scrolled_window.SetScrollRate(10, 0)  # El primer valor es la velocidad de scroll horizontal, el segundo es vertical y está seteado en 0 porque no queremos scroll vertical.
        self.scrolled_window = ScrolledPanel(self.panel, style=wx.HSCROLL | wx.TAB_TRAVERSAL)
        self.scrolled_window.SetupScrolling(scroll_x=True, scroll_y=False, rate_x=10, rate_y=0)
        
        
        # Crear un StaticText para mostrar la ruta del archivo dentro del ScrolledWindow
        self.lbl_file_path = wx.StaticText(self.scrolled_window, label="No file selected")

        # Crear un sizer para el ScrolledWindow y agregar el StaticText
        scrolled_sizer = wx.BoxSizer(wx.HORIZONTAL)
        scrolled_sizer.Add(self.lbl_file_path, 1, wx.EXPAND | wx.ALL, 5)

        # Asignar el sizer al ScrolledWindow y actualizar su tamaño
        self.scrolled_window.SetSizer(scrolled_sizer)
        self.scrolled_window.SetMinSize((-1, 45))  # Establecer un tamaño mínimo para el ScrolledWindow

        self.left_sizer.Add(self.scrolled_window, 0, wx.EXPAND | wx.ALL, 5)
        self.Layout()
        self.Refresh()             # Refresca el frame para mostrar los cambios
        self.Update()

        # Espectros
        self.sheet_spectra_panel, self.choice_sheet_spectra = self.create_sheet_dropdown_section("Spectra Sheet Name:", self)
        self.left_sizer.Add(self.sheet_spectra_panel, 0, wx.EXPAND | wx.ALL, 5)

        # Channels selector (Tauri-like): All / Custom
        channels_panel = wx.Panel(self.panel)
        channels_sizer = wx.BoxSizer(wx.HORIZONTAL)
        channels_label = wx.StaticText(channels_panel, label="Channels:")
        channels_sizer.Add(channels_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)

        self.choice_channels = wx.Choice(channels_panel, choices=["All", "Custom"])
        self.choice_channels.SetSelection(0)
        self.choice_channels.SetToolTip("All = use all channels; Custom = select specific channels")
        channels_sizer.Add(self.choice_channels, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)

        self.entry_channels_custom = wx.TextCtrl(channels_panel)
        self.entry_channels_custom.Enable(False)
        hint = "e.g. 250-450 or 300, 310, 320"
        if hasattr(self.entry_channels_custom, "SetHint"):
            self.entry_channels_custom.SetHint(hint)
        self.entry_channels_custom.SetToolTip(hint)
        channels_sizer.Add(self.entry_channels_custom, 1, wx.ALL | wx.EXPAND, 5)

        channels_panel.SetSizer(channels_sizer)
        self.left_sizer.Add(channels_panel, 0, wx.EXPAND | wx.ALL, 5)
        self.choice_channels.Bind(wx.EVT_CHOICE, self.on_channels_mode_changed)

        # Concentraciones
        self.sheet_conc_panel, self.choice_sheet_conc = self.create_sheet_dropdown_section("Concentration Sheet Name:", self)
        self.left_sizer.Add(self.sheet_conc_panel, 0, wx.EXPAND | wx.ALL, 5)

        dropdowns_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.left_sizer.Add(dropdowns_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # ScrolledWindow para los checkboxes de nombres de columnas (inicialmente vacío)
        #self.columns_names_panel = wx.ScrolledWindow(self.panel, style=wx.HSCROLL)
        #self.columns_names_panel.SetScrollRate(10, 0)  # Configurar la velocidad de desplazamiento horizontal
        self.columns_names_panel = ScrolledPanel(self.panel, style=wx.HSCROLL | wx.TAB_TRAVERSAL)
        self.columns_names_panel.SetupScrolling(scroll_x=True, scroll_y=False, rate_x=10, rate_y=0)
                
        # Sizer para los checkboxes dentro del ScrolledWindow
        self.columns_names_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # Etiqueta para los nombres de las columnas
        self.lbl_columns = wx.StaticText(self.columns_names_panel, label="Column names: ")
        self.columns_names_sizer.Add(self.lbl_columns, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)

        # Configurar el sizer en el ScrolledWindow y añadir el panel al sizer principal
        self.columns_names_panel.SetSizer(self.columns_names_sizer)
        self.left_sizer.Add(self.columns_names_panel, 0, wx.EXPAND | wx.ALL, 5)

        # Ajustar el tamaño del ScrolledWindow para asegurarse de que la barra de desplazamiento aparezca cuando sea necesario
        self.columns_names_panel.SetMinSize((-1, 40))  # Reemplaza 'altura_deseada' con la altura que quieras para el panel
        #self.columns_names_panel.SetMaxSize((400, -1))  # Reemplaza 'ancho_deseado' con el ancho máximo que quieras para el panel

        # Llama a Layout para actualizar la interfaz gráfica
        self.panel.Layout()
        self.Refresh()             # Refresca el frame para mostrar los cambios
        self.Update()

        self.choice_columns_panel = wx.Panel(self.panel)
        # Crear el sizer horizontal para los menús desplegables
        self.dropdown_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # Crear y añadir el menú desplegable para 'Receptor or Ligant'
        self.receptor_label = wx.StaticText(self.choice_columns_panel, label="Receptor or Ligant:")
        self.dropdown_sizer.Add(self.receptor_label, 0, wx.ALL|wx.CENTER, 5)
        self.receptor_choice = wx.Choice(self.choice_columns_panel)
        self.dropdown_sizer.Add(self.receptor_choice, 0, wx.ALL|wx.CENTER, 5)

        # Crear y añadir el menú desplegable para 'Guest, Metal or Titrant'
        self.guest_label = wx.StaticText(self.choice_columns_panel, label="Guest, Metal or Titrant:")
        self.dropdown_sizer.Add(self.guest_label, 0, wx.ALL|wx.CENTER, 5)
        self.guest_choice = wx.Choice(self.choice_columns_panel)
        self.dropdown_sizer.Add(self.guest_choice, 0, wx.ALL|wx.CENTER, 5)

        # Agregar el sizer horizontal al panel principal o sizer
        # Asumiendo que 'self.columns_names_panel' es el panel que contiene tus checkboxes
        self.choice_columns_panel.SetSizer(self.dropdown_sizer)
        self.left_sizer.Add(self.choice_columns_panel, 0, wx.EXPAND | wx.ALL, 5)

        # Vincular la función de manejo de selección duplicada a los eventos de selección de los menús desplegables
        self.receptor_choice.Bind(wx.EVT_CHOICE, self.on_dropdown_selection)
        self.guest_choice.Bind(wx.EVT_CHOICE, self.on_dropdown_selection)

            
        # Autovalores
        self.sheet_EV_panel, self.entry_EV = self.create_sheet_section("Eigenvalues:", "0", parent = None)

        # Crear el Checkbox para EFA y añadirlo al sizer de la sección "Eigenvalues"
        self.EFA_cb = wx.CheckBox(self.sheet_EV_panel, label='EFA')
        self._efa_tooltip_default = self.EFA_cb.GetToolTipText() if hasattr(self.EFA_cb, "GetToolTipText") else ""
        self.EFA_cb.SetValue(True)  # Marcar el checkbox por defecto
        self.sheet_EV_panel.GetSizer().Insert(0, self.EFA_cb, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        self.left_sizer.Add(self.sheet_EV_panel, 0, wx.ALL | wx.EXPAND, 5)

        # Creación del wx.Notebook
        notebook = wx.Notebook(self.panel)

        # Creación de los paneles para las pestañas
        #tab_modelo = wx.Panel(notebook)
        tab_modelo = ScrolledPanel(notebook, style=wx.VSCROLL | wx.TAB_TRAVERSAL)
        tab_modelo.SetupScrolling(scroll_x=False, scroll_y=True, rate_x=0, rate_y=10)
        tab_optimizacion = wx.Panel(notebook)
        tab_plots = PlotsTabPanel(notebook, technique_panel=self, module_key="spec")

        # Añadir los paneles al notebook
        notebook.AddPage(tab_modelo, "Model")
        notebook.AddPage(tab_optimizacion, "Optimization")
        notebook.AddPage(tab_plots, "Plots")
        self.plots_tab = tab_plots

        # Creación del CheckBox
        self.toggle_components = wx.Button(tab_modelo, label="Define Model Dimensions")
        #self.toggle_components.Bind(wx.EVT_BUTTON, self.on_define_model_dimensions_checked)
        self.toggle_components.Bind(wx.EVT_BUTTON, self._wrap_define_model_dimensions)

        # Creación de los TextCtrl para número de componentes y número de especies
        self.num_components_text, self.entry_nc = self.create_sheet_section("Number of Components:", "0", parent = tab_modelo)
        self.num_species_text, self.entry_nsp = self.create_sheet_section("Number of Species:", "0", parent = tab_modelo)
        
        self.sp_columns = wx.StaticText(tab_modelo, label="Select non-absorbent species: ")

        self.sp_select_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # Reemplazar el wx.ListCtrl por un wx.grid.Grid
        #self.model_panel = wx.Panel(tab_modelo)
        self.model_panel = ScrolledPanel(tab_modelo, style=wx.TAB_TRAVERSAL)
        self.model_panel.SetupScrolling(scroll_x=False, scroll_y=True)
        self.model_panel.SetScrollRate(0, 10)
        self.model_panel.SetMinSize((-1, 240))  # altura fija para que aparezca el scroll interno
        self.model_grid = wx.grid.Grid(self.model_panel)
        self.model_grid.CreateGrid(0, 4)  # Por ejemplo, crear una cuadrícula sin filas inicialmente, pero con 4 columnas
        self.model_grid.SetSelectionMode(wx.grid.Grid.SelectRows)  # Configurar para seleccionar filas completas

        self.model_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.model_sizer.Add(self.model_grid, 1, wx.ALL | wx.EXPAND, 5)
        self.model_panel.SetSizer(self.model_sizer)

        # Configurar sizer y añadir al panel de la pestaña 'Modelo'
        modelo_sizer = wx.BoxSizer(wx.VERTICAL)
        #modelo_sizer.Add(self.sheet_model_panel, 0, wx.EXPAND | wx.ALL, 5)
        # Añadirlos al sizer
        modelo_sizer.Add(self.toggle_components, 0, wx.ALL | wx.EXPAND, 5)
        modelo_sizer.Add(self.num_components_text, 0, wx.ALL | wx.EXPAND, 5)
        modelo_sizer.Add(self.num_species_text, 0, wx.ALL | wx.EXPAND, 5)
        modelo_sizer.Add(self.sp_columns, 0, wx.EXPAND | wx.ALL, 5)
        #modelo_sizer.Add(self.model_panel, 1, wx.EXPAND | wx.ALL, 5)
        modelo_sizer.Add(self.model_panel, 0, wx.EXPAND | wx.ALL, 5)
        tab_modelo.SetSizer(modelo_sizer)
        tab_modelo.SetupScrolling(scroll_x=False, scroll_y=True)
        tab_modelo.FitInside()


        # Crear los controles para la pestaña 'Optimización'
        algo_panel = wx.Panel(tab_optimizacion)
        algo_label = wx.StaticText(algo_panel, label='Algorithm for C')
        algo_sizer = wx.BoxSizer(wx.VERTICAL)
        algo_sizer.Add(algo_label, 0, wx.ALL, 5)
        algoritmo = ["Newton-Raphson", "Levenberg-Marquardt"]
        self.choice_algoritm = wx.Choice(algo_panel, choices=algoritmo)
        algo_sizer.Add(self.choice_algoritm, 0, wx.ALL, 5)
        self.choice_algoritm.SetSelection(0)
        algo_panel.SetSizer(algo_sizer)

        ajustes_panel = wx.Panel(tab_optimizacion)
        ajustes_label = wx.StaticText(ajustes_panel, label='Model settings')
        ajustes_sizer = wx.BoxSizer(wx.VERTICAL)
        ajustes_sizer.Add(ajustes_label, 0, wx.ALL, 5)
        ajustes_modelo_choices = ["Free", "Step by step", "Non-cooperative"]
        self.choice_model_settings = wx.Choice(ajustes_panel, choices=ajustes_modelo_choices)
        self.choice_model_settings.Bind(wx.EVT_CHOICE, lambda event: self.update_parameter_grid())
        ajustes_sizer.Add(self.choice_model_settings, 0, wx.ALL, 5)
        self.choice_model_settings.SetSelection(0)
        ajustes_panel.SetSizer(ajustes_sizer)

        optimizador_panel = wx.Panel(tab_optimizacion)
        optimizador_label = wx.StaticText(optimizador_panel, label='Optimizer')
        optimizador_sizer = wx.BoxSizer(wx.VERTICAL)
        optimizador_sizer.Add(optimizador_label, 0, wx.ALL, 5)
        optimizador_choices = ["powell", "nelder-mead", "trust-constr", "cg", "bfgs", "l-bfgs-b", "tnc", "cobyla", "slsqp", "differential_evolution"]
        self.choice_optimizer_settings = wx.Choice(optimizador_panel, choices=optimizador_choices)
        optimizador_sizer.Add(self.choice_optimizer_settings, 0, wx.ALL, 5)
        self.choice_optimizer_settings.SetSelection(0)
        optimizador_panel.SetSizer(optimizador_sizer)

        # Crear el Grid
        self.grid = gridlib.Grid(tab_optimizacion)
        self.grid.CreateGrid(0, 5)
        self.grid.SetRowLabelSize(0)
        self.grid.SetColLabelValue(0, "Parameter")
        self.grid.SetColLabelValue(1, "Value")
        self.grid.SetColLabelValue(2, "Min")
        self.grid.SetColLabelValue(3, "Max")
        self.grid.SetColLabelValue(4, "Fixed")
        self.grid.AutoSizeColumns()

        # Sizer principal para la pestaña 'Optimización'
        optimizacion_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # Sizer para los controles verticales (menús desplegables)
        controls_sizer = wx.BoxSizer(wx.VERTICAL)
        controls_sizer.Add(algo_panel, 0, wx.EXPAND | wx.ALL, 5)
        controls_sizer.Add(ajustes_panel, 0, wx.EXPAND | wx.ALL, 5)
        controls_sizer.Add(optimizador_panel, 0, wx.EXPAND | wx.ALL, 5)

        # Añadir el sizer de controles al sizer principal de 'Optimización'
        optimizacion_sizer.Add(controls_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # Añadir el grid al sizer principal de 'Optimización'
        optimizacion_sizer.Add(self.grid, 1, wx.EXPAND | wx.ALL, 5)

        # Configurar el sizer en el panel de la pestaña 'Optimización'
        tab_optimizacion.SetSizer(optimizacion_sizer)

        # Ajustar la distribución de los controles
        tab_optimizacion.Layout()
        # Añadir el notebook al left_sizer
        self.left_sizer.Add(notebook, 1, wx.EXPAND | wx.ALL, 5)
        # Configura el sizer en el panel y llama a Layout para organizar los controles
        self.SetSizer(self.left_sizer)
        #self.Fit()
        self.Show()
        self.Layout()
        self.Refresh()             # Refresca el frame para mostrar los cambios
        self.Update()


    ###############################################################################################
    def on_channels_mode_changed(self, evt):
        mode = (self.choice_channels.GetStringSelection() or "All").strip()
        is_custom = mode.lower() == "custom"

        if hasattr(self, "entry_channels_custom"):
            self.entry_channels_custom.Enable(bool(is_custom))
            if not is_custom:
                self.entry_channels_custom.SetValue("")

        efa_cb = getattr(self, "EFA_cb", None)
        if efa_cb is not None:
            if is_custom:
                if efa_cb.GetValue():
                    efa_cb.SetValue(False)
                efa_cb.Enable(False)
                efa_cb.SetToolTip("EFA requiere espectro completo (Channels=All).")
            else:
                efa_cb.Enable(True)
                if getattr(self, "_efa_tooltip_default", ""):
                    efa_cb.SetToolTip(self._efa_tooltip_default)
                else:
                    efa_cb.SetToolTip("")

        if evt is not None:
            evt.Skip()

    ###############################################################################################
    def _refresh_page_scroller(self):
        page = self.GetParent()
        while page is not None and not hasattr(page, "FitInside"):
            page = page.GetParent()
        if page is not None:
            page.Layout()
            page.FitInside()
            page.SendSizeEvent()
    ###############################################################################################    
    def _wrap_define_model_dimensions(self, evt):
        try:
            self.on_define_model_dimensions_checked(evt)
        finally:
            self.Layout()
            # refresca el scroller del bloque Model (usa el de la clase base)
            wx.CallAfter(BaseTechniquePanel._refresh_scroller, self, self.model_panel, False, True)
            # refresca también el scroller externo (el de la barra horizontal)
            target_outer = getattr(self, "scrolled_window", self.GetParent())
            wx.CallAfter(BaseTechniquePanel._refresh_scroller, self, target_outer, True, True)
            # y, si quieres, refresca toda la página como extra
            wx.CallAfter(self._refresh_page_scroller)


    def process_data(self, event):
        """Collect UI inputs, call the new backend runner, then render results."""
        import threading

        def set_controls_enabled(enabled: bool) -> None:
            for attr in ("btn_process_data", "btn_prev_figure", "btn_next_figure", "btn_save_results"):
                btn = getattr(self.app_ref, attr, None)
                if btn is not None:
                    btn.Enable(bool(enabled))

        file_path = getattr(self, "file_path", None)
        if not file_path:
            wx.MessageBox(
                "No se ha seleccionado un archivo .xlsx",
                "Seleccione un archivo .xlsx para trabajar.",
                wx.OK | wx.ICON_ERROR,
            )
            return

        spectra_sheet = self.choice_sheet_spectra.GetStringSelection()
        conc_sheet = self.choice_sheet_conc.GetStringSelection()
        if not spectra_sheet or not conc_sheet:
            wx.MessageBox(
                "Por favor, seleccione las hojas de Excel correctamente.",
                "Error en selección de hojas",
                wx.OK | wx.ICON_ERROR,
            )
            return

        column_names = [checkbox.GetLabel() for checkbox in self.vars_columnas.values() if checkbox.IsChecked()]
        if not column_names:
            wx.MessageBox(
                "Por favor, selecciona al menos una casilla para continuar.",
                "Advertencia",
                wx.OK | wx.ICON_WARNING,
            )
            return

        receptor_label = (self.receptor_choice.GetStringSelection() or "").strip()
        guest_label = (self.guest_choice.GetStringSelection() or "").strip()
        if not receptor_label or not guest_label:
            wx.MessageBox(
                "Seleccione columnas para Receptor y Guest (deben ser diferentes).",
                "Faltan selecciones",
                wx.OK | wx.ICON_WARNING,
            )
            return
        if receptor_label == guest_label:
            wx.MessageBox(
                "Receptor and Guest cannot be the same column. Please select different columns.",
                "Selection Error",
                wx.OK | wx.ICON_WARNING,
            )
            return

        try:
            efa_eigenvalues = int(str(self.entry_EV.GetValue() or "0").strip())
        except ValueError:
            efa_eigenvalues = 0

        efa_enabled = bool(getattr(self, "EFA_cb", None) and self.EFA_cb.GetValue())

        # Channels (All / Custom) → backend expects `channels_resolved` (axis values).
        channels_mode = "all"
        channels_raw = "All"
        channels_custom = []
        channels_resolved = []
        choice_channels = getattr(self, "choice_channels", None)
        if choice_channels is not None and (choice_channels.GetStringSelection() or "All").strip().lower() == "custom":
            channels_mode = "custom"
            custom_ctrl = getattr(self, "entry_channels_custom", None)
            channels_raw = (custom_ctrl.GetValue() if custom_ctrl is not None else "").strip()
            try:
                parsed = _parse_custom_channels(channels_raw)
                axis_values = _load_spectroscopy_axis_values(file_path, spectra_sheet)
                channels_resolved = _resolve_custom_channels(parsed, axis_values)
                channels_custom = list(parsed.get("custom") or [])
            except Exception as exc:
                wx.MessageBox(str(exc), "Channels error", wx.OK | wx.ICON_ERROR)
                return

            if not channels_resolved:
                wx.MessageBox("Custom channels matched 0 axis values.", "Channels error", wx.OK | wx.ICON_ERROR)
                return

            # EFA requires full spectrum; force off when using Custom.
            if efa_enabled:
                efa_enabled = False
                if getattr(self, "EFA_cb", None) is not None:
                    self.EFA_cb.SetValue(False)

        model_grid_data = self.extract_data_from_grid()
        if not model_grid_data:
            wx.MessageBox("Select a sheet model o create one.", "Advertencia", wx.OK | wx.ICON_WARNING)
            return

        modelo = []
        any_nonzero = False
        for row in model_grid_data:
            row_vals = []
            for v in row:
                if v is None or (isinstance(v, str) and not v.strip()):
                    fv = 0.0
                else:
                    try:
                        fv = float(v)
                    except (TypeError, ValueError):
                        fv = 0.0
                any_nonzero = any_nonzero or (fv != 0.0)
                row_vals.append(fv)
            modelo.append(row_vals)

        if not any_nonzero:
            wx.MessageBox("Select a sheet model o create one.", "Advertencia", wx.OK | wx.ICON_WARNING)
            return

        non_abs_species = list(getattr(self, "model_grid", None).GetSelectedRows() if hasattr(self, "model_grid") else [])

        algorithm = self.choice_algoritm.GetStringSelection() or "Newton-Raphson"
        model_settings = self.choice_model_settings.GetStringSelection() or "Free"
        optimizer = self.choice_optimizer_settings.GetStringSelection() or "powell"

        n_rows = self.grid.GetNumberRows() if hasattr(self, "grid") else 0
        if n_rows <= 0:
            wx.MessageBox(
                "Defina el número de constantes (en Model/Optimization) y agregue valores iniciales.",
                "Advertencia",
                wx.OK | wx.ICON_WARNING,
            )
            return

        initial_k = []
        bounds = []
        fixed_mask = []
        for row in range(n_rows):
            val_s = (self.grid.GetCellValue(row, 1) or "").strip()
            if not val_s:
                wx.MessageBox(
                    "Write an initial estimate for the constants.",
                    "Advertencia",
                    wx.OK | wx.ICON_WARNING,
                )
                return
            try:
                k_val = float(val_s)
                initial_k.append(k_val)
            except ValueError:
                wx.MessageBox(
                    f"Invalid constant value in row {row + 1}.",
                    "Error",
                    wx.OK | wx.ICON_ERROR,
                )
                return

            fixed = False
            if self.grid.GetNumberCols() > 4:
                fixed = _grid_cell_to_bool(self.grid.GetCellValue(row, 4))
            fixed_mask.append(bool(fixed))

            min_s = (self.grid.GetCellValue(row, 2) or "").strip()
            max_s = (self.grid.GetCellValue(row, 3) or "").strip()
            try:
                min_v = float(min_s) if min_s else None
            except ValueError:
                min_v = None
            try:
                max_v = float(max_s) if max_s else None
            except ValueError:
                max_v = None

            if fixed:
                min_v = k_val
                max_v = k_val
                # Keep bounds consistent for optimizers (esp. differential_evolution).
                try:
                    self.grid.SetCellValue(row, 2, str(k_val))
                    self.grid.SetCellValue(row, 3, str(k_val))
                except Exception:
                    pass
            bounds.append((min_v, max_v))

        config = {
            "file_path": file_path,
            "spectra_sheet": spectra_sheet,
            "conc_sheet": conc_sheet,
            "column_names": column_names,
            "receptor_label": receptor_label,
            "guest_label": guest_label,
            "efa_enabled": efa_enabled,
            "efa_eigenvalues": efa_eigenvalues,
            "modelo": modelo,
            "non_abs_species": non_abs_species,
            "algorithm": algorithm,
            "model_settings": model_settings,
            "optimizer": optimizer,
            "initial_k": initial_k,
            "bounds": bounds,
            "fixed_mask": fixed_mask,
            "channels_mode": channels_mode,
            "channels_custom": channels_custom,
            "channels_raw": channels_raw,
            "channels_resolved": channels_resolved,
        }
        self.last_config = dict(config)

        self.last_result = None
        self.figures = []
        self.current_figure_index = -1
        set_controls_enabled(False)
        print("Starting spectroscopy processing...")

        def progress_cb(msg: str) -> None:
            print(str(msg).rstrip())

        def on_success(result: dict) -> None:
            try:
                self.last_result = result
                if getattr(self, "plots_tab", None) is not None:
                    try:
                        self.plots_tab.set_result(result, config=config)
                    except Exception:
                        pass

                from hmfit_core.plots import figures_from_graphs

                self.figures = figures_from_graphs(result.get("graphs") or {})
                self.current_figure_index = 0 if self.figures else -1
                if self.figures:
                    self.update_canvas_figure(self.figures[self.current_figure_index])

                results_text = result.get("results_text") or ""
                if results_text:
                    print("\n" + str(results_text).rstrip() + "\n")
            finally:
                set_controls_enabled(True)

        def on_error(exc: Exception) -> None:
            try:
                print(f"ERROR: {exc}")
                wx.MessageBox(str(exc), "Error", wx.OK | wx.ICON_ERROR)
            finally:
                set_controls_enabled(True)

        def worker() -> None:
            try:
                from hmfit_core import run_spectroscopy

                result = run_spectroscopy(config, progress_cb=progress_cb)
                if isinstance(result, dict) and result.get("error"):
                    raise RuntimeError(str(result.get("error")))
                if isinstance(result, dict) and not result.get("success", True):
                    raise RuntimeError("Processing failed.")
            except Exception as exc:
                wx.CallAfter(on_error, exc)
                return
            wx.CallAfter(on_success, result)

        self._worker_thread = threading.Thread(target=worker, daemon=True)
        self._worker_thread.start()
        return
