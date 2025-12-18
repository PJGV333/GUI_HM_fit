import wx
import wx.grid as gridlib
from wx.lib.scrolledpanel import ScrolledPanel

from .Methods import BaseTechniquePanel
from .plots_tab import PlotsTabPanel


class NMR_controlsPanel(BaseTechniquePanel):
    def __init__(self, parent, app_ref):
        super().__init__(parent, app_ref=app_ref)
        self.app_ref = app_ref

        self.panel = self
        self.left_sizer = wx.BoxSizer(wx.VERTICAL)
        self.right_sizer = wx.BoxSizer(wx.VERTICAL)

        # Crear controles (botones, etiquetas, etc.) y añadirlos a left_sizer o right_sizer
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

        # desplazamientos químicos
        self.sheet_chemshift_panel, self.choice_chemshifts = self.create_sheet_dropdown_section("Sheet Name of Chemical Shift:", self)
        self.left_sizer.Add(self.sheet_chemshift_panel, 0, wx.EXPAND | wx.ALL, 5)

        # Sizer horizontal para los menús desplegables
        dropdowns_sizer_chemical_shift = wx.BoxSizer(wx.HORIZONTAL)
        self.left_sizer.Add(dropdowns_sizer_chemical_shift, 0, wx.EXPAND | wx.ALL, 5)

        # Sizer horizontal para los menús desplegables
        dropdowns_sizer_chemical_shift = wx.BoxSizer(wx.HORIZONTAL)
        self.left_sizer.Add(dropdowns_sizer_chemical_shift, 0, wx.EXPAND | wx.ALL, 5)

        # ScrolledWindow para los checkboxes de desplazamientos químicos (inicialmente vacío)
        #self.chemical_shifts_panel = wx.ScrolledWindow(self.panel, style=wx.HSCROLL)
        #self.chemical_shifts_panel.SetScrollRate(10, 0)  # El primer número es la velocidad de desplazamiento horizontal, el segundo es para vertical (seteado a 0, ya que no queremos scroll vertical).
        self.chemical_shifts_panel = ScrolledPanel(self.panel, style=wx.HSCROLL | wx.TAB_TRAVERSAL)
        self.chemical_shifts_panel.SetupScrolling(scroll_x=True, scroll_y=False, rate_x=10, rate_y=0)
        
        
        # Sizer para los checkboxes dentro de ScrolledWindow
        self.chemical_shifts_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # Etiqueta para los desplazamientos químicos
        self.lbl_chemical_shifts = wx.StaticText(self.chemical_shifts_panel, label="Chemical Shifts: ")
        self.chemical_shifts_sizer.Add(self.lbl_chemical_shifts, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)

        # Configurar el sizer en el ScrolledWindow y añadir el panel al sizer principal
        self.chemical_shifts_panel.SetSizer(self.chemical_shifts_sizer)
        self.left_sizer.Add(self.chemical_shifts_panel, 0, wx.EXPAND | wx.ALL, 5)

        # Ajustar el tamaño del ScrolledWindow para asegurarse de que la barra de desplazamiento aparezca cuando sea necesario
        self.chemical_shifts_panel.SetMinSize((-1, 50))  # Reemplaza 'altura_deseada' con la altura que quieras para el panel
        #self.chemical_shifts_panel.SetMaxSize((400, -1))  # Reemplaza 'ancho_deseado' con el ancho máximo que quieras para el panel

        # Llama a Layout para actualizar la interfaz gráfica
        self.panel.Layout()
        self.Refresh()             # Refresca el frame para mostrar los cambios
        self.Update()

        # Panel para los menús desplegables de desplazamientos químicos
        self.choice_chemical_shifts_panel = wx.Panel(self.panel)

        #self.choice_chemshifts.Bind(wx.EVT_CHOICE, self.on_chemical_shift_sheet_selected)

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
            

        # Creación del wx.Notebook
        notebook = wx.Notebook(self.panel)

        # Creación de los paneles para las pestañas
        #tab_modelo = wx.Panel(notebook)
        tab_modelo = ScrolledPanel(notebook, style=wx.VSCROLL | wx.TAB_TRAVERSAL)
        tab_modelo.SetupScrolling(scroll_x=False, scroll_y=True, rate_x=0, rate_y=10)
        tab_optimizacion = wx.Panel(notebook)
        tab_plots = PlotsTabPanel(notebook, technique_panel=self, module_key="nmr")

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
        self.model_panel = ScrolledPanel(tab_modelo, style=wx.TAB_TRAVERSAL)
        self.model_panel.SetupScrolling(scroll_x=False, scroll_y=True)
        self.model_panel.SetScrollRate(0, 10)
        self.model_panel.SetMinSize((-1, 240))
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
        self.SetSizer(self.left_sizer)
        #self.Fit()
        self.Show()
        self.Layout()

###################################################################################

    def _refresh_page_scroller(self):
        page = self.GetParent()
        while page is not None and not hasattr(page, "FitInside"):
            page = page.GetParent()
        if page is not None:
            page.Layout()
            page.FitInside()
            page.SendSizeEvent()

    def _wrap_define_model_dimensions(self, evt):
        try:
            self.on_define_model_dimensions_checked(evt)
        finally:
            self.Layout()
            wx.CallAfter(self._refresh_scroller, self.model_panel, False, True)
            target_outer = getattr(self, "scrolled_window", self.GetParent())
            wx.CallAfter(self._refresh_scroller, target_outer, True, True)


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

        nmr_sheet = self.choice_chemshifts.GetStringSelection()
        conc_sheet = self.choice_sheet_conc.GetStringSelection()
        if not nmr_sheet or not conc_sheet:
            wx.MessageBox(
                "Por favor, seleccione las hojas de Excel correctamente.",
                "Error en selección de hojas",
                wx.OK | wx.ICON_ERROR,
            )
            return

        signal_names = [checkbox.GetLabel() for checkbox in self.vars_chemshift.values() if checkbox.IsChecked()]
        if not signal_names:
            wx.MessageBox(
                "Por favor, selecciona al menos una casilla de desplazamientos químicos para continuar.",
                "Advertencia",
                wx.OK | wx.ICON_WARNING,
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
                s = str(self.grid.GetCellValue(row, 4) or "").strip().lower()
                fixed = s in {"1", "true", "yes", "y", "t"}
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
                try:
                    self.grid.SetCellValue(row, 2, str(k_val))
                    self.grid.SetCellValue(row, 3, str(k_val))
                except Exception:
                    pass
            bounds.append((min_v, max_v))

        config = {
            "file_path": file_path,
            "nmr_sheet": nmr_sheet,
            "conc_sheet": conc_sheet,
            "column_names": column_names,
            "signal_names": signal_names,
            "receptor_label": receptor_label,
            "guest_label": guest_label,
            "modelo": modelo,
            "non_abs_species": non_abs_species,
            "algorithm": algorithm,
            "model_settings": model_settings,
            "optimizer": optimizer,
            "initial_k": initial_k,
            "bounds": bounds,
            "fixed_mask": fixed_mask,
            "k_fixed": fixed_mask,
        }
        self.last_config = dict(config)

        self.last_result = None
        self.figures = []
        self.current_figure_index = -1
        set_controls_enabled(False)
        print("Starting NMR processing...")

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
                from hmfit_core import run_nmr

                result = run_nmr(config, progress_cb=progress_cb)
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
