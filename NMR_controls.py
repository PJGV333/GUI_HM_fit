import wx
from Methods import BaseTechniquePanel
import wx.grid as gridlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('WXAgg')
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy import optimize
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
import warnings
warnings.filterwarnings("ignore")
import timeit
from Methods import BaseTechniquePanel


# Clase para la técnica de NMR
class NMR_controlsPanel(BaseTechniquePanel):
    def __init__(self, parent, app_ref):
        super().__init__(parent, app_ref)
        self.app_ref = app_ref

        self.panel = self
        self.left_sizer = wx.BoxSizer(wx.VERTICAL)
        self.right_sizer = wx.BoxSizer(wx.VERTICAL)

        # Crear controles (botones, etiquetas, etc.) y añadirlos a left_sizer o right_sizer
        # Ejemplo:
        self.btn_select_file = wx.Button(self.panel, label="Select Excel File")
        self.btn_select_file.Bind(wx.EVT_BUTTON, self.select_file)
        self.left_sizer.Add(self.btn_select_file, 0, wx.ALL | wx.EXPAND, 5)
        
        # Crear un StaticText para mostrar la ruta del archivo
        self.lbl_file_path = wx.StaticText(self.panel, label="No file selected")
        self.left_sizer.Add(self.lbl_file_path, 0, wx.ALL | wx.EXPAND, 5)
                
        # desplazamientos químicos
        self.sheet_chemshift_panel, self.choice_chemshifts = self.create_sheet_dropdown_section("Sheet Name of Chemical Shift:", self)
        self.left_sizer.Add(self.sheet_chemshift_panel, 0, wx.EXPAND | wx.ALL, 5)

        # Sizer horizontal para los menús desplegables
        dropdowns_sizer_chemical_shift = wx.BoxSizer(wx.HORIZONTAL)
        self.left_sizer.Add(dropdowns_sizer_chemical_shift, 0, wx.EXPAND | wx.ALL, 5)

        # Panel para los checkboxes de desplazamientos químicos (inicialmente vacío)
        self.chemical_shifts_panel = wx.Panel(self.panel)
        self.chemical_shifts_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # Etiqueta para los desplazamientos químicos
        self.lbl_chemical_shifts = wx.StaticText(self.chemical_shifts_panel, label="Chemical Shifts: ")
        self.chemical_shifts_sizer.Add(self.lbl_chemical_shifts, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        self.chemical_shifts_panel.SetSizer(self.chemical_shifts_sizer)
        self.left_sizer.Add(self.chemical_shifts_panel, 0, wx.EXPAND | wx.ALL, 5)

        # Panel para los menús desplegables de desplazamientos químicos
        self.choice_chemical_shifts_panel = wx.Panel(self.panel)

        self.choice_chemshifts.Bind(wx.EVT_CHOICE, self.on_chemical_shift_sheet_selected)

        # Concentraciones
        self.sheet_conc_panel, self.choice_sheet_conc = self.create_sheet_dropdown_section("Concentration Sheet Name:", self)
        self.left_sizer.Add(self.sheet_conc_panel, 0, wx.EXPAND | wx.ALL, 5)

        dropdowns_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.left_sizer.Add(dropdowns_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # Sección para Checkboxes (inicialmente vacía)
        self.columns_names_panel = wx.Panel(self.panel)
        self.columns_names_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.lbl_columns = wx.StaticText(self.columns_names_panel, label="Column names: ")
        self.columns_names_sizer.Add(self.lbl_columns, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        self.columns_names_panel.SetSizer(self.columns_names_sizer)
        self.left_sizer.Add(self.columns_names_panel, 0, wx.EXPAND | wx.ALL, 5)

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
        tab_modelo = wx.Panel(notebook)
        tab_optimizacion = wx.Panel(notebook)

        # Añadir los paneles al notebook
        notebook.AddPage(tab_modelo, "Modelo")
        notebook.AddPage(tab_optimizacion, "Optimización")

        # Creación del CheckBox
        self.toggle_components = wx.Button(tab_modelo, label="Define Model Dimensions")
        self.toggle_components.Bind(wx.EVT_BUTTON, self.on_define_model_dimensions_checked)

        # Creación de los TextCtrl para número de componentes y número de especies
        self.num_components_text, self.entry_nc = self.create_sheet_section("Number of Components:", "0", parent = tab_modelo)
        self.num_species_text, self.entry_nsp = self.create_sheet_section("Number of Species:", "0", parent = tab_modelo)
        
        self.sp_columns = wx.StaticText(tab_modelo, label="Select non-absorbent species: ")

        self.sp_select_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # Reemplazar el wx.ListCtrl por un wx.grid.Grid
        self.model_panel = wx.Panel(tab_modelo)
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
        modelo_sizer.Add(self.model_panel, 1, wx.EXPAND | wx.ALL, 5)
        tab_modelo.SetSizer(modelo_sizer)

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
        optimizador_choices = ["powell", "nelder-mead", "trust-constr", "cg", "bfgs", "l-bfgs-b", "tnc", "cobyla", "slsqp"]
        self.choice_optimizer_settings = wx.Choice(optimizador_panel, choices=optimizador_choices)
        optimizador_sizer.Add(self.choice_optimizer_settings, 0, wx.ALL, 5)
        self.choice_optimizer_settings.SetSelection(0)
        optimizador_panel.SetSizer(optimizador_sizer)

        # Crear el Grid
        self.grid = gridlib.Grid(tab_optimizacion)
        self.grid.CreateGrid(0, 4)
        self.grid.SetRowLabelSize(0)
        self.grid.SetColLabelValue(0, "Parameter")
        self.grid.SetColLabelValue(1, "Value")
        self.grid.SetColLabelValue(2, "Min")
        self.grid.SetColLabelValue(3, "Max")
        self.grid.AutoSizeColumns()

        # Configurar sizer y añadir al panel de la pestaña 'Optimización'
        optimizacion_sizer = wx.BoxSizer(wx.VERTICAL)
        optimizacion_sizer.Add(algo_panel, 0, wx.EXPAND | wx.ALL, 5)
        optimizacion_sizer.Add(ajustes_panel, 0, wx.EXPAND | wx.ALL, 5)
        optimizacion_sizer.Add(optimizador_panel, 0, wx.EXPAND | wx.ALL, 5)
        optimizacion_sizer.Add(self.grid, 1, wx.EXPAND | wx.ALL, 5)
        tab_optimizacion.SetSizer(optimizacion_sizer)

        # Añadir el notebook al left_sizer
        self.left_sizer.Add(notebook, 1, wx.EXPAND | wx.ALL, 5)
        self.SetSizer(self.left_sizer)
        self.Fit()
        self.Show()
        self.Layout()

    def process_data(self, event):
        # Placeholder for the actual data processing
        # Would call the functions from the provided script and display output
        
        # Verificar si se han seleccionado hojas válidas
        if not hasattr(self, 'file_path'):
            wx.MessageBox("No se ha seleccionado un archivo .xlsx", 
                        "Seleccione un archivo .xlsx para trabajar.", wx.OK | wx.ICON_ERROR)
            return  # Detener la ejecución de la función

        chem_shift = self.choice_sheet_chemshift.GetStringSelection()
        conc_entry = self.choice_sheet_conc.GetStringSelection()

        # Verificar si se han seleccionado hojas válidas
        if not chem_shift or not conc_entry:
            wx.MessageBox("Por favor, seleccione las hojas de Excel correctamente.", 
                        "Error en selección de hojas", wx.OK | wx.ICON_ERROR)
            return  # Detener la ejecución de la función

        # Extraer espectros para trabajar
        cs = pd.read_excel(self.file_path, chem_shift, header=0, index_col=0)

        # Extraer datos de esas columnas
        concentracion = pd.read_excel(self.file_path, conc_entry, header=0)

        nombres_de_columnas = concentracion.columns
        
        # Obtener los nombres de las columnas seleccionadas
        columnas_seleccionadas = [col for col, checkbox in self.vars_columnas.items() if checkbox.IsChecked()]

        if not columnas_seleccionadas:
            # Si no se ha seleccionado ningún checkbox, mostrar un mensaje de advertencia
            wx.MessageBox('Por favor, selecciona al menos una casilla para continuar.', 'Advertencia', wx.OK | wx.ICON_WARNING)
            return  # Salir de la función para no continuar con el procesamiento
        
        print("process_data iniciada")
        
        C_T = concentracion[columnas_seleccionadas].to_numpy()

        # Crear un diccionario que mapea nombres de columnas a sus nuevos índices en C_T
        column_indices_in_C_T = {name: index for index, name in enumerate(columnas_seleccionadas)}

        # Obtener los nombres de las columnas seleccionadas para receptor y huésped
        receptor_name = self.receptor_choice.GetStringSelection()
        guest_name = self.guest_choice.GetStringSelection()

        # Usar el diccionario para obtener los índices correctos dentro de C_T
        receptor_index_in_C_T = column_indices_in_C_T.get(receptor_name, -1)
        guest_index_in_C_T = column_indices_in_C_T.get(guest_name, -1)

        # Ahora puedes usar estos índices para indexar en C_T
        if receptor_index_in_C_T != -1 and guest_index_in_C_T != -1:
            G = C_T[:, guest_index_in_C_T]
            H = C_T[:, receptor_index_in_C_T]
            
        nc = len(C_T)
        n_comp = len(C_T.T)
        nw = len(spec)
        nm = spec.index.to_numpy()
        
                
        C_T = pd.DataFrame(C_T)
        
        
        # Intentar leer desde el archivo Excel
        grid_data = self.extract_data_from_grid()
        modelo = np.array(grid_data).T
        nas = self.on_selection_changed(event)

                    
        modelo = np.array(modelo)

        if not np.any(modelo):
            # Si no se ha seleccionado ningún checkbox, mostrar un mensaje de advertencia
            wx.MessageBox('Select a sheet model o create one.', 'Advertencia', wx.OK | wx.ICON_WARNING)
            return  # Salir de la función para no continuar con el procesamiento

        work_algo = self.choice_algoritm.GetStringSelection()
        model_sett = self.choice_model_settings.GetStringSelection() 

        k, bnds = self.extract_constants_from_grid()
        k_ini = k

        if not np.any(k):
            # Si no se ha seleccionado ningún checkbox, mostrar un mensaje de advertencia
            wx.MessageBox('Write an initial estimate for the constants.', 'Advertencia', wx.OK | wx.ICON_WARNING)
            return  # Salir de la función para no continuar con el procesamiento

        if work_algo == "Newton-Raphson":
            
            from NR_conc_algoritm import NewtonRaphson

            res = NewtonRaphson(C_T, modelo, nas, model_sett)

        elif work_algo == "Levenberg-Marquardt": 
            
            from LM_conc_algoritm import LevenbergMarquardt

            res = LevenbergMarquardt(C_T, modelo, nas, model_sett) 