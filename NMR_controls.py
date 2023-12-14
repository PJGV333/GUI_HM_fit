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

        # Crear un TextCtrl en lugar de StaticText para mostrar la ruta del archivo
        self.lbl_file_path = wx.TextCtrl(self.panel, style=wx.TE_READONLY | wx.TE_MULTILINE | wx.HSCROLL)
        self.lbl_file_path.SetMinSize((-1, 45))  # Ajustar el tamaño mínimo para evitar que sea demasiado grande
        self.lbl_file_path.SetValue("No file selected")
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

        #self.choice_chemshifts.Bind(wx.EVT_CHOICE, self.on_chemical_shift_sheet_selected)

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

        chem_shift = self.choice_chemshifts.GetStringSelection()
        conc_entry = self.choice_sheet_conc.GetStringSelection()

        # Verificar si se han seleccionado hojas válidas
        if not chem_shift or not conc_entry:
            wx.MessageBox("Por favor, seleccione las hojas de Excel correctamente.", 
                        "Error en selección de hojas", wx.OK | wx.ICON_ERROR)
            return  # Detener la ejecución de la función

        
        chemshift_data = pd.read_excel(self.file_path, chem_shift, header=0)

        # Obtener los nombres de las columnas seleccionadas en los Scheckboxes de desplazamientos químicos
        # Verificar qué checkboxes están marcados
        for col, checkbox in self.vars_chemshift.items():
            print(f"Columna: {col}, Checkbox marcado: {checkbox.IsChecked()}")

        columnas_chemshift_seleccionadas = [col for col, checkbox in self.vars_chemshift.items() if checkbox.IsChecked()]

        if not columnas_chemshift_seleccionadas:
            # Si no se ha seleccionado ningún checkbox, mostrar un mensaje de advertencia
            wx.MessageBox('Por favor, selecciona al menos una casilla de desplazamientos químicos para continuar.', 'Advertencia', wx.OK | wx.ICON_WARNING)
            return  # Salir de la función para no continuar con el procesamiento

        # Extraer datos de las columnas seleccionadas de desplazamientos químicos
        Chem_Shift_T = chemshift_data[columnas_chemshift_seleccionadas].to_numpy()

        # Crear un diccionario que mapea nombres de columnas a sus nuevos índices en Chem_Shift_T
        column_indices_in_Chem_Shift_T = {name: index for index, name in enumerate(columnas_chemshift_seleccionadas)}

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
    
        C_T = pd.DataFrame(C_T)
        Chem_Shift_T = pd.DataFrame(Chem_Shift_T)
        dq1 = Chem_Shift_T - Chem_Shift_T.iloc[0]
        dq = np.array(dq1)

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

        def cal_delta(k):
            C = res.concentraciones(k)[0]
            xi = C / H[:, np.newaxis]
            coeficientes = np.linalg.pinv(xi) @ dq
            dq_cal = coeficientes.T @ xi.T
            return dq_cal

        def cal_coef(k):
            C = res.concentraciones(k)[0]
            xi = C / H[:, np.newaxis]
            coeficientes = np.linalg.pinv(xi) @ dq
            return coeficientes

        def f_m(k):
            dq_cal = cal_delta(k)
            r = dq - dq_cal.T
            rms = np.sqrt(np.mean(np.square(r)))
            self.res_consola("f(x)", rms)
            self.res_consola("x", k)
            return rms
        
        def f_m2(k):
            dq_cal = cal_delta(k)
            r = dq - dq_cal.T
            rms = np.sqrt(np.mean(np.square(r)))
            return rms

        # Registrar el tiempo de inicio
        inicio = timeit.default_timer()
        
        optimizer = self.choice_optimizer_settings.GetStringSelection()
        print(optimizer)
        print(bnds)

    
        r_0 = optimize.minimize(f_m, k, method=optimizer, bounds=bnds)

                    
        # Registrar el tiempo de finalización
        fin = timeit.default_timer()
        
        # Calcular el tiempo total de ejecución
        tiempo_total = fin - inicio
        
        print("El tiempo de ejecución de la función fue: ", tiempo_total, "segundos.")


        k = np.ravel(r_0.x) 


        # Calcular el SER
        n = len(H)
        p = len(k)
        SER = f_m2(k)


        # Calcular la matriz jacobiana de los residuos
        def residuals(k):
            dq_cal = cal_delta(k)
            r = dq - dq_cal.T
            return r.flatten()

        #epsilon = np.sqrt(np.finfo(float).eps)
        #jacobian = np.array([approx_fprime(k, lambda ki: residuals(ki)[i], epsilon) for i in range(n)])

        def finite(x, fun):
            dfdx = []
            delta = np.sqrt(np.finfo(float).eps)
            for i in range(len(x)):
                step = np.zeros(len(x))
                step[i] = delta
                dfdx.append((fun(x + step) - fun(x - step)) / (2 * delta))
            return np.array(dfdx)

        jacobian = finite(k, residuals)

        # Calcular la matriz de covarianza
        cov_matrix = SER**2 * np.linalg.inv(jacobian @ jacobian.T)

        # Calcular el error estándar de las constantes de asociación
        SE_k = np.sqrt(np.diag(cov_matrix))

        # Calcular el error porcentual
        error_percent = (SE_k / np.abs(k)) * 100

        coef = cal_coef(k)

        C, Co = res.concentraciones(k)

        self.figura((G/H), (C/np.max(C))*100, ":o", "Abundance (%)", "[G]/[H]", "Perfil de concentraciones")
        
        dq_cal = cal_delta(k).T


        self.figura2((G/H), dq, cal_delta(k).T, "o", ":", "$\Delta$$\delta$ (ppm)", "[G]/[H]", 1, "Ajuste")

        MAE = abs(SER / nc)
        dif_en_ct = round(max(100 - (np.sum(C, 1) * 100 / max(H))), 2)

        # 1. Calcular la varianza de los residuales
        residuals_array = residuals(k)
        var_residuals = np.var(residuals_array)

        # 2. Calcular la varianza de los datos experimentales
        var_data_original = np.var(dq)

        # 3. Calcular covfit
        covfit = var_residuals / var_data_original

         ####pasos para imprimir bonito los resultados. 
        # Función para calcular los anchos máximos necesarios para cada columna
        def calculate_max_column_widths(headers, data_rows):
            column_widths = [len(header) for header in headers]
            for row in data_rows:
                for i, item in enumerate(row):
                    # Considerar la longitud del item como cadena
                    column_widths[i] = max(column_widths[i], len(str(item)))
            return column_widths

        # Encabezados y datos de ejemplo
        headers = ["Constant", "log10(K) ± Error", "% Error", "RMS", "Covfit"]
        data = [
            [f"K{i+1}", f"{k[i]:.2e} ± {SE_k[i]:.2e}", f"{error_percent[i]:.2f}", f"{SER:.2e}" if i == 0 else "", f"{covfit:.2e}" if i == 0 else ""]
            for i in range(len(k))
        ]

        # Calcular los anchos máximos para las columnas
        max_widths = calculate_max_column_widths(headers, data)

        # Crear la tabla con los anchos ajustados
        table_lines = []

        # Encabezado
        header_line = " | ".join(f"{header.ljust(max_widths[i])}" for i, header in enumerate(headers))
        table_lines.append("-" * len(header_line))
        table_lines.append(header_line)
        table_lines.append("-" * len(header_line))

        # Filas de datos
        for row in data:
            line = " | ".join(f"{item.ljust(max_widths[i])}" for i, item in enumerate(row))
            table_lines.append(line)

        # Unir las líneas para formar la tabla
        adjusted_table = "\n".join(table_lines)
        print(adjusted_table)
        ###### Aqui terminan los pasos para la impresión bonita########

        nombres = [f"k{i}" for i in range(1, len(k)+1)]
        k_nombres = [f"{n}" for n in nombres]
        
        K = np.array([k, error_percent]).T
        
        modelo = pd.DataFrame(modelo)
        C = pd.DataFrame(C)
        Co = pd.DataFrame(Co)
        C_T = pd.DataFrame(C_T)
        dq = pd.DataFrame(dq)
        dq_cal = pd.DataFrame(dq_cal)
        k = pd.DataFrame(K, index = [k_nombres])
        k_ini = pd.DataFrame(k_ini)
        covfit = covfit
        coef = pd.DataFrame(coef)
        
        stats = np.array([SER, MAE, dif_en_ct, covfit, optimizer])
        stats = pd.DataFrame(stats, index= ["RMS", "Error absoluto medio", 
                                            "Diferencia en C total (%)", 
                                            "covfit", "optimizer"])
        
        # Generar nombres de columnas
        num_columns_C = len(C.columns)
        column_names_C = [f"sp_{i}" for i in range(1, num_columns_C + 1)]

        num_columns_Co = len(Co.columns)
        column_names_Co = [f"sp_{i}" for i in range(1, num_columns_Co + 1)]

        num_columns_dq1 = len(Chem_Shift_T.columns)
        column_names_dq1 = [f"dobs_{i}" for i in range(1, num_columns_dq1 + 1)]

        num_columns_dq_cal = len(dq_cal.columns)
        column_names_dq_cal = [f"dcal_{i}" for i in range(1, num_columns_dq_cal + 1)]

        num_columns_coef = len(coef.columns)
        column_names_coef = [f"coef_{i}" for i in range(1, num_columns_coef + 1)]

        num_columns_ct = len(C_T.columns)
        column_names_ct = [f"ct_{i}" for i in range(1, num_columns_ct + 1)]

        column_names_k = ["Constants", "Error (%)"]

        column_names_k_ini = ["Constants"]

        column_names_stats = ["Stats"]

        # Asignar nombres de columnas a los DataFrames
        C.columns = column_names_C
        Co.columns = column_names_Co
        dq.columns = column_names_dq1
        dq_cal.columns = column_names_dq_cal
        k.columns = column_names_k
        k_ini.columns = column_names_k_ini
        stats.columns = column_names_stats
        C_T.columns = column_names_ct
        coef.columns = column_names_coef

        self.C = C
        self.Co = Co
        self.dq = dq
        self.dq_cal = dq_cal
        self.k = k
        self.k_ini = k_ini
        self.stats = stats
        self.C_T = C_T
        self.coef = coef

