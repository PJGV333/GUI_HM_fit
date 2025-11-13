import wx
from Methods import BaseTechniquePanel
import wx.grid as gridlib
import numpy as onp  # NumPy “real”, no JAX
from np_backend import xp as np, jit, jacrev, vmap, lax
import pandas as pd
import matplotlib
matplotlib.use('WXAgg')
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy import optimize
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
import warnings
warnings.filterwarnings("ignore")
import timeit
from Methods import BaseTechniquePanel
from wx.lib.scrolledpanel import ScrolledPanel
from errors import compute_errors_spectro_varpro
from core_ad_probe import solve_A_nnls_pgd


def pinv_cs(A, rcond=1e-12):
    """
    Pseudoinversa estable (ahora sin complex-step).
    Thin wrapper sobre np.linalg.pinv con fallback por si SVD no converge.
    """
    try:
        return np.linalg.pinv(A, rcond=rcond)
    except np.linalg.LinAlgError:
        # Fallback regularizado tipo ridge
        ATA = A.T @ A + (rcond if np.isscalar(rcond) else 1e-12) * np.eye(A.shape[1], dtype=A.dtype)
        return np.linalg.solve(ATA, A.T)


def _solve_A(C, YT, rcond=1e-10):
    # C: (m×s), YT: (nw×m)  →  A: (s×nw)
    A, *_ = np.linalg.lstsq(C, YT, rcond=rcond)
    return A

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
        self.scrolled_window = ScrolledPanel(self.panel, style=wx.HSCROLL | wx.TAB_TRAVERSAL | wx.ALWAYS_SHOW_SB)
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

        # Concentraciones
        self.sheet_conc_panel, self.choice_sheet_conc = self.create_sheet_dropdown_section("Concentration Sheet Name:", self)
        self.left_sizer.Add(self.sheet_conc_panel, 0, wx.EXPAND | wx.ALL, 5)

        dropdowns_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.left_sizer.Add(dropdowns_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # ScrolledWindow para los checkboxes de nombres de columnas (inicialmente vacío)
        #self.columns_names_panel = wx.ScrolledWindow(self.panel, style=wx.HSCROLL)
        #self.columns_names_panel.SetScrollRate(10, 0)  # Configurar la velocidad de desplazamiento horizontal
        self.columns_names_panel = ScrolledPanel(self.panel, style=wx.HSCROLL | wx.TAB_TRAVERSAL | wx.ALWAYS_SHOW_SB)
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
        self.EFA_cb.SetValue(True)  # Marcar el checkbox por defecto
        self.sheet_EV_panel.GetSizer().Insert(0, self.EFA_cb, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        self.left_sizer.Add(self.sheet_EV_panel, 0, wx.ALL | wx.EXPAND, 5)

        # Creación del wx.Notebook
        notebook = wx.Notebook(self.panel)

        # Creación de los paneles para las pestañas
        #tab_modelo = wx.Panel(notebook)
        tab_modelo = ScrolledPanel(notebook, style=wx.VSCROLL | wx.TAB_TRAVERSAL | wx.ALWAYS_SHOW_SB)
        tab_modelo.SetupScrolling(scroll_x=False, scroll_y=True, rate_x=0, rate_y=10)
        tab_optimizacion = wx.Panel(notebook)

        # Añadir los paneles al notebook
        notebook.AddPage(tab_modelo, "Model")
        notebook.AddPage(tab_optimizacion, "Optimization")

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
        self.model_panel = ScrolledPanel(tab_modelo, style=wx.VSCROLL | wx.TAB_TRAVERSAL | wx.ALWAYS_SHOW_SB)
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
        self.grid.CreateGrid(0, 4)
        self.grid.SetRowLabelSize(0)
        self.grid.SetColLabelValue(0, "Parameter")
        self.grid.SetColLabelValue(1, "Value")
        self.grid.SetColLabelValue(2, "Min")
        self.grid.SetColLabelValue(3, "Max")
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
        # Placeholder for the actual data processing
        # Would call the functions from the provided script and display output
        
        # Verificar si se han seleccionado hojas válidas
        if not hasattr(self, 'file_path'):
            wx.MessageBox("No se ha seleccionado un archivo .xlsx", 
                        "Seleccione un archivo .xlsx para trabajar.", wx.OK | wx.ICON_ERROR)
            return  # Detener la ejecución de la función

        spec_entry = self.choice_sheet_spectra.GetStringSelection()
        conc_entry = self.choice_sheet_conc.GetStringSelection()

        # Verificar si se han seleccionado hojas válidas
        if not spec_entry or not conc_entry:
            wx.MessageBox("Por favor, seleccione las hojas de Excel correctamente.", 
                        "Error en selección de hojas", wx.OK | wx.ICON_ERROR)
            return  # Detener la ejecución de la función

        # Extraer espectros para trabajar
        spec = pd.read_excel(self.file_path, spec_entry, header=0, index_col=0)
        nm = spec.index.to_numpy()

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
        
        #spec = spec.to_numpy(dtype=float) if hasattr(spec, "to_numpy") else onp.asarray(spec, dtype=float)
        
        C_T = concentracion[columnas_seleccionadas].to_numpy()

        #C_T = C_T.to_numpy(dtype=float) if hasattr(C_T, "to_numpy") else onp.asarray(C_T, dtype=float)


        # Crear un diccionario que mapea nombres de columnas a sus nuevos índices en C_T
        column_indices_in_C_T = {name: index for index, name in enumerate(columnas_seleccionadas)}

        # Obtener los nombres de las columnas seleccionadas para receptor y huésped
        receptor_name = self.receptor_choice.GetStringSelection()
        guest_name = self.guest_choice.GetStringSelection()

        # Usar el diccionario para obtener los índices correctos dentro de C_T
        receptor_index_in_C_T = column_indices_in_C_T.get(receptor_name, -1)
        guest_index_in_C_T = column_indices_in_C_T.get(guest_name, -1)

      
        # Verifica si al menos uno de los índices es diferente de -1
        if receptor_index_in_C_T != -1 or guest_index_in_C_T != -1:
            
            # Si guest_index_in_C_T es válido, asigna la columna correspondiente a G
            if guest_index_in_C_T != -1:
                G = C_T[:, guest_index_in_C_T]

            # Si receptor_index_in_C_T es válido, asigna la columna correspondiente a H
            if receptor_index_in_C_T != -1:
                H = C_T[:, receptor_index_in_C_T]
            
        nc = len(C_T)
        n_comp = len(C_T.T)
        nw = len(spec)
        
        def SVD_EFA(spec, args = (nc)):
            u, s, v = onp.linalg.svd(spec, full_matrices=False)
            
            #EFA fijo
            
            L = range(1,(nc + 1), 1)
            L2 = range(0, nc, 1)
            
            X = []
            for i in L:
                uj, sj, vj = onp.linalg.svd(spec.T.iloc[:i,:], full_matrices=False)
                X.append(sj)
            
            ev_s = pd.DataFrame(X)
            ev_s0 = onp.array(ev_s)
            
            X2 = []
            for i in L2:
                ui, si, vi = onp.linalg.svd(spec.T.iloc[i:,:], full_matrices=False)
                X2.append(si)
            
            ev_s1 = pd.DataFrame(X2)
            ev_s10 = np.array(ev_s1) 
            
            self.figura(range(0, nc), np.log10(s), "o", "log(EV)", "# de autovalores", "Eigenvalues")      
            self.figura2(G, np.log10(ev_s0), np.log10(ev_s10), "k-o", "b:o", "log(EV)", "[G], M", 1, "EFA")
                
            EV = int(self.entry_EV.GetValue())

            if EV == 0:
                EV = nc

            print("Eigenvalues used: ", EV)      

            Y = u[:,0:EV] @ np.diag(s[0:EV:]) @ v[0:EV:]
            #Y_svd = pd.DataFrame(Y, index = [list(nm)])
            return Y, EV
        
        if self.EFA_cb.GetValue():
            Y, EV = SVD_EFA(spec)
        else:
            Y = np.array(spec)
        
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


        # Implementing the abortividades function
        def abortividades(k, Y):
            C, Co = res.concentraciones(k)  # Assuming the function concentraciones returns C and Co
            #A = np.linalg.pinv(C) @ Y.T 
            A = solve_A_nnls_pgd(C, Y.T, ridge=0.0, max_iters=300)
            return np.all(A >= 0)
        
        def f_m2(k):
            C = res.concentraciones(k)[0]  
            A = solve_A_nnls_pgd(C, Y.T, ridge=0.0, max_iters=300)  
            r = C @ A - Y.T #se cambio np.linag.pinv por np.linalg.pinv
            rms = np.sqrt(np.mean(np.square(r)))
            #print(f"f(x): {rms}")
            #print(f"x: {k}")
            return rms, r
            
        def f_m(k):
            C = res.concentraciones(k)[0]    
            A = solve_A_nnls_pgd(C, Y.T, ridge=0.0, max_iters=300)  
            r = C @ A - Y.T
            #r = C @ np.linalg.pinv(C) @ Y.T - Y.T 
            rms = np.sqrt(np.mean(np.square(r)))
            self.res_consola("f(x)", rms)
            self.res_consola("x", k)
            
            # Procesar eventos de la GUI para actualizar la consola en tiempo real
            wx.Yield()
            return rms
                
        #bnds = [(-20, 20)]*len(k.T) #Bounds(0, 1e15, keep_feasible=(True)) #
        
        # Registrar el tiempo de inicio
        inicio = timeit.default_timer()
        
        optimizer = self.choice_optimizer_settings.GetStringSelection()
        print(optimizer)
        print(bnds)

        def verificar_bounds(bounds):
            """
            Verifica si los límites contienen np.inf o -np.inf.
            Retorna True si los límites son válidos, False de lo contrario.
            """
            for (min_val, max_val) in bounds:
                if np.isinf(min_val) or np.isinf(max_val):
                    return False
            return True
    
        
        if optimizer  == "differential_evolution":
            if verificar_bounds(bnds) == False:
                # Mostrar un MessageBox en caso de valores infinitos en los límites
                wx.MessageBox('Los límites no deben contener valores infinitos.', 
                            'Error en los Límites', wx.OK | wx.ICON_ERROR)
                # Aquí puedes manejar el error o simplemente retornar para evitar seguir con el procesamiento

            else:
                self.r_0 = differential_evolution(f_m, bnds, x0 = k, strategy='best1bin', 
                          maxiter=1000, popsize=15, tol=0.01, 
                           mutation=(0.5, 1), recombination=0.7,
                           init='latinhypercube')
            
        else:
            self.r_0 = optimize.minimize(f_m, k, method=optimizer, bounds=bnds)

                    
        # Registrar el tiempo de finalización
        fin = timeit.default_timer()
        
        # Calcular el tiempo total de ejecución
        tiempo_total = fin - inicio
        
        print("El tiempo de ejecución de la función fue: ", tiempo_total, "segundos.")
        
        k = self.r_0.x 
        k = np.ravel(k)
        
        
        from errors import pinv_cs, percent_error_log10K, sensitivities_wrt_logK

        
        # Calcular la matriz jacobiana de los residuos
        def residuals(k):
            C = res.concentraciones(k)[0]
            A = solve_A_nnls_pgd(C, Y.T, ridge=0.0, max_iters=300)  
            r = C @ A - Y.T
            #r = C @ np.linalg.pinv(C) @ Y.T - Y.T 
            return r.ravel() #r.flatten()
                
        # --- al inicio del archivo asegúrate de tener:
        # from errors import pinv_cs, percent_error_log10K, sensitivities_wrt_logK

        def compute_errors_spectroscopy(k, res, Y, modelo, nas):
            """
            Flujo ÚNICO (NR/LM) para errores en Espectroscopia usando sensibilidades implícitas.
            k      : (p,)    parámetros óptimos (log10 K)
            res    : solver de concentraciones con .concentraciones(k)
            Y      : (nw × m) matriz de datos (tal como la usas; su .T es (m × nw))
            modelo : (nspec×n_comp) o (n_comp×nspec)
            nas    : lista de índices de especies NO absorbentes
            Devuelve: dict con percK, SE_K, SE_log10K, Cov_log10K, RMS, s2, A, J, yfit
            """
            import numpy as np

            k = np.ravel(k)
            m = Y.shape[1]                     # nº puntos (m)
            nw = Y.shape[0]                    # nº "observables" (longitudes de onda)
            C, Co = res.concentraciones(k)     # C: (m × n_abs), Co: (m × nspec)

            # Resolver A en el óptimo (mínimos cuadrados)
            def _solve_A(C, YT, rcond=1e-10):
                A, *_ = np.linalg.lstsq(C, YT, rcond=rcond)   # (n_abs × nw)
                return A

            A = _solve_A(C, Y.T)               # (n_abs × nw)

            npts, n_abs = C.shape
            p = len(k)
            nspec = Co.shape[1]

            # Normalizar orientación de 'modelo' y deducir n_comp
            M_in = np.asarray(modelo, dtype=float)
            if M_in.shape[0] == nspec:
                Ms = M_in                      # (nspec × n_comp)
                n_comp = Ms.shape[1]
            elif M_in.shape[1] == nspec:
                Ms = M_in.T                    # (nspec × n_comp)
                n_comp = Ms.shape[1]
            else:
                raise ValueError(f"modelo incompatible con Co: nspec={nspec}, modelo shape={M_in.shape}")

            # Índices de especies absorbentes (para pasar de Co -> C)
            abs_idx = [j for j in range(nspec) if j not in nas]

            # Mapeo parámetros -> columnas de especies (estándar: COMPLEJOS = n_comp..nspec-1)
            param_idx = list(range(n_comp, nspec))
            if len(param_idx) != p:
                if p <= nspec:
                    param_idx = list(range(nspec - p, nspec))  # fallback consistente con p
                else:
                    raise ValueError(f"p={p} mayor que nspec={nspec}.")

            # Construir Jacobiano de residuos J (p × m*nw)
            J = np.zeros((p, npts * nw), dtype=float)

            for i in range(npts):
                # dCspec/dlog10K (nspec × p) en la fila i
                dC_dlog10K = sensitivities_wrt_logK(Co[i], Ms, param_idx=param_idx)   # (nspec × p)

                # Sólo especies absorbentes
                dC_abs = dC_dlog10K[abs_idx, :]       # (n_abs × p)

                # y_i = C_i @ A  ⇒ dy_i/dlogK = (dC_abs)^T @ A  (p × nw)
                dyi_dk = (dC_abs.T @ A)               # (p × nw)

                # residuo: r_i = (C_i @ A) - Y_i  ⇒ dr_i/dlogK = dy_i/dlogK
                # Si prefieres r = Y_i - (C_i@A), usa el signo opuesto; la covarianza no cambia.
                J[:, i*nw:(i+1)*nw] = dyi_dk

            # Residuos y métricas
            yfit = C @ A                               # (m × nw)
            r = (yfit - Y.T).ravel()                   # (m*nw,)
            dof = max(r.size - p, 1)
            s2  = float((r @ r) / dof)
            Cov_log10K = s2 * pinv_cs(J @ J.T)         # (p × p)
            SE_log10K  = np.sqrt(np.clip(np.diag(Cov_log10K), 0.0, np.inf))
            percK, SE_K, _ = percent_error_log10K(k, SE_log10K)

            RMS = float(np.sqrt(np.mean(r*r)))

            return {
                "percK": percK, "SE_K": SE_K, "SE_log10K": SE_log10K, "Cov_log10K": Cov_log10K,
                "RMS": RMS, "s2": s2, "A": A, "J": J, "yfit": yfit
            }

        metrics = compute_errors_spectro_varpro(
        k=self.r_0.x, res=res, Y=Y, modelo=modelo, nas=nas,
        rcond=1e-10, use_projector=True
        )
        SE_log10K = metrics["SE_log10K"]
        SE_K      = metrics["SE_K"]
        percK     = metrics["percK"]
        rms       = metrics["RMS"]
        covfit    = metrics["s2"]
        A         = metrics["A"]
        yfit      = metrics["yfit"]
        cov_matrix= metrics["Cov_log10K"]


        C, Co = res.concentraciones(k)
        

        if n_comp == 1:
            self.figura(H, C, ":o", "[Especies], M", "[H], M", "Perfil de concentraciones")

            y_cal = C @ np.linalg.pinv(C) @ Y.T
                            
            ssq, r0 = f_m2(k)
            #rms = f_m(k)
            
            A = solve_A_nnls_pgd(C, Y.T, ridge=0.0, max_iters=300)  

            self.figura(nm, A.T, "-", "Epsilon (u. a.)", "$\lambda$ (nm)", "Absortividades molares")
            self.figura2(nm, Y, y_cal.T, "-k", "k:", "Y observada (u. a.)", "$\lambda$ (nm)", 0.5, "Ajuste")
            
        else:
            self.figura(G, C, ":o", "[Species], M", "[G], M", "Perfil de concentraciones")
                    
            y_cal = C @ np.linalg.pinv(C) @ Y.T
                            
            ssq, r0 = f_m2(k)
            #rms = f_m(k)
            
            A = solve_A_nnls_pgd(C, Y.T, ridge=0.0, max_iters=300) 
            
            if not self.EFA_cb.GetValue():
                self.figura2(G, Y.T, y_cal, "ko", ":", "Y observada (u. a.)", "[X], M", 1, "Ajuste")
            else:
                self.figura(nm, A.T, "-", "Epsilon (u. a.)", "$\lambda$ (nm)", "Absortividades molares")
                self.figura2(nm, Y, y_cal.T, "-k", "k:", "Y observada (u. a.)", "$\lambda$ (nm)", 0.5, "Ajuste")   

                
        ssq, r0 = f_m2(k)
        
        # Lack of fit (porcentaje)
        # Definido como 100 * SS_res / SS_tot, donde:
        #   SS_res = sum(r^2) y SS_tot = sum( (Y - mean(Y))^2 )
        Yvec = Y.T  # mismo arreglo que usamos para r0
        SS_res = float(np.sum(r0**2))
        SS_tot = float(np.sum((Yvec - np.mean(Yvec))**2))
        
        if SS_tot <= 1e-30:    # evita división por cero si no hay varianza
            lof = 0.0
        else:
            lof = 100.0 * SS_res / SS_tot
            
        MAE = np.mean(abs(r0))
        dif_en_ct = round(max(100 - (np.sum(C, 1) * 100 / max(H))), 2)
        
        # Lack of fit (%)
        Yvec = Y.T
        SS_res = float(np.sum(r0**2))
        SS_tot = float(np.sum((Yvec - np.mean(Yvec))**2))
        lof = 0.0 if SS_tot <= 1e-30 else 100.0 * SS_res / SS_tot
        
        dif_en_ct = round(max(100 - (np.sum(C, 1) * 100 / max(H))), 2)
            
            
        ####pasos para imprimir bonito los resultados. 
        # Función para calcular los anchos máximos necesarios para cada columna
        def calculate_max_column_widths(headers, data_rows):
            column_widths = [len(header) for header in headers]
            for row in data_rows:
                for i, item in enumerate(row):
                    # Considerar la longitud del item como cadena
                    column_widths[i] = max(column_widths[i], len(str(item)))
            return column_widths

        def fmt(x, pat): 
            return "—" if not np.isfinite(x) else pat.format(x)

        # Encabezados
        headers = [
            "Constant",
            "log10(K) ± SE(log10K)",
            "% Error (K, Δ-method)",
            "RMS",
            "lof",
            "s² (var. reducida)"
        ]

        # Filas
        data = [
            [
                f"K{i+1}",
                f"{k[i]:.2e} ± {SE_log10K[i]:.2e}",   # ± en log10(K)
                f"{percK[i]:.2f} %",                  # % en K (delta method)
                f"{rms:.2e}" if i == 0 else "",
                f"{lof:.2e}" if i == 0 else "",
                f"{covfit:.2e}" if i == 0 else "",
            ]
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
        ###### Aqui terminan los pasos para la impresión bonita

        nombres = [f"k{i}" for i in range(1, len(k)+1)]
        k_nombres = [f"{n}" for n in nombres]
        
        K = np.array([k, percK]).T
        
        Y = pd.DataFrame(Y, index = [list(nm)])
        modelo = pd.DataFrame(modelo)
        C = pd.DataFrame(C)
        Co = pd.DataFrame(Co)
        k = pd.DataFrame(K, index = [k_nombres])
        k_ini = pd.DataFrame(k_ini.T, index = [k_nombres])
        A = pd.DataFrame(A.T, index = [list(nm)])
        phi = pd.DataFrame(y_cal.T, index = [list(nm)])
        cov_matrix = covfit
        
        if not self.EFA_cb.GetValue():
            EV = nc
            
        stats = onp.array([rms, lof, MAE, dif_en_ct, EV, cov_matrix, optimizer], dtype=object)
        stats = pd.DataFrame(stats, index= ["RMS", "Falta de ajuste (%)",\
                                            "Error absoluto medio", "Diferencia en C total (%)", "# Autovalores", "covfit", "optimizer"])
        
        # Generar nombres de columnas
        num_columns_C = len(C.columns)
        column_names_C = [f"sp_{i}" for i in range(1, num_columns_C + 1)]
        
        num_columns_Co = len(Co.columns)
        column_names_Co = [f"sp_{i}" for i in range(1, num_columns_Co + 1)]
        
        num_columns_yobs = len(Y.columns)
        column_names_yobs = [f"yobs_{i}" for i in range(1, num_columns_yobs + 1)]
        
        num_columns_ycal = len(phi.columns)
        column_names_ycal = [f"ycal_{i}" for i in range(1, num_columns_ycal + 1)]
        
        num_columns_A = len(A.columns)
        column_names_A = [f"A_{i}" for i in range(1, num_columns_A + 1)]
        
        num_columns_ct = len(C_T.columns)
        column_names_ct = [f"ct_{i}" for i in range(1, num_columns_ct + 1)]
        
        column_names_k = ["Constants", "Error (%)"]
        column_names_k_ini = ["Constants"]
        
        column_names_stats = ["Stats"]
        
        # Asignar nombres de columnas a los DataFrames
        C.columns = column_names_C
        Co.columns = column_names_Co
        Y.columns = column_names_yobs
        phi.columns = column_names_ycal
        A.columns = column_names_A
        k.columns = column_names_k
        k_ini.columns = column_names_k_ini
        stats.columns = column_names_stats
        C_T.columns = column_names_ct

        self.modelo = modelo
        self.C = C
        self.Co = Co
        self.Y = Y
        self.phi = phi
        self.A = A
        self.k = k
        self.k_ini = k_ini
        self.stats = stats
        self.C_T = C_T


        