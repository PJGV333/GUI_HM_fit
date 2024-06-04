import wx
from Methods import BaseTechniquePanel
import wx.grid as gridlib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from Methods import BaseTechniquePanel

# Clase para simulación
class Simulation_controlsPanel(BaseTechniquePanel):
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

        # Creación del wx.Notebook
        notebook = wx.Notebook(self.panel)

        # Creación de los paneles para las pestañas
        tab_modelo = wx.Panel(notebook)
        tab_optimizacion = wx.Panel(notebook)

        # Añadir los paneles al notebook
        notebook.AddPage(tab_modelo, "Model")
        notebook.AddPage(tab_optimizacion, "Optimization")
        
        #Aquí crear los statictext.
        
        # Creación del CheckBox
        self.toggle_components = wx.Button(tab_modelo, label="Define Model Dimensions")
        self.toggle_components.Bind(wx.EVT_BUTTON, self.on_define_model_dimensions_checked)

        # Creación de los TextCtrl para número de componentes y número de especies
        self.num_components_text, self.entry_nc = self.create_sheet_section("Number of Components:", "0", parent = tab_modelo)
        self.num_species_text, self.entry_nsp = self.create_sheet_section("Number of Species:", "0", parent = tab_modelo)
        
        self.num_components_text.Bind(wx.EVT_TEXT, self.on_num_components_change)

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

    #####################################################################################################
        
    def on_num_components_change(self, event):
        """
        Manejador del evento que se dispara cuando cambia el texto del número de componentes.
        """
        current_panel = self.technique_notebook.GetCurrentPage()
        if current_panel is not None and current_panel.GetLabel() == "Simulation":
            try:
                num_components = int(self.num_components_text.GetValue())
            except ValueError:
                # Manejar el caso en que el valor no sea un número
                return

            # Eliminar TextCtrls anteriores si existen
            if hasattr(self, 'component_inputs'):
                for input_control in self.component_inputs:
                    input_control.Destroy()

            # Crear nuevos TextCtrls
            self.create_component_inputs(num_components)

    
    def on_confirm_components(self, event):
        """
        Manejador del evento que se dispara cuando se presiona el botón de confirmar componentes.
        """
        current_panel = self.technique_notebook.GetCurrentPage()
        if current_panel is not None and current_panel.GetLabel() == "Simulation":
            try:
                num_components = int(self.num_components_text.GetValue())
            except ValueError:
                # Manejar el caso en que el valor no sea un número
                wx.MessageBox("Please enter a valid number of components", "Error", wx.OK | wx.ICON_ERROR)
                return

            # Eliminar TextCtrls anteriores si existen
            if hasattr(self, 'component_inputs'):
                for input_control in self.component_inputs:
                    input_control.Destroy()
                del self.component_inputs

            # Crear nuevos TextCtrls
            self.create_component_inputs(num_components)


    def create_component_inputs(self, num_components):
        """
        Crea entradas para las concentraciones de los componentes.

        Args:
            num_components (int): Número de componentes ingresado por el usuario.
        """
        self.component_inputs = []
        for i in range(num_components):
            label = f"C{i+1} Concentration Range (e.g. 1e-3,2e-5,50):"
            default_value = "1e-3,2e-5,50"
            input_control = self.create_sheet_section(label, default_value, parent=self)
            self.component_inputs.append(input_control)
            self.left_sizer.Add(input_control, 0, wx.EXPAND)

        self.panel.Layout()  # Actualizar el layout para mostrar los nuevos controles

    #####################################################################################################
    
    def process_data(self, event):
        H = np.linspace(1e-3, 2e-5, 50)
        G = np.linspace(2e-3, 0, 50)
        P = np.linspace(0, 1e-3, 50)
        C_T = np.array([H, G, P]).T
       
        # Intentar leer desde el archivo Excel
        grid_data = self.extract_data_from_grid()
        modelo = np.array(grid_data).T
        nas = self.on_selection_changed(event)
        work_algo = self.choice_algoritm.GetStringSelection()
        model_sett = self.choice_model_settings.GetStringSelection() 

        k, bnds = self.extract_constants_from_grid()
        k = k
        if work_algo == "Newton-Raphson":
                
            from NR_conc_algoritm import NewtonRaphson

            res = NewtonRaphson(C_T, modelo, nas, model_sett)

        elif work_algo == "Levenberg-Marquardt": 
            
            from LM_conc_algoritm import LevenbergMarquardt

            res = LevenbergMarquardt(C_T, modelo, nas, model_sett) 

        plt.plot(G, C, ":o")
        plt.ylabel("[Especies], M", size = "xx-large")
        plt.xlabel("[G], M", size = "xx-large")
        plt.xticks(size = "large")
        plt.yticks(size = "large")
        plt.show()

        # Funcion gaussiana 

        def gauss(x, xmax, width):
            G = np.exp(-np.log(2)*((x-xmax)/(width/2))**2)
            return G
        
        # Molar component spectra A

        lam = np.linspace(190,800, 350)

        a = 30000 * gauss(lam, 405, 100)
        b = 25000 * gauss(lam, 480, 100)
        c = 18000 * gauss(lam, 390, 175)
        d = 14000 * gauss(lam, 360, 150)
        e = 10000 * gauss(lam, 340, 80)
        f = 15500 * gauss(lam, 430, 130)
        g = 13500 * gauss(lam, 375, 230)

        noise = np.random.normal(0.002, 5e-6,350) # ruido
        S = np.array([a, b, c, d,e, f, g]) * noise

        plt.plot(lam, S.T)
        plt.ylabel("Epsilon (u. a.)", size = "xx-large")
        plt.xlabel("$\lambda$ (nm)", size = "xx-large")
        plt.xticks(size = "large")
        plt.yticks(size = "large")
        plt.show()

        Y = C @ S

        plt.plot(lam, Y.T)
        plt.ylabel("Y observada (u. a.)", size = "xx-large")
        plt.xlabel("$\lambda$ (nm)", size = "xx-large")
        plt.xticks(size = "large")
        plt.yticks(size = "large")
        plt.show()

        nombres = [f"k{i}" for i in range(1, len(k)+1)]
        k_nombres = [f"{n}" for n in nombres]
        indice = lam.T
        ct_ind = ["H", "G"]

        modelo = pd.DataFrame(modelo)
        C = pd.DataFrame(C)
        Co = pd.DataFrame(Co)
        k = pd.DataFrame(k, index = [k_nombres])
        C_T = pd.DataFrame(C_T)
        Y = pd.DataFrame(Y.T, index = [list(indice)])