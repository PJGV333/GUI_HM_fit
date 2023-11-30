import wx
import sys
from wx import FileDialog
import wx.lib.scrolledpanel as scrolled
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('WXAgg')
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy import optimize
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, basinhopping
import sys
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
import warnings
warnings.filterwarnings("ignore")
import timeit

class TextRedirector:
    def __init__(self, text_ctrl):
        self.text_ctrl = text_ctrl

    def write(self, string):
        wx.CallAfter(self.text_ctrl.WriteText, string)

    def flush(self):
        pass

class CancelledByUserException(Exception):
    pass


class App(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, title="HM Fit", size=(800, 600))

        #self.panel = wx.Panel(self)
        # Utilizar ScrolledPanel
        self.panel = scrolled.ScrolledPanel(self)
        self.panel.SetupScrolling(scroll_x=True, scroll_y=True)

        self.vars_columnas = {} #lista para almacenar las columnas de la hoja de concentraciones
        self.figures = []  # Lista para almacenar figuras 
        self.current_figure_index = -1  # Índice inicial para navegación de figuras
       
        # Diseño usando Sizers
        self.main_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.left_sizer = wx.BoxSizer(wx.VERTICAL)
        self.right_sizer = wx.BoxSizer(wx.VERTICAL)

        # Añadir sizers al panel principal
        self.main_sizer.Add(self.left_sizer, 1, wx.EXPAND | wx.ALL, 5)
        self.main_sizer.Add(self.right_sizer, 1, wx.EXPAND | wx.ALL, 5)

        # Vincular el evento de redimensionamiento
        self.Bind(wx.EVT_SIZE, self.on_resize)

        
        tecnicas = ["Spectroscopy", "NMR", "Simulation"]
        self.choices_calctype = wx.Choice(self.panel, choices=tecnicas)
        self.left_sizer.Add(self.choices_calctype, 0, wx.ALL, 5)
        self.choices_calctype.SetSelection(0)  # Selecciona la primera opción por defect
       
        # Crear controles (botones, etiquetas, etc.) y añadirlos a left_sizer o right_sizer
        # Ejemplo:
        self.btn_select_file = wx.Button(self.panel, label="Select Excel File")
        self.btn_select_file.Bind(wx.EVT_BUTTON, self.select_file)
        self.left_sizer.Add(self.btn_select_file, 0, wx.ALL | wx.EXPAND, 5)

        # Crear un StaticText para mostrar la ruta del archivo
        self.lbl_file_path = wx.StaticText(self.panel, label="No file selected")
        self.left_sizer.Add(self.lbl_file_path, 0, wx.ALL | wx.EXPAND, 5)

        # Espectros
        self.sheet_spectra_panel, self.choice_sheet_spectra = self.create_sheet_dropdown_section("Spectra Sheet Name:")
        self.left_sizer.Add(self.sheet_spectra_panel, 0, wx.EXPAND | wx.ALL, 5)

        # Concentraciones
        self.sheet_conc_panel, self.choice_sheet_conc = self.create_sheet_dropdown_section("Concentration Sheet Name:")
        self.left_sizer.Add(self.sheet_conc_panel, 0, wx.EXPAND | wx.ALL, 5)

        # Sección para Checkboxes (inicialmente vacía)
        self.columns_names_panel = wx.Panel(self.panel)
        self.columns_names_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.lbl_columns = wx.StaticText(self.columns_names_panel, label="Column names: ")
        self.columns_names_sizer.Add(self.lbl_columns, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        self.columns_names_panel.SetSizer(self.columns_names_sizer)
        self.left_sizer.Add(self.columns_names_panel, 0, wx.EXPAND | wx.ALL, 5)

        # modelo
        self.sheet_model_panel, self.entry_sheet_model = self.create_sheet_dropdown_section("Model Sheet Name:")
        self.left_sizer.Add(self.sheet_model_panel, 0, wx.EXPAND | wx.ALL, 5)

        # Sección para Checkboxes (inicialmente vacía)
        self.sp_select_panel = wx.Panel(self.panel)
        self.sp_select_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # Autovalores
        self.sheet_EV_panel, self.entry_EV = self.create_sheet_section("Eigenvalues:", "0")

        # Crear el Checkbox para EFA y añadirlo al sizer de la sección "Eigenvalues"
        self.EFA_cb = wx.CheckBox(self.sheet_EV_panel, label='EFA')
        self.EFA_cb.SetValue(True)  # Marcar el checkbox por defecto
        self.sheet_EV_panel.GetSizer().Insert(0, self.EFA_cb, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)

        # Añadir la sección "Eigenvalues" con el Checkbox al left_sizer
        self.left_sizer.Add(self.sheet_EV_panel, 0, wx.ALL, 5)   

        self.sp_columns = wx.StaticText(self.sp_select_panel, label="Select non-absorbent species: ")
        self.sp_select_sizer.Add(self.sp_columns, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        self.sp_select_panel.SetSizer(self.sp_select_sizer)
        self.left_sizer.Add(self.sp_select_panel, 0, wx.EXPAND | wx.ALL, 5)

        # Panel y sizer para el modelo
        self.model_panel = wx.Panel(self.panel)
        self.model_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # Crear el wx.ListCtrl para mostrar el modelo
        self.model_list_ctrl = wx.ListCtrl(self.model_panel, style=wx.LC_REPORT | wx.BORDER_SUNKEN)
        self.model_sizer.Add(self.model_list_ctrl, 1, wx.ALL | wx.EXPAND, 5)

        # Configurar sizer y añadir al panel principal
        self.model_panel.SetSizer(self.model_sizer)

        # Crear un Sizer vertical para el panel del modelo y añadirlo
        model_vertical_sizer = wx.BoxSizer(wx.VERTICAL)
        model_vertical_sizer.Add(self.model_panel, 1, wx.EXPAND | wx.ALL, 5)
        

        # Sizer horizontal para alinear los paneles de ajustes
        #ajustes_optimizadores_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # Crear un Sizer horizontal para alinear el panel del modelo y los paneles de ajustes
        main_horizontal_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # Añadir el Sizer vertical del modelo al Sizer horizontal principal
        main_horizontal_sizer.Add(model_vertical_sizer, 1, wx.EXPAND | wx.ALL, 5)

        # Sizer vertical para los paneles de ajustes
        ajustes_vertical_sizer = wx.BoxSizer(wx.VERTICAL)

        # Panel y sizer para el algoritmo
        algo_panel = wx.Panel(self.panel)
        algo_sizer = wx.BoxSizer(wx.VERTICAL)
        algo_label = wx.StaticText(algo_panel, label='Algorithm for C')
        algo_sizer.Add(algo_label, 0, wx.ALL, 5)
        algortimo = ["Newton-Raphson", "Levenberg-Marquardt"]
        self.choice_algoritm = wx.Choice(algo_panel, choices=algortimo)
        algo_sizer.Add(self.choice_algoritm, 0, wx.ALL, 5)
        self.choice_algoritm.SetSelection(0)
        algo_panel.SetSizer(algo_sizer)

        # Panel y sizer para los ajustes de modelo
        ajustes_panel = wx.Panel(self.panel)
        ajustes_sizer = wx.BoxSizer(wx.VERTICAL)
        ajustes_label = wx.StaticText(ajustes_panel, label='Model settings')
        ajustes_sizer.Add(ajustes_label, 0, wx.ALL, 5)
        ajustes_modelo_choices = ["Free", "Step by step", "Non-cooperative"]
        self.choice_model_settings = wx.Choice(ajustes_panel, choices=ajustes_modelo_choices)
        ajustes_sizer.Add(self.choice_model_settings, 0, wx.ALL, 5)
        self.choice_model_settings.SetSelection(0)
        ajustes_panel.SetSizer(ajustes_sizer)

        # Panel y sizer para los ajustes del optimizador
        optimizador_panel = wx.Panel(self.panel)
        optimizador_sizer = wx.BoxSizer(wx.VERTICAL)
        optimizador_label = wx.StaticText(optimizador_panel, label='Optimizer')
        optimizador_sizer.Add(optimizador_label, 0, wx.ALL, 5)
        optimizador_choices = ["powell", "nelder-mead", "trust-constr", "cg", "bfgs", "l-bfgs-b", "tnc", "cobyla", "slsqp"]
        self.choice_optimizer_settings = wx.Choice(optimizador_panel, choices=optimizador_choices)
        optimizador_sizer.Add(self.choice_optimizer_settings, 0, wx.ALL, 5)
        self.choice_optimizer_settings.SetSelection(0)
        optimizador_panel.SetSizer(optimizador_sizer)

        # Añadir paneles de algoritmo, ajustes de modelo y ajustes del optimizador
        ajustes_vertical_sizer.Add(algo_panel, 1, wx.EXPAND | wx.ALL, 5)
        ajustes_vertical_sizer.Add(ajustes_panel, 1, wx.EXPAND | wx.ALL, 5)
        ajustes_vertical_sizer.Add(optimizador_panel, 1, wx.EXPAND | wx.ALL, 5)

        # Añadir el Sizer vertical de ajustes al Sizer horizontal principal
        main_horizontal_sizer.Add(ajustes_vertical_sizer, 1, wx.EXPAND | wx.ALL, 5)

        # Añadir el Sizer horizontal principal al left_sizer
        self.left_sizer.Add(main_horizontal_sizer, 1, wx.EXPAND | wx.ALL, 5)

        # Enlazar eventos de selección, si es necesario
        self.choice_model_settings.Bind(wx.EVT_CHOICE, self.on_model_settings_selected)
        self.choice_optimizer_settings.Bind(wx.EVT_CHOICE, self.on_optimizer_settings_selected)
        self.choices_calctype.Bind(wx.EVT_CHOICE, self.choice_type_calc)
        self.choice_algoritm.Bind(wx.EVT_CHOICE, self.choice_algoritm_type)

        self.bounds_panel = wx.Panel(self.panel)
        self.bounds_sizer = wx.FlexGridSizer(0, 3, 10, 10)  # Filas, Columnas, Espacio horizontal, Espacio vertical
        self.bounds_panel.SetSizer(self.bounds_sizer)

         # Crear el ListCtrl para los límites de parámetros
        self.param_limits_list_ctrl = wx.ListCtrl(self.panel, style=wx.LC_REPORT | wx.LC_EDIT_LABELS)
        self.param_limits_list_ctrl.InsertColumn(0, "Parámetro", wx.LIST_FORMAT_LEFT, width=100)
        self.param_limits_list_ctrl.InsertColumn(1, "Mínimo", wx.LIST_FORMAT_LEFT, width=100)
        self.param_limits_list_ctrl.InsertColumn(2, "Máximo", wx.LIST_FORMAT_LEFT, width=100)

        # Añadir el ListCtrl al sizer
        self.left_sizer.Add(self.param_limits_list_ctrl, 0, wx.ALL, 5)
    ##############################################################################################################
        """ Panel derecho del gui """

        # Crear un SplitterWindow en el panel derecho
        right_splitter = wx.SplitterWindow(self.panel)

        # Crear dos paneles para el splitter en el panel derecho
        canvas_panel = wx.Panel(right_splitter)
        console_panel = wx.Panel(right_splitter)

        # Sizer para el panel del canvas
        canvas_sizer = wx.BoxSizer(wx.VERTICAL)

        # Crear el canvas de gráficas en el canvas_panel
        self.fig = Figure()
        self.canvas = FigureCanvas(canvas_panel, -1, self.fig)
        canvas_sizer.Add(self.canvas, 1, wx.EXPAND | wx.ALL)
        canvas_panel.SetSizer(canvas_sizer)

        # Sizer para los botones
        buttons_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # Añadir un espaciador flexible al inicio
        buttons_sizer.AddStretchSpacer()

        # Botón "Prev"
        self.btn_prev_figure = wx.Button(self.panel, label="<< Prev")
        buttons_sizer.Add(self.btn_prev_figure, 0, wx.ALL, 5)
        self.btn_prev_figure.Bind(wx.EVT_BUTTON, self.show_prev_figure)

        # Añadir un espaciador flexible entre los botones
        buttons_sizer.AddStretchSpacer()

        # Botón "Process Data"
        self.btn_process_data = wx.Button(self.panel, label="Process Data")
        buttons_sizer.Add(self.btn_process_data, 0, wx.ALL | wx.LEFT, 5)
        self.btn_process_data.Bind(wx.EVT_BUTTON, self.process_data)

        # Añadir un espaciador flexible entre los botones
        buttons_sizer.AddStretchSpacer()

        # Botón "Next"
        self.btn_next_figure = wx.Button(self.panel, label="Next >>")
        buttons_sizer.Add(self.btn_next_figure, 0, wx.ALL | wx.LEFT, 5)
        self.btn_next_figure.Bind(wx.EVT_BUTTON, self.show_next_figure)

        # Añadir un espaciador flexible al final
        buttons_sizer.AddStretchSpacer()

        # Añadir el sizer de los botones al sizer principal del lado derecho
        self.right_sizer.Add(buttons_sizer, 0, wx.EXPAND)

        # Sizer para el panel de la consola
        console_sizer = wx.BoxSizer(wx.VERTICAL)

        # Crear la consola en el console_panel
        self.console = wx.TextCtrl(console_panel, style=wx.TE_MULTILINE | wx.TE_READONLY)
        self.console.SetFont(wx.Font(9, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))

        # Configuración de colores para la consola
        self.console.SetBackgroundColour(wx.BLACK)  # Fondo negro
        self.console.SetForegroundColour(wx.WHITE)  # Texto blanco

        console_sizer.Add(self.console, 2, wx.EXPAND | wx.ALL)
        console_panel.SetSizer(console_sizer)

        # Dividir el splitter entre los dos paneles del panel derecho
        right_splitter.SplitHorizontally(canvas_panel, console_panel)
        right_splitter.SetMinimumPaneSize(20)  # Tamaño mínimo de los paneles

        # Añadir el splitter al sizer del panel derecho
        self.right_sizer.Add(right_splitter, 1, wx.EXPAND)
        
        # Establecer la posición inicial del divisor (sash)
        # El valor específico dependerá del tamaño deseado de tus paneles
        right_splitter.SetSashPosition(700)  # Ejemplo con 200 píxeles
   
        # Redirigir stdout
        sys.stdout = TextRedirector(self.console)
        
        # Botón guardar resultados
        save_results_sizer = wx.BoxSizer(wx.HORIZONTAL)
        save_results_sizer.AddStretchSpacer()

        # Botón "Save Results"
        self.btn_save_results = wx.Button(self.panel, label="Save Results")
        save_results_sizer.Add(self.btn_save_results, 0, wx.ALL, 5)
        self.btn_save_results.Bind(wx.EVT_BUTTON, self.save_results)

        # Botón "Reset Calculation"
        reset_button = wx.Button(self.panel, label="Reset Calculation")
        reset_button.Bind(wx.EVT_BUTTON, lambda event: self.reset_calculation())
        save_results_sizer.Add(reset_button, 0, wx.ALL, 5)

        save_results_sizer.AddStretchSpacer()

        # Añadir el sizer al sizer principal del lado derecho
        self.right_sizer.Add(save_results_sizer, 0, wx.EXPAND)

        # Método de controlador de eventos para el botón irá aquí     

        self.panel.SetSizer(self.main_sizer)
        self.main_sizer.Layout()

    ####################################################################################################################

    #Redimensionar el tamaño de la ventana al 50% por cada panel.
    def on_resize(self, event):
        # Llama a este método para permitir que el evento de redimensionamiento continúe
        event.Skip()

        # Obtener el tamaño actual del frame
        frame_size = self.GetSize()

        # Calcular el nuevo ancho para los paneles
        new_width = frame_size[0] // 2

        # Ajustar los tamaños de los sizers
        self.left_sizer.SetMinSize((new_width, -1))
        self.right_sizer.SetMinSize((new_width, -1))

        # Actualizar el layout
        self.panel.Layout()


    # Definir tipo de calculo
    def choice_type_calc(self, event):
        select_calc = self.choices_calctype.GetStringSelection()
        return select_calc
    
    # Definir tipo de calculo
    def choice_algoritm_type(self, event):
        select_algo = self.choice_algoritm.GetStringSelection()
        return select_algo
    
    # Definir los controladores de eventos
    def on_model_settings_selected(self, event):
        selected_model = self.choice_model_settings.GetStringSelection()
        # Aquí puedes agregar la lógica para manejar la selección del modelo
        return selected_model

    def on_optimizer_settings_selected(self, event):
        selected_optimizer = self.choice_optimizer_settings.GetStringSelection()
        # Aquí puedes agregar la lógica para manejar la selección del optimizador
        return selected_optimizer    

    def create_sheet_section(self, label_text, default_value):
            # Crear un panel para esta sección
            panel = wx.Panel(self.panel)
            sizer = wx.BoxSizer(wx.HORIZONTAL)

            # Crear y añadir una etiqueta
            label = wx.StaticText(panel, label=label_text)
            sizer.Add(label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)

            # Crear y añadir un campo de texto
            text_ctrl = wx.TextCtrl(panel)
            text_ctrl.SetValue(default_value)
            sizer.Add(text_ctrl, 1, wx.ALL | wx.EXPAND, 5)

            # Configurar el sizer del panel y devolver el panel y el control de texto
            panel.SetSizer(sizer)
            return panel, text_ctrl

    def select_file(self, event):
        with FileDialog(self, "Select Excel file", wildcard="Excel files (*.xlsx)|*.xlsx|All files (*.*)|*.*",
                        style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return  # El usuario canceló la selección

            file_path = fileDialog.GetPath()
            self.lbl_file_path.SetLabel(file_path)
            self.file_path = file_path
            self.populate_sheet_choices(file_path)
    
    def create_sheet_dropdown_section(self, label_text):
        panel = wx.Panel(self.panel)
        sizer = wx.BoxSizer(wx.HORIZONTAL)

        # Crear y añadir la etiqueta
        label = wx.StaticText(panel, label=label_text)
        sizer.Add(label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)

        # Crear y añadir el menú desplegable
        choice = wx.Choice(panel)
        sizer.Add(choice, 1, wx.ALL | wx.EXPAND, 5)

        panel.SetSizer(sizer)
        return panel, choice
    
    def populate_sheet_choices(self, file_path):
        try:
            sheet_names = pd.ExcelFile(file_path).sheet_names
            self.choice_sheet_spectra.SetItems(sheet_names)
            #self.choice_sheet_spectra.SetSelection(0)  # Seleccionar la primera hoja por defecto

            self.choice_sheet_conc.SetItems(sheet_names)
            #self.choice_sheet_conc.SetSelection(0)  # Seleccionar la primera hoja por defecto
            self.choice_sheet_conc.Bind(wx.EVT_CHOICE, self.on_conc_sheet_selected)

            self.entry_sheet_model.SetItems(sheet_names)
            #self.choice_sheet_model.SetSelection(0)  # Seleccionar la primera hoja por defecto
            self.entry_sheet_model.Bind(wx.EVT_CHOICE, self.on_model_sheet_selected)

        except Exception as e:
            wx.MessageBox(f"Error al leer el archivo de Excel: {e}", "Error en archivo de Excel", wx.OK | wx.ICON_ERROR)

    def on_conc_sheet_selected(self, event):
        selected_sheet = self.choice_sheet_conc.GetStringSelection()
        try:
            df = pd.read_excel(self.file_path, sheet_name=selected_sheet)
            self.create_checkboxes(df.columns)
        except Exception as e:
            wx.MessageBox(f"Error al leer la hoja de Excel: {e}", "Error en la hoja de Excel", wx.OK | wx.ICON_ERROR)

    def on_model_sheet_selected(self, event):
        selected_sheet = self.entry_sheet_model.GetStringSelection()
        try:
            self.load_model_from_sheet(selected_sheet)
        except Exception as e:
            wx.MessageBox(f"Error al leer la hoja de Excel: {e}", "Error en la hoja de Excel", wx.OK | wx.ICON_ERROR)


    def save_results(self, event):
        # Usar FileDialog de wxPython para seleccionar el archivo donde guardar
        with wx.FileDialog(self, "Save Excel file", wildcard="Excel files (*.xlsx)|*.xlsx",
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return     # El usuario canceló la selección

            file_path = fileDialog.GetPath()

            # Asegúrate de que el file_path tiene la extensión '.xlsx'
            if not file_path.endswith('.xlsx'):
                file_path += '.xlsx'

            # Aquí va la lógica de guardado de tus datos
            with pd.ExcelWriter(file_path) as writer:
                if hasattr(self, 'modelo'):
                    self.modelo.to_excel(writer, sheet_name="Model")
                if hasattr(self, 'C'):
                    self.C.to_excel(writer, sheet_name="Absorbent_species")
                if hasattr(self, 'Co'):
                    self.Co.to_excel(writer, sheet_name="All_species")   
                if hasattr(self, 'concentracion'):
                    self.concentracion.to_excel(writer, sheet_name="Tot_con_comp")  
                if hasattr(self, 'A'):
                    self.A.to_excel(writer, sheet_name="Molar_Absortivities", index_label = 'nm', index = True)
                if hasattr(self, 'k'):
                    self.k.to_excel(writer, sheet_name="K_calculated")
                if hasattr(self, 'k_ini'):
                    self.k_ini.to_excel(writer, sheet_name="Init_guess_K")
                if hasattr(self, 'phi'):
                    self.phi.to_excel(writer, sheet_name="Y_calculated", index_label = 'nm', index = True)
                if hasattr(self, 'Y'):
                    self.Y.to_excel(writer, sheet_name="Y_observed", index_label = 'nm', index = True)
                if hasattr(self, 'stats'):
                    self.stats.to_excel(writer, sheet_name="Stats")

            # Mostrar un mensaje al finalizar el guardado
            wx.MessageBox(f"Results saved to {file_path}.", "Information", wx.OK | wx.ICON_INFORMATION)
    
    def create_checkboxes(self, column_names):
        # Convertir WindowList a una lista regular de Python y limpiar checkboxes antiguos
        children = list(self.columns_names_panel.GetChildren())
        for child in children[1:]:
            child.Destroy()

        # Asegúrate de que self.vars_columnas está vacío antes de empezar a añadir nuevos checkboxes
        self.vars_columnas = {}

        # Crear los checkboxes dentro del panel y añadirlos a self.vars_columnas
        for col in column_names:
            checkbox = wx.CheckBox(self.columns_names_panel, label=col)
            checkbox.Bind(wx.EVT_CHECKBOX, self.on_checkbox_select)
            self.columns_names_sizer.Add(checkbox, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
            self.vars_columnas[col] = checkbox  # Añadir cada checkbox al diccionario

        # Reorganizar los controles en el panel
        self.columns_names_panel.Layout()

    def on_checkbox_select(self, event):
        cb = event.GetEventObject()
        label = cb.GetLabel()
        if cb.IsChecked():
            print(f"Selected: {label}")
            # Añadir a la lista de seleccionados, o realizar otra acción
        else:
            print(f"Deselected: {label}")
            # Eliminar de la lista de seleccionados, o realizar otra acción

    def on_model_checkbox_select(self, event):
        # Este método se llama cuando se marca o desmarca un checkbox
        checkbox = event.GetEventObject()
        label = checkbox.GetLabel()
        if checkbox.IsChecked():
            print(f"Model row {label} está seleccionado.")
        else:
            print(f"Model row {label} no está seleccionado.")

  
    def figura(self, x, y, mark, ylabel, xlabel, title):
        fig = Figure(figsize=(4, 4), dpi=200)
        ax = fig.add_subplot(111)
        ax.plot(x, y, mark)
        ax.set_ylabel(ylabel, size="xx-large")
        ax.set_xlabel(xlabel, size="xx-large")
        ax.tick_params(axis='both', which='major', labelsize='large')
        self.figures.append(fig)  # Almacenar tanto fig como ax
        #self.add_figure_to_listbox(title)  # Añade título a la listbox
        #print("Figura añadida. Total de figuras:", len(self.figures))
        self.current_figure_index = len(self.figures) - 1
        self.update_canvas_figure(fig)

    def figura2(self, x, y, y2, mark1, mark2, ylabel, xlabel, alpha, title):
        fig = Figure(figsize=(4, 4), dpi=200)
        ax = fig.add_subplot(111)
        ax.plot(x, y, mark1, alpha=alpha)
        ax.plot(x, y2, mark2)
        ax.set_ylabel(ylabel, size="xx-large")
        ax.set_xlabel(xlabel, size="xx-large")
        ax.tick_params(axis='both', which='major', labelsize='large')
        self.figures.append(fig)  # Almacenar tanto fig como ax
        #self.add_figure_to_listbox(title)  # Añade título a la listbox
        self.current_figure_index = len(self.figures) - 1
        #print("Figura añadida. Total de figuras:", len(self.figures))
        self.update_canvas_figure(fig)
    
    def on_grafica_selected(self, event):
        seleccionado = self.graficas_listbox.curselection()
        if seleccionado:
            index = seleccionado[0]
            self.show_figure(index)

    def show_next_figure(self, event):
        if self.figures:
            self.current_figure_index = (self.current_figure_index + 1) % len(self.figures)
            #print("Mostrando figura:", self.current_figure_index + 1, "de", len(self.figures))
            self.update_canvas_figure(self.figures[self.current_figure_index])

    def show_prev_figure(self, event):
        if self.figures:
            self.current_figure_index = (self.current_figure_index - 1) % len(self.figures)
            #print("Mostrando figura:", self.current_figure_index + 1, "de", len(self.figures))
            self.update_canvas_figure(self.figures[self.current_figure_index])

    def update_canvas_figure(self, new_figure):
        current_axes = self.canvas.figure.gca()
        current_axes.clear()

        new_axes = new_figure.gca()
        current_axes._sharex = new_axes._sharex
        current_axes._sharey = new_axes._sharey

        for line in new_axes.get_lines():
            # Copiar datos del plot
            new_line, = current_axes.plot(line.get_xdata(), line.get_ydata(), marker=line.get_marker(), color=line.get_color())

            # Copiar estilos adicionales si están disponibles
            if line.get_linestyle() is not None:
                new_line.set_linestyle(line.get_linestyle())
            if line.get_alpha() is not None:
                new_line.set_alpha(line.get_alpha())
            # Agrega aquí más propiedades si es necesario

        current_axes.set_xlabel(new_axes.get_xlabel())
        current_axes.set_ylabel(new_axes.get_ylabel())
        current_axes.set_title(new_axes.get_title())

        self.canvas.draw()

    def show_figure(self, index):
        # Obtener la figura de la lista
        fig = self.figures[index]
        self.update_canvas_figure(fig)
    
    #Función para añadir los valores deseados para crear el modelo. 
    def ask_integer(self, message):
        dialog = wx.TextEntryDialog(self, message, "Input")
        if dialog.ShowModal() == wx.ID_OK:
            try:
                return int(dialog.GetValue())
            except ValueError:
                wx.MessageBox("Por favor, ingrese un número entero.", "Error", wx.OK | wx.ICON_ERROR)
        else:
            raise CancelledByUserException("La entrada fue cancelada por el usuario.")
        dialog.Destroy()
        #return self.reset_calculation() #None  # O manejar de otra manera si se cancela o ingresa un valor no vaiido
    
    #Función para añadir las constantes de asociación como floats.
    def ask_float(self, title, message):
        dialog = wx.TextEntryDialog(self, message, title)
        if dialog.ShowModal() == wx.ID_OK:
            try:
                return float(dialog.GetValue())
            except ValueError:
                wx.MessageBox("Por favor, ingrese un número válido.", "Error", wx.OK | wx.ICON_ERROR)
        else:
            raise CancelledByUserException("La entrada fue cancelada por el usuario.") #break  # Salir del bucle si el usuario cancela
        dialog.Destroy()
        #return self.reset_calculation()
        
    #Función para mostrar el modelo en el panel destinado para ello. 
    def load_model_from_sheet(self, sheet_name):
        df = pd.read_excel(self.file_path, sheet_name=sheet_name)
        df_transposed = df.T  # Transponer el DataFrame

        # Asegurarse de que el ListCtrl está limpio antes de añadir nuevos datos
        self.model_list_ctrl.DeleteAllItems()
        self.model_list_ctrl.ClearAll()

        # Crear el wx.ListCtrl para mostrar el modelo sin encabezados
        if hasattr(self, 'model_list_ctrl'):  # Verifica si el control ya existe
            self.model_list_ctrl.Destroy()  # Si es así, destrúyelo

        # Crea el ListCtrl con el estilo de no encabezado
        self.model_list_ctrl = wx.ListCtrl(self.model_panel, style=wx.LC_REPORT | wx.LC_NO_HEADER | wx.BORDER_SUNKEN)
        self.model_sizer.Add(self.model_list_ctrl, 1, wx.ALL | wx.EXPAND, 5)

         # Vincula los manejadores de eventos después de crear el ListCtrl
        self.model_list_ctrl.Bind(wx.EVT_LIST_ITEM_SELECTED, self.on_selection_changed)
        self.model_list_ctrl.Bind(wx.EVT_LIST_ITEM_DESELECTED, self.on_selection_changed)

        # Agregar columnas al ListCtrl
        for col_num in range(len(df_transposed.columns)):
            self.model_list_ctrl.InsertColumn(col_num, '')

       # Agregar filas al ListCtrl
        for i, row in df_transposed.iterrows():
            index = self.model_list_ctrl.InsertItem(self.model_list_ctrl.GetItemCount(), '')
            for j, value in enumerate(row):
                self.model_list_ctrl.SetItem(index, j, str(value))
            self.model_list_ctrl.SetColumnWidth(j, wx.LIST_AUTOSIZE)  # Ajusta el ancho de la columna

        # Actualizar el layout de los sizers
        self.model_panel.Layout()
        self.left_sizer.Layout()
    
    def load_generated_model(self, modelo):
        # Asegurarse de que el ListCtrl está limpio antes de añadir nuevos datos
        self.model_list_ctrl.DeleteAllItems()
        self.model_list_ctrl.ClearAll()

        # Crear el wx.ListCtrl para mostrar el modelo
        if not hasattr(self, 'model_list_ctrl'):
            self.model_list_ctrl = wx.ListCtrl(self.model_panel, style=wx.LC_REPORT | wx.LC_NO_HEADER | wx.BORDER_SUNKEN)
            self.model_sizer.Add(self.model_list_ctrl, 1, wx.ALL | wx.EXPAND, 5)

            # Vincula los manejadores de eventos después de crear el ListCtrl
            self.model_list_ctrl.Bind(wx.EVT_LIST_ITEM_SELECTED, self.on_selection_changed2)
            self.model_list_ctrl.Bind(wx.EVT_LIST_ITEM_DESELECTED, self.on_selection_changed2)

        # Agregar columnas al ListCtrl con nombres de las columnas seleccionadas
        selected_columns = [col for col in self.vars_columnas if self.vars_columnas[col].GetValue()]
        for col_name in selected_columns:
            self.model_list_ctrl.InsertColumn(selected_columns.index(col_name), col_name)

        # Agregar filas al ListCtrl
        num_rows = modelo.shape[1]
        for row_idx in range(num_rows):
            index = self.model_list_ctrl.InsertItem(self.model_list_ctrl.GetItemCount(), '')
            for col_idx, col_name in enumerate(selected_columns):
                self.model_list_ctrl.SetItem(index, col_idx, str(modelo[col_idx, row_idx]))
            self.model_list_ctrl.SetColumnWidth(col_idx, wx.LIST_AUTOSIZE)

        # Actualizar el layout de los sizers
        self.model_panel.Layout()
        self.left_sizer.Layout()

    # Funciones para seleccionar filas del modelo y transponer sus posiciones.
    def get_selected_rows(self):
        selection = []
        index = -1
        while True:
            index = self.model_list_ctrl.GetNextSelected(index)
            if index == -1:
                break
            selection.append(index)
        return selection
    
    def on_selection_changed(self, event):
        selected_rows = self.get_selected_rows()        
        df2 = pd.read_excel(self.file_path, sheet_name=self.entry_sheet_model.GetStringSelection())

        # Restar 1 de cada índice seleccionado para compensar el encabezado
        adjusted_selected_rows = [row - 1 for row in selected_rows]

        # Convertir nombres de columnas seleccionados a índices numéricos
        selected_indices = [df2.columns.get_loc(df2.columns[row]) for row in adjusted_selected_rows]
        print("Columnas seleccionadas:", selected_indices)
        return selected_indices
    
    def ask_indices(self, message):
        dialog = wx.TextEntryDialog(self, message, "Input")
        if dialog.ShowModal() == wx.ID_OK:
            input_text = dialog.GetValue()
            dialog.Destroy()
            try:
                # Convertir el texto ingresado en una lista de enteros
                if input_text == "":
                    return []
                else:
                    indices = [int(x.strip()) for x in input_text.split(',')]
                    return indices
            except ValueError:
                wx.MessageBox("Por favor, ingrese una lista válida de índices (e.g., '1, 2, 3').", "Error", wx.OK | wx.ICON_ERROR)
                return self.reset_calculation()  # O manejar de otra manera
        else:
            dialog.Destroy()
            return self.reset_calculation()  # Manejo cuando el usuario cancela


    def reset_calculation(self):
        # Reiniciar la ruta del archivo y la etiqueta correspondiente
        self.file_path = None
        self.lbl_file_path.SetLabel("No file selected")

        # Limpiar y reiniciar el DataFrame
        self.df = None

        # Limpiar el ListCtrl (si lo estás utilizando)
        if hasattr(self, 'model_list_ctrl'):
            self.model_list_ctrl.DeleteAllItems()
            self.model_list_ctrl.ClearAll()

        # Limpiar la lista de figuras
        if hasattr(self, 'figures'):
            self.figures.clear()
            # Si estás utilizando un canvas para mostrar las figuras, también debes limpiarlo
            if hasattr(self, 'canvas'):
                self.canvas.figure.clear()
                self.canvas.draw()

        # Aquí puedes agregar cualquier otro reinicio necesario, como limpiar campos de texto,
        # restablecer variables de estado, etc.

        # Finalmente, actualiza el layout si es necesario
        self.Layout()

    def res_consola(self, prefijo, resp):
        # Actualiza la GUI con los resultados de r_0
        # Por ejemplo, mostrar los resultados en un wx.TextCtrl
        self.console.AppendText(f"{prefijo}: {resp}\n")

    def create_titulant_dropdown(self):
        self.titulant_panel = wx.Panel(self.panel)
        titulant_sizer = wx.BoxSizer(wx.HORIZONTAL)

        titulant_label = wx.StaticText(self.titulant_panel, label="Select Titulant:")
        titulant_sizer.Add(titulant_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)

        self.choice_titulant = wx.Choice(self.titulant_panel)
        titulant_sizer.Add(self.choice_titulant, 1, wx.ALL | wx.EXPAND, 5)

        self.titulant_panel.SetSizer(titulant_sizer)
        self.left_sizer.Add(self.titulant_panel, 0, wx.EXPAND | wx.ALL, 5)

    def update_titulant_choices(self, column_names):
        self.choice_titulant.Clear()
        self.choice_titulant.SetItems(column_names)
        if column_names:
            self.choice_titulant.SetSelection(0)

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
        G = C_T[:,1]
        H = C_T[:,0]
        nc = len(C_T)
        n_comp = len(C_T.T)
        nw = len(spec)
        nm = spec.index.to_numpy()
        
        def SVD_EFA(spec, args = (nc)):
            u, s, v = np.linalg.svd(spec, full_matrices=False)
            
            #EFA fijo
            
            L = range(1,(nc + 1), 1)
            L2 = range(0, nc, 1)
            
            X = []
            for i in L:
                uj, sj, vj = np.linalg.svd(spec.T.iloc[:i,:], full_matrices=False)
                X.append(sj)
            
            ev_s = pd.DataFrame(X)
            ev_s0 = np.array(ev_s)
            
            X2 = []
            for i in L2:
                ui, si, vi = np.linalg.svd(spec.T.iloc[i:,:], full_matrices=False)
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
        
        try:
            modelo = pd.read_excel(self.file_path, self.entry_sheet_model.GetStringSelection(), header=0, index_col=0)
            
            #print(modelo)
            nas = self.on_selection_changed(event)
            
        except:
            try:
                m = self.ask_integer("Indique el coeficiente estequiométrico para el receptor:")
                n = self.ask_integer("Indique el coeficiente estequiométrico para el huesped:")
                # Continuar con el resto de tu lógica si las entradas son válidas
                #coef = np.array([m, n])
                def combinaciones(m, n):
                    combinaciones = []
                    for i in range(0, m + 1):
                        for j in range(0, n + 1):
                            if i == 0 and j > 1:
                                pass
                            elif i > 1 and j == 0:
                                pass
                            elif j == 1 and i == 0:
                                combinaciones.append([j, i])
                            elif i == 1 and j == 0:
                                combinaciones.append([j, i])
                            elif i + j != 0:
                                combinaciones.append([i, j])
                    return np.array(combinaciones).T
                
                modelo = combinaciones(m, n)
                self.load_generated_model(modelo)
                nas = self.ask_indices("Índice de especies no absorbentes (e.g., '1, 4' o dejar en blanco para []):")
                #print(modelo)

            except CancelledByUserException as e:
                # Imprimir el mensaje de excepción, que será redirigido a self.console
                print(str(e))
                return
            
        
        modelo = np.array(modelo)

        work_algo = self.choice_algoritm.GetStringSelection()
        model_sett = self.choice_model_settings.GetStringSelection()
        
        if model_sett == "Non-cooperative":
            n_K = len(modelo.T) - n_comp - 1
        
        else:
            n_K = len(modelo.T) - n_comp
        
        if n_K == 1:
            try:
                k_e = self.ask_float("K", "Indique un valor estimado para la constante de asociación:")
                k = np.array([k_e])
                k_ini = k
            except CancelledByUserException as e:
                # Imprimir el mensaje de excepción, que será redirigido a self.console
                print(str(e))
                return
        else:
            try:
                k_e = [] #[1., 1.]
                for i in range(n_K):
                    ks = "K" + str(i+1) 
                    i = self.ask_float(ks, "Indique un valor estimado para esta constante de asociación:")
                    print(ks + ":", i)
                    k_e.append(i)
            
                k = np.array([k_e])
                k_ini = k
                k = np.ravel(k)
            except CancelledByUserException as e:
                # Imprimir el mensaje de excepción, que será redirigido a self.console
                print(str(e))
                return



        if work_algo == "Newton-Raphson":
            
            from NR_conc_algoritm import NewtonRaphson

            res = NewtonRaphson(C_T, modelo, nas, model_sett)

        elif work_algo == "Levenberg-Marquardt": 
            
            from LM_conc_algoritm import LevenbergMarquardt

            res = LevenbergMarquardt(C_T, modelo, nas, model_sett) 


        # Implementing the abortividades function
        def abortividades(k, Y):
            C, Co = res.concentraciones(k)  # Assuming the function concentraciones returns C and Co
            A = np.linalg.pinv(C) @ Y.T
            return np.all(A >= 0)
        
        def f_m2(k):
            C = res.concentraciones(k)[0]    
            r = C @ np.linalg.pinv(C) @ Y.T - Y.T
            rms = np.sqrt(np.mean(np.square(r)))
            #print(f"f(x): {rms}")
            #print(f"x: {k}")
            return rms, r
            
        def f_m(k):
            C = res.concentraciones(k)[0]    
            r = C @ np.linalg.pinv(C) @ Y.T - Y.T
            rms = np.sqrt(np.mean(np.square(r)))
            self.res_consola("f(x)", rms)
            self.res_consola("x", k)
            
            # Procesar eventos de la GUI para actualizar la consola en tiempo real
            wx.Yield()
            return rms
        
                
        #bounds = [(-20, 20)]*len(k.T) #Bounds(0, 1e15, keep_feasible=(True)) #
        
        # Registrar el tiempo de inicio
        inicio = timeit.default_timer()
        
        optimizer = self.choice_optimizer_settings.GetStringSelection()
        print(optimizer)
        r_0 = optimize.minimize(f_m, k, method=optimizer)
                      
        # Registrar el tiempo de finalización
        fin = timeit.default_timer()
        
        # Calcular el tiempo total de ejecución
        tiempo_total = fin - inicio
        
        print("El tiempo de ejecución de la función fue: ", tiempo_total, "segundos.")
        
        k = r_0.x 
        k = np.ravel(k)
        
        # Calcular el SER
        
        n = len(G)
        p = len(k)
        
        SER = f_m(k)
        
        # Calcular la matriz jacobiana de los residuos
        def residuals(k):
            C = res.concentraciones(k)[0]
            r = C @ np.linalg.pinv(C) @ Y.T - Y.T
            return r.flatten()
                
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
        cov_matrix = SER**2 * np.linalg.pinv(jacobian @ jacobian.T)
        
        # Calcular el error estándar de las constantes de asociación
        SE_k = np.sqrt(np.diag(cov_matrix))
        
        # Calcular el error porcentual
        error_percent = (SE_k / np.abs(k)) #* 100
        
        C, Co = res.concentraciones(k)
        
        self.figura(G/np.max(H), C, ":o", "[Especies], M", "[G]/[H], M", "Perfil de concentraciones")
                
        Q, R = np.linalg.qr(C)
        y_cal = Q @ Q.T @ Y.T
        #y_cal = C @ np.linalg.pinv(C, rcond=1e-20) @ Y.T
                        
        ssq, r0 = f_m2(k)
        rms = f_m(k)
        
        A = np.linalg.pinv(C) @ Y.T 
        
        if not self.EFA_cb.GetValue():
            self.figura2(G, Y.T, y_cal, "ko", ":", "Y observada (u. a.)", "[X], M", 1, "Ajuste")
        else:
            self.figura(nm, A.T, "-", "Epsilon (u. a.)", "$\lambda$ (nm)", "Absortividades molares")
            self.figura2(nm, Y, y_cal.T, "-k", "k:", "Y observada (u. a.)", "$\lambda$ (nm)", 0.5, "Ajuste")   

        lof = (((sum(sum((r0**2))) / sum(sum((Y**2)))))**0.5) * 100
        #MAE = np.sqrt((sum(sum(r0**2)) / (nw - len(k))))
        MAE = np.mean(abs(r0))
        dif_en_ct = round(max(100 - (np.sum(C, 1) * 100 / max(H))), 2)
        
        # 1. Calcular la varianza de los residuales
        residuals_array = residuals(k)
        var_residuals = np.var(residuals_array)
        
        # 2. Calcular la varianza de los datos experimentales
        var_data_original = np.var(Y)
        
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
        headers = ["Constant", "log10(K) ± Error", "% Error", "LoF (%)", "RMS", "Diff C total (%)", "Covfit"]
        data = [
            [f"K{i+1}", f"{k[i]:.2e} ± {SE_k[i]:.2e}", f"{error_percent[i] * 100:.2f}", f"{lof:.2f}" if i == 0 else "", f"{rms:.2e}" if i == 0 else "", f"{dif_en_ct:.2f}" if i == 0 else "", f"{covfit:.2e}" if i == 0 else ""]
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
        
        K = np.array([k, error_percent]).T
        
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
            
        stats = np.array([rms, lof, MAE, dif_en_ct, EV, cov_matrix, optimizer])
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

        self.C = C
        self.Co = Co
        self.Y = Y
        self.phi = phi
        self.A = A
        self.k = k
        self.k_ini = k_ini
        self.stats = stats
        self.C_T = C_T


# Iniciar la aplicación
if __name__ == "__main__":
    app = wx.App(False)
    frame = App()
    frame.Show()
    app.MainLoop()
