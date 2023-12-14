import wx
import sys
from wx import FileDialog
import wx.grid as gridlib
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
from Spectroscopy_controls import Spectroscopy_controlsPanel
from NMR_controls import NMR_controlsPanel
from Methods import BaseTechniquePanel 
import importlib


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
        self.panel = wx.Panel(self)       
        
        #self.figures = []  # Lista para almacenar figuras 
        #self.current_figure_index = -1  # Índice inicial para navegación de figuras

        # Diseño usando Sizers
        self.main_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.left_sizer = wx.BoxSizer(wx.VERTICAL)
        self.right_sizer = wx.BoxSizer(wx.VERTICAL)

        # Añadir sizers al panel principal
        self.main_sizer.Add(self.left_sizer, 1, wx.EXPAND | wx.ALL, 5)
        self.main_sizer.Add(self.right_sizer, 2, wx.EXPAND | wx.ALL, 5)
        
        self.technique_notebook = wx.Notebook(self.panel)
                
        # Añadir paneles de técnica al wx.Notebook
        #self.spectroscopy_panel = SpectroscopyPanel(self.technique_notebook)
        self.spectroscopy_panel = Spectroscopy_controlsPanel(self.technique_notebook, app_ref=self)
        self.nmr_panel = NMR_controlsPanel(self.technique_notebook, app_ref=self)
        #self.pka_panel = pkaPanel(self.technique_notebook)
        #self.pka_panel = pkaPanel(self.technique_notebook)
        self.technique_notebook.AddPage(self.spectroscopy_panel, "Spectroscopy")
        self.technique_notebook.AddPage(self.nmr_panel, "NMR")
        #self.technique_notebook.AddPage(self.nmr_panel, "NMR") ... debo utilizar estas lineas: self.notebook = TechniqueNotebook(self)
        
        self.left_sizer.Add(self.technique_notebook,  1, wx.EXPAND | wx.ALL)
        # Establecer el sizer principal y ajustar el layout
        self.panel.SetSizer(self.main_sizer)
        self.panel.Layout()
       
    ##############################################################################################################
        """ Panel derecho del gui """

        self.current_technique_panel = None  # Inicializas la referencia al panel actual

        
        self.technique_panel = BaseTechniquePanel(self.panel, app_ref=self)

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
        #self.btn_process_data.Bind(wx.EVT_BUTTON, self.process_data)
        self.btn_process_data.Bind(wx.EVT_BUTTON, self.on_process_data)

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

        console_sizer.Add(self.console, 1, wx.EXPAND | wx.ALL)
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
        reset_button.Bind(wx.EVT_BUTTON, self.reset_calculation)
        save_results_sizer.Add(reset_button, 0, wx.ALL, 5)

        save_results_sizer.AddStretchSpacer()

        # Añadir el sizer al sizer principal del lado derecho
        self.right_sizer.Add(save_results_sizer, 0, wx.EXPAND)

        # Método de controlador de eventos para el botón irá aquí     

    
        self.panel.SetSizer(self.main_sizer)
        self.main_sizer.Layout()
    ####################################################################################################################
    def on_process_data(self, event):
        # Obtener el panel actualmente seleccionado
        current_panel = self.technique_notebook.GetCurrentPage()
        self.current_technique_panel = current_panel
  
        # Llamar al método `process_data` del panel actual
        if current_panel:
            current_panel.process_data(event)

    
    def show_next_figure(self, event):
        if self.current_technique_panel and self.current_technique_panel.figures:
            self.current_technique_panel.current_figure_index = (self.current_technique_panel.current_figure_index + 1) % len(self.current_technique_panel.figures)
            figure = self.current_technique_panel.figures[self.current_technique_panel.current_figure_index]
            self.current_technique_panel.update_canvas_figure(figure)

    def show_prev_figure(self, event):
        if self.current_technique_panel and self.current_technique_panel.figures:
            self.current_technique_panel.current_figure_index = (self.current_technique_panel.current_figure_index - 1) % len(self.current_technique_panel.figures)
            figure = self.current_technique_panel.figures[self.current_technique_panel.current_figure_index]
            self.current_technique_panel.update_canvas_figure(figure)


    def reset_calculation(self, event):
        # Reiniciar la ruta del archivo y la etiqueta correspondiente
        current_panel = self.current_technique_panel
        if hasattr(current_panel, 'file_path'):
            current_panel.file_path = None
        if hasattr(current_panel, 'lbl_file_path'):
            current_panel.lbl_file_path.SetLabel("No file selected")

        if hasattr(current_panel, "sheet_spectra_panel"):
            current_panel.choice_sheet_spectra = None

        if hasattr(current_panel, "choice_sheet_conc"):
            current_panel.choice_sheet_conc = None

        # Limpiar y reiniciar el DataFrame
        if hasattr(current_panel, 'df'):
            current_panel.df = None

        # Limpiar el grid (si lo estás utilizando)
        if hasattr(self, 'model_grid'):
            # Verificar si el grid tiene filas; si es así, eliminarlas todas
            if self.model_grid.GetNumberRows() > 0:
                self.model_grid.DeleteRows(0, self.model_grid.GetNumberRows())

            # Verificar si el grid tiene columnas; si es así, eliminarlas todas
            if self.model_grid.GetNumberCols() > 0:
                self.model_grid.DeleteCols(0, self.model_grid.GetNumberCols())

        # Limpiar la lista de figuras
        if hasattr(self, 'fig'):
            self.fig.clear()
            # Si estás utilizando un canvas para mostrar las figuras, también debes limpiarlo
            if hasattr(self, 'canvas'):
                self.canvas.figure.clear()
                self.canvas.draw()

        # Eliminar los checkboxes actuales
        if hasattr(current_panel, 'columns_names_panel'):
            children = list(current_panel.columns_names_panel.GetChildren())
            for child in children:  # Modificado para incluir todos los hijos
                if isinstance(child, wx.CheckBox):
                    child.Destroy()

       # Limpiar el diccionario de checkboxes
        self.vars_columnas = {}

        self.Layout()

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

             # Obtener el panel actualmente seleccionado
            current_panel = self.technique_notebook.GetCurrentPage()

            # Aquí va la lógica de guardado de tus datos
            with pd.ExcelWriter(file_path) as writer:
                if hasattr(current_panel, 'modelo'):
                    current_panel.modelo.to_excel(writer, sheet_name="Model")
                if hasattr(current_panel, 'C'):
                    current_panel.C.to_excel(writer, sheet_name="Absorbent_species")
                if hasattr(current_panel, 'Co'):
                    current_panel.Co.to_excel(writer, sheet_name="All_species")   
                if hasattr(current_panel, 'concentracion'):
                    current_panel.concentracion.to_excel(writer, sheet_name="Tot_con_comp")  
                if hasattr(current_panel, 'A'):
                    current_panel.A.to_excel(writer, sheet_name="Molar_Absortivities", index_label = 'nm', index = True)
                if hasattr(current_panel, 'dq'):
                    current_panel.dq.to_excel(writer, sheet_name="Chemical_Shifts")
                if hasattr(current_panel, 'dq_cal'):
                    current_panel.dq_cal.to_excel(writer, sheet_name="Calculated_Chemical_Shifts")
                if hasattr(current_panel, 'coef'):
                    current_panel.coef.to_excel(writer, sheet_name="Coefficients")
                if hasattr(current_panel, 'k'):
                    current_panel.k.to_excel(writer, sheet_name="K_calculated")
                if hasattr(current_panel, 'k_ini'):
                    current_panel.k_ini.to_excel(writer, sheet_name="Init_guess_K")
                if hasattr(current_panel, 'phi'):
                    current_panel.phi.to_excel(writer, sheet_name="Y_calculated", index_label = 'nm', index = True)
                if hasattr(current_panel, 'Y'):
                    current_panel.Y.to_excel(writer, sheet_name="Y_observed", index_label = 'nm', index = True)
                if hasattr(current_panel, 'stats'):
                    current_panel.stats.to_excel(writer, sheet_name="Stats")

            # Mostrar un mensaje al finalizar el guardado
            wx.MessageBox(f"Results saved to {file_path}.", "Information", wx.OK | wx.ICON_INFORMATION)



# Iniciar la aplicación
if __name__ == "__main__":
    app = wx.App(False)
    frame = App()
    frame.Show()
    app.MainLoop()
