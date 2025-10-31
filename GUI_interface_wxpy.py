import wx
import sys
from wx import FileDialog
import wx.grid as gridlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('WXAgg')
matplotlib.rcParams['keymap.quit'] = []
matplotlib.rcParams['keymap.quit_all'] = []
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy import optimize
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, basinhopping
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore")
import timeit
from Spectroscopy_controls import Spectroscopy_controlsPanel
from NMR_controls import NMR_controlsPanel
#from Simulation_controls import Simulation_controlsPanel
from Methods import BaseTechniquePanel, add_private_font_if_available, get_monospace_font
import importlib


class TextRedirector:
    def __init__(self, text_ctrl):
        self.text_ctrl = text_ctrl

    def write(self, string):
        wx.CallAfter(self.text_ctrl.WriteText, string.expandtabs(4))

    def flush(self):
        pass

class CancelledByUserException(Exception):
    pass

class App(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, title="HM Fit", size=(800, 600))
        add_private_font_if_available()
        self.Bind(wx.EVT_CHAR_HOOK, self._on_char_hook)
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
        #self.simulation_panel = Simulation_controlsPanel(self.technique_notebook, app_ref=self)
        #self.pka_panel = pkaPanel(self.technique_notebook)
        self.technique_notebook.AddPage(self.spectroscopy_panel, "Spectroscopy")
        self.technique_notebook.AddPage(self.nmr_panel, "NMR")
        #self.technique_notebook.AddPage(self.simulation_panel, "Simulation") #... debo utilizar estas lineas: self.notebook = TechniqueNotebook(self)
        
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
        self.console.SetFont(get_monospace_font(9))

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
        self.Refresh()             # Refresca el frame para mostrar los cambios
        self.Update()

    def _on_char_hook(self, evt):
        win = wx.Window.FindFocus()
        if isinstance(win, (wx.TextCtrl, wx.SpinCtrl, wx.ComboBox, gridlib.Grid)):
            evt.Skip()
            return

        key_code = evt.GetKeyCode()
        if key_code is not None and ord('0') <= key_code <= ord('9'):
            return

        evt.Skip()
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
        # Función para reiniciar un panel específico
        def reset_panel(panel):
            if hasattr(panel, 'file_path'):
                panel.file_path = None
                panel.Layout()
                self.Refresh()
                self.Update()

            if hasattr(panel, 'lbl_file_path'):
                panel.lbl_file_path.SetLabel("No file selected")
                panel.Layout()
                self.Refresh()
                self.Update()

            if hasattr(panel, 'scrolled_window'):
                panel.scrolled_window.Layout()
                self.Refresh()
                self.Update()

            # Reiniciar los menús desplegables
            if hasattr(panel, "choice_sheet_spectra"):
                if hasattr(panel.choice_sheet_spectra, 'Clear'):
                    panel.choice_sheet_spectra.Clear()
                if hasattr(panel.choice_sheet_spectra, 'SetSelection'):
                    panel.choice_sheet_spectra.SetSelection(-1)
                self.Refresh()
                self.Update()

            if hasattr(panel, "choice_sheet_conc"):
                if hasattr(panel.choice_sheet_conc, 'Clear'):
                    panel.choice_sheet_conc.Clear()
                if hasattr(panel.choice_sheet_conc, 'SetSelection'):
                    panel.choice_sheet_conc.SetSelection(-1)
                self.Refresh()
                self.Update()

            if hasattr(panel, 'receptor_choice'):
                if hasattr(panel.receptor_choice, 'Clear'):
                    panel.receptor_choice.Clear()
                if hasattr(panel.receptor_choice, 'SetSelection'):
                    panel.receptor_choice.SetSelection(-1)
                self.Refresh()
                self.Update()

            if hasattr(panel, 'guest_choice'):
                if hasattr(panel.guest_choice, 'Clear'):
                    panel.guest_choice.Clear()
                if hasattr(panel.guest_choice, 'SetSelection'):
                    panel.guest_choice.SetSelection(-1)
                self.Refresh()
                self.Update()

            # Limpiar y reiniciar el DataFrame
            if hasattr(panel, 'df'):
                panel.df = None
                self.Refresh()
                self.Update()

            # Limpiar el grid (si lo estás utilizando)
            if hasattr(panel, 'model_grid'):
                if panel.model_grid.GetNumberRows() > 0:
                    panel.model_grid.DeleteRows(0, panel.model_grid.GetNumberRows())
                if panel.model_grid.GetNumberCols() > 0:
                    panel.model_grid.DeleteCols(0, panel.model_grid.GetNumberCols())

            # Limpiar la lista de figuras
            if hasattr(self, 'fig'):
                self.fig.clear()
                if hasattr(self, 'canvas'):
                    self.canvas.figure.clear()
                    self.canvas.draw()

            # Eliminar los checkboxes actuales
            if hasattr(panel, 'columns_names_panel'):
                children = list(panel.columns_names_panel.GetChildren())
                for child in children:
                    if isinstance(child, wx.CheckBox):
                        child.Destroy()

            # Limpiar el diccionario de checkboxes
            self.vars_columnas = {}

            # Limpiar los elementos adicionales en el panel de Optimización
            if hasattr(panel, 'choice_algoritm'):
                panel.choice_algoritm.SetSelection(0)

            if hasattr(panel, 'choice_model_settings'):
                panel.choice_model_settings.SetSelection(0)

            if hasattr(panel, 'choice_optimizer_settings'):
                panel.choice_optimizer_settings.SetSelection(0)

            # Limpiar el grid de parámetros
            if hasattr(panel, 'grid'):
                panel.grid.ClearGrid()
                if panel.grid.GetNumberRows() > 0:
                    panel.grid.DeleteRows(0, panel.grid.GetNumberRows())

            # Resetear entradas de texto para número de componentes y especies
            if hasattr(panel, 'entry_nc'):
                panel.entry_nc.SetValue("0")

            if hasattr(panel, 'entry_nsp'):
                panel.entry_nsp.SetValue("0")

            # Reiniciar elementos específicos de la pestaña NMR
            if hasattr(panel, "choice_chemshifts"):
                if hasattr(panel.choice_chemshifts, 'Clear'):
                    panel.choice_chemshifts.Clear()
                if hasattr(panel.choice_chemshifts, 'SetSelection'):
                    panel.choice_chemshifts.SetSelection(-1)
                self.Refresh()
                self.Update()

            if hasattr(panel, 'chemical_shifts_panel'):
                children = list(panel.chemical_shifts_panel.GetChildren())
                for child in children:
                    if isinstance(child, wx.CheckBox):
                        child.Destroy()

            if hasattr(panel, 'choice_columns_panel'):
                children = list(panel.choice_columns_panel.GetChildren())
                for child in children:
                    if isinstance(child, wx.Choice):
                        child.Clear()
                        child.SetSelection(-1)
                    elif isinstance(child, wx.TextCtrl):
                        child.SetValue("")

            # Reiniciar elementos específicos de la pestaña Spectroscopy
            if hasattr(panel, 'sheet_spectra_panel'):
                if hasattr(panel.sheet_spectra_panel, 'Clear'):
                    panel.sheet_spectra_panel.Clear()
                if hasattr(panel.sheet_spectra_panel, 'SetSelection'):
                    panel.sheet_spectra_panel.SetSelection(-1)
                self.Refresh()
                self.Update()

            if hasattr(panel, 'sheet_conc_panel'):
                if hasattr(panel.sheet_conc_panel, 'Clear'):
                    panel.sheet_conc_panel.Clear()
                if hasattr(panel.sheet_conc_panel, 'SetSelection'):
                    panel.sheet_conc_panel.SetSelection(-1)
                self.Refresh()
                self.Update()

        # Determinar el panel activo y reiniciarlo
        active_panel = self.technique_notebook.GetCurrentPage()  # Asume que self.technique_notebook es el wx.Notebook que contiene las pestañas
        reset_panel(active_panel)

        self.Layout()
        self.Refresh()
        self.Update()


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
                if hasattr(current_panel, 'C_T'):
                    current_panel.C_T.to_excel(writer, sheet_name="Tot_con_comp")  
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
