import wx
import sys
import os
from wx import FileDialog
from wx.lib.scrolledpanel import ScrolledPanel
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
# from Simulation_controls import Simulation_controlsPanel
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

        self.SetMinSize((1100, 650))
        add_private_font_if_available()
        self.Bind(wx.EVT_CHAR_HOOK, self._on_char_hook)
        self.panel = wx.Panel(self)

        # Diseño usando Sizers
        self.main_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.left_sizer = wx.BoxSizer(wx.VERTICAL)
        self.right_sizer = wx.BoxSizer(wx.VERTICAL)

        # Añadir sizers al panel principal
        self.main_sizer.Add(self.left_sizer, 1, wx.EXPAND | wx.ALL, 5)
        self.main_sizer.Add(self.right_sizer, 1, wx.EXPAND | wx.ALL, 5)

        self.technique_notebook = wx.Notebook(self.panel)

        # --- Spectroscopy en ScrolledPanel ---
        self.spectro_page = ScrolledPanel(self.technique_notebook, style=wx.TAB_TRAVERSAL)
        self.spectroscopy_panel = Spectroscopy_controlsPanel(self.spectro_page, app_ref=self)
        _sizer_s = wx.BoxSizer(wx.VERTICAL)
        _sizer_s.Add(self.spectroscopy_panel, 1, wx.EXPAND)
        self.spectro_page.SetSizer(_sizer_s)
        self.spectro_page.SetupScrolling(scroll_x=True, scroll_y=True)
        self.spectro_page.SetMinSize((560, -1))
        self.spectro_page.inner_panel = self.spectroscopy_panel
        self.spectro_page.Bind(
            wx.EVT_SIZE,
            lambda e: (self.spectro_page.Layout(), self.spectro_page.FitInside(), e.Skip())
        )
        self.technique_notebook.AddPage(self.spectro_page, "Spectroscopy")

        # --- NMR en ScrolledPanel ---
        self.nmr_page = ScrolledPanel(self.technique_notebook, style=wx.TAB_TRAVERSAL)
        self.nmr_panel = NMR_controlsPanel(self.nmr_page, app_ref=self)
        _sizer_n = wx.BoxSizer(wx.VERTICAL)
        _sizer_n.Add(self.nmr_panel, 1, wx.EXPAND)
        self.nmr_page.SetSizer(_sizer_n)
        self.nmr_page.SetupScrolling(scroll_x=True, scroll_y=True)
        self.nmr_page.SetMinSize((560, -1))
        self.nmr_page.inner_panel = self.nmr_panel
        self.nmr_page.Bind(
            wx.EVT_SIZE,
            lambda e: (self.nmr_page.Layout(), self.nmr_page.FitInside(), e.Skip())
        )
        self.technique_notebook.AddPage(self.nmr_page, "NMR")
        # self.technique_notebook.AddPage(self.simulation_panel, "Simulation")

        self.left_sizer.Add(self.technique_notebook, 1, wx.EXPAND | wx.ALL)

        # Establecer el sizer principal y ajustar el layout
        self.panel.SetSizer(self.main_sizer)
        self.panel.Layout()
        self.Bind(wx.EVT_IDLE, self._fit_pages_on_idle)

        # ---------------------- Panel derecho ----------------------
        self.current_technique_panel = None
        self.technique_panel = BaseTechniquePanel(self.panel, app_ref=self)

        # Sizer con botones superiores
        buttons_sizer = wx.BoxSizer(wx.HORIZONTAL)
        buttons_sizer.AddStretchSpacer()

        self.btn_prev_figure = wx.Button(self.panel, label="<< Prev")
        buttons_sizer.Add(self.btn_prev_figure, 0, wx.ALL, 5)
        self.btn_prev_figure.Bind(wx.EVT_BUTTON, self.show_prev_figure)

        buttons_sizer.AddStretchSpacer()

        self.btn_process_data = wx.Button(self.panel, label="Process Data")
        buttons_sizer.Add(self.btn_process_data, 0, wx.ALL | wx.LEFT, 5)
        self.btn_process_data.Bind(wx.EVT_BUTTON, self.on_process_data)

        buttons_sizer.AddStretchSpacer()

        self.btn_next_figure = wx.Button(self.panel, label="Next >>")
        buttons_sizer.Add(self.btn_next_figure, 0, wx.ALL | wx.LEFT, 5)
        self.btn_next_figure.Bind(wx.EVT_BUTTON, self.show_next_figure)

        buttons_sizer.AddStretchSpacer()
        self.right_sizer.Add(buttons_sizer, 0, wx.EXPAND)

        # -------- Splitter único (canvas arriba, consola abajo) --------
        self.right_splitter = wx.SplitterWindow(self.panel, style=wx.SP_LIVE_UPDATE | wx.SP_3D)

        # Panel superior: gráfica
        canvas_panel = wx.Panel(self.right_splitter)
        canvas_sizer = wx.BoxSizer(wx.VERTICAL)

        # Figura sin constrained_layout; manejaremos márgenes manualmente
        self.fig = Figure()
        self.canvas = FigureCanvas(canvas_panel, -1, self.fig)
        canvas_sizer.Add(self.canvas, 1, wx.EXPAND | wx.ALL, 0)
        canvas_panel.SetSizer(canvas_sizer)
        canvas_panel.Bind(wx.EVT_SIZE, self._on_canvas_resize)
        self.canvas.Bind(wx.EVT_SIZE, self._on_canvas_resize)  # redibujo cuando el canvas cambia tamaño

        # Panel inferior: consola
        console_panel = wx.Panel(self.right_splitter)
        console_sizer = wx.BoxSizer(wx.VERTICAL)
        self.console = wx.TextCtrl(
            console_panel,
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_DONTWRAP | wx.HSCROLL
        )
        self.console.SetFont(get_monospace_font(9))
        self.console.SetBackgroundColour(wx.BLACK)
        self.console.SetForegroundColour(wx.WHITE)
        console_sizer.Add(self.console, 1, wx.EXPAND | wx.ALL, 0)
        console_panel.SetSizer(console_sizer)

        # Configuración del splitter
        self.right_splitter.SplitHorizontally(canvas_panel, console_panel)
        self.right_splitter.SetSashGravity(0.80)                  # conserva ~80% para la gráfica
        self.right_splitter.SetMinimumPaneSize(80)                # permite achicar el terminal
        self.right_splitter.SetSashPosition(int(self.GetSize().GetHeight() * 0.75))
        self.right_splitter.Bind(wx.EVT_SPLITTER_SASH_POS_CHANGED,  self._on_splitter_sash)
        self.right_splitter.Bind(wx.EVT_SPLITTER_SASH_POS_CHANGING, self._on_splitter_sash)

        # Añadir el splitter al sizer derecho
        self.right_sizer.Add(self.right_splitter, 1, wx.EXPAND | wx.ALL, 5)

        # Redirigir stdout hacia la consola
        sys.stdout = TextRedirector(self.console)

        # Botones inferiores (Save / Reset)
        save_results_sizer = wx.BoxSizer(wx.HORIZONTAL)
        save_results_sizer.AddStretchSpacer()

        self.btn_save_results = wx.Button(self.panel, label="Save Results")
        save_results_sizer.Add(self.btn_save_results, 0, wx.ALL, 5)
        self.btn_save_results.Bind(wx.EVT_BUTTON, self.save_results)

        reset_button = wx.Button(self.panel, label="Reset Calculation")
        reset_button.Bind(wx.EVT_BUTTON, self.reset_calculation)
        save_results_sizer.Add(reset_button, 0, wx.ALL, 5)

        save_results_sizer.AddStretchSpacer()
        self.right_sizer.Add(save_results_sizer, 0, wx.EXPAND)

        # Layout final
        self.panel.SetSizer(self.main_sizer)
        self.main_sizer.Layout()
        self.Refresh()
        self.Update()
        # Asegurar tamaño/márgenes correctos al arrancar
        wx.CallAfter(self._redraw_canvas)

    # ----------------- Handlers para teclado/resize -----------------
    def _on_char_hook(self, evt):
        win = wx.Window.FindFocus()
        if isinstance(win, (wx.TextCtrl, wx.SpinCtrl, wx.ComboBox, gridlib.Grid)):
            evt.Skip()
            return
        key_code = evt.GetKeyCode()
        if key_code is not None and ord('0') <= key_code <= ord('9'):
            return
        evt.Skip()

    def _fit_pages_on_idle(self, evt):
        # Mantiene actualizados los scrollbars cuando cambian paneles internos
        for page in getattr(self, 'spectro_page', []), getattr(self, 'nmr_page', []):
            if isinstance(page, list):
                continue
            try:
                page.Layout()
                page.FitInside()
            except Exception:
                pass
        evt.Skip()

    def _redraw_canvas(self):
        """Ajusta tamaño y márgenes para que no se recorte y se adapte al sash."""
        try:
            # Forzar al canvas a ocupar exactamente el área disponible del panel
            parent_sz = self.canvas.GetParent().GetClientSize()
            self.canvas.SetMinSize(parent_sz)
            self.canvas.SetSize(parent_sz)

            # Márgenes seguros (ajústalos si quieres más/menos borde)
            self.fig.subplots_adjust(left=0.14, right=0.985, bottom=0.20, top=0.985)

            # Redibujo eficiente
            self.canvas.draw_idle()
        except Exception:
            pass

    def _on_splitter_sash(self, evt):
        self._redraw_canvas()
        evt.Skip()

    def _on_canvas_resize(self, evt):
        # Redibuja la figura al cambiar tamaño del panel/canvas
        try:
            evt.GetEventObject().Layout()
        except Exception:
            pass
        self._redraw_canvas()
        evt.Skip()

    # ----------------- Botones navegación y cálculo -----------------
    def on_process_data(self, event):
        current_page = self.technique_notebook.GetCurrentPage()
        current_panel = getattr(current_page, "inner_panel", current_page)
        self.current_technique_panel = current_panel
        if current_panel:
            current_panel.process_data(event)

    def show_next_figure(self, event):
        if self.current_technique_panel and self.current_technique_panel.figures:
            self.current_technique_panel.current_figure_index = (
                self.current_technique_panel.current_figure_index + 1
            ) % len(self.current_technique_panel.figures)
            figure = self.current_technique_panel.figures[self.current_technique_panel.current_figure_index]
            self.current_technique_panel.update_canvas_figure(figure)

    def show_prev_figure(self, event):
        if self.current_technique_panel and self.current_technique_panel.figures:
            self.current_technique_panel.current_figure_index = (
                self.current_technique_panel.current_figure_index - 1
            ) % len(self.current_technique_panel.figures)
        figure = self.current_technique_panel.figures[self.current_technique_panel.current_figure_index]
        self.current_technique_panel.update_canvas_figure(figure)

    # ----------------- Reset & Guardado -----------------
    def reset_calculation(self, event):
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

            # Reiniciar menús desplegables
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

            # Limpiar el grid
            if hasattr(panel, 'model_grid'):
                if panel.model_grid.GetNumberRows() > 0:
                    panel.model_grid.DeleteRows(0, panel.model_grid.GetNumberRows())
                if panel.model_grid.GetNumberCols() > 0:
                    panel.model_grid.DeleteCols(0, panel.model_grid.GetNumberCols())

            # Limpiar lista de figuras
            if hasattr(self, 'fig'):
                self.fig.clear()
                if hasattr(self, 'canvas'):
                    self.canvas.figure.clear()
                    self.canvas.draw()

            # Eliminar checkboxes actuales
            if hasattr(panel, 'columns_names_panel'):
                children = list(panel.columns_names_panel.GetChildren())
                for child in children:
                    if isinstance(child, wx.CheckBox):
                        child.Destroy()

            # Limpiar diccionario de checkboxes
            self.vars_columnas = {}

            # Limpiar elementos adicionales en Optimización
            if hasattr(panel, 'choice_algoritm'):
                panel.choice_algoritm.SetSelection(0)

            if hasattr(panel, 'choice_model_settings'):
                panel.choice_model_settings.SetSelection(0)

            if hasattr(panel, 'choice_optimizer_settings'):
                panel.choice_optimizer_settings.SetSelection(0)

            # Limpiar grid de parámetros
            if hasattr(panel, 'grid'):
                panel.grid.ClearGrid()
                if panel.grid.GetNumberRows() > 0:
                    panel.grid.DeleteRows(0, panel.grid.GetNumberRows())

            # Resetear entradas
            if hasattr(panel, 'entry_nc'):
                panel.entry_nc.SetValue("0")
            if hasattr(panel, 'entry_nsp'):
                panel.entry_nsp.SetValue("0")

            # NMR
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

            # Spectroscopy
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

        active_page = self.technique_notebook.GetCurrentPage()
        active_panel = getattr(active_page, "inner_panel", active_page)
        reset_panel(active_panel)

        self.Layout()
        self.Refresh()
        self.Update()

    def save_results(self, event):
        with wx.FileDialog(
            self, "Save Excel file", wildcard="Excel files (*.xlsx)|*.xlsx",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT
        ) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return
            file_path = fileDialog.GetPath()
            if not file_path.endswith('.xlsx'):
                file_path += '.xlsx'

            current_page = self.technique_notebook.GetCurrentPage()
            current_panel = getattr(current_page, "inner_panel", current_page)

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
                    current_panel.A.to_excel(writer, sheet_name="Molar_Absortivities",
                                             index_label='nm', index=True)
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
                    current_panel.phi.to_excel(writer, sheet_name="Y_calculated",
                                               index_label='nm', index=True)
                if hasattr(current_panel, 'Y'):
                    current_panel.Y.to_excel(writer, sheet_name="Y_observed",
                                             index_label='nm', index=True)
                if hasattr(current_panel, 'stats'):
                    current_panel.stats.to_excel(writer, sheet_name="Stats")

            wx.MessageBox(f"Results saved to {file_path}.", "Information", wx.OK | wx.ICON_INFORMATION)


# Iniciar la aplicación
if __name__ == "__main__":
    if sys.platform.startswith("win"):
        os.environ.setdefault("WX_HIGH_DPI_AWARE", "1")
    app = wx.App(False)
    frame = App()
    frame.Show()
    app.MainLoop()
