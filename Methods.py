import os
import wx
import sys
from wx import FileDialog
from matplotlib.figure import Figure
import pandas as pd
import numpy as np


def add_private_font_if_available():
    """Register an embedded monospace font if it ships with the project."""
    here = os.path.dirname(os.path.abspath(__file__))
    font_path = os.path.join(here, "assets", "fonts", "DejaVuSansMono.ttf")
    try:
        if os.path.isfile(font_path):
            wx.Font.AddPrivateFont(font_path)
    except Exception:
        # Best effort: failing to register the font should not break the UI.
        pass


def get_monospace_font(point_size=None):
    """Return a robust monospace font with fallbacks for portable bundles."""
    base_font = wx.SystemSettings.GetFont(wx.SYS_DEFAULT_GUI_FONT)
    default_size = base_font.GetPointSize() if base_font and base_font.IsOk() else 10
    size = point_size or default_size

    candidates = [
        "DejaVu Sans Mono",
        "Liberation Mono",
        "Noto Sans Mono",
        "Courier New",
        "Monospace",
        "monospace",
    ]

    for face in candidates:
        font_info = wx.FontInfo(size).FaceName(face).Family(wx.FONTFAMILY_TELETYPE)
        candidate = wx.Font(font_info)
        if candidate.IsOk():
            return candidate

    return wx.Font(wx.FontInfo(size).Family(wx.FONTFAMILY_TELETYPE))


class BaseTechniquePanel(wx.Panel):
    def __init__(self, parent, app_ref):
        super().__init__(parent)
        self.app_ref = app_ref
        
        self.vars_columnas = {} #lista para almacenar las columnas de la hoja de concentraciones
        self.figures = []  # Lista para almacenar figuras 
        self.current_figure_index = -1  # Índice inicial para navegación de figuras
        self.vars_chemshift = {}

    
    # --- en Methods.py, dentro de BaseTechniquePanel ---
    def _refresh_scroller(self, sw, scroll_x=True, scroll_y=False):
        """Recalcula tamaño virtual, layout y fuerza el repintado inmediato."""
        from wx.lib.scrolledpanel import ScrolledPanel
    
        # Si es ScrolledPanel usamos su helper
        if isinstance(sw, ScrolledPanel):
            sw.FitInside()
            sw.SetupScrolling(scroll_x=scroll_x, scroll_y=scroll_y)
        else:
            # ScrolledWindow clásico: fijar tamaño virtual manualmente
            if sw.GetSizer():
                sw.SetVirtualSize(sw.GetSizer().CalcMin())
    
        sw.Layout()
        sw.Refresh()
        sw.Update()
        sw.SendSizeEvent()
    
        tlw = sw.GetTopLevelParent()
        if tlw:
            tlw.Layout()
            tlw.SendSizeEvent()



    # añadir parametros al cuadro de las constantes. Editar los valores por defecto.
    def add_parameter_bounds(self, num_parameters):
        self.grid.ClearGrid()  # Limpiar el Grid antes de añadir nuevos elementos
        self.grid.SetRowLabelSize(0)  # Ocultar la columna de números de fila

        # Asegurarse de que el Grid tiene suficientes filas
        if self.grid.GetNumberRows() < num_parameters:
            self.grid.AppendRows(num_parameters - self.grid.GetNumberRows())
        elif self.grid.GetNumberRows() > num_parameters:
            self.grid.DeleteRows(num_parameters, self.grid.GetNumberRows() - num_parameters)

        # Rellenar las filas con los datos
        for i in range(num_parameters):
            self.grid.SetCellValue(i, 0, f"K {i + 1}")  # Nombre del parámetro
            self.grid.SetCellValue(i, 1, "")  # Valor por defecto para "Valor"
            self.grid.SetCellValue(i, 2, "")  # Valor por defecto para "Mín"
            self.grid.SetCellValue(i, 3, "")  # Valor por defecto para "Máx"
 
    # Función para extraer los datos del grid y crear x0 y bonds para el optimizador

    def extract_constants_from_grid(self):
        n_rows = self.grid.GetNumberRows()
        x0 = []
        bounds = []
        for row in range(n_rows):
            # Intentar convertir los valores de las celdas a flotantes, o asignar infinitos según corresponda
            try:
                value = float(self.grid.GetCellValue(row, 1)) if self.grid.GetCellValue(row, 1) else None
                min_val = float(self.grid.GetCellValue(row, 2)) if self.grid.GetCellValue(row, 2) else -np.inf
                max_val = float(self.grid.GetCellValue(row, 3)) if self.grid.GetCellValue(row, 3) else np.inf
            except ValueError:  # Captura el error si la conversión a flotante falla
                continue  # Puede agregar aquí un manejo de errores si es necesario
            
            # Añadir el valor a la lista de valores iniciales si no es None
            if value is not None:
                x0.append(value)
            
            # Añadir la tupla de límites a la lista de límites
            # Aquí ya no necesitamos comprobar si min_val o max_val son None
            bounds.append((min_val, max_val))

        return np.array(x0), bounds


    def on_optimize_button_click(self, event):
        parameter_bounds = self.get_parameter_bounds()
        if parameter_bounds is not None:
            print(parameter_bounds)  # Proceder con la optimización
        else:
            # Manejar el caso en que los valores no son válidos
            wx.MessageBox("Por favor, corrija los valores de los límites antes de continuar.", "Error", wx.OK | wx.ICON_WARNING)

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

    def create_sheet_section(self, label_text, default_value, parent):
            
            if parent is None:
                parent = self.panel

            panel = wx.Panel(parent)
            # Crear un panel para esta sección
            #panel = wx.Panel(self.panel)
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
            self._refresh_scroller(self.scrolled_window, scroll_x=True, scroll_y=False)
            self.file_path = file_path
            self.populate_sheet_choices(file_path)
    
    def create_sheet_dropdown_section(self, label_text, parent=None):
            if parent is None:
                parent = self.panel

            panel = wx.Panel(parent)
            sizer = wx.BoxSizer(wx.HORIZONTAL)

            # Crear y añadir la etiqueta
            label = wx.StaticText(panel, label=label_text)
            sizer.Add(label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)

            # Crear y añadir el menú desplegable
            choice = wx.Choice(panel)
            sizer.Add(choice, 1, wx.ALL | wx.EXPAND, 5)

            panel.SetSizer(sizer)
            return panel, choice
    
    def update_dropdown_choices(self):
        # Obtener las columnas seleccionadas de los checkboxes
        selected_columns = [checkbox.GetLabel() for checkbox in self.vars_columnas.values() if checkbox.IsChecked()]
        # Vaciar las selecciones actuales
        self.receptor_choice.Clear()
        self.guest_choice.Clear()
        # Establecer las columnas seleccionadas como elementos de los menús desplegables
        self.receptor_choice.AppendItems(selected_columns)
        self.guest_choice.AppendItems(selected_columns)
        # Reiniciar la selección
        self.receptor_choice.SetSelection(-1)
        self.guest_choice.SetSelection(-1)

    # Función para manejar la selección duplicada
    def on_dropdown_selection(self, event):
        # Obtener las selecciones actuales de los menús desplegables
        receptor_selection = self.receptor_choice.GetStringSelection()
        guest_selection = self.guest_choice.GetStringSelection()
        # Comprobar si el usuario ha seleccionado la misma columna para ambos
        if receptor_selection == guest_selection and receptor_selection != "":
            wx.MessageBox("Receptor and Guest cannot be the same column. Please select different columns.", "Selection Error", wx.OK | wx.ICON_WARNING)
            # Reiniciar las selecciones
            self.receptor_choice.SetSelection(-1)
            self.guest_choice.SetSelection(-1)
            # Puedes volver a llamar a la función para actualizar los menús desplegables si es necesario
            self.update_dropdown_choices()


    # Vincular la función de actualización a los eventos de los checkboxes
    def bind_checkbox_events(self):
        for child in self.columns_names_panel.GetChildren():
            if isinstance(child, wx.CheckBox):
                child.Bind(wx.EVT_CHECKBOX, self.on_checkbox_select)

    def on_checkbox_select(self, event):
        # Llamar a la función que actualiza los menús desplegables
        self.update_dropdown_choices()
        # También puedes manejar otras acciones necesarias cuando un checkbox es seleccionado o deseleccionado


    def populate_sheet_choices(self, file_path):
        try:
            # Obtener los nombres de las hojas del archivo Excel
            sheet_names = pd.ExcelFile(file_path).sheet_names
            
            # Añadir una opción en blanco al principio de la lista de nombres de las hojas
            sheet_names_with_blank = [""] + sheet_names
            
            # Verificar y configurar las opciones del wx.Choice para espectros
            if hasattr(self, 'choice_sheet_spectra'):
                self.choice_sheet_spectra.SetItems(sheet_names_with_blank)
                self.choice_sheet_spectra.SetSelection(0)  # La opción en blanco será seleccionada por defecto

            # Verificar y configurar las opciones del wx.Choice para concentraciones
            if hasattr(self, 'choice_chemshifts'):
                self.choice_chemshifts.SetItems(sheet_names_with_blank)
                self.choice_chemshifts.SetSelection(0)  # La opción en blanco será seleccionada por defecto
                self.choice_chemshifts.Bind(wx.EVT_CHOICE, self.on_chemical_shift_sheet_selected)

            # Verificar y configurar las opciones del wx.Choice para concentraciones
            if hasattr(self, 'choice_sheet_conc'):
                self.choice_sheet_conc.SetItems(sheet_names_with_blank)
                self.choice_sheet_conc.SetSelection(0)  # La opción en blanco será seleccionada por defecto
                self.choice_sheet_conc.Bind(wx.EVT_CHOICE, self.on_conc_sheet_selected)

            # Similarmente, puedes agregar verificaciones y configuraciones para otros menús desplegables
            # que podrías tener en futuras pestañas

        except Exception as e:
            wx.MessageBox(f"Error al leer el archivo Excel: {e}", "Error", wx.OK | wx.ICON_ERROR)

    def clear_model_grid(self):
        if hasattr(self, 'model_grid'):
            # Verificar si el grid tiene filas; si es así, eliminarlas todas
            if self.model_grid.GetNumberRows() > 0:
                self.model_grid.DeleteRows(0, self.model_grid.GetNumberRows())

            # Verificar si el grid tiene columnas; si es así, eliminarlas todas
            if self.model_grid.GetNumberCols() > 0:
                self.model_grid.DeleteCols(0, self.model_grid.GetNumberCols())

            self.model_grid.Refresh()  # Opcionalmente refresca el grid después de borrarlo

    def on_conc_sheet_selected(self, event):
        selected_sheet = self.choice_sheet_conc.GetStringSelection()
        try:
            df = pd.read_excel(self.file_path, sheet_name=selected_sheet)
            # Crear un mapeo de nombres de columna a índices
            self.column_indices = {col: i for i, col in enumerate(df.columns)}
            # Crear checkboxes y actualizar menús desplegables
            self.create_checkboxes(df.columns)
            self.update_dropdown_choices()
        except Exception as e:
            wx.MessageBox(f"Error al leer la hoja de Excel: {e}", "Error en la hoja de Excel", wx.OK | wx.ICON_ERROR)
    

    ### Funciones para Checkboxes de desplazamientos químicos. 
    def on_chemical_shift_sheet_selected(self, event):
        selected_sheet = self.choice_chemshifts.GetStringSelection()
        try:
            df = pd.read_excel(self.file_path, sheet_name=selected_sheet)
            self.create_chemshift_checkboxes(df.columns)
        except Exception as e:
            wx.MessageBox(f"Error al leer la hoja de Excel: {e}", "Error en la hoja de Excel", wx.OK | wx.ICON_ERROR)

    def create_chemshift_checkboxes(self, column_names):
        # Limpiar checkboxes antiguos
        children = list(self.chemical_shifts_panel.GetChildren())
        for child in children:
            if isinstance(child, wx.CheckBox):
                child.Destroy()

        # Asegurarse de que self.vars_chemshift está vacío antes de empezar a añadir nuevos checkboxes
        self.vars_chemshift = {}

        # Crear los checkboxes dentro del panel y añadirlos a self.vars_chemshift
        for col in column_names:
            checkbox = wx.CheckBox(self.chemical_shifts_panel, label=col)
            checkbox.Bind(wx.EVT_CHECKBOX, self.on_chemshift_checkbox_select)
            self.chemical_shifts_sizer.Add(checkbox, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
            self.vars_chemshift[col] = checkbox

        # Reorganizar los controles en el panel
        self.chemical_shifts_panel.Layout()
        self._refresh_scroller(self.chemical_shifts_panel, scroll_x=True, scroll_y=False)

    
    def on_chemshift_checkbox_select(self, event):
        cb = event.GetEventObject()
        label = cb.GetLabel()
        if cb.IsChecked():
            print(f"Selected: {label}")
        else:
            print(f"Deselected: {label}")
    #### Termina control de checkboxes panel NMR de desplazamientos químicos. 

    # Función para obtener el índice de la columna seleccionada para el receptor
    def get_receptor_column_index(self):
        column_name = self.receptor_choice.GetStringSelection()
        return self.column_indices.get(column_name, -1)  # Retorna -1 si no se encuentra

    # Función para obtener el índice de la columna seleccionada para el huésped
    def get_guest_column_index(self):
        column_name = self.guest_choice.GetStringSelection()
        return self.column_indices.get(column_name, -1)  # Retorna -1 si no se encuentra

    
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
        
        self.bind_checkbox_events() 
        #self.columns_names_panel.Layout()     
        self._refresh_scroller(self.columns_names_panel, scroll_x=True, scroll_y=False)

    def on_checkbox_select(self, event):
        cb = event.GetEventObject()
        label = cb.GetLabel()
        if cb.IsChecked():
            print(f"Selected: {label}")
            # Añadir a la lista de seleccionados, o realizar otra acción
            self.update_dropdown_choices()
        else:
            print(f"Deselected: {label}")
            # Eliminar de la lista de seleccionados, o realizar otra acción
            self.update_dropdown_choices()

    def figura(self, x, y, mark, ylabel, xlabel, title):
        fig = Figure(figsize=(4, 4), dpi=200)
        ax = fig.add_subplot(111)
        ax.plot(x, y, mark)
        ax.set_ylabel(ylabel, size="xx-large")
        ax.set_xlabel(xlabel, size="xx-large")
        ax.tick_params(axis='both', which='major', labelsize='large')
        self.figures.append(fig)  # Almacenar tanto fig como ax
        #print(f"Figura añadida en {id(self)}. Total de figuras: {len(self.figures)}")
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
        self.current_figure_index = len(self.figures) - 1
        #print(f"Figura añadida en {id(self)}. Total de figuras: {len(self.figures)}")
        self.update_canvas_figure(fig)
    
    def update_canvas_figure(self, new_figure):
        #print("Mostrando figura:", self.figures)
        current_axes = self.app_ref.canvas.figure.gca()
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

        self.app_ref.canvas.draw()
    
    # Manejador de eventos para la selección de celdas o filas
    def on_selection_changed(self, event):
        # Intenta cargar los datos desde el archivo Excel si está disponible
        selected_rows = self.model_grid.GetSelectedRows()
            #print("Filas seleccionadas en la interfaz de usuario (corresponden a columnas en el DataFrame):", selected_rows)
            # Crea un DataFrame a partir de los datos del grid si hay un error al cargar desde Excel
        df = pd.DataFrame(self.extract_data_from_grid())
        
        print("Columnas seleccionadas:", selected_rows)
        return selected_rows

        
    def on_define_model_dimensions_checked(self, event=None):
        
        try:
            num_components = int(self.entry_nc.GetValue())
            num_species = int(self.entry_nsp.GetValue())
            total_rows = num_components + num_species
            
            # Limpiar el grid existente
            self.model_grid.ClearGrid()
            if self.model_grid.GetNumberRows() > 0:
                self.model_grid.DeleteRows(0, self.model_grid.GetNumberRows())
            if self.model_grid.GetNumberCols() > 0:
                self.model_grid.DeleteCols(0, self.model_grid.GetNumberCols())

            # Establecer nuevas dimensiones del grid
            self.model_grid.AppendCols(num_components)
            self.model_grid.AppendRows(total_rows)

            # Asignar nombres a las columnas y filas
            for col in range(num_components):
                self.model_grid.SetColLabelValue(col, f"C{col + 1}")
            for row in range(total_rows):
                self.model_grid.SetRowLabelValue(row, f"sp{row + 1}")

            # Llenar el grid con la matriz de identidad para las componentes
            identity_matrix = np.eye(num_components)
            for row in range(num_components):
                for col in range(num_components):
                    self.model_grid.SetCellValue(row, col, str(identity_matrix[row, col]))

            # Las filas adicionales se dejan en blanco o con otro valor predeterminado si se desea
            for row in range(num_components, total_rows):
                for col in range(num_components):
                    self.model_grid.SetCellValue(row, col, "0")  # o cualquier otro valor predeterminado

            # Llamada a la función para añadir los límites de los parámetros
            self.add_parameter_bounds(num_species)

            self.model_grid.AutoSize()  # Ajustar el tamaño de las celdas para mostrar el contenido
            self.model_panel.Layout()  # Actualizar el layout del panel que contiene el grid
        except ValueError:
            # Mostrar mensaje de error si los valores no son enteros
            wx.MessageBox("Please enter valid integers for Number of Components and Number of Species.",
                        "Error", wx.OK | wx.ICON_ERROR)
        
        self.model_panel.Layout()  # Actualiza el layout del panel que contiene el grid
        self.model_panel.Fit()     # Ajusta el tamaño del panel para coincidir con el tamaño de sus hijos
        self.Layout()              # Actualiza el layout del frame si 'self' es el frame
        #self.Fit()                 # Ajusta el tamaño del frame para coincidir con el tamaño de sus hijos
        self.Refresh()             # Refresca el frame para mostrar los cambios
        self.Update()              # Fuerza la repintura inmediata del frame


    def extract_data_from_grid(self):
        num_rows = self.model_grid.GetNumberRows()
        num_cols = self.model_grid.GetNumberCols()
        data_matrix = []

        for row in range(num_rows):
            row_data = []
            for col in range(num_cols):
                value_str = self.model_grid.GetCellValue(row, col)
                # Intentar convertir el valor a float, si no es posible dejarlo como está
                try:
                    value = float(value_str) if value_str else None  # Convertir a float si la celda no está vacía
                except ValueError:
                    value = value_str  # Mantener como cadena si no puede convertirse a float
                row_data.append(value)
            data_matrix.append(row_data)

        return data_matrix
    
    def update_parameter_grid(self):
        # Obtener el valor seleccionado en el menú desplegable
        model_setting = self.choice_model_settings.GetStringSelection()

        # Calcular el número de constantes de asociación
        if model_setting == "Non-cooperative":
            n_K = int(self.entry_nsp.GetValue()) - 1
        else:
            n_K = int(self.entry_nsp.GetValue())

        # Llamar a la función que actualiza el grid con el número correcto de constantes
        self.add_parameter_bounds(n_K)


    def res_consola(self, prefijo, resp):
        # Actualiza la GUI con los resultados de r_0
        # Por ejemplo, mostrar los resultados en un wx.TextCtrl
        self.app_ref.console.AppendText(f"{prefijo}: {resp}\n")

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

    
