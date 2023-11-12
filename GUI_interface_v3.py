# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 13:45:43 2023

@author: lap_PJGV
"""

import tkinter as tk
from tkinter import messagebox, filedialog, simpledialog
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import optimize
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, basinhopping
from scipy.optimize import least_squares
import code
import sys
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
import warnings
warnings.filterwarnings("ignore")
from tkinter import Tk, ttk
import timeit


class TextRedirector:
    def __init__(self, widget):
        self.widget = widget

    def write(self, str):
        self.widget.insert(tk.END, str)
        self.widget.see(tk.END)  # Auto-scroll to the end

    def flush(self):
        pass

# Main application window
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Data Analysis Tool")
        
        self.vars_columnas = {} 

        # Configuración de dos columnas en la ventana principal
        self.grid_columnconfigure(0, weight=1)  # Columna izquierda para controles
        self.grid_columnconfigure(1, weight=2)  # Columna derecha para gráficas y output

        # Columna Izquierda: Controles y Opciones
        #left_frame = tk.Frame(self)
        #left_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
       
        self.left_frame = tk.Frame(self)
        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # File Selection Frame
        file_frame = tk.Frame(self.left_frame)  # Moved inside left_frame
        file_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        self.btn_select_file = tk.Button(file_frame, text="Select Excel File", command=self.select_file)
        self.btn_select_file.pack(side='left')
        self.lbl_file = tk.Label(file_frame, text="No file selected")
        self.lbl_file.pack(side='left')

        # Sheet Names Frame
        # Spectra sheet
        sheet_frame = tk.Frame(self.left_frame)  # Moved inside left_frame
        sheet_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        self.lbl_sheet_spectra = tk.Label(sheet_frame, text="Spectra Sheet Name:")
        self.lbl_sheet_spectra.pack(side='left')
        
        #sheet_spectra = tk.Frame(self.left_frame)
        #sheet_spectra.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        self.entry_sheet_spectra = tk.Entry(sheet_frame)
        self.entry_sheet_spectra.pack(side='left')
        self.entry_sheet_spectra.insert(0, 'datos_titulacion')
        
        # concentrations sheet
        sheet_conc = tk.Frame(self.left_frame)
        sheet_conc.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        self.lbl_sheet_conc = tk.Label(sheet_conc, text="Concentration Sheet Name:")
        self.lbl_sheet_conc.pack(side='left')

        self.entry_sheet_conc = tk.Entry(sheet_conc)
        self.entry_sheet_conc.pack(side='left')
        self.entry_sheet_conc.insert(0, 'conc')

        sheet_model = tk.Frame(self.left_frame)
        sheet_model.grid(row=3, column=0, sticky="ew", padx=5, pady=5)

        self.lbl_sheet_model = tk.Label(sheet_model, text="Model Sheet Name:")
        self.lbl_sheet_model.pack(side='left')

        self.entry_sheet_model = tk.Entry(sheet_model)
        self.entry_sheet_model.pack(side='left')
        self.entry_sheet_model.insert(0, 'modelo')

        # Configurar frame para nombres de columnas y checkboxes
        self.columns_names_frame = tk.Frame(self.left_frame)
        self.columns_names_frame.grid(row=5, column=0, sticky="ew")

        # Configurar grid para manejar múltiples columnas
        self.columns_names_frame.grid_columnconfigure(1, weight=1)  # Para checkboxes

        # Añadir texto
        self.lbl_columns = tk.Label(self.columns_names_frame, text="Columns names: ")
        self.lbl_columns.grid(row=0, column=0, sticky="w")
        
        EV_val = tk.Frame(self.left_frame)
        EV_val.grid(row=6, column=0, sticky="ew", padx=5, pady=5)

        self.lbl_EV = tk.Label(EV_val, text="EV:")
        self.lbl_EV.pack(side='left')
        
        self.entry_EV = tk.Entry(EV_val)
        self.entry_EV.pack(side='left')
        self.entry_EV.insert(0, 0)

        # Ajustes de Modelo Frame
        ajustes_frame = tk.Frame(self.left_frame)  # Moved inside left_frame
        ajustes_frame.grid(row=7, column=0, sticky="ew", padx=5, pady=5)

        ajustes_label = tk.Label(ajustes_frame, text='Ajustes al modelo')
        ajustes_label.pack(anchor='w')

        self.ajustes_modelo = tk.StringVar(value='Libre')  # Setting a default value
        paso_a_paso_rb = tk.Radiobutton(ajustes_frame, text='Paso a Paso', variable=self.ajustes_modelo, value='Paso a Paso')
        paso_a_paso_rb.pack(anchor='w')
        libre_rb = tk.Radiobutton(ajustes_frame, text='Libre', variable=self.ajustes_modelo, value='Libre')
        libre_rb.pack(anchor='w')
        no_cooperativo_rb = tk.Radiobutton(ajustes_frame, text='No cooperativo', variable=self.ajustes_modelo, value='No cooperativo')
        no_cooperativo_rb.pack(anchor='w')

        # Optimizer Selection Frame
        optimizador_frame = tk.Frame(self.left_frame)  # Moved inside left_frame
        optimizador_frame.grid(row=8, column=0, sticky="ew", padx=5, pady=5)

        optimizador_label = tk.Label(optimizador_frame, text='Seleccione optimizador')
        optimizador_label.pack(anchor='w')

        self.optimizador = tk.StringVar(value='Powell')  # Setting a default value
        powell_rb = tk.Radiobutton(optimizador_frame, text='Powell', variable=self.optimizador, value='Powell')
        powell_rb.pack(anchor='w')
        nelder_mead_rb = tk.Radiobutton(optimizador_frame, text='Nelder-Mead', variable=self.optimizador, value='Nelder-Mead')
        nelder_mead_rb.pack(anchor='w')
        trust_constr_rb = tk.Radiobutton(optimizador_frame, text='Trust-Constr', variable=self.optimizador, value='Trust-Constr')
        trust_constr_rb.pack(anchor='w')
        differential_evolution_rb = tk.Radiobutton(optimizador_frame, text='Differential Evolution', variable=self.optimizador, value='Differential Evolution')
        differential_evolution_rb.pack(anchor='w')
        basinhopping_rb = tk.Radiobutton(optimizador_frame, text='Basinhopping', variable=self.optimizador, value='Basinhopping')
        basinhopping_rb.pack(anchor='w')
                
        # Columna Derecha: Canvas para Gráficas y Prompt de Output
        right_frame = tk.Frame(self)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        # Configuración de peso para columnas y filas en right_frame
        right_frame.grid_columnconfigure(0, weight=1)
        right_frame.grid_rowconfigure(0, weight=1)  # Para el canvas
        right_frame.grid_rowconfigure(1, weight=1)  # Para la consola de Python

        # Canvas para las gráficas
        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Scrollbar para el Canvas
        scrollbar = ttk.Scrollbar(right_frame, orient="vertical", command=self.canvas.get_tk_widget().yview)
        scrollbar.grid(row=0, column=1, sticky="ns")

        # Button to process the data
        self.btn_process_data = tk.Button(right_frame, text="Process Data", command=self.process_data)
        self.btn_process_data.grid(row=2, column=0, sticky="ew", padx=5, pady=5)

        # Create a Text widget for console output with specific dimensions
        self.output_text = tk.Text(right_frame, height=33, width=50, bg="black", fg="white")
        self.output_text.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        # Redirect stdout
        sys.stdout = TextRedirector(self.output_text)

        # Create an instance of TkinterConsoleOutput and redirect sys.stdout
        #self.console_output = TextRedirector(self.output_text)
        #sys.stdout = self.console_output

        # Save Button (move to right_frame for better organization)
        self.btn_save_file = tk.Button(right_frame, text="Save Results to Excel", command=self.save_results)
        self.btn_save_file.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        
        # Configurar una función para actualizar la ventana del GUI periódicamente
        self.after(1, self.update_gui)

        # Crear una instancia del intérprete de Python
        self.python_interpreter = code.InteractiveConsole(locals=globals())

    #def select_file(self):
    #    file_path = filedialog.askopenfilename(initialdir="/", title="Select file",
    #                                           filetypes=(("Excel files", "*.xlsx"), ("all files", "*.*")))
    #    if file_path:
    #        self.lbl_file.config(text=file_path)
    #       self.file_path = file_path
    
    def select_file(self):
        file_path = filedialog.askopenfilename(initialdir="/", title="Select file",
                                            filetypes=(("Excel files", "*.xlsx"), ("all files", "*.*")))
        if file_path:
            self.lbl_file.config(text=file_path)
            self.file_path = file_path

            # Leer el DataFrame desde el archivo seleccionado
            # Suponiendo que tus datos están en la primera hoja, ajusta según sea necesario
            df = pd.read_excel(file_path, sheet_name=self.entry_sheet_conc.get())

            # Crear casillas de verificación para cada columna
            self.create_checkboxes(df.columns)
            
    def configure_canvas(self, event):
        # Configurar el área de desplazamiento del Canvas
        self.graph_canvas.configure(scrollregion=self.graph_canvas.bbox("all"))

    def show_figure(self, figure):
        # Agregar la figura al Frame interior y actualizar el Canvas
        self.figures.append(figure)
        canvas = FigureCanvasTkAgg(figure, master=self.graph_frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        canvas.draw()
        self.configure_canvas(None)
        
    def update_gui(self):
        # Esta función se llama periódicamente para actualizar el GUI
        # Puedes personalizar cómo deseas que se actualice la ventana aquí

        # Por ejemplo, puedes hacer que se desplace automáticamente hacia la parte inferior del Text widget
        self.output_text.see(tk.END)

        # Llamar a esta función nuevamente después de un cierto intervalo
        self.after(10, self.update_gui)
            
    def save_results(self):
        # Placeholder for save functionality
        file_path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                                 filetypes=[("Excel files", "*.xlsx")])
        if file_path:
            messagebox.showinfo("Information", f"Results saved to {file_path}.")
    
    def create_checkboxes(self, column_names):
        # Limpiar checkboxes anteriores si existen
        for widget in self.columns_names_frame.winfo_children():
            widget.destroy()

        # Configurar grid para manejar múltiples columnas
        for i in range(len(column_names) + 1):
            self.columns_names_frame.grid_columnconfigure(i, weight=1)

        # Añadir texto
        self.lbl_columns = tk.Label(self.columns_names_frame, text="Columns names: ")
        self.lbl_columns.grid(row=0, column=0, sticky="w")

        # Crear los checkboxes dentro del mismo frame
        self.vars_columnas = {col: tk.BooleanVar() for col in column_names}
        for i, (col, var) in enumerate(self.vars_columnas.items(), start=1):
            checkbox = tk.Checkbutton(self.columns_names_frame, text=col, variable=var)
            checkbox.grid(row=0, column=i, sticky="")


    def on_checkbox_select(self, name, var):
        if var.get():
            if name not in self.selected_columns:
                self.selected_columns.append(name)
        else:
            if name in self.selected_columns:
                self.selected_columns.remove(name)
    
    def figura(self, x, y, mark, ylabel, xlabel):
        plt.plot(x, y, mark)
        plt.ylabel(ylabel, size = "xx-large")
        plt.xlabel(xlabel, size = "xx-large")
        plt.xticks(size = "large")
        plt.yticks(size = "large")
        plt.show()
    
    def figura2(self, x, y, y2, mark1, mark2, ylabel, xlabel, alpha):
        plt.plot(x, y, mark1, alpha)
        plt.plot(x, y2, mark2)
        plt.ylabel(ylabel, size = "xx-large")
        plt.xlabel(xlabel, size = "xx-large")
        plt.xticks(size = "large")
        plt.yticks(size = "large")
        plt.show()

    def figura(self, x, y, mark, ylabel, xlabel):
        fig = Figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(x, y, mark)
        ax.set_ylabel(ylabel, size="xx-large")
        ax.set_xlabel(xlabel, size="xx-large")
        ax.tick_params(axis='both', which='major', labelsize='large')

        canvas = FigureCanvasTkAgg(fig, master=self.right_frame)  # Asumiendo que right_frame es el contenedor para tus gráficas
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def figura2(self, x, y, y2, mark1, mark2, ylabel, xlabel, alpha):
        fig = Figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(x, y, mark1, alpha=alpha)
        ax.plot(x, y2, mark2)
        ax.set_ylabel(ylabel, size="xx-large")
        ax.set_xlabel(xlabel, size="xx-large")
        ax.tick_params(axis='both', which='major', labelsize='large')

        canvas = FigureCanvasTkAgg(fig, master=self.right_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


    def process_data(self):
        # Placeholder for the actual data processing
        # Would call the functions from the provided script and display output

        print("process_data iniciada")
        
        n_archivo = self.file_path
        datos = n_archivo 
        spec_entry = self.entry_sheet_spectra.get()
        conc_entry = self.entry_sheet_conc.get()
        spec = pd.read_excel(datos, spec_entry, header=0, index_col=0)
        #concentracion = pd.read_excel(datos,conc_entry, header=0)
        # Extraer datos de esas columnas
        concentracion = pd.read_excel(self.file_path, conc_entry, header=0)

        nombres_de_columnas = concentracion.columns

        # Obtener los nombres de las columnas seleccionadas
        columnas_seleccionadas = [col for col, var in self.vars_columnas.items() if var.get()]
        
        C_T = concentracion[columnas_seleccionadas].to_numpy()
        G = C_T[:,1]
        H = C_T[:,0]
        nc = len(C_T)
        n_comp = len(C_T.T)
        nw = len(spec)
        nm = spec.index.to_numpy()
        
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
        
        self.figura(range(0, nc), np.log10(s), "o", "log(EV)", "# de autovalores")      
        self.figura2(G, np.log10(ev_s0), np.log10(ev_s10), "k-o", "b:o", "log(EV)", "[G], M", 1)

        #self.show_figure(F1)
        #self.show_figure(F2)
            
        EV = int(self.entry_EV.get())

        if EV == 0:
            EV = nc

        print(EV)      

        Y = u[:,0:EV] @ np.diag(s[0:EV:]) @ v[0:EV:]
        Y_svd = pd.DataFrame(Y, index = [list(nm)])
        C_T = pd.DataFrame(C_T)
        
        try:
            modelo = pd.read_excel(self.file_path, self.entry_sheet_model.get(), header=0, index_col=0)
            print(modelo)
            
        except:
            #m = int(input("Indique el coeficiente estequiométrico para el receptor : ", ))
            m = simpledialog.askinteger("Input", "Indique el coeficiente estequiométrico para el receptor:")
            #n = int(input("Indique el coeficiente estequiométrico para el huesped : ", ))
            n = simpledialog.askinteger("Input", "Indique el coeficiente estequiométrico para el huesped:")
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
            print(modelo)
        
        modelo = np.array(modelo)

        n_K = len(modelo.T) - n_comp #- 1
        
        if n_K == 1:
            k_e = float(simpledialog.askinteger("K", "Indique un valor estimado para la constante de asociación:"))
        else:
            k_e = [] #[1., 1.]
            for i in range(n_K):
                ks = "K" + str(i+1) 
                i = float(simpledialog.askinteger(ks, "Indique un valor estimado para esta constante de asociación:"))
                print(ks + ":", i)
                k_e.append(i)
        
        k = np.array([k_e])
        k_ini = k
        k = np.ravel(k)
                
        def concentraciones(K, args = (C_T, modelo)):
        
            """
            Calcula las concentraciones para una serie de reacciones químicas utilizando el método de Levenberg-Marquardt.
        
            Argumentos:
            K -- una lista de constantes de equilibrio para cada reacción química.
            C_T -- una matriz de dimensiones (n_reacciones, n_componentes) que contiene las concentraciones totales de cada componente en cada reacción.
            modelo -- una matriz de dimensiones (n_componentes, n_reacciones) que contiene los coeficientes estequiométricos de cada componente en cada reacción.
            max_iter -- el número máximo de iteraciones que se permiten antes de detenerse (por defecto 2000).
            tol -- la tolerancia para considerar que el cálculo ha convergido (por defecto 1e-15).
            lmbda -- el parámetro de Levenberg-Marquardt (por defecto 0.01).
        
            Devuelve:
            Un array de dimensiones (n_reacciones, n_componentes-2) que contiene las concentraciones de cada componente para cada reacción.
            """
            ctot = np.array(C_T)
            n_reacciones, n_componentes = ctot.shape
            pre_ko = np.zeros(n_componentes)
            K = np.concatenate((pre_ko, K))
            #K_1 = np.array([K[2] - np.log10(4)])
            #K = np.concatenate((K, K_1))
            K = np.cumsum(K)
        
            #K[:-1] = np.cumsum(K[:-1])
            K = 10**K
            
            nspec = len(K)
        
            def calcular_concentraciones(ctot_i, c_guess):
                def residuals(c):
                    c_spec = np.prod(np.power(np.tile(c, (nspec, 1)).T, modelo), axis=0) * K
                    c_tot_cal = np.sum(modelo * np.tile(c_spec, (n_componentes, 1)), axis=1)
                    d = ctot_i - c_tot_cal
                    return d
                
                def jacobian(c):
                    c_spec = np.prod(np.power(np.tile(c, (nspec, 1)).T, modelo), axis=0) * K
                    jacobian_mat = np.empty((n_componentes, n_componentes))
                    for j in range(n_componentes):
                        for h in range(n_componentes):
                            jacobian_mat[j, h] = np.sum(modelo.T[:, j] * modelo.T[:, h] * c_spec) / c[j]
                    
                    J = np.dot(np.linalg.pinv(-jacobian_mat), np.diagflat(c_guess))
                    return J
                
                c_guess = least_squares(residuals, c_guess, jac=jacobian, method='lm', xtol=1e-8).x
                
                c_spec = np.prod(np.power(np.tile(c_guess, (nspec, 1)).T, modelo), axis=0) * K
                return c_guess, c_spec
        
            c_calculada = np.zeros((n_reacciones, nspec))
            for i in range(n_reacciones):
                c_guess = np.ones(n_componentes) * 1e-10
                c_guess, c_spec = calcular_concentraciones(ctot[i], c_guess)
                c_calculada[i] = c_spec
            
            C = np.delete(c_calculada, [1], axis = 1)
            return C, c_calculada
        
        
        """
        Y = (M, W)
        C = (M, N)
        A = (N, W)
        
        U = (M, N)
        V = (N, W)
        S = (N, N)
        """    
        
        V = u[:,0:(EV)].T
        S = np.diag(s[0:(EV):])
        U = v[0:(EV):].T
           
        # Implementing the abortividades function
        def abortividades(k, Y):
            C, Co = concentraciones(k)  # Assuming the function concentraciones returns C and Co
            A = np.linalg.pinv(C) @ Y.T
            return np.all(A >= 0)
        
        def f_m2(k):
            C = concentraciones(k)[0]
            y_c = U @ S
            r = C @ np.linalg.pinv(C) @ y_c - y_c    
            rms = np.sum(np.square(r))
            print(f"f(x): {rms}")
            print(f"x: {k}")
            return rms, r
        
        # Modifying f_m to use abortividades
        def f_m(k):
            C = concentraciones(k)[0]
            y_c = U @ S
            r = C @ np.linalg.pinv(C) @ y_c - y_c    
            rms = np.sqrt(np.mean(np.square(r)))
            print(f"f(x): {rms}")
            print(f"x: {k}")
            return rms
        
        bounds = [(-20, 20)]*len(k.T) #Bounds(0, 1e15, keep_feasible=(True)) #
        
        # Registrar el tiempo de inicio
        inicio = timeit.default_timer()
        
        optimizer = "powell" 
        r_0 = optimize.minimize(f_m, k, method=optimizer)
        
        # Applying the callback to differential_evolution
        # =============================================================================
        # r_0 = differential_evolution(
        #     f_m, 
        #     bounds, 
        #     x0=r_0.x, 
        #     strategy='best1bin', 
        #     maxiter=1000, 
        #     popsize=15, 
        #     tol=0.01,
        #     mutation=(0.5, 1), 
        #     recombination=0.7,
        #     init='latinhypercube', 
        #     #callback=abortividades_callback
        # )
        # =============================================================================
        
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
            C = concentraciones(k)[0]
            y_c = U @ S
            r = C @ np.linalg.pinv(C) @ y_c - y_c
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
        cov_matrix = SER**2 * np.linalg.pinv(jacobian @ jacobian.T)
        
        # Calcular el error estándar de las constantes de asociación
        SE_k = np.sqrt(np.diag(cov_matrix))
        
        # Calcular el error porcentual
        error_percent = (SE_k / np.abs(k)) * 100
        
        C, Co = concentraciones(k)
        
        self.figura(G/np.max(H), C, ":o", "[Especies], M", "[G]/[H], M")
                
        Q, R = np.linalg.qr(C)
        y_cal = Q @ Q.T @ Y.T
        #y_cal = C @ np.linalg.pinv(C, rcond=1e-20) @ Y.T
        
                
        plt.plot(nm, Y, "k", alpha = 0.5)
        plt.plot(nm, y_cal.T, "k:")
        plt.ylabel("Y observada (u. a.)", size = "xx-large")
        plt.xlabel("$\lambda$ (nm)", size = "xx-large")
        plt.xticks(size = "large")
        plt.yticks(size = "large")
        plt.show()
                
        ssq, r0 = f_m2(k)
        rms = f_m(k)
        
        A = np.linalg.pinv(C) @ Y.T 
        
        self.figura(nm, A.T, "-", "Epsilon (u. a.)", "$\lambda$ (nm)")
                
        lof = (((sum(sum((r0**2))) / sum(sum((Y**2)))))**0.5) * 100
        MAE = np.sqrt((sum(sum(r0**2)) / (nw - len(k))))
        dif_en_ct = round(max(100 - (np.sum(C, 1) * 100 / max(H))), 2)
        
        # 1. Calcular la varianza de los residuales
        residuals_array = residuals(k)
        var_residuals = np.var(residuals_array)
        
        # 2. Calcular la varianza de los datos experimentales
        var_data_original = np.var(Y)
        
        # 3. Calcular covfit
        covfit = var_residuals / var_data_original
        
        print("="*50)
        print("RMS: ",rms)
        print("Falta de ajuste (%): ",lof)
        print("Error absoluto medio: ",MAE)
        print("Constante de asociación :", k)
        print("Error estándar de las constantes de asociación:", SE_k)
        print("Error porcentual de las constantes de asociación:", error_percent)
        print("diferencia en C total (%): ", dif_en_ct)
        print("covfit: ", covfit)
        print("="*50)
        
        nombres = [f"k{i}" for i in range(1, len(k)+1)]
        k_nombres = [f"{n}" for n in nombres]
        
        K = np.array([k, error_percent]).T
        
        modelo = pd.DataFrame(modelo)
        C = pd.DataFrame(C)
        Co = pd.DataFrame(Co)
        k = pd.DataFrame(K, index = [k_nombres])
        k_ini = pd.DataFrame(k_ini.T, index = [k_nombres])
        A = pd.DataFrame(A.T, index = [list(nm)])
        phi = pd.DataFrame(y_cal.T, index = [list(nm)])
        cov_matrix = covfit
        
        stats = np.array([rms, lof, MAE, dif_en_ct, EV, cov_matrix, optimizer])
        stats = pd.DataFrame(stats, index= ["RMS", "Falta de ajuste (%)",\
                                            "Error absoluto medio", "Diferencia en C total (%)", "# Autovalores", "covfit", "optimizer"])
        
        # Generar nombres de columnas
        num_columns_C = len(C.columns)
        column_names_C = [f"sp_{i}" for i in range(1, num_columns_C + 1)]
        
        num_columns_Co = len(Co.columns)
        column_names_Co = [f"sp_{i}" for i in range(1, num_columns_Co + 1)]
        
        num_columns_yobs = len(Y_svd.columns)
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
        Y_svd.columns = column_names_yobs
        phi.columns = column_names_ycal
        A.columns = column_names_A
        k.columns = column_names_k
        k_ini.columns = column_names_k_ini
        stats.columns = column_names_stats
        C_T.columns = column_names_ct
            
        def save_file():
            root = Tk()
            root.withdraw()
            file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
            with pd.ExcelWriter(file_path) as writer:
                modelo.to_excel(writer, sheet_name="modelo")
                C.to_excel(writer, sheet_name="c_especies")
                Co.to_excel(writer, sheet_name="especies_totales")
                concentracion.to_excel(writer, sheet_name="C_totales")
                A.to_excel(writer, sheet_name="A_calculada", index_label = "nm", index = True)
                k.to_excel(writer, sheet_name="Constantes_de_asociación")
                k_ini.to_excel(writer, sheet_name="Constantes_iniciales")
                phi.to_excel(writer, sheet_name="Y_cal", index_label = "nm", index = True)
                Y_svd.to_excel(writer, sheet_name= "Y_svd", index_label = "nm", index = True)
                stats.to_excel(writer, sheet_name="Estadísticos")
                
        #save_file()
        
        
        self.display_results()

        
# Run the application
if __name__ == "__main__":
    app = App()
    app.mainloop()
    