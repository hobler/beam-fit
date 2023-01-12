import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
import numpy as np
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure

import beamfit as bf
import json


def increase_n():
    global n

    n += 1
    sumNP.increase_n()
    func_widget = FuncParamWidget(rt)
    param_widgets.append(func_widget)
    func_widget.set_start_values(sumNP)
    param_note.add(func_widget, text=str(n))
    # Change to new tab
    param_note.select(n-1)


def decrease_n():
    global n

    if n > 1:
        n -= 1
        param_note.forget(param_widgets.pop())
        sumNP.decrease_n()


def n_changed():
    if sumNP is None:
        n_spinbox.selection_clear()
        messagebox.showwarning(
            title="No measurement",
            message="No measurements loaded! Please load measurement data.")
        return

    if n > int(n_spinbox.get()):
        decrease_n()
    elif n < int(n_spinbox.get()):
        increase_n()


def print_fitted_params():
    """Prints the fitted parameters in the field in the bottom right"""
    cs = sumNP.get_cs()
    betas = sumNP.get_betas()
    sigmas = sumNP.get_sigmas()

    fitted_params_field.config(state="normal")
    fitted_params_field.delete(1.0, tk.END)
    for i in range(0, n):
        fitted_params_field.insert(tk.END,
                                   "Func " + str(i + 1) + ":\n"
                                   + "\tc: " + str(cs[i])
                                   + "\n\tsigma: " + str(sigmas[i])
                                   + "\n\tbeta: " + str(betas[i]) + "\n\n")
    fitted_params_field.config(state="disabled")


def use_fitted_params():
    if sumNP is None:
        messagebox.showwarning(
            title="No measurement",
            message="No measurements loaded! Please load measurement data.")
        return

    cs = sumNP.get_cs()
    sigmas = sumNP.get_sigmas()
    betas = sumNP.get_betas()

    for i, paramW in enumerate(param_widgets):
        if i != 0:
            paramW.c_param.set_value(cs[i])
        paramW.sigma_param.set_value(sigmas[i])
        paramW.beta_param.set_value(betas[i])


def fit():
    """Starts the fitting process when the fit button is pressed"""
    if sumNP is None:
        messagebox.showwarning(
            title="No measurement",
            message="No measurements loaded! Please load measurement data.")
        return
    try:
        for i, paramW in enumerate(param_widgets):
            if i != 0:
                if paramW.is_c_fixed():
                    sumNP.fix_c_i(i, paramW.get_c_value())
                else:
                    sumNP.unfix_c_i(i)
                    sumNP.set_startpoint_c_i(i, paramW.get_c_value())
            if paramW.is_beta_fixed():
                sumNP.fix_beta_i(i, paramW.get_beta_value())
            else:
                sumNP.unfix_beta_i(i)
                sumNP.set_startpoint_beta_i(i, paramW.get_beta_value())
            if paramW.is_sigma_fixed():
                sumNP.fix_sigma_i(i, paramW.get_sigma_value())
            else:
                sumNP.unfix_sigma_i(i)
                sumNP.set_startpoint_sigma_i(i, paramW.get_sigma_value())

        sumNP.fit()
        draw_pearson()
        print_fitted_params()
    except Exception as err:
        messagebox.showwarning(
            title="Wrong input",
            message="There was a problem with the parameter input.\n Error: "
                    + str(err))


def apply_graph_options():
    """Applies all graphic options for the graph """
    if int(semilog_lin_check.get()):
        ax.set_yscale("log")
    if int(legend_check.get()):
        ax.legend()
    if int(grid_check.get()):
        ax.grid()


def draw_init():
    """Initializes the graph """
    global ax, canvas, x, f_meas

    ax.cla()
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.plot(x, f_meas, marker="x", linestyle="None", label="$f^{exp}(x)$")
    apply_graph_options()
    canvas.draw()


def draw_pearson():
    """Draws the fitted function, measurement data and the single functions"""
    global ax, canvas, sumNP, f_meas, x, x_fine, n

    if sumNP is None:
        return

    cs = sumNP.get_cs()
    betas = sumNP.get_betas()
    sigmas = sumNP.get_sigmas()

    ax.cla()
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")

    ax.plot(x, f_meas, marker="x", linestyle="None", label="$f^{exp}(x)$")
    ax.plot(x_fine, bf.sum_n_pearson(x_fine, cs, betas, sigmas), label="$f(x)$")
    if int(single_func_check.get()):
        for k in range(0, n):
            ax.plot(x_fine,
                    cs[k] * bf.pearson_function_fast(x_fine,
                                                     betas[k],
                                                     sigmas[k]),
                    label="$f_{}(x)$".format(str(k + 1)),
                    linestyle="--")

    apply_graph_options()
    toolbar.update()
    canvas.draw()


def swap_semilog_lin():
    if int(semilog_lin_check.get()):
        ax.set_yscale("log")
    else:
        ax.set_yscale("linear")

    toolbar.update()
    canvas.draw()


def swap_legend():
    if int(legend_check.get()):
        ax.legend()
    else:
        ax.get_legend().remove()
    canvas.draw()


def swap_grid():
    if int(grid_check.get()):
        ax.grid(True)
    else:
        ax.grid(False)
    canvas.draw()


def open_x_options():
    """Opens a window to change the x values"""
    x_option = tk.Toplevel(rt)
    x_option.title("X value options")
    tk.Label(x_option, text="Start value:").grid(column=0, row=0)
    tk.Entry(x_option, textvariable=x_start).grid(column=1, row=0)
    tk.Label(x_option, text="End value:").grid(column=0, row=1)
    tk.Entry(x_option, textvariable=x_end).grid(column=1, row=1)
    tk.Label(x_option, text="Number of steps:").grid(column=0, row=2)
    tk.Entry(x_option, textvariable=x_num).grid(column=1, row=2)
    tk.Button(x_option,
              text="Save",
              command=change_x_options).grid(column=0, row=3, columnspan=2)


def change_x_options():
    global x_fine
    x_fine = np.linspace(float(x_start.get()),
                         float(x_end.get()),
                         int(x_num.get()))
    draw_pearson()


def save_function_data():
    """Opens a dialog to save the fitted function data in a csv file"""
    save_filename = filedialog.asksaveasfilename(
        filetypes=(('Comma seperated value', '.csv'),
                   ('All files', '.*')))
    if not save_filename:
        print("Aborted")
        return
    cs = sumNP.get_cs()
    betas = sumNP.get_betas()
    sigmas = sumNP.get_sigmas()
    data = np.column_stack((x_fine,
                            bf.sum_n_pearson(x_fine, cs, betas, sigmas)))
    np.savetxt(save_filename, data, delimiter=",")


def save_parameters():
    """Opens a dialog to save the fitted parameters in a csv file """
    if sumNP is None:
        messagebox.showwarning(
            title="No measurement",
            message="No measurements loaded! Please load measurement data.")
        return
    parameter_filename = filedialog.asksaveasfilename(
        filetypes=(('Comma seperated value', '.csv'),
                   ('All files', '.*')))
    if not parameter_filename:
        print("Aborted")
        return
    cs = sumNP.get_cs()
    betas = sumNP.get_betas()
    sigmas = sumNP.get_sigmas()
    parameter_data = np.column_stack((cs, betas, sigmas))

    np.savetxt(parameter_filename, parameter_data,
               delimiter=",", header="c,beta,sigma", comments="")


def load_measurement_data(event=None):
    """Opens a dialog to load in the measurement data """
    global x, f_meas, sumNP, n, x_fine
    measurement_filename = filedialog.askopenfilename(
        filetypes=(('Comma seperated value', '.csv'),
                   ('All files', '.*')))
    if not measurement_filename:
        print("Aborted")
        return

    if sumNP is not None:
        result = messagebox.askokcancel(title="Warning",
                                        message="All options which are not "
                                                "saved will be deleted! Do you "
                                                "want to continue?")
        if not result:
            return

    load_data = np.genfromtxt(measurement_filename, delimiter=",")
    x = load_data[:, 0]
    f_meas = load_data[:, 1]
    x_fine = np.linspace(np.min(x), np.max(x), np.size(x))
    x_start.set(str(np.min(x)))
    x_end.set(str(np.max(x)))
    x_num.set(str(np.size(x)))

    sumNP = bf.SumNPearson(f_meas, x)
    while n != 1:
        param_note.forget(param_widgets.pop())
        n -= 1
    n_spinbox.config(state="readonly")
    n_spin_value.set(1)
    fitted_params_field.config(state="normal")
    fitted_params_field.delete(1.0, tk.END)
    fitted_params_field.insert(tk.END, "No fitted params yet")
    fitted_params_field.config(state="disabled")
    param_widgets[0].beta_param.set_value(sumNP.start_beta)
    param_widgets[0].sigma_param.set_value(sumNP.start_sigma)
    param_widgets[0].sigma_param.set_fixed(False)
    param_widgets[0].beta_param.set_fixed(False)
    draw_init()


def save_json():
    """Saves all the parameter options in a json file """
    option_filename = filedialog.asksaveasfilename(
        filetypes=(('JavaScript Object Notation', '.json'),
                   ('All files', '.*')))
    if not option_filename:
        return
    func_params = [{"beta": {"value": param_widgets[0].get_beta_value(),
                             "fixed": param_widgets[0].is_beta_fixed()},
                    "sigma": {"value": param_widgets[0].get_sigma_value(),
                              "fixed": param_widgets[0].is_sigma_fixed()}}]

    for i in range(1, n):
        func_params.append(
            {"c": {"value": param_widgets[i].get_c_value(),
                   "fixed": param_widgets[i].is_c_fixed()},
             "beta": {"value": param_widgets[i].get_beta_value(),
                      "fixed": param_widgets[i].is_beta_fixed()},
             "sigma": {"value": param_widgets[i].get_sigma_value(),
                       "fixed": param_widgets[i].is_sigma_fixed()}})

    dict_output = {"functions": func_params}
    with open(option_filename, "w") as json_file:
        json.dump(dict_output, json_file, indent=1)


def load_json():
    """Loads and parses the parameter options out of a json file """
    if sumNP is None:
        messagebox.showwarning(
            title="No measurement",
            message="No measurements loaded! Please load measurement data.")
        return

    option_filename = filedialog.askopenfilename(
        filetypes=(('JavaScript Object Notation', '.json'),
                   ('All files', '.*')))
    if not option_filename:
        return

    with open(option_filename, "r") as json_file:
        dict_options = json.load(json_file)
    try:
        function_params = dict_options["functions"]
        while n != len(function_params):
            if n > len(function_params):
                decrease_n()
            else:
                increase_n()

        # Set the spinbox to the current n value to make it coherent
        n_spin_value.set(n)

        for i, params in enumerate(function_params):
            if i != 0:
                param_widgets[i].c_param.set_value(params["c"]["value"])
                param_widgets[i].c_param.set_fixed(params["c"]["fixed"])
            param_widgets[i].beta_param.set_value(params["beta"]["value"])
            param_widgets[i].beta_param.set_fixed(params["beta"]["fixed"])
            param_widgets[i].sigma_param.set_value(params["sigma"]["value"])
            param_widgets[i].sigma_param.set_fixed(params["sigma"]["fixed"])

        messagebox.showinfo(title="Load options",
                            message="Loaded options successfully")
    except Exception as err:
        messagebox.showwarning(
            title="Option file error",
            message="Could not read option file!\n Error:" + str(err))


class ParamWidget(ttk.Frame):
    """Widget which is used to get the value of a parameter and if its fixed"""
    def __init__(self, parent, param_name):
        ttk.Frame.__init__(self, parent)

        self.value_label = ttk.Label(self, text=param_name+str(": "))
        self.value = ttk.Entry(self)
        self.value_label.pack(side="left")
        self.value.pack(side="left")

        self.check_var = tk.IntVar()
        self.fix_check = ttk.Checkbutton(self,
                                         text=str("Fix ") + param_name,
                                         variable=self.check_var)
        self.fix_check.pack(side="left")

    def is_fixed(self):
        return self.check_var.get() == 1

    def get_value(self):
        return float(self.value.get())

    def set_value(self, value):
        self.value.delete(0, tk.END)
        self.value.insert(0, value)

    def set_fixed(self, is_fixed):
        self.check_var.set(is_fixed)


class FuncParamWidget(ttk.Frame):
    """Widget to combine the three parameters for a pearson function """
    def __init__(self, parent):
        ttk.Frame.__init__(self, parent)
        self.beta_param = ParamWidget(self, "beta")
        self.beta_param.pack(side="bottom")
        self.sigma_param = ParamWidget(self, "sigma")
        self.sigma_param.pack(side="bottom")
        self.c_param = ParamWidget(self, "c")
        self.c_param.pack(side="bottom")

    def is_c_fixed(self):
        return self.c_param.is_fixed()

    def get_c_value(self):
        return self.c_param.get_value()

    def is_beta_fixed(self):
        return self.beta_param.is_fixed()

    def get_beta_value(self):
        return self.beta_param.get_value()

    def is_sigma_fixed(self):
        return self.sigma_param.is_fixed()

    def get_sigma_value(self):
        return self.sigma_param.get_value()

    def set_start_values(self, sum_np):
        self.c_param.set_value(0)
        self.beta_param.set_value(sum_np.start_beta)
        self.sigma_param.set_value(sum_np.start_sigma)


rt = tk.Tk()
rt.wm_title("Beamfit")
rt.resizable(False, False)

# Create menus
menu = tk.Menu(rt)
rt.config(menu=menu)
rt.option_add('*tearOff', False)
file_menu = tk.Menu(menu)
graph_menu = tk.Menu(menu)
menu.add_cascade(label="File", menu=file_menu)
menu.add_cascade(label="Graph", menu=graph_menu)

# File menu
file_menu.add_command(label="Load measurement data",
                      command=load_measurement_data, accelerator="Ctrl+O")
file_menu.bind_all("<Control-o>", load_measurement_data)
file_menu.add_command(label="Load options", command=load_json)
file_menu.add_separator()
file_menu.add_command(label="Save fitted function", command=save_function_data)
file_menu.add_command(label="Save parameters", command=save_parameters)
file_menu.add_command(label="Save options", command=save_json)


# Graph menu
semilog_lin_check = tk.IntVar()
graph_menu.add_checkbutton(label="Semilog",
                           command=swap_semilog_lin,
                           variable=semilog_lin_check)
legend_check = tk.IntVar()
legend_check.set(1)
graph_menu.add_checkbutton(label="Legend",
                           command=swap_legend,
                           variable=legend_check)
grid_check = tk.IntVar()
graph_menu.add_checkbutton(label="Grid",
                           command=swap_grid,
                           variable=grid_check)
single_func_check = tk.IntVar()
graph_menu.add_checkbutton(label="Individual functions",
                           command=draw_pearson,
                           variable=single_func_check)

graph_menu.add_command(label="Change x values", command=open_x_options)


# Init var
x = None
x_fine = None
f_meas = None
n = 1
sumNP: bf.SumNPearson = None
x_start = tk.StringVar()
x_end = tk.StringVar()
x_num = tk.StringVar()

# Create figure
fig = Figure(figsize=(5, 4), dpi=100)
ax = fig.add_subplot()

# Add matplotlib inside the frame
graph_frame = ttk.Labelframe(rt, text="Graph")
canvas = FigureCanvasTkAgg(fig, master=graph_frame)

toolbar = NavigationToolbar2Tk(canvas, graph_frame)
toolbar.update()
toolbar.pack(side="bottom", padx=(5, 5))
canvas.get_tk_widget().pack(side="top", padx=(5, 5))
graph_frame.grid(column=0, row=0, columnspan=2, rowspan=4,
                 padx=(10, 10), pady=(10, 10))

# Add the number of func control inside the frame
n_frame = ttk.Frame(rt)
n_label = ttk.Label(n_frame, text="n: ")
n_label.pack(side="left")
n_spin_value = tk.IntVar()  # This is only needed to reset the spinbox
n_spinbox = tk.Spinbox(n_frame, from_=1, to=99, state="disabled",
                       highlightthickness=0,
                       command=n_changed,
                       textvariable=n_spin_value,
                       width=3)
n_spinbox.pack(side="left")
n_frame.grid(column=4, row=0)


# Add fit button inside the frame
buttons_frame = ttk.Frame(rt)
fit_bt = ttk.Button(buttons_frame, text='Fit', command=fit)
fit_bt.pack(side="left", padx=(5, 5))

use_fitted_params_button = ttk.Button(buttons_frame,
                                      text='Use fitted parameters',
                                      command=use_fitted_params)
use_fitted_params_button.pack(side="right", padx=(5, 5))
buttons_frame.grid(column=4, row=3, pady=(5, 5))

# Add function params
param_frame = ttk.LabelFrame(rt, text="Initial Parameters",
                             width=340, height=110)
param_frame.pack_propagate(False)
param_note = ttk.Notebook(param_frame)
param_note.pack(anchor="n", fill="both", expand=True)
param_frame.grid(column=4, row=1)
param_widgets = [FuncParamWidget(rt)]
param_widgets[0].c_param.destroy()
param_note.add(param_widgets[0], text=str(n))

# Add output of the fitted parameters for quick access
fitted_params_frame = ttk.Labelframe(rt, text="Fitted parameters")
fitted_params_scroll = ttk.Scrollbar(fitted_params_frame)
fitted_params_field = tk.Text(fitted_params_frame,
                              width=40, height=10, cursor="arrow",
                              yscrollcommand=fitted_params_scroll.set)
fitted_params_scroll.config(command=fitted_params_field.yview)
fitted_params_field.insert(tk.END, "No fitted params yet")
fitted_params_field.config(state="disabled")
fitted_params_field.pack(side="left")
fitted_params_scroll.pack(side="right", fill="y")
fitted_params_frame.grid(column=4, row=2, padx=(10, 10))

rt.mainloop()
