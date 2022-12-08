import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import numpy as np
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure

import beamfit as bf


def increase_n():
    global n
    n += 1
    n_text.set("n: " + str(n))
    sumNP.increase_n()
    func_widget = FuncParamWidget(rt)
    param_widgets.append(func_widget)
    param_note.add(func_widget, text=str(n))
    # Change to new tab
    param_note.select(n-1)


def decrease_n():
    global n
    if n > 1:
        n -= 1
        n_text.set("n: " + str(n))
        param_note.forget(param_widgets.pop())
        sumNP.decrease_n()


def print_fitted_params():
    cs = sumNP.get_cs()
    betas = sumNP.get_betas()
    sigmas = sumNP.get_sigmas()

    fitted_params_field.config(state="normal")
    fitted_params_field.delete(1.0, tk.END)
    for i in range(0, n):
        fitted_params_field.insert(tk.END,
                                   "Func " + str(i + 1) + ":\n"
                                   + "\tc: " + str(cs[i])
                                   + "\n\tbeta: " + str(betas[i])
                                   + "\n\tsigma: " + str(sigmas[i]) + "\n\n")
    fitted_params_field.config(state="disabled")


def fit():
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


def draw_pearson():
    global ax, canvas, sumNP, f_meas, x, x_fine, n
    cs = sumNP.get_cs()
    betas = sumNP.get_betas()
    sigmas = sumNP.get_sigmas()

    ax.cla()
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    if int(semilog_lin_check.get()):
        ax.set_yscale("log")

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

    if int(legend_check.get()):
        ax.legend()
    if int(grid_check.get()):
        ax.grid()
    canvas.draw()


def swap_semilog_lin():
    if int(semilog_lin_check.get()):
        ax.set_yscale("log")
    else:
        ax.set_yscale("linear")
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


def save_function_data():
    save_filename = filedialog.asksaveasfile(
        filetypes=(('Comma seperated value', '.csv'),
                   ('All files', '.*')))
    cs = sumNP.get_cs()
    betas = sumNP.get_betas()
    sigmas = sumNP.get_sigmas()
    data = np.column_stack((x_fine,
                            bf.sum_n_pearson(x_fine, cs, betas, sigmas)))
    np.savetxt(save_filename.name, data, delimiter=",")


def save_parameters():
    parameter_filename = filedialog.asksaveasfile(
        filetypes=(('Comma seperated value', '.csv'),
                   ('All files', '.*')))
    cs = sumNP.get_cs()
    betas = sumNP.get_betas()
    sigmas = sumNP.get_sigmas()
    parameter_data = np.column_stack((cs, betas, sigmas))
    """
    So apparently if I use the File and not the name of the file the last
    20 or so lines are not saved... I don't know why I guess it has something to
    do with the filedialog but I don't know. Could write my own file writer and
    look into it....
    """
    np.savetxt(parameter_filename.name, parameter_data,
               delimiter=",", header="c,beta,sigma", comments="")


class ParamWidget(ttk.Frame):
    def __init__(self, parent, param_name):
        ttk.Frame.__init__(self, parent)

        self.value_label = ttk.Label(self, text=param_name+str(" value:"))
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


class FuncParamWidget(ttk.Frame):
    def __init__(self, parent):
        ttk.Frame.__init__(self, parent)

        self.c_param = ParamWidget(self, "c")
        self.c_param.pack(side="top", fill="y")
        self.beta_param = ParamWidget(self, "beta")
        self.beta_param.pack(side="top", fill="y")
        self.sigma_param = ParamWidget(self, "sigma")
        self.sigma_param.pack(side="top", fill="y")

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
file_menu.add_command(label="Save fitted function", command=save_function_data)
file_menu.add_command(label="Save parameters", command=save_parameters)

# Graph menu
semilog_lin_check = tk.StringVar()
semilog_lin_check.set("0")
graph_menu.add_checkbutton(label="line/semilog",
                           command=swap_semilog_lin,
                           variable=semilog_lin_check)
legend_check = tk.StringVar()
legend_check.set("0")
graph_menu.add_checkbutton(label="Legend",
                           command=swap_legend,
                           variable=legend_check)
grid_check = tk.StringVar()
grid_check.set("0")
graph_menu.add_checkbutton(label="Grid",
                           command=swap_grid,
                           variable=grid_check)
single_func_check = tk.StringVar()
single_func_check.set("0")
graph_menu.add_checkbutton(label="Show single functions",
                           command=draw_pearson,
                           variable=single_func_check)

# Create figure
fig = Figure(figsize=(5, 4), dpi=100)
ax = fig.add_subplot()
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
x = np.arange(-5, 5, 0.5)
x_fine = np.arange(-5, 5, 0.01)
n = 1
f_meas = bf.create_measurement(x, d=0.5, sigma1=1, sigma2=1)
ax.plot(x, f_meas, marker="x", linestyle="None", label="$f^{exp}(x)$")
sumNP = bf.SumNPearson(f_meas, x)

# Add matplotlib inside the frame
graph_frame = ttk.Labelframe(rt, text="Graph")
canvas = FigureCanvasTkAgg(fig, master=graph_frame)

toolbar = NavigationToolbar2Tk(canvas, graph_frame, pack_toolbar=False)
toolbar.update()
toolbar.pack(side="bottom")
canvas.get_tk_widget().pack(side="top", padx=(5, 5))
graph_frame.grid(column=0, row=0, columnspan=2, rowspan=3, padx=(10, 10))

# Add the number of func control inside the frame
n_text = tk.StringVar()
n_text.set("n: 1")
n_frame = ttk.Frame(rt)
n_label = ttk.Label(n_frame, text="n: 1", textvariable=n_text)
n_label.pack(side="left")
n_decrease_bt = ttk.Button(n_frame, text='<', command=decrease_n)
n_decrease_bt.pack(side="left")
n_increase_bt = ttk.Button(n_frame, text='>', command=increase_n)
n_increase_bt.pack(side="left")
n_frame.grid(column=4, row=0)

# Add fit button inside the frame
fit_bt = ttk.Button(rt, text='Fit', command=fit)
fit_bt.grid(column=4, row=3, pady=(5, 5))

# Add function params
param_note = ttk.Notebook(rt)
param_note.grid(column=4, row=1)
param_widgets = [FuncParamWidget(rt)]
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






