import beamfit as bf
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
import tkinter as tk


def increase_n():
    global n
    n += 1
    n_text.set("n: " + str(n))
    sumNP.increase_n()
    print("Increase")


def decrease_n():
    global n
    if n > 1:
        n -= 1
        n_text.set("n: " + str(n))
        sumNP.decrease_n()
        print("Decrease")


def fit():
    sumNP.fit()
    draw_pearson()


def draw_pearson():
    global ax, canvas, sumNP, f_meas, x, x_fine, n
    cs = sumNP.get_cs()
    betas = sumNP.get_betas()
    sigmas = sumNP.get_sigmas()
    ax.cla()
    ax.plot(x, f_meas, marker="x", linestyle="None", label="$f^{exp}(x)$")
    ax.plot(x_fine, bf.sum_n_pearson(x_fine, cs, betas, sigmas), label="$f(x)$")
    for k in range(0, n):
        ax.plot(x_fine,
                cs[k] * bf.pearson_function_fast(x_fine,
                                                 betas[k],
                                                 sigmas[k]),
                label="$f_{}(x)$".format(str(k + 1)),
                linestyle="--")
    ax.legend()
    canvas.draw()


rt = tk.Tk()
rt.wm_title("Beamfit")

# Create figure
fig = Figure(figsize=(5, 4), dpi=100)
ax = fig.add_subplot()
x = np.arange(-4, 4, 0.5)
x_fine = np.arange(-4, 4, 0.01)
n = 1
f_meas = bf.create_measurement(x, d=0.8, sigma1=2, sigma2=0.5)
sumNP = bf.SumNPearson(f_meas, x)
sumNP.fit()

# Add matplotlib inside the frame
canvas = FigureCanvasTkAgg(fig, master=rt)
draw_pearson()

toolbar = NavigationToolbar2Tk(canvas, rt, pack_toolbar=False)
toolbar.update()
toolbar.grid(column=0, row=2, columnspan=2)
canvas.get_tk_widget().grid(column=0, row=0, columnspan=2, rowspan=2)

# Add the number of func control inside the frame
n_text = tk.StringVar()
n_text.set("n: 1")
n_label = tk.Label(rt, text="n: 1", textvariable=n_text)
n_label.grid(column=3, row=0)
n_increase_bt = tk.Button(rt, text='>', command=increase_n)
n_increase_bt.grid(column=5, row=0)
n_decrease_bt = tk.Button(rt, text='<', command=decrease_n)
n_decrease_bt.grid(column=4, row=0)

# Add fit button inside the frame
fit_bt = tk.Button(rt, text='Fit', command=fit)
fit_bt.grid(column=4, row=2)

rt.mainloop()






