# BeamFit
`BeamFit` is a GUI based program to fit a Sum of Pearson functions to a one 
dimensional intensity distribution of an ion beam.

## Requirements
`BeamFit` requires the following to run:

* [NumPy](https://numpy.org/)
* [Matplotlib](https://matplotlib.org/)
* [SciPy](https://scipy.org/)

## Startup
To start the `BeamFit` GUI, clone it locally and run it with a Python 3
interpreter like so:

    $ git clone https://github.com/hobler/beam-fit
    $ cd beam-fit/beamfit
    $ python3 beamfit_gui.py

## Usage
### How to start
To start working with `BeamFit` a measurement data file has to be loaded up 
either through the corresponding option under **File** or by pressing **Ctrl + O**.
This file has to be a **.csv** file to work. After the file is loaded the 
parameters for the fitting can be altered on the right. In the top right the 
number of pearson functions can be increased or decreased. When increasing the
number of functions, a new tab will appear with the parameters for the new 
function.
### How to change the parameters
The fitting parameters can be changed in the middle section on the right side.
Those parameters are the **kurtoses** (beta), the **standard deviation** (sigma)
and the **scalar factor** (c) with which the function will be multiplied before
it gets added. Their start value for the fitting can be changed by changing the
value in the corresponding field of the parameter. By checking the box on 
the right of the parameter, the parameter gets fixed and the value in the field
is now the fixed value. After changing the parameter options the fitting can be
started with the **Fit** button in the bottom right corner. When the fitting is
completed the fitted parameters will appear in the field below the parameter 
options.
### How to save
The result of the fitting can be saved in many ways. The function values can be
saved through the **File** menu. These will be saved as a **.csv** file. Also, 
just the values of the fitted parameters can be saved via the **File** menu as a
**.csv** file. The graph which is displayed on the left side can be saved through
the toolbar below the graph. Furthermore, the current work-setup can be saved 
with the **Save options** option in the **File** menu. All the current fitting
options for the parameters will be saved and can be later loaded up to continue
work from the last session.
### Menu
This is a brief summary of the option menus

#### File:
* **Save fitted function**: Saves fitted function as **.csv** file
* **Save parameters**: Saves parameters as **.csv** file
* **Save options**: Saves fitting options as **.json** file
* **Load measurement data**: Loads **.csv** file with measurement data
* **Load options**: Loads options file from previous session

#### Graph:
* **line/semilog**: swaps between linear and semi-logarithmic representation
* **Legend**: turns on and off the legend
* **Show single functions**: Shows or hides all the individual pearson functions
* **Change x values**: Opens window to change the x values which are used to plot the fitted function
