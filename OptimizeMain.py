import numpy as np
import matplotlib.pyplot as plt
from py_wake.deficit_models.gaussian import IEA37SimpleBastankhahGaussian     #wake model
from py_wake.examples.data.iea37 import IEA37_WindTurbines, IEA37Site         #wind turbines and site used
from topfarm.cost_models.py_wake_wrapper import PyWakeAEPCostModelComponent   #cost model

from topfarm import TopFarmProblem
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.examples.iea37 import get_iea37_initial, get_iea37_constraints, get_iea37_cost
from topfarm.plotting import NoPlot, XYPlotComp
from py_wake import NOJ
from py_wake.site import XRSite
import xarray as xr

from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
import numpy as np
from topfarm.constraint_components.boundary import XYBoundaryConstraint
import random
import matplotlib.pyplot as plt
from initializeTurbines import initializeTurbines


n_wt = 100
n_wd = 12

#finn på egne tall her
boundaries = [(-10000, 0), (10000, 0), (10000, 20000), (500, 22000), (-10000, 20000)]

initial = initializeTurbines(boundaries, n_wt, 1500)

constraint = XYBoundaryConstraint(boundaries, 'polygon')


f = [0.084,0.084,0.084,0.084,0.083,0.083,0.083,0.083,0.083,0.083,0.083,0.083]
A = [10,8,6,10,10,10,10,10,8,12,13,13]
k = [2.5,2.5,2.5,2.5,2.5,2.5,2.5,2.5,2.5,2.5,2.5,2.5]
wd = np.linspace(0, 360, len(f), endpoint=False)
ti = 0.1

site = XRSite(ds = xr.Dataset(data_vars={'Sector_frequency': ('wd', f), 'Weibull_A': ('wd', A), 'Weibull_k': ('wd', k), 'TI': ti},
                            coords={'wd': wd}))

#Defining the wind turbine object
u = [0,3,12,25,30]
ct = [0,8/9,8/9,.3, 0]
power = [0,0,15000,15000,0]

wind_turbines = WindTurbine(name='MyWT',
                    diameter=240,
                    hub_height=150,
                    powerCtFunction=PowerCtTabular(u,power,'kW',ct))

wfmodel = NOJ(site, wind_turbines)   #PyWake's wind farm model

cost_comp = PyWakeAEPCostModelComponent(wfmodel, n_wt, wd=wd)

driver = EasyScipyOptimizeDriver()

design_vars = dict(zip('xy', (initial[:, :2]).T))

tf_problem = TopFarmProblem(
            design_vars,
            cost_comp,
            constraints=constraint,
            driver=driver,
            plot_comp=XYPlotComp())

_, state, _ = tf_problem.optimize()


positions = state

print(state)

opt_x = np.round(positions['x'], 2)
opt_y = np.round(positions['y'], 2)

# Print out the current positions
print("Optimized Positions: (x,y)")
for i, (x, y) in enumerate(zip(opt_x, opt_y)):
    print(f"Wind Turbine {i+1}: ({x}, {y})")


init_x = initial[:, 0]
init_y = initial[:, 1]


# Plot the coordinates
plt.figure(figsize=(8, 6))

# Plot initial coordinates in red
plt.plot(init_x, init_y, 'ro', label='Initial Positions')

# Plot optimized coordinates in blue
plt.plot(opt_x, opt_y, 'bo', label='Optimized Positions')

# Plot the boundary
boundary_x = [point[0] for point in boundaries]  # Extract x coordinates
boundary_y = [point[1] for point in boundaries]  # Extract y coordinates
plt.plot(boundary_x + [boundary_x[0]], boundary_y + [boundary_y[0]], 'k-', label='Boundary')

#Plotting
plt.title('Wind Turbine Positions')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True)
plt.legend()
plt.show()






