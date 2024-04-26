import streamlit as st
from streamlit_folium import folium_static
import folium
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from windrose import WindroseAxes
from scipy.stats import weibull_min
import requests
from initializeTurbines import initializeTurbines
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from py_wake import NOJ
from py_wake.site import XRSite
import xarray as xr
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from topfarm.cost_models.py_wake_wrapper import PyWakeAEPCostModelComponent
from topfarm import TopFarmProblem
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.plotting import NoPlot, XYPlotComp
from MinimumDistanceMultiRotor import MinimumDistanceMultiRotor
from optimizeSingleTurbines import optimizeSingleTurbines
from positionMultiRotor import positionMultiRotor
from PerformWindAnalysis import perform_wind_analysis

def main():


    st.title("Wind analysis")

    st.write('''come on, idiot, I ain't got all day...''')
    # Create a default map centered at Oslo
    m = folium.Map(location=[59.0, 5.11], zoom_start=6)

    #Markerer Utsira Nord
    utsira_koordinater = [
    (59.4480556, 4.2691667),
    (59.0694444, 4.4075),
    (59.105, 4.8122222),
    (59.4822222, 4.6736111),
    (59.4480556, 4.2691667),]

    SNii_koordinater = [
    (56.8233333,4.3466667),
    (57.0933333,5.1680556),
    (56.7380556,5.4975),
    (56.5916667,5.0336111),
    (56.4838889,4.6413889),
    (56.8233333,4.3466667),]

    # Capture start and end dates from user input
    #start_date = st.date_input("Enter start date:")
    #end_date = st.date_input("Enter end date:")
    start_date_formatted = 20180101
    end_date_formatted = 20230101

    folium.PolyLine(utsira_koordinater, tooltip="Utsira Nord").add_to(m)
    folium.PolyLine(SNii_koordinater, tooltip="Sørlige Norsjø II").add_to(m)

    folium_static(m)

    if 'f' not in st.session_state:
        st.session_state['f'] = list

    if 'A' not in st.session_state:
        st.session_state['A'] = list

    if 'k' not in st.session_state:
        st.session_state['k'] = list

    if 'initial' not in st.session_state:
        st.session_state['initial'] = list

    if 'state' not in st.session_state:
        st.session_state['state'] = list
    
        

    option = st.selectbox(
    "What area do you want to analyze?",
    ("Utsira Nord", "Sørlige Nordsjø II"),index=None, placeholder="Select area...",)
    st.write('You selected:', option)

    if option == 'Utsira Nord':
        latitude, longitude = 59.4822222, 4.6736111
        boundaries = [(1265384.397, 6625141.556), (1287449.855, 6633798.704), (1270957.191, 6673574.952), (1249123.155, 6665157.463)]
        constraint = XYBoundaryConstraint(boundaries, 'polygon')
        st.session_state['f'], st.session_state['A'], st.session_state['k'] = perform_wind_analysis(option, latitude, longitude, start_date_formatted, end_date_formatted)
        print(st.session_state['f'])

    elif option == 'Sørlige Nordsjø II':
        latitude, longitude = 56.8233333, 4.3466667
        boundaries = [(1336965.579, 6343643.86), (1358365.539, 6360396.720), (1383019.537, 6382447.185), (1354868.111, 6417238.752), (1311612.014, 6377417.797)]
        constraint = XYBoundaryConstraint(boundaries, 'polygon')
        st.session_state['f'], st.session_state['A'], st.session_state['k'] = perform_wind_analysis(option, latitude, longitude, start_date_formatted, end_date_formatted)
        print(st.session_state['f'])
    


    rotorType = st.selectbox(
        "Choose wind turbine type",
        ("Single Rotor", "Multi Rotor"), index=None, placeholder="Select turbine type..."
    )
    st.write('You selected:', rotorType)

    if rotorType == "Multi Rotor":
        wd = np.linspace(0, 360, len(st.session_state['f']), endpoint=False)
        ti = 0.1
        site = XRSite(ds = xr.Dataset(data_vars={'Sector_frequency': ('wd', st.session_state['f']), 'Weibull_A': ('wd', st.session_state['A']), 'Weibull_k': ('wd', st.session_state['k']), 'TI': ti},
                        coords={'wd': wd}))

        #Defining the wind turbine object
        u = [0,3,12,25,30]
        ct = [0,8/9,8/9,.3, 0]
        power = [0,0,1000,1000,0]

        wind_turbines = WindTurbine(name='MyWT',
                            diameter=30,
                            hub_height=40,
                            powerCtFunction=PowerCtTabular(u,power,'kW',ct))
        
        wfmodel = NOJ(site, wind_turbines)

        rows = st.selectbox(
            "Choose how many rows the multi rotor systems will include",
            (2,3,4,5,6,7,8), index=None, placeholder="Select number rows..."
        )
        st.write('You selected:', str(rows), "number of rows")
        collumns = st.selectbox(
            "Choose how many rows the multi rotor systems will include",
            (2,3,4,5,6,7,8), index=None, placeholder="Select number collumns..."
        )
        st.write('You selected:', str(collumns), "number of rows...")
        n_mr = st.number_input(
            "Choose how many multi rotor systems the wind park will include",
            1,375, placeholder="Select number of turbines..."
        )
    elif rotorType == "Single Rotor":
        
        wd = np.linspace(0, 360, len(st.session_state['f']), endpoint=False)
        ti = 0.1
        site = XRSite(ds = xr.Dataset(data_vars={'Sector_frequency': ('wd', st.session_state['f']), 'Weibull_A': ('wd', st.session_state['A']), 'Weibull_k': ('wd', st.session_state['k']), 'TI': ti},
                        coords={'wd': wd}))
        

        #Defining the wind turbine object
        u = [0,3,12,25,30]
        ct = [0,8/9,8/9,.3, 0]
        power = [0,0,15000,15000,0]

        wind_turbines = WindTurbine(name='MyWT',
                            diameter=240,
                            hub_height=150,
                            powerCtFunction=PowerCtTabular(u,power,'kW',ct))

        wfmodel = NOJ(site, wind_turbines)

        n_wt = st.number_input("Choose how many turbines the windpark will inhabit",
                            1,200, placeholder="Select number of turbines..."
                            )
        st.write('You selected:', str(n_wt), "number of turbines")
    
    # Button to trigger optimization
    if st.button("Optimize Wind Farm"):
        if rotorType == 'Single Rotor':
            with st.spinner('Wait for it...'):
                st.session_state['initial'], st.session_state['state'] = optimizeSingleTurbines(boundaries, n_wt, 1000)
                st.write(st.session_state['state'])
            st.success('Done!')
            

            #Optimize according to single Rotor
        elif rotorType == 'Multi Rotor':
            #Optimize according to multi Rotor
            minimumDistance = MinimumDistanceMultiRotor(st.session_state['f'],st.session_state['A'],st.session_state['k'], wd, rows, collumns)
            st.write("The minimum distance between multi rotors should be: ", str(minimumDistance))
            st.session_state['state'] = positionMultiRotor(boundaries, minimumDistance, n_mr)
            st.write(st.session_state['state'])


if __name__ == "__main__":
    main()

# streamlit run --server.fileWatcherType=poll streamlit.py
