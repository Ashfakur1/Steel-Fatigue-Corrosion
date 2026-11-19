# -*- coding: mbcs -*-
#
# Abaqus/CAE Release 2023 replay file
# Internal Version: 2022_09_29-00.11.55 183150
# Run by it on Mon Nov 16 10:55:24 2026
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(1.55469, 1.55556), width=228.85, 
    height=154.311)
session.viewports['Viewport: 1'].makeCurrent()
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
execfile('1_fem_pipeline.py', __main__.__dict__)
#: PARSED PARAMS: corrosion_depth=2.000, stress=350.000, job_name=mesh_c2p0_s350_m0p5, mesh=0.50, mesh_study=True
#: RESULT FILE: E:\Ashfakur\MSC\Thesis\Steel Fatigue PINN\Codes\Results\fem_mesh_study.csv
#: The model "CorrosionFatigueModel" has been created.
#: DIAGNOSTIC: total elements in instance = 128000
#: DIAGNOSTIC: instance node X range = 0.000 to 100.000
#: DIAGNOSTIC: gauge_elements found via inst.elements = 64000
#: Gauge set created with 64000 elements (X: 25.0 to 75.0)
#: Model: E:/Ashfakur/MSC/Thesis/Steel Fatigue PINN/Codes/mesh_c2p0_s350_m0p5.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       3
#: Number of Node Sets:          2
#: Number of Steps:              1
#: GAUGE-SECTION STRESS: max=350.44 MPa, mean=350.00 MPa (n=64000 values)
#: DIAGNOSTIC: reaction force check failed: arg1; found 'generator', expecting a recognized type
#: DONE: mesh_c2p0_s350_m0p5 -> max=350.44 MPa, mean=350.00 MPa (written to E:\Ashfakur\MSC\Thesis\Steel Fatigue PINN\Codes\Results\fem_mesh_study.csv)
print 'RT script done'
#: RT script done
