
############################configure the julia project and load the interface######################
import os
import julia

# Specify the path to your Julia project or environment
os.environ["JULIA_PROJECT"] = "/home/c/chenqian3/ACEhamiltonians/H2O_PASHA"
julia.install()
from julia.api import Julia
jl = Julia(compiled_modules=False)

from julia import Main, Serialization, Base
Main.include("/home/c/chenqian3/ACEhamiltonians/H2O_PASHA/H2O_Pasha/data_interface/function.jl")



##########################################load the model#############################################
model_path = "/home/c/chenqian3/ACEhamiltonians/H2O_PASHA/H2O_Pasha/Result/H_H2O_1_rcut_6/H2O_H_aims.bin"
model = Serialization.deserialize(model_path)


################################define a python format ase.atoms object###############################
from ase import Atoms
import numpy as np
O_H_distance = 0.96  
angle = 104.5  

angle_rad = np.radians(angle / 2) 

x = O_H_distance * np.sin(angle_rad)
y = O_H_distance * np.cos(angle_rad)
z = 0.0

positions = [
    (0, 0, 0),          # Oxygen
    (x, y, 0),          # Hydrogen 1
    (x, -y, 0)          # Hydrogen 2
]
water = Atoms('OH2', positions=positions)



#######################################################################################################
predicted = Main.J2P.predict([water]*64, model)   
predicted = [h for h in predicted] 
np.save("/home/c/chenqian3/ACEhamiltonians/H2O_PASHA/H2O_Pasha/data_interface/predicted.npy", predicted)
