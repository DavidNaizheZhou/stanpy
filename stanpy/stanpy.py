"""Main module."""

import numpy as np
###### Functions ############

def qs(b:np.array,h:np.array,zsi:np.array,ysi:np.array) -> dict: 
    return dict()

def mat(E:float=None, G:float=None,I:float=None,A:float=None, EI:float=np.nan, GA:float=np.nan, EA:float=np.nan, qs:dict=dict()) -> dict:
    return dict()

def load_integrals(qs:dict,mat:dict, load:dict) -> np.array:
    return np.empty(5,1)

def transfer_relation(qs:dict, mat:dict, load:dict) -> np.array:
    return np.zeros((5,5))

def boundary_condition(fi:np.array, fk:np.array) -> dict:
    return dict()

