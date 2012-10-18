import numpy as N
import pylab as P
from assimulo.problem import Explicit_Problem  #Imports the problem formulation from Assimulo
from assimulo.solvers.sundials import CVode #Imports the solver CVode from Assimulo

def rhs(t,y):
    A =N.array([[0,1],[-2,-1]])
    yd=N.dot(A,y)

    return yd

y0=N.array([1.0,1.0])
t0=0.0

model = Explicit_Problem(rhs, y0, t0) #Create an Assimulo problem
model.name = 'Linear Test ODE'        #Specifies the name of problem (optional)

sim = CVode(model)

tfinal = 10.0        #Specify the final time

t, y = sim.simulate(tfinal) #Use the .simulate method to simulate and provide the final time

#Plots the result
P.plot(t,y)
P.show()
