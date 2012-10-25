from assimulo.explicit_ode import Explicit_ODE
import assimulo
import assimulo.ode

import scipy.optimize as opt
import numpy as np

class HHT(Explicit_ODE):
    
    def __init__(self, problem):
        Explicit_ODE.__init__(self, problem)
        
        self.options["g"] = 0.5 
        self.options["b"] = 0.25 
        self.options["a"] = 0.
        self.options["h"] = 0.05
        
    def step(self, t, y, tf, opts):
        pass
        
    def integrate(self,t,y,tf,opts):
        pass
        
    
