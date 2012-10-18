from assimulo.problem import Explicit_Problem  #Imports the problem formulation from Assimulo
from assimulo.solvers.sundials import CVode #Imports the solver CVode from Assimulo
import numpy

class SecondOrderExplicit_Problem(Explicit_Problem):
	def __init__(self,rhs,y0,yd0,t0=0,sw0=None,p0=None)
		"""y'' = rhs(t,y,y')
			transform into

			v' = rhs(t,y,v)
			y' = v
		"""
		def newrhs(t,y):
			n=len(y)/2
			return numpy.array([
				rhs(t,y[0:n],y[n:]),
				y[n:] ])

		super(SecondOrderExplicit_Problem,self).__init__(newrhs,y0,t0,sw0,p0)
