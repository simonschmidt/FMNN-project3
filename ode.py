#coding: utf-8
from assimulo.problem import Explicit_Problem  #Imports the problem formulation from Assimulo
from assimulo.solvers.sundials import CVode #Imports the solver CVode from Assimulo
import numpy
import pylab

class SecondOrderExplicit_Problem(Explicit_Problem):
	def __init__(self,rhs,y0,yd0,t0=0,sw0=None,p0=None):
		"""y'' = rhs(t,y,y')
			transform into

			y' = v
			v' = rhs(t,y,v)
		"""
		def newrhs(t,yyd,**kwargs):
			n=len(yyd)/2

			y=yyd[:n]
			v=yyd[n:]

			dv = rhs(t,y,v)
			dy = v

			res = numpy.hstack((dy,dv))
			return res

		yyd0 = numpy.hstack((y0,yd0))
		super(SecondOrderExplicit_Problem,self).__init__(newrhs,yyd0,t0)


def so_test(arrows=True):
	def rhs(t,y,dy):                
	    A =numpy.array([[0,1],[-2,-1]])
	    B =numpy.array([[-1,2],[-2,0]]) 
	    return numpy.dot(A,y) +numpy.dot(B,dy)

	y0=numpy.array([1,1])
	yd0=numpy.array([-2,1])

	model = SecondOrderExplicit_Problem(rhs,y0,yd0)

	sim = CVode(model)

	t,y = sim.simulate(10.0)
	(y,dy) = (y[:,0:2],y[:,2:])

	# Plot components relative time
	pylab.subplot(2,1,1)
	pylab.hold(True)

	pylab.plot(t,y[:,0],label='$y_0$')
	pylab.plot(t,y[:,1],label='$y_1$')
	pylab.xlabel('t')
	pylab.legend(loc=0)


	# Plot y₀ relative y₁
	pylab.subplot(2,1,2)
	pylab.hold(True)
	pylab.grid(True)

	pylab.plot(y[:,0],y[:,1])
	pylab.xlabel('$y_0$')
	pylab.ylabel('$y_1$')

	# Arrows in tangent direction (similar to phase portrait, but only on the solution)
	if arrows:
		scale = numpy.max(map(numpy.linalg.norm,dy))
		print scale
		for x,dx in zip(y[::5],dy[::5]):
			dx=dx/scale
			print numpy.linalg.norm(dx)
			pylab.arrow(x[0],x[1],dx[0],dx[1],head_width=0.1,alpha=0.8,fc='k')



	pylab.show()

	return (t,y)