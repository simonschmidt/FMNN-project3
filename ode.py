#coding: utf-8
from assimulo.problem import Explicit_Problem  #Imports the problem formulation from Assimulo
from assimulo.solvers.sundials import CVode #Imports the solver CVode from Assimulo
import numpy
import pylab

class SecondOrderExplicit_Problem(Explicit_Problem):
    def __init__(self,rhs,y0,yd0,t0=0,sw0=None,p0=None, isDamped=True):
        """y'' = rhs(t,y,y')
            
            Note: the result of a simulation returns value of derivative as well
                To resolve, say y has n components:
                t,y = sim.simulate(10.0)
                (y,dy) = (y[:,:n],y[:,n:])

        """
        # TODO
        # Handle sw0 and p0 in newrhs and when passing to Explicit_Problem


        def newrhs(t,yyd,**kwargs):
            """
            transform y'' = rhs(t,y,y') into

            y' = v
            v' = rhs(t,y,v)

            and pass that into Explicit_Problem
            """
            n=len(yyd)/2

            if newrhs.isDamped:
                dv = rhs(t,yyd[:n],yyd[n:])
            else:
                dv = rhs(t,yyd[:n])

            return numpy.hstack((yyd[n:],dv))
        newrhs.isDamped = isDamped

        # need to stack y0 and yd0 together as initial condition for newrhs
        yyd0 = numpy.hstack((y0,yd0))
        super(SecondOrderExplicit_Problem,self).__init__(newrhs,yyd0,t0)



def so_test(rhs=None,y0=None,yd0=None,tfinal=10.0,solver=CVode,arrows=True,arrow_distance=5,arrow_scaling=1.,arrow_head_width=0.1):
    """
        Function to simplify testing of SecondOrderExplicit_Problem

        rhs: function s.t. y'' = rhs(t,y,y')
        y0,yd0: initials
        tfinal: stop time
        solver: default CVode


        arrows: Plot arrows in solution curve plot
        arrow_distance: Number of solution ponits to skip between each arrow
        arrow_scaling: maximum length of an arrow
        arrow_head_width:


        returns the solution vector (t,y,dy)

    """

    # Default rhs, note time-dependences
    if rhs is None:
        def rhs(t,y,dy):                
            A =numpy.array([[0,1],[-2,-1]])
            B =numpy.array([[-1,2],[-2,0]]) 
            return numpy.dot(A,y) +numpy.dot(B,(1+numpy.cos(t))*dy)
    if y0 is None:
        y0=numpy.array([1,1])
    if yd0 is None:
        yd0=numpy.array([-2,1])

    # Set up model and run a solver
    model = SecondOrderExplicit_Problem(rhs,y0,yd0)

    sim = solver(model)

    t,y = sim.simulate(tfinal)
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
        scale = arrow_scaling/numpy.max(map(numpy.linalg.norm,dy))
        for x,dx in zip(y[::arrow_distance],dy[::arrow_distance]):
            dx=dx*scale
            pylab.arrow(x[0],x[1],dx[0],dx[1],head_width=arrow_head_width,alpha=0.8,fc='k')


    pylab.show()

    return (t,y,dy)

