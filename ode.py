#coding: utf-8
from assimulo.problem import Explicit_Problem  #Imports the problem formulation from Assimulo
from assimulo.solvers.sundials import CVode #Imports the solver CVode from Assimulo
from assimulo.explicit_ode import Explicit_ODE
from assimulo.solvers import ExplicitEuler
import assimulo
import assimulo.ode
import scipy.optimize
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

        self.rhs_orig = rhs
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
        arrow_distance: Number of solution points to skip between each arrow
        arrow_scaling: maximum length of an arrow
        arrow_head_width:


        returns the solution matrix (t,y,dy)

    """

    # Default rhs, note time-dependency
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


# Used src/solvers/euler.pyx to figure out behaviour
# Could use some touching up, like get/set functions for g,b like h 
# already has from ExplicitEuler
class Newmark(ExplicitEuler):
    def __init__(self,problem):
        super(Newmark,self).__init__(problem)

        # Set gamma and beta and stepsize options
        self.options["g"] = 0.6  
        self.options["b"] = 0.4 
        self.options["h"] = 0.1

        self.n = len(self.y0)/2
        self.a_old = None

        self.rhs = problem.rhs_orig

        self.supports["one_step_mode"] = True


    def step(self,t,y,tf,opts):
        h = self.options["h"]

        if t+h < tf:
            t, y = self._step(t,y,h)
            return assimulo.ode.ID_PY_OK, t, y
        else:
            h = min(h, abs(tf-t))
            t, y = self._step(t,y,h)
            return assimulo.ode.ID_PY_COMPLETE, t, y


    # Not sure why I needed this wrapper, without it _step from ExplicitEuler 
    # is used instead
    def integrate(self,t,y,tf,opts):
        h = self.options["h"]
        h = min(h, abs(tf-t))

        k = numpy.floor((tf-t)/h)+1
        tr = numpy.zeros(k)
        yr = numpy.zeros((k,self.n*2))
        i=0
        while t+h < tf:
            t,y = self._step(t,y,h)
            tr[i]=t
            yr[i]=y
            i=i+1
            h=min(h,abs(tf-t))
        else:
            t,y= self._step(t,y,h)
            tr[i]=t
            yr[i]=y
            i=i+1

        return assimulo.ode.ID_PY_COMPLETE, tr, yr

    # Some functions to de-uglify _step
    # almost half the time is spent in this function :(
    def _newmark(self,y,t_new,h,a_new):
        """
            given old y and a a_new guess,
            returns p_new and v_new
        """
        p_old = y[:self.n]
        v_old = y[self.n:]
        p_new = p_old + h*v_old + 0.5*h*h*((1-2*self.b)*self.a_old + 2*self.b*a_new)
        v_new = v_old + h*((1-self.g)*self.a_old + self.g*a_new)
        return (p_new,v_new)

    def _newmarkError(self,y,t_new,h,a_new):
        """
            ||a_new - rhs(t_new, p_new, v_new)||
        """
        (p_new,v_new) = self._newmark(y,t_new,h,a_new)
        return numpy.linalg.norm(a_new - self.rhs(t_new,p_new,v_new))

    def _newmarkUpdate(self,y,t_new,h,a_new):
        return numpy.hstack(self._newmark(y,t_new,h,a_new))

    def _step(self,t,y,h):
        """
        This function ties the newmark process together to give the next values

        Done by guessing a a_new, using that to get p_new and v_new,
        if everything is perfect:
        |a_new - rhs(t_new, p_new, v_new)| = 0 
        so normal optimization methods to improve upon a_new
        (currently scipy.optimize.fmin)
        """

        # Used as starting value when finding min below
        if self.a_old is None:
            self.a_old = self.rhs(t,y[:self.n],y[self.n:])

        self.b = self.options["b"]
        self.g = self.options["g"]

        t_new = t+h
        # Any better ways to solve this?
        a_new = scipy.optimize.fmin(lambda a: self._newmarkError(y,t_new,h,a), self.a_old ,disp=False)

        y_new = self._newmarkUpdate(y,t_new,h,a_new)

        self.a_old = a_new # save for next initial guess
        return (t_new, y_new)

    def print_statistics(self, verbose=assimulo.ode.NORMAL):
        self.log_message('Final Run Statistics: %s \n' % self.problem.name,        verbose)
        self.log_message(' Step-length          : %s '%(self.options["h"]), verbose)
        self.log_message(' Newmark gamma        : %s '%(self.options["g"]), verbose)
        self.log_message(' Newmark beta         : %s '%(self.options["b"]), verbose)
        self.log_message('\nSolver options:\n',                                    verbose)
        self.log_message(' Solver            : Newmark',                     verbose)
        self.log_message(' Solver type       : Fixed step\n',                      verbose)

