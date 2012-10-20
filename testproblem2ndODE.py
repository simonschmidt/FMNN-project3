# -*- coding: UTF-8 -*-
from __future__ import division

"""
This file contains two test problems for software to
solve 2nd Order ODEs.

To generate the right hand side function of the testproblems do
the following:
    
truck=Truck()
f1=truck.fcn  <---- right hand side function of Truck problem
x0=truck.initial_conditions()  <--- initial conditions for this problem

pend=Pendlum2nd() 
f2=pend.fcn  <---- right hand side function of Pendulum problem
x0=pend.initial_conditions() <--- initial conditions for this problem

    


Edda Eich-Soellner/Claus Führer     1997
coded in Python by Olivier Verdier and Claus Führer 2009/2011
"""

from numpy import array, empty, zeros, dot, pi, sqrt, cos, sin, hstack, vstack, column_stack, exp, eye, diag, eye, zeros_like, arctan, linspace, identity, r_, ones

import numpy as np
import numpy.testing as npt

from numpy.linalg import norm, inv, solve
from scipy.linalg import eigvals
from pylab import plot, legend
import pylab as PL

#--------------- Truck Class and functions

def package_in_arrays(relat):
    def new_relat(self, fr1,fr2,fra,frd1,frd2,frda,frp1,frp2,to1,to2,toa,tod1,tod2,toda,top1,top2):
        return relat(self,array([fr1,fr2]),fra,array([frd1,frd2]),frda,array([frp1,frp2]),
                array([to1,to2]),toa,array([tod1,tod2]),toda,array([top1,top2]))
    return new_relat

class Truck(object):
    """
    Truck class. 
    Mainly a conversion from matlab files from the `homepage of the book 
    "Numerical Methods in Multibody Dynamics" <http://www.maths.lth.se/na/staff/claus/NMMD2/>`_.
    """
    
    def __init__(self, unconstrained=True):
              self.unconstrained = unconstrained
              self.initialise()
    
    def initialise(self):
        """
        Sets all the constant for the truck.
        """
        self.m1 = 1450;self.m2 = 3335;self.m3 = 600;self.m4 = 1100;self.m5 = 11515;self.l2 = 14313;self.l4 = 948;self.l5 = 33000;
        #
        self.a12 = -2.06;self.a23 = 2.44;self.a24 = 1.94;self.a42 = 3.64;self.a25 = 0.98;self.a52=-3.07;
        self.b42 = 0.9 ;self.b24 = -0.8 ;
        self.c25 = 2.44;self.c1d=-1.91;self.c2d=-1.31;
        self.ac1 = -3.07;self.ac2 = 0.15;self.cc1 = -1.61;self.cc2 = -0.75;self.h1 = -1.34;self.h2 = 0;self.h3 = 0;
        #
        self.k10 = 44e5;self.k12 = 8.247E+5;self.k30 = 22e5;self.k23 = 2.711E+5;
        self.k42 = 1.357e5;self.k24 = self.k42;self.k25 = 9.0e5;
        self.k1d=7.70E5;self.k2d=self.k1d;
        self.d10 = 600;self.d12 = 21593;self.d30 = 300;self.d23 = 38537;
        self.d24 = 12218;self.d42 = 12218;self.d25 = 38500;
        # springs
        self.d1d=33013;self.d2d=self.d1d;
        # gravity
        self.ggr = 9.81;
        
        # forces
        self.f10N =     -2.32914516200000e+006;
        self.f12N =     -0.24687266200000e+006;
        self.f30N =     -1.14743483800000e+006;
        self.f23N =     -0.08492483800000e+006;
        self.f24N =     -0.12784288235294e+006;
        self.f42N =     -0.12720811764706e+006;
        self.f25N =     -0.85490594111111e+006;
        # springs
        self.f2dN =     -0.76635491099974e+6;
        self.f1dN =     -0.76635491099974e+6;    

    def initial_conditions(self):
        """
        Initial conditions for the unconstrained or constrained truck
        """
        if self.unconstrained:
            x=[0.5,2.0,0.,0.5,2.9,0.,self.a25-self.c25,2.9,0.]
            xd=9*[0.]
            x=np.array(x+xd)
            
            return x
        else:
            p2=2.0
            x=[0.5,p2,0.,0.5,p2+self.ac2-self.cc2,0.,self.ac1-self.cc1,
                          p2+self.ac2-self.cc2,0.]
            la=[0.,6.80562088888888e+4]
            xd=9*[0.]
            x=np.array(x+xd+la)
            return x
            
    def fcn(self, t, x, xdot_param=None):
        """
        Right hand side function (unconstrained)
              or residual function      (constrained)
              for the 
        un/constrained truck model
        
        
        Parameters
        ----------
        t : scalar
            time
        x : array, shape(N,)
            x=[p;v;λ]: positions, velocities, lambda (possibly)
            if N == 20, the *constrained* truck is used
            if N == 18, the *unconstrained* truck is used
        xdot_param : array, shape(N,)
            has to be left to None in case of unconstrained problem
        
        Returns
        -------
        Either:
            the residual `f(x) - M*xdot`
        or
            the right hand side `f(x)/M`
        
        """
        constrained=  not self.unconstrained
        if xdot_param is None:
            xdot = zeros_like(x)
        else:
            xdot = xdot_param
        xlen = len(x)
        
        res = empty(xlen)
        
        m1=self.m1;m2=self.m2;m3=self.m3;m4=self.m4;m5=self.m5;l2=self.l2;l4=self.l4;l5=self.l5;
        #
        a12=self.a12;a23=self.a23;a24=self.a24;a42=self.a42;a25=self.a25;a52=self.a52;
        b42=self.b42;b24=self.b24; 
        c25=self.c25;c1d=self.c1d;c2d=self.c2d;
        ac1=self.ac1;ac2=self.ac2;cc1=self.cc1;cc2=self.cc2;h1=self.h1;h2=self.h2;h3=self.h3;
        #
        k10=self.k10;k12=self.k12;k30=self.k30;k23=self.k23;
        k42=self.k42;k24=k42;k25=self.k25;
        #
        k1d=self.k1d;k2d=self.k2d;
        #
        d10=self.d10;d12=self.d12;d30=self.d30;d23=self.d23;
        d24=self.d24;d42=self.d42;d25=self.d25;
        #
        d1d=self.d1d;d2d=self.d2d;
        #
        #
        # excitation (assumed truck speed 15 m/s
        # 1 m takes tm secs)
        v = 15.;
        tm=1/15;
        piv=pi*v;
        ts= 2./15.;
        td=(-a12+a23)*tm; # time delay between the wheels
        #
        [u1,u1d] = self.excite(t,ts+td,[piv,tm]);
        [u2,u2d] = self.excite(t,ts,[piv,tm]);
        #
        p = x[:9]
##        p1 = x( 1); p2 = x( 2); p3    = x( 3);
##        p4 = x( 4); p5 = x( 5); p6    = x( 6);
##        p7 = x( 7); p8 = x( 8); p9    = x( 9);
        #
        v = x[9:18]
##        v1    = x(10); v2     = x(11); v3  = x(12);
##        v4    = x(13); v5     = x(14); v6  = x(15);
##        v7    = x(16); v8     = x(17); v9  = x(18);
        #
##        la1 = x(19); la2 = x(20);
        if constrained:
            la = x[18:20]
        #
        # relative vectors and their derivatives
        #
        [rho10,rho10d]=self.rel(0.,u1,0.,0.,u1d,0.,a12,0.,
                a12,p[1-1],0.,0.,v[1-1],0.,0.,0.)
        [rho12,rho12d]=self.rel(a12,p[1-1],0.,0.,v[1-1], 0.,0., 0.,
                0.,p[2-1],p[3-1],0.,v[2-1],v[3-1],a12,h1)
        [rho23,rho23d]=self.rel(a23,p[4-1],0.,0.,v[4-1], 0.,0., 0.,
                0.,p[2-1],p[3-1],0.,v[2-1],v[3-1],a23,h1)
        [rho30,rho30d]=self.rel(0.,u2,0.,0.,u2d,0.,a23,0.,
                a23,p[4-1],0.,0.,v[4-1],0.,0.,0.)
        [rho24,rho24d]=self.rel(0.,p[2-1],p[3-1],0.,v[2-1], v[3-1],a24,h2,
                a24-b24,p[5-1],p[6-1],0.,v[5-1],v[6-1],b24,h3)
        [rho42,rho42d]=self.rel(0.,p[2-1],p[3-1],0.,v[2-1], v[3-1],a42,h2,
                a24-b24,p[5-1],p[6-1],0.,v[5-1],v[6-1],b42,h3)
        [rho25,rho25d]=self.rel(0.,p[2-1],p[3-1],0.,v[2-1], v[3-1],a25,h2,
                p[7-1],p[8-1],p[9-1],v[7-1],v[8-1],v[9-1],c25,h3)
        # constraint
        [rho52,rho52d]=self.rel(0.,p[2-1],p[3-1],0.,v[2-1], v[3-1],ac1,ac2,
                p[7-1],p[8-1],p[9-1],v[7-1],v[8-1],v[9-1],cc1,cc2)
        # springs
        [rhod2,rhod2d]=self.rel(0.,p[2-1],p[3-1],0.,v[2-1], v[3-1],a52,h2,
                     p[7-1],     p[8-1],p[9-1],v[7-1],v[8-1], v[9-1],c2d,h3);
        [rhod1,rhod1d]=self.rel(0.,        p[2-1],p[3-1],0.,v[2-1], v[3-1],a52,h2,
                     p[7-1],     p[8-1],p[9-1],v[7-1],v[8-1], v[9-1],c1d,h3);

        # 
        f10N = self.f10N
        f12N = self.f12N
        f30N = self.f30N
        f23N = self.f23N
        f24N = self.f24N
        f42N = self.f42N
        f25N = self.f25N
        # springs
        f2dN = self.f2dN
        f1dN = self.f1dN
        #
        [(f101,f102),e10]=self.spridamp(k10,d10,rho10,rho10d,f10N);
        [(f121,f122),e12]=self.spridamp(k12,d12,rho12,rho12d,f12N); 
        [(f301,f302),e30]=self.spridamp(k30,d30,rho30,rho30d,f30N);
        [(f241,f242),e24]=self.spridamp(k24,d24,rho24,rho24d,f24N);
        [(f231,f232),e23]=self.spridamp(k23,d23,rho23,rho23d,f23N);
        [(f421,f422),e42]=self.spridamp(k42,d42,rho42,rho42d,f42N);
        [(f251,f252),e25]=self.spridamp(k25,d25,rho25,rho25d,f25N);
        # springs:
        [(f2d1,f2d2),e2d]=self.spridamp(k2d,d2d,rhod2,rhod2d,f2dN);
        [(f1d1,f1d2),e1d]=self.spridamp(k1d,d1d,rhod1,rhod1d,f1dN);
        #
        #    Kinematic equation
        #
##        for i= in range(9):
##            res[i] = xdot[i] - x[i+9]; # 
        res[:9] = xdot[:9] - x[9:18]
        # remove spring forces if constrained:
        if constrained:
            f1d1 = f1d2 = f2d1 = f2d2 = 0
        #
        #    Dynamic Equation
        #
        # spring:
        f1 = f1d1 + f2d1
        f2 = f1d2 + f2d2
        #
        co3=cos(p[3-1]);si3=sin(p[3-1]);co6=cos(p[6-1]);si6=sin(p[6-1]);
        co9=cos(p[9-1]);si9=sin(p[9-1]);
        res[10-1] = m1*xdot[10-1]-((-f102+f122)-m1*self.ggr);
        res[11-1] = m2*xdot[11-1]-((-f122-f232+f242+f422+f252 + f2)-m2*self.ggr);
        res[12-1] = l2*xdot[12-1] - (
                    (a23*(-f232)+a12*(-f122)+h1*(-f231-f121))*co3-
                    (a23*(-f231)+a12*(-f121)+h1*(-f232-f122))*si3+
                    (a25*f252+ a52*f2 + h2*(f251 + f1))*co3-
                    (a25*f251+ a52*f1 + h2*(f252 + f2))*si3+
                    (a42*f422+ a24*f242     +h2*(f421+f241))*co3-
                    (a42*f421+ a24*f241     +h2*(f422+f242))*si3);
        res[13-1] = m3*xdot[13-1]-((-f302+f232)-m3*self.ggr); 
        res[14-1] = m4*xdot[14-1]-((-f422-f242)-m4*self.ggr);
        res[15-1] = l4*xdot[15-1]-(
                    (b42*(-f422)+b24*(-f242)+h3*(-f421-f241))*co6-
                    (b42*(-f421)+b24*(-f241)+h3*(-f422-f242))*si6);
        res[16-1] = m5*xdot[16-1]-(-f251-f1);
        res[17-1] = m5*xdot[17-1]-(-f252-f2-m5*self.ggr);
        res[18-1] = l5*xdot[18-1]-(
                    (c25*(-f252)+c2d*(-f2d2)+c1d*(-f1d2)+h3*(-f251-f1))*co9-
                    (c25*(-f251)+c2d*(-f2d1)+c1d*(-f1d1)+h3*(-f252-f2))*si9);
        if constrained:
            #
            #    Adding the constraint matrix and the Lagrange multipliers
            #
            G=array([[0, 0,+ac1*si3+ac2*co3,0,0,0,1,0, -cc1*si9-cc2*co9],
               [0,-1, -ac1*co3+ac2*si3,0,0,0,0,1,+cc1*co9-cc2*si9]]);
            res[9:18] -= dot(G.T, la) # (G'*[la1;la2])';
            #
            #    Constraints
            #
            res[18:20]=rho52d;
        
        if xdot_param is None and self.unconstrained: # looking for a right hand side
            mass = ones(18)
            mass[-9:] = array([m1, m2, l2, m3, m4, l4, m5, m5, l5])
            res2= -res/mass
            return res2
        else:
            return res

    def spridamp(self, k,d,rho,rhod,fN):
        """
        [f1,f2,e]=spridamp(k,d,rho,rhod,fN)
        forces of a planar spring and damper in
        parallel, where
        k,d      are the stiffness and damping ratio
        rho      is the relative displacement vector and
        rhod  is its time derivative
        fN              is the nominal force
        """
        length    = norm(rho)
        lengthd = dot(rho,rhod)/length
        f = k*length + d*lengthd + fN
        e = rho/length
        return f*e,e
    
    def relat(self, fr,fra,frd,frda,frp,to,toa,tod,toda,top):
        """ [rho,rhod]=relat(fr,frd,fr1,fr2,to,tod,to1,to2)
              computes the relative vector and its derivative between
              the point frp on body "from" described in the
              body fixed coordinate system located in fr and
              rotated with the angle fra.
              the other point is located on body "to" and described 
              correspondingly.
        """
        def rotation(theta):
            return array([[cos(theta),-sin(theta)],[sin(theta),cos(theta)]])
        def rotationd(theta):
            return rotation(theta + pi/2)

        rotmto = rotation(toa)
        rotmfr = rotation(fra)

        rho = to + dot(rotmto, top)-(fr + dot(rotmfr, frp))
        
        rhod= tod + dot(rotationd(toa)*toda,top) - (frd + dot(rotationd(fra)*frda,frp))
        return rho, rhod
    
    rel = package_in_arrays(relat)
    
    def excite(self, t,t0,param):
        """
        function [u,ud]=excite1(t,t0,param)
        Excitation model 1 
        u = 0.001*(t-t0)**6*exp(-(t-t0));    a ca 12 cm hump 
        """
        if t < t0:
            u=0.0
            ud=0.0
        else:
            u = 0.001*(t-t0)**6*exp(-(t-t0))  # a ca 12 cm hump 
            ud= 0.006*(t-t0)**5*exp(-(t-t0))-0.001*(t-t0)**6*exp(-(t-t0))

        return u,ud    

class Pendulum2nd(object):
    """
    The pendulum as second order DAE in angle coordinates
    Initial condition alpha=pi/2, alpha=0.
    corresponds to an initial condition with the pendulum bar horizontal
    """
    def __init__(self):
        self.ggr=array([9.81])
        self.m=1.
        self.L=1.
    def initial_condition(self):
        x=[np.pi/2.,0.]
        return x
    def fcn(self,t,x):
        alpha=x[0]   # position
        alphad=x[1]  # velocity
        alphadd=-self.ggr/self.L * sin(alpha)
        return alphadd
