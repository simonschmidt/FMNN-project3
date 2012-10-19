FMNN-project3
=============

http://www.maths.lth.se/na/courses/FMNN25/media/material/project03_.pdf

# Newmark 
test with 
```
t,y,yd = so_test(solver=Newmark,tfinal=10.)
```

# Assimulo installation
When compiling sundials make sure to have -fPIC in the CFLAGS
and before running setup.py in assimulo; in the file src/solvers/sundials.pyx change
```
include "sundials_constants.pxi" #Sundials related constants
include "sundials_callbacks.pxi"
```
to
```
include "../lib/sundials_constants.pxi" #Sundials related constants
include "../lib/sundials_callbacks.pxi"
```
To test installation:
```
from assimulo.examples import *

cvode_with_jac.run_example()
```
you should get a plot
