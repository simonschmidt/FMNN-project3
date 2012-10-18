FMNN-project3
=============

http://www.maths.lth.se/na/courses/FMNN25/media/material/project03_.pdf

# Assimulo installation
in the file src/solvers/sundials.pyx change
```
include "sundials_constants.pxi" #Sundials related constants
include "sundials_callbacks.pxi"
```
to
```
include "../lib/sundials_constants.pxi" #Sundials related constants
include "../lib/sundials_callbacks.pxi"
```