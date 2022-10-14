import numpy as np
import adios2
import sys

if len(sys.argv) < 2:
    print("Usage: %s array_size " %(sys.argv[0]))
    exit(1)

nx = int(sys.argv[1])
shape = [nx, 2]
start = [0, 0]
count = [nx, 2]

# with-as will call adios2.close on fh at the end
# if only one rank is active pass MPI.COMM_SELF
with adios2.open("cfd.bp", "r") as fh:
    print("Total steps:", fh.steps())
    for fstep in fh:
        if fstep.current_step() != fh.steps()-1:
            continue

        # inspect variables in current step
        step_vars = fstep.available_variables()

        # print variables information
        print("Printing step", fstep.current_step())
        for name, info in step_vars.items():
            print("variable_name: " + name)
            for key, value in info.items():
                print("\t" + key + ": " + value)
        print("\n")

    # track current step
    step = fstep.current_step()
    if( step == 0 ):
      size_in = fstep.read("size")

    # read variables return a numpy array with corresponding selection
    pressure = fstep.read("pressure", start, count)
