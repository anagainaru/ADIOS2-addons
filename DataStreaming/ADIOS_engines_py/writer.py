import numpy as np
import adios2
import sys

if len(sys.argv) < 3:
    print("Usage: %s array_size steps" %(sys.argv[0]))
    exit(1)

nx = int(sys.argv[1])
shape = [nx]
start = [0]
count = [nx]
NSteps = int(sys.argv[2])

# with-as will call adios2.close on fh at the end
with adios2.open("cfd.sst", "w", config_file="adios.xml", io_in_config_file="test") as fh:
   # NSteps from application
   for i in range(0, NSteps):
      fh.write("physical_time", np.array([100]) )
      pressure = np.random.rand(nx)
      fh.write("pressure", pressure, shape, start, count, end_step=True)
