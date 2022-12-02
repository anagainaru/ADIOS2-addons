#BSUB -P project
#BSUB -W 00:01
#BSUB -nnodes 1
#BSUB -q debug
#BSUB -J adios_gpu
#BSUB -o test%J.out
#BSUB -e test%J.out

module load cuda

jsrun -n1 -a1 -c4 -g1 /path/to/ADIOS2/build/bin/CudaBP4WriteRead_cuda 
