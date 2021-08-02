#BSUB -P CSC143
#BSUB -W 0:30
#BSUB -nnodes 1
#BSUB -q debug
#BSUB -J adios_sst_test
#BSUB -o sst_test%J.out
#BSUB -e sst_test%J.out

module load  ibm-wml-ce/1.6.2-1
conda activate cloned-ibm-env
module load hdf5
#ADIOS2
export PYTHONPATH=/gpfs/alpine/world-shared/csc143/ganyushin/quip_app/ADIOS2-Python-fast/build/lib/python3.6/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=/gpfs/alpine/world-shared/csc143/ganyushin/quip_app/ADIOS2-Python-fast/build/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/gpfs/alpine/world-shared/csc143/ganyushin/ADIOS2-Python-fast/build/thirdparty/EVPath/EVPath/lib64:$LD_LIBRARY_PATH

jsrun --erf_input batch_sst.erf
