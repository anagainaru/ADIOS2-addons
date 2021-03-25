procs=$1
if [ -z "$1" ]
  then
    procs=1
fi
cores=$(( $procs * 24 ))
num_variables=$2
if [ -z "$2" ]
  then
    num_variables=1
fi

for i in 100 512 1024 2048 4096; do

array_size=$(( $(( $i * 1024 )) / 4 ))
echo "
#!/bin/bash -l
#BSUB -P CSC143
#BSUB -W 00:30
#BSUB -nnodes $procs
#BSUB -J sscTest
#BSUB -o output_ssc.%J
#BSUB -e output_ssc.%J

module load gcc

export LD_PRELOAD=$LD_PRELOAD:/usr/lib64/libibverbs.so.1:/usr/lib64/librdmacm.so.1

jsrun -n$cores /gpfs/alpine/csc143/proj-shared/againaru/adios/engine_perf/build/sscReadWriter $array_size $num_variables
" > submit_temp$i.sh

bsub submit_temp$i.sh
done
