procs=$1
if [ -z "$1" ]
  then
    procs=2
fi
num_variables=$2
if [ -z "$2" ]
  then
    num_variables=1
fi
writers=$(( $procs / 2 ))

for i in 100 512 1024 2048 4096; do

echo "
app 0: /gpfs/alpine/csc143/proj-shared/againaru/adios/engine_perf/build/sstWriter $array_size $num_variables
app 1: /gpfs/alpine/csc143/proj-shared/againaru/adios/engine_perf/build/sstReader $array_size $num_variables
overlapping_rs: warn
oversubscribe_cpu: warn
oversubscribe_mem: allow
oversubscribe_gpu: allow
launch_distribution: packed
" > submit_sst_temp$i.erf

cnt=0
apid=0
for (( p=0; p<$procs; p++ )); do
if [ $p -eq $writers ]; then
apid=1
fi
echo "rank: $cnt: { host: $p; cpu: {0-3} } : app $apid
rank: $(( $cnt + 1 )): { host: $p; cpu: {4-7} } : app $apid
rank: $(( $cnt + 2 )): { host: $p; cpu: {8-11} } : app $apid
rank: $(( $cnt + 3 )): { host: $p; cpu: {12-15} } : app $apid
rank: $(( $cnt + 4 )): { host: $p; cpu: {16-19} } : app $apid
rank: $(( $cnt + 5 )): { host: $p; cpu: {20-23} } : app $apid
" >> submit_sst_temp$i.erf
cnt=$(( $cnt + 6 ))
done

array_size=$(( $(( $i * 1024 )) / 4 ))
echo "
#!/bin/bash -l
#BSUB -P CSC143
#BSUB -W 00:30
#BSUB -nnodes $procs
#BSUB -J sstTest
#BSUB -o output_sst.%J
#BSUB -e output_sst.%J

module load gcc

export LD_PRELOAD=$LD_PRELOAD:/usr/lib64/libibverbs.so.1:/usr/lib64/librdmacm.so.1

jsrun --erf_input submit_sst_temp$i.erf
" > submit_temp$i.sh

bsub submit_temp$i.sh
done
