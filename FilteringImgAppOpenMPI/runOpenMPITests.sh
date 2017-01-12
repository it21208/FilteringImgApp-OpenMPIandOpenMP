#!/bin/bash

# Check if script is executed with no parameters
if [ $# -eq 0 ]; then
    echo "|--> ERROR: output filename is required!"
    exit 1
fi

OutputFile="${1}_${HOSTNAME}.txt"

echo `hostname` > $OutputFile
echo `date` >> $OutputFile
echo `lscpu` >> $OutputFile
for IMG in 4 20 400 1000
do
  for TASKS in 2 4 8 16
  do
    echo >> $OutputFile
    mpirun -np ${TASKS} ./filteringimgappopenmpi ${IMG} image${IMG}x${IMG}.dat openMPIConv${IMG}x${IMG}_${TASKS}.dat 2>/dev/null 1>> $OutputFile
  done
done
export OMP_NUM_THREADS=
