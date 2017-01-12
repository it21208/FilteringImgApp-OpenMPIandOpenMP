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
  echo >> $OutputFile
  ./filteringimgapp ${IMG} image${IMG}x${IMG}.dat serialConv${IMG}x${IMG}.dat 2>/dev/null 1>> $OutputFile
done
