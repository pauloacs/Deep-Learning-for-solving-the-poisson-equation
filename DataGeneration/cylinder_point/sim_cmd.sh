#!/bin/bash

cd simulation_data

shopt -s nullglob
numdirs=(*/)
numdirs=${#numdirs[@]}
((max= numdirs-1))


for i in `seq 0 $max`
do
  if [ $(($i%5)) -eq 0 ]; then
    sleep 5s
  fi
  cd $i
  blockMesh > log.blockMesh
  mirrorMesh -overwrite > log.mirror
  renumberMesh -overwrite > log.renumber
  echo "----------$i/$max simulation is running------------------"
  pisoFoam > log.piso
  postProcess -func writeCellCentres > log.cells
  foamToVTK -ascii > log.foamtoVTK
  echo "----------$i/$max is done--------------------------------"
  cd ..
  echo "$i/$max is done" > ls-output_0.txt
done
