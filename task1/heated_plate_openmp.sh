#! /bin/bash
#
gcc -c -Wall -fopenmp origin.c
if [ $? -ne 0 ]; then
  echo "Compile error."
  exit
fi
#
gcc -fopenmp origin.o -lm
if [ $? -ne 0 ]; then
  echo "Load error."
  exit
fi
rm origin.o
mv a.out $HOME/binc/origin
#
echo "Normal end of execution."
