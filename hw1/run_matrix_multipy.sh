#!/bin/bash

# Read matrix dimensions from the command line arguments
M_SIZE=$1
N_SIZE=$2
K_SIZE=$3

# Ensure that the required arguments are provided
if [ -z "$M_SIZE" ] || [ -z "$N_SIZE" ] || [ -z "$K_SIZE" ]; then
    echo "Usage: $0 M_SIZE N_SIZE K_SIZE"| tee -a"results.txt"
    exit 1
fi

# Define output file
OUTPUT_FILE="results.txt"

echo "$M_SIZE $N_SIZE $K_SIZE " |tee -a "results.txt"
# Run Python matrix multiplication
python3 p_m.py $M_SIZE $N_SIZE $K_SIZE | tee -a "results.txt"



# Compile and run Java matrix multiplication

javac j_m.java | tee -a "OUTPUT_FILE"
java j_m $M_SIZE $N_SIZE $K_SIZE | tee -a "results.txt"

# Compile and run C matrix multiplication
gcc -O c_m.c -o c_m | tee -a "results.txt"
./c_m $M_SIZE $N_SIZE $K_SIZE | tee -a "results.txt"

gcc -O1 c_m.c -o c_m1 | tee -a "results.txt"
./c_m1 $M_SIZE $N_SIZE $K_SIZE | tee -a "results.txt"

gcc -O2 c_m.c -o c_m2 | tee -a "results.txt"
./c_m2 $M_SIZE $N_SIZE $K_SIZE | tee -a "results.txt"

gcc -O3 c_m.c -o c_m3 | tee -a "results.txt"
./c_m3 $M_SIZE $N_SIZE $K_SIZE | tee -a "results.txt"

echo "Results have been saved to results.txt" | tee -a "results.txt"
echo " " >> temp | tee -a "results.txt"
 