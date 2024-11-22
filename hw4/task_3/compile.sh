gcc -c -fPIC parallel.c -o parallel.o
gcc -shared -o libpa_for.so parallel.o
cp libpa_for.so ../lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/hongjie/ml/lib
gcc -o main task3.c -L. -lpa_for