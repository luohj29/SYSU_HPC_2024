源文件： 
c_m.c 串行矩阵计算； 
mp_v1.cpp: mpi点对点通信
mp_v2.cpp: mpi集合通信
cmo.py: 计算加速比和并行效率的脚本
matrix_multiply_s.cpp: 串行矩阵计算乘法函数

可执行文件
matrix_multiply_s.o: 矩阵乘法函数object
libmm.so: 串行矩阵计算乘法函数动态库
mp_v1: 点对点并行矩阵乘法
mp_v2: 集合通信矩阵乘法

编译指令(或直接make)：
matrix_multiply_s.o:
g++ -c -fPIC matrix_multiply_s.cpp 

libmm.so:
g++ -shared -o libmm.so matrix_multiply_s.o

库函数修改系统路径：
sudo cp libmm.so /usr/local/lib

mp_v1：链接动态库
mpic++ -L./ -Wall mp_v1.cpp -o mp_v1 -lmm


mp_v2：链接动态库
mpic++ -L./ -Wall mp_v2.cpp -o mp_v2 -lmm

运行指令（M=N=K）或直接make run, 结果会在output.txt显示：
./c_m 521 512 512
mpirun -n 4 mp_v1 512 
mpirun -n 4 mp_v2 512 

output.txt:记录了Make输出结果