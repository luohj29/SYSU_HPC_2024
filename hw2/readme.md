Դ�ļ��� 
c_m.c ���о�����㣻 
mp_v1.cpp: mpi��Ե�ͨ��
mp_v2.cpp: mpi����ͨ��
cmo.py: ������ٱȺͲ���Ч�ʵĽű�
matrix_multiply_s.cpp: ���о������˷�����

��ִ���ļ�
matrix_multiply_s.o: ����˷�����object
libmm.so: ���о������˷�������̬��
mp_v1: ��Ե㲢�о���˷�
mp_v2: ����ͨ�ž���˷�

����ָ��(��ֱ��make)��
matrix_multiply_s.o:
g++ -c -fPIC matrix_multiply_s.cpp 

libmm.so:
g++ -shared -o libmm.so matrix_multiply_s.o

�⺯���޸�ϵͳ·����
sudo cp libmm.so /usr/local/lib

mp_v1�����Ӷ�̬��
mpic++ -L./ -Wall mp_v1.cpp -o mp_v1 -lmm


mp_v2�����Ӷ�̬��
mpic++ -L./ -Wall mp_v2.cpp -o mp_v2 -lmm

����ָ�M=N=K����ֱ��make run, �������output.txt��ʾ��
./c_m 521 512 512
mpirun -n 4 mp_v1 512 
mpirun -n 4 mp_v2 512 

output.txt:��¼��Make������