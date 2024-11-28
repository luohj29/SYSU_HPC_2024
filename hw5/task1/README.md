# **��ɽ��ѧ�����Ժ������ʵ�鱨��**

**��2024ѧ���＾ѧ�ڣ�**

�γ����ƣ������ܼ��������� **�����ˣ�**

|       |     |              |     |
|-------|-----|--------------|-----|
| ʵ��  |   heated_plate_openmp����diyʵ��  | רҵ������ |   ��Ϣ�����ѧ  |
| ѧ��  |    22336173 | ����         |  �޺��   |
| Email |   3133974071@qq.com  | �������     |   2024-11-27  |
## Heated_plate_openmp�����Pthread����
### ԭʼ�汾��Openmp��

    Number of processors available = 16
    Number of threads =              16
    MEAN = 74.949900

    Iteration  Change

         1  18.737475
         2  9.368737
         4  4.098823
         8  2.289577
        16  1.136604
        32  0.568201
        64  0.282805
       128  0.141777
       256  0.070808
       512  0.035427
      1024  0.017707
      2048  0.008856
      4096  0.004428
      8192  0.002210
     16384  0.001043

     16955  0.001000

    Error tolerance achieved.
    Wallclock time = 60.274613

ʹ��֮ǰ�޸ĵ�parallel for�Ŀ�ܡ�������ѭ���ڵĶ�u����ֵ�ĺ���(matrix_map_matrix)������ԭ����openmp���������ּ�������ʱ�����࣬ʱ��������300���롣
``` cpp
/*
    �����ά���������index���Լ���ά����ĵ�ַ
*/
void *matrix_map_matrix(void *args){ //a function to map value to a matrix
    struct index_in *index = (struct index_in *)args;
    struct matrix2matrix *argues = (struct matrix2matrix *)index->args;
    int i, j;
    int row_increment = index->index_diy->row_increment;
    int col_increment = index->index_diy->col_increment;
    int row_start = index->index_diy->row_start;
    int col_start = index->index_diy->col_start;
    int row_end = index->index_diy->row_end;
    int col_end = index->index_diy->col_end;
    double **dst = argues->dst;
    double **src = argues->src;

    for (i = row_start; i < row_end; i += row_increment)
    {  
        for (j = col_start; j < col_end;j+=col_increment)
        {
            dst[i][j] = src[i][j] ;
        }
    }
    return NULL;
}
```

         1  18.737475
         2  9.368737
         4  4.098823
         8  2.289577
        16  1.136604
        32  0.568201
        64  0.282805
       128  0.141777
       256  0.070808
       512  0.035430
      1024  0.017712
      2048  0.008824
      4096  0.004677
      8192  0.002722

     11578  0.000931

    Error tolerance achieved.
    Wallclock time = 375.170502


������Ϊpthreadд��parallel for�����ڲ��ϵĴ����������̣߳��������ϵͳ���ܡ� \
���ǿ���ʹ���̳߳���ʵ��pthread�̵߳Ĵ��������֣�ʹ��α�����̣߳������乤�������١�\
�̳߳ذ�������������notify������Ϊ��������������ʱ�򣬹����߳��ܼ�ʱ���ѡ����������Ϊ�յ�ʱ�򣬹����̼߳�ʱ���ߡ�
�̳߳ػ���������������unfinished������������������һ�������Ƿ���ɣ�һ����ɲ���������һ�������Ͷ�ţ���Ϊ������м�������ǰ��ļ���������**˳����ص�**��
�̳߳ش�github�ϻ�ȡhttps://github.com/JunhuaiYang/PthreadPool��������pool_wait()��ʵ�֣�ȷ����������һ��һ���Ĳ��д�����
```cpp
/*
    �̳߳ع�������֤pthread�̲߳���Ƶ���Ĵ��������١�
*/
class PthreadPool
{
private:
    pthread_mutex_t lock;                                   // ������,��֤���������ٽ�����ȫ
    pthread_cond_t notify;                                  // ��������������Ƿ����������
    std::queue<Pthreadpool_Runable> thread_queue;           // �������
    pthread_t *threads;                                     // ��������
    int shutdown;                                           // ��ʾ�̳߳��Ƿ�ر�
    static void *threadpool_thread(void *threadpool);       // ���к���

public:
    PthreadPool();
    ~PthreadPool();
    bool is_queue_empty();
    int thread_num;                                                  // �߳�����
    int running_num;                                                 // �������е��߳���
    int unfinished_task;                                               //δ��ɵ�����
    pthread_cond_t cond;                                             // ���һ�������Ƿ����
    int Init(unsigned int num);                                       // ��ʼ���̳߳�
    int AddTask(void* (*function)(void *), void *argument = nullptr); // ��������
    int Destory(PthreadPool_Shutdown flag = graceful_shutdown);      // ֹͣ���ڽ��е����񲢴ݻ��̳߳�
    void pool_wait();  //�ȴ����е����̳߳ص����񶼱�ʵ�֣���������
};
```
���յļ���Ч������ͼ��ʾ�����Կ������ٻ��ǲ����ģ��ﵽopenmpһ��������ˮƽ��


         1  18.737475
         2  9.368737
         4  4.098823
         8  2.289577
        16  1.136604
        32  0.568201
        64  0.282805
       128  0.141777
       256  0.070808
       512  0.035427
      1024  0.017707
      2048  0.008856
      4096  0.004428
      8192  0.002210
     16384  0.001043

     16955  0.001000

    Error tolerance achieved.
    Wallclock time = 54.366369

�����������½�25��Ԫ�أ���֤��ȷ�� \
HEATED_PLATE_OPENMP: \
  Normal end of execution. \
99.9699 99.9774 99.9850 99.9925 100.0000  \
99.9774 99.9831 99.9887 99.9944 100.0000  \
99.9850 99.9887 99.9925 99.9962 100.0000  \
99.9925 99.9944 99.9962 99.9981 100.0000  \
100.0000 100.0000 100.0000 100.0000 100.0000  


   

## MPiʵ�ֲ��м���

###  ����
���������֣�0�Ž��̸�������ʼ���󣬹����������̵�ͨ�ţ�ֵ��ע�����
- mpi����ʱһ���������ڴ��ȡ���ڷ�������ʱ������������ķ�������Ҫdouble new(new����)��
- ����ļ����߼��� 
    1. 0�̷߳�����󣬼���Mean.
    2. 0���̷��;���w�ķֿ飨���ݽ�����Ŀƽ�����䣩�������������̡�
    3. �������̻��w�ķֿ飬��ֵ��u����������Ԫ��������㹫ʽ������w�������Ƚ�u��w����Ԫ��֮��� \
    ���ͬmax_diff������max_diff�ͼ���֮���w�����0���̡�
    4. 0���̼�������max_diff������һ������ֵ��diff�����صľ���w����ֵ������w,�Ƚ�diff��epsilon�Ĵ�С��������diff < eps,���������������0���̷���signal = -1�����н��̣����������յ�-1�źž͵ȴ������������ظ����㡣�����ظ���2����
``` cpp
if (my_rank ==0){
    ...

// ���������ڴ�
    double *w = (double *)malloc(M * N * sizeof(double));
    ... //�����ʼmean�ľ���
    while (epsilon <= diff)
    {
        diff = 0;
        // ���;���u, w��������ū�����̼���Mean, ����Ԫ������㣬��ֵ������max_diff
        for (int i = 0; i < comm_sz - 1; i++) { // �������A����
            // my_diff[i] = 0;
            begin_Arow = std::max(i * avg_rows - 1, 0);                                               
            end_Arow = (i + 1 == comm_sz - 1) ? M : (i + 1) * avg_rows;
            end_Arow = std::min(end_Arow + 1, M);
            // Ϊ��ʹ����Χ�����ݣ���Ҫ�������������ж�����ݴ��͹�ȥ��
            
            MPI_Send(&end_Arow, 1, MPI_INT, i + 1, 10, MPI_COMM_WORLD); // end_Arow������Ϊ�źţ������-1����ū�����̽���

            MPI_Send(&w[begin_Arow * N], (end_Arow - begin_Arow) * N, MPI_DOUBLE, i + 1, 1, MPI_COMM_WORLD);

        }
        for (int i = 0; i < comm_sz - 1; i++) {  //return the std::max diff[i]
            begin_Arow = std::max(i * avg_rows - 1, 0);                                                                                                                                                                                                                                                                                                                      
            end_Arow = (i + 1 == comm_sz - 1) ? M : (i + 1) * avg_rows;
            end_Arow = std::min(end_Arow + 1, M);
            MPI_Recv(&my_diff[i], 1, MPI_DOUBLE, i + 1, 3, MPI_COMM_WORLD, &status);
            MPI_Recv(&w[(begin_Arow+1) * N], (end_Arow - begin_Arow -2) * N, MPI_DOUBLE, i + 1, 4, MPI_COMM_WORLD, &status);
            diff = std::max(my_diff[i], diff);
        }

        iterations++;
        if ( iterations == iterations_print )
        {
            printf ( "  %8d  %f\n", iterations, diff );
            iterations_print = 2 * iterations_print;
        }
    }
    signal = -1;
    printf("At %d , episilon err reached, gonna kill all processes\n", iterations);
    for(int i = 0; i < comm_sz - 1; i++){
        MPI_Send(&signal, 1, MPI_INT, i + 1, 10, MPI_COMM_WORLD); // end_Arow������Ϊ�źţ������-1����ū�����̽���
    }
    free(w);
    free(u);
}
```
ע��������Ҫ��Ԫ��4�ܵ�����Ԫ����Ϊ������㣬���н��ܾ���Ҫ����ά�����������С�

``` cpp
else { // �� 0 �Ž��̵Ĵ��� ���е�������������ҿ�Ҫ��
    int real_begin_Arow = my_rank == 1 ? 1 : avg_rows * (my_rank - 1);  //ʵ�ʼ����Ԫ�س�ʼ
    int virtual_begin_Arow = real_begin_Arow - 1; //��Ҫ���ϼ���Ԫ��
    int real_end_Arow = my_rank == comm_sz - 1 ? M - 1 : (avg_rows * my_rank); //ʵ�ʼ���Ԫ�ص��Ͻ�
    int virtual_end_Arow = real_end_Arow + 1;  //��Ҫ���¼���Ԫ��
    double *local_w = (double *)malloc((virtual_end_Arow - virtual_begin_Arow) * N * sizeof(double)); // �����Ӿ��� w ���ڴ�
    double *local_u = (double *)malloc((virtual_end_Arow - virtual_begin_Arow) * N * sizeof(double)); // �����Ӿ��� u ���ڴ�
    double max_diff;
    int M_len = virtual_end_Arow - virtual_begin_Arow;  //���ܾ���ά��
    while (true)
    {
        // ���շ��������Ϣ���ж��Ƿ����������������ȷ������У�������ʱ�������ź�-1
        MPI_Recv(&signal, 1, MPI_INT, 0, 10, MPI_COMM_WORLD, &status);
        
        if (virtual_end_Arow != signal)
        { // �����ź�
            printf("%d process received quit signal %d\n", my_rank, signal);
            break;
        }

        // �����Ӿ������ݵ�local_u,�������Ľ������local_w
  l    MPI_Recv(local_u, (virtual_end_Arow - virtual_begin_Arow) * N, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);

        max_diff = 0.0;
        for (int i = 0; i < M_len; i++)
        {
            if(i == 0 || i ==M_len-1){  //local_w�߽�
                for (int j = 0; j < N ; j++) {
                    local_w[i * N + j] = local_u[i * N + j];
                }
                continue;
            }
            for (int j = 0; j < N ; j++) {
                if(j == 0||j==N-1){ //local_w�߽�
                    local_w[i * N + j] = local_u[i * N + j];
                    continue;
                }
                // ����Ϊ���������
                local_w[i * N + j] = 0.25 * (local_u[(i - 1) * N + j] + local_u[(i + 1) * N + j] +  
                                             local_u[i * N + (j - 1)] + local_u[i * N + (j + 1)]);
                double diff = fabs(local_u[i * N + j] - local_w[i * N + j]);
                if (diff > max_diff) {
                    max_diff = diff; // ����������
                }
            }
        }
        MPI_Send(&max_diff, 1, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);                                                         // ����������
        MPI_Send(&local_w[N], (real_end_Arow - real_begin_Arow) * N, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD); // ���͸��º���Ӿ��� w
    }
    printf("%d process ending\n", my_rank);
    free(local_w); // �ͷ��Ӿ����ڴ�
    free(local_u);
}
```

### ���н��
         1  18.737475
         2  9.368737
         4  4.098823
         8  2.289577
        16  1.136604
        32  0.568201
        64  0.282805
       128  0.141777
       256  0.070808
       512  0.035427
      1024  0.017707
      2048  0.008856
      4096  0.004428
      8192  0.002210
     16384  0.001043
    At 16955 , episilon err reached, gonna kill all processes
    Elapsed time: 17.741727 seconds

�Ա�openmp�汾��

         1  18.737475
         2  9.368737
         4  4.098823
         8  2.289577
        16  1.136604
        32  0.568201
        64  0.282805
       128  0.141777
       256  0.070808
       512  0.035427
      1024  0.017707
      2048  0.008856
      4096  0.004428
      8192  0.002210
     16384  0.001043
     16955  0.001000
    Error tolerance achieved.
    Wallclock time = 60.274613

���Կ�����ȷ�Եõ���֤��ͬʱ����ʱ����졣

### �����Ż�
���ǵ�����Ծ���Ĳ������Ƕ�ά������Ȼ��Ծ���Ԫ�ؽ��в������������Կ���ʹ��һ��lambda�������ε�һ����ά��������traverse()ʵ�ָ����ɵĲ��� 

## ���ܲ���
������һ��Pthreadʵ��parallel_for�������ܲ��ԣ�����ʱ����ԣ��ڴ���ԡ�
### ʱ�����
���Գ����ڲ�ͬ�����ģ�ͼ����ģ������£�����ʱ���������������ʱ��̫���޷���� 
``` makefile
run: //makefile
	for n in 1, 2, 4, 8, 16; do \
		for size in 128 256 512 1024 2048 4096 8192; do \
			echo "Running program with size $$size and $$n threads" | tee -a output.txt; \
			./$(PROGRAM) $$size $$size $$n | tee -a output.txt; \
		done; \
		echo "End of program with $$n threads" | tee -a output.txt; \
		echo "" | tee -a output.txt; \
	done; \
	echo "End of program" | tee -a output.txt; \
	echo "" | tee -a output.txt;
```
| ����ά�� | 1 ���� (��) | 2 ���� (��) | 4 ���� (��) | 8 ���� (��) | 16 ���� (��) |
|----------|-------------|-------------|-------------|-------------|--------------|
| 128      | 1.472873    | 0.979901    | 0.795760    | 0.891577    | 1.884200     |
| 256      | 11.536146   | 6.826970    | 4.473302    | 8.035092    | 8.782962     |
| 512      | 78.093475   | 42.615858   | 21.921012   | 15.068684   | 27.455385    |
| 1024     | 322.881929  | 174.309762  | 96.825203   | 59.636986   | 74.984984    |
| 2048     | 1273.401172 | 666.447201  | 347.250282  | 214.609258  | 246.510050   |
| 4096     | 2930.082439 | 2654.926696 | 1391.575813 | 791.794789  | 822.905625   |
| 8192     |             | 10927.505105|             | 2950.624220 | 2230.558220  |
### �ڴ����
ʹ�� Valgrind �����ڴ����
����ʹ���������������� Valgrind������������ڴ�ʹ������������Ƕ��ڴ����ģ�

�������500 500 8�̵߳ļ�������
```bash
valgrind --tool=massif --time-unit=B --stacks=yes ./main 500 500 8
```
���������
- --tool=massif��ָ��ʹ�� massif �����������ڴ����ġ�
- --time-unit=B�����ֽڣ�B��Ϊ��λ����ʾ�ڴ���������
- --stacks=yes������ջ�ڴ�ķ�����������׽ջ�ϵ��ڴ����ġ�
- ./your_exe��������Ҫ�����Ŀ�ִ���ļ���


massif �����־ (massif.out.pid) �� ms_print ��ӡ
����������ʱ��massif ������һ������ļ���ͨ������Ϊ massif.out.pid������ pid �ǽ��̵� ID������ļ������˳����ڲ�ͬʱ�����ڴ��������ݡ�

ʹ�� ms_print �����������Ϳ��ӻ�����־�ļ���
``` bash
ms_print massif.out.pid
```

�ڼ���500* 500������ ʹ��8���߳�
![alt text](image.png)
��ƵĲ��з�ʽ��һ��ʼ������2��500*500��double�������Ĺ�4MB���ң�ͼƬ˵������û�ж�����ķ��ա�
����500 * 500������ʹ��4���߳�
![alt text](image-1.png)
���Կ����ڴ����Ļ���һ�£�˵���߳��������ڴ�������Ӱ�첻��
�ٿ���256* 256��������8�̼߳���
![alt text](image-2.png)
���Կ����ڴ����ļ��ٵ���ԭ����1/4
����M * N�������ģ���ڴ�������O(M * N)
���鿴�ڴ�������Դ
![alt text](image-3.png)
���Կ���98%���ϵ��ڴ涼�Ƕѷ�������ģ���֮��Ĵ󲿷����ǿ����������������