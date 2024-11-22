import sys
import numpy as np
import time

def matrix_multiplication(m, n, k):

    a = np.random.uniform(0, 10000, (m, n))  
    b = np.random.uniform(0, 10000, (n, k))  
    c = np.zeros((m, k), dtype=np.float64)  

 
    start = time.time()
    for i in range(m):
        for l in range(n):
            for j in range(k):  
                c[i, j] += a[i, l] * b[l, j]
    end = time.time()


    time_use = end - start
    print(f"Python Matrix Multiplying of {m}x{n} and {n}x{k} used time of {time_use:.6f} seconds")

if __name__ == "__main__":

    m = int(sys.argv[1])
    n = int(sys.argv[2])
    k = int(sys.argv[3])
    matrix_multiplication(m, n, k)
