# CUDA GEMM and CONV for RGB_IMAGE
## illustration
gemm: cpu, gpu (global mem), gpu (shared mem), cublas;
conv: sliding window, im2col(using cublas for gemm), cudnn.
## source code
in the /src
conv_main, gemm_main for convolution, gemm respectively
other files are for used.
## build
make all:for compile the src file.
make run1:test the gemm for block size, and matrixs size. the output is in gemm_ouput.txt;
make run2:test the convolution for strde and the matrixs size. The output is in conv_output.txt.
