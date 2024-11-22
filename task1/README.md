using diy openmp replace the origin one
    这是原版的实现，时间控制在20秒以内
HEATED_PLATE_OPENMP
  C/OpenMP version
  A program to solve for the steady state temperature distribution
  over a rectangular plate.

  Spatial grid of 500 by 500 points.
  The iteration will be repeated until the change is <= 1.000000e-03
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

HEATED_PLATE_OPENMP:
  Normal end of execution.

    更改了循环内的对u矩阵赋值的函数，导致消耗时间增多，
    HEATED_PLATE_OPENMP
  C/OpenMP version
  A program to solve for the steady state temperature distribution
  over a rectangular plate.

  Spatial grid of 500 by 500 points.
  The iteration will be repeated until the change is <= 1.000000e-03
  Number of processors available = 16
  Number of threads =              16

  MEAN = 74.949900
0x7fff751fdb50
0.0000

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
       512  0.035430
      1024  0.017712
      2048  0.008824
      4096  0.004677
      8192  0.002722

     11578  0.000931

  Error tolerance achieved.
  Wallclock time = 375.170502

HEATED_PLATE_OPENMP:
  Normal end of execution.