import java.util.Random;

public class j_m {

    public static void j_m(int mSize, int nSize, int kSize) {

        double[][] a = new double[mSize][nSize];
        double[][] b = new double[nSize][kSize];
        double[][] c = new double[mSize][kSize];
        Random random = new Random();

 
        for (int i = 0; i < mSize; i++) {
            for (int j = 0; j < nSize; j++) {
                a[i][j] = random.nextDouble() * 10000; 
            }
        }
        for (int i = 0; i < nSize; i++) {
            for (int j = 0; j < kSize; j++) {
                b[i][j] = random.nextDouble() * 10000; 
            }
        }

  
        long startTime = System.currentTimeMillis();


        for (int i = 0; i < mSize; i++) {
            for (int k = 0; k < nSize; k++) {
                for (int j = 0; j < kSize; j++) {
                    c[i][j] += a[i][k] * b[k][j];
                }
            }
        }


        long endTime = System.currentTimeMillis();

        double timeUsed = (endTime - startTime) / 1000.0;
        System.out.printf("Java Matrix Multiplying of %dx%d and %dx%d took %.6f seconds\n",
                          mSize, nSize, nSize, kSize, timeUsed);
    }

    public static void main(String[] args) {
        if (args.length != 3) {
            System.err.println("Usage: java JMatrixMultiplication M_SIZE N_SIZE K_SIZE");
            return;
        }
        int mSize = Integer.parseInt(args[0]);
        int nSize = Integer.parseInt(args[1]);
        int kSize = Integer.parseInt(args[2]);
        j_m(mSize, nSize, kSize);
    }
}
