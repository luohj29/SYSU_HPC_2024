#include <cstdio>
#include <cmath>

// Define the function
double funct(double x) {
    return -5 * x;
}

// Function to perform the Euler method
double sim_Eruler(double step, double (*funct)(double), double x_0, double y_0, double x) { 
    for(double i = x_0; i < x; i += step) {
        y_0 += step * funct(y_0);  // Use i instead of y_0 to call funct correctly
        printf("%f\n", y_0);
    }
    return y_0;
}

// Function to get the predicted value for the improved Euler method
double get_y_predict(double step, double y_0, double (*funct)(double)) {
    double y_predict = y_0;
    double delta_base = 0;
    int i = 1;
    while (i++)
    {                                                                // iteration to make the y_predict precise!!!
        double delta = ((funct(y_0) + funct(y_predict)) / 2) * step; // Corrected parentheses
        if (i ==1) printf("\n%f", std::abs(delta));
        if (std::abs(delta-delta_base) < 0.005) {          
            // printf("out\n");
            break;
        } else {
            delta_base = delta;
            y_predict = y_0 + delta;
        }
        // i--;
    }
    return y_predict;
}

// Improved Euler method implementation
double improved_Eruler(double step, double (*funct)(double), double x_0, double y_0, double x) { 
    double y_predict;
    for(double i = x_0; i < x; i += step) {
        y_predict = get_y_predict(step, y_0, funct);  // Call the function correctly
        printf("%f %f\n", y_0, y_predict);
        y_0 += step * 0.5 * (funct(y_0) + funct(y_predict));  // Corrected formula
    }
    return y_0;
}

int main() {
    double step = 0.01; // Step size
    double x_0 = 0;     // Initial x value
    double y_0 = 1;     // Initial y value
    double x = 1;     // Final x value


    // Call the Euler method function with the function pointer
    double result_sim_Eruler = sim_Eruler(step, funct, x_0, y_0, x);
    double result_improved_Eruler = improved_Eruler(step, funct, x_0, y_0, x);

    // Output the result
    printf("info:\nstep: %f, x: %f\n", step, x);
    printf("The result from sim_Eruler is: %lf\n", result_sim_Eruler);
    printf("The result from improved_Eruler is: %lf\n", result_improved_Eruler); 
    printf("The real result is: %lf\n", exp(-5*x));
    return 0;
}
