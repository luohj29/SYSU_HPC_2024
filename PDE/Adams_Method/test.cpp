/*
    u' = -5u
    u(n+1) = u(n) +inerval(from n to n+1) f(x)dx
    三阶adams法:先用三阶牛顿外插法获得插值的近似函数:f(n+1) = -f(n-3) + 4f(n-2) -6f(n-1) +4f(n),再代入积分方程,用梯形积分法近似积分
    代入原始,得到u(n+1) = u(n) +h/2*(5f(n)-6f(n-1)+4f(n-2)-f(n-3))
    注意初始三个点公式 u(n+1) = u(n) + f(n)*h
*/

#include<iostream>
#include<cmath>
#include<vector>
#include<iomanip>
// Define the function
double U_0 = 1.0;
double U_1 = 1.0;
double X_0 = 0.0;
double H = 0.001;

double funct(double x)
{
    return  x;
}
void test(std::vector<double> &x,std::vector<double> &f,std::vector<double> &u, int start, int end){
// u(n+1) = u(n) +h/2*(5f(n)-6f(n-1)+4f(n-2)-f(n-3))
    for (int i = start; i<end-2; i++){
        u[i + 2] = u[i + 1] + h * f[i + 1];
        f[i + 2] = u[i + 2];
        u[i + 2] = 2 * u[i+1] -u[i]+ H / 4 * (2* f[i+2] - 2*f[i]);
        f[i + 2] = u[i + 2];
        x[i + 2] = x[i] + H;
    }
    return ;
}


void print_result(std::vector<double> &x,std::vector<double> &f,std::vector<double> &u, int start, int end){
    for (int i = start; i<=end; i++){
        std::cout <<std::fixed <<std::setprecision(7) << x[i] << '\t' << f[i] << '\t' << u[i] << std::endl;
    }
    return;
}
int main(){
    std::vector<double> x_points;  //record the x points, x_ponts[0] = 0,and the step is h
    std::vector<double> f_points;  //if  3 order, then 4 previous is used to predict the next one(using the adams method), the first 3 maybe out by sim_Eruler
    std::vector<double> u_points;
    x_points.push_back(X_0);
    x_points.push_back(x_points[i] + H);
    f_points.push_back(funct(U_0));
    f_points.push_back(funct(U_1));
    u_points.push_back(U_0);
    u_points.pop_back(U_1);
    // for (int i = 0; i < 2; i++)
    // { // using the eruler method to predict the first 3 f_points
    //     u_points.push_back(u_points[i]+H*f_points[i]);
    //     f_points.push_back(funct(u_points[i+1]));
    //     x_points.push_back(x_points[i] + H);
    // }
    double end = 1 / H;
    x_points.resize(end+1);
    u_points.resize(end+1);
    f_points.resize(end+1);

    test(x_points, f_points, u_points, 0, (int)end);
    std::cout << "info: step = " << H << std::endl;
    std::cout << "test at 1: " << u_points[end] << std::endl;
    double adam_bashfort_result = u_points[end];


    // std::fill(u_points.begin() + 4, u_points.end(), 0);
    // std::fill(x_points.begin() + 4, x_points.end(), 0);
    // std::fill(f_points.begin() + 4, f_points.end(), 0);
    // Adams_Moulton_3Order(x_points, f_points, u_points, 4, (int)end);
    // print_result(x_points, f_points, u_points, 0, (int)end);
    std::cout << "test at 1: " << u_points[end] << std::endl;
    std::cout << "Real result at 1: " << 3/2<< std::endl;
    std::cout << "Loss is:" <<std::fixed <<std::setprecision(7)<< std::abs(adam_bashfort_result - exp(-5 * 1)) << std::endl;
    // std::cout << "Loss2 is:" <<std::fixed <<std::setprecision(7)<< std::abs(u_points[end] - exp(-5 * 1)) << std::endl;
    return 0;
}