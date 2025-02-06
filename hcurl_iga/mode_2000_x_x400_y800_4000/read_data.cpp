#include <iostream>
#include <fstream>
#include <vector>
 
int main() {
    std::ifstream file_Ex_real("E_x_real.txt");
    std::ifstream file_Ex_imag("E_x_imag.txt");
    std::ifstream file_Ey_real("E_y_real.txt");
    std::ifstream file_Ey_imag("E_y_imag.txt");
    std::ifstream file_Ez_real("E_z_real.txt");
    std::ifstream file_Ez_imag("E_z_imag.txt");

    std::vector<double> data_Ex_real;
    std::vector<double> data_Ex_imag;
    std::vector<double> data_Ey_real;
    std::vector<double> data_Ey_imag;
    std::vector<double> data_Ez_real;
    std::vector<double> data_Ez_imag;

    double temp_Ex_real;
    double temp_Ex_imag;
    double temp_Ey_real;
    double temp_Ey_imag;
    double temp_Ez_real;
    double temp_Ez_imag;

    // 检查文件是否成功打开
    if (!file_Ex_real.is_open()) {
        std::cerr << "无法打开文件" << std::endl;
        return 1;
    }
    if (!file_Ex_imag.is_open()) {
        std::cerr << "无法打开文件" << std::endl;
        return 1;
    }
    if (!file_Ey_real.is_open()) {
        std::cerr << "无法打开文件" << std::endl;
        return 1;
    }
    if (!file_Ey_imag.is_open()) {
        std::cerr << "无法打开文件" << std::endl;
        return 1;
    }
    if (!file_Ez_real.is_open()) {
        std::cerr << "无法打开文件" << std::endl;
        return 1;
    }
    if (!file_Ez_imag.is_open()) {
        std::cerr << "无法打开文件" << std::endl;
        return 1;
    }
 
    // 读取文件中的数字并存入vector
    while (file_Ex_real >> temp_Ex_real) {
        data_Ex_real.push_back(temp_Ex_real);
    }
    while (file_Ex_imag >> temp_Ex_imag) {
        data_Ex_imag.push_back(temp_Ex_imag);
    }
    while (file_Ey_real >> temp_Ey_real) {
        data_Ey_real.push_back(temp_Ey_real);
    }
    while (file_Ey_imag >> temp_Ey_imag) {
        data_Ey_imag.push_back(temp_Ey_imag);
    }
    while (file_Ez_real >> temp_Ez_real) {
        data_Ez_real.push_back(temp_Ez_real);
    }
    while (file_Ez_imag >> temp_Ez_imag) {
        data_Ez_imag.push_back(temp_Ez_imag);
    }

    file_Ex_real.close();
    file_Ex_imag.close();
    file_Ey_real.close();
    file_Ey_imag.close();
    file_Ez_real.close();
    file_Ez_imag.close();
 
    double x, y, z;
    double dx = 20;
    double dy = 20;
    int I = 200,J = 200;
    z = 0;
    x = 2010.0;
    y = 2010.0;

    int Ix = (int)(x/dx);
    int Iy = (int)(y/dy);

    double addx = 1.0*(x - Ix*dx)/dx;
    double addy = 1.0*(y - Iy*dy)/dy;

    double Ex_real = (1.0-addy)*((1.0-addx)*data_Ex_real[I*Iy + Ix] + addx*data_Ex_real[I*Iy + Ix+1])+
                     addy*((1.0-addx)*data_Ex_real[I*(Iy+1) + Ix] + addx*data_Ex_real[I*(Iy+1) + Ix+1]);

    std::cout<<Ex_real<<" "<<data_Ex_real[I*Iy + Ix]<<" "<<data_Ex_real[I*Iy + Ix+1]<<" "<<data_Ex_real[I*(Iy+1) + Ix]<<" "<<data_Ex_real[I*(Iy+1) + Ix+1]<<" "<<addx<<" "<<addy<<std::endl;
    
    return 0;
}