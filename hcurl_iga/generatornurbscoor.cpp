#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

int main() {

    // open a file for outputting the matrix
    ofstream outputfile;
    outputfile.open ("kaipengtestnurbs2.mesh");

    if (outputfile.is_open()) 
    {
        double dx = 0.1;//1.0/(kv_point-1);
        double x = 0.0; 
        double y = 0.0;
        double z = 0.0;

        for(int i = 1; i <= 9; i++)
        {
            x = i * dx;
            y = 0;
            z = 0;
            outputfile<<x<<" "<<y<<" "<<z<<endl;
        }

        for(int i = 1; i <= 9; i++)
        {
            x = 1-i * dx;
            y = 0.5;
            z = 0;
            outputfile<<x<<" "<<y<<" "<<z<<endl;
        }

        for(int i = 1; i <= 9; i++)
        {
            x = i * dx;
            y = 0;
            z = 0.1;
            outputfile<<x<<" "<<y<<" "<<z<<endl;
        }

        for(int i = 1; i <= 9; i++)
        {
            x = 1-i * dx;
            y = 0.5;
            z = 0.1;
            outputfile<<x<<" "<<y<<" "<<z<<endl;
        }

        for(int i = 1; i <= 4; i++)
        {
            x = 0;
            y = i * dx;
            z = 0;
            outputfile<<x<<" "<<y<<" "<<z<<endl;
        }

        for(int i = 1; i <= 4; i++)
        {
            x = 1;
            y = i * dx;
            z = 0;
            outputfile<<x<<" "<<y<<" "<<z<<endl;
        }

        for(int i = 1; i <= 4; i++)
        {
            x = 0;
            y = i * dx;
            z = 0.1;
            outputfile<<x<<" "<<y<<" "<<z<<endl;
        }

        for(int i = 1; i <= 4; i++)
        {
            x = 1;
            y = i * dx;
            z = 0.1;
            outputfile<<x<<" "<<y<<" "<<z<<endl;
        }


        for(int j = 1; j <= 4; j++)
            for(int i = 1; i <= 9; i++)
            {
                x = i*dx;
                y = 0.5-j*dx;
                z = 0;
                outputfile<<x<<" "<<y<<" "<<z<<endl;
            }

        for(int j = 1; j <= 4; j++)
            for(int i = 1; i <= 9; i++)
            {
                x = i*dx;
                y = j*dx;
                z = 0.1;
                outputfile<<x<<" "<<y<<" "<<z<<endl;
            }

    }

    outputfile.close();

    return 0;
}