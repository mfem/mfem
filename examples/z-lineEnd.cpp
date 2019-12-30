#include <fstream>
#include <iostream>

using namespace std;

int main(int argc, char *argv[])
{
   std::ofstream iterfile;
   iterfile.open("iters.txt", std::ios_base::app);
   iterfile << std::endl;
   iterfile.close();
   return 0;
}