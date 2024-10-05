#include <iomanip>
#include <iostream>
#include <vector>
#include "mfem.hpp"

class TableLogger
{
public:
   enum dtype { DOUBLE, INT };

protected:
   // Double data to be printed.
   std::vector<double *> data_double;
   // Int data to be printed
   std::vector<int *> data_int;
   // Data type for each column
   std::vector<dtype> data_order;
   // Name of each monitored data
   std::vector<std::string> names;
   // Output stream
   std::ostream &os;
   // Column width
   int w;
   // Whether the variable name row has been printed or not
   bool var_name_printed;
   bool isRoot; // true if serial or root in parallel
   std::unique_ptr<std::ofstream> file;

private:
public:
   // Create a logger that prints a row of variables for each call of Print
   TableLogger(std::ostream &os = std::cout);
   // Set column width of the table to be printed
   void setw(const int column_width) { w = column_width; }
   // Add double data to be monitored
   void Append(const std::string name, double &val);
   // Add double data to be monitored
   void Append(const std::string name, int &val);
   // Print a row of currently monitored data. If it is called
   void Print(bool print_valname=false);
   // Save data to a file whenever Print is called.
   void SaveWhenPrint(std::string filename,
                      std::ios::openmode mode = std::ios::out);
   // Close file manually.
   void CloseFile() { if (file) { file.reset(nullptr); } }
};

