#pragma once
#include "mfem.hpp"
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
namespace mfem
{
// Make a new (parallel) grid function from a given (parallel) finite element space.
GridFunction *MakeGridFunction(FiniteElementSpace *fes);
// Make a new (parallel) linear form from a given (parallel) finite element space.
LinearForm *MakeLinearForm(FiniteElementSpace *fes);
// Make a new (parallel) nonlinear form from a given (parallel) finite element space.
NonlinearForm *MakeNonlinearForm(FiniteElementSpace *fes);
// Make a new (parallel) bilinear form from a given (parallel) finite element space.
BilinearForm *MakeBilinearForm(FiniteElementSpace *fes);
// Make a new (parallel) mixed-bilinear form from a given (parallel) finite element spaces.
MixedBilinearForm *MakeMixedBilinearForm(FiniteElementSpace *trial_fes,
                                         FiniteElementSpace *test_fes);

class TableLogger
{
public:
   enum dtype {DOUBLE, INT};
protected:
   // Double data to be printed.
   std::vector<double*> monitored_double_data;
   // Int data to be printed
   std::vector<int*> monitored_int_data;
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
   // my_rank = IsSerial ? 0 : Mpi::WorldRank()
   int my_rank;
   std::unique_ptr<std::ofstream> file;
private:

public:
   // Create a logger that prints a row of variables for each call of Print
   TableLogger(std::ostream &os=mfem::out);
   // Set column width of the table to be printed
   void setw(const int column_width) {w = column_width;}
   // Add double data to be monitored
   void Append(const std::string name, double &val);
   // Add double data to be monitored
   void Append(const std::string name, int &val);
   // Print a row of currently monitored data. If it is called
   void Print();
   // Save data to a file whenever Print is called.
   void SaveWhenPrint(const char *filename, std::ios::openmode mode=std::ios::trunc);
   // Close file manually.
   void CloseFile() {if (file) file.reset(nullptr); }


};
}