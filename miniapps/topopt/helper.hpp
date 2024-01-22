#pragma once
#include "mfem.hpp"
#include <vector>
#include <string>
#include <iostream>
namespace mfem
{
GridFunction *MakeGridFunction(FiniteElementSpace *fes);
LinearForm *MakeLinearForm(FiniteElementSpace *fes);
NonlinearForm *MakeNonlinearForm(FiniteElementSpace *fes);
BilinearForm *MakeBilinearForm(FiniteElementSpace *fes);
MixedBilinearForm *MakeMixedBilinearForm(FiniteElementSpace *trial_fes,
                                         FiniteElementSpace *test_fes);

class TableLogger
{
public:
   enum dtype {DOUBLE, INT};
protected:
   // Array<double*> monitored_double_data;
   std::vector<double*> monitored_double_data;
   std::vector<int*> monitored_int_data;
   std::vector<dtype> data_order;
   std::vector<std::string> names;
   std::ostream &os;
   int w;
   bool var_name_printed;
   int my_rank;
private:

public:
   TableLogger(std::ostream &os=mfem::out);
   void setw(const int column_width){w = column_width;}
   void Append(const std::string name, double &val);
   void Append(const std::string name, int &val);
   void Print();


};
}