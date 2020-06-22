
#include<mfem.hpp>
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

class ConvergenceRates
{
private:

#ifdef MFEM_USE_MPI
   MPI_Comm comm;
   int comm_flag; // 0 - local, 1 - global over 'comm'
#endif
   int counter=0;
   int print_flag=0;
   Array<double> error;
   Array<double> rates;
   Array<int> ndofs;
public:
   ConvergenceRates();
#ifdef MFEM_USE_MPI
   ConvergenceRates(MPI_Comm _comm);
#endif
   // Clear any internal data
   void Clear();

   // Add Scalar Solution 
   void AddSolution(GridFunction * gf, Coefficient * u_ex);

   // Add Vector Solution 
   void AddSolution(GridFunction * gf, VectorCoefficient * U_ex);

   // Get L2 error for step n
   double GetL2Error(int n);

   // Get all L2 errors
   Array<double> * GetL2Errors();

   // Get rates for all refinements
   Array<double> * GetRates();

   // Print rates and errors
   void Print();

   ~ConvergenceRates(){};
};