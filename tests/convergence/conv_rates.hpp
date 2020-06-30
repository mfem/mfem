
#include<mfem.hpp>
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

class Convergence
{
private:

#ifdef MFEM_USE_MPI
   MPI_Comm comm;
   int comm_flag; // 0 - local, 1 - global over 'comm'
#endif
   int counter=0;
   int dcounter=0;
   int cont_type;
   int print_flag=0;
   Array<double> L2Errors;
   Array<double> DErrors;
   Array<double> EnErrors;
   Array<double> L2Rates;
   Array<double> DRates;
   Array<double> EnRates;
   Array<int> ndofs;
   double CoeffNorm;
   double CoeffDNorm;
   void AddL2Error(GridFunction * gf, Coefficient * u, VectorCoefficient * U);
   void AddGf(GridFunction * gf, Coefficient * u, VectorCoefficient * grad);
   void AddGf(GridFunction * gf, VectorCoefficient * u, 
                        VectorCoefficient * curl, Coefficient * div);

   double GetNorm(GridFunction * gf, Coefficient * u, VectorCoefficient * U);
public:
   Convergence();
#ifdef MFEM_USE_MPI
   Convergence(MPI_Comm _comm);
#endif
   // Clear any internal data
   void Clear();

   void AddGridFunction(GridFunction * gf, Coefficient * u, VectorCoefficient * grad = NULL)
   {
      AddGf(gf, u, grad);
   }
   void AddGridFunction(GridFunction * gf, VectorCoefficient * u)
   {
      AddGf(gf,u, nullptr, nullptr);
   }
   void AddGridFunction(GridFunction * gf, VectorCoefficient * u, VectorCoefficient * curl)
   {
      AddGf(gf,u, curl, nullptr);
   }
   void AddGridFunction(GridFunction * gf, VectorCoefficient * u, Coefficient * div) 
   {
      AddGf(gf,u, nullptr, div);
   }
   // Get L2 error for step n
   double GetL2Error(int n);

   // Get all L2 errors
   Array<double> * GetL2Errors();

   // Get rates for all refinements
   Array<double> * GetRates();

   // Print rates and errors
   void Print(bool relative = false);

   ~Convergence(){};
};