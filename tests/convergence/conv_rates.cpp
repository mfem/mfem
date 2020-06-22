
#include "conv_rates.hpp"

ConvergenceRates::ConvergenceRates()
{
#ifdef MFEM_USE_MPI
   comm_flag = 0;
#endif
   print_flag = 1;
}

#ifdef MFEM_USE_MPI
ConvergenceRates::ConvergenceRates(MPI_Comm _comm) : comm(_comm)
{
   comm_flag = 1;
   int rank;
   MPI_Comm_rank(comm, &rank);
   if (rank==0) 
   {
      print_flag = 1;
   }
}
#endif

void ConvergenceRates::Clear()
{
   counter=0;
   rates.SetSize(0);
}

void ConvergenceRates::AddSolution(GridFunction * gf, Coefficient * u)
{
   double L2err = gf->ComputeL2Error(*u);
   error.Append(L2err);
   int tdofs = gf->FESpace()->GetTrueVSize();
   if (comm_flag)
   {
      MPI_Allreduce(&tdofs,&tdofs,1,MPI_INT,MPI_SUM,comm);
   }
   ndofs.Append(tdofs);

   double val = (counter) ? log(error[counter-1]/L2err)/log(2.0) : 0.0;
   rates.Append(val);
   
   counter++;
}
void ConvergenceRates::AddSolution(GridFunction * gf, VectorCoefficient * U)
{
   double L2err = gf->ComputeL2Error(*U);
   error.Append(L2err);
   int tdofs = gf->FESpace()->GetTrueVSize();
   if (comm_flag)
   {
      MPI_Allreduce(&tdofs,&tdofs,1,MPI_INT,MPI_SUM,comm);
   }
   ndofs.Append(tdofs);

   double val = (counter) ? log(error[counter-1]/L2err)/log(2.0) : 0.0;
   rates.Append(val);
   
   counter++;
}
double ConvergenceRates::GetL2Error(int n)
{
   MFEM_VERIFY(n<= counter,"Step out of bounds")
   return error[n];
}

Array<double> * ConvergenceRates::GetRates()
{
   return &rates;
}

Array<double> * ConvergenceRates::GetL2Errors()
{
   return &error;
}

void ConvergenceRates::Print()
{
   if (print_flag)
   {
      cout << " -----------------------------------------"
           << endl;
      cout  << right<< setw(11)<< "DOFs "<< setw(15) << "L^2 error "<< setw(15);
      cout << "L^2 rate " << endl;
      cout << " -----------------------------------------"
           << endl;
      cout << setprecision(4);
      for (int i =0; i<counter; i++)
      {
         cout << right << setw(10)<< ndofs[i] << setw(16) << scientific << error[i] 
              << setw(13)  << fixed << rates[i] << endl;
      }
   }
}
