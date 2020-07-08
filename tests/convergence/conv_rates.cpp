
#include "conv_rates.hpp"

Convergence::Convergence()
{
#ifdef MFEM_USE_MPI
   comm_flag = 0;
#endif
   print_flag = 1;
}

#ifdef MFEM_USE_MPI
Convergence::Convergence(MPI_Comm _comm) : comm(_comm)
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

void Convergence::Clear()
{
   counter=0;
   dcounter=0;
   L2Errors.SetSize(0);
   L2Rates.SetSize(0);
   DErrors.SetSize(0);
   DRates.SetSize(0);
   EnErrors.SetSize(0);
   EnRates.SetSize(0);
}


double Convergence::GetNorm(GridFunction * gf, Coefficient * u, VectorCoefficient * U)
{
   double norm;
   int intorder = 2*gf->FESpace()->GetFE(0)->GetOrder();
   int order = gf->FESpace()->GetOrder(0);
   int order_quad = max(2, 2*order+1);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }
#ifdef MFEM_USE_MPI   
   if (comm_flag)
   {
      ParMesh * pmesh = (dynamic_cast<ParGridFunction *>(gf))->ParFESpace()->GetParMesh();
      if (u)
      {
         norm = ComputeGlobalLpNorm(2.0,*u,*pmesh,irs);
      }
      else if (U)
      {
         norm = ComputeGlobalLpNorm(2.0,*U,*pmesh,irs);
      }
   }
#else
   Mesh * mesh = gf->FESpace()->GetMesh();
   if (u)
   {
      norm = ComputeGlobalLpNorm(2.0,*u,*pmesh,irs);
   }
   else if (U)
   {
      norm = ComputeGlobalLpNorm(2.0,*U,*pmesh,irs);
   }
#endif   
   return norm;
}

void Convergence::AddL2Error(GridFunction * gf, 
                             Coefficient * u, VectorCoefficient * U)
{
   int tdofs = gf->FESpace()->GetTrueVSize();
#ifdef MFEM_USE_MPI   
   if (comm_flag)
   {
      MPI_Allreduce(&tdofs,&tdofs,1,MPI_INT,MPI_SUM,comm);
   }
#endif   
   ndofs.Append(tdofs);
   double L2Err;
   double L2Norm;
   if (u)
   {
      L2Err = gf->ComputeL2Error(*u);
      CoeffNorm = GetNorm(gf,u,nullptr);
   }
   else if (U) 
   {
      L2Err = gf->ComputeL2Error(*U);
      CoeffNorm = GetNorm(gf,nullptr,U);
   }
   else
   {
      MFEM_ABORT("Exact Solution Coefficient pointer is NULL");
   }
   L2Errors.Append(L2Err);
   double val = (counter) ? log(L2Errors[counter-1]/L2Err)/log(2.0) : 0.0;
   L2Rates.Append(val);
   counter++;
}


void Convergence::AddGf(GridFunction * gf, Coefficient * u, 
                  VectorCoefficient * grad, 
                  Coefficient * ell_coeff, double Nu)
{
   cont_type = gf->FESpace()->FEColl()->GetContType();
   
   MFEM_VERIFY((cont_type == mfem::FiniteElementCollection::CONTINUOUS) ||
               (cont_type == mfem::FiniteElementCollection::DISCONTINUOUS),
               "This Constructor is intented for H1 or L2 Elements")
   
   AddL2Error(gf,u, nullptr);

   if (grad)
   {
      double GradErr = gf->ComputeGradError(grad);
      DErrors.Append(GradErr);
      double err = sqrt(L2Errors[counter-1]*L2Errors[counter-1] + GradErr*GradErr);
      EnErrors.Append(err);
      double val = (dcounter) ? log(DErrors[dcounter-1]/GradErr)/log(2.0) : 0.0;
      double eval = (dcounter) ? log(EnErrors[dcounter-1]/err)/log(2.0) : 0.0;
      DRates.Append(val);
      EnRates.Append(eval);
      CoeffDNorm = GetNorm(gf,nullptr,grad);
      dcounter++;
      MFEM_VERIFY(counter == dcounter, "Number of Added solutions and derivatives do not match")
   }

   if (cont_type == mfem::FiniteElementCollection::DISCONTINUOUS)
   {
      if (ell_coeff)
      {
         double DGErr = gf->ComputeDGFaceJumpError(u,ell_coeff,Nu);
         DGFaceErrors.Append(DGErr);
         double val = (fcounter) ? log(DGFaceErrors[fcounter-1]/DGErr)/log(2.0) : 0.0;
         DGFaceRates.Append(val);
         fcounter++;
         MFEM_VERIFY(fcounter == dcounter, "Number of Added solutions missmatch");
      }
   }

}
void Convergence::AddGf(GridFunction * gf, VectorCoefficient * U, 
                  VectorCoefficient * curl, Coefficient * div)
{
   cont_type = gf->FESpace()->FEColl()->GetContType();

   AddL2Error(gf,nullptr,U);
   double DErr = 0.0;
   int derivative = 0;
   if (curl)
   {
      DErr = gf->ComputeCurlError(curl);
      CoeffDNorm = GetNorm(gf,nullptr,curl);
      derivative = 1;
   }
   else if (div)
   {
      DErr = gf->ComputeDivError(div);
      CoeffDNorm = GetNorm(gf,div,nullptr); // update coefficient norm
      derivative = 1;
   }
   if (derivative)
   {
      double err = sqrt(L2Errors[counter-1]*L2Errors[counter-1] + DErr*DErr);
      DErrors.Append(DErr);
      EnErrors.Append(err);
      double val = (dcounter) ? log(DErrors[dcounter-1]/DErr)/log(2.0) : 0.0;
      double eval = (dcounter) ? log(EnErrors[dcounter-1]/err)/log(2.0) : 0.0;
      DRates.Append(val);
      EnRates.Append(eval);
      dcounter++;
      MFEM_VERIFY(counter == dcounter, "Number of added solutions and derivatives do not match")
   }
}


double Convergence::GetL2Error(int n)
{
   MFEM_VERIFY(n<= counter,"Step out of bounds")
   return L2Errors[n];
}

Array<double> * Convergence::GetRates()
{
   return &L2Rates;
}

Array<double> * Convergence::GetL2Errors()
{
   return &L2Errors;
}

void Convergence::Print(bool relative)
{
   if (print_flag)
   {
      string title = (relative) ? "Relative " : "Absolute ";
      cout << endl;
      cout << " -------------------------------------------" << endl;
      cout << "            " << title << " L2 Error              " << endl;
      cout << " -------------------------------------------"
           << endl;
      cout << right<< setw(11)<< "DOFs "<< setw(13) << "Error ";
      cout <<  setw(15) << "Rate " << endl;
      cout << " -------------------------------------------"
           << endl;
      cout << setprecision(4);
      double d = (relative) ? CoeffNorm : 1.0;
      for (int i =0; i<counter; i++)
      {
         cout << right << setw(10)<< ndofs[i] << setw(16) 
              << scientific << L2Errors[i]/d << setw(13)  
              << fixed << L2Rates[i] << endl;
      }
      cout << endl;
      if (dcounter == counter)
      {
         string dname;
         switch (cont_type)
         {
            case 0: dname = "Grad"; break;
            case 1: dname = "Curl"; break;
            case 2: dname = "Div ";  break;
            case 3: dname = "DG Grad ";  break;
            default: break;
         }
         cout << " -------------------------------------------" << endl;
         cout << "              " << title << dname << " Error        " << endl;
         cout << " -------------------------------------------" << endl;
         cout << right<< setw(11)<< "DOFs "<< setw(13) << "Error ";
         cout <<  setw(15) << "Rate " << endl;
         cout << " -------------------------------------------"
           << endl;
         cout << setprecision(4);
         d = (relative) ? CoeffDNorm : 1.0;
         for (int i =0; i<dcounter; i++)
         {
            cout << right << setw(10)<< ndofs[i] << setw(16) 
                 << scientific << DErrors[i]/d << setw(13)  
                 << fixed << DRates[i] << endl;
         }
         cout << endl;
         switch (cont_type)
         {
            case 0: dname = "H1"; break;
            case 1: dname = "H(Curl)"; break;
            case 2: dname = "H(Div)";  break;
            case 3: dname = "DG H1";  break;
            default: break;
         }

         d = (relative) ? sqrt(CoeffNorm*CoeffNorm + CoeffDNorm*CoeffDNorm) : 1.0;

         cout << " -------------------------------------------" << endl;
         cout << "              " << title << dname << " Error        " << endl;
         cout << " -------------------------------------------" << endl;
         cout << right<< setw(11)<< "DOFs "<< setw(13) << "Error ";
         cout <<  setw(15) << "Rate " << endl;
         cout << " -------------------------------------------"
           << endl;
         cout << setprecision(4);
         for (int i =0; i<dcounter; i++)
         {
            cout << right << setw(10)<< ndofs[i] << setw(16) 
                 << scientific << EnErrors[i]/d << setw(13)  
                 << fixed << EnRates[i] << endl;
         }
         cout << endl;

         if (cont_type == 3)
         {
            cout << " -------------------------------------------" << endl;
            cout << "            DG Face Jump Error          " << endl;
            cout << " -------------------------------------------"
                 << endl;
            cout << right<< setw(11)<< "DOFs "<< setw(13) << "Error ";
            cout <<  setw(15) << "Rate " << endl;
            cout << " -------------------------------------------"
                 << endl;
            cout << setprecision(4);
            for (int i =0; i<counter; i++)
            {
               cout << right << setw(10)<< ndofs[i] << setw(16) 
                    << scientific << DGFaceErrors[i] << setw(13)  
                    << fixed << DGFaceRates[i] << endl;
            }
            cout << endl;
         }
      }
   }
}
