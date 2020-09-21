#include "convergence.hpp"

using namespace std;


namespace mfem
{

void ConvergenceStudy::Reset()
{
   counter=0;
   dcounter=0;
   fcounter=0;
   cont_type=-1;
   print_flag=1;
   L2Errors.SetSize(0);
   L2Rates.SetSize(0);
   DErrors.SetSize(0);
   DRates.SetSize(0);
   EnErrors.SetSize(0);
   EnRates.SetSize(0);
   DGFaceErrors.SetSize(0);
   DGFaceRates.SetSize(0);
   ndofs.SetSize(0);
}

double ConvergenceStudy::GetNorm(GridFunction *gf, Coefficient *scalar_u,
                                 VectorCoefficient *vector_u)
{
   bool norm_set = false;
   double norm=0.0;
   int order = gf->FESpace()->GetOrder(0);
   int order_quad = std::max(2, 2*order+1);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }
#ifdef MFEM_USE_MPI
   ParGridFunction *pgf = dynamic_cast<ParGridFunction *>(gf);
   if (pgf)
   {
      ParMesh *pmesh = pgf->ParFESpace()->GetParMesh();
      if (scalar_u)
      {
         norm = ComputeGlobalLpNorm(2.0,*scalar_u,*pmesh,irs);
      }
      else if (vector_u)
      {
         norm = ComputeGlobalLpNorm(2.0,*vector_u,*pmesh,irs);
      }
      norm_set = true;
   }
#endif
   if (!norm_set)
   {
      Mesh *mesh = gf->FESpace()->GetMesh();
      if (scalar_u)
      {
         norm = ComputeLpNorm(2.0,*scalar_u,*mesh,irs);
      }
      else if (vector_u)
      {
         norm = ComputeLpNorm(2.0,*vector_u,*mesh,irs);
      }
   }
   return norm;
}

void ConvergenceStudy::AddL2Error(GridFunction *gf,
                                  Coefficient *scalar_u, VectorCoefficient *vector_u)
{
   int tdofs=0;
#ifdef MFEM_USE_MPI
   ParGridFunction *pgf = dynamic_cast<ParGridFunction *>(gf);
   if (pgf)
   {
      MPI_Comm comm = pgf->ParFESpace()->GetComm();
      int rank;
      MPI_Comm_rank(comm, &rank);
      print_flag = 0;
      if (rank==0) { print_flag = 1; }
      tdofs = pgf->ParFESpace()->GlobalTrueVSize();
   }
#endif
   if (!tdofs) { tdofs = gf->FESpace()->GetTrueVSize(); }
   ndofs.Append(tdofs);
   double L2Err;
   if (scalar_u)
   {
      L2Err = gf->ComputeL2Error(*scalar_u);
      CoeffNorm = GetNorm(gf,scalar_u,nullptr);
   }
   else if (vector_u)
   {
      L2Err = gf->ComputeL2Error(*vector_u);
      CoeffNorm = GetNorm(gf,nullptr,vector_u);
   }
   else
   {
      MFEM_ABORT("Exact Solution Coefficient pointer is NULL");
   }
   L2Errors.Append(L2Err);
   // Compute the rate of convergence by:
   // rate = log (||u - u_h|| / ||u - u_{h/2}||)/log(2)
   double val = (counter) ? log(L2Errors[counter-1]/L2Err)/log(2.0) : 0.0;
   L2Rates.Append(val);
   counter++;
}

void ConvergenceStudy::AddGf(GridFunction *gf, Coefficient *scalar_u,
                             VectorCoefficient *grad,
                             Coefficient *ell_coeff,
                             JumpScaling jump_scaling)
{
   cont_type = gf->FESpace()->FEColl()->GetContType();

   MFEM_VERIFY((cont_type == mfem::FiniteElementCollection::CONTINUOUS) ||
               (cont_type == mfem::FiniteElementCollection::DISCONTINUOUS),
               "This constructor is intended for H1 or L2 Elements")

   AddL2Error(gf,scalar_u, nullptr);

   if (grad)
   {
      double GradErr = gf->ComputeGradError(grad);
      DErrors.Append(GradErr);
      double err = sqrt(L2Errors[counter-1]*L2Errors[counter-1]+GradErr*GradErr);
      EnErrors.Append(err);
      // Compute the rate of convergence by:
      // rate = log (||u - u_h|| / ||u - u_{h/2}||)/log(2)
      double val = (dcounter) ? log(DErrors[dcounter-1]/GradErr)/log(2.0) : 0.0;
      double eval = (dcounter) ? log(EnErrors[dcounter-1]/err)/log(2.0) : 0.0;
      DRates.Append(val);
      EnRates.Append(eval);
      CoeffDNorm = GetNorm(gf,nullptr,grad);
      dcounter++;
      MFEM_VERIFY(counter == dcounter,
                  "Number of added solutions and derivatives do not match")
   }

   if (cont_type == mfem::FiniteElementCollection::DISCONTINUOUS && ell_coeff)
   {
      double DGErr = gf->ComputeDGFaceJumpError(scalar_u,ell_coeff,jump_scaling);
      DGFaceErrors.Append(DGErr);
      // Compute the rate of convergence by:
      // rate = log (||u - u_h|| / ||u - u_{h/2}||)/log(2)
      double val=(fcounter) ? log(DGFaceErrors[fcounter-1]/DGErr)/log(2.0):0.;
      DGFaceRates.Append(val);
      fcounter++;
      MFEM_VERIFY(fcounter == counter, "Number of added solutions mismatch");
   }
}

void ConvergenceStudy::AddGf(GridFunction *gf, VectorCoefficient *vector_u,
                             VectorCoefficient *curl, Coefficient *div)
{
   cont_type = gf->FESpace()->FEColl()->GetContType();

   AddL2Error(gf,nullptr,vector_u);
   double DErr = 0.0;
   bool derivative = false;
   if (curl)
   {
      DErr = gf->ComputeCurlError(curl);
      CoeffDNorm = GetNorm(gf,nullptr,curl);
      derivative = true;
   }
   else if (div)
   {
      DErr = gf->ComputeDivError(div);
      // update coefficient norm
      CoeffDNorm = GetNorm(gf,div,nullptr);
      derivative = true;
   }
   if (derivative)
   {
      double err = sqrt(L2Errors[counter-1]*L2Errors[counter-1] + DErr*DErr);
      DErrors.Append(DErr);
      EnErrors.Append(err);
      // Compute the rate of convergence by:
      // rate = log (||u - u_h|| / ||u - u_{h/2}||)/log(2)
      double val = (dcounter) ? log(DErrors[dcounter-1]/DErr)/log(2.0) : 0.0;
      double eval = (dcounter) ? log(EnErrors[dcounter-1]/err)/log(2.0) : 0.0;
      DRates.Append(val);
      EnRates.Append(eval);
      dcounter++;
      MFEM_VERIFY(counter == dcounter,
                  "Number of added solutions and derivatives do not match")
   }
}

void ConvergenceStudy::Print(bool relative, std::ostream &out)
{
   if (print_flag)
   {
      std::string title = (relative) ? "Relative " : "Absolute ";
      out << "\n";
      out << " -------------------------------------------" << "\n";
      out <<  std::setw(21) << title << "L2 Error " << "\n";
      out << " -------------------------------------------"
          << "\n";
      out << std::right<< std::setw(11)<< "DOFs "<< std::setw(13) << "Error ";
      out <<  std::setw(15) << "Rate " << "\n";
      out << " -------------------------------------------"
          << "\n";
      out << std::setprecision(4);
      double d = (relative) ? CoeffNorm : 1.0;
      for (int i =0; i<counter; i++)
      {
         out << std::right << std::setw(10)<< ndofs[i] << std::setw(16)
             << std::scientific << L2Errors[i]/d << std::setw(13)
             << std::fixed << L2Rates[i] << "\n";
      }
      out << "\n";
      if (dcounter == counter)
      {
         std::string dname;
         switch (cont_type)
         {
            case 0: dname = "Grad"; break;
            case 1: dname = "Curl"; break;
            case 2: dname = "Div";  break;
            case 3: dname = "DG Grad";  break;
            default: break;
         }
         out << " -------------------------------------------" << "\n";
         out <<  std::setw(21) << title << dname << " Error  " << "\n";
         out << " -------------------------------------------" << "\n";
         out << std::right<<std::setw(11)<< "DOFs "<< std::setw(13) << "Error";
         out << std::setw(15) << "Rate " << "\n";
         out << " -------------------------------------------"
             << "\n";
         out << std::setprecision(4);
         d = (relative) ? CoeffDNorm : 1.0;
         for (int i =0; i<dcounter; i++)
         {
            out << std::right << std::setw(10)<< ndofs[i] << std::setw(16)
                << std::scientific << DErrors[i]/d << std::setw(13)
                << std::fixed << DRates[i] << "\n";
         }
         out << "\n";
         switch (cont_type)
         {
            case 0: dname = "H1"; break;
            case 1: dname = "H(Curl)"; break;
            case 2: dname = "H(Div)";  break;
            case 3: dname = "DG H1";  break;
            default: break;
         }

         if (dcounter)
         {
            d = (relative) ?
                sqrt(CoeffNorm*CoeffNorm + CoeffDNorm*CoeffDNorm):1.0;

            out << " -------------------------------------------" << "\n";
            out << std::setw(21) << title << dname << " Error   " << "\n";
            out << " -------------------------------------------" << "\n";
            out << std::right<< std::setw(11)<< "DOFs "<< std::setw(13);
            out << "Error ";
            out << std::setw(15) << "Rate " << "\n";
            out << " -------------------------------------------"
                << "\n";
            out << std::setprecision(4);
            for (int i =0; i<dcounter; i++)
            {
               out << std::right << std::setw(10)<< ndofs[i] << std::setw(16)
                   << std::scientific << EnErrors[i]/d << std::setw(13)
                   << std::fixed << EnRates[i] << "\n";
            }
            out << "\n";
         }
      }
      if (cont_type == 3 && fcounter)
      {
         out << " -------------------------------------------" << "\n";
         out << "            DG Face Jump Error          " << "\n";
         out << " -------------------------------------------"
             << "\n";
         out << std::right<< std::setw(11)<< "DOFs "<< std::setw(13);
         out << "Error ";
         out << std::setw(15) << "Rate " << "\n";
         out << " -------------------------------------------"
             << "\n";
         out << std::setprecision(4);
         for (int i =0; i<fcounter; i++)
         {
            out << std::right << std::setw(10)<< ndofs[i] << std::setw(16)
                << std::scientific << DGFaceErrors[i] << std::setw(13)
                << std::fixed << DGFaceRates[i] << "\n";
         }
         out << "\n";
      }
   }
}

} // namespace mfem
