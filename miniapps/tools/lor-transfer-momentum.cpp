#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int problem = 1; // problem type

// Exact functions to project
double rho_exact(const Vector &xvec);
void vel_exact(const Vector &xvec, Vector &vel);

// Helper functions
void ReportConservationErrors(GridFunction &rho_ho, GridFunction &rho_lor,
                              GridFunction &vel_ho, GridFunction &vel_lor);
void SaveAndIncrement(ParaViewDataCollection &dc, GridFunction &rho,
                      GridFunction &vel);

int main(int argc, char *argv[])
{
   // Parse command-line options.
   const char *mesh_file = "../../data/star.mesh";
   int ho_order = 3;
   int lref = ho_order+1;
   int lor_order = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem type (see the RHO_exact function).");
   args.AddOption(&ho_order, "-o", "--order",
                  "High-order polynomial degree.");
   args.AddOption(&lref, "-lref", "--lor-ref-level", "LOR refinement level.");
   args.AddOption(&lor_order, "-lo", "--lor-order",
                  "Low-order polynomial degree for density. "
                  "Velocity has degree lor-order + 1.");
   args.ParseCheck();

   // Read the mesh from the given mesh file
   Mesh mesh(mesh_file);
   int dim = mesh.Dimension();

   int vdim = dim;

   // Create the low-order refined mesh
   int basis_lor = BasisType::GaussLobatto; // BasisType::ClosedUniform;
   Mesh mesh_lor = Mesh::MakeRefined(mesh, lref, basis_lor);

   // Create a ParaView data collection that we can use for visualization
   ParaViewDataCollection dc("LORTransferMomentum");
   dc.SetPrefixPath("ParaView");
   dc.SetHighOrderOutput(true);

   // Create L2 and H1 low-order and high-order finite element collections
   L2_FECollection fec_l2_ho(ho_order, dim);
   L2_FECollection fec_l2_lor(lor_order, dim);

   H1_FECollection fec_h1_ho(ho_order, dim);
   H1_FECollection fec_h1_lor(lor_order+1, dim);

   // Create the scalar L2 spaces, used for density
   FiniteElementSpace fes_l2_ho(&mesh, &fec_l2_ho);
   FiniteElementSpace fes_l2_lor(&mesh_lor, &fec_l2_lor);

   // Create the vector H1 spaces, used for velocity
   FiniteElementSpace fes_h1_ho(&mesh, &fec_h1_ho, vdim);
   FiniteElementSpace fes_h1_lor(&mesh_lor, &fec_h1_lor, vdim);

   // Create the density and velocity grid functions
   GridFunction rho_ho(&fes_l2_ho), rho_lor(&fes_l2_lor);
   GridFunction vel_ho(&fes_h1_ho), vel_lor(&fes_h1_lor);

   FunctionCoefficient rho_coeff(rho_exact);
   VectorFunctionCoefficient vel_coeff(vdim, vel_exact);

   rho_ho.ProjectCoefficient(rho_coeff);
   vel_ho.ProjectCoefficient(vel_coeff);

   L2ProjectionGridTransfer proj_rho(fes_l2_ho, fes_l2_lor);
   proj_rho.ForwardOperator().Mult(rho_ho, rho_lor);

   // Use the density grid functions for the weighted L2 projection for velocity
   GridFunctionCoefficient rho_coeff_ho(&rho_ho);
   GridFunctionCoefficient rho_coeff_lor(&rho_lor);
   L2ProjectionGridTransfer proj_vel(fes_h1_ho, fes_h1_lor,
                                     rho_coeff_ho, rho_coeff_lor);
   proj_vel.ForwardOperator().Mult(vel_ho, vel_lor);

   printf("\n==== HO to LOR conservation ===========\n");
   ReportConservationErrors(rho_ho, rho_lor, vel_ho, vel_lor);
   SaveAndIncrement(dc, rho_ho, vel_ho);
   SaveAndIncrement(dc, rho_lor, vel_lor);

   GridFunction rho_ho_2(&fes_l2_ho), vel_ho_2(&fes_h1_ho);
   proj_rho.BackwardOperator().Mult(rho_lor, rho_ho_2);
   proj_vel.BackwardOperator().Mult(vel_lor, vel_ho_2);

   SaveAndIncrement(dc, rho_ho_2, vel_ho_2);

   VectorGridFunctionCoefficient vel_coeff_ho(&vel_ho);

   printf("\n==== Check PR = I ===================== \n\n");
   printf(" || PR(rho) - rho || = % 10.6e\n",
          rho_ho_2.ComputeL2Error(rho_coeff_ho));
   printf(" || PR(vel) - vel || = % 10.6e\n",
          vel_ho_2.ComputeL2Error(vel_coeff_ho));

   printf("\n==== LOR to HO conservation ========== \n");
   rho_lor.ProjectCoefficient(rho_coeff);
   vel_lor.ProjectCoefficient(vel_coeff);
   proj_rho.BackwardOperator().Mult(rho_lor, rho_ho);

   proj_vel.SetWeightedProjectionCoefficinets(rho_coeff_ho, rho_coeff_lor);
   proj_vel.BackwardOperator().Mult(vel_lor, vel_ho);

   ReportConservationErrors(rho_ho, rho_lor, vel_ho, vel_lor);
   SaveAndIncrement(dc, rho_lor, vel_lor);
   SaveAndIncrement(dc, rho_ho, vel_ho);

   return 0;
}

double TotalMass(GridFunction &rho)
{
   FiniteElementSpace &fes = *rho.FESpace();
   LinearForm lf(&fes);
   ConstantCoefficient one(1.0);
   lf.AddDomainIntegrator(new DomainLFIntegrator(one));
   lf.Assemble();
   return lf*rho;
}

Vector TotalMomentum(GridFunction &rho, GridFunction &vel)
{
   FiniteElementSpace &fes = *vel.FESpace();
   FiniteElementSpace fes_scalar(fes.GetMesh(), fes.FEColl());
   int vdim = fes.GetVDim();
   Vector total_momentum(vdim);

   GridFunctionCoefficient rho_coeff(&rho);

   LinearForm lf(&fes_scalar);
   lf.AddDomainIntegrator(new DomainLFIntegrator(rho_coeff));
   lf.Assemble();

   GridFunction vel_comp(&fes_scalar);
   for (int d=0; d<vdim; ++d)
   {
      Array<int> dofs, vdofs;
      for (int i=0; i<fes.GetNE(); ++i)
      {
         fes_scalar.GetElementDofs(i, dofs);
         fes.GetElementVDofs(i, vdofs);
         int ndofs = dofs.Size();
         for (int j=0; j<ndofs; ++j)
         {
            vel_comp[dofs[j]] = vel[vdofs[j + d*ndofs]];
         }
      }
      total_momentum[d] = lf*vel_comp;
   }

   return total_momentum;
}

void ReportConservationErrors(GridFunction &rho_ho, GridFunction &rho_lor,
                              GridFunction &vel_ho, GridFunction &vel_lor)
{
   double mass_ho = TotalMass(rho_ho);
   double mass_lor = TotalMass(rho_lor);
   printf("\n");
   printf("Total mass (HO):  % 10.6e\n", mass_ho);
   printf("Total mass (LOR): % 10.6e\n", mass_lor);
   printf("Total mass error: % 10.6e\n", std::fabs(mass_ho - mass_lor));

   int vdim = vel_ho.FESpace()->GetVDim();
   Vector momentum_ho, momentum_lor;
   momentum_ho = TotalMomentum(rho_ho, vel_ho);
   momentum_lor = TotalMomentum(rho_lor, vel_lor);
   printf("\n");
   printf("Dim    Total momentum (HO)    Total momentum (LOR)    Error\n");
   for (int d=0; d<vdim; ++d)
   {
      double m_ho = momentum_ho[d];
      double m_lor = momentum_lor[d];
      double m_err = std::fabs(m_ho - m_lor);
      printf(" %d     % 10.6e          % 10.6e          % 10.6e\n",
             d, m_ho, m_lor, m_err);
   }
}

void SaveAndIncrement(ParaViewDataCollection &dc, GridFunction &rho,
                      GridFunction &vel)
{
   dc.SetCycle(dc.GetCycle() + 1);
   dc.SetMesh(rho.FESpace()->GetMesh());
   dc.RegisterField("rho", &rho);
   dc.RegisterField("velocity", &vel);
   dc.SetLevelsOfDetail(std::max(1, rho.FESpace()->GetMaxElementOrder()));
   dc.Save();
   dc.SetTime(dc.GetTime() + 1);
}

double rho_exact(const Vector &xvec)
{
   switch (problem)
   {
      case 1: // smooth field
         return 1.0 + 0.25*cos(2*M_PI*xvec.Norml2());
      case 2: // cubic function
         return std::fabs(xvec(1)*xvec(1)*xvec(1) + 2*xvec(0)*xvec(1) + xvec(0));
      default:
         return 1.0;
   }
}

void vel_exact(const Vector &xvec, Vector &vel)
{
   int dim = vel.Size();
   const double w = M_PI/2;

   double x, y, z;
   x = xvec[0];
   if (dim >= 2) { y = xvec[1]; }
   if (dim >= 3) { z = xvec[2]; }

   if (problem == 1)
   {
      if (dim == 1) { vel(0) = 1.0; }
      if (dim == 2) { vel(0) = 1.0 + w*y; vel(1) = 2.0 - w*x; }
      if (dim == 3) { vel(0) = w*y; vel(1) = -w*x; vel(2) = 0.0; }
   }
   else if (problem == 2)
   {

      double d = max((x+1.)*(1.-x),0.) * max((y+1.)*(1.-y),0.);
      d = d*d;
      if (dim == 1) { vel(0) = 1.0; }
      if (dim == 2) { vel(0) = d*w*y; vel(1) = -d*w*x; }
      if (dim == 3) { vel(0) = d*w*y; vel(1) = -d*w*x; vel(2) = 0.0; }
   }
   else
   {
      vel = 1.0;
   }
}
