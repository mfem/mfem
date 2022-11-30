#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "fem/bilininteg_diffusion_pa.hpp"
#include "fem/kernel_dispatch.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   const char *mesh_file = "../data/star.mesh";
   const int order = 1;

   // KernelDispatchTable<DiffusionApplyPAKernels> apply;

   // Array<double> a1;
   // Vector v1;
   // apply.Run(2, 1, 1, 1, true, a1, a1, a1, a1, v1, v1, v1, 1, 1);

   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   H1_FECollection fec(order, dim);
   FiniteElementSpace fespace(&mesh, &fec);

   Vector B(fespace.GetTrueVSize()), X(fespace.GetTrueVSize());
   X.Randomize(1);

   BilinearForm a(&fespace);
   a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   a.AddDomainIntegrator(new DiffusionIntegrator);
   a.AddDomainIntegrator(new MassIntegrator);

   const Geometry::Type geom = mesh.GetElementGeometry(0);
   const IntegrationRule &ir = IntRules.Get(geom, 2*order + 2);
   (*a.GetDBFI())[0]->SetIntegrationRule(ir);
   (*a.GetDBFI())[1]->SetIntegrationRule(ir);

   a.Assemble();

   Array<int> ess_dofs;
   fespace.GetBoundaryTrueDofs(ess_dofs);

   Vector diag(fespace.GetTrueVSize());

   OperatorPtr A;
   a.FormSystemMatrix(ess_dofs, A);
   A->Mult(X, B);
   a.AssembleDiagonal(diag);

   DiffusionIntegrator::AddSpecialization<2,2,3>();
   MassIntegrator::AddSpecialization<2,2,3>();
   A->Mult(X, B);
   a.AssembleDiagonal(diag);

   return 0;
}
