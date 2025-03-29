#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "J_field_vec_coeffs.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   const char *mesh_file = "mesh/2d_mesh.mesh";
   bool visualization = true;

   Mesh mesh(mesh_file, 1, 1);
   // mesh.UniformRefinement();
   int dim = mesh.Dimension();
   cout << "Mesh loaded" << endl;

   // make a L2 space with the mesh
   H1_FECollection fec(1, dim);
   FiniteElementSpace fespace(&mesh, &fec);

   GridFunction MMS_gf(&fespace);
   cout << MMS_gf.FESpace()->GetTrueVSize() << endl;
   MMS_gf = 0.0;

   LinearForm b(&fespace);
   WeightedHarmonicGridFunctionCoefficient MMS_coeff(POLYNOMIAL);
   b.AddDomainIntegrator(new DomainLFIntegrator(MMS_coeff));
   b.Assemble();

   // 2. make the bilinear form
   BilinearForm a(&fespace);
   ConstantCoefficient one(1.0);
   a.AddDomainIntegrator(new MassIntegrator(one));
   a.Assemble();
   a.Finalize();

   // 3. solve the system
   CGSolver M_solver;
   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-24);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(1e5);
   M_solver.SetPrintLevel(1);
   M_solver.SetOperator(a.SpMat());

   Vector X(MMS_gf.Size());
   X = 0.0;
   M_solver.Mult(b, X);

   MMS_gf.SetFromTrueDofs(X);

   if (visualization)
   {
      char vishost[] = "localhost";
      int visport = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n"
               << mesh << MMS_gf << flush;
   }

   // paraview
   {
      ParaViewDataCollection paraview_dc("MMS_gf", &mesh);
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(1);
      paraview_dc.SetCycle(0);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetTime(0.0); // set the time
      paraview_dc.RegisterField("MMS_gf", &MMS_gf);
      paraview_dc.Save();
   }

   ofstream sol_ofs("output/MMS_gf.gf");
   sol_ofs.precision(8);
   MMS_gf.Save(sol_ofs);

   return 0;
}
