//
// Compile with: make findpsi
//
// Sample runs:
//  ./findpsi 
//  ./findpsi -m ./meshes/RegPoloidalQuadMeshNonAligned_true.mesh -g ./interpolated.gf

#include "mfem.hpp"

using namespace mfem;
using namespace std;

double  alpha = 1.44525792e-01,
        r_x = 4.91157885e+00,
        z_x = -3.61688204e+00,
        psi_x = 1.28863812e+00,
        f_x = -32.86000000;

int main (int argc, char *argv[])
{
   // Set the method's default parameters.
   const char *mesh_file = "./solution/build-field.mesh";
   const char *Bperp_file = "./solution/Bperp.gf";
   const char *Btor_file = "./solution/Btor.gf";
   int order             = 2;
   bool visualization    = false;

   // Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Vector finite element order (polynomial degree).");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // Initialize and refine the starting mesh.
   Mesh mesh(mesh_file, 1, 1);
   const int dim = mesh.Dimension();
   int dim3D = mesh.Dimension();
   int sdim3D = mesh.SpaceDimension();
   if (dim3D!=3 || sdim3D!=3){
      cout << "wrong dimensions in mesh!"<<endl;
      return 1;
   }

   // Read the solution psi
   ifstream Bperp_stream(Bperp_file);
   ifstream Btor_stream(Btor_file);
   GridFunction Bperp(&mesh, Bperp_stream);
   GridFunction Btotal(&mesh, Btor_stream);

   FiniteElementSpace *Bfespace=Bperp.FESpace();
   int size = Bfespace->GetVSize();     //expect VSize and TrueVSize are the same
   cout << "Number of elements: "<<mesh.GetNE()<<endl;
   cout << "Number of finite element unknowns: " << size << endl;

   Btotal+=Bperp;

   FiniteElementCollection *Jfec=new RT_FECollection(order-1, dim3D);
   FiniteElementSpace *Jfespace=new FiniteElementSpace(&mesh, Jfec);
   int sizeJ = Jfespace->GetVSize();     //expect VSize and TrueVSize are the same
   cout << "Number of finite element unknowns in J: " << sizeJ << endl;

   GridFunction J(Jfespace);

   ConstantCoefficient one(1.0);
   BilinearForm mass(Jfespace);
   mass.AddDomainIntegrator(new VectorFEMassIntegrator(one));
   mass.Assemble();
   mass.Finalize();

   CGSolver M_solver;
   DSmoother M_prec(mass.SpMat()); 
   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-8);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(100);
   M_solver.SetPrintLevel(1);
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(mass.SpMat());

   MixedBilinearForm a_mixed(Bfespace, Jfespace);
   a_mixed.AddDomainIntegrator(new MixedVectorCurlIntegrator(one));
   a_mixed.Assemble();

   Vector x(sizeJ), rhs(sizeJ);
   a_mixed.Mult(Btotal, rhs);
   x = 0.0;
   M_solver.Mult(rhs, x);

   J.SetFromTrueDofs(x);

   if (visualization){
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << mesh << J << flush;
   }

   if (true){
      ofstream sol_ofs("Btotal.gf");
      sol_ofs.precision(8);
      Btotal.Save(sol_ofs);

      ofstream sol_ofs1("J.gf");
      sol_ofs1.precision(8);
      J.Save(sol_ofs1);

      ParaViewDataCollection paraview_dc("computeJ", &mesh);
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(order);
      paraview_dc.SetCycle(0);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetTime(0.0); // set the time
      paraview_dc.RegisterField("Btotal",&Btotal);
      paraview_dc.RegisterField("J",&J);
      paraview_dc.Save();
   }

   return 0;
}
