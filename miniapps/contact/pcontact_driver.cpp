//                               Parallel contact example
//
// Compile with: make pcontact_driver
// sample run
// mpirun -np 6 ./pcontact_driver -sr 2 -pr 2

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "ipsolver/ParIPsolver.hpp"

using namespace std;
using namespace mfem;


int main(int argc, char *argv[])
{
   Mpi::Init();
   int myid = Mpi::WorldRank();
   int num_procs = Mpi::WorldSize();
   Hypre::Init();
   // 1. Parse command-line options.
   const char *mesh_file1 = "meshes/block1.mesh";
   const char *mesh_file2 = "meshes/rotatedblock2.mesh";
   // const char *mesh_file2 = "meshes/block2.mesh";
   int order = 1;
   int sref = 0;
   int pref = 0;
   Array<int> attr;
   Array<int> m_attr;
   bool visualization = true;
   bool paraview = false;
   double linsolvertol = 1e-6;
   int relax_type = 8;
   double optimizer_tol = 1e-6;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file1, "-m1", "--mesh1",
                  "First mesh file to use.");
   args.AddOption(&mesh_file2, "-m2", "--mesh2",
                  "Second mesh file to use.");
   args.AddOption(&attr, "-at", "--attributes-surf",
                  "Attributes of boundary faces on contact surface for mesh 2.");
   args.AddOption(&sref, "-sr", "--serial-refinements",
                  "Number of uniform refinements.");
   args.AddOption(&pref, "-pr", "--parallel-refinements",
                  "Number of uniform refinements.");
   args.AddOption(&linsolvertol, "-stol", "--solver-tol",
                  "Linear Solver Tolerance.");
   args.AddOption(&optimizer_tol, "-otol", "--optimizer-tol",
                  "Interior Point Solver Tolerance.");                  
   args.AddOption(&relax_type, "-rt", "--relax-type",
                  "Selection of Smoother for AMG");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&paraview, "-paraview", "--paraview", "-no-paraview",
                  "--no-paraview",
                  "Enable or disable ParaView visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   ParElasticityProblem * prob1 = new ParElasticityProblem(MPI_COMM_WORLD,
                                                           mesh_file1,sref,pref,order);
   Vector lambda1(prob1->GetMesh()->attributes.Max()); lambda1 = 57.6923076923;
   Vector mu1(prob1->GetMesh()->attributes.Max()); mu1 = 38.4615384615;
   ParElasticityProblem * prob2 = new ParElasticityProblem(MPI_COMM_WORLD,
                                                           mesh_file2,sref,pref,order);
   Vector lambda2(prob2->GetMesh()->attributes.Max()); lambda2 = 57.6923076923;
   Vector mu2(prob2->GetMesh()->attributes.Max()); mu2 = 38.4615384615;

   prob1->SetLambda(lambda1); prob1->SetMu(mu1);
   prob2->SetLambda(lambda2); prob2->SetMu(mu2);

   ParContactProblem contact(prob1,prob2);
   QPOptParContactProblem qpopt(&contact);
   int numconstr = contact.GetGlobalNumConstraints();

   ParInteriorPointSolver optimizer(&qpopt);

   optimizer.SetTol(optimizer_tol);
   optimizer.SetMaxIter(50);

   int linsolver = 2;
   optimizer.SetLinearSolver(linsolver);
   optimizer.SetLinearSolveTol(linsolvertol);
   optimizer.SetLinearSolveRelaxType(relax_type);

   ParGridFunction x1 = prob1->GetDisplacementGridFunction();
   ParGridFunction x2 = prob2->GetDisplacementGridFunction();

   int ndofs1 = prob1->GetNumTDofs();
   int ndofs2 = prob2->GetNumTDofs();
   int gndofs1 = prob1->GetGlobalNumDofs();
   int gndofs2 = prob2->GetGlobalNumDofs();
   int ndofs = ndofs1 + ndofs2;

   Vector X1 = x1.GetTrueVector();
   Vector X2 = x2.GetTrueVector();

   Vector x0(ndofs); x0 = 0.0;
   x0.SetVector(X1,0);
   x0.SetVector(X2,X1.Size());

   Vector xf(ndofs); xf = 0.0;
   optimizer.Mult(x0, xf);

   double Einitial = contact.E(x0);
   double Efinal = contact.E(xf);
   Array<int> & CGiterations = optimizer.GetCGIterNumbers();
   if (Mpi::Root())
   {
      mfem::out << endl;
      mfem::out << " Initial Energy objective     = " << Einitial << endl;
      mfem::out << " Final Energy objective       = " << Efinal << endl;
      mfem::out << " Global number of dofs        = " << gndofs1 + gndofs2 << endl;
      mfem::out << " Global number of constraints = " << numconstr << endl;
      mfem::out << " CG iteration numbers         = " ;
      CGiterations.Print(mfem::out, CGiterations.Size());
   }

   MFEM_VERIFY(optimizer.GetConverged(),
               "Interior point solver did not converge.");


   if (visualization || paraview)
   {
      ParFiniteElementSpace * fes1 = prob1->GetFESpace();
      ParFiniteElementSpace * fes2 = prob2->GetFESpace();

      ParMesh * mesh1 = fes1->GetParMesh();
      ParMesh * mesh2 = fes2->GetParMesh();

      Vector X1_new(xf.GetData(),fes1->GetTrueVSize());
      Vector X2_new(&xf.GetData()[fes1->GetTrueVSize()],fes2->GetTrueVSize());

      ParGridFunction x1_gf(fes1);
      ParGridFunction x2_gf(fes2);

      x1_gf.SetFromTrueDofs(X1_new);
      x2_gf.SetFromTrueDofs(X2_new);

      mesh1->MoveNodes(x1_gf);
      mesh2->MoveNodes(x2_gf);

      if (paraview)
      {
         ParaViewDataCollection paraview_dc1("QPContactBody1", mesh1);
         paraview_dc1.SetPrefixPath("ParaView");
         paraview_dc1.SetLevelsOfDetail(1);
         paraview_dc1.SetDataFormat(VTKFormat::BINARY);
         paraview_dc1.SetHighOrderOutput(true);
         paraview_dc1.SetCycle(0);
         paraview_dc1.SetTime(0.0);
         paraview_dc1.RegisterField("Body1", &x1_gf);
         paraview_dc1.Save();

         ParaViewDataCollection paraview_dc2("QPContactBody2", mesh2);
         paraview_dc2.SetPrefixPath("ParaView");
         paraview_dc2.SetLevelsOfDetail(1);
         paraview_dc2.SetDataFormat(VTKFormat::BINARY);
         paraview_dc2.SetHighOrderOutput(true);
         paraview_dc2.SetCycle(0);
         paraview_dc2.SetTime(0.0);
         paraview_dc2.RegisterField("Body2", &x2_gf);
         paraview_dc2.Save();
      }

      if (visualization)
      {
         char vishost[] = "localhost";
         int visport = 19916;

         {
            socketstream sol_sock(vishost, visport);
            sol_sock.precision(8);
            sol_sock << "parallel " << 2*num_procs << " " << myid << "\n"
                     << "solution\n" << *mesh1 << x1_gf << flush;
         }
         {
            socketstream sol_sock(vishost, visport);
            sol_sock.precision(8);
            sol_sock << "parallel " << 2*num_procs << " " << myid+num_procs << "\n"
                     << "solution\n" << *mesh2 << x2_gf << flush;
         }
      }
   }

   delete prob1;
   delete prob2;

   return 0;
}
