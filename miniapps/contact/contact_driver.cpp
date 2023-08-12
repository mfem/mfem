//                                Contact example
//
// Compile with: make contact
//
// Sample runs:  ./contact -m1 block1.mesh -m2 block2.mesh -at "5 6 7 8"
// Sample runs:  ./contact -m1 block1_d.mesh -m2 block2_d.mesh -at "5 6 7 8"

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "ipsolver/IPsolver.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file1 = "meshes/block1.mesh";
   const char *mesh_file2 = "meshes/rotatedblock2.mesh";
   int order = 1;
   int ref = 0;
   Array<int> attr;
   Array<int> m_attr;
   int linSolver = 2;
   bool paraview = false;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file1, "-m1", "--mesh1",
                  "First mesh file to use.");
   args.AddOption(&mesh_file2, "-m2", "--mesh2",
                  "Second mesh file to use.");
   args.AddOption(&attr, "-at", "--attributes-surf",
                  "Attributes of boundary faces on contact surface for mesh 2.");
   args.AddOption(&ref, "-r", "--refinements",
                  "Number of uniform refinements.");     
   args.AddOption(&paraview, "-paraview", "--paraview", "-no-paraview",
                  "--no-paraview",
                  "Enable or disable ParaView visualization.");        
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

   ElasticityProblem prob1(mesh_file1,ref,order);
   ElasticityProblem prob2(mesh_file2,ref,order);

   ContactProblem contact(&prob1, &prob2);
   QPOptContactProblem qpopt(&contact);
   int numconstr = contact.GetNumConstraints();

   InteriorPointSolver optimizer(&qpopt);
   optimizer.SetTol(1e-6);
   optimizer.SetMaxIter(50);
   optimizer.SetLinearSolver(linSolver);
   optimizer.SetLinearSolveTol(1e-10);

   GridFunction x1 = prob1.GetDisplacementGridFunction();
   GridFunction x2 = prob2.GetDisplacementGridFunction();

   int ndofs1 = prob1.GetNumDofs();
   int ndofs2 = prob2.GetNumDofs();
   int ndofs = ndofs1 + ndofs2;

   Vector x0(ndofs); x0 = 0.0;
   x0.SetVector(x1,0);
   x0.SetVector(x2,x1.Size());

   Vector xf(ndofs); xf = 0.0;
   optimizer.Mult(x0, xf);
   Array<int> & CGiterations = optimizer.GetCGIterNumbers(); 

   double Einitial = contact.E(x0);
   double Efinal = contact.E(xf);
   
   mfem::out << endl;
   mfem::out << " Initial Energy objective     = " << Einitial << endl;
   mfem::out << " Final Energy objective       = " << Efinal << endl;
   mfem::out << " Global number of dofs        = " << ndofs1 + ndofs2 << endl;
   mfem::out << " Global number of constraints = " << numconstr << endl;
   mfem::out << " CG iteration numbers         = " ; CGiterations.Print(mfem::out, CGiterations.Size());

   MFEM_VERIFY(optimizer.GetConverged(), "Interior point solver did not converge.");

   if (visualization || paraview)
   {
      FiniteElementSpace * fes1 = prob1.GetFESpace();
      FiniteElementSpace * fes2 = prob2.GetFESpace();
   
      Mesh * mesh1 = fes1->GetMesh();
      Mesh * mesh2 = fes2->GetMesh();

      GridFunction x1_gf(fes1,xf.GetData());
      GridFunction x2_gf(fes2,&xf.GetData()[fes1->GetTrueVSize()]);

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
            sol_sock << "parallel " << 2 << " " << 0 << "\n"
                     << "solution\n" << *mesh1 << x1_gf << flush;
         }
         {
            socketstream sol_sock(vishost, visport);
            sol_sock.precision(8);
            sol_sock << "parallel " << 2 << " " << 1 << "\n"
                     << "solution\n" << *mesh2 << x2_gf << flush;                     
         }
      }
   }

   return 0;
}
