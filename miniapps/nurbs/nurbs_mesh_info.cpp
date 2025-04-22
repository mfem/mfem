
//
//                          MFEM Print info of NURBS mesh
//
//
// Compile with: make nurbs_mesh_info
//
// Sample runs:
//    nurbs_mesh_info -m ../../data/cube-nurbs.mesh -o 0 -r 2

// Description:  This code prints detailed mesh information such as:
//                - Print separate patch info
//                - 1D shape functions associated knot vectors
//                - Give Greville, Botella and Demko points of the knot vectors
//

#include <iostream>
#include "mfem.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // Read parameters from command line
   const char *mesh_file = "../../data/square-nurbs.mesh";
   const char *ref_file  = "";
   int ref_levels = -1;
   int order = 1;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly, -1 for auto.");
   args.AddOption(&ref_file, "-rf", "--ref-file",
                  "File with refinement data");
   args.AddOption(&order, "-o", "--order",
                  "NURBS order (polynomial degree) or -1 for");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization."); // Dummy arg for `make test`
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }

   // Read the mesh
   Mesh mesh(mesh_file, 1, 1);
   NURBSExtension *ext = mesh.NURBSext;

   if (!ext)
   {
      mfem_error("Mesh is not a NURBS mesh.");
   }

   // Refine the mesh as specified
   mesh.DegreeElevate(16, order);

   if (mesh.NURBSext && (strlen(ref_file) != 0))
   {
      mesh.RefineNURBSFromFile(ref_file);
   }

   for (int l = 0; l < ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   // Print mesh info
   mesh.PrintInfo();

   // Print patch info
   mfem::out<<"=======================================;"<<endl;
   mfem::out<<" Patch info"<<endl;
   mfem::out<<"=======================================;"<<endl;
   for (int p = 0; p < ext->GetNP(); p++)
   {
      Array<const KnotVector *> kv;
      ext->GetPatchKnotVectors(p, kv);

      mfem::out<<p<<": Order = "<<kv[0]->GetOrder();
      for (int k = 1; k < kv.Size(); k++)
      {
         mfem::out<<"x"<<kv[k]->GetOrder();
      }
      mfem::out<<" : DOFs = "<<kv[0]->GetNCP();
      for (int k = 1; k < kv.Size(); k++)
      {
         mfem::out<<"x"<<kv[k]->GetNCP();
      }
      mfem::out<<endl;
   }

   // Print knotvector info
   for (int k = 0; k < ext->GetNKV() ; k++)
   {
      mfem::out<<"=======================================;"<<endl;
      mfem::out<<" KnotVector "<<k<<endl;
      mfem::out<<"=======================================;"<<endl;
      const KnotVector &kv = *ext->GetKnotVector(k);
      mfem::out<<"Knotvector : "; kv.Print(mfem::out);

      std::string gnuplot = "plot 0";
      Vector a(kv.GetNCP());
      for (int i = 0; i < kv.GetNCP(); i++)
      {
         a = 0.0;
         a[i] = 1.0;
         std::string filename = "k" + std::to_string(k) +"_n"  + std::to_string(
                                   i) + ".dat";
         mfem::out<<"Write shape function to: "<<filename<<"\n";
         std::ofstream ofs(filename);
         kv.PrintFunction(ofs, a, 201);
         ofs.close();
         gnuplot += ", '" + filename+"' u 1:2 w l";
      }
      mfem::out<<gnuplot<<endl;

      // Greville
      Vector greville(kv.GetNCP());
      for (int i = 0; i < kv.GetNCP(); i++)
      {
         greville[i] = kv.GetGreville(i);
      }
      mfem::out<<"Greville points : "; greville.Print(mfem::out, 32);

      // Botella
      Vector botella(kv.GetNCP());
      for (int i = 0; i < kv.GetNCP(); i++)
      {
         botella[i] = kv.GetBotella(i);
      }
      mfem::out<<"Botella  points : "; botella.Print(mfem::out, 32);

      // Demko
      Vector demko(kv.GetNCP());
      for (int i = 0; i < kv.GetNCP(); i++)
      {
         demko[i] = kv.GetDemko(i);
      }
      mfem::out<<"Demko    points : "; demko.Print(mfem::out, 32);

      // Chebyshev spline
      Vector x(kv.GetNCP());
      for ( int i = 0; i <kv.GetNCP(); i++)
      {
         x[i] = std::pow(-1.0, i);
      }
      kv.GetInterpolant(x, demko, a);
      mfem::out<<"Chebyshev spline coeff : "; a.Print(mfem::out, 32);

      std::string filename = "k" + std::to_string(k) +"_cheby.dat";
      mfem::out<<"Write Chebyshev spline to: "<<filename<<"\n";
      std::ofstream ofs(filename);
      kv.PrintFunction(ofs, a, 201);
      ofs.close();
   }

}
