
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "mesh_partition.hpp"

using namespace std;
using namespace mfem;

double sin_func(const Vector & x);

int main(int argc, char *argv[])
{
   // 1. Parse command line options
   const char *mesh_file = "../../data/periodic-annulus-sector.msh";
   int order = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.ParseCheck();

   // 2. Read the mesh from the given mesh file, and refine once uniformly.
   Mesh mesh(mesh_file);
   mesh.EnsureNodes();
   // mesh.RemoveInternalBoundaries();
   // mesh.SetCurvature(3,true);
   // mesh.GetNodes()->Print();
   // Array<int> elems0({1,3,18,13,23});
   Array<int> elems0({0,1,2,3});
   Subdomain subdomain0(mesh);
   // Mesh * submesh0 = subdomain0.GetSubMesh(elems0);

   cout << "number of boundary elements = " << mesh.GetNBE() << endl;
   // Array<int> bdrelems0({3,4,5,6,7});
   // Array<int> bdrelems0({0,1,2,3,4});
   Array<int> bdrelems0(mesh.GetNBE());
   for (int i = 0; i<mesh.GetNBE(); i++)
   {
      bdrelems0[i] = i;
   }
   // Array<int> bdrelems0(1); bdrelems0 = 0;
   cout << "number of vertices = " << mesh.GetNV() << endl;
   for (int i = 0; i< bdrelems0.Size(); i++)
   {
      Array<int> vertices;
      mesh.GetBdrElementVertices(bdrelems0[i], vertices);
      cout << "Boundary element: " << i <<", vertices = " ; vertices.Print() ;
   }
   
   Mesh * bdrmesh0 = subdomain0.GetBdrSurfaceMesh(bdrelems0);

   H1_FECollection fec(order, mesh.Dimension());
   FiniteElementSpace fespace(&mesh, &fec);
   FunctionCoefficient coeff(sin_func);
   GridFunction gf(&fespace);
   gf.ProjectCoefficient(coeff);
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream mesh_sock(vishost, visport);
      mesh_sock.precision(8);
      // mesh_sock << "mesh\n" << mesh << "keys n \n" << flush;
      mesh_sock << "solution\n" << mesh << gf <<  "keys jnmR \n" 
      << "valuerange 0 1.0 \n" << flush;
   }

   subdomain0.SetFESpace(fespace);

   // SparseMatrix * P = subdomain0.GetProlonationMatrix();
   // FiniteElementSpace * elem_fes = 
   //             subdomain0.GetSubFESpace(Subdomain::entity_type::volume);
   // GridFunction gf_e(elem_fes);
   // P->MultTranspose(gf,gf_e);

   SparseMatrix * Pb = subdomain0.GetBdrProlonationMatrix();
   FiniteElementSpace * bdr_elem_fes = 
               subdomain0.GetSubFESpace(Subdomain::entity_type::bdr);
   GridFunction gf_b(bdr_elem_fes);
   Pb->MultTranspose(gf,gf_b);

   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      // if (submesh0)
      // {
      //    socketstream mesh0_sock(vishost, visport);
      //    mesh0_sock.precision(8);
      //    // mesh0_sock << "mesh\n" << *submesh0 << "keys n \n" << flush;
      //    mesh0_sock << "solution\n" << *submesh0 << gf_e << "keys nmR \n"
      //    << "valuerange 0 1.0 \n" << flush;
      // }
      if (bdrmesh0 && mesh.Dimension() == 3)
      {
         socketstream mesh1_sock(vishost, visport);
         mesh1_sock.precision(8);
         // mesh1_sock << "mesh\n" << *bdrmesh0 << "keys n \n" << flush;
         mesh1_sock << "solution\n" << *bdrmesh0 << gf_b
         << "valuerange 0 1.0 \n" << flush;
      }
   }

   ParaViewDataCollection paraview_dc("mesh_partition", bdrmesh0);
   paraview_dc.SetPrefixPath("ParaView");
   const FiniteElementSpace * fes_ = bdrmesh0->GetNodalFESpace();
   int ord = (fes_) ? fes_->GetOrder(0) : order;
   paraview_dc.SetLevelsOfDetail(5);
   paraview_dc.SetCycle(0);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);
   paraview_dc.SetTime(0.0); // set the time
   paraview_dc.RegisterField("solution",&gf_b);
   paraview_dc.Save();

   return 0;
}

double sin_func(const Vector & x)
{
   Vector c(x.Size());
   c.Randomize();
   // double dotp = c*x;
   // return (sin(10.0*M_PI*dotp));
   // return sin(2.*M_PI*x[0]);
   // return 1.-x[1]*x[1]/4.0;
   return (0.5-x[1])*(0.5-x[1]);
   // double r = sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
   // return r;
}
