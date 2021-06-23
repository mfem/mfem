
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
   const char *mesh_file = "../data/star.mesh";
   int order = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.ParseCheck();

   // 2. Read the mesh from the given mesh file, and refine once uniformly.
   Mesh mesh(mesh_file);
   // mesh.Print(cout);
   // mesh.UniformRefinement();

   // mesh.EnsureNodes();

   // Array<int> elems0({0,4,8,12,16});
   // Array<int> elems0({0,4,8,12,16});
   // Array<int> elems0({6,7,8});
   Array<int> elems0({0,1,2,3,4});
   // Array<int> elems0({7,6,17,20,21,22});
   // Array<int> elems0({0,1,2,3,4});
   // Array<int> elems0({104,103,86,109});
   // elems0.Print(cout, elems0.Size());
   Subdomain subdomain0(mesh);
   Mesh * submesh0 = subdomain0.GetSubMesh(elems0);

   // Array<int> bdrelems0({8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23});
   cout << "number of boundary elements = " << mesh.GetNBE() << endl;
   Array<int> bdrelems0({0,1});
   
   // Array<int> bdrelems0(1); bdrelems0 = 1;
   Mesh * bdrmesh0 = subdomain0.GetBdrSurfaceMesh(bdrelems0);

   // // Array<int> faceelems0({153,151}); 
   // Array<int> faceelems0({3,4}); 
   // Mesh * surfmesh0 = subdomain0.GetSurfaceMesh(faceelems0);


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
      mesh_sock << "solution\n" << mesh << gf 
      << "valuerange -1.0 1.0 \n" << flush;
   }

   subdomain0.SetFESpace(fespace);

   SparseMatrix * P = subdomain0.GetProlonationMatrix();
   FiniteElementSpace * elem_fes = 
               subdomain0.GetSubFESpace(Subdomain::entity_type::volume);
   GridFunction gf_e(elem_fes);
   if (!P)
   {
      cout << "P is null" << endl;
   }
   cout << "P->size = " << P->Height() << " x " << P->Width() << endl;
   cout << "gf.Size = " << gf.Size() << endl;
   cout << "gf_e.Size = " << gf_e.Size() << endl;
   P->MultTranspose(gf,gf_e);


   SparseMatrix * Pb = subdomain0.GetBdrProlonationMatrix();
   FiniteElementSpace * bdr_elem_fes = 
               subdomain0.GetSubFESpace(Subdomain::entity_type::bdr);
   GridFunction gf_b(bdr_elem_fes);
   Pb->MultTranspose(gf,gf_b);
   // SparseMatrix * Pf = subdomain0.GetFaceProlonationMatrix();

   // FiniteElementSpace * face_elem_fes = 
   //          subdomain0.GetSubFESpace(Subdomain::entity_type::face);     
   // GridFunction gf_f(face_elem_fes);
   // Pf->MultTranspose(gf,gf_f);

   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      if (submesh0)
      {
         socketstream mesh0_sock(vishost, visport);
         mesh0_sock.precision(8);
         // mesh0_sock << "mesh\n" << *submesh0 << "keys n \n" << flush;
         mesh0_sock << "solution\n" << *submesh0 << gf_e 
         << "valuerange -1.0 1.0 \n" << flush;
      }
      if (bdrmesh0)
      {
         socketstream mesh1_sock(vishost, visport);
         mesh1_sock.precision(8);
         // mesh1_sock << "mesh\n" << *bdrmesh0 << "keys n \n" << flush;
         mesh1_sock << "solution\n" << *bdrmesh0 << gf_b 
         << "valuerange -1.0 1.0 \n" << flush;
      }
      // if (surfmesh0)
      // {
      //    socketstream mesh2_sock(vishost, visport);
      //    mesh2_sock.precision(8);
      //    // mesh2_sock << "mesh\n" << *surfmesh0 << "keys n \n" << flush;
      //    mesh2_sock << "solution\n" << *surfmesh0 << gf_f << 
         // << "keys \n" 
         // << "valuerange -1.0 1.0 \n" << flush;
      // }
   }

   // Array<int> bdr_elems;
   // for (int i =0; i<mesh.GetNBE(); i++)
   // {
   //    int attr = mesh.GetBdrAttribute(i);
   //    if (attr == 4)
   //    {
   //       bdr_elems.Append(i);
   //    }
   // }
   // // bdr_elems.SetSize(1);
   // // bdr_elems[0] = 1;
   // // bdr_elems.Print();

   // H1_FECollection fec(order, mesh.Dimension());
   // FiniteElementSpace fespace(&mesh, &fec);
   // FunctionCoefficient coeff(sin_func);
   // GridFunction gf(&fespace);
   // gf.ProjectCoefficient(coeff);
   // {
   //    char vishost[] = "localhost";
   //    int  visport   = 19916;
   //    socketstream mesh_sock(vishost, visport);
   //    mesh_sock.precision(8);
   //    // mesh_sock << "mesh\n" << mesh << "keys n \n" << flush;
   //    mesh_sock << "solution\n" << mesh << gf << flush; 
   //    // << "valuerange -5000.0 5000.0 \n" << flush;
   //    // << "valuerange -1.0 1.0 \n" << flush;
   // }

   // Subdomain subdomain1(mesh);
   // subdomain1.SetFESpace(fespace);
   // Mesh * bdrmesh0 = subdomain1.GetBdrSurfaceMesh(bdr_elems);
   // SparseMatrix * Pb = subdomain1.GetBdrProlonationMatrix();
   // FiniteElementSpace * bdr_elem_fes = 
   //             subdomain1.GetSubFESpace(Subdomain::entity_type::bdr);
   // GridFunction gf_b(bdr_elem_fes);
   // Pb->MultTranspose(gf,gf_b);



   // // gf.Print();
   // // gf_b.Print();
   // // bdrmesh0->Print(cout);

   // if (bdrmesh0)
   // {
   //    char vishost[] = "localhost";
   //    int  visport   = 19916;
   //    socketstream mesh1_sock(vishost, visport);
   //    mesh1_sock.precision(8);
   //    // mesh1_sock << "mesh\n" << *bdrmesh0 << "keys n \n" << flush;
   //    mesh1_sock << "solution\n" << *bdrmesh0 << gf_b << flush;
   //    // << "valuerange -5000.0 5000.0 \n" << flush;
   // }

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
   double dotp = x.Sum();
   // return (sin(10.0*M_PI*dotp));
   // return sin(2.*M_PI*x[0]);
   // return 1.-x[1]*x[1]/4.0;
   return (0.5-x[1])*(0.5-x[1]);
   // double r = sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
   // return r;
}
