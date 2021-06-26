
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
   // mesh.EnsureNodes();
   // mesh.UniformRefinement();
   // Array<int> elems0({0,1,2,3});
   int nel = mesh.GetNE();
   // int nel = 5;
   Array<int> elems0(nel/2);
   for (int i = 0; i<nel/2; i++)
   {
      elems0[i] = i;
   }
   elems0.Print();
   // elems0.Append(24);
   // elems0.Append(23);
   // elems0.Append(26);
   Subdomain subdomain0(mesh);

   Mesh * submesh = subdomain0.GetSubMesh(elems0);


   // cout << "number of boundary elements = " << mesh.GetNBE() << endl;
   Array<int> faces(mesh.GetNBE()/2);
   for (int i = 0; i<mesh.GetNBE()/2; i++)
   {
      faces[i] = mesh.GetBdrFace(i);
   }
   
   Mesh * surfmesh = subdomain0.GetSurfaceMesh(faces);

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
         // << flush;
   }

   subdomain0.SetFESpace(fespace);

   SparseMatrix * P = subdomain0.GetProlonationMatrix();
   FiniteElementSpace * elem_fes = 
               subdomain0.GetSubFESpace(Subdomain::entity_type::volume);
   GridFunction gf_e(elem_fes);
   cout << "Size P = " << P->Height() << " x " << P->Width() << endl;
   cout << "gf_e.Size = " << gf_e.Size() << endl;
   cout << "gf.Size = " << gf.Size() << endl;
   P->MultTranspose(gf,gf_e);

   SparseMatrix * Pb = subdomain0.GetSurfaceProlonationMatrix();
   FiniteElementSpace * bdr_elem_fes = 
               subdomain0.GetSubFESpace(Subdomain::entity_type::surface);
   GridFunction gf_b(bdr_elem_fes);
   Pb->MultTranspose(gf,gf_b);

   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      if (submesh)
      {
         socketstream mesh0_sock(vishost, visport);
         mesh0_sock.precision(8);
         // mesh0_sock << "mesh\n" << *submesh << "keys n \n" << flush;
         mesh0_sock << "solution\n" << *submesh << gf_e << "keys nmR \n"
         << "valuerange 0 1.0 \n" << flush;
         // << flush;
      }
      if (surfmesh && mesh.Dimension() == 3)
      {
         socketstream mesh1_sock(vishost, visport);
         mesh1_sock.precision(8);
         // mesh1_sock << "mesh\n" << *bdrmesh0 << "keys n \n" << flush;
         mesh1_sock << "solution\n" << *surfmesh << gf_b
         << "valuerange 0 1.0 \n" << flush;
         // << flush;
      }
   }

   // ParaViewDataCollection paraview_dc("mesh_partition", surfmesh);
   // paraview_dc.SetPrefixPath("ParaView");
   // const FiniteElementSpace * fes_ = surfmesh->GetNodalFESpace();
   // int ord = (fes_) ? fes_->GetOrder(0) : order;
   // paraview_dc.SetLevelsOfDetail(ord);
   // paraview_dc.SetCycle(0);
   // paraview_dc.SetDataFormat(VTKFormat::BINARY);
   // paraview_dc.SetHighOrderOutput(true);
   // paraview_dc.SetTime(0.0); // set the time
   // paraview_dc.RegisterField("solution",&gf_b);
   // paraview_dc.Save();



   // // ---------------------------------------------------------
   // FiniteElementCollection *fec1 = new H1_FECollection(order, submesh->Dimension());
   // FiniteElementSpace fespace1(submesh, fec1);
   // Array<int> ess_tdof_list;
   // if (submesh->bdr_attributes.Size())
   // {
   //    Array<int> ess_bdr(mesh.bdr_attributes.Max());
   //    ess_bdr = 1;
   //    fespace1.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   // }

   // LinearForm b(&fespace1);
   // ConstantCoefficient one(1.0);
   // b.AddDomainIntegrator(new DomainLFIntegrator(one));
   // b.Assemble();

   // GridFunction x(&fespace1);
   // x = 0.0;

   // // 9. Set up the bilinear form a(.,.) on the finite element space
   // //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   // //    domain integrator.
   // BilinearForm a(&fespace1);
   // a.AddDomainIntegrator(new DiffusionIntegrator(one));
   // a.Assemble();
   // OperatorPtr A;
   // Vector B, X;
   // a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
   // cout << "Size of linear system: " << A->Height() << endl;
   // // Use a simple symmetric Gauss-Seidel preconditioner with PCG.
   // GSSmoother M((SparseMatrix&)(*A));
   // PCG(*A, M, B, X, 1, 200, 1e-12, 0.0);
   // // 12. Recover the solution as a finite element grid function.
   // a.RecoverFEMSolution(X, b, x);
   // {
   //    char vishost[] = "localhost";
   //    int  visport   = 19916;
   //    socketstream sol_sock2(vishost, visport);
   //    sol_sock2.precision(8);
   //    sol_sock2 << "solution\n" << *submesh << x << flush;
   // }

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
   // return (0.5-x[1])*(0.5-x[1]);
   return x[1];
   // double r = sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
   // return r;
}
