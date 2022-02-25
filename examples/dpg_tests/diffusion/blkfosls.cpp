//                 MFEM Fosls 1
//
// Compile with: make blkfosls
//
//     - Δ u = f, in Ω
//         u = 0, on ∂Ω

// First Order System

//   ∇ u - σ = 0, in Ω
// - ∇⋅σ     = f, in Ω
//        u  = 0, in ∂Ω

// FOSLS:
//       minimize  1/2(||∇u - σ||^2 + ||∇ ⋅ σ - f||^2)


// -------------------------------------------------
// |   |    u    |         σ           |    RHS    |
// -------------------------------------------------
// | v | (∇u,∇v) |      -(σ,∇v)        |     0     |
// |   |         |                     |           |
// | τ | -(∇u,τ) | (∇⋅σ, ∇⋅τ) + (σ,τ)  | -(f,∇⋅τ ) |

// where (u,τ) ∈ H^1(Ω) × H(div,Ω)
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../../../data/inline-quad.mesh";
   int order = 1;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
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

   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   // 5. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order. If order < 1, we
   //    instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec0 = new H1_FECollection(order, dim);
   FiniteElementCollection *fec1 = new RT_FECollection(order-1, dim);
   FiniteElementSpace fespace0(&mesh, fec0);
   FiniteElementSpace fespace1(&mesh, fec1);

   Array<FiniteElementSpace *> fespaces(2);
   fespaces[0] = &fespace0;
   fespaces[1] = &fespace1;

   Array<int> ess_bdr;
   Array<int> ess_tdof_list;
   if (mesh.bdr_attributes.Size())
   {
      ess_bdr.SetSize(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      fespaces[0]->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   BlockBilinearForm a(fespaces);
   a.SetDiagonalPolicy(mfem::Operator::DIAG_KEEP);

   cout << "H1 fespace = " << fespace0.GetVSize() << endl;
   cout << "RT fespace = " << fespace1.GetVSize() << endl;

   FiniteElementCollection *fec2 = new RT_Trace_FECollection(order-1, dim);
   FiniteElementSpace RT_trace_fes(&mesh, fec2);
   cout << "RT trace = " << RT_trace_fes.GetVSize() << endl;

   // for (int i = 0; i<mesh.GetNE(); i++)
   // {
   //    // const FiniteElement * fe = fespace1.GetFE(i);
   //    // fespace1.GetTraceElement()
   //    Array<int> faces, ori;
   //    mesh.GetElementEdges(i, faces, ori);
   //    for (int f = 0; f<faces.Size(); f++)
   //    {
   //       const FiniteElement * fe_trace = RT_trace_fes.GetFaceElement(faces[f]);
   //       cout << fe_trace->GetDof() << endl;
   //       Array<int> face_dofs;
   //       RT_trace_fes.GetFaceDofs(faces[f],face_dofs);
   //       cout << "face dofs = " << endl;
   //       face_dofs.Print();
   //    }

   //    // cout << fe->GetGeomType() << endl;
   //    Array<int> vdofs;
   //    RT_trace_fes.GetElementVDofs(i, vdofs);
   //    cout << "trace dofs = " << endl;
   //    vdofs.Print();
   //    fespace1.GetElementVDofs(i, vdofs);
   //    cout << "elem dofs = " << endl;
   //    vdofs.Print();
   //    cin.get();
   // }


   ConstantCoefficient one(1.0);
   ConstantCoefficient negone(-1.0);

   Array2D<BilinearFormIntegrator * > blfi(2,2);
   blfi(0,0) = new DiffusionIntegrator(one);
   blfi(0,1) = new MixedVectorWeakDivergenceIntegrator(one);
   blfi(1,0) = new MixedVectorGradientIntegrator(negone);

   BilinearFormIntegrator * divdiv = new DivDivIntegrator(one);
   BilinearFormIntegrator * mass = new VectorFEMassIntegrator(one);
   SumIntegrator * suminteg = new SumIntegrator();
   suminteg->AddIntegrator(divdiv);
   suminteg->AddIntegrator(mass);
   blfi(1,1) = suminteg;

   TestBlockBilinearFormIntegrator * integ = new TestBlockBilinearFormIntegrator();
   integ->SetIntegrators(blfi);
   a.AddDomainIntegrator(integ);
   a.Assemble();


   BlockLinearForm b(fespaces);

   TestBlockLinearFormIntegrator * lininteg = new TestBlockLinearFormIntegrator();
   Array<LinearFormIntegrator * > lfi(2);
   lfi[0] = nullptr;
   lfi[1] = new VectorFEDomainLFDivIntegrator(negone);
   lininteg->SetIntegrators(lfi);
   b.AddDomainIntegrator(lininteg);
   b.Assemble();


   // need to implement blkgridfunction later but for now Vector would do
   int size = 0;
   for (int i = 0; i<fespaces.Size(); i++)
   {
      size += fespaces[i]->GetVSize();
   }

   Vector x(size);
   x = 0.0;

   OperatorPtr A;
   Vector X,B;
   a.FormLinearSystem(ess_tdof_list,x,b,A,X,B);

   GSSmoother M((SparseMatrix&)(*A));
   CGSolver cg;
   cg.SetRelTol(1e-6);
   cg.SetMaxIter(200);
   cg.SetPrintLevel(1);
   cg.SetPreconditioner(M);
   cg.SetOperator(*A);
   cg.Mult(B, X);

   a.RecoverFEMSolution(X,b,x);

   GridFunction u_gf, sigma_gf;
   double *data = x.GetData();
   u_gf.MakeRef(fespaces[0],&data[0]);
   sigma_gf.MakeRef(fespaces[1],&data[fespaces[0]->GetVSize()]);

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream solu_sock(vishost, visport);
      solu_sock.precision(8);
      solu_sock << "solution\n" << mesh << u_gf <<
                "window_title 'Numerical u' "
                << flush;
      socketstream sols_sock(vishost, visport);
      sols_sock.precision(8);
      sols_sock << "solution\n" << mesh << sigma_gf <<
                "window_title 'Numerical sigma' "
                << flush;
   }

   delete fec0;

   return 0;
}
