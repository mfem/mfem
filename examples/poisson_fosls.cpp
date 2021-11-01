//                                MFEM Example 1
//
// Compile with: make poisson_fosls
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
   const char *mesh_file = "../data/inline-quad.mesh";
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

   FiniteElementCollection *H1fec = new H1_FECollection(order,dim); 
   FiniteElementSpace *H1fes = new FiniteElementSpace(&mesh, H1fec);

   FiniteElementCollection *RTfec = new RT_FECollection(order,dim); 
   FiniteElementSpace *RTfes = new FiniteElementSpace(&mesh, RTfec);


   // Coefficients 
   ConstantCoefficient one(1.0);
   ConstantCoefficient negone(-1.0);

   // Linear forms 
   LinearForm b_0(H1fes);
   // (f,∇⋅τ )
   LinearForm b_1(RTfes);
   b_1.AddDomainIntegrator(new VectorFEDomainLFDivIntegrator(negone));

   // Bilinear forms
   // (∇u,∇v)
   BilinearForm a_00(H1fes);
   a_00.AddDomainIntegrator(new DiffusionIntegrator(one));

   //   -(σ,∇v)  
   MixedBilinearForm a_01(RTfes, H1fes);
   a_01.AddDomainIntegrator(new MixedVectorWeakDivergenceIntegrator(one)); // (-1 is included)

   //    // -(∇u,τ)
   // MixedBilinearForm()
   MixedBilinearForm a_10(H1fes, RTfes);
   a_10.AddDomainIntegrator(new MixedVectorGradientIntegrator(negone));

   // (∇⋅σ, ∇⋅τ) + (σ,τ) 

   BilinearForm a_11(RTfes);
   a_11.AddDomainIntegrator(new DivDivIntegrator(one));
   a_11.AddDomainIntegrator(new VectorFEMassIntegrator(one));


   Array<int> ess_bdr;
   Array<int> ess_tdof_list;
   if (mesh.bdr_attributes.Size())
   {
      ess_bdr.SetSize(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      H1fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   Array<int> block_offsets(3);
   block_offsets[0] = 0;
   block_offsets[1] = H1fes->GetVSize();
   block_offsets[2] = RTfes->GetVSize();
   block_offsets.PartialSum();


   BlockVector x(block_offsets), rhs(block_offsets);
   x = 0.0;  rhs = 0.0;

   GridFunction u_gf, sigma_gf;
   u_gf.MakeRef(H1fes,x.GetBlock(0));
   u_gf = 0.0;
   sigma_gf.MakeRef(RTfes,x.GetBlock(1));
   sigma_gf = 0.0;


   b_0.Update(H1fes,rhs.GetBlock(0),0);
   b_0.Assemble();

   // b_0.Print();


   b_1.Update(RTfes,rhs.GetBlock(1),0);
   b_1.Assemble();


   // b_1.Print();

   // Assembly and BC
   a_00.Assemble();
   a_00.EliminateEssentialBC(ess_bdr,x.GetBlock(0), rhs.GetBlock(0));
   a_00.Finalize();
   SparseMatrix &A_00 = a_00.SpMat();

   a_01.Assemble();
   a_01.EliminateTestDofs(ess_bdr);
   a_01.Finalize();
   SparseMatrix &A_01 = a_01.SpMat();

   a_10.Assemble();
   a_10.EliminateTrialDofs(ess_bdr,x.GetBlock(0), rhs.GetBlock(1));
   a_10.Finalize();
   SparseMatrix &A_10 = a_10.SpMat();

   a_11.Assemble();
   a_11.Finalize();
   
   SparseMatrix &A_11 = a_11.SpMat();

   BlockMatrix BlockA(block_offsets);
   BlockA.SetBlock(0,0,&A_00);
   BlockA.SetBlock(0,1,&A_01);
   BlockA.SetBlock(1,0,&A_10);
   BlockA.SetBlock(1,1,&A_11);

   SparseMatrix * A = BlockA.CreateMonolithic();

   GSSmoother M(*A);

   CGSolver cg;
   cg.SetRelTol(1e-6);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(1);
   cg.SetPreconditioner(M);
   cg.SetOperator(*A);
   cg.Mult(rhs, x);

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

   return 0;
}
