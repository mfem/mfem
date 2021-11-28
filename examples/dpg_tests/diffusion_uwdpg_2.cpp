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

// UW-DPG:
// 
// u ∈ L^2(Ω), σ ∈ (L^2(Ω))^dim 
// û ∈ H^1/2, σ̂ ∈ H^-1/2  
// -(u , ∇⋅v) + < û, v⋅n> - (σ , v) = 0,      ∀ v ∈ H(div,Ω)      
//  (σ , ∇ τ) - < σ̂, τ  >           = (f,τ)   ∀ τ ∈ H^1(Ω)
//            û = 0        on ∂Ω 



// ----------------------------------------------------------------------
// |   |     u     |     σ      |     û      |    σ̂    |  RHS    |
// ----------------------------------------------------------------------
// | v | -(u,∇⋅v)  |  -(σ,v)    |  < û, v⋅n> |         |    0    |
// |   |           |                         |         |
// | τ |           |  (σ,∇ τ)   |            | -<σ̂,τ>  |  (f,τ)  |  

// where (v,τ) ∈  H(div,Ω) × H^1(Ω) 

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

double uexact(const Vector & x)
{
   return 1.;
}

void sigma_exact(const Vector & x, Vector & sigma)
{
   sigma.SetSize(2);
   sigma = 0.;
}

double fexact(const Vector & x)
{
   return 0;
}

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   int order = 1;
   bool visualization = true;

   OptionsParser args(argc, argv);
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

   Mesh mesh = Mesh::MakeCartesian2D(1,1,mfem::Element::QUADRILATERAL);


   int dim = mesh.Dimension();


   // Define spaces
   // L2 space for u
   FiniteElementCollection *u_fec = new L2_FECollection(order-1,dim);
   FiniteElementSpace *u_fes = new FiniteElementSpace(&mesh,u_fec);

   // L2 space for σ_x
   FiniteElementCollection *sigma_x_fec = new L2_FECollection(order-1,dim);
   FiniteElementSpace *sigma_x_fes = new FiniteElementSpace(&mesh,sigma_x_fec); 


   // L2 space for σ_y
   FiniteElementCollection *sigma_y_fec = new L2_FECollection(order-1,dim);
   FiniteElementSpace *sigma_y_fes = new FiniteElementSpace(&mesh,sigma_y_fec); 

   // H^1/2 space for û 
   FiniteElementCollection * hatu_fec = new H1_Trace_FECollection(order,dim);
   FiniteElementSpace *hatu_fes = new FiniteElementSpace(&mesh,hatu_fec);

   // H^-1/2 space for σ̂ 
   FiniteElementCollection * hatsigma_fec = new RT_Trace_FECollection(order-1,dim);   
   FiniteElementSpace *hatsigma_fes = new FiniteElementSpace(&mesh,hatsigma_fec);

   // testspace fe collections
   int test_order = order;
   FiniteElementCollection * v_fec = new RT_FECollection(test_order-1, dim);
   FiniteElementSpace *v_fes = new FiniteElementSpace(&mesh,v_fec);


   FiniteElementCollection * tau_fec = new H1_FECollection(test_order, dim);
   FiniteElementSpace *tau_fes = new FiniteElementSpace(&mesh,tau_fec);

   // Coefficients
   ConstantCoefficient one(1.0);
   ConstantCoefficient negone(-1.0);
   Vector negone_x_v(dim); negone_x_v = 0.0; negone_x_v(0) = -1.;
   Vector negone_y_v(dim); negone_y_v = 0.0; negone_y_v(1) = -1.;
   Vector one_x_v(dim); one_x_v = 0.0; one_x_v(0) = 1.0;
   Vector one_y_v(dim); one_y_v = 0.0; one_y_v(1) = 1.0;

   VectorConstantCoefficient negone_x(negone_x_v);
   VectorConstantCoefficient negone_y(negone_y_v);
   VectorConstantCoefficient one_x(one_x_v);
   VectorConstantCoefficient one_y(one_y_v);

   Array<int> empty;


   // -(u,∇⋅v)
   MixedBilinearForm * a00 = new MixedBilinearForm(u_fes,v_fes);
   a00->AddDomainIntegrator(new MixedScalarWeakGradientIntegrator(one));
   a00->Assemble();
   SparseMatrix A00;
   a00->FormRectangularSystemMatrix(empty,empty,A00);


   // -(σ,v) 
   MixedBilinearForm * a01 = new MixedBilinearForm(sigma_x_fes,v_fes);
   // -((1,0)σ_x,v) 
   a01->AddDomainIntegrator(new MixedVectorProductIntegrator(negone_x));
   a01->Assemble();
   SparseMatrix A01;
   a01->FormRectangularSystemMatrix(empty,empty,A01);

   // -((0,1)σ_y,v) 
   MixedBilinearForm * a02 = new MixedBilinearForm(sigma_y_fes,v_fes);
   a02->AddDomainIntegrator(new MixedVectorProductIntegrator(negone_y));
   a02->Assemble();
   SparseMatrix A02;
   a02->FormRectangularSystemMatrix(empty,empty,A02);


   // ((1,0)σ_x,∇ τ)
   MixedBilinearForm * a11 = new MixedBilinearForm(sigma_x_fes,tau_fes);
   a11->AddDomainIntegrator(new MixedScalarWeakDivergenceIntegrator(negone_x));
   a11->Assemble();
   SparseMatrix A11;
   a11->FormRectangularSystemMatrix(empty,empty,A11);

   // ((0,1)σ_y,∇ τ)
   MixedBilinearForm * a12 = new MixedBilinearForm(sigma_y_fes,tau_fes);
   a12->AddDomainIntegrator(new MixedScalarWeakDivergenceIntegrator(negone_y));
   a12->Assemble();
   SparseMatrix A12;
   a12->FormRectangularSystemMatrix(empty,empty,A12);


   // < û, v⋅n>
   MixedBilinearForm * a03 = new MixedBilinearForm(hatu_fes,v_fes);
   a03->AddTraceFaceIntegrator(new NormalTraceJumpIntegrator);
   a03->Assemble();
   SparseMatrix A03;
   a03->FormRectangularSystemMatrix(empty,empty,A03);

   // // -<σ̂,τ> (sign is included in the variable)
   // < û, v⋅n>
   MixedBilinearForm * a14 = new MixedBilinearForm(hatsigma_fes,tau_fes);
   a14->AddTraceFaceIntegrator(new TraceJumpIntegrator);
   a14->Assemble();
   SparseMatrix A14;
   a14->FormRectangularSystemMatrix(empty,empty,A14);

   BilinearForm * g00 = new BilinearForm(v_fes);
   SumIntegrator * suminteg1 = new SumIntegrator;
   suminteg1->AddIntegrator(new DivDivIntegrator(one));
   suminteg1->AddIntegrator(new VectorFEMassIntegrator(one));

   InverseIntegrator * integ1 = new InverseIntegrator(suminteg1);
   g00->AddDomainIntegrator(integ1);
   g00->Assemble();
   SparseMatrix G00;
   g00->FormSystemMatrix(empty,G00);

   BilinearForm * g11 = new BilinearForm(tau_fes);
   SumIntegrator * suminteg2 = new SumIntegrator;
   suminteg2->AddIntegrator(new DiffusionIntegrator(one));
   suminteg2->AddIntegrator(new MassIntegrator(one));
   InverseIntegrator * integ2 = new InverseIntegrator(suminteg2);
   g11->AddDomainIntegrator(integ2);
   g11->Assemble();
   SparseMatrix G11;
   g11->FormSystemMatrix(empty,G11);

   LinearForm *b0 = new LinearForm(v_fes);
   b0->Assemble();


   LinearForm *b1 = new LinearForm(tau_fes);
   b1->AddDomainIntegrator(new DomainLFIntegrator(one));
   b1->Assemble();


   Array<int> offsets(6);
   offsets[0] = 0;
   offsets[1] = u_fes->GetTrueVSize();
   offsets[2] = sigma_x_fes->GetTrueVSize();
   offsets[3] = sigma_y_fes->GetTrueVSize();
   offsets[4] = hatu_fes->GetTrueVSize();
   offsets[5] = hatsigma_fes->GetTrueVSize();
   offsets.PartialSum();


   Array<int> offsetsG(3);
   offsetsG[0] = 0;
   offsetsG[1] = v_fes->GetTrueVSize();
   offsetsG[2] = tau_fes->GetTrueVSize();
   offsetsG.PartialSum();

   BlockMatrix B0(offsetsG,offsets);
   B0.SetBlock(0,0,&A00);
   B0.SetBlock(0,1,&A01);
   B0.SetBlock(0,2,&A02);
   B0.SetBlock(0,3,&A03);
   B0.SetBlock(1,1,&A11);
   B0.SetBlock(1,2,&A12);
   B0.SetBlock(1,4,&A14);

   SparseMatrix * B = B0.CreateMonolithic();

   BlockMatrix G0(offsetsG);
   G0.SetBlock(0,0,&G00);
   G0.SetBlock(1,1,&G11);

   SparseMatrix * G = G0.CreateMonolithic();



   Vector b(v_fes->GetTrueVSize() + tau_fes->GetTrueVSize());
   b = 0.;
   b.SetVector(*b1,v_fes->GetTrueVSize());

   SparseMatrix * A = RAP(*B, *G, *B);


   Vector L(b.Size());
   G->Mult(b,L);

   Vector BtGl(B->Width());
   B->MultTranspose(L,BtGl);
   
   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
   if (mesh.bdr_attributes.Size())
   {
      ess_bdr.SetSize(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      hatu_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // shift the ess_tdofs
   for (int i = 0; i < ess_tdof_list.Size(); i++)
   {
      ess_tdof_list[i] += u_fes->GetTrueVSize() + sigma_x_fes->GetTrueVSize()
                                                + sigma_y_fes->GetTrueVSize();
   }

   Vector X(BtGl.Size());
   X = 0.0;

   for (int i = 0; i<ess_tdof_list.Size(); i++)
   {
      int j = ess_tdof_list[i];
      A->EliminateRowCol(j,X[j],BtGl);
   }

   CGSolver cg;
   GSSmoother M(*A);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(3);
   cg.SetPreconditioner(M);
   cg.SetOperator(*A);
   cg.Mult(BtGl, X);

   GridFunction u_gf;
   double *data = X.GetData();
   u_gf.MakeRef(u_fes,data);


   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream solu_sock(vishost, visport);
   solu_sock.precision(8);
   solu_sock << "solution\n" << mesh << u_gf <<
             "window_title 'Numerical u' "
             << flush;

   // delete a;
   delete tau_fec;
   delete v_fec;
   delete hatsigma_fes;
   delete hatsigma_fec;
   delete hatu_fes;
   delete hatu_fec;
   delete sigma_x_fec;
   delete sigma_y_fes;
   delete u_fec;
   delete u_fes;


   return 0;
}
