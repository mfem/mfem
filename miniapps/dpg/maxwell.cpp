//                   MFEM Ultraweak DPG example for Maxwell
//
// Compile with: make maxwell
//
// Sample runs
// maxwell -m ../../data/inline-tri.mesh -ref 4 -o 1 -rnum 1.0
// maxwell -m ../../data/amr-quad.mesh -ref 3 -o 2 -rnum 1.6 -sc
// maxwell -m ../../data/inline-quad.mesh -ref 2 -o 3 -rnum 4.2 -sc
// maxwell -m ../../data/inline-hex.mesh -ref 1 -o 2 -sc -rnum 1.0

// Description:
// This example code demonstrates the use of MFEM to define and solve
// the "ultraweak" (UW) DPG formulation for the (indefinite) Maxwell problem

//      ∇×(1/μ ∇×E) - ω² ϵ E = Ĵ ,   in Ω
//                       E×n = E₀, on ∂Ω

// It solves a problem with a manufactured solution E_exact being a plane wave
// in the x-component and zero in y (and z) component.
// This example computes and prints out convergence rates for the L² error.

// The DPG UW deals with the First Order System
//  i ω μ H + ∇ × E = 0,   in Ω
// -i ω ϵ E + ∇ × H = J,   in Ω        (1)
//            E × n = E₀,  on ∂Ω

// Note: Ĵ = -iωJ
// in 2D
// E is vector valued and H is scalar.
//    (∇ × E, F) = (E, ∇ × F) + < n × E , F>
// or (∇ ⋅ AE , F) = (AE, ∇ F) + < AE ⋅ n, F>
// where A = [0 1; -1 0];

// E ∈ (L²(Ω))² , H ∈ L²(Ω)
// Ê ∈ H^-1/2(Ω)(Γₕ), Ĥ ∈ H^1/2(Γₕ)
//  i ω μ (H,F) + (E, ∇ × F) + < AÊ, F > = 0,      ∀ F ∈ H¹
// -i ω ϵ (E,G) + (H,∇ × G)  + < Ĥ, G × n > = (J,G)   ∀ G ∈ H(curl,Ω)
//                                    Ê = E₀      on ∂Ω
// -------------------------------------------------------------------------
// |   |       E      |      H      |      Ê       |       Ĥ      |  RHS    |
// -------------------------------------------------------------------------
// | F |  (E,∇ × F)   | i ω μ (H,F) |   < Ê, F >   |              |         |
// |   |              |             |              |              |         |
// | G | -i ω ϵ (E,G) |  (H,∇ × G)  |              | < Ĥ, G × n > |  (J,G)  |
// where (F,G) ∈  H¹ × H(curl,Ω)

// in 3D
// E,H ∈ ((L²(Ω)))³
// Ê ∈ H_0^1/2(Ω)(curl, Γₕ), Ĥ ∈ H^-1/2(curl, Γₕ)
//  i ω μ (H,F) + (E,∇ × F) + < Ê, F × n > = 0,      ∀ F ∈ H(curl,Ω)
// -i ω ϵ (E,G) + (H,∇ × G) + < Ĥ, G × n > = (J,G)   ∀ G ∈ H(curl,Ω)
//                                   Ê × n = E_0     on ∂Ω
// -------------------------------------------------------------------------
// |   |       E      |      H      |      Ê       |       Ĥ      |  RHS    |
// -------------------------------------------------------------------------
// | F |  (E,∇ × F)   | i ω μ (H,F) | < n × Ê, F > |              |         |
// |   |              |             |              |              |         |
// | G | -i ω ϵ (E,G) |  (H,∇ × G)  |              | < n × Ĥ, G > |  (J,G)  |
// where (F,G) ∈  H(curl,Ω) × H(curl,Ω)

// Here we use the "Adjoint Graph" norm on the test space i.e.,
// ||(F,G)||²ᵥ  = ||A^*(F,G)||² + ||(F,G)||² where A is the
// Maxwell operator defined by (1)

// For more information see https://doi.org/10.1016/j.camwa.2021.01.017

#include "mfem.hpp"
#include "util/complexweakform.hpp"
#include "../common/mfem-common.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mfem::common;

void maxwell_solution(const Vector & X, std::vector<complex<double>> &E);

void maxwell_solution_curl(const Vector & X,
                           std::vector<complex<double>> &curlE);

void maxwell_solution_curlcurl(const Vector & X,
                               std::vector<complex<double>> &curlcurlE);

void E_exact_r(const Vector &x, Vector & E_r);
void E_exact_i(const Vector &x, Vector & E_i);

void H_exact_r(const Vector &x, Vector & H_r);
void H_exact_i(const Vector &x, Vector & H_i);

void curlE_exact_r(const Vector &x, Vector &curlE_r);
void curlE_exact_i(const Vector &x, Vector &curlE_i);
void curlH_exact_r(const Vector &x,Vector &curlH_r);
void curlH_exact_i(const Vector &x,Vector &curlH_i);

void curlcurlE_exact_r(const Vector &x, Vector & curlcurlE_r);
void curlcurlE_exact_i(const Vector &x, Vector & curlcurlE_i);

void hatE_exact_r(const Vector & X, Vector & hatE_r);
void hatE_exact_i(const Vector & X, Vector & hatE_i);

void hatH_exact_r(const Vector & X, Vector & hatH_r);
void hatH_exact_i(const Vector & X, Vector & hatH_i);

double hatH_exact_scalar_r(const Vector & X);
double hatH_exact_scalar_i(const Vector & X);

void rhs_func_r(const Vector &x, Vector & J_r);
void rhs_func_i(const Vector &x, Vector & J_i);

int dim;
int dimc;
double omega;
double mu = 1.0;
double epsilon = 1.0;

int main(int argc, char *argv[])
{
   const char *mesh_file = "../../data/inline-quad.mesh";
   int order = 1;
   int delta_order = 1;
   bool visualization = true;
   double rnum=1.0;
   int ref = 0;
   bool static_cond = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree)");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&rnum, "-rnum", "--number-of-wavelengths",
                  "Number of wavelengths");
   args.AddOption(&mu, "-mu", "--permeability",
                  "Permeability of free space (or 1/(spring constant)).");
   args.AddOption(&epsilon, "-eps", "--permittivity",
                  "Permittivity of free space (or mass constant).");
   args.AddOption(&delta_order, "-do", "--delta-order",
                  "Order enrichment for DPG test space.");
   args.AddOption(&ref, "-ref", "--serial-ref",
                  "Number of serial refinements.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   omega = 2.*M_PI*rnum;

   Mesh mesh(mesh_file, 1, 1);
   dim = mesh.Dimension();
   MFEM_VERIFY(dim > 1, "Dimension = 1 is not supported in this example");

   dimc = (dim == 3) ? 3 : 1;
   int test_order = order+delta_order;

   // Define spaces
   enum TrialSpace
   {
      E_space     = 0,
      H_space     = 1,
      hatE_space  = 2,
      hatH_space  = 3
   };
   enum TestSpace
   {
      F_space = 0,
      G_space = 1
   };
   // L2 space for E
   FiniteElementCollection *E_fec = new L2_FECollection(order-1,dim);
   FiniteElementSpace *E_fes = new FiniteElementSpace(&mesh,E_fec,dim);

   // Vector L2 space for H
   FiniteElementCollection *H_fec = new L2_FECollection(order-1,dim);
   FiniteElementSpace *H_fes = new FiniteElementSpace(&mesh,H_fec, dimc);

   // H^-1/2 (curl) space for Ê
   FiniteElementCollection * hatE_fec = nullptr;
   FiniteElementCollection * hatH_fec = nullptr;
   FiniteElementCollection * F_fec = nullptr;
   if (dim == 3)
   {
      hatE_fec = new ND_Trace_FECollection(order,dim);
      hatH_fec = new ND_Trace_FECollection(order,dim);
      F_fec = new ND_FECollection(test_order, dim);
   }
   else
   {
      hatE_fec = new RT_Trace_FECollection(order-1,dim);
      hatH_fec = new H1_Trace_FECollection(order,dim);
      F_fec = new H1_FECollection(test_order, dim);
   }
   FiniteElementSpace *hatE_fes = new FiniteElementSpace(&mesh,hatE_fec);
   FiniteElementSpace *hatH_fes = new FiniteElementSpace(&mesh,hatH_fec);
   FiniteElementCollection * G_fec = new ND_FECollection(test_order, dim);

   // Coefficients
   ConstantCoefficient one(1.0);
   ConstantCoefficient eps2omeg2(epsilon*epsilon*omega*omega);
   ConstantCoefficient mu2omeg2(mu*mu*omega*omega);
   ConstantCoefficient muomeg(mu*omega);
   ConstantCoefficient negepsomeg(-epsilon*omega);
   ConstantCoefficient epsomeg(epsilon*omega);
   ConstantCoefficient negmuomeg(-mu*omega);

   DenseMatrix rot_mat(2);
   rot_mat(0,0) = 0.; rot_mat(0,1) = 1.;
   rot_mat(1,0) = -1.; rot_mat(1,1) = 0.;
   MatrixConstantCoefficient rot(rot_mat);
   ScalarMatrixProductCoefficient epsrot(epsomeg,rot);
   ScalarMatrixProductCoefficient negepsrot(negepsomeg,rot);

   Array<FiniteElementSpace * > trial_fes;
   Array<FiniteElementCollection * > test_fec;

   trial_fes.Append(E_fes);
   trial_fes.Append(H_fes);
   trial_fes.Append(hatE_fes);
   trial_fes.Append(hatH_fes);

   test_fec.Append(F_fec);
   test_fec.Append(G_fec);

   ComplexDPGWeakForm * a = new ComplexDPGWeakForm(trial_fes,test_fec);
   a->StoreMatrices();

   // (E,∇ × F)
   a->AddTrialIntegrator(new TransposeIntegrator(new MixedCurlIntegrator(one)),
                         nullptr,TrialSpace::E_space,TestSpace::F_space);
   // -i ω ϵ (E , G)
   a->AddTrialIntegrator(nullptr,
                         new TransposeIntegrator(new VectorFEMassIntegrator(negepsomeg)),
                         TrialSpace::E_space,TestSpace::G_space);
   //  (H,∇ × G)
   a->AddTrialIntegrator(new TransposeIntegrator(new MixedCurlIntegrator(one)),
                         nullptr, TrialSpace::H_space,TestSpace::G_space);

   if (dim == 3)
   {
      // i ω μ (H, F)
      a->AddTrialIntegrator(nullptr,
                            new TransposeIntegrator(new VectorFEMassIntegrator(muomeg)),
                            TrialSpace::H_space,TestSpace::F_space);
      // < n×Ê,F>
      a->AddTrialIntegrator(new TangentTraceIntegrator,nullptr,
                            TrialSpace::hatE_space,TestSpace::F_space);
   }
   else
   {
      // i ω μ (H, F)
      a->AddTrialIntegrator(nullptr,new MixedScalarMassIntegrator(muomeg),
                            TrialSpace::H_space,TestSpace::F_space);
      // <Ê,F>
      a->AddTrialIntegrator(new TraceIntegrator,nullptr,
                            TrialSpace::hatE_space, TestSpace::F_space);
   }
   // < n×Ĥ ,G>
   a->AddTrialIntegrator(new TangentTraceIntegrator,nullptr,
                         TrialSpace::hatH_space, TestSpace::G_space);

   // test integrators for the adjoint graph norm on the test space
   // (∇×G ,∇× δG)
   a->AddTestIntegrator(new CurlCurlIntegrator(one),nullptr,
                        TestSpace::G_space,TestSpace::G_space);
   // (G,δG)
   a->AddTestIntegrator(new VectorFEMassIntegrator(one),nullptr,
                        TestSpace::G_space, TestSpace::G_space);

   if (dim == 3)
   {
      // (∇×F,∇×δF)
      a->AddTestIntegrator(new CurlCurlIntegrator(one),nullptr,
                           TestSpace::F_space, TestSpace::F_space);
      // (F,δF)
      a->AddTestIntegrator(new VectorFEMassIntegrator(one),nullptr,
                           TestSpace::F_space,TestSpace::F_space);
      // μ^2 ω^2 (F,δF)
      a->AddTestIntegrator(new VectorFEMassIntegrator(mu2omeg2),nullptr,
                           TestSpace::F_space, TestSpace::F_space);
      // -i ω μ (F,∇ × δG) = (F, ω μ ∇ × δ G)
      a->AddTestIntegrator(nullptr,new MixedVectorWeakCurlIntegrator(negmuomeg),
                           TestSpace::F_space, TestSpace::G_space);
      // -i ω ϵ (∇ × F, δG)
      a->AddTestIntegrator(nullptr,new MixedVectorCurlIntegrator(negepsomeg),
                           TestSpace::F_space, TestSpace::G_space);
      // i ω μ (∇ × G,δF)
      a->AddTestIntegrator(nullptr,new MixedVectorCurlIntegrator(epsomeg),
                           TestSpace::G_space, TestSpace::F_space);
      // i ω ϵ (G, ∇ × δF )
      a->AddTestIntegrator(nullptr,new MixedVectorWeakCurlIntegrator(muomeg),
                           TestSpace::G_space, TestSpace::F_space);
      // ϵ^2 ω^2 (G,δG)
      a->AddTestIntegrator(new VectorFEMassIntegrator(eps2omeg2),nullptr,
                           TestSpace::G_space, TestSpace::G_space);
   }
   else
   {
      // (∇F,∇δF)
      a->AddTestIntegrator(new DiffusionIntegrator(one),nullptr,
                           TestSpace::F_space, TestSpace::F_space);
      // (F,δF)
      a->AddTestIntegrator(new MassIntegrator(one),nullptr,
                           TestSpace::F_space, TestSpace::F_space);
      // μ^2 ω^2 (F,δF)
      a->AddTestIntegrator(new MassIntegrator(mu2omeg2),nullptr,
                           TestSpace::F_space, TestSpace::F_space);
      // -i ω μ (F,∇ × δG) = i (F, -ω μ ∇ × δ G)
      a->AddTestIntegrator(nullptr,
                           new TransposeIntegrator(new MixedCurlIntegrator(negmuomeg)),
                           TestSpace::F_space, TestSpace::G_space);
      // -i ω ϵ (∇ × F, δG) = i (- ω ϵ A ∇ F,δG), A = [0 1; -1; 0]
      a->AddTestIntegrator(nullptr,new MixedVectorGradientIntegrator(negepsrot),
                           TestSpace::F_space, TestSpace::G_space);
      // i ω μ (∇ × G,δF) = i (ω μ ∇ × G, δF )
      a->AddTestIntegrator(nullptr,new MixedCurlIntegrator(muomeg),
                           TestSpace::G_space, TestSpace::F_space);
      // i ω ϵ (G, ∇ × δF ) =  i (ω ϵ G, A ∇ δF) = i ( G , ω ϵ A ∇ δF)
      a->AddTestIntegrator(nullptr,
                           new TransposeIntegrator(new MixedVectorGradientIntegrator(epsrot)),
                           TestSpace::G_space, TestSpace::F_space);
      // ϵ^2 ω^2 (G,δG)
      a->AddTestIntegrator(new VectorFEMassIntegrator(eps2omeg2),nullptr,
                           TestSpace::G_space, TestSpace::G_space);
   }

   // RHS
   VectorFunctionCoefficient f_rhs_r(dim,rhs_func_r);
   VectorFunctionCoefficient f_rhs_i(dim,rhs_func_i);
   a->AddDomainLFIntegrator(new VectorFEDomainLFIntegrator(f_rhs_r),
                            new VectorFEDomainLFIntegrator(f_rhs_i),
                            TestSpace::G_space);

   VectorFunctionCoefficient hatEex_r(dim,hatE_exact_r);
   VectorFunctionCoefficient hatEex_i(dim,hatE_exact_i);

   socketstream E_out_r;
   socketstream E_out_i;

   double err0 = 0.;
   int dof0;

   std::cout << "\n  Ref |"
             << "    Dofs    |"
             << "    ω    |"
             << "  L2 Error  |"
             << "  Rate  |"
             << " PCG it |" << endl;
   std::cout << std::string(60,'-')
             << endl;

   for (int it = 0; it<=ref; it++)
   {
      if (static_cond) { a->EnableStaticCondensation(); }
      a->Assemble();

      Array<int> ess_tdof_list;
      Array<int> ess_bdr;
      if (mesh.bdr_attributes.Size())
      {
         ess_bdr.SetSize(mesh.bdr_attributes.Max());
         ess_bdr = 1;
         hatE_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }

      // shift the ess_tdofs
      for (int j = 0; j < ess_tdof_list.Size(); j++)
      {
         ess_tdof_list[j] += E_fes->GetTrueVSize() + H_fes->GetTrueVSize();
      }

      Array<int> offsets(5);
      offsets[0] = 0;
      offsets[1] = E_fes->GetVSize();
      offsets[2] = H_fes->GetVSize();
      offsets[3] = hatE_fes->GetVSize();
      offsets[4] = hatH_fes->GetVSize();
      offsets.PartialSum();

      Vector x(2*offsets.Last());
      x = 0.;

      GridFunction hatE_gf_r(hatE_fes, x, offsets[2]);
      GridFunction hatE_gf_i(hatE_fes, x, offsets.Last() + offsets[2]);
      if (dim == 3)
      {
         hatE_gf_r.ProjectBdrCoefficientTangent(hatEex_r, ess_bdr);
         hatE_gf_i.ProjectBdrCoefficientTangent(hatEex_i, ess_bdr);
      }
      else
      {
         hatE_gf_r.ProjectBdrCoefficientNormal(hatEex_r, ess_bdr);
         hatE_gf_i.ProjectBdrCoefficientNormal(hatEex_i, ess_bdr);
      }
      OperatorPtr Ah;
      Vector X,B;
      a->FormLinearSystem(ess_tdof_list,x,Ah, X,B);

      ComplexOperator * Ahc = Ah.As<ComplexOperator>();

      BlockMatrix * A_r = dynamic_cast<BlockMatrix *>(&Ahc->real());
      BlockMatrix * A_i = dynamic_cast<BlockMatrix *>(&Ahc->imag());

      int num_blocks = A_r->NumRowBlocks();
      Array<int> tdof_offsets(2*num_blocks+1);
      tdof_offsets[0] = 0;
      int k = (static_cond) ? 2 : 0;
      for (int i=0; i<num_blocks; i++)
      {
         tdof_offsets[i+1] = trial_fes[i+k]->GetTrueVSize();
         tdof_offsets[num_blocks+i+1] = trial_fes[i+k]->GetTrueVSize();
      }
      tdof_offsets.PartialSum();

      BlockOperator A(tdof_offsets);
      for (int i = 0; i<num_blocks; i++)
      {
         for (int j = 0; j<num_blocks; j++)
         {
            A.SetBlock(i,j,&A_r->GetBlock(i,j));
            A.SetBlock(i,j+num_blocks,&A_i->GetBlock(i,j), -1.0);
            A.SetBlock(i+num_blocks,j+num_blocks,&A_r->GetBlock(i,j));
            A.SetBlock(i+num_blocks,j,&A_i->GetBlock(i,j));
         }
      }

      BlockDiagonalPreconditioner M(tdof_offsets);
      M.owns_blocks = 1;
      for (int i = 0; i<num_blocks; i++)
      {
         M.SetDiagonalBlock(i, new GSSmoother((SparseMatrix&)A_r->GetBlock(i,i)));
         M.SetDiagonalBlock(num_blocks+i, new GSSmoother((SparseMatrix&)A_r->GetBlock(i,
                                                                                      i)));
      }

      CGSolver cg;
      cg.SetRelTol(1e-10);
      cg.SetMaxIter(2000);
      cg.SetPrintLevel(0);
      cg.SetPreconditioner(M);
      cg.SetOperator(A);
      cg.Mult(B, X);

      a->RecoverFEMSolution(X,x);

      GridFunction E_r(E_fes, x, 0);
      GridFunction E_i(E_fes, x, offsets.Last());

      VectorFunctionCoefficient E_ex_r(dim,E_exact_r);
      VectorFunctionCoefficient E_ex_i(dim,E_exact_i);

      GridFunction H_r(H_fes, x, offsets[1]);
      GridFunction H_i(H_fes, x, offsets.Last() + offsets[1]);

      VectorFunctionCoefficient H_ex_r(dimc,H_exact_r);
      VectorFunctionCoefficient H_ex_i(dimc,H_exact_i);

      int dofs = 0;
      for (int i = 0; i<trial_fes.Size(); i++)
      {
         dofs += trial_fes[i]->GetTrueVSize();
      }

      double E_err_r = E_r.ComputeL2Error(E_ex_r);
      double E_err_i = E_i.ComputeL2Error(E_ex_i);
      double H_err_r = H_r.ComputeL2Error(H_ex_r);
      double H_err_i = H_i.ComputeL2Error(H_ex_i);

      double L2Error = sqrt(E_err_r*E_err_r + E_err_i*E_err_i
                            + H_err_r*H_err_r + H_err_i*H_err_i);

      double rate_err = (it) ? dim*log(err0/L2Error)/log((double)dof0/dofs) : 0.0;

      err0 = L2Error;
      dof0 = dofs;

      std::ios oldState(nullptr);
      oldState.copyfmt(std::cout);
      std::cout << std::right << std::setw(5) << it << " | "
                << std::setw(10) <<  dof0 << " | "
                << std::setprecision(1) << std::fixed
                << std::setw(4) <<  2*rnum << " π  | "
                << std::setprecision(3)
                << std::setw(10) << std::scientific << err0 << " | "
                << std::setprecision(2)
                << std::setw(6) << std::fixed << rate_err << " | "
                << std::setw(6) << std::fixed << cg.GetNumIterations() << " | "
                << std::endl;
      std::cout.copyfmt(oldState);

      if (visualization)
      {
         const char * keys = (it == 0 && dim == 2) ? "jRcml\n" : nullptr;
         char vishost[] = "localhost";
         int  visport   = 19916;
         VisualizeField(E_out_r,vishost, visport, E_r,
                        "Numerical Electric field (real part)", 0, 0, 500, 500, keys);
         VisualizeField(E_out_i,vishost, visport, E_i,
                        "Numerical Electric field (imaginary part)", 501, 0, 500, 500, keys);
      }

      if (it == ref)
      {
         break;
      }

      mesh.UniformRefinement();
      for (int i =0; i<trial_fes.Size(); i++)
      {
         trial_fes[i]->Update(false);
      }
      a->Update();
   }

   delete a;
   delete F_fec;
   delete G_fec;
   delete hatH_fes;
   delete hatH_fec;
   delete hatE_fes;
   delete hatE_fec;
   delete H_fec;
   delete E_fec;
   delete H_fes;
   delete E_fes;

   return 0;
}

void E_exact_r(const Vector &x, Vector & E_r)
{
   std::vector<std::complex<double>> E;
   maxwell_solution(x, E);
   E_r.SetSize(E.size());
   for (unsigned i = 0; i < E.size(); i++)
   {
      E_r[i]= E[i].real();
   }
}

void E_exact_i(const Vector &x, Vector & E_i)
{
   std::vector<std::complex<double>> E;
   maxwell_solution(x, E);
   E_i.SetSize(E.size());
   for (unsigned i = 0; i < E.size(); i++)
   {
      E_i[i]= E[i].imag();
   }
}

void curlE_exact_r(const Vector &x, Vector &curlE_r)
{
   std::vector<std::complex<double>> curlE;
   maxwell_solution_curl(x, curlE);
   curlE_r.SetSize(curlE.size());
   for (unsigned i = 0; i < curlE.size(); i++)
   {
      curlE_r[i]= curlE[i].real();
   }
}

void curlE_exact_i(const Vector &x, Vector &curlE_i)
{
   std::vector<std::complex<double>> curlE;
   maxwell_solution_curl(x, curlE);
   curlE_i.SetSize(curlE.size());
   for (unsigned i = 0; i < curlE.size(); i++)
   {
      curlE_i[i]= curlE[i].imag();
   }
}

void curlcurlE_exact_r(const Vector &x, Vector & curlcurlE_r)
{
   std::vector<std::complex<double>> curlcurlE;
   maxwell_solution_curlcurl(x, curlcurlE);
   curlcurlE_r.SetSize(curlcurlE.size());
   for (unsigned i = 0; i < curlcurlE.size(); i++)
   {
      curlcurlE_r[i]= curlcurlE[i].real();
   }
}

void curlcurlE_exact_i(const Vector &x, Vector & curlcurlE_i)
{
   std::vector<std::complex<double>> curlcurlE;
   maxwell_solution_curlcurl(x, curlcurlE);
   curlcurlE_i.SetSize(curlcurlE.size());
   for (unsigned i = 0; i < curlcurlE.size(); i++)
   {
      curlcurlE_i[i]= curlcurlE[i].imag();
   }
}

void H_exact_r(const Vector &x, Vector & H_r)
{
   // H = i ∇ × E / ω μ
   // H_r = - ∇ × E_i / ω μ
   Vector curlE_i;
   curlE_exact_i(x,curlE_i);
   H_r.SetSize(dimc);
   for (int i = 0; i<dimc; i++)
   {
      H_r(i) = - curlE_i(i) / (omega * mu);
   }
}

void H_exact_i(const Vector &x, Vector & H_i)
{
   // H = i ∇ × E / ω μ
   // H_i =  ∇ × E_r / ω μ
   Vector curlE_r;
   curlE_exact_r(x,curlE_r);
   H_i.SetSize(dimc);
   for (int i = 0; i<dimc; i++)
   {
      H_i(i) = curlE_r(i) / (omega * mu);
   }
}

void curlH_exact_r(const Vector &x,Vector &curlH_r)
{
   // ∇ × H_r = - ∇ × ∇ × E_i / ω μ
   Vector curlcurlE_i;
   curlcurlE_exact_i(x,curlcurlE_i);
   curlH_r.SetSize(dim);
   for (int i = 0; i<dim; i++)
   {
      curlH_r(i) = -curlcurlE_i(i) / (omega * mu);
   }
}

void curlH_exact_i(const Vector &x,Vector &curlH_i)
{
   // ∇ × H_i = ∇ × ∇ × E_r / ω μ
   Vector curlcurlE_r;
   curlcurlE_exact_r(x,curlcurlE_r);
   curlH_i.SetSize(dim);
   for (int i = 0; i<dim; i++)
   {
      curlH_i(i) = curlcurlE_r(i) / (omega * mu);
   }
}

void hatE_exact_r(const Vector & x, Vector & hatE_r)
{
   if (dim == 3)
   {
      E_exact_r(x,hatE_r);
   }
   else
   {
      Vector E_r;
      E_exact_r(x,E_r);
      hatE_r.SetSize(hatE_r.Size());
      // rotate E_hat
      hatE_r[0] = E_r[1];
      hatE_r[1] = -E_r[0];
   }
}

void hatE_exact_i(const Vector & x, Vector & hatE_i)
{
   if (dim == 3)
   {
      E_exact_i(x,hatE_i);
   }
   else
   {
      Vector E_i;
      E_exact_i(x,E_i);
      hatE_i.SetSize(hatE_i.Size());
      // rotate E_hat
      hatE_i[0] = E_i[1];
      hatE_i[1] = -E_i[0];
   }
}

void hatH_exact_r(const Vector & x, Vector & hatH_r)
{
   H_exact_r(x,hatH_r);
}

void hatH_exact_i(const Vector & x, Vector & hatH_i)
{
   H_exact_i(x,hatH_i);
}

double hatH_exact_scalar_r(const Vector & x)
{
   Vector hatH_r;
   H_exact_r(x,hatH_r);
   return hatH_r[0];
}

double hatH_exact_scalar_i(const Vector & x)
{
   Vector hatH_i;
   H_exact_i(x,hatH_i);
   return hatH_i[0];
}

// J = -i ω ϵ E + ∇ × H
// J_r + iJ_i = -i ω ϵ (E_r + i E_i) + ∇ × (H_r + i H_i)
void rhs_func_r(const Vector &x, Vector & J_r)
{
   // J_r = ω ϵ E_i + ∇ × H_r
   Vector E_i, curlH_r;
   E_exact_i(x,E_i);
   curlH_exact_r(x,curlH_r);
   J_r.SetSize(dim);
   for (int i = 0; i<dim; i++)
   {
      J_r(i) = omega * epsilon * E_i(i) + curlH_r(i);
   }
}

void rhs_func_i(const Vector &x, Vector & J_i)
{
   // J_i = - ω ϵ E_r + ∇ × H_i
   Vector E_r, curlH_i;
   E_exact_r(x,E_r);
   curlH_exact_i(x,curlH_i);
   J_i.SetSize(dim);
   for (int i = 0; i<dim; i++)
   {
      J_i(i) = -omega * epsilon * E_r(i) + curlH_i(i);
   }
}

void maxwell_solution(const Vector & X, std::vector<complex<double>> &E)
{
   E.resize(dim);
   std::complex<double> zi(0,1);
   std::complex<double> pw = exp(-zi * omega * (X.Sum()));
   E[0] = pw;
   E[1] = 0.0;
   if (dim == 3) { E[2] = 0.0; }
}

void maxwell_solution_curl(const Vector & X,
                           std::vector<complex<double>> &curlE)
{
   curlE.resize(dimc);
   std::complex<double> zi(0,1);
   std::complex<double> pw = exp(-zi * omega * (X.Sum()));
   if (dim == 3)
   {
      curlE[0] = 0.0;
      curlE[1] = -zi * omega * pw;
      curlE[2] =  zi * omega * pw;
   }
   else
   {
      curlE[0] = zi * omega * pw;
   }
}

void maxwell_solution_curlcurl(const Vector & X,
                               std::vector<complex<double>> &curlcurlE)
{
   curlcurlE.resize(dim);
   std::complex<double> zi(0,1);
   std::complex<double> pw = exp(-zi * omega * (X.Sum()));
   if (dim == 3)
   {
      curlcurlE[0] = 2.0 * omega * omega * pw;
      curlcurlE[1] = - omega * omega * pw;
      curlcurlE[2] = - omega * omega * pw;
   }
   else
   {
      curlcurlE[0] =   omega * omega * pw;
      curlcurlE[1] = - omega * omega * pw ;
   }
}
