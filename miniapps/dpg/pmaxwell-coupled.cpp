// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
//
//                   MFEM Ultraweak DPG Maxwell parallel example
//
// Compile with: make pmaxwell-coupled
//
// mpirun -np 4 ./pmaxwell-coupled -sref 1 -pref 2 -o 2 -rnum 0.5 -m ../../data/ref-cube.mesh -sc

// ∇×(1/μ ∇×E) - ω² ϵ₀ϵ E + cj*iϵ₀ ω²(J₁+J₂) = F̃ ,   in Ω
//                      -Δ∥ J₁ + c₁ J₁  - c₁P(r) E∥  = G₁ ,  in Ω
//                      -Δ∥ J₂ + c₂ J₂  + c₂P(r) E∥  = G₂ ,  in Ω
//                                       E×n = E₀ , on ∂Ω
//                                        J₁ = J̃₁ , on ∂Ω
//                                        J₂ = J̃₂ , on ∂Ω
// The DPG UW deals with the First Order System
//  i ω μ₀ H + ∇ × E        = 0,   in Ω
// -i ω ϵ₀ϵ E + ∇ × H  - cj*ωϵ₀ (J₁+J₂) = F,   in Ω
//         -Δ∥ J₁ + c₁ J₁ - c₁ P(r) E∥ = G₁,  in Ω
//         -Δ∥ J₂ + c₂ J₂ + c₂ P(r) E∥ = G₂,  in Ω
//                              E × n = E₀,  on ∂Ω
//                                 J₁ = J̃₁,  on ∂Ω
//                                 J₂ = J̃₂,  on ∂Ω

// in 2D
// E is vector valued and H is scalar.
//    (∇ × E, δE) = (E, ∇ × δE ) + < n × E , δE >
// or (∇ ⋅ AE , δE) = (AE, ∇ δE ) + < AE ⋅ n, δE >
// where A = [0 1; -1 0];

// E ∈ (L²(Ω))² , H ∈ L²(Ω), J ∈ (H¹(Ω))²
// Ê ∈ H^-1/2(Γₕ), Ĥ ∈ H^1/2(Γₕ), Ĵ ∈ (H^-1/2(Γₕ))²
//  i ω μ (H,δE) + (E, ∇ × δE) + < AÊ, δE > = 0,      ∀ δE ∈ H¹(Ω)
// -i ω ϵ₀ϵ (E,δH) + (H,∇ × δH)  + < Ĥ, δH × n > - cj ω (J₁+J₂,δH) = (F,δH)   ∀ δH ∈ H(curl,Ω)
// ((b⋅∇)J₁, (b⋅∇)δJ₁) + <Ĵ₁,δJ₁> + c₁(J₁,δJ₁) - c₁(P(r) E∥,δJ₁) = (G₁,δJ₁),    ∀ δJ₁ ∈ (H¹(Ω))²
// ((b⋅∇)J₂, (b⋅∇)δJ₂) + <Ĵ₂,δJ₂> + c₂(J₂,δJ₂) + c₂(P(r) E∥,δJ₂) = (G₂,δJ₂),    ∀ δJ₂ ∈ (H¹(Ω))²
//                                    Ê = E₀, on ∂Ω
//                                    J₁= J̃₁, on ∂Ω
//                                    J₂= J̃₂, on ∂Ω
// --------------------------------------------------------------------------------------------------------------------
// |    |       E        |        H       |         Jᵢ             |      Ê        |       Ĥ       |    Ĵᵢ    |     RHS    |
// --------------------------------------------------------------------------------------------------------------------
// | δE |   (E,∇ × δE)   |  i ω μ (H,δE)  |                        |    < Ê, δE >  |               |          |            |
// |    |                |                |                        |               |               |          |            |
// | δH | -iωϵ₀ϵ (E,δH)  |   (H,∇ × δH)   |     -ϵ₀ω (Jᵢ,δH)       |               | < Ĥ, δH × n > |          |   (F,δH)   |
// |    |                |                |                        |               |               |          |            |
// | δJᵢ|  ± cᵢ(E, δJᵢ)  |                |(∇Jᵢ,∇δJᵢ) + cᵢ(Jᵢ,δJᵢ) |               |               |<Ĵᵢ, δJᵢ> |   (G,δJ)   |
//
// where (δE,δH,δJ) ∈  H¹(Ω) × H(curl,Ω) × (H¹(Ω))ᵈ

// E,H ∈ (L^2(Ω))³, J ∈ (H¹(Ω))³
// Ê ∈ H_0^1/2(curl, Γₕ), Ĥ ∈ H^-1/2(curl, Γₕ), Ĵ ∈ (H^-1/2(Γₕ))³
//  i ω μ (H,δE) + (E,∇ × δE) + < Ê, δE × n >              = 0,      ∀ δE ∈ H(curl,Ω)
// -i ωϵ₀ϵ(E,δH) + (H,∇ × δH) + < Ĥ, δH × n > - ω ∑(Jᵢ,δH) = (F,δH),   ∀ δH ∈ H(curl,Ω)
// ((b⋅∇)J₁, (b⋅∇)δJ₁) + <Ĵ₁,δJ₁> + c₁(J₁,δJ₁) - c₁(P(r) E∥,δJ₁) = (G₁,δJ₁),      ∀ δJ₁ ∈ (H¹(Ω))²
// ((b⋅∇)J₂, (b⋅∇)δJ₂) + <Ĵ₂,δJ₂> + c₂(J₂,δJ₂) + c₂(P(r) E∥,δJ₂) = (G₂,δJ₂),      ∀ δJ₂ ∈ (H¹(Ω))²
//
//                                   Ê × n = E₀,      on ∂Ω
//                                       J = J₀,      on ∂Ω
// --------------------------------------------------------------------------------------------------------------------
// |    |       E        |        H       |      J            |      Ê        |       Ĥ       |    Ĵ     |     RHS    |
// --------------------------------------------------------------------------------------------------------------------
// | δE |   (E,∇ × δE)   |  i ω μ (H,δE)  |                   | < Ê, δE × n > |               |          |            |
// |    |                |                |                   |               |               |          |            |
// | δH | -iωϵ₀ϵ (E,δH)  |   (H,∇ × δH)   |    -ϵ₀ω (J,δH)     |               | < Ĥ, δH × n > |          |   (F,δH)   |
// |    |                |                |                   |               |               |          |            |
// | δJ |   cᵢ (E, δJ)   |                | (∇J,∇δJ)+cᵢ(J,δJ) |               |               |  <Ĵ, δJ> |   (G,δJ)   |
//
// where (δE,δH,δJ) ∈  H(curl,Ω) × H(curl,Ω) × (H¹(Ω))ᵈ

#include "mfem.hpp"
#include "util/pcomplexweakform.hpp"
#include "../common/mfem-common.hpp"
#include "./util/maxwell_utils.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mfem::common;

real_t delta = 0.01;
real_t a0 = -1.0;
real_t a1 = 5.0;

int dim;
int dimc;
real_t omega;
real_t mu0  = 1.257; // × 1e-6
real_t eps0 = 8.8541878128; // × 1e-12
real_t cfactor = 1e-6;
real_t c1 = 25.0; // × 1e-6
real_t c2 = 1.0; // × 1e-6
real_t cj = 1.0;

void maxwell_solution(const Vector &x, std::vector<complex<double>> &E);
void maxwell_solution_curl(const Vector &x,
                           std::vector<complex<double>> &curlE);
void maxwell_solution_curlcurl(const Vector &x,
                               std::vector<complex<double>> &curlcurlE);
void J1_solution(const Vector &x,std::vector<complex<double>> &J);
void J1_solution_grad(const Vector &x,
                      std::vector<std::vector<complex<double>>> &gradJ);
void J1_solution_laplace(const Vector &x,
                         std::vector<complex<double>> &laplaceJ);
void J1_solution_directional_laplace(const Vector &x, const Vector &b,
                                     std::vector<complex<double>> &laplaceJ);

void J2_solution(const Vector &x,std::vector<complex<double>> &J);
void J2_solution_grad(const Vector &x,
                      std::vector<std::vector<complex<double>>> &gradJ);
void J2_solution_laplace(const Vector &x,
                         std::vector<complex<double>> &laplaceJ);
void J2_solution_directional_laplace(const Vector &x, const Vector &b,
                                     std::vector<complex<double>> &laplaceJ);

void E_exact_r(const Vector &x, Vector & E_r);
void E_exact_i(const Vector &x, Vector & E_i);
void H_exact_r(const Vector &x, Vector & H_r);
void H_exact_i(const Vector &x, Vector & H_i);
void J1_exact_r(const Vector &x, Vector & J_r);
void J1_exact_i(const Vector &x, Vector & J_i);
void J2_exact_r(const Vector &x, Vector & J_r);
void J2_exact_i(const Vector &x, Vector & J_i);

void curlE_exact_r(const Vector &x, Vector &curlE_r);
void curlE_exact_i(const Vector &x, Vector &curlE_i);
void curlH_exact_r(const Vector &x,Vector &curlH_r);
void curlH_exact_i(const Vector &x,Vector &curlH_i);
void gradJ1_exact_r(const Vector &x, DenseMatrix &gradJ_r);
void gradJ1_exact_i(const Vector &x, DenseMatrix &gradJ_i);
void gradJ2_exact_r(const Vector &x, DenseMatrix &gradJ_r);
void gradJ2_exact_i(const Vector &x, DenseMatrix &gradJ_i);

void curlcurlE_exact_r(const Vector &x, Vector & curlcurlE_r);
void curlcurlE_exact_i(const Vector &x, Vector & curlcurlE_i);
void LaplaceJ1_exact_r(const Vector &x, Vector & d2J_r);
void LaplaceJ1_exact_i(const Vector &x, Vector & d2J_i);
void LaplaceJ2_exact_r(const Vector &x, Vector & d2J_r);
void LaplaceJ2_exact_i(const Vector &x, Vector & d2J_i);
void DirectionalLaplaceJ1_exact_r(const Vector &x, const Vector &b,
                                  Vector & d2J_r);
void DirectionalLaplaceJ1_exact_i(const Vector &x, const Vector &b,
                                  Vector & d2J_i);
void DirectionalLaplaceJ2_exact_r(const Vector &x, const Vector &b,
                                  Vector & d2J_r);
void DirectionalLaplaceJ2_exact_i(const Vector &x, const Vector &b,
                                  Vector & d2J_i);

void hatE_exact_r(const Vector & X, Vector & hatE_r);
void hatE_exact_i(const Vector & X, Vector & hatE_i);

void  rhs_dH_func_r(const Vector &x, Vector & rhs_dH_r);
void  rhs_dH_func_i(const Vector &x, Vector & rhs_dH_i);
void  rhs_dJ1_func_r(const Vector &x, Vector & rhs_dJ1_r);
void  rhs_dJ1_func_i(const Vector &x, Vector & rhs_dJ1_i);
void  rhs_dJ2_func_r(const Vector &x, Vector & rhs_dJ2_r);
void  rhs_dJ2_func_i(const Vector &x, Vector & rhs_dJ2_i);

real_t pfunc_r(const Vector &x)
{
   real_t r = std::sqrt(x(0) * x(0) + x(1) * x(1));
   return a0 + a1 *(r-0.9);
}

real_t pfunc_i(const Vector &x)
{
   return delta;
}

real_t sfunc_r(const Vector &x)
{
   return 1.0;
}

real_t sfunc_i(const Vector &x)
{
   return delta;
}

void bfunc(const Vector &x, Vector &b)
{
   real_t r = std::sqrt(x(0) * x(0) + x(1) * x(1));
   int dim = x.Size();
   b.SetSize(dim); b = 0.0;
   b(0) = -x(1) / r;
   b(1) =  x(0) / r;
   if (dim == 3) { b(2) = 0.0; }
}

void bcrossb(const Vector &x, DenseMatrix &bb)
{
   Vector b;
   bfunc(x, b);
   bb.SetSize(b.Size());
   MultVVt(b, bb);
}

void epsilon_r(const Vector &x, DenseMatrix &eps_r)
{
   eps_r.SetSize(dim);
   eps_r = 0.0;
   DenseMatrix bb;
   bcrossb(x, bb);
   DenseMatrix Id(dim); Id = 0.0;
   for (int i = 0; i < dim; i++)
   {
      Id(i,i) = 1.0;
   }
   Id-= bb;
   real_t s = sfunc_r(x);
   real_t p = pfunc_r(x);
   Add(s,Id,p,bb,eps_r);

   // eps_r = 0.0;
   // eps_r(0,0)=1.0;
   // eps_r(1,1)=1.0;
}

void epsilon_i(const Vector &x, DenseMatrix &eps_i)
{
   eps_i.SetSize(dim);
   eps_i = 0.0;
   DenseMatrix bb;
   bcrossb(x, bb);
   DenseMatrix Id(dim); Id = 0.0;
   for (int i = 0; i < dim; i++)
   {
      Id(i,i) = 1.0;
   }
   Id-= bb;
   real_t s = sfunc_i(x);
   real_t p = pfunc_i(x);
   Add(s,Id,p,bb,eps_i);

   // eps_i = 0.0;
}

int main(int argc, char *argv[])
{
   Mpi::Init();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   const char *mesh_file = "../../data/inline-hex.mesh";
   int order = 1;
   int delta_order = 1;
   real_t rnum=1.5; // × 1e9
   int sr = 0;
   int pr = 1;
   bool static_cond = false;
   bool visualization = true;
   bool mumps_solver = false;
   int problem = 0; // 0 - square, 1 - LHhot
   bool mms = false; // Method of Manufactured Solutions
   bool paraview = false;


   OptionsParser args(argc, argv);
   args.AddOption(&problem, "-prob", "--prob",
                  "Choice between 0 (square) and 1 (LHhot). mesh");
   args.AddOption(&mms, "-mms", "--mms", "-no-mms", "--no-mms",
                  "Enable or disable MMS (method of manufactured solutions).");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree)");
   args.AddOption(&rnum, "-rnum", "--number-of-wavelengths",
                  "Number of wavelengths");
   args.AddOption(&delta_order, "-do", "--delta-order",
                  "Order enrichment for DPG test space.");
   args.AddOption(&sr, "-sref", "--serial-ref",
                  "Number of parallel refinements.");
   args.AddOption(&cj, "-cj", "--cj",
                  "Coefficient scale for J coupling.");
   args.AddOption(&delta, "-delta", "--delta",
                  "Value of delta in S(r) and P(r) imaginary parts.");
   args.AddOption(&pr, "-pref", "--parallel-ref",
                  "Number of parallel refinements.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&paraview, "-paraview", "--paraview", "-no-paraview",
                  "--no-paraview",
                  "Enable or disable ParaView visualization.");
   args.AddOption(&mumps_solver, "-mumps", "--mumps", "-no-mumps",
                  "--no-mumps",
                  "Enable or disable MUMPS solver.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }

   if (!mms && problem !=1)
   {
      problem = 1;
      if (Mpi::Root())
      {
         mfem::out << "Problem is set to 1 (LH-hot)" << endl;
      }
   }

   omega = 2.*M_PI*rnum;

   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   if (problem == 0)
   {
      mesh_file = "../../data/inline-quad.mesh";
   }
   else
   {
      // mesh_file = "./plasma/data/LH_hot.msh";
      mesh_file = "./plasma/data/quad.msh";
   }

   Mesh mesh(mesh_file, 1, 1);
   mesh.RemoveInternalBoundaries();
   dim = mesh.Dimension();
   MFEM_VERIFY(dim > 1, "Dimension = 1 is not supported in this example");

   if (problem == 0)
   {
      mesh.EnsureNodes();
      GridFunction *nodes = mesh.GetNodes();
      (*nodes) += 1.0;
   }

   dimc = (dim == 3) ? 3 : 1;

   for (int i = 0; i<sr; i++)
   {
      mesh.UniformRefinement();
   }

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   // Define spaces
   enum TrialSpace
   {
      E_space     = 0,
      H_space     = 1,
      J1_space    = 2,
      J2_space    = 3,
      hatE_space  = 4,
      hatH_space  = 5,
      hatJ1_space  = 6,
      hatJ2_space  = 7
   };
   enum TestSpace
   {
      dE_space = 0,
      dH_space = 1,
      dJ1_space = 2,
      dJ2_space = 3
   };
   int test_order = order+delta_order;

   // Vector L2 space for E
   FiniteElementCollection *E_fec = new L2_FECollection(order-1,dim);
   ParFiniteElementSpace *E_fes = new ParFiniteElementSpace(&pmesh,E_fec,dim);

   // Vector L2 space for H
   FiniteElementCollection *H_fec = new L2_FECollection(order-1,dim);
   ParFiniteElementSpace *H_fes = new ParFiniteElementSpace(&pmesh,H_fec, dimc);

   // Vector H1 space for J1
   FiniteElementCollection *J1_fec = new H1_FECollection(order,dim);
   ParFiniteElementSpace *J1_fes = new ParFiniteElementSpace(&pmesh,J1_fec, dim);

   // Vector H1 space for J2
   FiniteElementCollection *J2_fec = new H1_FECollection(order,dim);
   ParFiniteElementSpace *J2_fes = new ParFiniteElementSpace(&pmesh,J2_fec, dim);

   FiniteElementCollection * hatE_fec = nullptr;
   FiniteElementCollection * hatH_fec = nullptr;
   FiniteElementCollection * dE_fec = nullptr;

   if (dim == 3)
   {
      // H^-1/2 (curl) space for Ê
      hatE_fec = new ND_Trace_FECollection(order,dim);
      // H^-1/2 (curl) space for Ĥ
      hatH_fec = new ND_Trace_FECollection(order,dim);
      dE_fec = new ND_FECollection(test_order, dim);
   }
   else
   {
      hatE_fec = new RT_Trace_FECollection(order-1,dim);
      hatH_fec = new H1_Trace_FECollection(order,dim);
      dE_fec = new H1_FECollection(test_order, dim);
   }

   ParFiniteElementSpace *hatE_fes = new ParFiniteElementSpace(&pmesh,hatE_fec);
   ParFiniteElementSpace *hatH_fes = new ParFiniteElementSpace(&pmesh,hatH_fec);

   // H^-1/2 space for Ĵ₁
   FiniteElementCollection * hatJ1_fec = new RT_Trace_FECollection(order-1,dim);
   ParFiniteElementSpace *hatJ1_fes = new ParFiniteElementSpace(&pmesh, hatJ1_fec,
                                                                dim);

   // H^-1/2 space for Ĵ₂
   FiniteElementCollection * hatJ2_fec = new RT_Trace_FECollection(order-1,dim);
   ParFiniteElementSpace *hatJ2_fes = new ParFiniteElementSpace(&pmesh, hatJ2_fec,
                                                                dim);

   FiniteElementCollection * dH_fec = new ND_FECollection(test_order, dim);
   FiniteElementCollection * dJ1_fec = new H1_FECollection(test_order, dim);
   FiniteElementCollection * dJ2_fec = new H1_FECollection(test_order, dim);

   Array<ParFiniteElementSpace * > trial_fes;
   Array<FiniteElementCollection * > test_fec;
   trial_fes.Append(E_fes);
   trial_fes.Append(H_fes);
   trial_fes.Append(J1_fes);
   trial_fes.Append(J2_fes);
   trial_fes.Append(hatE_fes);
   trial_fes.Append(hatH_fes);
   trial_fes.Append(hatJ1_fes);
   trial_fes.Append(hatJ2_fes);
   test_fec.Append(dE_fec);
   test_fec.Append(dH_fec);
   test_fec.Append(dJ1_fec);
   test_fec.Append(dJ2_fec);

   ParComplexDPGWeakForm * a = new ParComplexDPGWeakForm(trial_fes,test_fec);
   a->SetTestFECollVdim(TestSpace::dJ1_space,dim);
   a->SetTestFECollVdim(TestSpace::dJ2_space,dim);
   a->StoreMatrices();

   // // Bilinear form coefficients
   ConstantCoefficient one(1.0);
   ConstantCoefficient rec_omega(1.0/omega);
   ConstantCoefficient mu0omeg_cf(mu0*omega);
   ConstantCoefficient mu02omeg2_cf(mu0*mu0*omega*omega);
   ConstantCoefficient eps2omeg2_cf(eps0*eps0*omega*omega);
   ConstantCoefficient negepsomeg_cf(-eps0*omega);
   ConstantCoefficient epsomeg_cf(eps0*omega);
   ConstantCoefficient negmu0omeg_cf(-mu0*omega);
   ConstantCoefficient negomega(-omega);
   ConstantCoefficient c1_cf(c1);
   ConstantCoefficient negc1_cf(-c1);
   ConstantCoefficient c2_cf(c2);
   ConstantCoefficient negc2_cf(-c2);

   DenseMatrix Mone(dim);
   Mone = 0.0; Mone(0,0) = Mone(1,1) = 1.0;
   MatrixConstantCoefficient Mone_cf(Mone);

   // ε = S(r) (I - b⊗b) + P(r) b⊗b
   MatrixFunctionCoefficient eps_r(dim,epsilon_r);
   MatrixFunctionCoefficient eps_i(dim,epsilon_i);

   ScalarMatrixProductCoefficient eps0epsomeg_cf_r(epsomeg_cf,eps_r);
   ScalarMatrixProductCoefficient negeps0epsomeg_cf_r(negepsomeg_cf,eps_r);
   ScalarMatrixProductCoefficient eps0epsomeg_cf_i(epsomeg_cf,eps_i);
   ScalarMatrixProductCoefficient negeps0epsomeg_cf_i(negepsomeg_cf,eps_i);

   TransposeMatrixCoefficient eps0epsomeg_cf_r_t(eps0epsomeg_cf_r);
   TransposeMatrixCoefficient eps0epsomeg_cf_i_t(eps0epsomeg_cf_i);
   MatrixProductCoefficient MrMrt_cf(eps0epsomeg_cf_r, eps0epsomeg_cf_r_t);
   MatrixProductCoefficient MiMit_cf(eps0epsomeg_cf_i, eps0epsomeg_cf_i_t);
   MatrixProductCoefficient MiMrt_cf(eps0epsomeg_cf_i, eps0epsomeg_cf_r_t);
   MatrixProductCoefficient MrMit_cf(eps0epsomeg_cf_r, eps0epsomeg_cf_i_t);

   // (MᵣMᵣᵗ + MᵢMᵢᵗ) + i (MᵢMᵣᵗ - MᵣMᵢᵗ)
   MatrixSumCoefficient Mreal_cf(MrMrt_cf,MiMit_cf);
   MatrixSumCoefficient Mimag_cf(MiMrt_cf,MrMit_cf,1.0,-1.0);


   // for the 2D case
   DenseMatrix rot_mat(2);
   rot_mat(0,0) = 0.; rot_mat(0,1) = 1.;
   rot_mat(1,0) = -1.; rot_mat(1,1) = 0.;
   MatrixConstantCoefficient rot(rot_mat);
   ScalarMatrixProductCoefficient epsrot_cf(epsomeg_cf,rot);
   ScalarMatrixProductCoefficient negepsrot_cf(negepsomeg_cf,rot);

   // ω ϵ₀ ϵᵣ A
   MatrixProductCoefficient eps0epsomeg_cf_r_rot(eps0epsomeg_cf_r, rot);
   // ω ϵ₀ ϵᵢ A
   MatrixProductCoefficient eps0epsomeg_cf_i_rot(eps0epsomeg_cf_i, rot);
   // -ω ϵ₀ ϵᵣ A
   MatrixProductCoefficient negeps0epsomeg_cf_r_rot(negeps0epsomeg_cf_r, rot);
   // -ω ϵ₀ ϵᵢ A
   MatrixProductCoefficient negeps0epsomeg_cf_i_rot(negeps0epsomeg_cf_i, rot);


   // const IntegrationRule *irs[Geometry::NumGeom];
   // int order_quad = 2*order + 2;
   // for (int i = 0; i < Geometry::NumGeom; ++i)
   // {
   //    irs[i] = &(IntRules.Get(i, order_quad));
   // }
   const IntegrationRule &ir = IntRules.Get(pmesh.GetElementGeometry(0),
                                            2*test_order + 2);


   // (E,∇ × δE)
   a->AddTrialIntegrator(new TransposeIntegrator(new MixedCurlIntegrator(one)),
                         nullptr,TrialSpace::E_space, TestSpace::dE_space);
   //  -i ω ϵ₀ϵ (E,δH) =  - iωϵ₀ ( (ϵᵣ + i ϵᵢ) E, δH)
   //                  = (ω ϵ₀ ϵᵢ E, δH) + i (-ω ϵ₀ ϵᵣ E, δH)
   {
      VectorFEMassIntegrator * integ_r = new VectorFEMassIntegrator(eps0epsomeg_cf_i);
      VectorFEMassIntegrator * integ_i = new VectorFEMassIntegrator(
         negeps0epsomeg_cf_r);
      integ_r->SetIntRule(&ir);
      integ_i->SetIntRule(&ir);

      a->AddTrialIntegrator(new TransposeIntegrator(integ_r),
                            new TransposeIntegrator(integ_i),
                            TrialSpace::E_space,TestSpace::dH_space);
   }

   MatrixFunctionCoefficient bb_cf(dim,bcrossb);
   FunctionCoefficient pfunc_r_cf(pfunc_r);
   FunctionCoefficient pfunc_i_cf(pfunc_i);
   ScalarMatrixProductCoefficient pbb_r_cf(pfunc_r_cf,bb_cf);
   ScalarMatrixProductCoefficient pbb_i_cf(pfunc_i_cf,bb_cf);
   ScalarMatrixProductCoefficient negc1pbb_r_cf(negc1_cf,pbb_r_cf);
   ScalarMatrixProductCoefficient negc1pbb_i_cf(negc1_cf,pbb_i_cf);

   // -c₁ (P(r) E∥, δJ₁) = -c₁ (P(r) b⊗b E , δJ₁)
   {
      VectorMassIntegrator * integ_r = new VectorMassIntegrator(negc1pbb_r_cf);
      VectorMassIntegrator * integ_i = new VectorMassIntegrator(negc1pbb_i_cf);
      integ_r->SetIntRule(&ir);
      integ_i->SetIntRule(&ir);

      a->AddTrialIntegrator(integ_r, integ_i,
                            TrialSpace::E_space, TestSpace::dJ1_space);
   }

   ScalarMatrixProductCoefficient c2pbb_r_cf(c2_cf,pbb_r_cf);
   ScalarMatrixProductCoefficient c2pbb_i_cf(c2_cf,pbb_i_cf);

   // c₂  (P(r) E∥, δJ₂) = c₂ (P(r) b⊗b E , δJ₂)
   {
      VectorMassIntegrator * integ_r = new VectorMassIntegrator(c2pbb_r_cf);
      VectorMassIntegrator * integ_i = new VectorMassIntegrator(c2pbb_i_cf);
      integ_r->SetIntRule(&ir);
      integ_i->SetIntRule(&ir);

      a->AddTrialIntegrator(integ_r, integ_i,
                            TrialSpace::E_space, TestSpace::dJ2_space);
   }

   if (dim == 3)
   {
      // i ω μ (H, δE)
      a->AddTrialIntegrator(nullptr,new TransposeIntegrator(
                               new VectorFEMassIntegrator(mu0omeg_cf)),
                            TrialSpace::H_space, TestSpace::dE_space);
      // < Ê, δE × n >
      a->AddTrialIntegrator(new TangentTraceIntegrator,nullptr,
                            TrialSpace::hatE_space, TestSpace::dE_space);
      // test integrators
      // (∇×δE ,∇× δE)
      a->AddTestIntegrator(new CurlCurlIntegrator(one),nullptr,
                           TestSpace::dE_space,TestSpace::dE_space);
      // (δE,δE)
      a->AddTestIntegrator(new VectorFEMassIntegrator(one),nullptr,
                           TestSpace::dE_space,TestSpace::dE_space);
      // μ^2 ω^2 (δE ,δE)
      a->AddTestIntegrator(new VectorFEMassIntegrator(mu02omeg2_cf),nullptr,
                           TestSpace::dE_space, TestSpace::dE_space);
      // -i ω μ (δE ,∇ × δH) = i (F, -ω μ ∇ × δ G)
      a->AddTestIntegrator(nullptr,new MixedVectorWeakCurlIntegrator(negmu0omeg_cf),
                           TestSpace::dE_space, TestSpace::dH_space);
      // -i ω ϵ₀(ϵᵣ + i ϵᵢ ) (∇ × δE, δH)
      // ω ϵ₀ ϵᵢ (∇ × δE, δH) + i (-ω ϵ₀ ϵᵣ (∇ × δE, δH))
      a->AddTestIntegrator(new MixedVectorCurlIntegrator(eps0epsomeg_cf_i),
                           new MixedVectorCurlIntegrator(negeps0epsomeg_cf_r),
                           TestSpace::dE_space, TestSpace::dH_space);
      // i ω μ (∇ × δH ,δE)
      a->AddTestIntegrator(nullptr,new MixedVectorCurlIntegrator(mu0omeg_cf),
                           TestSpace::dH_space, TestSpace::dE_space);
      // i ω ϵ₀(ϵᵣ + i ϵᵢ ) (δH, ∇ × δE )
      //  (-ω ϵ₀ ϵᵢ δH, ∇ × δE) + i (ω ϵ₀ ϵᵣ δH, ∇ × δE)
      a->AddTestIntegrator(new MixedVectorWeakCurlIntegrator(negeps0epsomeg_cf_i),
                           new MixedVectorWeakCurlIntegrator(eps0epsomeg_cf_r),
                           TestSpace::dH_space, TestSpace::dE_space);
      // (ωϵ₀ϵ)(ωϵ₀ϵ)^*  (δH, δH)
      // (MᵣMᵣᵗ + MᵢMᵢᵗ) + i (MᵢMᵣᵗ - MᵣMᵢᵗ)
      a->AddTestIntegrator(new VectorFEMassIntegrator(Mreal_cf),
                           new VectorFEMassIntegrator(Mimag_cf),
                           TestSpace::dH_space, TestSpace::dH_space);
   }
   else
   {
      // i ω μ₀ (H, δE)
      a->AddTrialIntegrator(nullptr,new MixedScalarMassIntegrator(mu0omeg_cf),
                            TrialSpace::H_space, TestSpace::dE_space);
      // < n×Ê,δE>
      a->AddTrialIntegrator(new TraceIntegrator,nullptr,
                            TrialSpace::hatE_space, TestSpace::dE_space);
      // test integrators
      // (∇δE,∇δE)
      a->AddTestIntegrator(new DiffusionIntegrator(one),nullptr,
                           TestSpace::dE_space, TestSpace::dE_space);
      // (δE,δE)
      a->AddTestIntegrator(new MassIntegrator(one),nullptr,
                           TestSpace::dE_space, TestSpace::dE_space);
      // μ^2 ω^2 (δE,δE)
      a->AddTestIntegrator(new MassIntegrator(mu02omeg2_cf),nullptr,
                           TestSpace::dE_space, TestSpace::dE_space);
      // -i ω μ (δE,∇ × δH) = i (δE, -ω μ ∇ × δ H)
      a->AddTestIntegrator(nullptr,
                           new TransposeIntegrator(new MixedCurlIntegrator(negmu0omeg_cf)),
                           TestSpace::dE_space, TestSpace::dH_space);
      //  -i ω ϵ₀ϵ(∇ × δE, δH) = -i (ωϵ₀(ϵᵣ+iϵᵢ) A ∇ δE,δE), A = [0 1; -1 0]
      //  =  (ω ϵ₀ ϵᵢ A ∇ δE,δE) + i (-ω ϵ₀ ϵᵣ A ∇ δE,δE)
      {
         MixedVectorGradientIntegrator * integ_r
            = new MixedVectorGradientIntegrator(eps0epsomeg_cf_i_rot);
         MixedVectorGradientIntegrator * integ_i
            = new MixedVectorGradientIntegrator(negeps0epsomeg_cf_r_rot);
         integ_r->SetIntRule(&ir);
         integ_i->SetIntRule(&ir);
         a->AddTestIntegrator(integ_r, integ_i,
                              TestSpace::dE_space, TestSpace::dH_space);
      }
      // i ω μ₀ (∇ × δH ,δE) = i (ω μ₀ ∇ × δH, δE )
      a->AddTestIntegrator(nullptr,new MixedCurlIntegrator(mu0omeg_cf),
                           TestSpace::dH_space, TestSpace::dE_space);

      // i ω ϵ₀ϵ (δH, ∇ × δE ) = i (ω ϵ₀(ϵᵣ - i ϵᵢ) δH, A ∇ δE)
      //  = (δH, ω ϵ₀ ϵᵢ A ∇ δE) + i (δH, ω ϵ₀ ϵᵣ A ∇ δE)
      {
         MixedVectorGradientIntegrator * integ_r =
            new MixedVectorGradientIntegrator(eps0epsomeg_cf_i_rot);
         MixedVectorGradientIntegrator * integ_i =
            new MixedVectorGradientIntegrator(eps0epsomeg_cf_r_rot);
         integ_r->SetIntRule(&ir);
         integ_i->SetIntRule(&ir);

         a->AddTestIntegrator(new TransposeIntegrator(integ_r),
                              new TransposeIntegrator(integ_i),
                              TestSpace::dH_space, TestSpace::dE_space);
      }
      // (ωϵ₀ϵ)(ωϵ₀ϵ)^*  (δH, δH)
      // (MᵣMᵣᵗ + MᵢMᵢᵗ) + i (MᵢMᵣᵗ - MᵣMᵢᵗ)
      {
         VectorFEMassIntegrator * integ_r =
            new VectorFEMassIntegrator(Mreal_cf);
         VectorFEMassIntegrator * integ_i =
            new VectorFEMassIntegrator(Mimag_cf);
         integ_r->SetIntRule(&ir);
         integ_i->SetIntRule(&ir);

         a->AddTestIntegrator(integ_r, integ_i,
                              TestSpace::dH_space, TestSpace::dH_space);
      }
   }
   // (H,∇ × δH)
   a->AddTrialIntegrator(new TransposeIntegrator(new MixedCurlIntegrator(one)),
                         nullptr,TrialSpace::H_space, TestSpace::dH_space);
   //  -cj ω ϵ₀ (J₁,δH)
   ProductCoefficient scaled_negomegaeps0(cj*eps0,negomega);
   {
      VectorFEMassIntegrator * integ_r =
         new VectorFEMassIntegrator(scaled_negomegaeps0);
      integ_r->SetIntRule(&ir);

      a->AddTrialIntegrator(new TransposeIntegrator(integ_r),nullptr,
                            TrialSpace::J1_space, TestSpace::dH_space);
   }
   //  -cj ω ϵ₀ (J₂,δH)
   {
      VectorFEMassIntegrator * integ_r =
         new VectorFEMassIntegrator(scaled_negomegaeps0);
      integ_r->SetIntRule(&ir);

      a->AddTrialIntegrator(new TransposeIntegrator(integ_r),nullptr,
                            TrialSpace::J2_space, TestSpace::dH_space);
   }
   // scaled diff coefficients
   ConstantCoefficient scaled_diff_cf(cfactor);
   VectorFunctionCoefficient b_cf(dim,bfunc);
   ScalarVectorProductCoefficient scaled_b_cf(sqrt(cfactor),b_cf);
   // ( (b⋅∇)J₁,(b⋅∇)δJ₁ )
   {
      DirectionalDiffusionIntegrator * integ_r =
         new DirectionalDiffusionIntegrator(scaled_b_cf);
      integ_r->SetIntRule(&ir);

      a->AddTrialIntegrator(integ_r,nullptr,
                            TrialSpace::J1_space,TestSpace::dJ1_space);
   }

   // ( (b⋅∇)J₂,(b⋅∇)δJ₂ )
   {
      DirectionalDiffusionIntegrator * integ_r =
         new DirectionalDiffusionIntegrator(scaled_b_cf);
      integ_r->SetIntRule(&ir);

      a->AddTrialIntegrator(integ_r,nullptr,
                            TrialSpace::J2_space,TestSpace::dJ2_space);
   }

   // c₁ (J₁,δJ₁)
   a->AddTrialIntegrator(new VectorMassIntegrator(c1_cf),nullptr,
                         TrialSpace::J1_space,TestSpace::dJ1_space);

   // c₂ (J₂,δJ₂)
   a->AddTrialIntegrator(new VectorMassIntegrator(c2_cf),nullptr,
                         TrialSpace::J2_space,TestSpace::dJ2_space);

   // < Ĥ, δH × n >
   a->AddTrialIntegrator(new TangentTraceIntegrator,nullptr,
                         TrialSpace::hatH_space, TestSpace::dH_space);

   // <Ĵ₁, δJ₁>
   a->AddTrialIntegrator(new VectorTraceIntegrator,nullptr,
                         TrialSpace::hatJ1_space,TestSpace::dJ1_space);

   // <Ĵ₂, δJ₂>
   a->AddTrialIntegrator(new VectorTraceIntegrator,nullptr,
                         TrialSpace::hatJ2_space,TestSpace::dJ2_space);

   // test integrators
   // (∇×δH ,∇×δH)
   a->AddTestIntegrator(new CurlCurlIntegrator(one),nullptr,
                        TestSpace::dH_space,TestSpace::dH_space);
   // (δH,δH)
   a->AddTestIntegrator(new VectorFEMassIntegrator(one),nullptr,
                        TestSpace::dH_space,TestSpace::dH_space);

   // ----------------------------------------------------------------

   // ----------------------------------------------------------------
   // (∇δJ₁,∇δJ₁)
   a->AddTestIntegrator(new VectorDiffusionIntegrator(one),nullptr,
                        TestSpace::dJ1_space,TestSpace::dJ1_space);
   // (δJ₁,δJ₁)
   a->AddTestIntegrator(new VectorMassIntegrator(one),nullptr,
                        TestSpace::dJ1_space,
                        TestSpace::dJ1_space);
   // (∇δJ₂,∇δJ₂)
   a->AddTestIntegrator(new VectorDiffusionIntegrator(one),nullptr,
                        TestSpace::dJ2_space,TestSpace::dJ2_space);
   // (δJ₂,δJ₂)
   a->AddTestIntegrator(new VectorMassIntegrator(one),nullptr,
                        TestSpace::dJ2_space,
                        TestSpace::dJ2_space);

   VectorFunctionCoefficient f_rhs_dH_r(dim,rhs_dH_func_r);
   VectorFunctionCoefficient f_rhs_dH_i(dim,rhs_dH_func_i);
   VectorFunctionCoefficient f_rhs_dJ1_r(dim,rhs_dJ1_func_r);
   VectorFunctionCoefficient f_rhs_dJ1_i(dim,rhs_dJ1_func_i);
   VectorFunctionCoefficient f_rhs_dJ2_r(dim,rhs_dJ2_func_r);
   VectorFunctionCoefficient f_rhs_dJ2_i(dim,rhs_dJ2_func_i);

   if (mms)
   {
      a->AddDomainLFIntegrator(new VectorFEDomainLFIntegrator(f_rhs_dH_r),
                               new VectorFEDomainLFIntegrator(f_rhs_dH_i),
                               TestSpace::dH_space);
      a->AddDomainLFIntegrator(new VectorDomainLFIntegrator(f_rhs_dJ1_r),
                               new VectorDomainLFIntegrator(f_rhs_dJ1_i),
                               TestSpace::dJ1_space);
      a->AddDomainLFIntegrator(new VectorDomainLFIntegrator(f_rhs_dJ2_r),
                               new VectorDomainLFIntegrator(f_rhs_dJ2_i),
                               TestSpace::dJ2_space);
   }

   socketstream E_out_r, E_out_i, Eex_out_r, Eex_out_i;
   socketstream H_out_r, H_out_i, Hex_out_r, Hex_out_i;
   socketstream J1_out_r, J1_out_i, J1ex_out_r, J1ex_out_i;
   socketstream J2_out_r, J2_out_i, J2ex_out_r, J2ex_out_i;

   if (myid == 0)
   {
      std::cout << "\n  Ref |"
                << "    Dofs    |"
                << "    ω    |" ;
      std::cout  << "  L2 Error  |"
                 << "  Rate  |" ;
      std::cout << "  Residual  |"
                << "  Rate  |"
                << " PCG it |" << endl;
      std::cout << std::string(82,'-') << endl;
   }

   ParGridFunction E_r(E_fes), E_i(E_fes);
   ParGridFunction H_r(H_fes), H_i(H_fes);
   ParGridFunction J1_r(J1_fes), J1_i(J1_fes);
   ParGridFunction J2_r(J2_fes), J2_i(J2_fes);
   ParGridFunction Eex_r(E_fes), Eex_i(E_fes);
   ParGridFunction Hex_r(H_fes), Hex_i(H_fes);
   ParGridFunction J1ex_r(J1_fes), J1ex_i(J1_fes);
   ParGridFunction J2ex_r(J2_fes), J2ex_i(J2_fes);

   double res0 = 0.;
   double err0 = 0.;
   int dof0;

   if (static_cond) { a->EnableStaticCondensation(); }
   for (int it = 0; it<=pr; it++)
   {
      a->Assemble();

      Array<int> ess_tdof_list;
      Array<int> ess_tdof_listE;
      Array<int> ess_tdof_listJ1;
      Array<int> ess_tdof_listJ2;
      Array<int> ess_bdr;
      Array<int> one_r_bdr;
      Array<int> one_i_bdr;
      Array<int> negone_r_bdr;
      Array<int> negone_i_bdr;

      if (pmesh.bdr_attributes.Size())
      {
         ess_bdr.SetSize(pmesh.bdr_attributes.Max());
         ess_bdr = 1;
         hatE_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_listE);
         J1_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_listJ1);
         J2_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_listJ2);
      }

      // shift the ess_tdofs
      for (int j = 0; j < ess_tdof_listE.Size(); j++)
      {
         ess_tdof_listE[j] += E_fes->GetTrueVSize()  + H_fes->GetTrueVSize() +
                              J1_fes->GetTrueVSize() + J2_fes->GetTrueVSize();
      }
      for (int j = 0; j < ess_tdof_listJ1.Size(); j++)
      {
         ess_tdof_listJ1[j] += E_fes->GetTrueVSize() + H_fes->GetTrueVSize();
         ess_tdof_listJ2[j] += E_fes->GetTrueVSize() + H_fes->GetTrueVSize() +
                               J1_fes->GetTrueVSize();
      }
      ess_tdof_list.Append(ess_tdof_listE);
      ess_tdof_list.Append(ess_tdof_listJ1);
      ess_tdof_list.Append(ess_tdof_listJ2);

      Array<int> offsets(9);
      offsets[0] = 0;
      offsets[1] = E_fes->GetVSize();
      offsets[2] = H_fes->GetVSize();
      offsets[3] = J1_fes->GetVSize();
      offsets[4] = J2_fes->GetVSize();
      offsets[5] = hatE_fes->GetVSize();
      offsets[6] = hatH_fes->GetVSize();
      offsets[7] = hatJ1_fes->GetVSize();
      offsets[8] = hatJ2_fes->GetVSize();
      offsets.PartialSum();

      Vector x(2*offsets.Last());
      x = 0.;

      ParGridFunction hatE_gf_r(hatE_fes, x, offsets[4]);
      ParGridFunction hatE_gf_i(hatE_fes, x, offsets.Last() + offsets[4]);
      VectorFunctionCoefficient hatEex_r(dim,hatE_exact_r);
      VectorFunctionCoefficient hatEex_i(dim,hatE_exact_i);
      VectorFunctionCoefficient Ecf_r(dim,E_exact_r);
      VectorFunctionCoefficient Ecf_i(dim,E_exact_i);
      VectorFunctionCoefficient Hcf_r(dimc,H_exact_r);
      VectorFunctionCoefficient Hcf_i(dimc,H_exact_i);
      VectorFunctionCoefficient J1cf_r(dim,J1_exact_r);
      VectorFunctionCoefficient J1cf_i(dim,J1_exact_i);
      VectorFunctionCoefficient J2cf_r(dim,J2_exact_r);
      VectorFunctionCoefficient J2cf_i(dim,J2_exact_i);


      Vector rot_one_x(dim); rot_one_x = 0.0; rot_one_x(1) = -1.0;
      Vector rot_negone_x(dim); rot_negone_x = 0.0; rot_negone_x(1) = 1.0;
      VectorConstantCoefficient rot_one_x_cf(rot_one_x);
      VectorConstantCoefficient rot_negone_x_cf(rot_negone_x);
      ParGridFunction J1_gf_r(J1_fes, x, offsets[2]);
      ParGridFunction J1_gf_i(J1_fes, x, offsets.Last() + offsets[2]);
      ParGridFunction J2_gf_r(J2_fes, x, offsets[3]);
      ParGridFunction J2_gf_i(J2_fes, x, offsets.Last() + offsets[3]);
      J1_gf_r = 0.0; J1_gf_i = 0.0;
      J2_gf_r = 0.0; J2_gf_i = 0.0;
      if (mms)
      {
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
         J1_gf_r.ProjectBdrCoefficient(J1cf_r,ess_bdr);
         J1_gf_i.ProjectBdrCoefficient(J1cf_i,ess_bdr);
         J2_gf_r.ProjectBdrCoefficient(J2cf_r,ess_bdr);
         J2_gf_i.ProjectBdrCoefficient(J2cf_i,ess_bdr);
      }
      else
      {
         one_r_bdr.SetSize(pmesh.bdr_attributes.Max());
         one_i_bdr.SetSize(pmesh.bdr_attributes.Max());
         negone_r_bdr.SetSize(pmesh.bdr_attributes.Max());
         negone_i_bdr.SetSize(pmesh.bdr_attributes.Max());

         one_r_bdr = 0;  one_i_bdr = 0;
         negone_r_bdr = 0;  negone_i_bdr = 0;
         // attr = 30,2 (real)
         one_r_bdr[30-1] = 1;  one_r_bdr[2-1] = 1;
         // attr = 26,6 (imag)
         one_i_bdr[26-1] = 1;  one_i_bdr[6-1] = 1;
         // attr = 22,10 (real)
         negone_r_bdr[22-1] = 1; negone_r_bdr[10-1] = 1;
         // attr = 18,14 (imag)
         negone_i_bdr[18-1] = 1; negone_i_bdr[14-1] = 1;

         hatE_gf_r.ProjectBdrCoefficientNormal(rot_one_x_cf, one_r_bdr);
         hatE_gf_r.ProjectBdrCoefficientNormal(rot_negone_x_cf, negone_r_bdr);
         hatE_gf_i.ProjectBdrCoefficientNormal(rot_one_x_cf, one_i_bdr);
         hatE_gf_i.ProjectBdrCoefficientNormal(rot_negone_x_cf, negone_i_bdr);

      }


      OperatorPtr Ah;
      Vector X,B;
      a->FormLinearSystem(ess_tdof_list,x,Ah, X,B);

      ComplexOperator * Ahc = Ah.As<ComplexOperator>();

      BlockOperator * BlockA_r = dynamic_cast<BlockOperator *>(&Ahc->real());
      BlockOperator * BlockA_i = dynamic_cast<BlockOperator *>(&Ahc->imag());

      int num_blocks = BlockA_r->NumRowBlocks();

      Array2D<const HypreParMatrix*> A_r_matrices(num_blocks, num_blocks);
      Array2D<const HypreParMatrix*> A_i_matrices(num_blocks, num_blocks);
      for (int i = 0; i < num_blocks; i++)
      {
         for (int j = 0; j < num_blocks; j++)
         {
            A_r_matrices(i,j) = dynamic_cast<HypreParMatrix*>(&BlockA_r->GetBlock(i,j));
            A_i_matrices(i,j) = dynamic_cast<HypreParMatrix*>(&BlockA_i->GetBlock(i,j));
         }
      }

      HypreParMatrix * Ahr = HypreParMatrixFromBlocks(A_r_matrices);
      HypreParMatrix * Ahi = HypreParMatrixFromBlocks(A_i_matrices);

      ComplexHypreParMatrix * Ahc_hypre =
         new ComplexHypreParMatrix(Ahr, Ahi,false, false);

      if (Mpi::Root())
      {
         mfem::out << "Assembly finished successfully." << endl;
      }

#ifdef MFEM_USE_MUMPS
      if (mumps_solver)
      {
         HypreParMatrix *A = Ahc_hypre->GetSystemMatrix();
         auto solver = new MUMPSSolver(MPI_COMM_WORLD);
         solver->SetMatrixSymType(MUMPSSolver::MatType::UNSYMMETRIC);
         solver->SetPrintLevel(1);
         solver->SetOperator(*A);
         solver->Mult(B,X);
         delete A;
         delete solver;
      }
#else
      if (mumps_solver)
      {
         MFEM_WARNING("MFEM compiled without mumps. Switching to an iterative solver");
      }
      mumps_solver = false;
#endif
      int num_iter = -1;

      if (!mumps_solver)
      {
         Array<int> tdof_offsets(2*num_blocks+1);

         tdof_offsets[0] = 0;
         int skip = (static_cond) ? 0 : 2;
         for (int i=0; i<num_blocks; i++)
         {
            tdof_offsets[i+1] = BlockA_r->GetBlock(i,i).Height();
            tdof_offsets[num_blocks+i+1] = BlockA_r->GetBlock(i,i).Height();
         }
         tdof_offsets.PartialSum();

         X = 0.;
         BlockDiagonalPreconditioner M(tdof_offsets);
         // BlockTriangularSymmetricPreconditioner M(tdof_offsets);
         // for (int i = 0; i < num_blocks; i++)
         // {
         //    for (int j = 0; j < num_blocks; j++)
         //    {
         //       if (i != j)
         //       {
         //          M.SetBlock(i,j, &BlockA_r->GetBlock(i,j));
         //          M.SetBlock(i,j+num_blocks,&BlockA_i->GetBlock(i,j), -1.0);
         //          M.SetBlock(i+num_blocks,j+num_blocks,&BlockA_r->GetBlock(i,j));
         //          M.SetBlock(i+num_blocks,j,&BlockA_i->GetBlock(i,j));
         //       }
         //    }
         // }
         if (!static_cond)
         {
            HypreBoomerAMG * solver_E = new HypreBoomerAMG((HypreParMatrix &)
                                                           BlockA_r->GetBlock(0,0));
            solver_E->SetPrintLevel(0);
            solver_E->SetSystemsOptions(dim);
            HypreBoomerAMG * solver_H = new HypreBoomerAMG((HypreParMatrix &)
                                                           BlockA_r->GetBlock(1,1));
            solver_H->SetPrintLevel(0);
            solver_H->SetSystemsOptions(dim);
            M.SetDiagonalBlock(0,solver_E);
            M.SetDiagonalBlock(1,solver_H);
            M.SetDiagonalBlock(num_blocks,solver_E);
            M.SetDiagonalBlock(num_blocks+1,solver_H);
         }

         HypreBoomerAMG * solver_J1 = new HypreBoomerAMG((HypreParMatrix &)
                                                         BlockA_r->GetBlock(skip,skip));
         solver_J1->SetPrintLevel(0);
         solver_J1->SetSystemsOptions(dim);
         M.SetDiagonalBlock(skip,solver_J1);
         M.SetDiagonalBlock(skip+num_blocks,solver_J1);

         HypreBoomerAMG * solver_J2 = new HypreBoomerAMG((HypreParMatrix &)
                                                         BlockA_r->GetBlock(skip+1,skip+1));
         solver_J2->SetPrintLevel(0);
         solver_J2->SetSystemsOptions(dim);
         M.SetDiagonalBlock(skip+1,solver_J2);
         M.SetDiagonalBlock(skip+1+num_blocks,solver_J2);

         HypreAMS * solver_hatE = new HypreAMS((HypreParMatrix &)BlockA_r->GetBlock(
                                                  skip+2,skip+2),
                                               hatE_fes);
         solver_hatE->SetPrintLevel(0);
         HypreSolver * solver_hatH = nullptr;
         if (dim == 2)
         {
            solver_hatH = new HypreBoomerAMG((HypreParMatrix &)BlockA_r->GetBlock(skip+3,
                                                                                  skip+3));
            dynamic_cast<HypreBoomerAMG*>(solver_hatH)->SetPrintLevel(0);
         }
         else
         {
            solver_hatH = new HypreAMS((HypreParMatrix &)BlockA_r->GetBlock(
                                          skip+4,skip+4),
                                       hatH_fes);
            dynamic_cast<HypreAMS*>(solver_hatH)->SetPrintLevel(0);
         }
         HypreBoomerAMG * solver_hatJ1 = new HypreBoomerAMG((HypreParMatrix &)
                                                            BlockA_r->GetBlock(skip+4,skip+4));
         solver_hatJ1->SetPrintLevel(0);
         solver_hatJ1->SetSystemsOptions(dim);
         solver_hatJ1->SetRelaxType(88);

         HypreBoomerAMG * solver_hatJ2 = new HypreBoomerAMG((HypreParMatrix &)
                                                            BlockA_r->GetBlock(skip+5,skip+5));
         solver_hatJ2->SetPrintLevel(0);
         solver_hatJ2->SetSystemsOptions(dim);
         solver_hatJ2->SetRelaxType(88);

         M.SetDiagonalBlock(skip+2,solver_hatE);
         M.SetDiagonalBlock(skip+3,solver_hatH);
         M.SetDiagonalBlock(skip+4,solver_hatJ1);
         M.SetDiagonalBlock(skip+5,solver_hatJ2);
         M.SetDiagonalBlock(num_blocks+skip+2,solver_hatE);
         M.SetDiagonalBlock(num_blocks+skip+3,solver_hatH);
         M.SetDiagonalBlock(num_blocks+skip+4,solver_hatJ1);
         M.SetDiagonalBlock(num_blocks+skip+5,solver_hatJ2);

         CGSolver cg(MPI_COMM_WORLD);
         cg.SetRelTol(1e-10);
         cg.SetMaxIter(1000);
         cg.SetPrintLevel(1);
         cg.SetPreconditioner(M);
         cg.SetOperator(*Ahc_hypre);
         cg.Mult(B, X);
         num_iter = cg.GetNumIterations();

         for (int i = 0; i<num_blocks; i++)
         {
            delete &M.GetDiagonalBlock(i);
         }
      }

      a->RecoverFEMSolution(X,x);
      Vector & residuals = a->ComputeResidual(x);
      double residual = residuals.Norml2();
      double maxresidual = residuals.Max();
      double globalresidual = residual * residual;
      MPI_Allreduce(MPI_IN_PLACE,&maxresidual,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE,&globalresidual,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

      globalresidual = sqrt(globalresidual);

      E_r.MakeRef(E_fes,x, 0);
      E_i.MakeRef(E_fes,x, offsets.Last());

      H_r.MakeRef(H_fes,x, offsets[1]);
      H_i.MakeRef(H_fes,x, offsets.Last()+offsets[1]);

      J1_r.MakeRef(J1_fes,x, offsets[2]);
      J1_i.MakeRef(J1_fes,x, offsets.Last()+offsets[2]);

      J2_r.MakeRef(J2_fes,x, offsets[3]);
      J2_i.MakeRef(J2_fes,x, offsets.Last()+offsets[3]);

      int dofs = 0;
      for (int i = 0; i<trial_fes.Size(); i++)
      {
         dofs += trial_fes[i]->GlobalTrueVSize();
      }

      double L2Error = 0.0;
      double rate_err = 0.0;

      // compute errors
      // E error
      double E_err_r = E_r.ComputeL2Error(Ecf_r);
      double E_err_i = E_i.ComputeL2Error(Ecf_i);
      double H_err_r = H_r.ComputeL2Error(Hcf_r);
      double H_err_i = H_i.ComputeL2Error(Hcf_i);
      double J1_err_r = J1_r.ComputeL2Error(J1cf_r);
      double J1_err_i = J1_i.ComputeL2Error(J1cf_i);
      double J2_err_r = J2_r.ComputeL2Error(J2cf_r);
      double J2_err_i = J2_i.ComputeL2Error(J2cf_i);

      Eex_r.ProjectCoefficient(Ecf_r);
      Eex_i.ProjectCoefficient(Ecf_i);
      Hex_r.ProjectCoefficient(Hcf_r);
      Hex_i.ProjectCoefficient(Hcf_i);
      J1ex_r.ProjectCoefficient(J1cf_r);
      J1ex_i.ProjectCoefficient(J1cf_i);
      J2ex_r.ProjectCoefficient(J2cf_r);
      J2ex_i.ProjectCoefficient(J2cf_i);

      L2Error = sqrt(E_err_r*E_err_r + E_err_i*E_err_i
                     + H_err_r*H_err_r + H_err_i*H_err_i
                     + J1_err_r*J1_err_r + J1_err_i*J1_err_i
                     + J2_err_r*J2_err_r + J2_err_i*J2_err_i);
      rate_err = (it) ? dim*log(err0/L2Error)/log((double)dof0/dofs) : 0.0;
      err0 = L2Error;

      double rate_res = (it) ? dim*log(res0/globalresidual)/log((
                                                                   double)dof0/dofs) : 0.0;

      res0 = globalresidual;
      dof0 = dofs;

      if (myid == 0)
      {
         std::ios oldState(nullptr);
         oldState.copyfmt(std::cout);
         std::cout << std::right << std::setw(5) << it << " | "
                   << std::setw(10) <<  dof0 << " | "
                   << std::setprecision(1) << std::fixed
                   << std::setw(4) <<  2.0*rnum << " π  | "
                   << std::setprecision(3);
         std::cout << std::setw(10) << std::scientific <<  err0 << " | "
                   << std::setprecision(2)
                   << std::setw(6) << std::fixed << rate_err << " | " ;
         std::cout << std::setprecision(3)
                   << std::setw(10) << std::scientific <<  res0 << " | "
                   << std::setprecision(2)
                   << std::setw(6) << std::fixed << rate_res << " | "
                   << std::setw(6) << std::fixed << num_iter << " | "
                   << std::endl;
         std::cout.copyfmt(oldState);
      }

      if (visualization)
      {
         // const char * keys = (it == 0 && dim == 2) ? "jRcml\n" : nullptr;
         const char * keys = nullptr;
         char vishost[] = "localhost";
         int  visport   = 19916;
         VisualizeField(E_out_r,vishost, visport, E_r,
                        "Numerical Electric field (real part)", 0, 0, 500, 500, keys);
         // VisualizeField(Eex_out_r,vishost, visport, Eex_r,
         //                "Exact Electric field (real part)", 501, 0, 500, 500, keys);
         // VisualizeField(H_out_r,vishost, visport, H_r,
         //                "Numerical Magnetic field (real part)", 0, 0, 500, 500, keys);
         // VisualizeField(Hex_out_r,vishost, visport, Hex_r,
         //                "Exact Magnetic field (real part)", 501, 0, 500, 500, keys);
         // VisualizeField(J1_out_r,vishost, visport, J1_r,
         //                "Numerical J1 field (real part)", 0, 0, 500, 500, keys);
         // VisualizeField(J1ex_out_r,vishost, visport, J1ex_r,
         //                "Exact J1 field (real part)", 501, 0, 500, 500, keys);
         // VisualizeField(J2_out_r,vishost, visport, J2_r,
         //                "Numerical J2 field (real part)", 0, 0, 500, 500, keys);
         // VisualizeField(J2ex_out_r,vishost, visport, J2ex_r,
         // "Exact J2 field (real part)", 501, 0, 500, 500, keys);

         // VisualizeField(E_out_i,vishost, visport, E_i,
         //                "Numerical Electric field (imag part)", 0, 0, 500, 500, keys);
         // VisualizeField(Eex_out_i,vishost, visport, Eex_i,
         //                "Exact Electric field (imag part)", 501, 0, 500, 500, keys);
         // VisualizeField(H_out_i,vishost, visport, H_i,
         //                "Numerical Magnetic field (imag part)", 0, 0, 500, 500, keys);
         // VisualizeField(Hex_out_i,vishost, visport, Hex_i,
         //                "Exact Magnetic field (imag part)", 501, 0, 500, 500, keys);
         // VisualizeField(J_out_i,vishost, visport, J_i,
         //                "Numerical J field (imag part)", 0, 0, 500, 500, keys);
         // VisualizeField(Jex_out_i,vishost, visport, Jex_i,
         //                "Exact J field (imag part)", 501, 0, 500, 500, keys);
      }

      if (it == pr)
      {
         break;
      }

      pmesh.UniformRefinement();
      for (int i =0; i<trial_fes.Size(); i++)
      {
         trial_fes[i]->Update(false);
      }
      a->Update();
      Eex_r.Update();
      Eex_i.Update();
      Hex_r.Update();
      Hex_i.Update();
      J1ex_r.Update();
      J1ex_i.Update();
      J2ex_r.Update();
      J2ex_i.Update();
   }

   delete a;
   // delete F_fec;
   // delete G_fec;
   // delete hatH_fes;
   // delete hatH_fec;
   // delete hatE_fes;
   // delete hatE_fec;
   // delete H_fec;
   // delete E_fec;
   // delete H_fes;
   // delete E_fes;

   return 0;
}

void maxwell_solution(const Vector & X, std::vector<complex<double>> &E)
{
   complex<double> zi = complex<double>(0., 1.);
   E.resize(dim);
   for (int i = 0; i < dim; ++i)
   {
      E[i] = 0.0;
   }
   E[0] = exp(zi * omega * (X.Sum()));
}

void maxwell_solution_curl(const Vector & X,
                           std::vector<complex<double>> &curlE)
{
   complex<double> zi = complex<double>(0., 1.);
   curlE.resize(dimc);
   for (int i = 0; i < dimc; ++i)
   {
      curlE[i] = 0.0;
   }

   std::complex<double> pw = exp(zi * omega * (X.Sum()));
   if (dim == 3)
   {
      curlE[0] = 0.0;
      curlE[1] = zi * omega * pw;
      curlE[2] = -zi * omega * pw;
   }
   else
   {
      curlE[0] = -zi * omega * pw;
   }
}

void maxwell_solution_curlcurl(const Vector & X,
                               std::vector<complex<double>> &curlcurlE)
{
   complex<double> zi = complex<double>(0., 1.);
   curlcurlE.resize(dim);
   for (int i = 0; i < dim; ++i)
   {
      curlcurlE[i] = 0.0;;
   }
   std::complex<double> pw = exp(zi * omega * (X.Sum()));
   if (dim == 3)
   {
      curlcurlE[0] = 2.0 * omega * omega * pw;
      curlcurlE[1] = - omega * omega * pw;
      curlcurlE[2] = - omega * omega * pw;
   }
   else
   {
      curlcurlE[0] = omega * omega * pw;
      curlcurlE[1] = -omega * omega * pw;
   }
}

void J1_solution(const Vector &x,std::vector<complex<double>> &J)
{
   complex<double> zi = complex<double>(0., 1.);
   J.resize(dim);
   for (int i = 0; i < dim; ++i)
   {
      J[i] = 0.0;
   }
   J[0] = x[0] * x[0];
}

void J1_solution_grad(const Vector &x,
                      std::vector<std::vector<complex<double>>> &gradJ)
{
   gradJ.resize(dim);
   for (int i = 0; i < dim; ++i)
   {
      gradJ[i].resize(dim);
      for (int j = 0; j < dim; ++j)
      {
         gradJ[i][j] = 0.0;
      }
   }
   gradJ[0][0] = 2*x[0];
}

void J1_solution_laplace(const Vector &x,
                         std::vector<complex<double>> &laplaceJ)
{
   laplaceJ.resize(dim);
   for (int i = 0; i < dim; ++i)
   {
      laplaceJ[i] = 0.0;
   }
   laplaceJ[0] = 2.0;
}

void J1_solution_directional_laplace(const Vector &x, const Vector &b,
                                     std::vector<complex<double>> &blaplaceJ)
{
   blaplaceJ.resize(dim);
   blaplaceJ[0] = b(0)*b(0)*2.0;
   blaplaceJ[1] = 0.0;
}

void J2_solution(const Vector &x,std::vector<complex<double>> &J)
{
   complex<double> zi = complex<double>(0., 1.);
   J.resize(dim);
   for (int i = 0; i < dim; ++i)
   {
      J[i] = 0.0;
   }
   J[1] = x[1] * x[1];
}

void J2_solution_grad(const Vector &x,
                      std::vector<std::vector<complex<double>>> &gradJ)
{
   gradJ.resize(dim);
   for (int i = 0; i < dim; ++i)
   {
      gradJ[i].resize(dim);
      for (int j = 0; j < dim; ++j)
      {
         gradJ[i][j] = 0.0;
      }
   }
   gradJ[1][1] = 2*x[1];
}

void J2_solution_laplace(const Vector &x,
                         std::vector<complex<double>> &laplaceJ)
{
   laplaceJ.resize(dim);
   for (int i = 0; i < dim; ++i)
   {
      laplaceJ[i] = 0.0;
   }
   laplaceJ[1] = 2.0;
}

void J2_solution_directional_laplace(const Vector &x, const Vector &b,
                                     std::vector<complex<double>> &blaplaceJ)
{
   blaplaceJ.resize(dim);
   blaplaceJ[0] = 0.0;
   blaplaceJ[1] = b(1)*b(1)*2.0;
}

void E_exact_r(const Vector &x, Vector & E_r)
{
   std::vector<std::complex<double>> E;
   maxwell_solution(x,E);
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

void H_exact_r(const Vector &x, Vector & H_r)
{
   // H = i ∇ × E / ω μ
   // H_r = - ∇ × E_i / ω μ
   Vector curlE_i;
   curlE_exact_i(x,curlE_i);
   H_r.SetSize(dimc);
   for (int i = 0; i<dimc; i++)
   {
      H_r(i) = - curlE_i(i) / (omega * mu0);
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
      H_i(i) = curlE_r(i) / (omega * mu0);
   }
}

void J1_exact_r(const Vector &x, Vector & J_r)
{
   std::vector<std::complex<double>> J;
   J1_solution(x,J);
   J_r.SetSize(J.size());
   for (unsigned i = 0; i < J.size(); i++)
   {
      J_r[i]= J[i].real();
   }
}

void J1_exact_i(const Vector &x, Vector & J_i)
{
   std::vector<std::complex<double>> J;
   J1_solution(x,J);
   J_i.SetSize(J.size());
   for (unsigned i = 0; i < J.size(); i++)
   {
      J_i[i]= J[i].imag();
   }
}

void J2_exact_r(const Vector &x, Vector & J_r)
{
   std::vector<std::complex<double>> J;
   J2_solution(x,J);
   J_r.SetSize(J.size());
   for (unsigned i = 0; i < J.size(); i++)
   {
      J_r[i]= J[i].real();
   }
}

void J2_exact_i(const Vector &x, Vector & J_i)
{
   std::vector<std::complex<double>> J;
   J2_solution(x,J);
   J_i.SetSize(J.size());
   for (unsigned i = 0; i < J.size(); i++)
   {
      J_i[i]= J[i].imag();
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

void curlH_exact_r(const Vector &x,Vector &curlH_r)
{
   // ∇ × H_r = - ∇ × ∇ × E_i / ω μ
   Vector curlcurlE_i;
   curlcurlE_exact_i(x,curlcurlE_i);
   curlH_r.SetSize(dim);
   for (int i = 0; i<dim; i++)
   {
      curlH_r(i) = -curlcurlE_i(i) / (omega * mu0);
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
      curlH_i(i) = curlcurlE_r(i) / (omega * mu0);
   }
}

void gradJ1_exact_r(const Vector &x, DenseMatrix &gradJ_r)
{
   std::vector<std::vector<std::complex<double>>> gradJ;
   J1_solution_grad(x, gradJ);
   gradJ_r.SetSize(gradJ.size());
   for (unsigned i = 0; i < gradJ.size(); i++)
   {
      for (unsigned j = 0; j < gradJ.size(); j++)
      {
         gradJ_r(i,j)= gradJ[i][j].real();
      }
   }
}

void gradJ1_exact_i(const Vector &x, DenseMatrix &gradJ_i)
{
   std::vector<std::vector<std::complex<double>>> gradJ;
   J1_solution_grad(x, gradJ);
   gradJ_i.SetSize(gradJ.size());
   for (unsigned i = 0; i < gradJ.size(); i++)
   {
      for (unsigned j = 0; j < gradJ.size(); j++)
      {
         gradJ_i(i,j)= gradJ[i][j].imag();
      }
   }
}

void gradJ2_exact_r(const Vector &x, DenseMatrix &gradJ_r)
{
   std::vector<std::vector<std::complex<double>>> gradJ;
   J2_solution_grad(x, gradJ);
   gradJ_r.SetSize(gradJ.size());
   for (unsigned i = 0; i < gradJ.size(); i++)
   {
      for (unsigned j = 0; j < gradJ.size(); j++)
      {
         gradJ_r(i,j)= gradJ[i][j].real();
      }
   }
}

void gradJ2_exact_i(const Vector &x, DenseMatrix &gradJ_i)
{
   std::vector<std::vector<std::complex<double>>> gradJ;
   J2_solution_grad(x, gradJ);
   gradJ_i.SetSize(gradJ.size());
   for (unsigned i = 0; i < gradJ.size(); i++)
   {
      for (unsigned j = 0; j < gradJ.size(); j++)
      {
         gradJ_i(i,j)= gradJ[i][j].imag();
      }
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


void LaplaceJ1_exact_r(const Vector &x, Vector & d2J_r)
{
   std::vector<std::complex<double>> d2J;
   J1_solution_laplace(x, d2J);
   d2J_r.SetSize(d2J.size());
   for (unsigned i = 0; i < d2J.size(); i++)
   {
      d2J_r[i]= d2J[i].real();
   }
}

void LaplaceJ1_exact_i(const Vector &x, Vector & d2J_i)
{
   std::vector<std::complex<double>> d2J;
   J1_solution_laplace(x, d2J);
   d2J_i.SetSize(d2J.size());
   for (unsigned i = 0; i < d2J.size(); i++)
   {
      d2J_i[i]= d2J[i].imag();
   }
}

void LaplaceJ2_exact_r(const Vector &x, Vector & d2J_r)
{
   std::vector<std::complex<double>> d2J;
   J2_solution_laplace(x, d2J);
   d2J_r.SetSize(d2J.size());
   for (unsigned i = 0; i < d2J.size(); i++)
   {
      d2J_r[i]= d2J[i].real();
   }
}

void LaplaceJ2_exact_i(const Vector &x, Vector & d2J_i)
{
   std::vector<std::complex<double>> d2J;
   J2_solution_laplace(x, d2J);
   d2J_i.SetSize(d2J.size());
   for (unsigned i = 0; i < d2J.size(); i++)
   {
      d2J_i[i]= d2J[i].imag();
   }
}

void DirectionalLaplaceJ1_exact_r(const Vector &x, const Vector &b,
                                  Vector & d2J_r)
{
   std::vector<std::complex<double>> d2J;
   J1_solution_directional_laplace(x, b, d2J);
   d2J_r.SetSize(d2J.size());
   for (unsigned i = 0; i < d2J.size(); i++)
   {
      d2J_r[i]= d2J[i].real();
   }
}

void DirectionalLaplaceJ1_exact_i(const Vector &x, const Vector &b,
                                  Vector & d2J_i)
{
   std::vector<std::complex<double>> d2J;
   J1_solution_directional_laplace(x, b, d2J);
   d2J_i.SetSize(d2J.size());
   for (unsigned i = 0; i < d2J.size(); i++)
   {
      d2J_i[i]= d2J[i].imag();
   }
}

void DirectionalLaplaceJ2_exact_r(const Vector &x, const Vector &b,
                                  Vector & d2J_r)
{
   std::vector<std::complex<double>> d2J;
   J2_solution_directional_laplace(x, b, d2J);
   d2J_r.SetSize(d2J.size());
   for (unsigned i = 0; i < d2J.size(); i++)
   {
      d2J_r[i]= d2J[i].real();
   }
}

void DirectionalLaplaceJ2_exact_i(const Vector &x, const Vector &b,
                                  Vector & d2J_i)
{
   std::vector<std::complex<double>> d2J;
   J2_solution_directional_laplace(x, b, d2J);
   d2J_i.SetSize(d2J.size());
   for (unsigned i = 0; i < d2J.size(); i++)
   {
      d2J_i[i]= d2J[i].imag();
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
      hatE_r.SetSize(E_r.Size());
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
      hatE_i.SetSize(E_i.Size());
      // rotate E_hat
      hatE_i[0] = E_i[1];
      hatE_i[1] = -E_i[0];
   }
}

// F = -i ω ϵ₀ ϵ E + ∇ × H - cj ϵ₀ ω J
// F_r + i F_i = -i ω ϵ₀(ϵᵣ + i ϵᵢ)(E_r + i E_i) + ∇ × (H_r + i H_i) - cj ϵ₀ ω (J_r + i J_i)
void  rhs_dH_func_r(const Vector &x, Vector & F_r)
{
   // F_r = ω ϵ₀ (ϵᵣE_i + ϵᵢE_r) + ∇ × H_r - cj ω J_r
   Vector E_i, E_r, curlH_r, J1_r, J2_r;
   E_exact_r(x,E_r);
   E_exact_i(x,E_i);
   curlH_exact_r(x,curlH_r);
   J1_exact_r(x,J1_r);
   J2_exact_r(x,J2_r);
   F_r.SetSize(dim);
   DenseMatrix eps_r, eps_i;
   epsilon_r(x, eps_r);
   epsilon_i(x, eps_i);

   Vector eps_iE_r(dim); eps_i.Mult(E_r, eps_iE_r);
   Vector eps_rE_i(dim); eps_r.Mult(E_i, eps_rE_i);

   for (int i = 0; i<dim; i++)
   {
      F_r(i) = omega * eps0 * (eps_rE_i(i) + eps_iE_r(i)) + curlH_r(
                  i) - cj *  eps0 * omega * (J1_r(i)+J2_r(i));
   }
}

void  rhs_dH_func_i(const Vector &x, Vector & F_i)
{
   // F_i = - ω (ϵᵣ E_r - ϵᵢE_i) + ∇ × H_i - cj*ϵ₀*ω J_i
   Vector E_r, E_i, curlH_i, J1_i, J2_i;
   E_exact_r(x,E_r);
   E_exact_i(x,E_i);
   curlH_exact_i(x,curlH_i);
   J1_exact_i(x,J1_i);
   J2_exact_i(x,J2_i);
   F_i.SetSize(dim);
   DenseMatrix eps_r, eps_i;
   epsilon_r(x, eps_r);
   epsilon_i(x, eps_i);

   Vector eps_rE_r(dim); eps_r.Mult(E_r, eps_rE_r);
   Vector eps_iE_i(dim); eps_i.Mult(E_i, eps_iE_i);

   for (int i = 0; i<dim; i++)
   {
      F_i(i) = -omega * eps0 * (eps_rE_r(i) - eps_iE_i(i)) + curlH_i(
                  i) - cj * eps0 * omega * (J1_i(i)+J2_i(i));
   }
}

// G = -ΔJ + c₁ J - c₁ P(r) b⊗b  E
// G_r + i G_i = - Δ (J_r + i J_i) + c₁ (J_r + i J_i) - c₁ (Pᵣ + i Pᵢ) b⊗b (E_r + i E_i)
void  rhs_dJ1_func_r(const Vector &x, Vector & G_r)
{
   // G_r = - Δ J_r + c₁ J_r - c₁ (Pᵣ b⊗b E_r - Pᵢ b⊗b E_i)
   Vector E_r, E_i, J_r, d2J_r;
   E_exact_r(x,E_r);
   E_exact_i(x,E_i);
   J1_exact_r(x,J_r);
   // LaplaceJ1_exact_r(x,d2J_r);
   Vector b; bfunc(x,b);
   DirectionalLaplaceJ1_exact_r(x, b, d2J_r);
   DenseMatrix bb; bcrossb(x, bb);
   real_t P_r = pfunc_r(x);
   real_t P_i = pfunc_i(x);
   Vector bbE_r(dim), bbE_i(dim);
   bb.Mult(E_r, bbE_r);
   bb.Mult(E_i, bbE_i);
   G_r.SetSize(dim);
   for (int i = 0; i<dim; i++)
   {
      G_r(i) = -d2J_r[i]*cfactor + c1*J_r[i] - c1 * (P_r * bbE_r[i] - P_i * bbE_i[i]);
   }
}

void  rhs_dJ1_func_i(const Vector &x, Vector & G_i)
{
   // G_i = - Δ J_i + c₁ J_i - c₁ (Pᵢ b⊗b E_r + Pᵣ b⊗b E_i)
   Vector E_i, E_r, J_i, d2J_i;
   E_exact_i(x,E_i);
   E_exact_r(x,E_r);
   J1_exact_i(x,J_i);
   // LaplaceJ1_exact_i(x,d2J_i);
   Vector b; bfunc(x,b);
   DirectionalLaplaceJ1_exact_i(x, b, d2J_i);
   DenseMatrix bb; bcrossb(x, bb);
   real_t P_r = pfunc_r(x);
   real_t P_i = pfunc_i(x);
   Vector bbE_r(dim), bbE_i(dim);
   bb.Mult(E_r, bbE_r);
   bb.Mult(E_i, bbE_i);
   G_i.SetSize(dim);
   for (int i = 0; i<dim; i++)
   {
      G_i(i) = -d2J_i[i]*cfactor + c1*J_i[i] - c1 * (P_i * bbE_r[i] + P_r * bbE_i[i]);
   }
}

void  rhs_dJ2_func_r(const Vector &x, Vector & G_r)
{
   // G_r = - Δ J_r + c₂ J_r + c₂ (Pᵣ b⊗b E_r - Pᵢ b⊗b E_i)
   Vector E_r, E_i, J_r, d2J_r;
   E_exact_r(x,E_r);
   E_exact_i(x,E_i);
   J2_exact_r(x,J_r);
   // LaplaceJ2_exact_r(x,d2J_r);
   Vector b; bfunc(x,b);
   DirectionalLaplaceJ2_exact_r(x, b, d2J_r);
   DenseMatrix bb; bcrossb(x, bb);
   real_t P_r = pfunc_r(x);
   real_t P_i = pfunc_i(x);
   Vector bbE_r(dim), bbE_i(dim);
   bb.Mult(E_r, bbE_r);
   bb.Mult(E_i, bbE_i);
   G_r.SetSize(dim);
   for (int i = 0; i<dim; i++)
   {
      G_r(i) = -d2J_r[i]*cfactor + c2*J_r[i] + c2 * (P_r * bbE_r[i] - P_i * bbE_i[i]);
   }
}

void  rhs_dJ2_func_i(const Vector &x, Vector & G_i)
{
   // G_i = - Δ J_i + c₂ J_i + c₂ (Pᵢ b⊗b E_r + Pᵣ b⊗b E_i)
   Vector E_i, E_r, J_i, d2J_i;
   E_exact_i(x,E_i);
   E_exact_r(x,E_r);
   J2_exact_i(x,J_i);
   // LaplaceJ2_exact_i(x,d2J_i);
   Vector b; bfunc(x,b);
   DirectionalLaplaceJ2_exact_i(x, b, d2J_i);
   DenseMatrix bb; bcrossb(x, bb);
   real_t P_r = pfunc_r(x);
   real_t P_i = pfunc_i(x);
   Vector bbE_r(dim), bbE_i(dim);
   bb.Mult(E_r, bbE_r);
   bb.Mult(E_i, bbE_i);
   G_i.SetSize(dim);
   for (int i = 0; i<dim; i++)
   {
      G_i(i) = -d2J_i[i]*cfactor + c2*J_i[i] + c2 * (P_i * bbE_r[i] + P_r * bbE_i[i]);
   }
}