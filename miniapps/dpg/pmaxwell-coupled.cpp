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
// Compile with: make pmaxwell
//
// mpirun -np 4 pmaxwell -m ../../data/inline-quad.mesh

//      ∇×(1/μ ∇×E) - ω² ϵ E + J = F̃ ,   in Ω
//             -ΔJ + α² J + c² E = G ,   in Ω
//                       E×n = E₀ , on ∂Ω
//                         J = J₀ , on ∂Ω

// The DPG UW deals with the First Order System
//  i ω μ H + ∇ × E         = 0,   in Ω
// -i ω ϵ E + ∇ × H + i/ω J = F,   in Ω
//        -ΔJ + α² J + c² E = G ,   in Ω
//                    E × n = E_0, on ∂Ω
//                        J = J₀ , on ∂Ω

// E,H ∈ (L^2(Ω))³
// Ê ∈ H_0^1/2(Ω)(curl, Γₕ), Ĥ ∈ H^-1/2(curl, Γₕ)
//  i ω μ (H,δE) + (E,∇ × δE) + < Ê, δE × n >              = 0,      ∀ δE ∈ H(curl,Ω)
// -i ω ϵ (E,δH) + (H,∇ × δH) + < Ĥ, δH × n > + i/ω (J,δH) = (F,δH),   ∀ δH ∈ H(curl,Ω)
//            (∇J, ∇δJ) + <Ĵ, δJ> + α² (J,δJ) + c² (E, δJ) = (G,δJ),      ∀ δJ ∈ (H¹(Ω))ᵈ
//
//                                   Ê × n = E₀,      on ∂Ω
//                                       J = J₀,      on ∂Ω
// --------------------------------------------------------------------------------------------------------------------
// |    |       E        |        H       |      J            |      Ê        |       Ĥ       |    Ĵ     |     RHS    |
// --------------------------------------------------------------------------------------------------------------------
// | δE |   (E,∇ × δE)   |  i ω μ (H,δE)  |                   | < Ê, δE × n > |               |          |            |
// |    |                |                |                   |               |               |          |            |
// | δH | -i ω ϵ (E,δH)  |   (H,∇ × δH)   |    i/ω (J,δH)     |               | < Ĥ, δH × n > |          |   (F,δH)   |
// |    |                |                |                   |               |               |          |            |
// | δJ |   c² (E, δJ)   |                | (∇J,∇δJ)+α²(J,δJ) |               |               |  <Ĵ, δJ> |   (G,δJ)   |
//
// where (δE,δH,δJ) ∈  H(curl,Ω) × H(curl,Ω) × (H¹(Ω))ᵈ

#include "mfem.hpp"
#include "util/pcomplexweakform.hpp"
#include "../common/mfem-common.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mfem::common;

void maxwell_solution(const Vector &x, std::vector<complex<double>> &E);
void maxwell_solution_curl(const Vector &x,
                           std::vector<complex<double>> &curlE);
void maxwell_solution_curlcurl(const Vector &x,
                               std::vector<complex<double>> &curlcurlE);
void J_solution(const Vector &x,std::vector<complex<double>> &J);
void J_solution_grad(const Vector &x,
                     std::vector<std::vector<complex<double>>> &gradJ);
void J_solution_laplace(const Vector &x,
                        std::vector<complex<double>> &laplaceJ);


void E_exact_r(const Vector &x, Vector & E_r);
void E_exact_i(const Vector &x, Vector & E_i);
void H_exact_r(const Vector &x, Vector & H_r);
void H_exact_i(const Vector &x, Vector & H_i);
void J_exact_r(const Vector &x, Vector & J_r);
void J_exact_i(const Vector &x, Vector & J_i);

void curlE_exact_r(const Vector &x, Vector &curlE_r);
void curlE_exact_i(const Vector &x, Vector &curlE_i);
void curlH_exact_r(const Vector &x,Vector &curlH_r);
void curlH_exact_i(const Vector &x,Vector &curlH_i);
void gradJ_exact_r(const Vector &x, DenseMatrix &gradJ_r);
void gradJ_exact_i(const Vector &x, DenseMatrix &gradJ_i);

void curlcurlE_exact_r(const Vector &x, Vector & curlcurlE_r);
void curlcurlE_exact_i(const Vector &x, Vector & curlcurlE_i);
void LaplaceJ_exact_r(const Vector &x, Vector & d2J_r);
void LaplaceJ_exact_i(const Vector &x, Vector & d2J_i);


void hatE_exact_r(const Vector & X, Vector & hatE_r);
void hatE_exact_i(const Vector & X, Vector & hatE_i);

void  rhs1_func_r(const Vector &x, Vector & rhs1_r);
void  rhs1_func_i(const Vector &x, Vector & rhs1_i);
void  rhs2_func_r(const Vector &x, Vector & rhs2_r);
void  rhs2_func_i(const Vector &x, Vector & rhs2_i);


int dim;
int dimc;
double omega;
double mu = 1.0;
double epsilon = 1.0;
double alpha = 1.0;
double c = 1.0;

int main(int argc, char *argv[])
{
   Mpi::Init();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   const char *mesh_file = "../../data/inline-hex.mesh";
   int order = 1;
   int delta_order = 1;
   double rnum=1.0;
   int sr = 0;
   int pr = 1;
   bool static_cond = false;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree)");
   args.AddOption(&rnum, "-rnum", "--number-of-wavelengths",
                  "Number of wavelengths");
   args.AddOption(&delta_order, "-do", "--delta-order",
                  "Order enrichment for DPG test space.");
   args.AddOption(&sr, "-sref", "--serial-ref",
                  "Number of parallel refinements.");
   args.AddOption(&pr, "-pref", "--parallel-ref",
                  "Number of parallel refinements.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }

   omega = 2.*M_PI*rnum;

   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   Mesh mesh(mesh_file, 1, 1);
   dim = mesh.Dimension();
   MFEM_VERIFY(dim > 1, "Dimension = 1 is not supported in this example");

   dimc = 1;

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
      J_space     = 2,
      hatE_space  = 3,
      hatH_space  = 4,
      hatJ_space  = 5
   };
   enum TestSpace
   {
      dE_space = 0,
      dH_space = 1,
      dJ_space = 2
   };
   // Vector L2 space for E
   FiniteElementCollection *E_fec = new L2_FECollection(order-1,dim);
   ParFiniteElementSpace *E_fes = new ParFiniteElementSpace(&pmesh,E_fec,dim);

   // Vector L2 space for H
   FiniteElementCollection *H_fec = new L2_FECollection(order-1,dim);
   ParFiniteElementSpace *H_fes = new ParFiniteElementSpace(&pmesh,H_fec, dim);

   // Vector H1 space for J
   FiniteElementCollection *J_fec = new H1_FECollection(order,dim);
   ParFiniteElementSpace *J_fes = new ParFiniteElementSpace(&pmesh,J_fec, dim);


   // H^-1/2 (curl) space for Ê
   FiniteElementCollection * hatE_fec = new ND_Trace_FECollection(order,dim);
   ParFiniteElementSpace *hatE_fes = new ParFiniteElementSpace(&pmesh,hatE_fec);

   // H^-1/2 (curl) space for Ĥ
   FiniteElementCollection * hatH_fec = new ND_Trace_FECollection(order,dim);
   ParFiniteElementSpace *hatH_fes = new ParFiniteElementSpace(&pmesh,hatH_fec);

   // H^-1/2 space for Ĵ
   FiniteElementCollection * hatJ_fec = new RT_Trace_FECollection(order-1,dim);
   ParFiniteElementSpace *hatJ_fes = new ParFiniteElementSpace(&pmesh, hatJ_fec,
                                                               dim);

   int test_order = order+delta_order;
   FiniteElementCollection * dE_fec = new ND_FECollection(test_order, dim);
   FiniteElementCollection * dH_fec = new ND_FECollection(test_order, dim);
   FiniteElementCollection * dJ_fec = new H1_FECollection(test_order, dim);

   Array<ParFiniteElementSpace * > trial_fes;
   Array<FiniteElementCollection * > test_fec;
   trial_fes.Append(E_fes);
   trial_fes.Append(H_fes);
   trial_fes.Append(J_fes);
   trial_fes.Append(hatE_fes);
   trial_fes.Append(hatH_fes);
   trial_fes.Append(hatJ_fes);
   test_fec.Append(dE_fec);
   test_fec.Append(dH_fec);
   test_fec.Append(dJ_fec);

   ParComplexDPGWeakForm * a = new ParComplexDPGWeakForm(trial_fes,test_fec);
   a->SetTestFECollVdim(TestSpace::dJ_space,dim);
   a->StoreMatrices();

   // // Bilinear form coefficients
   ConstantCoefficient one(1.0);
   ConstantCoefficient rec_omega(1.0/omega);
   ConstantCoefficient muomeg_cf(mu*omega);
   ConstantCoefficient negepsomeg_cf(-epsilon*omega);
   ConstantCoefficient c2_cf(c*c);
   ConstantCoefficient a2_cf(alpha*alpha);

   // (E,∇ × δE)
   a->AddTrialIntegrator(new TransposeIntegrator(new MixedCurlIntegrator(one)),
                         nullptr,TrialSpace::E_space, TestSpace::dE_space);
   //  -i ω ϵ (E,δH) = i (- ω ϵ E, δH)
   a->AddTrialIntegrator(nullptr,
                         new TransposeIntegrator(new VectorFEMassIntegrator(negepsomeg_cf)),
                         TrialSpace::E_space,TestSpace::dH_space);
   // c² (E, δJ)
   a->AddTrialIntegrator(new VectorMassIntegrator(c2_cf), nullptr,
                         TrialSpace::E_space, TestSpace::dJ_space);

   // i ω μ (H, δE)
   a->AddTrialIntegrator(nullptr,new TransposeIntegrator(
                            new VectorFEMassIntegrator(muomeg_cf)),
                         TrialSpace::H_space, TestSpace::dE_space);

   // (H,∇ × δH)
   a->AddTrialIntegrator(new TransposeIntegrator(new MixedCurlIntegrator(one)),
                         nullptr,TrialSpace::H_space, TestSpace::dH_space);

   //  i/ω (J,δH)
   a->AddTrialIntegrator(nullptr,
                         new TransposeIntegrator(new VectorFEMassIntegrator(rec_omega)),
                         TrialSpace::J_space, TestSpace::dH_space);

   // (∇J,∇δJ)
   a->AddTrialIntegrator(new VectorDiffusionIntegrator(one),nullptr,
                         TrialSpace::J_space,TestSpace::dJ_space);

   // α²(J,δJ)
   a->AddTrialIntegrator(new VectorMassIntegrator(a2_cf),nullptr,
                         TrialSpace::J_space,TestSpace::dJ_space);

   // < Ê, δE × n >
   a->AddTrialIntegrator(new TangentTraceIntegrator,nullptr,
                         TrialSpace::hatE_space, TestSpace::dE_space);

   // < Ĥ, δH × n >
   a->AddTrialIntegrator(new TangentTraceIntegrator,nullptr,
                         TrialSpace::hatH_space, TestSpace::dH_space);

   // <Ĵ, δJ>
   a->AddTrialIntegrator(new VectorTraceIntegrator,nullptr,
                         TrialSpace::hatJ_space,TestSpace::dJ_space);

   // test integrators
   // (∇×δE ,∇× δE)
   a->AddTestIntegrator(new CurlCurlIntegrator(one),nullptr,
                        TestSpace::dE_space,TestSpace::dE_space);
   // (δE,δE)
   a->AddTestIntegrator(new VectorFEMassIntegrator(one),nullptr,
                        TestSpace::dE_space,TestSpace::dE_space);

   // (∇×δH ,∇×δH)
   a->AddTestIntegrator(new CurlCurlIntegrator(one),nullptr,
                        TestSpace::dH_space,TestSpace::dH_space);
   // (δH,δH)
   a->AddTestIntegrator(new VectorFEMassIntegrator(one),nullptr,
                        TestSpace::dH_space,TestSpace::dH_space);

   // (∇δJ,∇δJ)
   a->AddTestIntegrator(new VectorDiffusionIntegrator(one),nullptr,
                        TestSpace::dJ_space,TestSpace::dJ_space);
   // (δJ,δJ)
   a->AddTestIntegrator(new VectorMassIntegrator(one),nullptr, TestSpace::dJ_space,
                        TestSpace::dJ_space);


   VectorFunctionCoefficient f_rhs1_r(dim,rhs1_func_r);
   VectorFunctionCoefficient f_rhs1_i(dim,rhs1_func_i);
   a->AddDomainLFIntegrator(new VectorFEDomainLFIntegrator(f_rhs1_r),
                            new VectorFEDomainLFIntegrator(f_rhs1_i),
                            TestSpace::dH_space);

   VectorFunctionCoefficient f_rhs2_r(dim,rhs2_func_r);
   VectorFunctionCoefficient f_rhs2_i(dim,rhs2_func_i);
   a->AddDomainLFIntegrator(new VectorDomainLFIntegrator(f_rhs2_r),
                            new VectorDomainLFIntegrator(f_rhs2_i),
                            TestSpace::dJ_space);

   socketstream E_out_r, E_out_i, Eex_out_r, Eex_out_i;
   socketstream H_out_r, H_out_i, Hex_out_r, Hex_out_i;
   socketstream J_out_r, J_out_i, Jex_out_r, Jex_out_i;

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
   ParGridFunction J_r(J_fes), J_i(J_fes);
   ParGridFunction Eex_r(E_fes), Eex_i(E_fes);
   ParGridFunction Hex_r(H_fes), Hex_i(H_fes);
   ParGridFunction Jex_r(J_fes), Jex_i(J_fes);

   double res0 = 0.;
   double err0 = 0.;
   int dof0;

   if (static_cond) { a->EnableStaticCondensation(); }
   for (int it = 0; it<=pr; it++)
   {
      a->Assemble();

      Array<int> ess_tdof_list;
      Array<int> ess_tdof_listE;
      Array<int> ess_tdof_listJ;
      Array<int> ess_bdr;
      if (pmesh.bdr_attributes.Size())
      {
         ess_bdr.SetSize(pmesh.bdr_attributes.Max());
         ess_bdr = 1;
         hatE_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_listE);
         J_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_listJ);
      }

      // shift the ess_tdofs
      for (int j = 0; j < ess_tdof_listE.Size(); j++)
      {
         ess_tdof_listE[j] += E_fes->GetTrueVSize() + H_fes->GetTrueVSize() +
                              J_fes->GetTrueVSize();
      }
      for (int j = 0; j < ess_tdof_listJ.Size(); j++)
      {
         ess_tdof_listJ[j] += E_fes->GetTrueVSize() + H_fes->GetTrueVSize();
      }
      ess_tdof_list.Append(ess_tdof_listE);
      ess_tdof_list.Append(ess_tdof_listJ);

      Array<int> offsets(7);
      offsets[0] = 0;
      offsets[1] = E_fes->GetVSize();
      offsets[2] = H_fes->GetVSize();
      offsets[3] = J_fes->GetVSize();
      offsets[4] = hatE_fes->GetVSize();
      offsets[5] = hatH_fes->GetVSize();
      offsets[6] = hatJ_fes->GetVSize();
      offsets.PartialSum();

      Vector x(2*offsets.Last());
      x = 0.;

      ParGridFunction hatE_gf_r(hatE_fes, x, offsets[3]);
      ParGridFunction hatE_gf_i(hatE_fes, x, offsets.Last() + offsets[3]);
      VectorFunctionCoefficient hatEex_r(dim,hatE_exact_r);
      VectorFunctionCoefficient hatEex_i(dim,hatE_exact_i);
      hatE_gf_r.ProjectBdrCoefficientTangent(hatEex_r, ess_bdr);
      hatE_gf_i.ProjectBdrCoefficientTangent(hatEex_i, ess_bdr);

      VectorFunctionCoefficient Ecf_r(dim,E_exact_r);
      VectorFunctionCoefficient Ecf_i(dim,E_exact_i);
      VectorFunctionCoefficient Hcf_r(dim,H_exact_r);
      VectorFunctionCoefficient Hcf_i(dim,H_exact_i);
      VectorFunctionCoefficient Jcf_r(dim,J_exact_r);
      VectorFunctionCoefficient Jcf_i(dim,J_exact_i);

      ParGridFunction J_gf_r(J_fes, x, offsets[2]);
      ParGridFunction J_gf_i(J_fes, x, offsets.Last() + offsets[2]);
      J_gf_r.ProjectBdrCoefficient(Jcf_r,ess_bdr);
      J_gf_i.ProjectBdrCoefficient(Jcf_i,ess_bdr);

      OperatorPtr Ah;
      Vector X,B;
      a->FormLinearSystem(ess_tdof_list,x,Ah, X,B);

      ComplexOperator * Ahc = Ah.As<ComplexOperator>();

      BlockOperator * BlockA_r = dynamic_cast<BlockOperator *>(&Ahc->real());
      BlockOperator * BlockA_i = dynamic_cast<BlockOperator *>(&Ahc->imag());

      int num_blocks = BlockA_r->NumRowBlocks();
      Array<int> tdof_offsets(2*num_blocks+1);

      tdof_offsets[0] = 0;
      int skip = (static_cond) ? 0 : 2;
      for (int i=0; i<num_blocks; i++)
      {
         tdof_offsets[i+1] = BlockA_r->GetBlock(i,i).Height();
         tdof_offsets[num_blocks+i+1] = BlockA_r->GetBlock(i,i).Height();
      }
      tdof_offsets.PartialSum();
      BlockOperator blockA(tdof_offsets);

      for (int i = 0; i<num_blocks; i++)
      {
         for (int j = 0; j<num_blocks; j++)
         {
            blockA.SetBlock(i,j,&BlockA_r->GetBlock(i,j));
            blockA.SetBlock(i,j+num_blocks,&BlockA_i->GetBlock(i,j), -1.0);
            blockA.SetBlock(i+num_blocks,j+num_blocks,&BlockA_r->GetBlock(i,j));
            blockA.SetBlock(i+num_blocks,j,&BlockA_i->GetBlock(i,j));
         }
      }

      X = 0.;
      BlockDiagonalPreconditioner M(tdof_offsets);
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

      HypreBoomerAMG * solver_J = new HypreBoomerAMG((HypreParMatrix &)
                                                     BlockA_r->GetBlock(skip,skip));
      solver_J->SetPrintLevel(0);
      solver_J->SetSystemsOptions(dim);
      M.SetDiagonalBlock(skip,solver_J);
      M.SetDiagonalBlock(skip+num_blocks,solver_J);

      HypreAMS * solver_hatE = new HypreAMS((HypreParMatrix &)BlockA_r->GetBlock(
                                               skip+1,skip+1),
                                            hatE_fes);
      solver_hatE->SetPrintLevel(0);
      HypreAMS * solver_hatH = new HypreAMS((HypreParMatrix &)BlockA_r->GetBlock(
                                               skip+2,skip+2),
                                            hatH_fes);
      solver_hatH->SetPrintLevel(0);
      HypreBoomerAMG * solver_hatJ = new HypreBoomerAMG((HypreParMatrix &)
                                                        BlockA_r->GetBlock(skip+3,skip+3));
      solver_hatJ->SetPrintLevel(0);

      M.SetDiagonalBlock(skip+1,solver_hatE);
      M.SetDiagonalBlock(skip+2,solver_hatH);
      M.SetDiagonalBlock(skip+3,solver_hatJ);
      M.SetDiagonalBlock(num_blocks+skip+1,solver_hatE);
      M.SetDiagonalBlock(num_blocks+skip+2,solver_hatH);
      M.SetDiagonalBlock(num_blocks+skip+3,solver_hatJ);

      CGSolver cg(MPI_COMM_WORLD);
      cg.SetRelTol(1e-6);
      cg.SetMaxIter(10000);
      cg.SetPrintLevel(0);
      cg.SetPreconditioner(M);
      cg.SetOperator(blockA);
      cg.Mult(B, X);
      int num_iter = cg.GetNumIterations();

      for (int i = 0; i<num_blocks; i++)
      {
         delete &M.GetDiagonalBlock(i);
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

      J_r.MakeRef(J_fes,x, offsets[2]);
      J_i.MakeRef(J_fes,x, offsets.Last()+offsets[2]);

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
      double J_err_r = J_r.ComputeL2Error(Jcf_r);
      double J_err_i = J_i.ComputeL2Error(Jcf_i);

      Eex_r.ProjectCoefficient(Ecf_r);
      Eex_i.ProjectCoefficient(Ecf_i);
      Hex_r.ProjectCoefficient(Hcf_r);
      Hex_i.ProjectCoefficient(Hcf_i);
      Jex_r.ProjectCoefficient(Jcf_r);
      Jex_i.ProjectCoefficient(Jcf_i);

      L2Error = sqrt(E_err_r*E_err_r + E_err_i*E_err_i
                     + H_err_r*H_err_r + H_err_i*H_err_i
                     + J_err_r*J_err_r + J_err_i*J_err_i);
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
         VisualizeField(Eex_out_r,vishost, visport, Eex_r,
                        "Exact Electric field (real part)", 501, 0, 500, 500, keys);
         VisualizeField(H_out_r,vishost, visport, H_r,
                        "Numerical Magnetic field (real part)", 0, 0, 500, 500, keys);
         VisualizeField(Hex_out_r,vishost, visport, Hex_r,
                        "Exact Magnetic field (real part)", 501, 0, 500, 500, keys);
         VisualizeField(J_out_r,vishost, visport, J_r,
                        "Numerical J field (real part)", 0, 0, 500, 500, keys);
         VisualizeField(Jex_out_r,vishost, visport, Jex_r,
                        "Exact J field (real part)", 501, 0, 500, 500, keys);

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
      Jex_r.Update();
      Jex_i.Update();
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
   curlE.resize(dim);
   std::complex<double> pw = exp(zi * omega * (X.Sum()));
   curlE[0] = 0.0;
   curlE[1] = zi * omega * pw;
   curlE[2] = -zi * omega * pw;
}

void maxwell_solution_curlcurl(const Vector & X,
                               std::vector<complex<double>> &curlcurlE)
{
   complex<double> zi = complex<double>(0., 1.);
   curlcurlE.resize(dim);
   std::complex<double> pw = exp(zi * omega * (X.Sum()));
   curlcurlE[0] = 2.0 * omega * omega * pw;
   curlcurlE[1] = - omega * omega * pw;
   curlcurlE[2] = - omega * omega * pw;
}

void J_solution(const Vector &x,std::vector<complex<double>> &J)
{
   complex<double> zi = complex<double>(0., 1.);
   J.resize(dim);
   for (int i = 0; i < dim; ++i)
   {
      J[i] = 0.0;
   }
   J[0] = x[0] * x[0];
}

void J_solution_grad(const Vector &x,
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

void J_solution_laplace(const Vector &x, std::vector<complex<double>> &laplaceJ)
{
   laplaceJ.resize(dim);
   for (int i = 0; i < dim; ++i)
   {
      laplaceJ[i] = 0.0;
   }
   laplaceJ[0] = 2.0;
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
   H_r.SetSize(dim);
   for (int i = 0; i<dim; i++)
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
   H_i.SetSize(dim);
   for (int i = 0; i<dim; i++)
   {
      H_i(i) = curlE_r(i) / (omega * mu);
   }
}

void J_exact_r(const Vector &x, Vector & J_r)
{
   std::vector<std::complex<double>> J;
   J_solution(x,J);
   J_r.SetSize(J.size());
   for (unsigned i = 0; i < J.size(); i++)
   {
      J_r[i]= J[i].real();
   }
}

void J_exact_i(const Vector &x, Vector & J_i)
{
   std::vector<std::complex<double>> J;
   J_solution(x,J);
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

void gradJ_exact_r(const Vector &x, DenseMatrix &gradJ_r)
{
   std::vector<std::vector<std::complex<double>>> gradJ;
   J_solution_grad(x, gradJ);
   gradJ_r.SetSize(gradJ.size());
   for (unsigned i = 0; i < gradJ.size(); i++)
   {
      for (unsigned j = 0; j < gradJ.size(); j++)
      {
         gradJ_r(i,j)= gradJ[i][j].real();
      }
   }
}

void gradJ_exact_i(const Vector &x, DenseMatrix &gradJ_i)
{
   std::vector<std::vector<std::complex<double>>> gradJ;
   J_solution_grad(x, gradJ);
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


void LaplaceJ_exact_r(const Vector &x, Vector & d2J_r)
{
   std::vector<std::complex<double>> d2J;
   J_solution_laplace(x, d2J);
   d2J_r.SetSize(d2J.size());
   for (unsigned i = 0; i < d2J.size(); i++)
   {
      d2J_r[i]= d2J[i].real();
   }
}

void LaplaceJ_exact_i(const Vector &x, Vector & d2J_i)
{
   std::vector<std::complex<double>> d2J;
   J_solution_laplace(x, d2J);
   d2J_i.SetSize(d2J.size());
   for (unsigned i = 0; i < d2J.size(); i++)
   {
      d2J_i[i]= d2J[i].imag();
   }
}

void hatE_exact_r(const Vector & x, Vector & hatE_r)
{
   E_exact_r(x,hatE_r);
}

void hatE_exact_i(const Vector & x, Vector & hatE_i)
{
   E_exact_i(x,hatE_i);
}

// F = -i ω ϵ E + ∇ × H + i/ω J
// F_r + i F_i = -i ω ϵ (E_r + i E_i) + ∇ × (H_r + i H_i) + i/ω (J_r + i J_i)
void  rhs1_func_r(const Vector &x, Vector & F_r)
{
   // F_r = ω ϵ E_i + ∇ × H_r - 1/ω J_i
   Vector E_i, curlH_r, J_i;
   E_exact_i(x,E_i);
   curlH_exact_r(x,curlH_r);
   J_exact_i(x,J_i);
   F_r.SetSize(dim);
   for (int i = 0; i<dim; i++)
   {
      F_r(i) = omega * epsilon * E_i(i) + curlH_r(i) - 1.0/omega * J_i(i);
   }
}

void  rhs1_func_i(const Vector &x, Vector & F_i)
{
   // F_i = - ω ϵ E_r + ∇ × H_i + 1/ω J_r
   Vector E_r, curlH_i, J_r;
   E_exact_r(x,E_r);
   curlH_exact_i(x,curlH_i);
   J_exact_r(x,J_r);
   F_i.SetSize(dim);
   for (int i = 0; i<dim; i++)
   {
      F_i(i) = -omega * epsilon * E_r(i) + curlH_i(i) + 1.0/omega * J_r(i);
   }
}


// G = -ΔJ + α² J + c² E
// G_r + i G_i = - Δ (J_r + i J_i) + α² (J_r + i J_i) + c² (E_r + i E_i)
void  rhs2_func_r(const Vector &x, Vector & G_r)
{
   // G_r = - Δ J_r + α² J_r + c² E_r
   Vector E_r, J_r, d2J_r;
   E_exact_r(x,E_r);
   J_exact_r(x,J_r);
   LaplaceJ_exact_r(x,d2J_r);
   G_r.SetSize(dim);
   for (int i = 0; i<dim; i++)
   {
      G_r(i) = -d2J_r[i] + alpha*alpha*J_r[i] + c*c * E_r[i];
   }
}

void  rhs2_func_i(const Vector &x, Vector & G_i)
{
   // G_i = - Δ J_i + α² J_i + c² E_i
   Vector E_i, J_i, d2J_i;
   E_exact_i(x,E_i);
   J_exact_i(x,J_i);
   LaplaceJ_exact_i(x,d2J_i);
   G_i.SetSize(dim);
   for (int i = 0; i<dim; i++)
   {
      G_i(i) = -d2J_i[i] + alpha*alpha*J_i[i] + c*c * E_i[i];
   }
}