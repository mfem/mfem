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

//      ∇×(1/μ ∇×E) - ω² ϵ E + J = F̃ ,   in Ω
//             -ΔJ + α² J + c² E = G ,   in Ω
//                       E×n = E₀ , on ∂Ω
//                         J = J₀ , on ∂Ω

#include "mfem.hpp"
#include "util/pcomplexweakform.hpp"
#include "util/pcomplexblockform.hpp"
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

   const char *mesh_file = "../../data/inline-square.mesh";
   int order = 1;
   double rnum=1.0;
   int sr = 0;
   int pr = 1;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree)");
   args.AddOption(&rnum, "-rnum", "--number-of-wavelengths",
                  "Number of wavelengths");
   args.AddOption(&sr, "-sref", "--serial-ref",
                  "Number of parallel refinements.");
   args.AddOption(&pr, "-pref", "--parallel-ref",
                  "Number of parallel refinements.");
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
      J_space     = 1
   };

   // Vector L2 space for E
   FiniteElementCollection *E_fec = new ND_FECollection(order,dim);
   ParFiniteElementSpace *E_fes = new ParFiniteElementSpace(&pmesh,E_fec);

   // Vector H1 space for J
   FiniteElementCollection *J_fec = new H1_FECollection(order,dim);
   ParFiniteElementSpace *J_fes = new ParFiniteElementSpace(&pmesh,J_fec, dim);

   Array<ParFiniteElementSpace * > trial_fes;
   trial_fes.Append(E_fes);
   trial_fes.Append(J_fes);

   ParComplexBlockForm * a = new ParComplexBlockForm(trial_fes);

   // // Bilinear form coefficients
   ConstantCoefficient one(1.0);
   ConstantCoefficient muinv(1./mu);
   ConstantCoefficient rec_omega(1.0/omega);
   ConstantCoefficient muomeg_cf(mu*omega);
   ConstantCoefficient mu2omeg2_cf(mu*mu*omega*omega);
   ConstantCoefficient eps2omeg2_cf(epsilon*epsilon*omega*omega);
   ConstantCoefficient negepsomeg_cf(-epsilon*omega);
   ConstantCoefficient epsomeg_cf(epsilon*omega);
   ConstantCoefficient negmuomeg_cf(-mu*omega);
   ConstantCoefficient c2_cf(c*c);
   ConstantCoefficient a2_cf(alpha*alpha);

   ConstantCoefficient negepsomeg2_cf(-epsilon*omega*omega);


   // (1/μ ∇ × E,∇ × δE)
   a->AddDomainIntegrator(new CurlCurlIntegrator(muinv), nullptr,
                          TrialSpace::E_space, TrialSpace::E_space);
   // (- ω² ϵ E, δE)
   a->AddDomainIntegrator(new VectorFEMassIntegrator(negepsomeg2_cf), nullptr,
                          TrialSpace::E_space, TrialSpace::E_space);
   // //  (J, δE)
   a->AddDomainIntegrator(new TransposeIntegrator(new VectorFEMassIntegrator(one)),
                          nullptr,
                          TrialSpace::J_space, TrialSpace::E_space);

   // // (∇ J, ∇ δJ)
   a->AddDomainIntegrator(new VectorDiffusionIntegrator(one), nullptr,
                          TrialSpace::J_space, TrialSpace::J_space);
   // // (α²J,   δJ)
   a->AddDomainIntegrator(new VectorMassIntegrator(a2_cf), nullptr,
                          TrialSpace::J_space, TrialSpace::J_space);
   // // (c²E,   δJ)
   a->AddDomainIntegrator(new VectorFEMassIntegrator(c2_cf), nullptr,
                          TrialSpace::E_space, TrialSpace::J_space);

   Array<int> offsets(3);
   offsets[0] = 0;
   offsets[1] = E_fes->GetVSize();
   offsets[2] = J_fes->GetVSize();
   offsets.PartialSum();

   Vector x(2*offsets.Last());
   x = 0.;
   Vector b(2*offsets.Last());
   real_t * bdata = b.GetData();

   ParLinearForm bE_r(E_fes, bdata);
   ParLinearForm bE_i(E_fes, &bdata[offsets.Last()]);
   ParLinearForm bJ_r(J_fes, &bdata[offsets[1]]);
   ParLinearForm bJ_i(J_fes, &bdata[offsets[2]]);

   VectorFunctionCoefficient f_rhs1_r(dim,rhs1_func_r);
   VectorFunctionCoefficient f_rhs1_i(dim,rhs1_func_i);

   bE_r.AddDomainIntegrator(new VectorFEDomainLFIntegrator(f_rhs1_r));
   bE_i.AddDomainIntegrator(new VectorFEDomainLFIntegrator(f_rhs1_i));



   VectorFunctionCoefficient f_rhs2_r(dim,rhs2_func_r);
   VectorFunctionCoefficient f_rhs2_i(dim,rhs2_func_i);

   bJ_r.AddDomainIntegrator(new VectorDomainLFIntegrator(f_rhs2_r));
   bJ_i.AddDomainIntegrator(new VectorDomainLFIntegrator(f_rhs2_i));
   bE_r.Assemble();
   bE_i.Assemble();
   bJ_r.Assemble();
   bJ_i.Assemble();


   socketstream E_out_r, E_out_i, Eex_out_r, Eex_out_i;
   socketstream J_out_r, J_out_i, Jex_out_r, Jex_out_i;


   ParGridFunction E_r(E_fes), E_i(E_fes);
   ParGridFunction J_r(J_fes), J_i(J_fes);
   ParGridFunction Eex_r(E_fes), Eex_i(E_fes);
   ParGridFunction Jex_r(J_fes), Jex_i(J_fes);


   a->Assemble();

   // cin.get();

   Array<int> ess_tdof_list;
   Array<int> ess_tdof_listE;
   Array<int> ess_tdof_listJ;
   Array<int> ess_bdr;
   if (pmesh.bdr_attributes.Size())
   {
      ess_bdr.SetSize(pmesh.bdr_attributes.Max());
      ess_bdr = 1;
      E_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_listE);
      J_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_listJ);
   }

   // shift the ess_tdofs
   for (int j = 0; j < ess_tdof_listJ.Size(); j++)
   {
      ess_tdof_listJ[j] += E_fes->GetTrueVSize();
   }
   ess_tdof_list.Append(ess_tdof_listE);
   ess_tdof_list.Append(ess_tdof_listJ);

   ParGridFunction E_gf_r(E_fes, x, offsets[0]); E_gf_r = 0.0;
   ParGridFunction E_gf_i(E_fes, x, offsets.Last()); E_gf_i = 0.0;
   VectorFunctionCoefficient Eex_cf_r(dim,E_exact_r);
   VectorFunctionCoefficient Eex_cf_i(dim,E_exact_i);
   E_gf_r.ProjectBdrCoefficientTangent(Eex_cf_r, ess_bdr);
   E_gf_i.ProjectBdrCoefficientTangent(Eex_cf_i, ess_bdr);

   VectorFunctionCoefficient Ecf_r(dim,E_exact_r);
   VectorFunctionCoefficient Ecf_i(dim,E_exact_i);
   VectorFunctionCoefficient Jcf_r(dim,J_exact_r);
   VectorFunctionCoefficient Jcf_i(dim,J_exact_i);

   ParGridFunction J_gf_r(J_fes, x, offsets[1]); J_gf_r = 0.0;
   ParGridFunction J_gf_i(J_fes, x, offsets.Last() + offsets[1]); J_gf_i = 0.0;
   J_gf_r.ProjectBdrCoefficient(Jcf_r,ess_bdr);
   J_gf_i.ProjectBdrCoefficient(Jcf_i,ess_bdr);

   OperatorPtr Ah;
   Vector X,B;
   a->FormLinearSystem(ess_tdof_list,x,b, Ah, X,B);

   ComplexOperator * Ahc = Ah.As<ComplexOperator>();

   BlockOperator * BlockA_r = dynamic_cast<BlockOperator *>(&Ahc->real());
   BlockOperator * BlockA_i = dynamic_cast<BlockOperator *>(&Ahc->imag());

   int nblocks = BlockA_r->NumRowBlocks();
   Array2D<HypreParMatrix*> A_r_matrices(nblocks, nblocks);
   Array2D<HypreParMatrix*> A_i_matrices(nblocks, nblocks);
   for (int i = 0; i < nblocks; i++)
   {
      for (int j = 0; j < nblocks; j++)
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
   HypreParMatrix *A = Ahc_hypre->GetSystemMatrix();
   // auto cpardiso = new CPardisoSolver(A->GetComm());
   auto solver = new MUMPSSolver(MPI_COMM_WORLD);
   solver->SetMatrixSymType(MUMPSSolver::MatType::UNSYMMETRIC);
   solver->SetPrintLevel(1);
   solver->SetOperator(*A);
   solver->Mult(B,X);
   delete A;
   delete solver;
#else
   MFEM_ABORT("MFEM compiled without mumps");
#endif

   a->RecoverFEMSolution(X, x);


   E_r.MakeRef(E_fes,x, 0);
   E_i.MakeRef(E_fes,x, offsets.Last());

   J_r.MakeRef(J_fes,x, offsets[1]);
   J_i.MakeRef(J_fes,x, offsets.Last()+offsets[1]);

   if (visualization)
   {
      // const char * keys = (it == 0 && dim == 2) ? "jRcml\n" : nullptr;
      const char * keys = nullptr;
      char vishost[] = "localhost";
      int  visport   = 19916;
      VisualizeField(E_out_r,vishost, visport, E_r,
                     "Numerical Electric field (real part)", 0, 0, 500, 500, keys);
      VisualizeField(J_out_r,vishost, visport, J_r,
                     "Numerical J field (real part)", 0, 0, 500, 500, keys);

   }

   delete a;

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

// F = ∇×(1/μ ∇×E) - ω² ϵ E + J
void  rhs1_func_r(const Vector &x, Vector & F_r)
{
   Vector E_r, curlcurlE_r, J_r;
   E_exact_r(x,E_r);
   curlcurlE_exact_r(x,curlcurlE_r);
   J_exact_r(x,J_r);
   F_r.SetSize(dim);
   for (int i = 0; i<dim; i++)
   {
      F_r(i) = 1.0/mu * curlcurlE_r(i)
               - omega * omega * epsilon * E_r(i)
               + J_r(i);
   }
}

void  rhs1_func_i(const Vector &x, Vector & F_i)
{
   Vector E_i, curlcurlE_i, J_i;
   E_exact_i(x,E_i);
   curlcurlE_exact_i(x,curlcurlE_i);
   J_exact_i(x,J_i);
   F_i.SetSize(dim);
   for (int i = 0; i<dim; i++)
   {
      F_i(i) = 1.0/mu * curlcurlE_i(i)
               - omega * omega * epsilon * E_i(i)
               + J_i(i);
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