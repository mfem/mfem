// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include <memory>

#include "mtop_solvers.hpp"

#if defined(__has_include) && __has_include("general/nvtx.hpp") && !defined(_WIN32)
#undef NVTX_COLOR
#define NVTX_COLOR ::nvtx::kNvidia
#include "general/nvtx.hpp"
#else
#define dbg(...)
#endif

using namespace mfem;

IsoLinElasticSolver::IsoLinElasticSolver(ParMesh *mesh, int vorder, bool pa):
   pmesh(mesh),
   pa(pa),
   vorder(vorder),
   dim(mesh->Dimension()),
   spaceDim(mesh->SpaceDimension()),
   vfec(new H1_FECollection(vorder, dim)),
   vfes(new ParFiniteElementSpace(pmesh, vfec, dim,
                                  // Elasticity PA only implemented for byNODES ordering
                                  pa ? Ordering::byNODES :Ordering::byVDIM)),
   sol(vfes->GetTrueVSize()),
   adj(vfes->GetTrueVSize()),
   rhs(vfes->GetTrueVSize()),
   fdisp(vfes),
   adisp(vfes),
   prec(nullptr),
   ls(nullptr),
   lor_block_offsets(dim + 1),
   lvforce(nullptr),
   volforce(nullptr),
   E(nullptr),
   nu(nullptr),
   lambda(nullptr),
   mu(nullptr),
   bf(nullptr),
   lf(nullptr)
{
   dbg("sol size:{}", sol.Size());
   sol = 0.0;
   rhs = 0.0;
   adj = 0.0;

   fdisp = 0.0;
   adisp = 0.0;

   SetLinearSolver();

   Operator::width = vfes->GetTrueVSize();
   Operator::height = vfes->GetTrueVSize();

   lcsurf_load = std::make_unique<SurfaceLoad>(dim, load_coeff);
   glsurf_load = std::make_unique<SurfaceLoad>(dim, surf_loads);
}

IsoLinElasticSolver::~IsoLinElasticSolver()
{
   dbg();
   delete prec;
   delete ls;

   delete bf;
   delete lf;

   delete vfes;
   delete vfec;

   delete lvforce;

   for (auto it = load_coeff.begin(); it != load_coeff.end(); it++)
   {
      delete it->second;
   }

   delete lambda;
   delete mu;
}

void IsoLinElasticSolver::SetLinearSolver(const real_t rtol,
                                          const real_t atol,
                                          const int miter)
{
   linear_rtol = rtol;
   linear_atol = atol;
   linear_iter = miter;
}

void IsoLinElasticSolver::AddDispBC(const int id, const int dir,
                                    const real_t val)
{
   if (dir == 0)
   {
      bcx[id] = ConstantCoefficient(val);
      AddDispBC(id, dir, bcx[id]);
   }
   if (dir == 1)
   {
      bcy[id] = ConstantCoefficient(val);
      AddDispBC(id, dir, bcy[id]);
   }
   if (dir == 2)
   {
      bcz[id] = ConstantCoefficient(val);
      AddDispBC(id, dir, bcz[id]);
   }
   if (dir == 4)
   {
      bcx[id] = ConstantCoefficient(val);
      bcy[id] = ConstantCoefficient(val);
      bcz[id] = ConstantCoefficient(val);
      AddDispBC(id, 0, bcx[id]);
      AddDispBC(id, 1, bcy[id]);
      AddDispBC(id, 2, bcz[id]);
   }
}

void IsoLinElasticSolver::DelDispBC()
{
   bccx.clear();
   bccy.clear();
   bccz.clear();

   bcx.clear();
   bcy.clear();
   bcz.clear();

   ess_tdofv.DeleteAll();
}

void IsoLinElasticSolver::AddDispBC(int id, int dir, Coefficient &val)
{
   if (dir == 0) { bccx[id] = &val; }
   if (dir == 1) { bccy[id] = &val; }
   if (dir == 2) { bccz[id] = &val; }
   if (dir == 4)
   {
      bccx[id] = &val;
      bccy[id] = &val;
      bccz[id] = &val;
   }
   if (pmesh->Dimension() == 2) { bccz.clear(); }
}

void IsoLinElasticSolver::SetVolForce(real_t fx, real_t fy, real_t fz)
{
   delete lvforce;
   Vector ff(dim);
   ff(0) = fx;
   ff(1) = fy;
   if (dim == 3) { ff(2) = fz; }
   lvforce = new VectorConstantCoefficient(ff);
   volforce = lvforce;
}

void IsoLinElasticSolver::SetVolForce(VectorCoefficient &fv)
{
   volforce = &fv;
}

void IsoLinElasticSolver::SetEssTDofs(Vector &bsol, Array<int> &ess_dofs)
{
   // Set the BC
   ess_tdofv.DeleteAll();

   Array<int> ess_tdofx, ess_tdofy, ess_tdofz;

   dbg("X direction");
   for (auto it = bccx.begin(); it != bccx.end(); it++)
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 0;
      ess_bdr[it->first - 1] = 1;
      Array<int> ess_tdof_list;
      vfes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list, 0);
      ess_tdofx.Append(ess_tdof_list);

      VectorArrayCoefficient pcoeff(dim);
      pcoeff.Set(0, it->second, false);
      fdisp.ProjectBdrCoefficient(pcoeff, ess_bdr);
   }
   // copy tdofsx from velocity grid function
   {
      Vector &vc = fdisp.GetTrueVector();
      const int Net = ess_tdofx.Size();
      const auto d_vc = vc.Read();
      const auto d_ess_tdofx = ess_tdofx.Read();
      auto d_bsol = bsol.ReadWrite();
      mfem::forall(Net, [=] MFEM_HOST_DEVICE(int ii)
      {
         d_bsol[d_ess_tdofx[ii]] = d_vc[d_ess_tdofx[ii]];
      });
   }
   ess_tdofx.HostReadWrite(), ess_dofs.HostReadWrite();
   ess_dofs.Append(ess_tdofx);
   ess_tdofx.DeleteAll();

   dbg("Y direction");
   for (auto it = bccy.begin(); it != bccy.end(); it++)
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 0;
      ess_bdr[it->first - 1] = 1;
      Array<int> ess_tdof_list;
      vfes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list, 1);
      ess_tdofy.Append(ess_tdof_list);

      VectorArrayCoefficient pcoeff(dim);
      pcoeff.Set(1, it->second, false);
      fdisp.ProjectBdrCoefficient(pcoeff, ess_bdr);
   }
   // copy tdofsy from velocity grid function
   {
      Vector &vc = fdisp.GetTrueVector();
      const int Net = ess_tdofy.Size();
      const auto d_vc = vc.Read();
      const auto d_ess_tdofy = ess_tdofy.Read();
      auto d_bsol = bsol.ReadWrite();
      mfem::forall(Net, [=] MFEM_HOST_DEVICE(int ii)
      {
         d_bsol[d_ess_tdofy[ii]] = d_vc[d_ess_tdofy[ii]];
      });
      ess_tdofy.HostReadWrite(), ess_dofs.HostReadWrite();
   }
   ess_dofs.Append(ess_tdofy);
   ess_tdofy.DeleteAll();

   if (dim == 3)
   {
      dbg("Z direction");
      for (auto it = bccz.begin(); it != bccz.end(); it++)
      {
         Array<int> ess_bdr(pmesh->bdr_attributes.Max());
         ess_bdr = 0;
         ess_bdr[it->first - 1] = 1;
         Array<int> ess_tdof_list;
         vfes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list, 2);
         ess_tdofz.Append(ess_tdof_list);

         VectorArrayCoefficient pcoeff(dim);
         pcoeff.Set(2, it->second, false);
         fdisp.ProjectBdrCoefficient(pcoeff, ess_bdr);
      }

      // copy tdofsz from velocity grid function
      {
         Vector &vc = fdisp.GetTrueVector();
         for (int ii = 0; ii < ess_tdofz.Size(); ii++)
         {
            bsol[ess_tdofz[ii]] = vc[ess_tdofz[ii]];
         }
      }
      ess_dofs.Append(ess_tdofz);
      ess_tdofz.DeleteAll();
   }
}

void IsoLinElasticSolver::Mult(const Vector &x, Vector &y) const
{
   dbg();
   // the rhs x is assumed to have the contribution of the BC set in advance
   // the BC values are not modified here
   ls->Mult(x, y);

   int N = ess_tdofv.Size();
   real_t *yp = y.ReadWrite();
   const real_t *sp = sol.Read();
   const int *ep = ess_tdofv.Read();
   forall(N, [=] MFEM_HOST_DEVICE(int i) { yp[ep[i]] = sp[ep[i]]; });
}

void IsoLinElasticSolver::MultTranspose(const Vector &x,
                                        Vector &y) const
{
   dbg();
   // the adjoint rhs is assumed to be corrected for the BC
   // K is symmetric
   ls->Mult(x, y);

   int N = ess_tdofv.Size();
   ess_tdofv.Read();

   auto yp = y.Write();
   const auto ep = ess_tdofv.Read();

   mfem::forall(N, [=] MFEM_HOST_DEVICE(int i) { yp[ep[i]] = 0.0; });
}

void IsoLinElasticSolver::Assemble()
{
   dbg();
   if (bf == nullptr) { return; }

   // set BC
   sol = real_t(0.0);
   SetEssTDofs(sol, ess_tdofv);

   real_t nr = ParNormlp(sol, 2, pmesh->GetComm());
   if (pmesh->GetMyRank() == 0) { mfem::out << "sol=" << nr << std::endl; }

   if (pa)
   {
      dbg("Assemble/OperatorHandle/EliminateRowsCols");
      bf->Assemble();
      Operator *Kop;
      bf->FormSystemOperator(ess_tdofv, Kop);
      Kh = std::make_unique<OperatorHandle>(Kop);
      Kc = dynamic_cast<mfem::ConstrainedOperator*>(Kop);
      dbg("Kc size: {}x{}", Kc->Width(), Kc->Height());
   }
   else
   {
      dbg("Assemble/Finalize/ParallelAssemble/EliminateRowsCols");
      bf->Assemble(0);
      bf->Finalize();
      // bf->FormSystemMatrix(ess_tdofv,K);
      K.reset(bf->ParallelAssemble());
      Ke.reset(K->EliminateRowsCols(ess_tdofv));
   }

   if (ls == nullptr)
   {
      dbg("new CGSolver/HypreBoomerAMG");
      ls = new CGSolver(pmesh->GetComm());
      ls->SetAbsTol(linear_atol);
      ls->SetRelTol(linear_rtol);
      ls->SetMaxIter(linear_iter);
      if (!pa)
      {
         prec = new HypreBoomerAMG();
         // set the rigid body modes
         prec->SetElasticityOptions(vfes);
         prec->SetPrintLevel(1);
         ls->SetPreconditioner(*prec);
      }
      else
      {
#if 0
         auto prec_pa = new OperatorJacobiSmoother(*bf, ess_tdofv);
         ls->SetPreconditioner(*prec_pa);
#else
         // PA LOR lor_disc & scalar_lor_fespace setup
         dbg("new ParLORDiscretization");
         lor_disc = std::make_unique<ParLORDiscretization>(*vfes);
         ParFiniteElementSpace &lor_space = lor_disc->GetParFESpace();
         const FiniteElementCollection &lor_fec = *lor_space.FEColl();
         ParMesh &lor_mesh = *lor_space.GetParMesh();
         lor_scalar_fespace = std::make_unique<ParFiniteElementSpace>(
                                 &lor_mesh, &lor_fec, 1, Ordering::byNODES);
         lor_block_offsets[0] = 0;
         lor_integrator = std::make_unique<ElasticityIntegrator>(*lambda, *mu);
         lor_integrator->AssemblePA(lor_disc->GetParFESpace());

         for (int j = 0; j < dim; j++)
         {
            auto *block = new ElasticityComponentIntegrator(*lor_integrator, j, j);
            // create the LOR matrix and corresponding AMG preconditioners.
            lor_bilinear_forms.emplace_back(new ParBilinearForm(lor_scalar_fespace.get()));
            lor_bilinear_forms[j]->SetAssemblyLevel(AssemblyLevel::FULL);
            lor_bilinear_forms[j]->EnableSparseMatrixSorting(Device::IsEnabled());
            lor_bilinear_forms[j]->AddDomainIntegrator(block);
            lor_bilinear_forms[j]->Assemble();
            // get block essential boundary info
            Array<int> ess_tdof_list_block, ess_bdr_block(pmesh->bdr_attributes.Max());
            ess_bdr_block = 0;
            ess_bdr_block[0] = 1;
            lor_scalar_fespace->GetEssentialTrueDofs(ess_bdr_block, ess_tdof_list_block);
            lor_block.emplace_back(lor_bilinear_forms[j]->ParallelAssemble());
            lor_block[j]->EliminateBC(ess_tdof_list_block,
                                      Operator::DiagonalPolicy::DIAG_ONE);
            lor_amg_blocks.emplace_back(new HypreBoomerAMG);
            lor_amg_blocks[j]->SetStrengthThresh(0.25);
            lor_amg_blocks[j]->SetRelaxType(16);  // Chebyshev
            lor_amg_blocks[j]->SetOperator(*lor_block[j]);
            lor_block_offsets[j+1] = lor_amg_blocks[j]->Height();
         }

         lor_block_offsets.PartialSum();
         lor_blockDiag = std::make_unique<BlockDiagonalPreconditioner>
                         (lor_block_offsets);
         for (int i = 0; i < dim; i++)
         {
            lor_blockDiag->SetDiagonalBlock(i, lor_amg_blocks[i].get());
         }
         lor_pa_prec.reset(lor_blockDiag.release());
         ls->SetPreconditioner(*lor_pa_prec);
#endif
      }
      ls->SetOperator(pa ? *Kh->Ptr() : *K);
      ls->SetPrintLevel(1);
   }
   else
   {
      ls->SetOperator(pa ? *Kh->Ptr() : *K);
   }

   mfem::out << pmesh->GetMyRank() << " LSW=" << ls->Width()
             << " LSH=" << ls->Height() << " KFW=" << (pa?Kc->Width():K->Width())
             << " KFH=" << (pa?Kc->Height():K->Height()) << std::endl;
}

void IsoLinElasticSolver::FSolve()
{
   dbg();
   if (lf == nullptr)
   {
      lf = new ParLinearForm(vfes);
      if (volforce != nullptr)
      {
         lf->AddDomainIntegrator(new VectorDomainLFIntegrator(*volforce));
      }
      // add surface loads
      lf->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(*lcsurf_load));
   }

   (*lf) = real_t(0.0);

   if (pa) { lf->UseFastAssembly(true); }
   lf->Assemble();
   lf->ParallelAssemble(rhs);

   /*
   Ke->AddMult(sol,rhs,1.0);
   for(int i=0;i<ess_tdofv.Size();i++){
       rhs[ess_tdofv[i]]=-sol[ess_tdofv[i]];
   }*/

   if (!pa)
   {
      K->EliminateBC(*Ke, ess_tdofv, sol, rhs);
   }
   else
   {
      Kc->EliminateRHS(sol, rhs);
   }

   {
      real_t nr = ParNormlp(rhs, 2, pmesh->GetComm());
      if (Mpi::Root()) { mfem::out << "rhs=" << nr << std::endl; }
   }

   ls->Mult(rhs, sol);

   delete lf;
   lf = nullptr;
}
