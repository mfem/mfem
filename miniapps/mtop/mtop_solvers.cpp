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

using namespace mfem;

using mfem::future::dual;
using mfem::future::tuple;
using mfem::future::tensor;

using mfem::future::Weight;
using mfem::future::Gradient;
using mfem::future::Identity;

///////////////////////////////////////////////////////////////////////////////
/// \brief The QFunction struct defining the linear elasticity operator at
/// integration points which is valid in 2D and 3D
template <int DIM> struct QFunction
{
   using matd_t = tensor<real_t, DIM, DIM>;

   struct Elasticity
   {
      MFEM_HOST_DEVICE inline auto operator()(const matd_t &dudxi,
                                              const real_t &L, const real_t &M,
                                              const matd_t &J,
                                              const real_t &w) const
      {
         const matd_t JxW = transpose(inv(J)) * det(J) * w;
         constexpr auto I = mfem::future::IsotropicIdentity<DIM>();
         const auto eps = mfem::future::sym(dudxi * mfem::future::inv(J));
         return tuple{(L * tr(eps) * I + 2.0 * M * eps) * JxW};
      }
   };
};

IsoLinElasticSolver::IsoLinElasticSolver(ParMesh *mesh, int vorder,
                                         bool pa, bool dfem):
   pmesh(mesh),
   pa(pa),
   dfem(dfem),
   dim(mesh->Dimension()),
   spaceDim(mesh->SpaceDimension()),
   vfec(new H1_FECollection(vorder, dim)),
   vfes(new ParFiniteElementSpace(pmesh, vfec, dim,
                                  // PA Elasticity only implemented for byNODES ordering
                                  pa||dfem ? Ordering::byNODES : Ordering::byVDIM)),
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
   fe(vfes->GetFE(0)),
   nodes((pmesh->EnsureNodes(),
          static_cast<ParGridFunction *>(pmesh->GetNodes()))),
   mfes(nodes->ParFESpace()),
   ir(IntRules.Get(fe->GetGeomType(),
                   fe->GetOrder() + fe->GetOrder() + fe->GetDim() - 1)),
   qs(*pmesh, ir),
   Lambda_ps(*pmesh, ir, 1),
   Mu_ps(*pmesh, ir, 1),
   lf(nullptr)
{
   MFEM_VERIFY(qs.GetSize() == Lambda_ps.GetTrueVSize(),
               "QuadratureSpace and ParameterSpace size mismatch");

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

   if (pmesh->attributes.Size() > 0)
   {
      domain_attributes.SetSize(pmesh->attributes.Max());
      domain_attributes = 1;
   }
}

IsoLinElasticSolver::~IsoLinElasticSolver()
{
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

void IsoLinElasticSolver::SetLinearSolver(real_t rtol,
                                          real_t atol,
                                          int miter)
{
   linear_rtol = rtol;
   linear_atol = atol;
   linear_iter = miter;
}

void IsoLinElasticSolver::AddDispBC(int id, int dir,
                                    real_t val)
{
   if (dir == 0)
   {
      bcx[id] = ConstantCoefficient(val);
      AddDispBC(id, dir, bcx[id]);
   }
   else if (dir == 1)
   {
      bcy[id] = ConstantCoefficient(val);
      AddDispBC(id, dir, bcy[id]);
   }
   else if (dir == 2)
   {
      bcz[id] = ConstantCoefficient(val);
      AddDispBC(id, dir, bcz[id]);
   }
   else if (dir == -1)
   {
      bcx[id] = ConstantCoefficient(val);
      bcy[id] = ConstantCoefficient(val);
      bcz[id] = ConstantCoefficient(val);
      AddDispBC(id, 0, bcx[id]);
      AddDispBC(id, 1, bcy[id]);
      AddDispBC(id, 2, bcz[id]);
   }
   else
   {
      MFEM_ABORT("Invalid BC direction: "
                 "0(x), 1(y), 2(z), or -1(all), got " << dir);
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
   else if (dir == 1) { bccy[id] = &val; }
   else if (dir == 2) { bccz[id] = &val; }
   else if (dir == -1)
   {
      bccx[id] = &val;
      bccy[id] = &val;
      bccz[id] = &val;
   }
   else
   {
      MFEM_ABORT("Invalid BC direction: "
                 "0(x), 1(y), 2(z), or -1(all), got " << dir);
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


void IsoLinElasticSolver::SetEssTDofs(int j,
                                      ParFiniteElementSpace& scalar_space,
                                      Array<int> &ess_dofs)
{
   // Set the BC
   ess_dofs.DeleteAll();

   auto cbcc = &bccx;
   if (j == 1) { cbcc = &bccy; }
   else if (j == 2) { cbcc = &bccz; }

   Array<int> ess_bdr(pmesh->bdr_attributes.Max()); ess_bdr = 0;

   for (auto it = cbcc->begin(); it != cbcc->end(); it++)
   {
      ess_bdr[it->first - 1] = 1;
   }
   scalar_space.GetEssentialTrueDofs(ess_bdr,ess_dofs);
}


void IsoLinElasticSolver::SetEssTDofs(Vector &bsol, Array<int> &ess_dofs)
{
   // Set the BC
   ess_tdofv.DeleteAll();

   Array<int> ess_tdofx, ess_tdofy, ess_tdofz;

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
   // copy tdofsx from displacement grid function
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
   // copy tdofsy from displacement grid function
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

      // copy tdofsz from displacement grid function
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

   // the rhs x is assumed to have the contribution of the BC set in advance
   // the BC values are not modified here
   ls->Mult(x, y);

   int N = ess_tdofv.Size();
   real_t *yp = y.ReadWrite();
   const real_t *sp = sol.Read();
   const int *ep = ess_tdofv.Read();
   mfem::forall(N, [=] MFEM_HOST_DEVICE(int i) { yp[ep[i]] = sp[ep[i]]; });
}

void IsoLinElasticSolver::MultTranspose(const Vector &x,
                                        Vector &y) const
{
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
   delete bf; bf=nullptr;

   if (dfem)
   {
#ifdef MFEM_USE_DOUBLE
      // define the differentiable operator
      dop = std::make_unique<mfem::future::DifferentiableOperator>(
      std::vector<mfem::future::FieldDescriptor> {{ U, vfes }},
      std::vector<mfem::future::FieldDescriptor>
      {
         { LCoeff, &Lambda_ps},
         { MuCoeff, &Mu_ps},
         { Coords, mfes }
      },
      *pmesh);

      // sample lambda on the integration points
      Lambda_cv = std::make_unique<CoefficientVector>(*lambda, qs);

      // sample mu on the integration points
      Mu_cv = std::make_unique<CoefficientVector>(*mu, qs);

      // set the parameters of the differentiable operator
      dop->SetParameters({ Lambda_cv.get(), Mu_cv.get(), nodes });

      // define the q-function for dimensions 2 and 3
      const auto inputs =
         mfem::future::tuple{ Gradient<U>{},
                              Identity<LCoeff>{}, Identity<MuCoeff>{},
                              Gradient<Coords>{},
                              Weight{} };
      const auto output = mfem::future::tuple{ Gradient<U>{} };
      if (2 == spaceDim)
      {
         typename QFunction<2>::Elasticity e2qf;
         dop->AddDomainIntegrator(e2qf, inputs, output, ir, domain_attributes);
      }
      else if (3 == spaceDim)
      {
         typename QFunction<3>::Elasticity e3qf;
         dop->AddDomainIntegrator(e3qf, inputs, output, ir, domain_attributes);
      }
      else { MFEM_ABORT("Space dimension not supported"); }
#else
      MFEM_ABORT("Differentiable operator is only supported in double precision");
#endif
   }
   else
   {
      // define standard bilinear form
      bf = new mfem::ParBilinearForm(vfes);
      bf->AddDomainIntegrator(new mfem::ElasticityIntegrator(*lambda, *mu));
      if (pa) { bf->SetAssemblyLevel(mfem::AssemblyLevel::PARTIAL); }
   }

   // set BC
   sol = real_t(0.0);
   SetEssTDofs(sol, ess_tdofv);

   if (pa)
   {
      bf->Assemble();
      Operator *Kop;
      bf->FormSystemOperator(ess_tdofv, Kop);
      Kh = std::make_unique<OperatorHandle>(Kop);
      Kc = dynamic_cast<mfem::ConstrainedOperator*>(Kop);
   }
   else if (dfem)
   {
      Operator *Kop;
      dop->FormSystemOperator(ess_tdofv, Kop);
      Kh = std::make_unique<OperatorHandle>(Kop);
      Kc = dynamic_cast<mfem::ConstrainedOperator*>(Kop);
   }
   else
   {
      bf->Assemble(0);
      bf->Finalize();
      K.reset(bf->ParallelAssemble());
      Ke.reset(K->EliminateRowsCols(ess_tdofv));
   }

   if (ls == nullptr)
   {
      ls = new CGSolver(pmesh->GetComm());
      if (pa || dfem)
      {
         // PA LOR lor_disc & scalar_lor_fespace setup
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
            // set the essential boundaries
            Array<int> ess_tdof_list_block;
            SetEssTDofs(j,*lor_scalar_fespace,ess_tdof_list_block);
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
         lor_blockDiag =
            std::make_unique<BlockDiagonalPreconditioner>(lor_block_offsets);
         for (int i = 0; i < dim; i++)
         {
            lor_blockDiag->SetDiagonalBlock(i, lor_amg_blocks[i].get());
         }
         lor_pa_prec.reset(lor_blockDiag.release());
         ls->SetPreconditioner(*lor_pa_prec);
      }
      else
      {
         prec = new HypreBoomerAMG();
         // set the rigid body modes
         prec->SetElasticityOptions(vfes);
         ls->SetPreconditioner(*prec);
      }
      ls->SetOperator(((pa||dfem) ? *Kh->Ptr() : *K));
      ls->SetPrintLevel(1);
   }
   else
   {
      ls->SetOperator((pa||dfem) ? *Kh->Ptr() : *K);
   }
}

void IsoLinElasticSolver::FSolve()
{
   ls->SetAbsTol(linear_atol);
   ls->SetRelTol(linear_rtol);
   ls->SetMaxIter(linear_iter);

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

   if (pa || dfem) { lf->UseFastAssembly(true); }
   lf->Assemble();
   lf->ParallelAssemble(rhs);

   if (pa || dfem) { Kc->EliminateRHS(sol, rhs); }
   else
   {
      K->EliminateBC(*Ke, ess_tdofv, sol, rhs);
   }

   ls->Mult(rhs, sol);

   delete lf;
   lf = nullptr;
}

template <int DIM, typename scalar_t=real_t> struct FilterQFunction
{
   using matd_t = tensor<real_t, DIM, DIM>;
   using vecd_t = tensor<scalar_t, DIM>;


   struct Diffusion
   {
      real_t diff_coeff=1.0;

      void SetDiffusion(real_t val_)
      {
         diff_coeff=val_;
      }

      MFEM_HOST_DEVICE inline auto operator()(const vecd_t &dfdxi,
                                              //const real_t &diff,
                                              const matd_t &J,
                                              const real_t &w) const
      {
         const auto invJ = mfem::future::inv(J);
         const auto TinJ = mfem::future::transpose(invJ);
         const auto detJ = mfem::future::det(J);
         return tuple{ (dfdxi * invJ) * TinJ * detJ * w * diff_coeff};
      }
   };

   struct Mass
   {
      real_t density=1.0;

      void SetDensity(real_t val_)
      {
         density=val_;
      }

      MFEM_HOST_DEVICE inline auto operator()(const scalar_t &frho,
                                              //const real_t &diff,
                                              const matd_t &J,
                                              const real_t &w) const
      {
         const auto detJ = mfem::future::det(J);
         return tuple{ density * frho * detJ * w };
      }
   };


   struct RHS
   {
      MFEM_HOST_DEVICE inline auto operator()(const scalar_t &frho,
                                              const scalar_t &urho,
                                              //const real_t &diff,
                                              const matd_t &J,
                                              const real_t &w) const
      {
         const auto detJ = mfem::future::det(J);
         return tuple{ urho* detJ * w };
      }
   };
};

PDEFilter::PDEFilter(ParMesh *mesh, real_t r, int order):
   pmesh(mesh),
   dim(mesh->Dimension()),
   spaceDim(mesh->SpaceDimension()),
   filter_radius(r),
   ffec(new H1_FECollection(order, dim)),
   ffes(new ParFiniteElementSpace(pmesh, ffec, 1, Ordering::byNODES)),
   ifec(new L2_FECollection(order-1, dim)),
   ifes(new ParFiniteElementSpace(pmesh, ifec, 1, Ordering::byNODES)),
   filtered_field(ffes),
   input_field(ifes),
   h1_gradient(ffes),
   prec(nullptr),
   ls(nullptr),
   ess_tdofv(),
   fe(ffes->GetFE(0)),
   nodes((pmesh->EnsureNodes(),
          static_cast<ParGridFunction *>(pmesh->GetNodes()))),
   mfes(nodes->ParFESpace()),
   ir(IntRules.Get(fe->GetGeomType(),
                   fe->GetOrder() + fe->GetOrder() + fe->GetDim() - 1)),
   qs(*pmesh, ir),
   diff_ps(*pmesh, ir, 1),
   K(nullptr)
{
   filtered_field = 0.0;
   input_field = 0.0;   
   h1_gradient = 0.0;

   SetLinearSolver();

   Operator::width = ifes->GetTrueVSize();
   Operator::height = ffes->GetTrueVSize();


   if (pmesh->attributes.Size() > 0)
   {
      domain_attributes.SetSize(pmesh->attributes.Max());
      domain_attributes = 1;
   }
}

PDEFilter::PDEFilter(ParFiniteElementSpace *fespace, real_t r, int order):
   pmesh(fespace->GetParMesh()),
   dim(fespace->GetParMesh()->Dimension()),
   spaceDim(fespace->GetParMesh()->SpaceDimension()),
   filter_radius(r),
   ffec(new H1_FECollection(order, dim)),
   ffes(new ParFiniteElementSpace(pmesh, ffec, 1, Ordering::byNODES)),
   ifec(nullptr),
   ifes(new ParFiniteElementSpace(*fespace)),
   filtered_field(ffes),
   input_field(ifes),
   h1_gradient(ffes),
   prec(nullptr),
   ls(nullptr),
   ess_tdofv(),
   fe(ffes->GetFE(0)),
   nodes((pmesh->EnsureNodes(),
          static_cast<ParGridFunction *>(pmesh->GetNodes()))),
   mfes(nodes->ParFESpace()),
   ir(IntRules.Get(fe->GetGeomType(),
                   fe->GetOrder() + fe->GetOrder() + fe->GetDim() - 1)),
   qs(*pmesh, ir),
   diff_ps(*pmesh, ir, 1),
   K(nullptr)
{

   filtered_field = 0.0;
   input_field = 0.0;   
   h1_gradient = 0.0;

   SetLinearSolver();

   Operator::width = ifes->GetTrueVSize();
   Operator::height = ffes->GetTrueVSize();


   if (pmesh->attributes.Size() > 0)
   {
      domain_attributes.SetSize(pmesh->attributes.Max());
      domain_attributes = 1;
   }
}


PDEFilter::~PDEFilter()
{
   delete prec;
   delete ls;

   delete ffes;
   delete ffec;

   delete ifes;
   delete ifec;

   delete K;
}  

void PDEFilter::SetFilterRadius(real_t r)
{
   filter_radius = r;
}

void PDEFilter::SetLinearSolver(real_t rtol,
                                 real_t atol,
                                 int miter)
{
   linear_rtol = rtol;
   linear_atol = atol;
   linear_iter = miter;
}

void PDEFilter::Assemble()
{
   delete prec; prec=nullptr;
   delete ls; ls=nullptr;

   // define the differentiable operator
   // defined the matrix vector product
   dop= std::make_unique<mfem::future::DifferentiableOperator>(
      std::vector<mfem::future::FieldDescriptor> {{ FSol, ffes }},
      std::vector<mfem::future::FieldDescriptor>
      {
         //{ DiffCoeff, &diff_ps },
         { Coords, mfes }
      },
      *pmesh);

   ConstantCoefficient diffusion_coeff(filter_radius*filter_radius); 
   // sample filter coefficient on the integration points
   diff_cv = std::make_unique<CoefficientVector>(diffusion_coeff, qs);

   // set the parameters of the differentiable operator
   //dop->SetParameters({ diff_cv.get(), nodes });
   dop->SetParameters({nodes});

   // define the q-function for dimensions 2 and 3
   // diffusion term
   const auto dinputs =
      mfem::future::tuple{ mfem::future::Gradient<FSol>{},
                           //mfem::future::Identity<DiffCoeff>{},
                           mfem::future::Gradient<Coords>{},
                           mfem::future::Weight{} };
   // mass term
   const auto minputs =
      mfem::future::tuple{ mfem::future::Value<FSol>{},
                           //mfem::future::Identity<DiffCoeff>{},
                           mfem::future::Gradient<Coords>{},
                           mfem::future::Weight{} };

   // output of the diffusion term
   const auto doutput = mfem::future::tuple{ mfem::future::Gradient<FSol>{} };

   // output of the mass term
   const auto moutput = mfem::future::tuple{ mfem::future::Value<FSol>{} };  

   
   if (2 == spaceDim)
   {
     typename FilterQFunction<2>::Diffusion diff_qf;
     diff_qf.SetDiffusion(filter_radius*filter_radius);
     dop->AddDomainIntegrator(diff_qf, dinputs, doutput, ir, 
                               domain_attributes);

     typename FilterQFunction<2>::Mass mass_qf;
     mass_qf.SetDensity(1.0);
     dop->AddDomainIntegrator(mass_qf, minputs, moutput, ir, 
                                 domain_attributes);
   }
   else if (3 == spaceDim)
   {
      typename FilterQFunction<3>::Diffusion diff_qf;
      diff_qf.SetDiffusion(filter_radius*filter_radius);
      dop->AddDomainIntegrator(diff_qf, dinputs, doutput, ir,
                                 domain_attributes);

      typename FilterQFunction<3>::Mass mass_qf;
      mass_qf.SetDensity(1.0);
      dop->AddDomainIntegrator(mass_qf, minputs, moutput, ir, 
                                 domain_attributes);
   }
   else { MFEM_ABORT("Space dimension not supported"); }
   

   // set BC
   // TODO:: Do not forget to set the BCs here 

   Operator *Kop;
   dop->FormSystemOperator(ess_tdofv, Kop);
   Kh = std::make_unique<OperatorHandle>(Kop);
   Kc = dynamic_cast<mfem::ConstrainedOperator*>(Kop);

   // delete old assembled Jacobian if it exists
   delete K; K=nullptr;
   //LOR Preconditioner
   {
      std::unique_ptr<mfem::ParLORDiscretization> lor_disc;
      lor_disc = std::make_unique<ParLORDiscretization>(*ffes);
      ParFiniteElementSpace &lor_space = lor_disc->GetParFESpace();
      ParMesh &lor_mesh = *lor_space.GetParMesh();
      lor_mesh.EnsureNodes();
      ParGridFunction* lor_nodes=static_cast<ParGridFunction *>(lor_mesh.GetNodes());
      ParFiniteElementSpace* lor_nodes_fes = lor_nodes->ParFESpace();

      /*
      // Get the LOR Jacobian of the differentiable operator
      std::unique_ptr<mfem::future::DifferentiableOperator> dopd;
      // define the differentiable operator
      dopd= std::make_unique<mfem::future::DifferentiableOperator>(
         std::vector<mfem::future::FieldDescriptor> {{FSol, &lor_space}},
         //std::vector<mfem::future::FieldDescriptor> {{FSol, ffes}},
         std::vector<mfem::future::FieldDescriptor>
         {
            //{ DiffCoeff, &diff_ps },
            { Coords, lor_nodes_fes }
         },
         lor_mesh);

      // set the parameters of the differentiable operator
      //dopd->SetParameters({ diff_cv.get(), nodes });
      dopd->SetParameters({lor_nodes});
      
      auto derivatives = std::integer_sequence<size_t, FSol> {};
      // define the q-function for dimensions 2 and 3
      using mfem::future::dual;
      using dual_t = dual<real_t, real_t>;
      if (2 == spaceDim)
      {
         typename FilterQFunction<2,dual_t>::Diffusion diff_qfd;
         diff_qfd.SetDiffusion(filter_radius*filter_radius);
         dopd->AddDomainIntegrator(diff_qfd, dinputs, doutput, ir,
                                    domain_attributes, derivatives);

         typename FilterQFunction<2,dual_t>::Mass mass_qfd;
         mass_qfd.SetDensity(1.0);
         dopd->AddDomainIntegrator(mass_qfd, minputs, moutput, ir, 
                                    domain_attributes, derivatives);

         //typename FilterQFunction<2,dual_t>::Diffusion diff_qfd;
         //diff_qfd.SetDiffusion(filter_radius*filter_radius);
         //dopd->AddDomainIntegrator(diff_qfd, dinputs, doutput, ir,
         //                           domain_attributes, derivatives);
      }
      else if (3 == spaceDim)
      {
         typename FilterQFunction<3,dual_t>::Diffusion diff_qfd;
         diff_qfd.SetDiffusion(filter_radius*filter_radius);
         dopd->AddDomainIntegrator(diff_qfd, dinputs, doutput, ir,
                                    domain_attributes, derivatives);

         typename FilterQFunction<3,dual_t>::Mass mass_qfd;
         mass_qfd.SetDensity(1.0);
         dopd->AddDomainIntegrator(mass_qfd, minputs, moutput, ir, 
                                    domain_attributes, derivatives);
      }

      std::shared_ptr<mfem::future::DerivativeOperator> dres_du;
      // set parameters using grid functions
      ParGridFunction lor_gf(&lor_space); lor_gf=0.0;

      dres_du = dopd->GetDerivative(FSol, {&lor_gf}, 
                              { lor_nodes });

      // get the Jacobian
      dres_du->Assemble(K);
      */      

      
      ParBilinearForm bf_lor(&lor_space);
      ConstantCoefficient diff_coeff_lor(filter_radius*filter_radius);
      bf_lor.AddDomainIntegrator(new DiffusionIntegrator(diff_coeff_lor));
      bf_lor.AddDomainIntegrator(new MassIntegrator());
      bf_lor.Assemble();
      bf_lor.Finalize();
      K=bf_lor.ParallelAssemble();
      

      K->EliminateBC(ess_tdofv,Operator::DiagonalPolicy::DIAG_ONE);
   }

    // set the linear solver
   ls = new CGSolver(pmesh->GetComm());
   prec = new HypreBoomerAMG();
   ls->SetOperator(*Kh->Ptr());
   ls->SetPrintLevel(1); 
   ls->SetPreconditioner(*prec);

   ls->SetRelTol(linear_rtol);
   ls->SetAbsTol(linear_atol);
   ls->SetMaxIter(linear_iter);

   // set the preconditioner for the linear solver
   prec->SetOperator(*K);

   // defined the RHS operator
   {
      drh = std::make_unique<mfem::future::DifferentiableOperator>(
         std::vector<mfem::future::FieldDescriptor> {{ FSol, ffes }},
         std::vector<mfem::future::FieldDescriptor>
         {
            { USol, ifes },
            { Coords, mfes }
         },
         *pmesh);

      // set the parameters of the differentiable operator
      drh->SetParameters({&input_field,nodes});
      // define the q-function for dimensions 2 and 3
      // mass term
      const auto rhsinputs =
         mfem::future::tuple{ mfem::future::Value<FSol>{},
                              mfem::future::Value<USol>{},
                              mfem::future::Gradient<Coords>{},
                              mfem::future::Weight{} };
      // output of the mass term
      const auto rhsoutput = mfem::future::tuple{ mfem::future::Value<FSol>{} };

      if (2 == spaceDim)
      {
         typename FilterQFunction<2>::RHS rhs_qf;
         drh->AddDomainIntegrator(rhs_qf, rhsinputs, rhsoutput, ir,
                                    domain_attributes);
      }
      else if (3 == spaceDim)
      {
         typename FilterQFunction<3>::RHS rhs_qf;
         drh->AddDomainIntegrator(rhs_qf, rhsinputs, rhsoutput, ir,
                                    domain_attributes);
      }
      else { MFEM_ABORT("Space dimension not supported"); }

      Operator *Kop;
      drh->FormSystemOperator(ess_tdofv, Kop);
      Rh = std::make_unique<OperatorHandle>(Kop);
      Rc = dynamic_cast<mfem::ConstrainedOperator*>(Kop);
   }

}

void PDEFilter::Mult(const Vector &x, Vector &y) const
{
   input_field.SetFromTrueDofs(x);
   dop->SetParameters({ nodes });
   drh->SetParameters({ &input_field, nodes });
   rhs.SetSize(ffes->GetTrueVSize());
   drh->Mult(y, rhs);

   {
      ConstantCoefficient diffusion_coeff(filter_radius*filter_radius);
      ParBilinearForm bf(ffes);
      bf.AddDomainIntegrator(
         new mfem::DiffusionIntegrator(diffusion_coeff));
      bf.AddDomainIntegrator(new mfem::MassIntegrator());
      bf.Assemble();
      bf.Finalize();
      std::unique_ptr<mfem::HypreParMatrix> Kb(bf.ParallelAssemble());
      Kb->EliminateBC(ess_tdofv,Operator::DiagonalPolicy::DIAG_ONE);


      Vector ost(ffes->GetTrueVSize()); ost.Randomize();
      Vector tst1(ffes->GetTrueVSize());
      Vector tst2(ffes->GetTrueVSize());
      Kb->Mult(ost, tst1);
      dop->Mult(ost,tst2);

      real_t norm1 = 0.0;
      real_t norm2 = 0.0;
      norm1= InnerProduct(pmesh->GetComm(), tst1, tst1);
      norm2= InnerProduct(pmesh->GetComm(), tst2, tst2);

      if(0==pmesh->GetMyRank())
      {   
         std::cout<< "PDE Filter check: ||K*o|| = " << sqrt(norm1)
               << ", ||Dop*o|| = " << sqrt(norm2) << std::endl;
      }

   }

   // set BC to the solution vector y
   // set BC in the rhs
   // Kc->EliminateRHS(y, rhs);

   ls->SetRelTol(linear_rtol);
   ls->SetAbsTol(linear_atol);
   ls->SetMaxIter(linear_iter);

   ls->Mult(rhs, y);

   filtered_field.SetFromTrueDofs(y);
}

void PDEFilter::MultTranspose(const Vector &x,
                                        Vector &y) const
{
}


DFEMLinElasticSolver::DFEMLinElasticSolver(ParMesh *mesh, int vorder)
   : Operator(0, 0),
     pmesh(mesh),
     dim(mesh->Dimension()),
     spaceDim(mesh->SpaceDimension()),
     vfec(std::make_unique<H1_FECollection>(vorder, dim)),
     // dfem path in upstream forces Ordering::byNODES
     vfes(std::make_unique<ParFiniteElementSpace>(pmesh, vfec.get(), dim,
                                                  Ordering::byNODES)),
     fdisp(vfes.get())
{
   height = width = vfes->GetTrueVSize();
   sol.SetSize(width);
   rhs.SetSize(width);
   sol = 0.0;
   rhs = 0.0;

   // default body force = 0
   vol_force_vec.SetSize(spaceDim);
   vol_force_vec = 0.0;

}

DFEMLinElasticSolver::~DFEMLinElasticSolver()
{
   // clean any owned BC coefficients
   for (auto &kv : disp_bcs)
   {
      if (kv.second.owned && kv.second.coeff) { delete kv.second.coeff; }
   }
   disp_bcs.clear();

   delete ls;
   ls = nullptr;
}

void DFEMLinElasticSolver::SetLinearSolver(real_t rtol, real_t atol, int miter)
{
   linear_rtol = rtol;
   linear_atol = atol;
   linear_iter = miter;
}


void DFEMLinElasticSolver::SetVolForce(real_t fx, real_t fy, real_t fz)
{
   vol_force_vec.SetSize(spaceDim);
   vol_force_vec = 0.0;
   vol_force_vec[0] = fx;
   vol_force_vec[1] = fy;
   if (spaceDim == 3) { vol_force_vec[2] = fz; }

   volforce_owned = std::make_unique<VectorConstantCoefficient>(vol_force_vec);
   volforce = volforce_owned.get();
}

void DFEMLinElasticSolver::SetVolForce(VectorCoefficient &ff)
{
   volforce_owned.reset();
   volforce = &ff;
}

void DFEMLinElasticSolver::AddSurfLoad(int id, real_t fx, real_t fy, real_t fz)
{
 
}

void DFEMLinElasticSolver::AddSurfLoad(int id, VectorCoefficient &ff)
{

}

void DFEMLinElasticSolver::AddDispBC(int id, int dir, real_t val)
{
   auto *c = new ConstantCoefficient(val);
   disp_bcs.emplace(id, DispBC{dir, c, true});
}

void DFEMLinElasticSolver::AddDispBC(int id, int dir, Coefficient &val)
{
   disp_bcs.emplace(id, DispBC{dir, &val, false});
}

void DFEMLinElasticSolver::DelDispBC()
{
   for (auto &kv : disp_bcs)
   {
      if (kv.second.owned && kv.second.coeff) { delete kv.second.coeff; }
   }
   disp_bcs.clear();
}

void DFEMLinElasticSolver::SetEssTDofs(Vector &bsol, Array<int> &ess_dofs) const
{
  
}

void DFEMLinElasticSolver::Assemble()
{


}

void DFEMLinElasticSolver::FSolve()
{
   MFEM_VERIFY(ls != nullptr && Kc != nullptr, "Call Assemble() before FSolve().");

   ls->SetAbsTol(linear_atol);
   ls->SetRelTol(linear_rtol);
   ls->SetMaxIter(linear_iter);

   // Eliminate RHS using constrained operator (dfem path)
   Kc->EliminateRHS(sol, rhs);

   ls->Mult(rhs, sol);
}

void DFEMLinElasticSolver::Mult(const Vector &x, Vector &y) const
{
   MFEM_VERIFY(ls != nullptr, "Call Assemble() before Mult().");
   ls->Mult(x, y);

   int N = ess_tdofv.Size();
   real_t *yp = y.ReadWrite();
   const real_t *sp = sol.Read();
   const int *ep = ess_tdofv.Read();
   mfem::forall(N, [=] MFEM_HOST_DEVICE(int i) { yp[ep[i]] = sp[ep[i]]; });
}

void DFEMLinElasticSolver::MultTranspose(const Vector &x, Vector &y) const
{
   ls->Mult(x, y);

   int N = ess_tdofv.Size();
   ess_tdofv.Read();

   auto yp = y.Write();
   const auto ep = ess_tdofv.Read();

   mfem::forall(N, [=] MFEM_HOST_DEVICE(int i) { yp[ep[i]] = 0.0; });
}


