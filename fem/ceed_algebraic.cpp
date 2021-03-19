// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "ceed_algebraic.hpp"

#ifdef MFEM_USE_CEED
#include "../fem/bilinearform.hpp"
#include "../fem/fespace.hpp"
#include "../fem/libceed/ceedsolvers-atpmg.h"
#include "../fem/libceed/ceedsolvers-interpolation.h"
#include "../fem/libceed/ceed-assemble.hpp"
#include "../fem/libceed/ceedsolvers-qcoarsen.h"
#include "../fem/libceed/ceedsolvers-sparsify.h"
#include "../fem/pfespace.hpp"

namespace mfem
{

Solver *BuildSmootherFromCeed(MFEMCeedOperator &op, bool chebyshev)
{
   CeedOperator ceed_op = op.GetCeedOperator();
   const Array<int> &ess_tdofs = op.GetEssentialTrueDofs();
   const Operator *P = op.GetProlongation();
   // Assemble the a local diagonal, in the sense of L-vector
   CeedVector diagceed;
   CeedInt length;
   CeedOperatorGetSize(ceed_op, &length);
   CeedVectorCreate(internal::ceed, length, &diagceed);
   CeedMemType mem;
   CeedGetPreferredMemType(internal::ceed, &mem);
   if (!Device::Allows(Backend::CUDA) || mem != CEED_MEM_DEVICE)
   {
      mem = CEED_MEM_HOST;
   }
   Vector local_diag(length);
   CeedScalar *ptr = (mem == CEED_MEM_HOST) ? local_diag.HostWrite() :
                     local_diag.Write(true);
   CeedVectorSetArray(diagceed, mem, CEED_USE_POINTER, ptr);
   CeedOperatorLinearAssembleDiagonal(ceed_op, diagceed, CEED_REQUEST_IMMEDIATE);
   CeedVectorTakeArray(diagceed, mem, NULL);

   Vector t_diag;
   if (P)
   {
      t_diag.SetSize(P->Width());
      P->MultTranspose(local_diag, t_diag);
   }
   else
   {
      t_diag.NewMemoryAndSize(local_diag.GetMemory(), length, false);
   }
   Solver *out = NULL;
   if (chebyshev)
   {
      const int cheb_order = 3;
      out = new OperatorChebyshevSmoother(&op, t_diag, ess_tdofs, cheb_order);
   }
   else
   {
      const double jacobi_scale = 0.65;
      out = new OperatorJacobiSmoother(t_diag, ess_tdofs, jacobi_scale);
   }
   CeedVectorDestroy(&diagceed);
   return out;
}

#ifdef MFEM_USE_MPI

class CeedAMG : public Solver
{
public:
   CeedAMG(MFEMCeedOperator &oper, HypreParMatrix *P, bool amgx=false,
           const std::string amgx_config_file="")
   {
      MFEM_ASSERT(P != NULL, "");
      const Array<int> ess_tdofs = oper.GetEssentialTrueDofs();
      height = width = oper.Height();

      CeedOperatorFullAssemble(oper.GetCeedOperator(), &mat_local);

      {
         HypreParMatrix hypre_local(
            P->GetComm(), P->GetGlobalNumRows(), P->RowPart(), mat_local);
         op_assembled = RAP(&hypre_local, P);
      }
      HypreParMatrix *mat_e = op_assembled->EliminateRowsCols(ess_tdofs);
      delete mat_e;

#ifdef MFEM_USE_AMGX
      if (amgx)
      {
         if (amgx_config_file == "")
         {
            bool amgx_verbose = false;
            amg = new AmgXSolver(op_assembled->GetComm(),
                                 AmgXSolver::PRECONDITIONER, amgx_verbose);
         }
         else
         {
            AmgXSolver * amgx_prec = new AmgXSolver;
            amgx_prec->ReadParameters(amgx_config_file, AmgXSolver::EXTERNAL);
            amgx_prec->InitExclusiveGPU(MPI_COMM_WORLD);
            amgx_prec->SetOperator(*op_assembled);
            amg = amgx_prec;
         }
      }
      else
#endif
      {
         HypreBoomerAMG * hypre_amg = new HypreBoomerAMG(*op_assembled);
         hypre_amg->SetPrintLevel(0);
         amg = hypre_amg;
      }
   }

   void SetOperator(const Operator &op) { amg->SetOperator(op); }
   void Mult(const Vector &x, Vector &y) const { amg->Mult(x, y); }

   ~CeedAMG()
   {
      delete op_assembled;
      delete amg;
      delete mat_local;
   }

private:
   SparseMatrix *mat_local;
   HypreParMatrix *op_assembled;
   Solver *amg;
};

/**
   Too much copied code; this and CeedAMG should probably inherit
   from a common base class.
*/
class CeedSparsifyAMG : public Solver
{
public:
   CeedSparsifyAMG(MFEMCeedOperator &oper, HypreParMatrix *P, bool amgx=false)
   {
      MFEM_ASSERT(P != NULL, "");
      const Array<int> ess_tdofs = oper.GetEssentialTrueDofs();
      height = width = oper.Height();

      CeedSparsifySimple(oper.GetCeedOperator(), &sparse_basis, &sparse_oper);
      CeedOperatorFullAssemble(sparse_oper, &mat_local);

      {
         HypreParMatrix hypre_local(
            P->GetComm(), P->GetGlobalNumRows(), P->RowPart(), mat_local);
         op_assembled = RAP(&hypre_local, P);
      }
      HypreParMatrix *mat_e = op_assembled->EliminateRowsCols(ess_tdofs);
      delete mat_e;

#ifdef MFEM_USE_AMGX
      if (amgx)
      {
         bool amgx_verbose = false;
         amg = new AmgXSolver(op_assembled->GetComm(),
                              AmgXSolver::PRECONDITIONER, amgx_verbose);
         amg->SetOperator(*op_assembled);
      }
      else
#endif
      {
         HypreBoomerAMG * hypre_amg = new HypreBoomerAMG(*op_assembled);
         hypre_amg->SetPrintLevel(0);
         amg = hypre_amg;
      }
   }

   void SetOperator(const Operator &op) { amg->SetOperator(op); }
   void Mult(const Vector &x, Vector &y) const { amg->Mult(x, y); }

   ~CeedSparsifyAMG()
   {
      CeedBasisDestroy(&sparse_basis);
      CeedOperatorDestroy(&sparse_oper);

      delete op_assembled;
      delete amg;
      delete mat_local;
   }

private:
   CeedOperator sparse_oper;
   CeedBasis sparse_basis;
   SparseMatrix *mat_local;
   HypreParMatrix *op_assembled;
   Solver *amg;
};

#endif

void CoarsenEssentialDofs(const Operator &interp,
                          const Array<int> &ho_ess_tdofs,
                          Array<int> &alg_lo_ess_tdofs)
{
   Vector ho_boundary_ones(interp.Height());
   ho_boundary_ones = 0.0;
   const int *ho_ess_tdofs_h = ho_ess_tdofs.HostRead();
   for (int i=0; i<ho_ess_tdofs.Size(); ++i)
   {
      ho_boundary_ones[ho_ess_tdofs_h[i]] = 1.0;
   }
   Vector lo_boundary_ones(interp.Width());
   interp.MultTranspose(ho_boundary_ones, lo_boundary_ones);
   auto lobo = lo_boundary_ones.HostRead();
   for (int i = 0; i < lo_boundary_ones.Size(); ++i)
   {
      if (lobo[i] > 0.9)
      {
         alg_lo_ess_tdofs.Append(i);
      }
   }
}

template <typename INTEG>
int TryToAddCeedSubOperator(BilinearFormIntegrator *integ_in, CeedOperator op)
{
   INTEG *integ = dynamic_cast<INTEG*>(integ_in);
   if (integ != NULL)
   {
      CeedCompositeOperatorAddSub(op, integ->GetCeedData()->oper);
      return 1;
   }
   return 0;
}

CeedOperator CreateCeedCompositeOperatorFromBilinearForm(BilinearForm &form)
{
   CeedOperator op;
   CeedCompositeOperatorCreate(internal::ceed, &op);

   // Get the domain bilinear form integrators (DBFIs)
   Array<BilinearFormIntegrator*> *bffis = form.GetDBFI();
   int num_integrators = bffis->Size();

   int count = 0;
   for (int i = 0; i < num_integrators; ++i)
   {
      BilinearFormIntegrator *integ = (*bffis)[i];
      count += TryToAddCeedSubOperator<DiffusionIntegrator>(integ, op);
      count += TryToAddCeedSubOperator<MassIntegrator>(integ, op);
      count += TryToAddCeedSubOperator<VectorDiffusionIntegrator>(integ, op);
      count += TryToAddCeedSubOperator<VectorMassIntegrator>(integ, op);
   }
   if (count != num_integrators)
   {
      mfem_error("Some integrator does not support Ceed!");
   }
   return op;
}

CeedOperator CoarsenCeedCompositeOperator(
   CeedOperator op, CeedElemRestriction er,
   CeedBasis c2f, int order_reduction,
   int qorder_reduction,
   CeedQuadMode fine_qmode, CeedQuadMode coarse_qmode
)
{
   bool isComposite;
   CeedOperatorIsComposite(op, &isComposite);
   MFEM_ASSERT(isComposite, "");

   CeedOperator op_coarse;
   CeedCompositeOperatorCreate(internal::ceed, &op_coarse);

   int nsub;
   CeedOperatorGetNumSub(op, &nsub);
   CeedOperator *subops;
   CeedOperatorGetSubList(op, &subops);
   for (int isub=0; isub<nsub; ++isub)
   {
      CeedOperator subop = subops[isub];
      CeedBasis basis_coarse, basis_c2f;
      CeedOperator subop_coarse, t_subop_coarse;
      CeedATPMGOperator(subop, order_reduction, er, &basis_coarse, &basis_c2f,
                        &t_subop_coarse);
      if (qorder_reduction == 0)
      {
         subop_coarse = t_subop_coarse;
      }
      else
      {
         CeedVector qcoarsen_assembledqf;
         CeedQFunctionContext qcoarsen_context;
         CeedOperatorQCoarsen(t_subop_coarse, qorder_reduction, &subop_coarse,
                              &qcoarsen_assembledqf, &qcoarsen_context,
                              fine_qmode, coarse_qmode);
         CeedVectorDestroy(&qcoarsen_assembledqf); // todo: delete inside previous function?
         CeedQFunctionContextDestroy(&qcoarsen_context);
         CeedOperatorDestroy(&t_subop_coarse);
      }
      CeedBasisDestroy(&basis_coarse); // refcounted by subop_coarse
      CeedBasisDestroy(&basis_c2f);
      CeedCompositeOperatorAddSub(op_coarse, subop_coarse);
      CeedOperatorDestroy(&subop_coarse); // refcounted by composite operator
   }
   return op_coarse;
}

AlgebraicCeedMultigrid::AlgebraicCeedMultigrid(
   AlgebraicSpaceHierarchy &hierarchy,
   BilinearForm &form,
   const Array<int> &ess_tdofs,
   int print_level,
   double contrast_threshold,
   int switch_amg_order,
   bool collocate_coarse,
   bool sparsification,
   const std::string amgx_config_file
) : GeometricMultigrid(hierarchy)
{
   // Construct finest level
   ceed_operators.Prepend(CreateCeedCompositeOperatorFromBilinearForm(form));
   essentialTrueDofs.Prepend(new Array<int>);
   *essentialTrueDofs[0] = ess_tdofs;

   int current_order = hierarchy.GetFESpaceAtLevel(0).GetOrder(0);

   // Construct interpolation, operators, at all levels of hierarchy by coarsening
   int level_counter = 0;
   while (current_order > 1)
   {
      double minq, maxq, absmin;
      CeedOperatorGetHeuristics(ceed_operators[0], &minq, &maxq, &absmin);
      // TODO: in principle we need to communicate heuristics across
      // processors!
      double heuristic = std::max(std::abs(minq), std::abs(maxq)) / absmin;

      int order_reduction;
      if (heuristic > contrast_threshold && current_order <= switch_amg_order)
      {
         // assemble at this level
         break;
      }
      else if (heuristic > contrast_threshold || current_order == 3)
      {
         // coarsening directly from 3 to 1 appears to be bad
         order_reduction = 1;
      }
      else
      {
         order_reduction = current_order - (current_order/2);
      }
      if (print_level > 0)
      {
         std::cout << "  lc: " << level_counter << " heuristic = " << heuristic
                   << ", coarsening from order " << current_order 
                   << " to " << current_order - order_reduction << std::endl;
      }
      hierarchy.PrependPCoarsenedLevel(current_order, order_reduction);
      current_order = current_order - order_reduction;

      AlgebraicCoarseSpace &space = hierarchy.GetAlgebraicCoarseSpace(0);
      // int qor = (level_counter == 0) ? order_reduction + 1 : order_reduction;
      int qor = order_reduction;
      if (collocate_coarse)
      {
         if (level_counter == 0)
         {
            ceed_operators.Prepend(
               CoarsenCeedCompositeOperator(
                  ceed_operators[0], space.GetCeedElemRestriction(),
                  space.GetCeedCoarseToFine(), space.GetOrderReduction(),
                  qor, CEED_GAUSS, CEED_GAUSS_LOBATTO)
               );
         }
         else
         {
            ceed_operators.Prepend(
               CoarsenCeedCompositeOperator(
                  ceed_operators[0], space.GetCeedElemRestriction(),
                  space.GetCeedCoarseToFine(), space.GetOrderReduction(),
                  qor, CEED_GAUSS_LOBATTO, CEED_GAUSS_LOBATTO)
               );
         }
      }
      else
      {
         ceed_operators.Prepend(
            CoarsenCeedCompositeOperator(
               ceed_operators[0], space.GetCeedElemRestriction(),
               space.GetCeedCoarseToFine(), space.GetOrderReduction(),
               qor, CEED_GAUSS, CEED_GAUSS)
            );
      }

      Operator *P = hierarchy.GetProlongationAtLevel(0);
      essentialTrueDofs.Prepend(new Array<int>);
      CoarsenEssentialDofs(*P, *essentialTrueDofs[1],
                           *essentialTrueDofs[0]);

      level_counter++;
   }

   int nlevels = fespaces.GetNumLevels();
   // Add the operators and smoothers to the hierarchy, from coarse to fine
   for (int ilevel=0; ilevel<nlevels; ++ilevel)
   {
      FiniteElementSpace &space = hierarchy.GetFESpaceAtLevel(ilevel);
      const Operator *P = space.GetProlongationMatrix();
      MFEMCeedOperator *op = new MFEMCeedOperator(
         ceed_operators[ilevel], *essentialTrueDofs[ilevel], P);
      Solver *smoother;
      if (ilevel != 0)
      {
         smoother = BuildSmootherFromCeed(*op, true);
      }
      else
      {
         bool assemble_matrix = false;
#ifdef MFEM_USE_MPI
#ifdef MFEM_USE_AMGX
	 assemble_matrix = true;
#else
         if (!Device::Allows(Backend::CUDA)) { assemble_matrix = true; }
#endif
         HypreParMatrix *P_mat = NULL;
         if (assemble_matrix)
         {
            if (nlevels == 1)
            {
               // Only one level -- no coarsening, finest level
               ParFiniteElementSpace *pfes
                  = dynamic_cast<ParFiniteElementSpace*>(&space);
               if (pfes) { P_mat = pfes->Dof_TrueDof_Matrix(); }
            }
            else
            {
               ParAlgebraicCoarseSpace *pspace
                  = dynamic_cast<ParAlgebraicCoarseSpace*>(&space);
               if (pspace) { P_mat = pspace->GetProlongationHypreParMatrix(); }
            }
         }
         if (P_mat)
         {
            if (current_order > 1 && sparsification)
            {
               if (print_level >= 1)
               {
                  std::cout << "      sparsify AMG." << std::endl;
               }
               smoother = new CeedSparsifyAMG(*op, P_mat, Device::Allows(Backend::CUDA));
            }
            else
            {
               if (print_level >= 1)
               {
                  std::cout << "      no-sparsify AMG." << std::endl;
               }
               smoother = new CeedAMG(*op, P_mat, Device::Allows(Backend::CUDA),
                                      amgx_config_file);
            }
         }
         else
#endif
         {
            smoother = BuildSmootherFromCeed(*op, true);
         }
      }
      AddLevel(op, smoother, true, true);
   }
}

AlgebraicCeedMultigrid::~AlgebraicCeedMultigrid()
{
   for (int i=0; i<ceed_operators.Size(); ++i)
   {
      CeedOperatorDestroy(&ceed_operators[i]);
   }
}

void AlgebraicSpaceHierarchy::AddCoarseLevel(AlgebraicCoarseSpace* space,
                                             CeedElemRestriction er)
{
   MFEM_VERIFY(meshes.Size() >= 1, "At least one level must exist!");
   Mesh* finemesh = meshes[0];
   meshes.Prepend(finemesh); // every entry of meshes points to finest mesh
   ownedMeshes.Prepend(false);
   fespaces.Prepend(space);
   ownedFES.Prepend(true); // owns all but finest
   ceed_interpolations.Prepend(new MFEMCeedInterpolation(
                                  internal::ceed,
                                  space->GetCeedCoarseToFine(),
                                  space->GetCeedElemRestriction(),
                                  er)
      );
   const SparseMatrix *R = fespaces[1]->GetRestrictionMatrix();
   if (R)
   {
      R->BuildTranspose();
      R_tr.Prepend(new TransposeOperator(*R));
   }
   else
   {
      R_tr.Prepend(NULL);
   }
   prolongations.Prepend(ceed_interpolations[0]->SetupRAP(
                            space->GetProlongationMatrix(), R_tr[0]));
   ownedProlongations.Prepend(prolongations[0] != ceed_interpolations[0]);   
}

// the ifdefs and dynamic casts are very ugly, but the interface is kinda nice?
void AlgebraicSpaceHierarchy::PrependPCoarsenedLevel(
   int current_order, int order_reduction)
{
   MFEM_VERIFY(fespaces.Size() >= 1, "At least one level must exist!");
   int dim = meshes[0]->Dimension();

   AlgebraicCoarseSpace *fine_alg_space =
      dynamic_cast<AlgebraicCoarseSpace*>(fespaces[0]);
   CeedElemRestriction current_er;
#ifdef MFEM_USE_MPI
   GroupCommunicator *gc = NULL;
#endif
   if (fine_alg_space)
   {
      current_er = fine_alg_space->GetCeedElemRestriction();
#ifdef MFEM_USE_MPI
      ParAlgebraicCoarseSpace *par_alg_space =
         dynamic_cast<ParAlgebraicCoarseSpace*>(fine_alg_space);
      if (par_alg_space) { gc = par_alg_space->GetGroupCommunicator(); }
#endif
   }
   else
   {
      current_er = fine_er;
#ifdef MFEM_USE_MPI
      ParFiniteElementSpace *pfes =
         dynamic_cast<ParFiniteElementSpace*>(fespaces[0]);
      if (pfes) { gc = &pfes->GroupComm(); }
#endif
   }
   AlgebraicCoarseSpace *space;
#ifdef MFEM_USE_MPI
   if (gc)
   {
      space = new ParAlgebraicCoarseSpace(
         *fespaces[0], current_er, current_order, dim, order_reduction, gc);
   }
   else
#endif
   {
      space = new AlgebraicCoarseSpace(
         *fespaces[0], current_er, current_order, dim, order_reduction);
   }
   AddCoarseLevel(space, current_er);
}

AlgebraicSpaceHierarchy::AlgebraicSpaceHierarchy(FiniteElementSpace &fes)
{
   int order = fes.GetOrder(0);

   meshes.Prepend(fes.GetMesh());
   ownedMeshes.Prepend(false);
   fespaces.Prepend(&fes);
   ownedFES.Prepend(false);

   Ceed ceed = internal::ceed;
   InitCeedTensorRestriction(fes, ceed, &fine_er);
}

AlgebraicCoarseSpace::AlgebraicCoarseSpace(
   FiniteElementSpace &fine_fes,
   CeedElemRestriction fine_er,
   int order,
   int dim,
   int order_reduction_
) : order_reduction(order_reduction_)
{
   order_reduction = order_reduction_;

   CeedATPMGElemRestriction(order, order_reduction, fine_er,
                            &ceed_elem_restriction, dof_map );
   CeedBasisATPMGCoarseToFine(internal::ceed, order+1, dim,
                              order_reduction, &coarse_to_fine );
   CeedElemRestrictionGetLVectorSize(ceed_elem_restriction, &ndofs);
   mesh = fine_fes.GetMesh();
}

AlgebraicCoarseSpace::~AlgebraicCoarseSpace()
{
   free(dof_map);
   CeedBasisDestroy(&coarse_to_fine);
   CeedElemRestrictionDestroy(&ceed_elem_restriction);
}

#ifdef MFEM_USE_MPI

ParAlgebraicCoarseSpace::ParAlgebraicCoarseSpace(
   FiniteElementSpace &fine_fes,
   CeedElemRestriction fine_er,
   int order,
   int dim,
   int order_reduction_,
   GroupCommunicator *gc_fine)
   : AlgebraicCoarseSpace(fine_fes, fine_er, order, dim, order_reduction_)
{
   int lsize;
   CeedElemRestrictionGetLVectorSize(ceed_elem_restriction, &lsize);
   const Table &group_ldof_fine = gc_fine->GroupLDofTable();

   ldof_group.SetSize(lsize);
   ldof_group = 0;

   GroupTopology &group_topo = gc_fine->GetGroupTopology();
   gc = new GroupCommunicator(group_topo);
   Table &group_ldof = gc->GroupLDofTable();
   group_ldof.MakeI(group_ldof_fine.Size());
   for (int g=1; g<group_ldof_fine.Size(); ++g)
   {
      int nldof_fine_g = group_ldof_fine.RowSize(g);
      const int *ldof_fine_g = group_ldof_fine.GetRow(g);
      for (int i=0; i<nldof_fine_g; ++i)
      {
         int icoarse = dof_map[ldof_fine_g[i]];
         if (icoarse >= 0)
         {
            group_ldof.AddAColumnInRow(g);
            ldof_group[icoarse] = g;
         }
      }
   }
   group_ldof.MakeJ();
   for (int g=1; g<group_ldof_fine.Size(); ++g)
   {
      int nldof_fine_g = group_ldof_fine.RowSize(g);
      const int *ldof_fine_g = group_ldof_fine.GetRow(g);
      for (int i=0; i<nldof_fine_g; ++i)
      {
         int icoarse = dof_map[ldof_fine_g[i]];
         if (icoarse >= 0)
         {
            group_ldof.AddConnection(g, icoarse);
         }
      }
   }
   group_ldof.ShiftUpI();
   gc->Finalize();
   ldof_ltdof.SetSize(lsize);
   ldof_ltdof = -2;
   int ltsize = 0;
   for (int i=0; i<lsize; ++i)
   {
      int g = ldof_group[i];
      if (group_topo.IAmMaster(g))
      {
         ldof_ltdof[i] = ltsize;
         ++ltsize;
      }
   }
   gc->SetLTDofTable(ldof_ltdof);
   gc->Bcast(ldof_ltdof);

   R_mat = new SparseMatrix(ltsize, lsize);
   for (int j=0; j<lsize; ++j)
   {
      if (group_topo.IAmMaster(ldof_group[j]))
      {
         int i = ldof_ltdof[j];
         R_mat->Set(i,j,1.0);
      }
   }
   R_mat->Finalize();

   if (Device::Allows(Backend::DEVICE_MASK))
   {
      P = new DeviceConformingProlongationOperator(*gc, R_mat);
   }
   else
   {
      P = new ConformingProlongationOperator(lsize, *gc);
   }
   P_mat = NULL;
}

HypreParMatrix *ParAlgebraicCoarseSpace::GetProlongationHypreParMatrix()
{
   if (P_mat) { return P_mat; }

   ParMesh *pmesh = dynamic_cast<ParMesh*>(mesh);
   MFEM_VERIFY(pmesh != NULL, "");
   Array<HYPRE_Int> dof_offsets, tdof_offsets, tdof_nb_offsets;
   Array<HYPRE_Int> *offsets[2] = {&dof_offsets, &tdof_offsets};
   int lsize = P->Height();
   int ltsize = P->Width();
   HYPRE_Int loc_sizes[2] = {lsize, ltsize};
   pmesh->GenerateOffsets(2, loc_sizes, offsets);

   MPI_Comm comm = pmesh->GetComm();

   const GroupTopology &group_topo = gc->GetGroupTopology();

   if (HYPRE_AssumedPartitionCheck())
   {
      // communicate the neighbor offsets in tdof_nb_offsets
      int nsize = group_topo.GetNumNeighbors()-1;
      MPI_Request *requests = new MPI_Request[2*nsize];
      MPI_Status  *statuses = new MPI_Status[2*nsize];
      tdof_nb_offsets.SetSize(nsize+1);
      tdof_nb_offsets[0] = tdof_offsets[0];

      // send and receive neighbors' local tdof offsets
      int request_counter = 0;
      for (int i = 1; i <= nsize; i++)
      {
         MPI_Irecv(&tdof_nb_offsets[i], 1, HYPRE_MPI_INT,
                   group_topo.GetNeighborRank(i), 5365, comm,
                   &requests[request_counter++]);
      }
      for (int i = 1; i <= nsize; i++)
      {
         MPI_Isend(&tdof_nb_offsets[0], 1, HYPRE_MPI_INT,
                   group_topo.GetNeighborRank(i), 5365, comm,
                   &requests[request_counter++]);
      }
      MPI_Waitall(request_counter, requests, statuses);

      delete [] statuses;
      delete [] requests;
   }

   HYPRE_Int *i_diag = Memory<HYPRE_Int>(lsize+1);
   HYPRE_Int *j_diag = Memory<HYPRE_Int>(ltsize);
   int diag_counter;

   HYPRE_Int *i_offd = Memory<HYPRE_Int>(lsize+1);
   HYPRE_Int *j_offd = Memory<HYPRE_Int>(lsize-ltsize);
   int offd_counter;

   HYPRE_Int *cmap   = Memory<HYPRE_Int>(lsize-ltsize);

   HYPRE_Int *col_starts = tdof_offsets;
   HYPRE_Int *row_starts = dof_offsets;

   Array<Pair<HYPRE_Int, int> > cmap_j_offd(lsize-ltsize);

   i_diag[0] = i_offd[0] = 0;
   diag_counter = offd_counter = 0;
   for (int i_ldof = 0; i_ldof < lsize; i_ldof++)
   {
      int g = ldof_group[i_ldof];
      int i_ltdof = ldof_ltdof[i_ldof];
      if (group_topo.IAmMaster(g))
      {
         j_diag[diag_counter++] = i_ltdof;
      }
      else
      {
         HYPRE_Int global_tdof_number;
         int g = ldof_group[i_ldof];
         if (HYPRE_AssumedPartitionCheck())
         {
            global_tdof_number
               = i_ltdof + tdof_nb_offsets[group_topo.GetGroupMaster(g)];
         }
         else
         {
            global_tdof_number
               = i_ltdof + tdof_offsets[group_topo.GetGroupMasterRank(g)];
         }

         cmap_j_offd[offd_counter].one = global_tdof_number;
         cmap_j_offd[offd_counter].two = offd_counter;
         offd_counter++;
      }
      i_diag[i_ldof+1] = diag_counter;
      i_offd[i_ldof+1] = offd_counter;
   }

   SortPairs<HYPRE_Int, int>(cmap_j_offd, offd_counter);

   for (int i = 0; i < offd_counter; i++)
   {
      cmap[i] = cmap_j_offd[i].one;
      j_offd[cmap_j_offd[i].two] = i;
   }

   P_mat = new HypreParMatrix(
      comm, pmesh->GetMyRank(), pmesh->GetNRanks(),
      row_starts, col_starts,
      i_diag, j_diag, i_offd, j_offd,
      cmap, offd_counter
   );

   P_mat->CopyRowStarts();
   P_mat->CopyColStarts();

   return P_mat;
}

ParAlgebraicCoarseSpace::~ParAlgebraicCoarseSpace()
{
   delete P;
   delete R_mat;
   delete P_mat;
   delete gc;
}

#endif


} // namespace mfem
#endif // MFEM_USE_CEED
