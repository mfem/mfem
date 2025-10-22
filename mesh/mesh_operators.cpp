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

#include "mesh_operators.hpp"
#include "pmesh.hpp"

namespace mfem
{

MeshOperatorSequence::~MeshOperatorSequence()
{
   // delete in reverse order
   for (int i = sequence.Size()-1; i >= 0; i--)
   {
      delete sequence[i];
   }
}

int MeshOperatorSequence::ApplyImpl(Mesh &mesh)
{
   if (sequence.Size() == 0) { return NONE; }
next_step:
   step = (step + 1) % sequence.Size();
   bool last = (step == sequence.Size() - 1);
   int mod = sequence[step]->ApplyImpl(mesh);
   switch (mod & MASK_ACTION)
   {
      case NONE:     if (last) { return NONE; } goto next_step;
      case CONTINUE: return last ? mod : (REPEAT | (mod & MASK_INFO));
      case STOP:     return STOP;
      case REPEAT:    --step; return mod;
   }
   return NONE;
}

void MeshOperatorSequence::Reset()
{
   for (int i = 0; i < sequence.Size(); i++)
   {
      sequence[i]->Reset();
   }
   step = 0;
}


ThresholdRefiner::ThresholdRefiner(ErrorEstimator &est)
   : estimator(est)
{
   aniso_estimator = dynamic_cast<AnisotropicErrorEstimator*>(&estimator);
   total_norm_p = infinity();
   total_err_goal = 0.0;
   total_fraction = 0.5;
   local_err_goal = 0.0;
   max_elements = std::numeric_limits<long long>::max();

   threshold = 0.0;
   num_marked_elements = 0LL;
   current_sequence = -1;

   non_conforming = -1;
   nc_limit = 0;
}

real_t ThresholdRefiner::GetNorm(const Vector &local_err, Mesh &mesh) const
{
#ifdef MFEM_USE_MPI
   ParMesh *pmesh = dynamic_cast<ParMesh*>(&mesh);
   if (pmesh)
   {
      return ParNormlp(local_err, total_norm_p, pmesh->GetComm());
   }
#endif
   return local_err.Normlp(total_norm_p);
}

int ThresholdRefiner::MarkWithoutRefining(Mesh & mesh,
                                          Array<Refinement> & refinements)
{
   threshold = 0.0;
   num_marked_elements = 0LL;
   refinements.SetSize(0);
   current_sequence = mesh.GetSequence();

   const long long num_elements = mesh.GetGlobalNE();
   if (num_elements >= max_elements) { return STOP; }

   const int NE = mesh.GetNE();
   const Vector &local_err = estimator.GetLocalErrors();
   MFEM_ASSERT(local_err.Size() == NE, "invalid size of local_err");

   const real_t total_err = GetNorm(local_err, mesh);
   if (total_err <= total_err_goal) { return STOP; }

   if (total_norm_p < infinity())
   {
      threshold = std::max((real_t) (total_err * total_fraction *
                                     std::pow(num_elements, -1.0/total_norm_p)),
                           local_err_goal);
   }
   else
   {
      threshold = std::max(total_err * total_fraction, local_err_goal);
   }

   for (int el = 0; el < NE; el++)
   {
      if (local_err(el) > threshold)
      {
         refinements.Append(Refinement(el));
      }
   }

   if (aniso_estimator)
   {
      const Array<int> &aniso_flags = aniso_estimator->GetAnisotropicFlags();
      if (aniso_flags.Size() > 0)
      {
         for (int i = 0; i < refinements.Size(); i++)
         {
            Refinement &ref = refinements[i];
            ref.SetType(aniso_flags[ref.index]);
         }
      }
   }

   return NONE;
}

int ThresholdRefiner::ApplyImpl(Mesh &mesh)
{
   const int action = MarkWithoutRefining(mesh, marked_elements);
   if (action == STOP) { return STOP; }

   num_marked_elements = mesh.ReduceInt(marked_elements.Size());
   if (num_marked_elements == 0LL) { return STOP; }

   mesh.GeneralRefinement(marked_elements, non_conforming, nc_limit);
   return CONTINUE + REFINED;
}

void ThresholdRefiner::Reset()
{
   estimator.Reset();
   current_sequence = -1;
   num_marked_elements = 0LL;
   // marked_elements.SetSize(0); // not necessary
}


int ThresholdDerefiner::ApplyImpl(Mesh &mesh)
{
   if (mesh.Conforming()) { return NONE; }

   const Vector &local_err = estimator.GetLocalErrors();
   bool derefs = mesh.DerefineByError(local_err, threshold, nc_limit, op);

   return derefs ? CONTINUE + DEREFINED : NONE;
}


int CoefficientRefiner::ApplyImpl(Mesh &mesh)
{
   int max_it = 1;
   return PreprocessMesh(mesh, max_it);
}

int CoefficientRefiner::PreprocessMesh(Mesh &mesh, int max_it)
{
   int rank = 0;
   MFEM_VERIFY(max_it > 0, "max_it must be strictly positive")

   int dim = mesh.Dimension();
   L2_FECollection l2fec(order, dim);
   FiniteElementSpace* l2fes = NULL;

   bool par = false;
   GridFunction *gf = NULL;

#ifdef MFEM_USE_MPI
   ParMesh* pmesh = dynamic_cast<ParMesh*>(&mesh);
   if (pmesh && pmesh->Nonconforming())
   {
      par = true;
      l2fes = new ParFiniteElementSpace(pmesh, &l2fec);
      gf = new ParGridFunction(static_cast<ParFiniteElementSpace*>(l2fes));
   }
#endif
   if (!par)
   {
      l2fes = new FiniteElementSpace(&mesh, &l2fec);
      gf = new GridFunction(l2fes);
   }

   // If custom integration rule has not been set,
   // then use the default integration rule
   if (!irs)
   {
      int order_quad = 2*order + 3;
      for (int i=0; i < Geometry::NumGeom; ++i)
      {
         ir_default[i] = &(IntRules.Get(i, order_quad));
      }
      irs = ir_default;
   }

   for (int i = 0; i < max_it; i++)
   {
      // Compute number of elements and L2-norm of f.
      int NE = mesh.GetNE();
      int globalNE = 0;
      real_t norm_of_coeff = 0.0;
      if (par)
      {
#ifdef MFEM_USE_MPI
         globalNE = pmesh->GetGlobalNE();
         norm_of_coeff = ComputeGlobalLpNorm(2.0,*coeff,*pmesh,irs);
#endif
      }
      else
      {
         globalNE = NE;
         norm_of_coeff = ComputeLpNorm(2.0,*coeff,mesh,irs);
      }

      // Compute average L2-norm of f
      real_t av_norm_of_coeff = norm_of_coeff / sqrt(globalNE);

      // Compute element-wise L2-norms of (I - Π) f
      Vector element_norms_of_fine_scale(NE);
      gf->SetSpace(l2fes);
      gf->ProjectCoefficient(*coeff);
      gf->ComputeElementL2Errors(*coeff,element_norms_of_fine_scale,irs);

      // Define osc_K(f) := || h ⋅ (I - Π) f ||_K and select elements
      // for refinement based on threshold. Also record relative osc(f).
      global_osc = 0.0;
      mesh_refinements.SetSize(0);
      element_oscs.Destroy();
      element_oscs.SetSize(NE);
      element_oscs = 0.0;
      for (int j = 0; j < NE; j++)
      {
         real_t h = mesh.GetElementSize(j);
         real_t element_osc = h * element_norms_of_fine_scale(j);
         if ( element_osc > threshold * av_norm_of_coeff )
         {
            mesh_refinements.Append(j);
         }
         element_oscs(j) = element_osc/(norm_of_coeff + 1e-10);
         global_osc += element_osc*element_osc;
      }
#ifdef MFEM_USE_MPI
      if (par)
      {
         MPI_Comm comm = pmesh->GetComm();
         MPI_Allreduce(MPI_IN_PLACE, &global_osc, 1, MPITypeMap<real_t>::mpi_type,
                       MPI_SUM, comm);
         MPI_Comm_rank(comm, &rank);
      }
#endif
      global_osc = sqrt(global_osc)/(norm_of_coeff + 1e-10);

      // Exit if the global threshold or maximum number of elements is reached.
      if (global_osc < threshold || globalNE > max_elements)
      {
         if (global_osc > threshold && globalNE > max_elements && rank == 0 &&
             print_level)
         {
            MFEM_WARNING("Reached maximum number of elements "
                         "before resolving data to tolerance.");
         }
         delete l2fes;
         delete gf;
         return STOP;
      }

      // Refine elements.
      mesh.GeneralRefinement(mesh_refinements, nonconforming, nc_limit);
      l2fes->Update(false);
      gf->Update();

   }
   delete l2fes;
   delete gf;
   return CONTINUE + REFINED;

}

void CoefficientRefiner::Reset()
{
   element_oscs.Destroy();
   global_osc = 0.0;
   coeff = NULL;
   irs = NULL;
}


int Rebalancer::ApplyImpl(Mesh &mesh)
{
#ifdef MFEM_USE_MPI
   ParMesh *pmesh = dynamic_cast<ParMesh*>(&mesh);
   if (pmesh && pmesh->Nonconforming())
   {
      pmesh->Rebalance();
      return CONTINUE + REBALANCED;
   }
#endif
   return NONE;
}


} // namespace mfem
