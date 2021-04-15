// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_LOR
#define MFEM_LOR

#include "bilinearform.hpp"

namespace mfem
{

/// Abstract base class for LOR and ParLOR classes, which construct low-order
/// refined versions of bilinear forms.
class LORBase
{
private:
   using GetIntegratorsFn = Array<BilinearFormIntegrator*> *(BilinearForm::*)();
   using GetMarkersFn = Array<Array<int>*> *(BilinearForm::*)();
   using AddIntegratorFn = void (BilinearForm::*)(BilinearFormIntegrator*);
   using AddIntegratorMarkersFn =
      void (BilinearForm::*)(BilinearFormIntegrator*, Array<int>&);

   BilinearForm &a;
   IntegrationRules irs;
   const IntegrationRule *ir;
   std::map<BilinearFormIntegrator*, const IntegrationRule*> ir_map;

   void AddIntegrators(BilinearForm &a_to,
                       GetIntegratorsFn get_integrators,
                       AddIntegratorFn add_integrator);

   void AddIntegratorsAndMarkers(BilinearForm &a_to,
                                 GetIntegratorsFn get_integrators,
                                 GetMarkersFn get_markers,
                                 AddIntegratorMarkersFn add_integrator);


   void ResetIntegrationRules(GetIntegratorsFn get_integrators);

protected:
   /// Adds all the integrators from the member data @a a to the BilinearForm
   /// @a a_to. Marks @a a_to as using external integrators. If the mesh
   /// consists of tensor product elements, temporarily changes the integration
   /// rules of the integrators to use collocated quadrature for better
   /// conditioning of the LOR system.
   void AddIntegrators(BilinearForm &a_to);

   /// Resets the integration rules of the integrators of @a a to their original
   /// values.
   void ResetIntegrationRules();

   LORBase(BilinearForm &a_);
};

class LOR : LORBase
{
protected:
   Mesh mesh;
   FiniteElementCollection *fec;
   FiniteElementSpace *fes;
   BilinearForm *a;
   SparseMatrix A;

public:
   LOR(BilinearForm &a_ho, const Array<int> &ess_tdof_list,
       int ref_type=BasisType::GaussLobatto);

   SparseMatrix &GetAssembledMatrix();

   ~LOR();
};

template <typename SolverType>
class LORSolver : public Solver
{
protected:
   LOR lor;
   SolverType solver;
public:
   LORSolver(BilinearForm &a_ho, const Array<int> &ess_tdof_list,
             int ref_type=BasisType::GaussLobatto)
      : lor(a_ho, ess_tdof_list, ref_type),
        solver(lor.GetAssembledMatrix())
   { }

   void SetOperator(const Operator &op)
   {
      if (&op != &lor.GetAssembledMatrix())
      {
         MFEM_ABORT("LORSolver::SetOperator not supported.")
      }
   }

   void Mult(const Vector &x, Vector &y) const
   {
      solver.Mult(x, y);
   }
};

#ifdef MFEM_USE_MPI

class ParLOR : public LORBase
{
protected:
   ParMesh mesh;
   FiniteElementCollection *fec;
   ParFiniteElementSpace *fes;
   ParBilinearForm *a;
   HypreParMatrix A;

public:
   ParLOR(ParBilinearForm &a_ho, const Array<int> &ess_tdof_list,
          int ref_type=BasisType::GaussLobatto);

   HypreParMatrix &GetAssembledMatrix();

   ~ParLOR();
};

template <typename SolverType>
class ParLORSolver : public Solver
{
protected:
   ParLOR lor;
   SolverType solver;
public:
   ParLORSolver(ParBilinearForm &a_ho, const Array<int> &ess_tdof_list,
                int ref_type=BasisType::GaussLobatto)
      : lor(a_ho, ess_tdof_list, ref_type),
        solver(lor.GetAssembledMatrix())
   { }

   void SetOperator(const Operator &op)
   {
      if (&op != &lor.GetAssembledMatrix())
      {
         MFEM_ABORT("ParLORSolver::SetOperator not supported.")
      }
   }

   void Mult(const Vector &x, Vector &y) const
   {
      solver.Mult(x, y);
   }
};

#endif

} // namespace mfem

#endif
