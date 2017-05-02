// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_MAXWELL_SOLVER
#define MFEM_MAXWELL_SOLVER

#include "../../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "mfem.hpp"
#include "../common/pfem_extras.hpp"

using namespace std;
using namespace mfem;

// Physical Constants

// Permittivity of Free Space (units F/m)
static double epsilon0_ = 8.8541878176e-12;

// Permeability of Free Space (units H/m)
static double mu0_ = 4.0e-7*M_PI;

namespace mfem
{

using miniapps::ND_ParFESpace;
using miniapps::RT_ParFESpace;
using miniapps::ParDiscreteCurlOperator;

namespace electromagnetics
{

class MaxwellSolver : public TimeDependentOperator
{
public:
   MaxwellSolver(ParMesh & pmesh, int sOrder,
                 double (*eps     )(const Vector&),
                 double (*muInv   )(const Vector&),
                 void   (*j_src   )(const Vector&, double, Vector&),
                 Array<int> & dbcs,
                 void   (*dEdt_bc )(const Vector&, double, Vector&));

   ~MaxwellSolver();

   HYPRE_Int GetProblemSize();

   void PrintSizes();

   void SetInitialEField(VectorCoefficient & EFieldCoef);
   void SetInitialBField(VectorCoefficient & BFieldCoef);

   void Mult(const Vector &B, Vector &dE) const;

   double GetMaximumTimeStep() const;

   double GetEnergy() const;

   Operator & GetNegCurl() { return *NegCurl_; }

   Vector & GetEField() { return *E_; }
   Vector & GetBField() { return *B_; }

   void SyncGridFuncs();

   void RegisterVisItFields(VisItDataCollection & visit_dc);

   void WriteVisItFields(int it = 0);

   void InitializeGLVis();

   void DisplayToGLVis();

private:

   // double SnapTimeStep(int n, double dt);

   int myid_;
   int num_procs_;
   int order_;

   ParMesh * pmesh_;

   VisItDataCollection * visit_dc_;

   ND_ParFESpace * HCurlFESpace_;
   RT_ParFESpace * HDivFESpace_;

   ParBilinearForm * hCurlMassEps_;
   ParBilinearForm * hDivMassMuInv_;
   ParMixedBilinearForm * weakCurlMuInv_;

   ParDiscreteCurlOperator * Curl_;

   HypreDiagScale  * diagScale_;
   HyprePCG        * pcg_;

   ParGridFunction * e_;
   ParGridFunction * b_;
   ParGridFunction * j_;
   ParGridFunction * de_;
   ParGridFunction * rhs_;
   ParLinearForm   * jd_;

   HypreParMatrix * M1Eps_;
   HypreParMatrix * M2MuInv_;
   HypreParMatrix * NegCurl_;
   HypreParMatrix * WeakCurlMuInv_;
   HypreParVector * E_;
   HypreParVector * B_;
   mutable HypreParVector * HD_;
   mutable HypreParVector * J_;
   mutable HypreParVector * JD_;
   mutable HypreParVector * RHS_;

   Coefficient       * epsCoef_;   // Electric Permittivity Coefficient
   Coefficient       * muInvCoef_; // Magnetic Permeability Coefficient
   VectorCoefficient * eCoef_;
   VectorCoefficient * bCoef_;
   VectorCoefficient * jCoef_;
   VectorCoefficient * dEdtBCCoef_;

   double (*eps_    )(const Vector&);
   double (*muInv_  )(const Vector&);
   void   (*j_src_  )(const Vector&, double, Vector&);
   void   (*dEdt_bc_)(const Vector&, double, Vector&);

   Array<int> * dbcs_;
   Array<int>   dbc_dofs_;
   std::map<std::string,socketstream*> socks_;
};

} // namespace electromagnetics

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_MAXWELL_SOLVER
