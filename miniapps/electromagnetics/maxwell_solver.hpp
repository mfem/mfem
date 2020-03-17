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

#ifndef MFEM_MAXWELL_SOLVER
#define MFEM_MAXWELL_SOLVER

#include "../common/pfem_extras.hpp"
#include "electromagnetics.hpp"

#ifdef MFEM_USE_MPI

#include <string>
#include <map>

using namespace std;
using namespace mfem;

namespace mfem
{

using common::ND_ParFESpace;
using common::RT_ParFESpace;
using common::ParDiscreteCurlOperator;

namespace electromagnetics
{

class MaxwellSolver : public TimeDependentOperator
{
public:
   MaxwellSolver(ParMesh & pmesh, int sOrder,
                 double (*eps     )(const Vector&),
                 double (*muInv   )(const Vector&),
                 double (*sigma   )(const Vector&),
                 void   (*j_src   )(const Vector&, double, Vector&),
                 Array<int> & abcs, Array<int> & dbcs,
                 void   (*dEdt_bc )(const Vector&, double, Vector&));

   ~MaxwellSolver();

   int GetLogging() const { return logging_; }
   void SetLogging(int logging) { logging_ = logging; }

   HYPRE_Int GetProblemSize();

   void PrintSizes();

   void SetInitialEField(VectorCoefficient & EFieldCoef);
   void SetInitialBField(VectorCoefficient & BFieldCoef);

   void Mult(const Vector &B, Vector &dEdt) const;

   void ImplicitSolve(const double dt, const Vector &x, Vector &k);

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

   // This method alters mutable member data
   void setupSolver(const int idt, const double dt) const;

   void implicitSolve(const double dt, const Vector &x, Vector &k) const;

   int myid_;
   int num_procs_;
   int order_;
   int logging_;

   bool lossy_;

   double dtMax_;   // Maximum stable time step
   double dtScale_; // Used to scale dt before converting to an integer

   ParMesh * pmesh_;

   ND_ParFESpace * HCurlFESpace_;
   RT_ParFESpace * HDivFESpace_;

   ParBilinearForm * hDivMassMuInv_;
   ParBilinearForm * hCurlLosses_;
   ParMixedBilinearForm * weakCurlMuInv_;

   ParDiscreteCurlOperator * Curl_;

   ParGridFunction * e_;    // Electric Field (HCurl)
   ParGridFunction * b_;    // Magnetic Flux (HDiv)
   ParGridFunction * j_;    // Volumetric Current Density (HCurl)
   ParGridFunction * dedt_; // Time Derivative of Electric Field (HCurl)
   ParGridFunction * rhs_;  // Dual of displacement current, rhs vector (HCurl)
   ParLinearForm   * jd_;   // Dual of current density (HCurl)

   HypreParMatrix * M1Losses_;
   HypreParMatrix * M2MuInv_;
   HypreParMatrix * NegCurl_;
   HypreParMatrix * WeakCurlMuInv_;
   HypreParVector * E_; // Current value of the electric field DoFs
   HypreParVector * B_; // Current value of the magnetic flux DoFs
   mutable HypreParVector * HD_; // Used in energy calculation
   mutable HypreParVector * RHS_;

   Coefficient       * epsCoef_;    // Electric Permittivity Coefficient
   Coefficient       * muInvCoef_;  // Magnetic Permeability Coefficient
   Coefficient       * sigmaCoef_;  // Electric Conductivity Coefficient
   Coefficient       * etaInvCoef_; // Admittance Coefficient
   VectorCoefficient * eCoef_;      // Initial Electric Field
   VectorCoefficient * bCoef_;      // Initial Magnetic Flux
   VectorCoefficient * jCoef_;      // Time dependent current density
   VectorCoefficient * dEdtBCCoef_; // Time dependent boundary condition

   double (*eps_    )(const Vector&);
   double (*muInv_  )(const Vector&);
   double (*sigma_  )(const Vector&);
   void   (*j_src_  )(const Vector&, double, Vector&);

   // Array of 0's and 1's marking the location of absorbing surfaces
   Array<int> abc_marker_;

   // Array of 0's and 1's marking the location of Dirichlet boundaries
   Array<int> dbc_marker_;
   void   (*dEdt_bc_)(const Vector&, double, Vector&);

   // Dirichlet degrees of freedom
   Array<int>   dbc_dofs_;

   // High order symplectic integration requires partial time steps of differing
   // lengths. If losses are present the system matrix includes a portion scaled
   // by the time step. Consequently, high order time integration requires
   // different system matrices. The following maps contain various objects that
   // depend on the time step.
   mutable std::map<int, ParBilinearForm *> a1_;
   mutable std::map<int, HypreParMatrix  *> A1_;
   mutable std::map<int, Coefficient     *> dtCoef_;
   mutable std::map<int, Coefficient     *> dtSigmaCoef_;
   mutable std::map<int, Coefficient     *> dtEtaInvCoef_;
   mutable std::map<int, HypreDiagScale  *> diagScale_;
   mutable std::map<int, HyprePCG        *> pcg_;

   // Data collection used to write VisIt files
   VisItDataCollection * visit_dc_;

   // Sockets used to communicate with GLVis
   std::map<std::string, socketstream*> socks_;
};

} // namespace electromagnetics

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_MAXWELL_SOLVER
