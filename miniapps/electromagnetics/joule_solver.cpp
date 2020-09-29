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

#include "joule_solver.hpp"

#ifdef MFEM_USE_MPI

using namespace std;

namespace mfem
{

using namespace common;

namespace electromagnetics
{

MagneticDiffusionEOperator::MagneticDiffusionEOperator(
   int stateVectorLen,
   ParFiniteElementSpace &L2FES,
   ParFiniteElementSpace &HCurlFES,
   ParFiniteElementSpace &HDivFES,
   ParFiniteElementSpace &HGradFES,
   Array<int> &ess_bdr_arg,
   Array<int> &thermal_ess_bdr_arg,
   Array<int> &poisson_ess_bdr_arg,
   double mu_coef,
   std::map<int, double> sigmaAttMap,
   std::map<int, double> TcapacityAttMap,
   std::map<int, double> InvTcapAttMap,
   std::map<int, double> InvTcondAttMap)

   : TimeDependentOperator(stateVectorLen, 0.0),
     L2FESpace(L2FES), HCurlFESpace(HCurlFES), HDivFESpace(HDivFES),
     HGradFESpace(HGradFES),
     a0(NULL), a1(NULL), a2(NULL), m1(NULL), m2(NULL), m3(NULL),
     s1(NULL), s2(NULL), grad(NULL), curl(NULL), weakDiv(NULL), weakDivC(NULL),
     weakCurl(NULL),
     A0(NULL), A1(NULL), A2(NULL), M1(NULL), M2(NULL), M3(NULL),
     X0(NULL), X1(NULL), X2(NULL), B0(NULL), B1(NULL), B2(NULL), B3(NULL),
     v0(NULL), v1(NULL), v2(NULL),
     amg_a0(NULL), pcg_a0(NULL), ads_a2(NULL), pcg_a2(NULL), ams_a1(NULL),
     pcg_a1(NULL), dsp_m3(NULL),pcg_m3(NULL),
     dsp_m1(NULL), pcg_m1(NULL), dsp_m2(NULL), pcg_m2(NULL),
     mu(mu_coef), dt_A1(-1.0), dt_A2(-1.0)
{
   ess_bdr.SetSize(ess_bdr_arg.Size());
   for (int i=0; i<ess_bdr_arg.Size(); i++)
   {
      ess_bdr[i] = ess_bdr_arg[i];
   }
   thermal_ess_bdr.SetSize(thermal_ess_bdr_arg.Size());
   for (int i=0; i<thermal_ess_bdr_arg.Size(); i++)
   {
      thermal_ess_bdr[i] = thermal_ess_bdr_arg[i];
   }
   poisson_ess_bdr.SetSize(poisson_ess_bdr_arg.Size());
   for (int i=0; i<poisson_ess_bdr_arg.Size(); i++)
   {
      poisson_ess_bdr[i] = poisson_ess_bdr_arg[i];
   }

   sigma     = new MeshDependentCoefficient(sigmaAttMap);
   Tcapacity = new MeshDependentCoefficient(TcapacityAttMap);
   InvTcap   = new MeshDependentCoefficient(InvTcapAttMap);
   InvTcond  = new MeshDependentCoefficient(InvTcondAttMap);

   this->buildA0(*sigma);
   this->buildM3(*Tcapacity);
   this->buildM1(*sigma);
   this->buildM2(*InvTcond);
   this->buildS2(*InvTcap);
   this->buildS1(1.0/mu);
   this->buildCurl(1.0/mu);
   this->buildDiv(*InvTcap);
   this->buildGrad();

   v0 = new ParGridFunction(&HGradFESpace);
   v1 = new ParGridFunction(&HCurlFESpace);
   v2 = new ParGridFunction(&HDivFESpace);
   A0 = new HypreParMatrix;
   A1 = new HypreParMatrix;
   A2 = new HypreParMatrix;
   X0 = new Vector;
   X1 = new Vector;
   X2 = new Vector;
   B0 = new Vector;
   B1 = new Vector;
   B2 = new Vector;
   B3 = new Vector;
}

void MagneticDiffusionEOperator::Init(Vector &X)
{
   Vector zero_vec(3); zero_vec = 0.0;
   VectorConstantCoefficient Zero_vec(zero_vec);
   ConstantCoefficient Zero(0.0);

   // The big BlockVector stores the fields as follows:
   //    Temperature
   //    Temperature Flux
   //    P field
   //    E field
   //    B field
   //    Joule Heating

   int Vsize_l2 = L2FESpace.GetVSize();
   int Vsize_nd = HCurlFESpace.GetVSize();
   int Vsize_rt = HDivFESpace.GetVSize();
   int Vsize_h1 = HGradFESpace.GetVSize();

   Array<int> true_offset(7);
   true_offset[0] = 0;
   true_offset[1] = true_offset[0] + Vsize_l2;
   true_offset[2] = true_offset[1] + Vsize_rt;
   true_offset[3] = true_offset[2] + Vsize_h1;
   true_offset[4] = true_offset[3] + Vsize_nd;
   true_offset[5] = true_offset[4] + Vsize_rt;
   true_offset[6] = true_offset[5] + Vsize_l2;

   Vector* xptr = (Vector*) &X;
   ParGridFunction E, B, T, F, W, P;
   T.MakeRef(&L2FESpace,   *xptr,true_offset[0]);
   F.MakeRef(&HDivFESpace, *xptr,true_offset[1]);
   P.MakeRef(&HGradFESpace,*xptr,true_offset[2]);
   E.MakeRef(&HCurlFESpace,*xptr,true_offset[3]);
   B.MakeRef(&HDivFESpace, *xptr,true_offset[4]);
   W.MakeRef(&L2FESpace,   *xptr,true_offset[5]);

   E.ProjectCoefficient(Zero_vec);
   B.ProjectCoefficient(Zero_vec);
   F.ProjectCoefficient(Zero_vec);
   T.ProjectCoefficient(Zero);
   P.ProjectCoefficient(Zero);
   W.ProjectCoefficient(Zero);
}

/*
This is an experimental Mult() method for explicit integration.
Not recommended for actual use.

S0 P  = 0
M1 E  = WeakCurl^T B + Grad P
   dB = -Curl E
M2 F  = WeakDiv^T T
M3 dT = WeakDiv F + W

where W is the Joule heating.

Boundary conditions are applied to E.  No boundary conditions are applied to B.
Since we are using Hdiv, zero flux is an essential BC on F.  P is given by Div
sigma Grad P = 0 with appropriate BC's.
*/
void MagneticDiffusionEOperator::Mult(const Vector &X, Vector &dX_dt) const
{
   dX_dt = 0.0;

   // The big BlockVector stores the fields as follows:
   //    Temperature
   //    Temperature Flux
   //    P field
   //    E field
   //    B field
   //    Joule Heating

   int Vsize_l2 = L2FESpace.GetVSize();
   int Vsize_nd = HCurlFESpace.GetVSize();
   int Vsize_rt = HDivFESpace.GetVSize();
   int Vsize_h1 = HGradFESpace.GetVSize();

   Array<int> true_offset(7);
   true_offset[0] = 0;
   true_offset[1] = true_offset[0] + Vsize_l2;
   true_offset[2] = true_offset[1] + Vsize_rt;
   true_offset[3] = true_offset[2] + Vsize_h1;
   true_offset[4] = true_offset[3] + Vsize_nd;
   true_offset[5] = true_offset[4] + Vsize_rt;
   true_offset[6] = true_offset[5] + Vsize_l2;

   Vector* xptr = (Vector*) &X;
   ParGridFunction E, B, T, F, W, P;
   T.MakeRef(&L2FESpace,   *xptr,true_offset[0]);
   F.MakeRef(&HDivFESpace, *xptr,true_offset[1]);
   P.MakeRef(&HGradFESpace,*xptr,true_offset[2]);
   E.MakeRef(&HCurlFESpace,*xptr,true_offset[3]);
   B.MakeRef(&HDivFESpace, *xptr,true_offset[4]);
   W.MakeRef(&L2FESpace,   *xptr,true_offset[5]);

   ParGridFunction dE, dB, dT, dF, dW, dP;
   dT.MakeRef(&L2FESpace,   dX_dt,true_offset[0]);
   dF.MakeRef(&HDivFESpace, dX_dt,true_offset[1]);
   dP.MakeRef(&HGradFESpace,dX_dt,true_offset[2]);
   dE.MakeRef(&HCurlFESpace,dX_dt,true_offset[3]);
   dB.MakeRef(&HDivFESpace, dX_dt,true_offset[4]);
   dW.MakeRef(&L2FESpace,   dX_dt,true_offset[5]);

   // db = - Curl E
   curl->Mult(E, dB);
   dB *= -1.0;

   // form the Laplacian and solve it
   ParGridFunction Phi_gf(&HGradFESpace);

   // p_bc is given function defining electrostatic potential on surface
   FunctionCoefficient voltage(p_bc);
   voltage.SetTime(this->GetTime());
   Phi_gf = 0.0;

   // the line below is currently not fully supported on AMR meshes
   // Phi_gf.ProjectBdrCoefficient(voltage,poisson_ess_bdr);

   // this is a hack to get around the above issue
   Phi_gf.ProjectCoefficient(voltage);
   // end of hack

   // apply essential BC's and apply static condensation, the new system to
   // solve is A0 X0 = B0
   Array<int> poisson_ess_tdof_list;
   HGradFESpace.GetEssentialTrueDofs(poisson_ess_bdr, poisson_ess_tdof_list);

   *v0 = 0.0;
   a0->FormLinearSystem(poisson_ess_tdof_list,Phi_gf,*v0,*A0,*X0,*B0);

   if (amg_a0 == NULL) { amg_a0 = new HypreBoomerAMG(*A0); }
   if (pcg_a0 == NULL)
   {
      pcg_a0 = new HyprePCG(*A0);
      pcg_a0->SetTol(SOLVER_TOL);
      pcg_a0->SetMaxIter(SOLVER_MAX_IT);
      pcg_a0->SetPrintLevel(SOLVER_PRINT_LEVEL);
      pcg_a0->SetPreconditioner(*amg_a0);
   }
   // pcg "Mult" operation is a solve
   // X0 = A0^-1 * B0
   pcg_a0->Mult(*B0, *X0);

   // "undo" the static condensation using dP as a temporary variable, dP stores
   // Pnew
   a0->RecoverFEMSolution(*X0,*v0,P);
   dP = 0.0;

   // v1 = <1/mu v, curl u> B
   // B is a grid function but weakCurl is not parallel assembled so is OK
   weakCurl->MultTranspose(B, *v1);

   // now add Grad dPhi/dt term
   // use E as a temporary, E = Grad P
   // v1 = curl 1/mu B + M1 * Grad P
   // note: these two steps could be replaced by one step if we have the
   // bilinear form <sigma gradP, E>
   grad->Mult(P,E);
   m1->AddMult(E,*v1,1.0);

   // OK now v1 is the right hand side, just need to add essential BC's

   ParGridFunction J_gf(&HCurlFESpace);

   // edot_bc is time-derivative E-field on a boundary surface and then it is
   // used as a Dirichlet BC.

   VectorFunctionCoefficient Jdot(3, edot_bc);
   J_gf = 0.0;
   J_gf.ProjectBdrCoefficientTangent(Jdot,ess_bdr);

   // apply essential BC's and apply static condensation
   // the new system to solve is M1 X1 = B1
   Array<int> ess_tdof_list;
   HCurlFESpace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   m1->FormLinearSystem(ess_tdof_list,J_gf,*v1,*A1,*X1,*B1);

   if (dsp_m1 == NULL) { dsp_m1 = new HypreDiagScale(*A1); }
   if (pcg_m1 == NULL)
   {
      pcg_m1 = new HyprePCG(*A1);
      pcg_m1->SetTol(SOLVER_TOL);
      pcg_m1->SetMaxIter(SOLVER_MAX_IT);
      pcg_m1->SetPrintLevel(SOLVER_PRINT_LEVEL);
      pcg_m1->SetPreconditioner(*dsp_m1);
   }
   // pcg "Mult" operation is a solve
   // X1 = M1^-1 * B1 = M1^-1 (-S1 E)
   pcg_m1->Mult(*B1, *X1);

   // "undo" the static condensation and fill in grid function dE
   m1->RecoverFEMSolution(*X1,*v1,E);
   dE = 0.0;

   // the total field is E_tot = E_ind - Grad Phi
   // so we need to subtract out Grad Phi
   // E = E - grad (P)
   grad->AddMult(P,E,-1.0);

   // Compute Joule heating using the previous value of E
   this->GetJouleHeating(E,W);
   dW = 0.0;

   // Mult(x,y,alpha=1,beta=0)
   // y = alpha*A*x + beta*y
   // giving
   // v2 = <v, div u> * T
   weakDiv->Mult(T, *v2);

   // apply the thermal BC
   // recall for Hdiv formulation the essential BC is on the flux
   Vector zero_vec(3); zero_vec = 0.0;
   VectorConstantCoefficient Zero_vec(zero_vec);
   ParGridFunction F_gf(&HDivFESpace);
   F_gf = 0.0;
   F_gf.ProjectBdrCoefficientNormal(Zero_vec,thermal_ess_bdr);

   // apply essential BC's and apply static condensation
   // the new system to solve is M2 X2 = B2
   Array<int> thermal_ess_tdof_list;
   HDivFESpace.GetEssentialTrueDofs(thermal_ess_bdr, thermal_ess_tdof_list);

   m2->FormLinearSystem(thermal_ess_tdof_list,F_gf,*v2,*A2,*X2,*B2);

   if (dsp_m2 == NULL) { dsp_m2 = new HypreDiagScale(*A2); }
   if (pcg_m2 == NULL)
   {
      pcg_m2 = new HyprePCG(*A2);
      pcg_m2->SetTol(SOLVER_TOL);
      pcg_m2->SetMaxIter(SOLVER_MAX_IT);
      pcg_m2->SetPrintLevel(SOLVER_PRINT_LEVEL);
      pcg_m2->SetPreconditioner(*dsp_m2);
   }
   // X2 = m2^-1 * B2
   pcg_m2->Mult(*B2, *X2);

   // "undo" the static condensation and fill in grid function dF
   m2->RecoverFEMSolution(*X2,*v2,F);

   // Compute dT using previous value of flux
   // dT = [w - div F]
   //
   // <u,u> dT = <1/c W,u> - <1/c div v,u> F
   //
   // where W is Joule heating and F is the flux that we just computed
   //
   // note: if div is a BilinearForm, then W should be converted to a LoadVector

   GridFunctionCoefficient Wcoeff(&W);
   ParLinearForm temp_lf(&L2FESpace);

   // compute load vector < W, u>
   temp_lf.AddDomainIntegrator(new DomainLFIntegrator(Wcoeff));
   temp_lf.Assemble();
   // lf = lf - div F
   weakDiv->AddMult(F, temp_lf, -1.0);

   // if div is a BilinearForm, need to perform mass matrix solve to convert
   // energy cT to temperature T

   if (dsp_m3 == NULL) { dsp_m3 = new HypreDiagScale(*M3); }
   if (pcg_m3 == NULL)
   {
      pcg_m3 = new HyprePCG(*M3);
      pcg_m3->SetTol(SOLVER_TOL);
      pcg_m3->SetMaxIter(SOLVER_MAX_IT);
      pcg_m3->SetPrintLevel(SOLVER_PRINT_LEVEL);
      pcg_m3->SetPreconditioner(*dsp_m3);
   }
   // solve for dT from M3 dT = lf
   // no boundary conditions on this solve
   pcg_m3->Mult(temp_lf, dT);
}

/*
This is the main computational code that computes dX/dt implicitly
where X is the state vector containing P, E, B, F, T, and W

        S0 P = 0
(M1+dt S1) E = WeakCurl^T B + Grad P
          dB = -Curl E
(M2+dt S2) F = WeakDiv^T T
       M3 dT = WeakDiv F + W

where W is the Joule heating.

Boundary conditions are applied to E.  Boundary conditions are applied to F.  No
boundary conditions are applied to B or T.

The W term in the left hand side is the Joule heating which is a nonlinear
(quadratic) function of E.

P is solution of Div sigma Grad dP = 0.

The total E-field is given by E_tot = E_ind - Grad P, the big equation for E
above is really for E_ind (the induced, or solenoidal, component) and this is
corrected for.
*/
void MagneticDiffusionEOperator::ImplicitSolve(const double dt,
                                               const Vector &X, Vector &dX_dt)
{
   if ( A2 == NULL || fabs(dt-dt_A2) > 1.0e-12*dt )
   {
      this->buildA2(*InvTcond, *InvTcap, dt);
   }
   if ( A1 == NULL || fabs(dt-dt_A1) > 1.0e-12*dt )
   {
      this->buildA1(1.0/mu, *sigma, dt);
   }

   dX_dt = 0.0;

   // The big BlockVector stores the fields as follows:
   //    Temperature
   //    Temperature Flux
   //    P field
   //    E field
   //    B field
   //    Joule Heating

   int Vsize_l2 = L2FESpace.GetVSize();
   int Vsize_nd = HCurlFESpace.GetVSize();
   int Vsize_rt = HDivFESpace.GetVSize();
   int Vsize_h1 = HGradFESpace.GetVSize();

   Array<int> true_offset(7);
   true_offset[0] = 0;
   true_offset[1] = true_offset[0] + Vsize_l2;
   true_offset[2] = true_offset[1] + Vsize_rt;
   true_offset[3] = true_offset[2] + Vsize_h1;
   true_offset[4] = true_offset[3] + Vsize_nd;
   true_offset[5] = true_offset[4] + Vsize_rt;
   true_offset[6] = true_offset[5] + Vsize_l2;

   Vector* xptr  = (Vector*) &X;
   ParGridFunction E, B, T, F, W, P;
   T.MakeRef(&L2FESpace,   *xptr,true_offset[0]);
   F.MakeRef(&HDivFESpace, *xptr,true_offset[1]);
   P.MakeRef(&HGradFESpace,*xptr,true_offset[2]);
   E.MakeRef(&HCurlFESpace,*xptr,true_offset[3]);
   B.MakeRef(&HDivFESpace, *xptr,true_offset[4]);
   W.MakeRef(&L2FESpace,   *xptr,true_offset[5]);

   ParGridFunction dE, dB, dT, dF, dW, dP;
   dT.MakeRef(&L2FESpace,   dX_dt,true_offset[0]);
   dF.MakeRef(&HDivFESpace, dX_dt,true_offset[1]);
   dP.MakeRef(&HGradFESpace,dX_dt,true_offset[2]);
   dE.MakeRef(&HCurlFESpace,dX_dt,true_offset[3]);
   dB.MakeRef(&HDivFESpace, dX_dt,true_offset[4]);
   dW.MakeRef(&L2FESpace,   dX_dt,true_offset[5]);

   // form the Laplacian and solve it
   ParGridFunction Phi_gf(&HGradFESpace);

   // p_bc is given function defining electrostatic potential on surface
   FunctionCoefficient voltage(p_bc);
   voltage.SetTime(this->GetTime());
   Phi_gf = 0.0;

   // the function below is currently not fully supported on AMR meshes
   // Phi_gf.ProjectBdrCoefficient(voltage,poisson_ess_bdr);

   // this is a hack to get around the above issue
   Phi_gf.ProjectCoefficient(voltage);
   // end of hack

   // apply essential BC's and apply static condensation, the new system to
   // solve is A0 X0 = B0
   Array<int> poisson_ess_tdof_list;
   HGradFESpace.GetEssentialTrueDofs(poisson_ess_bdr, poisson_ess_tdof_list);

   *v0 = 0.0;
   a0->FormLinearSystem(poisson_ess_tdof_list,Phi_gf,*v0,*A0,*X0,*B0);

   if (amg_a0 == NULL) { amg_a0 = new HypreBoomerAMG(*A0); }
   if (pcg_a0 == NULL)
   {
      pcg_a0 = new HyprePCG(*A0);
      pcg_a0->SetTol(SOLVER_TOL);
      pcg_a0->SetMaxIter(SOLVER_MAX_IT);
      pcg_a0->SetPrintLevel(SOLVER_PRINT_LEVEL);
      pcg_a0->SetPreconditioner(*amg_a0);
   }
   // pcg "Mult" operation is a solve
   // X0 = A0^-1 * B0
   pcg_a0->Mult(*B0, *X0);

   // "undo" the static condensation saving result in grid function dP
   a0->RecoverFEMSolution(*X0,*v0,P);
   dP = 0.0;

   // v1 = <1/mu v, curl u> B
   // B is a grid function but weakCurl is not parallel assembled so is OK
   weakCurl->MultTranspose(B, *v1);

   // now add Grad dPhi/dt term
   // use E as a temporary, E = Grad P
   // v1 = curl 1/mu B + M1 * Grad P
   grad->Mult(P,E);
   m1->AddMult(E,*v1,1.0);

   ParGridFunction J_gf(&HCurlFESpace);

   // edot_bc is time-derivative E-field on a boundary surface
   // and then it is used as a Dirichlet BC
   // the vector v1 will be modified by the values Jtmp and
   // the part of the matrix m1 that hs been eliminated (but stored).
   VectorFunctionCoefficient Jdot(3, edot_bc);
   J_gf = 0.0;
   J_gf.ProjectBdrCoefficientTangent(Jdot,ess_bdr);

   // form the linear system, including eliminating essential BC's and applying
   // static condensation. The system to solve is A1 X1 = B1
   Array<int> ess_tdof_list;
   HCurlFESpace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   a1->FormLinearSystem(ess_tdof_list,J_gf,*v1,*A1,*X1,*B1);

   // We only need to create the solver and preconditioner once
   if ( ams_a1 == NULL )
   {
      ParFiniteElementSpace *prec_fespace =
         (a1->StaticCondensationIsEnabled() ? a1->SCParFESpace() : &HCurlFESpace);
      ams_a1 = new HypreAMS(*A1, prec_fespace);
   }
   if ( pcg_a1 == NULL )
   {
      pcg_a1 = new HyprePCG(*A1);
      pcg_a1->SetTol(SOLVER_TOL);
      pcg_a1->SetMaxIter(SOLVER_MAX_IT);
      pcg_a1->SetPrintLevel(SOLVER_PRINT_LEVEL);
      pcg_a1->SetPreconditioner(*ams_a1);
   }
   // solve the system
   // dE = (A1)^-1 [-S1 E]
   pcg_a1->Mult(*B1, *X1);

   // this is required because of static condensation, E is a grid function
   a1->RecoverFEMSolution(*X1,*v1,E);
   dE = 0.0;

   // the total field is E_tot = E_ind - Grad Phi
   // so we need to subtract out Grad Phi
   // E = E - grad (P)
   // note grad maps GF to GF
   grad->AddMult(P,E,-1.0);

   // Compute dB/dt = -Curl(E_{n+1})
   // note curl maps GF to GF
   curl->Mult(E, dB);
   dB *= -1.0;

   // Compute Energy Deposition
   this->GetJouleHeating(E,W);

   // v2 = Div^T * W, where W is the Joule heating computed above, and
   // Div is the matrix <div u, v>
   weakDivC->MultTranspose(W, *v2);
   *v2 *= dt;

   // v2 = <v, div u> T + (1.0)*v2
   weakDiv->AddMultTranspose(T, *v2, 1.0);

   // apply the thermal BC
   Vector zero_vec(3); zero_vec = 0.0;
   VectorConstantCoefficient Zero_vec(zero_vec);
   ParGridFunction F_gf(&HDivFESpace);
   F_gf = 0.0;
   F_gf.ProjectBdrCoefficientNormal(Zero_vec,thermal_ess_bdr);

   // form the linear system, including eliminating essential BC's and applying
   // static condensation. The system to solve is A2 X2 = B2
   Array<int> thermal_ess_tdof_list;
   HDivFESpace.GetEssentialTrueDofs(thermal_ess_bdr, thermal_ess_tdof_list);
   a2->FormLinearSystem(thermal_ess_tdof_list,F_gf,*v2,*A2,*X2,*B2);

   // We only need to create the solver and preconditioner once
   if ( ads_a2 == NULL )
   {
      ParFiniteElementSpace *prec_fespace =
         (a2->StaticCondensationIsEnabled() ? a2->SCParFESpace() : &HDivFESpace);
      ads_a2 = new HypreADS(*A2, prec_fespace);
   }
   if ( pcg_a2 == NULL )
   {
      pcg_a2 = new HyprePCG(*A2);
      pcg_a2->SetTol(SOLVER_TOL);
      pcg_a2->SetMaxIter(SOLVER_MAX_IT);
      pcg_a2->SetPrintLevel(SOLVER_PRINT_LEVEL);
      pcg_a2->SetPreconditioner(*ads_a2);
   }
   // solve for dF from a2 dF = v2
   // dF = (A2)^-1 [S2*F + rhs]
   pcg_a2->Mult(*B2, *X2);

   // this is required because of static condensation
   a2->RecoverFEMSolution(*X2,*v2,F);

   // c dT = [W - div F]
   //
   // <u,u> dT = <1/c W,u> - <1/c div v,u>
   //
   // where W is Joule heating and F is the flux that we just computed
   //
   // note: if div is a BilinearForm, then W should be converted to a LoadVector
   // compute load vector <1/c W, u> where W is the Joule heating GF

   // create the Coefficient 1/c W
   //ScaledGFCoefficient Wcoeff(&W, *InvTcap);
   GridFunctionCoefficient Wcoeff(&W);

   // compute <W,u>
   ParLinearForm temp_lf(&L2FESpace);
   temp_lf.AddDomainIntegrator(new DomainLFIntegrator(Wcoeff));
   temp_lf.Assemble();

   // lf = lf - div F
   weakDiv->AddMult(F, temp_lf, -1.0);

   // need to perform mass matrix solve to get temperature T
   // <c u, u> Tdot = -<div v, u> F +  <1/c W, u>
   // NOTE: supposedly we can just invert any L2 matrix, could do that here
   // instead of a solve

   if (dsp_m3 == NULL) { dsp_m3 = new HypreDiagScale(*M3); }
   if (pcg_m3 == NULL)
   {
      pcg_m3 = new HyprePCG(*M3);
      pcg_m3->SetTol(SOLVER_TOL);
      pcg_m3->SetMaxIter(SOLVER_MAX_IT);
      pcg_m3->SetPrintLevel(SOLVER_PRINT_LEVEL);
      pcg_m3->SetPreconditioner(*dsp_m3);
   }

   // solve for dT from M3 dT = lf
   // no boundary conditions on this solve
   pcg_m3->Mult(temp_lf, dT);
}

void MagneticDiffusionEOperator::buildA0(MeshDependentCoefficient &Sigma)
{
   if ( a0 != NULL ) { delete a0; }

   // First create and assemble the bilinear form.  For now we assume the mesh
   // isn't moving, the materials are time independent, and dt is constant. So
   // we only need to do this once.

   // ConstantCoefficient Sigma(sigma);
   a0 = new ParBilinearForm(&HGradFESpace);
   a0->AddDomainIntegrator(new DiffusionIntegrator(Sigma));
   if (STATIC_COND == 1) { a0->EnableStaticCondensation(); }
   a0->Assemble();

   // Don't finalize or parallel assemble this is done in FormLinearSystem.
}

void MagneticDiffusionEOperator::buildA1(double muInv,
                                         MeshDependentCoefficient &Sigma,
                                         double dt)
{
   if ( a1 != NULL ) { delete a1; }

   // First create and assemble the bilinear form.  For now we assume the mesh
   // isn't moving, the materials are time independent, and dt is constant. So
   // we only need to do this once.

   ConstantCoefficient dtMuInv(dt*muInv);
   a1 = new ParBilinearForm(&HCurlFESpace);
   a1->AddDomainIntegrator(new VectorFEMassIntegrator(Sigma));
   a1->AddDomainIntegrator(new CurlCurlIntegrator(dtMuInv));
   if (STATIC_COND == 1) { a1->EnableStaticCondensation(); }
   a1->Assemble();

   // Don't finalize or parallel assemble this is done in FormLinearSystem.

   dt_A1 = dt;
}

void MagneticDiffusionEOperator::buildA2(MeshDependentCoefficient &InvTcond,
                                         MeshDependentCoefficient &InvTcap,
                                         double dt)
{
   if ( a2 != NULL ) { delete a2; }

   InvTcap.SetScaleFactor(dt);
   a2 = new ParBilinearForm(&HDivFESpace);
   a2->AddDomainIntegrator(new VectorFEMassIntegrator(InvTcond));
   a2->AddDomainIntegrator(new DivDivIntegrator(InvTcap));
   if (STATIC_COND == 1) { a2->EnableStaticCondensation(); }
   a2->Assemble();

   // Don't finalize or parallel assemble this is done in FormLinearSystem.

   dt_A2 = dt;
}

void MagneticDiffusionEOperator::buildM1(MeshDependentCoefficient &Sigma)
{
   if ( m1 != NULL ) { delete m1; }

   m1 = new ParBilinearForm(&HCurlFESpace);
   m1->AddDomainIntegrator(new VectorFEMassIntegrator(Sigma));
   m1->Assemble();

   // Don't finalize or parallel assemble this is done in FormLinearSystem.
}

void MagneticDiffusionEOperator::buildM2(MeshDependentCoefficient &Alpha)
{
   if ( m2 != NULL ) { delete m2; }

   // ConstantCoefficient MuInv(muInv);
   m2 = new ParBilinearForm(&HDivFESpace);
   m2->AddDomainIntegrator(new VectorFEMassIntegrator(Alpha));
   m2->Assemble();

   // Don't finalize or parallel assemble this is done in FormLinearSystem.
}

void MagneticDiffusionEOperator::buildM3(MeshDependentCoefficient &Tcapacity)
{
   if ( m3 != NULL ) { delete m3; }

   // ConstantCoefficient Sigma(sigma);
   m3 = new ParBilinearForm(&L2FESpace);
   m3->AddDomainIntegrator(new MassIntegrator(Tcapacity));
   m3->Assemble();
   m3->Finalize();
   M3 = m3->ParallelAssemble();
}

void MagneticDiffusionEOperator::buildS1(double muInv)
{
   if ( s1 != NULL ) { delete s1; }

   ConstantCoefficient MuInv(muInv);
   s1 = new ParBilinearForm(&HCurlFESpace);
   s1->AddDomainIntegrator(new CurlCurlIntegrator(MuInv));
   s1->Assemble();
}

void MagneticDiffusionEOperator::buildS2(MeshDependentCoefficient &InvTcap)
{
   if ( s2 != NULL ) { delete s2; }

   // ConstantCoefficient param(a);
   s2 = new ParBilinearForm(&HDivFESpace);
   s2->AddDomainIntegrator(new DivDivIntegrator(InvTcap));
   s2->Assemble();
}

void MagneticDiffusionEOperator::buildCurl(double muInv)
{
   if ( curl != NULL ) { delete curl; }
   if ( weakCurl != NULL ) { delete weakCurl; }

   curl = new ParDiscreteLinearOperator(&HCurlFESpace, &HDivFESpace);
   curl->AddDomainInterpolator(new CurlInterpolator);
   curl->Assemble();

   ConstantCoefficient MuInv(muInv);
   weakCurl = new ParMixedBilinearForm(&HCurlFESpace, &HDivFESpace);
   weakCurl->AddDomainIntegrator(new VectorFECurlIntegrator(MuInv));
   weakCurl->Assemble();

   // no ParallelAssemble since this will be applied to GridFunctions
}

void MagneticDiffusionEOperator::buildDiv(MeshDependentCoefficient &InvTcap)
{
   if ( weakDiv != NULL ) { delete weakDiv; }
   if ( weakDivC != NULL ) { delete weakDivC; }

   weakDivC = new ParMixedBilinearForm(&HDivFESpace, &L2FESpace);
   weakDivC->AddDomainIntegrator(new VectorFEDivergenceIntegrator(InvTcap));
   weakDivC->Assemble();

   weakDiv = new ParMixedBilinearForm(&HDivFESpace, &L2FESpace);
   weakDiv->AddDomainIntegrator(new VectorFEDivergenceIntegrator());
   weakDiv->Assemble();

   // no ParallelAssemble since this will be applied to GridFunctions
}

void MagneticDiffusionEOperator::buildGrad()
{
   if ( grad != NULL ) { delete grad; }

   grad = new ParDiscreteLinearOperator(&HGradFESpace, &HCurlFESpace);
   grad->AddDomainInterpolator(new GradientInterpolator());
   grad->Assemble();

   // no ParallelAssemble since this will be applied to GridFunctions
}

double MagneticDiffusionEOperator::ElectricLosses(ParGridFunction &E_gf) const
{
   double el = m1->InnerProduct(E_gf,E_gf);

   double global_el;
   MPI_Allreduce(&el, &global_el, 1, MPI_DOUBLE, MPI_SUM,
                 m2->ParFESpace()->GetComm());

   return el;
}

// E is the input GF, w is the output GF which is assumed to be an L2 scalar
// representing the Joule heating
void MagneticDiffusionEOperator::GetJouleHeating(ParGridFunction &E_gf,
                                                 ParGridFunction &w_gf) const
{
   // The w_coeff object stashes a reference to sigma and E, and it has
   // an Eval method that will be used by ProjectCoefficient.
   JouleHeatingCoefficient w_coeff(*sigma, E_gf);

   // This applies the definition of the finite element degrees-of-freedom
   // to convert the function to a set of discrete values
   w_gf.ProjectCoefficient(w_coeff);
}

void MagneticDiffusionEOperator::SetTime(const double _t)
{ t = _t; }

MagneticDiffusionEOperator::~MagneticDiffusionEOperator()
{
   if ( ams_a1 != NULL ) { delete ams_a1; }
   if ( pcg_a1 != NULL ) { delete pcg_a1; }

   if ( dsp_m1 != NULL ) { delete dsp_m1; }
   if ( pcg_m1 != NULL ) { delete pcg_m1; }

   if ( dsp_m2 != NULL ) { delete dsp_m2; }
   if ( pcg_m2 != NULL ) { delete pcg_m2; }

   if ( curl != NULL ) { delete curl; }
   if ( weakDiv != NULL ) { delete weakDiv; }
   if ( weakDivC != NULL ) { delete weakDivC; }
   if ( weakCurl != NULL ) { delete weakCurl; }
   if ( grad != NULL ) { delete grad; }

   if ( a0 != NULL ) { delete a0; }
   if ( a1 != NULL ) { delete a1; }
   if ( a2 != NULL ) { delete a2; }
   if ( m1 != NULL ) { delete m1; }
   if ( m2 != NULL ) { delete m2; }
   if ( s1 != NULL ) { delete s1; }
   if ( s2 != NULL ) { delete s2; }

   if ( A0 != NULL ) { delete A0; }
   if ( X0 != NULL ) { delete X0; }
   if ( B0 != NULL ) { delete B0; }

   if ( A1 != NULL ) { delete A1; }
   if ( X1 != NULL ) { delete X1; }
   if ( B1 != NULL ) { delete B1; }

   if ( A2 != NULL ) { delete A2; }
   if ( X2 != NULL ) { delete X2; }
   if ( B2 != NULL ) { delete B2; }

   if ( v1 != NULL ) { delete v1; }
   if ( v2 != NULL ) { delete v2; }

   if (sigma     != NULL) { delete sigma; }
   if (Tcapacity != NULL) { delete Tcapacity; }
   if (InvTcap   != NULL) { delete InvTcap; }
   if (InvTcond  != NULL) { delete InvTcond; }

   delete amg_a0;
   delete pcg_a0;
   delete pcg_a2;
   delete ads_a2;
   delete m3;
   delete dsp_m3;
   delete pcg_m3;
   delete M1;
   delete M2;
   delete M3;
   delete v0;
   delete B3;
}

void MagneticDiffusionEOperator::Debug(const char *base, double)
{
   {
      hypre_ParCSRMatrixPrint(*A1,"A1_");
      HypreParVector tempB1(A1->GetComm(),A1->N(),B1->GetData(),A1->ColPart());
      tempB1.Print("B1_");
      HypreParVector tempX1(A1->GetComm(),A1->N(),X1->GetData(),A1->ColPart());
      tempX1.Print("X1_");
   }

   {
      hypre_ParCSRMatrixPrint(*A2,"A2_");
      HypreParVector tempB2(A2->GetComm(),A2->N(),B2->GetData(),A2->ColPart());
      tempB2.Print("B2_");
      HypreParVector tempX2(A2->GetComm(),A2->N(),X2->GetData(),A2->ColPart());
      tempX2.Print("X2_");
   }
}

double JouleHeatingCoefficient::Eval(ElementTransformation &T,
                                     const IntegrationPoint &ip)
{
   Vector E;
   double thisSigma;
   E_gf.GetVectorValue(T, ip, E);
   thisSigma = sigma.Eval(T, ip);
   return thisSigma*(E*E);
}

MeshDependentCoefficient::MeshDependentCoefficient(
   const std::map<int, double> &inputMap, double scale)
   : Coefficient()
{
   // make a copy of the magic attribute-value map for later use
   materialMap = new std::map<int, double>(inputMap);
   scaleFactor = scale;
}

MeshDependentCoefficient::MeshDependentCoefficient(
   const MeshDependentCoefficient &cloneMe)
   : Coefficient()
{
   // make a copy of the magic attribute-value map for later use
   materialMap = new std::map<int, double>(*(cloneMe.materialMap));
   scaleFactor = cloneMe.scaleFactor;
}

double MeshDependentCoefficient::Eval(ElementTransformation &T,
                                      const IntegrationPoint &ip)
{
   // given the attribute, extract the coefficient value from the map
   std::map<int, double>::iterator it;
   int thisAtt = T.Attribute;
   double value;
   it = materialMap->find(thisAtt);
   if (it != materialMap->end())
   {
      value = it->second;
   }
   else
   {
      value = 0.0; // avoid compile warning
      std::cerr << "MeshDependentCoefficient attribute " << thisAtt
                << " not found" << std::endl;
      mfem_error();
   }

   return value*scaleFactor;
}

ScaledGFCoefficient::ScaledGFCoefficient(GridFunction *gf,
                                         MeshDependentCoefficient &input_mdc)
   : GridFunctionCoefficient(gf), mdc(input_mdc) {}

double ScaledGFCoefficient::Eval(ElementTransformation &T,
                                 const IntegrationPoint &ip)
{
   return mdc.Eval(T,ip) * GridFunctionCoefficient::Eval(T,ip);
}

} // namespace electromagnetics

} // namespace mfem

#endif // MFEM_USE_MPI
