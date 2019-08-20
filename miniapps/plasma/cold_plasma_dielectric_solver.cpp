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

#include "cold_plasma_dielectric_solver.hpp"

#ifdef MFEM_USE_MPI

using namespace std;

namespace mfem
{
using namespace miniapps;

namespace plasma
{

// Used for combining scalar coefficients
double prodFunc(double a, double b) { return a * b; }

EnergyDensityCoef::EnergyDensityCoef(double omega,
                                     VectorCoefficient &Er,
                                     VectorCoefficient &Ei,
                                     VectorCoefficient &dEr,
                                     VectorCoefficient &dEi,
                                     MatrixCoefficient &epsr,
                                     MatrixCoefficient &epsi,
                                     Coefficient &muInv)
   : omega_(omega),
     ErCoef_(Er),
     EiCoef_(Ei),
     dErCoef_(dEr),
     dEiCoef_(dEi),
     epsrCoef_(epsr),
     epsiCoef_(epsi),
     muInvCoef_(muInv),
     Er_(3),
     Ei_(3),
     Dr_(3),
     Di_(3),
     Br_(3),
     Bi_(3),
     eps_r_(3),
     eps_i_(3)
{}

double EnergyDensityCoef::Eval(ElementTransformation &T,
                               const IntegrationPoint &ip)
{
   ErCoef_.Eval(Er_, T, ip);
   EiCoef_.Eval(Ei_, T, ip);

   dErCoef_.Eval(Bi_, T, ip); Bi_ /=  omega_;
   dEiCoef_.Eval(Br_, T, ip); Br_ /= -omega_;

   epsrCoef_.Eval(eps_r_, T, ip);
   epsiCoef_.Eval(eps_i_, T, ip);

   eps_r_.Mult(Er_, Dr_);
   eps_i_.AddMult_a(-1.0, Ei_, Dr_);

   eps_i_.Mult(Er_, Di_);
   eps_r_.AddMult(Ei_, Di_);

   double muInv = muInvCoef_.Eval(T, ip);

   double u = (Er_ * Dr_) + (Ei_ * Di_) + ((Br_ * Br_) + (Bi_ * Bi_)) * muInv;

   return 0.5 * u;
}

PoyntingVectorReCoef::PoyntingVectorReCoef(double omega,
                                           VectorCoefficient &Er,
                                           VectorCoefficient &Ei,
                                           VectorCoefficient &dEr,
                                           VectorCoefficient &dEi,
                                           Coefficient &muInv)
   : VectorCoefficient(3),
     omega_(omega),
     ErCoef_(Er),
     EiCoef_(Ei),
     dErCoef_(dEr),
     dEiCoef_(dEi),
     muInvCoef_(muInv),
     Er_(3),
     Ei_(3),
     Hr_(3),
     Hi_(3)
{}

void PoyntingVectorReCoef::Eval(Vector &S, ElementTransformation &T,
                                const IntegrationPoint &ip)
{
   ErCoef_.Eval(Er_, T, ip);
   EiCoef_.Eval(Ei_, T, ip);

   double muInv = muInvCoef_.Eval(T, ip);

   dErCoef_.Eval(Hi_, T, ip); Hi_ *=  muInv / omega_;
   dEiCoef_.Eval(Hr_, T, ip); Hr_ *= -muInv / omega_;

   S.SetSize(3);

   S[0] = Er_[1] * Hr_[2] - Er_[2] * Hr_[1] +
          Ei_[1] * Hi_[2] - Ei_[2] * Hi_[1] ;

   S[1] = Er_[2] * Hr_[0] - Er_[0] * Hr_[2] +
          Ei_[2] * Hi_[0] - Ei_[0] * Hi_[2] ;

   S[2] = Er_[0] * Hr_[1] - Er_[1] * Hr_[0] +
          Ei_[0] * Hi_[1] - Ei_[1] * Hi_[0] ;

   S *= 0.5;
}

PoyntingVectorImCoef::PoyntingVectorImCoef(double omega,
                                           VectorCoefficient &Er,
                                           VectorCoefficient &Ei,
                                           VectorCoefficient &dEr,
                                           VectorCoefficient &dEi,
                                           Coefficient &muInv)
   : VectorCoefficient(3),
     omega_(omega),
     ErCoef_(Er),
     EiCoef_(Ei),
     dErCoef_(dEr),
     dEiCoef_(dEi),
     muInvCoef_(muInv),
     Er_(3),
     Ei_(3),
     Hr_(3),
     Hi_(3)
{}

void PoyntingVectorImCoef::Eval(Vector &S, ElementTransformation &T,
                                const IntegrationPoint &ip)
{
   ErCoef_.Eval(Er_, T, ip);
   EiCoef_.Eval(Ei_, T, ip);

   double muInv = muInvCoef_.Eval(T, ip);

   dErCoef_.Eval(Hi_, T, ip); Hi_ *=  muInv / omega_;
   dEiCoef_.Eval(Hr_, T, ip); Hr_ *= -muInv / omega_;

   S.SetSize(3);

   S[0] = Er_[1] * Hi_[2] - Er_[2] * Hi_[1] -
          Ei_[1] * Hr_[2] + Ei_[2] * Hr_[1] ;

   S[1] = Er_[2] * Hi_[0] - Er_[0] * Hi_[2] -
          Ei_[2] * Hr_[0] + Ei_[0] * Hr_[2] ;

   S[2] = Er_[0] * Hi_[1] - Er_[1] * Hi_[0] -
          Ei_[0] * Hr_[1] + Ei_[1] * Hr_[0] ;

   S *= 0.5;
}

CPDSolver::CPDSolver(ParMesh & pmesh, int order, double omega,
                     CPDSolver::SolverType sol, SolverOptions & sOpts,
                     CPDSolver::PrecondType prec,
                     ComplexOperator::Convention conv,
                     MatrixCoefficient & epsReCoef,
                     MatrixCoefficient & epsImCoef,
                     MatrixCoefficient & epsAbsCoef,
                     Coefficient & muInvCoef,
                     Coefficient * etaInvCoef,
                     Coefficient * etaInvReCoef,
                     Coefficient * etaInvImCoef,
                     VectorCoefficient * kCoef,
                     Array<int> & abcs,
                     Array<int> & sbcs,
                     // Array<int> & dbcs,
                     Array<ComplexVectorCoefficientByAttr> & dbcs,
                     // void   (*e_r_bc )(const Vector&, Vector&),
                     // void   (*e_i_bc )(const Vector&, Vector&),
                     // VectorCoefficient & EReCoef,
                     // VectorCoefficient & EImCoef,
                     void (*j_r_src)(const Vector&, Vector&),
                     void (*j_i_src)(const Vector&, Vector&),
                     bool vis_u)
   : myid_(0),
     num_procs_(1),
     order_(order),
     logging_(1),
     sol_(sol),
     solOpts_(sOpts),
     prec_(prec),
     conv_(conv),
     ownsEtaInv_(etaInvCoef == NULL),
     vis_u_(vis_u),
     omega_(omega),
     // solNorm_(-1.0),
     pmesh_(&pmesh),
     L2FESpace_(NULL),
     L2VFESpace_(NULL),
     H1FESpace_(NULL),
     HCurlFESpace_(NULL),
     HDivFESpace_(NULL),
     a_(4 - pmesh.Dimension(), 4 - pmesh.Dimension()),
     am_(4 - pmesh.Dimension(), 4 - pmesh.Dimension()),
     b_(4 - pmesh.Dimension()),
     e_(4 - pmesh.Dimension()),
     j_(4 - pmesh.Dimension()),
     rhs_(4 - pmesh.Dimension()),
     e_t_(NULL),
     e_v_(4 - pmesh.Dimension()),
     j_v_(NULL),
     u_(NULL),
     S_(NULL),
     epsReCoef_(&epsReCoef),
     epsImCoef_(&epsImCoef),
     epsAbsCoef_(&epsAbsCoef),
     muInvCoef_(&muInvCoef),
     etaInvCoef_(etaInvCoef),
     etaInvReCoef_(etaInvReCoef),
     etaInvImCoef_(etaInvImCoef),
     kCoef_(kCoef),
     omegaCoef_(new ConstantCoefficient(omega_)),
     negOmegaCoef_(new ConstantCoefficient(-omega_)),
     omega2Coef_(new ConstantCoefficient(pow(omega_, 2))),
     negOmega2Coef_(new ConstantCoefficient(-pow(omega_, 2))),
     abcCoef_(NULL),
     sbcReCoef_(NULL),
     sbcImCoef_(NULL),
     sinkx_(NULL),
     coskx_(NULL),
     negsinkx_(NULL),
     negMuInvCoef_(NULL),
     massReCoef_(NULL),
     massImCoef_(NULL),
     posMassCoef_(NULL),
     negMuInvkxkxCoef_(NULL),
     massRe2x2Coef_(NULL),
     massRe2x1Coef_(NULL),
     massRe1x2Coef_(NULL),
     massReZZCoef_(NULL),
     massIm2x2Coef_(NULL),
     massIm2x1Coef_(NULL),
     massIm1x2Coef_(NULL),
     massImZZCoef_(NULL),
     posMass2x2Coef_(NULL),
     posMassZZCoef_(NULL),
     negMuInvkCoef_(NULL),
     jrCoef_(NULL),
     jiCoef_(NULL),
     rhsrCoef_(NULL),
     rhsiCoef_(NULL),
     jr2x1Coef_(NULL),
     ji2x1Coef_(NULL),
     jrZCoef_(NULL),
     jiZCoef_(NULL),
     rhsr2x1Coef_(NULL),
     rhsi2x1Coef_(NULL),
     rhsrZCoef_(NULL),
     rhsiZCoef_(NULL),
     // erCoef_(EReCoef),
     // eiCoef_(EImCoef),
     derCoef_(NULL),
     deiCoef_(NULL),
     uCoef_(omega_, erCoef_, eiCoef_, derCoef_, deiCoef_,
            *epsReCoef_, *epsImCoef_, *muInvCoef_),
     SrCoef_(omega_, erCoef_, eiCoef_, derCoef_, deiCoef_, *muInvCoef_),
     SiCoef_(omega_, erCoef_, eiCoef_, derCoef_, deiCoef_, *muInvCoef_),
     j_r_src_(j_r_src),
     j_i_src_(j_i_src),
     // e_r_bc_(e_r_bc),
     // e_i_bc_(e_i_bc),
     dbcs_(&dbcs),
     visit_dc_(NULL)
{
   // Initialize MPI variables
   MPI_Comm_size(pmesh_->GetComm(), &num_procs_);
   MPI_Comm_rank(pmesh_->GetComm(), &myid_);

   if ( myid_ == 0 && logging_ > 0 )
   {
      cout << "Constructing CPDSolver ..." << endl;
   }

   tic_toc.Clear();
   tic_toc.Start();

   a_  = NULL;
   am_ = NULL;
   b_  = NULL;
   e_  = NULL;
   j_  = NULL;
   rhs_ = NULL;

   // Define compatible parallel finite element spaces on the parallel
   // mesh. Here we use arbitrary order H1, Nedelec, and Raviart-Thomas finite
   // elements.
   if (pmesh_->Dimension() < 3)
   {
      H1FESpace_    = new H1_ParFESpace(pmesh_,order,pmesh_->Dimension());
   }
   HCurlFESpace_ = new ND_ParFESpace(pmesh_,order,pmesh_->Dimension());

   if (kCoef_)
   {
      L2VFESpace_ = new L2_ParFESpace(pmesh_,order,pmesh_->Dimension(),
                                      pmesh_->SpaceDimension());
      e_t_ = new ParGridFunction(L2VFESpace_);
      e_v_[0] = new ParComplexGridFunction(L2VFESpace_);
      j_v_ = new ParComplexGridFunction(L2VFESpace_);

      sinkx_ = new PhaseCoefficient(*kCoef_, &sin);
      coskx_ = new PhaseCoefficient(*kCoef_, &cos);
      negsinkx_ = new ProductCoefficient(-1.0, *sinkx_);

      negMuInvCoef_ = new ProductCoefficient(-1.0, *muInvCoef_);
      negMuInvkCoef_ = new ScalarVectorProductCoefficient(*negMuInvCoef_,
                                                          *kCoef_);
      negMuInvkxkxCoef_ = new CrossCrossCoefficient(*muInvCoef_, *kCoef_);
   }
   else
   {
      e_t_ = new ParGridFunction(HCurlFESpace_);
   }

   // HDivFESpace_  = new RT_ParFESpace(pmesh_,order,pmesh_->Dimension());
   if (false)
   {
      GridFunction * nodes = pmesh_->GetNodes();
      cout << "nodes is " << nodes << endl;
      for (int i=0; i<HCurlFESpace_->GetNBE(); i++)
      {
         const FiniteElement &be = *HCurlFESpace_->GetBE(i);
         ElementTransformation *eltrans =
            HCurlFESpace_->GetBdrElementTransformation (i);
         cout << i << '\t' << pmesh_->GetBdrAttribute(i)
              << '\t' << be.GetGeomType()
              << '\t' << eltrans->ElementNo
              << '\t' << eltrans->Attribute
              << endl;
      }
   }

   blockTrueOffsets_.SetSize(1 + 2 * (4 - pmesh_->Dimension()));
   blockTrueOffsets_[0] = 0;
   blockTrueOffsets_[1] = HCurlFESpace_->TrueVSize();
   blockTrueOffsets_[2] = HCurlFESpace_->TrueVSize();
   if (pmesh_->Dimension() < 3)
   {
      blockTrueOffsets_[3] = H1FESpace_->TrueVSize();
      blockTrueOffsets_[4] = H1FESpace_->TrueVSize();
   }
   if (pmesh_->Dimension() < 2)
   {
      blockTrueOffsets_[5] = H1FESpace_->TrueVSize();
      blockTrueOffsets_[6] = H1FESpace_->TrueVSize();
   }
   blockTrueOffsets_.PartialSum();

   // int irOrder = H1FESpace_->GetElementTransformation(0)->OrderW()
   //            + 2 * order;
   // int geom = H1FESpace_->GetFE(0)->GetGeomType();
   // const IntegrationRule * ir = &IntRules.Get(geom, irOrder);
   /*
   // Select surface attributes for Dirichlet BCs
   ess_bdr_.SetSize(pmesh.bdr_attributes.Max());
   non_k_bdr_.SetSize(pmesh.bdr_attributes.Max());
   ess_bdr_ = 1;   // All outer surfaces
   non_k_bdr_ = 1; // Surfaces without applied surface currents
   for (int i=0; i<kbcs.Size(); i++)
   {
      non_k_bdr_[kbcs[i]-1] = 0;
   }
   */

   ess_bdr_.SetSize(pmesh.bdr_attributes.Max());
   if ( dbcs_->Size() > 0 )
   {
      if ( dbcs_->Size() == 1 && (*dbcs_)[0].attr[0] == -1 )
      {
         ess_bdr_ = 1;
      }
      else
      {
         ess_bdr_ = 0;
         for (int i=0; i<dbcs_->Size(); i++)
         {
            for (int j=0; j<(*dbcs_)[i].attr.Size(); j++)
            {
               ess_bdr_[(*dbcs_)[i].attr[j]-1] = 1;
            }
         }
      }
      HCurlFESpace_->GetEssentialTrueDofs(ess_bdr_, ess_bdr_nd_tdofs_);
      if (H1FESpace_)
      {
         H1FESpace_->GetEssentialTrueDofs(ess_bdr_, ess_bdr_h1_tdofs_);
      }
   }

   // Setup various coefficients
   /*
   // Vector Potential on the outer surface
   if ( a_bc_ == NULL )
   {
      Vector Zero(3);
      Zero = 0.0;
      aBCCoef_ = new VectorConstantCoefficient(Zero);
   }
   else
   {
      aBCCoef_ = new VectorFunctionCoefficient(pmesh_->SpaceDimension(),
                                               *a_bc_);
   }
   */

   massReCoef_ = new ScalarMatrixProductCoefficient(*negOmega2Coef_,
                                                    *epsReCoef_);
   massImCoef_ = new ScalarMatrixProductCoefficient(*negOmega2Coef_,
                                                    *epsImCoef_);
   posMassCoef_ = new ScalarMatrixProductCoefficient(*omega2Coef_,
                                                     *epsAbsCoef_);

   if (pmesh_->Dimension() == 2)
   {
      massRe2x2Coef_ = new SubMatrixCoefficient(*massReCoef_, 2, 2, 0, 0);
      massIm2x2Coef_ = new SubMatrixCoefficient(*massImCoef_, 2, 2, 0, 0);

      massRe2x1Coef_ = new SubVectorCoefficient(*massReCoef_, 2, 1, 0, 2);
      massIm2x1Coef_ = new SubVectorCoefficient(*massImCoef_, 2, 1, 0, 2);
      massRe1x2Coef_ = new SubVectorCoefficient(*massReCoef_, 1, 2, 2, 0);
      massIm1x2Coef_ = new SubVectorCoefficient(*massImCoef_, 1, 2, 2, 0);

      massReZZCoef_  = new MatrixEntryCoefficient(*massReCoef_, 2, 2);
      massImZZCoef_  = new MatrixEntryCoefficient(*massImCoef_, 2, 2);

      posMass2x2Coef_ = new SubMatrixCoefficient(*posMassCoef_, 2, 2, 0, 0);
      posMassZZCoef_  = new MatrixEntryCoefficient(*posMassCoef_, 2, 2);
   }

   // Impedance of free space
   if ( abcs.Size() > 0 )
   {
      if ( myid_ == 0 && logging_ > 0 )
      {
         cout << "Creating Admittance Coefficient" << endl;
      }

      abc_marker_.SetSize(pmesh.bdr_attributes.Max());
      if ( abcs.Size() == 1 && abcs[0] < 0 )
      {
         // Mark all boundaries as absorbing
         abc_marker_ = 1;
      }
      else
      {
         // Mark select boundaries as absorbing
         abc_marker_ = 0;
         for (int i=0; i<abcs.Size(); i++)
         {
            abc_marker_[abcs[i]-1] = 1;
         }
      }
      if ( etaInvCoef_ == NULL )
      {
         etaInvCoef_ = new ConstantCoefficient(sqrt(epsilon0_/mu0_));
      }
      abcCoef_ = new TransformedCoefficient(negOmegaCoef_, etaInvCoef_,
                                            prodFunc);
   }

   // Complex Impedance
   if ( sbcs.Size() > 0 && etaInvReCoef_ != NULL && etaInvReCoef_ != NULL )
   {
      if ( myid_ == 0 && logging_ > 0 )
      {
         cout << "Creating Complex Admittance Coefficient" << endl;
      }

      sbc_marker_.SetSize(pmesh.bdr_attributes.Max());

      // Mark select boundaries as absorbing
      sbc_marker_ = 0;
      for (int i=0; i<sbcs.Size(); i++)
      {
         sbc_marker_[sbcs[i]-1] = 1;
      }

      sbcReCoef_ = new TransformedCoefficient(omegaCoef_, etaInvImCoef_,
                                              prodFunc);
      sbcImCoef_ = new TransformedCoefficient(negOmegaCoef_, etaInvReCoef_,
                                              prodFunc);
   }

   // Volume Current Density
   if ( j_r_src_ != NULL )
   {
      jrCoef_ = new VectorFunctionCoefficient(pmesh_->SpaceDimension(),
                                              j_r_src_);
   }
   else
   {
      Vector j(3); j = 0.0;
      jrCoef_ = new VectorConstantCoefficient(j);
   }
   if ( j_i_src_ != NULL )
   {
      jiCoef_ = new VectorFunctionCoefficient(pmesh_->SpaceDimension(),
                                              j_i_src_);
   }
   else
   {
      Vector j(3); j = 0.0;
      jiCoef_ = new VectorConstantCoefficient(j);
   }

   rhsrCoef_ = new ScalarVectorProductCoefficient(omega_, *jiCoef_);
   rhsiCoef_ = new ScalarVectorProductCoefficient(-omega_, *jrCoef_);
   if (pmesh_->Dimension() == 2)
   {
      jr2x1Coef_ = new SubVectorCoefficient(*jrCoef_, 2, 0);
      ji2x1Coef_ = new SubVectorCoefficient(*jiCoef_, 2, 0);
      jrZCoef_   = new VectorEntryCoefficient(*jrCoef_, 2);
      jiZCoef_   = new VectorEntryCoefficient(*jiCoef_, 2);

      rhsr2x1Coef_ = new SubVectorCoefficient(*rhsrCoef_, 2, 0);
      rhsi2x1Coef_ = new SubVectorCoefficient(*rhsiCoef_, 2, 0);
      rhsrZCoef_   = new VectorEntryCoefficient(*rhsrCoef_, 2);
      rhsiZCoef_   = new VectorEntryCoefficient(*rhsiCoef_, 2);
   }

   /*
   // Magnetization
   if ( m_src_ != NULL )
   {
      mCoef_ = new VectorFunctionCoefficient(pmesh_->SpaceDimension(),
                                             m_src_);
   }
   */

   // Bilinear Forms
   a_(0,0) = new ParSesquilinearForm(HCurlFESpace_, conv_);
   a_(0,0)->AddDomainIntegrator(new CurlCurlIntegrator(*muInvCoef_), NULL);

   b_[0] = new ParBilinearForm(HCurlFESpace_);
   b_[0]->AddDomainIntegrator(new CurlCurlIntegrator(*muInvCoef_));
   // b1_->AddDomainIntegrator(new VectorFEMassIntegrator(*epsAbsCoef_));
   //b1_->AddDomainIntegrator(new VectorFEMassIntegrator(*massImCoef_));

   rhs_[0] = new ParComplexLinearForm(HCurlFESpace_, conv_);

   switch (pmesh_->Dimension())
   {
      case 2:
         a_(0,0)->AddDomainIntegrator(new VectorFEMassIntegrator(*massRe2x2Coef_),
                                      new VectorFEMassIntegrator(*massIm2x2Coef_));

         a_(1,1) = new ParSesquilinearForm(H1FESpace_, conv_);
         a_(1,1)->AddDomainIntegrator(new DiffusionIntegrator(*muInvCoef_), NULL);
         a_(1,1)->AddDomainIntegrator(new MassIntegrator(*massReZZCoef_),
                                      new MassIntegrator(*massImZZCoef_));

         am_(0,1) = new ParMixedSesquilinearForm(H1FESpace_, HCurlFESpace_, conv_);
         am_(0,1)->AddDomainIntegrator(new MixedVectorProductIntegrator(*massRe2x1Coef_),
                                       new MixedVectorProductIntegrator(*massIm2x1Coef_));

         am_(1,0) = new ParMixedSesquilinearForm(HCurlFESpace_, H1FESpace_, conv_);
         am_(1,0)->AddDomainIntegrator(new MixedDotProductIntegrator(*massRe1x2Coef_),
                                       new MixedDotProductIntegrator(*massIm1x2Coef_));
         b_[0]->AddDomainIntegrator(new VectorFEMassIntegrator(*posMass2x2Coef_));

         b_[1] = new ParBilinearForm(H1FESpace_);
         b_[1]->AddDomainIntegrator(new DiffusionIntegrator(*muInvCoef_));
         b_[1]->AddDomainIntegrator(new MassIntegrator(*posMassZZCoef_));

         rhs_[0]->AddDomainIntegrator(new VectorFEDomainLFIntegrator(*rhsr2x1Coef_),
                                      new VectorFEDomainLFIntegrator(*rhsi2x1Coef_));

         rhs_[1] = new ParComplexLinearForm(H1FESpace_, conv_);
         rhs_[1]->AddDomainIntegrator(new DomainLFIntegrator(*rhsrZCoef_),
                                      new DomainLFIntegrator(*rhsiZCoef_));
         break;
      case 3:
         a_(0,0)->AddDomainIntegrator(new VectorFEMassIntegrator(*massReCoef_),
                                      new VectorFEMassIntegrator(*massImCoef_));

         b_[0]->AddDomainIntegrator(new VectorFEMassIntegrator(*posMassCoef_));

         rhs_[0]->AddDomainIntegrator(new VectorFEDomainLFIntegrator(*rhsrCoef_),
                                      new VectorFEDomainLFIntegrator(*rhsiCoef_));
         break;
   }

   for (int i=0; i<rhs_.Size(); i++)
   {
      rhs_[i]->real().Vector::operator=(0.0);
      rhs_[i]->imag().Vector::operator=(0.0);
   }

   if ( kCoef_)
   {
      a_(0,0)->AddDomainIntegrator(new VectorFEMassIntegrator(*negMuInvkxkxCoef_),
                                   NULL);
      a_(0,0)->AddDomainIntegrator(NULL,
                                   new MixedCrossCurlIntegrator(*negMuInvkCoef_));
      a_(0,0)->AddDomainIntegrator(NULL,
                                   new MixedWeakCurlCrossIntegrator(*negMuInvkCoef_));
   }
   if ( abcCoef_ )
   {
      a_(0,0)->AddBoundaryIntegrator(NULL, new VectorFEMassIntegrator(*abcCoef_),
                                     abc_marker_);
   }
   if ( sbcReCoef_ && sbcImCoef_ )
   {
      a_(0,0)->AddBoundaryIntegrator(new VectorFEMassIntegrator(*sbcReCoef_),
                                     new VectorFEMassIntegrator(*sbcImCoef_),
                                     sbc_marker_);
   }

   // Build grid functions
   e_[0]  = new ParComplexGridFunction(HCurlFESpace_);
   *e_[0] = 0.0;
   // solNorm_ = e_->ComputeL2Error(const_cast<VectorCoefficient&>(erCoef_),
   //                               const_cast<VectorCoefficient&>(eiCoef_));

   j_[0] = new ParComplexGridFunction(HCurlFESpace_);
   switch (pmesh_->Dimension())
   {
      case 2:
         e_[1]  = new ParComplexGridFunction(H1FESpace_);
         *e_[1] = 0.0;

         j_[0]->ProjectCoefficient(*jr2x1Coef_, *ji2x1Coef_);
         j_[1] = new ParComplexGridFunction(H1FESpace_);
         j_[1]->ProjectCoefficient(*jrZCoef_, *jiZCoef_);
         break;
      case 3:
         j_[0]->ProjectCoefficient(*jrCoef_, *jiCoef_);
         break;
   }

   if (vis_u_)
   {
      L2FESpace_ = new L2_ParFESpace(pmesh_,2*order-1,pmesh_->Dimension());
      u_ = new ParGridFunction(L2FESpace_);

      HDivFESpace_ = new RT_ParFESpace(pmesh_,2*order,pmesh_->Dimension());
      S_ = new ParComplexGridFunction(HDivFESpace_);

      erCoef_.SetGridFunction(&e_[0]->real());
      eiCoef_.SetGridFunction(&e_[0]->imag());

      derCoef_.SetGridFunction(&e_[0]->real());
      deiCoef_.SetGridFunction(&e_[0]->imag());
   }

   tic_toc.Stop();

   if ( myid_ == 0 && logging_ > 0 )
   {
      cout << " done in " << tic_toc.RealTime() << " seconds." << endl;
   }
}

CPDSolver::~CPDSolver()
{
   delete negMuInvkxkxCoef_;
   delete negMuInvkCoef_;
   delete negMuInvCoef_;
   delete negsinkx_;
   delete coskx_;
   delete sinkx_;
   delete rhsr2x1Coef_;
   delete rhsi2x1Coef_;
   delete rhsrZCoef_;
   delete rhsiZCoef_;
   delete jr2x1Coef_;
   delete ji2x1Coef_;
   delete jrZCoef_;
   delete jiZCoef_;
   delete rhsrCoef_;
   delete rhsiCoef_;
   delete jrCoef_;
   delete jiCoef_;
   // delete erCoef_;
   // delete eiCoef_;

   delete massRe2x2Coef_;
   delete massRe2x1Coef_;
   delete massRe1x2Coef_;
   delete massReZZCoef_;

   delete massIm2x2Coef_;
   delete massIm2x1Coef_;
   delete massIm1x2Coef_;
   delete massImZZCoef_;

   delete posMass2x2Coef_;
   delete posMassZZCoef_;

   delete massReCoef_;
   delete massImCoef_;
   delete posMassCoef_;
   delete abcCoef_;
   delete sbcReCoef_;
   delete sbcImCoef_;
   if ( ownsEtaInv_ ) { delete etaInvCoef_; }
   delete omegaCoef_;
   delete negOmegaCoef_;
   delete omega2Coef_;
   delete negOmega2Coef_;

   // delete DivFreeProj_;
   // delete SurfCur_;

   for (int i=0; i<e_v_.Size(); i++)
      if (e_v_[i] != e_[i]) { delete e_v_[i]; }
   if (j_v_ != j_[0]) { delete j_v_; }
   // delete e_r_;
   // delete e_i_;
   for (int i=0; i<e_.Size(); i++) { delete e_[i]; }
   // delete b_;
   // delete h_;
   for (int i=0; i<j_.Size(); i++) { delete j_[i]; }
   delete u_;
   // delete j_r_;
   // delete j_i_;
   // delete j_;
   // delete k_;
   // delete m_;
   // delete bd_;
   for (int i=0; i<rhs_.Size(); i++) { delete rhs_[i]; }
   delete e_t_;
   // delete jd_r_;
   // delete jd_i_;
   // delete grad_;
   // delete curl_;

   for (int i=0; i<a_.NumRows(); i++)
      for (int j=0; j<a_.NumCols(); j++)
      {
         delete a_(i,j);
      }

   for (int i=0; i<am_.NumRows(); i++)
      for (int j=0; j<am_.NumCols(); j++)
      {
         delete am_(i,j);
      }

   for (int i=0; i<b_.Size(); i++) { delete b_[i]; }
   // delete curlMuInvCurl_;
   // delete hCurlMass_;
   // delete hDivHCurlMuInv_;
   // delete weakCurlMuInv_;

   delete L2FESpace_;
   delete L2VFESpace_;
   delete H1FESpace_;
   delete HCurlFESpace_;
   delete HDivFESpace_;

   map<string,socketstream*>::iterator mit;
   for (mit=socks_.begin(); mit!=socks_.end(); mit++)
   {
      delete mit->second;
   }
}

HYPRE_Int
CPDSolver::GetProblemSize()
{
   if (pmesh_->Dimension() == 1)
   {
      return 2 * (HCurlFESpace_->GlobalTrueVSize() +
                  2 * H1FESpace_->GlobalTrueVSize());
   }
   else if (pmesh_->Dimension() == 2)
   {
      return 2 * (HCurlFESpace_->GlobalTrueVSize() +
                  H1FESpace_->GlobalTrueVSize());
   }
   return 2 * HCurlFESpace_->GlobalTrueVSize();
}

void
CPDSolver::PrintSizes()
{
   HYPRE_Int size_h1 = (H1FESpace_) ? H1FESpace_->GlobalTrueVSize() : 0;
   HYPRE_Int size_nd = HCurlFESpace_->GlobalTrueVSize();
   // HYPRE_Int size_rt = HDivFESpace_->GlobalTrueVSize();
   HYPRE_Int size_pr = this->GetProblemSize();
   if (myid_ == 0)
   {
      cout << "Number of H1      unknowns: " << size_h1 << endl;
      cout << "Number of H(Curl) unknowns: " << size_nd << endl;
      // cout << "Number of H(Div)  unknowns: " << size_rt << endl;
      cout << "Total Number of unknowns:   " << size_pr << endl;
   }
}

void
CPDSolver::Assemble()
{
   if ( myid_ == 0 && logging_ > 0 ) { cout << "Assembling ..." << flush; }

   tic_toc.Clear();
   tic_toc.Start();

   // a0_->Assemble();
   // a0_->Finalize();

   for (int i=0; i<a_.NumRows(); i++)
      for (int j=0; j<a_.NumCols(); j++)
         if (a_(i,j))
         {
            a_(i,j)->Assemble();
            a_(i,j)->Finalize();
         }

   for (int i=0; i<am_.NumRows(); i++)
      for (int j=0; j<am_.NumCols(); j++)
         if (am_(i,j))
         {
            am_(i,j)->Assemble();
            am_(i,j)->Finalize();
         }

   for (int i=0; i<b_.Size(); i++)
      if (b_[i])
      {
         b_[i]->Assemble();
         b_[i]->Finalize();
      }

   for (int i=0; i<rhs_.Size(); i++)
      if (rhs_[i])
      {
         rhs_[i]->Assemble();
      }
   /*
   curlMuInvCurl_->Assemble();
   curlMuInvCurl_->Finalize();
   hDivHCurlMuInv_->Assemble();
   hDivHCurlMuInv_->Finalize();
   hCurlMass_->Assemble();
   hCurlMass_->Finalize();
   curl_->Assemble();
   curl_->Finalize();
   if ( grad_ )
   {
      grad_->Assemble();
      grad_->Finalize();
   }
   if ( weakCurlMuInv_ )
   {
      weakCurlMuInv_->Assemble();
      weakCurlMuInv_->Finalize();
   }
   */
   tic_toc.Stop();

   if ( myid_ == 0 && logging_ > 0 )
   {
      cout << " done in " << tic_toc.RealTime() << " seconds." << endl;
   }
}

void
CPDSolver::Update()
{
   if ( myid_ == 0 && logging_ > 0 ) { cout << "Updating ..." << endl; }

   tic_toc.Clear();
   tic_toc.Start();

   // Inform the spaces that the mesh has changed
   // Note: we don't need to interpolate any GridFunctions on the new mesh
   // so we pass 'false' to skip creation of any transformation matrices.
   // H1FESpace_->Update(false);
   if (L2FESpace_) { L2FESpace_->Update(); }
   if (L2VFESpace_) { L2VFESpace_->Update(); }
   if (H1FESpace_) { H1FESpace_->Update(); }
   HCurlFESpace_->Update();
   if (HDivFESpace_) { HDivFESpace_->Update(false); }

   if ( ess_bdr_.Size() > 0 )
   {
      HCurlFESpace_->GetEssentialTrueDofs(ess_bdr_, ess_bdr_nd_tdofs_);
      if (H1FESpace_)
      {
         H1FESpace_->GetEssentialTrueDofs(ess_bdr_, ess_bdr_h1_tdofs_);
      }
   }

   blockTrueOffsets_[0] = 0;
   blockTrueOffsets_[1] = HCurlFESpace_->TrueVSize();
   blockTrueOffsets_[2] = HCurlFESpace_->TrueVSize();
   if (pmesh_->Dimension() < 3)
   {
      blockTrueOffsets_[3] = H1FESpace_->TrueVSize();
      blockTrueOffsets_[4] = H1FESpace_->TrueVSize();
   }
   if (pmesh_->Dimension() < 2)
   {
      blockTrueOffsets_[5] = H1FESpace_->TrueVSize();
      blockTrueOffsets_[6] = H1FESpace_->TrueVSize();
   }
   blockTrueOffsets_.PartialSum();

   // Inform the grid functions that the space has changed.
   for (int i=0; i<e_.Size(); i++) { e_[i]->Update(); }
   if (u_) { u_->Update(); }
   if (S_) { S_->Update(); }
   if (e_t_) { e_t_->Update(); }
   for (int i=0; i<e_v_.Size(); i++)
      if (e_v_[i] && e_v_[i] != e_[i]) { e_v_[i]->Update(); }
   if (j_v_) { j_v_->Update(); }
   // e_r_->Update();
   // e_i_->Update();
   // h_->Update();
   // b_->Update();
   // bd_->Update();
   for (int i=0; i<rhs_.Size(); i++)
      if (rhs_[i])
      {
         rhs_[i]->Update();
      }
   // jd_i_->Update();
   // if ( jr_ ) { jr_->Update(); }
   for (int i=0; i<j_.Size(); i++) if ( j_[i] ) {  j_[i]->Update(); }
   // if ( j_r_  ) {  j_r_->Update(); }
   // if ( j_i_  ) {  j_i_->Update(); }
   // if ( k_  ) {  k_->Update(); }
   // if ( m_  ) {  m_->Update(); }

   // Inform the bilinear forms that the space has changed.
   // a0_->Update();
   for (int i=0; i<a_.NumRows(); i++)
      for (int j=0; j<a_.NumCols(); j++)
         if (a_(i,j))
         {
            a_(i,j)->Update();
         }

   for (int i=0; i<am_.NumRows(); i++)
      for (int j=0; j<am_.NumCols(); j++)
         if (am_(i,j))
         {
            am_(i,j)->Update();
         }

   for (int i=0; i<b_.Size(); i++)
      if (b_[i])
      {
         b_[i]->Update();
      }
   // curlMuInvCurl_->Update();
   // hCurlMass_->Update();
   // hDivHCurlMuInv_->Update();
   // if ( weakCurlMuInv_ ) { weakCurlMuInv_->Update(); }

   // Inform the other objects that the space has changed.
   // curl_->Update();
   // if ( grad_        ) { grad_->Update(); }
   // if ( DivFreeProj_ ) { DivFreeProj_->Update(); }
   // if ( SurfCur_     ) { SurfCur_->Update(); }
   tic_toc.Stop();

   if ( myid_ == 0 && logging_ > 0 )
   {
      cout << " done in " << tic_toc.RealTime() << " seconds." << endl;
   }
}

void
CPDSolver::Solve()
{
   if ( myid_ == 0 && logging_ > 0 ) { cout << "Running solver ... " << endl; }

   // cout << "Norm of e (pre-fls): " << e_[0]->real().Norml2() << endl;
   if (dbcs_->Size() > 0)
   {
      Array<int> attr_marker(pmesh_->bdr_attributes.Max());
      for (int i = 0; i<dbcs_->Size(); i++)
      {
         attr_marker = 0;
         for (int j=0; j<(*dbcs_)[i].attr.Size(); j++)
         {
            attr_marker[(*dbcs_)[i].attr[j] - 1] = 1;
         }
         switch (pmesh_->Dimension())
         {
            case 2:
            {
               SubVectorCoefficient   ReXY(*(*dbcs_)[i].real, 2, 0);
               VectorEntryCoefficient ReZ(*(*dbcs_)[i].real, 2);
               SubVectorCoefficient   ImXY(*(*dbcs_)[i].imag, 2, 0);
               VectorEntryCoefficient ImZ(*(*dbcs_)[i].imag, 2);
               e_[0]->ProjectBdrCoefficientTangent(ReXY, ImXY,
                                                   attr_marker);
               e_[1]->ProjectBdrCoefficient(ReZ, ImZ,
                                            attr_marker);
            }
            break;
            case 3:
               e_[0]->ProjectBdrCoefficientTangent(*(*dbcs_)[i].real,
                                                   *(*dbcs_)[i].imag,
                                                   attr_marker);
               break;
         }
      }
   }

   // cout << "Norm of e (post-fls): " << e_[0]->real().Norml2() << endl;
   // cout << "Norm of RHS: " << RHS.Norml2() << endl;

   Array<int> op_sizes(b_.Size()+1); op_sizes = 0;
   Array<OperatorHandle> PCOp(b_.Size());
   b_[0]->FormSystemMatrix(ess_bdr_nd_tdofs_, PCOp[0]);
   if (pmesh_->Dimension() < 3)
   {
      b_[1]->FormSystemMatrix(ess_bdr_h1_tdofs_, PCOp[1]);
   }
   if (pmesh_->Dimension() < 2)
   {
      b_[2]->FormSystemMatrix(ess_bdr_h1_tdofs_, PCOp[2]);
   }

   for (int i=0; i<b_.Size(); i++)
   {
      op_sizes[i+1] = 2 * PCOp[i]->Height();
   }
   op_sizes.PartialSum();
   int prob_size = op_sizes[b_.Size()];

   Array<Vector> BlockE(b_.Size());
   Array<Vector> BlockRHS(b_.Size());
   Vector E(prob_size), RHS(prob_size);

   BlockE[0].SetDataAndSize(&E[0], op_sizes[1]);
   BlockRHS[0].SetDataAndSize(&RHS[0], op_sizes[1]);

   Array<OperatorHandle> A1(b_.Size());
   a_(0,0)->FormLinearSystem(ess_bdr_nd_tdofs_, *e_[0], *rhs_[0],
                             A1[0], BlockE[0], BlockRHS[0]);
   for (int i=1; i<b_.Size(); i++)
   {
      BlockE[i].SetDataAndSize(&E[op_sizes[i]], op_sizes[i+1] - op_sizes[i]);
      BlockRHS[i].SetDataAndSize(&RHS[op_sizes[i]], op_sizes[i+1] - op_sizes[i]);
      a_(i,i)->FormLinearSystem(ess_bdr_h1_tdofs_, *e_[i], *rhs_[i],
                                A1[i], BlockE[i], BlockRHS[i]);
   }

   tic_toc.Clear();
   tic_toc.Start();

   Array<Operator*> pcr(b_.Size()); pcr = NULL;
   Array<Operator*> pci(b_.Size()); pci = NULL;
   BlockDiagonalPreconditioner * BDP = NULL;

   if (sol_ == GMRES || sol_ == FGMRES || sol_ == MINRES)
   {
      bool havePC = false;
      switch (prec_)
      {
         case INVALID_PC:
            if ( myid_ == 0 && logging_ > 0 )
            {
               cout << "No Preconditioner Requested" << endl;
            }
            break;
         case DIAG_SCALE:
            if ( myid_ == 0 && logging_ > 0 )
            {
               cout << "Diagonal Scaling Preconditioner Requested" << endl;
            }
            for (int i=0; i<b_.Size(); i++)
            {
               pcr[i] = new HypreDiagScale(dynamic_cast<HypreParMatrix&>(*PCOp[i].Ptr()));
            }
            havePC = true;
            break;
         case PARASAILS:
            if ( myid_ == 0 && logging_ > 0 )
            {
               cout << "ParaSails Preconditioner Requested" << endl;
            }
            for (int i=0; i<b_.Size(); i++)
            {
               pcr[i] = new HypreParaSails(dynamic_cast<HypreParMatrix&>(*PCOp[i].Ptr()));
               dynamic_cast<HypreParaSails*>(pcr[i])->SetSymmetry(1);
            }
            havePC = true;
            break;
         case EUCLID:
            if ( myid_ == 0 && logging_ > 0 )
            {
               cout << "Euclid Preconditioner Requested" << endl;
            }
            for (int i=0; i<b_.Size(); i++)
            {
               pcr[i] = new HypreEuclid(dynamic_cast<HypreParMatrix&>(*PCOp[i].Ptr()));
               if (solOpts_.euLvl != 1)
               {
                  HypreSolver * pc = dynamic_cast<HypreSolver*>(pcr[i]);
                  HYPRE_EuclidSetLevel(*pc, solOpts_.euLvl);
               }
            }
            havePC = true;
            break;
         case AMS:
            if ( myid_ == 0 && logging_ > 0 )
            {
               cout << "AMS Preconditioner Requested" << endl;
            }
            for (int i=0; i<b_.Size(); i++)
               pcr[i] = new HypreAMS(dynamic_cast<HypreParMatrix&>(*PCOp[i].Ptr()),
                                     HCurlFESpace_);
            havePC = true;
            break;
         default:
            MFEM_ABORT("Requested preconditioner is not available.");
            break;
      }
      cout << "Building preconditioner for imaginary part" << endl;
      for (int i = 0; i<b_.Size(); i++)
      {
         if (pcr[i] && conv_ != ComplexOperator::HERMITIAN)
         {
            pci[i] = new ScaledOperator(pcr[i], -1.0);
         }
         else
         {
            pci[i] = pcr[i];
         }
      }
      if (havePC)
      {
         cout << "Building block preconditioner" << endl;
         BDP = new BlockDiagonalPreconditioner(blockTrueOffsets_);
         for (int i=0; i<b_.Size(); i++)
         {
            BDP->SetDiagonalBlock(2 * i + 0, pcr[i]);
            BDP->SetDiagonalBlock(2 * i + 1, pci[i]);
         }
         BDP->owns_blocks = 0;
      }
   }

   BlockOperator A(op_sizes);
   A.owns_blocks = false;

   for (int i=0; i<b_.Size(); i++)
   {
      cout << "op_sizes " << op_sizes[i] << " " << op_sizes[i+1] << " " <<
           A1[i]->Height() << endl;
      A.SetDiagonalBlock(i, A1[i].Ptr());
      /*
      for (int j=0; j<b_.Size(); j++)
      {
      if (am_(i,j) != NULL)
      {
        A.SetBlock(i, j, am_(i,j)->ParallelAssemble());
      }
           }
           */
   }

   switch (sol_)
   {
      case GMRES:
      {
         if ( myid_ == 0 && logging_ > 0 )
         {
            cout << "GMRES Solver Requested" << endl;
         }
         GMRESSolver gmres(HCurlFESpace_->GetComm());
         if (BDP) { gmres.SetPreconditioner(*BDP); }
         gmres.SetOperator(A);
         gmres.SetRelTol(solOpts_.relTol);
         gmres.SetMaxIter(solOpts_.maxIter);
         gmres.SetKDim(solOpts_.kDim);
         gmres.SetPrintLevel(solOpts_.printLvl);

         gmres.Mult(RHS, E);
      }
      break;
      case FGMRES:
      {
         if ( myid_ == 0 && logging_ > 0 )
         {
            cout << "FGMRES Solver Requested" << endl;
         }
         FGMRESSolver fgmres(HCurlFESpace_->GetComm());
         if (BDP) { fgmres.SetPreconditioner(*BDP); }
         fgmres.SetOperator(A);
         fgmres.SetRelTol(solOpts_.relTol);
         fgmres.SetMaxIter(solOpts_.maxIter);
         fgmres.SetKDim(solOpts_.kDim);
         fgmres.SetPrintLevel(solOpts_.printLvl);

         fgmres.Mult(RHS, E);

         // delete B1;
      }
      break;
      case MINRES:
      {
         if ( myid_ == 0 && logging_ > 0 )
         {
            cout << "MINRES Solver Requested" << endl;
         }
         MINRESSolver minres(HCurlFESpace_->GetComm());
         if (BDP) { minres.SetPreconditioner(*BDP); }
         minres.SetOperator(A);
         minres.SetRelTol(solOpts_.relTol);
         minres.SetMaxIter(solOpts_.maxIter);
         minres.SetPrintLevel(solOpts_.printLvl);

         minres.Mult(RHS, E);
      }
      break;
#ifdef MFEM_USE_SUPERLU
      case SUPERLU:
      {
         if ( myid_ == 0 && logging_ > 0 )
         {
            cout << "SuperLU Solver Requested" << endl;
         }
         ComplexHypreParMatrix * A1Z = A1[0].As<ComplexHypreParMatrix>();
         HypreParMatrix * A1C = A1Z->GetSystemMatrix();
         SuperLURowLocMatrix A_SuperLU(*A1C);
         SuperLUSolver solver(MPI_COMM_WORLD);
         solver.SetOperator(A_SuperLU);
         solver.Mult(RHS, E);
         delete A1C;
         // delete A1Z;
      }
      break;
#endif
#ifdef MFEM_USE_STRUMPACK
      case STRUMPACK:
      {
         if ( myid_ == 0 && logging_ > 0 )
         {
            cout << "STRUMPACK Solver Requested" << endl;
         }
         //A1.SetOperatorOwner(false);
         ComplexHypreParMatrix * A1Z = A1[0].As<ComplexHypreParMatrix>();
         HypreParMatrix * A1C = A1Z->GetSystemMatrix();
         STRUMPACKRowLocMatrix A_STRUMPACK(*A1C);
         STRUMPACKSolver solver(0, NULL, MPI_COMM_WORLD);
         solver.SetPrintFactorStatistics(true);
         solver.SetPrintSolveStatistics(false);
         solver.SetKrylovSolver(strumpack::KrylovSolver::DIRECT);
         solver.SetReorderingStrategy(strumpack::ReorderingStrategy::METIS);
         solver.DisableMatching();
         solver.SetOperator(A_STRUMPACK);
         solver.SetFromCommandLine();
         solver.Mult(RHS, E);
         delete A1C;
         // delete A1Z;
      }
      break;
#endif
      default:
         MFEM_ABORT("Requested solver is not available.");
         break;
   };

   tic_toc.Stop();

   e_[0]->Distribute(E);

   delete BDP;
   for (int i=0; i<pcr.Size(); i++)
   {
      if (pci[i] != pcr[i]) { delete pci[i]; }
      delete pcr[i];
   }
   if ( myid_ == 0 && logging_ > 0 )
   {
      cout << " Solver done in " << tic_toc.RealTime() << " seconds." << endl;
   }
}

double
CPDSolver::GetError(const VectorCoefficient & EReCoef,
                    const VectorCoefficient & EImCoef) const
{
   ParComplexGridFunction z(e_[0]->ParFESpace());
   z = 0.0;

   double solNorm = z.ComputeL2Error(const_cast<VectorCoefficient&>(EReCoef),
                                     const_cast<VectorCoefficient&>(EImCoef));


   double solErr = e_[0]->ComputeL2Error(const_cast<VectorCoefficient&>(EReCoef),
                                         const_cast<VectorCoefficient&>(EImCoef));

   return (solNorm > 0.0) ? solErr / solNorm : solErr;
}

void
CPDSolver::GetErrorEstimates(Vector & errors)
{
   if ( myid_ == 0 && logging_ > 0 )
   { cout << "Estimating Error ... " << flush; }

   // Space for the discontinuous (original) flux
   CurlCurlIntegrator flux_integrator(*muInvCoef_);
   RT_FECollection flux_fec(order_-1, pmesh_->SpaceDimension());
   ParFiniteElementSpace flux_fes(pmesh_, &flux_fec);

   // Space for the smoothed (conforming) flux
   double norm_p = 1;
   ND_FECollection smooth_flux_fec(order_, pmesh_->Dimension());
   ParFiniteElementSpace smooth_flux_fes(pmesh_, &smooth_flux_fec);

   L2ZZErrorEstimator(flux_integrator, e_[0]->real(),
                      smooth_flux_fes, flux_fes, errors, norm_p);

   if ( myid_ == 0 && logging_ > 0 ) { cout << "done." << endl; }
}

void
CPDSolver::RegisterVisItFields(VisItDataCollection & visit_dc)
{
   visit_dc_ = &visit_dc;

   visit_dc.RegisterField("Re(E)", &e_[0]->real());
   visit_dc.RegisterField("Im(E)", &e_[0]->imag());
   // visit_dc.RegisterField("Er", e_r_);
   // visit_dc.RegisterField("Ei", e_i_);
   // visit_dc.RegisterField("B", b_);
   // visit_dc.RegisterField("H", h_);
   if ( j_[0] )
   {
      visit_dc.RegisterField("Re(J)", &j_[0]->real());
      visit_dc.RegisterField("Im(J)", &j_[0]->imag());
   }
   if ( u_ )
   {
      visit_dc.RegisterField("U", u_);
      visit_dc.RegisterField("Re(S)", &S_->real());
      visit_dc.RegisterField("Im(S)", &S_->imag());
      // visit_dc.RegisterField("Im(u)", &u_->imag());
   }
   // if ( j_r_ ) { visit_dc.RegisterField("Jr", j_r_); }
   // if ( j_i_ ) { visit_dc.RegisterField("Ji", j_i_); }
   // if ( k_ ) { visit_dc.RegisterField("K", k_); }
   // if ( m_ ) { visit_dc.RegisterField("M", m_); }
   // if ( SurfCur_ ) { visit_dc.RegisterField("Psi", SurfCur_->GetPsi()); }
}

void
CPDSolver::WriteVisItFields(int it)
{
   if ( visit_dc_ )
   {
      if (myid_ == 0) { cout << "Writing VisIt files ..." << flush; }

      if ( j_[0] )
      {
         if (pmesh_->Dimension() == 3)
         {
            j_[0]->ProjectCoefficient(*jrCoef_, *jiCoef_);
         }
         else if (pmesh_->Dimension() == 2)
         {
            j_[0]->ProjectCoefficient(*jr2x1Coef_, *ji2x1Coef_);
            j_[1]->ProjectCoefficient(*jrZCoef_, *jiZCoef_);
         }
      }
      if ( u_ )
      {
         u_->ProjectCoefficient(uCoef_);
         S_->ProjectCoefficient(SrCoef_, SiCoef_);
      }

      HYPRE_Int prob_size = this->GetProblemSize();
      visit_dc_->SetCycle(it);
      visit_dc_->SetTime(prob_size);
      visit_dc_->Save();

      if (myid_ == 0) { cout << " done." << endl; }
   }
}

void
CPDSolver::InitializeGLVis()
{
   if ( myid_ == 0 ) { cout << "Opening GLVis sockets." << endl; }

   switch (pmesh_->Dimension())
   {
      case 2:
         socks_["Er_xy"] = new socketstream;
         socks_["Er_xy"]->precision(8);
         socks_["Er_z"] = new socketstream;
         socks_["Er_z"]->precision(8);

         socks_["Ei_xy"] = new socketstream;
         socks_["Ei_xy"]->precision(8);
         socks_["Ei_z"] = new socketstream;
         socks_["Ei_z"]->precision(8);
         break;
      case 3:
         socks_["Er"] = new socketstream;
         socks_["Er"]->precision(8);

         socks_["Ei"] = new socketstream;
         socks_["Ei"]->precision(8);
         break;
   }
   // socks_["B"] = new socketstream;
   // socks_["B"]->precision(8);

   // socks_["H"] = new socketstream;
   // socks_["H"]->precision(8);

   if ( j_[0] )
   {
      socks_["Jr"] = new socketstream;
      socks_["Jr"]->precision(8);

      socks_["Ji"] = new socketstream;
      socks_["Ji"]->precision(8);
   }

   if ( u_ )
   {
      socks_["U"] = new socketstream;
      socks_["U"]->precision(8);

      socks_["Sr"] = new socketstream;
      socks_["Sr"]->precision(8);

      socks_["Si"] = new socketstream;
      socks_["Si"]->precision(8);
   }
   /*
   if ( k_ )
   {
      socks_["K"] = new socketstream;
      socks_["K"]->precision(8);
      socks_["Psi"] = new socketstream;
      socks_["Psi"]->precision(8);
   }
   if ( m_ )
   {
      socks_["M"] = new socketstream;
      socks_["M"]->precision(8);
   }
   */
   if ( myid_ == 0 ) { cout << "GLVis sockets open." << endl; }
}

void
CPDSolver::DisplayToGLVis()
{
   if (myid_ == 0) { cout << "Sending data to GLVis ..." << flush; }

   char vishost[] = "localhost";
   int  visport   = 19916;

   int Wx = 0, Wy = 0; // window position
   int Ww = 350, Wh = 350; // window size
   int offx = Ww+10, offy = Wh+45; // window offsets

   if (kCoef_)
   {
      VectorGridFunctionCoefficient e_r(&e_[0]->real());
      VectorGridFunctionCoefficient e_i(&e_[0]->imag());
      VectorSumCoefficient erCoef(e_r, e_i, *coskx_, *sinkx_);
      VectorSumCoefficient eiCoef(e_i, e_r, *coskx_, *negsinkx_);

      e_v_[0]->ProjectCoefficient(erCoef, eiCoef);
   }
   else
   {
      for (int i=0; i<e_v_.Size(); i++)
      {
         e_v_[i] = e_[i];
      }
   }

   switch (pmesh_->Dimension())
   {
      case 2:
         VisualizeField(*socks_["Er_xy"], vishost, visport,
                        e_v_[0]->real(), "Electric Field, Re(E_xy)",
			Wx, Wy, Ww, Wh);
         Wx += offx;

         VisualizeField(*socks_["Er_z"], vishost, visport,
                        e_v_[1]->real(), "Electric Field, Re(E_z)",
			Wx, Wy, Ww, Wh);
         Wx += offx;

         VisualizeField(*socks_["Ei_xy"], vishost, visport,
                        e_v_[0]->imag(), "Electric Field, Im(E_xy)",
			Wx, Wy, Ww, Wh);
         Wx += offx;

         VisualizeField(*socks_["Ei_z"], vishost, visport,
                        e_v_[1]->imag(), "Electric Field, Im(E_z)",
			Wx, Wy, Ww, Wh);
         break;
      case 3:
         VisualizeField(*socks_["Er"], vishost, visport,
                        e_v_[0]->real(), "Electric Field, Re(E)",
			Wx, Wy, Ww, Wh);
         Wx += offx;

         VisualizeField(*socks_["Ei"], vishost, visport,
                        e_v_[0]->imag(), "Electric Field, Im(E)",
			Wx, Wy, Ww, Wh);
         break;
   }
   /*
   Wx += offx;
   VisualizeField(*socks_["B"], vishost, visport,
                  *b_, "Magnetic Flux Density (B)", Wx, Wy, Ww, Wh);
   Wx += offx;
   VisualizeField(*socks_["H"], vishost, visport,
                  *h_, "Magnetic Field (H)", Wx, Wy, Ww, Wh);
   Wx += offx;
   */
   if ( j_[0] )
   {
      Wx = 0; Wy += offy; // next line

      if (pmesh_->Dimension() == 3)
      {
         j_[0]->ProjectCoefficient(*jrCoef_, *jiCoef_);
      }
      else if (pmesh_->Dimension() == 3)
      {
         j_[0]->ProjectCoefficient(*jr2x1Coef_, *ji2x1Coef_);
         j_[1]->ProjectCoefficient(*jrZCoef_, *jiZCoef_);
      }

      if (kCoef_)
      {
         VectorGridFunctionCoefficient j_r(&j_[0]->real());
         VectorGridFunctionCoefficient j_i(&j_[0]->imag());
         VectorSumCoefficient jrCoef(j_r, j_i, *coskx_, *sinkx_);
         VectorSumCoefficient jiCoef(j_i, j_r, *coskx_, *negsinkx_);

         j_v_->ProjectCoefficient(jrCoef, jiCoef);
      }
      else
      {
         j_v_ = j_[0];
      }

      VisualizeField(*socks_["Jr"], vishost, visport,
                     j_v_->real(), "Current Density, Re(J)", Wx, Wy, Ww, Wh);
      Wx += offx;
      VisualizeField(*socks_["Ji"], vishost, visport,
                     j_v_->imag(), "Current Density, Im(J)", Wx, Wy, Ww, Wh);
   }
   Wx = 0; Wy += offy; // next line

   if ( u_ )
   {
      Wx = 0; Wy += offy; // next line

      u_->ProjectCoefficient(uCoef_);
      S_->ProjectCoefficient(SrCoef_, SiCoef_);

      VisualizeField(*socks_["U"], vishost, visport,
                     *u_, "Energy Density, U", Wx, Wy, Ww, Wh);

      Wx += offx;
      VisualizeField(*socks_["Sr"], vishost, visport,
                     S_->real(), "Poynting Vector, Re(S)", Wx, Wy, Ww, Wh);
      Wx += offx;
      VisualizeField(*socks_["Si"], vishost, visport,
                     S_->imag(), "Poynting Vector, Im(S)", Wx, Wy, Ww, Wh);
   }
   Wx = 0; Wy += offy; // next line
   /*
   if ( k_ )
   {
      VisualizeField(*socks_["K"], vishost, visport,
                     *k_, "Surface Current Density (K)", Wx, Wy, Ww, Wh);
      Wx += offx;
      VisualizeField(*socks_["Psi"], vishost, visport,
                     *SurfCur_->GetPsi(),
                     "Surface Current Potential (Psi)", Wx, Wy, Ww, Wh);
      Wx += offx;
   }
   if ( m_ )
   {
      VisualizeField(*socks_["M"], vishost, visport,
                     *m_, "Magnetization (M)", Wx, Wy, Ww, Wh);
      Wx += offx;
   }
   */
   if (myid_ == 0) { cout << " done." << endl; }
}

void
CPDSolver::DisplayAnimationToGLVis()
{
   if (myid_ == 0) { cout << "Sending animation data to GLVis ..." << flush; }

   if (kCoef_)
   {
      VectorGridFunctionCoefficient e_r(&e_[0]->real());
      VectorGridFunctionCoefficient e_i(&e_[0]->imag());
      VectorSumCoefficient erCoef(e_r, e_i, *coskx_, *sinkx_);
      VectorSumCoefficient eiCoef(e_i, e_r, *coskx_, *negsinkx_);

      e_v_[0]->ProjectCoefficient(erCoef, eiCoef);
   }
   else
   {
      for (int i=0; i<e_v_.Size(); i++)
      {
         e_v_[i] = e_[i];
      }
   }

   Vector zeroVec(3); zeroVec = 0.0;
   VectorConstantCoefficient zeroCoef(zeroVec);

   double norm_r = e_v_[0]->real().ComputeMaxError(zeroCoef);
   double norm_i = e_v_[0]->imag().ComputeMaxError(zeroCoef);

   *e_t_ = e_v_[0]->real();

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock << "parallel " << num_procs_ << " " << myid_ << "\n";
   sol_sock.precision(8);
   sol_sock << "solution\n" << *pmesh_ << *e_t_
            << "window_title 'Harmonic Solution (t = 0.0 T)'"
            << "valuerange 0.0 " << max(norm_r, norm_i) << "\n"
            << "autoscale off\n"
            << "keys cvvv\n"
            << "pause\n" << flush;
   if (myid_ == 0)
      cout << "GLVis visualization paused."
           << " Press space (in the GLVis window) to resume it.\n";
   int num_frames = 24;
   int i = 0;
   while (sol_sock)
   {
      double t = (double)(i % num_frames) / num_frames;
      ostringstream oss;
      oss << "Harmonic Solution (t = " << t << " T)";

      add( cos( 2.0 * M_PI * t), e_v_[0]->real(),
           -sin( 2.0 * M_PI * t), e_v_[0]->imag(), *e_t_);
      sol_sock << "parallel " << num_procs_ << " " << myid_ << "\n";
      sol_sock << "solution\n" << *pmesh_ << *e_t_
               << "window_title '" << oss.str() << "'" << flush;
      i++;
   }
}

} // namespace plasma

} // namespace mfem

#endif // MFEM_USE_MPI
