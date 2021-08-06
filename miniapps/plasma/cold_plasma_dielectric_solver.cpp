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
#include "cold_plasma_dielectric_coefs.hpp"

#ifdef MFEM_USE_MPI

using namespace std;

namespace mfem
{
using namespace common;

namespace plasma
{

// Used for combining scalar coefficients
double prodFunc(double a, double b) { return a * b; }

ElectricEnergyDensityCoef::ElectricEnergyDensityCoef(VectorCoefficient &Er,
                                                     VectorCoefficient &Ei,
                                                     MatrixCoefficient &epsr,
                                                     MatrixCoefficient &epsi)
   : ErCoef_(Er),
     EiCoef_(Ei),
     epsrCoef_(epsr),
     epsiCoef_(epsi),
     Er_(3),
     Ei_(3),
     Dr_(3),
     Di_(3),
     eps_r_(3),
     eps_i_(3)
{}

double ElectricEnergyDensityCoef::Eval(ElementTransformation &T,
                                       const IntegrationPoint &ip)
{
   ErCoef_.Eval(Er_, T, ip);
   EiCoef_.Eval(Ei_, T, ip);

   epsrCoef_.Eval(eps_r_, T, ip);
   epsiCoef_.Eval(eps_i_, T, ip);

   if (T.ElementNo == 1)
   {
      cout << "eps_r" << endl;
      eps_r_.Print(std::cout, 3);
      cout << "eps_i" << endl;
      eps_i_.Print(std::cout, 3);
   }

   eps_r_.Mult(Er_, Dr_);
   eps_i_.AddMult_a(-1.0, Ei_, Dr_);

   eps_i_.Mult(Er_, Di_);
   eps_r_.AddMult(Ei_, Di_);

   double u = (Er_ * Dr_) + (Ei_ * Di_);

   return 0.5 * u;
}

MagneticEnergyDensityCoef::MagneticEnergyDensityCoef(double omega,
                                                     VectorCoefficient &dEr,
                                                     VectorCoefficient &dEi,
                                                     Coefficient &muInv)
   : omega_(omega),
     dErCoef_(dEr),
     dEiCoef_(dEi),
     muInvCoef_(muInv),
     Br_(3),
     Bi_(3)
{}

double MagneticEnergyDensityCoef::Eval(ElementTransformation &T,
                                       const IntegrationPoint &ip)
{
   dErCoef_.Eval(Bi_, T, ip); Bi_ /=  omega_;
   dEiCoef_.Eval(Br_, T, ip); Br_ /= -omega_;

   double muInv = muInvCoef_.Eval(T, ip);

   double u = ((Br_ * Br_) + (Bi_ * Bi_)) * muInv;

   return 0.5 * u;
}

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
                     VectorCoefficient & BCoef,
                     MatrixCoefficient & epsReCoef,
                     MatrixCoefficient & epsImCoef,
                     MatrixCoefficient & epsAbsCoef,
                     Coefficient & muInvCoef,
                     Coefficient * etaInvCoef,
                     VectorCoefficient * kCoef,
                     Array<int> & abcs,
                     // Array<int> & dbcs,
                     Array<ComplexVectorCoefficientByAttr*> & dbcs,
                     Array<ComplexVectorCoefficientByAttr*> & nbcs,
                     Array<ComplexCoefficientByAttr*> & sbcs,
                     // void   (*e_r_bc )(const Vector&, Vector&),
                     // void   (*e_i_bc )(const Vector&, Vector&),
                     // VectorCoefficient & EReCoef,
                     // VectorCoefficient & EImCoef,
                     void (*j_r_src)(const Vector&, Vector&),
                     void (*j_i_src)(const Vector&, Vector&),
                     bool vis_u, bool pa)
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
     pa_(pa),
     omega_(omega),
     // solNorm_(-1.0),
     pmesh_(&pmesh),
     L2FESpace_(NULL),
     L2FESpace2p_(NULL),
     L2VFESpace_(NULL),
     H1FESpace_(NULL),
     HCurlFESpace_(NULL),
     HDivFESpace_(NULL),
     HDivFESpace2p_(NULL),
     a1_(NULL),
     b1_(NULL),
     m2_(NULL),
     m12EpsRe_(NULL),
     m12EpsIm_(NULL),
     m0_(NULL),
     n20ZRe_(NULL),
     n20ZIm_(NULL),
     e_(NULL),
     e_tmp_(NULL),
     d_(NULL),
     temp_(NULL),
     grad_(NULL),
     kOp_(NULL),
     phi_(NULL),
     phi_tmp_(NULL),
     rectPot_(NULL),
     j_(NULL),
     rhs_(NULL),
     e_t_(NULL),
     e_b_(NULL),
     e_v_(NULL),
     d_v_(NULL),
     phi_v_(NULL),
     j_v_(NULL),
     b_hat_(NULL),
     u_(NULL),
     uE_(NULL),
     uB_(NULL),
     S_(NULL),
     StixS_(NULL),
     StixD_(NULL),
     StixP_(NULL),
     EpsPara_(NULL),
     BCoef_(&BCoef),
     epsReCoef_(&epsReCoef),
     epsImCoef_(&epsImCoef),
     epsAbsCoef_(&epsAbsCoef),
     muInvCoef_(&muInvCoef),
     etaInvCoef_(etaInvCoef),
     kCoef_(kCoef),
     SReCoef_(NULL),
     SImCoef_(NULL),
     DReCoef_(NULL),
     DImCoef_(NULL),
     PReCoef_(NULL),
     PImCoef_(NULL),
     omegaCoef_(new ConstantCoefficient(omega_)),
     negOmegaCoef_(new ConstantCoefficient(-omega_)),
     omega2Coef_(new ConstantCoefficient(pow(omega_, 2))),
     negOmega2Coef_(new ConstantCoefficient(-pow(omega_, 2))),
     abcCoef_(NULL),
     sinkx_(NULL),
     coskx_(NULL),
     negsinkx_(NULL),
     negMuInvCoef_(NULL),
     massReCoef_(NULL),
     massImCoef_(NULL),
     posMassCoef_(NULL),
     negMuInvkxkxCoef_(NULL),
     negMuInvkCoef_(NULL),
     muInvkCoef_(NULL),
     jrCoef_(NULL),
     jiCoef_(NULL),
     rhsrCoef_(NULL),
     rhsiCoef_(NULL),
     // erCoef_(EReCoef),
     // eiCoef_(EImCoef),
     derCoef_(NULL),
     deiCoef_(NULL),
     uCoef_(omega_, erCoef_, eiCoef_, derCoef_, deiCoef_,
            *epsReCoef_, *epsImCoef_, *muInvCoef_),
     uECoef_(erCoef_, eiCoef_, *epsReCoef_, *epsImCoef_),
     uBCoef_(omega_, derCoef_, deiCoef_, *muInvCoef_),
     SrCoef_(omega_, erCoef_, eiCoef_, derCoef_, deiCoef_, *muInvCoef_),
     SiCoef_(omega_, erCoef_, eiCoef_, derCoef_, deiCoef_, *muInvCoef_),
     j_r_src_(j_r_src),
     j_i_src_(j_i_src),
     // e_r_bc_(e_r_bc),
     // e_i_bc_(e_i_bc),
     dbcs_(&dbcs),
     nbcs_(&nbcs),
     nkbcs_(NULL),
     sbcs_(&sbcs),
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

   // Define compatible parallel finite element spaces on the parallel
   // mesh. Here we use arbitrary order H1, Nedelec, and Raviart-Thomas finite
   // elements.
   H1FESpace_    = new H1_ParFESpace(pmesh_,order,pmesh_->Dimension());
   HCurlFESpace_ = new ND_ParFESpace(pmesh_,order,pmesh_->Dimension());
   HDivFESpace_  = new RT_ParFESpace(pmesh_,order,pmesh_->Dimension());
   L2FESpace_    = new L2_ParFESpace(pmesh_,order-1,pmesh_->Dimension());

   if (BCoef_)
   {
      e_b_ = new ParComplexGridFunction(L2FESpace_);
      *e_b_ = 0.0;
      b_hat_ = new ParGridFunction(HDivFESpace_);
      EpsPara_ = new ParComplexGridFunction(L2FESpace_);
   }
   if (kCoef_)
   {
      L2VFESpace_ = new L2_ParFESpace(pmesh_,order,pmesh_->Dimension(),
                                      pmesh_->SpaceDimension());
      e_t_ = new ParGridFunction(L2VFESpace_);
      e_v_ = new ParComplexGridFunction(L2VFESpace_);
      d_v_ = new ParComplexGridFunction(L2VFESpace_);
      j_v_ = new ParComplexGridFunction(L2VFESpace_);
      if (sbcs_->Size() > 0)
      {
         phi_v_ = new ParComplexGridFunction(L2FESpace_);
      }

      sinkx_ = new PhaseCoefficient(*kCoef_, &sin);
      coskx_ = new PhaseCoefficient(*kCoef_, &cos);
      negsinkx_ = new ProductCoefficient(-1.0, *sinkx_);

      negMuInvCoef_ = new ProductCoefficient(-1.0, *muInvCoef_);
      negMuInvkCoef_ = new ScalarVectorProductCoefficient(*negMuInvCoef_,
                                                          *kCoef_);
      muInvkCoef_ = new ScalarVectorProductCoefficient(*muInvCoef_,
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
         ElementTransformation *eltrans = HCurlFESpace_->GetBdrElementTransformation (i);
         cout << i << '\t' << pmesh_->GetBdrAttribute(i)
              << '\t' << be.GetGeomType()
              << '\t' << eltrans->ElementNo
              << '\t' << eltrans->Attribute
              << endl;
      }
   }

   blockTrueOffsets_.SetSize(3);
   blockTrueOffsets_[0] = 0;
   blockTrueOffsets_[1] = HCurlFESpace_->TrueVSize();
   blockTrueOffsets_[2] = HCurlFESpace_->TrueVSize();
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
   ess_bdr_ = 0;
   if ( dbcs_->Size() > 0 )
   {
      if ( dbcs_->Size() == 1 && (*dbcs_)[0]->attr[0] == -1 )
      {
         ess_bdr_ = 1;
      }
      else
      {
         for (int i=0; i<dbcs_->Size(); i++)
         {
            for (int j=0; j<(*dbcs_)[i]->attr.Size(); j++)
            {
               ess_bdr_[(*dbcs_)[i]->attr[j]-1] = 1;
            }
         }
      }
   }
   if ( sbcs_->Size() > 0 )
   {
      if ( sbcs_->Size() == 1 && (*sbcs_)[0]->attr[0] == -1 )
      {
         ess_bdr_ = 1;
      }
      else
      {
         for (int i=0; i<sbcs_->Size(); i++)
         {
            for (int j=0; j<(*sbcs_)[i]->attr.Size(); j++)
            {
               ess_bdr_[(*sbcs_)[i]->attr[j]-1] = 1;
            }
         }
      }
   }
   if ( sbcs_->Size() > 0 || dbcs_->Size() > 0)
   {
      HCurlFESpace_->GetEssentialTrueDofs(ess_bdr_, ess_bdr_tdofs_);
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
   rhsrCoef_ = new ScalarVectorProductCoefficient(-omega_, *jiCoef_);
   rhsiCoef_ = new ScalarVectorProductCoefficient(omega_, *jrCoef_);

   if (nbcs_->Size() > 0)
   {
      nkbcs_ = new Array<ComplexVectorCoefficientByAttr*>(nbcs_->Size());
      for (int i=0; i<nbcs_->Size(); i++)
      {
         (*nkbcs_)[i]->attr = (*nbcs_)[i]->attr;
         (*nkbcs_)[i]->attr_marker.SetSize(pmesh.bdr_attributes.Max());
         (*nkbcs_)[i]->attr_marker = 0;
         for (int j=0; j<(*nbcs_)[i]->attr.Size(); j++)
         {
            (*nkbcs_)[i]->attr_marker[(*nbcs_)[i]->attr[j] - 1] = 1;
         }

         (*nkbcs_)[i]->real =
            new ScalarVectorProductCoefficient(omega_, *(*nbcs_)[i]->imag);
         (*nkbcs_)[i]->imag =
            new ScalarVectorProductCoefficient(-omega_, *(*nbcs_)[i]->real);
      }
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
   a1_ = new ParSesquilinearForm(HCurlFESpace_, conv_);
   if (pa_) { a1_->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a1_->AddDomainIntegrator(new CurlCurlIntegrator(*muInvCoef_), NULL);
   a1_->AddDomainIntegrator(new VectorFEMassIntegrator(*massReCoef_),
                            new VectorFEMassIntegrator(*massImCoef_));
   if ( kCoef_ )
   {
      if (pa_)
      {
         MFEM_ABORT("kCoef_: Partial Assembly has not yet been implemented for "
                    "MixedCrossCurlIntegrator and MixedWeakCurlCrossIntegrator.");
      }
      a1_->AddDomainIntegrator(new VectorFEMassIntegrator(*negMuInvkxkxCoef_),
                               NULL);
      a1_->AddDomainIntegrator(NULL,
                               new MixedCrossCurlIntegrator(*muInvkCoef_));
      a1_->AddDomainIntegrator(NULL,
                               new MixedWeakCurlCrossIntegrator(*muInvkCoef_));
   }
   if ( abcCoef_ )
   {
      if (pa_)
      {
         MFEM_ABORT("abcCoef_: Partial Assembly has not yet been tested for "
                    "this BoundaryIntegrator.");
      }
      a1_->AddBoundaryIntegrator(NULL, new VectorFEMassIntegrator(*abcCoef_),
                                 abc_marker_);
   }
   /*
   if ( sbcReCoef_ && sbcImCoef_ )
   {
      if (pa_)
      {
         MFEM_ABORT("sbcCoef_: Partial Assembly has not yet been tested for "
                    "this BoundaryIntegrator.");
      }
      a1_->AddBoundaryIntegrator(new VectorFEMassIntegrator(*sbcReCoef_),
                                 new VectorFEMassIntegrator(*sbcImCoef_),
                                 sbc_marker_);
   }
   */
   b1_ = new ParBilinearForm(HCurlFESpace_);
   if (pa_) { b1_->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   b1_->AddDomainIntegrator(new CurlCurlIntegrator(*muInvCoef_));
   // b1_->AddDomainIntegrator(new VectorFEMassIntegrator(*epsAbsCoef_));
   b1_->AddDomainIntegrator(new VectorFEMassIntegrator(*posMassCoef_));
   //b1_->AddDomainIntegrator(new VectorFEMassIntegrator(*massImCoef_));

   m2_ = new ParBilinearForm(HDivFESpace_);
   if (pa_) { m2_->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   m2_->AddDomainIntegrator(new VectorFEMassIntegrator);

   m12EpsRe_ = new ParMixedBilinearForm(HCurlFESpace_, HDivFESpace_);
   m12EpsIm_ = new ParMixedBilinearForm(HCurlFESpace_, HDivFESpace_);
   m12EpsRe_->AddDomainIntegrator(new VectorFEMassIntegrator(*epsReCoef_));
   m12EpsIm_->AddDomainIntegrator(new VectorFEMassIntegrator(*epsImCoef_));
   if (pa_)
   {
      // TODO: PA
      //m12EpsRe_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      //m12EpsIm_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }

   if (sbcs_->Size() > 0)
   {
      // TODO: PA does not support BoundaryIntegrator yet
      m0_ = new ParBilinearForm(H1FESpace_);

      n20ZRe_ = new ParMixedBilinearForm(HDivFESpace_, H1FESpace_);
      n20ZIm_ = new ParMixedBilinearForm(HDivFESpace_, H1FESpace_);

      for (int i=0; i<sbcs_->Size(); i++)
      {
         ComplexCoefficientByAttr * sbc = (*sbcs_)[i];

         m0_->AddBoundaryIntegrator(new MassIntegrator, sbc->attr_marker);

         n20ZRe_->AddBoundaryIntegrator(new MassIntegrator(*sbc->real),
                                        sbc->attr_marker);
         n20ZIm_->AddBoundaryIntegrator(new MassIntegrator(*sbc->imag),
                                        sbc->attr_marker);
      }
   }

   // Build grid functions
   e_  = new ParComplexGridFunction(HCurlFESpace_);
   *e_ = 0.0;

   temp_ = new ParGridFunction(HCurlFESpace_);
   *temp_ = 0.0;

   d_  = new ParComplexGridFunction(HDivFESpace_);
   *d_ = 0.0;

   if (sbcs_->Size() > 0)
   {
      phi_  = new ParComplexGridFunction(H1FESpace_);
      *phi_ = 0.0;
      /*
      PlasmaProfile::Type dpt = PlasmaProfile::GRADIENT;
      Vector dpp(6);
      dpp[0] = 2e20; dpp[1] = 0; dpp[2] = 0; dpp[3] = 0; dpp[4] = 0; dpp[5] = 100; dpp[6] = 0;
      PlasmaProfile rhoCoef(dpt, dpp);
      phi_->ProjectCoefficient(rhoCoef, rhoCoef);
       */

      rectPot_ = new ParComplexGridFunction(H1FESpace_);
      *rectPot_ = 0.0;

      for (int i=0; i<sbcs_->Size(); i++)
      {
         ComplexCoefficientByAttr * sbc = (*sbcs_)[i];
         SheathImpedance * z_r = dynamic_cast<SheathImpedance*>(sbc->real);
         SheathImpedance * z_i = dynamic_cast<SheathImpedance*>(sbc->imag);

         if (z_r) { z_r->SetPotential(*phi_); }
         if (z_i) { z_i->SetPotential(*phi_); }
      }

      grad_ = new ParDiscreteGradOperator(H1FESpace_, HCurlFESpace_);

      if (kCoef_)
      {
	kOp_ = new ParDiscreteLinearOperator(H1FESpace_, HCurlFESpace_);
	kOp_->AddDomainInterpolator(new VectorScalarProductInterpolator(*kCoef_));
      }
   }
   // solNorm_ = e_->ComputeL2Error(const_cast<VectorCoefficient&>(erCoef_),
   //                               const_cast<VectorCoefficient&>(eiCoef_));

   j_ = new ParComplexGridFunction(HDivFESpace_);
   j_->ProjectCoefficient(*jrCoef_, *jiCoef_);

   rhs_ = new ParComplexLinearForm(HCurlFESpace_, conv_);
   rhs_->AddDomainIntegrator(new VectorFEDomainLFIntegrator(*rhsrCoef_),
                             new VectorFEDomainLFIntegrator(*rhsiCoef_));

   if (nkbcs_ != NULL)
   {
      for (int i=0; i<nkbcs_->Size(); i++)
      {
         rhs_->AddBoundaryIntegrator(new VectorFEBoundaryTangentLFIntegrator(*
                                                                             (*nkbcs_)[i]->real),
                                     new VectorFEBoundaryTangentLFIntegrator(*(*nkbcs_)[i]->imag),
                                     (*nkbcs_)[i]->attr_marker);

      }
   }
   rhs_->real().Vector::operator=(0.0);
   rhs_->imag().Vector::operator=(0.0);

   if (vis_u_)
   {
      if (L2FESpace2p_ == NULL)
      {
         L2FESpace2p_ = new L2_ParFESpace(pmesh_,2*order-1,pmesh_->Dimension());
      }
      u_ = new ParGridFunction(L2FESpace2p_);
      uE_ = new ParGridFunction(L2FESpace2p_);
      uB_ = new ParGridFunction(L2FESpace2p_);

      HDivFESpace2p_ = new RT_ParFESpace(pmesh_,2*order,pmesh_->Dimension());
      S_ = new ParComplexGridFunction(HDivFESpace2p_);

      erCoef_.SetGridFunction(&e_->real());
      eiCoef_.SetGridFunction(&e_->imag());

      derCoef_.SetGridFunction(&e_->real());
      deiCoef_.SetGridFunction(&e_->imag());
   }

   {
      StixCoefBase * s = dynamic_cast<StixCoefBase*>(epsReCoef_);

      if (s != NULL)
      {
         SReCoef_ = new StixSCoef(*s);
         SImCoef_ = new StixSCoef(*s);
         DReCoef_ = new StixDCoef(*s);
         DImCoef_ = new StixDCoef(*s);
         PReCoef_ = new StixPCoef(*s);
         PImCoef_ = new StixPCoef(*s);

         dynamic_cast<StixCoefBase*>(SImCoef_)->SetImaginaryPart();
         dynamic_cast<StixCoefBase*>(DImCoef_)->SetImaginaryPart();
         dynamic_cast<StixCoefBase*>(PImCoef_)->SetImaginaryPart();

         StixS_ = new ParComplexGridFunction(L2FESpace_);
         StixD_ = new ParComplexGridFunction(L2FESpace_);
         StixP_ = new ParComplexGridFunction(L2FESpace_);
      }
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
   delete muInvkCoef_;
   delete negMuInvCoef_;
   delete negsinkx_;
   delete coskx_;
   delete sinkx_;
   delete rhsrCoef_;
   delete rhsiCoef_;
   delete jrCoef_;
   delete jiCoef_;
   // delete erCoef_;
   // delete eiCoef_;
   delete SReCoef_;
   delete SImCoef_;
   delete DReCoef_;
   delete DImCoef_;
   delete PReCoef_;
   delete PImCoef_;
   delete massReCoef_;
   delete massImCoef_;
   delete posMassCoef_;
   delete abcCoef_;
   if ( ownsEtaInv_ ) { delete etaInvCoef_; }
   delete omegaCoef_;
   delete negOmegaCoef_;
   delete omega2Coef_;
   delete negOmega2Coef_;

   // delete DivFreeProj_;
   // delete SurfCur_;

   if (e_v_ != e_) { delete e_v_; }
   if (d_v_ != d_) { delete d_v_; }
   if (j_v_ != j_) { delete j_v_; }
   if (phi_v_ != phi_) {delete phi_v_;}
   delete e_b_;
   // delete e_r_;
   // delete e_i_;
   delete e_;
   delete d_;
   delete temp_;
   delete phi_;
   delete phi_tmp_;
   delete rectPot_;
   // delete b_;
   // delete h_;
   delete j_;
   delete u_;
   delete uE_;
   delete uB_;
   delete StixS_;
   delete StixD_;
   delete StixP_;
   delete EpsPara_;
   // delete j_r_;
   // delete j_i_;
   // delete j_;
   // delete k_;
   // delete m_;
   // delete bd_;
   delete rhs_;
   delete e_t_;
   // delete jd_r_;
   // delete jd_i_;
   delete grad_;
   // delete curl_;

   delete a1_;
   delete b1_;
   delete m2_;
   delete m12EpsRe_;
   delete m12EpsIm_;
   delete m0_;
   delete n20ZRe_;
   delete n20ZIm_;
   // delete curlMuInvCurl_;
   // delete hCurlMass_;
   // delete hDivHCurlMuInv_;
   // delete weakCurlMuInv_;

   delete L2FESpace_;
   delete L2FESpace2p_;
   delete L2VFESpace_;
   delete H1FESpace_;
   delete HCurlFESpace_;
   delete HDivFESpace_;
   delete HDivFESpace2p_;

   map<string,socketstream*>::iterator mit;
   for (mit=socks_.begin(); mit!=socks_.end(); mit++)
   {
      delete mit->second;
   }
}

HYPRE_Int
CPDSolver::GetProblemSize()
{
   return 2 * HCurlFESpace_->GlobalTrueVSize();
}

void
CPDSolver::PrintSizes()
{
   // HYPRE_Int size_h1 = H1FESpace_->GlobalTrueVSize();
   HYPRE_Int size_nd = HCurlFESpace_->GlobalTrueVSize();
   // HYPRE_Int size_rt = HDivFESpace_->GlobalTrueVSize();
   if (myid_ == 0)
   {
      // cout << "Number of H1      unknowns: " << size_h1 << endl;
      cout << "Number of H(Curl) unknowns: " << size_nd << endl;
      // cout << "Number of H(Div)  unknowns: " << size_rt << endl;
   }
}

void
CPDSolver::Assemble()
{
   if ( myid_ == 0 && logging_ > 0 ) { cout << "Assembling ..." << endl; }

   if ( myid_ == 0 && logging_ > 0 )
   { cout << "  Curl(1/mu Curl) - omega^2 epsilon ..." << flush; }
   tic_toc.Clear();
   tic_toc.Start();

   // a0_->Assemble();
   // a0_->Finalize();

   a1_->Assemble();
   if (!pa_) { a1_->Finalize(); }

   tic_toc.Stop();
   if ( myid_ == 0 && logging_ > 0 )
   {
      cout << " done in " << tic_toc.RealTime() << " seconds." << endl;
   }
   if ( myid_ == 0 && logging_ > 0 )
   { cout << "  Curl(1/mu Curl) + omega^2 |epsilon| ..." << flush; }
   tic_toc.Clear();
   tic_toc.Start();

   b1_->Assemble();
   if (!pa_) { b1_->Finalize(); }

   tic_toc.Stop();
   if ( myid_ == 0 && logging_ > 0 )
   {
      cout << " done in " << tic_toc.RealTime() << " seconds." << endl;
   }
   if ( myid_ == 0 && logging_ > 0 )
   { cout << "  H(Div) mass matrix ..." << flush; }
   tic_toc.Clear();
   tic_toc.Start();

   m2_->Assemble();
   if (!pa_) { m2_->Finalize(); }

   tic_toc.Stop();
   if ( myid_ == 0 && logging_ > 0 )
   {
      cout << " done in " << tic_toc.RealTime() << " seconds." << endl;
   }
   if ( myid_ == 0 && logging_ > 0 )
   { cout << "  (epsilon u, v), u in H(Curl), v in H(Div)..." << flush; }
   tic_toc.Clear();
   tic_toc.Start();

   // TODO: PA
   m12EpsRe_->Assemble();
   m12EpsRe_->Finalize();
   //if (!pa_) m12EpsRe_->Finalize();

   // TODO: PA
   m12EpsIm_->Assemble();
   m12EpsIm_->Finalize();
   //if (!pa_) m12EpsIm_->Finalize();

   if (m0_)
   {
      tic_toc.Stop();
      if ( myid_ == 0 && logging_ > 0 )
      {
         cout << " done in " << tic_toc.RealTime() << " seconds." << endl;
      }
      if ( myid_ == 0 && logging_ > 0 )
      { cout << "  H1 mass matrix ..." << flush; }
      tic_toc.Clear();
      tic_toc.Start();

      // TODO: PA
      m0_->Assemble();
      m0_->Finalize();
      //if (!pa_) m0_->Finalize();

      tic_toc.Stop();
      if ( myid_ == 0 && logging_ > 0 )
      {
         cout << " done in " << tic_toc.RealTime() << " seconds." << endl;
      }
      if ( myid_ == 0 && logging_ > 0 )
      { cout << "  (z n.u, v), u in H(Div), v in H1..." << flush; }
      tic_toc.Clear();
      tic_toc.Start();

      // n20ZRe_->Assemble();
      // n20ZRe_->Finalize();
      //if (!pa_) n20ZRe_->Finalize();

      // n20ZIm_->Assemble();
      // n20ZIm_->Finalize();
      //if (!pa_) n20ZIm_->Finalize();
   }

   tic_toc.Stop();
   if ( myid_ == 0 && logging_ > 0 )
   {
      cout << " done in " << tic_toc.RealTime() << " seconds." << endl;
   }
   if ( myid_ == 0 && logging_ > 0 )
   { cout << "  Source term..." << flush; }
   tic_toc.Clear();
   tic_toc.Start();

   rhs_->Assemble();

   if ( grad_ )
   {
      grad_->Assemble();
      grad_->Finalize();
   }
   if ( kOp_ )
   {
      kOp_->Assemble();
      kOp_->Finalize();
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
   if (L2FESpace2p_) { L2FESpace2p_->Update(false); }
   if (L2VFESpace_) { L2VFESpace_->Update(); }
   H1FESpace_->Update();
   HCurlFESpace_->Update();
   HDivFESpace_->Update();
   if (HDivFESpace2p_) { HDivFESpace2p_->Update(false); }

   if ( sbcs_->Size() > 0 || dbcs_->Size() > 0)
   {
      HCurlFESpace_->GetEssentialTrueDofs(ess_bdr_, ess_bdr_tdofs_);
   }

   blockTrueOffsets_[0] = 0;
   blockTrueOffsets_[1] = HCurlFESpace_->TrueVSize();
   blockTrueOffsets_[2] = HCurlFESpace_->TrueVSize();
   blockTrueOffsets_.PartialSum();

   // Inform the grid functions that the space has changed.
   e_->Update();
   d_->Update();
   temp_->Update();
   if (u_) { u_->Update(); }
   if (uE_) { uE_->Update(); }
   if (uB_) { uB_->Update(); }
   if (S_) { S_->Update(); }
   if (e_t_) { e_t_->Update(); }
   if (e_b_) { e_b_->Update(); }
   if (e_v_) { e_v_->Update(); }
   if (d_v_) { d_v_->Update(); }
   if (j_v_) { j_v_->Update(); }
   if (phi_) {phi_->Update(); }
   if (phi_v_) {phi_v_->Update(); }
   if (rectPot_) { rectPot_ ->Update();}
   // e_r_->Update();
   // e_i_->Update();
   // h_->Update();
   // b_->Update();
   // bd_->Update();
   rhs_->Update();
   // jd_i_->Update();
   // if ( jr_ ) { jr_->Update(); }
   if ( j_  ) {  j_->Update(); }
   // if ( j_r_  ) {  j_r_->Update(); }
   // if ( j_i_  ) {  j_i_->Update(); }
   // if ( k_  ) {  k_->Update(); }
   // if ( m_  ) {  m_->Update(); }

   // Inform the bilinear forms that the space has changed.
   // a0_->Update();
   a1_->Update();
   b1_->Update();
   m2_->Update();
   m12EpsRe_->Update();
   m12EpsIm_->Update();
   if (m0_)
   {
      m0_->Update();
      n20ZRe_->Update();
      n20ZIm_->Update();
   }
   // curlMuInvCurl_->Update();
   // hCurlMass_->Update();
   // hDivHCurlMuInv_->Update();
   // if ( weakCurlMuInv_ ) { weakCurlMuInv_->Update(); }

   // Inform the other objects that the space has changed.
   // curl_->Update();
   if ( grad_ ) { grad_->Update(); }
   if (kOp_) { kOp_->Update(); }
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

   double E_err = 1.0;

   if (dbcs_->Size() > 0)
   {
      Array<int> attr_marker(pmesh_->bdr_attributes.Max());
      for (int i = 0; i<dbcs_->Size(); i++)
      {
         attr_marker = 0;
         for (int j=0; j<(*dbcs_)[i]->attr.Size(); j++)
         {
            attr_marker[(*dbcs_)[i]->attr[j] - 1] = 1;
         }
         /*
              e_->ProjectBdrCoefficientTangent(*(*dbcs_)[i].real,
                                               *(*dbcs_)[i].imag,
                                               attr_marker);
         */
         e_->ProjectCoefficient(*(*dbcs_)[i]->real,
                                *(*dbcs_)[i]->imag);
      }
   }

   if (e_tmp_ == NULL)
   {
      e_tmp_ = new ParComplexGridFunction(HCurlFESpace_);
   }
   e_tmp_->real() = e_->real();
   e_tmp_->imag() = e_->imag();

   VectorGridFunctionCoefficient e_tmp_r_(&e_tmp_->real());
   VectorGridFunctionCoefficient e_tmp_i_(&e_tmp_->imag());

   Vector zeroVec(3); zeroVec = 0.0;
   VectorConstantCoefficient zeroVecCoef(zeroVec);

   int E_iter = 1;

   while ( E_err > 1e-4 && E_iter <= 10 )
   {
      //if (E_iter > 3){break;}
      OperatorHandle A1;
      Vector E, RHS;
      // cout << "Norm of jd (pre-fls): " << jd_->Norml2() << endl;
      if ( grad_ && phi_ != NULL)
      {
         grad_->Mult(phi_->real(), e_->real());
         grad_->Mult(phi_->imag(), e_->imag());
      }
      if ( kOp_ && phi_ != NULL)
      {
	kOp_->AddMult(phi_->imag(), e_->real(), -1.0);
	kOp_->AddMult(phi_->real(), e_->imag(),  1.0);
      }
      if (dbcs_->Size() > 0)
      {
         Array<int> attr_marker(pmesh_->bdr_attributes.Max());
         Array<int> ess_bdr_vdofs;
         for (int i = 0; i<dbcs_->Size(); i++)
         {
            attr_marker = 0;
            for (int j=0; j<(*dbcs_)[i]->attr.Size(); j++)
            {
               attr_marker[(*dbcs_)[i]->attr[j] - 1] = 1;
            }
            // Determine marker array in vDoF space
            HCurlFESpace_->GetEssentialVDofs(attr_marker, ess_bdr_vdofs);

            temp_->ProjectCoefficient(*(*dbcs_)[i]->real);
            for (int j=0; j<ess_bdr_vdofs.Size(); j++)
            {
               if (ess_bdr_vdofs[j]) { e_->real()[j] = (*temp_)[j]; }
            }

            temp_->ProjectCoefficient(*(*dbcs_)[i]->imag);
            for (int j=0; j<ess_bdr_vdofs.Size(); j++)
            {
               if (ess_bdr_vdofs[j]) { e_->imag()[j] = (*temp_)[j]; }
            }

            /*
                  e_->ProjectBdrCoefficientTangent(*(*dbcs_)[i].real,
                                                   *(*dbcs_)[i].imag,
                                                   attr_marker);

             e_->ProjectCoefficient(*(*dbcs_)[i].real,
                                    *(*dbcs_)[i].imag);
            */

         }
      }

      if (e_ == e_tmp_ ) { break;} // L2 norm errors, these are only the boundary values at this point

      a1_->FormLinearSystem(ess_bdr_tdofs_, *e_, *rhs_, A1, E, RHS);

      // cout << "Norm of jd (post-fls): " << jd_->Norml2() << endl;
      // cout << "Norm of RHS: " << RHS.Norml2() << endl;

      tic_toc.Clear();
      tic_toc.Start();

      Operator * pcr = NULL;
      Operator * pci = NULL;
      BlockDiagonalPreconditioner * BDP = NULL;

      if (pa_)
      {
         switch (prec_)
         {
            case INVALID_PC:
               if ( myid_ == 0 && logging_ > 0 )
               {
                  cout << "No Preconditioner Requested (PA)" << endl;
               }
               break;
            case DIAG_SCALE:
               if ( myid_ == 0 && logging_ > 0 )
               {
                  cout << "Diagonal Scaling Preconditioner Requested (PA)" << endl;
               }
               pcr = new OperatorJacobiSmoother(*b1_, ess_bdr_tdofs_);
               break;
            default:
               MFEM_ABORT("Requested preconditioner is not available with PA.");
               break;
         }
      }
      else if (sol_ == GMRES || sol_ == FGMRES || sol_ == MINRES)
      {
         OperatorHandle PCOp;
         b1_->FormSystemMatrix(ess_bdr_tdofs_, PCOp);
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
               pcr = new HypreDiagScale(dynamic_cast<HypreParMatrix&>(*PCOp.Ptr()));
               break;
            case PARASAILS:
               if ( myid_ == 0 && logging_ > 0 )
               {
                  cout << "ParaSails Preconditioner Requested" << endl;
               }
               pcr = new HypreParaSails(dynamic_cast<HypreParMatrix&>(*PCOp.Ptr()));
               dynamic_cast<HypreParaSails*>(pcr)->SetSymmetry(1);
               break;
            case EUCLID:
               if ( myid_ == 0 && logging_ > 0 )
               {
                  cout << "Euclid Preconditioner Requested" << endl;
               }
               pcr = new HypreEuclid(dynamic_cast<HypreParMatrix&>(*PCOp.Ptr()));
               if (solOpts_.euLvl != 1)
               {
                  HypreSolver * pc = dynamic_cast<HypreSolver*>(pcr);
                  HYPRE_EuclidSetLevel(*pc, solOpts_.euLvl);
               }
               break;
            case AMS:
               if ( myid_ == 0 && logging_ > 0 )
               {
                  cout << "AMS Preconditioner Requested" << endl;
               }
               pcr = new HypreAMS(dynamic_cast<HypreParMatrix&>(*PCOp.Ptr()),
                                  HCurlFESpace_);
               break;
            default:
               MFEM_ABORT("Requested preconditioner is not available.");
               break;
         }
      }

      if (pcr && conv_ != ComplexOperator::HERMITIAN)
      {
         pci = new ScaledOperator(pcr, -1.0);
      }
      else
      {
         pci = pcr;
      }
      if (pcr)
      {
         BDP = new BlockDiagonalPreconditioner(blockTrueOffsets_);
         BDP->SetDiagonalBlock(0, pcr);
         BDP->SetDiagonalBlock(1, pci);
         BDP->owns_blocks = 0;
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
            gmres.SetOperator(*A1.Ptr());
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
            fgmres.SetOperator(*A1.Ptr());
            fgmres.SetRelTol(solOpts_.relTol);
            fgmres.SetMaxIter(solOpts_.maxIter);
            fgmres.SetKDim(solOpts_.kDim);
            fgmres.SetPrintLevel(solOpts_.printLvl);

            fgmres.Mult(RHS, E);
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
            minres.SetOperator(*A1.Ptr());
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
            ComplexHypreParMatrix * A1Z = A1.As<ComplexHypreParMatrix>();
            HypreParMatrix * A1C = A1Z->GetSystemMatrix();
            SuperLURowLocMatrix A_SuperLU(*A1C);
            SuperLUSolver solver(MPI_COMM_WORLD);
            solver.SetOperator(A_SuperLU);
            solver.Mult(RHS, E);
            delete A1C;
            //delete A1Z;
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
            ComplexHypreParMatrix * A1Z = A1.As<ComplexHypreParMatrix>();
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

      a1_->RecoverFEMSolution(E, *rhs_, *e_);

      // Update D = epsilon E
      {
         OperatorPtr M2;
         Vector D, RHS2;

         ParComplexLinearForm rhs(HDivFESpace_, conv_);
         ParComplexLinearForm tmp(HDivFESpace_, conv_);

         m12EpsRe_->Mult(e_->real(), rhs.real());
         m12EpsIm_->Mult(e_->imag(), tmp.real());

         m12EpsRe_->Mult(e_->imag(), rhs.imag());
         m12EpsIm_->Mult(e_->real(), tmp.imag());

         rhs.real() -= tmp.real();
         rhs.imag() += tmp.imag();


         if (conv_ == ComplexOperator::Convention::BLOCK_SYMMETRIC)
         {
            rhs.imag() *= -1.0;
         }
         rhs.SyncAlias();
         tmp.SyncAlias();

         Array<int> ess_tdof;
         m2_->FormSystemMatrix(ess_tdof, M2);

         D.SetSize(HDivFESpace_->TrueVSize()); D = 0.0;
         RHS2.SetSize(HDivFESpace_->TrueVSize());

         Operator *diag = NULL;
         Operator *pcg = NULL;
         if (pa_)
         {
            diag = new OperatorJacobiSmoother(*m2_, ess_tdof);
            CGSolver *cg = new CGSolver(MPI_COMM_WORLD);
            cg->SetOperator(*M2);
            cg->SetPreconditioner(static_cast<OperatorJacobiSmoother&>(*diag));
            cg->SetRelTol(1e-12);
            cg->SetMaxIter(1000);
            pcg = cg;
         }
         else
         {
            diag = new HypreDiagScale(*M2.As<HypreParMatrix>());
            HyprePCG *cg = new HyprePCG(*M2.As<HypreParMatrix>());
            cg->SetPreconditioner(static_cast<HypreDiagScale&>(*diag));
            cg->SetTol(1e-12);
            cg->SetMaxIter(1000);
            pcg = cg;
         }
         rhs.real().ParallelAssemble(RHS2);
         pcg->Mult(RHS2, D);
         d_->real().Distribute(D);
         rhs.imag().ParallelAssemble(RHS2);
         pcg->Mult(RHS2, D);
         d_->imag().Distribute(D);

         if (conv_ == ComplexOperator::Convention::BLOCK_SYMMETRIC)
         {
            d_->imag() *= -1.0;
         }

         delete diag;
         delete pcg;

      }

      // Update phi = - i w z n.D on the boundary
      if (sbcs_->Size() > 0)
      {
         if (phi_tmp_ == NULL)
         {
            phi_tmp_ = new ParComplexGridFunction(H1FESpace_);
         }
         phi_tmp_->real() = phi_->real();
         phi_tmp_->imag() = phi_->imag();

         GridFunctionCoefficient phi_tmp_r_(&phi_tmp_->real());
         GridFunctionCoefficient phi_tmp_i_(&phi_tmp_->imag());

         ConstantCoefficient zeroScalarCoef(0.0);

         HypreParMatrix M0;
         Vector Phi, RHS0;

         Array<int> sbc_marker(pmesh_->bdr_attributes.Max());
         sbc_marker = 0;
         for (int i=0; i<sbcs_->Size(); i++)
         {
            ComplexCoefficientByAttr * sbc = (*sbcs_)[i];
            for (int j=0; j<sbc->attr_marker.Size(); j++)
            {
               sbc_marker[j] |= sbc->attr_marker[j];
            }
         }

         Array<int> sbc_tdof;
         H1FESpace_->GetEssentialTrueDofs(sbc_marker, sbc_tdof);

         Array<int> sbc_tdof_marker;
         H1FESpace_->ListToMarker(sbc_tdof, H1FESpace_->GetTrueVSize(),
                                  sbc_tdof_marker, 1);

         // Invert marker
         for (int i=0; i<sbc_tdof_marker.Size(); i++)
         {
            sbc_tdof_marker[i] = 1 - sbc_tdof_marker[i];
         }

         Array<int> ess_tdof;
         H1FESpace_->MarkerToList(sbc_tdof_marker, ess_tdof);

         m0_->FormSystemMatrix(ess_tdof, M0);

         Phi.SetSize(H1FESpace_->TrueVSize()); Phi = 0.0;
         RHS0.SetSize(H1FESpace_->TrueVSize());

         HypreDiagScale diag(M0);

         HyprePCG pcg(M0);
         pcg.SetPreconditioner(diag);
         pcg.SetTol(1e-12);
         pcg.SetMaxIter(1000);

         ParComplexLinearForm rhs(H1FESpace_, conv_);
         ParComplexLinearForm tmp(H1FESpace_, conv_);

         double Phi_err  = 1.0;
         int    Phi_iter = 1;
         while (Phi_err > 1e-4 && Phi_iter <= 1)
         {
            n20ZRe_->Update();
            n20ZIm_->Update();

            n20ZRe_->Assemble();
            n20ZRe_->Finalize();

            n20ZIm_->Assemble();
            n20ZIm_->Finalize();

            n20ZRe_->Mult(d_->imag(), rhs.real());
            n20ZIm_->Mult(d_->real(), tmp.real());

            n20ZRe_->Mult(d_->real(), tmp.imag());
            n20ZIm_->Mult(d_->imag(), rhs.imag());

            rhs.real() += tmp.real();
            rhs.imag() -= tmp.imag();

            rhs.real() *= omega_;
            rhs.imag() *= omega_;

            if (conv_ == ComplexOperator::Convention::BLOCK_SYMMETRIC)
            {
               rhs.imag() *= -1.0;
            }

            rhs.real().ParallelAssemble(RHS0);

            pcg.Mult(RHS0, Phi);

            phi_->real().Distribute(Phi);

            rhs.imag().ParallelAssemble(RHS0);

            pcg.Mult(RHS0, Phi);

            phi_->imag().Distribute(Phi);

	    if (conv_ == ComplexOperator::Convention::BLOCK_SYMMETRIC)
            {
               phi_->imag() *= -1.0;
            }

            // have to have some error tolerance ...
            double PhisolNorm = phi_tmp_->ComputeL2Error(zeroScalarCoef, zeroScalarCoef);
            double PhisolErr = phi_->ComputeL2Error(phi_tmp_r_, phi_tmp_i_);

            Phi_err = PhisolErr;
            if ( PhisolNorm != 0 ) { Phi_err = PhisolErr / PhisolNorm; }

            // if (PhisolNorm == 0 && Phi_err < 1e-3) { break; }

            if (myid_ == 0)
            {
               cout << "Phi pass: " << Phi_iter << " Error: " << Phi_err << '\n';
            }

            phi_tmp_->real() = phi_->real();
            phi_tmp_->imag() = phi_->imag();

            Phi_iter++;
         }
      }

      delete BDP;
      if (pci != pcr) { delete pci; }
      delete pcr;

      if ( myid_ == 0 && logging_ > 0 )
      {
         cout << " Solver done in " << tic_toc.RealTime() << " seconds." << endl;
      }
      double EsolNorm = e_tmp_->ComputeL2Error(zeroVecCoef, zeroVecCoef);
      double EsolErr = e_->ComputeL2Error(e_tmp_r_, e_tmp_i_);
      double EsolNorm_e = e_->ComputeL2Error(zeroVecCoef, zeroVecCoef);

      E_err = EsolErr;
      if ( EsolNorm != 0 ) { E_err = EsolErr / EsolNorm; }

      if (myid_ == 0)
      {
         cout << "EField pass: " << E_iter << " Error: " << E_err << '\n';
      }

      e_tmp_->real() = e_->real();
      e_tmp_->imag() = e_->imag();

      E_iter++;
   }
   if (myid_ == 0)
   {
      cout << " Outer E field calculation done in " << E_iter
           << " iteration(s)." << endl;
   }
}

double
CPDSolver::GetError(const VectorCoefficient & EReCoef,
                    const VectorCoefficient & EImCoef) const
{
   ParComplexGridFunction z(e_->ParFESpace());
   z = 0.0;

   double solNorm = z.ComputeL2Error(const_cast<VectorCoefficient&>(EReCoef),
                                     const_cast<VectorCoefficient&>(EImCoef));


   double solErr = e_->ComputeL2Error(const_cast<VectorCoefficient&>(EReCoef),
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

   L2ZZErrorEstimator(flux_integrator, e_->real(),
                      smooth_flux_fes, flux_fes, errors, norm_p);

   if ( myid_ == 0 && logging_ > 0 ) { cout << "done." << endl; }
}

void
CPDSolver::RegisterVisItFields(VisItDataCollection & visit_dc)
{
   visit_dc_ = &visit_dc;

   visit_dc.RegisterField("Re_E", &e_->real());
   visit_dc.RegisterField("Im_E", &e_->imag());

   visit_dc.RegisterField("Re_D", &d_->real());
   visit_dc.RegisterField("Im_D", &d_->imag());

   if (sbcs_->Size() > 0)
   {
      visit_dc.RegisterField("Re_Phi", &phi_->real());
      visit_dc.RegisterField("Im_Phi", &phi_->imag());
   }

   if ( rectPot_ )
   {
      visit_dc.RegisterField("Rec_Re_Phi", &rectPot_->real());
      visit_dc.RegisterField("Rec_Im_Phi", &rectPot_->imag());
   }

   if ( BCoef_)
   {
      visit_dc.RegisterField("B_hat", b_hat_);
      visit_dc.RegisterField("Re_EB", &e_b_->real());
      visit_dc.RegisterField("Im_EB", &e_b_->imag());
      visit_dc.RegisterField("Re_EpsPara", &EpsPara_->real());
      visit_dc.RegisterField("Im_EpsPara", &EpsPara_->imag());
   }

   // visit_dc.RegisterField("Er", e_r_);
   // visit_dc.RegisterField("Ei", e_i_);
   // visit_dc.RegisterField("B", b_);
   // visit_dc.RegisterField("H", h_);
   if ( j_ )
   {
      visit_dc.RegisterField("Re_J", &j_->real());
      visit_dc.RegisterField("Im_J", &j_->imag());
   }
   if ( u_ )
   {
      visit_dc.RegisterField("U", u_);
      visit_dc.RegisterField("U_E", uE_);
      visit_dc.RegisterField("U_B", uB_);
      visit_dc.RegisterField("Re_S", &S_->real());
      visit_dc.RegisterField("Im_S", &S_->imag());
      // visit_dc.RegisterField("Im(u)", &u_->imag());
   }
   if ( StixS_ )
   {
      visit_dc.RegisterField("Re_StixS", &StixS_->real());
      visit_dc.RegisterField("Im_StixS", &StixS_->imag());
      visit_dc.RegisterField("Re_StixD", &StixD_->real());
      visit_dc.RegisterField("Im_StixD", &StixD_->imag());
      visit_dc.RegisterField("Re_StixP", &StixP_->real());
      visit_dc.RegisterField("Im_StixP", &StixP_->imag());
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

      if ( j_ )
      {
         j_->ProjectCoefficient(*jrCoef_, *jiCoef_);
      }
      if ( u_ )
      {
         u_->ProjectCoefficient(uCoef_);
         uE_->ProjectCoefficient(uECoef_);
         uB_->ProjectCoefficient(uBCoef_);
         S_->ProjectCoefficient(SrCoef_, SiCoef_);
      }
      if ( StixS_ )
      {
         StixS_->ProjectCoefficient(*SReCoef_, *SImCoef_);
         StixD_->ProjectCoefficient(*DReCoef_, *DImCoef_);
         StixP_->ProjectCoefficient(*PReCoef_, *PImCoef_);
      }

      if ( rectPot_ )
      {

         ComplexCoefficientByAttr * sbc = (*sbcs_)[0];
         SheathBase * sb = dynamic_cast<SheathBase*>(sbc->real);


         if (sb != NULL)
         {
            RectifiedSheathPotential rectPotCoefR(*sb, true);
            RectifiedSheathPotential rectPotCoefI(*sb, false);

            rectPot_->ProjectCoefficient(rectPotCoefR, rectPotCoefI);
         }

      }

      if ( BCoef_)
      {
         b_hat_->ProjectCoefficient(*BCoef_);

         VectorGridFunctionCoefficient e_r(&e_->real());
         VectorGridFunctionCoefficient e_i(&e_->imag());
         InnerProductCoefficient ebrCoef(e_r, *BCoef_);
         InnerProductCoefficient ebiCoef(e_i, *BCoef_);

         e_b_->ProjectCoefficient(ebrCoef, ebiCoef);

         MatrixVectorProductCoefficient ReEpsB(*epsReCoef_, *BCoef_);
         MatrixVectorProductCoefficient ImEpsB(*epsImCoef_, *BCoef_);
         InnerProductCoefficient ReEpsParaCoef(*BCoef_, ReEpsB);
         InnerProductCoefficient ImEpsParaCoef(*BCoef_, ImEpsB);

         EpsPara_->ProjectCoefficient(ReEpsParaCoef, ImEpsParaCoef);
         *EpsPara_ *= 1.0 / epsilon0_;
      }

      //ComplexCoefficientByAttr & sbc = (*sbcs_)[0];
      //SheathBase * sb = dynamic_cast<SheathBase*>(sbc.real);

      /*
      if (sb != NULL)
      {
        RectifiedSheathPotential rectPotCoefR(*sb, true);
        RectifiedSheathPotential rectPotCoefI(*sb, false);

        rectPot_->ProjectCoefficient(rectPotCoefR, rectPotCoefI);
       }
       */



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

   socks_["Er"] = new socketstream;
   socks_["Er"]->precision(8);

   socks_["Ei"] = new socketstream;
   socks_["Ei"]->precision(8);

   socks_["Dr"] = new socketstream;
   socks_["Dr"]->precision(8);

   socks_["Di"] = new socketstream;
   socks_["Di"]->precision(8);

   if (sbcs_->Size() > 0)
   {
      socks_["Phir"] = new socketstream;
      socks_["Phir"]->precision(8);

      socks_["Phii"] = new socketstream;
      socks_["Phii"]->precision(8);

      socks_["RecPhir"] = new socketstream;
      socks_["RecPhir"]->precision(8);

      socks_["RecPhii"] = new socketstream;
      socks_["RecPhii"]->precision(8);
   }

   if (BCoef_)
   {
      socks_["EBr"] = new socketstream;
      socks_["EBr"]->precision(8);

      socks_["EBi"] = new socketstream;
      socks_["EBi"]->precision(8);
   }

   // socks_["B"] = new socketstream;
   // socks_["B"]->precision(8);

   // socks_["H"] = new socketstream;
   // socks_["H"]->precision(8);

   if ( j_ )
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

      socks_["U_E"] = new socketstream;
      socks_["U_E"]->precision(8);

      socks_["U_B"] = new socketstream;
      socks_["U_B"]->precision(8);

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
      VectorGridFunctionCoefficient e_r(&e_->real());
      VectorGridFunctionCoefficient e_i(&e_->imag());
      VectorSumCoefficient erCoef(e_r, e_i, *coskx_, *negsinkx_);
      VectorSumCoefficient eiCoef(e_i, e_r, *coskx_, *sinkx_);

      e_v_->ProjectCoefficient(erCoef, eiCoef);

      VectorGridFunctionCoefficient d_r(&d_->real());
      VectorGridFunctionCoefficient d_i(&d_->imag());
      VectorSumCoefficient drCoef(d_r, d_i, *coskx_, *negsinkx_);
      VectorSumCoefficient diCoef(d_i, d_r, *coskx_, *sinkx_);

      d_v_->ProjectCoefficient(drCoef, diCoef);

      if (sbcs_->Size() > 0)
      {
         GridFunctionCoefficient phi_r(&phi_->real());
         GridFunctionCoefficient phi_i(&phi_->imag());
         ProductCoefficient cosphi_r(phi_r, *coskx_);
         ProductCoefficient cosphi_i(phi_i, *coskx_);
         ProductCoefficient sinphi_i(phi_i, *sinkx_);
         ProductCoefficient negsinphi_r(phi_r, *negsinkx_);
         SumCoefficient phirCoef(cosphi_r, sinphi_i);
         SumCoefficient phiiCoef(cosphi_i, negsinphi_r);

         phi_v_->ProjectCoefficient(phirCoef, phiiCoef);
      }
   }
   else
   {
      e_v_ = e_;
      d_v_ = d_;
      phi_v_ = phi_;
   }

   VisualizeField(*socks_["Er"], vishost, visport,
                  e_v_->real(), "Electric Field, Re(E)", Wx, Wy, Ww, Wh);
   Wx += offx;

   VisualizeField(*socks_["Ei"], vishost, visport,
                  e_v_->imag(), "Electric Field, Im(E)", Wx, Wy, Ww, Wh);

   /*
   Wx += offx;
   VisualizeField(*socks_["Dr"], vishost, visport,
                 d_v_->real(), "Electric Flux, Re(D)", Wx, Wy, Ww, Wh);
   Wx += offx;

   VisualizeField(*socks_["Di"], vishost, visport,
                 d_v_->imag(), "Electric Flux, Im(D)", Wx, Wy, Ww, Wh);
    */

   if (sbcs_->Size() > 0)
   {

      Wx += offx;
      VisualizeField(*socks_["Phir"], vishost, visport,
                     phi_v_->real(), "Sheath Potential, Re(Phi)", Wx, Wy, Ww, Wh);
      Wx += offx;

      VisualizeField(*socks_["Phii"], vishost, visport,
                     phi_v_->imag(), "Sheath Potential, Im(Phi)", Wx, Wy, Ww, Wh);

      /*

      Wx += offx;
      VisualizeField(*socks_["RecPhir"], vishost, visport,
                    rectPot_->real(), "Rectified Potential, Re(RecPhi)", Wx, Wy, Ww, Wh);
      Wx += offx;

      VisualizeField(*socks_["RecPhii"], vishost, visport,
                    rectPot_->imag(), "Rectified Potential, Im(RecPhi)", Wx, Wy, Ww, Wh);
       */

   }

   if (BCoef_)
   {
      /*
      VectorGridFunctionCoefficient e_r(&e_v_->real());
      VectorGridFunctionCoefficient e_i(&e_v_->imag());
      InnerProductCoefficient ebrCoef(e_r, *BCoef_);
      InnerProductCoefficient ebiCoef(e_i, *BCoef_);

      e_b_->ProjectCoefficient(ebrCoef, ebiCoef);
       */


      VisualizeField(*socks_["EBr"], vishost, visport,
                     e_b_->real(), "Parallel Electric Field, Re(E.B)",
                     Wx, Wy, Ww, Wh);
      Wx += offx;

      VisualizeField(*socks_["EBi"], vishost, visport,
                     e_b_->imag(), "Parallel Electric Field, Im(E.B)",
                     Wx, Wy, Ww, Wh);
      Wx += offx;


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
   if ( j_ )
   {
      Wx = 0; Wy += offy; // next line

      j_->ProjectCoefficient(*jrCoef_, *jiCoef_);

      if (kCoef_)
      {
         VectorGridFunctionCoefficient j_r(&j_->real());
         VectorGridFunctionCoefficient j_i(&j_->imag());
         VectorSumCoefficient jrCoef(j_r, j_i, *coskx_, *negsinkx_);
         VectorSumCoefficient jiCoef(j_i, j_r, *coskx_, *sinkx_);

         j_v_->ProjectCoefficient(jrCoef, jiCoef);
      }
      else
      {
         j_v_ = j_;
      }

      /*
      VisualizeField(*socks_["Jr"], vishost, visport,
                    j_v_->real(), "Current Density, Re(J)", Wx, Wy, Ww, Wh);
      Wx += offx;
      VisualizeField(*socks_["Ji"], vishost, visport,
                    j_v_->imag(), "Current Density, Im(J)", Wx, Wy, Ww, Wh);
       */
   }
   Wx = 0; Wy += offy; // next line

   if ( u_ )
   {
      Wx = 0; Wy += offy; // next line

      u_->ProjectCoefficient(uCoef_);
      uE_->ProjectCoefficient(uECoef_);
      uB_->ProjectCoefficient(uBCoef_);
      S_->ProjectCoefficient(SrCoef_, SiCoef_);

      VisualizeField(*socks_["U"], vishost, visport,
                     *u_, "Energy Density, U", Wx, Wy, Ww, Wh);

      Wx += offx;
      VisualizeField(*socks_["U_E"], vishost, visport,
                     *uE_, "Energy Density, U_E", Wx, Wy, Ww, Wh);

      Wx += offx;
      VisualizeField(*socks_["U_B"], vishost, visport,
                     *uB_, "Energy Density, U_B", Wx, Wy, Ww, Wh);

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
      VectorGridFunctionCoefficient e_r(&e_->real());
      VectorGridFunctionCoefficient e_i(&e_->imag());
      VectorSumCoefficient erCoef(e_r, e_i, *coskx_, *negsinkx_);
      VectorSumCoefficient eiCoef(e_i, e_r, *coskx_, *sinkx_);

      e_v_->ProjectCoefficient(erCoef, eiCoef);
   }
   else
   {
      e_v_ = e_;
   }

   Vector zeroVec(3); zeroVec = 0.0;
   VectorConstantCoefficient zeroCoef(zeroVec);

   double norm_r = e_v_->real().ComputeMaxError(zeroCoef);
   double norm_i = e_v_->imag().ComputeMaxError(zeroCoef);

   *e_t_ = e_v_->real();

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

      add( cos( 2.0 * M_PI * t), e_v_->real(),
           -sin( 2.0 * M_PI * t), e_v_->imag(), *e_t_);
      sol_sock << "parallel " << num_procs_ << " " << myid_ << "\n";
      sol_sock << "solution\n" << *pmesh_ << *e_t_
               << "window_title '" << oss.str() << "'" << flush;
      i++;
   }
}

} // namespace plasma

} // namespace mfem

#endif // MFEM_USE_MPI
