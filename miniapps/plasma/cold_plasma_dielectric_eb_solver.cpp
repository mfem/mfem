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

#include "cold_plasma_dielectric_eb_solver.hpp"
#include "cold_plasma_dielectric_coefs.hpp"
#include "../common/mesh_extras.hpp"

#ifdef MFEM_USE_MPI

using namespace std;

namespace mfem
{
using namespace common;

namespace plasma
{

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

CPDSolverEB::CPDSolverEB(ParMesh & pmesh, int order, double omega,
                         CPDSolverEB::SolverType sol, SolverOptions & sOpts,
                         CPDSolverEB::PrecondType prec,
                         ComplexOperator::Convention conv,
                         VectorCoefficient & BCoef,
                         MatrixCoefficient & epsReCoef,
                         MatrixCoefficient & epsImCoef,
                         MatrixCoefficient & epsAbsCoef,
                         Coefficient & muInvCoef,
                         Coefficient * etaInvCoef,
                         VectorCoefficient * kReCoef,
                         VectorCoefficient * kImCoef,
                         Array<int> & abcs,
                         StixBCs & stixBCs,
                         void (*j_r_src)(const Vector&, Vector&),
                         void (*j_i_src)(const Vector&, Vector&),
                         bool vis_u, bool cyl, bool pa)
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
     pmesh_(&pmesh),
     L2FESpace_(NULL),
     L2FESpace2p_(NULL),
     L2VFESpace_(NULL),
     L2FESpace3D_(NULL),
     L2VFESpace3D_(NULL),
     HCurlFESpace_(NULL),
     HDivFESpace_(NULL),
     HDivFESpace2p_(NULL),
     a1_(NULL),
     b1_(NULL),
     m2_(NULL),
     m12EpsRe_(NULL),
     m12EpsIm_(NULL),
     curl_(NULL),
     kReCross_(NULL),
     kImCross_(NULL),
     e_(NULL),
     d_(NULL),
     j_(NULL),
     rhs_(NULL),
     e_t_(NULL),
     e_b_(NULL),
     e_v_(NULL),
     b_v_(NULL),
     d_v_(NULL),
     j_v_(NULL),
     b_hat_(NULL),
     u_(NULL),
     uE_(NULL),
     uB_(NULL),
     S_(NULL),
     StixS_(NULL),
     StixD_(NULL),
     StixP_(NULL),
     BCoef_(&BCoef),
     epsReCoef_(&epsReCoef),
     epsImCoef_(&epsImCoef),
     epsAbsCoef_(&epsAbsCoef),
     muInvCoef_(&muInvCoef),
     etaInvCoef_(etaInvCoef),
     kReCoef_(kReCoef),
     kImCoef_(kImCoef),
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
     sbcReCoef_(NULL),
     sbcImCoef_(NULL),
     sinkx_(NULL),
     coskx_(NULL),
     negsinkx_(NULL),
     // negMuInvCoef_(NULL),
     massReCoef_(NULL),
     massImCoef_(NULL),
     posMassCoef_(NULL),
     kmkReCoef_(kReCoef_, kImCoef_, muInvCoef_, true, -1.0),
     kmkImCoef_(kReCoef_, kImCoef_, muInvCoef_, false, -1.0),
     kmReCoef_(kReCoef_, muInvCoef_, 1.0),
     kmImCoef_(kImCoef_, muInvCoef_, -1.0),
     // negMuInvkxkxCoef_(NULL),
     // negMuInvkCoef_(NULL),
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
     // dbcs_(&dbcs),
     dbcs_(stixBCs.GetDirichletBCs()),
     // nbcs_(&nbcs),
     nbcs_(stixBCs.GetNeumannBCs()),
     // nkbcs_(NULL),
     sbcs_(stixBCs.GetSheathBCs()),
     visit_dc_(NULL)
{
   // Initialize MPI variables
   MPI_Comm_size(pmesh_->GetComm(), &num_procs_);
   MPI_Comm_rank(pmesh_->GetComm(), &myid_);

   if ( myid_ == 0 && logging_ > 0 )
   {
      cout << "Constructing CPDSolverEB ..." << endl;
   }

   tic_toc.Clear();
   tic_toc.Start();

   // Define compatible parallel finite element spaces on the parallel
   // mesh. Here we use arbitrary order H1, Nedelec, and Raviart-Thomas finite
   // elements.
   switch (pmesh_->Dimension())
   {
      case 1:
         HCurlFESpace_ = new ND_R1D_ParFESpace(pmesh_,order,pmesh_->Dimension());
         HDivFESpace_  = new RT_R1D_ParFESpace(pmesh_,order,pmesh_->Dimension());
         break;
      case 2:
         HCurlFESpace_ = new ND_R2D_ParFESpace(pmesh_,order,pmesh_->Dimension());
         HDivFESpace_  = new RT_R2D_ParFESpace(pmesh_,order,pmesh_->Dimension());
         break;
      case 3:
         HCurlFESpace_ = new ND_ParFESpace(pmesh_,order,pmesh_->Dimension());
         HDivFESpace_  = new RT_ParFESpace(pmesh_,order,pmesh_->Dimension());
         break;
   }

   if (BCoef_)
   {
      if (L2FESpace_ == NULL)
      {
         L2FESpace_ = new L2_ParFESpace(pmesh_,order-1,pmesh_->Dimension());
      }
      e_b_ = new ParComplexGridFunction(L2FESpace_);
      *e_b_ = 0.0;
      b_hat_ = new ParGridFunction(HDivFESpace_);
   }
   if (kReCoef_ || kImCoef_)
   {
      L2VFESpace_ = new L2_ParFESpace(pmesh_,order,pmesh_->Dimension(),
                                      pmesh_->SpaceDimension());
      e_t_ = new ParGridFunction(L2VFESpace_);
      // e_v_ = new ParComplexGridFunction(L2VFESpace_);
      // d_v_ = new ParComplexGridFunction(L2VFESpace_);
      // j_v_ = new ParComplexGridFunction(L2VFESpace_);

      // sinkx_ = new ComplexPhaseCoefficient(*kReCoef_, *kImCoef_, sin);
      // coskx_ = new ComplexPhaseCoefficient(*kReCoef_, *kImCoef_, cos);
      // negsinkx_ = new ProductCoefficient(-1.0, *sinkx_);
      /*
      negMuInvCoef_ = new ProductCoefficient(-1.0, *muInvCoef_);
      negMuInvkCoef_ = new ScalarVectorProductCoefficient(*negMuInvCoef_,
                                                          *kCoef_);
      negMuInvkxkxCoef_ = new CrossCrossCoefficient(*muInvCoef_, *kCoef_);
      */
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
   if ( dbcs_.Size() > 0 )
   {
      if ( dbcs_.Size() == 1 && dbcs_[0]->attr[0] == -1 )
      {
         ess_bdr_ = 1;
      }
      else
      {
         ess_bdr_ = 0;
         for (int i=0; i<dbcs_.Size(); i++)
         {
            for (int j=0; j<dbcs_[i]->attr.Size(); j++)
            {
               ess_bdr_[dbcs_[i]->attr[j]-1] = 1;
            }
         }
      }
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

      abc_bdr_marker_.SetSize(pmesh.bdr_attributes.Max());
      if ( abcs.Size() == 1 && abcs[0] < 0 )
      {
         // Mark all boundaries as absorbing
         abc_bdr_marker_ = 1;
      }
      else
      {
         // Mark select boundaries as absorbing
         abc_bdr_marker_ = 0;
         for (int i=0; i<abcs.Size(); i++)
         {
            abc_bdr_marker_[abcs[i]-1] = 1;
         }
      }
      if ( etaInvCoef_ == NULL )
      {
         etaInvCoef_ = new ConstantCoefficient(sqrt(epsilon0_/mu0_));
      }
      abcCoef_ = new TransformedCoefficient(negOmegaCoef_, etaInvCoef_,
                                            prodFunc);
   }
   /*
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
   */
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

   if (nbcs_.Size() > 0)
   {
      nkbcs_.SetSize(nbcs_.Size());
      for (int i=0; i<nbcs_.Size(); i++)
      {
         nkbcs_[i] = new ComplexVectorCoefficientByAttr;
         nkbcs_[i]->attr = nbcs_[i]->attr;
         nkbcs_[i]->attr_marker.SetSize(pmesh.bdr_attributes.Max());
         nkbcs_[i]->attr_marker = 0;
         for (int j=0; j<nbcs_[i]->attr.Size(); j++)
         {
            nkbcs_[i]->attr_marker[nbcs_[i]->attr[j] - 1] = 1;
         }

         VectorCoefficient * nk_re =
            new ComplexPhaseVectorCoefficient(kReCoef_, kImCoef_,
                                              nbcs_[i]->real, nbcs_[i]->imag,
                                              true, true);
         VectorCoefficient * nk_im =
            new ComplexPhaseVectorCoefficient(kReCoef_, kImCoef_,
                                              nbcs_[i]->real, nbcs_[i]->imag,
                                              false, true);

         vCoefs_.Append(nk_re);
         vCoefs_.Append(nk_im);

         VectorCoefficient * wnk_re =
            new ScalarVectorProductCoefficient(-omega_, *nk_im);
         VectorCoefficient * wnk_im =
            new ScalarVectorProductCoefficient(omega_, *nk_re);

         vCoefs_.Append(wnk_re);
         vCoefs_.Append(wnk_im);

         nkbcs_[i]->real = wnk_re;
         nkbcs_[i]->imag = wnk_im;
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
   curl_ = new ParDiscreteCurlOperator(HCurlFESpace_, HDivFESpace_);

   // Bilinear Forms
   a1_ = new ParSesquilinearForm(HCurlFESpace_, conv_);
   if (pa_) { a1_->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a1_->AddDomainIntegrator(new CurlCurlIntegrator(*muInvCoef_), NULL);
   a1_->AddDomainIntegrator(new VectorFEMassIntegrator(*massReCoef_),
                            new VectorFEMassIntegrator(*massImCoef_));
   if ( kReCoef_ || kImCoef_ )
   {
      if (pa_)
      {
         MFEM_ABORT("kCoef_: Partial Assembly has not yet been implemented for "
                    "MixedCrossCurlIntegrator and MixedWeakCurlCrossIntegrator.");
      }
      a1_->AddDomainIntegrator(new VectorFEMassIntegrator(kmkReCoef_),
                               new VectorFEMassIntegrator(kmkImCoef_));
      a1_->AddDomainIntegrator(new MixedVectorCurlIntegrator(kmImCoef_),
                               new MixedVectorCurlIntegrator(kmReCoef_));
      a1_->AddDomainIntegrator(new MixedVectorWeakCurlIntegrator(kmImCoef_),
                               new MixedVectorWeakCurlIntegrator(kmReCoef_));
      /*
      a1_->AddDomainIntegrator(new VectorFEMassIntegrator(*negMuInvkxkxCoef_),
                               NULL);
      a1_->AddDomainIntegrator(NULL,
                               new MixedCrossCurlIntegrator(*negMuInvkCoef_));
      a1_->AddDomainIntegrator(NULL,
                               new MixedWeakCurlCrossIntegrator(*negMuInvkCoef_));
      */
   }
   if (kReCoef_)
   {
      kReCross_ = new ParDiscreteLinearOperator(HCurlFESpace_, HDivFESpace_);
      kReCross_->AddDomainInterpolator(
         new VectorCrossProductInterpolator(*kReCoef_));
   }
   if (kImCoef_)
   {
      kImCross_ = new ParDiscreteLinearOperator(HCurlFESpace_, HDivFESpace_);
      kImCross_->AddDomainInterpolator(
         new VectorCrossProductInterpolator(*kImCoef_));
   }

   if ( abcCoef_ )
   {
      if (pa_)
      {
         MFEM_ABORT("abcCoef_: Partial Assembly has not yet been tested for "
                    "this BoundaryIntegrator.");
      }
      a1_->AddBoundaryIntegrator(NULL, new VectorFEMassIntegrator(*abcCoef_),
                                 abc_bdr_marker_);
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

   // Build grid functions
   e_  = new ParComplexGridFunction(HCurlFESpace_);
   *e_ = 0.0;

   d_  = new ParComplexGridFunction(HDivFESpace_);
   *d_ = 0.0;

   b_  = new ParComplexGridFunction(HDivFESpace_);
   *b_ = 0.0;
   // solNorm_ = e_->ComputeL2Error(const_cast<VectorCoefficient&>(erCoef_),
   //                               const_cast<VectorCoefficient&>(eiCoef_));

   j_ = new ParComplexGridFunction(HDivFESpace_);
   j_->ProjectCoefficient(*jrCoef_, *jiCoef_);

   rhs_ = new ParComplexLinearForm(HCurlFESpace_, conv_);
   rhs_->AddDomainIntegrator(new VectorFEDomainLFIntegrator(*rhsrCoef_),
                             new VectorFEDomainLFIntegrator(*rhsiCoef_));

   if (nkbcs_ != NULL)
   {
      for (int i=0; i<nkbcs_.Size(); i++)
      {
         rhs_->AddBoundaryIntegrator(new VectorFEBoundaryTangentialLFIntegrator(*
                                                                                nkbcs_[i]->real),
                                     new VectorFEBoundaryTangentialLFIntegrator(*nkbcs_[i]->imag),
                                     nkbcs_[i]->attr_marker);

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
      }
   }

   tic_toc.Stop();

   if ( myid_ == 0 && logging_ > 0 )
   {
      cout << " done in " << tic_toc.RealTime() << " seconds." << endl;
   }
}

CPDSolverEB::~CPDSolverEB()
{
   // delete negMuInvkxkxCoef_;
   // delete negMuInvkCoef_;
   // delete negMuInvCoef_;
   for (int i=0; i<vCoefs_.Size(); i++)
   {
      delete vCoefs_[i];
   }

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
   delete sbcReCoef_;
   delete sbcImCoef_;
   if ( ownsEtaInv_ ) { delete etaInvCoef_; }
   delete omegaCoef_;
   delete negOmegaCoef_;
   delete omega2Coef_;
   delete negOmega2Coef_;

   // delete DivFreeProj_;
   // delete SurfCur_;

   delete StixS_;
   delete StixD_;
   delete StixP_;
   delete e_v_;
   delete d_v_;
   delete j_v_;
   delete e_b_;
   delete b_hat_;
   delete b_hat_v_;
   // delete e_r_;
   // delete e_i_;
   delete e_;
   delete d_;
   delete b_;
   // delete h_;
   delete j_;
   delete u_;
   delete uE_;
   delete uB_;
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
   // delete grad_;
   delete curl_;

   delete a1_;
   delete b1_;
   delete m2_;
   delete m12EpsRe_;
   delete m12EpsIm_;

   // delete curlMuInvCurl_;
   // delete hCurlMass_;
   // delete hDivHCurlMuInv_;
   // delete weakCurlMuInv_;

   delete L2FESpace_;
   delete L2FESpace2p_;
   delete L2VFESpace_;
   // delete H1FESpace_;
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
CPDSolverEB::GetProblemSize()
{
   return 2 * HCurlFESpace_->GlobalTrueVSize();
}

void
CPDSolverEB::PrintSizes()
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
CPDSolverEB::Assemble()
{
   if ( myid_ == 0 && logging_ > 0 ) { cout << "Assembling ..." << flush; }

   tic_toc.Clear();
   tic_toc.Start();

   // a0_->Assemble();
   // a0_->Finalize();

   a1_->Assemble();
   if (!pa_) { a1_->Finalize(); }

   b1_->Assemble();
   if (!pa_) { b1_->Finalize(); }

   m2_->Assemble();
   if (!pa_) { m2_->Finalize(); }

   // TODO: PA
   m12EpsRe_->Assemble();
   m12EpsRe_->Finalize();
   //if (!pa_) m12EpsRe_->Finalize();

   // TODO: PA
   m12EpsIm_->Assemble();
   m12EpsIm_->Finalize();
   //if (!pa_) m12EpsIm_->Finalize();

   rhs_->Assemble();
   /*
   curlMuInvCurl_->Assemble();
   curlMuInvCurl_->Finalize();
   hDivHCurlMuInv_->Assemble();
   hDivHCurlMuInv_->Finalize();
   hCurlMass_->Assemble();
   hCurlMass_->Finalize();
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
   curl_->Assemble();
   curl_->Finalize();

   if (kReCross_)
   {
      kReCross_->Assemble();
      kReCross_->Finalize();
   }
   if (kImCross_)
   {
      kImCross_->Assemble();
      kImCross_->Finalize();
   }

   tic_toc.Stop();

   if ( myid_ == 0 && logging_ > 0 )
   {
      cout << " done in " << tic_toc.RealTime() << " seconds." << endl;
   }
}

void
CPDSolverEB::Update()
{
   if ( myid_ == 0 && logging_ > 0 ) { cout << "Updating ..." << endl; }

   tic_toc.Clear();
   tic_toc.Start();

   // Inform the spaces that the mesh has changed
   // Note: we don't need to interpolate any GridFunctions on the new mesh
   // so we pass 'false' to skip creation of any transformation matrices.
   // H1FESpace_->Update(false);
   if (L2FESpace_) { L2FESpace_->Update(); }
   if (L2FESpace2p_) { L2FESpace2p_->Update(); }
   if (L2VFESpace_) { L2VFESpace_->Update(); }
   HCurlFESpace_->Update();
   HDivFESpace_->Update();
   if (HDivFESpace2p_) { HDivFESpace2p_->Update(false); }

   if ( ess_bdr_.Size() > 0 )
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
   b_->Update();
   if (u_) { u_->Update(); }
   if (uE_) { uE_->Update(); }
   if (uB_) { uB_->Update(); }
   if (S_) { S_->Update(); }
   if (e_t_) { e_t_->Update(); }
   if (e_b_) { e_b_->Update(); }
   if (e_v_) { e_v_->Update(); }
   if (d_v_) { d_v_->Update(); }
   if (j_v_) { j_v_->Update(); }
   if (b_hat_) { b_hat_->Update(); }
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
   // curlMuInvCurl_->Update();
   // hCurlMass_->Update();
   // hDivHCurlMuInv_->Update();
   // if ( weakCurlMuInv_ ) { weakCurlMuInv_->Update(); }

   // Inform the other objects that the space has changed.
   curl_->Update();
   if (kReCross_) { kReCross_->Update(); }
   if (kImCross_) { kImCross_->Update(); }
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
CPDSolverEB::Solve()
{
   if ( myid_ == 0 && logging_ > 0 ) { cout << "Running solver ... " << endl; }

   OperatorHandle A1;
   Vector E, RHS;
   // cout << "Norm of jd (pre-fls): " << jd_->Norml2() << endl;
   if (dbcs_.Size() > 0)
   {
      Array<int> attr_marker(pmesh_->bdr_attributes.Max());
      for (int i = 0; i<dbcs_.Size(); i++)
      {
         attr_marker = 0;
         for (int j=0; j<dbcs_[i]->attr.Size(); j++)
         {
            attr_marker[dbcs_[i]->attr[j] - 1] = 1;
         }

         bool dbc_phase = true;
         if (dbc_phase)
         {
            ComplexPhaseVectorCoefficient re_e_dbc(kReCoef_, kImCoef_,
                                                   dbcs_[i]->real,
                                                   dbcs_[i]->imag, true, true);
            ComplexPhaseVectorCoefficient im_e_dbc(kReCoef_, kImCoef_,
                                                   dbcs_[i]->real,
                                                   dbcs_[i]->imag, false, true);

            e_->ProjectBdrCoefficientTangent(re_e_dbc, im_e_dbc, attr_marker);
         }
         else
         {
            e_->ProjectBdrCoefficientTangent(*dbcs_[i]->real, *dbcs_[i]->imag,
                                             attr_marker);
         }
      }
   }

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
#ifdef MFEM_USE_MUMPS
      case DMUMPS:
      {
         if ( myid_ == 0 && logging_ > 0 )
         {
            cout << "MUMPS (Real) Solver Requested" << endl;
         }
         ComplexHypreParMatrix * A1Z = A1.As<ComplexHypreParMatrix>();
         HypreParMatrix * A1C = A1Z->GetSystemMatrix();
         MUMPSSolver dmumps;
         dmumps.SetPrintLevel(1);
         dmumps.SetMatrixSymType(MUMPSSolver::MatType::UNSYMMETRIC);
         dmumps.SetOperator(*A1C);
         dmumps.Mult(RHS, E);
         delete A1C;
      }
      break;
      case ZMUMPS:
      {
         if ( myid_ == 0 && logging_ > 0 )
         {
            cout << "MUMPS (Complex) Solver Requested" << endl;
         }
         ComplexHypreParMatrix * A1Z = A1.As<ComplexHypreParMatrix>();
         ComplexMUMPSSolver zmumps;
         zmumps.SetPrintLevel(1);
         zmumps.SetOperator(*A1Z);
         zmumps.Mult(RHS, E);
      }
      break;
#endif
      default:
         MFEM_ABORT("Requested solver is not available.");
         break;
   };

   tic_toc.Stop();

   e_->Distribute(E);

   this->computeB(*e_, *b_);

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

      D.SetSize(HDivFESpace_->TrueVSize());
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

   delete BDP;
   if (pci != pcr) { delete pci; }
   delete pcr;

   if ( myid_ == 0 && logging_ > 0 )
   {
      cout << " Solver done in " << tic_toc.RealTime() << " seconds." << endl;
   }
}

// Compute B from E:  B = Curl(E) / (i omega)
void CPDSolverEB::computeB(const ParComplexGridFunction & e,
                           ParComplexGridFunction & b)
{
   // B = Curl(E) / (i omega) = -i Curl(E) / omega
   curl_->Mult(e.imag(), b.real()); b.real() *=  1.0 / omega_;
   curl_->Mult(e.real(), b.imag()); b.imag() *= -1.0 / omega_;

   if (kReCoef_ || kImCoef_)
   {
      // B += i k x E / (i omega) = k x E / omega
      kReCross_->AddMult(e.real(), b.real(),  1.0 / omega_);
      kReCross_->AddMult(e.imag(), b.imag(),  1.0 / omega_);

      kImCross_->AddMult(e.imag(), b.real(), -1.0 / omega_);
      kImCross_->AddMult(e.real(), b.imag(),  1.0 / omega_);
   }

   if (logging_ > 1)
   {
      Vector zeroVec(3); zeroVec = 0.0;
      VectorConstantCoefficient zeroVCoef(zeroVec);

      double nrmb = b.ComputeL2Error(zeroVCoef, zeroVCoef);
      if (myid_ == 0) { cout << "norm of B: " << nrmb << endl; }
   }
}

double
CPDSolverEB::GetEFieldError(const VectorCoefficient & EReCoef,
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
CPDSolverEB::GetErrorEstimates(Vector & errors)
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

void CPDSolverEB::prepareVectorVisField(const ParComplexGridFunction &u,
                                        ComplexGridFunction &v)
{
   if (kReCoef_ || kImCoef_)
   {
      VectorGridFunctionCoefficient u_r(&u.real());
      VectorGridFunctionCoefficient u_i(&u.imag());
      ComplexPhaseVectorCoefficient uk_r(kReCoef_, kImCoef_, &u_r, &u_i,
                                         true, false);
      ComplexPhaseVectorCoefficient uk_i(kReCoef_, kImCoef_, &u_r, &u_i,
                                         false, false);

      VectorR2DCoef uk_r_3D(uk_r, *pmesh_);
      VectorR2DCoef uk_i_3D(uk_i, *pmesh_);

      v.ProjectCoefficient(uk_r_3D, uk_i_3D);
   }
   else
   {
      VectorGridFunctionCoefficient u_r(&u.real());
      VectorGridFunctionCoefficient u_i(&u.imag());

      VectorR2DCoef u_r_3D(u_r, *pmesh_);
      VectorR2DCoef u_i_3D(u_i, *pmesh_);

      v.ProjectCoefficient(u_r_3D, u_i_3D);
   }
}

void CPDSolverEB::prepareVisFields()
{
   prepareVectorVisField(*e_, *e_v_);
   prepareVectorVisField(*d_, *d_v_);
   prepareVectorVisField(*b_, *b_v_);
   /*
   if (h_tilde_)
   {
      VectorGridFunctionCoefficient u_r(&h_->real());
      VectorGridFunctionCoefficient u_i(&h_->imag());

      VectorR2DCoef u_r_3D(u_r, *pmesh_);
      VectorR2DCoef u_i_3D(u_i, *pmesh_);

      h_tilde_->ProjectCoefficient(u_r_3D, u_i_3D);
   }
   */
   /*
   {
      VectorGridFunctionCoefficient d_r(&d_->real());
      VectorGridFunctionCoefficient d_i(&d_->real());

      DivergenceGridFunctionCoefficient div_d_r(&d_->real());
      DivergenceGridFunctionCoefficient div_d_i(&d_->imag());

      VectorR2DCoef dReCoef(d_r, *pmesh_);
      VectorR2DCoef dImCoef(d_i, *pmesh_);

      Vector ZeroVec(3); ZeroVec = 0.0;
      VectorConstantCoefficient ZeroCoef(ZeroVec);
      VectorCoefficient * kReCoef = (kReCoef_) ? kReCoef_ : &ZeroCoef;
      VectorCoefficient * kImCoef = (kImCoef_) ? kImCoef_ : &ZeroCoef;

      InnerProductCoefficient krdr(*kReCoef, dReCoef);
      InnerProductCoefficient krdi(*kReCoef, dImCoef);
      InnerProductCoefficient kidr(*kImCoef, dReCoef);
      InnerProductCoefficient kidi(*kImCoef, dImCoef);

      SumCoefficient ikd_r(krdi, kidr, -1.0, -1.0);
      SumCoefficient ikd_i(krdr, kidi,  1.0, -1.0);

      SumCoefficient div_d_r_3D(div_d_r, ikd_r);
      SumCoefficient div_d_i_3D(div_d_i, ikd_i);

      ComplexPhaseCoefficient ddk_r(kReCoef_, kImCoef_, &div_d_r_3D, &div_d_i_3D,
                                    true, false);
      ComplexPhaseCoefficient ddk_i(kReCoef_, kImCoef_, &div_d_r_3D, &div_d_i_3D,
                                    false, false);

      div_d_->ProjectCoefficient(ddk_r, ddk_i);
   }
   */
   if (BCoef_)
   {
      // VectorGridFunctionCoefficient b_hat(b_hat_);
      // VectorR2DCoef b_hat_3D(b_hat, *pmesh_);
      VectorR2DCoef b_hat_3D(*BCoef_, *pmesh_);
      b_hat_v_->ProjectCoefficient(b_hat_3D);
      /*
      VectorGridFunctionCoefficient e_r(&e_v_->real());
      VectorGridFunctionCoefficient e_i(&e_v_->imag());
      InnerProductCoefficient eb_r(e_r, *BCoef_);
      InnerProductCoefficient eb_i(e_i, *BCoef_);

      e_b_v_->ProjectCoefficient(eb_r, eb_i);
      */
   }
}

void
CPDSolverEB::RegisterVisItFields(VisItDataCollection & visit_dc)
{
   visit_dc_ = &visit_dc;

   if (L2FESpace3D_ == NULL)
   {
      L2FESpace3D_  = new L2_FESpace(visit_dc_->GetMesh(),order_-1,3);
   }
   if (L2VFESpace3D_ == NULL)
   {
      L2VFESpace3D_  = new L2_FESpace(visit_dc_->GetMesh(),order_-1,3,3);
   }

   StixS_ = new ComplexGridFunction(L2FESpace3D_);
   StixD_ = new ComplexGridFunction(L2FESpace3D_);
   StixP_ = new ComplexGridFunction(L2FESpace3D_);

   e_v_ = new ComplexGridFunction(L2VFESpace3D_);
   d_v_ = new ComplexGridFunction(L2VFESpace3D_);
   b_v_ = new ComplexGridFunction(L2VFESpace3D_);

   visit_dc.RegisterField("Re_E", &e_v_->real());
   visit_dc.RegisterField("Im_E", &e_v_->imag());

   visit_dc.RegisterField("Re_D", &d_v_->real());
   visit_dc.RegisterField("Im_D", &d_v_->imag());

   visit_dc.RegisterField("Re_B", &b_v_->real());
   visit_dc.RegisterField("Im_B", &b_v_->imag());

   // visit_dc.RegisterField("Er", e_r_);
   // visit_dc.RegisterField("Ei", e_i_);
   // visit_dc.RegisterField("B", b_);
   // visit_dc.RegisterField("H", h_);
   if ( BCoef_)
   {
      b_hat_v_ = new GridFunction(L2VFESpace3D_);

      visit_dc.RegisterField("B_hat", b_hat_v_);
   }

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
   // if ( j_r_ ) { visit_dc.RegisterField("Jr", j_r_); }
   // if ( j_i_ ) { visit_dc.RegisterField("Ji", j_i_); }
   // if ( k_ ) { visit_dc.RegisterField("K", k_); }
   // if ( m_ ) { visit_dc.RegisterField("M", m_); }
   // if ( SurfCur_ ) { visit_dc.RegisterField("Psi", SurfCur_->GetPsi()); }
   if ( StixS_ )
   {
      visit_dc.RegisterField("Re_StixS", &StixS_->real());
      visit_dc.RegisterField("Im_StixS", &StixS_->imag());
      visit_dc.RegisterField("Re_StixD", &StixD_->real());
      visit_dc.RegisterField("Im_StixD", &StixD_->imag());
      visit_dc.RegisterField("Re_StixP", &StixP_->real());
      visit_dc.RegisterField("Im_StixP", &StixP_->imag());
   }
}

void
CPDSolverEB::WriteVisItFields(int it)
{
   if ( visit_dc_ )
   {
      if (myid_ == 0) { cout << "Writing VisIt files ..." << flush; }

      prepareVisFields();

      if ( StixS_ )
      {
         StixS_->ProjectCoefficient(*SReCoef_, *SImCoef_);
         StixD_->ProjectCoefficient(*DReCoef_, *DImCoef_);
         StixP_->ProjectCoefficient(*PReCoef_, *PImCoef_);
      }
      /*
      curl_->Mult(e_->real(), b_->imag());
      curl_->Mult(e_->imag(), b_->real());
      if (kImCross_)
      {
         kImCross_->AddMult(e_->real(), b_->imag(), -1.0);
         kImCross_->AddMult(e_->imag(), b_->real(), -1.0);
      }
      if (kReCross_)
      {
         kReCross_->AddMult(e_->imag(), b_->imag(), -1.0);
         kReCross_->AddMult(e_->real(), b_->real(),  1.0);
      }
      b_->real() /= omega_;
      b_->imag() /= -omega_;

      if ( BCoef_)
      {
         b_hat_->ProjectCoefficient(*BCoef_);
      }

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
      */
      HYPRE_Int prob_size = this->GetProblemSize();
      visit_dc_->SetCycle(it);
      visit_dc_->SetTime(prob_size);
      visit_dc_->Save();

      if (myid_ == 0) { cout << " done." << endl; }
   }
}

void
CPDSolverEB::InitializeGLVis()
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
CPDSolverEB::DisplayToGLVis()
{
   if (myid_ == 0) { cout << "Sending data to GLVis ..." << flush; }

   char vishost[] = "localhost";
   int  visport   = 19916;

   int Wx = 0, Wy = 0; // window position
   int Ww = 350, Wh = 350; // window size
   int offx = Ww+10, offy = Wh+45; // window offsets

   prepareVisFields();
   /*
   if (kReCoef_ || kImCoef_)
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
   }
   else
   {
      e_v_ = e_;
      d_v_ = d_;
   }
   */
   ostringstream er_keys, ei_keys;
   er_keys << "aaAcppppp";
   ei_keys << "aaAcppppp";

   switch (pmesh_->Dimension())
   {
      case 1:
         break;
      case 2:
         break;
      case 3:
         VisualizeField(*socks_["Er"], vishost, visport,
                        e_v_->real(), "Electric Field, Re(E)", Wx, Wy, Ww, Wh,
                        er_keys.str().c_str());
         Wx += offx;

         VisualizeField(*socks_["Ei"], vishost, visport,
                        e_v_->imag(), "Electric Field, Im(E)", Wx, Wy, Ww, Wh,
                        ei_keys.str().c_str());
         break;
   }

   ostringstream br_keys, bi_keys;
   br_keys << "aaAcPPPP";
   bi_keys << "aaAcPPPP";

   switch (pmesh_->Dimension())
   {
      case 1:
         break;
      case 2:
         break;
      case 3:

         VisualizeField(*socks_["Br"], vishost, visport,
                        b_v_->real(), "Magnetic Field, Re(B)", Wx, Wy, Ww, Wh,
                        br_keys.str().c_str());
         Wx += offx;

         VisualizeField(*socks_["Bi"], vishost, visport,
                        b_v_->imag(), "Magnetic Field, Im(B)", Wx, Wy, Ww, Wh,
                        br_keys.str().c_str());

         Wx += offx;
         break;
   }

   /*
   VisualizeField(*socks_["Er"], vishost, visport,
                  e_v_->real(), "Electric Field, Re(E)", Wx, Wy, Ww, Wh);
   Wx += offx;

   VisualizeField(*socks_["Ei"], vishost, visport,
                  e_v_->imag(), "Electric Field, Im(E)", Wx, Wy, Ww, Wh);
   */
   /*
   Wx += offx;
   VisualizeField(*socks_["Dr"], vishost, visport,
                  d_v_->real(), "Electric Flux, Re(D)", Wx, Wy, Ww, Wh);
   Wx += offx;

   VisualizeField(*socks_["Di"], vishost, visport,
                  d_v_->imag(), "Electric Flux, Im(D)", Wx, Wy, Ww, Wh);
   */
   if (BCoef_)
   {
      /*
       VectorGridFunctionCoefficient e_r(&e_v_->real());
       VectorGridFunctionCoefficient e_i(&e_v_->imag());
       InnerProductCoefficient ebrCoef(e_r, *BCoef_);
       InnerProductCoefficient ebiCoef(e_i, *BCoef_);

       e_b_->ProjectCoefficient(ebrCoef, ebiCoef);
      */
      /*
       VisualizeField(*socks_["EBr"], vishost, visport,
                      e_b_->real(), "Parallel Electric Field, Re(E.B)",
                      Wx, Wy, Ww, Wh);
       Wx += offx;

       VisualizeField(*socks_["EBi"], vishost, visport,
                      e_b_->imag(), "Parallel Electric Field, Im(E.B)",
                      Wx, Wy, Ww, Wh);
       Wx += offx;
      */
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
      /*
      j_->ProjectCoefficient(*jrCoef_, *jiCoef_);

      if (kReCoef_ || kImCoef_)
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
      /*
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
      */
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
CPDSolverEB::DisplayAnimationToGLVis()
{
   if (myid_ == 0) { cout << "Sending animation data to GLVis ..." << flush; }

   if (kReCoef_ || kImCoef_)
   {
      VectorGridFunctionCoefficient e_r(&e_->real());
      VectorGridFunctionCoefficient e_i(&e_->imag());
      VectorSumCoefficient erCoef(e_r, e_i, *coskx_, *negsinkx_);
      VectorSumCoefficient eiCoef(e_i, e_r, *coskx_, *sinkx_);

      e_v_->ProjectCoefficient(erCoef, eiCoef);
   }
   else
   {
      VectorGridFunctionCoefficient e_r(&e_->real());
      VectorGridFunctionCoefficient e_i(&e_->imag());

      e_v_->ProjectCoefficient(e_r, e_i);
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
           sin( 2.0 * M_PI * t), e_v_->imag(), *e_t_);
      sol_sock << "parallel " << num_procs_ << " " << myid_ << "\n";
      sol_sock << "solution\n" << *pmesh_ << *e_t_
               << "window_title '" << oss.str() << "'" << flush;
      i++;
   }
}

} // namespace plasma

} // namespace mfem

#endif // MFEM_USE_MPI
