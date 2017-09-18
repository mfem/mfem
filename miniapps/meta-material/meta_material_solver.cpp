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

#include "../../config/config.hpp"
#include <cassert>

#ifdef MFEM_USE_MPI

#include "meta_material_solver.hpp"

using namespace std;

namespace mfem
{

using namespace miniapps;

namespace meta_material
{

LinearCoefficient::LinearCoefficient(GridFunction * gf, double c0, double c1)
   : GridFunctionCoefficient(gf),
     c0_(c0),
     c1_(c1)
{
}

double
LinearCoefficient::Eval(ElementTransformation &T,
                        const IntegrationPoint &ip)
{
   double vf = this->GridFunctionCoefficient::Eval(T, ip);
   return c0_ + (c1_ - c0_) * vf;
}
/*
PenaltyCoefficient::PenaltyCoefficient(GridFunction * gf, int penalty,
                                     double c0, double c1)
 : GridFunctionCoefficient(gf),
   penalty_(penalty),
   c0_(c0),
   c1_(c1)
{
}

double
PenaltyCoefficient::Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
{
 double vf = this->GridFunctionCoefficient::Eval(T, ip);
 return c0_ + (c1_ - c0_) * pow(vf, penalty_);
}

double
PenaltyCoefficient::GetSensitivity(ElementTransformation &T,
                                 const IntegrationPoint &ip)
{
 double vf = this->GridFunctionCoefficient::Eval(T, ip);
 return (c1_ - c0_) * penalty_* pow(vf, penalty_ - 1);
}
*/
Homogenization::Homogenization(MPI_Comm comm)
   : comm_(comm), newVF_(true), vf_(NULL)
{
   MPI_Comm_size(comm_, &numProcs_);
   MPI_Comm_rank(comm_, &myid_);
}

Density::Density(ParMesh & pmesh, double refDensity, double vol,
                 Coefficient &rhoCoef, double tol)
   : Homogenization(pmesh.GetComm()),
     pmesh_(&pmesh),
     L2FESpace_(NULL),
     rho_(NULL),
     cellVol_(NULL),
     rhoCoef_(&rhoCoef),
     one_(1.0),
     refDensity_(refDensity),
     vol_(vol),
     tol_(tol),
     vd_(NULL),
     sock_(NULL),
     sock2_(NULL)
{
   L2FESpace_  = new L2_ParFESpace(pmesh_, 0, pmesh_->Dimension());
   rho_        = new ParGridFunction(L2FESpace_);
   divGradRho_ = new ParGridFunction(L2FESpace_);
   cellVol_    = new ParLinearForm(L2FESpace_);

   cellVol_->AddDomainIntegrator(new DomainLFIntegrator(one_));
   cellVol_->Assemble();

   *rho_ = 0.0;
}

Density::~Density()
{
   delete sock_;
   delete sock2_;
   delete cellVol_;
   delete rho_;
   delete divGradRho_;
   delete L2FESpace_;
}
/*
void
Density::SetVolumeFraction(ParGridFunction & vf)
{
 this->Homogenization::SetVolumeFraction(vf);

 rhoCoef_.SetGridFunction(vf_);
}
*/
void
Density::updateRho()
{
   rho_->ProjectCoefficient(*rhoCoef_);
   double old_mass = cellVol_->operator()(*rho_);
   double mass = 2.0 * old_mass;
   int maxiter = 5;
   int iter = 0;
   double fill_tol = 1e-4;

   // If no mass appears in the unit cell or if the unit cell is full,
   // refine until some material is resolved
   while ((mass < refDensity_ * vol_ * fill_tol ||
           mass - refDensity_ * vol_ > refDensity_ * vol_ * (1.0-fill_tol)) &&
          iter < maxiter)
   {
      pmesh_->UniformRefinement();

      rho_->Update();
      rho_->ProjectCoefficient(*rhoCoef_);

      cellVol_->Update();
      cellVol_->Assemble();

      old_mass = cellVol_->operator()(*rho_);
      mass = 2.0 * old_mass;

      divGradRho_->Update();

      iter++;
   }

   // Refine elements elements which are partially full to better
   // capture the edges
   iter = 0;
   while (iter < maxiter && fabs(mass - old_mass) > tol_ * mass)
   {
      // rho_->ProjectCoefficient(*rhoCoef_);
      // double mass = cellVol_->operator()(*rho_);
      double nrm = rho_->Normlinf();
      cout << "max norm of rho: " << nrm << endl;
      {
         H1_ParFESpace H1FESpace(pmesh_, 1, pmesh_->Dimension());
         ParGridFunction rhoCont(&H1FESpace);
         rhoCont.ProjectCoefficient(*rhoCoef_);

         GridFunctionCoefficient rhoContFunc(&rhoCont);

         divGradRho_->ProjectCoefficient(rhoContFunc);
         *divGradRho_ -= *rho_;

         double nrmRho = rho_->Normlinf();
         if ( nrmRho == 0.0 ) { nrmRho = 1.0; }
         for (int j=0; j<divGradRho_->Size(); j++)
         {
            (*divGradRho_)[j] = fabs((*divGradRho_)[j]) / nrmRho;
         }
         cout << "Max error: " << divGradRho_->Normlinf() << endl;
      }
      // if ( i < niter-1 )
      {
         cout << "Refining " << iter+1 << endl;
         pmesh_->RefineByError(*divGradRho_, 0.001);
         cout << "Number of elements: " << pmesh_->GetNE() << endl;

         L2FESpace_->Update();

         rho_->Update();
         rho_->ProjectCoefficient(*rhoCoef_);

         cellVol_->Update();
         cellVol_->Assemble();
         old_mass = mass;
         mass = cellVol_->operator()(*rho_);
         cout << mass << ", |mass - old_mass|/mass = "
              << fabs(mass-old_mass)/mass << endl;
         divGradRho_->Update();
      }
      iter++;
   }
}

void
Density::GetHomogenizedProperties(vector<double> & p)
{
   if ( newVF_ ) { this->updateRho(); }
   p.resize(1);
   p[0] = cellVol_->operator()(*rho_);
   p[0] /= vol_;
}
/*
void
Density::GetPropertySensitivities(vector<ParGridFunction> & dp)
{
 if ( dp.size() < 1 )
 {
    dp.resize(1);
    dp[0].SetSpace(L2FESpace_);
 }

 Array<int> vdofs;
 ElementTransformation *eltrans;
 IntegrationPoint ip; ip.Init();
 for (int i=0; i<L2FESpace_->GetNE(); i++)
 {
    L2FESpace_->GetElementVDofs(i, vdofs);
    eltrans = L2FESpace_->GetElementTransformation(i);
    double dRho = rhoCoef_.GetSensitivity(*eltrans, ip);
    dp[0][vdofs[0]] = dRho * (*cellVol_)[vdofs[0]] / vol_;
 }
}
*/
void
Density::InitializeGLVis(VisData & vd)
{
   vd_ = &vd;

   if ( sock_ == NULL)
   {
      sock_ = new socketstream;
      sock_->precision(8);
   }
   if ( sock2_ == NULL)
   {
      sock2_ = new socketstream;
      sock2_->precision(8);
   }
}

void
Density::DisplayToGLVis()
{
   if (vd_ == NULL)
   {
      MFEM_WARNING("DisplayToGLVis being called before InitializeGLVis!");
      return;
   }
   VisualizeField(*sock_, *rho_, "Density", *vd_);
   vd_->IncrementWindow();
   VisualizeField(*sock2_, *divGradRho_, "Normalized DivGrad Density", *vd_);
   vd_->IncrementWindow();
}

void
Density::WriteVisItFields(const string & prefix,
                          const string & label)
{
   VisItDataCollection visit_dc(label.c_str(), pmesh_);
   visit_dc.SetPrefixPath(prefix.c_str());
   visit_dc.RegisterField("Density", rho_);
   visit_dc.RegisterField("DivGrad Density", divGradRho_);
   visit_dc.Save();
}
/*
StiffnessTensor::StiffnessTensor(ParMesh & pmesh, double vol,
                                 double lambda0, double mu0,
                                 double lambda1, double mu1)
   :  Homogenization(pmesh.GetComm()),
      dim_(pmesh.SpaceDimension()),
      amg_elast_(false),
      pmesh_(&pmesh),
      L2FESpace_(NULL),
      H1FESpace_(NULL),
      H1VFESpace_(NULL),
      HCurlFESpace_(NULL),
      HCurlVFESpace_(NULL),
      lambdaCoef_(NULL, lambda0, lambda1),
      muCoef_(NULL, mu0, mu1),
      xxCoef_(lambdaCoef_, muCoef_, 0),
      yyCoef_(lambdaCoef_, muCoef_, 1),
      zzCoef_(lambdaCoef_, muCoef_, 2),
      yzCoef_(lambdaCoef_, muCoef_, 1, 2),
      xzCoef_(lambdaCoef_, muCoef_, 0, 2),
      xyCoef_(lambdaCoef_, muCoef_, 0, 1),
      lambda_(NULL),
      mu_(NULL),
      a_(NULL),
      grad_(NULL),
      b_(NULL),
      tmp1_(NULL),
      vol_(vol),
      vd_(NULL),
      seqVF_(0)
*/
StiffnessTensor::StiffnessTensor(ParMesh & pmesh, double vol,
                                 Coefficient &lambdaCoef, Coefficient &muCoef,
                                 double tol)
   :  Homogenization(pmesh.GetComm()),
      dim_(pmesh.SpaceDimension()),
      amg_elast_(false),
      pmesh_(&pmesh),
      L2FESpace_(NULL),
      L2VFESpace_(NULL),
      H1FESpace_(NULL),
      H1VFESpace_(NULL),
      HCurlFESpace_(NULL),
      HCurlVFESpace_(NULL),
      HDivFESpace_(NULL),
      lambdaCoef_(&lambdaCoef),
      muCoef_(&muCoef),
      xxCoef_(*lambdaCoef_, *muCoef_, 0),
      yyCoef_(*lambdaCoef_, *muCoef_, 1),
      zzCoef_(*lambdaCoef_, *muCoef_, 2),
      yzCoef_(*lambdaCoef_, *muCoef_, 1, 2),
      xzCoef_(*lambdaCoef_, *muCoef_, 0, 2),
      xyCoef_(*lambdaCoef_, *muCoef_, 0, 1),
      lambda_(NULL),
      mu_(NULL),
      a_(NULL),
      grad_(NULL),
      errSol_(NULL),
      b_(NULL),
      // tmp1_(NULL),
      vol_(vol),
      tol_(tol),
      vd_(NULL),
      seqVF_(0)
{
   L2FESpace_  = new L2_ParFESpace(pmesh_, 0, pmesh_->Dimension());
   L2VFESpace_ = new L2_ParFESpace(pmesh_, 1, pmesh_->Dimension(),
                                   pmesh_->SpaceDimension());

   H1FESpace_  = new H1_ParFESpace(pmesh_, 1, pmesh_->Dimension());
   H1VFESpace_ = new H1_ParFESpace(pmesh_, 1, pmesh_->Dimension(),
                                   BasisType::GaussLobatto,
                                   pmesh_->SpaceDimension());

   HCurlFESpace_  = new ND_ParFESpace(pmesh_, 1, pmesh_->Dimension());
   HCurlVFESpace_ = new ND_ParFESpace(pmesh_, 1, pmesh_->Dimension(),
                                      pmesh_->SpaceDimension());
   HDivFESpace_   = new RT_ParFESpace(pmesh_, 1, pmesh_->Dimension());

   irOrder_ = H1FESpace_->GetElementTransformation(0)->OrderW() + 2;
   geom_ = H1FESpace_->GetFE(0)->GetGeomType();
   const IntegrationRule * ir = &IntRules.Get(geom_, irOrder_);

   grad_ = new ParDiscreteGradOperator(H1FESpace_, HCurlFESpace_);
   // grad_->Assemble();
   // grad_->Finalize();

   lambda_ = new ParGridFunction(L2FESpace_);
   mu_     = new ParGridFunction(L2FESpace_);

   a_ = new ParBilinearForm(H1VFESpace_);
   BilinearFormIntegrator * elasInteg =
      new ElasticityIntegrator(*lambdaCoef_, *muCoef_);
   elasInteg->SetIntRule(ir);
   a_->AddDomainIntegrator(elasInteg);
   // a_->Update();
   // a_->Assemble(0);
   // a_->Finalize(0);

   for (int i=0; i<3; i++)
   {
      E_[i]   = new ParGridFunction(HCurlFESpace_);
   }
   for (int i=0; i<6; i++)
   {
      m_[i]   = new ParBilinearForm(HCurlFESpace_);
      Chi_[i] = new ParGridFunction(H1VFESpace_);
      F_[i]   = new ParGridFunction(HCurlVFESpace_);
      // MF_[i]  = new ParGridFunction(HCurlVFESpace_);
      MF_[i]  = new ParLinearForm(HCurlVFESpace_);
   }

   BilinearFormIntegrator * massInteg[6];
   massInteg[0] = new VectorFEMassIntegrator(xxCoef_);
   massInteg[1] = new VectorFEMassIntegrator(yyCoef_);
   massInteg[2] = new VectorFEMassIntegrator(zzCoef_);
   massInteg[3] = new VectorFEMassIntegrator(yzCoef_);
   massInteg[4] = new VectorFEMassIntegrator(xzCoef_);
   massInteg[5] = new VectorFEMassIntegrator(xyCoef_);

   for (int i=0; i<6; i++)
   {
      massInteg[i]->SetIntRule(ir);
      m_[i]->AddDomainIntegrator(massInteg[i]);
      // m_[i]->Update();
      // m_[i]->Assemble(0);
      // m_[i]->Finalize(0);
   }

   errSol_ = new ParGridFunction(H1FESpace_, (double*)NULL);
   diffInteg_[0] = new DiffusionIntegrator(xxCoef_);
   diffInteg_[1] = new DiffusionIntegrator(yyCoef_);
   diffInteg_[2] = new DiffusionIntegrator(zzCoef_);
   for (int i=0; i<3; i++)
   {
      errors_[i] = new ParGridFunction(L2FESpace_);
      errEst_[i] = new L2ZienkiewiczZhuEstimator(*diffInteg_[i], *errSol_,
                                                 *L2VFESpace_ ,*HDivFESpace_);
   }
   errors_[3] = new ParGridFunction(L2FESpace_);

   b_    = new ParGridFunction(H1VFESpace_);
   // tmp1_ = new HypreParVector(HCurlFESpace_);

   /*
   Vector xhat(3); xhat = 0.0; xhat[0] = 1.0;
   Vector yhat(3); yhat = 0.0; yhat[1] = 1.0;
   Vector zhat(3); zhat = 0.0; zhat[2] = 1.0;

   VectorConstantCoefficient xHat(xhat);
   VectorConstantCoefficient yHat(yhat);
   VectorConstantCoefficient zHat(zhat);

   E_[0]->ProjectCoefficient(xHat);
   E_[1]->ProjectCoefficient(yHat);
   E_[2]->ProjectCoefficient(zHat);

   {
      ofstream ofs;
      ofs.precision(16);
      ofs.open("E_X.vec"); E_[0]->Print(ofs, 1); ofs.close();
      ofs.open("E_Y.vec"); E_[1]->Print(ofs, 1); ofs.close();
      ofs.open("E_Z.vec"); E_[2]->Print(ofs, 1); ofs.close();
   }
   */
}

StiffnessTensor::~StiffnessTensor()
{
   for (int i=0; i<3; i++)
   {
      delete E_[i];
      delete errEst_[i];
      delete diffInteg_[i];
      delete errors_[i];
   }
   delete errors_[3];

   for (int i=0; i<6; i++)
   {
      delete Chi_[i];
      delete F_[i];
      delete MF_[i];
      delete m_[i];
   }

   delete grad_;

   delete errSol_;
   // delete tmp1_;
   delete b_;
   delete a_;

   delete lambda_;
   delete mu_;

   delete L2FESpace_;
   delete L2VFESpace_;
   delete H1FESpace_;
   delete H1VFESpace_;
   delete HCurlFESpace_;
   delete HCurlVFESpace_;
   delete HDivFESpace_;
}
/*
void
StiffnessTensor::SetVolumeFraction(ParGridFunction & vf)
{
   seqVF_++;
   this->Homogenization::SetVolumeFraction(vf);

   // if ( firstVF_ )
   // {
   lambdaCoef_.SetGridFunction(vf_);
   muCoef_.SetGridFunction(vf_);
   // }

   a_->Update();
   a_->Assemble(0);
   a_->Finalize(0);

   for (int i=0; i<6; i++)
   {
      m_[i]->Update();
      m_[i]->Assemble(0);
      m_[i]->Finalize(0);

      if ( seqVF_ == 1 )
      {
         ostringstream oss;
         oss << "M_" << i << ".mat";
         ofstream ofs(oss.str().c_str());
         m_[i]->SpMat().Print(ofs,1);
         ofs.close();
      }
   }

   if ( false )
   {
      // Test our block mass matrix
      HypreParVector phi(H1VFESpace_);
      HypreParVector GradPhi(HCurlVFESpace_);
      HypreParVector Psi(HCurlVFESpace_);
      HypreParVector DivPsi0(H1VFESpace_);
      HypreParVector DivPsi1(H1VFESpace_);
      HypreParVector DiffPsi(H1VFESpace_);

      phi.Randomize(123);
      a_->Mult(phi, DivPsi0);

      this->TensorGradient(phi, GradPhi);
      this->TensorMassMatrix(GradPhi, Psi);
      this->TensorGradientTranspose(Psi, DivPsi1);

      DiffPsi = DivPsi0;
      DiffPsi -= DivPsi1;

      cout << "Norm of DivPsi0:  " << DivPsi0.Norml2() << endl;
      cout << "Norm of DivPsi1:  " << DivPsi1.Norml2() << endl;
      cout << "Norm of diff:     " << DiffPsi.Norml2() << endl;
   }
}
*/
void
StiffnessTensor::GetHomogenizedProperties(vector<double> & p)
{
   //cout << myid_ << ": Entering GetHomogenizedProperties" << endl;

   vector<double> old_p(21);
   p.resize(21);

   Vector xhat(3); xhat = 0.0; xhat[0] = 1.0;
   Vector yhat(3); yhat = 0.0; yhat[1] = 1.0;
   Vector zhat(3); zhat = 0.0; zhat[2] = 1.0;

   VectorConstantCoefficient xHat(xhat);
   VectorConstantCoefficient yHat(yhat);
   VectorConstantCoefficient zHat(zhat);

   bool newProb = true;
   int max_ref_its = 5;
   int ref_its = 0;
   while ( newProb )
   {
      ref_its++;

      grad_->Assemble();
      grad_->Finalize();

      a_->Assemble(0);
      a_->Finalize(0);

      for (int i=0; i<6; i++)
      {
         m_[i]->Assemble(0);
         m_[i]->Finalize(0);
      }

      E_[0]->ProjectCoefficient(xHat);
      E_[1]->ProjectCoefficient(yHat);
      E_[2]->ProjectCoefficient(zHat);

      HYPRE_Int h1_tsize = H1FESpace_->GetTrueVSize();

      Array<int> ess_tdof_list(0);
      if ( myid_ == 0 )
      {
         ess_tdof_list.SetSize(3);
         ess_tdof_list[0] = 0;
         ess_tdof_list[1] = h1_tsize;
         ess_tdof_list[2] = 2*h1_tsize;
      }

      HypreBoomerAMG * amg = NULL;
      HyprePCG       * pcg = NULL;
      HypreParMatrix   A;
      Vector B, X;

      for (int i=0; i<6; i++)
      {
         if ( myid_ == 0 )
         {
            cout << "Solving Problem " << i+1 << " of 6" << endl;
         }

         // The following magic formulae select two x, y, and z indices
         // for our tensor:
         //
         //  i | ib  ic  xyz
         // ================
         //  0 | 0   0   xx
         //  1 | 1   1   yy
         //  2 | 2   2   zz
         //  3 | 1   2   yz
         //  4 | 0   2   xz
         //  5 | 0   1   xy
         //
         int ib = i - 2 * (i / 3) - 2 * (i / 4) - (i / 5);
         int ic = i - (i / 3) - (i / 4) - 2 * (i / 5);

         //  Compute M * E_ij (using F_[i] as a temporary)
         // cout << myid_ << ": Calling RestrictedTensorMassMatrix, ib=" << ib
         //   << ", E_[" << ic << "], F_["<< i << "]" << endl;
         this->RestrictedTensorMassMatrix(ib, *E_[ic], *F_[i]);
         //  Compute b = Div * M * E_ij
         // cout << myid_ << ": Calling TensorGradietnTranspose" << endl;
         this->TensorGradientTranspose(*F_[i], *b_);

         // Compute b = - Div * M * E_ij
         *b_ *= -1.0; *Chi_[i] = 0.0;
         // cout << myid_ << ": Calling FormLinearSystem" << endl;
         a_->FormLinearSystem(ess_tdof_list, *Chi_[i], *b_, A, X, B);
         // cout << myid_ << ": Back From FormLinearSystem" << endl;
         /*
         if ( i == 0 && seqVF_ == 1 )
         {
            ofstream ofs;
            ofs.precision(16);
            ofs.open("ME_XX.vec"); F_[i]->Print(ofs, 1); ofs.close();
         // b_->Print("b_XX.vec");
         }
         */
         if ( i == 0 )
         {
            if ( false )
            {
               ostringstream oss;
               oss << "A_" << (int)floor(100.0*drand48()) << ".mat";
               A.Print(oss.str().c_str());
            }

            amg = new HypreBoomerAMG(A);
            if ( amg_elast_ )
            {
               amg->SetElasticityOptions(H1VFESpace_);
            }
            else
            {
               amg->SetSystemsOptions(pmesh_->SpaceDimension());
            }
            amg->SetPrintLevel(0);

            pcg = new HyprePCG(A);
            pcg->SetTol(1e-12);
            pcg->SetMaxIter(500);
            pcg->SetPrintLevel(0);
            pcg->SetPreconditioner(*amg);
         }

         // Solve for Chi_ij
         // cout << myid_ << ": Solving" << endl;
         pcg->Mult(B, X);
         // cout << myid_ << ": Calling RecoverFEMSolution" << endl;
         a_->RecoverFEMSolution(X, *b_, *Chi_[i]);
         // cout << myid_ << ": Back From RecoverFEMSolution" << endl;

         // Compute F_ij = E_ij + Grad * Chi_ij
         this->TensorGradient(*Chi_[i], *F_[i]);
         this->RestrictedVectorAdd(ib, *E_[ic], *F_[i]);

         // Compute M * F_ij
         this->TensorMassMatrix(*F_[i], *MF_[i]);
      }
      delete pcg;
      delete amg;

      int k=0;
      if ( ref_its > 1 )
      {
         for (k=0; k<21; k++)
         {
            old_p[k] = p[k];
         }
      }
      k = 0;
      for (int i=0; i<6; i++)
      {
         for (int j=i; j<6; j++)
         {
            // Compute 21 unique Elasticity Tensor components
            // as  F_i^T * M * F_j / domain_volume
            // p[k] = (*F_[i] * *MF_[j]) / vol_;
            p[k] = MF_[j]->operator()(*F_[i]) / vol_;
            k++;
         }
      }

      bool bigChange = true;
      newProb = false;
      if ( ref_its > 1 )
      {
         double norm_p = 0.0;
         double diff_p = 0.0;
         for (k=0; k<21; k++)
         {
            norm_p += pow(p[k], 2.0);
            diff_p += pow(p[k] - old_p[k], 2.0);
         }
         double rel_diff = sqrt(diff_p / norm_p);
         bigChange = rel_diff > tol_;
         cout << "diff/norm ratio " << sqrt(diff_p) << "/" << sqrt(norm_p)
              << " " << rel_diff << endl;
      }

      if ( bigChange && ref_its < max_ref_its )
      {
         cout << "Effective Elasticity Tensor:  " << endl;
         int k = 0;
         for (unsigned int i=0; i<6; i++)
         {
            for (unsigned int j=0; j<i; j++)
            {
               cout << " -----------";
            }
            for (unsigned int j=i; j<6; j++)
            {
               cout << " " << p[k];
               k++;
            }
            cout << endl;
         }
         cout << endl;

         HYPRE_Int  h1_size = H1FESpace_->GetNDofs();

         errSol_->SetData(&(Chi_[0]->GetData())[0*h1_size]);
         // const Vector & errXX = errEst_[0]->GetLocalErrors();
         *errors_[0] = errEst_[0]->GetLocalErrors();

         errSol_->SetData(&(Chi_[1]->GetData())[1*h1_size]);
         // const Vector & errYY = errEst_[1]->GetLocalErrors();
         *errors_[1] = errEst_[1]->GetLocalErrors();

         errSol_->SetData(&(Chi_[2]->GetData())[2*h1_size]);
         // const Vector & errZZ = errEst_[2]->GetLocalErrors();
         *errors_[2] = errEst_[2]->GetLocalErrors();

         for (int i=0; i<errors_[3]->Size(); i++)
         {
            double sum = pow((*errors_[0])[i], 2.0) +
                         pow((*errors_[1])[i], 2.0) +
                         pow((*errors_[2])[i], 2.0);
            (*errors_[3])[i] = pow(sum, 0.5);
         }

         double max_err = errors_[3]->Normlinf();
         pmesh_->RefineByError(*errors_[3], 0.9 * max_err);

         L2FESpace_->Update();
         L2VFESpace_->Update();
         H1FESpace_->Update();
         H1VFESpace_->Update();
         HCurlFESpace_->Update();
         HCurlVFESpace_->Update();
         HDivFESpace_->Update();

         lambda_->Update();
         mu_->Update();

         a_->Update();
         for (int i=0; i<6; i++) { m_[i]->Update(); }
         for (int i=0; i<6; i++) { Chi_[i]->Update(); }
         for (int i=0; i<3; i++) { E_[i]->Update(); }
         for (int i=0; i<6; i++) { F_[i]->Update(); }
         for (int i=0; i<6; i++) { MF_[i]->Update(); }
         grad_->Update();
         //errSol_->SetData(&(Chi_[0]->GetData())[0]);
         //errSol_->Update();
         errSol_->MakeRef(H1FESpace_, &(Chi_[0]->GetData())[0]);
         for (int i=0; i<4; i++) { errors_[i]->Update(); }
         b_->Update();

         newProb = true;
      }
   }

   if ( seqVF_ == 1 )
   {
      ofstream ofs;
      ofs.open("Chi_XX.vec"); Chi_[0]->Print(ofs, 1); ofs.close();
      ofs.open("Chi_YY.vec"); Chi_[1]->Print(ofs, 1); ofs.close();
      ofs.open("Chi_ZZ.vec"); Chi_[2]->Print(ofs, 1); ofs.close();
      ofs.open("Chi_YZ.vec"); Chi_[3]->Print(ofs, 1); ofs.close();
      ofs.open("Chi_XZ.vec"); Chi_[4]->Print(ofs, 1); ofs.close();
      ofs.open("Chi_XY.vec"); Chi_[5]->Print(ofs, 1); ofs.close();
      ofs.open("F_XX.vec"); F_[0]->Print(ofs, 1); ofs.close();
      ofs.open("F_YY.vec"); F_[1]->Print(ofs, 1); ofs.close();
      ofs.open("F_ZZ.vec"); F_[2]->Print(ofs, 1); ofs.close();
      ofs.open("F_YZ.vec"); F_[3]->Print(ofs, 1); ofs.close();
      ofs.open("F_XZ.vec"); F_[4]->Print(ofs, 1); ofs.close();
      ofs.open("F_XY.vec"); F_[5]->Print(ofs, 1); ofs.close();
   }
   //cout << myid_ << ": Leaving GetHomogenizedProperties" << endl;
}
/*
void
StiffnessTensor::GetPropertySensitivities(vector<ParGridFunction> & dp)
{
   dp.resize(21);

   int l = 0;
   for (int j=0; j<6; j++)
   {
      for (int k=j; k<6; k++)
      {
         dp[l].SetSpace(L2FESpace_);
         l++;
      }
   }

   Array<int> l2_vdofs, nd_vdofs;
   ElementTransformation *eltrans;
   IntegrationPoint ip; ip.Init();
   vector<DenseMatrix> elmat(6);

   HYPRE_Int  nd_size = HCurlFESpace_->GetNDofs();
   Vector fjx, fjy, fjz;
   Vector fkx, fky, fkz;
   Vector mfjx, mfjy, mfjz;

   const IntegrationRule * ir = &IntRules.Get(geom_, irOrder_);

   for (int i=0; i<L2FESpace_->GetNE(); i++)
   {
      L2FESpace_->GetElementVDofs(i, l2_vdofs);
      HCurlFESpace_->GetElementVDofs(i, nd_vdofs);

      eltrans = L2FESpace_->GetElementTransformation(i);
      double dLambda = lambdaCoef_.GetSensitivity(*eltrans, ip);
      double dMu     = muCoef_.GetSensitivity(*eltrans, ip);

      DenseTensor mat(3,3,6);
      mat = 0.0;

      mat(0,0,0) = dLambda + 2.0 * dMu;
      mat(1,1,0) = dMu;
      mat(2,2,0) = dMu;

      mat(0,0,1) = dMu;
      mat(1,1,1) = dLambda + 2.0 * dMu;
      mat(2,2,1) = dMu;

      mat(0,0,2) = dMu;
      mat(1,1,2) = dMu;
      mat(2,2,2) = dLambda + 2.0 * dMu;

      mat(1,2,3) = dLambda;
      mat(2,1,3) = dMu;

      mat(0,2,4) = dLambda;
      mat(2,0,4) = dMu;

      mat(0,1,5) = dLambda;
      mat(1,0,5) = dMu;

      if ( i == 0 && false )
      {
         mat(0).Print(cout); cout << endl;
         mat(1).Print(cout); cout << endl;
         mat(2).Print(cout); cout << endl;
         mat(3).Print(cout); cout << endl;
         mat(4).Print(cout); cout << endl;
         mat(5).Print(cout); cout << endl;
      }

      MatrixConstantCoefficient xxCoef(mat(0));
      MatrixConstantCoefficient yyCoef(mat(1));
      MatrixConstantCoefficient zzCoef(mat(2));
      MatrixConstantCoefficient yzCoef(mat(3));
      MatrixConstantCoefficient xzCoef(mat(4));
      MatrixConstantCoefficient xyCoef(mat(5));

      VectorFEMassIntegrator * m[6];
      m[0] = new VectorFEMassIntegrator(xxCoef);
      m[1] = new VectorFEMassIntegrator(yyCoef);
      m[2] = new VectorFEMassIntegrator(zzCoef);
      m[3] = new VectorFEMassIntegrator(yzCoef);
      m[4] = new VectorFEMassIntegrator(xzCoef);
      m[5] = new VectorFEMassIntegrator(xyCoef);

      const FiniteElement &fe = *HCurlFESpace_->GetFE(i);

      for (int j=0; j<6; j++)
      {
         m[j]->SetIntRule(ir);
         m[j]->AssembleElementMatrix( fe, *eltrans, elmat[j]);
      }

      l = 0;
      for (int j=0; j<6; j++)
      {
         Vector Fjx(&(F_[j]->GetData())[0*nd_size], nd_size);
         Vector Fjy(&(F_[j]->GetData())[1*nd_size], nd_size);
         Vector Fjz(&(F_[j]->GetData())[2*nd_size], nd_size);

         Fjx.GetSubVector(nd_vdofs, fjx);
         Fjy.GetSubVector(nd_vdofs, fjy);
         Fjz.GetSubVector(nd_vdofs, fjz);

         mfjx.SetSize(fjx.Size());
         mfjy.SetSize(fjx.Size());
         mfjz.SetSize(fjx.Size());

         if ( true )
         {
            elmat[0].Mult(fjx, mfjx);
            elmat[5].AddMult(fjy, mfjx);
            elmat[4].AddMult(fjz, mfjx);

            elmat[5].MultTranspose(fjx, mfjy);
            elmat[1].AddMult(fjy, mfjy);
            elmat[3].AddMult(fjz, mfjy);

            elmat[4].MultTranspose(fjx, mfjz);
            elmat[3].AddMultTranspose(fjy, mfjz);
            elmat[2].AddMult(fjz, mfjz);
         }
         else
         {
            elmat[0].Mult(fjx, mfjx);
            elmat[5].AddMultTranspose(fjy, mfjx);
            elmat[4].AddMultTranspose(fjz, mfjx);

            elmat[5].Mult(fjx, mfjy);
            elmat[1].AddMult(fjy, mfjy);
            elmat[3].AddMultTranspose(fjz, mfjy);

            elmat[4].Mult(fjx, mfjz);
            elmat[3].AddMult(fjy, mfjz);
            elmat[2].AddMult(fjz, mfjz);
         }

         for (int k=j; k<6; k++)
         {
            Vector Fkx(&(F_[k]->GetData())[0*nd_size], nd_size);
            Vector Fky(&(F_[k]->GetData())[1*nd_size], nd_size);
            Vector Fkz(&(F_[k]->GetData())[2*nd_size], nd_size);

            Fkx.GetSubVector(nd_vdofs, fkx);
            Fky.GetSubVector(nd_vdofs, fky);
            Fkz.GetSubVector(nd_vdofs, fkz);

            dp[l][l2_vdofs[0]] = (fkx * mfjx + fky * mfjy + fkz * mfjz) / vol_;

            l++;
         }
      }
      for (int j=0; j<6; j++)
      {
         delete m[j];
      }
   }
}
*/
void
StiffnessTensor::TensorGradient(const Vector & x, Vector & y)
{
   //cout << myid_ << ": Entering TensorGradient" << endl;
   HYPRE_Int  h1_size = H1FESpace_->GetNDofs();
   HYPRE_Int  nd_size = HCurlFESpace_->GetNDofs();

   Vector xx(&(x.GetData())[0*h1_size], h1_size);
   Vector xy(&(x.GetData())[1*h1_size], h1_size);
   Vector xz(&(x.GetData())[2*h1_size], h1_size);

   Vector yx(&(y.GetData())[0*nd_size], nd_size);
   Vector yy(&(y.GetData())[1*nd_size], nd_size);
   Vector yz(&(y.GetData())[2*nd_size], nd_size);

   grad_->Mult(xx, yx);
   grad_->Mult(xy, yy);
   grad_->Mult(xz, yz);
   //cout << myid_ << ": Leaving TensorGradient" << endl;
}

void
StiffnessTensor::TensorGradientTranspose(const Vector & x, Vector & y)
{
   //cout << myid_ << ": Entering TensorGradientTranspose" << endl;
   HYPRE_Int  h1_size = H1FESpace_->GetNDofs();
   HYPRE_Int  nd_size = HCurlFESpace_->GetNDofs();

   Vector xx(&(x.GetData())[0*nd_size], nd_size);
   Vector xy(&(x.GetData())[1*nd_size], nd_size);
   Vector xz(&(x.GetData())[2*nd_size], nd_size);

   Vector yx(&(y.GetData())[0*h1_size], h1_size);
   Vector yy(&(y.GetData())[1*h1_size], h1_size);
   Vector yz(&(y.GetData())[2*h1_size], h1_size);

   grad_->MultTranspose(xx, yx);
   grad_->MultTranspose(xy, yy);
   grad_->MultTranspose(xz, yz);
   //cout << myid_ << ": Leaving TensorGradientTranspose" << endl;
}

void
StiffnessTensor::TensorMassMatrix(const Vector & x, Vector & y)
{
   //cout << myid_ << ": Entering TensorMassMatrix" << endl;
   HYPRE_Int  nd_size = HCurlFESpace_->GetNDofs();

   Vector xx(&(x.GetData())[0*nd_size], nd_size);
   Vector xy(&(x.GetData())[1*nd_size], nd_size);
   Vector xz(&(x.GetData())[2*nd_size], nd_size);

   Vector yx(&(y.GetData())[0*nd_size], nd_size);
   Vector yy(&(y.GetData())[1*nd_size], nd_size);
   Vector yz(&(y.GetData())[2*nd_size], nd_size);

   // Perform Block Matrix Vector Multiply
   /*
       /yx\   / m0  m5  m4 \ /xx\
       |yy| = | m5T m1  m3 | |xy|
       \yz/   \ m4T m3T m2 / \xz/
   */

   m_[0]->Mult(xx, yx);
   m_[1]->Mult(xy, yy);
   m_[2]->Mult(xz, yz);
   m_[3]->AddMult(xz, yy);
   m_[4]->AddMult(xz, yx);
   m_[5]->AddMult(xy, yx);
   m_[3]->AddMultTranspose(xy, yz);
   m_[4]->AddMultTranspose(xx, yz);
   m_[5]->AddMultTranspose(xx, yy);
   //cout << myid_ << ": Leaving TensorMassMatrix" << endl;
}

void
StiffnessTensor::RestrictedTensorMassMatrix(int r, const Vector & x, Vector & y)
{
   // cout << myid_ << ": Entering RestrictedTensorMassMatrix" << endl;

   // cout << "x size: " << x.Size() << endl;
   // cout << "y size: " << y.Size() << endl;

   HYPRE_Int  nd_size = HCurlFESpace_->GetNDofs();

   // cout << "nd_size: " << nd_size << endl;

   Vector yx(&(y.GetData())[0*nd_size], nd_size);
   Vector yy(&(y.GetData())[1*nd_size], nd_size);
   Vector yz(&(y.GetData())[2*nd_size], nd_size);

   // Perform Block Matrix Vector Multiply
   /*
       /yx\   / m0  m5  m4 \ /xx\
       |yy| = | m5T m1  m3 | |xy|
       \yz/   \ m4T m3T m2 / \xz/

             /xx\   /x\    /0\    /0\
       Where |xy| = |0| or |x| or |0| for r = 0, 1, or 2 respectively.
             \xz/   \0/    \0/    \x/
   */
   for ( int i=0; i<6; i++)
   {
      if ( m_[i] == NULL)
      {
         cout << "m_[" << i << "] is NULL" << endl;
      }
      // else
      //  {
      //   cout << "m[" << i << "] is " << m_[i]->Height() << " x " << m_[i]->Width() << endl;
      //  }
   }

   switch (r)
   {
      case 0:
         m_[0]->Mult(x, yx);
         m_[5]->MultTranspose(x, yy);
         m_[4]->MultTranspose(x, yz);
         break;
      case 1:
         m_[5]->Mult(x, yx);
         m_[1]->Mult(x, yy);
         m_[3]->MultTranspose(x, yz);
         break;
      case 2:
         m_[4]->Mult(x, yx);
         m_[3]->Mult(x, yy);
         m_[2]->Mult(x, yz);
         break;
   }
   //cout << myid_ << ": Leaving RestrictedTensorMassMatrix" << endl;
}

void
StiffnessTensor::RestrictedVectorAdd(int r, const Vector & x, Vector & y)
{
   //cout << myid_ << ": Entering RestrictedVectorAdd" << endl;
   HYPRE_Int  nd_size = HCurlFESpace_->GetNDofs();

   Vector yx(&(y.GetData())[0*nd_size], nd_size);
   Vector yy(&(y.GetData())[1*nd_size], nd_size);
   Vector yz(&(y.GetData())[2*nd_size], nd_size);

   // Perform Block Matrix Vector Multiply
   /*
       /yx\    /xx\
       |yy| += |xy|
       \yz/    \xz/

             /xx\   /x\    /0\    /0\
       Where |xy| = |0| or |x| or |0| for r = 0, 1, or 2 respectively.
             \xz/   \0/    \0/    \x/
   */

   switch (r)
   {
      case 0:
         yx += x;
         break;
      case 1:
         yy += x;
         break;
      case 2:
         yz += x;
         break;
   }
   //cout << myid_ << ": Leaving RestrictedVectorAdd" << endl;
}

void
StiffnessTensor::InitializeGLVis(VisData & vd)
{
   vd_ = &vd;

   for (int i=0; i<8; i++)
   {
      socks_[i].precision(8);
   }
   for (int i=0; i<4; i++)
   {
      err_socks_[i].precision(8);
   }
}

void
StiffnessTensor::DisplayToGLVis()
{
   if (vd_ == NULL)
   {
      MFEM_WARNING("DisplayToGLVis being called before InitializeGLVis!");
      return;
   }

   lambda_->ProjectCoefficient(*lambdaCoef_);
   mu_->ProjectCoefficient(*muCoef_);

   VisualizeField(socks_[0], *lambda_, "Lambda", *vd_);
   vd_->IncrementWindow();
   VisualizeField(socks_[1], *mu_, "Mu", *vd_);
   vd_->IncrementWindow();

   VisualizeField(socks_[2], *Chi_[0], "Chi XX", *vd_);
   vd_->IncrementWindow();
   VisualizeField(socks_[3], *Chi_[1], "Chi YY", *vd_);
   vd_->IncrementWindow();
   VisualizeField(socks_[4], *Chi_[2], "Chi ZZ", *vd_);
   vd_->IncrementWindow();
   VisualizeField(socks_[5], *Chi_[3], "Chi YZ", *vd_);
   vd_->IncrementWindow();
   VisualizeField(socks_[6], *Chi_[4], "Chi XZ", *vd_);
   vd_->IncrementWindow();
   VisualizeField(socks_[7], *Chi_[5], "Chi XY", *vd_);
   vd_->IncrementWindow();

   VisualizeField(err_socks_[0], *errors_[0], "Error XX", *vd_);
   vd_->IncrementWindow();
   VisualizeField(err_socks_[1], *errors_[1], "Error YY", *vd_);
   vd_->IncrementWindow();
   VisualizeField(err_socks_[2], *errors_[2], "Error ZZ", *vd_);
   vd_->IncrementWindow();
   VisualizeField(err_socks_[3], *errors_[3], "Combined Error Estimate", *vd_);
   vd_->IncrementWindow();
}

void
StiffnessTensor::WriteVisItFields(const string & prefix,
                                  const string & label)
{
   lambda_->ProjectCoefficient(*lambdaCoef_);
   mu_->ProjectCoefficient(*muCoef_);

   VisItDataCollection visit_dc(label.c_str(), pmesh_);
   visit_dc.SetPrefixPath(prefix.c_str());
   visit_dc.RegisterField("Lambda", lambda_);
   visit_dc.RegisterField("Mu", mu_);
   visit_dc.RegisterField("Chi XX", Chi_[0]);
   visit_dc.RegisterField("Chi YY", Chi_[1]);
   visit_dc.RegisterField("Chi ZZ", Chi_[2]);
   visit_dc.RegisterField("Chi YZ", Chi_[3]);
   visit_dc.RegisterField("Chi XZ", Chi_[4]);
   visit_dc.RegisterField("Chi XY", Chi_[5]);
   visit_dc.Save();
}

StiffnessTensor::DiagElasticityCoef::DiagElasticityCoef(
   Coefficient & lambda,
   Coefficient & mu,
   int axis)
   : MatrixCoefficient(3),
     axis_(axis),
     lambda_(&lambda),
     mu_(&mu)
{
}

void
StiffnessTensor::DiagElasticityCoef::Eval(DenseMatrix &K,
                                          ElementTransformation &T,
                                          const IntegrationPoint &ip)
{
   int axis1 = (axis_+1)%3;
   int axis2 = (axis_+2)%3;

   double mu = mu_->Eval(T, ip);

   K.SetSize(3); K = 0.0;
   K(axis_, axis_) = 2.0 * mu + lambda_->Eval(T, ip);
   K(axis1, axis1) = mu;
   K(axis2, axis2) = mu;
}

StiffnessTensor::OffDiagElasticityCoef::OffDiagElasticityCoef(
   Coefficient & lambda,
   Coefficient & mu,
   int axis0, int axis1)
   : MatrixCoefficient(3),
     axis0_(axis0),
     axis1_(axis1),
     lambda_(&lambda),
     mu_(&mu)
{
}

void
StiffnessTensor::OffDiagElasticityCoef::Eval(DenseMatrix &K,
                                             ElementTransformation &T,
                                             const IntegrationPoint &ip)
{
   K.SetSize(3); K = 0.0;
   K(axis0_, axis1_) = lambda_->Eval(T, ip);
   K(axis1_, axis0_) = mu_->Eval(T, ip);
}

ParDiscreteVectorProductOperator::ParDiscreteVectorProductOperator(
   ParFiniteElementSpace *dfes,
   ParFiniteElementSpace *rfes,
   const Vector & v)
   : ParDiscreteInterpolationOperator(dfes, rfes),
     vCoef_(v)
{
   this->AddDomainInterpolator(new VectorScalarProductInterpolator(vCoef_));
}

ParDiscreteVectorCrossProductOperator::ParDiscreteVectorCrossProductOperator(
   ParFiniteElementSpace *dfes,
   ParFiniteElementSpace *rfes,
   const Vector & v)
   : ParDiscreteInterpolationOperator(dfes, rfes),
     vCoef_(v)
{
   this->AddDomainInterpolator(new VectorCrossProductInterpolator(vCoef_));
}

MaxwellBlochWaveEquation::MaxwellBlochWaveEquation(ParMesh & pmesh,
                                                   int order)
   : myid_(0),
     hcurl_loc_size_(-1),
     hdiv_loc_size_(-1),
     nev_(-1),
     // newAlpha_(true),
     newBeta_(true),
     newZeta_(true),
     newOmega_(true),
     newMCoef_(true),
     newKCoef_(true),
     pmesh_(&pmesh),
     H1FESpace_(NULL),
     HCurlFESpace_(NULL),
     HDivFESpace_(NULL),
     // L2FESpace_(NULL),
     // bravais_(NULL),
     // fourierHCurl_(NULL),
     // alpha_a_(0.0),
     // alpha_i_(90.0),
     atol_(1.0e-6),
     beta_(0.0),
     // omega_(-1.0),
     mCoef_(NULL),
     kCoef_(NULL),
     /*     cosCoef_(NULL),
       sinCoef_(NULL),*/
     A_(NULL),
     M_(NULL),
     C_(NULL),
     blkHCurl_(NULL),
     blkHDiv_(NULL),
     M1_(NULL),
     M2_(NULL),
     S1_(NULL),
     T1_(NULL),
     T12_(NULL),
     Z12_(NULL),
     DKZ_(NULL),
     // DKZT_(NULL),
     T1Inv_(NULL),
     Curl_(NULL),
     Zeta_(NULL),
     BDP_(NULL),
     Precond_(NULL),
     SubSpaceProj_(NULL),
     // tmpVecA_(NULL),
     // tmpVecB_(NULL),
     vecs_(NULL),
     vec0_(NULL),
     lobpcg_(NULL),
     ame_(NULL),
     energy_(NULL)
{
   // Initialize MPI variables
   comm_ = pmesh.GetComm();
   MPI_Comm_rank(comm_, &myid_);

   if ( myid_ == 0 )
   {
      cout << "Constructing MaxwellBlochWaveEquation" << endl;
   }

   int dim = pmesh.Dimension();

   zeta_.SetSize(dim);

   H1FESpace_    = new H1_ParFESpace(&pmesh,order,dim);
   HCurlFESpace_ = new ND_ParFESpace(&pmesh,order,dim);
   HDivFESpace_  = new RT_ParFESpace(&pmesh,order,dim);
   // L2FESpace_    = new L2_ParFESpace(&pmesh,0,dim);
}

MaxwellBlochWaveEquation::~MaxwellBlochWaveEquation()
{
   delete lobpcg_;
   // delete minres_;
   // delete gmres_;
   // delete B_;

   if ( vecs_ != NULL )
   {
      for (int i=0; i<nev_; i++) { delete vecs_[i]; }
      delete [] vecs_;
   }

   // delete tmpVecA_;
   // delete tmpVecB_;

   delete SubSpaceProj_;
   delete Precond_;

   delete blkHCurl_;
   delete blkHDiv_;
   delete A_;
   delete M_;
   delete C_;
   delete BDP_;
   delete T1Inv_;

   delete M1_;
   delete M2_;
   delete S1_;
   delete T1_;
   delete T12_;
   delete Z12_;
   delete DKZ_;
   // delete DKZT_;
   delete Curl_;
   delete Zeta_;

   // delete fourierHCurl_;

   delete H1FESpace_;
   delete HCurlFESpace_;
   delete HDivFESpace_;
   // delete L2FESpace_;
   /*
   for (int i=0; i<3; i++)
   {
      delete AvgHCurl_coskx_[i];
      delete AvgHCurl_sinkx_[i];

      delete AvgHDiv_coskx_[i];
      delete AvgHDiv_sinkx_[i];

      delete AvgHCurl_eps_coskx_[i];
      delete AvgHCurl_eps_sinkx_[i];
   }
   */
}

void
MaxwellBlochWaveEquation::SetKappa(const Vector & kappa)
{
   kappa_ = kappa;
   beta_  = kappa.Norml2();  newBeta_ = true;
   zeta_  = kappa;           newZeta_ = true;
   if ( fabs(beta_) > 0.0 )
   {
      zeta_ /= beta_;
   }
}

void
MaxwellBlochWaveEquation::SetBeta(double beta)
{
   beta_ = beta; newBeta_ = true;
}

void
MaxwellBlochWaveEquation::SetZeta(const Vector & zeta)
{
   zeta_ = zeta; newZeta_ = true;
}

void
MaxwellBlochWaveEquation::SetAbsoluteTolerance(double atol)
{
   atol_ = atol;
}

void
MaxwellBlochWaveEquation::SetNumEigs(int nev)
{
   nev_ = nev;
}

void
MaxwellBlochWaveEquation::SetMassCoef(Coefficient & m)
{
   mCoef_ = &m; newMCoef_ = true;
}

void
MaxwellBlochWaveEquation::SetStiffnessCoef(Coefficient & k)
{
   kCoef_ = &k; newKCoef_ = true;
}

void
MaxwellBlochWaveEquation::Setup()
{
   if ( hcurl_loc_size_ == -1 )
   {
      hcurl_loc_size_ = HCurlFESpace_->TrueVSize();
      hdiv_loc_size_  = HDivFESpace_->TrueVSize();

      if ( myid_ == 0 )
      {
         cout << "local sizes " << hcurl_loc_size_ << " and " << hdiv_loc_size_ << endl;
      }

      block_offsets_.SetSize(3);
      block_offsets_[0] = 0;
      block_offsets_[1] = HCurlFESpace_->GetVSize();
      block_offsets_[2] = HCurlFESpace_->GetVSize();
      block_offsets_.PartialSum();

      block_trueOffsets_.SetSize(3);
      block_trueOffsets_[0] = 0;
      block_trueOffsets_[1] = HCurlFESpace_->TrueVSize();
      block_trueOffsets_[2] = HCurlFESpace_->TrueVSize();
      block_trueOffsets_.PartialSum();

      block_trueOffsets2_.SetSize(3);
      block_trueOffsets2_[0] = 0;
      block_trueOffsets2_[1] = HDivFESpace_->TrueVSize();
      block_trueOffsets2_[2] = HDivFESpace_->TrueVSize();
      block_trueOffsets2_.PartialSum();

      tdof_offsets_.SetSize(HCurlFESpace_->GetNRanks()+1);
      HYPRE_Int * hcurl_tdof_offsets = HCurlFESpace_->GetTrueDofOffsets();
      for (int i=0; i<tdof_offsets_.Size(); i++)
      {
         tdof_offsets_[i] = 2 * hcurl_tdof_offsets[i];
      }

      blkHCurl_ = new BlockVector(block_trueOffsets_);
      blkHDiv_  = new BlockVector(block_trueOffsets2_);
   }

   if ( newKCoef_ )
   {
      if ( myid_ == 0 ) { cout << "Building M2(k)" << endl; }
      ParBilinearForm m2(HDivFESpace_);
      m2.AddDomainIntegrator(new VectorFEMassIntegrator(*kCoef_));
      m2.Assemble();
      m2.Finalize();
      delete M2_;
      M2_ = m2.ParallelAssemble();
   }

   if ( newZeta_ )
   {
      if ( myid_ == 0 ) { cout << "Building zeta cross operator" << endl; }
      delete Zeta_;
      Zeta_ = new ParDiscreteVectorCrossProductOperator(HCurlFESpace_,
                                                        HDivFESpace_,zeta_);
      Zeta_->Assemble();
      Zeta_->Finalize();
      Z12_ = Zeta_->ParallelAssemble();
   }

   if ( Curl_ == NULL )
   {
      if ( myid_ == 0 ) { cout << "Building Curl operator" << endl; }
      Curl_ = new ParDiscreteCurlOperator(HCurlFESpace_,HDivFESpace_);
      Curl_->Assemble();
      Curl_->Finalize();
      T12_ = Curl_->ParallelAssemble();
   }

   if ( newZeta_ || newBeta_ || newKCoef_ )
   {
      if ( myid_ == 0 ) { cout << "Forming CMC" << endl; }
      HypreParMatrix * CMC = RAP(M2_,T12_);

      if ( S1_ ) { delete S1_; }

      if ( fabs(beta_) > 0.0 )
      {
         if ( myid_ == 0 ) { cout << "Forming 2nd order operators" << endl; }
         HypreParMatrix * ZMZ = RAP(M2_, Z12_);
         HypreParMatrix * CMZ = RAP(T12_, M2_, Z12_);
         HypreParMatrix * ZMC = RAP(Z12_, M2_, T12_);

         if ( myid_ == 0 ) { cout << "Forming DKZ" << endl; }
         *ZMC *= -1.0;
         if ( DKZ_ ) { delete DKZ_; }
         DKZ_ = ParAdd(CMZ,ZMC);
         delete CMZ;
         delete ZMC;

         if ( myid_ == 0 ) { cout << "Scaling ZMZ" << endl; }
         // *ZMZ *= beta_*beta_/(a_*a_);
         *ZMZ *= beta_*beta_;

         if ( myid_ == 0 ) { cout << "Forming S1" << endl; }
         S1_ = ParAdd(CMC,ZMZ);
         if ( myid_ == 0 ) { cout << "Done forming S1" << endl; }
         delete CMC;
         delete ZMZ;
         if ( myid_ == 0 ) { cout << "Done forming 2nd order operators" << endl; }
      }
      else
      {
         S1_ = CMC;
      }
   }

   if ( newMCoef_ )
   {
      if ( myid_ == 0 ) { cout << "Building M1(m)" << endl; }
      ParBilinearForm m1(HCurlFESpace_);
      m1.AddDomainIntegrator(new VectorFEMassIntegrator(*mCoef_));
      m1.Assemble();
      m1.Finalize();
      delete M1_;
      M1_ = m1.ParallelAssemble();
   }

   if ( newZeta_ || newBeta_ || newKCoef_ )
   {
      if ( A_ == NULL )
      {
         if ( myid_ == 0 ) { cout << "Building Block A" << endl; }
         A_ = new BlockOperator(block_trueOffsets_);
      }
      A_->SetDiagonalBlock(0,S1_);
      A_->SetDiagonalBlock(1,S1_);
      if ( fabs(beta_) > 0.0 )
      {
         // A_->SetBlock(0,1,DKZ_, beta_*M_PI/(180.0*a_));
         // A_->SetBlock(1,0,DKZ_,-beta_*M_PI/(180.0*a_));
         // A_->SetBlock(0,1,DKZ_, beta_/a_);
         // A_->SetBlock(1,0,DKZ_,-beta_/a_);
         A_->SetBlock(0,1,DKZ_, beta_);
         A_->SetBlock(1,0,DKZ_,-beta_);
      }
      A_->owns_blocks = 0;
   }

   if ( newMCoef_ )
   {
      if ( M_ == NULL )
      {
         if ( myid_ == 0 ) { cout << "Building Block M" << endl; }
         M_ = new BlockOperator(block_trueOffsets_);
      }
      M_->SetDiagonalBlock(0,M1_);
      M_->SetDiagonalBlock(1,M1_);
      M_->owns_blocks = 0;
   }

   if ( newZeta_ || newBeta_ )
   {
      if ( C_ == NULL )
      {
         if ( myid_ == 0 ) { cout << "Building Block C" << endl; }
         C_ = new BlockOperator(block_trueOffsets2_, block_trueOffsets_);
      }
      C_->SetDiagonalBlock(0, T12_);
      C_->SetDiagonalBlock(1, T12_);
      if ( fabs(beta_) > 0.0 )
      {
         // C_->SetBlock(0,1,Zeta_->ParallelAssemble(), beta_*M_PI/(180.0*a_));
         // C_->SetBlock(1,0,Zeta_->ParallelAssemble(),-beta_*M_PI/(180.0*a_));
         // C_->SetBlock(0,1,Zeta_->ParallelAssemble(), beta_/a_);
         // C_->SetBlock(1,0,Zeta_->ParallelAssemble(),-beta_/a_);
         C_->SetBlock(0,1,Z12_, beta_);
         C_->SetBlock(1,0,Z12_,-beta_);
      }
      C_->owns_blocks = 0;
   }

   if ( newZeta_ || newBeta_ || newKCoef_ )
   {
      if ( myid_ == 0 ) { cout << "Building T1Inv" << endl; }
      delete T1Inv_;
      if ( fabs(beta_*180.0) < M_PI )
      {
         cout << "HypreAMS::SetSingularProblem()" << endl;
         T1Inv_ = new HypreAMS(*S1_,HCurlFESpace_);
         T1Inv_->SetSingularProblem();
      }
      else
      {
         T1Inv_ = new HypreAMS(*S1_,HCurlFESpace_);
         // T1Inv_->SetSingularProblem();
      }

      if ( true || fabs(beta_) > 0.0 )
      {
         if ( myid_ == 0 ) { cout << "Building BDP" << endl; }
         delete BDP_;
         BDP_ = new BlockDiagonalPreconditioner(block_trueOffsets_);
         BDP_->SetDiagonalBlock(0,T1Inv_);
         BDP_->SetDiagonalBlock(1,T1Inv_);
         BDP_->owns_blocks = 0;
      }
   }

   if ( ( newZeta_ || newBeta_ || newMCoef_ || newKCoef_ ) && nev_ > 0 )
   {
      if ( fabs(beta_) > 0.0 )
      {
         delete SubSpaceProj_;
         if ( myid_ == 0 ) { cout << "Building Subspace Projector" << endl; }
         MaxwellBlochWaveProjector * mbwProj =
            new MaxwellBlochWaveProjector(*HCurlFESpace_,
                                          *H1FESpace_,
                                          *M_,beta_,zeta_);
         mbwProj->Setup();

         SubSpaceProj_ = mbwProj;

         if ( myid_ == 0 ) { cout << "Building Preconditioner" << endl; }
         delete Precond_;
         Precond_ = new MaxwellBlochWavePrecond(*HCurlFESpace_,*BDP_,
                                                *SubSpaceProj_,0.5);
         Precond_->SetOperator(*A_);

         if ( myid_ == 0 ) { cout << "Building HypreLOBPCG solver" << endl; }
         delete lobpcg_;
         lobpcg_ = new HypreLOBPCG(comm_);

         lobpcg_->SetNumModes(nev_);
         lobpcg_->SetPreconditioner(*this->GetPreconditioner());
         lobpcg_->SetMaxIter(2000);
         lobpcg_->SetTol(atol_);
         lobpcg_->SetPrecondUsageMode(1);
         lobpcg_->SetPrintLevel(1);

         // Set the matrices which define the linear system
         lobpcg_->SetMassMatrix(*this->GetMOperator());
         lobpcg_->SetOperator(*this->GetAOperator());
         lobpcg_->SetSubSpaceProjector(*this->GetSubSpaceProjector());

         if ( false && vecs_ != NULL )
         {
            cout << "HypreLOBPCG::SetInitialVectors()" << endl;
            int n = 1 + (int)ceil(nev_/4);
            for (int i=nev_-n; i<nev_; i++) { vecs_[i]->Randomize(123); }
            lobpcg_->SetInitialVectors(nev_, vecs_);
         }
      }
      else
      {
         if ( myid_ == 0 ) { cout << "Building HypreAME solver" << endl; }
         delete ame_;
         ame_ = new HypreAME(comm_);
         ame_->SetNumModes(nev_/2);
         ame_->SetPreconditioner(*T1Inv_);
         ame_->SetMaxIter(2000);
         ame_->SetTol(atol_);
         ame_->SetRelTol(1e-8);
         ame_->SetPrintLevel(1);

         // Set the matrices which define the linear system
         ame_->SetMassMatrix(*M1_);
         ame_->SetOperator(*S1_);

         if ( vec0_ == NULL )
         {
            vec0_ = new HypreParVector(*M1_);
         }
         *vec0_ = 0.0;
      }
   }

   Vector xHat(3), yHat(3), zHat(3);
   xHat = yHat = zHat = 0.0;
   xHat(0) = 1.0; yHat(1) = 1.0; zHat(2) = 1.0;

   newZeta_  = false;
   newBeta_  = false;
   newOmega_ = false;
   newMCoef_ = false;
   newKCoef_ = false;

   if ( myid_ == 0 ) { cout << "Leaving Setup" << endl; }
}

void
MaxwellBlochWaveEquation::SetInitialVectors(int num_vecs,
                                            HypreParVector ** vecs)
{
   if ( lobpcg_ )
   {
      lobpcg_->SetInitialVectors(num_vecs, vecs);
   }
}

void
MaxwellBlochWaveEquation::Solve()
{
   if ( nev_ > 0 )
   {
      if ( fabs(beta_) > 0.0 )
      {
         lobpcg_->Solve();
         vecs_ = lobpcg_->StealEigenvectors();
         cout << "lobpcg done" << endl;
      }
      else
      {
         ame_->Solve();
         //vecs_ = ame_->StealEigenvectors();
         cout << "ame done" << endl;
      }
   }
   cout << "Solve done" << endl;
}

void
MaxwellBlochWaveEquation::GetEigenvalues(vector<double> & eigenvalues)
{
   if ( lobpcg_ )
   {
      Array<double> eigs;
      lobpcg_->GetEigenvalues(eigs);
      eigenvalues.resize(eigs.Size());
      for (int i=0; i<eigs.Size(); i++)
      {
         eigenvalues[i] = eigs[i];
      }
   }
   else if ( ame_ )
   {
      Array<double> eigs0;
      ame_->GetEigenvalues(eigs0);
      eigenvalues.resize(2*eigs0.Size());
      for (int i=0; i<eigs0.Size(); i++)
      {
         eigenvalues[2*i+0] = eigs0[i];
         eigenvalues[2*i+1] = eigs0[i];
      }
   }
}

void
MaxwellBlochWaveEquation::GetEigenvalues(int nev, const Vector & kappa,
                                         vector<HypreParVector*> & init_vecs,
                                         vector<double> & eigenvalues)
{
   this->SetNumEigs(nev);
   this->SetKappa(kappa);
   this->Setup();
   this->SetInitialVectors(nev, &init_vecs[0]);

   StopWatch chrono;
   chrono.Clear();
   chrono.Start();
   this->Solve();
   chrono.Stop();
   solve_times_.push_back(chrono.RealTime());

   this->GetEigenvalues(eigenvalues);

   cout << "Leaving LOBPCG block of GetEigenvalues" << endl << flush;

}

void
MaxwellBlochWaveEquation::GetEigenvector(unsigned int i,
                                         HypreParVector & Er,
                                         HypreParVector & Ei,
                                         HypreParVector & Br,
                                         HypreParVector & Bi)
{
   this->GetEigenvectorE(i, Er, Ei);
   this->GetEigenvectorB(i, Br, Bi);
}

void
MaxwellBlochWaveEquation::GetEigenvectorE(unsigned int i,
                                          HypreParVector & Er,
                                          HypreParVector & Ei)
{
   double * data = NULL;
   if ( vecs_ != NULL )
   {
      data = (double*)*vecs_[i];
   }
   else
   {
      if ( lobpcg_ )
      {
         data = (double*)lobpcg_->GetEigenvector(i);
      }
      else if ( ame_ )
      {
         if ( i%2 == 0 )
         {
            data = (double*)ame_->GetEigenvector(i/2);
         }
         else
         {
            data = (double*)ame_->GetEigenvector((i-1)/2);
         }
      }
   }

   if ( lobpcg_ )
   {
      Er.SetData(&data[0]);
      Ei.SetData(&data[hcurl_loc_size_]);
   }
   else if ( ame_ )
   {
      if ( i%2 == 0 )
      {
         Er.SetData(&data[0]);
         Ei.SetData(vec0_->GetData());
      }
      else
      {
         Er.SetData(vec0_->GetData());
         Ei.SetData(&data[0]);
      }
   }
}

void
MaxwellBlochWaveEquation::GetEigenvectorB(unsigned int i,
                                          HypreParVector & Br,
                                          HypreParVector & Bi)
{
   vector<double> eigenvalues;
   this->GetEigenvalues(eigenvalues);

   if ( lobpcg_ )
   {
      if ( vecs_ != NULL )
      {
         C_->Mult(*vecs_[i], *blkHDiv_);
      }
      else
      {
         C_->Mult(lobpcg_->GetEigenvector(i), *blkHDiv_);
      }
   }
   else if ( ame_ )
   {
      if ( i%2 == 0 )
      {
         blkHDiv_->GetBlock(1) = 0.0;
         Curl_->Mult(ame_->GetEigenvector(i/2),blkHDiv_->GetBlock(0));
      }
      else
      {
         Curl_->Mult(ame_->GetEigenvector((i-1)/2),blkHDiv_->GetBlock(1));
         blkHDiv_->GetBlock(0) = 0.0;
      }
   }

   if ( eigenvalues[i] != 0.0 ) { *blkHDiv_ /= sqrt(fabs(eigenvalues[i])); }

   double * data = (double*)*blkHDiv_;
   Bi.SetData(&data[0]);
   Br.SetData(&data[hdiv_loc_size_]); Br *= -1.0;
}

void
MaxwellBlochWaveEquation::IdentifyDegeneracies(double zero_tol, double rel_tol,
                                               vector<set<int> > & degen)
{
   // Get the eigenvalues
   vector<double> eigenvalues;
   this->GetEigenvalues(eigenvalues);

   // Assume no degeneracies
   degen.resize(eigenvalues.size());

   // No eigenvalues means no degeneracies
   if ( eigenvalues.size() == 0 )
   {
      return;
   }

   // Place the first eigenvalue in the first set
   int nd = 0;
   degen[nd].insert(0);

   // Switch to select between zero_tol and rel_tol
   bool zeroes = eigenvalues[0] < zero_tol;

   // Scan the eigenvalues
   for (unsigned int i=1; i<eigenvalues.size(); i++)
   {
      if ( zeroes )
      {
         // The previous eigenvalue was a zero
         if ( eigenvalues[i] > zero_tol )
         {
            // This eigenvalue is not a zero
            nd++;
            zeroes = false;
         }
         // Place this eigenvalue in the appropriate grouping
         degen[nd].insert(i);
      }
      else
      {
         // The previous eigenvalue was non-zero
         if ( fabs( eigenvalues[i] - eigenvalues[i-1] ) >
              ( eigenvalues[i] + eigenvalues[i-1] ) * 0.5 * rel_tol )
         {
            // This eigenvalue belongs to a new grouping
            nd++;
         }
         // Place this eigenvalue in the appropriate grouping
         degen[nd].insert(i);
      }
   }

   // Adjust size down to the number of degeneracies identified
   degen.resize( nd + 1 );
}

void
MaxwellBlochWaveEquation::DetermineBasis(const Vector & v1,
                                         std::vector<Vector> & e)
{
   e.resize(3);

   double kNorm = kappa_.Norml2();
   if ( kNorm < 1.0e-4 )
   {
      for (int i=0; i<3; i++)
      {
         e[i].SetSize(3); e[i] = 0.0; e[i][i] = 1.0;
      }
      return;
   }

   e[2].SetSize(3); e[2] = kappa_; e[2] /= kNorm;

   double e2dotv1 = e[2] * v1;
   e[1].SetSize(3); e[1] = v1; e[1].Add(-e2dotv1,v1);
   double e1norm  = e[1].Norml2();
   e[1] /= e1norm;

   e[0].SetSize(3);

   e[0][0] = e[1][1] * e[2][2] - e[1][2] * e[2][1];
   e[0][0] = e[1][2] * e[2][0] - e[1][0] * e[2][2];
   e[0][2] = e[1][0] * e[2][1] - e[1][1] * e[2][0];
}

void
MaxwellBlochWaveEquation::WriteVisitFields(const string & prefix,
                                           const string & label)
{
   cout << "Writing VisIt data to: " << prefix  << " " << label << endl;

   ParGridFunction Er(this->GetHCurlFESpace());
   ParGridFunction Ei(this->GetHCurlFESpace());

   ParGridFunction Br(this->GetHDivFESpace());
   ParGridFunction Bi(this->GetHDivFESpace());

   HypreParVector ErVec(this->GetHCurlFESpace()->GetComm(),
                        this->GetHCurlFESpace()->GlobalTrueVSize(),
                        NULL,
                        this->GetHCurlFESpace()->GetTrueDofOffsets());
   HypreParVector EiVec(this->GetHCurlFESpace()->GetComm(),
                        this->GetHCurlFESpace()->GlobalTrueVSize(),
                        NULL,
                        this->GetHCurlFESpace()->GetTrueDofOffsets());

   HypreParVector BrVec(this->GetHDivFESpace()->GetComm(),
                        this->GetHDivFESpace()->GlobalTrueVSize(),
                        NULL,
                        this->GetHDivFESpace()->GetTrueDofOffsets());
   HypreParVector BiVec(this->GetHDivFESpace()->GetComm(),
                        this->GetHDivFESpace()->GlobalTrueVSize(),
                        NULL,
                        this->GetHDivFESpace()->GetTrueDofOffsets());

   VisItDataCollection visit_dc(label.c_str(), pmesh_);
   visit_dc.SetPrefixPath(prefix.c_str());

   if ( dynamic_cast<GridFunctionCoefficient*>(mCoef_) )
   {
      GridFunctionCoefficient * gfc =
         dynamic_cast<GridFunctionCoefficient*>(mCoef_);
      visit_dc.RegisterField("epsilon", gfc->GetGridFunction() );
   }
   if ( dynamic_cast<GridFunctionCoefficient*>(kCoef_) )
   {
      GridFunctionCoefficient * gfc =
         dynamic_cast<GridFunctionCoefficient*>(kCoef_);
      visit_dc.RegisterField("muInv", gfc->GetGridFunction() );
   }
   /*
   if ( cosKappaX_ )
   {
     visit_dc.RegisterField("CosKappaX", cosKappaX_);
   }
   if ( sinKappaX_ )
   {
     visit_dc.RegisterField("SinKappaX", sinKappaX_);
   }
   */
   visit_dc.RegisterField("E_r", &Er);
   visit_dc.RegisterField("E_i", &Ei);
   visit_dc.RegisterField("B_r", &Br);
   visit_dc.RegisterField("B_i", &Bi);

   vector<double> eigenvalues;
   this->GetEigenvalues(eigenvalues);

   // cout << "Number of eigenmodes: " << nev_ << endl;
   for (int i=0; i<nev_; i++)
   {
      // cout << "Writing mode " << i << " corresponding to eigenvalue "
      //  << eigenvalues[i] << endl;
      this->GetEigenvector(i, ErVec, EiVec, BrVec, BiVec);

      Er = ErVec;
      Ei = EiVec;

      Br = BrVec;
      Bi = BiVec;

      visit_dc.SetCycle(i+1);
      if ( eigenvalues[i] > 0.0 )
      {
         visit_dc.SetTime(sqrt(eigenvalues[i]));
      }
      else if ( eigenvalues[i] > -1.0e-6 )
      {
         visit_dc.SetTime(0.0);
      }
      else
      {
         visit_dc.SetTime(-1.0);
      }

      visit_dc.Save();
   }
}

void
MaxwellBlochWaveEquation::GetSolverStats(double &meanTime, double &stdDevTime,
                                         double &meanIter, double &stdDevIter,
                                         int &nSolves)
{
   nSolves = (int)solve_times_.size();

   meanTime = 0.0;
   for (unsigned int i=0; i<solve_times_.size(); i++)
   {
      meanTime += solve_times_[i];
   }
   if ( nSolves > 0 ) { meanTime /= solve_times_.size(); }

   double var = 0.0;
   for (unsigned int i=0; i<solve_times_.size(); i++)
   {
      var += pow(solve_times_[i]-meanTime, 2.0);
   }
   if ( nSolves > 0 ) { var /= solve_times_.size(); }
   stdDevTime = sqrt(var);

   meanIter = 0.0;
   /*
   for (unsigned int i=0; i<solve_iters_.size(); i++)
   {
     meanIter += solve_iters_[i];
   }
   meanIter /= solve_iters_.size();
   */
   var = 0.0;
   /*
   for (unsigned int i=0; i<solve_iters_.size(); i++)
   {
     var += pow(solve_iters_[i]-meanIter, 2.0);
   }
   var /= solve_iters_.size();
   */
   stdDevIter = sqrt(var);
}

MaxwellBlochWaveEquation::MaxwellBlochWavePrecond::
MaxwellBlochWavePrecond(ParFiniteElementSpace & HCurlFESpace,
                        BlockDiagonalPreconditioner & BDP,
                        Operator & subSpaceProj,
                        //BlockOperator & LU,
                        double w)
   : Solver(2*HCurlFESpace.GlobalTrueVSize()),
     myid_(0), BDP_(&BDP), subSpaceProj_(&subSpaceProj), u_(NULL)
{
   // Initialize MPI variables
   MPI_Comm comm = HCurlFESpace.GetComm();
   MPI_Comm_rank(comm, &myid_);
   int numProcs = HCurlFESpace.GetNRanks();

   if ( myid_ == 0 ) { cout << "MaxwellBlochWavePrecond" << endl; }

   int locSize = 2*HCurlFESpace.TrueVSize();
   int glbSize = 0;

   HYPRE_Int * part = NULL;

   if (HYPRE_AssumedPartitionCheck())
   {
      part = new HYPRE_Int[2];

      MPI_Scan(&locSize, &part[1], 1, HYPRE_MPI_INT, MPI_SUM, comm);

      part[0] = part[1] - locSize;

      MPI_Allreduce(&locSize, &glbSize, 1, HYPRE_MPI_INT, MPI_SUM, comm);
   }
   else
   {
      part = new HYPRE_Int[numProcs+1];

      MPI_Allgather(&locSize, 1, MPI_INT,
                    &part[1], 1, HYPRE_MPI_INT, comm);

      part[0] = 0;
      for (int i=0; i<numProcs; i++)
      {
         part[i+1] += part[i];
      }

      glbSize = part[numProcs];
   }

   // r_ = new HypreParVector(comm,glbSize,part);
   u_ = new HypreParVector(comm,glbSize,part);
   // v_ = new HypreParVector(comm,glbSize,part);
}

MaxwellBlochWaveEquation::
MaxwellBlochWavePrecond::~MaxwellBlochWavePrecond()
{
   // delete r_;
   delete u_;
   // delete v_;
}

void
MaxwellBlochWaveEquation::
MaxwellBlochWavePrecond::Mult(const Vector & x, Vector & y) const
{
   if ( subSpaceProj_ )
   {
      BDP_->Mult(x,*u_);
      subSpaceProj_->Mult(*u_,y);
   }
   else
   {
      BDP_->Mult(x,y);
   }

}

void
MaxwellBlochWaveEquation::
MaxwellBlochWavePrecond::SetOperator(const Operator & A)
{
   A_ = &A;
}

MaxwellBlochWaveEquation::
MaxwellBlochWaveProjector::
MaxwellBlochWaveProjector(ParFiniteElementSpace & HCurlFESpace,
                          ParFiniteElementSpace & H1FESpace,
                          BlockOperator & M,
                          double beta, const Vector & zeta)
   : Operator(2*HCurlFESpace.GlobalTrueVSize()),
     newBeta_(true),
     newZeta_(true),
     HCurlFESpace_(&HCurlFESpace),
     H1FESpace_(&H1FESpace),
     beta_(beta),
     zeta_(zeta),
     T01_(NULL),
     Z01_(NULL),
     A0_(NULL),
     DKZ_(NULL),
     minres_(NULL),
     S0_(NULL),
     M_(&M),
     G_(NULL),
     urDummy_(NULL),
     uiDummy_(NULL),
     vrDummy_(NULL),
     viDummy_(NULL),
     u0_(NULL),
     v0_(NULL),
     u1_(NULL),
     v1_(NULL)
{
   // Initialize MPI variables
   MPI_Comm_rank(H1FESpace.GetParMesh()->GetComm(), &myid_);

   if ( myid_ == 0 )
   {
      cout << "Constructing MaxwellBlochWaveProjector" << endl;
   }

   block_offsets0_.SetSize(3);
   block_offsets0_[0] = 0;
   block_offsets0_[1] = H1FESpace.GetVSize();
   block_offsets0_[2] = H1FESpace.GetVSize();
   block_offsets0_.PartialSum();

   block_offsets1_.SetSize(3);
   block_offsets1_[0] = 0;
   block_offsets1_[1] = HCurlFESpace.GetVSize();
   block_offsets1_[2] = HCurlFESpace.GetVSize();
   block_offsets1_.PartialSum();

   block_trueOffsets0_.SetSize(3);
   block_trueOffsets0_[0] = 0;
   block_trueOffsets0_[1] = H1FESpace.TrueVSize();
   block_trueOffsets0_[2] = H1FESpace.TrueVSize();
   block_trueOffsets0_.PartialSum();

   block_trueOffsets1_.SetSize(3);
   block_trueOffsets1_[0] = 0;
   block_trueOffsets1_[1] = HCurlFESpace.TrueVSize();
   block_trueOffsets1_[2] = HCurlFESpace.TrueVSize();
   block_trueOffsets1_.PartialSum();

   locSize_ = HCurlFESpace.TrueVSize();

   u0_ = new BlockVector(block_trueOffsets0_);
   v0_ = new BlockVector(block_trueOffsets0_);
   u1_ = new BlockVector(block_trueOffsets1_);
   v1_ = new BlockVector(block_trueOffsets1_);

   if ( myid_ > 0 ) { cout << "done" << endl; }
}

MaxwellBlochWaveEquation::
MaxwellBlochWaveProjector::~MaxwellBlochWaveProjector()
{
   delete urDummy_; delete uiDummy_; delete vrDummy_; delete viDummy_;
   delete u0_; delete v0_;
   delete u1_; delete v1_;
   delete T01_;
   delete Z01_;
   delete A0_;
   delete DKZ_;
   delete S0_;
   delete G_;
   delete minres_;
}


void
MaxwellBlochWaveEquation::
MaxwellBlochWaveProjector::SetBeta(double beta)
{
   beta_ = beta; newBeta_ = true;
}

void
MaxwellBlochWaveEquation::
MaxwellBlochWaveProjector::SetZeta(const Vector & zeta)
{
   zeta_ = zeta; newZeta_ = true;
}

void
MaxwellBlochWaveEquation::
MaxwellBlochWaveProjector::Setup()
{
   if ( myid_ == 0 )
   {
      cout << "Setting up MaxwellBlochWaveProjector" << endl;
   }

   if ( T01_ == NULL )
   {
      if ( myid_ == 0 ) { cout << "Building Grad operator" << endl; }
      ParDiscreteGradOperator Grad(H1FESpace_,HCurlFESpace_);
      Grad.Assemble();
      Grad.Finalize();
      T01_ = Grad.ParallelAssemble();
   }

   if ( newZeta_ )
   {
      if ( Z01_ ) { delete Z01_; }
      if ( myid_ == 0 ) { cout << "Building zeta times operator" << endl; }
      ParDiscreteVectorProductOperator Zeta(H1FESpace_,
                                            HCurlFESpace_,zeta_);
      Zeta.Assemble();
      Zeta.Finalize();
      Z01_ = Zeta.ParallelAssemble();
   }

   if ( G_ == NULL )
   {
      if ( myid_ == 0 ) { cout << "Building Block G" << endl; }
      G_ = new BlockOperator(block_trueOffsets1_,block_trueOffsets0_);
   }
   G_->SetBlock(0,0,T01_);
   G_->SetBlock(1,1,T01_);
   if ( fabs(beta_) > 0.0 )
   {
      // G_->SetBlock(0,1,Zeta_->ParallelAssemble(), beta_*M_PI/180.0);
      // G_->SetBlock(1,0,Zeta_->ParallelAssemble(),-beta_*M_PI/180.0);
      G_->SetBlock(0,1,Z01_, beta_);
      G_->SetBlock(1,0,Z01_,-beta_);
   }
   G_->owns_blocks = 0;

   if ( newBeta_ || newZeta_ )
   {
      if ( myid_ == 0 ) { cout << "Forming GMG" << endl; }
      HypreParMatrix *  M1 = dynamic_cast<HypreParMatrix*>(&M_->GetBlock(0,0));
      HypreParMatrix * GMG = RAP(M1,T01_);

      if ( fabs(beta_) > 0.0 )
      {
         if ( myid_ == 0 ) { cout << "Forming 2nd order operators" << endl; }
         HypreParMatrix * ZMZ = RAP(M1,Z01_);

         HypreParMatrix * GMZ = RAP(T01_, M1, Z01_);
         HypreParMatrix * ZMG = RAP(Z01_, M1, T01_);
         *GMZ *= -1.0;

         if ( DKZ_ ) { delete DKZ_; }
         DKZ_ = ParAdd(GMZ,ZMG);

         delete GMZ;
         delete ZMG;

         // *ZMZ *= beta_*beta_*M_PI*M_PI/32400.0;
         *ZMZ *= beta_*beta_;
         if ( A0_ ) { delete A0_; }
         A0_ = ParAdd(GMG,ZMZ);
         delete GMG;
         delete ZMZ;
      }
      else
      {
         A0_ = GMG;
      }
   }

   if ( S0_ == NULL )
   {
      if ( myid_ > 0 ) { cout << "Building Block S0" << endl; }
      S0_ = new BlockOperator(block_trueOffsets0_);
   }
   S0_->SetDiagonalBlock(0,A0_,1.0);
   S0_->SetDiagonalBlock(1,A0_,1.0);
   if ( fabs(beta_) > 0.0 )
   {
      // S0_->SetBlock(0,1,DKZ_,-beta_*M_PI/180.0);
      // S0_->SetBlock(1,0,DKZ_, beta_*M_PI/180.0);
      S0_->SetBlock(0,1,DKZ_,-beta_);
      S0_->SetBlock(1,0,DKZ_, beta_);
   }
   S0_->owns_blocks = 0;

   if ( myid_ > 0 ) { cout << "Creating MINRES Solver" << endl; }
   delete minres_;
   minres_ = new MINRESSolver(H1FESpace_->GetComm());
   minres_->SetOperator(*S0_);
   minres_->SetRelTol(1e-13);
   minres_->SetMaxIter(3000);
   minres_->SetPrintLevel(0);

   newBeta_  = false;
   newZeta_  = false;

   if ( myid_ > 0 ) { cout << "done" << endl; }
}

void
MaxwellBlochWaveEquation::
MaxwellBlochWaveProjector::Mult(const Vector &x, Vector &y) const
{
   M_->Mult(x,y);
   G_->MultTranspose(y,*u0_);
   *v0_ = 0.0;
   minres_->Mult(*u0_,*v0_);
   G_->Mult(*v0_,y);
   y *= -1.0;
   y += x;
}

MaxwellBlochWaveSolver::MaxwellBlochWaveSolver(ParMesh & pmesh,
                                               BravaisLattice & bravais,
                                               Coefficient & epsCoef,
                                               Coefficient & muCoef,
                                               double tol)
   : max_lvl_(3)
   , nev_(24)
   , tol_(tol)
   , pmesh_(0)
   , mbwe_(0)
   , refineOp_(0)
   , initialVecs_(0)
   , locSize_(0)
   , part_(0)
   , epsCoef_(&epsCoef)
   , muInvCoef_(&muCoef)
{
   pmesh_.push_back(&pmesh);
   mbwe_.push_back(new MaxwellBlochWaveEquation(*pmesh_[0], 1));

   mbwe_[0]->SetMassCoef(*epsCoef_);
   mbwe_[0]->SetStiffnessCoef(muInvCoef_);

   ParFiniteElementSpace *  fespace = mbwe_[0]->GetHCurlFESpace();

   HypreParVector * Er = new HypreParVector(fespace->GetComm(),
                                            fespace->GlobalTrueVSize(),
                                            NULL,
                                            fespace->GetTrueDofOffsets());
   HypreParVector * Ei = new HypreParVector(fespace->GetComm(),
                                            fespace->GlobalTrueVSize(),
                                            NULL,
                                            fespace->GetTrueDofOffsets());

   EField_.push_back(std::make_pair(Er, Ei));
   /*
   EField_.push_back(std::make_pair(HypreParVector(fespace->GetComm(),
                                                   fespace->GlobalTrueVSize(),
                                                   NULL,
                                                   fespace->GetTrueDofOffsets()),
                                    HypreParVector(fespace->GetComm(),
                                                   fespace->GlobalTrueVSize(),
                                                   NULL,
                                                   fespace->GetTrueDofOffsets())
                                   )
                    );
   */
   /*
   efield_.resize(1);
   efield_[0].resize(nev_);
   for (int i=0; i<nev_; i++)
   {
     efield_[0][i] = make_pair(ParGridFunction(fespace),
             ParGridFunction(fespace));
   }
   */
   efield_.push_back(make_pair(new ParGridFunction(fespace),
                               new ParGridFunction(fespace)));

   initialVecs_.resize(1);
   initialVecs_[0].resize(nev_);

   locSize_.push_back(fespace->TrueVSize());
   int glbSize = 2 * fespace->GlobalTrueVSize();

   part_.push_back(NULL);
   this->createPartitioning(*fespace, part_[0]);

   for (int i=0; i<nev_; i++)
   {
      initialVecs_[0][i] = new HypreParVector(fespace->GetComm(),
                                              glbSize,
                                              part_[0]);
   }
}

MaxwellBlochWaveSolver::~MaxwellBlochWaveSolver()
{
   // Intentionally skip 0-th entry because it is not locally owned
   for (unsigned int i=1; i<pmesh_.size(); i++)
   {
      delete pmesh_[i]; pmesh_[i] = NULL;
   }
   for (unsigned int i=0; i<mbwe_.size(); i++)
   {
      delete mbwe_[i]; mbwe_[i] = NULL;
   }
   for (unsigned int i=0; i<refineOp_.size(); i++)
   {
      delete refineOp_[i]; refineOp_[i] = NULL;
   }
   for (unsigned int i=0; i<EField_.size(); i++)
   {
      delete EField_[i].first;  EField_[i].first = NULL;
      delete EField_[i].second; EField_[i].second = NULL;
   }
   for (unsigned int i=0; i<efield_.size(); i++)
   {
      delete efield_[i].first;  efield_[i].first = NULL;
      delete efield_[i].second; efield_[i].second = NULL;
   }
}

void
MaxwellBlochWaveSolver::SetKappa(const Vector & kappa)
{
   kappa_  = kappa;

   for (unsigned int i=0; i<mbwe_.size(); i++)
   {
      mbwe_[i]->SetKappa(kappa);
   }
}

void
MaxwellBlochWaveSolver::SetBeta(double beta)
{
   double oldBeta = kappa_.Norml2();

   kappa_ *= beta / (oldBeta > 0.0)?oldBeta:1.0;

   for (unsigned int i=0; i<mbwe_.size(); i++)
   {
      mbwe_[i]->SetBeta(beta);
   }
}

void
MaxwellBlochWaveSolver::SetZeta(const Vector & zeta)
{
   double oldBeta = kappa_.Norml2();
   kappa_ = zeta;
   kappa_ *= oldBeta;

   for (unsigned int i=0; i<mbwe_.size(); i++)
   {
      mbwe_[i]->SetZeta(zeta);
   }
}

void
MaxwellBlochWaveSolver::GetEigenfrequencies(std::vector<double> & omega)
{
   vector<double> coarse_eigs;
   vector<double> fine_eigs;

   mbwe_[0]->SetNumEigs(nev_);
   mbwe_[0]->Setup();
   mbwe_[0]->Solve();
   mbwe_[0]->GetEigenvalues(fine_eigs);
   /*
   for (unsigned int i=0; i<fine_eigs.size(); i++)
   {
     mbwe_[0]->GetEigenvectorE(i, EField_[0].first, EField_[0].second);
     efield_[0][i].first  = EField_[0].first;
     efield_[0][i].second = EField_[0].second;
   }
   */

   int lvl = 1;
   double err = 2.0 * tol_;
   while (lvl < max_lvl_ && err > tol_)
   {
      coarse_eigs.resize(fine_eigs.size());
      for (unsigned int i=0; i<fine_eigs.size(); i++)
      {
         coarse_eigs[i] = fine_eigs[i];
      }

      if ( lvl >= (int)mbwe_.size() )
      {
         pmesh_.push_back(new ParMesh(*pmesh_[lvl-1]));
         mbwe_.push_back(new MaxwellBlochWaveEquation(*pmesh_[lvl], 1));

         pmesh_[lvl]->UniformRefinement();

         mbwe_[lvl]->GetH1FESpace()->Update();
         mbwe_[lvl]->GetHDivFESpace()->Update();
         ParFiniteElementSpace *  fespace = mbwe_[lvl]->GetHCurlFESpace();

         refineOp_.push_back(fespace->GetUpdateOperator());
         fespace->SetUpdateOperatorOwner(false);

         HypreParVector * Er = new HypreParVector(fespace->GetComm(),
                                                  fespace->GlobalTrueVSize(),
                                                  NULL,
                                                  fespace->GetTrueDofOffsets());
         HypreParVector * Ei = new HypreParVector(fespace->GetComm(),
                                                  fespace->GlobalTrueVSize(),
                                                  NULL,
                                                  fespace->GetTrueDofOffsets());

         EField_.push_back(std::make_pair(Er, Ei));
         /*
              EField_.push_back(
                 std::make_pair(HypreParVector(fespace->GetComm(),
                                               fespace->GlobalTrueVSize(),
                                               NULL,
                                               fespace->GetTrueDofOffsets()),
                                HypreParVector(fespace->GetComm(),
                                               fespace->GlobalTrueVSize(),
                                               NULL,
                                               fespace->GetTrueDofOffsets()))
              );
         */
         /*
         efield_[lvl].resize(nev_);
         for (int i=0; i<nev_; i++)
         {
           efield_[lvl][i] = make_pair(ParGridFunction(fespace),
                       ParGridFunction(fespace));
         }
              */
         efield_.push_back(make_pair(new ParGridFunction(fespace),
                                     new ParGridFunction(fespace)));

         initialVecs_.resize(lvl+1);
         initialVecs_[lvl].resize(nev_);

         locSize_.push_back(fespace->TrueVSize());
         int glbSize = 2 * fespace->GlobalTrueVSize();

         part_.push_back(NULL);
         this->createPartitioning(*fespace, part_[lvl]);

         for (int i=0; i<nev_; i++)
         {
            initialVecs_[lvl][i] = new HypreParVector(fespace->GetComm(),
                                                      glbSize,
                                                      part_[lvl]);
         }

         mbwe_[lvl]->SetMassCoef(*epsCoef_);
         mbwe_[lvl]->SetStiffnessCoef(muInvCoef_);
         mbwe_[lvl]->SetKappa(kappa_);
         mbwe_[lvl]->SetNumEigs(nev_);
         mbwe_[lvl]->Setup();
      }


      for (int i=0; i<nev_; i++)
      {
         mbwe_[lvl-1]->GetEigenvectorE(i,
                                       *EField_[lvl-1].first,
                                       *EField_[lvl-1].second);
         *efield_[lvl-1].first  = *EField_[lvl-1].first;
         *efield_[lvl-1].second = *EField_[lvl-1].second;

         refineOp_[lvl-1]->Mult(*efield_[lvl-1].first,
                                *efield_[lvl].first);
         refineOp_[lvl-1]->Mult(*efield_[lvl-1].second,
                                *efield_[lvl].second);

         EField_[lvl].first->SetData(
            initialVecs_[lvl][i]->GetData());
         EField_[lvl].second->SetData(
            &initialVecs_[lvl][i]->GetData()[locSize_[lvl]]);

         efield_[lvl].first->ParallelProject(*EField_[lvl].first);
         efield_[lvl].second->ParallelProject(*EField_[lvl].second);
      }
      mbwe_[lvl]->SetInitialVectors(nev_, &initialVecs_[lvl][0]);
      mbwe_[lvl]->Solve();
      mbwe_[lvl]->GetEigenvalues(fine_eigs);
      /*
      for (unsigned int i=0; i<fine_eigs.size(); i++)
      {
         mbwe_[lvl]->GetEigenvectorE(i,
                 EField_[lvl].first,
                 EField_[lvl].second);
         efield_[lvl].first  = EField_[lvl].first;
         efield_[lvl].second = EField_[lvl].second;
      }
      */
      err = 0.0;
      for (unsigned int i=0; i<fine_eigs.size(); i++)
      {
         err += fabs(fine_eigs[i] - coarse_eigs[i]);// /
         //(fine_eigs[i] > 1.0e-4)?fine_eigs[i]:1.0;
      }
      // err = sqrt(err);
      cout << "Error norm: " << err << endl;
      lvl++;
   }

   omega.resize(fine_eigs.size());
   for (unsigned int i=0; i<fine_eigs.size(); i++)
   {
      omega[i] = sqrt(fabs(fine_eigs[i]));
   }
}

void
MaxwellBlochWaveSolver::createPartitioning(ParFiniteElementSpace & pfes,
                                           HYPRE_Int *& part)
{
   MPI_Comm comm = pfes.GetComm();
   int locSize = 2 * pfes.TrueVSize();
   int glbSize = 0;

   if (HYPRE_AssumedPartitionCheck())
   {
      part = new HYPRE_Int[2];

      MPI_Scan(&locSize, &part[1], 1, HYPRE_MPI_INT, MPI_SUM, comm);

      part[0] = part[1] - locSize;

      MPI_Allreduce(&locSize, &glbSize, 1, HYPRE_MPI_INT, MPI_SUM, comm);
   }
   else
   {
      int numProcs  = pfes.GetNRanks();
      part = new HYPRE_Int[numProcs+1];

      MPI_Allgather(&locSize, 1, MPI_INT,
                    &part[1], 1, HYPRE_MPI_INT, comm);

      part[0] = 0;
      for (int i=0; i<numProcs; i++)
      {
         part[i+1] += part[i];
      }

      glbSize = part[numProcs];
   }
}

void
MaxwellBlochWaveSolver::InitializeGLVis(VisData & vd)
{}

void
MaxwellBlochWaveSolver::DisplayToGLVis()
{}

void
WriteVisItFields(const std::string & prefix,
                 const std::string & label)
{}

MaxwellDispersion::MaxwellDispersion(ParMesh & pmesh,
                                     BravaisLattice & bravais,
                                     Coefficient & epsCoef,
                                     Coefficient & muCoef,
                                     double tol)
   : bravais_(&bravais)
{
   mbws_ = new MaxwellBlochWaveSolver(pmesh, bravais, epsCoef, muCoef, tol);
}

MaxwellDispersion::~MaxwellDispersion()
{
   delete mbws_;
}

void
MaxwellDispersion::GetDispersionPlot()
{
   this->traverseBrillouinZone();
}

void
MaxwellDispersion::InitializeGLVis(VisData & vd)
{}

void
MaxwellDispersion::DisplayToGLVis()
{}

void
MaxwellDispersion::WriteVisItFields(const std::string & prefix,
                                    const std::string & label)
{}

void
MaxwellDispersion::traverseBrillouinZone()
{
   Vector kappa(3);
   bravais_->GetSymmetryPoint(1, kappa);
   mbws_->SetKappa(kappa);

   vector<double> omega;
   mbws_->GetEigenfrequencies(omega);
}

MaxwellBandGap::MaxwellBandGap(ParMesh & pmesh,
                               BravaisLattice & bravais,
                               Coefficient & epsCoef,
                               Coefficient & muCoef,
                               double tol)
   : Homogenization(pmesh.GetComm())
     //, pmesh_(&pmesh)
     //, bravais_(&bravais)
     //, epsCoef_(&epsCoef)
     //, muCoef_(&muCoef)
{
   disp_ = new MaxwellDispersion(pmesh, bravais, epsCoef, muCoef, tol);
}

MaxwellBandGap::~MaxwellBandGap()
{
   delete disp_;
}

void
MaxwellBandGap::GetHomogenizedProperties(std::vector<double> & p)
{
   p.resize(0);

   disp_->GetDispersionPlot();
}

void
MaxwellBandGap::InitializeGLVis(VisData & vd)
{}

void
MaxwellBandGap::DisplayToGLVis()
{}

void
MaxwellBandGap::WriteVisItFields(const std::string & prefix,
                                 const std::string & label)
{}

} // namespace meta_material
} // namespace mfem

#endif // MFEM_USE_MPI
