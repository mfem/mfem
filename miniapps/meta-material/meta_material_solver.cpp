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
         pmesh_->RefineByError(*divGradRho_, 0.1);
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
   */
   {
      ofstream ofs;
      ofs.precision(16);
      ofs.open("E_X.vec"); E_[0]->Print(ofs, 1); ofs.close();
      ofs.open("E_Y.vec"); E_[1]->Print(ofs, 1); ofs.close();
      ofs.open("E_Z.vec"); E_[2]->Print(ofs, 1); ofs.close();
   }
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
	 for (int i=0; i<6; i++) m_[i]->Update();
	 for (int i=0; i<6; i++) Chi_[i]->Update();
	 for (int i=0; i<3; i++) E_[i]->Update();
	 for (int i=0; i<6; i++) F_[i]->Update();
	 for (int i=0; i<6; i++) MF_[i]->Update();
	 grad_->Update();
	 errSol_->SetData(&(Chi_[0]->GetData())[0]);
	 errSol_->Update();
	 for (int i=0; i<4; i++) errors_[i]->Update();
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

} // namespace meta_material
} // namespace mfem

#endif // MFEM_USE_MPI
