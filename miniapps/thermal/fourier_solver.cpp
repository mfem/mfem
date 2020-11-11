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

#include "fourier_solver.hpp"
#include "../common/mesh_extras.hpp"
#include "../../general/text.hpp"

#ifdef MFEM_USE_MPI

using namespace std;

namespace mfem
{

using namespace common;

namespace thermal
{

std::string FieldSymbol(FieldType t)
{
   switch (t)
   {
      case DENSITY:
         return "rho";
      case TEMPERATURE:
         return "T";
      default:
         return "N/A";
   }
}

CoefFactory::~CoefFactory()
{
   for (int i=0; i<coefs.Size(); i++)
   {
      delete coefs[i];
   }
}

Coefficient * CoefFactory::operator()(std::istream &input)
{
   string buff;

   skip_comment_lines(input, '#');
   input >> buff;
   return (*this)(buff, input);
}

Coefficient * CoefFactory::operator()(std::string &name, std::istream &input)
{
   if (name == "ConstantCoefficient")
   {
      double val;
      input >> val;
      int c = coefs.Append(new ConstantCoefficient(val));
      return coefs[--c];
   }
   else if (name == "PWConstCoefficient")
   {
      int nvals;
      input >> nvals;
      Vector vals(nvals);
      for (int i=0; i<nvals; i++)
      {
         input >> vals[i];
      }
      int c = coefs.Append(new PWConstCoefficient(vals));
      return coefs[--c];
   }
   else if (name == "FunctionCoefficient")
   {
      int index;
      input >> index;
      MFEM_VERIFY(index >=0 && index < ext_fn.Size(),
                  "Invalid Function index read by CoefFactory");
      int c = coefs.Append(new FunctionCoefficient(ext_fn[index]));
      return coefs[--c];
   }
   else if (name == "GridFunctionCoefficient")
   {
      int index;
      input >> index;
      MFEM_VERIFY(index >=0 && index < ext_gf.Size(),
                  "Invalid GridFunction index read by CoefFactory");
      int c = coefs.Append(new GridFunctionCoefficient(ext_gf[index]));
      return coefs[--c];
   }
   else if (name == "RestrictedCoefficient")
   {
      Coefficient * rc = (*this)(input);
      int nattr;
      input >> nattr;
      Array<int> attr(nattr);
      for (int i=0; i<nattr; i++)
      {
         input >> attr[i];
      }
      int c = coefs.Append(new RestrictedCoefficient(*rc, attr));
      return coefs[--c];
   }
   else
   {
      // Unrecognized coefficient type.  Try external factories.
      for (int i=0; i<ext_fac.Size(); i++)
      {
         Coefficient * c = (*ext_fac[i])(name, input);
         if (c != NULL) { return c; }
      }
      return NULL;
   }
}

AdvectionDiffusionBC::~AdvectionDiffusionBC()
{
   for (int i=0; i<dbc.Size(); i++)
   {
      if (dbc[i]->ownCoef)
      {
         delete dbc[i]->coef;
      }
      delete dbc[i];
   }
   for (int i=0; i<nbc.Size(); i++)
   {
      if (nbc[i]->ownCoef)
      {
         delete nbc[i]->coef;
      }
      delete nbc[i];
   }
   for (int i=0; i<rbc.Size(); i++)
   {
      for (int j=0; j<rbc[i]->coefs.Size(); j++)
      {
         if (rbc[i]->ownCoefs[j])
         {
            delete rbc[i]->coefs[j];
         }
      }
      delete rbc[i];
   }
}

const char * AdvectionDiffusionBC::GetBCTypeName(BCType bctype)
{
   switch (bctype)
   {
      case DIRICHLET_BC: return "Dirichlet";
      case NEUMANN_BC: return "Neumann";
      case ROBIN_BC: return "Robin";
   }
   return "Unknown";
}

void AdvectionDiffusionBC::ReadAttr(std::istream &input,
                                    BCType bctype,
                                    Array<int> &attr)
{
   int nbdr = 0;
   skip_comment_lines(input, '#');
   input >> nbdr;
   for (int i=0; i<nbdr; i++)
   {
      int b = 0;
      input >> b;
      if (bc_attr.count(b) == 0)
      {
         bc_attr.insert(b);
         if (bctype == DIRICHLET_BC)
         {
            dbc_attr.Append(b);
         }
      }
      else
      {
         MFEM_ABORT("Attempting to add a " << GetBCTypeName(bctype)
                    << " BC on boundary " << b
                    << " which already has a boundary condition defined.");
      }
      attr.Append(b);
   }
}

void AdvectionDiffusionBC::ReadCoefByAttr(std::istream &input,
                                          BCType bctype,
                                          CoefficientByAttr &cba)
{
   ReadAttr(input, bctype, cba.attr);
   cba.coef = (*coefFact)(input);
   cba.ownCoef = false;
}

void AdvectionDiffusionBC::ReadCoefsByAttr(std::istream &input,
                                           BCType bctype,
                                           CoefficientsByAttr &cba)
{
   ReadAttr(input, bctype, cba.attr);
   cba.coefs.SetSize(2);
   cba.coefs[0] = (*coefFact)(input);
   cba.coefs[1] = (*coefFact)(input);
   cba.ownCoefs.SetSize(2);
   cba.ownCoefs[0] = false;
   cba.ownCoefs[1] = false;
}

void AdvectionDiffusionBC::ReadBCs(std::istream &input)
{
   string buff;

   skip_comment_lines(input, '#');
   input >> buff;
   MFEM_VERIFY(buff == "scalar_bcs", "invalid BC file");

   while (input >> buff)
   {
      skip_comment_lines(input, '#');
      if (buff == "dirichlet")
      {
         CoefficientByAttr * c = new CoefficientByAttr;
         ReadCoefByAttr(input, DIRICHLET_BC, *c);
         dbc.Append(c);
      }
      else if (buff == "neumann")
      {
         CoefficientByAttr * c = new CoefficientByAttr;
         ReadCoefByAttr(input, NEUMANN_BC, *c);
         nbc.Append(c);
      }
      else if (buff == "robin")
      {
         CoefficientsByAttr * c = new CoefficientsByAttr;
         ReadCoefsByAttr(input, ROBIN_BC, *c);
         rbc.Append(c);
      }
   }
}

void AdvectionDiffusionBC::AddDirichletBC(const Array<int> & bdr,
                                          Coefficient &val)
{
   for (int i=0; i<bdr.Size(); i++)
   {
      if (bc_attr.count(bdr[i]) == 0)
      {
         bc_attr.insert(bdr[i]);
         dbc_attr.Append(bdr[i]);
      }
      else
      {
         MFEM_ABORT("Attempting to add a Dirichlet BC on boundary " << bdr[i]
                    << " which already has a boundary condition defined.");
      }
   }
   CoefficientByAttr * c = new CoefficientByAttr;
   c->attr = bdr;
   c->coef = &val;
   c->ownCoef = false;
   dbc.Append(c);
}

void AdvectionDiffusionBC::AddNeumannBC(const Array<int> & bdr,
                                        Coefficient &val)
{
   for (int i=0; i<bdr.Size(); i++)
   {
      if (bc_attr.count(bdr[i]) == 0)
      {
         bc_attr.insert(bdr[i]);
      }
      else
      {
         MFEM_ABORT("Attempting to add a Neumann BC on boundary " << bdr[i]
                    << " which already has a boundary condition defined.");
      }
   }
   CoefficientByAttr * c = new CoefficientByAttr;
   c->attr = bdr;
   c->coef = &val;
   c->ownCoef = false;
   nbc.Append(c);
}

void AdvectionDiffusionBC::AddRobinBC(const Array<int> & bdr, Coefficient &a,
                                      Coefficient &b)
{
   for (int i=0; i<bdr.Size(); i++)
   {
      if (bc_attr.count(bdr[i]) == 0)
      {
         bc_attr.insert(bdr[i]);
      }
      else
      {
         MFEM_ABORT("Attempting to add a Robin BC on boundary " << bdr[i]
                    << " which already has a boundary condition defined.");
      }
   }
   CoefficientsByAttr * c = new CoefficientsByAttr;
   c->attr = bdr;
   c->coefs.SetSize(2);
   c->coefs[0] = &a;
   c->coefs[1] = &b;
   c->ownCoefs.SetSize(2);
   c->ownCoefs = false;
   rbc.Append(c);
}

const Array<int> & AdvectionDiffusionBC::GetHomogeneousNeumannBDR() const
{
   if (hbc_attr.Size() != bdr_attr.Size() - bc_attr.size())
   {
      hbc_attr.SetSize(bdr_attr.Size() - bc_attr.size());
      int o = 0;
      for (int i=0; i<bdr_attr.Size(); i++)
      {
         if (bc_attr.count(bdr_attr[i]) == 0)
         {
            hbc_attr[o++] = bdr_attr[i];
         }
      }
   }
   return hbc_attr;
}

AdvectionTDO::AdvectionTDO(ParFiniteElementSpace &H1_FES,
                           VectorCoefficient &velCoef)
   : TimeDependentOperator(H1_FES.GetVSize(), 0.0),
     H1_FESpace_(H1_FES),
     velCoef_(velCoef),
     ess_bdr_tdofs_(0),
     m1_(&H1_FES),
     adv1_(&H1_FES),
     M1Inv_(NULL),
     M1Diag_(NULL),
     SOL_(H1_FES.GetTrueVSize()),
     RHS_(H1_FES.GetTrueVSize()),
     rhs_(H1_FES.GetVSize())
{
   m1_.AddDomainIntegrator(new MassIntegrator);
   m1_.Assemble();

   adv1_.AddDomainIntegrator(new MixedScalarWeakDivergenceIntegrator(velCoef_));
   adv1_.Assemble();
}

AdvectionTDO::~AdvectionTDO()
{
   delete M1Inv_;
   delete M1Diag_;
}

void
AdvectionTDO::initMult() const
{
   if ( M1Inv_ == NULL || M1Diag_ == NULL )
   {
      if ( M1Inv_ == NULL )
      {
         M1Inv_ = new HyprePCG(M1_);
         M1Inv_->SetTol(1e-12);
         M1Inv_->SetMaxIter(200);
         M1Inv_->SetPrintLevel(0);
      }
      else
      {
         M1Inv_->SetOperator(M1_);
      }
      if ( M1Diag_ == NULL )
      {
         M1Diag_ = new HypreDiagScale(M1_);
         M1Inv_->SetPreconditioner(*M1Diag_);
      }
      else
      {
         M1Diag_->SetOperator(M1_);
      }
   }
}

void AdvectionTDO::Mult(const Vector &y, Vector &dydt) const
{
   dydt_gf_.MakeRef(&H1_FESpace_, dydt);
   adv1_.Mult(y, rhs_);
   rhs_ *= -1.0;

   dydt_gf_ = 0.0;
   m1_.FormLinearSystem(ess_bdr_tdofs_, dydt_gf_, rhs_, M1_, SOL_, RHS_);

   this->initMult();

   M1Inv_->Mult(RHS_, SOL_);

   m1_.RecoverFEMSolution(SOL_, rhs_, dydt_gf_);

}

DiffusionTDO::DiffusionTDO(ParFiniteElementSpace &H1_FES,
                           Coefficient & dTdtBdr,
                           Array<int> & bdr_attr,
                           Coefficient & c, bool td_c,
                           Coefficient & k, bool td_k,
                           Coefficient & Q, bool td_Q)
   : TimeDependentOperator(H1_FES.GetVSize(), 0.0),
     myid_(H1_FES.GetMyRank()),
     init_(false), //initA_(false), initAInv_(false),
     multCount_(0), solveCount_(0),
     H1_FESpace_(&H1_FES),
     mC_(NULL), sK_(NULL), a_(NULL), dTdt_gf_(NULL), Qs_(NULL),
     MCInv_(NULL), MCDiag_(NULL),
     AInv_(NULL), APrecond_(NULL),
     rhs_(NULL),
     bdr_attr_(&bdr_attr), ess_bdr_tdofs_(0), dTdtBdrCoef_(&dTdtBdr),
     tdQ_(td_Q), tdC_(td_c), tdK_(td_k),
     QCoef_(&Q), CCoef_(&c), kCoef_(&k), KCoef_(NULL),
     // CInvCoef_(NULL), kInvCoef_(NULL), KInvCoef_(NULL)
     dtkCoef_(NULL), dtKCoef_(NULL)
{
   this->init();
}

DiffusionTDO::DiffusionTDO(ParFiniteElementSpace &H1_FES,
                           Coefficient & dTdtBdr,
                           Array<int> & bdr_attr,
                           Coefficient & c, bool td_c,
                           MatrixCoefficient & K, bool td_k,
                           Coefficient & Q, bool td_Q)
   : TimeDependentOperator(H1_FES.GetVSize(), 0.0),
     init_(false),
     multCount_(0), solveCount_(0),
     H1_FESpace_(&H1_FES),
     mC_(NULL), sK_(NULL), a_(NULL), dTdt_gf_(NULL), Qs_(NULL),
     MCInv_(NULL), MCDiag_(NULL),
     AInv_(NULL), APrecond_(NULL),
     rhs_(NULL),
     bdr_attr_(&bdr_attr), ess_bdr_tdofs_(0), dTdtBdrCoef_(&dTdtBdr),
     tdQ_(td_Q), tdC_(td_c), tdK_(td_k),
     QCoef_(&Q), CCoef_(&c), kCoef_(NULL), KCoef_(&K),
     dtkCoef_(NULL), dtKCoef_(NULL)
{
   this->init();
}

DiffusionTDO::~DiffusionTDO()
{
   delete a_;
   delete mC_;
   delete sK_;
   delete dTdt_gf_;
   delete Qs_;
   delete MCInv_;
   delete MCDiag_;
   delete AInv_;
   delete APrecond_;
}

void
DiffusionTDO::init()
{
   if ( init_ ) { return; }

   if ( mC_ == NULL )
   {
      mC_ = new ParBilinearForm(H1_FESpace_);
      mC_->AddDomainIntegrator(new MassIntegrator(*CCoef_));
      mC_->Assemble();
   }

   if ( sK_ == NULL )
   {
      sK_ = new ParBilinearForm(H1_FESpace_);
      if ( kCoef_ != NULL )
      {
         sK_->AddDomainIntegrator(new DiffusionIntegrator(*kCoef_));
      }
      else if ( KCoef_ != NULL )
      {
         sK_->AddDomainIntegrator(new DiffusionIntegrator(*KCoef_));
      }
      sK_->Assemble();
   }
   if ( dTdt_gf_ == NULL )
   {
      dTdt_gf_ = new ParGridFunction(H1_FESpace_);
   }
   if ( Qs_ == NULL && QCoef_ != NULL )
   {
      Qs_ = new ParLinearForm(H1_FESpace_);
      Qs_->AddDomainIntegrator(new DomainLFIntegrator(*QCoef_));
      Qs_->Assemble();
      rhs_ = new Vector(Qs_->Size());
   }
   /*
   CInvCoef_ = new InverseCoefficient(*CCoef_);
   if ( kCoef_ != NULL ) kInvCoef_ = new InverseCoefficient(*kCoef_);
   if ( KCoef_ != NULL ) KInvCoef_ = new MatrixInverseCoefficient(*KCoef_);
   */
   H1_FESpace_->GetEssentialTrueDofs(*bdr_attr_, ess_bdr_tdofs_);

   init_ = true;
}

void
DiffusionTDO::SetTime(const double time)
{
   this->TimeDependentOperator::SetTime(time);

   dTdtBdrCoef_->SetTime(t);

   if ( tdQ_ )
   {
      QCoef_->SetTime(t);
      Qs_->Assemble();
   }

   if ( tdC_ )
   {
      CCoef_->SetTime(t);
      mC_->Assemble();
   }

   if ( tdK_ )
   {
      if ( kCoef_ != NULL ) { kCoef_->SetTime(t); }
      if ( KCoef_ != NULL ) { KCoef_->SetTime(t); }
      sK_->Assemble();
   }

   if ( ( tdC_ || tdK_ ) && a_ != NULL )
   {
      a_->Assemble();
   }

   newTime_ = true;
}
/*
void
DiffusionTDOOperator::SetHeatSource(Coefficient & Q, bool time_dep)
{
if ( ownsQ_ )
  {
    delete QCoef_;
  }

tdQ_   = time_dep;
QCoef_ = &Q;
}

void
DiffusionTDOOperator::SetConductivityCoefficient(Coefficient & k,
                  bool time_dep)
{
if ( ownsK_ )
  {
    delete kCoef_;
    delete KCoef_;
  }

tdK_   = time_dep;
kCoef_ = &k;
KCoef_ = NULL;
}

void
DiffusionTDOOperator::SetConductivityCoefficient(MatrixCoefficient & K,
                  bool time_dep)
{
if ( ownsK_ )
  {
    delete kCoef_;
    delete KCoef_;
  }

tdK_   = time_dep;
kCoef_ = NULL;
KCoef_ = &K;
}

void
DiffusionTDOOperator::SetSpecificHeatCoefficient(Coefficient & c,
                  bool time_dep)
{
if ( ownsC_ )
  {
    delete CCoef_;
  }

tdC_   = time_dep;
CCoef_ = &c;
}
*/
void
DiffusionTDO::initMult() const
{
   if ( tdC_ || MCInv_ == NULL || MCDiag_ == NULL )
   {
      if ( MCInv_ == NULL )
      {
         MCInv_ = new HyprePCG(MC_);
         MCInv_->SetTol(1e-12);
         MCInv_->SetMaxIter(200);
         MCInv_->SetPrintLevel(0);
      }
      else
      {
         MCInv_->SetOperator(MC_);
      }
      if ( MCDiag_ == NULL )
      {
         MCDiag_ = new HypreDiagScale(MC_);
         MCInv_->SetPreconditioner(*MCDiag_);
      }
      else
      {
         MCDiag_->SetOperator(MC_);
      }
   }
}

void
DiffusionTDO::Mult(const Vector &T, Vector &dT_dt) const
{
   dT_dt = 0.0;

   sK_->Mult(T, *rhs_);

   *rhs_ -= *Qs_;
   rhs_->Neg();

   dTdt_gf_->ProjectBdrCoefficient(*dTdtBdrCoef_, *bdr_attr_);

   mC_->FormLinearSystem(ess_bdr_tdofs_, *dTdt_gf_, *rhs_, MC_, dTdt_, RHS_);

   this->initMult();

   MCInv_->Mult(RHS_, dTdt_);

   mC_->RecoverFEMSolution(dTdt_, *rhs_, dT_dt);

   multCount_++;
}

void
DiffusionTDO::initA(double dt)
{
   if ( kCoef_ != NULL )
   {
      if (dtkCoef_ == NULL)
      {
         dtkCoef_ = new ProductCoefficient(dt, *kCoef_);
      }
      dtkCoef_->SetAConst(dt);
   }
   else
   {
      if (dtKCoef_ == NULL)
      {
         dtKCoef_ = new ScalarMatrixProductCoefficient(dt, *KCoef_);
      }
      dtKCoef_->SetAConst(dt);
   }
   if ( a_ == NULL)
   {
      a_ = new ParBilinearForm(H1_FESpace_);
      a_->AddDomainIntegrator(new MassIntegrator(*CCoef_));
      if ( kCoef_ != NULL)
      {
         a_->AddDomainIntegrator(new DiffusionIntegrator(*dtkCoef_));
      }
      else
      {
         a_->AddDomainIntegrator(new DiffusionIntegrator(*dtKCoef_));
      }
   }
   a_->Update();
   a_->Assemble();
}

void
DiffusionTDO::initImplicitSolve()
{
   delete AInv_;
   AInv_ = new HyprePCG(A_);
   AInv_->SetTol(1e-12);
   AInv_->SetMaxIter(200);
   AInv_->SetPrintLevel(0);

   delete APrecond_;
   APrecond_ = new HypreBoomerAMG(A_);
   APrecond_->SetPrintLevel(0);
   AInv_->SetPreconditioner(*APrecond_);
   /*
      if ( tdC_ || tdK_ || AInv_ == NULL || APrecond_ == NULL )
      {
         if ( AInv_ == NULL )
         {
            AInv_ = new HyprePCG(A_);
            AInv_->SetTol(1e-12);
            AInv_->SetMaxIter(200);
            AInv_->SetPrintLevel(0);
         }
         else
         {
            AInv_->SetOperator(A_);
         }
         if ( APrecond_ == NULL )
         {
            APrecond_ = new HypreBoomerAMG(A_);
            APrecond_->SetPrintLevel(0);
            AInv_->SetPreconditioner(*APrecond_);
         }
         else
         {
            APrecond_->SetOperator(A_);
         }
      }
     */
}

void
DiffusionTDO::ImplicitSolve(const double dt,
                            const Vector &T, Vector &dT_dt)
{
   dT_dt = 0.0;
   // cout << "sK size: " << sK_->Width() << ", T size: " << T.Size() << ", rhs_ size: " << rhs_->Size() << endl;

   /*
   ostringstream ossT; ossT << "T_" << solveCount_ << ".vec";
   ofstream ofsT(ossT.str().c_str());
   T.Print(ofsT);
   ofsT.close();
   */
   sK_->Mult(T, *rhs_);
   /*
   ofstream ofsrhs("rhs.vec");
   rhs_->Print(ofsrhs);

   ofstream ofsQ("Q.vec");
   Qs_->Print(ofsQ);
   */
   *rhs_ -= *Qs_;
   *rhs_ *= -1.0;

   dTdt_gf_->ProjectBdrCoefficient(*dTdtBdrCoef_, *bdr_attr_);

   this->initA(dt);

   a_->FormLinearSystem(ess_bdr_tdofs_, *dTdt_gf_, *rhs_, A_, dTdt_, RHS_);
   /*
   A_.Print("A.mat");
   ofstream ofsB("b.vec");
   RHS_.Print(ofsB);
   */
   this->initImplicitSolve();

   AInv_->Mult(RHS_, dTdt_);

   a_->RecoverFEMSolution(dTdt_, *rhs_, dT_dt);

   solveCount_++;
}

AdvectionDiffusionTDO::AdvectionDiffusionTDO(ParFiniteElementSpace &H1_FES,
                                             Coefficient & dTdtBdr,
                                             Array<int> & bdr_attr,
                                             Coefficient & c, bool td_c,
                                             Coefficient & k, bool td_k,
                                             VectorCoefficient & V,
                                             bool td_v, double nu,
                                             Coefficient & Q, bool td_Q)
   : TimeDependentOperator(H1_FES.GetVSize(), 0.0),
     myid_(H1_FES.GetMyRank()),
     init_(false), //initA_(false), initAInv_(false),
     multCount_(0), solveCount_(0),
     H1_FESpace_(&H1_FES),
     mC_(NULL), sK_(NULL), aV_(NULL), a_(NULL), dTdt_gf_(NULL), Qs_(NULL),
     MCInv_(NULL), MCDiag_(NULL),
     AInv_(NULL), APrecond_(NULL),
     rhs_(NULL),
     bdr_attr_(&bdr_attr), ess_bdr_tdofs_(0), dTdtBdrCoef_(&dTdtBdr),
     tdQ_(td_Q), tdC_(td_c), tdK_(td_k), tdV_(td_v), nu_(nu),
     QCoef_(&Q), CCoef_(&c), kCoef_(&k), KCoef_(NULL), VCoef_(&V),
     // CInvCoef_(NULL), kInvCoef_(NULL), KInvCoef_(NULL)
     dtkCoef_(NULL), dtKCoef_(NULL), dtnuVCoef_(NULL)
{
   this->init();
}

AdvectionDiffusionTDO::AdvectionDiffusionTDO(ParFiniteElementSpace &H1_FES,
                                             Coefficient & dTdtBdr,
                                             Array<int> & bdr_attr,
                                             Coefficient & c, bool td_c,
                                             MatrixCoefficient & K, bool td_k,
                                             VectorCoefficient & V,
                                             bool td_v, double nu,
                                             Coefficient & Q, bool td_Q)
   : TimeDependentOperator(H1_FES.GetVSize(), 0.0),
     init_(false),
     multCount_(0), solveCount_(0),
     H1_FESpace_(&H1_FES),
     mC_(NULL), sK_(NULL), aV_(NULL), a_(NULL), dTdt_gf_(NULL), Qs_(NULL),
     MCInv_(NULL), MCDiag_(NULL),
     AInv_(NULL), APrecond_(NULL),
     rhs_(NULL),
     bdr_attr_(&bdr_attr), ess_bdr_tdofs_(0), dTdtBdrCoef_(&dTdtBdr),
     tdQ_(td_Q), tdC_(td_c), tdK_(td_k), tdV_(td_v), nu_(nu),
     QCoef_(&Q), CCoef_(&c), kCoef_(NULL), KCoef_(&K), VCoef_(&V),
     dtkCoef_(NULL), dtKCoef_(NULL), dtnuVCoef_(NULL)
{
   this->init();
}

AdvectionDiffusionTDO::~AdvectionDiffusionTDO()
{
   delete a_;
   delete aV_;
   delete mC_;
   delete sK_;
   delete dTdt_gf_;
   delete Qs_;
   delete MCInv_;
   delete MCDiag_;
   delete AInv_;
   delete APrecond_;
   delete dtkCoef_;
   delete dtKCoef_;
   delete dtnuVCoef_;
}

void
AdvectionDiffusionTDO::init()
{
   if ( init_ ) { return; }

   if ( mC_ == NULL )
   {
      mC_ = new ParBilinearForm(H1_FESpace_);
      mC_->AddDomainIntegrator(new MassIntegrator(*CCoef_));
      mC_->Assemble();
   }

   if ( sK_ == NULL )
   {
      sK_ = new ParBilinearForm(H1_FESpace_);
      if ( kCoef_ != NULL )
      {
         sK_->AddDomainIntegrator(new DiffusionIntegrator(*kCoef_));
      }
      else if ( KCoef_ != NULL )
      {
         sK_->AddDomainIntegrator(new DiffusionIntegrator(*KCoef_));
      }
      sK_->Assemble();
   }
   if ( aV_ == NULL )
   {
      aV_ = new ParBilinearForm(H1_FESpace_);
      aV_->AddDomainIntegrator(
         new MixedScalarWeakDivergenceIntegrator(*VCoef_));
      aV_->Assemble();
   }
   if ( dTdt_gf_ == NULL )
   {
      dTdt_gf_ = new ParGridFunction(H1_FESpace_);
   }
   if ( Qs_ == NULL && QCoef_ != NULL )
   {
      Qs_ = new ParLinearForm(H1_FESpace_);
      Qs_->AddDomainIntegrator(new DomainLFIntegrator(*QCoef_));
      Qs_->Assemble();
      rhs_ = new Vector(Qs_->Size());
   }
   /*
   CInvCoef_ = new InverseCoefficient(*CCoef_);
   if ( kCoef_ != NULL ) kInvCoef_ = new InverseCoefficient(*kCoef_);
   if ( KCoef_ != NULL ) KInvCoef_ = new MatrixInverseCoefficient(*KCoef_);
   */
   H1_FESpace_->GetEssentialTrueDofs(*bdr_attr_, ess_bdr_tdofs_);

   init_ = true;
}

void
AdvectionDiffusionTDO::SetTime(const double time)
{
   this->TimeDependentOperator::SetTime(time);

   dTdtBdrCoef_->SetTime(t);

   if ( tdQ_ )
   {
      QCoef_->SetTime(t);
      Qs_->Assemble();
   }

   if ( tdC_ )
   {
      CCoef_->SetTime(t);
      mC_->Assemble();
   }

   if ( tdK_ )
   {
      if ( kCoef_ != NULL ) { kCoef_->SetTime(t); }
      if ( KCoef_ != NULL ) { KCoef_->SetTime(t); }
      sK_->Assemble();
   }

   if ( tdV_ )
   {
      VCoef_->SetTime(t);
      aV_->Assemble();
   }

   if ( ( tdC_ || tdK_ || tdV_ ) && a_ != NULL )
   {
      a_->Assemble();
   }

   newTime_ = true;
}

void
AdvectionDiffusionTDO::initMult() const
{
   if ( tdC_ || MCInv_ == NULL || MCDiag_ == NULL )
   {
      if ( MCInv_ == NULL )
      {
         MCInv_ = new HyprePCG(MC_);
         MCInv_->SetTol(1e-12);
         MCInv_->SetMaxIter(200);
         MCInv_->SetPrintLevel(0);
      }
      else
      {
         MCInv_->SetOperator(MC_);
      }
      if ( MCDiag_ == NULL )
      {
         MCDiag_ = new HypreDiagScale(MC_);
         MCInv_->SetPreconditioner(*MCDiag_);
      }
      else
      {
         MCDiag_->SetOperator(MC_);
      }
   }
}

void
AdvectionDiffusionTDO::Mult(const Vector &T, Vector &dT_dt) const
{
   dT_dt = 0.0;

   sK_->Mult(T, *rhs_);

   *rhs_ -= *Qs_;
   rhs_->Neg();

   dTdt_gf_->ProjectBdrCoefficient(*dTdtBdrCoef_, *bdr_attr_);

   mC_->FormLinearSystem(ess_bdr_tdofs_, *dTdt_gf_, *rhs_, MC_, dTdt_, RHS_);

   this->initMult();

   MCInv_->Mult(RHS_, dTdt_);

   mC_->RecoverFEMSolution(dTdt_, *rhs_, dT_dt);

   multCount_++;
}

void
AdvectionDiffusionTDO::initA(double dt)
{
   if ( kCoef_ != NULL )
   {
      if (dtkCoef_ == NULL)
      {
         dtkCoef_ = new ProductCoefficient(dt, *kCoef_);
      }
      dtkCoef_->SetAConst(dt);
   }
   else
   {
      if (dtKCoef_ == NULL)
      {
         dtKCoef_ = new ScalarMatrixProductCoefficient(dt, *KCoef_);
      }
      dtKCoef_->SetAConst(dt);
   }
   if (nu_ != 0.0)
   {
      if (dtnuVCoef_ == NULL)
      {
         dtnuVCoef_ = new ScalarVectorProductCoefficient(dt * nu_, *VCoef_);
      }
      dtnuVCoef_->SetAConst(dt * nu_);
   }

   if ( a_ == NULL)
   {
      a_ = new ParBilinearForm(H1_FESpace_);
      a_->AddDomainIntegrator(new MassIntegrator(*CCoef_));
      if ( kCoef_ != NULL)
      {
         a_->AddDomainIntegrator(new DiffusionIntegrator(*dtkCoef_));
      }
      else
      {
         a_->AddDomainIntegrator(new DiffusionIntegrator(*dtKCoef_));
      }
      if (nu_ != 0.0)
      {
         a_->AddDomainIntegrator(
            new MixedScalarWeakDivergenceIntegrator(*dtnuVCoef_));
      }
   }
   a_->Update();
   a_->Assemble();
}

void
AdvectionDiffusionTDO::initImplicitSolve()
{
   delete AInv_;
   AInv_ = new HyprePCG(A_);
   AInv_->SetTol(1e-12);
   AInv_->SetMaxIter(200);
   AInv_->SetPrintLevel(0);

   delete APrecond_;
   APrecond_ = new HypreBoomerAMG(A_);
   APrecond_->SetPrintLevel(0);
   AInv_->SetPreconditioner(*APrecond_);
   /*
      if ( tdC_ || tdK_ || AInv_ == NULL || APrecond_ == NULL )
      {
         if ( AInv_ == NULL )
         {
            AInv_ = new HyprePCG(A_);
            AInv_->SetTol(1e-12);
            AInv_->SetMaxIter(200);
            AInv_->SetPrintLevel(0);
         }
         else
         {
            AInv_->SetOperator(A_);
         }
         if ( APrecond_ == NULL )
         {
            APrecond_ = new HypreBoomerAMG(A_);
            APrecond_->SetPrintLevel(0);
            AInv_->SetPreconditioner(*APrecond_);
         }
         else
         {
            APrecond_->SetOperator(A_);
         }
      }
     */
}

void
AdvectionDiffusionTDO::ImplicitSolve(const double dt,
                                     const Vector &T, Vector &dT_dt)
{
   dT_dt = 0.0;
   // cout << "sK size: " << sK_->Width() << ", T size: " << T.Size() << ", rhs_ size: " << rhs_->Size() << endl;

   /*
   ostringstream ossT; ossT << "T_" << solveCount_ << ".vec";
   ofstream ofsT(ossT.str().c_str());
   T.Print(ofsT);
   ofsT.close();
   */
   sK_->Mult(T, *rhs_);
   aV_->AddMult(T, *rhs_);
   /*
   ofstream ofsrhs("rhs.vec");
   rhs_->Print(ofsrhs);

   ofstream ofsQ("Q.vec");
   Qs_->Print(ofsQ);
   */
   *rhs_ -= *Qs_;
   *rhs_ *= -1.0;

   dTdt_gf_->ProjectBdrCoefficient(*dTdtBdrCoef_, *bdr_attr_);

   this->initA(dt);

   a_->FormLinearSystem(ess_bdr_tdofs_, *dTdt_gf_, *rhs_, A_, dTdt_, RHS_);
   /*
   A_.Print("A.mat");
   ofstream ofsB("b.vec");
   RHS_.Print(ofsB);
   */
   this->initImplicitSolve();

   AInv_->Mult(RHS_, dTdt_);

   a_->RecoverFEMSolution(dTdt_, *rhs_, dT_dt);

   solveCount_++;
}

DGAdvectionDiffusionTDO::DGAdvectionDiffusionTDO(const MPI_Session &mpi,
                                                 const DGParams &dg,
                                                 ParFiniteElementSpace &fes,
                                                 ParGridFunction &yGF,
                                                 ParGridFunction &kGF,
                                                 SumCoefficient &ykCoef,
                                                 const AdvectionDiffusionBC &bcs,
                                                 int term_flag,
                                                 int vis_flag,
                                                 bool imex,
                                                 int logging)
   : TimeDependentOperator(fes.GetTrueVSize()),
     mpi_(mpi),
     logging_(logging),
     fes_(fes),
     yGF_(yGF),
     newton_op_prec_(mpi, logging),
     newton_op_solver_(fes.GetComm()),
     newton_solver_(fes.GetComm()),
     op_(mpi, dg, fes, yGF, kGF, ykCoef, bcs,
         term_flag, vis_flag, logging)
{
   if ( mpi_.Root() && logging_ > 1)
   {
      cout << "Constructing DGAdvectionDiffusionTDO" << endl;
   }
   if (logging_ > 2)
   {
      cout << mpi_.WorldRank() << ": TDO size " << width << endl;
   }

   const double rel_tol = 1e-6;

   newton_op_solver_.SetRelTol(rel_tol * 1.0e-4);
   newton_op_solver_.SetAbsTol(0.0);
   newton_op_solver_.SetMaxIter(300);
   newton_op_solver_.SetPrintLevel(3);
   newton_op_solver_.SetPreconditioner(newton_op_prec_);

   newton_solver_.iterative_mode = false;
   newton_solver_.SetSolver(newton_op_solver_);
   newton_solver_.SetOperator(op_);
   newton_solver_.SetPrintLevel(1); // print Newton iterations
   newton_solver_.SetRelTol(rel_tol);
   newton_solver_.SetAbsTol(1e-6);
   newton_solver_.SetMaxIter(10);

   if (mpi_.Root() && logging_ > 1)
   {
      cout << "Done constructing DGAdvectionDiffusionTDO" << endl;
   }
}

DGAdvectionDiffusionTDO::~DGAdvectionDiffusionTDO()
{
   map<string, socketstream*>::iterator mit;
   for (mit=socks_.begin(); mit!=socks_.end(); mit++)
   {
      delete mit->second;
   }
}

void DGAdvectionDiffusionTDO::SetTime(const double _t)
{
   if (mpi_.Root() && logging_ > 1)
   {
      cout << "Entering DGAdvectionDiffusionTDO::SetTime with t = " << _t << endl;
   }
   this->TimeDependentOperator::SetTime(_t);

   if ( mpi_.Root() && logging_ > 1)
   {
      cout << "Leaving DGAdvectionDiffusionTDO::SetTime" << endl;
   }
}

void DGAdvectionDiffusionTDO::SetLogging(int logging)
{
   op_.SetLogging(logging);
}

void
DGAdvectionDiffusionTDO::RegisterDataFields(DataCollection & dc)
{
   dc_ = &dc;

   op_.RegisterDataFields(dc);
}

void
DGAdvectionDiffusionTDO::PrepareDataFields()
{
   op_.PrepareDataFields();
}

void
DGAdvectionDiffusionTDO::InitializeGLVis()
{
   if ( mpi_.Root() && logging_ > 0 )
   { cout << "Opening GLVis sockets." << endl << flush; }
   op_.InitializeGLVis();
}

void
DGAdvectionDiffusionTDO::DisplayToGLVis()
{
   if ( mpi_.Root() && logging_ > 1 )
   { cout << "Sending data to GLVis ..." << flush; }

   op_.DisplayToGLVis();

   if ( mpi_.Root() && logging_ > 1 ) { cout << " " << flush; }
}

void DGAdvectionDiffusionTDO::ImplicitSolve(const double dt, const Vector &y,
                                            Vector &k)
{
   if (mpi_.Root() && logging_ > 1)
   {
      cout << "Entering DGAdvectionDiffusionTDO::ImplicitSolve" << endl;
   }

   k = 0.0;

   // Coefficient inside the NLOperator classes use data from yGF_ to evaluate
   // fields.  We need to make sure this data accesses the provided vector y.
   yGF_.MakeRef(&fes_, y.GetData());
   yGF_.ExchangeFaceNbrData();

   if (mpi_.Root() && logging_ > 0)
   {
      cout << "Setting time step: " << dt << " in DGAdvectionDiffusionTDO"
           << endl;
   }
   op_.SetTimeStep(dt);

   Vector zero;
   newton_solver_.Mult(zero, k);

   if (mpi_.Root() && logging_ > 1)
   {
      cout << "Leaving DGAdvectionDiffusionTDO::ImplicitSolve" << endl;
   }
}

void DGAdvectionDiffusionTDO::Update()
{
   if (mpi_.Root() && logging_ > 1)
   {
      cout << "Entering DGAdvectionDiffusionTDO::Update" << endl;
   }

   height = width = fes_.GetTrueVSize();

   op_.Update();

   newton_solver_.SetOperator(op_);

   if (mpi_.Root() && logging_ > 1)
   {
      cout << "Leaving DGAdvectionDiffusionTDO::Update" << endl;
   }
}

void
DGAdvectionDiffusionTDO::ADPrec::SetOperator(const Operator &op)
{
   if (mpi_.Root() && logging_)
   {
      cout << "ADPrec::SetOperator"
           << endl;
   }

   height = width = op.Height();

   delete prec_;

   const HypreParMatrix & M =
      dynamic_cast<const HypreParMatrix&>(op);

   HypreBoomerAMG * amg =
      new HypreBoomerAMG(const_cast<HypreParMatrix&>(M));
   amg->SetPrintLevel(0);
   prec_ = amg;

}

DGAdvectionDiffusionTDO::NLOperator::NLOperator(const MPI_Session & mpi,
                                                const DGParams & dg,
                                                ParFiniteElementSpace &fes,
                                                ParGridFunction & yGF,
                                                ParGridFunction & kGF,
                                                SumCoefficient &ykCoef,
                                                int term_flag, int vis_flag,
                                                int logging,
                                                const string & log_prefix)
   : Operator(fes.GetTrueVSize(),
              fes.GetTrueVSize()),
     mpi_(mpi), dg_(dg),
     logging_(logging), log_prefix_(log_prefix),
     dt_(0.0),
     fes_(fes),
     pmesh_(*fes_.GetParMesh()),
     yGF_(yGF),
     kGF_(kGF),
     yCoef_(&yGF_),
     kCoef_(&kGF_),
     ykCoef_(ykCoef),
     dbfi_m_(NULL),
     blf_(NULL),
     term_flag_(term_flag),
     vis_flag_(vis_flag),
     dc_(NULL)
{
   if ( mpi_.Root() && logging_ > 1)
   {
      cout << "Constructing NLOperator" << endl;
   }

   MFEM_VERIFY(yGF.Size() == kGF.Size(), "Mismatch in yGF and kGF sizes");

   if (vis_flag_ < 0) { vis_flag_ = this->GetDefaultVisFlag(); }

   if ( mpi_.Root() && logging_ > 1)
   {
      cout << "Done constructing NLOperator" << endl;
   }
}

DGAdvectionDiffusionTDO::NLOperator::~NLOperator()
{
   delete blf_;
   delete dbfi_m_;

   for (int i=0; i<dbfi_.Size(); i++)
   {
      delete dbfi_[i];
   }
   for (int i=0; i<fbfi_.Size(); i++)
   {
      delete fbfi_[i];
   }
   for (int i=0; i<bfbfi_.Size(); i++)
   {
      delete bfbfi_[i];
   }
   for (int i=0; i<bfbfi_marker_.Size(); i++)
   {
      delete bfbfi_marker_[i];
   }
   for (int i=0; i<dlfi_.Size(); i++)
   {
      delete dlfi_[i];
   }
   for (int i=0; i<bflfi_.Size(); i++)
   {
      delete bflfi_[i];
   }
   for (int i=0; i<bflfi_marker_.Size(); i++)
   {
      delete bflfi_marker_[i];
   }
}

void DGAdvectionDiffusionTDO::NLOperator::SetLogging(int logging,
                                                     const string & prefix)
{
   logging_ = logging;
   log_prefix_ = prefix;
}

void DGAdvectionDiffusionTDO::NLOperator::SetTimeStep(double dt)
{
   dt_ = dt;

   ykCoef_.SetBeta(dt_);
}

void DGAdvectionDiffusionTDO::NLOperator::Mult(const Vector &k,
                                               Vector &r) const
{
   if (mpi_.Root() && logging_ > 1)
   {
      cout << log_prefix_ << "DGAdvectionDiffusionTDO::NLOperator::Mult"
	   << endl;
   }

   r = 0.0;

   kGF_.MakeRef(&fes_, k.GetData());
   kGF_.ExchangeFaceNbrData();

   ElementTransformation *eltrans = NULL;

   for (int i=0; i < fes_.GetNE(); i++)
   {
      fes_.GetElementVDofs(i, vdofs_);

      const FiniteElement &fe = *fes_.GetFE(i);
      eltrans = fes_.GetElementTransformation(i);

      int ndof = vdofs_.Size();

      elvec_.SetSize(ndof);
      locdvec_.SetSize(ndof);

      elvec_ = 0.0;

      if (dbfi_m_ != NULL)
      {
         kGF_.GetSubVector(vdofs_, locdvec_);

         dbfi_m_->AssembleElementMatrix(fe, *eltrans, elmat_);

         elmat_.AddMult(locdvec_, elvec_);
      }
      r.AddElementVector(vdofs_, elvec_);
   }

   if (mpi_.Root() && logging_ > 2)
   {
      cout << log_prefix_
           << "DGAdvectionDiffusionTDO::NLOperator::Mult element loop done"
	   << endl;
   }
   if (dbfi_.Size())
   {
      ElementTransformation *eltrans = NULL;

      for (int i=0; i < fes_.GetNE(); i++)
      {
         fes_.GetElementVDofs(i, vdofs_);

         const FiniteElement &fe = *fes_.GetFE(i);
         eltrans = fes_.GetElementTransformation(i);

         int ndof = vdofs_.Size();

         elvec_.SetSize(ndof);
         locvec_.SetSize(ndof);
         locdvec_.SetSize(ndof);

         yGF_.GetSubVector(vdofs_, locvec_);
         kGF_.GetSubVector(vdofs_, locdvec_);

         locvec_.Add(dt_, locdvec_);

         dbfi_[0]->AssembleElementMatrix(fe, *eltrans, elmat_);
         for (int j = 1; j < dbfi_.Size(); j++)
         {
            dbfi_[j]->AssembleElementMatrix(fe, *eltrans, elmat_j_);
            elmat_ += elmat_j_;
         }

         elmat_.Mult(locvec_, elvec_);

         r.AddElementVector(vdofs_, elvec_);
      }
   }

   if (mpi_.Root() && logging_ > 2)
   {
      cout << log_prefix_
           << "DGAdvectionDiffusionTDO::NLOperator::Mult element loop done"
	   << endl;
   }
   if (fbfi_.Size())
   {
      FaceElementTransformations *ftrans = NULL;

      for (int i = 0; i < pmesh_.GetNumFaces(); i++)
      {
         ftrans = pmesh_.GetInteriorFaceTransformations(i);
         if (ftrans != NULL)
         {
            fes_.GetElementVDofs(ftrans->Elem1No, vdofs_);
            fes_.GetElementVDofs(ftrans->Elem2No, vdofs2_);
            vdofs_.Append(vdofs2_);

            const FiniteElement &fe1 = *fes_.GetFE(ftrans->Elem1No);
            const FiniteElement &fe2 = *fes_.GetFE(ftrans->Elem2No);

            fbfi_[0]->AssembleFaceMatrix(fe1, fe2, *ftrans, elmat_);
            for (int j = 1; j < fbfi_.Size(); j++)
            {
               fbfi_[j]->AssembleFaceMatrix(fe1, fe2, *ftrans, elmat_j_);
               elmat_ += elmat_j_;
            }

            int ndof = vdofs_.Size();

            elvec_.SetSize(ndof);
            locvec_.SetSize(ndof);
            locdvec_.SetSize(ndof);

            yGF_.GetSubVector(vdofs_, locvec_);
            kGF_.GetSubVector(vdofs_, locdvec_);

            locvec_.Add(dt_, locdvec_);

            elmat_.Mult(locvec_, elvec_);

            r.AddElementVector(vdofs_, elvec_);
         }
      }

      Vector elvec(NULL, 0);
      Vector locvec1(NULL, 0);
      Vector locvec2(NULL, 0);
      Vector locdvec1(NULL, 0);
      Vector locdvec2(NULL, 0);

      // DenseMatrix elmat(NULL, 0, 0);

      int nsfaces = pmesh_.GetNSharedFaces();
      for (int i = 0; i < nsfaces; i++)
      {
         ftrans = pmesh_.GetSharedFaceTransformations(i);
         int nbr_el_no = ftrans->Elem2No - pmesh_.GetNE();
         fes_.GetElementVDofs(ftrans->Elem1No, vdofs_);
         fes_.GetFaceNbrElementVDofs(nbr_el_no, vdofs2_);

         for (int j = 0; j < fbfi_.Size(); j++)
         {
            fbfi_[j]->AssembleFaceMatrix(*fes_.GetFE(ftrans->Elem1No),
                                         *fes_.GetFaceNbrFE(nbr_el_no),
                                         *ftrans, elmat_);

            int ndof  = vdofs_.Size();
            int ndof2 = vdofs2_.Size();

            elvec_.SetSize(ndof+ndof2);
            locvec_.SetSize(ndof+ndof2);
            locdvec_.SetSize(ndof+ndof2);

            elvec.SetDataAndSize(&elvec_[0], ndof);

            locvec1.SetDataAndSize(&locvec_[0], ndof);
            locvec2.SetDataAndSize(&locvec_[ndof], ndof2);

            locdvec1.SetDataAndSize(&locdvec_[0], ndof);
            locdvec2.SetDataAndSize(&locdvec_[ndof], ndof2);

            yGF_.GetSubVector(vdofs_, locvec1);
            kGF_.GetSubVector(vdofs_, locdvec1);

            yGF_.FaceNbrData().GetSubVector(vdofs2_, locvec2);
            kGF_.FaceNbrData().GetSubVector(vdofs2_, locdvec2);

            locvec_.Add(dt_, locdvec_);

            elmat_.Mult(locvec_, elvec_);

            r.AddElementVector(vdofs_, elvec);
         }
      }
   }

   if (mpi_.Root() && logging_ > 2)
   {
      cout << log_prefix_
           << "DGAdvectionDiffusionTDO::NLOperator::Mult face loop done"
	   << endl;
   }
   if (bfbfi_.Size())
   {
      FaceElementTransformations *ftrans = NULL;

      // Which boundary attributes need to be processed?
      Array<int> bdr_attr_marker(pmesh_.bdr_attributes.Size() ?
                                 pmesh_.bdr_attributes.Max() : 0);
      bdr_attr_marker = 0;
      for (int j = 0; j < bfbfi_.Size(); j++)
      {
         if (bfbfi_marker_[j] == NULL)
         {
            bdr_attr_marker = 1;
            break;
         }
         const Array<int> &bdr_marker = *bfbfi_marker_[j];
         MFEM_ASSERT(bdr_marker.Size() == bdr_attr_marker.Size(),
                     "invalid boundary marker for boundary face integrator #"
                     << j << ", counting from zero");
         for (int i = 0; i < bdr_attr_marker.Size(); i++)
         {
            bdr_attr_marker[i] |= bdr_marker[i];
         }
      }

      for (int i = 0; i < fes_.GetNBE(); i++)
      {
         const int bdr_attr = pmesh_.GetBdrAttribute(i);
         if (bdr_attr_marker[bdr_attr-1] == 0) { continue; }

         ftrans = pmesh_.GetBdrFaceTransformations(i);
         if (ftrans != NULL)
         {
            fes_.GetElementVDofs(ftrans->Elem1No, vdofs_);

            int ndof = vdofs_.Size();

            const FiniteElement &fe1 = *fes_.GetFE(ftrans->Elem1No);
            const FiniteElement &fe2 = fe1;

            elmat_.SetSize(ndof);
            elmat_ = 0.0;

            for (int j = 0; j < bfbfi_.Size(); j++)
            {
               if (bfbfi_marker_[j] != NULL)
                  if ((*bfbfi_marker_[j])[bdr_attr-1] == 0) { continue; }

               bfbfi_[j]->AssembleFaceMatrix(fe1, fe2, *ftrans, elmat_j_);
               elmat_ += elmat_j_;
            }

            elvec_.SetSize(ndof);
            locvec_.SetSize(ndof);
            locdvec_.SetSize(ndof);

            yGF_.GetSubVector(vdofs_, locvec_);
            kGF_.GetSubVector(vdofs_, locdvec_);

            locvec_.Add(dt_, locdvec_);

            elmat_.Mult(locvec_, elvec_);

            r.AddElementVector(vdofs_, elvec_);
         }
      }
   }

   if (dlfi_.Size())
   {
      ElementTransformation *eltrans = NULL;
      const FiniteElement *fe = NULL;

      for (int i=0; i < fes_.GetNE(); i++)
      {
         fes_.GetElementVDofs(i, vdofs_);
         eltrans = fes_.GetElementTransformation(i);
         fe = fes_.GetFE(i);

         int ndof = vdofs_.Size();
         elvec_.SetSize(ndof);

         for (int j=0; j < dlfi_.Size(); j++)
         {
            dlfi_[j]->AssembleRHSElementVect(*fe, *eltrans, elvec_);
            elvec_ *= -1.0;
            r.AddElementVector (vdofs_, elvec_);
         }
      }
   }

   if (bflfi_.Size())
   {
      FaceElementTransformations *tr;
      Mesh *mesh = fes_.GetMesh();

      // Which boundary attributes need to be processed?
      Array<int> bdr_attr_marker(mesh->bdr_attributes.Size() ?
                                 mesh->bdr_attributes.Max() : 0);
      bdr_attr_marker = 0;
      for (int j = 0; j < bflfi_.Size(); j++)
      {
         if (bflfi_marker_[j] == NULL)
         {
            bdr_attr_marker = 1;
            break;
         }
         const Array<int> &bdr_marker = *bflfi_marker_[j];
         MFEM_ASSERT(bdr_marker.Size() == bdr_attr_marker.Size(),
                     "invalid boundary marker for boundary face integrator #"
                     << j << ", counting from zero");
         for (int i = 0; i < bdr_attr_marker.Size(); i++)
         {
            bdr_attr_marker[i] |= bdr_marker[i];
         }
      }

      for (int i = 0; i < mesh->GetNBE(); i++)
      {
         const int bdr_attr = mesh->GetBdrAttribute(i);
         if (bdr_attr_marker[bdr_attr-1] == 0) { continue; }

         tr = mesh->GetBdrFaceTransformations(i);
         if (tr != NULL)
         {
            fes_.GetElementVDofs (tr -> Elem1No, vdofs_);

            int ndof = vdofs_.Size();
            elvec_.SetSize(ndof);

            for (int j = 0; j < bflfi_.Size(); j++)
            {
               if (bflfi_marker_[j] &&
                   (*bflfi_marker_[j])[bdr_attr-1] == 0) { continue; }

               bflfi_[j] -> AssembleRHSElementVect (*fes_.GetFE(tr -> Elem1No),
                                                    *tr, elvec_);
               elvec_ *= -1.0;
               r.AddElementVector (vdofs_, elvec_);
            }
         }
      }
   }

   if (mpi_.Root() && logging_ > 1)
   {
      cout << log_prefix_
           << "DGAdvectionDiffusionTDO::NLOperator::Mult done" << endl;
   }
}

void DGAdvectionDiffusionTDO::NLOperator::Update()
{
   if (mpi_.Root() && logging_ > 1)
   {
      cout << "Entering DGAdvectionDiffusionTDO::NLOperator::Update" << endl;
   }

   height = fes_.GetTrueVSize();
   width  = fes_.GetTrueVSize();

   if (blf_ != NULL)
   {
      blf_->Update();
   }

   if (mpi_.Root() && logging_ > 1)
   {
      cout << "Leaving DGAdvectionDiffusionTDO::NLOperator::Update" << endl;
   }
}

Operator &
DGAdvectionDiffusionTDO::NLOperator::GetGradient(const Vector &k) const
{
   if (mpi_.Root() && logging_ > 1)
   {
      cout << "Entering DGAdvectionDiffusionTDO::NLOperator::GetGradient()"
           << endl;
   }

   MFEM_VERIFY(blf_ != NULL, "The Bilinear Form object is NULL!");

   kGF_.MakeRef(&fes_, k.GetData());
   kGF_.ExchangeFaceNbrData();

   blf_->Update();
   blf_->Assemble(0);
   blf_->Finalize(0);
   blf_op_ = blf_->ParallelAssemble();

   return *blf_op_;
}

void
DGAdvectionDiffusionTDO::NLOperator::RegisterDataFields(DataCollection & dc)
{
   dc_ = &dc;

   if (this->CheckVisFlag(0))
   {
      dc.RegisterField(field_name_, &yGF_);
   }
}

void
DGAdvectionDiffusionTDO::NLOperator::PrepareDataFields()
{
}

DGAdvectionDiffusionTDO::AdvectionDiffusionOp::AdvectionDiffusionOp(
   const MPI_Session & mpi, const DGParams & dg,
   ParFiniteElementSpace &fes,
   ParGridFunction & yGF,
   ParGridFunction & kGF,
   SumCoefficient &ykCoef,
   const AdvectionDiffusionBC & bcs,
   int term_flag, int vis_flag,
   int logging,
   const std::string & log_prefix)
   : NLOperator(mpi, dg, fes, yGF, kGF, ykCoef, term_flag, vis_flag,
                logging, log_prefix),
     coefGF_(&fes),
     bcs_(bcs)
{

}

DGAdvectionDiffusionTDO::AdvectionDiffusionOp::~AdvectionDiffusionOp()
{
   for (int i=0; i<coefs_.Size(); i++)
   {
      delete coefs_[i];
   }

   for (int i=0; i<dtSCoefs_.Size(); i++)
   {
      delete dtSCoefs_[i];
   }
   for (int i=0; i<negdtSCoefs_.Size(); i++)
   {
      delete negdtSCoefs_[i];
   }
   for (int i=0; i<dtVCoefs_.Size(); i++)
   {
      delete dtVCoefs_[i];
   }
   for (int i=0; i<dtMCoefs_.Size(); i++)
   {
      delete dtMCoefs_[i];
   }
   for (unsigned int i=0; i<sout_.size(); i++)
   {
      delete sout_[i];
   }
}

void DGAdvectionDiffusionTDO::AdvectionDiffusionOp::SetTimeStep(double dt)
{
   if (mpi_.Root() && logging_)
   {
      cout << "Setting time step: " << dt << " in AdvectionDiffusionOp"
           << endl;
   }

   NLOperator::SetTimeStep(dt);

   for (int i=0; i<dtSCoefs_.Size(); i++)
   {
      dtSCoefs_[i]->SetAConst(dt);
   }
   for (int i=0; i<negdtSCoefs_.Size(); i++)
   {
      negdtSCoefs_[i]->SetAConst(-dt);
   }
   for (int i=0; i<dtVCoefs_.Size(); i++)
   {
      dtVCoefs_[i]->SetAConst(dt);
   }
   for (int i=0; i<dtMCoefs_.Size(); i++)
   {
      dtMCoefs_[i]->SetAConst(dt);
   }
}

void DGAdvectionDiffusionTDO::AdvectionDiffusionOp::Update()
{
   if (mpi_.Root() && logging_ > 1)
   {
      cout << "Entering DGAdvectionDiffusionTDO::AdvectionDiffusionOp::Update"
           << endl;
   }

   coefGF_.Update();

   this->NLOperator::Update();

   if (mpi_.Root() && logging_ > 1)
   {
      cout << "Leaving DGAdvectionDiffusionTDO::AdvectionDiffusionOp::Update"
           << endl;
   }
}

void DGAdvectionDiffusionTDO::AdvectionDiffusionOp::SetTimeDerivativeTerm(
   StateVariableCoef &MCoef)
{
   if ( mpi_.Root() && logging_ > 0)
   {
      cout << field_name_
           << ": Adding time derivative term proportional to d "
           << FieldSymbol(FieldType::TEMPERATURE) << " / dt" << endl;
   }

   StateVariableCoef * coef = MCoef.Clone();
   // coef->SetDerivType(FieldType::TEMPERATURE);
   coefs_.Append(coef);

   delete dbfi_m_;
   dbfi_m_ = new MassIntegrator(*coef);

   if (blf_ == NULL)
   {
      blf_ = new ParBilinearForm(&fes_);
   }
   blf_->AddDomainIntegrator(new MassIntegrator(*coef));
}

void DGAdvectionDiffusionTDO::AdvectionDiffusionOp::SetDiffusionTerm(
   StateVariableCoef &DCoef)
{
   if ( mpi_.Root() && logging_ > 0)
   {
      cout << field_name_ << ": Adding isotropic diffusion term" << endl;
   }

   ProductCoefficient * dtDCoef = new ProductCoefficient(dt_, DCoef);
   dtSCoefs_.Append(dtDCoef);

   dbfi_.Append(new DiffusionIntegrator(DCoef));
   fbfi_.Append(new DGDiffusionIntegrator(DCoef,
					  dg_.sigma,
					  dg_.kappa));

   if (blf_ == NULL)
   {
      blf_ = new ParBilinearForm(&fes_);
   }

   blf_->AddDomainIntegrator(new DiffusionIntegrator(*dtDCoef));
   blf_->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(*dtDCoef,
							     dg_.sigma,
							     dg_.kappa));

   const Array<CoefficientByAttr*> & dbc = bcs_.GetDirichletBCs();
   for (unsigned int i=0; i<dbc.Size(); i++)
   {
     bfbfi_marker_.Append(new Array<int>);
     AttrToMarker(pmesh_.bdr_attributes.Max(), dbc[i]->attr,
		  *bfbfi_marker_.Last());

     blf_->AddBdrFaceIntegrator(new DGDiffusionIntegrator(*dtDCoef,
							  dg_.sigma,
							  dg_.kappa),
				*bfbfi_marker_.Last());
     
     bfbfi_.Append(new DGDiffusionIntegrator(DCoef,
					     dg_.sigma,
					     dg_.kappa));

     bflfi_marker_.Append(new Array<int>);
     AttrToMarker(pmesh_.bdr_attributes.Max(), dbc[i]->attr,
		  *bflfi_marker_.Last());
     
     bflfi_.Append(new DGDirichletLFIntegrator(*dbc[i]->coef, DCoef,
					       dg_.sigma,
					       dg_.kappa));
   }

   const Array<CoefficientByAttr*> & nbc = bcs_.GetNeumannBCs();
   for (unsigned int i=0; i<nbc.Size(); i++)
   {
      bflfi_marker_.Append(new Array<int>);
      AttrToMarker(pmesh_.bdr_attributes.Max(), nbc[i]->attr,
                   *bflfi_marker_.Last());
      coefs_.Append(new ProductCoefficient(DCoef, *nbc[i]->coef));
      bflfi_.Append(new BoundaryLFIntegrator(*coefs_.Last()));
   }

   const Array<CoefficientsByAttr*> & rbc = bcs_.GetRobinBCs();
   for (unsigned int i=0; i<rbc.Size(); i++)
   {
      bfbfi_marker_.Append(new Array<int>);
      AttrToMarker(pmesh_.bdr_attributes.Max(), rbc[i]->attr,
                   *bfbfi_marker_.Last());
      bfbfi_.Append(new BoundaryMassIntegrator(*rbc[i]->coefs[0]));

      bflfi_marker_.Append(new Array<int>);
      AttrToMarker(pmesh_.bdr_attributes.Max(), rbc[i]->attr,
                   *bflfi_marker_.Last());
      bflfi_.Append(new BoundaryLFIntegrator(*rbc[i]->coefs[1]));

      ProductCoefficient * dtaCoef = new ProductCoefficient(dt_,
                                                            *rbc[i]->coefs[0]);
      dtSCoefs_.Append(dtaCoef);

      blf_->AddBdrFaceIntegrator(new BoundaryMassIntegrator(*dtaCoef),
                                 *bfbfi_marker_.Last());
   }
}

void DGAdvectionDiffusionTDO::AdvectionDiffusionOp::SetDiffusionTerm(
   StateVariableMatCoef &DCoef)
{
   if ( mpi_.Root() && logging_ > 0)
   {
      cout << field_name_ << ": Adding anisotropic diffusion term" << endl;
   }

   ScalarMatrixProductCoefficient * dtDCoef =
      new ScalarMatrixProductCoefficient(dt_, DCoef);
   dtMCoefs_.Append(dtDCoef);

   dbfi_.Append(new DiffusionIntegrator(DCoef));
   fbfi_.Append(new DGDiffusionIntegrator(DCoef,
                                          dg_.sigma,
                                          dg_.kappa));

   if (blf_ == NULL)
   {
      blf_ = new ParBilinearForm(&fes_);
   }

   blf_->AddDomainIntegrator(new DiffusionIntegrator(*dtDCoef));
   blf_->AddInteriorFaceIntegrator(
      new DGDiffusionIntegrator(*dtDCoef,
                                dg_.sigma,
                                dg_.kappa));

   const Array<CoefficientByAttr*> & dbc = bcs_.GetDirichletBCs();
   for (unsigned int i=0; i<dbc.Size(); i++)
   {
      bfbfi_marker_.Append(new Array<int>);
      AttrToMarker(pmesh_.bdr_attributes.Max(), dbc[i]->attr,
                   *bfbfi_marker_.Last());
      bfbfi_.Append(new DGDiffusionIntegrator(DCoef,
                                              dg_.sigma,
                                              dg_.kappa));

      bflfi_marker_.Append(new Array<int>);
      AttrToMarker(pmesh_.bdr_attributes.Max(), dbc[i]->attr,
                   *bflfi_marker_.Last());
      bflfi_.Append(new DGDirichletLFIntegrator(*dbc[i]->coef, DCoef,
                                                dg_.sigma,
                                                dg_.kappa));

      blf_->AddBdrFaceIntegrator(new DGDiffusionIntegrator(*dtDCoef,
                                                           dg_.sigma,
                                                           dg_.kappa),
                                 *bfbfi_marker_.Last());
   }

   const Array<CoefficientByAttr*> & nbc = bcs_.GetNeumannBCs();
   for (unsigned int i=0; i<nbc.Size(); i++)
   {
      bflfi_marker_.Append(new Array<int>);
      AttrToMarker(pmesh_.bdr_attributes.Max(), nbc[i]->attr,
                   *bflfi_marker_.Last());
      bflfi_.Append(new BoundaryLFIntegrator(*nbc[i]->coef));
   }

   const Array<CoefficientsByAttr*> & rbc = bcs_.GetRobinBCs();
   for (unsigned int i=0; i<rbc.Size(); i++)
   {
      bfbfi_marker_.Append(new Array<int>);
      AttrToMarker(pmesh_.bdr_attributes.Max(), rbc[i]->attr,
                   *bfbfi_marker_.Last());
      bfbfi_.Append(new BoundaryMassIntegrator(*rbc[i]->coefs[0]));

      bflfi_marker_.Append(new Array<int>);
      AttrToMarker(pmesh_.bdr_attributes.Max(), rbc[i]->attr,
                   *bflfi_marker_.Last());
      bflfi_.Append(new BoundaryLFIntegrator(*rbc[i]->coefs[1]));

      ProductCoefficient * dtaCoef = new ProductCoefficient(dt_,
                                                            *rbc[i]->coefs[0]);
      dtSCoefs_.Append(dtaCoef);

      blf_->AddBdrFaceIntegrator(new BoundaryMassIntegrator(*dtaCoef),
                                 *bfbfi_marker_.Last());
   }
}

void DGAdvectionDiffusionTDO::AdvectionDiffusionOp::SetAdvectionTerm(
   StateVariableVecCoef &VCoef,
   bool bc)
{
   if ( mpi_.Root() && logging_ > 0)
   {
      cout << field_name_ << ": Adding advection term" << endl;
   }

   ScalarVectorProductCoefficient * dtVCoef =
      new ScalarVectorProductCoefficient(dt_, VCoef);
   dtVCoefs_.Append(dtVCoef);

   dbfi_.Append(new MixedScalarWeakDivergenceIntegrator(VCoef));
   fbfi_.Append(new DGTraceIntegrator(VCoef, 1.0, -0.5));

   if (bc)
   {
      bfbfi_.Append(new DGTraceIntegrator(VCoef, 1.0, -0.5));
      bfbfi_marker_.Append(NULL);
   }

   if (blf_ == NULL)
   {
      blf_ = new ParBilinearForm(&fes_);
   }

   blf_->AddDomainIntegrator(
      new MixedScalarWeakDivergenceIntegrator(*dtVCoef));
   blf_->AddInteriorFaceIntegrator(new DGTraceIntegrator(*dtVCoef,
                                                         1.0, -0.5));

   if (bc)
   {
      blf_->AddBdrFaceIntegrator(new DGTraceIntegrator(*dtVCoef,
                                                       1.0, -0.5));
   }
}

void DGAdvectionDiffusionTDO::AdvectionDiffusionOp::SetSourceTerm(
   StateVariableCoef &SCoef)
{
   if ( mpi_.Root() && logging_ > 0)
   {
      cout << field_name_ << ": Adding source term" << endl;
   }

   dlfi_.Append(new DomainLFIntegrator(SCoef));

   if (SCoef.NonTrivialValue(FieldType::TEMPERATURE))
   {
      StateVariableCoef * coef = SCoef.Clone();
      coef->SetDerivType(FieldType::TEMPERATURE);
      ProductCoefficient * dtdSCoef = new ProductCoefficient(-dt_, *coef);
      negdtSCoefs_.Append(dtdSCoef);

      if (blf_ == NULL)
      {
         blf_ = new ParBilinearForm(&fes_);
      }
      blf_->AddDomainIntegrator(new MassIntegrator(*dtdSCoef));
   }
}

void
DGAdvectionDiffusionTDO::AdvectionDiffusionOp::InitializeGLVis()
{
   if ((int)sout_.size() < coefs_.Size())
   {
      sout_.resize(coefs_.Size());
      for (int i=0; i<coefs_.Size(); i++)
      {
         sout_[i] = new socketstream;
      }
   }
}

void
DGAdvectionDiffusionTDO::AdvectionDiffusionOp::DisplayToGLVis()
{
   for (int i=0; i<coefs_.Size(); i++)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;

      int Wx = 0, Wy = 0; // window position
      int Ww = 275, Wh = 250; // window size
      int Dx = 3, Dy = 25;

      ostringstream oss;
      oss << "coef " << index_ << " " << i + 1;

      coefGF_.ProjectCoefficient(*coefs_[i]);

      int c = i % 4;
      int r = i / 4;
      VisualizeField(*sout_[i], vishost, visport, coefGF_, oss.str().c_str(),
                     Wx + c * (Ww + Dx), Wy + r * (Wh + Dy), Ww, Wh);
   }
}

} // namespace thermal

} // namespace mfem

#endif // MFEM_USE_MPI
