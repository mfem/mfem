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

#include "transport_solver.hpp"
#include "../../general/text.hpp"

#include "schwarz.hpp"

#ifdef MFEM_USE_MPI

using namespace std;
namespace mfem
{
using namespace common;

namespace plasma
{

namespace transport
{
/*
double tau_e(double Te, int ns, double * ni, int * zi, double lnLambda)
{
   double tau = 0.0;
   for (int i=0; i<ns; i++)
   {
      tau += meanElectronIonCollisionTime(Te, ni[i], zi[i], lnLambda);
   }
   return tau;
}

double tau_i(double ma, double Ta, int ion, int ns, double * ni, int * zi,
             double lnLambda)
{
   double tau = 0.0;
   for (int i=0; i<ns; i++)
   {
      tau += meanIonIonCollisionTime(ma, Ta, ni[i], zi[ion], zi[i], lnLambda);
   }
   return tau;
}
*/
std::string FieldSymbol(FieldType t)
{
   switch (t)
   {
      case NEUTRAL_DENSITY:
         return "n_n";
      case ION_DENSITY:
         return "n_i";
      case ION_PARA_VELOCITY:
         return "v_i";
      case ION_TEMPERATURE:
         return "T_i";
      case ELECTRON_TEMPERATURE:
         return "T_e";
      default:
         return "N/A";
   }
}

void ElementOrder(ParFiniteElementSpace &fes, Vector &elemOrder)
{
   MFEM_ASSERT(elemOrder.Size() == fes.GetNE(),
               "Incorrect size for result vector");

   for (int i = 0; i < fes.GetNE(); i++)
   {
      elemOrder[i] = fes.GetElementOrder(i);
   }
}

void DiscontinuitySensor(GridFunction &u, double uRef, double alpha,
                         Vector &disc)
{
   FiniteElementSpace *fes = u.FESpace();

   MFEM_ASSERT(disc.Size() == fes->GetNE(),
               "Incorrect size for result vector");

   const FiniteElement *fe, *fe_lo;
   const FiniteElementCollection *fec = fes->FEColl();
   ElementTransformation *T;
   Array<int> dofs;

   for (int i = 0; i < fes->GetNE(); i++)
   {
      fe = fes->GetFE(i);
      fe_lo = fec->GetFE(fe->GetGeomType(), fe->GetOrder() - 1);

      int dof = fe->GetDof();
      int dof_lo = fe_lo->GetDof();

      const IntegrationRule *ir;
      int intorder = 2*fe->GetOrder() + 3; // <----------
      ir = &(IntRules.Get(fe->GetGeomType(), intorder));

      T = fes->GetElementTransformation(i);

      Vector DofVal(dof), loc_data(dof);
      Vector DofVal_lo(dof_lo), loc_data_lo(dof_lo);

      DofTransformation * doftrans = fes->GetElementDofs(i, dofs);
      u.GetSubVector(dofs, loc_data);
      if (doftrans)
      {
         doftrans->InvTransformPrimal(loc_data);
      }

      DenseMatrix I;
      fe_lo->Project(*fe, *T, I);
      I.Mult(loc_data, loc_data_lo);

      double norm2 = 0.0;
      double disc_val = 0.0;

      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         T->SetIntPoint(&ip);

         double val = 0.0, val_lo = 0.0;
         if (fe->GetMapType() == FiniteElement::VALUE)
         {
            fe->CalcShape(ip, DofVal);
            val = DofVal * loc_data;

            fe_lo->CalcShape(ip, DofVal_lo);
            val_lo = DofVal_lo * loc_data_lo;
         }
         else
         {
            fe->CalcPhysShape(*T, DofVal);
            val = DofVal * loc_data;

            fe_lo->CalcPhysShape(*T, DofVal_lo);
            val_lo = DofVal_lo * loc_data_lo;
         }

         disc_val += ip.weight * T->Weight() * pow(fabs(val - val_lo), 2);
         norm2 += ip.weight * T->Weight() * val * val;
      }

      norm2 = fabs(norm2);
      if (norm2 > uRef * uRef * 1e-8)
      {
         disc_val /= norm2;
      }
      else
      {
         disc_val = DBL_MIN;
      }
      disc[i] = (1.0 - alpha) * disc[i] + alpha * disc_val;
   }
}

void ParallelMeshSpacing(ParFiniteElementSpace &fes, VectorCoefficient &B3Coef,
                         Vector &hEffective)
{
   Vector B3(3);
   Vector B2(B3.GetData(), 2);
   Vector JB(2);

   for (int i=0; i<fes.GetNE(); i++)
   {
      ElementTransformation &T = *fes.GetElementTransformation(i);
      const IntegrationPoint &ip = Geometries.GetCenter(T.GetGeometryType());

      int elemOrder = fes.GetElementOrder(i);

      B3Coef.Eval(B3, T, ip);
      double B2mag2 = B2 * B2;

      T.Jacobian().MultTranspose(B2, JB);
      double h = sqrt((JB * JB) / B2mag2);

      hEffective[i] = h / elemOrder;
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
   for (int i=0; i<obc.Size(); i++)
   {
      if (obc[i]->ownCoef)
      {
         delete obc[i]->coef;
      }
      delete obc[i];
   }
}

void AdvectionDiffusionBC::SetTime(double t) const
{
   for (int i=0; i<dbc.Size(); i++)
   {
      if (dbc[i]->coef)
      {
         dbc[i]->coef->SetTime(t);;
      }
   }
   for (int i=0; i<nbc.Size(); i++)
   {
      if (nbc[i]->coef)
      {
         nbc[i]->coef->SetTime(t);
      }
   }
   for (int i=0; i<rbc.Size(); i++)
   {
      for (int j=0; j<rbc[i]->coefs.Size(); j++)
      {
         if (rbc[i]->coefs[j])
         {
            rbc[i]->coefs[j]->SetTime(t);
         }
      }
   }
   for (int i=0; i<obc.Size(); i++)
   {
      if (obc[i]->coef)
      {
         obc[i]->coef->SetTime(t);
      }
   }
}

const char * AdvectionDiffusionBC::GetBCTypeName(BCType bctype)
{
   switch (bctype)
   {
      case DIRICHLET_BC: return "Dirichlet";
      case   NEUMANN_BC: return "Neumann";
      case     ROBIN_BC: return "Robin";
      case   OUTFLOW_BC: return "Outflow";
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
      if (bctype != OUTFLOW_BC)
      {
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
      }
      else
      {
         if (obc_attr.Find(b) == -1)
         {
            obc_attr.Append(b);
         }
         else
         {
            MFEM_ABORT("Attempting to add a " << GetBCTypeName(bctype)
                       << " BC on boundary " << b
                       << " which already has an outflow boundary condition"
                       << " defined.");
         }
      }
      attr.Append(b);
   }
}

void AdvectionDiffusionBC::ReadCoefByAttr(std::istream &input,
                                          BCType bctype,
                                          CoefficientByAttr &cba)
{
   ReadAttr(input, bctype, cba.attr);
   cba.coef = coefFact->GetScalarCoef(input);
   cba.ownCoef = false;
}

void AdvectionDiffusionBC::ReadCoefsByAttr(std::istream &input,
                                           BCType bctype,
                                           CoefficientsByAttr &cba)
{
   ReadAttr(input, bctype, cba.attr);
   cba.coefs.SetSize(2);
   cba.coefs[0] = coefFact->GetScalarCoef(input);
   cba.coefs[1] = coefFact->GetScalarCoef(input);
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
      else if (buff == "outflow")
      {
         CoefficientByAttr * c = new CoefficientByAttr;
         ReadCoefByAttr(input, OUTFLOW_BC, *c);
         obc.Append(c);
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

void AdvectionDiffusionBC::AddOutflowBC(const Array<int> & bdr,
                                        Coefficient &val)
{
   for (int i=0; i<bdr.Size(); i++)
   {
      if (obc_attr.Find(bdr[i]) == -1)
      {
         obc_attr.Append(bdr[i]);
      }
      else
      {
         MFEM_ABORT("Attempting to add an outflow BC on boundary " << bdr[i]
                    << " which already has an outflow boundary condition"
                    << " defined.");
      }
   }
   CoefficientByAttr * c = new CoefficientByAttr;
   c->attr = bdr;
   c->coef = &val;
   c->ownCoef = false;
   obc.Append(c);
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

RecyclingBC::~RecyclingBC()
{
   for (int i=0; i<bc.Size(); i++)
   {
      for (int j=0; j<bc[i]->coefs.Size(); j++)
      {
         if (bc[i]->ownCoefs[j])
         {
            delete bc[i]->coefs[j];
         }
      }
      delete bc[i];
   }
}

void RecyclingBC::SetTime(double t) const
{
   for (int i=0; i<bc.Size(); i++)
   {
      for (int j=0; j<bc[i]->coefs.Size(); j++)
      {
         if (bc[i]->coefs[j])
         {
            bc[i]->coefs[j]->SetTime(t);
         }
      }
   }
}
/*
void RecyclingBC::ReadBCs(std::istream &input)
{
   string buff;

   skip_comment_lines(input, '#');
   input >> buff;
   MFEM_VERIFY(buff == "coupled_bcs", "invalid BC file");

   while (input >> buff)
   {
      skip_comment_lines(input, '#');
      if (buff == "recycling")
      {
         CoefficientsByAttr * c = new CoefficientsByAttr;
         ReadCoefsByAttr(input, *c);
         bc.Append(c);
      }
   }
}
*/
void RecyclingBC::ReadBC(std::istream &input)
{
   // string buff;

   // skip_comment_lines(input, '#');
   // input >> buff;
   // MFEM_VERIFY(buff == "coupled_bcs", "invalid BC file");

   skip_comment_lines(input, '#');
   input >> ion_index;
   skip_comment_lines(input, '#');
   input >> vel_index;
   skip_comment_lines(input, '#');
   input >> neu_index;

   CoefficientsByAttr * c = new CoefficientsByAttr;
   ReadCoefsByAttr(input, *c);
   bc.Append(c);
}

void RecyclingBC::ReadAttr(std::istream &input,
                           Array<int> &attr)
{
   int nbdr = 0;
   skip_comment_lines(input, '#');
   input >> nbdr;
   for (int i=0; i<nbdr; i++)
   {
      int b = 0;
      input >> b;
      attr.Append(b);
   }
}

void RecyclingBC::ReadCoefsByAttr(std::istream &input,
                                  CoefficientsByAttr &cba)
{
   ReadAttr(input, cba.attr);
   cba.coefs.SetSize(2);
   cba.coefs[0] = coefFact->GetScalarCoef(input);
   cba.coefs[1] = coefFact->GetScalarCoef(input);
   cba.ownCoefs.SetSize(2);
   cba.ownCoefs[0] = false;
   cba.ownCoefs[1] = false;
}

void RecyclingBC::AddRecyclingBC(int ion, int vel, int neu,
                                 const Array<int> & bdr,
                                 Coefficient & ion_frac,
                                 Coefficient & neu_frac)
{
   ion_index = ion;
   vel_index = vel;
   neu_index = neu;

   CoefficientsByAttr * c = new CoefficientsByAttr;
   c->attr = bdr;
   c->coefs.SetSize(2);
   c->coefs[0] = &ion_frac;
   c->coefs[1] = &neu_frac;
   c->ownCoefs.SetSize(2);
   c->ownCoefs = false;
   bc.Append(c);
}

CoupledBCs::~CoupledBCs()
{
   for (int i=0; i<rbcs_.Size(); i++)
   {
      delete rbcs_[i];
   }
}

void CoupledBCs::ReadBCs(common::CoefFactory &cf, std::istream &input)
{
   string buff;

   skip_comment_lines(input, '#');
   input >> buff;
   MFEM_VERIFY(buff == "scalar_bcs", "invalid BC file");

   while (input >> buff)
   {
      skip_comment_lines(input, '#');
      if (buff == "recycling")
      {
         RecyclingBC * r = new RecyclingBC;
         r->LoadBCs(cf, input);
         // ReadCoefByAttr(input, DIRICHLET_BC, *c);
         rbcs_.Append(r);
      }
   }
}

TransportBCs::TransportBCs(const Array<int> & bdr_attr, int neqn)
   : neqn_(neqn),
     bcs_(neqn),
     bdr_attr_(bdr_attr)
{
   bcs_ = NULL;
   for (int i=0; i<neqn_; i++)
   {
      bcs_[i] = new AdvectionDiffusionBC(bdr_attr);
   }
}

TransportBCs::TransportBCs(const Array<int> & bdr_attr, int neqn,
                           CoefFactory &cf, std::istream &input)
   : neqn_(neqn),
     bcs_(neqn),
     bdr_attr_(bdr_attr),
     cbcs_()
{
   bcs_ = NULL;
   this->ReadBCs(cf, input);
}

TransportBCs::~TransportBCs()
{
   for (int i=0; i<neqn_; i++)
   {
      delete bcs_[i];
   }
}

void TransportBCs::ReadBCs(CoefFactory &cf, std::istream &input)
{
   string buff;

   skip_comment_lines(input, '#');
   input >> buff;
   MFEM_VERIFY(buff == "transport_bcs", "invalid BC file");

   vector<ios::streampos> pos(neqn_+2);
   for (int i=0; i<=neqn_+1; i++)
   {
      pos[i] = -1;
   }
   while (input >> buff)
   {
      pos[neqn_+1] = std::max(pos[neqn_+1], input.tellg());
      skip_comment_lines(input, '#');
      if (buff == "neutral_density")
      {
         pos[0] = input.tellg();
         mfem::out << "Found 'neutral_density' at position " << pos[0] << endl;
      }
      else if (buff == "ion_density")
      {
         pos[1] = input.tellg();
         mfem::out << "Found 'ion_density' at position " << pos[1] << endl;
      }
      else if (buff == "ion_parallel_velocity")
      {
         pos[2] = input.tellg();
         mfem::out << "Found 'ion_parallel_velocity' at position " << pos[2]
                   << endl;
      }
      else if (buff == "ion_temperature")
      {
         pos[3] = input.tellg();
         mfem::out << "Found 'ion_temperature' at position " << pos[3]
                   << endl;
      }
      else if (buff == "electron_temperature")
      {
         pos[4] = input.tellg();
         mfem::out << "Found 'electron_temperature' at position " << pos[4]
                   << endl;
      }
      else if (buff == "coupled_bcs")
      {
         pos[5] = input.tellg();
         mfem::out << "Found 'coupled_bcs' at position " << pos[5] << endl;
      }
   }
   for (int i=neqn_; i >= 0; i--)
   {
      if (pos[i] < 0) { pos[i] = pos[i+1]; }
   }

   input.clear();
   for (int i=0; i<neqn_; i++)
   {
      input.seekg(pos[i], std::ios::beg);
      int length = pos[i+1] - pos[i];
      if (length > 0)
      {
         char * buffer = new char[length];
         input.read(buffer, length);

         string buff_str(buffer, length);

         istringstream iss(buff_str);
         delete bcs_[i];
         bcs_[i] = new AdvectionDiffusionBC(bdr_attr_, cf, iss);

         delete [] buffer;
      }
      else
      {
         if (bcs_[i] == NULL)
         {
            bcs_[i] = new AdvectionDiffusionBC(bdr_attr_);
         }
      }
   }
   {
      input.seekg(pos[neqn_], std::ios::beg);
      int length = pos[neqn_+1] - pos[neqn_];
      if (length > 0)
      {
         char * buffer = new char[length];
         input.read(buffer, length);

         string buff_str(buffer, length);

         istringstream iss(buff_str);
         cbcs_.LoadBCs( cf, iss);

         delete [] buffer;
      }
      for (int i=0; i<cbcs_.GetNumRecyclingBCs(); i++)
      {
         const RecyclingBC & rbc = cbcs_.GetRecyclingBC(i);
         int ion_index = rbc.GetIonDensityIndex();
         const Array<CoefficientsByAttr*> & bc = rbc.GetRecyclingBCs();
         for (int j=0; j<bc.Size(); j++)
         {
            bcs_[ion_index]->AddOutflowBC(bc[j]->attr, *bc[j]->coefs[0]);
         }
      }
   }
}

void TransportICs::ReadICs(CoefFactory &cf, std::istream &input)
{
   string buff;

   skip_comment_lines(input, '#');
   input >> buff;
   MFEM_VERIFY(buff == "transport_ics", "invalid Initial Condition file");

   vector<ios::streampos> pos(neqn_+1);
   for (int i=0; i<=neqn_; i++)
   {
      pos[i] = -1;
   }
   while (input >> buff)
   {
      pos[neqn_] = std::max(pos[neqn_], input.tellg());
      skip_comment_lines(input, '#');
      if (buff == "neutral_density")
      {
         pos[0] = input.tellg();
         mfem::out << "Found 'neutral_density' at position " << pos[0] << endl;
      }
      else if (buff == "ion_density")
      {
         pos[1] = input.tellg();
         mfem::out << "Found 'ion_density' at position " << pos[1] << endl;
      }
      else if (buff == "ion_parallel_velocity")
      {
         pos[2] = input.tellg();
         mfem::out << "Found 'ion_parallel_velocity' at position " << pos[2]
                   << endl;
      }
      else if (buff == "ion_temperature")
      {
         pos[3] = input.tellg();
         mfem::out << "Found 'ion_temperature' at position " << pos[3]
                   << endl;
      }
      else if (buff == "electron_temperature")
      {
         pos[4] = input.tellg();
         mfem::out << "Found 'electron_temperature' at position " << pos[4]
                   << endl;
      }
   }
   for (int i=neqn_-1; i >= 0; i--)
   {
      if (pos[i] < 0) { pos[i] = pos[i+1]; }
   }

   input.clear();
   for (int i=0; i<neqn_; i++)
   {
      input.seekg(pos[i], std::ios::beg);
      int length = pos[i+1] - pos[i];
      if (length > 0)
      {
         char * buffer = new char[length];
         input.read(buffer, length);

         string buff_str(buffer, length);

         istringstream iss(buff_str);
         ics_[i] = cf.GetScalarCoef(iss);
         own_ics_[i] = false;

         delete [] buffer;
      }
      else
      {
         ics_[i] = NULL;
      }
   }
}

void TransportExactSolutions::Read(CoefFactory &cf, std::istream &input)
{
   string buff;

   skip_comment_lines(input, '#');
   input >> buff;
   MFEM_VERIFY(buff == "transport_ess", "invalid Exact Solutions file");

   vector<ios::streampos> pos(neqn_+1);
   for (int i=0; i<=neqn_; i++)
   {
      pos[i] = -1;
   }
   while (input >> buff)
   {
      pos[neqn_] = std::max(pos[neqn_], input.tellg());
      skip_comment_lines(input, '#');
      if (buff == "neutral_density")
      {
         pos[0] = input.tellg();
         mfem::out << "Found 'neutral_density' at position " << pos[0] << endl;
      }
      else if (buff == "ion_density")
      {
         pos[1] = input.tellg();
         mfem::out << "Found 'ion_density' at position " << pos[1] << endl;
      }
      else if (buff == "ion_parallel_velocity")
      {
         pos[2] = input.tellg();
         mfem::out << "Found 'ion_parallel_velocity' at position " << pos[2]
                   << endl;
      }
      else if (buff == "ion_temperature")
      {
         pos[3] = input.tellg();
         mfem::out << "Found 'ion_temperature' at position " << pos[3] << endl;
      }
      else if (buff == "electron_temperature")
      {
         pos[4] = input.tellg();
         mfem::out << "Found 'electron_temperature' at position " << pos[4]
                   << endl;
      }
   }
   for (int i=neqn_-1; i >= 0; i--)
   {
      if (pos[i] < 0) { pos[i] = pos[i+1]; }
   }

   input.clear();
   for (int i=0; i<neqn_; i++)
   {
      input.seekg(pos[i], std::ios::beg);
      int length = pos[i+1] - pos[i];
      if (length > 0)
      {
         char * buffer = new char[length];
         input.read(buffer, length);

         string buff_str(buffer, length);

         istringstream iss(buff_str);
         ess_[i] = cf.GetScalarCoef(iss);
         own_ess_[i] = false;

         delete [] buffer;
      }
      else
      {
         ess_[i] = NULL;
      }
   }
}

void TransportCoefs::ReadCoefs(CoefFactory &cf, std::istream &input)
{
   string buff;

   skip_comment_lines(input, '#');
   input >> buff;
   MFEM_VERIFY(buff == "transport_coefs", "invalid Coefficient file");

   vector<ios::streampos> pos(neqn_+2);
   for (int i=0; i<=neqn_+1; i++)
   {
      pos[i] = -1;
   }
   while (input >> buff)
   {
      pos[neqn_+1] = std::max(pos[neqn_+1], input.tellg());
      skip_comment_lines(input, '#');
      if (buff == "neutral_density")
      {
         pos[0] = input.tellg();
         mfem::out << "Found 'neutral_density' at position " << pos[0] << endl;
      }
      else if (buff == "ion_density")
      {
         pos[1] = input.tellg();
         mfem::out << "Found 'ion_density' at position " << pos[1] << endl;
      }
      else if (buff == "ion_parallel_momentum")
      {
         pos[2] = input.tellg();
         mfem::out << "Found 'ion_parallel_momentum' at position " << pos[2]
                   << endl;
      }
      else if (buff == "ion_static_pressure")
      {
         pos[3] = input.tellg();
         mfem::out << "Found 'ion_static_pressure' at position " << pos[3]
                   << endl;
         MFEM_ABORT("Outdated input file: switch to total energy equations");
      }
      else if (buff == "electron_static_pressure")
      {
         pos[4] = input.tellg();
         mfem::out << "Found 'electron_static_pressure' at position " << pos[4]
                   << endl;
         MFEM_ABORT("Outdated input file: switch to total energy equations");
      }
      else if (buff == "ion_total_energy")
      {
         pos[3] = input.tellg();
         mfem::out << "Found 'ion_total_energy' at position " << pos[3]
                   << endl;
      }
      else if (buff == "electron_total_energy")
      {
         pos[4] = input.tellg();
         mfem::out << "Found 'electron_total_energy' at position " << pos[4]
                   << endl;
      }
      else if (buff == "common_coefs")
      {
         pos[5] = input.tellg();
         mfem::out << "Found 'common_coefs' at position " << pos[5] << endl;
      }
   }
   for (int i=neqn_; i >= 0; i--)
   {
      if (pos[i] < 0) { pos[i] = pos[i+1]; }
   }

   input.clear();
   for (int i=0; i<=neqn_; i++)
   {
      input.seekg(pos[i], std::ios::beg);
      int length = pos[i+1] - pos[i];
      if (length > 0)
      {
         char * buffer = new char[length];
         input.read(buffer, length);

         string buff_str(buffer, length);

         istringstream iss(buff_str);
         eqnCoefs_[i]->LoadCoefs(cf, iss);

         delete [] buffer;
      }
   }
}

void EqnCoefficients::ReadCoefs(std::istream &input)
{
   int nSCoefs = sCoefs_.Size();
   int nVCoefs = vCoefs_.Size();
   int nMCoefs = mCoefs_.Size();
   int nCoefs = nSCoefs + nVCoefs + nMCoefs;
   string buff;

   skip_comment_lines(input, '#');

   vector<ios::streampos> pos(nCoefs+1);
   vector<int> ord(nCoefs+1);
   for (int i=0; i<=nCoefs; i++)
   {
      pos[i] = -1;
      ord[i] = nCoefs;
   }

   enum CoefType {INVALID_COEF_TYPE = -1, SCALAR, VECTOR, MATRIX};
   Array<CoefType> typ(nCoefs);
   typ = CoefType::INVALID_COEF_TYPE;

   int c = 0;
   while (input >> buff)
   {
      pos[nCoefs] = std::max(pos[nCoefs], input.tellg());
      skip_comment_lines(input, '#');

      for (unsigned int i=0; i<sCoefNames_.size(); i++)
      {
         if (buff == sCoefNames_[i])
         {
            pos[i] = input.tellg();
            typ[i] = SCALAR;
            ord[c] = i; c++;
            break;
         }
      }
      for (unsigned int i=0; i<vCoefNames_.size(); i++)
      {
         if (buff == vCoefNames_[i])
         {
            pos[i+nSCoefs] = input.tellg();
            typ[i+nSCoefs] = VECTOR;
            ord[c] = i+nSCoefs; c++;
            break;
         }
      }
      for (unsigned int i=0; i<mCoefNames_.size(); i++)
      {
         if (buff == mCoefNames_[i])
         {
            pos[i+nSCoefs+nVCoefs] = input.tellg();
            typ[i+nSCoefs+nVCoefs] = MATRIX;
            ord[c] = i+nSCoefs+nVCoefs; c++;
            break;
         }
      }
   }

   Coefficient       *sCoef = NULL;
   VectorCoefficient *vCoef = NULL;
   MatrixCoefficient *mCoef = NULL;

   StateVariableCoef    *svsCoef = NULL;
   StateVariableVecCoef *svvCoef = NULL;
   StateVariableMatCoef *svmCoef = NULL;

   input.clear();
   for (int i=0; i<c; i++)
   {
      input.seekg(pos[ord[i]], std::ios::beg);
      int length = pos[ord[i+1]] - pos[ord[i]];
      if (length > 0)
      {
         char * buffer = new char[length];
         input.read(buffer, length);

         string buff_str(buffer, length);

         istringstream iss(buff_str);
         switch (typ[ord[i]])
         {
            case SCALAR:
               sCoef = coefFact->GetScalarCoef(iss);
               svsCoef = dynamic_cast<StateVariableCoef*>(sCoef);
               if (svsCoef == NULL)
               {
                  svsCoef = new StateVariableStandardCoef(*sCoef);
               }
               sCoefs_[ord[i]] = svsCoef;
               break;
            case VECTOR:
               vCoef = coefFact->GetVectorCoef(iss);
               svvCoef = dynamic_cast<StateVariableVecCoef*>(vCoef);
               if (svvCoef == NULL)
               {
                  svvCoef = new StateVariableStandardVecCoef(*vCoef);
               }
               vCoefs_[ord[i]-nSCoefs] = svvCoef;
               break;
            case MATRIX:
               mCoef = coefFact->GetMatrixCoef(iss);
               svmCoef = dynamic_cast<StateVariableMatCoef*>(mCoef);
               if (svmCoef == NULL)
               {
                  svmCoef = new StateVariableStandardMatCoef(*mCoef);
               }
               mCoefs_[ord[i]-nSCoefs-nVCoefs] = svmCoef;
               break;
            default:
               MFEM_WARNING("Unrecognized coefficient type");
         }
         delete [] buffer;
      }
   }
}

NeutralDensityCoefs::NeutralDensityCoefs()
   : EqnCoefficients(sCoefNames::NUM_SCALAR_COEFS)
{
   sCoefNames_[DIFFUSION_COEF] = "diffusion_coef";
   sCoefNames_[SOURCE_COEF]    = "source_coef";
}

IonDensityCoefs::IonDensityCoefs()
   : EqnCoefficients(sCoefNames::NUM_SCALAR_COEFS)
{
   sCoefNames_[PARA_DIFFUSION_COEF] = "para_diffusion_coef";
   sCoefNames_[PERP_DIFFUSION_COEF] = "perp_diffusion_coef";
   sCoefNames_[SOURCE_COEF]         = "source_coef";
}

IonMomentumCoefs::IonMomentumCoefs()
   : EqnCoefficients(sCoefNames::NUM_SCALAR_COEFS)
{
   sCoefNames_[PARA_DIFFUSION_COEF] = "para_diffusion_coef";
   sCoefNames_[PERP_DIFFUSION_COEF] = "perp_diffusion_coef";
   sCoefNames_[SOURCE_COEF]         = "source_coef";
}

IonStaticPressureCoefs::IonStaticPressureCoefs()
   : EqnCoefficients(sCoefNames::NUM_SCALAR_COEFS)
{
   sCoefNames_[PARA_DIFFUSION_COEF] = "para_diffusion_coef";
   sCoefNames_[PERP_DIFFUSION_COEF] = "perp_diffusion_coef";
   sCoefNames_[SOURCE_COEF]         = "source_coef";
}

ElectronStaticPressureCoefs::ElectronStaticPressureCoefs()
   : EqnCoefficients(sCoefNames::NUM_SCALAR_COEFS)
{
   sCoefNames_[PARA_DIFFUSION_COEF] = "para_diffusion_coef";
   sCoefNames_[PERP_DIFFUSION_COEF] = "perp_diffusion_coef";
   sCoefNames_[SOURCE_COEF]         = "source_coef";
}

IonTotalEnergyCoefs::IonTotalEnergyCoefs()
   : EqnCoefficients(sCoefNames::NUM_SCALAR_COEFS)
{
   sCoefNames_[PARA_DIFFUSION_COEF] = "para_diffusion_coef";
   sCoefNames_[PERP_DIFFUSION_COEF] = "perp_diffusion_coef";
   sCoefNames_[SOURCE_COEF]         = "source_coef";
}

ElectronTotalEnergyCoefs::ElectronTotalEnergyCoefs()
   : EqnCoefficients(sCoefNames::NUM_SCALAR_COEFS)
{
   sCoefNames_[PARA_DIFFUSION_COEF] = "para_diffusion_coef";
   sCoefNames_[PERP_DIFFUSION_COEF] = "perp_diffusion_coef";
   sCoefNames_[SOURCE_COEF]         = "source_coef";
}

CommonCoefs::CommonCoefs()
   : EqnCoefficients(sCoefNames::NUM_SCALAR_COEFS, vCoefNames::NUM_VECTOR_COEFS)
{
   sCoefNames_[IONIZATION_COEF]      = "ionization_coef";
   sCoefNames_[RECOMBINATION_COEF]   = "recombination_coef";
   sCoefNames_[CHARGE_EXCHANGE_COEF] = "charge_exchange_coef";

   vCoefNames_[MAGNETIC_FIELD_COEF]  = "magnetic_field_coef";
}

TransportCoefFactory::TransportCoefFactory(
   const std::vector<std::string> & names,
   ParGridFunctionArray & pgfa)
{
   MFEM_VERIFY(names.size() == pgfa.Size(),
               "TransportCoefFactory constructor: "
               "Size mismatch in input arguments.");
   for (int i=0; i<pgfa.Size(); i++)
   {
      this->AddExternalGridFunction(names[i], *pgfa[i]);
   }
}

Coefficient *
TransportCoefFactory::GetScalarCoef(std::string &name, std::istream &input)
{
   int coef_idx = -1;
   if (name == "StateVariableConstantCoef")
   {
      double c;
      input >> c;
      coef_idx = sCoefs.Append(new StateVariableConstantCoef(c));
   }
   else if (name == "StateVariableGridFunctionCoef")
   {
      int ft;
      string gf_name;
      input >> ft >> gf_name;
      MFEM_VERIFY(ext_gf.find(gf_name) != ext_gf.end(), "TransportCoefFactory: "
                  "GridFunction named \"" << gf_name << "\" not found amongst "
                  "external GridFunctions.");
      coef_idx = sCoefs.Append(new StateVariableGridFunctionCoef(ext_gf[gf_name],
                                                                 (FieldType)ft));
   }
   else if (name == "StateVariableSumCoef")
   {
      Coefficient * ACoef = this->GetScalarCoef(input);
      Coefficient * BCoef = this->GetScalarCoef(input);

      StateVariableCoef * A = dynamic_cast<StateVariableCoef*>(ACoef);
      StateVariableCoef * B = dynamic_cast<StateVariableCoef*>(BCoef);

      MFEM_VERIFY(A != NULL, "TransportCoefFactory: first argument to "
                  "StateVariableSumCoef is not a StateVariableCoef.");
      MFEM_VERIFY(B != NULL, "TransportCoefFactory: second argument to "
                  "StateVariableSumCoef is not a StateVariableCoef.");

      coef_idx = sCoefs.Append(new StateVariableSumCoef(*A, *B));
   }
   else if (name == "StateVariableProductCoef")
   {
      Coefficient * ACoef = this->GetScalarCoef(input);
      Coefficient * BCoef = this->GetScalarCoef(input);

      StateVariableCoef * A = dynamic_cast<StateVariableCoef*>(ACoef);
      StateVariableCoef * B = dynamic_cast<StateVariableCoef*>(BCoef);

      MFEM_VERIFY(A != NULL, "TransportCoefFactory: first argument to "
                  "StateVariableProductCoef is not a StateVariableCoef.");
      MFEM_VERIFY(B != NULL, "TransportCoefFactory: second argument to "
                  "StateVariableProductCoef is not a StateVariableCoef.");

      coef_idx = sCoefs.Append(new StateVariableProductCoef(*A, *B));
   }
   else if (name == "StateVariablePowerCoef")
   {
      Coefficient * ACoef = this->GetScalarCoef(input);

      StateVariableCoef * A = dynamic_cast<StateVariableCoef*>(ACoef);

      MFEM_VERIFY(A != NULL, "TransportCoefFactory: first argument to "
                  "StateVariablePowerCoef is not a StateVariableCoef.");

      int p;
      input >> p;

      coef_idx = sCoefs.Append(new StateVariablePowerCoef(*A, p));
   }
   else if (name == "SoundSpeedCoef")
   {
      double mi;
      input >> mi;

      string TiCoefName;
      input >> TiCoefName;
      Coefficient * TiCoef = this->GetScalarCoef(TiCoefName, input);

      string TeCoefName;
      input >> TeCoefName;
      Coefficient * TeCoef = this->GetScalarCoef(TeCoefName, input);

      coef_idx = sCoefs.Append(new SoundSpeedCoef(mi, *TiCoef, *TeCoef));
   }
   else if (name == "ApproxIonizationRate")
   {
      Coefficient * TeCoef = this->GetScalarCoef(input);

      coef_idx = sCoefs.Append(new ApproxIonizationRate(*TeCoef));
   }
   else if (name == "ApproxRecombinationRate")
   {
      Coefficient * TeCoef = this->GetScalarCoef(input);

      coef_idx = sCoefs.Append(new ApproxRecombinationRate(*TeCoef));
   }
   else if (name == "ApproxChargeExchangeRate")
   {
      Coefficient * TiCoef = this->GetScalarCoef(input);

      coef_idx = sCoefs.Append(new ApproxRecombinationRate(*TiCoef));
   }
   else if (name == "IonizationSourceCoef")
   {
      Coefficient * neCoef = this->GetScalarCoef(input);
      Coefficient * nnCoef = this->GetScalarCoef(input);
      Coefficient * izCoef = this->GetScalarCoef(input);

      double nn0 = 1e10;
      input >> nn0;

      coef_idx = sCoefs.Append(new IonizationSourceCoef(*neCoef, *nnCoef,
                                                        *izCoef, nn0));
   }
   else if (name == "RecombinationSinkCoef")
   {
      Coefficient * neCoef = this->GetScalarCoef(input);
      Coefficient * niCoef = this->GetScalarCoef(input);
      Coefficient * rcCoef = this->GetScalarCoef(input);

      double ni0 = 1e10;
      input >> ni0;

      coef_idx = sCoefs.Append(new RecombinationSinkCoef(*neCoef, *niCoef,
                                                         *rcCoef, ni0));
   }
   else if (name == "ChargeExchangeSinkCoef")
   {
      Coefficient * nnCoef = this->GetScalarCoef(input);
      Coefficient * niCoef = this->GetScalarCoef(input);
      Coefficient * cxCoef = this->GetScalarCoef(input);

      coef_idx = sCoefs.Append(new RecombinationSinkCoef(*nnCoef, *niCoef,
                                                         *cxCoef));
   }
   else
   {
      return CoefFactory::GetScalarCoef(name, input);
   }
   return sCoefs[--coef_idx];
}

VectorCoefficient *
TransportCoefFactory::GetVectorCoef(std::string &name, std::istream &input)
{
   int coef_idx = -1;
   if (name == "__dummy_name__")
   {
   }
   else
   {
      return CoefFactory::GetVectorCoef(name, input);
   }
   return vCoefs[--coef_idx];
}
/*
void ElectronStaticPressureCoefs::ReadCoefs(std::istream &input)
{
   string buff;

   skip_comment_lines(input, '#');

   Array<ios::streampos> pos(NUM_COEFS+1);
   pos = -1;
   while (input >> buff)
   {
      pos[NUM_COEFS] = std::max(pos[NUM_COEFS], input.tellg());
      skip_comment_lines(input, '#');
      if (buff == "perp_diffusion_coef")
      {
         pos[PERP_DIFFUSION_COEF] = input.tellg();
      }
      else if (buff == "para_diffusion_coef")
      {
         pos[PARA_DIFFUSION_COEF] = input.tellg();
      }
      else if (buff == "source_coef")
      {
         pos[SOURCE_COEF] = input.tellg();
      }
   }
   for (int i=NUM_COEFS-1; i >= 0; i--)
   {
      if (pos[i] < 0) { pos[i] = pos[i+1]; }
   }

   input.clear();
   for (int i=0; i<NUM_COEFS; i++)
   {
      input.seekg(pos[i], std::ios::beg);
      int length = pos[i+1] - pos[i];
      if (length > 0)
      {
         char * buffer = new char[length];
         input.read(buffer, length);

         string buff_str(buffer, length);

         istringstream iss(buff_str);
         coefs_[i] = (*coefFact)(iss);

         delete [] buffer;
      }
   }
}
*/
/*
ChiParaCoefficient::ChiParaCoefficient(BlockVector & nBV, Array<int> & z)
   : nBV_(nBV),
     ion_(-1),
     z_(z),
     m_(NULL),
     n_(z.Size())
{}

ChiParaCoefficient::ChiParaCoefficient(BlockVector & nBV, int ion_species,
                                       Array<int> & z, Vector & m)
   : nBV_(nBV),
     ion_(ion_species),
     z_(z),
     m_(&m),
     n_(z.Size())
{}

void ChiParaCoefficient::SetT(ParGridFunction & T)
{
   TCoef_.SetGridFunction(&T);
   nGF_.MakeRef(T.ParFESpace(), nBV_.GetBlock(0));
}

double
ChiParaCoefficient::Eval(ElementTransformation &T, const IntegrationPoint &ip)
{
   double temp = TCoef_.Eval(T, ip);

   for (int i=0; i<z_.Size(); i++)
   {
      nGF_.SetData(nBV_.GetBlock(i));
      nCoef_.SetGridFunction(&nGF_);
      n_[i] = nCoef_.Eval(T, ip);
   }

   if (ion_ < 0)
   {
      return chi_e_para(temp, z_.Size(), n_, z_);
   }
   else
   {
      return chi_i_para((*m_)[ion_], temp, ion_, z_.Size(), n_, z_);
   }
}

ChiPerpCoefficient::ChiPerpCoefficient(BlockVector & nBV, Array<int> & z)
   : ion_(-1)
{}

ChiPerpCoefficient::ChiPerpCoefficient(BlockVector & nBV, int ion_species,
                                       Array<int> & z, Vector & m)
   : ion_(ion_species)
{}

double
ChiPerpCoefficient::Eval(ElementTransformation &T, const IntegrationPoint &ip)
{
   if (ion_ < 0)
   {
      return chi_e_perp();
   }
   else
   {
      return chi_i_perp();
   }
}

ChiCrossCoefficient::ChiCrossCoefficient(BlockVector & nBV, Array<int> & z)
   : ion_(-1)
{}

ChiCrossCoefficient::ChiCrossCoefficient(BlockVector & nBV, int ion_species,
                                         Array<int> & z, Vector & m)
   : ion_(ion_species)
{}

double
ChiCrossCoefficient::Eval(ElementTransformation &T, const IntegrationPoint &ip)
{
   if (ion_ < 0)
   {
      return chi_e_cross();
   }
   else
   {
      return chi_i_cross();
   }
}

ChiCoefficient::ChiCoefficient(int dim, BlockVector & nBV, Array<int> & charges)
   : MatrixCoefficient(dim),
     chiParaCoef_(nBV, charges),
     chiPerpCoef_(nBV, charges),
     chiCrossCoef_(nBV, charges),
     bHat_(dim)
{}

ChiCoefficient::ChiCoefficient(int dim, BlockVector & nBV, int ion_species,
                               Array<int> & charges, Vector & masses)
   : MatrixCoefficient(dim),
     chiParaCoef_(nBV, ion_species, charges, masses),
     chiPerpCoef_(nBV, ion_species, charges, masses),
     chiCrossCoef_(nBV, ion_species, charges, masses),
     bHat_(dim)
{}

void ChiCoefficient::SetT(ParGridFunction & T)
{
   chiParaCoef_.SetT(T);
}

void ChiCoefficient::SetB(ParGridFunction & B)
{
   BCoef_.SetGridFunction(&B);
}

void ChiCoefficient::Eval(DenseMatrix &K, ElementTransformation &T,
                          const IntegrationPoint &ip)
{
   double chi_para  = chiParaCoef_.Eval(T, ip);
   double chi_perp  = chiPerpCoef_.Eval(T, ip);
   double chi_cross = (width > 2) ? chiCrossCoef_.Eval(T, ip) : 0.0;

   BCoef_.Eval(bHat_, T, ip);
   bHat_ /= bHat_.Norml2();

   K.SetSize(bHat_.Size());

   if (width == 2)
   {
      K(0,0) = bHat_[0] * bHat_[0] * (chi_para - chi_perp) + chi_perp;
      K(0,1) = bHat_[0] * bHat_[1] * (chi_para - chi_perp);
      K(1,0) = K(0,1);
      K(1,1) = bHat_[1] * bHat_[1] * (chi_para - chi_perp) + chi_perp;
   }
   else
   {
      K(0,0) = bHat_[0] * bHat_[0] * (chi_para - chi_perp) + chi_perp;
      K(0,1) = bHat_[0] * bHat_[1] * (chi_para - chi_perp);
      K(0,2) = bHat_[0] * bHat_[2] * (chi_para - chi_perp);
      K(1,0) = K(0,1);
      K(1,1) = bHat_[1] * bHat_[1] * (chi_para - chi_perp) + chi_perp;
      K(1,2) = bHat_[1] * bHat_[2] * (chi_para - chi_perp);
      K(2,0) = K(0,2);
      K(2,1) = K(1,2);
      K(2,2) = bHat_[2] * bHat_[2] * (chi_para - chi_perp) + chi_perp;

      K(1,2) -= bHat_[0] * chi_cross;
      K(2,0) -= bHat_[1] * chi_cross;
      K(0,1) -= bHat_[2] * chi_cross;
      K(2,1) += bHat_[0] * chi_cross;
      K(0,2) += bHat_[1] * chi_cross;
      K(1,0) += bHat_[2] * chi_cross;

   }
}

EtaParaCoefficient::EtaParaCoefficient(BlockVector & nBV, Array<int> & z)
   : nBV_(nBV),
     ion_(-1),
     z_(z),
     m_(NULL),
     n_(z.Size())
{}

EtaParaCoefficient::EtaParaCoefficient(BlockVector & nBV, int ion_species,
                                       Array<int> & z, Vector & m)
   : nBV_(nBV),
     ion_(ion_species),
     z_(z),
     m_(&m),
     n_(z.Size())
{}

void EtaParaCoefficient::SetT(ParGridFunction & T)
{
   TCoef_.SetGridFunction(&T);
   nGF_.MakeRef(T.ParFESpace(), nBV_.GetBlock(0));
}

double
EtaParaCoefficient::Eval(ElementTransformation &T, const IntegrationPoint &ip)
{
   double temp = TCoef_.Eval(T, ip);

   for (int i=0; i<z_.Size(); i++)
   {
      nGF_.SetData(nBV_.GetBlock(i));
      nCoef_.SetGridFunction(&nGF_);
      n_[i] = nCoef_.Eval(T, ip);
   }

   if (ion_ < 0)
   {
      return eta_e_para(temp, z_.Size(), z_.Size(), n_, z_);
   }
   else
   {
      return eta_i_para((*m_)[ion_], temp, ion_, z_.Size(), n_, z_);
   }
}
*/
/*
ElementSkewGridFunction::ElementSkewGridFunction(ParMesh &_pmesh)
   : ParGridFunction(),
     pmesh(_pmesh),
     fes(&_pmesh, 0, pmesh.SpaceDimension())
{
   computeSkew();
}

void ElementSkewGridFunction::Update()
{
   fes.Update(false);
   ParGridFunction::Update();
   computeSkew();
}

void ElementSkewGridFunction::computeSkew()
{
   const int dim = pmesh.Dimension();
   // DenseMatrix Jac(dim);

   Array<int> vdofs;
   ElementTransformation *T = NULL;
   // Vector el_x;
   // DenseMatrix PMatI;
   // DenseMatrix DSh;
   // DenseMatrix Jpr(dim);

   for (int i = 0; i < pmesh.GetNE(); i++)
   {
      // const FiniteElement *fe = fes.GetFE(i);
      //const int dof = fe->GetDof();
      // fes.GetElementVDofs(i, vdofs);
      T = fes.GetElementTransformation(i);
      // x.GetSubVector(vdofs, el_x);
      // PMatI.UseExternalData(el_x.GetData(), dof, dim);
      // DSh.SetSize(dof, dim);

      const IntegrationRule &ir = IntRules.Get(pmesh.GetElementBaseGeometry(i),
                                               T->OrderJ());
      for (int j = 0; j < ir.GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir.IntPoint(j);

         T->SetIntPoint(&ip);
         const DenseMatrix Jpr = T->Jacobian();
         // fe->CalcDShape(ip, DSh);
         // MultAtB(PMatI, DSh, Jpr);

         Vector col1, col2, col3;
         Jpr.GetColumn(0, col1);
         Jpr.GetColumn(1, col2);
         Jpr.GetColumn(2, col3);
         double norm_c1 = col1.Norml2(),
                norm_c2 = col2.Norml2(),
                norm_c3 = dim == 3 ? col3.Norml2() : 0.;

         double cos_Jpr_12 = (col1 * col2) / (norm_c1 * norm_c2),
                cos_Jpr_13 = dim == 3 ?  (col1 * col3) / (norm_c1 * norm_c3) : 0,
                cos_Jpr_23 = dim == 3 ?  (col2 * col3) / (norm_c2 * norm_c3) : 0.;

         double th12 = acos(cos_Jpr_12),
                th13 = dim == 3 ? acos(cos_Jpr_13) : 0.,
                th23 = dim == 3 ? acos(cos_Jpr_23) : 0.;

         double max_angle = std::max(1.0, fabs(cot(th12)));
         if (dim == 3)
         {
            max_angle = std::max(max_angle, fabs(cot(th13)));
            max_angle = std::max(max_angle, fabs(cot(th23)));
         }
         (*this)(i) = max_angle;
      }
   }
}
*/
DGAdvectionDiffusionTDO::DGAdvectionDiffusionTDO(const DGParams & dg,
                                                 ParFiniteElementSpace &fes,
                                                 ParGridFunctionArray &pgf,
                                                 Coefficient &CCoef,
                                                 bool imex)
   : TimeDependentOperator(fes.GetVSize()),
     dg_(dg),
     imex_(imex),
     logging_(0),
     log_prefix_(""),
     dt_(-1.0),
     fes_(&fes),
     pgf_(&pgf),
     CCoef_(&CCoef),
     VCoef_(NULL),
     dCoef_(NULL),
     DCoef_(NULL),
     SCoef_(NULL),
     negVCoef_(NULL),
     dtNegVCoef_(NULL),
     dtdCoef_(NULL),
     dtDCoef_(NULL),
     dbcAttr_(0),
     dbcCoef_(NULL),
     nbcAttr_(0),
     nbcCoef_(NULL),
     m_(fes_),
     a_(NULL),
     b_(NULL),
     s_(NULL),
     k_(NULL),
     q_exp_(NULL),
     q_imp_(NULL),
     M_(NULL),
     // B_(NULL),
     // S_(NULL),
     rhs_(fes_),
     RHS_(fes_->GetTrueVSize()),
     X_(fes_->GetTrueVSize())
{
   m_.AddDomainIntegrator(new MassIntegrator(*CCoef_));

   M_prec_.SetType(HypreSmoother::Jacobi);
   M_solver_.SetPreconditioner(M_prec_);

   M_solver_.iterative_mode = false;
   M_solver_.SetRelTol(1e-9);
   M_solver_.SetAbsTol(0.0);
   M_solver_.SetMaxIter(100);
   M_solver_.SetPrintLevel(0);
}

DGAdvectionDiffusionTDO::~DGAdvectionDiffusionTDO()
{
   delete M_;
   // delete B_;
   // delete S_;

   delete a_;
   delete b_;
   delete s_;
   delete k_;
   delete q_exp_;
   delete q_imp_;

   delete negVCoef_;
   delete dtNegVCoef_;
   delete dtdCoef_;
   delete dtDCoef_;
}

void DGAdvectionDiffusionTDO::initM()
{
   m_.Assemble();
   m_.Finalize();

   delete M_;
   M_ = m_.ParallelAssemble();
   M_solver_.SetOperator(*M_);
}

void DGAdvectionDiffusionTDO::initA()
{
   if (a_ == NULL)
   {
      a_ = new ParBilinearForm(fes_);

      a_->AddDomainIntegrator(new MassIntegrator(*CCoef_));
      if (dCoef_)
      {
         a_->AddDomainIntegrator(new DiffusionIntegrator(*dtdCoef_));
         a_->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(*dtdCoef_,
                                                                 dg_.sigma,
                                                                 dg_.kappa));
         // a_->AddBdrFaceIntegrator(new DGDiffusionIntegrator(*dtdCoef_,
         //                                                 dg_.sigma,
         //                                                 dg_.kappa));
      }
      else if (DCoef_)
      {
         a_->AddDomainIntegrator(new DiffusionIntegrator(*dtDCoef_));
         a_->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(*dtDCoef_,
                                                                 dg_.sigma,
                                                                 dg_.kappa));
         // a_->AddBdrFaceIntegrator(new DGDiffusionIntegrator(*dtDCoef_,
         //                                                 dg_.sigma,
         //                                                 dg_.kappa));
      }
      if (negVCoef_ && !imex_)
      {
         a_->AddDomainIntegrator(new ConvectionIntegrator(*dtNegVCoef_, -1.0));
         a_->AddInteriorFaceIntegrator(
            new TransposeIntegrator(new DGTraceIntegrator(*dtNegVCoef_,
                                                          1.0, -0.5)));
         a_->AddBdrFaceIntegrator(
            new TransposeIntegrator(new DGTraceIntegrator(*dtNegVCoef_,
                                                          1.0, -0.5)));
      }
   }
}

void DGAdvectionDiffusionTDO::initB()
{
   if (b_ == NULL && (dCoef_ || DCoef_ || VCoef_))
   {
      b_ = new ParBilinearForm(fes_);

      if (dCoef_)
      {
         b_->AddDomainIntegrator(new DiffusionIntegrator(*dCoef_));
         b_->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(*dCoef_,
                                                                 dg_.sigma,
                                                                 dg_.kappa));
         // b_->AddBdrFaceIntegrator(new DGDiffusionIntegrator(*dCoef_,
         //                                                 dg_.sigma,
         //                                                 dg_.kappa));
      }
      else if (DCoef_)
      {
         b_->AddDomainIntegrator(new DiffusionIntegrator(*DCoef_));
         b_->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(*DCoef_,
                                                                 dg_.sigma,
                                                                 dg_.kappa));
         // b_->AddBdrFaceIntegrator(new DGDiffusionIntegrator(*DCoef_,
         //                                                 dg_.sigma,
         //                                                 dg_.kappa));
      }
      if (negVCoef_)
      {
         b_->AddDomainIntegrator(new ConvectionIntegrator(*negVCoef_, -1.0));
         b_->AddInteriorFaceIntegrator(
            new TransposeIntegrator(new DGTraceIntegrator(*negVCoef_,
                                                          1.0, -0.5)));
         b_->AddBdrFaceIntegrator(
            new TransposeIntegrator(new DGTraceIntegrator(*negVCoef_,
                                                          1.0, -0.5)));
      }
      /*
      b_->Assemble();
      b_->Finalize();

      delete B_;
      B_ = b_->ParallelAssemble();
      */
   }
}

void DGAdvectionDiffusionTDO::initS()
{
   if (s_ == NULL && (dCoef_ || DCoef_))
   {
      s_ = new ParBilinearForm(fes_);

      if (dCoef_)
      {
         s_->AddDomainIntegrator(new DiffusionIntegrator(*dCoef_));
         s_->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(*dCoef_,
                                                                 dg_.sigma,
                                                                 dg_.kappa));
         // s_->AddBdrFaceIntegrator(new DGDiffusionIntegrator(*dCoef_,
         //                                                 dg_.sigma,
         //                                                 dg_.kappa));
      }
      else if (DCoef_)
      {
         s_->AddDomainIntegrator(new DiffusionIntegrator(*DCoef_));
         s_->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(*DCoef_,
                                                                 dg_.sigma,
                                                                 dg_.kappa));
         // s_->AddBdrFaceIntegrator(new DGDiffusionIntegrator(*DCoef_,
         //                                                 dg_.sigma,
         //                                                 dg_.kappa));
      }
      /*
      s_->Assemble();
      s_->Finalize();

      delete S_;
      S_ = s_->ParallelAssemble();
      */
   }
}

void DGAdvectionDiffusionTDO::initK()
{
   if (k_ == NULL && VCoef_)
   {
      k_ = new ParBilinearForm(fes_);

      if (negVCoef_)
      {
         k_->AddDomainIntegrator(new ConvectionIntegrator(*negVCoef_, -1.0));
         k_->AddInteriorFaceIntegrator(
            new TransposeIntegrator(new DGTraceIntegrator(*negVCoef_,
                                                          1.0, -0.5)));
         k_->AddBdrFaceIntegrator(
            new TransposeIntegrator(new DGTraceIntegrator(*negVCoef_,
                                                          1.0, -0.5)));
      }
      k_->Assemble();
      k_->Finalize();
   }
}

void DGAdvectionDiffusionTDO::initQ()
{
   if (imex_)
   {
      if (q_exp_ == NULL &&
          (SCoef_ || (dbcCoef_ && (dCoef_ || DCoef_ || VCoef_))))
      {
         q_exp_ = new ParLinearForm(fes_);

         if (SCoef_)
         {
            q_exp_->AddDomainIntegrator(new DomainLFIntegrator(*SCoef_));
         }
         if (dbcCoef_ && VCoef_ && !(dCoef_ || DCoef_))
         {
            q_exp_->AddBdrFaceIntegrator(
               new BoundaryFlowIntegrator(*dbcCoef_, *negVCoef_,
                                          -1.0, -0.5),
               dbcAttr_);
         }
         q_exp_->Assemble();
      }
      if (q_imp_ == NULL &&
          (SCoef_ || (dbcCoef_ && (dCoef_ || DCoef_ || VCoef_))))
      {
         q_imp_ = new ParLinearForm(fes_);

         if (dbcCoef_ && dCoef_)
         {
            q_imp_->AddBdrFaceIntegrator(
               new DGDirichletLFIntegrator(*dbcCoef_, *dCoef_,
                                           dg_.sigma, dg_.kappa),
               dbcAttr_);
         }
         else if (dbcCoef_ && DCoef_)
         {
            q_imp_->AddBdrFaceIntegrator(
               new DGDirichletLFIntegrator(*dbcCoef_, *DCoef_,
                                           dg_.sigma, dg_.kappa),
               dbcAttr_);
         }
         q_imp_->Assemble();
      }
   }
   else
   {
      if (q_imp_ == NULL &&
          (SCoef_ || (dbcCoef_ && (dCoef_ || DCoef_ || VCoef_))))
      {
         q_imp_ = new ParLinearForm(fes_);

         if (SCoef_)
         {
            q_imp_->AddDomainIntegrator(new DomainLFIntegrator(*SCoef_));
         }
         if (dbcCoef_ && dCoef_)
         {
            q_imp_->AddBdrFaceIntegrator(
               new DGDirichletLFIntegrator(*dbcCoef_, *dCoef_,
                                           dg_.sigma, dg_.kappa),
               dbcAttr_);
         }
         else if (dbcCoef_ && DCoef_)
         {
            q_imp_->AddBdrFaceIntegrator(
               new DGDirichletLFIntegrator(*dbcCoef_, *DCoef_,
                                           dg_.sigma, dg_.kappa),
               dbcAttr_);
         }
         else if (dbcCoef_ && VCoef_)
         {
            q_imp_->AddBdrFaceIntegrator(
               new BoundaryFlowIntegrator(*dbcCoef_, *negVCoef_,
                                          -1.0, -0.5),
               dbcAttr_);
         }
         q_imp_->Assemble();
      }
   }
}

void DGAdvectionDiffusionTDO::SetTime(const double _t)
{
   this->TimeDependentOperator::SetTime(_t);

   if (fes_->GetMyRank() == 0 && logging_)
   {
      cout << log_prefix_ << "SetTime with t = " << _t << endl;
   }

   this->initM();

   this->initA();

   if (imex_)
   {
      this->initS();
      this->initK();
   }
   else
   {
      this->initB();
   }

   this->initQ();
}

void DGAdvectionDiffusionTDO::SetLogging(int logging, const string & prefix)
{
   logging_ = logging;
   log_prefix_ = prefix;
}

void DGAdvectionDiffusionTDO::SetAdvectionCoefficient(VectorCoefficient &VCoef)
{
   VCoef_ = &VCoef;
   if (negVCoef_ == NULL)
   {
      negVCoef_ = new ScalarVectorProductCoefficient(-1.0, VCoef);
   }
   else
   {
      negVCoef_->SetBCoef(VCoef);
   }
   if (dtNegVCoef_ == NULL)
   {
      dtNegVCoef_ = new ScalarVectorProductCoefficient(dt_, *negVCoef_);
   }
   if (imex_)
   {
      delete k_; k_ = NULL;
   }
   else
   {
      delete a_; a_ = NULL;
      delete b_; b_ = NULL;
   }
}

void DGAdvectionDiffusionTDO::SetDiffusionCoefficient(Coefficient &dCoef)
{
   dCoef_ = &dCoef;
   if (dtdCoef_ == NULL)
   {
      dtdCoef_ = new ProductCoefficient(dt_, dCoef);
   }
   else
   {
      dtdCoef_->SetBCoef(dCoef);
   }
   if (imex_)
   {
      delete a_; a_ = NULL;
      delete s_; s_ = NULL;
   }
   else
   {
      delete a_; a_ = NULL;
      delete b_; b_ = NULL;
   }
}

void DGAdvectionDiffusionTDO::SetDiffusionCoefficient(MatrixCoefficient &DCoef)
{
   DCoef_ = &DCoef;
   if (dtDCoef_ == NULL)
   {
      dtDCoef_ = new ScalarMatrixProductCoefficient(dt_, DCoef);
   }
   else
   {
      dtDCoef_->SetBCoef(DCoef);
   }
   if (imex_)
   {
      delete a_; a_ = NULL;
      delete s_; s_ = NULL;
   }
   else
   {
      delete a_; a_ = NULL;
      delete b_; b_ = NULL;
   }
}

void DGAdvectionDiffusionTDO::SetSourceCoefficient(Coefficient &SCoef)
{
   SCoef_ = &SCoef;
   delete q_exp_; q_exp_ = NULL;
   delete q_imp_; q_imp_ = NULL;
}

void DGAdvectionDiffusionTDO::SetDirichletBC(Array<int> &dbc_attr,
                                             Coefficient &dbc)
{
   dbcAttr_ = dbc_attr;
   dbcCoef_ = &dbc;
   delete q_exp_; q_exp_ = NULL;
   delete q_imp_; q_imp_ = NULL;
}

void DGAdvectionDiffusionTDO::SetNeumannBC(Array<int> &nbc_attr,
                                           Coefficient &nbc)
{
   nbcAttr_ = nbc_attr;
   nbcCoef_ = &nbc;
   delete q_exp_; q_exp_ = NULL;
   delete q_imp_; q_imp_ = NULL;
}

void DGAdvectionDiffusionTDO::ExplicitMult(const Vector &x, Vector &fx) const
{
   MFEM_VERIFY(imex_, "Unexpected call to ExplicitMult for non-IMEX method!");

   pgf_->ExchangeFaceNbrData();

   if (q_exp_)
   {
      rhs_ = *q_exp_;
   }
   else
   {
      rhs_ = 0.0;
   }
   if (k_) { k_->AddMult(x, rhs_, -1.0); }

   rhs_.ParallelAssemble(RHS_);
   // double nrmR = InnerProduct(M_->GetComm(), RHS_, RHS_);
   // cout << "Norm^2 RHS: " << nrmR << endl;
   M_solver_.Mult(RHS_, X_);

   // double nrmX = InnerProduct(M_->GetComm(), X_, X_);
   // cout << "Norm^2 X: " << nrmX << endl;

   ParGridFunction fx_gf(fes_, (double*)NULL);
   fx_gf.MakeRef(fes_, &fx[0]);
   fx_gf = X_;
}

void DGAdvectionDiffusionTDO::ImplicitSolve(const double dt,
                                            const Vector &u, Vector &dudt)
{
   pgf_->ExchangeFaceNbrData();

   if (fes_->GetMyRank() == 0 && logging_)
   {
      cout << log_prefix_ << "ImplicitSolve with dt = " << dt << endl;
   }

   if (fabs(dt - dt_) > 1e-4 * dt_)
   {
      // cout << "Setting time step" << endl;
      if (dtdCoef_   ) { dtdCoef_->SetAConst(dt); }
      if (dtDCoef_   ) { dtDCoef_->SetAConst(dt); }
      if (dtNegVCoef_) { dtNegVCoef_->SetAConst(dt); }

      dt_ = dt;
   }
   // cout << "applying q_imp" << endl;
   if (q_imp_)
   {
      rhs_ = *q_imp_;
   }
   else
   {
      rhs_ = 0.0;
   }
   rhs_.ParallelAssemble(RHS_);

   fes_->Dof_TrueDof_Matrix()->Mult(u, X_);

   if (imex_)
   {
      if (s_)
      {
         s_->Assemble();
         s_->Finalize();

         HypreParMatrix * S = s_->ParallelAssemble();

         S->Mult(-1.0, X_, 1.0, RHS_);
         delete S;
      }
   }
   else
   {
      //cout << "applying b" << endl;
      if (b_)
      {
         // cout << "DGA::ImplicitSolve assembling b" << endl;
         b_->Assemble();
         b_->Finalize();

         HypreParMatrix * B = b_->ParallelAssemble();

         B->Mult(-1.0, X_, 1.0, RHS_);
         delete B;
      }
   }
   // cout << "DGA::ImplicitSolve assembling a" << endl;
   a_->Assemble();
   a_->Finalize();
   HypreParMatrix *A = a_->ParallelAssemble();

   HypreBoomerAMG *A_prec = new HypreBoomerAMG(*A);
   A_prec->SetPrintLevel(0);
   if (imex_)
   {
      // cout << "solving with CG" << endl;
      CGSolver *A_solver = new CGSolver(A->GetComm());
      A_solver->SetOperator(*A);
      A_solver->SetPreconditioner(*A_prec);

      A_solver->iterative_mode = false;
      A_solver->SetRelTol(1e-9);
      A_solver->SetAbsTol(0.0);
      A_solver->SetMaxIter(100);
      A_solver->SetPrintLevel(0);
      A_solver->Mult(RHS_, X_);
      delete A_solver;
   }
   else
   {
      // cout << "solving with GMRES" << endl;
      GMRESSolver *A_solver = new GMRESSolver(A->GetComm());
      A_solver->SetOperator(*A);
      A_solver->SetPreconditioner(*A_prec);

      A_solver->iterative_mode = false;
      A_solver->SetRelTol(1e-9);
      A_solver->SetAbsTol(0.0);
      A_solver->SetMaxIter(100);
      A_solver->SetPrintLevel(0);
      A_solver->Mult(RHS_, X_);
      delete A_solver;
   }
   // cout << "done with solve" << endl;

   ParGridFunction dudt_gf(fes_, (double*)NULL);
   dudt_gf.MakeRef(fes_, &dudt[0]);
   dudt_gf = X_;

   //delete A_solver;
   delete A_prec;
   delete A;
}

void DGAdvectionDiffusionTDO::Update()
{
   height = width = fes_->GetVSize();

   m_.Update(); m_.Assemble(); m_.Finalize();
   a_->Update(); // a_->Assemble(); a_->Finalize();

   if (b_) { b_->Update(); /*b_->Assemble(); b_->Finalize();*/ }
   if (s_) { s_->Update(); /*s_->Assemble(); s_->Finalize();*/ }
   if (k_) { k_->Update(); k_->Assemble(); k_->Finalize(); }
   if (q_exp_) { q_exp_->Update(); q_exp_->Assemble(); }
   if (q_imp_) { q_imp_->Update(); q_imp_->Assemble(); }

   rhs_.Update();
   RHS_.SetSize(fes_->GetTrueVSize());
   X_.SetSize(fes_->GetTrueVSize());
}

DGTransportTDO::
TransportLeftPrec::TransportLeftPrec(const Array<int> &offsets,
                                     const TransPrecParams &p,
                                     DGTransportTDO::CombinedOp &combOp)
   : BlockDiagonalPreconditioner(offsets),
     diag_prec_(5),
#ifdef MFEM_USE_SUPERLU
     slu_mat_(5),
#endif
     comb_op_(combOp),
     p_(p)
{
   diag_prec_ = NULL;
#ifdef MFEM_USE_SUPERLU
   slu_mat_ = NULL;
#endif
}

DGTransportTDO::TransportLeftPrec::~TransportLeftPrec()
{
   for (int i=0; i<diag_prec_.Size(); i++)
   {
      delete diag_prec_[i];
#ifdef MFEM_USE_SUPERLU
      delete slu_mat_[i];
#endif
   }
}

void DGTransportTDO::TransportLeftPrec::SetOperator(const Operator &op)
{
   height = width = op.Height();

   const BlockOperator *blk_op = dynamic_cast<const BlockOperator*>(&op);

   if (blk_op)
   {
      this->Offsets() = blk_op->RowOffsets();

      for (int i=0; i<diag_prec_.Size(); i++)
      {
         if (!blk_op->IsZeroBlock(i,i))
         {
            delete diag_prec_[i];

            const Operator & diag_op = blk_op->GetBlock(i,i);
            const HypreParMatrix & M =
               dynamic_cast<const HypreParMatrix&>(diag_op);

#ifdef MFEM_USE_SUPERLU
            if (p_.type == 2)
            {
               delete slu_mat_[i];
               slu_mat_[i] = new SuperLURowLocMatrix(M);
               SuperLUSolver * slu = new SuperLUSolver(MPI_COMM_WORLD);
               slu->SetOperator(*slu_mat_[i]);
               diag_prec_[i] = slu;
            }
            else
#endif
            {
               diag_prec_[i] = comb_op_.GetPreconditionerBlock(i);
            }
            SetDiagonalBlock(i, diag_prec_[i]);
         }
      }
   }
}

CG2DG::CG2DG(const ParFiniteElementSpace &fes_dg_,
             const Array<int> &cg_ess_tdof_list)
   : Operator(fes_dg_.GetTrueVSize()),
     fes_dg(fes_dg_),
     fec_cg(fes_dg.GetOrder(0), fes_dg.GetParMesh()->Dimension()),
     fes_cg(fes_dg.GetParMesh(), &fec_cg)
{
   width = fes_cg.GetTrueVSize();

   int nldof_dg = fes_dg.GetNDofs();
   int nldof_cg = fes_cg.GetNDofs();
   SparseMatrix mat(nldof_dg, nldof_cg);

   // Assembly
   DenseMatrix elmat;
   Array<int> vdofs_dg, vdofs_cg;
   for (int iel=0; iel<fes_dg.GetNE(); ++iel)
   {
      const FiniteElement &fe_dg = *fes_dg.GetFE(iel);
      const FiniteElement &fe_cg = *fes_cg.GetFE(iel);

      int ndof_dg = fe_dg.GetDof();
      int ndof_cg = fe_cg.GetDof();

      fes_dg.GetElementVDofs(iel, vdofs_dg);
      fes_cg.GetElementVDofs(iel, vdofs_cg);

      const Array<int> &dof_map =
         dynamic_cast<const NodalFiniteElement&>(fe_cg).GetLexicographicOrdering();

      elmat.SetSize(ndof_dg, ndof_cg);
      elmat = 0.0;
      for (int i=0; i<ndof_dg; ++i)
      {
         elmat(i,dof_map[i]) = 1.0;
      }
      int skip_zeros = 1;
      mat.SetSubMatrix(vdofs_dg, vdofs_cg, elmat, skip_zeros);
   }

   // Zero out the boundary for homogeneous Dirichlet BC

   Vector column_scaling(nldof_cg);
   column_scaling = 1.0;
   for (auto it = cg_ess_tdof_list.begin(); it != cg_ess_tdof_list.end(); ++it)
   {
      column_scaling(*it) = 0.0;
   }

   mat.ScaleColumns(column_scaling);

   mat.Finalize();
   C.Swap(mat);

   P = fes_cg.Dof_TrueDof_Matrix();
   z.SetSize(P->Height());
}

void CG2DG::Mult(const Vector &x, Vector &y) const
{
   if (P)
   {
      P->Mult(x, z);
      C.Mult(z, y);
   }
   else
   {
      C.Mult(x, y);
   }
}

void CG2DG::MultTranspose(const Vector &x, Vector &y) const
{
   if (P)
   {
      C.MultTranspose(x, z);
      P->MultTranspose(z, y);
   }
   else
   {
      C.MultTranspose(x, y);
   }
}

HypreParMatrix *CG2DG::ParallelAssemble()
{
   HypreParMatrix C_par(fes_dg.GetComm(),
                        fes_dg.GlobalTrueVSize(),
                        fes_cg.GlobalVSize(),
                        fes_dg.GetTrueDofOffsets(),
                        fes_cg.GetDofOffsets(),
                        &C);
   return ParMult(&C_par, P);
}

DiscontPSCPreconditioner::DiscontPSCPreconditioner(
   const CG2DG &cg2dg_,
   const Solver &cg_solver_,
   const Solver &smoother_)
   : Solver(cg2dg_.fes_dg.GetNDofs()),
     cg2dg(cg2dg_),
     cg_solver(cg_solver_),
     smoother(smoother_)
{
   b_cg.SetSize(cg2dg.fes_cg.GetTrueVSize());
   x_cg.SetSize(cg2dg.fes_cg.GetTrueVSize());
   x_z.SetSize(cg2dg.fes_dg.GetNDofs());
}

void DiscontPSCPreconditioner::Mult(const Vector &b, Vector &x) const
{
   // Smoother
   smoother.Mult(b, x_z);
   // Restrict residual
   cg2dg.MultTranspose(b, b_cg);
   // Approximate coarse solve
   cg_solver.Mult(b_cg, x_cg);
   // Prolongate
   cg2dg.Mult(x_cg, x);
   // Add edge correction term
   x += x_z;
}

inline void DGTransportTDO::GMRESRPCSolver::
GeneratePlaneRotation(double &dx, double &dy,
                      double &cs, double &sn)
{
   if (dy == 0.0)
   {
      cs = 1.0;
      sn = 0.0;
   }
   else if (fabs(dy) > fabs(dx))
   {
      double temp = dx / dy;
      sn = 1.0 / sqrt( 1.0 + temp*temp );
      cs = temp * sn;
   }
   else
   {
      double temp = dy / dx;
      cs = 1.0 / sqrt( 1.0 + temp*temp );
      sn = temp * cs;
   }
}

inline void DGTransportTDO::GMRESRPCSolver::
ApplyPlaneRotation(double &dx, double &dy, double &cs, double &sn)
{
   double temp = cs * dx + sn * dy;
   dy = -sn * dx + cs * dy;
   dx = temp;
}

inline void DGTransportTDO::GMRESRPCSolver::
Update(Vector &x, int k, DenseMatrix &h, Vector &s,
       Array<Vector*> &v)
{
   Vector y(s);

   // Backsolve:
   for (int i = k; i >= 0; i--)
   {
      y(i) /= h(i,i);
      for (int j = i - 1; j >= 0; j--)
      {
         y(j) -= h(j,i) * y(i);
      }
   }

   for (int j = 0; j <= k; j++)
   {
      x.Add(y(j), *v[j]);
   }
}

void DGTransportTDO::GMRESRPCSolver::Mult(const Vector &b, Vector &x) const
{
   // Generalized Minimum Residual method following the algorithm
   // on p. 20 of the SIAM Templates book.

   int n = width;

   DenseMatrix H(m+1, m);
   Vector s(m+1), cs(m+1), sn(m+1);
   Vector r(n), y(n), u(n), w(n);
   Array<Vector *> v;

   double resid;
   int i, j, k;

   if (iterative_mode)
   {
      oper->Mult(x, r);

      if (r_prec)
      {
         r_prec->Mult(x, y);
      }
      else
      {
         y = x;
      }
   }
   else
   {
      y = 0.0;
   }

   if (prec)
   {
      if (iterative_mode)
      {
         subtract(b, r, w);
         prec->Mult(w, r);    // r = M (b - A x)
      }
      else
      {
         prec->Mult(b, r);
      }
   }
   else
   {
      if (iterative_mode)
      {
         subtract(b, r, r);
      }
      else
      {
         r = b;
      }
   }
   double beta = Norm(r);  // beta = ||r||
   MFEM_ASSERT(IsFinite(beta), "beta = " << beta);

   final_norm = std::max(rel_tol*beta, abs_tol);

   if (beta <= final_norm)
   {
      final_norm = beta;
      final_iter = 0;
      converged = true;
      goto finish;
   }

   if (print_options.iterations || print_options.first_and_last)
   {
      mfem::out << "   Pass : " << setw(2) << 1
                << "   Iteration : " << setw(3) << 0
                << "  ||B r|| = " << beta
                << (print_options.first_and_last ? " ...\n" : "\n");
   }

   if (r_prec)
   {
      r_prec->InverseMult(y, x);
   }
   else
   {
      x = y;
   }
   Monitor(0, beta, r, x);

   v.SetSize(m+1, NULL);

   for (j = 1; j <= max_iter; )
   {
      if (v[0] == NULL) { v[0] = new Vector(n); }
      v[0]->Set(1.0/beta, r);
      s = 0.0; s(0) = beta;

      for (i = 0; i < m && j <= max_iter; i++, j++)
      {
         if (r_prec)
         {
            r_prec->InverseMult(*v[i], u);
         }
         else
         {
            u = *v[i];
         }
         if (prec)
         {
            oper->Mult(u, r);
            prec->Mult(r, w);        // w = M A v[i]
         }
         else
         {
            oper->Mult(u, w);
         }

         for (k = 0; k <= i; k++)
         {
            H(k,i) = Dot(w, *v[k]);  // H(k,i) = w * v[k]
            w.Add(-H(k,i), *v[k]);   // w -= H(k,i) * v[k]
         }

         H(i+1,i) = Norm(w);           // H(i+1,i) = ||w||
         MFEM_ASSERT(IsFinite(H(i+1,i)), "Norm(w) = " << H(i+1,i));
         if (v[i+1] == NULL) { v[i+1] = new Vector(n); }
         v[i+1]->Set(1.0/H(i+1,i), w); // v[i+1] = w / H(i+1,i)

         for (k = 0; k < i; k++)
         {
            ApplyPlaneRotation(H(k,i), H(k+1,i), cs(k), sn(k));
         }

         GeneratePlaneRotation(H(i,i), H(i+1,i), cs(i), sn(i));
         ApplyPlaneRotation(H(i,i), H(i+1,i), cs(i), sn(i));
         ApplyPlaneRotation(s(i), s(i+1), cs(i), sn(i));

         resid = fabs(s(i+1));
         MFEM_ASSERT(IsFinite(resid), "resid = " << resid);

         if (resid <= final_norm)
         {
            Update(y, i, H, s, v);
            final_norm = resid;
            final_iter = j;
            converged = true;
            goto finish;
         }

         if (print_options.iterations)
         {
            mfem::out << "   Pass : " << setw(2) << (j-1)/m+1
                      << "   Iteration : " << setw(3) << j
                      << "  ||B r|| = " << resid << '\n';
         }

         if (r_prec)
         {
            r_prec->InverseMult(y, x);
         }
         else
         {
            x = y;
         }
         Monitor(j, resid, r, x);
      }

      if (print_options.iterations && j <= max_iter)
      {
         mfem::out << "Restarting..." << '\n';
      }

      Update(y, i-1, H, s, v);

      if (r_prec)
      {
         r_prec->InverseMult(y, u);
      }
      else
      {
         u = y;
      }
      oper->Mult(u, r);
      if (prec)
      {
         subtract(b, r, w);
         prec->Mult(w, r);    // r = M (b - A x)
      }
      else
      {
         subtract(b, r, r);
      }
      beta = Norm(r);         // beta = ||r||
      MFEM_ASSERT(IsFinite(beta), "beta = " << beta);
      if (beta <= final_norm)
      {
         final_norm = beta;
         final_iter = j;
         converged = true;
         goto finish;
      }
   }

   final_norm = beta;
   final_iter = max_iter;
   converged = false;

finish:
   if ((print_options.iterations && converged) || print_options.first_and_last)
   {
      mfem::out << "   Pass : " << setw(2) << (j-1)/m+1
                << "   Iteration : " << setw(3) << final_iter
                << "  ||B r|| = " << resid << '\n';
   }
   if (print_options.summary || (print_options.warnings && !converged))
   {
      mfem::out << "GMRES: Number of iterations: " << final_iter << '\n';
   }
   if (print_options.warnings && !converged)
   {
      mfem::out << "GMRES: No convergence!\n";
   }

   if (r_prec)
   {
      r_prec->InverseMult(y, x);
   }
   else
   {
      x = y;
   }
   Monitor(final_iter, final_norm, r, x, true);

   for (i = 0; i < v.Size(); i++)
   {
      delete v[i];
   }
}

void DiscontPSCPreconditioner::SetOperator(const Operator &op) { }

AdditivePreconditioner::AdditivePreconditioner(const Solver &P1_,
                                               const Solver &P2_)
   : Solver(P1_.Height()),
     P1(P1_),
     P2(P2_),
     v(P1_.Height())
{
}

void AdditivePreconditioner::Mult(const Vector &b, Vector &x) const
{
   // Precondition by P1: x = Pb
   P1.Mult(b, x);

   // Precondition by P2: v = Pb
   P2.Mult(b, v);

   x += v;
}

void AdditivePreconditioner::SetOperator(const Operator &op)
{
   A = &op;
}

MultiplicativePreconditioner::MultiplicativePreconditioner(const Solver &P1_,
                                                           const Solver &P2_)
   : Solver(P1_.Height()),
     P1(P1_),
     P2(P2_),
     r(P1_.Height()),
     v(P1_.Height())
{
}

void MultiplicativePreconditioner::Mult(const Vector &b, Vector &x) const
{
   // Precondition by P1: x = Pb
   P1.Mult(b, x);

   // Compute residual r = Ax - b = APb - b
   A->Mult(x, r);
   r -= b;

   // Precondition by P2: v = Pr
   P2.Mult(r, v);

   // Now v approximates the error e = A^{-1} r = x - A^{-1} b
   x -= v;
}

void MultiplicativePreconditioner::SetOperator(const Operator &op)
{
   A = &op;
}

DGTransportTDO::DGTransportTDO(const MPI_Session &mpi, const DGParams &dg,
                               const std::vector<ArtViscParams> & av,
                               const PlasmaParams &plasma,
                               const SolverParams & tol,
                               const Vector &eqn_weights,
                               const Vector &fld_weights,
                               ParFiniteElementSpace &fes,
                               ParFiniteElementSpace &vfes,
                               ParFiniteElementSpace &ffes,
                               ParFiniteElementSpace &h1_fes,
                               Array<int> &offsets,
                               ParGridFunctionArray &yGF,
                               ParGridFunctionArray &kGF,
                               const TransportBCs & bcs,
                               const TransportCoefs & coefs,
                               double Di_perp, double Xi_perp, double Xe_perp,
                               const Array<int> &term_flags,
                               const Array<int> &vis_flags,
                               bool imex, unsigned int op_flag, int logging)
   : TimeDependentOperator(ffes.GetVSize()),
     mpi_(mpi),
     logging_(logging),
     op_flag_(op_flag),
     fes_(fes),
     ffes_(ffes),
     yGF_(yGF),
     kGF_(kGF),
     offsets_(offsets),
     tol_(tol),
     op_(mpi, dg, av, plasma, eqn_weights, vfes, h1_fes, yGF, kGF,
         bcs, coefs, offsets_,
         Di_perp, Xi_perp, Xe_perp,
         term_flags, vis_flags, op_flag, logging),
     newton_op_l_prec_(offsets, tol.prec, op_),
     newton_op_r_prec_(offsets, fld_weights, tol.prec, op_),
     newton_op_solver_(fes.GetComm()),
     newton_solver_(fes.GetComm())
{
   if ( mpi_.Root() && logging_ > 1)
   {
      cout << "Constructing DGTransportTDO" << endl;
   }

   newton_op_solver_.SetRelTol(tol_.lin_rel_tol);
   newton_op_solver_.SetAbsTol(tol_.lin_abs_tol);
   newton_op_solver_.SetMaxIter(tol_.lin_max_iter);
   newton_op_solver_.SetPrintLevel(tol_.lin_log_lvl);
   newton_op_solver_.SetPreconditioner(newton_op_l_prec_);
   newton_op_solver_.SetRightPreconditioner(newton_op_r_prec_);

   newton_solver_.iterative_mode = false;
   newton_solver_.SetSolver(newton_op_solver_);
   newton_solver_.SetOperator(op_);
   newton_solver_.SetPrintLevel(tol_.newt_log_lvl); // print Newton iterations
   newton_solver_.SetRelTol(tol_.newt_rel_tol);
   newton_solver_.SetAbsTol(tol_.newt_abs_tol);
   newton_solver_.SetMaxIter(tol_.newt_max_iter);

   if (tol_.ss_abs_tol > 0.0 || tol_.ss_rel_tol > 0.0)
   {
      kMax_.SetSize(kGF_.Size());
      kMax_ = 0.0;

      ss_.SetSize(kGF_.Size());
      ss_ = false;
   }

   if (mpi_.Root() && logging_ > 1)
   {
      cout << "Done constructing DGTransportTDO" << endl;
   }
}

DGTransportTDO::~DGTransportTDO()
{
   map<string, socketstream*>::iterator mit;
   for (mit=socks_.begin(); mit!=socks_.end(); mit++)
   {
      delete mit->second;
   }
}

void DGTransportTDO::SetTime(const double _t)
{
   if (mpi_.Root() && logging_ > 1)
   {
      cout << "Entering DGTransportTDO::SetTime with t = " << _t << endl;
   }

   op_.SetTime(_t);

   this->TimeDependentOperator::SetTime(_t);

   if ( mpi_.Root() && logging_ > 1)
   {
      cout << "Leaving DGTransportTDO::SetTime" << endl;
   }
}

void DGTransportTDO::SetLogging(int logging)
{
   op_.SetLogging(logging);
}

double DGTransportTDO::CheckGradient()
{
   int tsize = newton_solver_.Height();
   int fsize = tsize / 5;
   Vector k(tsize);
   Vector h(tsize);

   k.Randomize(1234);
   h.Randomize(5678);

   ConstantCoefficient zeroCoef(0.0);
   for (int i=0; i<5; i++)
   {
      double nrm = ((op_flag_ >> i) & 1) ?
                   yGF_[i]->ComputeMaxError(zeroCoef) : 0.0;

      for (int j=0; j<fsize; j++)
      {
         k[i*fsize + j] = 1e-2 * nrm * (2.0 * k[i*fsize + j] - 1.0);
         h[i*fsize + j] = 1e-4 * nrm * (2.0 * h[i*fsize + j] - 1.0);
      }
   }

   yGF_.ExchangeFaceNbrData();
   double f = newton_solver_.CheckGradient(k, h);
   return f;
}

bool DGTransportTDO::CheckForSteadyState()
{
   bool ss = ss_.Size() > 0;

   for (int i=0; i<ss_.Size(); i++)
   {
      ss &= ss_[i];
   }

   return ss;
}

void
DGTransportTDO::RegisterDataFields(DataCollection & dc)
{
   dc_ = &dc;

   op_.RegisterDataFields(dc);
}

void
DGTransportTDO::PrepareDataFields()
{
   op_.PrepareDataFields();
}

void
DGTransportTDO::InitializeGLVis()
{
   if ( mpi_.Root() && logging_ > 0 )
   { cout << "Opening GLVis sockets." << endl << flush; }
   op_.InitializeGLVis();
}

void
DGTransportTDO::DisplayToGLVis()
{
   if ( mpi_.Root() && logging_ > 1 )
   { cout << "Sending data to GLVis ..." << flush; }

   op_.DisplayToGLVis();

   if ( mpi_.Root() && logging_ > 1 ) { cout << " " << flush; }
}

void DGTransportTDO::ImplicitSolve(const double dt, const Vector &y,
                                   Vector &k)
{
   if (mpi_.Root() && logging_ > 1)
   {
      cout << "Entering DGTransportTDO::ImplicitSolve" << endl;
   }

   k = 0.0;

   // Coefficient inside the NLOperator classes use data from yGF_ to evaluate
   // fields.  We need to make sure this data accesses the provided vector y.
   double *prev_y = yGF_[0]->GetData();

   for (int i=0; i<offsets_.Size() - 1; i++)
   {
      yGF_[i]->MakeRef(&fes_, y.GetData() + offsets_[i]);
   }
   yGF_.ExchangeFaceNbrData();

   double *prev_k = kGF_[0]->GetData();

   for (int i=0; i<offsets_.Size() - 1; i++)
   {
      kGF_[i]->MakeRef(&fes_, k.GetData() + offsets_[i]);
   }
   kGF_.ExchangeFaceNbrData();

   if (mpi_.Root() && logging_ > 0)
   {
      cout << "Setting time step: " << dt << " in DGTransportTDO" << endl;
   }
   op_.SetTimeStep(dt);

   Vector zeroVec;
   newton_solver_.Mult(zeroVec, k);

   if (kMax_.Size() == kGF_.Size())
   {
      ConstantCoefficient zeroCoef(0.0);
      for (int i=0; i<kGF_.Size(); i++)
      {
         if ((op_flag_ >> i) & 1)
         {
            double kNrm = kGF_[i]->ComputeL2Error(zeroCoef);
            kMax_[i] = std::max(kMax_[i], kNrm);

            ss_[i] = kNrm < tol_.ss_abs_tol ||
                     (kNrm < tol_.ss_rel_tol * kMax_[i]);
         }
         else
         {
            ss_[i] = true;
         }
      }
   }

   // Restore the data arrays to those used before this method was called.
   for (int i=0; i<offsets_.Size() - 1; i++)
   {
      yGF_[i]->MakeRef(&fes_, prev_y + offsets_[i]);
   }
   yGF_.ExchangeFaceNbrData();

   for (int i=0; i<offsets_.Size() - 1; i++)
   {
      kGF_[i]->MakeRef(&fes_, prev_k + offsets_[i]);
   }
   if (prev_k != NULL)
   {
      kGF_.ExchangeFaceNbrData();
   }

   if (mpi_.Root() && logging_ > 1)
   {
      cout << "Leaving DGTransportTDO::ImplicitSolve" << endl;
   }
}

void DGTransportTDO::Update()
{
   if (mpi_.Root() && logging_ > 1)
   {
      cout << "Entering DGTransportTDO::Update" << endl;
   }

   height = width = ffes_.GetVSize();

   op_.Update();

   newton_solver_.SetOperator(op_);

   if (mpi_.Root() && logging_ > 1)
   {
      cout << "Leaving DGTransportTDO::Update" << endl;
   }
}

DGTransportTDO::NLOperator::NLOperator(const MPI_Session & mpi,
                                       const DGParams & dg,
                                       int index,
                                       const string & eqn_name,
                                       const string & field_name,
                                       ParGridFunctionArray & yGF,
                                       ParGridFunctionArray & kGF,
                                       int term_flag, int vis_flag,
                                       int logging,
                                       const string & log_prefix)
   : Operator(yGF[0]->ParFESpace()->GetVSize(),
              5*(yGF[0]->ParFESpace()->GetVSize())),
     mpi_(mpi), dg_(dg),
     logging_(logging), log_prefix_(log_prefix),
     index_(index),
     eqn_name_(eqn_name),
     field_name_(field_name),
     dt_(0.0),
     fes_(*yGF[0]->ParFESpace()),
     pmesh_(*fes_.GetParMesh()),
     yGF_(yGF),
     kGF_(kGF),
     yCoefPtrs_(yGF_.Size()),
     kCoefPtrs_(kGF_.Size()),
     ykCoefPtrs_(kGF_.Size()),
     dbfi_m_(5),
     dbfi_(5),
     fbfi_(5),
     bfbfi_(5),
     bfbfi_marker_(5),
     blf_(5),
     cgblf_(5),
     term_flag_(term_flag),
     vis_flag_(vis_flag),
     dc_(NULL)
{
   if ( mpi_.Root() && logging_ > 1)
   {
      cout << "Constructing NLOperator for " << eqn_name << endl;
   }

   MFEM_VERIFY(yGF.Size() == kGF.Size(), "Mismatch in yGF and kGF sizes");

   for (int i=0; i<yGF_.Size(); i++)
   {
      yCoefPtrs_[i] = new StateVariableGridFunctionCoef(yGF_[i], (FieldType)i);
      kCoefPtrs_[i] = new StateVariableGridFunctionCoef(kGF_[i],
                                                        INVALID_FIELD_TYPE);

      // y + dt k
      // Note that dt_ has not been set yet but we use it here for emphasis
      ykCoefPtrs_[i] = new StateVariableSumCoef(*yCoefPtrs_[i],
                                                *kCoefPtrs_[i],
                                                1.0, dt_);
   }

   blf_ = NULL;
   cgblf_ = NULL;

   if ( mpi_.Root() && logging_ > 1)
   {
      cout << "Done constructing NLOperator for " << eqn_name << endl;
   }
}

DGTransportTDO::NLOperator::~NLOperator()
{
   for (int i=0; i<blf_.Size(); i++)
   {
      delete blf_[i];
   }
   for (int i=0; i<cgblf_.Size(); i++)
   {
      delete cgblf_[i];
   }

   for (int i=0; i<dbfi_m_.size(); i++)
   {
      for (int j=0; j<dbfi_m_[i].Size(); j++)
      {
         delete dbfi_m_[i][j];
      }
   }
   for (int i=0; i<dbfi_.size(); i++)
   {
      for (int j=0; j<dbfi_[i].Size(); j++)
      {
         delete dbfi_[i][j];
      }
   }
   for (int i=0; i<fbfi_.size(); i++)
   {
      for (int j=0; j<fbfi_[i].Size(); j++)
      {
         delete fbfi_[i][j];
      }
   }
   for (int i=0; i<bfbfi_.size(); i++)
   {
      for (int j=0; j<bfbfi_[i].Size(); j++)
      {
         delete bfbfi_[i][j];
      }
   }
   for (int i=0; i<bfbfi_marker_.size(); i++)
   {
      for (int j=0; j<bfbfi_marker_[i].Size(); j++)
      {
         delete bfbfi_marker_[i][j];
      }
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

   delete D_smoother_;
   delete D_amg_;
   delete cg2dg_;
}

void DGTransportTDO::NLOperator::SetLogging(int logging, const string & prefix)
{
   logging_ = logging;
   log_prefix_ = prefix;
}

void DGTransportTDO::NLOperator::SetTimeStep(double dt)
{
   dt_ = dt;

   for (int i=0; i<ykCoefPtrs_.Size(); i++)
   {
      ykCoefPtrs_[i]->SetBeta(dt);
   }
}

void DGTransportTDO::NLOperator::Mult(const Vector &, Vector &r) const
{
   if (mpi_.Root() && logging_ > 1)
   {
      cout << log_prefix_ << "DGTransportTDO::NLOperator::Mult" << endl;
   }

   r = 0.0;

   for (int i=0; i < fes_.GetNE(); i++)
   {
      fes_.GetElementVDofs(i, vdofs_);

      const FiniteElement &fe = *fes_.GetFE(i);
      ElementTransformation *eltrans = fes_.GetElementTransformation(i);

      int ndof = vdofs_.Size();

      elvec_.SetSize(ndof);
      locdvec_.SetSize(ndof);

      elvec_ = 0.0;

      for (int j=0; j<5; j++)
      {
         if (dbfi_m_[j].Size() > 0)
         {
            kGF_[j]->GetSubVector(vdofs_, locdvec_);

            dbfi_m_[j][0]->AssembleElementMatrix(fe, *eltrans, elmat_);
            for (int k = 1; k < dbfi_m_[j].Size(); k++)
            {
               dbfi_m_[j][k]->AssembleElementMatrix(fe, *eltrans, elmat_k_);
               elmat_ += elmat_k_;
            }

            elmat_.AddMult(locdvec_, elvec_);
         }
      }
      r.AddElementVector(vdofs_, elvec_);
   }

   if (mpi_.Root() && logging_ > 2)
   {
      cout << log_prefix_
           << "DGTransportTDO::NLOperator::Mult element loop done" << endl;
   }

   int dbfi_size = 0;
   for (int j=0; j<5; j++)
   {
      dbfi_size += dbfi_[j].Size();
   }
   if (dbfi_size)
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

         elvec_ = 0.0;

         for (int j=0; j<5; j++)
         {
            if (dbfi_[j].Size() > 0)
            {
               yGF_[j]->GetSubVector(vdofs_, locvec_);
               kGF_[j]->GetSubVector(vdofs_, locdvec_);

               locvec_.Add(dt_, locdvec_);

               dbfi_[j][0]->AssembleElementMatrix(fe, *eltrans, elmat_);
               for (int k = 1; k < dbfi_[j].Size(); k++)
               {
                  dbfi_[j][k]->AssembleElementMatrix(fe, *eltrans, elmat_k_);
                  elmat_ += elmat_k_;
               }

               elmat_.AddMult(locvec_, elvec_);
            }
         }
         r.AddElementVector(vdofs_, elvec_);
      }
   }

   if (mpi_.Root() && logging_ > 2)
   {
      cout << log_prefix_
           << "DGTransportTDO::NLOperator::Mult element loop done" << endl;
   }
   int fbfi_size = 0;
   for (int j=0; j<5; j++)
   {
      fbfi_size += fbfi_[j].Size();
   }
   if (fbfi_size)
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

            int ndof = vdofs_.Size();

            elvec_.SetSize(ndof);
            locvec_.SetSize(ndof);
            locdvec_.SetSize(ndof);

            elvec_ = 0.0;

            for (int j=0; j<5; j++)
            {
               if (fbfi_[j].Size() > 0)
               {
                  yGF_[j]->GetSubVector(vdofs_, locvec_);
                  kGF_[j]->GetSubVector(vdofs_, locdvec_);

                  locvec_.Add(dt_, locdvec_);

                  fbfi_[j][0]->AssembleFaceMatrix(fe1, fe2, *ftrans, elmat_);
                  for (int k = 1; k < fbfi_[j].Size(); k++)
                  {
                     fbfi_[j][k]->AssembleFaceMatrix(fe1, fe2, *ftrans, elmat_k_);
                     elmat_ += elmat_k_;
                  }

                  elmat_.AddMult(locvec_, elvec_);
               }
            }
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

         const FiniteElement &fe1 = *fes_.GetFE(ftrans->Elem1No);
         const FiniteElement &fe2 = *fes_.GetFaceNbrFE(nbr_el_no);

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

         elvec_ = 0.0;

         for (int j=0; j<5; j++)
         {
            if (fbfi_[j].Size() > 0)
            {
               yGF_[j]->GetSubVector(vdofs_, locvec1);
               kGF_[j]->GetSubVector(vdofs_, locdvec1);

               yGF_[j]->FaceNbrData().GetSubVector(vdofs2_, locvec2);
               kGF_[j]->FaceNbrData().GetSubVector(vdofs2_, locdvec2);

               locvec_.Add(dt_, locdvec_);

               fbfi_[j][0]->AssembleFaceMatrix(fe1, fe2, *ftrans, elmat_);
               for (int k = 1; k < fbfi_[j].Size(); k++)
               {
                  fbfi_[j][k]->AssembleFaceMatrix(fe1, fe2, *ftrans, elmat_k_);
                  elmat_ += elmat_k_;
               }

               elmat_.AddMult(locvec_, elvec_);
            }
         }
         r.AddElementVector(vdofs_, elvec);
      }
   }

   if (mpi_.Root() && logging_ > 2)
   {
      cout << log_prefix_
           << "DGTransportTDO::NLOperator::Mult face loop done" << endl;
   }
   int bfbfi_size = 0;
   for (int j=0; j<5; j++)
   {
      bfbfi_size += bfbfi_[j].Size();
   }
   if (bfbfi_size)
   {
      FaceElementTransformations *ftrans = NULL;

      // Which boundary attributes need to be processed?
      Array<int> bdr_attr_marker(pmesh_.bdr_attributes.Size() ?
                                 pmesh_.bdr_attributes.Max() : 0);
      bdr_attr_marker = 0;
      for (int j=0; j<5; j++)
      {
         if (bfbfi_[j].Size() > 0)
         {
            for (int k = 0; k < bfbfi_[j].Size(); k++)
            {
               if (bfbfi_marker_[j][k] == NULL)
               {
                  bdr_attr_marker = 1;
                  break;
               }
               const Array<int> &bdr_marker = *bfbfi_marker_[j][k];
               MFEM_ASSERT(bdr_marker.Size() == bdr_attr_marker.Size(),
                           "invalid boundary marker for boundary face "
                           "integrator #"
                           << k << ", counting from zero");
               for (int i = 0; i < bdr_attr_marker.Size(); i++)
               {
                  bdr_attr_marker[i] |= bdr_marker[i];
               }
            }
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

            const FiniteElement &fe1 = *fes_.GetFE(ftrans->Elem1No);
            const FiniteElement &fe2 = fe1;

            int ndof = vdofs_.Size();

            elmat_.SetSize(ndof);
            elvec_.SetSize(ndof);
            locvec_.SetSize(ndof);
            locdvec_.SetSize(ndof);

            elvec_ = 0.0;

            for (int j=0; j<5; j++)
            {
               if (bfbfi_[j].Size() > 0)
               {
                  yGF_[j]->GetSubVector(vdofs_, locvec_);
                  kGF_[j]->GetSubVector(vdofs_, locdvec_);

                  locvec_.Add(dt_, locdvec_);

                  elmat_ = 0.0;

                  for (int k = 0; k < bfbfi_[j].Size(); k++)
                  {
                     if (bfbfi_marker_[j][k] != NULL)
                        if ((*bfbfi_marker_[j][k])[bdr_attr-1] == 0) { continue; }

                     bfbfi_[j][k]->AssembleFaceMatrix(fe1, fe2, *ftrans, elmat_k_);
                     elmat_ += elmat_k_;
                  }

                  elmat_.AddMult(locvec_, elvec_);
               }
            }
            r.AddElementVector(vdofs_, elvec_);
         }
      }
   }

   if (dlfi_.Size())
   {
      ElementTransformation *eltrans = NULL;

      for (int i=0; i < fes_.GetNE(); i++)
      {
         fes_.GetElementVDofs(i, vdofs_);
         eltrans = fes_.GetElementTransformation(i);

         int ndof = vdofs_.Size();
         elvec_.SetSize(ndof);

         for (int k=0; k < dlfi_.Size(); k++)
         {
            dlfi_[k]->AssembleRHSElementVect(*fes_.GetFE(i), *eltrans, elvec_);
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
      for (int k = 0; k < bflfi_.Size(); k++)
      {
         if (bflfi_marker_[k] == NULL)
         {
            bdr_attr_marker = 1;
            break;
         }
         const Array<int> &bdr_marker = *bflfi_marker_[k];
         MFEM_ASSERT(bdr_marker.Size() == bdr_attr_marker.Size(),
                     "invalid boundary marker for boundary face integrator #"
                     << k << ", counting from zero");
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

            for (int k = 0; k < bflfi_.Size(); k++)
            {
               if (bflfi_marker_[k] &&
                   (*bflfi_marker_[k])[bdr_attr-1] == 0) { continue; }

               bflfi_[k] -> AssembleRHSElementVect (*fes_.GetFE(tr -> Elem1No),
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
           << "DGTransportTDO::NLOperator::Mult done" << endl;
   }
}

void DGTransportTDO::NLOperator::Update()
{
   if (mpi_.Root() && logging_ > 1)
   {
      cout << "Entering DGTransportTDO::NLOperator::Update" << endl;
   }

   height = fes_.GetVSize();
   width  = 5 * fes_.GetVSize();

   for (int i=0; i<5; i++)
   {
      if (blf_[i] != NULL)
      {
         blf_[i]->Update();
      }
      if (cgblf_[i] != NULL)
      {
         cgblf_[i]->Update();
      }
   }

   if (mpi_.Root() && logging_ > 1)
   {
      cout << "Leaving DGTransportTDO::NLOperator::Update" << endl;
   }
}

Operator *DGTransportTDO::NLOperator::GetGradientBlock(int i)
{
   if (mpi_.Root() && logging_ > 1)
   {
      cout << "Entering DGTransportTDO::NLOperator::GetGradientBlock("
           << i << ") for equation " << index_ << endl;
   }

   if ( blf_[i] != NULL)
   {
      //cout << "Non-null gradient block " << i << endl;
      blf_[i]->Update(); // Clears the matrix so we start from 0 again
      blf_[i]->Assemble(0);
      blf_[i]->Finalize(0);

      HypreParMatrix * Dmat = blf_[i]->ParallelAssemble();
      Operator * D = Dmat;

      if (index_ == i)  // Diagonal term
      {
         if (cgblf_[i])
         {
            // Construct CG space operator and preconditioner
            ParFiniteElementSpace *fes_dg = blf_[i]->ParFESpace();
            delete cg2dg_;
            delete D_amg_;
            delete D_smoother_;
            // Note that dg_precond_ should not be deleted here,
            // as it becomes diag_prec_[i] for IonDensity, which
            // gets deleted elsewhere.

            cg2dg_ = new CG2DG(*fes_dg, cg_ess_tdof_list);
            delete CG2DGmat_;

            const bool algebraic = false;
            if (algebraic) // Algebraic version
            {
               CG2DGmat_ = cg2dg_->ParallelAssemble();
               D_cg_ = RAP(Dmat, CG2DGmat_);
               D_cg_->EliminateZeroRows();
            }
            else // CG discretization version
            {
               MFEM_VERIFY(cgblf_[i], "");
               cgblf_[i]->Update(); // Clears the matrix so we start from 0 again
               cgblf_[i]->Assemble(0);
               cgblf_[i]->Finalize(0);

               OperatorPtr A_cg;
               cgblf_[i]->FormSystemMatrix(cg_ess_tdof_list, A_cg);
               D_cg_ = A_cg.As<HypreParMatrix>();
            }

            // Set up the preconditioner in CG space
            if (use_lor_cg)
            {
               delete D_lor_;
               D_lor_ = new ParLORDiscretization(*cgblf_[i], cg_ess_tdof_list);
               pmesh_.ExchangeFaceNbrNodes();
               if (use_air_cg)
               {
                  MFEM_VERIFY(cgblf_[i], "");
                  const int block_size = cgblf_[i]->ParFESpace()->GetFE(0)->GetDof();
                  D_amg_ = new LORSolver<AIR_prec>(*D_lor_, block_size);
               }
               else
               {
                  D_amg_ = new LORSolver<HypreBoomerAMG>(*D_lor_);
               }
            }
            else
            {
               if (use_air_cg)
               {
                  MFEM_VERIFY(cgblf_[i], "");
                  const int block_size = cgblf_[i]->ParFESpace()->GetFE(0)->GetDof();
                  D_amg_ = new AIR_prec(block_size);
                  D_amg_->SetOperator(*D_cg_);
               }
               else
               {
                  D_amg_ = new HypreBoomerAMG(*D_cg_);
               }
            }

            D_smoother_ = new HypreSmoother(*Dmat, HypreSmoother::Jacobi);
            if (use_schwarz)
            {

               //D_schwarz_ = new SchwarzSmoother(cgblf_[i]->ParFESpace()->GetParMesh(),
               //                                 0, cgblf_[i]->ParFESpace(), D_cg_);  // TODO: delete this pointer
               D_mult_ = new MultiplicativePreconditioner(*D_amg_,
                                                          *D_schwarz_);  // TODO: delete this pointer

               D_mult_->SetOperator(*D_cg_);
               dg_precond_ = new DiscontPSCPreconditioner(*cg2dg_, *D_mult_, *D_smoother_);
               //dg_precond_ = new DiscontPSCPreconditioner(*cg2dg_, *D_schwarz_, *D_smoother_);
               //dg_precond_ = new DiscontPSCPreconditioner(*cg2dg_, *D_amg_, *D_smoother_);
            }
            else
            {
               dg_precond_ = new DiscontPSCPreconditioner(*cg2dg_, *D_amg_, *D_smoother_);
            }
         }
         else
         {
            // The operator is just a DG mass matrix so return a simple precond
            dg_precond_ = new HypreDiagScale(*Dmat);
         }
      }

      return D;
   }
   else
   {
      return NULL;
   }
}

void
DGTransportTDO::NLOperator::RegisterDataFields(DataCollection & dc)
{
   dc_ = &dc;

   if (this->CheckVisFlag(0))
   {
      dc.RegisterField(field_name_, yGF_[index_]);
   }
}

void
DGTransportTDO::NLOperator::PrepareDataFields()
{
}

DGTransportTDO::TransportOp::TransportOp(const MPI_Session & mpi,
                                         const DGParams & dg,
                                         const PlasmaParams & plasma, int index,
                                         const std::string &eqn_name,
                                         const std::string &field_name,
                                         ParFiniteElementSpace * vfes,
                                         ParFiniteElementSpace * h1_fes,
                                         ParGridFunctionArray & yGF,
                                         ParGridFunctionArray & kGF,
                                         ParGridFunction & elOrdGF,
                                         ParGridFunction & hGF,
                                         const AdvectionDiffusionBC & bcs,
                                         const CoupledBCs & cbcs,
                                         const CommonCoefs & common_coefs,
                                         VectorCoefficient & B3Coef,
                                         int term_flag, int vis_flag,
                                         int logging,
                                         const std::string & log_prefix)
   : NLOperator(mpi, dg, index, eqn_name, field_name,
                yGF, kGF, term_flag, vis_flag, logging, log_prefix),
     coefGF_(yGF[0]->ParFESpace()),
     vfes_(vfes),
     h1_fes_(h1_fes),
     plasma_(plasma),
     m_n_kg_(plasma.m_n_kg),
     T_n_eV_(plasma.T_n_eV),
     v_n_(sqrt(8.0 * T_n_eV_ * J_per_eV_ / (M_PI * m_n_kg_))),
     m_i_kg_(plasma.m_i_kg),
     z_i_(plasma.z_i),
     nnCoef_(*ykCoefPtrs_[NEUTRAL_DENSITY]),
     niCoef_(*ykCoefPtrs_[ION_DENSITY]),
     viCoef_(*ykCoefPtrs_[ION_PARA_VELOCITY]),
     TiCoef_(*ykCoefPtrs_[ION_TEMPERATURE]),
     TeCoef_(*ykCoefPtrs_[ELECTRON_TEMPERATURE]),
     vnAvgCoef_(plasma.v_n_avg_m_per_s),
     vnBarCoef_(plasma.v_n_bar_m_per_s),
     TnCoef_(plasma.T_n_eV),
     ziCoef_((double)z_i_),
     neCoef_(ziCoef_, niCoef_),
     dTe0Coef_(*kCoefPtrs_[ELECTRON_TEMPERATURE]),
     lnLambda_(TeCoef_, neCoef_, plasma.z_i),
     // CsCoef_(m_i_kg_, TiCoef_, TeCoef_),
     bcs_(bcs),
     cbcs_(cbcs),
     cmncoefs_(common_coefs),
     elOrdCoef_(&elOrdGF),
     hCoef_(&hGF),
     B3Coef_(B3Coef),
     BxyCoef_(B3Coef),
     UnitBxyCoef_(B3Coef),
     massCoef_(NULL),
     diffusionCoef_(NULL),
     diffusionMatrixCoef_(NULL),
     // advectionCoef_(NULL),
     izCoef_(TeCoef_),
     rcCoef_(TeCoef_),
     cxCoef_(TiCoef_),
     SizDefCoef_(neCoef_, nnCoef_, izCoef_),
     SrcDefCoef_(neCoef_, niCoef_, rcCoef_),
     ScxDefCoef_(nnCoef_, niCoef_, cxCoef_),
     SizCoef_((cmncoefs_(CmnCoefs::IONIZATION_COEF) != NULL)
              ? const_cast<StateVariableCoef&>
              (*cmncoefs_(CmnCoefs::IONIZATION_COEF))
              : SizDefCoef_),
     SrcCoef_((cmncoefs_(CmnCoefs::RECOMBINATION_COEF) != NULL)
              ? const_cast<StateVariableCoef&>
              (*cmncoefs_(CmnCoefs::RECOMBINATION_COEF))
              : SrcDefCoef_),
     ScxCoef_((cmncoefs_(CmnCoefs::CHARGE_EXCHANGE_COEF) != NULL)
              ? const_cast<StateVariableCoef&>
              (*cmncoefs_(CmnCoefs::CHARGE_EXCHANGE_COEF))
              : ScxDefCoef_)
{
}

DGTransportTDO::TransportOp::~TransportOp()
{
   for (int i=0; i<svscoefs_.Size(); i++)
   {
      delete svscoefs_[i];
   }
   for (int i=0; i<svvcoefs_.Size(); i++)
   {
      delete svvcoefs_[i];
   }
   for (int i=0; i<svmcoefs_.Size(); i++)
   {
      delete svmcoefs_[i];
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
   for (int i=0; i<negdtVCoefs_.Size(); i++)
   {
      delete negdtVCoefs_[i];
   }
   for (int i=0; i<dtMCoefs_.Size(); i++)
   {
      delete dtMCoefs_[i];
   }
   for (int i=0; i<sCoefs_.Size(); i++)
   {
      delete sCoefs_[i];
   }
   for (int i=0; i<vCoefs_.Size(); i++)
   {
      delete vCoefs_[i];
   }
   for (int i=0; i<mCoefs_.Size(); i++)
   {
      delete mCoefs_[i];
   }
   for (int i=0; i<yGF_.Size(); i++)
   {
      delete yCoefPtrs_[i];
      delete kCoefPtrs_[i];
      delete ykCoefPtrs_[i];
   }

   for (unsigned int i=0; i<sout_.size(); i++)
   {
      delete sout_[i];
   }
}

void DGTransportTDO::TransportOp::SetTime(double t)
{
   if (mpi_.Root() && logging_ > 1)
   {
      cout << "Setting time: " << t << " in TransportOp (" << eqn_name_ << ")"
           << endl;
   }

   // NLOperator::SetTime(t);

   bcs_.SetTime(t);

   for (int i=0; i<svscoefs_.Size(); i++)
   {
      svscoefs_[i]->SetTime(t);
   }
   for (int i=0; i<svvcoefs_.Size(); i++)
   {
      svvcoefs_[i]->SetTime(t);
   }
   for (int i=0; i<svmcoefs_.Size(); i++)
   {
      svmcoefs_[i]->SetTime(t);
   }
   for (int i=0; i<dtSCoefs_.Size(); i++)
   {
      dtSCoefs_[i]->SetTime(t);
   }
   for (int i=0; i<negdtSCoefs_.Size(); i++)
   {
      negdtSCoefs_[i]->SetTime(t);
   }
   for (int i=0; i<dtVCoefs_.Size(); i++)
   {
      dtVCoefs_[i]->SetTime(t);
   }
   for (int i=0; i<negdtVCoefs_.Size(); i++)
   {
      negdtVCoefs_[i]->SetTime(t);
   }
   for (int i=0; i<dtMCoefs_.Size(); i++)
   {
      dtMCoefs_[i]->SetTime(t);
   }
   for (int i=0; i<sCoefs_.Size(); i++)
   {
      sCoefs_[i]->SetTime(t);
   }
   for (int i=0; i<mCoefs_.Size(); i++)
   {
      mCoefs_[i]->SetTime(t);
   }

   if (massCoef_) { massCoef_->SetTime(t); }
   if (diffusionCoef_) { diffusionCoef_->SetTime(t); }
   if (diffusionMatrixCoef_) { diffusionMatrixCoef_->SetTime(t); }
   // if (advectionCoef_) { advectionCoef_->SetTime(t); }
}

void DGTransportTDO::TransportOp::SetTimeStep(double dt)
{
   if (mpi_.Root() && logging_ > 1)
   {
      cout << "Setting time step: " << dt << " in TransportOp ("
           << eqn_name_ << ")"
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
   for (int i=0; i<negdtVCoefs_.Size(); i++)
   {
      negdtVCoefs_[i]->SetAConst(-dt);
   }
   for (int i=0; i<dtMCoefs_.Size(); i++)
   {
      dtMCoefs_[i]->SetAConst(dt);
   }
}

void DGTransportTDO::TransportOp::SetTimeDerivativeTerm(
   StateVariableCoef &MCoef)
{
   for (int i=0; i<5; i++)
   {
      if (MCoef.NonTrivialValue((FieldType)i))
      {
         if ( mpi_.Root() && logging_ > 0)
         {
            cout << eqn_name_
                 << ": Adding time derivative term proportional to d "
                 << FieldSymbol((FieldType)i) << " / dt" << endl;
         }

         StateVariableCoef * coef = MCoef.Clone();
         coef->SetDerivType((FieldType)i);
         svscoefs_.Append(coef);
         dbfi_m_[i].Append(new MassIntegrator(*coef));

         if (blf_[i] == NULL)
         {
            blf_[i] = new ParBilinearForm(&fes_);
         }
         blf_[i]->AddDomainIntegrator(new MassIntegrator(*coef));

         if (index_ == i && h1_fes_ != NULL )
         {
            if (cgblf_[i] == NULL)
            {
               cgblf_[i] = new ParBilinearForm(h1_fes_);
               //cout << "Number of CG dofs " << h1_fes_.GlobalTrueVSize() << endl;
               //cout << "Number of DG dofs " << fes_.GlobalTrueVSize() << endl;
            }
            cgblf_[i]->AddDomainIntegrator(new MassIntegrator(*coef));
         }
      }
   }

   massCoef_ = &MCoef;
}

void DGTransportTDO::TransportOp::SetDiffusionTerm(StateVariableCoef &DCoef)
{
   if ( mpi_.Root() && logging_ > 0)
   {
      cout << eqn_name_ << ": Adding isotropic diffusion term" << endl;
   }

   diffusionCoef_ = &DCoef;

   ProductCoefficient * dtDCoef = new ProductCoefficient(dt_, DCoef);
   dtSCoefs_.Append(dtDCoef);

   dbfi_[index_].Append(new DiffusionIntegrator(DCoef));
   fbfi_[index_].Append(new DGDiffusionIntegrator(DCoef,
                                                  dg_.sigma,
                                                  dg_.kappa));

   if (blf_[index_] == NULL)
   {
      blf_[index_] = new ParBilinearForm(&fes_);
   }

   blf_[index_]->AddDomainIntegrator(new DiffusionIntegrator(*dtDCoef));
   blf_[index_]->AddInteriorFaceIntegrator(
      new DGDiffusionIntegrator(*dtDCoef,
                                dg_.sigma,
                                dg_.kappa));

   if (h1_fes_ != NULL)
   {
      if (cgblf_[index_] == NULL)
      {
         cgblf_[index_] = new ParBilinearForm(h1_fes_);
      }
      cgblf_[index_]->AddDomainIntegrator(new DiffusionIntegrator(*dtDCoef));
   }

   Array<int> ess_bdr(pmesh_.bdr_attributes.Max());
   ess_bdr = 0;

   const Array<CoefficientByAttr*> & dbc = bcs_.GetDirichletBCs();
   for (int i=0; i<dbc.Size(); i++)
   {
      for (int j=0; j<dbc[i]->attr.Size(); ++j)
      {
         ess_bdr[dbc[i]->attr[j] - 1] = 1;
      }

      bfbfi_marker_[index_].Append(new Array<int>);
      AttrToMarker(pmesh_.bdr_attributes.Max(), dbc[i]->attr,
                   *bfbfi_marker_[index_].Last());

      bfbfi_[index_].Append(new DGDiffusionIntegrator(DCoef,
                                                      dg_.sigma,
                                                      dg_.kappa));

      bflfi_marker_.Append(new Array<int>);
      AttrToMarker(pmesh_.bdr_attributes.Max(), dbc[i]->attr,
                   *bflfi_marker_.Last());
      bflfi_.Append(new DGDirichletLFIntegrator(*dbc[i]->coef, DCoef,
                                                dg_.sigma,
                                                dg_.kappa));

      blf_[index_]->AddBdrFaceIntegrator(new DGDiffusionIntegrator(*dtDCoef,
                                                                   dg_.sigma,
                                                                   dg_.kappa),
                                         *bfbfi_marker_[index_].Last());
   }

   if (h1_fes_ != NULL)
   {
      h1_fes_->GetEssentialTrueDofs(ess_bdr, cg_ess_tdof_list);
   }

   const Array<CoefficientByAttr*> & nbc = bcs_.GetNeumannBCs();
   for (int i=0; i<nbc.Size(); i++)
   {
      bflfi_marker_.Append(new Array<int>);
      AttrToMarker(pmesh_.bdr_attributes.Max(), nbc[i]->attr,
                   *bflfi_marker_.Last());
      sCoefs_.Append(new ProductCoefficient(DCoef, *nbc[i]->coef));
      bflfi_.Append(new BoundaryLFIntegrator(*sCoefs_.Last()));
   }

   const Array<CoefficientsByAttr*> & rbc = bcs_.GetRobinBCs();
   for (int i=0; i<rbc.Size(); i++)
   {
      bfbfi_marker_[index_].Append(new Array<int>);
      AttrToMarker(pmesh_.bdr_attributes.Max(), rbc[i]->attr,
                   *bfbfi_marker_[index_].Last());
      bfbfi_[index_].Append(new BoundaryMassIntegrator(*rbc[i]->coefs[0]));

      bflfi_marker_.Append(new Array<int>);
      AttrToMarker(pmesh_.bdr_attributes.Max(), rbc[i]->attr,
                   *bflfi_marker_.Last());
      bflfi_.Append(new BoundaryLFIntegrator(*rbc[i]->coefs[1]));

      ProductCoefficient * dtaCoef = new ProductCoefficient(dt_,
                                                            *rbc[i]->coefs[0]);
      dtSCoefs_.Append(dtaCoef);

      blf_[index_]->AddBdrFaceIntegrator(new BoundaryMassIntegrator(*dtaCoef),
                                         *bfbfi_marker_[index_].Last());
   }
}

void DGTransportTDO::TransportOp::SetDiffusionTerm(StateVariableMatCoef &DCoef)
{
   if ( mpi_.Root() && logging_ > 0)
   {
      cout << eqn_name_ << ": Adding anisotropic diffusion term v1" << endl;
   }

   diffusionMatrixCoef_ = &DCoef;

   ScalarMatrixProductCoefficient * dtDCoef =
      new ScalarMatrixProductCoefficient(dt_, DCoef);
   dtMCoefs_.Append(dtDCoef);

   dbfi_[index_].Append(new DiffusionIntegrator(DCoef));
   fbfi_[index_].Append(new DGDiffusionIntegrator(DCoef,
                                                  dg_.sigma,
                                                  dg_.kappa));

   if (blf_[index_] == NULL)
   {
      blf_[index_] = new ParBilinearForm(&fes_);
   }

   blf_[index_]->AddDomainIntegrator(new DiffusionIntegrator(*dtDCoef));
   blf_[index_]->AddInteriorFaceIntegrator(
      new DGDiffusionIntegrator(*dtDCoef,
                                dg_.sigma,
                                dg_.kappa));

   if (h1_fes_ != NULL)
   {
      if (cgblf_[index_] == NULL)
      {
         cgblf_[index_] = new ParBilinearForm(h1_fes_);
      }
      cgblf_[index_]->AddDomainIntegrator(new DiffusionIntegrator(*dtDCoef));
   }

   Array<int> ess_bdr(pmesh_.bdr_attributes.Max());
   ess_bdr = 0;

   const Array<CoefficientByAttr*> & dbc = bcs_.GetDirichletBCs();
   for (int i=0; i<dbc.Size(); i++)
   {
      for (int j=0; j<dbc[i]->attr.Size(); ++j)
      {
         ess_bdr[dbc[i]->attr[j] - 1] = 1;
      }

      bfbfi_marker_[index_].Append(new Array<int>);
      AttrToMarker(pmesh_.bdr_attributes.Max(), dbc[i]->attr,
                   *bfbfi_marker_[index_].Last());
      bfbfi_[index_].Append(new DGDiffusionIntegrator(DCoef,
                                                      dg_.sigma,
                                                      dg_.kappa));

      bflfi_marker_.Append(new Array<int>);
      AttrToMarker(pmesh_.bdr_attributes.Max(), dbc[i]->attr,
                   *bflfi_marker_.Last());
      bflfi_.Append(new DGDirichletLFIntegrator(*dbc[i]->coef, DCoef,
                                                dg_.sigma,
                                                dg_.kappa));

      blf_[index_]->AddBdrFaceIntegrator(new DGDiffusionIntegrator(*dtDCoef,
                                                                   dg_.sigma,
                                                                   dg_.kappa),
                                         *bfbfi_marker_[index_].Last());
   }

   if (h1_fes_ != NULL)
   {
      h1_fes_->GetEssentialTrueDofs(ess_bdr, cg_ess_tdof_list);
   }

   const Array<CoefficientByAttr*> & nbc = bcs_.GetNeumannBCs();
   for (int i=0; i<nbc.Size(); i++)
   {
      bflfi_marker_.Append(new Array<int>);
      AttrToMarker(pmesh_.bdr_attributes.Max(), nbc[i]->attr,
                   *bflfi_marker_.Last());
      bflfi_.Append(new BoundaryLFIntegrator(*nbc[i]->coef));
   }

   const Array<CoefficientsByAttr*> & rbc = bcs_.GetRobinBCs();
   for (int i=0; i<rbc.Size(); i++)
   {
      bfbfi_marker_[index_].Append(new Array<int>);
      AttrToMarker(pmesh_.bdr_attributes.Max(), rbc[i]->attr,
                   *bfbfi_marker_[index_].Last());
      bfbfi_[index_].Append(new BoundaryMassIntegrator(*rbc[i]->coefs[0]));

      bflfi_marker_.Append(new Array<int>);
      AttrToMarker(pmesh_.bdr_attributes.Max(), rbc[i]->attr,
                   *bflfi_marker_.Last());
      bflfi_.Append(new BoundaryLFIntegrator(*rbc[i]->coefs[1]));

      ProductCoefficient * dtaCoef = new ProductCoefficient(dt_,
                                                            *rbc[i]->coefs[0]);
      dtSCoefs_.Append(dtaCoef);

      blf_[index_]->AddBdrFaceIntegrator(new BoundaryMassIntegrator(*dtaCoef),
                                         *bfbfi_marker_[index_].Last());
   }
}
/*
void
DGTransportTDO::TransportOp::SetAnisoDiffusionTerm(StateVariableMatCoef &DCoef,
                                                   Coefficient &skewCoef,
                                                   double D_min,
                                                   double D_max)
{
   if ( mpi_.Root() && logging_ > 0)
   {
      cout << eqn_name_ << ": Adding anisotropic diffusion term" << endl;
   }

   diffusionMatrixCoef_ = &DCoef;

   ScalarMatrixProductCoefficient * dtDCoef =
      new ScalarMatrixProductCoefficient(dt_, DCoef);
   dtMCoefs_.Append(dtDCoef);

   ProductCoefficient * dtSkewCoef =
      new ProductCoefficient(dt_, skewCoef);
   dtSCoefs_.Append(dtSkewCoef);

   dbfi_.Append(new DiffusionIntegrator(DCoef));
   fbfi_.Append(new DGAnisoDiffusionIntegrator(DCoef,
                                               skewCoef,
                                               D_min, D_max,
                                               dg_.sigma,
                                               dg_.kappa));

   if (blf_[index_] == NULL)
   {
      blf_[index_] = new ParBilinearForm(&fes_);
   }

   blf_[index_]->AddDomainIntegrator(new DiffusionIntegrator(*dtDCoef));
   blf_[index_]->AddInteriorFaceIntegrator(
      new DGAnisoDiffusionIntegrator(*dtDCoef,
                                     *dtSkewCoef,
                                     D_min, D_max,
                                     dg_.sigma,
                                     dg_.kappa));

   const Array<CoefficientByAttr*> & dbc = bcs_.GetDirichletBCs();
   for (int i=0; i<dbc.Size(); i++)
   {
      bfbfi_marker_.Append(new Array<int>);
      AttrToMarker(pmesh_.bdr_attributes.Max(), dbc[i]->attr,
                   *bfbfi_marker_.Last());
      bfbfi_.Append(new DGAnisoDiffusionIntegrator(DCoef,
                                                   skewCoef,
                                                   D_min, D_max,
                                                   dg_.sigma,
                                                   dg_.kappa));

      bflfi_marker_.Append(new Array<int>);
      AttrToMarker(pmesh_.bdr_attributes.Max(), dbc[i]->attr,
                   *bflfi_marker_.Last());
      bflfi_.Append(new DGDirichletLFIntegrator(*dbc[i]->coef, DCoef,
                                                dg_.sigma,
                                                dg_.kappa));

      blf_[index_]->AddBdrFaceIntegrator(new DGAnisoDiffusionIntegrator(*dtDCoef,
                                                                        *dtSkewCoef,
                                                                        D_min, D_max,
                                                                        dg_.sigma,
                                                                        dg_.kappa),
                                         *bfbfi_marker_.Last());
   }

   const Array<CoefficientByAttr*> & nbc = bcs_.GetNeumannBCs();
   for (int i=0; i<nbc.Size(); i++)
   {
      bflfi_marker_.Append(new Array<int>);
      AttrToMarker(pmesh_.bdr_attributes.Max(), nbc[i]->attr,
                   *bflfi_marker_.Last());
      bflfi_.Append(new BoundaryLFIntegrator(*nbc[i]->coef));
   }

   const Array<CoefficientsByAttr*> & rbc = bcs_.GetRobinBCs();
   for (int i=0; i<rbc.Size(); i++)
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

      blf_[index_]->AddBdrFaceIntegrator(new BoundaryMassIntegrator(*dtaCoef),
                                         *bfbfi_marker_.Last());
   }
}
*/
void
DGTransportTDO::TransportOp::SetAnisotropicDiffusionTerm(
   StateVariableMatCoef &DCoef,
   Coefficient *DParaCoef,
   Coefficient *DPerpCoef)
{
   if ( mpi_.Root() && logging_ > 0)
   {
      cout << eqn_name_ << ": Adding anisotropic diffusion term v2" << endl;
   }

   diffusionMatrixCoef_ = &DCoef;

   ScalarMatrixProductCoefficient * dtDCoef =
      new ScalarMatrixProductCoefficient(dt_, DCoef);
   dtMCoefs_.Append(dtDCoef);

   ProductCoefficient * dtDParaCoef = NULL;
   if (DParaCoef != NULL)
   {
      dtDParaCoef = new ProductCoefficient(dt_, *DParaCoef);
      dtSCoefs_.Append(dtDParaCoef);
   }

   ProductCoefficient * dtDPerpCoef = NULL;
   if (DPerpCoef != NULL)
   {
      dtDPerpCoef =  new ProductCoefficient(dt_, *DPerpCoef);
      dtSCoefs_.Append(dtDPerpCoef);
   }

   dbfi_[index_].Append(new DiffusionIntegrator(DCoef));
   fbfi_[index_].Append(new DGAnisoDiffIntegrator(DCoef,
                                                  DParaCoef,
                                                  DPerpCoef,
                                                  dg_.sigma,
                                                  dg_.kappa));

   if (blf_[index_] == NULL)
   {
      blf_[index_] = new ParBilinearForm(&fes_);
   }

   blf_[index_]->AddDomainIntegrator(new DiffusionIntegrator(*dtDCoef));
   blf_[index_]->AddInteriorFaceIntegrator(
      new DGAnisoDiffIntegrator(*dtDCoef,
                                dtDParaCoef,
                                dtDPerpCoef,
                                dg_.sigma,
                                dg_.kappa));

   if (h1_fes_ != NULL)
   {
      if (cgblf_[index_] == NULL)
      {
         cgblf_[index_] = new ParBilinearForm(h1_fes_);
      }
      cgblf_[index_]->AddDomainIntegrator(new DiffusionIntegrator(*dtDCoef));
   }

   Array<int> ess_bdr(pmesh_.bdr_attributes.Max());
   ess_bdr = 0;

   const Array<CoefficientByAttr*> & dbc = bcs_.GetDirichletBCs();
   for (int i=0; i<dbc.Size(); i++)
   {
      for (int j=0; j<dbc[i]->attr.Size(); ++j)
      {
         ess_bdr[dbc[i]->attr[j] - 1] = 1;
      }

      bfbfi_marker_[index_].Append(new Array<int>);
      AttrToMarker(pmesh_.bdr_attributes.Max(), dbc[i]->attr,
                   *bfbfi_marker_[index_].Last());
      bfbfi_[index_].Append(new DGAnisoDiffBdrIntegrator(DCoef,
                                                         DParaCoef,
                                                         DPerpCoef,
                                                         dg_.sigma,
                                                         dg_.kappa));

      bflfi_marker_.Append(new Array<int>);
      AttrToMarker(pmesh_.bdr_attributes.Max(), dbc[i]->attr,
                   *bflfi_marker_.Last());
      bflfi_.Append(new DGAnisoDiffDirichletLFIntegrator(*dbc[i]->coef,
                                                         DCoef,
                                                         DParaCoef,
                                                         DPerpCoef,
                                                         dg_.sigma,
                                                         dg_.kappa));

      blf_[index_]->AddBdrFaceIntegrator(
         new DGAnisoDiffBdrIntegrator(*dtDCoef,
                                      dtDParaCoef,
                                      dtDPerpCoef,
                                      dg_.sigma,
                                      dg_.kappa),
         *bfbfi_marker_[index_].Last());
   }

   if (h1_fes_ != NULL)
   {
      h1_fes_->GetEssentialTrueDofs(ess_bdr, cg_ess_tdof_list);
   }

   const Array<CoefficientByAttr*> & nbc = bcs_.GetNeumannBCs();
   for (int i=0; i<nbc.Size(); i++)
   {
      bflfi_marker_.Append(new Array<int>);
      AttrToMarker(pmesh_.bdr_attributes.Max(), nbc[i]->attr,
                   *bflfi_marker_.Last());
      bflfi_.Append(new BoundaryLFIntegrator(*nbc[i]->coef));
   }

   const Array<CoefficientsByAttr*> & rbc = bcs_.GetRobinBCs();
   for (int i=0; i<rbc.Size(); i++)
   {
      bfbfi_marker_[index_].Append(new Array<int>);
      AttrToMarker(pmesh_.bdr_attributes.Max(), rbc[i]->attr,
                   *bfbfi_marker_[index_].Last());
      bfbfi_[index_].Append(new BoundaryMassIntegrator(*rbc[i]->coefs[0]));

      bflfi_marker_.Append(new Array<int>);
      AttrToMarker(pmesh_.bdr_attributes.Max(), rbc[i]->attr,
                   *bflfi_marker_.Last());
      bflfi_.Append(new BoundaryLFIntegrator(*rbc[i]->coefs[1]));

      ProductCoefficient * dtaCoef = new ProductCoefficient(dt_,
                                                            *rbc[i]->coefs[0]);
      dtSCoefs_.Append(dtaCoef);

      blf_[index_]->AddBdrFaceIntegrator(new BoundaryMassIntegrator(*dtaCoef),
                                         *bfbfi_marker_[index_].Last());
   }
}

void
DGTransportTDO::TransportOp::SetDiffusionTermGradient(
   StateVariableMatCoef &DCoef)
{
   if ( mpi_.Root() && logging_ > 0)
   {
      cout << eqn_name_ << ": Adding diffusion term to gradient" << endl;
   }

   VectorCoefficient * dyCoef =
      new GradientGridFunctionCoefficient(yGF_[index_]);
   vCoefs_.Append(dyCoef);

   for (int i=0; i<5; i++)
   {
      if (DCoef.NonTrivialValue((FieldType)i))
      {
         if ( mpi_.Root() && logging_ > 0)
         {
            cout << eqn_name_
                 << ": Adding diffusion term proportional to d "
                 << FieldSymbol((FieldType)i) << " / dt "
                 << "in the gradient" << endl;
         }

         StateVariableMatCoef * coef = DCoef.Clone();
         coef->SetDerivType((FieldType)i);
         svmcoefs_.Append(coef);

         VectorCoefficient * DdyCoef = new MatVecCoefficient(*coef, *dyCoef);
         vCoefs_.Append(DdyCoef);

         ScalarVectorProductCoefficient * negdtDdyCoef =
            new ScalarVectorProductCoefficient(-dt_, *DdyCoef);
         negdtVCoefs_.Append(negdtDdyCoef);

         if (blf_[i] == NULL)
         {
            blf_[i] = new ParBilinearForm(&fes_);
         }
         blf_[i]->AddDomainIntegrator(
            new ConservativeConvectionIntegrator(*negdtDdyCoef, 1.0));
         blf_[i]->AddInteriorFaceIntegrator(
            new DGTraceIntegrator(*negdtDdyCoef, 1.0, -0.5));

         if (h1_fes_ != NULL)
         {
            if (cgblf_[i] == NULL)
            {
               cgblf_[i] = new ParBilinearForm(h1_fes_);
            }
            cgblf_[i]->AddDomainIntegrator(
               new ConservativeConvectionIntegrator(*negdtDdyCoef, 1.0));
         }
      }
   }
}

void DGTransportTDO::TransportOp::SetSourceTerm(StateVariableCoef &SCoef,
                                                double s)
{
   if ( mpi_.Root() && logging_ > 0)
   {
      cout << eqn_name_ << ": Adding source term" << endl;
   }

   Coefficient * coef = &SCoef;
   if (s != 1.0)
   {
      coef = new ProductCoefficient(s, SCoef);
      sCoefs_.Append(coef);
   }

   dlfi_.Append(new DomainLFIntegrator(*coef));

   this->SetSourceTermGradient(SCoef, s);
}

void DGTransportTDO::TransportOp::SetSourceTermGradient(
   StateVariableCoef &SCoef,
   double s)
{
   if ( mpi_.Root() && logging_ > 0)
   {
      cout << eqn_name_ << ": Adding source term to gradient" << endl;
   }

   for (int i=0; i<5; i++)
   {
      if (SCoef.NonTrivialValue((FieldType)i))
      {
         if ( mpi_.Root() && logging_ > 0)
         {
            cout << eqn_name_
                 << ": Adding source term proportional to d "
                 << FieldSymbol((FieldType)i) << " / dt "
                 << "in the gradient" << endl;
         }

         StateVariableCoef * coef = NULL;
         if (s == 1.0)
         {
            coef = SCoef.Clone();
         }
         else
         {
            StateVariableConstantCoef sCoef(s);
            coef = new StateVariableProductCoef(sCoef, SCoef);
         }
         coef->SetDerivType((FieldType)i);
         svscoefs_.Append(coef);
         ProductCoefficient * dtdSCoef = new ProductCoefficient(-dt_, *coef);
         negdtSCoefs_.Append(dtdSCoef);

         if (blf_[i] == NULL)
         {
            blf_[i] = new ParBilinearForm(&fes_);
         }
         blf_[i]->AddDomainIntegrator(new MassIntegrator(*dtdSCoef));

      }
   }
}

void DGTransportTDO::TransportOp::SetBdrSourceTerm(StateVariableCoef &SCoef,
                                                   StateVariableVecCoef &VCoef)
{
   if ( mpi_.Root() && logging_ > 0)
   {
      cout << eqn_name_ << ": Adding boundary source term" << endl;
   }

   bflfi_.Append(new BoundaryFlowIntegrator(SCoef, VCoef, 1.0, -0.5));
   bflfi_marker_.Append(NULL);

   for (int i=0; i<5; i++)
   {
      if (SCoef.NonTrivialValue((FieldType)i))
      {
         if ( mpi_.Root() && logging_ > 0)
         {
            cout << eqn_name_
                 << ": Adding boundary source term proportional to d "
                 << FieldSymbol((FieldType)i) << " / dt "
                 << "in the gradient" << endl;
         }

         StateVariableCoef * coef = SCoef.Clone();
         coef->SetDerivType((FieldType)i);
         ProductCoefficient * dtdSCoef =
            new ProductCoefficient(dt_, *coef);
         dtSCoefs_.Append(dtdSCoef);

         if (blf_[i] == NULL)
         {
            blf_[i] = new ParBilinearForm(&fes_);
         }
         // blf_[i]->AddBdrFaceIntegrator(new DGTraceIntegrator(*dtdSCoef, VCoef,
         //                                                            1.0, 0.5));

      }
      if (VCoef.NonTrivialValue((FieldType)i))
      {
         if ( mpi_.Root() && logging_ > 0)
         {
            cout << eqn_name_
                 << ": Adding boundary source term proportional to d "
                 << FieldSymbol((FieldType)i) << " / dt "
                 << "in the gradient" << endl;
         }

         StateVariableVecCoef * coef = VCoef.Clone();
         coef->SetDerivType((FieldType)i);
         ScalarVectorProductCoefficient * dtdVCoef =
            new ScalarVectorProductCoefficient(dt_, *coef);
         dtVCoefs_.Append(dtdVCoef);

         if (blf_[i] == NULL)
         {
            blf_[i] = new ParBilinearForm(&fes_);
         }
         // blf_[i]->AddBdrFaceIntegrator(new DGTraceIntegrator(SCoef, *dtdVCoef,
         //                                                            1.0, 0.5));

      }
   }
}

void
DGTransportTDO::TransportOp::SetOutflowBdrTerm(
   StateVariableVecCoef &VCoef,
   const Array<CoefficientByAttr*> & obc)
{
   if ( mpi_.Root() && logging_ > 0)
   {
      cout << eqn_name_ << ": Adding outflow boundary term" << endl;
   }

   for (int i=0; i<obc.Size(); i++)
   {
      ScalarVectorProductCoefficient * rVCoef =
         new ScalarVectorProductCoefficient(*obc[i]->coef, VCoef);
      vCoefs_.Append(rVCoef);

      ScalarVectorProductCoefficient * dtrVCoef =
         new ScalarVectorProductCoefficient(dt_, *rVCoef);
      dtVCoefs_.Append(dtrVCoef);

      bfbfi_[index_].Append(new DGTraceIntegrator(*rVCoef, 1.0, 0.5));
      bfbfi_marker_[index_].Append(new Array<int>);
      AttrToMarker(pmesh_.bdr_attributes.Max(), obc[i]->attr,
                   *bfbfi_marker_[index_].Last());

      if (blf_[index_] == NULL)
      {
         blf_[index_] = new ParBilinearForm(&fes_);
      }
      blf_[index_]->AddBdrFaceIntegrator(new DGTraceIntegrator(*dtrVCoef,
                                                               1.0, 0.5),
                                         *bfbfi_marker_[index_].Last());
   }
}

void
DGTransportTDO::TransportOp::SetRecyclingBdrSourceTerm(const RecyclingBC & rbc)
{
   if ( mpi_.Root() && logging_ > 0)
   {
      cout << eqn_name_ << ": Adding recycling boundary source term" << endl;
   }

   int ion_index = rbc.GetIonDensityIndex();
   int vel_index = rbc.GetIonVelocityIndex();

   const Array<CoefficientsByAttr*> & bcs = rbc.GetRecyclingBCs();

   StateVariableCoef & SCoef(*ykCoefPtrs_[ion_index]);
   StateVariableVecCoef * VCoefPtr =
      new IonAdvectionCoef(*ykCoefPtrs_[vel_index], B3Coef_);
   svvcoefs_.Append(VCoefPtr);

   for (int j = 0; j< bcs.Size(); j++)
   {
      bflfi_marker_.Append(new Array<int>);
      AttrToMarker(pmesh_.bdr_attributes.Max(), bcs[j]->attr,
                   *bflfi_marker_.Last());

      Coefficient * rCoef = bcs[j]->coefs[1];

      StateVariableCoef * rSVCoef = dynamic_cast<StateVariableCoef*>(rCoef);
      if (rSVCoef == NULL)
      {
         rSVCoef = new StateVariableStandardCoef(*rCoef);
      }

      StateVariableCoef * RSCoef =
         new StateVariableProductCoef(*rSVCoef, SCoef);
      svscoefs_.Append(RSCoef);

      if (rSVCoef != rCoef)
      {
         delete rSVCoef;
      }

      bflfi_.Append(new BoundaryFlowIntegrator(*RSCoef, *VCoefPtr, 1.0, -0.5));

      for (int i=0; i<5; i++)
      {
         if (RSCoef->NonTrivialValue((FieldType)i))
         {
            if ( mpi_.Root() && logging_ > 0)
            {
               cout << eqn_name_
                    << ": Adding boundary source term proportional to d "
                    << FieldSymbol((FieldType)i) << " / dt "
                    << "in the gradient" << endl;
            }

            StateVariableCoef * coef = RSCoef->Clone();
            coef->SetDerivType((FieldType)i);
            ProductCoefficient * dtdSCoef =
               new ProductCoefficient(dt_, *coef);
            dtSCoefs_.Append(dtdSCoef);

            if (blf_[i] == NULL)
            {
               blf_[i] = new ParBilinearForm(&fes_);
            }
            // blf_[i]->AddBdrFaceIntegrator(new DGTraceIntegrator(*dtdSCoef, VCoef,
            //                                                            1.0, 0.5));

         }
         if (VCoefPtr->NonTrivialValue((FieldType)i))
         {
            if ( mpi_.Root() && logging_ > 0)
            {
               cout << eqn_name_
                    << ": Adding boundary source term proportional to d "
                    << FieldSymbol((FieldType)i) << " / dt "
                    << "in the gradient" << endl;
            }

            StateVariableVecCoef * coef = VCoefPtr->Clone();
            coef->SetDerivType((FieldType)i);
            ScalarVectorProductCoefficient * dtdVCoef =
               new ScalarVectorProductCoefficient(dt_, *coef);
            dtVCoefs_.Append(dtdVCoef);

            if (blf_[i] == NULL)
            {
               blf_[i] = new ParBilinearForm(&fes_);
            }
            // blf_[i]->AddBdrFaceIntegrator(new DGTraceIntegrator(SCoef, *dtdVCoef,
            //                                                            1.0, 0.5));

         }
      }
   }
}

void
DGTransportTDO::TransportOp::InitializeGLVis()
{
   /*
    if ((int)sout_.size() < coefs_.Size())
    {
       sout_.resize(coefs_.Size());
       for (int i=0; i<coefs_.Size(); i++)
       {
          sout_[i] = new socketstream;
       }
    }
   */
}

void
DGTransportTDO::TransportOp::DisplayToGLVis()
{
   /*
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
   */
}

DGTransportTDO::AdvTransportOp::AdvTransportOp(const MPI_Session & mpi,
                                               const DGParams & dg,
                                               const PlasmaParams & plasma,
                                               int index,
                                               const std::string &eqn_name,
                                               const std::string &field_name,
                                               ParFiniteElementSpace * vfes,
                                               ParFiniteElementSpace * h1_fes,
                                               ParGridFunctionArray & yGF,
                                               ParGridFunctionArray & kGF,
                                               ParGridFunction & elOrdGF,
                                               ParGridFunction & hGF,
                                               const AdvectionDiffusionBC & bcs,
                                               const CoupledBCs & cbcs,
                                               const CommonCoefs & common_coefs,
                                               VectorCoefficient & B3Coef,
                                               int term_flag, int vis_flag,
                                               int logging,
                                               const std::string & log_prefix)
   : TransportOp(mpi, dg, plasma, index, eqn_name, field_name, vfes, h1_fes,
                 yGF, kGF, elOrdGF, hGF, bcs, cbcs, common_coefs, B3Coef,
                 term_flag, vis_flag, logging, log_prefix),
     advectionCoef_(NULL),
     l2_fes_0_(new L2_ParFESpace(&pmesh_, 0, pmesh_.SpaceDimension())),
     h1_fes_1_(new H1_ParFESpace(&pmesh_, 1, pmesh_.SpaceDimension())),
     elOrdDiscGF_(new ParGridFunction(l2_fes_0_)),
     elOrdContGF_(new ParGridFunction(h1_fes_1_)),
     OscDiscGF_(new ParGridFunction(l2_fes_0_)),
     OscContGF_(new ParGridFunction(h1_fes_1_)),
     hDiscGF_(new ParGridFunction(l2_fes_0_)),
     hContGF_(new ParGridFunction(h1_fes_1_)),
     elOrdDiscCoef_(elOrdDiscGF_),
     elOrdContCoef_(elOrdContGF_),
     OscDiscCoef_(OscDiscGF_),
     OscContCoef_(OscContGF_),
     hDiscCoef_(hDiscGF_),
     hContCoef_(hContGF_),
     CsCoef_(m_i_kg_, TiCoef_, TeCoef_)
{
}

DGTransportTDO::AdvTransportOp::~AdvTransportOp()
{
   delete elOrdDiscGF_;
   delete elOrdContGF_;
   delete OscDiscGF_;
   delete OscContGF_;
   delete hDiscGF_;
   delete hContGF_;
   delete l2_fes_0_;
   delete h1_fes_1_;
}

void DGTransportTDO::AdvTransportOp::SetTime(double t)
{
   if (mpi_.Root() && logging_ > 1)
   {
      cout << "Setting time: " << t << " in AdvTransportOp (" << eqn_name_
           << ")"
           << endl;
   }

   TransportOp::SetTime(t);

   CsCoef_.SetTime(t);

   if (advectionCoef_) { advectionCoef_->SetTime(t); }
}

void DGTransportTDO::AdvTransportOp::Update()
{
   if (mpi_.Root() && logging_ > 1)
   {
      cout << "Entering DGTransportTDO::AdvTransportOp::Update" << endl;
   }

   TransportOp::Update();

   if (l2_fes_0_  != NULL) { l2_fes_0_->Update(); }
   if (h1_fes_1_  != NULL) { h1_fes_1_->Update(); }

   if (elOrdDiscGF_ != NULL) { elOrdDiscGF_->Update(); }
   if (elOrdContGF_ != NULL) { elOrdContGF_->Update(); }
   if (OscDiscGF_   != NULL) { OscDiscGF_->Update(); }
   if (OscContGF_   != NULL) { OscContGF_->Update(); }
   if (hDiscGF_     != NULL) { hDiscGF_->Update(); }
   if (hContGF_     != NULL) { hContGF_->Update(); }
}

void
DGTransportTDO::AdvTransportOp::SetAdvectionDiffusionTerm(
   StateVariableMatCoef &DCoef,
   StateVariableVecCoef &VCoef,
   Coefficient *DParaCoef,
   Coefficient *DPerpCoef)
{
   if ( mpi_.Root() && logging_ > 0)
   {
      cout << eqn_name_ << ": Adding advection-diffusion term" << endl;
   }

   diffusionMatrixCoef_ = &DCoef;
   advectionCoef_ = &VCoef;

   ScalarMatrixProductCoefficient * dtDCoef =
      new ScalarMatrixProductCoefficient(dt_, DCoef);
   dtMCoefs_.Append(dtDCoef);

   ScalarVectorProductCoefficient * dtVCoef =
      new ScalarVectorProductCoefficient(dt_, VCoef);
   dtVCoefs_.Append(dtVCoef);

   ProductCoefficient * dtDParaCoef = NULL;
   if (DParaCoef != NULL)
   {
      dtDParaCoef = new ProductCoefficient(dt_, *DParaCoef);
      dtSCoefs_.Append(dtDParaCoef);
   }

   ProductCoefficient * dtDPerpCoef = NULL;
   if (DPerpCoef != NULL)
   {
      dtDPerpCoef = new ProductCoefficient(dt_, *DPerpCoef);
      dtSCoefs_.Append(dtDPerpCoef);
   }

   double lambda = 1.0;
   double kappa2 = 0.0;

   dbfi_[index_].Append(new DiffusionIntegrator(DCoef));
   dbfi_[index_].Append(new ConservativeConvectionIntegrator(VCoef));
   fbfi_[index_].Append(new DGAdvDiffIntegrator(DCoef,
                                                VCoef,
                                                DParaCoef,
                                                DPerpCoef,
                                                lambda,
                                                dg_.sigma,
                                                dg_.kappa,
                                                kappa2));
   // bfbfi_.Append(new DGTraceIntegrator(VCoef, 1.0, 0.5));
   // bfbfi_marker_.Append(NULL);

   if (blf_[index_] == NULL)
   {
      blf_[index_] = new ParBilinearForm(&fes_);
   }

   blf_[index_]->AddDomainIntegrator(new DiffusionIntegrator(*dtDCoef));
   blf_[index_]->AddDomainIntegrator(new ConservativeConvectionIntegrator(
                                        *dtVCoef));

   blf_[index_]->AddInteriorFaceIntegrator(
      new DGAdvDiffIntegrator(*dtDCoef,
                              *dtVCoef,
                              dtDParaCoef,
                              dtDPerpCoef,
                              lambda,
                              dg_.sigma,
                              dg_.kappa,
                              kappa2));

   // blf_[index_]->AddBdrFaceIntegrator(new DGTraceIntegrator(*dtVCoef, 1.0, 0.5));

   if (h1_fes_ != NULL)
   {
      if (cgblf_[index_] == NULL)
      {
         cgblf_[index_] = new ParBilinearForm(h1_fes_);
      }
      cgblf_[index_]->AddDomainIntegrator(new DiffusionIntegrator(*dtDCoef));
      cgblf_[index_]->AddDomainIntegrator(new ConservativeConvectionIntegrator(
                                             *dtVCoef));
   }

   Array<int> ess_bdr;
   if (pmesh_.bdr_attributes.Size() > 0)
   {
      ess_bdr.SetSize(pmesh_.bdr_attributes.Max());
      ess_bdr = 0;
   }

   const Array<CoefficientByAttr*> & dbc = bcs_.GetDirichletBCs();
   for (int i=0; i<dbc.Size(); i++)
   {
      for (int j=0; j<dbc[i]->attr.Size(); ++j)
      {
         ess_bdr[dbc[i]->attr[j] - 1] = 1;
      }

      bfbfi_marker_[index_].Append(new Array<int>);
      AttrToMarker(pmesh_.bdr_attributes.Max(), dbc[i]->attr,
                   *bfbfi_marker_[index_].Last());
      bfbfi_[index_].Append(new DGAdvDiffBdrIntegrator(DCoef,
                                                       VCoef,
                                                       DParaCoef,
                                                       DPerpCoef,
                                                       lambda,
                                                       dg_.sigma,
                                                       dg_.kappa,
                                                       kappa2));

      bflfi_marker_.Append(new Array<int>);
      AttrToMarker(pmesh_.bdr_attributes.Max(), dbc[i]->attr,
                   *bflfi_marker_.Last());
      bflfi_.Append(new DGAdvDiffDirichletLFIntegrator(*dbc[i]->coef,
                                                       DCoef,
                                                       VCoef,
                                                       DParaCoef,
                                                       DPerpCoef,
                                                       lambda,
                                                       dg_.sigma,
                                                       dg_.kappa,
                                                       kappa2));

      blf_[index_]->AddBdrFaceIntegrator(
         new DGAdvDiffBdrIntegrator(*dtDCoef,
                                    *dtVCoef,
                                    dtDParaCoef,
                                    dtDPerpCoef,
                                    lambda,
                                    dg_.sigma,
                                    dg_.kappa,
                                    kappa2),
         *bfbfi_marker_[index_].Last());
   }

   if (h1_fes_ != NULL)
   {
      h1_fes_->GetEssentialTrueDofs(ess_bdr, cg_ess_tdof_list);
   }

   const Array<CoefficientByAttr*> & nbc = bcs_.GetNeumannBCs();
   for (int i=0; i<nbc.Size(); i++)
   {
      bflfi_marker_.Append(new Array<int>);
      AttrToMarker(pmesh_.bdr_attributes.Max(), nbc[i]->attr,
                   *bflfi_marker_.Last());
      bflfi_.Append(new BoundaryLFIntegrator(*nbc[i]->coef));
   }

   const Array<CoefficientsByAttr*> & rbc = bcs_.GetRobinBCs();
   for (int i=0; i<rbc.Size(); i++)
   {
      bfbfi_marker_[index_].Append(new Array<int>);
      AttrToMarker(pmesh_.bdr_attributes.Max(), rbc[i]->attr,
                   *bfbfi_marker_[index_].Last());
      bfbfi_[index_].Append(new BoundaryMassIntegrator(*rbc[i]->coefs[0]));

      bflfi_marker_.Append(new Array<int>);
      AttrToMarker(pmesh_.bdr_attributes.Max(), rbc[i]->attr,
                   *bflfi_marker_.Last());
      bflfi_.Append(new BoundaryLFIntegrator(*rbc[i]->coefs[1]));

      ProductCoefficient * dtaCoef = new ProductCoefficient(dt_,
                                                            *rbc[i]->coefs[0]);
      dtSCoefs_.Append(dtaCoef);

      blf_[index_]->AddBdrFaceIntegrator(new BoundaryMassIntegrator(*dtaCoef),
                                         *bfbfi_marker_[index_].Last());
   }

   this->SetDiffusionTermGradient(DCoef);
}

void DGTransportTDO::AdvTransportOp::SetAdvectionTerm(StateVariableVecCoef
                                                      &VCoef/*,
                              bool bc*/)
{
   if ( mpi_.Root() && logging_ > 0)
   {
      cout << eqn_name_ << ": Adding advection term" << endl;
   }

   advectionCoef_ = &VCoef;

   ScalarVectorProductCoefficient * dtVCoef =
      new ScalarVectorProductCoefficient(dt_, VCoef);
   dtVCoefs_.Append(dtVCoef);

   // dbfi_.Append(new MixedScalarWeakDivergenceIntegrator(VCoef));
   dbfi_[index_].Append(new ConservativeConvectionIntegrator(VCoef, 1.0));
   fbfi_[index_].Append(new DGTraceIntegrator(VCoef, 1.0, -0.5));
   /*
   if (bc)
   {
      bfbfi_.Append(new DGTraceIntegrator(VCoef, 1.0, -0.5));
      bfbfi_marker_.Append(NULL);
   }
   */
   if (blf_[index_] == NULL)
   {
      blf_[index_] = new ParBilinearForm(&fes_);
   }

   blf_[index_]->AddDomainIntegrator(
      // new MixedScalarWeakDivergenceIntegrator(*dtVCoef));
      new ConservativeConvectionIntegrator(*dtVCoef, 1.0));
   blf_[index_]->AddInteriorFaceIntegrator(new DGTraceIntegrator(*dtVCoef,
                                                                 1.0, -0.5));
   if (h1_fes_ != NULL)
   {
      if (cgblf_[index_] == NULL)
      {
         cgblf_[index_] = new ParBilinearForm(h1_fes_);
      }
      cgblf_[index_]->AddDomainIntegrator(
         new ConservativeConvectionIntegrator(*dtVCoef, 1.0));
   }
   /*
   if (bc)
   {
     blf_[index_]->AddBdrFaceIntegrator(new DGTraceIntegrator(*dtVCoef,
                                                              1.0, -0.5));
   }
   */
}

DGTransportTDO::CombinedOp::CombinedOp(const MPI_Session & mpi,
                                       const DGParams & dg,
                                       const std::vector<ArtViscParams> & av,
                                       const PlasmaParams & plasma,
                                       const Vector &eqn_weights,
                                       ParFiniteElementSpace & vfes,
                                       ParFiniteElementSpace & h1_fes,
                                       ParGridFunctionArray & yGF,
                                       ParGridFunctionArray & kGF,
                                       const TransportBCs & bcs,
                                       const TransportCoefs & coefs,
                                       Array<int> & offsets,
                                       double DiPerp,
                                       double XiPerp,
                                       double XePerp,
                                       const Array<int> & term_flags,
                                       const Array<int> & vis_flags,
                                       unsigned int op_flag, int logging)
   : mpi_(mpi),
     neq_(5),
     logging_(logging),
     fes_(*yGF[0]->ParFESpace()),
     // yGF_(yGF),
     kGF_(kGF),
     op_(neq_ + 1),
     wgts_(eqn_weights),
     offsets_(offsets),
     grad_(NULL),
     l2_fes_0_(fes_.GetParMesh(), 0, fes_.GetParMesh()->SpaceDimension()),
     h1_fes_1_(fes_.GetParMesh(), 1, fes_.GetParMesh()->SpaceDimension()),
     B3Coef_(const_cast<StateVariableVecCoef&>(*coefs[5].GetVectorCoefficient(
                                                  CommonCoefs::MAGNETIC_FIELD_COEF))),
     elOrdDiscGF_(&l2_fes_0_),
     elOrdDiscCoef_(&elOrdDiscGF_),
     elOrdContGF_(&h1_fes_1_),
     hDiscGF_(&l2_fes_0_),
     hDiscCoef_(&hDiscGF_),
     hContGF_(&h1_fes_1_)
{
   if ( mpi_.Root() && logging_ > 1)
   {
      cout << "Constructing CombinedOp" << endl;
   }

   op_ = NULL;

   // VectorCoefficient *B3Coef = const_cast<StateVariableVecCoef*>
   //                          (coefs[5].GetVectorCoefficient(CommonCoefs::MAGNETIC_FIELD_COEF));

   ElementOrder(fes_, elOrdDiscGF_);
   elOrdDiscGF_.ExchangeFaceNbrData();
   elOrdContGF_.ProjectDiscCoefficient(elOrdDiscCoef_, GridFunction::MINIMUM);
   elOrdContGF_.ExchangeFaceNbrData();

   ParallelMeshSpacing(fes_, B3Coef_, hDiscGF_);
   hDiscGF_.ExchangeFaceNbrData();
   hContGF_.ProjectDiscCoefficient(hDiscCoef_, GridFunction::MAXIMUM);
   hContGF_.ExchangeFaceNbrData();

   if ((op_flag >> 0) & 1)
   {
      op_[0] = new NeutralDensityOp(mpi, dg, plasma, h1_fes, yGF, kGF,
                                    elOrdContGF_, hContGF_,
                                    bcs[0], bcs.GetCoupledBCs(),
                                    coefs.GetNeutralDensityCoefs(),
                                    coefs.GetCommonCoefs(), B3Coef_,
                                    term_flags[0], vis_flags[0],
                                    logging, "n_n: ");
   }
   else
   {
      op_[0] = new DummyOp(mpi, dg, plasma, yGF, kGF,
                           elOrdContGF_, hContGF_,
                           bcs[0], bcs.GetCoupledBCs(),
                           coefs.GetCommonCoefs(), B3Coef_, 0,
                           "Neutral Density", "Neutral Density",
                           term_flags[0], vis_flags[0],
                           logging, "n_n (dummy): ");
   }

   if ((op_flag >> 1) & 1)
   {
      op_[1] = new IonDensityOp(mpi, dg, av[1], plasma, vfes, h1_fes, yGF, kGF,
                                elOrdContGF_, hContGF_,
                                bcs[1], bcs.GetCoupledBCs(),
                                coefs.GetIonDensityCoefs(),
                                coefs.GetCommonCoefs(), B3Coef_, DiPerp,
                                term_flags[1], vis_flags[1],
                                logging, "n_i: ");
   }
   else
   {
      op_[1] = new DummyOp(mpi, dg, plasma, yGF, kGF,
                           elOrdContGF_, hContGF_,
                           bcs[1], bcs.GetCoupledBCs(),
                           coefs.GetCommonCoefs(), B3Coef_, 1,
                           "Ion Density", "Ion Density",
                           term_flags[1], vis_flags[1],
                           logging, "n_i (dummy): ");
   }

   if ((op_flag >> 2) & 1)
   {
      op_[2] = new IonMomentumOp(mpi, dg, av[2], plasma, vfes, h1_fes, yGF, kGF,
                                 elOrdContGF_, hContGF_,
                                 bcs[2], bcs.GetCoupledBCs(),
                                 coefs.GetIonMomentumCoefs(),
                                 coefs.GetCommonCoefs(), B3Coef_, DiPerp,
                                 term_flags[2], vis_flags[2],
                                 logging, "v_i: ");
   }
   else
   {
      op_[2] = new DummyOp(mpi, dg, plasma, yGF, kGF,
                           elOrdContGF_, hContGF_,
                           bcs[2], bcs.GetCoupledBCs(),
                           coefs.GetCommonCoefs(), B3Coef_, 2,
                           "Ion Parallel Momentum", "Ion Parallel Velocity",
                           term_flags[2], vis_flags[2],
                           logging, "v_i (dummy): ");
   }
   /*
   if ((op_flag >> 3) & 1)
   {
      op_[3] = new IonStaticPressureOp(mpi, dg, plasma, yGF, kGF,
                                       bcs[3], bcs.GetCoupledBCs(),
                                       coefs.GetIonStaticPressureCoefs(),
                                       coefs.GetCommonCoefs(), *B3Coef, XiPerp,
                                       term_flags[3], vis_flags[3],
                                       logging, "T_i: ");
   }
   else
   {
      op_[3] = new DummyOp(mpi, dg, plasma, yGF, kGF,
                           bcs[3], bcs.GetCoupledBCs(),
                           coefs.GetCommonCoefs(), *B3Coef, 3,
                           "Ion Static Pressure", "Ion Temperature",
                           term_flags[3], vis_flags[3],
                           logging, "T_i (dummy): ");
   }

   if ((op_flag >> 4) & 1)
   {
      op_[4] = new ElectronStaticPressureOp(mpi, dg, plasma, yGF, kGF,
                                            bcs[4], bcs.GetCoupledBCs(),
                                            coefs.GetElectronStaticPressureCoefs(),
                                            coefs.GetCommonCoefs(), *B3Coef, XePerp,
                                            term_flags[4], vis_flags[4],
                                            logging, "T_e: ");
   }
   else
   {
      op_[4] = new DummyOp(mpi, dg, plasma, yGF, kGF,
                           bcs[4], bcs.GetCoupledBCs(),
                           coefs.GetCommonCoefs(), *B3Coef, 4,
                           "Electron Static Pressure", "Electron Temperature",
                           term_flags[4], vis_flags[4],
                           logging, "T_e (dummy): ");
   }
   */
   if ((op_flag >> 3) & 1)
   {
      op_[3] = new IonTotalEnergyOp(mpi, dg, plasma, vfes, h1_fes, yGF, kGF,
                                    elOrdContGF_, hContGF_,
                                    bcs[3], bcs.GetCoupledBCs(),
                                    coefs.GetIonTotalEnergyCoefs(),
                                    coefs.GetCommonCoefs(), B3Coef_, XiPerp,
                                    term_flags[3], vis_flags[3],
                                    logging, "T_i: ");
   }
   else
   {
      op_[3] = new DummyOp(mpi, dg, plasma, yGF, kGF,
                           elOrdContGF_, hContGF_,
                           bcs[3], bcs.GetCoupledBCs(),
                           coefs.GetCommonCoefs(), B3Coef_, 3,
                           "Total Ion Energy", "Ion Temperature",
                           term_flags[3], vis_flags[3],
                           logging, "T_i (dummy): ");
   }

   if ((op_flag >> 4) & 1)
   {
      op_[4] = new ElectronTotalEnergyOp(mpi, dg, plasma, vfes, h1_fes,
                                         yGF, kGF,
                                         elOrdContGF_, hContGF_,
                                         bcs[4], bcs.GetCoupledBCs(),
                                         coefs.GetElectronTotalEnergyCoefs(),
                                         coefs.GetCommonCoefs(), B3Coef_, XePerp,
                                         term_flags[4], vis_flags[4],
                                         logging, "T_e: ");
   }
   else
   {
      op_[4] = new DummyOp(mpi, dg, plasma, yGF, kGF,
                           elOrdContGF_, hContGF_,
                           bcs[4], bcs.GetCoupledBCs(),
                           coefs.GetCommonCoefs(), B3Coef_, 4,
                           "Total Electron Energy", "Electron Temperature",
                           term_flags[4], vis_flags[4],
                           logging, "T_e (dummy): ");
   }

   op_[5] = new VisualizationOp(mpi, dg, plasma, vfes, yGF, kGF,
                                elOrdContGF_, hContGF_,
                                bcs[0], bcs.GetCoupledBCs(),
                                coefs.GetCommonCoefs(), B3Coef_,
                                vis_flags[5],
                                logging, "Common: ");

   this->updateOffsets();

   if (mpi_.Root() && logging_ > 1)
   {
      cout << "Done constructing CombinedOp" << endl;
   }
}

DGTransportTDO::CombinedOp::~CombinedOp()
{
   delete grad_;

   for (int i=0; i<op_.Size(); i++) { delete op_[i]; }
   op_.SetSize(0);
}

void DGTransportTDO::CombinedOp::updateOffsets()
{
   if (mpi_.Root() && logging_ > 1)
   {
      cout << "Entering DGTransportTDO::CombinedOp::updateOffsets" << endl;
   }

   offsets_[0] = 0;

   for (int i=0; i<neq_; i++)
   {
      offsets_[i+1] = op_[i]->Height();
   }

   offsets_.PartialSum();

   height = width = offsets_[neq_];

   if (mpi_.Root() && logging_ > 1)
   {
      cout << "Leaving DGTransportTDO::CombinedOp::updateOffsets" << endl;
   }
}

void DGTransportTDO::CombinedOp::SetTime(double t)
{
   if ( mpi_.Root() && logging_ > 0)
   {
      cout << "Setting time: " << t << " in CombinedOp" << endl;
   }
   for (int i=0; i<op_.Size(); i++)
   {
      op_[i]->SetTime(t);
   }
}

void DGTransportTDO::CombinedOp::SetTimeStep(double dt)
{
   if ( mpi_.Root() && logging_ > 0)
   {
      cout << "Setting time step: " << dt << " in CombinedOp" << endl;
   }
   for (int i=0; i<op_.Size(); i++)
   {
      op_[i]->SetTimeStep(dt);
   }
}

void DGTransportTDO::CombinedOp::SetLogging(int logging)
{
   logging_ = logging;

   op_[0]->SetLogging(logging, "n_n: ");
   op_[1]->SetLogging(logging, "n_i: ");
   op_[2]->SetLogging(logging, "v_i: ");
   op_[3]->SetLogging(logging, "T_i: ");
   op_[4]->SetLogging(logging, "T_e: ");
   op_[5]->SetLogging(logging, "Common: ");
}

void
DGTransportTDO::CombinedOp::RegisterDataFields(DataCollection & dc)
{
   for (int i=0; i<op_.Size(); i++)
   {
      op_[i]->RegisterDataFields(dc);
   }
}

void
DGTransportTDO::CombinedOp::PrepareDataFields()
{
   double *prev_k = kGF_[0]->GetData();

   Vector k(offsets_[neq_]); k = 0.0;

   for (int i=0; i<kGF_.Size(); i++)
   {
      kGF_[i]->MakeRef(&fes_, k.GetData() + offsets_[i]);
   }
   kGF_.ExchangeFaceNbrData();


   for (int i=0; i<op_.Size(); i++)
   {
      op_[i]->PrepareDataFields();
   }

   for (int i=0; i<offsets_.Size() - 1; i++)
   {
      kGF_[i]->MakeRef(&fes_, prev_k + offsets_[i]);
   }
   if (prev_k != NULL)
   {
      kGF_.ExchangeFaceNbrData();
   }
}

void
DGTransportTDO::CombinedOp::InitializeGLVis()
{
   for (int i=0; i<op_.Size(); i++)
   {
      op_[i]->InitializeGLVis();
   }
}

void
DGTransportTDO::CombinedOp::DisplayToGLVis()
{
   for (int i=0; i<op_.Size(); i++)
   {
      op_[i]->DisplayToGLVis();
   }
}

void DGTransportTDO::CombinedOp::Update()
{
   if (mpi_.Root() && logging_ > 1)
   {
      cout << "Entering DGTransportTDO::CombinedOp::Update" << endl;
   }

   l2_fes_0_.Update();
   h1_fes_1_.Update();

   elOrdDiscGF_.Update();
   elOrdContGF_.Update();
   hDiscGF_.Update();
   hContGF_.Update();

   for (int i=0; i<op_.Size(); i++)
   {
      op_[i]->Update();
   }

   this->updateOffsets();

   if (mpi_.Root() && logging_ > 1)
   {
      cout << "Leaving DGTransportTDO::CombinedOp::Update" << endl;
   }
}

void DGTransportTDO::CombinedOp::UpdateGradient(const Vector &k) const
{
   if ( mpi_.Root() && logging_ > 1)
   {
      cout << "DGTransportTDO::CombinedOp::UpdateGradient" << endl;
   }

   delete grad_;

   double *prev_k = kGF_[0]->GetData();

   for (int i=0; i<kGF_.Size(); i++)
   {
      kGF_[i]->MakeRef(&fes_, k.GetData() + offsets_[i]);
   }
   kGF_.ExchangeFaceNbrData();

   ElementOrder(fes_, elOrdDiscGF_);
   elOrdDiscGF_.ExchangeFaceNbrData();
   elOrdContGF_.ProjectDiscCoefficient(elOrdDiscCoef_, GridFunction::MINIMUM);
   elOrdContGF_.ExchangeFaceNbrData();

   ParallelMeshSpacing(fes_, B3Coef_, hDiscGF_);
   hDiscGF_.ExchangeFaceNbrData();
   hContGF_.ProjectDiscCoefficient(hDiscCoef_, GridFunction::MAXIMUM);
   hContGF_.ExchangeFaceNbrData();

   grad_ = new BlockOperator(offsets_);
   grad_->owns_blocks = true;

   for (int i=0; i<neq_; i++)
   {
      op_[i]->PrepareGradient();

      for (int j=0; j<neq_; j++)
      {
         Operator * gradIJ = op_[i]->GetGradientBlock(j);
         if (gradIJ)
         {
            if ( mpi_.Root() && logging_ > 2)
            {
               cout << "Grad has block " << i << ", " << j << endl;
            }
            grad_->SetBlock(i, j, gradIJ, wgts_[i]);
         }
      }
   }

   for (int i=0; i<offsets_.Size() - 1; i++)
   {
      kGF_[i]->MakeRef(&fes_, prev_k + offsets_[i]);
   }
   if (prev_k != NULL)
   {
      kGF_.ExchangeFaceNbrData();
   }

   if ( mpi_.Root() && logging_ > 1)
   {
      cout << "DGTransportTDO::CombinedOp::UpdateGradient done" << endl;
   }
}

void DGTransportTDO::CombinedOp::Mult(const Vector &k, Vector &r) const
{
   if ( mpi_.Root() && logging_ > 1)
   {
      cout << "DGTransportTDO::CombinedOp::Mult" << endl;
   }

   double *prev_k = kGF_[0]->GetData();

   for (int i=0; i<kGF_.Size(); i++)
   {
      kGF_[i]->MakeRef(&fes_, k.GetData() + offsets_[i]);
   }
   kGF_.ExchangeFaceNbrData();

   for (int i=0; i<neq_; i++)
   {
      int size = offsets_[i+1] - offsets_[i];

      Vector r_i(&r[offsets_[i]], size);

      op_[i]->Mult(k, r_i);

      r_i *= wgts_[i];

      double norm_r = sqrt(InnerProduct(MPI_COMM_WORLD, r_i, r_i));
      if (mpi_.Root())
      {
         cout << "norm(r_" << i << ") " << norm_r << endl;
      }
   }

   for (int i=0; i<offsets_.Size() - 1; i++)
   {
      kGF_[i]->MakeRef(&fes_, prev_k + offsets_[i]);
   }
   if (prev_k != NULL)
   {
      kGF_.ExchangeFaceNbrData();
   }

   if ( mpi_.Root() && logging_ > 1)
   {
      cout << "DGTransportTDO::CombinedOp::Mult done" << endl;
   }
}

DGTransportTDO::NeutralDensityOp::NeutralDensityOp(const MPI_Session & mpi,
                                                   const DGParams & dg,
                                                   const PlasmaParams & plasma,
                                                   ParFiniteElementSpace & h1_fes,
                                                   ParGridFunctionArray & yGF,
                                                   ParGridFunctionArray & kGF,
                                                   ParGridFunction & elOrdGF,
                                                   ParGridFunction & hGF,
                                                   const AdvectionDiffusionBC & bcs,
                                                   const CoupledBCs & cbcs,
                                                   const NDCoefs & ndcoefs,
                                                   const CommonCoefs & cmncoefs,
                                                   VectorCoefficient & B3Coef,
                                                   int term_flag,
                                                   int vis_flag,
                                                   int logging,
                                                   const string & log_prefix)
   : TransportOp(mpi, dg, plasma, 0, "Neutral Density", "Neutral Density",
                 NULL, &h1_fes, yGF, kGF, elOrdGF, hGF,
                 bcs, cbcs, cmncoefs, B3Coef,
                 term_flag, vis_flag, logging, log_prefix),
     ndcoefs_(ndcoefs),
     DDefCoef_(neCoef_, vnBarCoef_, izCoef_, cxCoef_),
     DCoef_((ndcoefs_(NDCoefs::DIFFUSION_COEF) != NULL)
            ? const_cast<StateVariableCoef&>
            (*ndcoefs_(NDCoefs::DIFFUSION_COEF))
            : DDefCoef_),
     DGF_(NULL),
     SrcGF_(NULL),
     SizGF_(NULL),
     SGF_(NULL)
{
   if ( mpi_.Root() && logging_ > 1)
   {
      cout << "Constructing NeutralDensityOp" << endl;
      cout << "   Neutral mass:        " << m_n_kg_ << " kg"
           << " (" << m_n_kg_ / kg_per_amu_ << " amu)" << endl;
      cout << "   Neutral temperature: " << T_n_eV_ << " eV" << endl;
      cout << "   Neutral velocity:    " << v_n_ << " m/s" << endl;
   }

   if (term_flag_ < 0)
   {
      // Set default terms
      term_flag_ = 1023;
   }
   if (vis_flag_ < 0)
   {
      // Set default visualization fields
      vis_flag_ = (logging_ > 1) ? 1023 : this->GetDefaultVisFlag();
   }
   cout << "NeutralDensityOp term flag " << term_flag_ << endl;

   // Time derivative term: dn_n / dt
   SetTimeDerivativeTerm(nnCoef_);

   if (this->CheckTermFlag(DIFFUSION_TERM))
   {
      // Diffusion term: -Div(D_n Grad n_n)
      SetDiffusionTerm((ndcoefs_(NDCoefs::DIFFUSION_COEF) != NULL)
                       ? dynamic_cast<StateVariableCoef&>(DCoef_) : DDefCoef_);
   }

   if (this->CheckTermFlag(RECOMBINATION_SOURCE_TERM))
   {
      // Source term: Src
      SetSourceTerm(SrcCoef_, 1.0);
   }
   if (this->CheckTermFlag(IONIZATION_SINK_TERM))
   {
      // Source term: -Siz
      SetSourceTerm(SizCoef_, -1.0);
   }
   if (this->CheckTermFlag(SOURCE_TERM) &&
       ndcoefs_(NDCoefs::SOURCE_COEF) != NULL)
   {
      // Source term from command line
      SetSourceTerm(const_cast<StateVariableCoef&>(*ndcoefs_(NDCoefs::SOURCE_COEF)));
   }
   if (this->CheckTermFlag(RECYCLING_BDR_SOURCE_TERM))
   {
      // cout << "RECYCLING_BDR_SOURCE_TERM is on" << endl;
      /*
       ScalarVectorProductCoefficient * dtVCoef =
         new ScalarVectorProductCoefficient(dt_, ViCoef_);
       dtVCoefs_.Append(dtVCoef);

       if (blf_[1] == NULL)
       {
         blf_[1] = new ParBilinearForm(&fes_);
       }
       blf_[1]->AddBdrFaceIntegrator(new DGTraceIntegrator(*dtVCoef), -1.0, -0.5);
       */
      // SetBdrSourceTerm(niCoef_, ViCoef_);
      // cout << "Num recycling bcs " << cbcs_.GetNumRecyclingBCs() << endl;
      for (int i=0; i<cbcs_.GetNumRecyclingBCs(); i++)
      {
         const RecyclingBC & rbc = cbcs_.GetRecyclingBC(i);
         // cout << i << " neutral index  " << rbc.GetNeutralDensityIndex() << endl;
         // cout << i << " ion index      " << rbc.GetIonDensityIndex() << endl;
         // cout << i << " velocity index " << rbc.GetIonVelocityIndex() << endl;
         if (rbc.GetNeutralDensityIndex() == index_)
         {
            SetRecyclingBdrSourceTerm(rbc);
         }
      }
   }

   if (this->CheckVisFlag(DIFFUSION_COEF))
   {
      DGF_ = new ParGridFunction(&fes_);
   }
   if (this->CheckVisFlag(RECOMBINATION_SOURCE_COEF))
   {
      SrcGF_ = new ParGridFunction(&fes_);
   }
   if (this->CheckVisFlag(IONIZATION_SINK_COEF))
   {
      SizGF_ = new ParGridFunction(&fes_);
   }
   if (this->CheckVisFlag(SOURCE_COEF) &&
       ndcoefs_(NDCoefs::SOURCE_COEF) != NULL)
   {
      SGF_ = new ParGridFunction(&fes_);
   }
   if ( mpi_.Root() && logging_ > 1)
   {
      cout << "Done constructing NeutralDensityOp" << endl;
   }
}

DGTransportTDO::NeutralDensityOp::~NeutralDensityOp()
{
   delete DGF_;
   delete SrcGF_;
   delete SizGF_;
   delete SGF_;
}

void DGTransportTDO::NeutralDensityOp::SetTime(double t)
{
   if (mpi_.Root() && logging_)
   {
      cout << "Setting time: " << t << " in NeutralDensityOp" << endl;
   }
   TransportOp::SetTime(t);

   DDefCoef_.SetTime(t);
}

void DGTransportTDO::NeutralDensityOp::SetTimeStep(double dt)
{
   if (mpi_.Root() && logging_)
   {
      cout << "Setting time step: " << dt << " in NeutralDensityOp" << endl;
   }
   TransportOp::SetTimeStep(dt);
}

void DGTransportTDO::NeutralDensityOp::RegisterDataFields(DataCollection & dc)
{
   NLOperator::RegisterDataFields(dc);

   if (this->CheckVisFlag(DIFFUSION_COEF))
   {
      ostringstream oss;
      oss << eqn_name_ << " D_n";
      dc.RegisterField(oss.str(), DGF_);
   }
   if (this->CheckVisFlag(RECOMBINATION_SOURCE_COEF))
   {
      ostringstream oss;
      oss << eqn_name_ << " Src_n";
      dc.RegisterField(oss.str(), SrcGF_);
   }
   if (this->CheckVisFlag(IONIZATION_SINK_COEF))
   {
      ostringstream oss;
      oss << eqn_name_ << " Siz_n";
      dc.RegisterField(oss.str(), SizGF_);
   }
   if (this->CheckVisFlag(SOURCE_COEF) && SGF_ != NULL)
   {
      ostringstream oss;
      oss << eqn_name_ << " S_n";
      dc.RegisterField(oss.str(), SGF_);
   }
}

void DGTransportTDO::
NeutralDensityOp::PrepareDataFields()
{
   if (this->CheckVisFlag(DIFFUSION_COEF))
   {
      DGF_->ProjectCoefficient(DCoef_);
   }
   if (this->CheckVisFlag(RECOMBINATION_SOURCE_COEF))
   {
      if (this->CheckTermFlag(RECOMBINATION_SOURCE_TERM))
      {
         SrcGF_->ProjectCoefficient(SrcCoef_);
      }
      else
      {
         *SrcGF_ = 0.0;
      }
   }
   if (this->CheckVisFlag(IONIZATION_SINK_COEF))
   {
      if (this->CheckTermFlag(IONIZATION_SINK_TERM))
      {
         SizGF_->ProjectCoefficient(SizCoef_);
      }
      else
      {
         *SizGF_ = 0.0;
      }
   }
   if (this->CheckVisFlag(SOURCE_COEF) && SGF_ != NULL)
   {
      if (this->CheckTermFlag(SOURCE_TERM))
      {
         SGF_->ProjectCoefficient(
            const_cast<StateVariableCoef&>(*ndcoefs_(NDCoefs::SOURCE_COEF)));
      }
      else
      {
         *SGF_ = 0.0;
      }
   }
}

void DGTransportTDO::NeutralDensityOp::Update()
{
   if (mpi_.Root() && logging_ > 1)
   {
      cout << "Entering DGTransportTDO::NeutralDensityOp::Update" << endl;
   }

   TransportOp::Update();

   if (DGF_   != NULL) { DGF_->Update(); }
   if (SrcGF_ != NULL) { SrcGF_->Update(); }
   if (SizGF_ != NULL) { SizGF_->Update(); }
   if (SGF_   != NULL) { SGF_->Update(); }

   if (mpi_.Root() && logging_ > 1)
   {
      cout << "Leaving DGTransportTDO::NeutralDensityOp::Update" << endl;
   }
}

DGTransportTDO::IonDensityOp::IonDensityOp(const MPI_Session & mpi,
                                           const DGParams & dg,
                                           const ArtViscParams & av,
                                           const PlasmaParams & plasma,
                                           ParFiniteElementSpace & vfes,
                                           ParFiniteElementSpace & h1_fes,
                                           ParGridFunctionArray & yGF,
                                           ParGridFunctionArray & kGF,
                                           ParGridFunction & elOrdGF,
                                           ParGridFunction & hGF,
                                           const AdvectionDiffusionBC & bcs,
                                           const CoupledBCs & cbcs,
                                           const IDCoefs & idcoefs,
                                           const CmnCoefs & cmncoefs,
                                           VectorCoefficient & B3Coef,
                                           double DPerp,
                                           int term_flag, int vis_flag,
                                           int logging,
                                           const string & log_prefix)
   : AdvTransportOp(mpi, dg, plasma, 1, "Ion Density", "Ion Density",
                    &vfes, &h1_fes, yGF, kGF, elOrdGF, hGF,
                    bcs, cbcs, cmncoefs, B3Coef,
                    term_flag, vis_flag, logging, log_prefix),
     idcoefs_(idcoefs),
     av_(av),
     DPerpConstCoef_(DPerp),
     DParaCoef_(CsCoef_,
                &elOrdCoef_, &OscContCoef_, &hCoef_, av_),
     DParaCoefPtr_((idcoefs_(IDCoefs::PARA_DIFFUSION_COEF) != NULL)
                   ? const_cast<StateVariableCoef*>
                   (idcoefs_(IDCoefs::PARA_DIFFUSION_COEF))
                   : NULL),
     DPerpCoefPtr_((idcoefs_(IDCoefs::PERP_DIFFUSION_COEF) != NULL)
                   ? const_cast<StateVariableCoef*>
                   (idcoefs_(IDCoefs::PERP_DIFFUSION_COEF))
                   : &DPerpConstCoef_),
     DCoef_(DParaCoefPtr_,
            DPerpCoefPtr_, B3Coef_),
     ViCoef_(viCoef_, B3Coef_),
     DParaGF_(NULL),
     DPerpGF_(NULL),
     AdvGF_(NULL),
     SizGF_(NULL),
     SrcGF_(NULL),
     SGF_(NULL)
{
   if ( mpi_.Root() && logging_ > 1)
   {
      cout << "Constructing IonDensityOp" << endl;
      cout << "   Ion mass:   " << m_i_kg_ << " kg "
           << "(" << m_i_kg_ / kg_per_amu_ << " amu)" << endl;
      cout << "   Ion charge: " << z_i_ << " e" << endl;
   }

   if (term_flag_ < 0)
   {
      // Set default terms
      term_flag_ = 1023;
   }
   if (vis_flag_ < 0)
   {
      // Set default visualization fields
      vis_flag_ = (logging_ > 1) ? 1023 : this->GetDefaultVisFlag();
   }

   cout << "IonDensityOp term flag " << term_flag_ << endl;

   // Time derivative term: dn_i / dt
   // dbfi_m_[1].Append(new MassIntegrator);
   SetTimeDerivativeTerm(niCoef_);

   if (this->CheckTermFlag(DIFFUSION_TERM) &&
       this->CheckTermFlag(ADVECTION_TERM))
   {
      // Advection-Diffusion term: -Div(D_i Grad n_i - v_i n_i)
      SetAdvectionDiffusionTerm(DCoef_, ViCoef_, DParaCoefPtr_, DPerpCoefPtr_);
   }
   else
   {
      if (this->CheckTermFlag(DIFFUSION_TERM))
      {
         // Diffusion term: -Div(D_i Grad n_i)
         SetAnisotropicDiffusionTerm(DCoef_, DParaCoefPtr_, DPerpCoefPtr_);
      }

      if (this->CheckTermFlag(ADVECTION_TERM))
      {
         // Advection term: Div(v_i n_i)
         SetAdvectionTerm(ViCoef_/*, true*/);
      }
   }

   if (this->CheckTermFlag(ADVECTION_TERM) && bcs_.GetOutflowBCs().Size() > 0)
   {
      SetOutflowBdrTerm(ViCoef_, bcs_.GetOutflowBCs());
   }

   if (this->CheckTermFlag(IONIZATION_SOURCE_TERM))
   {
      // Source term: Siz
      SetSourceTerm(SizCoef_, 1.0);
   }
   if (this->CheckTermFlag(RECOMBINATION_SINK_TERM))
   {
      // Source term: -Src
      SetSourceTerm(SrcCoef_, -1.0);
   }

   if (this->CheckTermFlag(SOURCE_TERM) &&
       idcoefs_(IDCoefs::SOURCE_COEF) != NULL)
   {
      // Source term from command line or input file
      SetSourceTerm(const_cast<StateVariableCoef&>(*idcoefs_(IDCoefs::SOURCE_COEF)));
   }

   if (this->CheckVisFlag(DIFFUSION_PARA_COEF))
   {
      DParaGF_ = new ParGridFunction(&fes_);
   }
   if (this->CheckVisFlag(DIFFUSION_PERP_COEF))
   {
      DPerpGF_ = new ParGridFunction(&fes_);
   }
   if (this->CheckVisFlag(ADVECTION_COEF))
   {
      AdvGF_ = new ParGridFunction(vfes_);
   }
   if (this->CheckVisFlag(IONIZATION_SOURCE_COEF))
   {
      SizGF_ = new ParGridFunction(&fes_);
   }
   if (this->CheckVisFlag(RECOMBINATION_SINK_COEF))
   {
      SrcGF_ = new ParGridFunction(&fes_);
   }
   if (this->CheckVisFlag(SOURCE_COEF) &&
       idcoefs_(IDCoefs::SOURCE_COEF) != NULL)
   {
      SGF_ = new ParGridFunction(&fes_);
   }
   if ( mpi_.Root() && logging_ > 1)
   {
      cout << "Done constructing IonDensityOp" << endl;
   }
}


void DGTransportTDO::IonDensityOp::SetTime(double t)
{
   if (mpi_.Root() && logging_)
   {
      cout << "Setting time: " << t << " in IonDensityOp" << endl;
   }
   TransportOp::SetTime(t);

   DCoef_.SetTime(t);
   ViCoef_.SetTime(t);
}

void DGTransportTDO::IonDensityOp::SetTimeStep(double dt)
{
   if (mpi_.Root() && logging_)
   {
      cout << "Setting time step: " << dt << " in IonDensityOp" << endl;
   }
   TransportOp::SetTimeStep(dt);
}

DGTransportTDO::IonDensityOp::~IonDensityOp()
{
   delete DParaGF_;
   delete DPerpGF_;
   delete AdvGF_;
   delete SizGF_;
   delete SrcGF_;
   delete SGF_;
}

void DGTransportTDO::IonDensityOp::RegisterDataFields(DataCollection & dc)
{
   NLOperator::RegisterDataFields(dc);

   if (this->CheckVisFlag(DIFFUSION_PARA_COEF))
   {
      ostringstream oss;
      oss << eqn_name_ << " D_i Parallel";
      dc.RegisterField(oss.str(), DParaGF_);
   }
   if (this->CheckVisFlag(DIFFUSION_PERP_COEF))
   {
      ostringstream oss;
      oss << eqn_name_ << " D_i Perpendicular";
      dc.RegisterField(oss.str(), DPerpGF_);
   }
   if (this->CheckVisFlag(ADVECTION_COEF))
   {
      ostringstream oss;
      oss << eqn_name_ << " V_i";
      dc.RegisterField(oss.str(), AdvGF_);
   }
   if (this->CheckVisFlag(IONIZATION_SOURCE_COEF))
   {
      ostringstream oss;
      oss << eqn_name_ << " Siz_i";
      dc.RegisterField(oss.str(), SizGF_);
   }
   if (this->CheckVisFlag(RECOMBINATION_SINK_COEF))
   {
      ostringstream oss;
      oss << eqn_name_ << " Src_i";
      dc.RegisterField(oss.str(), SrcGF_);
   }
   if (this->CheckVisFlag(SOURCE_COEF) && SGF_ != NULL)
   {
      ostringstream oss;
      oss << eqn_name_ << " S_i";
      dc.RegisterField(oss.str(), SGF_);
   }
}

void DGTransportTDO::IonDensityOp::PrepareDataFields()
{
   if (this->CheckVisFlag(DIFFUSION_PARA_COEF))
   {
      if (DParaCoefPtr_ != NULL)
      {
         DParaGF_->ProjectCoefficient(*DParaCoefPtr_);
      }
      else
      {
         *DParaGF_ = 0.0;
      }
   }
   if (this->CheckVisFlag(DIFFUSION_PERP_COEF))
   {
      if (DPerpCoefPtr_ != NULL)
      {
         DPerpGF_->ProjectCoefficient(*DPerpCoefPtr_);
      }
      else
      {
         *DPerpGF_ = 0.0;
      }
   }
   if (this->CheckVisFlag(ADVECTION_COEF))
   {
      AdvGF_->ProjectCoefficient(ViCoef_);
   }
   if (this->CheckVisFlag(IONIZATION_SOURCE_COEF))
   {
      if (this->CheckTermFlag(IONIZATION_SOURCE_TERM))
      {
         SizGF_->ProjectCoefficient(SizCoef_);
      }
      else
      {
         *SizGF_ = 0.0;
      }
   }
   if (this->CheckVisFlag(RECOMBINATION_SINK_COEF))
   {
      if (this->CheckTermFlag(RECOMBINATION_SINK_TERM))
      {
         SrcGF_->ProjectCoefficient(SrcCoef_);
      }
      else
      {
         *SrcGF_ = 0.0;
      }
   }
   if (this->CheckVisFlag(SOURCE_COEF) && SGF_ != NULL)
   {
      SGF_->ProjectCoefficient(
         const_cast<StateVariableCoef&>(*idcoefs_(IDCoefs::SOURCE_COEF)));
   }
}

void DGTransportTDO::IonDensityOp::Update()
{
   if (mpi_.Root() && logging_ > 1)
   {
      cout << "Entering DGTransportTDO::IonDensityOp::Update" << endl;
   }

   AdvTransportOp::Update();

   if (DParaGF_ != NULL) { DParaGF_->Update(); }
   if (DPerpGF_ != NULL) { DPerpGF_->Update(); }
   if (AdvGF_   != NULL) { AdvGF_->Update(); }
   if (SizGF_   != NULL) { SizGF_->Update(); }
   if (SrcGF_   != NULL) { SrcGF_->Update(); }
   if (SGF_     != NULL) { SGF_->Update(); }

   if (mpi_.Root() && logging_ > 1)
   {
      cout << "Leaving DGTransportTDO::IonDensityOp::Update" << endl;
   }
}

DGTransportTDO::IonMomentumOp::IonMomentumOp(const MPI_Session & mpi,
                                             const DGParams & dg,
                                             const ArtViscParams & av,
                                             const PlasmaParams & plasma,
                                             ParFiniteElementSpace & vfes,
                                             ParFiniteElementSpace & h1_fes,
                                             ParGridFunctionArray & yGF,
                                             ParGridFunctionArray & kGF,
                                             ParGridFunction & elOrdGF,
                                             ParGridFunction & hGF,
                                             const AdvectionDiffusionBC & bcs,
                                             const CoupledBCs & cbcs,
                                             const IMCoefs & imcoefs,
                                             const CmnCoefs & cmncoefs,
                                             VectorCoefficient & B3Coef,
                                             double DPerp,
                                             int term_flag, int vis_flag,
                                             int logging,
                                             const string & log_prefix)
   : AdvTransportOp(mpi, dg, plasma, 2, "Ion Parallel Momentum",
                    "Ion Parallel Velocity", &vfes, &h1_fes,
                    yGF, kGF, elOrdGF, hGF,
                    bcs, cbcs, cmncoefs, B3Coef, term_flag, vis_flag,
                    logging, log_prefix),
     imcoefs_(imcoefs),
     av_(av),
     /*
     l2_fes_0_(new L2_ParFESpace(&pmesh_, 0, pmesh_.SpaceDimension())),
     h1_fes_1_(new H1_ParFESpace(&pmesh_, 1, pmesh_.SpaceDimension())),
     elOrdDiscGF_(new ParGridFunction(l2_fes_0_)),
     elOrdContGF_(new ParGridFunction(h1_fes_1_)),
     OscDiscGF_(new ParGridFunction(l2_fes_0_)),
     OscContGF_(new ParGridFunction(h1_fes_1_)),
     hDiscGF_(new ParGridFunction(l2_fes_0_)),
     hContGF_(new ParGridFunction(h1_fes_1_)),
     elOrdDiscCoef_(elOrdDiscGF_),
     elOrdContCoef_(elOrdContGF_),
     OscDiscCoef_(OscDiscGF_),
     OscContCoef_(OscContGF_),
     hDiscCoef_(hDiscGF_),
     hContCoef_(hContGF_),
     */
     // CsCoef_(m_i_kg_, TiCoef_, TeCoef_),
     DPerpConst_(DPerp),
     DPerpCoef_(DPerp),
     momCoef_(m_i_kg_, niCoef_, viCoef_),
     EtaParaCoef_(z_i_, m_i_kg_, lnLambda_, TiCoef_,
                  niCoef_, CsCoef_,
                  &elOrdCoef_, &OscContCoef_, &hCoef_,
                  av),
     EtaPerpCoef_(DPerpConst_, m_i_kg_, niCoef_),
     EtaParaCoefPtr_((imcoefs_(IMCoefs::PARA_DIFFUSION_COEF) != NULL)
                     ? const_cast<StateVariableCoef*>
                     (imcoefs_(IMCoefs::PARA_DIFFUSION_COEF))
                     : &EtaParaCoef_),
     EtaPerpCoefPtr_((imcoefs_(IMCoefs::PERP_DIFFUSION_COEF) != NULL)
                     ? const_cast<StateVariableCoef*>
                     (imcoefs_(IMCoefs::PERP_DIFFUSION_COEF))
                     : &EtaPerpCoef_),
     EtaCoef_(EtaParaCoefPtr_,
              EtaPerpCoefPtr_, B3Coef_),
     miniViCoef_(niCoef_, viCoef_, m_i_kg_, DPerpCoef_, B3Coef_),
     negGradPCoef_(yGF, kGF, z_i_, B3Coef_),
     negBPCoef_(yGF, kGF, z_i_, B3Coef_),
     negPCoef_(yGF, kGF, z_i_),
     SIZCoef_(z_i_, m_i_kg_, nnCoef_, niCoef_, vnAvgCoef_, izCoef_),
     SRCCoef_(z_i_, m_i_kg_, niCoef_, viCoef_, rcCoef_),
     SCXCoef_(m_i_kg_, nnCoef_, niCoef_, vnAvgCoef_, viCoef_, cxCoef_),
     EtaParaGF_(NULL),
     EtaPerpGF_(NULL),
     AdvGF_(NULL),
     MomParaGF_(NULL),
     SGPGF_(NULL),
     SIZGF_(NULL),
     SRCGF_(NULL),
     SCXGF_(NULL),
     SGF_(NULL)
{
   if ( mpi_.Root() && logging_ > 1)
   {
      cout << "Constructing IonMomentumOp" << endl;
   }

   if (term_flag_ < 0)
   {
      // Set default terms
      term_flag_ = 1023;
   }
   if (vis_flag_ < 0)
   {
      // Set default visualization fields
      vis_flag_ = (logging_ > 1) ? 1023 : this->GetDefaultVisFlag();
   }
   /*
   ElementOrder(*vfes_, *elOrdDiscGF_);
   elOrdDiscGF_->ExchangeFaceNbrData();
   elOrdContGF_->ProjectDiscCoefficient(elOrdDiscCoef_, GridFunction::MINIMUM);
   elOrdContGF_->ExchangeFaceNbrData();
   */
   (*OscDiscGF_) = 0.0;
   DiscontinuitySensor(*yGF_[2], av_.osc_thresh, 1.0, *OscDiscGF_);
   OscDiscGF_->ExchangeFaceNbrData();
   OscContGF_->ProjectDiscCoefficient(OscDiscCoef_, GridFunction::MAXIMUM);
   OscContGF_->ExchangeFaceNbrData();
   /*
   ParallelMeshSpacing(*vfes_, B3Coef_, *hDiscGF_);
   hDiscGF_->ExchangeFaceNbrData();
   hContGF_->ProjectDiscCoefficient(hDiscCoef_, GridFunction::MAXIMUM);
   hContGF_->ExchangeFaceNbrData();
   */
   // Time derivative term: d(m_i n_i v_i)/dt
   SetTimeDerivativeTerm(momCoef_);

   if (this->CheckTermFlag(DIFFUSION_TERM) &&
       this->CheckTermFlag(ADVECTION_TERM))
   {
      // Advection-Diffusion term: -Div(Eta_i Grad v_i - v_i m_i n_i v_i)
      SetAdvectionDiffusionTerm(EtaCoef_, miniViCoef_,
                                EtaParaCoefPtr_, EtaPerpCoefPtr_);
   }
   else
   {
      if (this->CheckTermFlag(DIFFUSION_TERM))
      {
         // Diffusion term: -Div(Eta_i Grad v_i)
         SetAnisotropicDiffusionTerm(EtaCoef_, EtaParaCoefPtr_, EtaPerpCoefPtr_);
      }

      if (this->CheckTermFlag(ADVECTION_TERM))
      {
         // Advection term: Div(m_i n_i v_i v_i)
         SetAdvectionTerm(miniViCoef_/*, true*/);
      }
   }

   if (this->CheckTermFlag(ADVECTION_TERM) && bcs_.GetOutflowBCs().Size() > 0)
   {
      SetOutflowBdrTerm(miniViCoef_, bcs_.GetOutflowBCs());
   }
   /*
   if (this->CheckTermFlag(DIFFUSION_TERM))
   {
      // Diffusion term: -Div(eta Grad v_i)
      SetDiffusionTerm(EtaCoef_);
   }

   if (this->CheckTermFlag(ADVECTION_TERM))
   {
      // Advection term: Div(m_i n_i v_i v_i)
      SetAdvectionTerm(miniViCoef_, true);
   }
   */
   if (this->CheckTermFlag(GRADP_SOURCE_TERM))
   {
      // Source term: - b . Grad(p_i + p_e)
      // dlfi_.Append(new DomainLFIntegrator(negGradPCoef_));
      SetSourceTerm(negGradPCoef_, 1.0);
   }
   if (this->CheckTermFlag(DIVBP_SOURCE_TERM))
   {
      // Source term: - Div(b (p_i + p_e))
      // SetDivergenceTerm(negBPCoef_);
      SetDivergenceTerm(negPCoef_, UnitBxyCoef_);
   }
   if (this->CheckTermFlag(DIVBP_SOURCE_TERM) &&
       bcs_.GetOutflowBCs().Size() > 0)
   {
      // SetOutflowBdrTerm(negBPCoef_, bcs_.GetOutflowBCs());
      const Array<CoefficientByAttr*> & obc = bcs_.GetOutflowBCs();
      for (int j=0; j<obc.Size(); j++)
      {
         for (int i=0; i<5; i++)
         {
            if (negBPCoef_.NonTrivialValue((FieldType)i))
            {
               if ( mpi_.Root() && logging_ > 0)
               {
                  cout << eqn_name_
                       << ": Adding outflow BC for divergence term proportional to "
                       << FieldSymbol((FieldType)i) << endl;
               }

               StateVariableVecCoef * coef = negBPCoef_.Clone();
               coef->SetDerivType((FieldType)i);
               svvcoefs_.Append(coef);

               ScalarVectorProductCoefficient * rVCoef =
                  new ScalarVectorProductCoefficient(*obc[j]->coef, *coef);
               vCoefs_.Append(rVCoef);

               ScalarVectorProductCoefficient * dtrVCoef =
                  new ScalarVectorProductCoefficient(dt_, *rVCoef);
               dtVCoefs_.Append(dtrVCoef);

               bfbfi_[i].Append(new DGTraceIntegrator(*rVCoef, 2.0, 0.0));
               bfbfi_marker_[i].Append(new Array<int>);
               AttrToMarker(pmesh_.bdr_attributes.Max(), obc[j]->attr,
                            *bfbfi_marker_[i].Last());

               if (blf_[i] == NULL)
               {
                  blf_[i] = new ParBilinearForm(&fes_);
               }
               blf_[i]->AddBdrFaceIntegrator(new DGTraceIntegrator(*dtrVCoef,
                                                                   2.0, 0.0),
                                             *bfbfi_marker_[i].Last());
            }
         }
      }
   }
   if (this->CheckTermFlag(IONIZATION_SOURCE_TERM))
   {
      SetSourceTerm(SIZCoef_, 1.0);
   }
   if (this->CheckTermFlag(RECOMBINATION_SINK_TERM))
   {
      SetSourceTerm(SRCCoef_, -1.0);
   }
   if (this->CheckTermFlag(CHARGE_EXCHANGE_SOURCE_TERM))
   {
      SetSourceTerm(SCXCoef_, 1.0);
   }
   if (this->CheckTermFlag(SOURCE_TERM) &&
       imcoefs_(IMCoefs::SOURCE_COEF) != NULL)
   {
      // Source term from command line
      SetSourceTerm(const_cast<StateVariableCoef&>(*imcoefs_(IMCoefs::SOURCE_COEF)));
   }

   if (this->CheckVisFlag(DIFFUSION_PARA_COEF))
   {
      EtaParaGF_ = new ParGridFunction(&fes_);
   }
   if (this->CheckVisFlag(DIFFUSION_PERP_COEF))
   {
      EtaPerpGF_ = new ParGridFunction(&fes_);
   }
   if (this->CheckVisFlag(ADVECTION_COEF))
   {
      AdvGF_ = new ParGridFunction(vfes_);
   }
   if (this->CheckVisFlag(GRADP_SOURCE_COEF))
   {
      SGPGF_ = new ParGridFunction(&fes_);
   }
   if (this->CheckVisFlag(IONIZATION_SOURCE_COEF))
   {
      SIZGF_ = new ParGridFunction(&fes_);
   }
   if (this->CheckVisFlag(RECOMBINATION_SINK_COEF))
   {
      SRCGF_ = new ParGridFunction(&fes_);
   }
   if (this->CheckVisFlag(CHARGE_EXCHANGE_SOURCE_COEF))
   {
      SCXGF_ = new ParGridFunction(&fes_);
   }
   if (this->CheckVisFlag(SOURCE_COEF) &&
       imcoefs_(IMCoefs::SOURCE_COEF) != NULL)
   {
      SGF_ = new ParGridFunction(&fes_);
   }
   if (this->CheckVisFlag(ION_PARA_MOMENTUM))
   {
      MomParaGF_ = new ParGridFunction(&vfes);
   }
   if ( mpi_.Root() && logging_ > 1)
   {
      cout << "Done constructing IonMomentumOp" << endl;
   }
}

DGTransportTDO::IonMomentumOp::~IonMomentumOp()
{
   delete EtaPerpGF_;
   delete EtaParaGF_;
   delete AdvGF_;
   delete MomParaGF_;
   delete SGPGF_;
   delete SIZGF_;
   delete SRCGF_;
   delete SCXGF_;
   delete SGF_;
   // delete elOrdDiscGF_;
   // delete elOrdContGF_;
   // delete OscDiscGF_;
   // delete OscContGF_;
   // delete hDiscGF_;
   // delete hContGF_;
   // delete l2_fes_0_;
   // delete h1_fes_1_;
}

void DGTransportTDO::IonMomentumOp::SetDivergenceTerm(StateVariableVecCoef
                                                      &VCoef)
{
   for (int i=0; i<5; i++)
   {
      if (VCoef.NonTrivialValue((FieldType)i))
      {
         if ( mpi_.Root() && logging_ > 0)
         {
            cout << eqn_name_
                 << ": Adding divergence term proportional to "
                 << FieldSymbol((FieldType)i) << endl;
         }

         StateVariableVecCoef * coef = VCoef.Clone();
         coef->SetDerivType((FieldType)i);
         svvcoefs_.Append(coef);

         ScalarVectorProductCoefficient * dtVCoef =
            new ScalarVectorProductCoefficient(dt_, *coef);
         dtVCoefs_.Append(dtVCoef);

         dbfi_[i].Append(new ConservativeConvectionIntegrator(*coef, 1.0));
         // fbfi_[i].Append(new DGTraceIntegrator(*coef, 1.0, -0.5));
         // fbfi_[i].Append(new DGTraceIntegrator(*coef, 2.0, 0.0));
         fbfi_[i].Append(new DGTraceIntegrator(*coef, 1.0, 0.5));

         if (blf_[i] == NULL)
         {
            blf_[i] = new ParBilinearForm(&fes_);
         }

         blf_[i]->AddDomainIntegrator(
            new ConservativeConvectionIntegrator(*dtVCoef, 1.0));
         // blf_[i]->AddInteriorFaceIntegrator(new DGTraceIntegrator(*dtVCoef,
         //                       1.0, -0.5));
         // blf_[i]->AddInteriorFaceIntegrator(new DGTraceIntegrator(*dtVCoef,
         //                       2.0, 0.0));
         blf_[i]->AddInteriorFaceIntegrator(new DGTraceIntegrator(*dtVCoef,
                                                                  1.0, 0.5));
         if (h1_fes_ != NULL)
         {
            if (cgblf_[i] == NULL)
            {
               cgblf_[i] = new ParBilinearForm(h1_fes_);
            }
            cgblf_[i]->AddDomainIntegrator(
               new ConservativeConvectionIntegrator(*dtVCoef, 1.0));
         }
      }
   }
}

void DGTransportTDO::IonMomentumOp::SetDivergenceTerm(StateVariableCoef &Coef,
                                                      VectorCoefficient &VCoef)
{
   for (int i=0; i<5; i++)
   {
      if (Coef.NonTrivialValue((FieldType)i))
      {
         if ( mpi_.Root() && logging_ > 0)
         {
            cout << eqn_name_
                 << ": Adding divergence term proportional to "
                 << FieldSymbol((FieldType)i) << endl;
         }

         StateVariableCoef * coef = Coef.Clone();
         coef->SetDerivType((FieldType)i);
         svscoefs_.Append(coef);

         VectorCoefficient *vcoef =
            new ScalarVectorProductCoefficient(*coef, VCoef);
         vCoefs_.Append(vcoef);

         ProductCoefficient * dtCoef =
            new ProductCoefficient(dt_, *coef);
         dtSCoefs_.Append(dtCoef);

         ScalarVectorProductCoefficient * dtVCoef =
            new ScalarVectorProductCoefficient(dt_, *vcoef);
         dtVCoefs_.Append(dtVCoef);

         dbfi_[i].Append(new ConservativeConvectionIntegrator(*vcoef, 1.0));
         // fbfi_[i].Append(new DGTraceIntegrator(*coef, VCoef, 1.0, -0.5));
         fbfi_[i].Append(new DGTraceIntegrator(*coef, VCoef, 1.0, 0.0));
         // fbfi_[i].Append(new DGTraceIntegrator(*coef, VCoef, 1.0, 0.5));

         if (blf_[i] == NULL)
         {
            blf_[i] = new ParBilinearForm(&fes_);
         }

         blf_[i]->AddDomainIntegrator(
            new ConservativeConvectionIntegrator(*dtVCoef, 1.0));
         // blf_[i]->AddInteriorFaceIntegrator(new DGTraceIntegrator(*dtCoef,
         //                       VCoef,
         //                       1.0, -0.5));
         blf_[i]->AddInteriorFaceIntegrator(new DGTraceIntegrator(*dtCoef,
                                                                  VCoef,
                                                                  1.0, 0.0));
         // blf_[i]->AddInteriorFaceIntegrator(new DGTraceIntegrator(*dtCoef,
         //                       VCoef,
         //                       1.0, 0.5));
         if (h1_fes_ != NULL)
         {
            if (cgblf_[i] == NULL)
            {
               cgblf_[i] = new ParBilinearForm(h1_fes_);
            }
            cgblf_[i]->AddDomainIntegrator(
               new ConservativeConvectionIntegrator(*dtVCoef, 1.0));
         }
      }
   }
}

void DGTransportTDO::IonMomentumOp::SetTime(double t)
{
   if (mpi_.Root() && logging_)
   {
      cout << "Setting time: " << t << " in IonMomentumOp" << endl;
   }
   TransportOp::SetTime(t);

   momCoef_.SetTime(t);
   EtaParaCoef_.SetTime(t);
   EtaPerpCoef_.SetTime(t);
   EtaCoef_.SetTime(t);
   miniViCoef_.SetTime(t);
   negGradPCoef_.SetTime(t);
   SIZCoef_.SetTime(t);
   SRCCoef_.SetTime(t);
   SCXCoef_.SetTime(t);
}

void DGTransportTDO::IonMomentumOp::SetTimeStep(double dt)
{
   if (mpi_.Root() && logging_)
   {
      cout << "Setting time step: " << dt << " in IonMomentumOp" << endl;
   }
   TransportOp::SetTimeStep(dt);
}

void DGTransportTDO::
IonMomentumOp::RegisterDataFields(DataCollection & dc)
{
   NLOperator::RegisterDataFields(dc);

   if (this->CheckVisFlag(DIFFUSION_PARA_COEF))
   {
      ostringstream oss;
      oss << eqn_name_ << " Eta_i Parallel";
      dc.RegisterField(oss.str(), EtaParaGF_);
   }
   if (this->CheckVisFlag(DIFFUSION_PERP_COEF))
   {
      ostringstream oss;
      oss << eqn_name_ << " Eta_i Perpendicular";
      dc.RegisterField(oss.str(), EtaPerpGF_);
   }
   if (this->CheckVisFlag(ADVECTION_COEF))
   {
      ostringstream oss;
      oss << eqn_name_ << " mnV_i";
      dc.RegisterField(oss.str(), AdvGF_);
   }
   if (this->CheckVisFlag(GRADP_SOURCE_COEF))
   {
      ostringstream oss;
      oss << eqn_name_ << " bGradP_i";
      dc.RegisterField(oss.str(), SGPGF_);
   }
   if (this->CheckVisFlag(IONIZATION_SOURCE_COEF))
   {
      ostringstream oss;
      oss << eqn_name_ << " Ionization Source";
      dc.RegisterField(oss.str(), SIZGF_);
   }
   if (this->CheckVisFlag(RECOMBINATION_SINK_COEF))
   {
      ostringstream oss;
      oss << eqn_name_ << " Recombination Sink";
      dc.RegisterField(oss.str(), SRCGF_);
   }
   if (this->CheckVisFlag(CHARGE_EXCHANGE_SOURCE_COEF))
   {
      ostringstream oss;
      oss << eqn_name_ << " Charge Exchange Source";
      dc.RegisterField(oss.str(), SCXGF_);
   }
   if (this->CheckVisFlag(SOURCE_COEF) && SGF_ != NULL)
   {
      ostringstream oss;
      oss << eqn_name_ << " S_im";
      dc.RegisterField(oss.str(), SGF_);
   }
   if (this->CheckVisFlag(ION_PARA_MOMENTUM))
   {
      ostringstream oss;
      oss << eqn_name_;
      dc.RegisterField(oss.str(), MomParaGF_);
   }
   /*
   {
      ostringstream oss;
      oss << eqn_name_ << " ElOrd L2";
      dc.RegisterField(oss.str(), elOrdDiscGF_);
   }
   {
      ostringstream oss;
      oss << eqn_name_ << " ElOrd H1";
      dc.RegisterField(oss.str(), elOrdContGF_);
   }
   */
   {
      ostringstream oss;
      oss << eqn_name_ << " Osc L2";
      dc.RegisterField(oss.str(), OscDiscGF_);
   }
   {
      ostringstream oss;
      oss << eqn_name_ << " Osc H1";
      dc.RegisterField(oss.str(), OscContGF_);
   }
   /*
   {
      ostringstream oss;
      oss << eqn_name_ << " Parallel h L2";
      dc.RegisterField(oss.str(), hDiscGF_);
   }
   {
      ostringstream oss;
      oss << eqn_name_ << " Parallel h H1";
      dc.RegisterField(oss.str(), hContGF_);
   }
   */
}

void DGTransportTDO::
IonMomentumOp::PrepareDataFields()
{
   if (this->CheckVisFlag(DIFFUSION_PARA_COEF))
   {
      if (EtaParaCoefPtr_ != NULL)
      {
         EtaParaGF_->ProjectCoefficient(*EtaParaCoefPtr_);
      }
      else
      {
         *EtaParaGF_ = 0.0;
      }
   }
   if (this->CheckVisFlag(DIFFUSION_PERP_COEF))
   {
      if (EtaPerpCoefPtr_ != NULL)
      {
         EtaPerpGF_->ProjectCoefficient(*EtaPerpCoefPtr_);
      }
      else
      {
         *EtaPerpGF_ = 0.0;
      }
   }
   if (this->CheckVisFlag(ADVECTION_COEF))
   {
      AdvGF_->ProjectCoefficient(miniViCoef_);
   }
   if (this->CheckVisFlag(GRADP_SOURCE_COEF))
   {
      SGPGF_->ProjectCoefficient(negGradPCoef_);
   }
   if (this->CheckVisFlag(IONIZATION_SOURCE_COEF))
   {
      SIZGF_->ProjectCoefficient(SIZCoef_);
   }
   if (this->CheckVisFlag(RECOMBINATION_SINK_COEF))
   {
      SRCGF_->ProjectCoefficient(SRCCoef_);
   }
   if (this->CheckVisFlag(CHARGE_EXCHANGE_SOURCE_COEF))
   {
      SCXGF_->ProjectCoefficient(SCXCoef_);
   }
   if (this->CheckVisFlag(SOURCE_COEF) && SGF_ != NULL)
   {
      SGF_->ProjectCoefficient(
         const_cast<StateVariableCoef&>(*imcoefs_(IMCoefs::SOURCE_COEF)));
   }
   if (this->CheckVisFlag(ION_PARA_MOMENTUM))
   {
      MomParaGF_->ProjectCoefficient(miniViCoef_);
   }
   /*
   {
      DiscontinuitySensor(*yGF_[2], *OscDiscGF_);
      OscDiscGF_->ExchangeFaceNbrData();
      OscContGF_->ProjectDiscCoefficient(OscDiscCoef_, GridFunction::MAXIMUM);

      ParallelMeshSpacing(*vfes_, B3Coef_, *hDiscGF_);
      hDiscGF_->ExchangeFaceNbrData();
      hContGF_->ProjectDiscCoefficient(hDiscCoef_, GridFunction::MAXIMUM);
   }
   */
}

void DGTransportTDO::IonMomentumOp::Update()
{
   if (mpi_.Root() && logging_ > 1)
   {
      cout << "Entering DGTransportTDO::IonMomentumOp::Update" << endl;
   }
   /*
   if (l2_fes_0_  != NULL) { l2_fes_0_->Update(); }
   if (h1_fes_1_  != NULL) { h1_fes_1_->Update(); }

   if (elOrdDiscGF_ != NULL) { elOrdDiscGF_->Update(); }
   if (elOrdContGF_ != NULL) { elOrdContGF_->Update(); }

   if (OscDiscGF_   != NULL) { OscDiscGF_->Update(); }
   if (OscContGF_   != NULL) { OscContGF_->Update(); }

   if (hDiscGF_     != NULL) { hDiscGF_->Update(); }
   if (hContGF_     != NULL) { hContGF_->Update(); }
   */
   AdvTransportOp::Update();

   if (EtaParaGF_ != NULL) { EtaParaGF_->Update(); }
   if (EtaPerpGF_ != NULL) { EtaPerpGF_->Update(); }
   if (AdvGF_     != NULL) { AdvGF_->Update(); }
   if (MomParaGF_ != NULL) { MomParaGF_->Update(); }
   if (SGPGF_     != NULL) { SGPGF_->Update(); }
   if (SIZGF_     != NULL) { SIZGF_->Update(); }
   if (SRCGF_     != NULL) { SRCGF_->Update(); }
   if (SCXGF_     != NULL) { SCXGF_->Update(); }
   if (SGF_       != NULL) { SGF_->Update(); }

   if (mpi_.Root() && logging_ > 1)
   {
      cout << "Leaving DGTransportTDO::IonMomentumOp::Update" << endl;
   }
}

void DGTransportTDO::IonMomentumOp::PrepareGradient()
{
   if (mpi_.Root() && logging_ > 1)
   {
      cout << "Entering DGTransportTDO::IonMomentumOp::PrepareGradient" << endl;
   }
   /*
   ElementOrder(*vfes_, *elOrdDiscGF_);
   elOrdDiscGF_->ExchangeFaceNbrData();
   elOrdContGF_->ProjectDiscCoefficient(elOrdDiscCoef_, GridFunction::MINIMUM);
   elOrdContGF_->ExchangeFaceNbrData();
   */
   DiscontinuitySensor(*yGF_[2], av_.osc_thresh, 0.1, *OscDiscGF_);
   OscDiscGF_->ExchangeFaceNbrData();
   OscContGF_->ProjectDiscCoefficient(OscDiscCoef_, GridFunction::MAXIMUM);
   OscContGF_->ExchangeFaceNbrData();
   /*
   ParallelMeshSpacing(*vfes_, B3Coef_, *hDiscGF_);
   hDiscGF_->ExchangeFaceNbrData();
   hContGF_->ProjectDiscCoefficient(hDiscCoef_, GridFunction::MAXIMUM);
   hContGF_->ExchangeFaceNbrData();
   */
   if (mpi_.Root() && logging_ > 1)
   {
      cout << "Leaving DGTransportTDO::IonMomentumOp::PrepareGradient" << endl;
   }
}

DGTransportTDO::IonStaticPressureOp::
IonStaticPressureOp(const MPI_Session & mpi,
                    const DGParams & dg,
                    const PlasmaParams & plasma,
                    ParGridFunctionArray & yGF,
                    ParGridFunctionArray & kGF,
                    ParGridFunction & elOrdGF,
                    ParGridFunction & hGF,
                    const AdvectionDiffusionBC & bcs,
                    const CoupledBCs & cbcs,
                    const ISPCoefs & ispcoefs,
                    const CmnCoefs & cmncoefs,
                    VectorCoefficient & B3Coef,
                    double ChiPerp,
                    int term_flag, int vis_flag,
                    int logging,
                    const string & log_prefix)
   : AdvTransportOp(mpi, dg, plasma, 3, "Ion Static Pressure", "Ion Temperature",
                    NULL, NULL, yGF, kGF, elOrdGF, hGF,
                    bcs, cbcs, cmncoefs, B3Coef,
                    term_flag, vis_flag, logging, log_prefix),
     ispcoefs_(ispcoefs),
     ChiPerpConst_(ChiPerp),
     presCoef_(niCoef_, TiCoef_),
     aniViCoef_(niCoef_, viCoef_, 2.5, B3Coef_),
     ChiParaCoef_(plasma.z_i, plasma.m_i_kg, lnLambda_, niCoef_, TiCoef_),
     ChiPerpCoef_(ChiPerpConst_),
     ChiParaCoefPtr_((ispcoefs_(ISPCoefs::PARA_DIFFUSION_COEF) != NULL)
                     ? const_cast<StateVariableCoef*>
                     (ispcoefs_(ISPCoefs::PARA_DIFFUSION_COEF))
                     : &ChiParaCoef_),
     ChiPerpCoefPtr_((ispcoefs_(ISPCoefs::PERP_DIFFUSION_COEF) != NULL)
                     ? const_cast<StateVariableCoef*>
                     (ispcoefs_(ISPCoefs::PERP_DIFFUSION_COEF))
                     : &ChiPerpCoef_),
     ChiCoef_(ChiParaCoefPtr_,
              ChiPerpCoefPtr_, B3Coef_),
     nChiParaCoef_(niCoef_, *ChiParaCoefPtr_),
     nChiPerpCoef_(niCoef_, *ChiPerpCoefPtr_),
     nChiCoef_(niCoef_, ChiCoef_),
     ChiParaGF_(NULL),
     ChiPerpGF_(NULL),
     SGF_(NULL)
{
   if ( mpi_.Root() && logging_ > 1)
   {
      cout << "Constructing IonStaticPressureOp" << endl;
   }

   if (term_flag_ < 0)
   {
      // Set default terms
      term_flag_ = 1023;
   }
   if (vis_flag_ < 0)
   {
      // Set default visualization fields
      vis_flag_ = (logging_ > 1) ? 1023 : this->GetDefaultVisFlag();
   }

   // Time derivative term:  d(1.5 n_i T_i) / dt
   SetTimeDerivativeTerm(presCoef_);

   if (this->CheckTermFlag(DIFFUSION_TERM) &&
       this->CheckTermFlag(ADVECTION_TERM))
   {
      // Advection-Diffusion term: -Div(n_i Chi_i Grad T_i - 2.5 n_i v_i T_i)
      SetAdvectionDiffusionTerm(nChiCoef_, aniViCoef_,
                                &nChiParaCoef_, &nChiPerpCoef_);
   }
   else
   {
      if (this->CheckTermFlag(DIFFUSION_TERM))
      {
         // Diffusion term: -Div(n_i chi Grad T_i)
         SetDiffusionTerm(nChiCoef_);
      }

      if (this->CheckTermFlag(ADVECTION_TERM))
      {
         // Advection term: Div(2.5 n_i v_i T_i)
         SetAdvectionTerm(aniViCoef_/*, true*/);
      }
   }

   if (this->CheckTermFlag(SOURCE_TERM) &&
       ispcoefs_(ISPCoefs::SOURCE_COEF) != NULL)
   {
      // Source term from command line
      SetSourceTerm(const_cast<StateVariableCoef&>
                    (*ispcoefs_(ISPCoefs::SOURCE_COEF)));
   }

   if (this->CheckVisFlag(DIFFUSION_PARA_COEF))
   {
      ChiParaGF_ = new ParGridFunction(&fes_);
   }
   if (this->CheckVisFlag(DIFFUSION_PERP_COEF))
   {
      ChiPerpGF_ = new ParGridFunction(&fes_);
   }
   if (this->CheckVisFlag(SOURCE_COEF) &&
       ispcoefs_(ISPCoefs::SOURCE_COEF) != NULL)
   {
      SGF_ = new ParGridFunction(&fes_);
   }

   if ( mpi_.Root() && logging_ > 1)
   {
      cout << "Done constructing IonStaticPressureOp" << endl;
   }
}

DGTransportTDO::IonStaticPressureOp::~IonStaticPressureOp()
{
   delete ChiPerpGF_;
   delete ChiParaGF_;
   delete SGF_;
}

void DGTransportTDO::IonStaticPressureOp::SetTimeStep(double dt)
{
   if (mpi_.Root() && logging_)
   {
      cout << "Setting time step: " << dt << " in IonStaticPressureOp"
           << endl;
   }
   TransportOp::SetTimeStep(dt);
}

void DGTransportTDO::
IonStaticPressureOp::RegisterDataFields(DataCollection & dc)
{
   NLOperator::RegisterDataFields(dc);

   if (this->CheckVisFlag(DIFFUSION_PARA_COEF))
   {
      dc.RegisterField("n_i Chi_i Parallel",      ChiParaGF_);
   }
   if (this->CheckVisFlag(DIFFUSION_PERP_COEF))
   {
      dc.RegisterField("n_i Chi_i Perpendicular", ChiPerpGF_);
   }
   if (this->CheckVisFlag(SOURCE_COEF) && SGF_ != NULL)
   {
      dc.RegisterField("Ion Static Pressure Source",    SGF_);
   }
}

void DGTransportTDO::
IonStaticPressureOp::PrepareDataFields()
{
   if (this->CheckVisFlag(DIFFUSION_PARA_COEF))
   {
      if (ispcoefs_(ISPCoefs::PARA_DIFFUSION_COEF) != NULL)
      {
         ChiParaGF_->ProjectCoefficient
         (const_cast<StateVariableCoef&>
          (*ispcoefs_(ISPCoefs::PARA_DIFFUSION_COEF)));
      }
      else
      {
         ChiParaGF_->ProjectCoefficient(ChiParaCoef_);
      }
   }
   if (this->CheckVisFlag(DIFFUSION_PERP_COEF))
   {
      if (ispcoefs_(ISPCoefs::PERP_DIFFUSION_COEF) != NULL)
      {
         ChiPerpGF_->ProjectCoefficient
         (const_cast<StateVariableCoef&>
          (*ispcoefs_(ISPCoefs::PERP_DIFFUSION_COEF)));
      }
      else
      {
         ChiPerpGF_->ProjectCoefficient(ChiPerpCoef_);
      }
   }
   if (this->CheckVisFlag(SOURCE_COEF))
   {
      if (ispcoefs_(ISPCoefs::SOURCE_COEF) != NULL)
      {
         SGF_->ProjectCoefficient
         (const_cast<StateVariableCoef&>
          (*ispcoefs_(ISPCoefs::SOURCE_COEF)));
      }
   }
}

void DGTransportTDO::IonStaticPressureOp::Update()
{
   if (mpi_.Root() && logging_ > 1)
   {
      cout << "Entering DGTransportTDO::IonStaticPressureOp::Update" << endl;
   }

   NLOperator::Update();

   if (ChiParaGF_) { ChiParaGF_->Update(); }
   if (ChiPerpGF_) { ChiPerpGF_->Update(); }
   if (SGF_) { SGF_->Update(); }

   if (mpi_.Root() && logging_ > 1)
   {
      cout << "Leaving DGTransportTDO::IonStaticPressureOp::Update" << endl;
   }
}

DGTransportTDO::ElectronStaticPressureOp::
ElectronStaticPressureOp(const MPI_Session & mpi,
                         const DGParams & dg,
                         const PlasmaParams & plasma,
                         ParGridFunctionArray & yGF,
                         ParGridFunctionArray & kGF,
                         ParGridFunction & elOrdGF,
                         ParGridFunction & hGF,
                         const AdvectionDiffusionBC & bcs,
                         const CoupledBCs & cbcs,
                         const ESPCoefs & espcoefs,
                         const CmnCoefs & cmncoefs,
                         VectorCoefficient & B3Coef,
                         double ChiPerp,
                         int term_flag, int vis_flag,
                         int logging,
                         const string & log_prefix)
   : AdvTransportOp(mpi, dg, plasma, 4, "Electron Static Pressure",
                    "Electron Temperature", NULL, NULL, yGF, kGF, elOrdGF, hGF,
                    bcs, cbcs, cmncoefs, B3Coef, term_flag, vis_flag,
                    logging, log_prefix),
     espcoefs_(espcoefs),
     ChiPerpConst_(ChiPerp),
     presCoef_(z_i_, niCoef_, TeCoef_),
     aneViCoef_(neCoef_, viCoef_, 2.5, B3Coef_),
     ChiParaCoef_(plasma.z_i, lnLambda_, neCoef_, TeCoef_),
     ChiPerpCoef_(ChiPerpConst_),
     ChiParaCoefPtr_((espcoefs_(ESPCoefs::PARA_DIFFUSION_COEF) != NULL)
                     ? const_cast<StateVariableCoef*>
                     (espcoefs_(ESPCoefs::PARA_DIFFUSION_COEF))
                     : &ChiParaCoef_),
     ChiPerpCoefPtr_((espcoefs_(ESPCoefs::PERP_DIFFUSION_COEF) != NULL)
                     ? const_cast<StateVariableCoef*>
                     (espcoefs_(ESPCoefs::PERP_DIFFUSION_COEF))
                     : &ChiPerpCoef_),
     ChiCoef_(ChiParaCoefPtr_,
              ChiPerpCoefPtr_, B3Coef_),
     nChiParaCoef_(neCoef_, *ChiParaCoefPtr_),
     nChiPerpCoef_(neCoef_, *ChiPerpCoefPtr_),
     nChiCoef_(neCoef_, ChiCoef_),
     ChiParaGF_(NULL),
     ChiPerpGF_(NULL),
     SGF_(NULL)
{
   if ( mpi_.Root() && logging_ > 1)
   {
      cout << "Constructing ElectronStaticPressureOp" << endl;
   }

   if (term_flag_ < 0)
   {
      // Set default terms
      term_flag_ = 1023;
   }
   if (vis_flag_ < 0)
   {
      // Set default visualization fields
      vis_flag_ = (logging_ > 1) ? 1023 : this->GetDefaultVisFlag();
   }

   // Time derivative term:  d(1.5 z_i n_i T_e) / dt
   SetTimeDerivativeTerm(presCoef_);

   if (this->CheckTermFlag(DIFFUSION_TERM) &&
       this->CheckTermFlag(ADVECTION_TERM))
   {
      // Advection-Diffusion term: -Div(n_e Chi_e Grad T_e - 2.5 n_e v_i T_e)
      SetAdvectionDiffusionTerm(nChiCoef_, aneViCoef_,
                                &nChiParaCoef_, &nChiPerpCoef_);
   }
   else
   {
      if (this->CheckTermFlag(DIFFUSION_TERM))
      {
         // Diffusion term: -Div(n_e chi Grad T_e)
         SetDiffusionTerm(nChiCoef_);
      }

      if (this->CheckTermFlag(ADVECTION_TERM))
      {
         // Advection term: Div(2.5 n_e v_i T_e)
         SetAdvectionTerm(aneViCoef_/*, true*/);
      }
   }

   if (this->CheckTermFlag(SOURCE_TERM) &&
       espcoefs_(ESPCoefs::SOURCE_COEF) != NULL)
   {
      // Source term from command line
      SetSourceTerm(const_cast<StateVariableCoef&>
                    (*espcoefs_(ESPCoefs::SOURCE_COEF)));
   }

   if (this->CheckVisFlag(DIFFUSION_PARA_COEF))
   {
      ChiParaGF_ = new ParGridFunction(&fes_);
   }
   if (this->CheckVisFlag(DIFFUSION_PERP_COEF))
   {
      ChiPerpGF_ = new ParGridFunction(&fes_);
   }
   if (this->CheckVisFlag(SOURCE_COEF) &&
       espcoefs_(ESPCoefs::SOURCE_COEF) != NULL)
   {
      SGF_ = new ParGridFunction(&fes_);
   }

   if ( mpi_.Root() && logging_ > 1)
   {
      cout << "Done constructing ElectronStaticPressureOp" << endl;
   }
}

DGTransportTDO::ElectronStaticPressureOp::~ElectronStaticPressureOp()
{
   delete ChiPerpGF_;
   delete ChiParaGF_;
   delete SGF_;
}

void DGTransportTDO::ElectronStaticPressureOp::SetTimeStep(double dt)
{
   if (mpi_.Root() && logging_)
   {
      cout << "Setting time step: " << dt << " in ElectronStaticPressureOp"
           << endl;
   }
   TransportOp::SetTimeStep(dt);
}

void DGTransportTDO::
ElectronStaticPressureOp::RegisterDataFields(DataCollection & dc)
{
   NLOperator::RegisterDataFields(dc);

   if (this->CheckVisFlag(DIFFUSION_PARA_COEF))
   {
      dc.RegisterField("n_e Chi_e Parallel",      ChiParaGF_);
   }
   if (this->CheckVisFlag(DIFFUSION_PERP_COEF))
   {
      dc.RegisterField("n_e Chi_e Perpendicular", ChiPerpGF_);
   }
   if (this->CheckVisFlag(SOURCE_COEF) && SGF_ != NULL)
   {
      dc.RegisterField("Electron Static Pressure Source", SGF_);
   }
}

void DGTransportTDO::
ElectronStaticPressureOp::PrepareDataFields()
{
   if (this->CheckVisFlag(DIFFUSION_PARA_COEF))
   {
      if (espcoefs_(ESPCoefs::PARA_DIFFUSION_COEF) != NULL)
      {
         ChiParaGF_->ProjectCoefficient
         (const_cast<StateVariableCoef&>
          (*espcoefs_(ESPCoefs::PARA_DIFFUSION_COEF)));
      }
      else
      {
         ChiParaGF_->ProjectCoefficient(ChiParaCoef_);
      }
   }
   if (this->CheckVisFlag(DIFFUSION_PERP_COEF))
   {
      if (espcoefs_(ESPCoefs::PERP_DIFFUSION_COEF) != NULL)
      {
         ChiPerpGF_->ProjectCoefficient
         (const_cast<StateVariableCoef&>
          (*espcoefs_(ESPCoefs::PERP_DIFFUSION_COEF)));
      }
      else
      {
         ChiPerpGF_->ProjectCoefficient(ChiPerpCoef_);
      }
   }
   if (espcoefs_(ESPCoefs::SOURCE_COEF) != NULL)
   {
      SGF_->ProjectCoefficient
      (const_cast<StateVariableCoef&>
       (*espcoefs_(ESPCoefs::SOURCE_COEF)));
   }
}

void DGTransportTDO::ElectronStaticPressureOp::Update()
{
   if (mpi_.Root() && logging_ > 1)
   {
      cout << "Entering DGTransportTDO::ElectronStaticPressureOp::Update"
           << endl;
   }

   NLOperator::Update();

   if (ChiParaGF_) { ChiParaGF_->Update(); }
   if (ChiPerpGF_) { ChiPerpGF_->Update(); }
   if (SGF_) { SGF_->Update(); }

   if (mpi_.Root() && logging_ > 1)
   {
      cout << "Leaving DGTransportTDO::ElectronStaticPressureOp::Update"
           << endl;
   }
}

DGTransportTDO::TotalEnergyOp::
TotalEnergyOp(const MPI_Session & mpi, const DGParams & dg,
              const PlasmaParams & plasma, int index,
              const std::string &eqn_name,
              const std::string &field_name,
              ParFiniteElementSpace & vfes,
              ParFiniteElementSpace & h1_fes,
              ParGridFunctionArray & yGF,
              ParGridFunctionArray & kGF,
              ParGridFunction & elOrdGF,
              ParGridFunction & hGF,
              const AdvectionDiffusionBC & bcs,
              const CoupledBCs & cbcs,
              const CommonCoefs & common_coefs,
              VectorCoefficient & B3Coef,
              int term_flag, int vis_flag,
              int logging,
              const std::string & log_prefix)
   : AdvTransportOp(mpi, dg, plasma, index, eqn_name, field_name,
                    &vfes, &h1_fes, yGF, kGF, elOrdGF, hGF,
                    bcs, cbcs, common_coefs, B3Coef,
                    term_flag, vis_flag, logging, log_prefix),
     QiCoef_(plasma.z_i, plasma.m_i_kg, lnLambda_,
             niCoef_, TiCoef_, TeCoef_),
     kBCoef_(J_per_eV_),
     phiIZCoef_(13.6),
     kBphiIZCoef_(kBCoef_, phiIZCoef_),
     BSVCoef_(BxyCoef_)
{
   if ( mpi_.Root() && logging_ > 1)
   {
      cout << "Constructing TotalEnergyOp for the " << field_name << endl;
   }

   if (term_flag_ < 0)
   {
      // Set default terms
      term_flag_ = 1023;
   }
}

void DGTransportTDO::TotalEnergyOp::SetTime(double t)
{
   if (mpi_.Root() && logging_)
   {
      cout << "Setting time: " << t << " in TotalEnergyOp (" << eqn_name_
           << ")" << endl;
   }
   TransportOp::SetTime(t);

   QiCoef_.SetTime(t);
   kBCoef_.SetTime(t);
   phiIZCoef_.SetTime(t);
   kBphiIZCoef_.SetTime(t);
   BSVCoef_.SetTime(t);
}

void
DGTransportTDO::TotalEnergyOp::
SetKineticEnergyAdvectionTerm(StateVariableVecCoef &VCoef)
{
   if ( mpi_.Root() && logging_ > 0)
   {
      cout << eqn_name_ << ": Adding kinetic energy advection term" << endl;
   }

   // advectionCoef_ = &VCoef;

   ScalarVectorProductCoefficient * dtVCoef =
      new ScalarVectorProductCoefficient(dt_, VCoef);
   dtVCoefs_.Append(dtVCoef);

   dbfi_[ION_PARA_VELOCITY].Append(
      new ConservativeConvectionIntegrator(VCoef, 1.0));
   fbfi_[ION_PARA_VELOCITY].Append(
      new DGTraceIntegrator(VCoef, 1.0, -0.5));

   if (blf_[ION_PARA_VELOCITY] == NULL)
   {
      blf_[ION_PARA_VELOCITY] = new ParBilinearForm(&fes_);
   }

   blf_[ION_PARA_VELOCITY]->AddDomainIntegrator(
      new ConservativeConvectionIntegrator(*dtVCoef, 1.0));
   blf_[ION_PARA_VELOCITY]->AddInteriorFaceIntegrator(
      new DGTraceIntegrator(*dtVCoef, 1.0, -0.5));
}

DGTransportTDO::IonTotalEnergyOp::
IonTotalEnergyOp(const MPI_Session & mpi, const DGParams & dg,
                 const PlasmaParams & plasma,
                 ParFiniteElementSpace & vfes,
                 ParFiniteElementSpace & h1_fes,
                 ParGridFunctionArray & yGF,
                 ParGridFunctionArray & kGF,
                 ParGridFunction & elOrdGF,
                 ParGridFunction & hGF,
                 const AdvectionDiffusionBC & bcs,
                 const CoupledBCs & cbcs,
                 const ITECoefs & itecoefs,
                 const CmnCoefs & cmncoefs,
                 VectorCoefficient & B3Coef,
                 double ChiPerp,
                 int term_flag, int vis_flag,
                 int logging,
                 const std::string & log_prefix)
   : TotalEnergyOp(mpi, dg, plasma, 3, "Total Ion Energy", "Ion Temperature",
                   vfes, h1_fes, yGF, kGF, elOrdGF, hGF,
                   bcs, cbcs, cmncoefs, B3Coef,
                   term_flag, vis_flag, logging, log_prefix),
     itecoefs_(itecoefs),
     ChiPerpConst_(ChiPerp),
     totEnergyCoef_(plasma.m_i_kg,
                    niCoef_, viCoef_, TiCoef_),
     kinEnergyCoef_(plasma.m_i_kg, niCoef_, viCoef_),
     advFluxCoef_(plasma.m_i_kg,
                  niCoef_, viCoef_, TiCoef_, B3Coef),
     aniViCoef_(niCoef_, viCoef_, 2.5 * J_per_eV_, B3Coef_),
     ChiParaCoef_(plasma.z_i, plasma.m_i_kg, lnLambda_, niCoef_, TiCoef_),
     ChiPerpCoef_(ChiPerpConst_),
     ChiParaCoefPtr_((itecoefs_(ITECoefs::PARA_DIFFUSION_COEF) != NULL)
                     ? const_cast<StateVariableCoef*>
                     (itecoefs_(ITECoefs::PARA_DIFFUSION_COEF))
                     : &ChiParaCoef_),
     ChiPerpCoefPtr_((itecoefs_(ITECoefs::PERP_DIFFUSION_COEF) != NULL)
                     ? const_cast<StateVariableCoef*>
                     (itecoefs_(ITECoefs::PERP_DIFFUSION_COEF))
                     : &ChiPerpCoef_),
     ChiCoef_(ChiParaCoefPtr_,
              ChiPerpCoefPtr_, B3Coef_),
     nChiParaCoef_(niCoef_, *ChiParaCoefPtr_),
     nChiPerpCoef_(niCoef_, *ChiPerpCoefPtr_),
     nkChiParaCoef_(kBCoef_, nChiParaCoef_),
     nkChiPerpCoef_(kBCoef_, nChiPerpCoef_),
     nChiCoef_(niCoef_, ChiCoef_),
     nkChiCoef_(kBCoef_, nChiCoef_),
     keVCoef_(kinEnergyCoef_, BSVCoef_),
     SIZCoef_(m_n_kg_, m_i_kg_, vnAvgCoef_, TnCoef_, SizCoef_),
     SRCCoef_(m_i_kg_, viCoef_, SrcCoef_),
     SCXCoef_(m_i_kg_, vnAvgCoef_, viCoef_, ScxCoef_),
     ChiParaGF_(NULL),
     ChiPerpGF_(NULL),
     AdvGF_(NULL),
     SGF_(NULL),
     QiGF_(NULL),
     totEnergyGF_(NULL)
{
   if ( mpi_.Root() && logging_ > 1)
   {
      cout << "Constructing IonTotalEnergyOp" << endl;
   }

   if (vis_flag_ < 0)
   {
      // Set default visualization fields
      vis_flag_ = (logging_ > 1) ? 1023 : this->GetDefaultVisFlag();
   }

   // Time derivative term:  d(1.5 n T + 0.5 m n v^2) / dt
   SetTimeDerivativeTerm(totEnergyCoef_);

   if (this->CheckTermFlag(DIFFUSION_TERM) &&
       this->CheckTermFlag(ADVECTION_TERM))
   {
      // Advection-Diffusion term: -Div(n_i Chi_i Grad T_i - 2.5 n_i T_i v_i)
      SetAdvectionDiffusionTerm(nkChiCoef_, aniViCoef_,
                                &nkChiParaCoef_, &nkChiPerpCoef_);
   }
   else
   {
      if (this->CheckTermFlag(DIFFUSION_TERM))
      {
         // Diffusion term: -Div(n_i Chi_i Grad T_i)
         SetAnisotropicDiffusionTerm(nkChiCoef_,
                                     &nkChiParaCoef_, &nkChiPerpCoef_);
      }

      if (this->CheckTermFlag(ADVECTION_TERM))
      {
         // Advection term: Div(2.5 n_i T_i v_i)
         SetAdvectionTerm(aniViCoef_/*, true*/);
      }
   }
   if (this->CheckTermFlag(KE_ADVECTION_TERM))
   {
      // Advection term: Div(0.5 m_i n_i v_i^2 bHat)
      SetKineticEnergyAdvectionTerm(keVCoef_);
   }

   if (this->CheckTermFlag(ADVECTION_TERM) && bcs_.GetOutflowBCs().Size() > 0)
   {
      SetOutflowBdrTerm(advFluxCoef_, bcs_.GetOutflowBCs());
   }

   if (this->CheckTermFlag(EQUIPARTITION_SOURCE_TERM))
   {
      // Source term: Qi
      SetSourceTerm(QiCoef_, 1.0);
   }

   if (this->CheckTermFlag(IONIZATION_SOURCE_TERM))
   {
      // Source term: Siz * (3/2 m_i/m_n k_B T_n + 1/2 m_i v_n^2)
      SetSourceTerm(SIZCoef_, 1.0);
   }

   if (this->CheckTermFlag(RECOMBINATION_SINK_TERM))
   {
      // Sink term: Src * (1/2 m_i v_i^2)
      SetSourceTerm(SRCCoef_, -1.0);
   }

   if (this->CheckTermFlag(CHARGE_EXCHANGE_SOURCE_TERM))
   {
      // Source term: Scx * (1/2 m_i v_n^2 - 1/2 m_i v_i^2)
      SetSourceTerm(SCXCoef_, 1.0);
   }

   if (this->CheckTermFlag(SOURCE_TERM) &&
       itecoefs_(ITECoefs::SOURCE_COEF) != NULL)
   {
      // Source term from command line
      SetSourceTerm(const_cast<StateVariableCoef&>
                    (*itecoefs_(ITECoefs::SOURCE_COEF)));
   }

   if (this->CheckVisFlag(DIFFUSION_PARA_COEF))
   {
      ChiParaGF_ = new ParGridFunction(&fes_);
   }
   if (this->CheckVisFlag(DIFFUSION_PERP_COEF))
   {
      ChiPerpGF_ = new ParGridFunction(&fes_);
   }
   if (this->CheckVisFlag(ADVECTION_COEF))
   {
      AdvGF_ = new ParGridFunction(vfes_);
   }
   if (this->CheckVisFlag(SOURCE_COEF) &&
       itecoefs_(ITECoefs::SOURCE_COEF) != NULL)
   {
      SGF_ = new ParGridFunction(&fes_);
   }
   if (this->CheckVisFlag(IONIZATION_SOURCE_COEF))
   {
      SIZGF_ = new ParGridFunction(&fes_);
   }
   if (this->CheckVisFlag(RECOMBINATION_SINK_COEF))
   {
      SRCGF_ = new ParGridFunction(&fes_);
   }
   if (this->CheckVisFlag(CHARGE_EXCHANGE_SOURCE_COEF))
   {
      SCXGF_ = new ParGridFunction(&fes_);
   }
   if (this->CheckVisFlag(EQUIPARTITION_SOURCE_COEF))
   {
      QiGF_ = new ParGridFunction(&fes_);
   }
   if (this->CheckVisFlag(ION_TOTAL_ENERGY))
   {
      totEnergyGF_ = new ParGridFunction(&fes_);
   }
   if ( mpi_.Root() && logging_ > 1)
   {
      cout << "Done constructing IonTotalEnergyOp" << endl;
   }
}

DGTransportTDO::IonTotalEnergyOp::~IonTotalEnergyOp()
{
   delete ChiPerpGF_;
   delete ChiParaGF_;
   delete AdvGF_;
   delete SIZGF_;
   delete SRCGF_;
   delete SCXGF_;
   delete SGF_;
   delete QiGF_;
   delete totEnergyGF_;
}

void DGTransportTDO::IonTotalEnergyOp::SetTime(double t)
{
   if (mpi_.Root() && logging_)
   {
      cout << "Setting time: " << t << " in IonTotalEnergyOp" << endl;
   }
   TotalEnergyOp::SetTime(t);

   totEnergyCoef_.SetTime(t);
   kinEnergyCoef_.SetTime(t);
   advFluxCoef_.SetTime(t);
   aniViCoef_.SetTime(t);
   ChiParaCoef_.SetTime(t);
   ChiPerpCoef_.SetTime(t);
   ChiCoef_.SetTime(t);
   nChiParaCoef_.SetTime(t);
   nChiPerpCoef_.SetTime(t);
   nkChiParaCoef_.SetTime(t);
   nkChiPerpCoef_.SetTime(t);
   nChiCoef_.SetTime(t);
   nkChiCoef_.SetTime(t);
   keVCoef_.SetTime(t);
   SIZCoef_.SetTime(t);
   SRCCoef_.SetTime(t);
   SCXCoef_.SetTime(t);
}

void DGTransportTDO::IonTotalEnergyOp::Update()
{
   if (mpi_.Root() && logging_ > 1)
   {
      cout << "Entering DGTransportTDO::IonTotalEnergyOp::Update" << endl;
   }

   TotalEnergyOp::Update();

   if (ChiParaGF_   != NULL) { ChiParaGF_->Update(); }
   if (ChiPerpGF_   != NULL) { ChiPerpGF_->Update(); }
   if (AdvGF_       != NULL) { AdvGF_->Update(); }
   if (SIZGF_       != NULL) { SIZGF_->Update(); }
   if (SRCGF_       != NULL) { SRCGF_->Update(); }
   if (SCXGF_       != NULL) { SCXGF_->Update(); }
   if (SGF_         != NULL) { SGF_->Update(); }
   if (QiGF_        != NULL) { QiGF_->Update(); }
   if (totEnergyGF_ != NULL) { totEnergyGF_->Update(); }

   if (mpi_.Root() && logging_ > 1)
   {
      cout << "Leaving DGTransportTDO::IonTotalEnergyOp::Update" << endl;
   }
}

void DGTransportTDO::IonTotalEnergyOp::RegisterDataFields(DataCollection & dc)
{
   TotalEnergyOp::RegisterDataFields(dc);

   if (this->CheckVisFlag(DIFFUSION_PARA_COEF))
   {
      dc.RegisterField("Chi_i Parallel", ChiParaGF_);
   }
   if (this->CheckVisFlag(DIFFUSION_PERP_COEF))
   {
      dc.RegisterField("Chi_i Perpendicular", ChiPerpGF_);
   }
   if (this->CheckVisFlag(ADVECTION_COEF) && AdvGF_ != NULL)
   {
      dc.RegisterField("Ion Energy Advection", AdvGF_);
   }
   if (this->CheckVisFlag(IONIZATION_SOURCE_COEF) && SIZGF_ != NULL)
   {
      dc.RegisterField("Ion Energy Ionization Source", SIZGF_);
   }
   if (this->CheckVisFlag(RECOMBINATION_SINK_COEF) && SRCGF_ != NULL)
   {
      dc.RegisterField("Ion Energy Recombination Sink", SRCGF_);
   }
   if (this->CheckVisFlag(CHARGE_EXCHANGE_SOURCE_COEF) && SCXGF_ != NULL)
   {
      dc.RegisterField("Ion Energy Charge Exchange Source", SCXGF_);
   }
   if (this->CheckVisFlag(SOURCE_COEF) && SGF_ != NULL)
   {
      dc.RegisterField("Ion Energy Source", SGF_);
   }
   if (this->CheckVisFlag(EQUIPARTITION_SOURCE_COEF))
   {
      ostringstream oss;
      oss << eqn_name_ << " Qi";
      dc.RegisterField(oss.str(), QiGF_);
   }
   if (this->CheckVisFlag(ION_TOTAL_ENERGY))
   {
      ostringstream oss;
      oss << eqn_name_;
      dc.RegisterField(oss.str(), totEnergyGF_);
   }
}

void DGTransportTDO::IonTotalEnergyOp::PrepareDataFields()
{
   if (this->CheckVisFlag(DIFFUSION_PARA_COEF))
   {
      if (itecoefs_(ITECoefs::PARA_DIFFUSION_COEF) != NULL)
      {
         ChiParaGF_->ProjectCoefficient
         (const_cast<StateVariableCoef&>
          (*itecoefs_(ITECoefs::PARA_DIFFUSION_COEF)));
      }
      else
      {
         ChiParaGF_->ProjectCoefficient(ChiParaCoef_);
      }
   }
   if (this->CheckVisFlag(DIFFUSION_PERP_COEF))
   {
      if (itecoefs_(ITECoefs::PERP_DIFFUSION_COEF) != NULL)
      {
         ChiPerpGF_->ProjectCoefficient
         (const_cast<StateVariableCoef&>
          (*itecoefs_(ITECoefs::PERP_DIFFUSION_COEF)));
      }
      else
      {
         ChiPerpGF_->ProjectCoefficient(ChiPerpCoef_);
      }
   }
   if (this->CheckVisFlag(SOURCE_COEF))
   {
      if (itecoefs_(ITECoefs::SOURCE_COEF) != NULL)
      {
         SGF_->ProjectCoefficient
         (const_cast<StateVariableCoef&>
          (*itecoefs_(ITECoefs::SOURCE_COEF)));
      }
   }
   if (this->CheckVisFlag(IONIZATION_SOURCE_COEF))
   {
      if (this->CheckTermFlag(IONIZATION_SOURCE_TERM))
      {
         SIZGF_->ProjectCoefficient(SIZCoef_);
      }
      else
      {
         *SIZGF_ = 0.0;
      }
   }
   if (this->CheckVisFlag(RECOMBINATION_SINK_COEF))
   {
      if (this->CheckTermFlag(RECOMBINATION_SINK_TERM))
      {
         SRCGF_->ProjectCoefficient(SRCCoef_);
      }
      else
      {
         *SRCGF_ = 0.0;
      }
   }
   if (this->CheckVisFlag(CHARGE_EXCHANGE_SOURCE_COEF))
   {
      if (this->CheckTermFlag(CHARGE_EXCHANGE_SOURCE_TERM))
      {
         SCXGF_->ProjectCoefficient(SCXCoef_);
      }
      else
      {
         *SCXGF_ = 0.0;
      }
   }
   if (this->CheckVisFlag(EQUIPARTITION_SOURCE_COEF))
   {
      if (this->CheckTermFlag(EQUIPARTITION_SOURCE_TERM))
      {
         QiGF_->ProjectCoefficient(QiCoef_);
      }
      else
      {
         *QiGF_ = 0.0;
      }
   }
   if (this->CheckVisFlag(ADVECTION_COEF))
   {
      AdvGF_->ProjectCoefficient(aniViCoef_);
   }
   if (this->CheckVisFlag(ION_TOTAL_ENERGY))
   {
      totEnergyGF_->ProjectCoefficient(totEnergyCoef_);
   }
}

DGTransportTDO::ElectronTotalEnergyOp::
ElectronTotalEnergyOp(const MPI_Session & mpi, const DGParams & dg,
                      const PlasmaParams & plasma,
                      ParFiniteElementSpace & vfes,
                      ParFiniteElementSpace & h1_fes,
                      ParGridFunctionArray & yGF,
                      ParGridFunctionArray & kGF,
                      ParGridFunction & elOrdGF,
                      ParGridFunction & hGF,
                      const AdvectionDiffusionBC & bcs,
                      const CoupledBCs & cbcs,
                      const ETECoefs & etecoefs,
                      const CmnCoefs & cmncoefs,
                      VectorCoefficient & B3Coef,
                      double ChiPerp,
                      int term_flag, int vis_flag,
                      int logging,
                      const std::string & log_prefix)
   : TotalEnergyOp(mpi, dg, plasma, 4, "Total Electron Energy",
                   "Electron Temperature",
                   vfes, h1_fes, yGF, kGF, elOrdGF, hGF,
                   bcs, cbcs, cmncoefs, B3Coef,
                   term_flag, vis_flag, logging, log_prefix),
     etecoefs_(etecoefs),
     ChiPerpConst_(ChiPerp),
     totEnergyCoef_(plasma.z_i, me_kg_, niCoef_, viCoef_, TeCoef_),
     kinEnergyCoef_(plasma.z_i, me_kg_, niCoef_, viCoef_),
     advFluxCoef_(plasma.z_i, me_kg_,
                  niCoef_, viCoef_, TeCoef_, B3Coef),
     aneViCoef_(neCoef_, viCoef_, 2.5 * J_per_eV_, B3Coef_),
     ChiParaCoef_(plasma.z_i, lnLambda_, neCoef_, TeCoef_),
     ChiPerpCoef_(ChiPerpConst_),
     ChiParaCoefPtr_((etecoefs_(ETECoefs::PARA_DIFFUSION_COEF) != NULL)
                     ? const_cast<StateVariableCoef*>
                     (etecoefs_(ETECoefs::PARA_DIFFUSION_COEF))
                     : &ChiParaCoef_),
     ChiPerpCoefPtr_((etecoefs_(ETECoefs::PERP_DIFFUSION_COEF) != NULL)
                     ? const_cast<StateVariableCoef*>
                     (etecoefs_(ETECoefs::PERP_DIFFUSION_COEF))
                     : &ChiPerpCoef_),
     ChiCoef_(ChiParaCoefPtr_,
              ChiPerpCoefPtr_, B3Coef_),
     nChiParaCoef_(neCoef_, *ChiParaCoefPtr_),
     nChiPerpCoef_(neCoef_, *ChiPerpCoefPtr_),
     nkChiParaCoef_(kBCoef_, nChiParaCoef_),
     nkChiPerpCoef_(kBCoef_, nChiPerpCoef_),
     nChiCoef_(neCoef_, ChiCoef_),
     nkChiCoef_(kBCoef_, nChiCoef_),
     keVCoef_(kinEnergyCoef_, BSVCoef_),
     SIZCoef_(m_n_kg_, phiIZCoef_, vnAvgCoef_, TnCoef_, SizCoef_),
     SRCCoef_(viCoef_, SrcCoef_),
     ChiParaGF_(NULL),
     ChiPerpGF_(NULL),
     AdvGF_(NULL),
     SGF_(NULL),
     QiGF_(NULL),
     totEnergyGF_(NULL)
{
   if ( mpi_.Root() && logging_ > 1)
   {
      cout << "Constructing ElectronTotalEnergyOp" << endl;
   }

   if (vis_flag_ < 0)
   {
      // Set default visualization fields
      vis_flag_ = (logging_ > 1) ? 1023 : this->GetDefaultVisFlag();
   }

   // Time derivative term:  d(1.5 n T + 0.5 m n v^2) / dt
   SetTimeDerivativeTerm(totEnergyCoef_);

   if (this->CheckTermFlag(DIFFUSION_TERM) &&
       this->CheckTermFlag(ADVECTION_TERM))
   {
      // Advection-Diffusion term: -Div(n_e Chi_i Grad T_i - 2.5 n_e T_e v_i)
      SetAdvectionDiffusionTerm(nkChiCoef_, aneViCoef_,
                                &nkChiParaCoef_, &nkChiPerpCoef_);
   }
   else
   {
      if (this->CheckTermFlag(DIFFUSION_TERM))
      {
         // Diffusion term: -Div(n_e Chi_i Grad T_i)
         SetAnisotropicDiffusionTerm(nkChiCoef_,
                                     &nkChiParaCoef_, &nkChiPerpCoef_);
      }

      if (this->CheckTermFlag(ADVECTION_TERM))
      {
         // Advection term: Div(2.5 n_e T_e v_i)
         SetAdvectionTerm(aneViCoef_/*, true*/);
      }
   }
   if (this->CheckTermFlag(KE_ADVECTION_TERM))
   {
      // Advection term: Div(0.5 m_e n_e v_i^2 bHat)
      SetKineticEnergyAdvectionTerm(keVCoef_);
   }

   if (this->CheckTermFlag(ADVECTION_TERM) && bcs_.GetOutflowBCs().Size() > 0)
   {
      SetOutflowBdrTerm(advFluxCoef_, bcs_.GetOutflowBCs());
   }

   if (this->CheckTermFlag(EQUIPARTITION_SOURCE_TERM))
   {
      // Source term: -Qi
      SetSourceTerm(QiCoef_, -1.0);
   }

   if (this->CheckTermFlag(IONIZATION_SINK_TERM))
   {
      // Source term: Siz * (phi_iz - 3/2 m_e/m_n k_B T_n - 1/2 m_e v_n^2)
      SetSourceTerm(SIZCoef_, -1.0);
   }

   if (this->CheckTermFlag(RECOMBINATION_SINK_TERM))
   {
      // Sink term: Src * (1/2 m_e v_i^2)
      SetSourceTerm(SRCCoef_, -1.0);
   }

   if (this->CheckTermFlag(SOURCE_TERM) &&
       etecoefs_(ETECoefs::SOURCE_COEF) != NULL)
   {
      // Source term from command line
      SetSourceTerm(const_cast<StateVariableCoef&>
                    (*etecoefs_(ETECoefs::SOURCE_COEF)));
   }

   if (this->CheckVisFlag(DIFFUSION_PARA_COEF))
   {
      ChiParaGF_ = new ParGridFunction(&fes_);
   }
   if (this->CheckVisFlag(DIFFUSION_PERP_COEF))
   {
      ChiPerpGF_ = new ParGridFunction(&fes_);
   }
   if (this->CheckVisFlag(ADVECTION_COEF))
   {
      AdvGF_ = new ParGridFunction(vfes_);
   }
   if (this->CheckVisFlag(SOURCE_COEF) &&
       etecoefs_(ETECoefs::SOURCE_COEF) != NULL)
   {
      SGF_ = new ParGridFunction(&fes_);
   }
   if (this->CheckVisFlag(IONIZATION_SINK_COEF))
   {
      SIZGF_ = new ParGridFunction(&fes_);
   }
   if (this->CheckVisFlag(RECOMBINATION_SINK_COEF))
   {
      SRCGF_ = new ParGridFunction(&fes_);
   }
   if (this->CheckVisFlag(EQUIPARTITION_SOURCE_COEF))
   {
      QiGF_ = new ParGridFunction(&fes_);
   }
   if (this->CheckVisFlag(ELECTRON_TOTAL_ENERGY))
   {
      totEnergyGF_ = new ParGridFunction(&fes_);
   }
   if ( mpi_.Root() && logging_ > 1)
   {
      cout << "Done constructing ElectronTotalEnergyOp" << endl;
   }
}

DGTransportTDO::ElectronTotalEnergyOp::~ElectronTotalEnergyOp()
{
   delete ChiPerpGF_;
   delete ChiParaGF_;
   delete AdvGF_;
   delete SIZGF_;
   delete SRCGF_;
   delete SGF_;
   delete QiGF_;
   delete totEnergyGF_;
}

void DGTransportTDO::ElectronTotalEnergyOp::SetTime(double t)
{
   if (mpi_.Root() && logging_)
   {
      cout << "Setting time: " << t << " in ElectronTotalEnergyOp" << endl;
   }
   TotalEnergyOp::SetTime(t);

   totEnergyCoef_.SetTime(t);
   kinEnergyCoef_.SetTime(t);
   advFluxCoef_.SetTime(t);
   aneViCoef_.SetTime(t);
   ChiParaCoef_.SetTime(t);
   ChiPerpCoef_.SetTime(t);
   ChiCoef_.SetTime(t);
   nChiParaCoef_.SetTime(t);
   nChiPerpCoef_.SetTime(t);
   nkChiParaCoef_.SetTime(t);
   nkChiPerpCoef_.SetTime(t);
   nChiCoef_.SetTime(t);
   nkChiCoef_.SetTime(t);
   keVCoef_.SetTime(t);
   SIZCoef_.SetTime(t);
   SRCCoef_.SetTime(t);
}

void DGTransportTDO::ElectronTotalEnergyOp::Update()
{
   if (mpi_.Root() && logging_ > 1)
   {
      cout << "Entering DGTransportTDO::ElectronTotalEnergyOp::Update" << endl;
   }

   TotalEnergyOp::Update();

   if (ChiParaGF_   != NULL) { ChiParaGF_->Update(); }
   if (ChiPerpGF_   != NULL) { ChiPerpGF_->Update(); }
   if (AdvGF_       != NULL) { AdvGF_->Update(); }
   if (SIZGF_       != NULL) { SIZGF_->Update(); }
   if (SRCGF_       != NULL) { SRCGF_->Update(); }
   if (SGF_         != NULL) { SGF_->Update(); }
   if (QiGF_        != NULL) { QiGF_->Update(); }
   if (totEnergyGF_ != NULL) { totEnergyGF_->Update(); }

   if (mpi_.Root() && logging_ > 1)
   {
      cout << "Leaving DGTransportTDO::ElectronTotalEnergyOp::Update" << endl;
   }
}

void DGTransportTDO::ElectronTotalEnergyOp::RegisterDataFields(
   DataCollection & dc)
{
   TotalEnergyOp::RegisterDataFields(dc);

   if (this->CheckVisFlag(DIFFUSION_PARA_COEF))
   {
      dc.RegisterField("Chi_e Parallel", ChiParaGF_);
   }
   if (this->CheckVisFlag(DIFFUSION_PERP_COEF))
   {
      dc.RegisterField("Chi_e Perpendicular", ChiPerpGF_);
   }
   if (this->CheckVisFlag(ADVECTION_COEF) && AdvGF_ != NULL)
   {
      dc.RegisterField("Electron Energy Advection", AdvGF_);
   }
   if (this->CheckVisFlag(IONIZATION_SINK_COEF) && SIZGF_ != NULL)
   {
      dc.RegisterField("Electron Energy Ionization Sink", SIZGF_);
   }
   if (this->CheckVisFlag(RECOMBINATION_SINK_COEF) && SRCGF_ != NULL)
   {
      dc.RegisterField("Electron Energy Recombination Sink", SRCGF_);
   }
   if (this->CheckVisFlag(SOURCE_COEF) && SGF_ != NULL)
   {
      dc.RegisterField("Electron Energy Source", SGF_);
   }
   if (this->CheckVisFlag(EQUIPARTITION_SOURCE_COEF))
   {
      ostringstream oss;
      oss << eqn_name_ << " Qi";
      dc.RegisterField(oss.str(), QiGF_);
   }
   if (this->CheckVisFlag(ELECTRON_TOTAL_ENERGY))
   {
      ostringstream oss;
      oss << eqn_name_;
      dc.RegisterField(oss.str(), totEnergyGF_);
   }
}

void DGTransportTDO::ElectronTotalEnergyOp::PrepareDataFields()
{
   if (this->CheckVisFlag(DIFFUSION_PARA_COEF))
   {
      if (etecoefs_(ETECoefs::PARA_DIFFUSION_COEF) != NULL)
      {
         ChiParaGF_->ProjectCoefficient
         (const_cast<StateVariableCoef&>
          (*etecoefs_(ETECoefs::PARA_DIFFUSION_COEF)));
      }
      else
      {
         ChiParaGF_->ProjectCoefficient(ChiParaCoef_);
      }
   }
   if (this->CheckVisFlag(DIFFUSION_PERP_COEF))
   {
      if (etecoefs_(ETECoefs::PERP_DIFFUSION_COEF) != NULL)
      {
         ChiPerpGF_->ProjectCoefficient
         (const_cast<StateVariableCoef&>
          (*etecoefs_(ETECoefs::PERP_DIFFUSION_COEF)));
      }
      else
      {
         ChiPerpGF_->ProjectCoefficient(ChiPerpCoef_);
      }
   }
   if (this->CheckVisFlag(SOURCE_COEF))
   {
      if (etecoefs_(ETECoefs::SOURCE_COEF) != NULL)
      {
         SGF_->ProjectCoefficient
         (const_cast<StateVariableCoef&>
          (*etecoefs_(ETECoefs::SOURCE_COEF)));
      }
   }
   if (this->CheckVisFlag(IONIZATION_SINK_COEF))
   {
      if (this->CheckTermFlag(IONIZATION_SINK_TERM))
      {
         SIZGF_->ProjectCoefficient(SIZCoef_);
      }
      else
      {
         *SIZGF_ = 0.0;
      }
   }
   if (this->CheckVisFlag(RECOMBINATION_SINK_COEF))
   {
      if (this->CheckTermFlag(RECOMBINATION_SINK_TERM))
      {
         SRCGF_->ProjectCoefficient(SRCCoef_);
      }
      else
      {
         *SRCGF_ = 0.0;
      }
   }
   if (this->CheckVisFlag(EQUIPARTITION_SOURCE_COEF))
   {
      if (this->CheckTermFlag(EQUIPARTITION_SOURCE_TERM))
      {
         QiGF_->ProjectCoefficient(QiCoef_);
      }
      else
      {
         *QiGF_ = 0.0;
      }
   }
   if (this->CheckVisFlag(ADVECTION_COEF))
   {
      AdvGF_->ProjectCoefficient(aneViCoef_);
   }
   if (this->CheckVisFlag(ELECTRON_TOTAL_ENERGY))
   {
      totEnergyGF_->ProjectCoefficient(totEnergyCoef_);
   }
}

DGTransportTDO::DummyOp::DummyOp(const MPI_Session & mpi, const DGParams & dg,
                                 const PlasmaParams & plasma,
                                 ParGridFunctionArray & yGF,
                                 ParGridFunctionArray & kGF,
                                 ParGridFunction & elOrdGF,
                                 ParGridFunction & hGF,
                                 const AdvectionDiffusionBC & bcs,
                                 const CoupledBCs & cbcs,
                                 const CmnCoefs & cmncoefs,
                                 VectorCoefficient & B3Coef,
                                 int index,
                                 const string & eqn_name,
                                 const string & field_name,
                                 int term_flag, int vis_flag, int logging,
                                 const string & log_prefix)
   : TransportOp(mpi, dg, plasma, index, eqn_name, field_name,
                 NULL, NULL, yGF, kGF, elOrdGF, hGF,
                 bcs, cbcs, cmncoefs, B3Coef,
                 term_flag, vis_flag, logging, log_prefix)
{
   if ( mpi_.Root() && logging_ > 1)
   {
      cout << "Constructing DummyOp for equation " << index_ << endl;
   }

   dbfi_m_[index_].Append(new MassIntegrator);

   if (blf_[index_] == NULL)
   {
      blf_[index_] = new ParBilinearForm(&fes_);
   }
   blf_[index_]->AddDomainIntegrator(new MassIntegrator);

   if ( mpi_.Root() && logging_ > 1)
   {
      cout << "Done constructing DummyOp" << endl;
   }
}

void DGTransportTDO::DummyOp::Update()
{
   NLOperator::Update();
}

DGTransportTDO::VisualizationOp::VisualizationOp(const MPI_Session & mpi,
                                                 const DGParams & dg,
                                                 const PlasmaParams & plasma,
                                                 ParFiniteElementSpace & vfes,
                                                 ParGridFunctionArray & yGF,
                                                 ParGridFunctionArray & kGF,
                                                 ParGridFunction & elOrdGF,
                                                 ParGridFunction & hGF,
                                                 const AdvectionDiffusionBC & bcs,
                                                 const CoupledBCs & cbcs,
                                                 const CmnCoefs & cmncoefs,
                                                 VectorCoefficient & B3Coef,
                                                 int vis_flag,
                                                 int logging,
                                                 const string & log_prefix)
   : AdvTransportOp(mpi, dg, plasma, 5, "N/A", "N/A",
                    &vfes, NULL, yGF, kGF, elOrdGF, hGF,
                    bcs, cbcs, cmncoefs, B3Coef,
                    -1, vis_flag, logging, log_prefix),
     //     B3Coef_(const_cast<VectorCoefficient&>
     //        (*coefs(5).GetVectorCoefficient
     //         (CommonCoefs::MAGNETIC_FIELD_COEF))),
     BxyCoef_(B3Coef_),
     BzCoef_(B3Coef_),
     lnLambdaCoef_(TeCoef_, neCoef_, plasma.z_i),
     TauICoef_(plasma.z_i, plasma.m_i_kg, niCoef_, TiCoef_, lnLambdaCoef_),
     TauECoef_(plasma.z_i, niCoef_, TeCoef_, lnLambdaCoef_),
     SigmaIZCoef_(TeCoef_),
     SigmaRCCoef_(TeCoef_),
     SigmaCXCoef_(TiCoef_),
     // CsCoef_(plasma.m_i_kg, TiCoef_, TeCoef_),
     BxyGF_(NULL),
     BzGF_(NULL),
     lnLambdaGF_(),
     TauIGF_(NULL),
     TauEGF_(NULL),
     SigmaIZGF_(NULL),
     SigmaRCGF_(NULL),
     SigmaCXGF_(NULL),
     CsGF_(NULL)
{
   if ( mpi_.Root() && logging_ > 1)
   {
      cout << "Constructing VisualizationOp" << endl;
   }

   if (vis_flag_ < 0)
   {
      // Set default visualization fields
      vis_flag_ = (logging_ > 1) ? 1023 : this->GetDefaultVisFlag();
   }

   if (this->CheckVisFlag(B_POLOIDAL))
   {
      BxyGF_ = new ParGridFunction(vfes_);
   }
   if (this->CheckVisFlag(B_TOROIDAL))
   {
      BzGF_ = new ParGridFunction(&fes_);
   }
   if (this->CheckVisFlag(COULOMB_LOG))
   {
      lnLambdaGF_ = new ParGridFunction(&fes_);
   }
   if (this->CheckVisFlag(TAU_I))
   {
      TauIGF_ = new ParGridFunction(&fes_);
   }
   if (this->CheckVisFlag(TAU_E))
   {
      TauEGF_ = new ParGridFunction(&fes_);
   }
   if (this->CheckVisFlag(IONIZATION_RATE))
   {
      SigmaIZGF_ = new ParGridFunction(&fes_);
   }
   if (this->CheckVisFlag(RECOMBINATION_RATE))
   {
      SigmaRCGF_ = new ParGridFunction(&fes_);
   }
   if (this->CheckVisFlag(CHARGE_EXCHANGE_RATE))
   {
      SigmaCXGF_ = new ParGridFunction(&fes_);
   }
   if (this->CheckVisFlag(ION_SOUND_SPEED))
   {
      CsGF_ = new ParGridFunction(&fes_);
   }

   if ( mpi_.Root() && logging_ > 1)
   {
      cout << "Done constructing VisualizationOp" << endl;
   }
}

DGTransportTDO::VisualizationOp::~VisualizationOp()
{
   delete BxyGF_;
   delete BzGF_;
   delete lnLambdaGF_;
   delete TauIGF_;
   delete TauEGF_;
   delete SigmaIZGF_;
   delete SigmaRCGF_;
   delete SigmaCXGF_;
   delete CsGF_;
}

void DGTransportTDO::VisualizationOp::Update()
{
   if (mpi_.Root() && logging_ > 1)
   {
      cout << "Entering DGTransportTDO::VisualizationOp::Update" << endl;
   }

   AdvTransportOp::Update();

   if (BxyGF_      != NULL) { BxyGF_->Update(); }
   if (BzGF_       != NULL) { BzGF_->Update(); }
   if (lnLambdaGF_ != NULL) { lnLambdaGF_->Update(); }
   if (TauIGF_     != NULL) { TauIGF_->Update(); }
   if (TauEGF_     != NULL) { TauEGF_->Update(); }
   if (SigmaIZGF_  != NULL) { SigmaIZGF_->Update(); }
   if (SigmaRCGF_  != NULL) { SigmaRCGF_->Update(); }
   if (SigmaCXGF_  != NULL) { SigmaCXGF_->Update(); }
   if (CsGF_       != NULL) { CsGF_->Update(); }

   if (mpi_.Root() && logging_ > 1)
   {
      cout << "Leaving DGTransportTDO::VisualizationOp::Update" << endl;
   }
}

void DGTransportTDO::VisualizationOp::RegisterDataFields(DataCollection & dc)
{
   if (mpi_.Root() && logging_ > 1)
   {
      cout << "Entering VisualizationOp::RegisterDataFields with vis_flag: "
           << vis_flag_ << endl;
   }
   if (this->CheckVisFlag(B_POLOIDAL))
   {
      if (mpi_.Root() && logging_ > 1)
      {
         cout << "Registering B Poloidal" << endl;
      }
      dc.RegisterField("B Poloidal", BxyGF_);
   }
   if (this->CheckVisFlag(B_TOROIDAL))
   {
      dc.RegisterField("B Toroidal", BzGF_);
   }
   if (this->CheckVisFlag(COULOMB_LOG))
   {
      dc.RegisterField("Coulomb Logarithm", lnLambdaGF_);
   }
   if (this->CheckVisFlag(TAU_I))
   {
      dc.RegisterField("Ion Collision Time", TauIGF_);
   }
   if (this->CheckVisFlag(TAU_E))
   {
      dc.RegisterField("Electron Collision Time",TauEGF_);
   }
   if (this->CheckVisFlag(IONIZATION_RATE))
   {
      dc.RegisterField("Ionization Rate",SigmaIZGF_);
   }
   if (this->CheckVisFlag(RECOMBINATION_RATE))
   {
      dc.RegisterField("Recombination Rate",SigmaRCGF_);
   }
   if (this->CheckVisFlag(CHARGE_EXCHANGE_RATE))
   {
      dc.RegisterField("Charge Exchange Rate",SigmaCXGF_);
   }
   if (this->CheckVisFlag(ION_SOUND_SPEED))
   {
      dc.RegisterField("Ion Sound Speed", CsGF_);
   }
}

void DGTransportTDO::VisualizationOp::PrepareDataFields()
{
   if (this->CheckVisFlag(B_POLOIDAL))
   {
      BxyGF_->ProjectCoefficient(BxyCoef_);
   }
   if (this->CheckVisFlag(B_TOROIDAL))
   {
      BzGF_->ProjectCoefficient(BzCoef_);
   }
   if (this->CheckVisFlag(COULOMB_LOG))
   {
      lnLambdaGF_->ProjectCoefficient(lnLambdaCoef_);
   }
   if (this->CheckVisFlag(TAU_I))
   {
      TauIGF_->ProjectCoefficient(TauICoef_);
   }
   if (this->CheckVisFlag(TAU_E))
   {
      TauEGF_->ProjectCoefficient(TauECoef_);
   }
   if (this->CheckVisFlag(IONIZATION_RATE))
   {
      SigmaIZGF_->ProjectCoefficient(SigmaIZCoef_);
   }
   if (this->CheckVisFlag(RECOMBINATION_RATE))
   {
      SigmaRCGF_->ProjectCoefficient(SigmaRCCoef_);
   }
   if (this->CheckVisFlag(CHARGE_EXCHANGE_RATE))
   {
      SigmaCXGF_->ProjectCoefficient(SigmaCXCoef_);
   }
   if (this->CheckVisFlag(ION_SOUND_SPEED))
   {
      CsGF_->ProjectCoefficient(CsCoef_);
   }
}

/*
TransportSolver::TransportSolver(ODESolver * implicitSolver,
                                 ODESolver * explicitSolver,
                                 ParFiniteElementSpace & sfes,
                                 ParFiniteElementSpace & vfes,
                                 ParFiniteElementSpace & ffes,
                                 BlockVector & nBV,
                                 ParGridFunction & B,
                                 Array<int> & charges,
                                 Vector & masses)
   : impSolver_(implicitSolver),
     expSolver_(explicitSolver),
     sfes_(sfes),
     vfes_(vfes),
     ffes_(ffes),
     nBV_(nBV),
     B_(B),
     charges_(charges),
     masses_(masses),
     msDiff_(NULL)
{
   this->initDiffusion();
}

TransportSolver::~TransportSolver()
{
   delete msDiff_;
}

void TransportSolver::initDiffusion()
{
   msDiff_ = new MultiSpeciesDiffusion(sfes_, vfes_, nBV_, charges_, masses_);
}

void TransportSolver::Update()
{
   msDiff_->Update();
}

void TransportSolver::Step(Vector &x, double &t, double &dt)
{
   msDiff_->Assemble();
   impSolver_->Step(x, t, dt);
}
*/
/*
MultiSpeciesDiffusion::MultiSpeciesDiffusion(ParFiniteElementSpace & sfes,
                                             ParFiniteElementSpace & vfes,
                                             BlockVector & nBV,
                                             Array<int> & charges,
                                             Vector & masses)
   : sfes_(sfes),
     vfes_(vfes),
     nBV_(nBV),
     charges_(charges),
     masses_(masses)
{}

MultiSpeciesDiffusion::~MultiSpeciesDiffusion()
{}

void MultiSpeciesDiffusion::initCoefficients()
{}

void MultiSpeciesDiffusion::initBilinearForms()
{}

void MultiSpeciesDiffusion::Assemble()
{}

void MultiSpeciesDiffusion::Update()
{}

void MultiSpeciesDiffusion::ImplicitSolve(const double dt,
                                          const Vector &x, Vector &y)
{}
*/
DiffusionTDO::DiffusionTDO(ParFiniteElementSpace &fes,
                           ParFiniteElementSpace &dfes,
                           ParFiniteElementSpace &vfes,
                           MatrixCoefficient & nuCoef,
                           double dg_sigma,
                           double dg_kappa)
   : TimeDependentOperator(vfes.GetTrueVSize()),
     dim_(vfes.GetFE(0)->GetDim()),
     dt_(0.0),
     dg_sigma_(dg_sigma),
     dg_kappa_(dg_kappa),
     fes_(fes),
     // dfes_(dfes),
     vfes_(vfes),
     m_(&fes_),
     d_(&fes_),
     rhs_(&fes_),
     x_(&vfes_),
     M_(NULL),
     D_(NULL),
     RHS_(fes_.GetTrueVSize()),
     X_(fes_.GetTrueVSize()),
     solver_(NULL),
     amg_(NULL),
     nuCoef_(nuCoef),
     dtNuCoef_(0.0, nuCoef_)
{
   m_.AddDomainIntegrator(new MassIntegrator);
   m_.AddDomainIntegrator(new DiffusionIntegrator(dtNuCoef_));
   m_.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(dtNuCoef_,
                                                          dg_sigma_,
                                                          dg_kappa_));
   m_.AddBdrFaceIntegrator(new DGDiffusionIntegrator(dtNuCoef_,
                                                     dg_sigma_, dg_kappa_));
   d_.AddDomainIntegrator(new DiffusionIntegrator(nuCoef_));
   d_.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(nuCoef_,
                                                          dg_sigma_,
                                                          dg_kappa_));
   d_.AddBdrFaceIntegrator(new DGDiffusionIntegrator(nuCoef_,
                                                     dg_sigma_, dg_kappa_));
   d_.Assemble();
   d_.Finalize();
   D_ = d_.ParallelAssemble();
}

void DiffusionTDO::ImplicitSolve(const double dt, const Vector &x, Vector &y)
{
   y = 0.0;

   this->initSolver(dt);

   for (int d=0; d<dim_; d++)
   {
      ParGridFunction xd(&fes_, &(x.GetData()[(d+1) * fes_.GetVSize()]));
      ParGridFunction yd(&fes_, &(y.GetData()[(d+1) * fes_.GetVSize()]));

      D_->Mult(xd, rhs_);
      rhs_ *= -1.0;
      rhs_.ParallelAssemble(RHS_);

      X_ = 0.0;
      solver_->Mult(RHS_, X_);

      yd = X_;
   }
}

void DiffusionTDO::initSolver(double dt)
{
   bool newM = false;
   if (fabs(dt - dt_) > 1e-4 * dt)
   {
      dt_ = dt;
      dtNuCoef_.SetAConst(dt);
      m_.Assemble(0);
      m_.Finalize(0);
      if (M_ != NULL)
      {
         delete M_;
      }
      M_ = m_.ParallelAssemble();
      newM = true;
   }

   if (amg_ == NULL || newM)
   {
      if (amg_ != NULL) { delete amg_; }
      amg_ = new HypreBoomerAMG(*M_);
   }
   if (solver_ == NULL || newM)
   {
      if (solver_ != NULL) { delete solver_; }
      if (dg_sigma_ == -1.0)
      {
         HyprePCG * pcg = new HyprePCG(*M_);
         pcg->SetTol(1e-12);
         pcg->SetMaxIter(200);
         pcg->SetPrintLevel(0);
         pcg->SetPreconditioner(*amg_);
         solver_ = pcg;
      }
      else
      {
         HypreGMRES * gmres = new HypreGMRES(*M_);
         gmres->SetTol(1e-12);
         gmres->SetMaxIter(200);
         gmres->SetKDim(10);
         gmres->SetPrintLevel(0);
         gmres->SetPreconditioner(*amg_);
         solver_ = gmres;
      }

   }
}

// Implementation of class FE_Evolution
AdvectionTDO::AdvectionTDO(ParFiniteElementSpace &vfes,
                           Operator &A, SparseMatrix &Aflux, int num_equation,
                           double specific_heat_ratio)
   : TimeDependentOperator(A.Height()),
     dim_(vfes.GetFE(0)->GetDim()),
     num_equation_(num_equation),
     specific_heat_ratio_(specific_heat_ratio),
     vfes_(vfes),
     A_(A),
     Aflux_(Aflux),
     Me_inv_(vfes.GetFE(0)->GetDof(), vfes.GetFE(0)->GetDof(), vfes.GetNE()),
     state_(num_equation_),
     f_(num_equation_, dim_),
     flux_(vfes.GetNDofs(), dim_, num_equation_),
     z_(A.Height())
{
   // Standard local assembly and inversion for energy mass matrices.
   const int dof = vfes_.GetFE(0)->GetDof();
   DenseMatrix Me(dof);
   DenseMatrixInverse inv(&Me);
   MassIntegrator mi;
   for (int i = 0; i < vfes_.GetNE(); i++)
   {
      mi.AssembleElementMatrix(*vfes_.GetFE(i),
                               *vfes_.GetElementTransformation(i), Me);
      inv.Factor();
      inv.GetInverseMatrix(Me_inv_(i));
   }
}

void AdvectionTDO::Mult(const Vector &x, Vector &y) const
{
   // 0. Reset wavespeed computation before operator application.
   max_char_speed_ = 0.;

   // 1. Create the vector z with the face terms -<F.n(u), [w]>.
   A_.Mult(x, z_);

   // 2. Add the element terms.
   // i.  computing the flux approximately as a grid function by interpolating
   //     at the solution nodes.
   // ii. multiplying this grid function by a (constant) mixed bilinear form for
   //     each of the num_equation, computing (F(u), grad(w)) for each equation.

   DenseMatrix xmat(x.GetData(), vfes_.GetNDofs(), num_equation_);
   GetFlux(xmat, flux_);

   for (int k = 0; k < num_equation_; k++)
   {
      Vector fk(flux_(k).GetData(), dim_ * vfes_.GetNDofs());
      Vector zk(z_.GetData() + k * vfes_.GetNDofs(), vfes_.GetNDofs());
      Aflux_.AddMult(fk, zk);
   }

   // 3. Multiply element-wise by the inverse mass matrices.
   Vector zval;
   Array<int> vdofs;
   const int dof = vfes_.GetFE(0)->GetDof();
   DenseMatrix zmat, ymat(dof, num_equation_);

   for (int i = 0; i < vfes_.GetNE(); i++)
   {
      // Return the vdofs ordered byNODES
      vfes_.GetElementVDofs(i, vdofs);
      z_.GetSubVector(vdofs, zval);
      zmat.UseExternalData(zval.GetData(), dof, num_equation_);
      mfem::Mult(Me_inv_(i), zmat, ymat);
      y.SetSubVector(vdofs, ymat.GetData());
   }
}

// Physicality check (at end)
bool StateIsPhysical(const Vector &state, const int dim,
                     const double specific_heat_ratio);

// Pressure (EOS) computation
inline double ComputePressure(const Vector &state, int dim,
                              double specific_heat_ratio)
{
   const double den = state(0);
   const Vector den_vel(state.GetData() + 1, dim);
   const double den_energy = state(1 + dim);

   double den_vel2 = 0;
   for (int d = 0; d < dim; d++) { den_vel2 += den_vel(d) * den_vel(d); }
   den_vel2 /= den;

   return (specific_heat_ratio - 1.0) * (den_energy - 0.5 * den_vel2);
}

// Compute the vector flux F(u)
void ComputeFlux(const Vector &state, int dim, double specific_heat_ratio,
                 DenseMatrix &flux)
{
   const double den = state(0);
   const Vector den_vel(state.GetData() + 1, dim);
   const double den_energy = state(1 + dim);

   MFEM_ASSERT(StateIsPhysical(state, dim, specific_heat_ratio), "");

   const double pres = ComputePressure(state, dim, specific_heat_ratio);

   for (int d = 0; d < dim; d++)
   {
      flux(0, d) = den_vel(d);
      for (int i = 0; i < dim; i++)
      {
         flux(1+i, d) = den_vel(i) * den_vel(d) / den;
      }
      flux(1+d, d) += pres;
   }

   const double H = (den_energy + pres) / den;
   for (int d = 0; d < dim; d++)
   {
      flux(1+dim, d) = den_vel(d) * H;
   }
}

// Compute the scalar F(u).n
void ComputeFluxDotN(const Vector &state, const Vector &nor,
                     double specific_heat_ratio,
                     Vector &fluxN)
{
   // NOTE: nor in general is not a unit normal
   const int dim = nor.Size();
   const double den = state(0);
   const Vector den_vel(state.GetData() + 1, dim);
   const double den_energy = state(1 + dim);

   MFEM_ASSERT(StateIsPhysical(state, dim, specific_heat_ratio), "");

   const double pres = ComputePressure(state, dim, specific_heat_ratio);

   double den_velN = 0;
   for (int d = 0; d < dim; d++) { den_velN += den_vel(d) * nor(d); }

   fluxN(0) = den_velN;
   for (int d = 0; d < dim; d++)
   {
      fluxN(1+d) = den_velN * den_vel(d) / den + pres * nor(d);
   }

   const double H = (den_energy + pres) / den;
   fluxN(1 + dim) = den_velN * H;
}

// Compute the maximum characteristic speed.
inline double ComputeMaxCharSpeed(const Vector &state,
                                  int dim, double specific_heat_ratio)
{
   const double den = state(0);
   const Vector den_vel(state.GetData() + 1, dim);

   double den_vel2 = 0;
   for (int d = 0; d < dim; d++) { den_vel2 += den_vel(d) * den_vel(d); }
   den_vel2 /= den;

   const double pres = ComputePressure(state, dim, specific_heat_ratio);
   const double sound = sqrt(specific_heat_ratio * pres / den);
   const double vel = sqrt(den_vel2 / den);

   return vel + sound;
}

// Compute the flux at solution nodes.
void AdvectionTDO::GetFlux(const DenseMatrix &x, DenseTensor &flux) const
{
   const int dof = flux.SizeI();
   const int dim = flux.SizeJ();

   for (int i = 0; i < dof; i++)
   {
      for (int k = 0; k < num_equation_; k++) { state_(k) = x(i, k); }
      ComputeFlux(state_, dim, specific_heat_ratio_, f_);

      for (int d = 0; d < dim; d++)
      {
         for (int k = 0; k < num_equation_; k++)
         {
            flux(i, d, k) = f_(k, d);
         }
      }

      // Update max char speed
      const double mcs = ComputeMaxCharSpeed(state_, dim, specific_heat_ratio_);
      if (mcs > max_char_speed_) { max_char_speed_ = mcs; }
   }
}

// Implementation of class RiemannSolver
RiemannSolver::RiemannSolver(int num_equation, double specific_heat_ratio) :
   num_equation_(num_equation),
   specific_heat_ratio_(specific_heat_ratio),
   flux1_(num_equation),
   flux2_(num_equation) { }

double RiemannSolver::Eval(const Vector &state1, const Vector &state2,
                           const Vector &nor, Vector &flux)
{
   // NOTE: nor in general is not a unit normal
   const int dim = nor.Size();

   MFEM_ASSERT(StateIsPhysical(state1, dim, specific_heat_ratio_), "");
   MFEM_ASSERT(StateIsPhysical(state2, dim, specific_heat_ratio_), "");

   const double maxE1 = ComputeMaxCharSpeed(state1, dim, specific_heat_ratio_);
   const double maxE2 = ComputeMaxCharSpeed(state2, dim, specific_heat_ratio_);

   const double maxE = max(maxE1, maxE2);

   ComputeFluxDotN(state1, nor, specific_heat_ratio_, flux1_);
   ComputeFluxDotN(state2, nor, specific_heat_ratio_, flux2_);

   double normag = 0;
   for (int i = 0; i < dim; i++)
   {
      normag += nor(i) * nor(i);
   }
   normag = sqrt(normag);

   for (int i = 0; i < num_equation_; i++)
   {
      flux(i) = 0.5 * (flux1_(i) + flux2_(i))
                - 0.5 * maxE * (state2(i) - state1(i)) * normag;
   }

   return maxE;
}

// Implementation of class DomainIntegrator
DomainIntegrator::DomainIntegrator(const int dim, int num_equation)
   : flux_(num_equation, dim) { }

void DomainIntegrator::AssembleElementMatrix2(const FiniteElement &trial_fe,
                                              const FiniteElement &test_fe,
                                              ElementTransformation &Tr,
                                              DenseMatrix &elmat)
{
   // Assemble the form (vec(v), grad(w))

   // Trial space = vector L2 space (mesh dim)
   // Test space  = scalar L2 space

   const int dof_trial = trial_fe.GetDof();
   const int dof_test = test_fe.GetDof();
   const int dim = trial_fe.GetDim();

   shape_.SetSize(dof_trial);
   dshapedr_.SetSize(dof_test, dim);
   dshapedx_.SetSize(dof_test, dim);

   elmat.SetSize(dof_test, dof_trial * dim);
   elmat = 0.0;

   const int maxorder = max(trial_fe.GetOrder(), test_fe.GetOrder());
   const int intorder = 2 * maxorder;
   const IntegrationRule *ir = &IntRules.Get(trial_fe.GetGeomType(), intorder);

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      // Calculate the shape functions
      trial_fe.CalcShape(ip, shape_);
      shape_ *= ip.weight;

      // Compute the physical gradients of the test functions
      Tr.SetIntPoint(&ip);
      test_fe.CalcDShape(ip, dshapedr_);
      Mult(dshapedr_, Tr.AdjugateJacobian(), dshapedx_);

      for (int d = 0; d < dim; d++)
      {
         for (int j = 0; j < dof_test; j++)
         {
            for (int k = 0; k < dof_trial; k++)
            {
               elmat(j, k + d * dof_trial) += shape_(k) * dshapedx_(j, d);
            }
         }
      }
   }
}

// Implementation of class FaceIntegrator
FaceIntegrator::FaceIntegrator(RiemannSolver &rsolver, const int dim,
                               const int num_equation) :
   num_equation_(num_equation),
   max_char_speed_(0.0),
   rsolver_(rsolver),
   funval1_(num_equation_),
   funval2_(num_equation_),
   nor_(dim),
   fluxN_(num_equation_) { }

void FaceIntegrator::AssembleFaceVector(const FiniteElement &el1,
                                        const FiniteElement &el2,
                                        FaceElementTransformations &Tr,
                                        const Vector &elfun, Vector &elvect)
{
   // Compute the term <F.n(u),[w]> on the interior faces.
   const int dof1 = el1.GetDof();
   const int dof2 = el2.GetDof();

   shape1_.SetSize(dof1);
   shape2_.SetSize(dof2);

   elvect.SetSize((dof1 + dof2) * num_equation_);
   elvect = 0.0;

   DenseMatrix elfun1_mat(elfun.GetData(), dof1, num_equation_);
   DenseMatrix elfun2_mat(elfun.GetData() + dof1 * num_equation_, dof2,
                          num_equation_);

   DenseMatrix elvect1_mat(elvect.GetData(), dof1, num_equation_);
   DenseMatrix elvect2_mat(elvect.GetData() + dof1 * num_equation_, dof2,
                           num_equation_);

   // Integration order calculation from DGTraceIntegrator
   int intorder;
   if (Tr.Elem2No >= 0)
      intorder = (min(Tr.Elem1->OrderW(), Tr.Elem2->OrderW()) +
                  2*max(el1.GetOrder(), el2.GetOrder()));
   else
   {
      intorder = Tr.Elem1->OrderW() + 2*el1.GetOrder();
   }
   if (el1.Space() == FunctionSpace::Pk)
   {
      intorder++;
   }
   const IntegrationRule *ir = &IntRules.Get(Tr.FaceGeom, intorder);

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.Loc1.Transform(ip, eip1_);
      Tr.Loc2.Transform(ip, eip2_);

      // Calculate basis functions on both elements at the face
      el1.CalcShape(eip1_, shape1_);
      el2.CalcShape(eip2_, shape2_);

      // Interpolate elfun at the point
      elfun1_mat.MultTranspose(shape1_, funval1_);
      elfun2_mat.MultTranspose(shape2_, funval2_);

      Tr.Face->SetIntPoint(&ip);

      // Get the normal vector and the flux on the face
      CalcOrtho(Tr.Face->Jacobian(), nor_);
      const double mcs = rsolver_.Eval(funval1_, funval2_, nor_, fluxN_);

      // Update max char speed
      if (mcs > max_char_speed_) { max_char_speed_ = mcs; }

      fluxN_ *= ip.weight;
      for (int k = 0; k < num_equation_; k++)
      {
         for (int s = 0; s < dof1; s++)
         {
            elvect1_mat(s, k) -= fluxN_(k) * shape1_(s);
         }
         for (int s = 0; s < dof2; s++)
         {
            elvect2_mat(s, k) += fluxN_(k) * shape2_(s);
         }
      }
   }
}
/*
void DGAnisoDiffusionIntegrator::AssembleFaceMatrix(
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Trans, DenseMatrix &elmat)
{
   int dim, ndof1, ndof2, ndofs;
   bool kappa_is_nonzero = (kappa != 0.);
   double w, wq = 0.0;

   dim = el1.GetDim();
   ndof1 = el1.GetDof();

   nor.SetSize(dim);
   nh.SetSize(dim);
   ni.SetSize(dim);
   adjJ.SetSize(dim);
   if (MQ)
   {
      mq.SetSize(dim);
   }

   shape1.SetSize(ndof1);
   dshape1.SetSize(ndof1, dim);
   dshape1dn.SetSize(ndof1);
   if (Trans.Elem2No >= 0)
   {
      ndof2 = el2.GetDof();
      shape2.SetSize(ndof2);
      dshape2.SetSize(ndof2, dim);
      dshape2dn.SetSize(ndof2);
   }
   else
   {
      ndof2 = 0;
   }

   ndofs = ndof1 + ndof2;
   elmat.SetSize(ndofs);
   elmat = 0.0;
   if (kappa_is_nonzero)
   {
      jmat.SetSize(ndofs);
      jmat = 0.;
   }

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // a simple choice for the integration order; is this OK?
      int order;
      if (ndof2)
      {
         order = 2*max(el1.GetOrder(), el2.GetOrder());
      }
      else
      {
         order = 2*el1.GetOrder();
      }
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
   }

   // assemble: < {(Q \nabla u).n},[v] >      --> elmat
   //           kappa < {h^{-1} Q} [u],[v] >  --> jmat
   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);

      // Access the neighboring elements' integration points
      // Note: eip2 will only contain valid data if Elem2 exists
      const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();
      const IntegrationPoint &eip2 = Trans.GetElement2IntPoint();

      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Jacobian(), nor);
      }

      el1.CalcShape(eip1, shape1);
      el1.CalcDShape(eip1, dshape1);
      w = ip.weight/Trans.Elem1->Weight();
      if (ndof2)
      {
         w /= 2;
      }

      nh.Set(w, nor);
      MQ->Eval(mq, *Trans.Elem1, eip1);
      mq.MultTranspose(nh, ni);

      CalcAdjugate(Trans.Elem1->Jacobian(), adjJ);
      adjJ.Mult(ni, nh);
      if (kappa_is_nonzero)
      {
         wq = nh * nor * CotTheta->Eval(*Trans.Elem1, eip1);
      }
      // Note: in the jump term, we use 1/h1 = |nor|/det(J1) which is
      // independent of Loc1 and always gives the size of element 1 in
      // direction perpendicular to the face. Indeed, for linear transformation
      //     |nor|=measure(face)/measure(ref. face),
      //   det(J1)=measure(element)/measure(ref. element),
      // and the ratios measure(ref. element)/measure(ref. face) are
      // compatible for all element/face pairs.
      // For example: meas(ref. tetrahedron)/meas(ref. triangle) = 1/3, and
      // for any tetrahedron vol(tet)=(1/3)*height*area(base).
      // For interior faces: q_e/h_e=(q1/h1+q2/h2)/2.

      dshape1.Mult(nh, dshape1dn);
      for (int i = 0; i < ndof1; i++)
         for (int j = 0; j < ndof1; j++)
         {
            elmat(i, j) += shape1(i) * dshape1dn(j);
         }

      if (ndof2)
      {
         el2.CalcShape(eip2, shape2);
         el2.CalcDShape(eip2, dshape2);
         w = ip.weight/2/Trans.Elem2->Weight();

         nh.Set(w, nor);
         MQ->Eval(mq, *Trans.Elem2, eip2);
         mq.MultTranspose(nh, ni);

         CalcAdjugate(Trans.Elem2->Jacobian(), adjJ);
         adjJ.Mult(ni, nh);
         if (kappa_is_nonzero)
         {
            wq += nh * nor * CotTheta->Eval(*Trans.Elem2, eip2);
         }

         dshape2.Mult(nh, dshape2dn);

         for (int i = 0; i < ndof1; i++)
            for (int j = 0; j < ndof2; j++)
            {
               elmat(i, ndof1 + j) += shape1(i) * dshape2dn(j);
            }

         for (int i = 0; i < ndof2; i++)
            for (int j = 0; j < ndof1; j++)
            {
               elmat(ndof1 + i, j) -= shape2(i) * dshape1dn(j);
            }

         for (int i = 0; i < ndof2; i++)
            for (int j = 0; j < ndof2; j++)
            {
               elmat(ndof1 + i, ndof1 + j) -= shape2(i) * dshape2dn(j);
            }
      }

      if (kappa_is_nonzero)
      {
         // only assemble the lower triangular part of jmat
         wq *= kappa * q1 * q1 / q0;
         for (int i = 0; i < ndof1; i++)
         {
            const double wsi = wq*shape1(i);
            for (int j = 0; j <= i; j++)
            {
               jmat(i, j) += wsi * shape1(j);
            }
         }
         if (ndof2)
         {
            for (int i = 0; i < ndof2; i++)
            {
               const int i2 = ndof1 + i;
               const double wsi = wq*shape2(i);
               for (int j = 0; j < ndof1; j++)
               {
                  jmat(i2, j) -= wsi * shape1(j);
               }
               for (int j = 0; j <= i; j++)
               {
                  jmat(i2, ndof1 + j) += wsi * shape2(j);
               }
            }
         }
      }
   }

   // elmat := -elmat + sigma*elmat^t + jmat
   if (kappa_is_nonzero)
   {
      for (int i = 0; i < ndofs; i++)
      {
         for (int j = 0; j < i; j++)
         {
            double aij = elmat(i,j), aji = elmat(j,i), mij = jmat(i,j);
            elmat(i,j) = sigma*aji - aij + mij;
            elmat(j,i) = sigma*aij - aji + mij;
         }
         elmat(i,i) = (sigma - 1.)*elmat(i,i) + jmat(i,i);
      }
   }
   else
   {
      for (int i = 0; i < ndofs; i++)
      {
         for (int j = 0; j < i; j++)
         {
            double aij = elmat(i,j), aji = elmat(j,i);
            elmat(i,j) = sigma*aji - aij;
            elmat(j,i) = sigma*aij - aji;
         }
         elmat(i,i) *= (sigma - 1.);
      }
   }
}
*/
double DGAdvDiffIntegrator::ComputeUpwindingParam(double epsilon,
                                                  double betaMag)
{
   // Diffusion dominated
   if (lambda * epsilon >= betaMag) { return 0.1; }

   // Advection dominated
   if (lambda * epsilon < 1e-5 * betaMag) { return 0.8; }

   // Transitional region
   return 0.1 - 0.14 * log10(lambda * epsilon / betaMag);
}

void DGAdvDiffIntegrator::AssembleFaceMatrix(const FiniteElement &el1,
                                             const FiniteElement &el2,
                                             FaceElementTransformations &Trans,
                                             DenseMatrix &elmat)
{
   int dim, ndof1, ndof2;

   double w, a00, a01, a10, q1, q2, t1, t2, t, b1n, nQ1n, nQ2n,
          alpha1, alpha2, bnrm;

   dim = el1.GetDim();
   Vector nor(dim);
   Vector B1(dim), B2(dim), B(dim);
   DenseMatrix Q1(dim), Q2(dim);
   Vector nQ1(dim), nQ2(dim);

   MFEM_VERIFY(Trans.Elem2No >= 0, "Designed for interior faces")

   ndof1 = el1.GetDof();
   ndof2 = el2.GetDof();

#ifdef MFEM_THREAD_SAFE
   Vector shape1(ndof1);
   Vector shape2(ndof2);
   DenseMatrix dshape1(ndof1, dim);
   DenseMatrix dshape2(ndof2, dim);
   Vector nQdshape1(ndof1);
   Vector nQdshape2(nodf2);
#else
   shape1.SetSize(ndof1);
   shape2.SetSize(ndof2);
   dshape1.SetSize(ndof1, dim);
   dshape2.SetSize(ndof2, dim);
   nQdshape1.SetSize(ndof1);
   nQdshape2.SetSize(ndof2);
#endif

   elmat.SetSize(ndof1 + ndof2);
   elmat = 0.0;

   Vector x(dim);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order;
      // Assuming order(u)==order(mesh)
      if (Trans.Elem2No >= 0)
         order = (0*min(Trans.Elem1->OrderW(), Trans.Elem2->OrderW()) +
                  2*max(el1.GetOrder(), el2.GetOrder()));
      else
      {
         order = Trans.Elem1->OrderW() + 2*el1.GetOrder();
      }
      if (el1.Space() == FunctionSpace::Pk)
      {
         order++;
      }
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
      if (Trans.Elem2No < 0 && false)
      {
         mfem::out << "DGTrace order " << order
                   << ", num pts = " << ir->GetNPoints() << std::endl;
      }
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);

      {
         Trans.Transform(ip, x);
      }

      double vol1 = Trans.Elem1->Weight();
      double vol2 = Trans.Elem2->Weight();

      // Access the neighboring elements' integration points
      // Note: eip2 will only contain valid data if Elem2 exists
      const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();
      const IntegrationPoint &eip2 = Trans.GetElement2IntPoint();

      el1.CalcShape(eip1, shape1);
      el2.CalcShape(eip2, shape2);

      el1.CalcPhysDShape(*Trans.Elem1, dshape1);
      el2.CalcPhysDShape(*Trans.Elem2, dshape2);

      if (Q)
      {
         q1 = Q->Eval(*Trans.Elem1, eip1);
         q2 = Q->Eval(*Trans.Elem2, eip2);
      }
      else if (MQ)
      {
         MQ->Eval(Q1, *Trans.Elem1, eip1);
         MQ->Eval(Q2, *Trans.Elem2, eip2);

         double qPara1 = 0.0, qPara2 = 0.0;
         double qPerp1 = 0.0, qPerp2 = 0.0;
         if (QPara)
         {
            qPara1 = QPara->Eval(*Trans.Elem1, eip1);
            qPara2 = QPara->Eval(*Trans.Elem2, eip2);
         }
         if (QPerp)
         {
            qPerp1 = QPerp->Eval(*Trans.Elem1, eip1);
            qPerp2 = QPerp->Eval(*Trans.Elem2, eip2);
         }
         q1 = std::max(qPara1, qPerp1);
         q2 = std::max(qPara2, qPerp2);
      }

      beta->Eval(B1, *Trans.Elem1, eip1);
      beta->Eval(B2, *Trans.Elem2, eip2);

      // beta should be continuous but it may not be
      add(0.5, B1, 0.5, B2, B);

      bnrm = sqrt(B * B);

      t1 = ComputeUpwindingParam(q1, bnrm);
      t2 = ComputeUpwindingParam(q2, bnrm);

      // tau should be continuous but just in case...
      t = 0.5 * (t1 + t2);

      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Jacobian(), nor);
      }

      if (MQ)
      {
         Q1.MultTranspose(nor, nQ1);
         Q2.MultTranspose(nor, nQ2);
      }
      else
      {
         nQ1.Set(q1, nor);
         nQ2.Set(q2, nor);
      }

      nQ1n = q1 * (nor * nor) / vol1;
      nQ2n = q2 * (nor * nor) / vol2;

      b1n = B * nor;

      double pen = 0.5 * kappa1 * (nQ1n + nQ2n) + kappa2 * fabs(b1n);

      alpha1 = 0.5 * (1.0 + copysign(t, b1n));
      alpha2 = 1.0 - alpha1;

      dshape1.Mult(nQ1, nQdshape1);
      dshape2.Mult(nQ2, nQdshape2);

      w = ip.weight;

      a00 = pen + (0.5 * sigma + alpha1 * (1.0 - sigma)) * b1n;
      a10 = -alpha1;
      a01 = alpha1 * sigma;
      for (int i = 0; i < ndof1; i++)
         for (int j = 0; j < ndof1; j++)
         {
            elmat(i, j) += w * (a00 * shape1(j) * shape1(i) +
                                a10 * nQdshape1(j) * shape1(i));
         }
      if (sigma != 0.0)
      {
         for (int i = 0; i < ndof1; i++)
            for (int j = 0; j < ndof1; j++)
            {
               elmat(i, j) += w * a01 * shape1(j) * nQdshape1(i);
            }
      }

      a00 = -pen + (0.5 * sigma - alpha1 - alpha2 * sigma) * b1n;
      a10 = alpha1;
      a01 = alpha2 * sigma;
      for (int i = 0; i < ndof2; i++)
         for (int j = 0; j < ndof1; j++)
         {
            elmat(ndof1+i, j) += w * (a00 * shape1(j) * shape2(i) +
                                      a10 * nQdshape1(j) * shape2(i));
         }
      if (sigma != 0.0)
      {
         for (int i = 0; i < ndof2; i++)
            for (int j = 0; j < ndof1; j++)
            {
               elmat(ndof1+i, j) += w * a01 * shape1(j) * nQdshape2(i);
            }
      }

      a00 = -pen + (-0.5 * sigma + alpha2 + alpha1 * sigma) * b1n;
      a10 = -alpha2;
      a01 = -alpha1 * sigma;
      for (int i = 0; i < ndof1; i++)
         for (int j = 0; j < ndof2; j++)
         {
            elmat(i, ndof1+j) += w * (a00 * shape2(j) * shape1(i) +
                                      a10 * nQdshape2(j) * shape1(i));
         }
      if (sigma != 0.0)
      {
         for (int i = 0; i < ndof1; i++)
            for (int j = 0; j < ndof2; j++)
            {
               elmat(i, ndof1+j) += w * a01 * shape2(j) * nQdshape1(i);
            }
      }

      a00 = pen + (-0.5 * sigma - alpha2 * (1.0 - sigma)) * b1n;
      a10 = alpha2;
      a01 = -alpha2 * sigma;
      for (int i = 0; i < ndof2; i++)
         for (int j = 0; j < ndof2; j++)
         {
            elmat(ndof1+i, ndof1+j) += w * (a00 * shape2(j) * shape2(i) +
                                            a10 * nQdshape2(j) * shape2(i));
         }
      if (sigma != 0.0)
      {
         for (int i = 0; i < ndof2; i++)
            for (int j = 0; j < ndof2; j++)
            {
               elmat(ndof1+i, ndof1+j) += w * a01 * shape2(j) * nQdshape2(i);
            }
      }
   }
}

void
DGAdvDiffBdrIntegrator::AssembleFaceMatrix(const FiniteElement &el1,
                                           const FiniteElement &el2,
                                           FaceElementTransformations &Trans,
                                           DenseMatrix &elmat)
{
   int dim, ndof1;

   double w, a00, a01, a10, q1, b1n, nQ1n;

   dim = el1.GetDim();
   Vector nor(dim);
   Vector B1(dim);
   DenseMatrix Q1(dim);
   Vector nQ1(dim);

   MFEM_VERIFY(Trans.Elem2No < 0, "Designed for bdr faces")

   ndof1 = el1.GetDof();

#ifdef MFEM_THREAD_SAFE
   Vector shape1(ndof1);
   DenseMatrix dshape1(ndof1, dim);
   Vector nQdshape1(ndof1);
#else
   shape1.SetSize(ndof1);
   dshape1.SetSize(ndof1, dim);
   nQdshape1.SetSize(ndof1);
#endif

   elmat.SetSize(ndof1);
   elmat = 0.0;

   Vector x(dim);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order;
      // Assuming order(u)==order(mesh)
      order = Trans.Elem1->OrderW() + 2*el1.GetOrder();

      if (el1.Space() == FunctionSpace::Pk)
      {
         order++;
      }
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
      if (Trans.Elem2No < 0 && false)
      {
         mfem::out << "DGTrace order " << order
                   << ", num pts = " << ir->GetNPoints() << std::endl;
      }
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);

      {
         Trans.Transform(ip, x);
      }

      double vol1 = Trans.Elem1->Weight();

      // Access the neighboring elements' integration points
      // Note: eip2 will only contain valid data if Elem2 exists
      const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();

      el1.CalcShape(eip1, shape1);

      el1.CalcPhysDShape(*Trans.Elem1, dshape1);

      if (Q)
      {
         q1 = Q->Eval(*Trans.Elem1, eip1);
      }
      else if (MQ)
      {
         MQ->Eval(Q1, *Trans.Elem1, eip1);

         double qPara1 = 0.0, qPerp1 = 0.0;
         if (QPara)
         {
            qPara1 = QPara->Eval(*Trans.Elem1, eip1);
         }
         if (QPerp)
         {
            qPerp1 = QPerp->Eval(*Trans.Elem1, eip1);
         }
         q1 = std::max(qPara1, qPerp1);
      }

      beta->Eval(B1, *Trans.Elem1, eip1);

      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Jacobian(), nor);
      }

      if (MQ)
      {
         Q1.MultTranspose(nor, nQ1);
      }
      else
      {
         nQ1.Set(q1, nor);
      }

      nQ1n = q1 * (nor * nor) / vol1;

      b1n = B1 * nor;

      double pen = kappa1 * nQ1n + kappa2 * fabs(b1n);

      dshape1.Mult(nQ1, nQdshape1);

      w = ip.weight;

      a00 = pen;
      a10 = -1.0;
      a01 = sigma;
      for (int i = 0; i < ndof1; i++)
         for (int j = 0; j < ndof1; j++)
         {
            elmat(i, j) += w * (a00 * shape1(j) * shape1(i) +
                                a10 * nQdshape1(j) * shape1(i));
         }
      if (sigma != 0.0)
      {
         for (int i = 0; i < ndof1; i++)
            for (int j = 0; j < ndof1; j++)
            {
               elmat(i, j) += w * a01 * shape1(j) * nQdshape1(i);
            }
      }

   }
}

void
DGAdvDiffDirichletLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el,
   FaceElementTransformations &Tr,
   Vector &elvect)
{
   int dim, ndof;
   bool kappa_is_nonzero = (kappa1 != 0.);
   double w, bn, q, u;

   dim = el.GetDim();
   ndof = el.GetDof();

   nor.SetSize(dim);
   nh.SetSize(dim);
   ni.SetSize(dim);
   adjJ.SetSize(dim);
   if (MQ)
   {
      mq.SetSize(dim);
   }
   vb.SetSize(dim);

   shape.SetSize(ndof);
   dshape.SetSize(ndof, dim);
   dshape_dn.SetSize(ndof);

   elvect.SetSize(ndof);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // a simple choice for the integration order; is this OK?
      int order = 2*el.GetOrder();
      ir = &IntRules.Get(Tr.GetGeometryType(), order);
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring element
      Tr.SetAllIntPoints(&ip);

      // Access the neighboring element's integration point
      const IntegrationPoint &eip = Tr.GetElement1IntPoint();

      if (dim == 1)
      {
         nor(0) = 2*eip.x - 1.0;
      }
      else
      {
         CalcOrtho(Tr.Jacobian(), nor);
      }

      el.CalcShape(eip, shape);
      el.CalcDShape(eip, dshape);

      // compute uD through the face transformation
      u = uD->Eval(Tr, ip);
      w = ip.weight * u / Tr.Elem1->Weight();
      if (Q)
      {
         q = Q->Eval(*Tr.Elem1, eip);
         w *= q;
         ni.Set(w, nor);
      }
      else
      {
         nh.Set(w, nor);
         MQ->Eval(mq, *Tr.Elem1, eip);
         mq.MultTranspose(nh, ni);

         double qPara = 0.0, qPerp = 0.0;
         if (QPara)
         {
            qPara = QPara->Eval(*Tr.Elem1, eip);
         }
         if (QPerp)
         {
            qPerp = QPerp->Eval(*Tr.Elem1, eip);
         }
         q = std::max(qPara, qPerp);
      }
      beta->Eval(vb, *Tr.Elem1, eip);
      bn = vb * nor;

      CalcAdjugate(Tr.Elem1->Jacobian(), adjJ);
      adjJ.Mult(ni, nh);

      dshape.Mult(nh, dshape_dn);
      elvect.Add(sigma, dshape_dn);

      if (kappa_is_nonzero)
      {
         elvect.Add(kappa1*q*w*(nor*nor), shape);
      }
      if (bn < 0.0)
      {
         elvect.Add(-ip.weight * bn * u, shape);

      }
   }
}

void
DGAnisoDiffIntegrator::AssembleFaceMatrix(const FiniteElement &el1,
                                          const FiniteElement &el2,
                                          FaceElementTransformations &Trans,
                                          DenseMatrix &elmat)
{
   int dim, ndof1, ndof2;

   double w, a00, a01, a10, q1, q2, nQ1n, nQ2n;

   dim = el1.GetDim();
   Vector nor(dim);
   DenseMatrix Q1(dim), Q2(dim);
   Vector nQ1(dim), nQ2(dim);

   MFEM_VERIFY(Trans.Elem2No >= 0, "Designed for interior faces")

   ndof1 = el1.GetDof();
   ndof2 = el2.GetDof();

#ifdef MFEM_THREAD_SAFE
   Vector shape1(ndof1);
   Vector shape2(ndof2);
   DenseMatrix dshape1(ndof1, dim);
   DenseMatrix dshape2(ndof2, dim);
   Vector nQdshape1(ndof1);
   Vector nQdshape2(nodf2);
#else
   shape1.SetSize(ndof1);
   shape2.SetSize(ndof2);
   dshape1.SetSize(ndof1, dim);
   dshape2.SetSize(ndof2, dim);
   nQdshape1.SetSize(ndof1);
   nQdshape2.SetSize(ndof2);
#endif

   elmat.SetSize(ndof1 + ndof2);
   elmat = 0.0;

   Vector x(dim);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order;
      // Assuming order(u)==order(mesh)
      if (Trans.Elem2No >= 0)
         order = (0*min(Trans.Elem1->OrderW(), Trans.Elem2->OrderW()) +
                  2*max(el1.GetOrder(), el2.GetOrder()));
      else
      {
         order = Trans.Elem1->OrderW() + 2*el1.GetOrder();
      }
      if (el1.Space() == FunctionSpace::Pk)
      {
         order++;
      }
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
      if (Trans.Elem2No < 0 && false)
      {
         mfem::out << "DGTrace order " << order
                   << ", num pts = " << ir->GetNPoints() << std::endl;
      }
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);

      {
         Trans.Transform(ip, x);
      }

      double vol1 = Trans.Elem1->Weight();
      double vol2 = Trans.Elem2->Weight();

      // Access the neighboring elements' integration points
      // Note: eip2 will only contain valid data if Elem2 exists
      const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();
      const IntegrationPoint &eip2 = Trans.GetElement2IntPoint();

      el1.CalcShape(eip1, shape1);
      el2.CalcShape(eip2, shape2);

      el1.CalcPhysDShape(*Trans.Elem1, dshape1);
      el2.CalcPhysDShape(*Trans.Elem2, dshape2);

      if (MQ)
      {
         MQ->Eval(Q1, *Trans.Elem1, eip1);
         MQ->Eval(Q2, *Trans.Elem2, eip2);

         double qPara1 = 0.0, qPara2 = 0.0;
         double qPerp1 = 0.0, qPerp2 = 0.0;
         if (QPara)
         {
            qPara1 = QPara->Eval(*Trans.Elem1, eip1);
            qPara2 = QPara->Eval(*Trans.Elem2, eip2);
         }
         if (QPerp)
         {
            qPerp1 = QPerp->Eval(*Trans.Elem1, eip1);
            qPerp2 = QPerp->Eval(*Trans.Elem2, eip2);
         }
         q1 = std::max(qPara1, qPerp1);
         q2 = std::max(qPara2, qPerp2);
      }

      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Jacobian(), nor);
      }

      if (MQ)
      {
         Q1.MultTranspose(nor, nQ1);
         Q2.MultTranspose(nor, nQ2);
      }
      else
      {
         nQ1.Set(q1, nor);
         nQ2.Set(q2, nor);
      }

      nQ1n = q1 * (nor * nor) / vol1;
      nQ2n = q2 * (nor * nor) / vol2;

      double pen = 0.5 * kappa * (nQ1n + nQ2n);

      dshape1.Mult(nQ1, nQdshape1);
      dshape2.Mult(nQ2, nQdshape2);

      w = ip.weight;

      a00 = pen;
      a10 = -0.5;
      a01 = 0.5 * sigma;
      for (int i = 0; i < ndof1; i++)
         for (int j = 0; j < ndof1; j++)
         {
            elmat(i, j) += w * (a00 * shape1(j) * shape1(i) +
                                a10 * nQdshape1(j) * shape1(i));
         }
      if (sigma != 0.0)
      {
         for (int i = 0; i < ndof1; i++)
            for (int j = 0; j < ndof1; j++)
            {
               elmat(i, j) += w * a01 * shape1(j) * nQdshape1(i);
            }
      }

      a00 = -pen;
      a10 = 0.5;
      a01 = 0.5 * sigma;
      for (int i = 0; i < ndof2; i++)
         for (int j = 0; j < ndof1; j++)
         {
            elmat(ndof1+i, j) += w * (a00 * shape1(j) * shape2(i) +
                                      a10 * nQdshape1(j) * shape2(i));
         }
      if (sigma != 0.0)
      {
         for (int i = 0; i < ndof2; i++)
            for (int j = 0; j < ndof1; j++)
            {
               elmat(ndof1+i, j) += w * a01 * shape1(j) * nQdshape2(i);
            }
      }

      a00 = -pen;
      a10 = -0.5;
      a01 = -0.5 * sigma;
      for (int i = 0; i < ndof1; i++)
         for (int j = 0; j < ndof2; j++)
         {
            elmat(i, ndof1+j) += w * (a00 * shape2(j) * shape1(i) +
                                      a10 * nQdshape2(j) * shape1(i));
         }
      if (sigma != 0.0)
      {
         for (int i = 0; i < ndof1; i++)
            for (int j = 0; j < ndof2; j++)
            {
               elmat(i, ndof1+j) += w * a01 * shape2(j) * nQdshape1(i);
            }
      }

      a00 = pen;
      a10 = 0.5;
      a01 = -0.5 * sigma;
      for (int i = 0; i < ndof2; i++)
         for (int j = 0; j < ndof2; j++)
         {
            elmat(ndof1+i, ndof1+j) += w * (a00 * shape2(j) * shape2(i) +
                                            a10 * nQdshape2(j) * shape2(i));
         }
      if (sigma != 0.0)
      {
         for (int i = 0; i < ndof2; i++)
            for (int j = 0; j < ndof2; j++)
            {
               elmat(ndof1+i, ndof1+j) += w * a01 * shape2(j) * nQdshape2(i);
            }
      }
   }
}

void
DGAnisoDiffBdrIntegrator::AssembleFaceMatrix(const FiniteElement &el1,
                                             const FiniteElement &el2,
                                             FaceElementTransformations &Trans,
                                             DenseMatrix &elmat)
{
   int dim, ndof1;

   double w, a00, a01, a10, q1, nQ1n;

   dim = el1.GetDim();
   Vector nor(dim);
   DenseMatrix Q1(dim);
   Vector nQ1(dim);

   MFEM_VERIFY(Trans.Elem2No < 0, "Designed for bdr faces")

   ndof1 = el1.GetDof();

#ifdef MFEM_THREAD_SAFE
   Vector shape1(ndof1);
   DenseMatrix dshape1(ndof1, dim);
   Vector nQdshape1(ndof1);
#else
   shape1.SetSize(ndof1);
   dshape1.SetSize(ndof1, dim);
   nQdshape1.SetSize(ndof1);
#endif

   elmat.SetSize(ndof1);
   elmat = 0.0;

   Vector x(dim);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order;
      // Assuming order(u)==order(mesh)
      order = Trans.Elem1->OrderW() + 2*el1.GetOrder();

      if (el1.Space() == FunctionSpace::Pk)
      {
         order++;
      }
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
      if (Trans.Elem2No < 0 && false)
      {
         mfem::out << "DGTrace order " << order
                   << ", num pts = " << ir->GetNPoints() << std::endl;
      }
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);

      {
         Trans.Transform(ip, x);
      }

      double vol1 = Trans.Elem1->Weight();

      // Access the neighboring elements' integration points
      // Note: eip2 will only contain valid data if Elem2 exists
      const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();

      el1.CalcShape(eip1, shape1);

      el1.CalcPhysDShape(*Trans.Elem1, dshape1);

      if (MQ)
      {
         MQ->Eval(Q1, *Trans.Elem1, eip1);

         double qPara1 = 0.0, qPerp1 = 0.0;
         if (QPara)
         {
            qPara1 = QPara->Eval(*Trans.Elem1, eip1);
         }
         if (QPerp)
         {
            qPerp1 = QPerp->Eval(*Trans.Elem1, eip1);
         }
         q1 = std::max(qPara1, qPerp1);
      }

      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Jacobian(), nor);
      }

      if (MQ)
      {
         Q1.MultTranspose(nor, nQ1);
      }
      else
      {
         nQ1.Set(q1, nor);
      }

      nQ1n = q1 * (nor * nor) / vol1;

      double pen = kappa * nQ1n;

      dshape1.Mult(nQ1, nQdshape1);

      w = ip.weight;

      a00 = pen;
      a10 = -1.0;
      a01 = sigma;
      for (int i = 0; i < ndof1; i++)
         for (int j = 0; j < ndof1; j++)
         {
            elmat(i, j) += w * (a00 * shape1(j) * shape1(i) +
                                a10 * nQdshape1(j) * shape1(i));
         }
      if (sigma != 0.0)
      {
         for (int i = 0; i < ndof1; i++)
            for (int j = 0; j < ndof1; j++)
            {
               elmat(i, j) += w * a01 * shape1(j) * nQdshape1(i);
            }
      }

   }
}

void
DGAnisoDiffDirichletLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el,
   FaceElementTransformations &Tr,
   Vector &elvect)
{
   int dim, ndof;
   bool kappa_is_nonzero = (kappa != 0.);
   double w, q, u;

   dim = el.GetDim();
   ndof = el.GetDof();

   nor.SetSize(dim);
   nh.SetSize(dim);
   ni.SetSize(dim);
   adjJ.SetSize(dim);
   if (MQ)
   {
      mq.SetSize(dim);
   }
   vb.SetSize(dim);

   shape.SetSize(ndof);
   dshape.SetSize(ndof, dim);
   dshape_dn.SetSize(ndof);

   elvect.SetSize(ndof);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // a simple choice for the integration order; is this OK?
      int order = 2*el.GetOrder();
      ir = &IntRules.Get(Tr.GetGeometryType(), order);
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring element
      Tr.SetAllIntPoints(&ip);

      // Access the neighboring element's integration point
      const IntegrationPoint &eip = Tr.GetElement1IntPoint();

      if (dim == 1)
      {
         nor(0) = 2*eip.x - 1.0;
      }
      else
      {
         CalcOrtho(Tr.Jacobian(), nor);
      }

      el.CalcShape(eip, shape);
      el.CalcDShape(eip, dshape);

      // compute uD through the face transformation
      u = uD->Eval(Tr, ip);
      w = ip.weight * u / Tr.Elem1->Weight();
      if (MQ)
      {
         nh.Set(w, nor);
         MQ->Eval(mq, *Tr.Elem1, eip);
         mq.MultTranspose(nh, ni);

         double qPara = 0.0, qPerp = 0.0;
         if (QPara)
         {
            qPara = QPara->Eval(*Tr.Elem1, eip);
         }
         if (QPerp)
         {
            qPerp = QPerp->Eval(*Tr.Elem1, eip);
         }
         q = std::max(qPara, qPerp);
      }

      CalcAdjugate(Tr.Elem1->Jacobian(), adjJ);
      adjJ.Mult(ni, nh);

      dshape.Mult(nh, dshape_dn);
      elvect.Add(sigma, dshape_dn);

      if (kappa_is_nonzero)
      {
         elvect.Add(kappa*q*w*(nor*nor), shape);
      }
   }
}

// Check that the state is physical - enabled in debug mode
bool StateIsPhysical(const Vector &state, int dim,
                     double specific_heat_ratio)
{
   const double den = state(0);
   const Vector den_vel(state.GetData() + 1, dim);
   const double den_energy = state(1 + dim);

   if (den < 0)
   {
      cout << "Negative density: ";
      for (int i = 0; i < state.Size(); i++)
      {
         cout << state(i) << " ";
      }
      cout << endl;
      return false;
   }
   if (den_energy <= 0)
   {
      cout << "Negative energy: ";
      for (int i = 0; i < state.Size(); i++)
      {
         cout << state(i) << " ";
      }
      cout << endl;
      return false;
   }

   double den_vel2 = 0;
   for (int i = 0; i < dim; i++) { den_vel2 += den_vel(i) * den_vel(i); }
   den_vel2 /= den;

   const double pres = (specific_heat_ratio - 1.0) *
                       (den_energy - 0.5 * den_vel2);

   if (pres <= 0)
   {
      cout << "Negative pressure: " << pres << ", state: ";
      for (int i = 0; i < state.Size(); i++)
      {
         cout << state(i) << " ";
      }
      cout << endl;
      return false;
   }
   return true;
}

} // namespace transport

} // namespace plasma

} // namespace mfem

#endif // MFEM_USE_MPI
