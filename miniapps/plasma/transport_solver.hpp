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

#ifndef MFEM_TRANSPORT_SOLVER
#define MFEM_TRANSPORT_SOLVER

#include "../common/fem_extras.hpp"
#include "../common/pfem_extras.hpp"
#include "../common/mesh_extras.hpp"
#include "plasma.hpp"

#ifdef MFEM_USE_MPI

namespace mfem
{

namespace plasma
{

namespace transport
{

/** Multispecies Electron-Ion Collision Time in seconds
 Te is the electron temperature in eV
 ns is the number of ion species
 ni is the density of ions (assuming ni=ne) in particles per meter^3
 zi is the charge number of the ion species
 lnLambda is the Coulomb Logarithm
*/
//double tau_e(double Te, int ns, double * ni, int * zi, double lnLambda);

/** Multispecies Ion-Ion Collision Time in seconds
   ma is the ion mass in a.m.u
   Ta is the ion temperature in eV
   ion is the ion species index for the desired collision time
   ns is the number of ion species
   ni is the density of ions (assuming ni=ne) in particles per meter^3
   zi is the charge number of the ion species
   lnLambda is the Coulomb Logarithm
*/
//double tau_i(double ma, double Ta, int ion, int ns, double * ni, int * zi,
//             double lnLambda);

/**
  Thermal diffusion coefficient along B field for electrons
  Return value is in m^2/s.
   Te is the electron temperature in eV
   ns is the number of ion species
   ni is the density of ions (assuming ni=ne) in particles per meter^3
   zi is the charge number of the ion species
*/
/*
inline double chi_e_para(double Te, int ns, double * ni, int * zi)
{
 // The factor of q_ is included to convert Te from eV to Joules
 return 3.16 * (q_ * Te / me_kg_) * tau_e(Te, ns, ni, zi, 17.0);
}
*/
/**
  Thermal diffusion coefficient perpendicular to B field for electrons
  Return value is in m^2/s.
*/
/*
inline double chi_e_perp()
{
 // The factor of q_ is included to convert Te from eV to Joules
 return 1.0;
}
*/
/**
  Thermal diffusion coefficient perpendicular to both B field and
  thermal gradient for electrons.
  Return value is in m^2/s.
   Te is the electron temperature in eV
   ni is the density of ions (assuming ni=ne) in particles per meter^3
   z is the charge number of the ion species
*/
/*
inline double chi_e_cross()
{
 // The factor of q_ is included to convert Te from eV to Joules
 return 0.0;
}
*/
/**
  Thermal diffusion coefficient along B field for ions
  Return value is in m^2/s.
   ma is the ion mass in a.m.u.
   Ta is the ion temperature in eV
   ion is the ion species index for the desired coefficient
   ns is the number of ion species
   nb is the density of ions in particles per meter^3
   zb is the charge number of the ion species
*/
/*
inline double chi_i_para(double ma, double Ta,
                       int ion, int ns, double * nb, int * zb)
{
 // The factor of q_ is included to convert Ta from eV to Joules
 // The factor of u_ is included to convert ma from a.m.u to kg
 return 3.9 * (q_ * Ta / (ma * kg_per_amu_ ) ) *
        tau_i(ma, Ta, ion, ns, nb, zb, 17.0);
}
*/
/**
  Thermal diffusion coefficient perpendicular to B field for ions
  Return value is in m^2/s.
*/
/*
inline double chi_i_perp()
{
 // The factor of q_ is included to convert Ti from eV to Joules
 // The factor of u_ is included to convert mi from a.m.u to kg
 return 1.0;
}
*/
/**
  Thermal diffusion coefficient perpendicular to both B field and
  thermal gradient for ions
  Return value is in m^2/s.
*/
/*
inline double chi_i_cross()
{
 // The factor of q_ is included to convert Ti from eV to Joules
 // The factor of u_ is included to convert mi from a.m.u to kg
 return 0.0;
}
*/
/**
  Viscosity coefficient along B field for electrons
  Return value is in (a.m.u)*m^2/s.
   ne is the density of electrons in particles per meter^3
   Te is the electron temperature in eV
   ns is the number of ion species
   ni is the density of ions (assuming ni=ne) in particles per meter^3
   zi is the charge number of the ion species
*/
/*
inline double eta_e_para(double ne, double Te, int ns, double * ni, int * zi)
{
 // The factor of q_ is included to convert Te from eV to Joules
 // The factor of u_ is included to convert from kg to a.m.u
 return 0.73 * ne * (q_ * Te / kg_per_amu_) * tau_e(Te, ns, ni, zi, 17.0);
}
*/
/**
  Viscosity coefficient along B field for ions
  Return value is in (a.m.u)*m^2/s.
   ma is the ion mass in a.m.u.
   Ta is the ion temperature in eV
   ion is the ion species index for the desired coefficient
   ns is the number of ion species
   nb is the density of ions in particles per meter^3
   zb is the charge number of the ion species
*/
/*
inline double eta_i_para(double ma, double Ta,
                       int ion, int ns, double * nb, int * zb)
{
 // The factor of q_ is included to convert Ti from eV to Joules
 // The factor of u_ is included to convert from kg to a.m.u
 return 0.96 * nb[ion] * (q_ * Ta / kg_per_amu_) *
        tau_i(ma, Ta, ion, ns, nb, zb, 17.0);
}
*/

void DiscontinuitySensor(GridFunction &u, Vector &error);

struct CoefficientByAttr
{
   Array<int> attr;
   Coefficient * coef;
   bool ownCoef;
};

struct CoefficientsByAttr
{
   Array<int> attr;
   Array<Coefficient*> coefs;
   Array<bool> ownCoefs;
};

class AdvectionDiffusionBC
{
public:
   enum BCType {DIRICHLET_BC, NEUMANN_BC, ROBIN_BC, OUTFLOW_BC};

private:
   Array<CoefficientByAttr*>  dbc; // Dirichlet BC data
   Array<CoefficientByAttr*>  nbc; // Neumann BC data
   Array<CoefficientsByAttr*> rbc; // Robin BC data
   Array<CoefficientByAttr*>  obc; // Outflow BC data
   mutable Array<int>  hbc_attr; // Homogeneous Neumann BC boundary attributes
   Array<int>  dbc_attr; // Dirichlet BC boundary attributes
   Array<int>  obc_attr; // Outflow BC boundary attributes

   std::set<int> bc_attr;
   const Array<int> & bdr_attr;

   common::CoefFactory * coefFact;

   void ReadBCs(std::istream &input);

   void ReadAttr(std::istream &input,
                 BCType bctype,
                 Array<int> &attr);

   void ReadCoefByAttr(std::istream &input,
                       BCType bctype,
                       CoefficientByAttr &cba);

   void ReadCoefsByAttr(std::istream &input,
                        BCType bctype,
                        CoefficientsByAttr &cba);

public:
   AdvectionDiffusionBC(const Array<int> & bdr)
      : bdr_attr(bdr), coefFact(NULL) {}

   AdvectionDiffusionBC(const Array<int> & bdr,
                        common::CoefFactory &cf, std::istream &input)
      : bdr_attr(bdr), coefFact(&cf) { ReadBCs(input); }

   ~AdvectionDiffusionBC();

   void SetTime(double t) const;

   static const char * GetBCTypeName(BCType bctype);

   void LoadBCs(common::CoefFactory &cf, std::istream &input)
   { coefFact = &cf; ReadBCs(input); }

   // Enforce u = val on boundaries with attributes in bdr
   void AddDirichletBC(const Array<int> & bdr, Coefficient &val);

   // Enforce du/dn = val on boundaries with attributes in bdr
   void AddNeumannBC(const Array<int> & bdr, Coefficient &val);

   // Enforce du/dn + a u = b on boundaries with attributes in bdr
   void AddRobinBC(const Array<int> & bdr, Coefficient &a, Coefficient &b);

   // Allows restricted outflow of the fluid through the boundary
   /** An outflow boundary condition is zero on portions of the
       boundary where the advection is directed into the domain. On
       portions where the advection is directed outward a val = 1
       would allow all incident fluid to flow out of the domain. If
       val < 1 the outflow is restricted leading to a buildup of fluid
       at the boundary.
   */
   void AddOutflowBC(const Array<int> & bdr, Coefficient &val);

   const Array<CoefficientByAttr*> & GetDirichletBCs() const { return dbc; }
   const Array<CoefficientByAttr*> & GetNeumannBCs() const { return nbc; }
   const Array<CoefficientsByAttr*> & GetRobinBCs() const { return rbc; }
   const Array<CoefficientByAttr*> & GetOutflowBCs() const { return obc; }

   const Array<int> & GetHomogeneousNeumannBDR() const;
   const Array<int> & GetDirichletBDR() const { return dbc_attr; }
   const Array<int> & GetOutflowBDR() const { return obc_attr; }
};

/** A RecyclingBC describes recombination at a boundary

  In a Recycling boundary condition an ion species recombines with
  electrons contained within the surface of the domain boundary and
  the resulting neutral atoms are added to the population of neutrals.

  We will assume that diffusion into the wall can be neglected
  i.e. only advection of ions towards the wall will lead to
  recycling. It is possible that some fraction of ions will remain
  ionized. The `ion_frac` coefficient should return the fraction
  (between 0 and 1) of incident ions which will be absorbed by the
  boundary. The `neu_frac` coefficient should return the fraction of
  incident ions which will be recycled as neutrals. In summary
  `ion_frac` and `n eu_frac` should be chosen so that:
   0 <= neu_frac <= ion_frac <= 1.

*/
class RecyclingBC
{
private:
   int ion_index;
   int vel_index;
   int neu_index;

   Array<CoefficientsByAttr*> bc; // Recycling BC data

   common::CoefFactory * coefFact;

   // void ReadBCs(std::istream &input);
   void ReadBC(std::istream &input);

   void ReadAttr(std::istream &input,
                 Array<int> &attr);

   void ReadCoefsByAttr(std::istream &input,
                        CoefficientsByAttr &cba);

public:
   RecyclingBC()
      : coefFact(NULL) {}

   // RecyclingBC(common::CoefFactory &cf, std::istream &input)
   //  : coefFact(&cf) { ReadBCs(input); }

   ~RecyclingBC();

   void SetTime(double t) const;

   void LoadBCs(common::CoefFactory &cf, std::istream &input)
   { coefFact = &cf; ReadBC(input); }

   void AddRecyclingBC(int ion, int vel, int neu, const Array<int> & bdr,
                       Coefficient & ion_frac, Coefficient & neu_frac);

   int GetIonDensityIndex() const { return ion_index; }
   int GetIonVelocityIndex() const { return vel_index; }
   int GetNeutralDensityIndex() const { return neu_index; }

   const Array<CoefficientsByAttr*> & GetRecyclingBCs() const { return bc; }
};

class CoupledBCs
{
public:
   enum BCType {RECYCLING_BC};

private:
   Array<RecyclingBC*> rbcs_;

   void ReadBCs(common::CoefFactory &cf, std::istream &input);

public:
   CoupledBCs() {}
   ~CoupledBCs();

   void LoadBCs(common::CoefFactory &cf, std::istream &input)
   { ReadBCs(cf, input); }

   int GetNumRecyclingBCs() const { return rbcs_.Size(); }
   RecyclingBC & GetRecyclingBC(int i) { return *rbcs_[i]; }
   const RecyclingBC & GetRecyclingBC(int i) const { return *rbcs_[i]; }
};

class TransportBCs
{
private:
   int neqn_;
   Array<AdvectionDiffusionBC*> bcs_;
   const Array<int> bdr_attr_;

   CoupledBCs cbcs_;

   void ReadBCs(common::CoefFactory &cf, std::istream &input);

public:
   TransportBCs(const Array<int> & bdr_attr, int neqn);

   TransportBCs(const Array<int> & bdr_attr, int neqn,
                common::CoefFactory &cf, std::istream &input);

   ~TransportBCs();

   void LoadBCs(common::CoefFactory &cf, std::istream &input)
   { ReadBCs(cf, input); }

   AdvectionDiffusionBC & operator()(int i) { return *bcs_[i]; }
   const AdvectionDiffusionBC & operator()(int i) const { return *bcs_[i]; }

   AdvectionDiffusionBC & operator[](int i) { return *bcs_[i]; }
   const AdvectionDiffusionBC & operator[](int i) const { return *bcs_[i]; }

   CoupledBCs & GetCoupledBCs() { return cbcs_; }
   const CoupledBCs & GetCoupledBCs() const { return cbcs_; }
};
/*
class GeneralCoefficient
{
public:
  enum GenCoefType {SCALAR_COEF, VECTOR_COEF, MATRIX_COEF};

private:
   CoefFactory * coefFact;

   GenCoefType type;
   Coefficient       * sCoef;
   VectorCoefficient * vCoef;
   MatrixCoefficient * mCoef;

   void ReadCoef(std::istream &input);

public:
  GeneralCoefficient()
    : coefFact(NULL), sCoef(NULL), vCoef(NULL), mCoef(NULL) {}
  GeneralCoefficient(CoefFactory &cf, std::istream &input)
    : coefFact(&cf), sCoef(NULL), vCoef(NULL), mCoef(NULL)
  { ReadCoef(input); }

   void LoadCoef(CoefFactory &cf, std::istream &input)
   { coefFact = &cf; ReadCoef(input); }

   void AddCoefficient(Coefficient &c) { type = SCALAR_COEF; sCoef = &c; }
   void AddCoefficient(VectorCoefficient &c) { type = VECTOR_COEF; vCoef = &c; }
   void AddCoefficient(MatrixCoefficient &c) { type = MATRIX_COEF; mCoef = &c; }

   GenCoefType GetCoefficientType() const { return type; }
   Coefficient * GetCoefficient() const { return sCoef; }
   VectorCoefficient * GetVectorCoefficient() const { return vCoef; }
   MatrixCoefficient * GetMatrixCoefficient() const { return mCoef; }
};
*/
class TransportICs
{
private:
   int neqn_;
   Array<Coefficient *> ics_;
   Array<bool> own_ics_;

   void ReadICs(common::CoefFactory &cf, std::istream &input);

public:
   TransportICs(int neqn)
      : neqn_(neqn),
        ics_(neqn),
        own_ics_(neqn)
   {
      ics_ = NULL;
      own_ics_ = false;
   }

   TransportICs(int neqn, common::CoefFactory &cf, std::istream &input);

   ~TransportICs()
   {
      for (int i=0; i<neqn_; i++)
      {
         if (own_ics_[i]) { delete ics_[i]; }
      }
   }

   void LoadICs(common::CoefFactory &cf, std::istream &input)
   { ReadICs(cf, input); }

   void SetOwnership(int i, bool own) { own_ics_[i] = own; }

   Coefficient *& operator()(int i) { return ics_[i]; }
   const Coefficient * operator()(int i) const { return ics_[i]; }

   Coefficient *& operator[](int i) { return ics_[i]; }
   const Coefficient * operator[](int i) const { return ics_[i]; }
};

class TransportExactSolutions
{
private:
   int neqn_;
   Array<Coefficient *> ess_;
   Array<bool> own_ess_;

   void Read(common::CoefFactory &cf, std::istream &input);

public:
   TransportExactSolutions(int neqn)
      : neqn_(neqn),
        ess_(neqn),
        own_ess_(neqn)
   {
      ess_ = NULL;
      own_ess_ = false;
   }

   TransportExactSolutions(int neqn, common::CoefFactory &cf,
                           std::istream &input);

   ~TransportExactSolutions()
   {
      for (int i=0; i<neqn_; i++)
      {
         if (own_ess_[i]) { delete ess_[i]; }
      }
   }

   void LoadExactSolutions(common::CoefFactory &cf, std::istream &input)
   { Read(cf, input); }

   void SetOwnership(int i, bool own) { own_ess_[i] = own; }

   Coefficient *& operator()(int i) { return ess_[i]; }
   const Coefficient * operator()(int i) const { return ess_[i]; }

   Coefficient *& operator[](int i) { return ess_[i]; }
   const Coefficient * operator[](int i) const { return ess_[i]; }
};

class StateVariableCoef;
class StateVariableVecCoef;
class StateVariableMatCoef;

class EqnCoefficients
{
protected:

   Array<StateVariableCoef *> sCoefs_;
   Array<StateVariableVecCoef *> vCoefs_;
   Array<StateVariableMatCoef *> mCoefs_;

   std::vector<std::string> sCoefNames_;
   std::vector<std::string> vCoefNames_;
   std::vector<std::string> mCoefNames_;

   common::CoefFactory * coefFact;

   virtual void ReadCoefs(std::istream &input);

public:
   EqnCoefficients(int nSCoefs, int nVCoefs = 0, int nMCoefs = 0)
      : sCoefs_(nSCoefs), vCoefs_(nVCoefs), mCoefs_(nMCoefs),
        sCoefNames_(nSCoefs), vCoefNames_(nVCoefs), mCoefNames_(nMCoefs)
   {
      sCoefs_ = NULL;
      vCoefs_ = NULL;
      mCoefs_ = NULL;
   }

   virtual ~EqnCoefficients() {}

   void LoadCoefs(common::CoefFactory &cf, std::istream &input)
   { coefFact = &cf; ReadCoefs(input); }

   StateVariableCoef *& operator()(int i) { return sCoefs_[i]; }
   const StateVariableCoef * operator()(int i) const { return sCoefs_[i]; }

   StateVariableCoef *& operator[](int i) { return sCoefs_[i]; }
   const StateVariableCoef * operator[](int i) const { return sCoefs_[i]; }

   StateVariableCoef *& GetScalarCoefficient(int i) { return sCoefs_[i]; }
   const StateVariableCoef * GetScalarCoefficient(int i) const
   { return sCoefs_[i]; }

   StateVariableVecCoef *& GetVectorCoefficient(int i) { return vCoefs_[i]; }
   const StateVariableVecCoef * GetVectorCoefficient(int i) const
   { return vCoefs_[i]; }

   StateVariableMatCoef *& GetMatrixCoefficient(int i) { return mCoefs_[i]; }
   const StateVariableMatCoef * GetMatrixCoefficient(int i) const
   { return mCoefs_[i]; }
};

class NeutralDensityCoefs : public EqnCoefficients
{
public:
   enum sCoefNames {DIFFUSION_COEF = 0, SOURCE_COEF, NUM_SCALAR_COEFS};

   NeutralDensityCoefs();
};

class IonDensityCoefs : public EqnCoefficients
{
public:
   enum sCoefNames {PERP_DIFFUSION_COEF = 0, PARA_DIFFUSION_COEF,
                    SOURCE_COEF, NUM_SCALAR_COEFS
                   };

   IonDensityCoefs();
};

class IonMomentumCoefs : public EqnCoefficients
{
public:
   enum sCoefNames {PARA_DIFFUSION_COEF = 0, PERP_DIFFUSION_COEF,
                    SOURCE_COEF, NUM_SCALAR_COEFS
                   };

   IonMomentumCoefs();
};

class IonStaticPressureCoefs : public EqnCoefficients
{
public:
   enum sCoefNames {PARA_DIFFUSION_COEF = 0, PERP_DIFFUSION_COEF,
                    SOURCE_COEF, NUM_SCALAR_COEFS
                   };

   IonStaticPressureCoefs();
};

class ElectronStaticPressureCoefs : public EqnCoefficients
{
public:
   enum sCoefNames {PERP_DIFFUSION_COEF = 0, PARA_DIFFUSION_COEF,
                    SOURCE_COEF, NUM_SCALAR_COEFS
                   };

   ElectronStaticPressureCoefs();
};

class IonTotalEnergyCoefs : public EqnCoefficients
{
public:
   enum sCoefNames {PARA_DIFFUSION_COEF = 0, PERP_DIFFUSION_COEF,
                    SOURCE_COEF, NUM_SCALAR_COEFS
                   };

   IonTotalEnergyCoefs();
};

class ElectronTotalEnergyCoefs : public EqnCoefficients
{
public:
   enum sCoefNames {PERP_DIFFUSION_COEF = 0, PARA_DIFFUSION_COEF,
                    SOURCE_COEF, NUM_SCALAR_COEFS
                   };

   ElectronTotalEnergyCoefs();
};

class CommonCoefs : public EqnCoefficients
{
public:
   enum sCoefNames {IONIZATION_COEF = 0, RECOMBINATION_COEF,
                    CHARGE_EXCHANGE_COEF,
                    NUM_SCALAR_COEFS
                   };
   enum vCoefNames {MAGNETIC_FIELD_COEF = 0, NUM_VECTOR_COEFS};

   CommonCoefs();
};

typedef NeutralDensityCoefs NDCoefs;
typedef IonDensityCoefs IDCoefs;
typedef IonMomentumCoefs IMCoefs;
typedef IonStaticPressureCoefs ISPCoefs;
typedef ElectronStaticPressureCoefs ESPCoefs;
typedef IonTotalEnergyCoefs ITECoefs;
typedef ElectronTotalEnergyCoefs ETECoefs;
typedef CommonCoefs CmnCoefs;

class TransportCoefs
{
private:
   int neqn_;
   Array<EqnCoefficients *> eqnCoefs_;

   void ReadCoefs(common::CoefFactory &cf, std::istream &input);

public:
   TransportCoefs(int neqn)
      : neqn_(neqn),
        eqnCoefs_(neqn+1)
   {
      eqnCoefs_ = NULL;
      eqnCoefs_[0] = new NeutralDensityCoefs;
      eqnCoefs_[1] = new IonDensityCoefs;
      eqnCoefs_[2] = new IonMomentumCoefs;
      // eqnCoefs_[3] = new IonStaticPressureCoefs;
      // eqnCoefs_[4] = new ElectronStaticPressureCoefs;
      eqnCoefs_[3] = new IonTotalEnergyCoefs;
      eqnCoefs_[4] = new ElectronTotalEnergyCoefs;
      eqnCoefs_[5] = new CommonCoefs;
   }

   TransportCoefs(int neqn, common::CoefFactory &cf, std::istream &input);

   ~TransportCoefs()
   {
      for (int i=0; i<=neqn_; i++)
      {
         delete eqnCoefs_[i];
      }
   }

   void LoadCoefs(common::CoefFactory &cf, std::istream &input)
   { ReadCoefs(cf, input); }

   EqnCoefficients & operator()(int i) { return *eqnCoefs_[i]; }
   const EqnCoefficients & operator()(int i) const { return *eqnCoefs_[i]; }

   EqnCoefficients & operator[](int i) { return *eqnCoefs_[i]; }
   const EqnCoefficients & operator[](int i) const { return *eqnCoefs_[i]; }

   NDCoefs & GetNeutralDensityCoefs()
   { return dynamic_cast<NDCoefs&>(*eqnCoefs_[0]); }
   const NDCoefs & GetNeutralDensityCoefs() const
   { return dynamic_cast<const NDCoefs&>(*eqnCoefs_[0]); }

   IDCoefs & GetIonDensityCoefs()
   { return dynamic_cast<IDCoefs&>(*eqnCoefs_[1]); }
   const IDCoefs & GetIonDensityCoefs() const
   { return dynamic_cast<const IDCoefs&>(*eqnCoefs_[1]); }

   IMCoefs & GetIonMomentumCoefs()
   { return dynamic_cast<IMCoefs&>(*eqnCoefs_[2]); }
   const IMCoefs & GetIonMomentumCoefs() const
   { return dynamic_cast<const IMCoefs&>(*eqnCoefs_[2]); }
   /*
   ISPCoefs & GetIonStaticPressureCoefs()
   { return dynamic_cast<ISPCoefs&>(*eqnCoefs_[3]); }
   const ISPCoefs & GetIonStaticPressureCoefs() const
   { return dynamic_cast<const ISPCoefs&>(*eqnCoefs_[3]); }

   ESPCoefs & GetElectronStaticPressureCoefs()
   { return dynamic_cast<ESPCoefs&>(*eqnCoefs_[4]); }
   const ESPCoefs & GetElectronStaticPressureCoefs() const
   { return dynamic_cast<const ESPCoefs&>(*eqnCoefs_[4]); }
   */
   ITECoefs & GetIonTotalEnergyCoefs()
   { return dynamic_cast<ITECoefs&>(*eqnCoefs_[3]); }
   const ITECoefs & GetIonTotalEnergyCoefs() const
   { return dynamic_cast<const ITECoefs&>(*eqnCoefs_[3]); }

   ETECoefs & GetElectronTotalEnergyCoefs()
   { return dynamic_cast<ETECoefs&>(*eqnCoefs_[4]); }
   const ETECoefs & GetElectronTotalEnergyCoefs() const
   { return dynamic_cast<const ETECoefs&>(*eqnCoefs_[4]); }

   CommonCoefs & GetCommonCoefs()
   { return dynamic_cast<CommonCoefs&>(*eqnCoefs_[5]); }
   const CommonCoefs & GetCommonCoefs() const
   { return dynamic_cast<const CommonCoefs&>(*eqnCoefs_[5]); }
};

class ParGridFunctionArray : public Array<ParGridFunction*>
{
private:
   bool owns_data;

public:
   ParGridFunctionArray() : owns_data(true) {}
   ParGridFunctionArray(int size)
      : Array<ParGridFunction*>(size), owns_data(true) {}
   ParGridFunctionArray(int size, ParFiniteElementSpace *pf)
      : Array<ParGridFunction*>(size), owns_data(true)
   {
      for (int i=0; i<size; i++)
      {
         data[i] = new ParGridFunction(pf);
      }
   }

   ~ParGridFunctionArray()
   {
      if (owns_data)
      {
         for (int i=0; i<size; i++)
         {
            delete data[i];
         }
      }
   }

   void SetOwner(bool owner) { owns_data = owner; }
   bool GetOwner() const { return owns_data; }

   void ProjectCoefficient(Array<Coefficient*> &coeff)
   {
      for (int i=0; i<size; i++)
      {
         if (coeff[i] != NULL)
         {
            data[i]->ProjectCoefficient(*coeff[i]);
         }
         else
         {
            *data[i] = 0.0;
         }
      }
   }

   void Update()
   {
      for (int i=0; i<size; i++)
      {
         data[i]->Update();
      }
   }

   void ExchangeFaceNbrData()
   {
      for (int i=0; i<size; i++)
      {
         data[i]->ExchangeFaceNbrData();
      }
   }
};

enum FieldType {INVALID_FIELD_TYPE   = -1,
                NEUTRAL_DENSITY      = 0,
                ION_DENSITY          = 1,
                ION_PARA_VELOCITY    = 2,
                ION_TEMPERATURE      = 3,
                ELECTRON_TEMPERATURE = 4
               };

std::string FieldSymbol(FieldType t);

class TransportCoefFactory : public common::CoefFactory
{
public:
   TransportCoefFactory() {}
   TransportCoefFactory(const std::vector<std::string> & names,
                        ParGridFunctionArray & pgfa);

   using common::CoefFactory::GetScalarCoef;
   using common::CoefFactory::GetVectorCoef;
   using common::CoefFactory::GetMatrixCoef;

   Coefficient * GetScalarCoef(std::string &name, std::istream &input);
   VectorCoefficient * GetVectorCoef(std::string &name, std::istream &input);
};

class StateVariableFunc
{
public:

   virtual bool NonTrivialValue(FieldType deriv) const = 0;

   void SetDerivType(FieldType deriv) { derivType_ = deriv; }
   FieldType GetDerivType() const { return derivType_; }

protected:
   StateVariableFunc(FieldType deriv = INVALID_FIELD_TYPE)
      : derivType_(deriv) {}

   FieldType derivType_;
};


class StateVariableCoef : public StateVariableFunc, public Coefficient
{
public:

   virtual StateVariableCoef * Clone() const = 0;

   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
   {
      switch (derivType_)
      {
         case INVALID_FIELD_TYPE:
            return Eval_Func(T, ip);
         case NEUTRAL_DENSITY:
            return Eval_dNn(T, ip);
         case ION_DENSITY:
            return Eval_dNi(T, ip);
         case ION_PARA_VELOCITY:
            return Eval_dVi(T, ip);
         case ION_TEMPERATURE:
            return Eval_dTi(T, ip);
         case ELECTRON_TEMPERATURE:
            return Eval_dTe(T, ip);
         default:
            return 0.0;
      }
   }

   /// Implementation of the coefficient function
   virtual double Eval_Func(ElementTransformation &T,
                            const IntegrationPoint &ip) { return 0.0; }

   /// Implementation of the genderalized derivative of the function
   virtual double Eval_dFunc(FieldType deriv,
                             ElementTransformation &T,
                             const IntegrationPoint &ip) { return 0.0; }

   /// The following can be overridden for efficiency when appropriate
   virtual double Eval_dNn(ElementTransformation &T,
                           const IntegrationPoint &ip)
   {
      return this->Eval_dFunc(NEUTRAL_DENSITY, T, ip);
   }

   virtual double Eval_dNi(ElementTransformation &T,
                           const IntegrationPoint &ip)
   {
      return this->Eval_dFunc(ION_DENSITY, T, ip);
   }

   virtual double Eval_dVi(ElementTransformation &T,
                           const IntegrationPoint &ip)
   {
      return this->Eval_dFunc(ION_PARA_VELOCITY, T, ip);
   }

   virtual double Eval_dTi(ElementTransformation &T,
                           const IntegrationPoint &ip)
   {
      return this->Eval_dFunc(ION_TEMPERATURE, T, ip);
   }

   virtual double Eval_dTe(ElementTransformation &T,
                           const IntegrationPoint &ip)
   {
      return this->Eval_dFunc(ELECTRON_TEMPERATURE, T, ip);
   }

protected:
   StateVariableCoef(FieldType deriv = INVALID_FIELD_TYPE)
      : StateVariableFunc(deriv) {}
};

class StateVariableVecCoef : public StateVariableFunc,
   public VectorCoefficient
{
public:
   virtual StateVariableVecCoef * Clone() const = 0;

   virtual void Eval(Vector &V,
                     ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      V.SetSize(vdim);

      switch (derivType_)
      {
         case INVALID_FIELD_TYPE:
            return Eval_Func(V, T, ip);
         case NEUTRAL_DENSITY:
            return Eval_dNn(V, T, ip);
         case ION_DENSITY:
            return Eval_dNi(V, T, ip);
         case ION_PARA_VELOCITY:
            return Eval_dVi(V, T, ip);
         case ION_TEMPERATURE:
            return Eval_dTi(V, T, ip);
         case ELECTRON_TEMPERATURE:
            return Eval_dTe(V, T, ip);
         default:
            V = 0.0;
            return;
      }
   }

   virtual void Eval_Func(Vector &V,
                          ElementTransformation &T,
                          const IntegrationPoint &ip) { V = 0.0; }

   virtual void Eval_dNn(Vector &V,
                         ElementTransformation &T,
                         const IntegrationPoint &ip) { V = 0.0; }

   virtual void Eval_dNi(Vector &V,
                         ElementTransformation &T,
                         const IntegrationPoint &ip) { V = 0.0; }

   virtual void Eval_dVi(Vector &V,
                         ElementTransformation &T,
                         const IntegrationPoint &ip) { V = 0.0; }

   virtual void Eval_dTi(Vector &V,
                         ElementTransformation &T,
                         const IntegrationPoint &ip) { V = 0.0; }

   virtual void Eval_dTe(Vector &V,
                         ElementTransformation &T,
                         const IntegrationPoint &ip) { V = 0.0; }

protected:
   StateVariableVecCoef(int dim, FieldType deriv = INVALID_FIELD_TYPE)
      : StateVariableFunc(deriv), VectorCoefficient(dim) {}
};

class StateVariableMatCoef : public StateVariableFunc,
   public MatrixCoefficient
{
public:

   virtual StateVariableMatCoef * Clone() const = 0;

   virtual void Eval(DenseMatrix &M,
                     ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      M.SetSize(height, width);

      switch (derivType_)
      {
         case INVALID_FIELD_TYPE:
            return Eval_Func(M, T, ip);
         case NEUTRAL_DENSITY:
            return Eval_dNn(M, T, ip);
         case ION_DENSITY:
            return Eval_dNi(M, T, ip);
         case ION_PARA_VELOCITY:
            return Eval_dVi(M, T, ip);
         case ION_TEMPERATURE:
            return Eval_dTi(M, T, ip);
         case ELECTRON_TEMPERATURE:
            return Eval_dTe(M, T, ip);
         default:
            M = 0.0;
            return;
      }
   }

   virtual void Eval_Func(DenseMatrix &M,
                          ElementTransformation &T,
                          const IntegrationPoint &ip) { M = 0.0; }

   virtual void Eval_dNn(DenseMatrix &M,
                         ElementTransformation &T,
                         const IntegrationPoint &ip) { M = 0.0; }

   virtual void Eval_dNi(DenseMatrix &M,
                         ElementTransformation &T,
                         const IntegrationPoint &ip) { M = 0.0; }

   virtual void Eval_dVi(DenseMatrix &M,
                         ElementTransformation &T,
                         const IntegrationPoint &ip) { M = 0.0; }

   virtual void Eval_dTi(DenseMatrix &M,
                         ElementTransformation &T,
                         const IntegrationPoint &ip) { M = 0.0; }

   virtual void Eval_dTe(DenseMatrix &M,
                         ElementTransformation &T,
                         const IntegrationPoint &ip) { M = 0.0; }

protected:
   StateVariableMatCoef(int dim, FieldType deriv = INVALID_FIELD_TYPE)
      : StateVariableFunc(deriv), MatrixCoefficient(dim) {}

   StateVariableMatCoef(int h, int w, FieldType deriv = INVALID_FIELD_TYPE)
      : StateVariableFunc(deriv), MatrixCoefficient(h, w) {}
};

class StateVariableStandardCoef : public StateVariableCoef
{
private:
   Coefficient & c_;

public:
   StateVariableStandardCoef(Coefficient & c)
      : c_(c)
   {}

   StateVariableStandardCoef(const StateVariableStandardCoef &other)
      : c_(other.c_)
   {}

   virtual StateVariableStandardCoef * Clone() const
   {
      return new StateVariableStandardCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE);
   }

   virtual double Eval_Func(ElementTransformation &T,
                            const IntegrationPoint &ip)
   {
      return c_.Eval(T, ip);
   }
};

class StateVariableStandardVecCoef : public StateVariableVecCoef
{
private:
   VectorCoefficient & V_;

public:
   StateVariableStandardVecCoef(VectorCoefficient & V)
      : StateVariableVecCoef(V.GetVDim()), V_(V)
   {}

   StateVariableStandardVecCoef(const StateVariableStandardVecCoef &other)
      : StateVariableVecCoef(other.vdim), V_(other.V_)
   {}

   virtual StateVariableStandardVecCoef * Clone() const
   {
      return new StateVariableStandardVecCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE);
   }

   virtual void Eval_Func(Vector & V, ElementTransformation &T,
                          const IntegrationPoint &ip)
   {
      V_.Eval(V, T, ip);
   }
};

class StateVariableStandardMatCoef : public StateVariableMatCoef
{
private:
   MatrixCoefficient & M_;

public:
   StateVariableStandardMatCoef(MatrixCoefficient & M)
      : StateVariableMatCoef(M.GetHeight(), M.GetWidth()), M_(M)
   {}

   StateVariableStandardMatCoef(const StateVariableStandardMatCoef &other)
      : StateVariableMatCoef(other.height, other.width), M_(other.M_)
   {}

   virtual StateVariableStandardMatCoef * Clone() const
   {
      return new StateVariableStandardMatCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE);
   }

   virtual void Eval_Func(DenseMatrix & M, ElementTransformation &T,
                          const IntegrationPoint &ip)
   {
      M_.Eval(M, T, ip);
   }
};

class StateVariableConstantCoef : public StateVariableCoef
{
private:
   ConstantCoefficient c_;

public:
   StateVariableConstantCoef(double c)
      : c_(c)
   {}

   StateVariableConstantCoef(const StateVariableConstantCoef &other)
      : c_(other.c_)
   {}

   virtual StateVariableConstantCoef * Clone() const
   {
      return new StateVariableConstantCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE);
   }

   virtual double Eval_Func(ElementTransformation &T,
                            const IntegrationPoint &ip)
   {
      return c_.Eval(T, ip);
   }
};

class StateVariableGridFunctionCoef : public StateVariableCoef
{
private:
   GridFunctionCoefficient gfc_;
   FieldType fieldType_;

public:
   StateVariableGridFunctionCoef(FieldType field)
      : fieldType_(field)
   {}

   StateVariableGridFunctionCoef(GridFunction *gf, FieldType field)
      : gfc_(gf), fieldType_(field)
   {}

   StateVariableGridFunctionCoef(const StateVariableGridFunctionCoef &other)
   {
      derivType_ = other.derivType_;
      fieldType_ = other.fieldType_;
      gfc_       = other.gfc_;
   }

   virtual StateVariableGridFunctionCoef * Clone() const
   {
      return new StateVariableGridFunctionCoef(*this);
   }

   void SetGridFunction(GridFunction *gf) { gfc_.SetGridFunction(gf); }
   const GridFunction * GetGridFunction() const
   { return gfc_.GetGridFunction(); }

   void SetFieldType(FieldType field) { fieldType_ = field; }
   FieldType GetFieldType() const { return fieldType_; }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE || deriv == fieldType_);
   }

   virtual double Eval_Func(ElementTransformation &T,
                            const IntegrationPoint &ip)
   {
      return gfc_.Eval(T, ip);
   }

   virtual double Eval_dNn(ElementTransformation &T,
                           const IntegrationPoint &ip)
   { return (fieldType_ == NEUTRAL_DENSITY) ? 1.0 : 0.0; }

   virtual double Eval_dNi(ElementTransformation &T,
                           const IntegrationPoint &ip)
   { return (fieldType_ == ION_DENSITY) ? 1.0 : 0.0; }

   virtual double Eval_dVi(ElementTransformation &T,
                           const IntegrationPoint &ip)
   { return (fieldType_ == ION_PARA_VELOCITY) ? 1.0 : 0.0; }

   virtual double Eval_dTi(ElementTransformation &T,
                           const IntegrationPoint &ip)
   { return (fieldType_ == ION_TEMPERATURE) ? 1.0 : 0.0; }

   virtual double Eval_dTe(ElementTransformation &T,
                           const IntegrationPoint &ip)
   { return (fieldType_ == ELECTRON_TEMPERATURE) ? 1.0 : 0.0; }
};

class StateVariableSumCoef : public StateVariableCoef
{
private:
   StateVariableCoef *a;
   StateVariableCoef *b;

   double alpha;
   double beta;

public:
   // Result is _alpha * A + _beta * B
   StateVariableSumCoef(StateVariableCoef &A, StateVariableCoef &B,
                        double _alpha = 1.0, double _beta = 1.0)
      : a(A.Clone()), b(B.Clone()), alpha(_alpha), beta(_beta) {}

   ~StateVariableSumCoef()
   {
      if (a != NULL) { delete a; }
      if (b != NULL) { delete b; }
   }

   virtual StateVariableSumCoef * Clone() const
   {
      return new StateVariableSumCoef(*a, *b, alpha, beta);
   }

   void SetACoef(StateVariableCoef &A) { a = &A; }
   StateVariableCoef * GetACoef() const { return a; }

   void SetBCoef(StateVariableCoef &B) { b = &B; }
   StateVariableCoef * GetBCoef() const { return b; }

   void SetAlpha(double _alpha) { alpha = _alpha; }
   double GetAlpha() const { return alpha; }

   void SetBeta(double _beta) { beta = _beta; }
   double GetBeta() const { return beta; }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return a->NonTrivialValue(deriv) || b->NonTrivialValue(deriv);
   }

   /// Evaluate the coefficient
   virtual double Eval_Func(ElementTransformation &T,
                            const IntegrationPoint &ip)
   { return alpha * a->Eval_Func(T, ip) + beta * b->Eval_Func(T, ip); }

   virtual double Eval_dNn(ElementTransformation &T,
                           const IntegrationPoint &ip)
   { return alpha * a->Eval_dNn(T, ip) + beta * b->Eval_dNn(T, ip); }

   virtual double Eval_dNi(ElementTransformation &T,
                           const IntegrationPoint &ip)
   { return alpha * a->Eval_dNi(T, ip) + beta * b->Eval_dNi(T, ip); }

   virtual double Eval_dVi(ElementTransformation &T,
                           const IntegrationPoint &ip)
   { return alpha * a->Eval_dVi(T, ip) + beta * b->Eval_dVi(T, ip); }

   virtual double Eval_dTi(ElementTransformation &T,
                           const IntegrationPoint &ip)
   { return alpha * a->Eval_dTi(T, ip) + beta * b->Eval_dTi(T, ip); }

   virtual double Eval_dTe(ElementTransformation &T,
                           const IntegrationPoint &ip)
   { return alpha * a->Eval_dTe(T, ip) + beta * b->Eval_dTe(T, ip); }
};

class StateVariableProductCoef : public StateVariableCoef
{
private:
   StateVariableCoef *a;
   StateVariableCoef *b;

public:
   // Result is A * B
   StateVariableProductCoef(StateVariableCoef &A, StateVariableCoef &B)
      : a(A.Clone()), b(B.Clone()) {}

   ~StateVariableProductCoef()
   {
      if (a != NULL) { delete a; }
      if (b != NULL) { delete b; }
   }

   virtual StateVariableProductCoef * Clone() const
   {
      return new StateVariableProductCoef(*a, *b);
   }

   void SetACoef(StateVariableCoef &A) { a = &A; }
   StateVariableCoef * GetACoef() const { return a; }

   void SetBCoef(StateVariableCoef &B) { b = &B; }
   StateVariableCoef * GetBCoef() const { return b; }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return a->NonTrivialValue(deriv) || b->NonTrivialValue(deriv);
   }

   /// Evaluate the coefficient
   virtual double Eval_Func(ElementTransformation &T,
                            const IntegrationPoint &ip)
   { return a->Eval_Func(T, ip) * b->Eval_Func(T, ip); }

   virtual double Eval_dNn(ElementTransformation &T,
                           const IntegrationPoint &ip)
   {
      return a->Eval_dNn(T, ip) * b->Eval_Func(T, ip) +
             a->Eval_Func(T, ip) * b->Eval_dNn(T, ip);
   }

   virtual double Eval_dNi(ElementTransformation &T,
                           const IntegrationPoint &ip)
   {
      return a->Eval_dNi(T, ip) * b->Eval_Func(T, ip) +
             a->Eval_Func(T, ip) * b->Eval_dNi(T, ip);
   }

   virtual double Eval_dVi(ElementTransformation &T,
                           const IntegrationPoint &ip)
   {
      return a->Eval_dVi(T, ip) * b->Eval_Func(T, ip) +
             a->Eval_Func(T, ip) * b->Eval_dVi(T, ip);
   }

   virtual double Eval_dTi(ElementTransformation &T,
                           const IntegrationPoint &ip)
   {
      return a->Eval_dTi(T, ip) * b->Eval_Func(T, ip) +
             a->Eval_Func(T, ip) * b->Eval_dTi(T, ip);
   }

   virtual double Eval_dTe(ElementTransformation &T,
                           const IntegrationPoint &ip)
   {
      return a->Eval_dTe(T, ip) * b->Eval_Func(T, ip) +
             a->Eval_Func(T, ip) * b->Eval_dTe(T, ip);
   }
};

class StateVariablePowerCoef : public StateVariableCoef
{
private:
   StateVariableCoef *a;
   int p;

public:
   // Result is A^p
   StateVariablePowerCoef(StateVariableCoef &A, int p)
      : a(A.Clone()), p(p) {}

   ~StateVariablePowerCoef()
   {
      if (a != NULL) { delete a; }
   }

   virtual StateVariablePowerCoef * Clone() const
   {
      return new StateVariablePowerCoef(*a, p);
   }

   void SetACoef(StateVariableCoef &A) { a = &A; }
   StateVariableCoef * GetACoef() const { return a; }

   void SetPower(int new_p) { p = new_p; }
   int GetPower() const { return p; }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return a->NonTrivialValue(deriv);
   }

   /// Evaluate the coefficient
   virtual double Eval_Func(ElementTransformation &T,
                            const IntegrationPoint &ip)
   { return pow(a->Eval_Func(T, ip), p); }

   virtual double Eval_dNn(ElementTransformation &T,
                           const IntegrationPoint &ip)
   {
      if (p == 0) { return 0; }
      if (p == 1) { return a->Eval_dNn(T, ip); }
      return (double)p * pow(a->Eval_Func(T, ip), p-1) * a->Eval_dNn(T, ip);
   }

   virtual double Eval_dNi(ElementTransformation &T,
                           const IntegrationPoint &ip)
   {
      if (p == 0) { return 0; }
      if (p == 1) { return a->Eval_dNi(T, ip); }
      return (double)p * pow(a->Eval_Func(T, ip), p-1) * a->Eval_dNi(T, ip);
   }

   virtual double Eval_dVi(ElementTransformation &T,
                           const IntegrationPoint &ip)
   {
      if (p == 0) { return 0; }
      if (p == 1) { return a->Eval_dVi(T, ip); }
      return (double)p * pow(a->Eval_Func(T, ip), p-1) * a->Eval_dVi(T, ip);
   }

   virtual double Eval_dTi(ElementTransformation &T,
                           const IntegrationPoint &ip)
   {
      if (p == 0) { return 0; }
      if (p == 1) { return a->Eval_dTi(T, ip); }
      return (double)p * pow(a->Eval_Func(T, ip), p-1) * a->Eval_dTi(T, ip);
   }

   virtual double Eval_dTe(ElementTransformation &T,
                           const IntegrationPoint &ip)
   {
      if (p == 0) { return 0; }
      if (p == 1) { return a->Eval_dTe(T, ip); }
      return (double)p * pow(a->Eval_Func(T, ip), p-1) * a->Eval_dTe(T, ip);
   }
};

class StateVariableScalarVectorProductCoef : public StateVariableVecCoef
{
private:
   StateVariableCoef    *a;
   StateVariableVecCoef *b;

   VectorCoefficient *b_std;

   mutable Vector dV_;

public:
   // Result is A * B
   StateVariableScalarVectorProductCoef(StateVariableCoef &A,
                                        StateVariableVecCoef &B)
      : StateVariableVecCoef(B.GetVDim()),
        a(A.Clone()), b(B.Clone()), b_std(NULL), dV_(B.GetVDim()) {}

   StateVariableScalarVectorProductCoef(StateVariableCoef &A,
                                        VectorCoefficient &B)
      : StateVariableVecCoef(B.GetVDim()),
        a(A.Clone()), b(NULL), b_std(&B) {}

   ~StateVariableScalarVectorProductCoef()
   {
      if (a != NULL) { delete a; }
      if (b != NULL) { delete b; }
   }

   virtual StateVariableScalarVectorProductCoef * Clone() const
   {
      if (b != NULL)
      {
         return new StateVariableScalarVectorProductCoef(*a, *b);
      }
      else
      {
         return new StateVariableScalarVectorProductCoef(*a, *b_std);
      }
   }

   void SetACoef(StateVariableCoef &A) { a = &A; }
   StateVariableCoef * GetACoef() const { return a; }

   void SetBCoef(StateVariableVecCoef &B) { b = &B; b_std = NULL; }
   StateVariableVecCoef * GetBCoef() const { return b; }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return a->NonTrivialValue(deriv) ||
             (b ? b->NonTrivialValue(deriv) : false);
   }

   /// Evaluate the coefficient
   virtual void Eval_Func(Vector &V, ElementTransformation &T,
                          const IntegrationPoint &ip)
   {
      b ? b->Eval_Func(V, T, ip) : b_std->Eval(V, T, ip);
      V *= a->Eval_Func(T, ip);
   }

   virtual void Eval_dNn(Vector &V, ElementTransformation &T,
                         const IntegrationPoint &ip)
   {
      b ? b->Eval_Func(V, T, ip) : b_std->Eval(V, T, ip);
      V *= a->Eval_dNn(T, ip);
      if (b)
      {
         b->Eval_dNn(dV_, T, ip); dV_ *= a->Eval_Func(T, ip);
         V += dV_;
      }
   }

   virtual void Eval_dNi(Vector &V, ElementTransformation &T,
                         const IntegrationPoint &ip)
   {
      b ? b->Eval_Func(V, T, ip) : b_std->Eval(V, T, ip);
      V *= a->Eval_dNi(T, ip);
      if (b)
      {
         b->Eval_dNi(dV_, T, ip); dV_ *= a->Eval_Func(T, ip);
         V += dV_;
      }
   }

   virtual void Eval_dVi(Vector &V, ElementTransformation &T,
                         const IntegrationPoint &ip)
   {
      b ? b->Eval_Func(V, T, ip) : b_std->Eval(V, T, ip);
      V *= a->Eval_dVi(T, ip);
      if (b)
      {
         b->Eval_dVi(dV_, T, ip); dV_ *= a->Eval_Func(T, ip);
         V += dV_;
      }
   }

   virtual void Eval_dTi(Vector &V, ElementTransformation &T,
                         const IntegrationPoint &ip)
   {
      b ? b->Eval_Func(V, T, ip) : b_std->Eval(V, T, ip);
      V *= a->Eval_dTi(T, ip);
      if (b)
      {
         b->Eval_dTi(dV_, T, ip); dV_ *= a->Eval_Func(T, ip);
         V += dV_;
      }
   }

   virtual void Eval_dTe(Vector &V, ElementTransformation &T,
                         const IntegrationPoint &ip)
   {
      b ? b->Eval_Func(V, T, ip) : b_std->Eval(V, T, ip);
      V *= a->Eval_dTe(T, ip);
      if (b)
      {
         b->Eval_dTe(dV_, T, ip); dV_ *= a->Eval_Func(T, ip);
         V += dV_;
      }
   }
};

class StateVariableScalarMatrixProductCoef : public StateVariableMatCoef
{
private:
   StateVariableCoef    *a;
   StateVariableMatCoef *b;

   mutable DenseMatrix dM_;

public:
   // Result is A * B
   StateVariableScalarMatrixProductCoef(StateVariableCoef &A,
                                        StateVariableMatCoef &B)
      : StateVariableMatCoef(B.GetHeight(), B.GetWidth()),
        a(A.Clone()), b(B.Clone()), dM_(B.GetHeight(), B.GetWidth()) {}

   ~StateVariableScalarMatrixProductCoef()
   {
      if (a != NULL) { delete a; }
      if (b != NULL) { delete b; }
   }

   virtual StateVariableScalarMatrixProductCoef * Clone() const
   {
      return new StateVariableScalarMatrixProductCoef(*a, *b);
   }

   void SetACoef(StateVariableCoef &A) { a = &A; }
   StateVariableCoef * GetACoef() const { return a; }

   void SetBCoef(StateVariableMatCoef &B) { b = &B; }
   StateVariableMatCoef * GetBCoef() const { return b; }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return a->NonTrivialValue(deriv) || b->NonTrivialValue(deriv);
   }

   /// Evaluate the coefficient
   virtual void Eval_Func(DenseMatrix &M, ElementTransformation &T,
                          const IntegrationPoint &ip)
   { b->Eval_Func(M, T, ip); M *= a->Eval_Func(T, ip); }

   virtual void Eval_dNn(DenseMatrix &M, ElementTransformation &T,
                         const IntegrationPoint &ip)
   {
      b->Eval_Func(M, T, ip); M *= a->Eval_dNn(T, ip);
      b->Eval_dNn(dM_, T, ip); dM_ *= a->Eval_Func(T, ip);
      M += dM_;
   }

   virtual void Eval_dNi(DenseMatrix &M, ElementTransformation &T,
                         const IntegrationPoint &ip)
   {
      b->Eval_Func(M, T, ip); M *= a->Eval_dNi(T, ip);
      b->Eval_dNi(dM_, T, ip); dM_ *= a->Eval_Func(T, ip);
      M += dM_;
   }

   virtual void Eval_dVi(DenseMatrix &M, ElementTransformation &T,
                         const IntegrationPoint &ip)
   {
      b->Eval_Func(M, T, ip); M *= a->Eval_dVi(T, ip);
      b->Eval_dVi(dM_, T, ip); dM_ *= a->Eval_Func(T, ip);
      M += dM_;
   }

   virtual void Eval_dTi(DenseMatrix &M, ElementTransformation &T,
                         const IntegrationPoint &ip)
   {
      b->Eval_Func(M, T, ip); M *= a->Eval_dTi(T, ip);
      b->Eval_dTi(dM_, T, ip); dM_ *= a->Eval_Func(T, ip);
      M += dM_;
   }

   virtual void Eval_dTe(DenseMatrix &M, ElementTransformation &T,
                         const IntegrationPoint &ip)
   {
      b->Eval_Func(M, T, ip); M *= a->Eval_dTe(T, ip);
      b->Eval_dTe(dM_, T, ip); dM_ *= a->Eval_Func(T, ip);
      M += dM_;
   }
};

class CoulombLogEICoef : public StateVariableCoef
{
private:
   int zi_;
   Coefficient *TeCoef_;
   Coefficient *neCoef_;

public:
   CoulombLogEICoef(Coefficient &TeCoef, Coefficient &neCoef, int zi)
      : zi_(zi), TeCoef_(&TeCoef), neCoef_(&neCoef) {}

   CoulombLogEICoef(const CoulombLogEICoef &other)
   {
      derivType_ = other.derivType_;
      zi_        = other.zi_;
      TeCoef_    = other.TeCoef_;
      neCoef_    = other.neCoef_;
   }

   virtual CoulombLogEICoef * Clone() const
   {
      return new CoulombLogEICoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE ||
              deriv == ION_DENSITY ||
              deriv == ELECTRON_TEMPERATURE);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double Te = std::max(TeCoef_->Eval(T, ip), 1.0);
      double ne = std::max(neCoef_->Eval(T, ip), 1.0);

      // MFEM_VERIFY(ne > 0.0,
      //          "CoulombLogEICoef::Eval_Func: "
      //          "Electron density (" << ne << ") "
      //          "less than or equal to zero.");
      // MFEM_VERIFY(Te > 0.0,
      //          "CoulombLogEICoef::Eval_Func: "
      //          "Electron temperature (" << Te << ") "
      //          "less than or equal to zero.");

      return lambda_ei(Te, ne, (double)zi_);
   }

   double Eval_dNi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double ne = std::max(neCoef_->Eval(T, ip), 1.0);

      // MFEM_VERIFY(ne > 0.0,
      //           "CoulombLogEICoef::Eval_dNi: "
      //           "Electron density (" << ne << ") "
      //           "less than or equal to zero.");

      return dlambda_ei_dne(1.0, ne, (double)zi_) * (double)zi_;
   }

   double Eval_dTe(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double Te = std::max(TeCoef_->Eval(T, ip), 1.0);

      // MFEM_VERIFY(Te > 0.0,
      //           "CoulombLogEICoef::Eval_dTe: "
      //           "Electron temperature (" << Te << ") "
      //           "less than or equal to zero.");

      return dlambda_ei_dne(Te, 1.0, (double)zi_);
   }
};

class IonCollisionTimeCoef : public StateVariableCoef
{
private:
   int z_i_;
   double m_i_;
   Coefficient &niCoef_;
   Coefficient &TiCoef_;
   StateVariableCoef &lnLambda_;

public:
   IonCollisionTimeCoef(int zi, double mi,
                        Coefficient &niCoef,
                        Coefficient &TiCoef,
                        StateVariableCoef &lnLambda)
      : z_i_(zi), m_i_(mi), niCoef_(niCoef), TiCoef_(TiCoef),
        lnLambda_(lnLambda) {}

   IonCollisionTimeCoef(const IonCollisionTimeCoef &other)
      : niCoef_(other.niCoef_),
        TiCoef_(other.TiCoef_),
        lnLambda_(other.lnLambda_)
   {
      derivType_ = other.derivType_;
      z_i_       = other.z_i_;
      m_i_       = other.m_i_;
   }

   virtual IonCollisionTimeCoef * Clone() const
   {
      return new IonCollisionTimeCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE ||
              deriv == ION_DENSITY ||
              deriv == ION_TEMPERATURE ||
              deriv == ELECTRON_TEMPERATURE);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double Ti = std::max(TiCoef_.Eval(T, ip), 1.0);
      double ni = niCoef_.Eval(T, ip);
      double lnLambda = lnLambda_.Eval(T, ip);

      // MFEM_VERIFY(Ti >= 0.0, "IonCollisionTimeCoef::Eval_Func: "
      //          "Negative temperature found");

      return tau_i(m_i_, (double)z_i_, ni, Ti, lnLambda);
   }

   double Eval_dNi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double Ti = std::max(TiCoef_.Eval(T, ip), 1.0);
      double ni = niCoef_.Eval(T, ip);
      double lnLambda = lnLambda_.Eval(T, ip);
      double dl = lnLambda_.Eval_dNi(T, ip);

      // MFEM_VERIFY(Ti >= 0.0, "IonCollisionTimeCoef::Eval_dNi: "
      //          "Negative temperature found");

      double dtau_dn = dtau_i_dni(m_i_, (double)z_i_, ni, Ti, lnLambda);
      double dtau_dl = dtau_i_dlambda(m_i_, (double)z_i_, ni, Ti, lnLambda);

      return dtau_dn + dtau_dl * dl;
   }

   double Eval_dTi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double Ti = std::max(TiCoef_.Eval(T, ip), 1.0);
      double ni = niCoef_.Eval(T, ip);
      double lnLambda = lnLambda_.Eval(T, ip);

      // MFEM_VERIFY(Ti >= 0.0, "IonCollisionTimeCoef::Eval_dTi: "
      //          "Negative temperature found");

      return dtau_i_dTi(m_i_, (double)z_i_, ni, Ti, lnLambda);
   }

   double Eval_dTe(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double Ti = std::max(TiCoef_.Eval(T, ip), 1.0);
      double ni = niCoef_.Eval(T, ip);
      double lnLambda = lnLambda_.Eval(T, ip);
      double dl = lnLambda_.Eval_dTe(T, ip);

      // MFEM_VERIFY(Ti >= 0.0, "IonCollisionTimeCoef::Eval_dTe: "
      //          "Negative temperature found");

      double dtau_dl = dtau_i_dlambda(m_i_, (double)z_i_, ni, Ti, lnLambda);

      return dtau_dl * dl;
   }
};

class ElectronCollisionTimeCoef : public StateVariableCoef
{
private:
   int z_i_;
   Coefficient &niCoef_;
   Coefficient &TeCoef_;
   StateVariableCoef &lnLambda_;

public:
   ElectronCollisionTimeCoef(int zi,
                             Coefficient &niCoef,
                             Coefficient &TeCoef,
                             StateVariableCoef &lnLambda)
      : z_i_(zi), niCoef_(niCoef), TeCoef_(TeCoef),
        lnLambda_(lnLambda) {}

   ElectronCollisionTimeCoef(const ElectronCollisionTimeCoef &other)
      : niCoef_(other.niCoef_),
        TeCoef_(other.TeCoef_),
        lnLambda_(other.lnLambda_)
   {
      derivType_ = other.derivType_;
      z_i_       = other.z_i_;
   }

   virtual ElectronCollisionTimeCoef * Clone() const
   {
      return new ElectronCollisionTimeCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE ||
              deriv == ION_DENSITY ||
              deriv == ELECTRON_TEMPERATURE);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double Te = std::max(TeCoef_.Eval(T, ip), 1.0);
      double ni = niCoef_.Eval(T, ip);
      double lnLambda = lnLambda_.Eval(T, ip);

      // MFEM_VERIFY(Te >= 0.0, "ElectronCollisionTimeCoef::Eval_Func: "
      //          "Negative temperature found");

      return tau_e(Te, (double)z_i_, ni, lnLambda);
   }

   double Eval_dNi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double Te = std::max(TeCoef_.Eval(T, ip), 1.0);
      double ni = niCoef_.Eval(T, ip);
      double lnLambda = lnLambda_.Eval(T, ip);
      double dl = lnLambda_.Eval_dNi(T, ip);

      // MFEM_VERIFY(Te >= 0.0, "ElectronCollisionTimeCoef::Eval_dNi: "
      //          "Negative temperature found");

      double dtau_dn = dtau_e_dni(Te, (double)z_i_, ni, lnLambda);
      double dtau_dl = dtau_e_dlambda(Te, (double)z_i_, ni, lnLambda);

      return dtau_dn + dtau_dl * dl;
   }

   double Eval_dTe(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double Te = std::max(TeCoef_.Eval(T, ip), 1.0);
      double ni = niCoef_.Eval(T, ip);
      double lnLambda = lnLambda_.Eval(T, ip);
      double dl = lnLambda_.Eval_dTe(T, ip);

      // MFEM_VERIFY(Te >= 0.0, "ElectronCollisionTimeCoef::Eval_dTe: "
      //          "Negative temperature found");

      double dtau_dT = dtau_e_dTe(Te, (double)z_i_, ni, lnLambda);
      double dtau_dl = dtau_e_dlambda(Te, (double)z_i_, ni, lnLambda);

      return dtau_dT + dtau_dl * dl;
   }
};

/** Given the ion and electron temperatures in eV this coefficient returns an
    approximation to the sound speed in m/s.
*/
class SoundSpeedCoef : public StateVariableCoef
{
private:
   double mi_kg_;
   Coefficient *TiCoef_;
   Coefficient *TeCoef_;

public:
   SoundSpeedCoef(double ion_mass_kg, Coefficient &TiCoef, Coefficient &TeCoef)
      : mi_kg_(ion_mass_kg), TiCoef_(&TiCoef), TeCoef_(&TeCoef) {}

   SoundSpeedCoef(const SoundSpeedCoef &other)
   {
      derivType_ = other.derivType_;
      mi_kg_     = other.mi_kg_;
      TiCoef_    = other.TiCoef_;
      TeCoef_    = other.TeCoef_;
   }

   virtual SoundSpeedCoef * Clone() const
   {
      return new SoundSpeedCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE ||
              deriv == ION_TEMPERATURE ||
              deriv == ELECTRON_TEMPERATURE);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double Ti = TiCoef_->Eval(T, ip);
      double Te = TeCoef_->Eval(T, ip);

      // MFEM_VERIFY(Ti + Te >= 0.0,
      //          "SoundSpeedCoef::Eval_Func: Negative temperature found");

      return sqrt(J_per_eV_ * std::max(Ti + Te, 0.0) / mi_kg_);
   }

   double Eval_dTi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double Ti = TiCoef_->Eval(T, ip);
      double Te = TeCoef_->Eval(T, ip);

      // MFEM_VERIFY(Ti + Te >= 0.0,
      //          "SoundSpeedCoef::Eval_dTi: Negative temperature found");

      return 0.5 * sqrt(J_per_eV_ / (mi_kg_ * std::max(Ti + Te, 1.0)));
   }

   double Eval_dTe(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double Ti = TiCoef_->Eval(T, ip);
      double Te = TeCoef_->Eval(T, ip);

      // MFEM_VERIFY(Ti + Te >= 0.0,
      //          "SoundSpeedCoef::Eval_dTe: Negative temperature found");

      return 0.5 * sqrt(J_per_eV_ / (mi_kg_ * std::max(Ti + Te, 1.0)));
   }

};

/** Given the electron temperature in eV this coefficient returns an
    approximation to the expected ionization rate in m^3/s.
*/
class ApproxIonizationRate : public StateVariableCoef
{
private:
   Coefficient *TeCoef_;

public:
   ApproxIonizationRate(Coefficient &TeCoef)
      : TeCoef_(&TeCoef) {}

   ApproxIonizationRate(const ApproxIonizationRate &other)
   {
      derivType_ = other.derivType_;
      TeCoef_    = other.TeCoef_;
   }

   virtual ApproxIonizationRate * Clone() const
   {
      return new ApproxIonizationRate(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE || deriv == ELECTRON_TEMPERATURE);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double Te2 = pow(TeCoef_->Eval(T, ip), 2);

      return 3.0e-16 * Te2 / (3.0 + 0.01 * Te2);
   }

   double Eval_dTe(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double Te = TeCoef_->Eval(T, ip);

      return 2.0 * 3.0 * 3.0e-16 * Te / pow(3.0 + 0.01 * Te * Te, 2);
   }

};

/** Given the electron temperature in eV this coefficient returns an
    approximation to the expected recombination rate in m^3/s.
*/
class ApproxRecombinationRate : public StateVariableCoef
{
private:
   Coefficient *TeCoef_;

public:
   ApproxRecombinationRate(Coefficient &TeCoef)
      : TeCoef_(&TeCoef) {}

   ApproxRecombinationRate(const ApproxRecombinationRate &other)
   {
      derivType_ = other.derivType_;
      TeCoef_    = other.TeCoef_;
   }

   virtual ApproxRecombinationRate * Clone() const
   {
      return new ApproxRecombinationRate(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE || deriv == ELECTRON_TEMPERATURE);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double Te = std::max(TeCoef_->Eval(T, ip), 1.0);

      int    pqn_c = 1;
      int    xsi_c = 2;
      double Ry    = 13.6;
      double chi_c = 13.61;
      double E_ion = Ry;

      // MFEM_VERIFY(Te >= 0.0, "ApproxRecombinationRate::Eval_Func: "
      //          "Negative temperature found");

      return 5.2e-20 / 2.0 * sqrt(E_ion / Te)
             * (1.0 - exp(-chi_c / Te
                          * (1.0 + 1.0 / pqn_c * (xsi_c / pow(pqn_c, 2) - 1.0))))
             * sqrt(pow(log(chi_c / Te), 2) + 2.0);

      // double Te2 = pow(Te, 2);
      // return 3.0e-19 * Te2 / (3.0 + 0.01 * Te2);
   }

   double Eval_dTe(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double Te = std::max(TeCoef_->Eval(T, ip), 1.0);

      // MFEM_VERIFY(Te >= 0.0, "ApproxRecombinationRate::Eval_dTe: "
      //          "Negative temperature found");

      int    pqn_c = 1;
      int    xsi_c = 2;
      double Ry    = 13.6;
      double chi_c = 13.61;
      double E_ion = Ry;

      double a = log(chi_c / Te);
      double b = 2.0 + pow(a, 2);
      double c = exp(-chi_c / Te
                     * (1.0 + 1.0 / pqn_c * (xsi_c / pow(pqn_c, 2) - 1.0)));

      return -5.2e-20  * 0.5 * sqrt(E_ion / (b * pow(Te, 3))) *
             ((a + 0.5 * b) * (1.0 - c)
              + c * (xsi_c - b * chi_c * pqn_c * pqn_c
                     * (1.0 - pqn_c) / (pow(pqn_c, 3) * Te)));

      // return 2.0 * 3.0 * 3.0e-19 * Te / pow(3.0 + 0.01 * Te * Te, 2);
   }

};

/** Given the ion temperature in eV this coefficient returns an
    approximation to the expected charge exchange rate in m^3/s.
*/
class ApproxChargeExchangeRate : public StateVariableCoef
{
private:
   Coefficient *TiCoef_;

public:
   ApproxChargeExchangeRate(Coefficient &TiCoef)
      : TiCoef_(&TiCoef) {}

   ApproxChargeExchangeRate(const ApproxChargeExchangeRate &other)
   {
      derivType_ = other.derivType_;
      TiCoef_    = other.TiCoef_;
   }

   virtual ApproxChargeExchangeRate * Clone() const
   {
      return new ApproxChargeExchangeRate(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE || deriv == ION_TEMPERATURE);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double Ti = std::max(TiCoef_->Eval(T, ip), 0.0);

      // MFEM_VERIFY(Ti >= 0.0, "ApproxChargeExchangeRate::Eval_Func: "
      //            "Negative temperature found");

      return 1.0e-14 * pow(Ti, 0.318);
   }

   double Eval_dTi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double Ti = std::max(TiCoef_->Eval(T, ip), 0.0);

      // MFEM_VERIFY(Ti >= 0.0, "ApproxChargeExchangeRate::Eval_dTi: "
      //            "Negative temperature found");

      return 0.318 * 1.0e-14 * pow(Ti, 0.318 - 1.0);
   }

};

/** Diffusion coefficient used in the Neutral density equation

   Dn = v_n^2 / [3 n_e (<sigma nu>_{iz} + <sigma nu>_{cx})]

   Where:
      v_n is the average speed of the Neutrals often labeled \bar{v}_n and
          given by the formula \bar{v}_n = sqrt{8 T_n / (pi m_n)}
      n_e is the electron density
      <sigma nu>_{iz} is the ionization rate
      <sigma nu>_{cx} is the charge exchange rate
*/
class NeutralDiffusionCoef : public StateVariableCoef
{
private:
   Coefficient       * ne_;
   Coefficient       * vn_;
   Coefficient       * iz_;
   Coefficient       * cx_;

   StateVariableCoef * ne_sv_;
   StateVariableCoef * vn_sv_;
   StateVariableCoef * iz_sv_;
   StateVariableCoef * cx_sv_;

public:
   NeutralDiffusionCoef(Coefficient &neCoef, Coefficient &vnBarCoef,
                        Coefficient &izCoef, Coefficient &cxCoef)
      : ne_(&neCoef), vn_(&vnBarCoef), iz_(&izCoef), cx_(&cxCoef)
   {
      ne_sv_ = dynamic_cast<StateVariableCoef*>(ne_);
      vn_sv_ = dynamic_cast<StateVariableCoef*>(vn_);
      iz_sv_ = dynamic_cast<StateVariableCoef*>(iz_);
      cx_sv_ = dynamic_cast<StateVariableCoef*>(cx_);
   }

   NeutralDiffusionCoef(const NeutralDiffusionCoef &other)
   {
      derivType_ = other.derivType_;
      ne_ = other.ne_;
      vn_ = other.vn_;
      iz_ = other.iz_;
      cx_ = other.cx_;

      ne_sv_ = dynamic_cast<StateVariableCoef*>(ne_);
      vn_sv_ = dynamic_cast<StateVariableCoef*>(vn_);
      iz_sv_ = dynamic_cast<StateVariableCoef*>(iz_);
      cx_sv_ = dynamic_cast<StateVariableCoef*>(cx_);
   }

   virtual NeutralDiffusionCoef * Clone() const
   {
      return new NeutralDiffusionCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      bool dne = (ne_sv_) ? ne_sv_->NonTrivialValue(deriv) : false;
      bool dvn = (vn_sv_) ? vn_sv_->NonTrivialValue(deriv) : false;
      bool diz = (iz_sv_) ? iz_sv_->NonTrivialValue(deriv) : false;
      bool dcx = (cx_sv_) ? cx_sv_->NonTrivialValue(deriv) : false;

      return (deriv == INVALID_FIELD_TYPE || dne || dvn || diz || dcx);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double ne = ne_->Eval(T, ip);
      double vn = vn_->Eval(T, ip);
      double iz = iz_->Eval(T, ip);
      double cx = cx_->Eval(T, ip);

      return vn * vn / (3.0 * ne * (iz + cx));
   }

   double Eval_dFunc(FieldType deriv,
                     ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      bool dne = (ne_sv_) ? ne_sv_->NonTrivialValue(deriv) : false;
      bool dvn = (vn_sv_) ? vn_sv_->NonTrivialValue(deriv) : false;
      bool diz = (iz_sv_) ? iz_sv_->NonTrivialValue(deriv) : false;
      bool dcx = (cx_sv_) ? cx_sv_->NonTrivialValue(deriv) : false;

      if (!dne && !dvn && !diz && !dcx)
      {
         return 0.0;
      }

      double ne = ne_->Eval(T, ip);
      double vn = vn_->Eval(T, ip);
      double iz = iz_->Eval(T, ip);
      double cx = cx_->Eval(T, ip);

      if (ne_sv_) { ne_sv_->SetDerivType(deriv); }
      if (vn_sv_) { vn_sv_->SetDerivType(deriv); }
      if (iz_sv_) { iz_sv_->SetDerivType(deriv); }
      if (cx_sv_) { cx_sv_->SetDerivType(deriv); }

      double dne_df = (ne_sv_) ? ne_sv_->Eval(T, ip) : 0.0;
      double dvn_df = (vn_sv_) ? vn_sv_->Eval(T, ip) : 0.0;
      double diz_df = (iz_sv_) ? iz_sv_->Eval(T, ip) : 0.0;
      double dcx_df = (cx_sv_) ? cx_sv_->Eval(T, ip) : 0.0;

      if (ne_sv_) { ne_sv_->SetDerivType(INVALID_FIELD_TYPE); }
      if (vn_sv_) { vn_sv_->SetDerivType(INVALID_FIELD_TYPE); }
      if (iz_sv_) { iz_sv_->SetDerivType(INVALID_FIELD_TYPE); }
      if (cx_sv_) { cx_sv_->SetDerivType(INVALID_FIELD_TYPE); }

      // vn * vn / (3.0 * ne * (iz + cx));
      return (2.0 * dvn_df - vn * (dne_df / ne + (diz_df + dcx_df) / (iz + cx)))
             * vn / (3.0 * ne * (iz + cx));
   }
};

class IonDiffusionCoef : public StateVariableMatCoef
{
private:
   Coefficient       * Dperp_;
   VectorCoefficient * B3_;

   mutable Vector B_;

public:
   IonDiffusionCoef(Coefficient &DperpCoef, VectorCoefficient &B3Coef)
      : StateVariableMatCoef(2), Dperp_(&DperpCoef), B3_(&B3Coef), B_(3) {}

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE);
   }

   void Eval_Func(DenseMatrix & M,
                  ElementTransformation &T,
                  const IntegrationPoint &ip)
   {
      M.SetSize(2);

      double Dperp = Dperp_->Eval(T, ip);

      B3_->Eval(B_, T, ip);

      double Bmag2 = B_ * B_;

      M(0,0) = (B_[1] * B_[1] + B_[2] * B_[2]) * Dperp / Bmag2;
      M(0,1) = -B_[0] * B_[1] * Dperp / Bmag2;
      M(1,0) = M(0,1);
      M(1,1) = (B_[0] * B_[0] + B_[2] * B_[2]) * Dperp / Bmag2;
   }

};

class IonAdvectionCoef : public StateVariableVecCoef
{
private:
   // double dt_;

   StateVariableCoef &vi_;
   // GridFunctionCoefficient vi0_;
   // GridFunctionCoefficient dvi0_;

   VectorCoefficient * B3_;

   mutable Vector B_;

public:
   IonAdvectionCoef(StateVariableCoef &vi,
                    VectorCoefficient &B3Coef)
      : StateVariableVecCoef(2),
        vi_(vi),
        B3_(&B3Coef), B_(3) {}

   IonAdvectionCoef(const IonAdvectionCoef &other)
      : StateVariableVecCoef(other.vdim),
        vi_(other.vi_),
        B3_(other.B3_),
        B_(3)
   {}

   virtual IonAdvectionCoef * Clone() const
   {
      return new IonAdvectionCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE || deriv == ION_PARA_VELOCITY);
   }

   void Eval_Func(Vector & V,
                  ElementTransformation &T,
                  const IntegrationPoint &ip)
   {
      V.SetSize(2);

      double vi = vi_.Eval(T, ip);

      B3_->Eval(B_, T, ip);

      double Bmag = sqrt(B_ * B_);

      V[0] = vi * B_[0] / Bmag;
      V[1] = vi * B_[1] / Bmag;
   }

   void Eval_dVi(Vector &V, ElementTransformation &T,
                 const IntegrationPoint &ip)
   {
      V.SetSize(2);

      B3_->Eval(B_, T, ip);

      double Bmag = sqrt(B_ * B_);

      V[0] = B_[0] / Bmag;
      V[1] = B_[1] / Bmag;
   }
};

class IonizationSourceCoef : public StateVariableCoef
{
private:
   Coefficient        * ne_;
   Coefficient        * nn_;
   Coefficient        * iz_;

   StateVariableCoef  * ne_sv_;
   StateVariableCoef  * iz_sv_;

   double nn0_;

public:
   IonizationSourceCoef(Coefficient &neCoef, Coefficient &nnCoef,
                        Coefficient &izCoef,
                        double nn0 = 1e10)
      : ne_(&neCoef), nn_(&nnCoef), iz_(&izCoef), nn0_(nn0)
   {
      ne_sv_ = dynamic_cast<StateVariableCoef*>(ne_);
      iz_sv_ = dynamic_cast<StateVariableCoef*>(iz_);
   }

   IonizationSourceCoef(const IonizationSourceCoef &other)
   {
      derivType_ = other.derivType_;
      ne_  = other.ne_;
      nn_  = other.nn_;
      iz_  = other.iz_;
      nn0_ = other.nn0_;

      ne_sv_ = dynamic_cast<StateVariableCoef*>(ne_);
      iz_sv_ = dynamic_cast<StateVariableCoef*>(iz_);
   }

   virtual IonizationSourceCoef * Clone() const
   {
      return new IonizationSourceCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      bool dne = (ne_sv_) ? ne_sv_->NonTrivialValue(deriv) : false;
      bool diz = (iz_sv_) ? iz_sv_->NonTrivialValue(deriv) : false;

      return (deriv == INVALID_FIELD_TYPE || deriv == NEUTRAL_DENSITY || dne || diz);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double nn = nn_->Eval(T, ip);
      if (nn < nn0_) { return 0.0; }

      double ne = ne_->Eval(T, ip);
      double iz = iz_->Eval(T, ip);

      return ne * nn * iz;
   }

   double Eval_dFunc(FieldType deriv, ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      bool dnn = (deriv == NEUTRAL_DENSITY);
      bool dne = (ne_sv_) ? ne_sv_->NonTrivialValue(deriv) : false;
      bool diz = (iz_sv_) ? iz_sv_->NonTrivialValue(deriv) : false;

      if (!dnn && !dne && !diz)
      {
         return 0.0;
      }

      double nn = nn_->Eval(T, ip);
      if (nn < nn0_) { return 0.0; }

      double ne = ne_->Eval(T, ip);
      double iz = iz_->Eval(T, ip);

      if (ne_sv_) { ne_sv_->SetDerivType(deriv); }
      if (iz_sv_) { iz_sv_->SetDerivType(deriv); }

      double dnn_df = (deriv == NEUTRAL_DENSITY) ? 1.0 : 0.0;
      double dne_df = (ne_sv_) ? ne_sv_->Eval(T, ip) : 0.0;
      double diz_df = (iz_sv_) ? iz_sv_->Eval(T, ip) : 0.0;

      if (ne_sv_) { ne_sv_->SetDerivType(INVALID_FIELD_TYPE); }
      if (iz_sv_) { iz_sv_->SetDerivType(INVALID_FIELD_TYPE); }

      //  s_ * ne * nn * iz;
      return dne_df * nn * iz + ne * dnn_df * iz + ne * nn * diz_df;
   }
};

class RecombinationSinkCoef : public StateVariableCoef
{
private:
   Coefficient        * ne_;
   Coefficient        * ni_;
   Coefficient        * rc_;

   StateVariableCoef  * ne_sv_;
   StateVariableCoef  * rc_sv_;

   double ni0_;

public:
   RecombinationSinkCoef(Coefficient &neCoef, Coefficient &niCoef,
                         Coefficient &rcCoef, double ni0 = 1e10)
      : ne_(&neCoef), ni_(&niCoef), rc_(&rcCoef), ni0_(ni0)
   {
      ne_sv_ = dynamic_cast<StateVariableCoef*>(ne_);
      rc_sv_ = dynamic_cast<StateVariableCoef*>(rc_);
   }

   RecombinationSinkCoef(const RecombinationSinkCoef &other)
   {
      derivType_ = other.derivType_;
      ne_ = other.ne_;
      ni_ = other.ni_;
      rc_ = other.rc_;

      ne_sv_ = dynamic_cast<StateVariableCoef*>(ne_);
      rc_sv_ = dynamic_cast<StateVariableCoef*>(rc_);

      ni0_ = other.ni0_;
   }

   virtual RecombinationSinkCoef * Clone() const
   {
      return new RecombinationSinkCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      bool dne = (ne_sv_) ? ne_sv_->NonTrivialValue(deriv) : false;
      bool drc = (rc_sv_) ? rc_sv_->NonTrivialValue(deriv) : false;

      return (deriv == INVALID_FIELD_TYPE ||
              deriv == ION_DENSITY || dne || drc);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double ni = ni_->Eval(T, ip);
      if (ni < ni0_) { return 0.0; }

      double ne = ne_->Eval(T, ip);
      double rc = rc_->Eval(T, ip);

      return ne * ni * rc;
   }

   double Eval_dFunc(FieldType deriv,
                     ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      bool dni = (deriv == ION_DENSITY);
      bool dne = (ne_sv_) ? ne_sv_->NonTrivialValue(deriv) : false;
      bool drc = (rc_sv_) ? rc_sv_->NonTrivialValue(deriv) : false;

      if (!dni && !dne && !drc)
      {
         return 0.0;
      }

      double ni = ni_->Eval(T, ip);
      if (ni < ni0_) { return 0.0; }

      double ne = ne_->Eval(T, ip);
      double rc = rc_->Eval(T, ip);

      if (ne_sv_) { ne_sv_->SetDerivType(deriv); }
      if (rc_sv_) { rc_sv_->SetDerivType(deriv); }

      double dni_df = (deriv == ION_DENSITY) ? 1.0 : 0.0;
      double dne_df = (ne_sv_) ? ne_sv_->Eval(T, ip) : 0.0;
      double drc_df = (rc_sv_) ? rc_sv_->Eval(T, ip) : 0.0;

      if (ne_sv_) { ne_sv_->SetDerivType(INVALID_FIELD_TYPE); }
      if (rc_sv_) { rc_sv_->SetDerivType(INVALID_FIELD_TYPE); }

      /// ne * ni * rc
      return dne_df * ni * rc + ne * dni_df * rc + ne * ni * drc_df;
   }
};

class ChargeExchangeSinkCoef : public StateVariableCoef
{
private:
   Coefficient        & nn_;
   Coefficient        & ni_;
   Coefficient        & cx_;

   StateVariableCoef  * cx_sv_;

public:
   ChargeExchangeSinkCoef(Coefficient &nnCoef, Coefficient &niCoef,
                          Coefficient &cxCoef)
      : nn_(nnCoef), ni_(niCoef), cx_(cxCoef)
   {
      cx_sv_ = dynamic_cast<StateVariableCoef*>(&cx_);
   }

   ChargeExchangeSinkCoef(const ChargeExchangeSinkCoef &other)
      : nn_(other.nn_), ni_(other.ni_), cx_(other.cx_)
   {
      derivType_ = other.derivType_;
      cx_sv_ = dynamic_cast<StateVariableCoef*>(&cx_);
   }

   virtual ChargeExchangeSinkCoef * Clone() const
   {
      return new ChargeExchangeSinkCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      bool dcx = cx_sv_->NonTrivialValue(deriv);

      return (deriv == INVALID_FIELD_TYPE ||
              deriv == ION_DENSITY || dcx);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double nn = nn_.Eval(T, ip);
      double ni = ni_.Eval(T, ip);
      double cx = cx_.Eval(T, ip);

      return nn * ni * cx;
   }

   double Eval_dFunc(FieldType deriv,
                     ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      bool dni = (deriv == ION_DENSITY);
      bool dcx = cx_sv_->NonTrivialValue(deriv);

      if (!dni && !dcx)
      {
         return 0.0;
      }

      double nn = nn_.Eval(T, ip);
      double ni = ni_.Eval(T, ip);
      double cx = cx_.Eval(T, ip);

      double dni_df = (deriv == ION_DENSITY) ? 1.0 : 0.0;
      double dcx_df = cx_sv_->Eval_dFunc(deriv, T, ip);

      /// nn * ni * cx
      return nn * (dni_df * cx + ni * dcx_df);
   }
};

class IonMomentumParaCoef : public StateVariableCoef
{
private:
   double m_i_;
   StateVariableCoef &niCoef_;
   StateVariableCoef &viCoef_;

public:
   IonMomentumParaCoef(double m_i,
                       StateVariableCoef &niCoef,
                       StateVariableCoef &viCoef)
      : m_i_(m_i), niCoef_(niCoef), viCoef_(viCoef) {}

   IonMomentumParaCoef(const IonMomentumParaCoef &other)
      : niCoef_(other.niCoef_),
        viCoef_(other.viCoef_)
   {
      derivType_ = other.derivType_;
      m_i_ = other.m_i_;
   }

   virtual IonMomentumParaCoef * Clone() const
   {
      return new IonMomentumParaCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE ||
              deriv == ION_DENSITY || deriv == ION_PARA_VELOCITY);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double ni = niCoef_.Eval_Func(T, ip);
      double vi = viCoef_.Eval_Func(T, ip);

      return m_i_ * ni * vi;
   }

   double Eval_dNi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double vi = viCoef_.Eval_Func(T, ip);

      return m_i_ * vi;
   }

   double Eval_dVi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double ni = niCoef_.Eval_Func(T, ip);

      return m_i_ * ni;
   }
};

/**
   The momentum equation uses a mass specified in kg and particle
   density in particles per meter cubed. Therefore, the diffusion
   coefficient must have units of kg / (meter * second).
*/
class IonMomentumParaDiffusionCoef : public StateVariableCoef
{
private:
   double z_i_;
   double m_i_kg_;
   const double a_;

   StateVariableCoef & lnLambda_;
   StateVariableCoef & TiCoef_;

   StateVariableCoef & niCoef_;
   StateVariableCoef & CsCoef_;
   VectorCoefficient & B3Coef_;
   ParFiniteElementSpace * fes_;
   Coefficient       * OscCoef_;
   const double width_;

   mutable Vector B3_;
   mutable Vector B2_;
   mutable Vector JB_;

public:
   IonMomentumParaDiffusionCoef(int ion_charge_number, double ion_mass_kg,
                                StateVariableCoef &lnLambda,
                                StateVariableCoef &TiCoef,
                                StateVariableCoef &niCoef,
                                StateVariableCoef &CsCoef,
                                VectorCoefficient &B3Coef,
                                ParFiniteElementSpace * fes,
                                Coefficient * OscCoef,
                                double width)
      : z_i_((double)ion_charge_number), m_i_kg_(ion_mass_kg),
        a_(0.96 * tau_i(m_i_kg_, z_i_, 1.0, 1.0/J_per_eV_, 1.0)),
        lnLambda_(lnLambda),
        TiCoef_(TiCoef),
        niCoef_(niCoef),
        CsCoef_(CsCoef),
        B3Coef_(B3Coef),
        fes_(fes),
        OscCoef_(OscCoef),
        width_(width),
        B3_(3),
        B2_(B3_.GetData(), 2),
        JB_(2)
   {}

   IonMomentumParaDiffusionCoef(const IonMomentumParaDiffusionCoef &other)
      : z_i_(other.z_i_), m_i_kg_(other.m_i_kg_),
        a_(0.96 * tau_i(m_i_kg_, z_i_, 1.0, 1.0/J_per_eV_, 1.0)),
        lnLambda_(other.lnLambda_),
        TiCoef_(other.TiCoef_),
        niCoef_(other.niCoef_),
        CsCoef_(other.CsCoef_),
        B3Coef_(other.B3Coef_),
        fes_(other.fes_),
        OscCoef_(other.OscCoef_),
        width_(other.width_),
        B3_(3),
        B2_(B3_.GetData(), 2),
        JB_(2)
   {
      derivType_ = other.derivType_;
   }

   virtual IonMomentumParaDiffusionCoef * Clone() const
   {
      return new IonMomentumParaDiffusionCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE || deriv == ION_TEMPERATURE);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double lnLambda = lnLambda_.Eval(T, ip);
      double Ti_J = std::max(TiCoef_.Eval_Func(T, ip), 1.0) * J_per_eV_;

      // MFEM_VERIFY(Ti_J >= 0.0, "IonMomentumParaDiffusionCoef::Eval_Func: "
      //          "Negative temperature found");

      double EtaPara = a_ * sqrt(pow(Ti_J, 5)) / lnLambda;

      if (OscCoef_)
      {
         // std::cout << "Element Type: " << T.ElementType
         //           << ", ElementNo: " << T.ElementNo << std::endl;
         int elemOrder = 2;//fes_->GetElementOrder(T.ElementNo);
         double s0 = -4.0 * log10(elemOrder);

         double Se = OscCoef_->Eval(T, ip);
         double se = log10(Se);

         double eps = 0.0;
         if (se >= s0 - width_)
         {
            double ni = niCoef_.Eval(T, ip);
            double Cs = CsCoef_.Eval(T, ip);

            B3Coef_.Eval(B3_, T, ip);
            double B2mag2 = B2_ * B2_;

            JB_.SetSize(2);
            T.Jacobian().MultTranspose(B2_, JB_);
            double h = sqrt((JB_ * JB_) / B2mag2);

            // double h = pow(T.Weight(), 1.0 / T.GetDimension() );

            // std::cout << "Estimates of h: " << h << " (by area) vs " << hJ << " (by J*B)" << std::endl;

            double eps0 = m_i_kg_ * ni * Cs * h / elemOrder;

            if (se > s0 + width_)
            {
               eps = eps0;
            }
            else
            {
               eps = 0.5 * eps0 * (1.0 + sin(0.5 * M_PI * (se - s0) / width_));
            }
         }

         EtaPara += eps;
      }
      return EtaPara;
   }

   double Eval_dTi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double lnLambda = lnLambda_.Eval(T, ip);
      double Ti_J = std::max(TiCoef_.Eval_Func(T, ip), 1.0) * J_per_eV_;

      // MFEM_VERIFY(Ti_J >= 0.0, "IonMomentumParaDiffusionCoef::Eval_dTi: "
      //          "Negative temperature found");

      double dEtaPara = 2.5 * a_ * sqrt(pow(Ti_J, 3)) * J_per_eV_ / lnLambda;
      return dEtaPara;
   }

};

class IonMomentumPerpDiffusionCoef : public StateVariableCoef
{
private:
   double Dperp_;
   double m_i_kg_;

   StateVariableCoef * niCoef_;

public:
   IonMomentumPerpDiffusionCoef(double Dperp, double ion_mass_kg,
                                StateVariableCoef &niCoef)
      : Dperp_(Dperp), m_i_kg_(ion_mass_kg),
        niCoef_(&niCoef)
   {}

   IonMomentumPerpDiffusionCoef(const IonMomentumPerpDiffusionCoef &other)
      : Dperp_(other.Dperp_), m_i_kg_(other.m_i_kg_),
        niCoef_(other.niCoef_)
   {
      derivType_ = other.derivType_;
   }

   virtual IonMomentumPerpDiffusionCoef * Clone() const
   {
      return new IonMomentumPerpDiffusionCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE || deriv == ION_DENSITY);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double ni = niCoef_->Eval_Func(T, ip);
      double EtaPerp = Dperp_ * m_i_kg_ * ni;
      return EtaPerp;
   }

   double Eval_dNi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double dEtaPerp = Dperp_ * m_i_kg_;
      return dEtaPerp;
   }

};
/*
class IonMomentumDiffusionCoef : public StateVariableMatCoef
{
private:
   double zi_;
   double mi_;
   const double lnLambda_;
   const double a_;

   Coefficient       * Dperp_;
   Coefficient       * niCoef_;
   Coefficient       * TiCoef_;
   VectorCoefficient * B3_;

   mutable Vector B_;

public:
   IonMomentumDiffusionCoef(int ion_charge, double ion_mass,
                            Coefficient &DperpCoef,
                            Coefficient &niCoef, Coefficient &TiCoef,
                            VectorCoefficient &B3Coef)
      : StateVariableMatCoef(2), zi_((double)ion_charge), mi_(ion_mass),
        lnLambda_(17.0),
        a_(0.96 * 0.75 * pow(4.0 * M_PI * epsilon0_, 2) *
           sqrt(mi_ * kg_per_amu_ * pow(J_per_eV_, 5) / M_PI) /
           (lnLambda_ * pow(q_ * zi_, 4))),
        Dperp_(&DperpCoef), niCoef_(&niCoef), TiCoef_(&TiCoef),
        B3_(&B3Coef), B_(3) {}

   bool NonTrivialValue(DerivType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE);
   }

   void Eval_Func(DenseMatrix & M,
                  ElementTransformation &T,
                  const IntegrationPoint &ip)
   {
      M.SetSize(2);

      double Dperp = Dperp_->Eval(T, ip);

      double ni = niCoef_->Eval(T, ip);
      double Ti = TiCoef_->Eval(T, ip);

      B3_->Eval(B_, T, ip);

      double Bmag2 = B_ * B_;

      double EtaPerp = mi_ * ni * Dperp;

      M(0,0) = (B_[1] * B_[1] + B_[2] * B_[2]) * EtaPerp / Bmag2;
      M(0,1) = -B_[0] * B_[1] * EtaPerp / Bmag2;
      M(1,0) = M(0,1);
      M(1,1) = (B_[0] * B_[0] + B_[2] * B_[2]) * EtaPerp / Bmag2;

      double EtaPara = a_ * sqrt(pow(Ti, 5));

      M(0,0) += B_[0] * B_[0] * EtaPara / Bmag2;
      M(0,1) += B_[0] * B_[1] * EtaPara / Bmag2;
      M(1,0) += B_[0] * B_[1] * EtaPara / Bmag2;
      M(1,1) += B_[1] * B_[1] * EtaPara / Bmag2;
   }

};
*/
class IonMomentumAdvectionCoef : public StateVariableVecCoef
{
private:
   double mi_;

   StateVariableCoef &ni_;
   StateVariableCoef &vi_;

   GradientGridFunctionCoefficient grad_ni0_;
   GradientGridFunctionCoefficient grad_dni0_;
   double dt_;

   Coefficient       * Dperp_;
   VectorCoefficient * B3_;

   mutable Vector gni_;
   mutable Vector gdni_;

   mutable Vector B_;

public:
   IonMomentumAdvectionCoef(StateVariableCoef &ni,
                            StateVariableCoef &vi,
                            double ion_mass,
                            Coefficient &DperpCoef,
                            VectorCoefficient &B3Coef)
      : StateVariableVecCoef(2),
        mi_(ion_mass),
        ni_(ni),
        vi_(vi),
        grad_ni0_(dynamic_cast<StateVariableGridFunctionCoef*>
                  (dynamic_cast<StateVariableSumCoef&>(ni).GetACoef())->
                  GetGridFunction()),
        grad_dni0_(dynamic_cast<StateVariableGridFunctionCoef*>
                   (dynamic_cast<StateVariableSumCoef&>(ni).GetBCoef())->
                   GetGridFunction()),
        dt_(dynamic_cast<StateVariableSumCoef&>(ni).GetBeta()),
        Dperp_(&DperpCoef),
        B3_(&B3Coef), B_(3)
   {}

   IonMomentumAdvectionCoef(const IonMomentumAdvectionCoef &other)
      : StateVariableVecCoef(other.vdim),
        mi_(other.mi_),
        ni_(other.ni_),
        vi_(other.vi_),
        grad_ni0_(other.grad_ni0_.GetGridFunction()),
        grad_dni0_(other.grad_dni0_.GetGridFunction()),
        dt_(other.dt_),
        B3_(other.B3_),
        B_(3)
   {}

   virtual IonMomentumAdvectionCoef * Clone() const
   {
      return new IonMomentumAdvectionCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE);
   }

   void Eval_Func(Vector & V,
                  ElementTransformation &T,
                  const IntegrationPoint &ip)
   {
      V.SetSize(2);

      double ni = ni_.Eval(T, ip);
      double vi = vi_.Eval(T, ip);

      // double dni0 = dni0_.Eval(T, ip);
      // double dvi0 = dvi0_.Eval(T, ip);

      // double ni1 = ni0 + dt_ * dni0;
      // double vi1 = vi0 + dt_ * dvi0;

      grad_ni0_.Eval(gni_, T, ip);
      grad_dni0_.Eval(gdni_, T, ip);
      // grad_dni0_.Eval(gdni0_, T, ip);
      gni_.Add(dt_, gdni_);

      // gni1_.SetSize(gni0_.Size());
      // add(gni0_, dt_, gdni0_, gni1_);

      double Dperp = Dperp_->Eval(T, ip);

      B3_->Eval(B_, T, ip);

      double Bmag2 = B_ * B_;
      double Bmag = sqrt(Bmag2);

      V[0] = mi_ * (ni * vi * B_[0] / Bmag +
                    Dperp * ((B_[1] * B_[1] + B_[2] * B_[2]) * gni_[0] -
                             B_[0] * B_[1] * gni_[1]) / Bmag2);
      V[1] = mi_ * (ni * vi * B_[1] / Bmag +
                    Dperp * ((B_[0] * B_[0] + B_[2] * B_[2]) * gni_[1] -
                             B_[0] * B_[1] * gni_[0]) / Bmag2);
   }
};

class IonMomentumIonizationCoef : public StateVariableCoef
{
private:
   int zi_;
   double mi_;
   Coefficient &nnCoef_;
   Coefficient &niCoef_;
   Coefficient &vnCoef_;
   StateVariableCoef &izCoef_;

public:
   IonMomentumIonizationCoef(int z_i, double m_i_kg,
                             Coefficient &nnCoef,
                             Coefficient &niCoef,
                             Coefficient &vnCoef,
                             StateVariableCoef &izCoef)
      : zi_(z_i), mi_(m_i_kg),
        nnCoef_(nnCoef), niCoef_(niCoef), vnCoef_(vnCoef), izCoef_(izCoef)
   {}

   IonMomentumIonizationCoef(const IonMomentumIonizationCoef &other)
      : nnCoef_(other.nnCoef_), niCoef_(other.niCoef_),
        vnCoef_(other.vnCoef_), izCoef_(other.izCoef_)
   {
      derivType_ = other.derivType_;
      zi_        = other.zi_;
      mi_        = other.mi_;
   }

   virtual IonMomentumIonizationCoef * Clone() const
   {
      return new IonMomentumIonizationCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE ||
              deriv == NEUTRAL_DENSITY ||
              deriv == ION_DENSITY ||
              izCoef_.NonTrivialValue(deriv) );
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double nn = nnCoef_.Eval(T, ip);
      double ni = niCoef_.Eval(T, ip);
      double vn = vnCoef_.Eval(T, ip);
      double iz = izCoef_.Eval_Func(T, ip);

      return mi_ * zi_ * nn * ni * vn * iz;
   }

   double Eval_dNn(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double nn = nnCoef_.Eval(T, ip);
      double ni = niCoef_.Eval(T, ip);
      double vn = vnCoef_.Eval(T, ip);
      double iz = izCoef_.Eval_Func(T, ip);
      double diz = izCoef_.Eval_dNn(T, ip);

      return mi_ * zi_ * ni * vn * (iz + nn * diz);
   }

   double Eval_dNi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double nn = nnCoef_.Eval(T, ip);
      double ni = niCoef_.Eval(T, ip);
      double vn = vnCoef_.Eval(T, ip);
      double iz = izCoef_.Eval_Func(T, ip);
      double diz = izCoef_.Eval_dNi(T, ip);

      return mi_ * zi_ * vn * nn * (iz + ni * diz);
   }

   double Eval_dVi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double nn = nnCoef_.Eval(T, ip);
      double ni = niCoef_.Eval(T, ip);
      double vn = vnCoef_.Eval(T, ip);
      double diz = izCoef_.Eval_dVi(T, ip);

      return mi_ * zi_ * nn * ni * vn * diz;
   }

   double Eval_dTi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double nn = nnCoef_.Eval(T, ip);
      double ni = niCoef_.Eval(T, ip);
      double vn = vnCoef_.Eval(T, ip);
      double diz = izCoef_.Eval_dTi(T, ip);

      return mi_ * zi_ * nn * ni * vn * diz;
   }

   double Eval_dTe(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double nn = nnCoef_.Eval(T, ip);
      double ni = niCoef_.Eval(T, ip);
      double vn = vnCoef_.Eval(T, ip);
      double diz = izCoef_.Eval_dTe(T, ip);

      return mi_ * zi_ * nn * ni * vn * diz;
   }
};

class IonMomentumRecombinationCoef : public StateVariableCoef
{
private:
   int zi_;
   double mi_;
   Coefficient &niCoef_;
   Coefficient &viCoef_;
   StateVariableCoef &rcCoef_;

public:
   IonMomentumRecombinationCoef(int z_i, double m_i_kg,
                                Coefficient &niCoef,
                                Coefficient &viCoef,
                                StateVariableCoef &rcCoef)
      : zi_(z_i), mi_(m_i_kg),
        niCoef_(niCoef), viCoef_(viCoef), rcCoef_(rcCoef)
   {}

   IonMomentumRecombinationCoef(const IonMomentumRecombinationCoef &other)
      : niCoef_(other.niCoef_), viCoef_(other.viCoef_), rcCoef_(other.rcCoef_)
   {
      derivType_ = other.derivType_;
      zi_        = other.zi_;
      mi_        = other.mi_;
   }

   virtual IonMomentumRecombinationCoef * Clone() const
   {
      return new IonMomentumRecombinationCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE ||
              deriv == ION_DENSITY ||
              deriv == ION_PARA_VELOCITY ||
              rcCoef_.NonTrivialValue(deriv) );
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double ni = niCoef_.Eval(T, ip);
      double vi = viCoef_.Eval(T, ip);
      double rc = rcCoef_.Eval_Func(T, ip);

      return mi_ * zi_ * ni * ni * vi * rc;
   }

   double Eval_dNn(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double ni = niCoef_.Eval(T, ip);
      double vi = viCoef_.Eval(T, ip);
      double drc = rcCoef_.Eval_dNn(T, ip);

      return mi_ * zi_ * ni * ni * vi * drc;
   }

   double Eval_dNi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double ni = niCoef_.Eval(T, ip);
      double vi = viCoef_.Eval(T, ip);
      double rc = rcCoef_.Eval_Func(T, ip);
      double drc = rcCoef_.Eval_dNi(T, ip);

      return mi_ * zi_ * vi * ni * (2.0 * rc + ni * drc);
   }

   double Eval_dVi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double ni = niCoef_.Eval(T, ip);
      double vi = viCoef_.Eval(T, ip);
      double rc = rcCoef_.Eval_Func(T, ip);
      double drc = rcCoef_.Eval_dVi(T, ip);

      return mi_ * zi_ * ni * ni * (rc + vi * drc);
   }

   double Eval_dTi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double ni = niCoef_.Eval(T, ip);
      double vi = viCoef_.Eval(T, ip);
      double drc = rcCoef_.Eval_dTi(T, ip);

      return mi_ * zi_ * ni * ni * vi * drc;
   }

   double Eval_dTe(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double ni = niCoef_.Eval(T, ip);
      double vi = viCoef_.Eval(T, ip);
      double drc = rcCoef_.Eval_dTe(T, ip);

      return mi_ * zi_ * ni * ni * vi * drc;
   }
};

class IonMomentumChargeExchangeCoef : public StateVariableCoef
{
private:
   double mi_;
   Coefficient &nnCoef_;
   Coefficient &niCoef_;
   Coefficient &vnCoef_;
   Coefficient &viCoef_;
   StateVariableCoef &cxCoef_;

public:
   IonMomentumChargeExchangeCoef(double m_i_kg,
                                 Coefficient &nnCoef,
                                 Coefficient &niCoef,
                                 Coefficient &vnCoef,
                                 Coefficient &viCoef,
                                 StateVariableCoef &cxCoef)
      : mi_(m_i_kg), nnCoef_(nnCoef), niCoef_(niCoef),
        vnCoef_(vnCoef), viCoef_(viCoef), cxCoef_(cxCoef)
   {}

   IonMomentumChargeExchangeCoef(const IonMomentumChargeExchangeCoef &other)
      : nnCoef_(other.nnCoef_), niCoef_(other.niCoef_),
        vnCoef_(other.vnCoef_), viCoef_(other.viCoef_),
        cxCoef_(other.cxCoef_)
   {
      derivType_ = other.derivType_;
      mi_        = other.mi_;
   }

   virtual IonMomentumChargeExchangeCoef * Clone() const
   {
      return new IonMomentumChargeExchangeCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE ||
              deriv == NEUTRAL_DENSITY ||
              deriv == ION_DENSITY ||
              deriv == ION_PARA_VELOCITY ||
              cxCoef_.NonTrivialValue(deriv) );
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double nn = nnCoef_.Eval(T, ip);
      double ni = niCoef_.Eval(T, ip);
      double vn = vnCoef_.Eval(T, ip);
      double vi = viCoef_.Eval(T, ip);
      double cx = cxCoef_.Eval_Func(T, ip);

      return mi_ * nn * ni * (vn - vi) * cx;
   }

   double Eval_dNn(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double nn = nnCoef_.Eval(T, ip);
      double ni = niCoef_.Eval(T, ip);
      double vn = vnCoef_.Eval(T, ip);
      double vi = viCoef_.Eval(T, ip);
      double cx = cxCoef_.Eval_Func(T, ip);
      double dcx = cxCoef_.Eval_dNn(T, ip);

      return mi_ * ni * (vn - vi) * (cx + nn * dcx);
   }

   double Eval_dNi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double nn = nnCoef_.Eval(T, ip);
      double ni = niCoef_.Eval(T, ip);
      double vn = vnCoef_.Eval(T, ip);
      double vi = viCoef_.Eval(T, ip);
      double cx = cxCoef_.Eval_Func(T, ip);
      double dcx = cxCoef_.Eval_dNi(T, ip);

      return mi_ * nn * (vn - vi) * (cx + ni * dcx);
   }

   double Eval_dVi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double nn = nnCoef_.Eval(T, ip);
      double ni = niCoef_.Eval(T, ip);
      double vn = vnCoef_.Eval(T, ip);
      double vi = viCoef_.Eval(T, ip);
      double cx = cxCoef_.Eval_Func(T, ip);
      double dcx = cxCoef_.Eval_dVi(T, ip);

      return mi_ * nn * ni * ((vn - vi) * dcx - cx);
   }

   double Eval_dTi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double nn = nnCoef_.Eval(T, ip);
      double ni = niCoef_.Eval(T, ip);
      double vn = vnCoef_.Eval(T, ip);
      double vi = viCoef_.Eval(T, ip);
      double dcx = cxCoef_.Eval_dTi(T, ip);

      return mi_ * nn * ni * (vn - vi) * dcx;
   }

   double Eval_dTe(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double nn = nnCoef_.Eval(T, ip);
      double ni = niCoef_.Eval(T, ip);
      double vn = vnCoef_.Eval(T, ip);
      double vi = viCoef_.Eval(T, ip);
      double dcx = cxCoef_.Eval_dTe(T, ip);

      return mi_ * nn * ni * (vn - vi) * dcx;
   }
};

class StaticPressureCoef : public StateVariableCoef
{
private:
   FieldType fieldType_;
   int z_i_;
   StateVariableCoef &niCoef_;
   StateVariableCoef &TCoef_;

public:
   StaticPressureCoef(StateVariableCoef &niCoef,
                      StateVariableCoef &TCoef)
      : fieldType_(ION_TEMPERATURE),
        z_i_(1), niCoef_(niCoef), TCoef_(TCoef) {}

   StaticPressureCoef(int z_i,
                      StateVariableCoef &niCoef,
                      StateVariableCoef &TCoef)
      : fieldType_(ELECTRON_TEMPERATURE),
        z_i_(z_i), niCoef_(niCoef), TCoef_(TCoef) {}

   StaticPressureCoef(const StaticPressureCoef &other)
      : niCoef_(other.niCoef_),
        TCoef_(other.TCoef_)
   {
      derivType_ = other.derivType_;
      fieldType_ = other.fieldType_;
      z_i_       = other.z_i_;
   }

   virtual StaticPressureCoef * Clone() const
   {
      return new StaticPressureCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE ||
              deriv == ION_DENSITY || deriv == fieldType_);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double ni = niCoef_.Eval(T, ip);
      double Ts = TCoef_.Eval(T, ip) * J_per_eV_;

      return 1.5 * z_i_ * ni * Ts;
   }

   double Eval_dNi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double Ts = TCoef_.Eval(T, ip) * J_per_eV_;

      return 1.5 * z_i_ * Ts;
   }

   double Eval_dTi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      if (fieldType_ == ION_TEMPERATURE)
      {
         double ni = niCoef_.Eval(T, ip);

         return 1.5 * z_i_ * ni * J_per_eV_;
      }
      else
      {
         return 0.0;
      }
   }

   double Eval_dTe(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      if (fieldType_ == ELECTRON_TEMPERATURE)
      {
         double ni = niCoef_.Eval(T, ip);

         return 1.5 * z_i_ * ni * J_per_eV_;
      }
      else
      {
         return 0.0;
      }
   }
};

class StaticPressureAdvectionCoef : public StateVariableVecCoef
{
private:
   double a_;

   StateVariableCoef &ni_;
   StateVariableCoef &vi_;

   VectorCoefficient * B3_;

   mutable Vector B_;

public:
   StaticPressureAdvectionCoef(StateVariableCoef &ni,
                               StateVariableCoef &vi,
                               double a,
                               VectorCoefficient &B3Coef)
      : StateVariableVecCoef(2),
        a_(a),
        ni_(ni),
        vi_(vi),
        B3_(&B3Coef), B_(3)
   {}

   StaticPressureAdvectionCoef(const StaticPressureAdvectionCoef &other)
      : StateVariableVecCoef(other.vdim),
        a_(other.a_),
        ni_(other.ni_),
        vi_(other.vi_),
        B3_(other.B3_),
        B_(3)
   {}

   virtual StaticPressureAdvectionCoef * Clone() const
   {
      return new StaticPressureAdvectionCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE);
   }

   void Eval_Func(Vector & V,
                  ElementTransformation &T,
                  const IntegrationPoint &ip)
   {
      V.SetSize(2);

      double ni = ni_.Eval(T, ip);
      double vi = vi_.Eval(T, ip);

      B3_->Eval(B_, T, ip);

      double Bmag2 = B_ * B_;
      double Bmag = sqrt(Bmag2);

      V[0] = a_ * (ni * vi * B_[0] / Bmag);
      V[1] = a_ * (ni * vi * B_[1] / Bmag);
   }
};

class TotalEnergyCoef : public StateVariableCoef
{
private:
   FieldType fieldType_;
   int z_i_;
   double m_kg_;
   StateVariableCoef &niCoef_;
   StateVariableCoef &viCoef_;
   StateVariableCoef &TCoef_;

public:
   TotalEnergyCoef(double m_kg,
                   StateVariableCoef &niCoef,
                   StateVariableCoef &viCoef,
                   StateVariableCoef &TCoef)
      : fieldType_(ION_TEMPERATURE),
        z_i_(1), m_kg_(m_kg), niCoef_(niCoef), viCoef_(viCoef), TCoef_(TCoef)
   {}

   TotalEnergyCoef(int z_i,
                   double m_kg,
                   StateVariableCoef &niCoef,
                   StateVariableCoef &viCoef,
                   StateVariableCoef &TCoef)
      : fieldType_(ELECTRON_TEMPERATURE),
        z_i_(z_i), m_kg_(m_kg), niCoef_(niCoef), viCoef_(viCoef), TCoef_(TCoef)
   {}

   TotalEnergyCoef(const TotalEnergyCoef &other)
      : niCoef_(other.niCoef_),
        viCoef_(other.viCoef_),
        TCoef_(other.TCoef_)
   {
      derivType_ = other.derivType_;
      fieldType_ = other.fieldType_;
      z_i_       = other.z_i_;
      m_kg_      = other.m_kg_;
   }

   virtual TotalEnergyCoef * Clone() const
   {
      return new TotalEnergyCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE ||
              deriv == ION_DENSITY ||
              deriv == ION_PARA_VELOCITY || deriv == fieldType_);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double ni  = niCoef_.Eval(T, ip);
      double vi  = viCoef_.Eval(T, ip);
      double T_J = TCoef_.Eval(T, ip) * J_per_eV_;

      return 0.5 * z_i_ * ni * (3.0 * T_J + m_kg_ * vi * vi);
   }

   double Eval_dNi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double vi = viCoef_.Eval(T, ip);
      double T_J = TCoef_.Eval(T, ip) * J_per_eV_;

      return 0.5 * z_i_ * (3.0 * T_J + m_kg_ * vi * vi);
   }

   double Eval_dVi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double ni = niCoef_.Eval(T, ip);
      double vi = viCoef_.Eval(T, ip);

      return z_i_ * ni * m_kg_ * vi;
   }

   double Eval_dTi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      if (fieldType_ == ION_TEMPERATURE)
      {
         double ni = niCoef_.Eval(T, ip);

         return 1.5 * z_i_ * ni * J_per_eV_;
      }
      else
      {
         return 0.0;
      }
   }

   double Eval_dTe(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      if (fieldType_ == ELECTRON_TEMPERATURE)
      {
         double ni = niCoef_.Eval(T, ip);

         return 1.5 * z_i_ * ni * J_per_eV_;
      }
      else
      {
         return 0.0;
      }
   }
};

class KineticEnergyCoef : public StateVariableCoef
{
private:
   int z_i_;
   double m_kg_;
   StateVariableCoef &niCoef_;
   StateVariableCoef &viCoef_;

public:
   KineticEnergyCoef(double m_kg,
                     StateVariableCoef &niCoef,
                     StateVariableCoef &viCoef)
      : z_i_(1), m_kg_(m_kg), niCoef_(niCoef), viCoef_(viCoef) {}

   KineticEnergyCoef(int z_i,
                     double m_kg,
                     StateVariableCoef &niCoef,
                     StateVariableCoef &viCoef)
      : z_i_(z_i), m_kg_(m_kg), niCoef_(niCoef), viCoef_(viCoef) {}

   KineticEnergyCoef(const KineticEnergyCoef &other)
      : niCoef_(other.niCoef_),
        viCoef_(other.viCoef_)
   {
      derivType_ = other.derivType_;
      z_i_       = other.z_i_;
      m_kg_      = other.m_kg_;
   }

   virtual KineticEnergyCoef * Clone() const
   {
      return new KineticEnergyCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE ||
              deriv == ION_DENSITY ||
              deriv == ION_PARA_VELOCITY);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double ni = niCoef_.Eval(T, ip);
      double vi = viCoef_.Eval(T, ip);

      return 0.5 * z_i_ * ni * m_kg_ * vi * vi;
   }

   double Eval_dNi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double vi = viCoef_.Eval(T, ip);

      return 0.5 * z_i_ * m_kg_ * vi * vi;
   }

   double Eval_dVi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double ni = niCoef_.Eval(T, ip);
      double vi = viCoef_.Eval(T, ip);

      return z_i_ * ni * m_kg_ * vi;
   }
};

class TotalEnergyAdvectionCoef : public StateVariableVecCoef
{
private:
   FieldType fieldType_;
   int z_i_;
   double m_kg_;
   StateVariableCoef &niCoef_;
   StateVariableCoef &viCoef_;
   StateVariableCoef &TCoef_;

   VectorCoefficient * B3_;

   mutable Vector B_;

public:
   TotalEnergyAdvectionCoef(double m_kg,
                            StateVariableCoef &niCoef,
                            StateVariableCoef &viCoef,
                            StateVariableCoef &TCoef,
                            VectorCoefficient &B3Coef)
      : StateVariableVecCoef(2),
        fieldType_(ION_TEMPERATURE),
        z_i_(1), m_kg_(m_kg), niCoef_(niCoef), viCoef_(viCoef), TCoef_(TCoef),
        B3_(&B3Coef), B_(3)
   {}

   TotalEnergyAdvectionCoef(int z_i,
                            double m_kg,
                            StateVariableCoef &niCoef,
                            StateVariableCoef &viCoef,
                            StateVariableCoef &TCoef,
                            VectorCoefficient &B3Coef)
      : StateVariableVecCoef(2),
        fieldType_(ELECTRON_TEMPERATURE),
        z_i_(z_i), m_kg_(m_kg), niCoef_(niCoef), viCoef_(viCoef), TCoef_(TCoef),
        B3_(&B3Coef), B_(3)
   {}

   TotalEnergyAdvectionCoef(const TotalEnergyAdvectionCoef &other)
      : StateVariableVecCoef(other.vdim),
        niCoef_(other.niCoef_),
        viCoef_(other.viCoef_),
        TCoef_(other.TCoef_),
        B3_(other.B3_),
        B_(3)
   {
      derivType_ = other.derivType_;
      fieldType_ = other.fieldType_;
      z_i_       = other.z_i_;
      m_kg_      = other.m_kg_;
   }

   virtual TotalEnergyAdvectionCoef * Clone() const
   {
      return new TotalEnergyAdvectionCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE ||
              deriv == ION_DENSITY ||
              deriv == ION_PARA_VELOCITY || deriv == fieldType_);
   }

   void Eval_Func(Vector & V,
                  ElementTransformation &T,
                  const IntegrationPoint &ip)
   {
      V.SetSize(2);

      double ni  = niCoef_.Eval(T, ip);
      double vi  = viCoef_.Eval(T, ip);
      double T_J = TCoef_.Eval(T, ip) * J_per_eV_;

      B3_->Eval(B_, T, ip);

      double Bmag2 = B_ * B_;
      double Bmag = sqrt(Bmag2);

      V[0] = B_[0] / Bmag;
      V[1] = B_[1] / Bmag;
      V *= 0.5 * z_i_ * ni * vi * (5.0 * T_J + m_kg_ * vi * vi);
   }

   void Eval_dNi(Vector & V,
                 ElementTransformation &T,
                 const IntegrationPoint &ip)
   {
      V.SetSize(2);

      double vi  = viCoef_.Eval(T, ip);
      double T_J = TCoef_.Eval(T, ip) * J_per_eV_;

      B3_->Eval(B_, T, ip);

      double Bmag2 = B_ * B_;
      double Bmag = sqrt(Bmag2);

      V[0] = B_[0] / Bmag;
      V[1] = B_[1] / Bmag;
      V *= 0.5 * z_i_ * vi * (5.0 * T_J + m_kg_ * vi * vi);
   }

   void Eval_dVi(Vector & V,
                 ElementTransformation &T,
                 const IntegrationPoint &ip)
   {
      V.SetSize(2);

      double ni  = niCoef_.Eval(T, ip);
      double vi  = viCoef_.Eval(T, ip);
      double T_J = TCoef_.Eval(T, ip) * J_per_eV_;

      B3_->Eval(B_, T, ip);

      double Bmag2 = B_ * B_;
      double Bmag = sqrt(Bmag2);

      V[0] = B_[0] / Bmag;
      V[1] = B_[1] / Bmag;
      V *= 0.5 * z_i_ * ni * (5.0 * T_J + 3.0 * m_kg_ * vi * vi);
   }

   void Eval_dTi(Vector & V,
                 ElementTransformation &T,
                 const IntegrationPoint &ip)
   {
      V.SetSize(2);

      if (fieldType_ == ION_TEMPERATURE)
      {
         double ni = niCoef_.Eval(T, ip);
         double vi = viCoef_.Eval(T, ip);

         B3_->Eval(B_, T, ip);

         double Bmag2 = B_ * B_;
         double Bmag = sqrt(Bmag2);

         V[0] = B_[0] / Bmag;
         V[1] = B_[1] / Bmag;
         V *= 2.5 * z_i_ * ni * vi * J_per_eV_;
      }
      else
      {
         V = 0.0;
      }
   }

   void Eval_dTe(Vector & V,
                 ElementTransformation &T,
                 const IntegrationPoint &ip)
   {
      V.SetSize(2);

      if (fieldType_ == ELECTRON_TEMPERATURE)
      {
         double ni = niCoef_.Eval(T, ip);
         double vi = viCoef_.Eval(T, ip);

         B3_->Eval(B_, T, ip);

         double Bmag2 = B_ * B_;
         double Bmag = sqrt(Bmag2);

         V[0] = B_[0] / Bmag;
         V[1] = B_[1] / Bmag;
         V *= 2.5 * z_i_ * ni * vi * J_per_eV_;
      }
      else
      {
         V = 0.0;
      }
   }
};

class IonElectronHeatExchangeCoef : public StateVariableCoef
{
private:
   int z_i_;
   double m_i_kg_;
   StateVariableCoef &lnLambda_;
   StateVariableCoef &niCoef_;
   StateVariableCoef &TiCoef_;
   StateVariableCoef &TeCoef_;

public:
   IonElectronHeatExchangeCoef(int z_i, double m_i_kg,
                               StateVariableCoef &lnLambda,
                               StateVariableCoef &niCoef,
                               StateVariableCoef &TiCoef,
                               StateVariableCoef &TeCoef)
      : z_i_(z_i), m_i_kg_(m_i_kg), lnLambda_(lnLambda),
        niCoef_(niCoef), TiCoef_(TiCoef), TeCoef_(TeCoef) {}

   IonElectronHeatExchangeCoef(const IonElectronHeatExchangeCoef &other)
      : lnLambda_(other.lnLambda_),
        niCoef_(other.niCoef_),
        TiCoef_(other.TiCoef_),
        TeCoef_(other.TeCoef_)
   {
      derivType_ = other.derivType_;
      z_i_       = other.z_i_;
      m_i_kg_    = other.m_i_kg_;
   }

   virtual IonElectronHeatExchangeCoef * Clone() const
   {
      return new IonElectronHeatExchangeCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE ||
              deriv == ION_DENSITY ||
              deriv == ION_TEMPERATURE ||
              deriv == ELECTRON_TEMPERATURE);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double lnLambda = lnLambda_.Eval_Func(T, ip);
      double ni = niCoef_.Eval(T, ip);
      double Ti = std::max(TiCoef_.Eval(T, ip), 1.0);
      double Te = std::max(TeCoef_.Eval(T, ip), 1.0);

      // MFEM_VERIFY(Te >= 0.0, "IonElectronHeatExchangeCoef::Eval_Func: "
      //          "Negative temperature found");

      double tau = tau_e(Te, z_i_, ni, lnLambda);

      return 3.0 * (Te - Ti) * J_per_eV_ * me_kg_ * z_i_ * ni / (m_i_kg_ * tau);
   }

   double Eval_dNi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double lnLambda = lnLambda_.Eval_Func(T, ip);
      double dl = lnLambda_.Eval_dNi(T, ip);
      double ni = niCoef_.Eval(T, ip);
      double Ti = std::max(TiCoef_.Eval(T, ip), 1.0);
      double Te = std::max(TeCoef_.Eval(T, ip), 1.0);

      // MFEM_VERIFY(Te >= 0.0, "IonElectronHeatExchangeCoef::Eval_dNi: "
      //          "Negative temperature found");

      double tau = tau_e(Te, z_i_, ni, lnLambda);
      double dtau_dn = dtau_e_dni(Te, z_i_, ni, lnLambda);
      double dtau_dl = dtau_e_dlambda(Te, z_i_, ni, lnLambda);

      double a = 3.0 * (Te - Ti) * J_per_eV_ * me_kg_ * z_i_ / m_i_kg_;

      return a * (tau - ni * (dtau_dn + dtau_dl * dl)) / (tau * tau);
   }

   double Eval_dTi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double lnLambda = lnLambda_.Eval(T, ip);
      double ni = niCoef_.Eval(T, ip);
      double Te = std::max(TeCoef_.Eval(T, ip), 1.0);

      // MFEM_VERIFY(Te >= 0.0, "IonElectronHeatExchangeCoef::Eval_dTi: "
      //          "Negative temperature found");

      double tau = tau_e(Te, z_i_, ni, lnLambda);

      return -3.0 * J_per_eV_ * me_kg_ * z_i_ * ni / (m_i_kg_ * tau);
   }

   double Eval_dTe(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double lnLambda = lnLambda_.Eval_Func(T, ip);
      double dl = lnLambda_.Eval_dTe(T, ip);
      double ni = niCoef_.Eval(T, ip);
      double Ti = std::max(TiCoef_.Eval(T, ip), 1.0);
      double Te = std::max(TeCoef_.Eval(T, ip), 1.0);

      // MFEM_VERIFY(Te >= 0.0, "IonElectronHeatExchangeCoef::Eval_dTe: "
      //          "Negative temperature found");

      double tau = tau_e(Te, z_i_, ni, lnLambda);
      double dtau_dT = dtau_e_dTe(Te, z_i_, ni, lnLambda);
      double dtau_dl = dtau_e_dlambda(Te, z_i_, ni, lnLambda);

      double a = 3.0 * me_kg_ * z_i_ * ni * J_per_eV_ / m_i_kg_;

      return a * (tau - (Te - Ti) * (dtau_dT + dtau_dl * dl)) / (tau * tau);
   }

};

class IonThermalParaDiffusionCoef : public StateVariableCoef
{
private:
   // double lnLambda_;
   double z_i_;
   double m_i_kg_;

   Coefficient * lnLambda_;
   Coefficient * niCoef_;
   Coefficient * TiCoef_;

public:
   IonThermalParaDiffusionCoef(double ion_charge,
                               double ion_mass_kg,
                               Coefficient &lnLambda,
                               Coefficient &niCoef,
                               Coefficient &TiCoef)
      : StateVariableCoef(),
        z_i_(ion_charge),
        m_i_kg_(ion_mass_kg),
        lnLambda_(&lnLambda), niCoef_(&niCoef), TiCoef_(&TiCoef)
   {}

   IonThermalParaDiffusionCoef(const IonThermalParaDiffusionCoef &other)
   {
      derivType_ = other.derivType_;
      z_i_       = other.z_i_;
      m_i_kg_    = other.m_i_kg_;
      lnLambda_  = other.lnLambda_;
      niCoef_    = other.niCoef_;
      TiCoef_    = other.TiCoef_;

   }

   virtual IonThermalParaDiffusionCoef * Clone() const
   {
      return new IonThermalParaDiffusionCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double ni  = std::max(niCoef_->Eval(T, ip), 1.0);
      double Ti_eV = std::max(TiCoef_->Eval(T, ip), 1.0);
      // MFEM_VERIFY(ni > 0.0,
      //          "Ion density (" << ni << ") "
      //          "less than or equal to zero in "
      //          "IonThermalParaDiffusionCoef.");
      // MFEM_VERIFY(Ti_eV > 0.0,
      //          "Ion temperature (" << Ti_eV << ") "
      //          "less than or equal to zero in "
      //          "IonThermalParaDiffusionCoef.");

      double lnLambda = lnLambda_->Eval(T, ip);
      double tau = tau_i(m_i_kg_, z_i_, ni, Ti_eV, lnLambda);
      // std::cout << "Chi_e parallel: " << 3.16 * ne * Te * J_per_eV_ * tau / me_kg_
      // << ", n_e: " << ne << ", T_e: " << Te << std::endl;
      return 3.9 * Ti_eV * J_per_eV_ * tau / m_i_kg_;
   }

};

class IonEnergyIonizationCoef : public StateVariableCoef
{
private:
   double mn_;
   double mi_;
   Coefficient &vnCoef_;
   Coefficient &TnCoef_;
   Coefficient &SizCoef_;
   StateVariableCoef *Siz_sv_;

public:
   IonEnergyIonizationCoef(double m_n_kg, double m_i_kg,
                           Coefficient &vnCoef,
                           Coefficient &TnCoef,
                           Coefficient &SizCoef)
      : mn_(m_n_kg), mi_(m_i_kg),
        vnCoef_(vnCoef), TnCoef_(TnCoef), SizCoef_(SizCoef)
   {
      Siz_sv_ = dynamic_cast<StateVariableCoef*>(&SizCoef_);
   }

   IonEnergyIonizationCoef(const IonEnergyIonizationCoef &other)
      : vnCoef_(other.vnCoef_), TnCoef_(other.TnCoef_),
        SizCoef_(other.SizCoef_)
   {
      derivType_ = other.derivType_;
      mn_        = other.mn_;
      mi_        = other.mi_;

      Siz_sv_ = dynamic_cast<StateVariableCoef*>(&SizCoef_);
   }

   virtual IonEnergyIonizationCoef * Clone() const
   {
      return new IonEnergyIonizationCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      bool diz = (Siz_sv_) ? Siz_sv_->NonTrivialValue(deriv) : false;
      return (deriv == INVALID_FIELD_TYPE || diz);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double vn = vnCoef_.Eval(T, ip);
      double Tn = TnCoef_.Eval(T, ip) * J_per_eV_;
      double Siz = SizCoef_.Eval(T, ip);

      return 0.5 * mi_ * Siz * (3.0 * Tn / mn_ + vn * vn);
   }

   double Eval_dFunc(FieldType deriv,
                     ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      bool diz = (Siz_sv_) ? Siz_sv_->NonTrivialValue(deriv) : false;
      if (!diz) { return 0.0; }

      double vn = vnCoef_.Eval(T, ip);
      double Tn = TnCoef_.Eval(T, ip) * J_per_eV_;
      double dSiz = Siz_sv_->Eval_dFunc(deriv, T, ip);

      return 0.5 * mi_ * dSiz * (3.0 * Tn / mn_ + vn * vn);
   }
};

class IonEnergyRecombinationCoef : public StateVariableCoef
{
private:
   double mi_;
   Coefficient &viCoef_;
   Coefficient &SrcCoef_;
   StateVariableCoef *Src_sv_;

public:
   IonEnergyRecombinationCoef(double m_i_kg,
                              Coefficient &viCoef,
                              Coefficient &SrcCoef)
      : mi_(m_i_kg),
        viCoef_(viCoef), SrcCoef_(SrcCoef)
   {
      Src_sv_ = dynamic_cast<StateVariableCoef*>(&SrcCoef_);
   }

   IonEnergyRecombinationCoef(const IonEnergyRecombinationCoef &other)
      : viCoef_(other.viCoef_), SrcCoef_(other.SrcCoef_)
   {
      derivType_ = other.derivType_;
      mi_        = other.mi_;

      Src_sv_ = dynamic_cast<StateVariableCoef*>(&SrcCoef_);
   }

   virtual IonEnergyRecombinationCoef * Clone() const
   {
      return new IonEnergyRecombinationCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      bool drc = (Src_sv_) ? Src_sv_->NonTrivialValue(deriv) : false;
      return (deriv == INVALID_FIELD_TYPE ||
              deriv == ION_PARA_VELOCITY ||
              drc);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double vi = viCoef_.Eval(T, ip);
      double Src = SrcCoef_.Eval(T, ip);

      return 0.5 * mi_ * Src * vi * vi;
   }

   double Eval_dFunc(FieldType deriv,
                     ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      bool drc = (Src_sv_) ? Src_sv_->NonTrivialValue(deriv) : false;
      if (!drc || deriv == ION_PARA_VELOCITY)
      { return 0.0; }

      double vi = viCoef_.Eval(T, ip);
      double dvi = (deriv == ION_PARA_VELOCITY) ? 1.0 : 0.0;
      double Src = SrcCoef_.Eval(T, ip);
      double dSrc = (Src_sv_) ? Src_sv_->Eval_dFunc(deriv, T, ip) : 0.0;

      return 0.5 * mi_ * vi * (dSrc * vi + 2.0 * Src * dvi);
   }
};

class IonEnergyChargeExchangeCoef : public StateVariableCoef
{
private:
   double mi_;
   Coefficient &vnCoef_;
   Coefficient &viCoef_;
   Coefficient &ScxCoef_;
   StateVariableCoef *Scx_sv_;

public:
   IonEnergyChargeExchangeCoef(double m_i_kg,
                               Coefficient &vnCoef,
                               Coefficient &viCoef,
                               Coefficient &ScxCoef)
      : mi_(m_i_kg),
        vnCoef_(vnCoef), viCoef_(viCoef), ScxCoef_(ScxCoef)
   {
      Scx_sv_ = dynamic_cast<StateVariableCoef*>(&ScxCoef_);
   }

   IonEnergyChargeExchangeCoef(const IonEnergyChargeExchangeCoef &other)
      : vnCoef_(other.vnCoef_), viCoef_(other.viCoef_), ScxCoef_(other.ScxCoef_)
   {
      derivType_ = other.derivType_;
      mi_        = other.mi_;

      Scx_sv_ = dynamic_cast<StateVariableCoef*>(&ScxCoef_);
   }

   virtual IonEnergyChargeExchangeCoef * Clone() const
   {
      return new IonEnergyChargeExchangeCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      bool dcx = (Scx_sv_) ? Scx_sv_->NonTrivialValue(deriv) : false;
      return (deriv == INVALID_FIELD_TYPE ||
              deriv == ION_PARA_VELOCITY ||
              dcx );
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double vn = vnCoef_.Eval(T, ip);
      double vi = viCoef_.Eval(T, ip);
      double Scx = ScxCoef_.Eval(T, ip);

      return 0.5 * mi_ * Scx * (vn * vn - vi * vi);
   }

   double Eval_dFunc(FieldType deriv,
                     ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      bool dcx = (Scx_sv_) ? Scx_sv_->NonTrivialValue(deriv) : false;
      if (!dcx || deriv == ION_PARA_VELOCITY)
      { return 0.0; }

      double vn = vnCoef_.Eval(T, ip);
      double vi = viCoef_.Eval(T, ip);
      double dvi = (deriv == ION_PARA_VELOCITY) ? 1.0 : 0.0;
      double Scx = ScxCoef_.Eval(T, ip);
      double dScx = (Scx_sv_) ? Scx_sv_->Eval_dFunc(deriv, T, ip) : 0.0;

      return 0.5 * mi_ * (dScx * (vn * vn - vi * vi) - 2.0 * Scx * vi * dvi);
   }
};

class ElectronThermalParaDiffusionCoef : public StateVariableCoef
{
private:
   double z_i_;

   Coefficient *lnLambda_;
   Coefficient *neCoef_;
   Coefficient *TeCoef_;

public:
   ElectronThermalParaDiffusionCoef(double ion_charge,
                                    Coefficient &lnLambda,
                                    Coefficient &neCoef,
                                    Coefficient &TeCoef,
                                    FieldType deriv = INVALID_FIELD_TYPE)
      : StateVariableCoef(deriv),
        z_i_(ion_charge), lnLambda_(&lnLambda),
        neCoef_(&neCoef), TeCoef_(&TeCoef)
   {}

   ElectronThermalParaDiffusionCoef(
      const ElectronThermalParaDiffusionCoef &other)
   {
      derivType_ = other.derivType_;
      z_i_       = other.z_i_;
      lnLambda_  = other.lnLambda_;
      neCoef_    = other.neCoef_;
      TeCoef_    = other.TeCoef_;
   }

   virtual ElectronThermalParaDiffusionCoef * Clone() const
   {
      return new ElectronThermalParaDiffusionCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE || deriv == ELECTRON_TEMPERATURE);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double ne    = std::max(neCoef_->Eval(T, ip), 1.0);
      double Te_eV = std::max(TeCoef_->Eval(T, ip), 1.0);
      // MFEM_VERIFY(ne > 0.0,
      //          "ElectronThermalParaDiffusionCoef::Eval_Func: "
      //          "Electron density (" << ne << ") "
      //          "less than or equal to zero.");
      // MFEM_VERIFY(Te_eV > 0.0,
      //          "ElectronThermalParaDiffusionCoef::Eval_Func: "
      //          "Electron temperature (" << Te_eV << ") "
      //          "less than or equal to zero.");

      double lnLambda = lnLambda_->Eval(T, ip);
      double tau = tau_e(Te_eV, z_i_, ne, lnLambda);
      // std::cout << "Chi_e parallel: " << 3.16 * ne * Te * J_per_eV_ * tau / me_kg_
      // << ", n_e: " << ne << ", T_e: " << Te << std::endl;
      return 3.16 * Te_eV * J_per_eV_ * tau / me_kg_;
   }
   /*
   double Eval_dTe(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      double ne = neCoef_->Eval(T, ip);
      double Te = TeCoef_->Eval(T, ip);
      MFEM_VERIFY(ne > 0.0,
                  "Electron density (" << ne << ") "
                  "less than or equal to zero in "
                  "ElectronThermalParaDiffusionCoef.");
      MFEM_VERIFY(Te > 0.0,
                  "Electron temperature (" << Te << ") "
                  "less than or equal to zero in "
                  "ElectronThermalParaDiffusionCoef.");

      double tau = tau_e(Te, z_i_, ne, 17.0);
      double dtau = dtau_e_dT(Te, z_i_, ne, 17.0);
      // std::cout << "Chi_e parallel: " << 3.16 * ne * Te * J_per_eV_ * tau / me_kg_
      // << ", n_e: " << ne << ", T_e: " << Te << std::endl;
      return 3.16 * ne * J_per_eV_ * (tau + Te * dtau)/ me_kg_;
   }
   */
};

class ElectronEnergyIonizationCoef : public StateVariableCoef
{
private:
   double mn_;
   Coefficient &phiIZCoef_;
   Coefficient &vnCoef_;
   Coefficient &TnCoef_;
   Coefficient &SizCoef_;
   StateVariableCoef *Siz_sv_;

public:
   ElectronEnergyIonizationCoef(double m_n_kg,
                                Coefficient &phiIZCoef,
                                Coefficient &vnCoef,
                                Coefficient &TnCoef,
                                Coefficient &SizCoef)
      : mn_(m_n_kg), phiIZCoef_(phiIZCoef), vnCoef_(vnCoef),
        TnCoef_(TnCoef), SizCoef_(SizCoef)
   {
      Siz_sv_ = dynamic_cast<StateVariableCoef*>(&SizCoef_);
   }

   ElectronEnergyIonizationCoef(const ElectronEnergyIonizationCoef &other)
      : phiIZCoef_(other.phiIZCoef_), vnCoef_(other.vnCoef_),
        TnCoef_(other.TnCoef_), SizCoef_(other.SizCoef_)
   {
      derivType_ = other.derivType_;
      mn_        = other.mn_;

      Siz_sv_ = dynamic_cast<StateVariableCoef*>(&SizCoef_);
   }

   virtual ElectronEnergyIonizationCoef * Clone() const
   {
      return new ElectronEnergyIonizationCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      bool diz = (Siz_sv_) ? Siz_sv_->NonTrivialValue(deriv) : false;
      return (deriv == INVALID_FIELD_TYPE || diz);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double phiIZ = phiIZCoef_.Eval(T, ip) * J_per_eV_;
      double vn = vnCoef_.Eval(T, ip);
      double Tn = TnCoef_.Eval(T, ip) * J_per_eV_;
      double Siz = SizCoef_.Eval(T, ip);

      return Siz * (phiIZ - 0.5 * me_kg_ * (3.0 * Tn / mn_ + vn * vn));
   }

   double Eval_dFunc(FieldType deriv,
                     ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      bool diz = (Siz_sv_) ? Siz_sv_->NonTrivialValue(deriv) : false;
      if (!diz) { return 0.0; }

      double phiIZ = phiIZCoef_.Eval(T, ip) * J_per_eV_;
      double vn = vnCoef_.Eval(T, ip);
      double Tn = TnCoef_.Eval(T, ip) * J_per_eV_;
      double dSiz = Siz_sv_->Eval_dFunc(deriv, T, ip);

      return dSiz * (phiIZ - 0.5 * me_kg_ * (3.0 * Tn / mn_ + vn * vn));
   }
};

class ElectronEnergyRecombinationCoef : public StateVariableCoef
{
private:
   Coefficient &viCoef_;
   Coefficient &SrcCoef_;
   StateVariableCoef *Src_sv_;

public:
   ElectronEnergyRecombinationCoef(Coefficient &viCoef,
                                   Coefficient &SrcCoef)
      : viCoef_(viCoef), SrcCoef_(SrcCoef)
   {
      Src_sv_ = dynamic_cast<StateVariableCoef*>(&SrcCoef_);
   }

   ElectronEnergyRecombinationCoef(const ElectronEnergyRecombinationCoef &other)
      : viCoef_(other.viCoef_), SrcCoef_(other.SrcCoef_)
   {
      derivType_ = other.derivType_;

      Src_sv_ = dynamic_cast<StateVariableCoef*>(&SrcCoef_);
   }

   virtual ElectronEnergyRecombinationCoef * Clone() const
   {
      return new ElectronEnergyRecombinationCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      bool drc = (Src_sv_) ? Src_sv_->NonTrivialValue(deriv) : false;
      return (deriv == INVALID_FIELD_TYPE ||
              deriv == ION_PARA_VELOCITY ||
              drc);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double vi = viCoef_.Eval(T, ip);
      double Src = SrcCoef_.Eval(T, ip);

      return 0.5 * me_kg_ * Src * vi * vi;
   }

   double Eval_dFunc(FieldType deriv,
                     ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      bool drc = (Src_sv_) ? Src_sv_->NonTrivialValue(deriv) : false;
      if (!drc || deriv == ION_PARA_VELOCITY)
      { return 0.0; }

      double vi = viCoef_.Eval(T, ip);
      double dvi = (deriv == ION_PARA_VELOCITY) ? 1.0 : 0.0;
      double Src = SrcCoef_.Eval(T, ip);
      double dSrc = (Src_sv_) ? Src_sv_->Eval_dFunc(deriv, T, ip) : 0.0;

      return 0.5 * me_kg_ * vi * (dSrc * vi + 2.0 * Src * dvi);
   }
};

class VectorXYCoefficient : public VectorCoefficient
{
private:
   VectorCoefficient * V_;
   Vector v3_;

public:
   VectorXYCoefficient(VectorCoefficient &V)
      : VectorCoefficient(2), V_(&V), v3_(3) {}

   void Eval(Vector &v,
             ElementTransformation &T,
             const IntegrationPoint &ip)
   { v.SetSize(2); V_->Eval(v3_, T, ip); v[0] = v3_[0]; v[1] = v3_[1]; }
};

class VectorZCoefficient : public Coefficient
{
private:
   VectorCoefficient * V_;
   Vector v3_;

public:
   VectorZCoefficient(VectorCoefficient &V)
      : V_(&V), v3_(3) {}

   double Eval(ElementTransformation &T,
               const IntegrationPoint &ip)
   { V_->Eval(v3_, T, ip); return v3_[2]; }
};

class Aniso2DDiffusionCoef : public StateVariableMatCoef
{
private:
   StateVariableCoef * Para_;
   StateVariableCoef * Perp_;
   VectorCoefficient * B3_;

   mutable Vector B_;

   void FormMat(double para, double perp,
                ElementTransformation &T,
                const IntegrationPoint &ip,
                DenseMatrix & M)
   {
      M.SetSize(2);

      B3_->Eval(B_, T, ip);

      double Bmag2 = B_ * B_;

      M(0,0) = (B_[0] * B_[0] * para +
                (B_[1] * B_[1] + B_[2] * B_[2]) * perp ) / Bmag2;
      M(0,1) = B_[0] * B_[1] * (para - perp) / Bmag2;
      M(1,0) = M(0,1);
      M(1,1) = (B_[1] * B_[1] * para +
                (B_[0] * B_[0] + B_[2] * B_[2]) * perp ) / Bmag2;
   }

public:
   Aniso2DDiffusionCoef(StateVariableCoef *ParaCoef,
                        StateVariableCoef *PerpCoef,
                        VectorCoefficient &B3Coef)
      : StateVariableMatCoef(2),
        Para_(ParaCoef), Perp_(PerpCoef),
        B3_(&B3Coef), B_(3) {}

   Aniso2DDiffusionCoef(bool para, StateVariableCoef &Coef,
                        VectorCoefficient &B3Coef)
      : StateVariableMatCoef(2),
        Para_(para ? &Coef : NULL), Perp_(para ? NULL : &Coef),
        B3_(&B3Coef), B_(3) {}

   Aniso2DDiffusionCoef(const Aniso2DDiffusionCoef &other)
      : StateVariableMatCoef(2),
        Para_(other.Para_),
        Perp_(other.Perp_),
        B3_(other.B3_),
        B_(3) {}

   virtual Aniso2DDiffusionCoef * Clone() const
   {
      return new Aniso2DDiffusionCoef(*this);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      bool ntv = deriv == INVALID_FIELD_TYPE;
      if (Para_)
      {
         ntv |= Para_->NonTrivialValue(deriv);
      }
      if (Perp_)
      {
         ntv |= Perp_->NonTrivialValue(deriv);
      }
      return ntv;
   }

   void Eval_Func(DenseMatrix & M,
                  ElementTransformation &T,
                  const IntegrationPoint &ip)
   {
      double para = Para_ ? Para_->Eval(T, ip) : 0.0;
      double perp = Perp_ ? Perp_->Eval(T, ip) : 0.0;

      this->FormMat(para, perp, T, ip, M);
   }

   void Eval_dNn(DenseMatrix & M,
                 ElementTransformation &T,
                 const IntegrationPoint &ip)
   {
      double para = Para_ ? Para_->Eval_dNn(T, ip) : 0.0;
      double perp = Perp_ ? Perp_->Eval_dNn(T, ip) : 0.0;

      this->FormMat(para, perp, T, ip, M);
   }

   void Eval_dNi(DenseMatrix & M,
                 ElementTransformation &T,
                 const IntegrationPoint &ip)
   {
      double para = Para_ ? Para_->Eval_dNi(T, ip) : 0.0;
      double perp = Perp_ ? Perp_->Eval_dNi(T, ip) : 0.0;

      this->FormMat(para, perp, T, ip, M);
   }

   void Eval_dVi(DenseMatrix & M,
                 ElementTransformation &T,
                 const IntegrationPoint &ip)
   {
      double para = Para_ ? Para_->Eval_dVi(T, ip) : 0.0;
      double perp = Perp_ ? Perp_->Eval_dVi(T, ip) : 0.0;

      this->FormMat(para, perp, T, ip, M);
   }

   void Eval_dTi(DenseMatrix & M,
                 ElementTransformation &T,
                 const IntegrationPoint &ip)
   {
      double para = Para_ ? Para_->Eval_dTi(T, ip) : 0.0;
      double perp = Perp_ ? Perp_->Eval_dTi(T, ip) : 0.0;

      this->FormMat(para, perp, T, ip, M);
   }

   void Eval_dTe(DenseMatrix & M,
                 ElementTransformation &T,
                 const IntegrationPoint &ip)
   {
      double para = Para_ ? Para_->Eval_dTe(T, ip) : 0.0;
      double perp = Perp_ ? Perp_->Eval_dTe(T, ip) : 0.0;

      this->FormMat(para, perp, T, ip, M);
   }
};

class NegGradPressureCoefficient : public StateVariableCoef
{
private:
   double zi_; // Stored as a double to avoid type casting in Eval methods
   double dt_;

   GridFunctionCoefficient ni0_;
   GridFunctionCoefficient Ti0_;
   GridFunctionCoefficient Te0_;

   GridFunctionCoefficient dni0_;
   GridFunctionCoefficient dTi0_;
   GridFunctionCoefficient dTe0_;

   GradientGridFunctionCoefficient grad_ni0_;
   GradientGridFunctionCoefficient grad_Ti0_;
   GradientGridFunctionCoefficient grad_Te0_;

   GradientGridFunctionCoefficient grad_dni0_;
   GradientGridFunctionCoefficient grad_dTi0_;
   GradientGridFunctionCoefficient grad_dTe0_;

   VectorCoefficient * B3_;

   mutable Vector gni0_;
   mutable Vector gTi0_;
   mutable Vector gTe0_;

   mutable Vector gdni0_;
   mutable Vector gdTi0_;
   mutable Vector gdTe0_;

   mutable Vector gni1_;
   mutable Vector gTi1_;
   mutable Vector gTe1_;

   mutable Vector B_;

public:
   NegGradPressureCoefficient(ParGridFunctionArray &yGF,
                              ParGridFunctionArray &kGF,
                              int zi, VectorCoefficient & B3Coef)
      : zi_((double)zi), dt_(0.0),
        ni0_(yGF[ION_DENSITY]),
        Ti0_(yGF[ION_TEMPERATURE]),
        Te0_(yGF[ELECTRON_TEMPERATURE]),
        dni0_(kGF[ION_DENSITY]),
        dTi0_(kGF[ION_TEMPERATURE]),
        dTe0_(kGF[ELECTRON_TEMPERATURE]),
        grad_ni0_(yGF[ION_DENSITY]),
        grad_Ti0_(yGF[ION_TEMPERATURE]),
        grad_Te0_(yGF[ELECTRON_TEMPERATURE]),
        grad_dni0_(kGF[ION_DENSITY]),
        grad_dTi0_(kGF[ION_TEMPERATURE]),
        grad_dTe0_(kGF[ELECTRON_TEMPERATURE]),
        B3_(&B3Coef), B_(3) {}

   NegGradPressureCoefficient(const NegGradPressureCoefficient & other)
      : ni0_(other.ni0_),
        Ti0_(other.Ti0_),
        Te0_(other.Te0_),
        dni0_(other.dni0_),
        dTi0_(other.dTi0_),
        dTe0_(other.dTe0_),
        grad_ni0_(ni0_.GetGridFunction()),
        grad_Ti0_(Ti0_.GetGridFunction()),
        grad_Te0_(Te0_.GetGridFunction()),
        grad_dni0_(dni0_.GetGridFunction()),
        grad_dTi0_(dTi0_.GetGridFunction()),
        grad_dTe0_(dTe0_.GetGridFunction()),
        B_(3)
   {
      derivType_ = other.derivType_;
      zi_ = other.zi_;
      dt_ = other.dt_;
      B3_ = other.B3_;
   }

   virtual NegGradPressureCoefficient * Clone() const
   {
      return new NegGradPressureCoefficient(*this);
   }

   void SetTimeStep(double dt) { dt_ = dt; }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return (deriv == INVALID_FIELD_TYPE ||
              deriv == ION_DENSITY || deriv == ION_TEMPERATURE ||
              deriv == ELECTRON_TEMPERATURE);
   }

   double Eval_Func(ElementTransformation &T,
                    const IntegrationPoint &ip)
   {
      double ni0 = ni0_.Eval(T, ip);
      double Ti0 = Ti0_.Eval(T, ip);
      double Te0 = Te0_.Eval(T, ip);

      double dni0 = dni0_.Eval(T, ip);
      double dTi0 = dTi0_.Eval(T, ip);
      double dTe0 = dTe0_.Eval(T, ip);

      double ni1 = ni0 + dt_ * dni0;
      double Ti1 = Ti0 + dt_ * dTi0;
      double Te1 = Te0 + dt_ * dTe0;

      grad_ni0_.Eval(gni0_, T, ip);
      grad_Ti0_.Eval(gTi0_, T, ip);
      grad_Te0_.Eval(gTe0_, T, ip);

      grad_dni0_.Eval(gdni0_, T, ip);
      grad_dTi0_.Eval(gdTi0_, T, ip);
      grad_dTe0_.Eval(gdTe0_, T, ip);

      gni1_.SetSize(gni0_.Size());
      gTi1_.SetSize(gTi0_.Size());
      gTe1_.SetSize(gTe0_.Size());

      add(gni0_, dt_, gdni0_, gni1_);
      add(gTi0_, dt_, gdTi0_, gTi1_);
      add(gTe0_, dt_, gdTe0_, gTe1_);

      B3_->Eval(B_, T, ip);

      double Bmag = sqrt(B_ * B_);

      return -J_per_eV_ * ((zi_ * Te1 + Ti1) * (B_[0] * gni1_[0] + B_[1] * gni1_[1]) +
                           (zi_ * (B_[0] * gTe1_[0] + B_[1] * gTe1_[1]) +
                            (B_[0] * gTi1_[0] + B_[1] * gTi1_[1])) * ni1) / Bmag;
   }

   double Eval_dNi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      grad_Ti0_.Eval(gTi0_, T, ip);
      grad_Te0_.Eval(gTe0_, T, ip);

      grad_dTi0_.Eval(gdTi0_, T, ip);
      grad_dTe0_.Eval(gdTe0_, T, ip);

      gTi1_.SetSize(gTi0_.Size());
      gTe1_.SetSize(gTe0_.Size());

      add(gTi0_, dt_, gdTi0_, gTi1_);
      add(gTe0_, dt_, gdTe0_, gTe1_);

      B3_->Eval(B_, T, ip);

      double Bmag = sqrt(B_ * B_);

      return -J_per_eV_ * (dt_ * zi_ * (B_[0] * gTe1_[0] + B_[1] * gTe1_[1]) +
                           (B_[0] * gTi1_[0] + B_[1] * gTi1_[1])) / Bmag;
   }

   double Eval_dTi(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      grad_ni0_.Eval(gni0_, T, ip);

      grad_dni0_.Eval(gdni0_, T, ip);

      gni1_.SetSize(gni0_.Size());

      add(gni0_, dt_, gdni0_, gni1_);

      B3_->Eval(B_, T, ip);

      double Bmag = sqrt(B_ * B_);

      return -J_per_eV_ * dt_ * (B_[0] * gni1_[0] + B_[1] * gni1_[1]) / Bmag;
   }

   double Eval_dTe(ElementTransformation &T,
                   const IntegrationPoint &ip)
   {
      grad_ni0_.Eval(gni0_, T, ip);

      grad_dni0_.Eval(gdni0_, T, ip);

      gni1_.SetSize(gni0_.Size());

      add(gni0_, dt_, gdni0_, gni1_);

      B3_->Eval(B_, T, ip);

      double Bmag = sqrt(B_ * B_);

      return -J_per_eV_ * dt_ * zi_ * (B_[0] * gni1_[0] + B_[1] * gni1_[1]) / Bmag;
   }
};
/*
class ElementSkewGridFunction : public ParGridFunction
{
private:
   ParMesh &pmesh;
   common::L2_FESpace fes;

   inline double cot(double th)
   {
      double s = sin(th);
      if (fabs(s) < 1e-14) { return 1e14; }
      return cos(th) / s;
   }

   void computeSkew();

public:
   ElementSkewGridFunction(ParMesh &_pmesh);

   virtual void Update();
};
*/
#if MFEM_HYPRE_VERSION >= 21800
// Algebraic multigrid preconditioner for advective problems based on
// approximate ideal restriction (AIR). Most effective when matrix is
// first scaled by DG block inverse, and AIR applied to scaled matrix.
// See https://doi.org/10.1137/17M1144350.
class AIR_prec : public Solver
{
private:
   const HypreParMatrix *A;
   // Copy of A scaled by block-diagonal inverse
   HypreParMatrix A_s;

   HypreBoomerAMG *AIR_solver;
   int blocksize;

public:
   AIR_prec(int blocksize_) : AIR_solver(NULL), blocksize(blocksize_) { }

   void SetOperator(const Operator &op)
   {
      width = op.Width();
      height = op.Height();

      A = dynamic_cast<const HypreParMatrix *>(&op);
      MFEM_VERIFY(A != NULL, "AIR_prec requires a HypreParMatrix.")

      // Scale A by block-diagonal inverse
      BlockInverseScale(A, &A_s, NULL, NULL, blocksize,
                        BlockInverseScaleJob::MATRIX_ONLY);
      delete AIR_solver;
      AIR_solver = new HypreBoomerAMG(A_s);
      AIR_solver->SetAdvectiveOptions(1, "", "FA");
      AIR_solver->SetPrintLevel(0);
      AIR_solver->SetMaxLevels(50);
   }

   virtual void Mult(const Vector &x, Vector &y) const
   {
      // Scale the rhs by block inverse and solve system
      HypreParVector z_s;
      BlockInverseScale(A, NULL, &x, &z_s, blocksize,
                        BlockInverseScaleJob::RHS_ONLY);
      AIR_solver->Mult(z_s, y);
   }

   ~AIR_prec()
   {
      delete AIR_solver;
   }
};
#endif


struct DGParams
{
   double sigma;
   double kappa;
   double width;
};

struct PlasmaParams
{
   double m_n_amu;
   double m_n_kg;
   double v_n_avg_m_per_s; // Magnitude of average neutral velocity
   double v_n_bar_m_per_s; // Average neutral speed
   double T_n_eV;
   double m_i_amu;
   double m_i_kg;
   int    z_i;
};

class DGAdvectionDiffusionTDO : public TimeDependentOperator
{
private:
   const DGParams & dg_;

   bool imex_;
   int logging_;
   std::string log_prefix_;
   double dt_;

   ParFiniteElementSpace *fes_;
   ParGridFunctionArray  *pgf_;

   Coefficient       *CCoef_;    // Scalar coefficient in front of du/dt
   VectorCoefficient *VCoef_;    // Velocity coefficient
   Coefficient       *dCoef_;    // Scalar diffusion coefficient
   MatrixCoefficient *DCoef_;    // Tensor diffusion coefficient
   Coefficient       *SCoef_;    // Source coefficient

   ScalarVectorProductCoefficient *negVCoef_;   // -1  * VCoef
   ScalarVectorProductCoefficient *dtNegVCoef_; // -dt * VCoef
   ProductCoefficient             *dtdCoef_;    //  dt * dCoef
   ScalarMatrixProductCoefficient *dtDCoef_;    //  dt * DCoef

   Array<int>   dbcAttr_;
   Coefficient *dbcCoef_; // Dirichlet BC coefficient

   Array<int>   nbcAttr_;
   Coefficient *nbcCoef_; // Neumann BC coefficient

   ParBilinearForm  m_;
   ParBilinearForm *a_;
   ParBilinearForm *b_;
   ParBilinearForm *s_;
   ParBilinearForm *k_;
   ParLinearForm   *q_exp_;
   ParLinearForm   *q_imp_;

   HypreParMatrix * M_;
   HypreSmoother M_prec_;
   CGSolver M_solver_;

   // HypreParMatrix * B_;
   // HypreParMatrix * S_;

   mutable ParLinearForm rhs_;
   mutable Vector RHS_;
   mutable Vector X_;

   void initM();
   void initA();
   void initB();
   void initS();
   void initK();
   void initQ();

public:
   DGAdvectionDiffusionTDO(const DGParams & dg,
                           ParFiniteElementSpace &fes,
                           ParGridFunctionArray &pgf,
                           Coefficient &CCoef, bool imex = true);

   ~DGAdvectionDiffusionTDO();

   void SetTime(const double _t);

   void SetLogging(int logging, const std::string & prefix = "");

   void SetAdvectionCoefficient(VectorCoefficient &VCoef);
   void SetDiffusionCoefficient(Coefficient &dCoef);
   void SetDiffusionCoefficient(MatrixCoefficient &DCoef);
   void SetSourceCoefficient(Coefficient &SCoef);

   void SetDirichletBC(Array<int> &dbc_attr, Coefficient &dbc);
   void SetNeumannBC(Array<int> &nbc_attr, Coefficient &nbc);

   virtual void ExplicitMult(const Vector &x, Vector &y) const;
   virtual void ImplicitSolve(const double dt, const Vector &u, Vector &dudt);

   void Update();
};

struct TransPrecParams
{
   int type;
   int log_lvl;
};

struct SolverParams
{
   // Linear solver tolerances
   double lin_abs_tol;
   double lin_rel_tol;
   int lin_max_iter;
   int lin_log_lvl;

   // Newton Solver tolerances
   double newt_abs_tol;
   double newt_rel_tol;
   int newt_max_iter;
   int newt_log_lvl;

   // Steady State tolerances
   double ss_abs_tol;
   double ss_rel_tol;

   TransPrecParams prec;
};

struct CG2DG : Operator
{
   const ParFiniteElementSpace &fes_dg;
   H1_FECollection fec_cg;
   ParFiniteElementSpace fes_cg;
   const HypreParMatrix *P;
   SparseMatrix C;
   mutable Vector z;
   CG2DG(const ParFiniteElementSpace &fes_dg, const Array<int> &cg_ess_tdof_list);
   void Mult(const Vector &x, Vector &y) const;
   void MultTranspose(const Vector &x, Vector &y) const;
   HypreParMatrix *ParallelAssemble(); // Caller must delete returned matrix
};

struct DiscontPSCPreconditioner : Solver
{
   const CG2DG &cg2dg;
   const Solver &cg_solver;
   const Solver &smoother;

   mutable Vector x_z, b_cg, x_cg;
   mutable Vector x_sm;

   DiscontPSCPreconditioner(const CG2DG &cg2dg_,
                            const Solver &cg_solver_,
                            const Solver &smoother_);
   virtual void Mult(const Vector &b, Vector &x) const;
   virtual void SetOperator(const Operator &op);
};

struct AdditivePreconditioner : Solver
{
   const Operator *A;
   const Solver &P1;
   const Solver &P2;

   mutable Vector v;

   AdditivePreconditioner(const Solver &P1_,
                          const Solver &P2_);

   virtual void Mult(const Vector &b, Vector &x) const;
   virtual void SetOperator(const Operator &op);
};

struct MultiplicativePreconditioner : Solver
{
   const Operator *A;
   const Solver &P1;
   const Solver &P2;

   mutable Vector r, v;

   MultiplicativePreconditioner(const Solver &P1_,
                                const Solver &P2_);

   virtual void Mult(const Vector &b, Vector &x) const;
   virtual void SetOperator(const Operator &op);
};

/** The DGTransportTDO class is designed to be used with an implicit
    ODESolver to solve a specific set of coupled transport equations
    using a Discontinuous Galerkin (DG) discretization of the relevant
    PDEs.

    This system of transport equations consists of mass conservation
    equations for one species of neutrals and one species of ions,
    momentum conservation for the ion species, and energy equations
    for ions and electrons.

    The field variables are density of neutrals, density of ions,
    velocity of ions in a direction parallel to the magnetic field,
    and the ion and electron temperatures. The electron density is
    assumed to depend on the ion density in such a way that
    quasi-neutrality is maintained e.g. n_e = z_i n_i.

    The ODE solvers in MFEM integrate, in time, equations of the form
       du/dt = f(u,t)

    Where f(u,t) can be a non-linear function of the field quantities,
    u(x,t). The primary function of this class is to implement the
    ImplicitSolve method which finds a du/dt which satisfies
       du/dt = f(u + dt * du/dt, t + dt)
    Since f(u,t) is generally non-linear this is computed using a
    Newton solver.
*/
class DGTransportTDO : public TimeDependentOperator
{
   // friend class TransportPrec;
private:
   const MPI_Session & mpi_;
   int logging_;
   int op_flag_;

   ParFiniteElementSpace &fes_;
   ParFiniteElementSpace &ffes_;
   ParGridFunctionArray  &yGF_;
   ParGridFunctionArray  &kGF_;

   Array<int> &offsets_;

   SolverParams tol_;

   Vector kMax_;
   Array<bool> ss_;

   // Data collection used to write data files
   DataCollection * dc_;

   // Sockets used to communicate with GLVis
   std::map<std::string, socketstream*> socks_;

   class NLOperator : public Operator
   {
   protected:
      const MPI_Session &mpi_;
      const DGParams &dg_;

      int logging_;
      std::string log_prefix_;

      int index_;
      std::string eqn_name_;
      std::string field_name_;
      double dt_;
      ParFiniteElementSpace &fes_;
      ParMesh               &pmesh_;
      ParGridFunctionArray  &yGF_;
      ParGridFunctionArray  &kGF_;

      Array<StateVariableGridFunctionCoef*>  yCoefPtrs_;
      Array<StateVariableGridFunctionCoef*>  kCoefPtrs_;
      Array<StateVariableSumCoef*>          ykCoefPtrs_;

      mutable Array<int> vdofs_;
      mutable Array<int> vdofs2_;
      mutable DenseMatrix elmat_;
      mutable DenseMatrix elmat_k_;
      mutable Vector elvec_;
      mutable Vector locvec_;
      mutable Vector locdvec_;

      typedef std::vector<Array<BilinearFormIntegrator*> > vArrayBFI;

      // Domain integrators for time derivatives of field variables
      vArrayBFI dbfi_m_;  // Domain Integrators
      // Array<Array<StateVariableCoef*> >      dbfi_mc_; // Domain Integrators

      // Domain integrators for field variables at next time step
      vArrayBFI dbfi_;  // Domain Integrators
      vArrayBFI fbfi_;  // Interior Face Integrators
      vArrayBFI bfbfi_; // Boundary Face Integrators
      std::vector<Array<Array<int>*> > bfbfi_marker_; ///< Entries are owned.

      // Domain integrators for source terms
      Array<LinearFormIntegrator*> dlfi_;  // Domain Integrators
      Array<LinearFormIntegrator*> bflfi_; // Boundary Face Integrators
      Array<Array<int>*>           bflfi_marker_; ///< Entries are owned.

      Array<ParBilinearForm*> blf_; // Bilinear Form Objects for Gradients
      Array<ParBilinearForm*> cgblf_; // Bilinear Form Objects for Gradients

      CG2DG *cg2dg_ = NULL;
      HypreParMatrix *CG2DGmat_ = NULL;
      Solver *D_amg_ = NULL;
      ParLORDiscretization *D_lor_ = NULL;
      HypreSmoother *D_smoother_ = NULL;
      HypreParMatrix *D_cg_ = NULL;
      Solver *dg_precond_ = NULL;

      Solver *D_mult_ = NULL;
      Solver *D_schwarz_ = NULL;
      Vector D_diag_;

      Array<int> cg_ess_tdof_list;

      bool use_lor_cg = true;
      bool use_air_cg = true;
      bool use_schwarz = false;

      int term_flag_;
      int vis_flag_;

      // Data collection used to write data files
      DataCollection * dc_;

      // Sockets used to communicate with GLVis
      std::map<std::string, socketstream*> socks_;

      NLOperator(const MPI_Session & mpi, const DGParams & dg,
                 int index,
                 const std::string &eqn_name,
                 const std::string &field_name,
                 ParGridFunctionArray & yGF,
                 ParGridFunctionArray & kGF,
                 int term_flag, int vis_flag, int logging,
                 const std::string & log_prefix);


   public:

      virtual ~NLOperator();

      void SetLogging(int logging, const std::string & prefix = "");

      virtual void SetTimeStep(double dt);

      virtual void Mult(const Vector &k, Vector &y) const;

      virtual void Update();
      virtual Operator *GetGradientBlock(int i);

      virtual Solver *GetPreconditioner() { return dg_precond_; }

      inline bool CheckTermFlag(int flag) { return (term_flag_>> flag) & 1; }

      inline bool CheckVisFlag(int flag) { return (vis_flag_>> flag) & 1; }

      virtual int GetDefaultVisFlag() = 0;

      virtual void RegisterDataFields(DataCollection & dc);

      virtual void PrepareDataFields();

      virtual void InitializeGLVis() = 0;

      virtual void DisplayToGLVis() = 0;
   };

   class TransportOp : public NLOperator
   {
   protected:
      Array<StateVariableCoef*>    svscoefs_;
      Array<StateVariableVecCoef*> svvcoefs_;
      Array<StateVariableMatCoef*> svmcoefs_;
      Array<ProductCoefficient*>             dtSCoefs_;
      Array<ProductCoefficient*>             negdtSCoefs_;
      Array<ScalarVectorProductCoefficient*> dtVCoefs_;
      Array<ScalarVectorProductCoefficient*> negdtVCoefs_;
      Array<ScalarMatrixProductCoefficient*> dtMCoefs_;
      Array<Coefficient*>       sCoefs_;
      Array<VectorCoefficient*> vCoefs_;
      Array<MatrixCoefficient*> mCoefs_;
      std::vector<socketstream*> sout_;
      ParGridFunction coefGF_;

      ParFiniteElementSpace *vfes_;
      ParFiniteElementSpace *h1_fes_;

      const PlasmaParams &plasma_;

      double m_n_kg_;
      double T_n_eV_;
      double v_n_;
      double m_i_kg_;
      int    z_i_;
      /*
      StateVariableGridFunctionCoef &nn0Coef_;
      StateVariableGridFunctionCoef &ni0Coef_;
      StateVariableGridFunctionCoef &vi0Coef_;
      StateVariableGridFunctionCoef &Ti0Coef_;
      StateVariableGridFunctionCoef &Te0Coef_;
      */
      StateVariableCoef &nnCoef_;
      StateVariableCoef &niCoef_;
      StateVariableCoef &viCoef_;
      StateVariableCoef &TiCoef_;
      StateVariableCoef &TeCoef_;

      StateVariableConstantCoef vnAvgCoef_; // Magnitude of average velocity
      StateVariableConstantCoef vnBarCoef_; // Average speed of neutrals
      StateVariableConstantCoef TnCoef_;
      StateVariableConstantCoef ziCoef_;
      StateVariableProductCoef  neCoef_;

      StateVariableGridFunctionCoef &dTe0Coef_;

      CoulombLogEICoef lnLambda_;

      const AdvectionDiffusionBC & bcs_;
      const CoupledBCs & cbcs_;

      const CommonCoefs & cmncoefs_;

      VectorCoefficient & B3Coef_;
      VectorXYCoefficient BxyCoef_;

      Coefficient       * massCoef_;
      Coefficient       * diffusionCoef_;
      MatrixCoefficient * diffusionMatrixCoef_;
      VectorCoefficient * advectionCoef_;

      ApproxIonizationRate    izCoef_;
      ApproxRecombinationRate rcCoef_;
      ApproxChargeExchangeRate cxCoef_;

      IonizationSourceCoef    SizDefCoef_;
      RecombinationSinkCoef   SrcDefCoef_;
      ChargeExchangeSinkCoef  ScxDefCoef_;

      StateVariableCoef & SizCoef_;
      StateVariableCoef & SrcCoef_;
      StateVariableCoef & ScxCoef_;

      TransportOp(const MPI_Session & mpi, const DGParams & dg,
                  const PlasmaParams & plasma, int index,
                  const std::string &eqn_name,
                  const std::string &field_name,
                  ParFiniteElementSpace * vfes,
                  ParFiniteElementSpace * h1_fes,
                  ParGridFunctionArray & yGF,
                  ParGridFunctionArray & kGF,
                  const AdvectionDiffusionBC & bcs,
                  const CoupledBCs & cbcs,
                  const CommonCoefs & common_coefs,
                  VectorCoefficient & B3Coef,
                  int term_flag, int vis_flag,
                  int logging,
                  const std::string & log_prefix);

      /** Sets the time derivative on the left hand side of the equation to be:
             d MCoef / dt
      */
      void SetTimeDerivativeTerm(StateVariableCoef &MCoef);

      /** Sets the diffusion term on the right hand side of the equation
          to be:
             Div(DCoef Grad y[index])
          where index is the index of the equation.
       */
      void SetDiffusionTerm(StateVariableCoef &DCoef);
      void SetDiffusionTerm(StateVariableMatCoef &DCoef);
      /*
       void SetAnisoDiffusionTerm(StateVariableMatCoef &DCoef,
                                  Coefficient &SkewCoef,
                                  double D_min, double D_max);
      */
      /** Sets the advection-diffusion term on the right hand side of the
          equation to be:
             Div(DCoef Grad y[index] - VCoef y[index])
           where index is the index of the equation.
       */
      void SetAnisotropicDiffusionTerm(StateVariableMatCoef &DCoef,
                                       Coefficient *DParaCoef,
                                       Coefficient *DPerpCoef);

      /** Sets the advection-diffusion term on the right hand side of the
          equation to be:
             Div(DCoef Grad y[index] - VCoef y[index])
           where index is the index of the equation.
       */
      void SetAdvectionDiffusionTerm(StateVariableMatCoef &DCoef,
                                     StateVariableVecCoef &VCoef,
                                     Coefficient *DParaCoef,
                                     Coefficient *DPerpCoef);
      void SetDiffusionTermGradient(StateVariableMatCoef &DCoef);

      /** Sets the advection term on the right hand side of the
      equation to be:
             Div(VCoef y[index])
          where index is the index of the equation.
       */
      void SetAdvectionTerm(StateVariableVecCoef &VCoef/*, bool bc = false*/);

      void SetSourceTerm(StateVariableCoef &SCoef, double s = 1.0);
      void SetSourceTermGradient(StateVariableCoef &SCoef, double s = 1.0);
      void SetBdrSourceTerm(StateVariableCoef &SCoef,
                            StateVariableVecCoef &VCoef);

      void SetOutflowBdrTerm(StateVariableVecCoef &VCoef,
                             const Array<CoefficientByAttr*> & obc);
      void SetRecyclingBdrSourceTerm(const RecyclingBC & rbc);

   public:
      virtual ~TransportOp();

      virtual void SetTime(double t);
      virtual void SetTimeStep(double dt);

      virtual void InitializeGLVis();
      virtual void DisplayToGLVis();

      inline Coefficient       * GetMassCoef() { return massCoef_; }
      inline Coefficient       * GetDiffusionCoef() { return diffusionCoef_; }
      inline MatrixCoefficient * GetDiffusionMatrixCoef()
      { return diffusionMatrixCoef_; }
      inline VectorCoefficient * GetAdvectionCoef() { return advectionCoef_; }
   };

   /** The NeutralDensityOp is an mfem::Operator designed to work with
       a NewtonSolver as one row in a block system of non-linear
       transport equations.  Specifically, this operator models the
       mass conservation equation for a neutral species.

          d n_n / dt = Div(D_n Grad(n_n)) + S_n

       Where the diffusion coefficient D_n is a function of n_e and T_e
       (the electron density and temperature respectively) and the
       source term S_n is a function of n_e, T_e, and n_n.  Note that n_e is
       not a state variable but is related to n_i by the simple relation
          n_e = z_i n_i
       where z_i is the charge of the ions and n_i is the ion density.

       To advance this equation in time we need to find k_nn = d n_n / dt
       which satisfies:
          k_nn - Div(D_n(n_e, T_e) Grad(n_n + dt k_nn))
               - S_n(n_e, T_e, n_n + dt k_nn) = 0
       Where n_e and T_e are also evaluated at the next time step.  This is
       done with a Newton solver which needs the Jacobian of this block of
       equations.

       The diagonal block is given by:
          1 - dt Div(D_n Grad) - dt d S_n / d n_n

       The other non-trivial blocks are:
          - dt Div(d D_n / d n_i Grad(n_n)) - dt d S_n / d n_i
          - dt Div(d D_n / d T_e Grad(n_n)) - dt d S_n / d T_e

       The blocks of the Jacobian will be assembled finite element matrices.
       For the diagonal block we need a mass integrator with coefficient
       (1 - dt d S_n / d n_n), and a set of integrators to model the DG
       diffusion operator with coefficient (dt D_n).

       The off-diagonal blocks will consist of a mass integrator with
       coefficient (-dt d S_n / d n_i) or (-dt d S_n / d
       T_e). Currently, (-dt d S_n / d T_e) is not implemented.
    */
   class NeutralDensityOp : public TransportOp
   {
   private:
      enum TermFlag {DIFFUSION_TERM = 0,
                     RECOMBINATION_SOURCE_TERM,
                     IONIZATION_SINK_TERM,
                     SOURCE_TERM,
                     RECYCLING_BDR_SOURCE_TERM
                    };
      enum VisField {DIFFUSION_COEF = 0,
                     RECOMBINATION_SOURCE_COEF,
                     IONIZATION_SINK_COEF,
                     SOURCE_COEF
                    };

      const NeutralDensityCoefs & ndcoefs_;

      NeutralDiffusionCoef      DDefCoef_; // Default diffusion coef
      StateVariableStandardCoef DCoef_;

      ParGridFunction * DGF_;
      ParGridFunction * SrcGF_;
      ParGridFunction * SizGF_;
      ParGridFunction * SGF_;

   public:
      NeutralDensityOp(const MPI_Session & mpi, const DGParams & dg,
                       const PlasmaParams & plasma,
                       ParFiniteElementSpace & h1_fes,
                       ParGridFunctionArray & yGF,
                       ParGridFunctionArray & kGF,
                       const AdvectionDiffusionBC & bcs,
                       const CoupledBCs & cbcs,
                       const NDCoefs & ndcoefs,
                       const CommonCoefs & cmncoefs,
                       VectorCoefficient & B3Coef,
                       int term_flag,
                       int vis_flag, int logging,
                       const std::string & log_prefix);

      virtual ~NeutralDensityOp();

      virtual void SetTime(double t);
      virtual void SetTimeStep(double dt);

      void Update();

      virtual int GetDefaultVisFlag() { return 7; }

      virtual void RegisterDataFields(DataCollection & dc);

      virtual void PrepareDataFields();
   };

   /** The IonDensityOp is an mfem::Operator designed to worth with a
       NewtonSolver as one row in a block system of non-linear
       transport equations.  Specifically, this operator models the
       mass conservation equation for a single ion species.

       d n_i / dt = Div(D_i Grad n_i)) - Div(v_i n_i b_hat) + S_i

       Where the diffusion coefficient D_i is a function of the
       magnetic field direction, v_i is the velocity of the ions
       parallel to B, and the source term S_i is a function of the
       electron and neutral densities as well as the electron
       temperature.

       To advance this equation in time we need to find k_ni = d n_i / dt
       which satisfies:
          k_ni - Div(D_i Grad(n_i + dt k_ni)) + Div(v_i (n_i + dt k_ni) b_hat)
               - S_i(n_e + z_i dt k_ni, T_e, n_n) = 0
       Where n_n and T_e are also evaluated at the next time step.  This is
       done with a Newton solver which needs the Jacobian of this block of
       equations.

       The diagonal block is given by:
          1 - dt Div(D_i Grad) + dt Div(v_i b_hat) - dt d S_i / d n_i

       The other non-trivial blocks are:
          - dt d S_i / d n_n
          + dt Div(n_i b_hat)
          - dt d S_i / d T_e

       The blocks of the Jacobian will be assembled finite element
       matrices.  For the diagonal block we need a mass integrator
       with coefficient (1 - dt d S_i / d n_i), a set of integrators
       to model the DG diffusion operator with coefficient (dt D_i),
       and a weak divergence integrator with coefficient (dt v_i).

       The off-diagonal blocks will consist of a mass integrator with
       coefficient (-dt d S_i / d n_n) or (-dt d S_i / d T_e).
       Currently, (dt Div(n_i b_hat)) and (-dt d S_i / d T_e) are not
       implemented.
    */
   class IonDensityOp : public TransportOp
   {
   private:
      enum TermFlag {DIFFUSION_TERM = 0,
                     ADVECTION_TERM,
                     IONIZATION_SOURCE_TERM,
                     RECOMBINATION_SINK_TERM,
                     SOURCE_TERM,
                     RECYCLING_BDR_SINK_TERM
                    };
      enum VisField {DIFFUSION_PARA_COEF = 0,
                     DIFFUSION_PERP_COEF,
                     ADVECTION_COEF,
                     IONIZATION_SOURCE_COEF,
                     RECOMBINATION_SINK_COEF,
                     SOURCE_COEF
                    };

      const IonDensityCoefs & idcoefs_;

      StateVariableConstantCoef DPerpConstCoef_;
      StateVariableCoef *       DParaCoefPtr_;
      StateVariableCoef *       DPerpCoefPtr_;
      Aniso2DDiffusionCoef      DCoef_;

      IonAdvectionCoef        ViCoef_;

      ParGridFunction * DParaGF_;
      ParGridFunction * DPerpGF_;
      ParGridFunction * AdvGF_;
      ParGridFunction * SizGF_;
      ParGridFunction * SrcGF_;
      ParGridFunction * SGF_;

   public:
      IonDensityOp(const MPI_Session & mpi, const DGParams & dg,
                   const PlasmaParams & plasma,
                   ParFiniteElementSpace & vfes,
                   ParFiniteElementSpace & h1_fes,
                   ParGridFunctionArray & yGF,
                   ParGridFunctionArray & kGF,
                   const AdvectionDiffusionBC & bcs,
                   const CoupledBCs & cbcs,
                   const IDCoefs & idcoefs,
                   const CommonCoefs & cmncoefs,
                   VectorCoefficient & B3Coef,
                   double DPerp,
                   int term_flag, int vis_flag, int logging,
                   const std::string & log_prefix);

      virtual ~IonDensityOp();

      virtual void SetTime(double t);
      virtual void SetTimeStep(double dt);

      void Update();

      virtual int GetDefaultVisFlag() { return 27; }

      virtual void RegisterDataFields(DataCollection & dc);

      virtual void PrepareDataFields();
   };

   /** The IonMomentumOp is an mfem::Operator designed to work with a
       NewtonSolver as one row in a block system of non-linear
       transport equations.  Specifically, this operator models the
       momentum conservation equation for a single ion species.

       m_i v_i d n_i / dt + m_i n_i d v_i / dt
          = Div(Eta Grad v_i) - Div(m_i n_i v_i v_i) - b.Grad(p_i + p_e)
          + m_i v_n S_iz - m_i v_i S_rc + m_i (v_n - v_i) S_cx

       Where the diffusion coefficient Eta is a function of the
       magnetic field, ion density, and ion temperature.

       To advance this equation in time we need to find k_vi = d v_i / dt
       which satisfies:
          m_i n_i k_vi - Div(Eta Grad(v_i + dt k_vi))
             + Div(m_i n_i v_i (v_i + dt k_vi)) + b.Grad(p_i + p_e)
             - m_i v_n S_iz + m_i (v_i + dt k_vi) S_rc
             - m_i (v_n - v_i - dt k_vi) S_cx = 0
       Where n_i, p_i, and p_e are also evaluated at the next time
       step.  This is done with a Newton solver which needs the
       Jacobian of this block of equations.

       The diagonal block is given by:
          m_i n_i - dt Div(Eta Grad) + dt Div(m_i n_i v_i)
             + dt m_i (S_rc + S_cx)
       MLS: Why is the advection term not doubled?

       The other non-trivial blocks are:
          m_i v_i - dt Div(d Eta / d n_i Grad(v_i)) + dt Div(m_i v_i v_i)
          - dt Div(d Eta / d T_i Grad(v_i)) + dt b.Grad(d p_i / d T_i)
          + dt b.Grad(d p_e / d T_e)

       Currently, the static pressure terms and the derivatives of Eta
       do not contribute to the Jacobian.
    */
   class IonMomentumOp : public TransportOp
   {
   private:
      enum TermFlag {DIFFUSION_TERM = 0,
                     ADVECTION_TERM, GRADP_SOURCE_TERM,
                     IONIZATION_SOURCE_TERM,
                     RECOMBINATION_SINK_TERM, CHARGE_EXCHANGE_SOURCE_TERM,
                     SOURCE_TERM
                    };
      enum VisField {DIFFUSION_PARA_COEF = 0, DIFFUSION_PERP_COEF,
                     ADVECTION_COEF, GRADP_SOURCE_COEF,
                     IONIZATION_SOURCE_COEF,
                     RECOMBINATION_SINK_COEF, CHARGE_EXCHANGE_SOURCE_COEF,
                     SOURCE_COEF,
                     ION_PARA_MOMENTUM
                    };

      const IonMomentumCoefs & imcoefs_;

      common::L2_ParFESpace * l2_fes_0_;
      common::H1_ParFESpace * h1_fes_1_;
      ParGridFunction       * OscDiscGF_;
      ParGridFunction       * OscContGF_;
      GridFunctionCoefficient OscDiscCoef_;
      GridFunctionCoefficient OscContCoef_;
      SoundSpeedCoef          CsCoef_;

      double DPerpConst_;
      StateVariableConstantCoef DPerpCoef_;

      IonMomentumParaCoef            momCoef_;
      IonMomentumParaDiffusionCoef   EtaParaCoef_;
      IonMomentumPerpDiffusionCoef   EtaPerpCoef_;
      StateVariableCoef *            EtaParaCoefPtr_;
      StateVariableCoef *            EtaPerpCoefPtr_;
      Aniso2DDiffusionCoef           EtaCoef_;

      IonMomentumAdvectionCoef miniViCoef_;

      NegGradPressureCoefficient negGradPCoef_;

      IonMomentumIonizationCoef     SIZCoef_;
      IonMomentumRecombinationCoef  SRCCoef_;
      IonMomentumChargeExchangeCoef SCXCoef_;

      ParGridFunction * EtaParaGF_;
      ParGridFunction * EtaPerpGF_;
      ParGridFunction * AdvGF_;
      ParGridFunction * MomParaGF_;
      ParGridFunction * SGPGF_;
      ParGridFunction * SIZGF_;
      ParGridFunction * SRCGF_;
      ParGridFunction * SCXGF_;
      ParGridFunction * SGF_;

   public:
      IonMomentumOp(const MPI_Session & mpi, const DGParams & dg,
                    const PlasmaParams & plasma,
                    ParFiniteElementSpace & vfes,
                    ParFiniteElementSpace & h1_fes,
                    ParGridFunctionArray & yGF, ParGridFunctionArray & kGF,
                    const AdvectionDiffusionBC & bcs,
                    const CoupledBCs & cbcs,
                    const IMCoefs & imcoefs,
                    const CmnCoefs & cmncoefs,
                    VectorCoefficient & B3Coef,
                    double DPerp,
                    int term_flag, int vis_flag, int logging,
                    const std::string & log_prefix);

      virtual ~IonMomentumOp();

      virtual void SetTime(double t);
      virtual void SetTimeStep(double dt);

      void Update();

      virtual int GetDefaultVisFlag() { return 127; }

      virtual void RegisterDataFields(DataCollection & dc);

      virtual void PrepareDataFields();
   };

   /** The IonStaticPressureOp is an mfem::Operator designed to work
       with a NewtonSolver as one row in a block system of non-linear
       transport equations.  Specifically, this operator models the
       static pressure equation (related to conservation of energy)
       for a single ion species.

       1.5 T_i d n_i / dt + 1.5 n_i d T_i / dt
          = Div(n_i Chi_i Grad(T_i) - 2.5 n_i v_i T_i)

       Where the diffusion coefficient Chi_i is a function of the
       magnetic field direction, ion density and temperature.

       MLS: Clearly this equation is incomplete.  We stopped at this
       point to focus on implementing a non-linear Robin boundary
       condition.

       To advance this equation in time we need to find
       k_Ti = d T_i / dt which satisfies:
          (3/2)(T_i d n_i / dt + n_i k_Ti) - Div(Chi_i Grad T_i) = 0
       Where n_i is also evaluated at the next time step.  This is
       done with a Newton solver which needs the Jacobian of this
       block of equations.

       The diagonal block is given by:
          1.5 n_i - dt Div(Chi_i Grad)

       The other non-trivial blocks are:
          1.5 T_i - dt Div(d Chi_i / d n_i Grad(T_i))

       MLS: Many more terms will arise once the full equation is implemented.
   */
   class IonStaticPressureOp : public TransportOp
   {
   private:
      enum TermFlag {DIFFUSION_TERM = 0, ADVECTION_TERM, SOURCE_TERM};
      enum VisField {DIFFUSION_PARA_COEF = 0, DIFFUSION_PERP_COEF,
                     SOURCE_COEF
                    };

      const IonStaticPressureCoefs & ispcoefs_;

      double ChiPerpConst_;

      StaticPressureCoef               presCoef_;
      StaticPressureAdvectionCoef      aniViCoef_;
      IonThermalParaDiffusionCoef      ChiParaCoef_;
      StateVariableConstantCoef        ChiPerpCoef_;
      StateVariableCoef *              ChiParaCoefPtr_;
      StateVariableCoef *              ChiPerpCoefPtr_;
      Aniso2DDiffusionCoef             ChiCoef_;
      ProductCoefficient               nChiParaCoef_;
      ProductCoefficient               nChiPerpCoef_;
      StateVariableScalarMatrixProductCoef nChiCoef_;

      ParGridFunction * ChiParaGF_;
      ParGridFunction * ChiPerpGF_;
      ParGridFunction * SGF_;

   public:
      IonStaticPressureOp(const MPI_Session & mpi, const DGParams & dg,
                          const PlasmaParams & plasma,
                          ParGridFunctionArray & yGF,
                          ParGridFunctionArray & kGF,
                          const AdvectionDiffusionBC & bcs,
                          const CoupledBCs & cbcs,
                          const ISPCoefs & ispcoefs,
                          const CmnCoefs & cmncoefs,
                          VectorCoefficient & B3Coef,
                          double ChiPerp,
                          int term_flag, int vis_flag, int logging,
                          const std::string & log_prefix);

      virtual ~IonStaticPressureOp();

      virtual void SetTimeStep(double dt);

      void Update();

      virtual int GetDefaultVisFlag() { return 4; }

      virtual void RegisterDataFields(DataCollection & dc);

      virtual void PrepareDataFields();
   };

   /** The ElectronStaticPressureOp is an mfem::Operator designed to
       work with a NewtonSolver as one row in a block system of
       non-linear transport equations.  Specifically, this operator
       models the static pressure equation (related to conservation of
       energy) for the flow of electrons.

       1.5 T_e d n_e / dt + 1.5 n_e d T_e / dt
          = Div(Chi_e Grad(T_e) - 2.5 n_e v_i T_e)

       Where the diffusion coefficient Chi_e is a function of the
       magnetic field direction, electron density and temperature.

       MLS: Clearly this equation is incomplete.  We stopped at this
       point to focus on implementing a non-linear Robin boundary
       condition.

       To advance this equation in time we need to find
       k_Te = d T_e / dt which satisfies:
          (3/2)(T_e d n_e / dt + n_e k_Te) - Div(Chi_e Grad T_e) = 0
       Where n_e is also evaluated at the next time step.  This is
       done with a Newton solver which needs the Jacobian of this
       block of equations.

       The diagonal block is given by:
          1.5 n_e - dt Div(Chi_e Grad)

       The other non-trivial blocks are:
          1.5 T_e - dt Div(d Chi_e / d n_e Grad(T_e))

       MLS: Many more terms will arise once the full equation is implemented.
   */
   class ElectronStaticPressureOp : public TransportOp
   {
   private:
      enum TermFlag {DIFFUSION_TERM = 0, ADVECTION_TERM, SOURCE_TERM};
      enum VisField {DIFFUSION_PARA_COEF = 0, DIFFUSION_PERP_COEF,
                     SOURCE_COEF = 1
                    };

      const ElectronStaticPressureCoefs & espcoefs_;

      double ChiPerpConst_;

      StaticPressureCoef               presCoef_;
      StaticPressureAdvectionCoef      aneViCoef_;
      ElectronThermalParaDiffusionCoef ChiParaCoef_;
      StateVariableConstantCoef        ChiPerpCoef_;
      StateVariableCoef *              ChiParaCoefPtr_;
      StateVariableCoef *              ChiPerpCoefPtr_;
      Aniso2DDiffusionCoef             ChiCoef_;
      ProductCoefficient               nChiParaCoef_;
      ProductCoefficient               nChiPerpCoef_;
      StateVariableScalarMatrixProductCoef nChiCoef_;

      ParGridFunction * ChiParaGF_;
      ParGridFunction * ChiPerpGF_;
      ParGridFunction * SGF_;

   public:
      ElectronStaticPressureOp(const MPI_Session & mpi, const DGParams & dg,
                               const PlasmaParams & plasma,
                               ParGridFunctionArray & yGF,
                               ParGridFunctionArray & kGF,
                               const AdvectionDiffusionBC & bcs,
                               const CoupledBCs & cbcs,
                               const ESPCoefs & espcoefs,
                               const CmnCoefs & cmncoefs,
                               VectorCoefficient & B3Coef,
                               double ChiPerp,
                               int term_flag, int vis_flag,
                               int logging,
                               const std::string & log_prefix);

      virtual ~ElectronStaticPressureOp();

      virtual void SetTimeStep(double dt);

      void RegisterDataFields(DataCollection & dc);

      void PrepareDataFields();

      void Update();

      virtual int GetDefaultVisFlag() { return 4; }
   };

   class TotalEnergyOp : public TransportOp
   {
   protected:

      IonElectronHeatExchangeCoef QiCoef_;

      StateVariableConstantCoef kBCoef_;
      StateVariableConstantCoef phiIZCoef_;
      StateVariableProductCoef kBphiIZCoef_;
      StateVariableStandardVecCoef BSVCoef_;

      TotalEnergyOp(const MPI_Session & mpi, const DGParams & dg,
                    const PlasmaParams & plasma, int index,
                    const std::string &eqn_name,
                    const std::string &field_name,
                    ParFiniteElementSpace & vfes,
                    ParFiniteElementSpace & h1_fes,
                    ParGridFunctionArray & yGF,
                    ParGridFunctionArray & kGF,
                    const AdvectionDiffusionBC & bcs,
                    const CoupledBCs & cbcs,
                    const CommonCoefs & common_coefs,
                    VectorCoefficient & B3Coef,
                    int term_flag, int vis_flag,
                    int logging,
                    const std::string & log_prefix);

   public:

      void SetKineticEnergyAdvectionTerm(StateVariableVecCoef &VCoef);

      virtual void SetTime(double t);
   };

   class IonTotalEnergyOp : public TotalEnergyOp
   {
   private:
      enum TermFlag {DIFFUSION_TERM = 0,
                     ADVECTION_TERM, KE_ADVECTION_TERM,
                     IONIZATION_SOURCE_TERM, RECOMBINATION_SINK_TERM,
                     CHARGE_EXCHANGE_SOURCE_TERM,
                     EQUIPARTITION_SOURCE_TERM, SOURCE_TERM
                    };

      enum VisField {DIFFUSION_PARA_COEF = 0, DIFFUSION_PERP_COEF,
                     ADVECTION_COEF, IONIZATION_SOURCE_COEF,
                     RECOMBINATION_SINK_COEF, CHARGE_EXCHANGE_SOURCE_COEF,
                     EQUIPARTITION_SOURCE_COEF, SOURCE_COEF,
                     ION_TOTAL_ENERGY
                    };

      const IonTotalEnergyCoefs & itecoefs_;

      double ChiPerpConst_;

      TotalEnergyCoef                  totEnergyCoef_;
      KineticEnergyCoef                kinEnergyCoef_;
      TotalEnergyAdvectionCoef         advFluxCoef_;
      StaticPressureAdvectionCoef      aniViCoef_;
      IonThermalParaDiffusionCoef      ChiParaCoef_;
      StateVariableConstantCoef        ChiPerpCoef_;
      StateVariableCoef *              ChiParaCoefPtr_;
      StateVariableCoef *              ChiPerpCoefPtr_;
      Aniso2DDiffusionCoef             ChiCoef_;
      ProductCoefficient               nChiParaCoef_;
      ProductCoefficient               nChiPerpCoef_;
      ProductCoefficient               nkChiParaCoef_;
      ProductCoefficient               nkChiPerpCoef_;
      StateVariableScalarMatrixProductCoef nChiCoef_;
      StateVariableScalarMatrixProductCoef nkChiCoef_;
      StateVariableScalarVectorProductCoef keVCoef_;

      IonEnergyIonizationCoef     SIZCoef_;
      IonEnergyRecombinationCoef  SRCCoef_;
      IonEnergyChargeExchangeCoef SCXCoef_;

      ParGridFunction * ChiParaGF_;
      ParGridFunction * ChiPerpGF_;
      ParGridFunction * AdvGF_;
      ParGridFunction * SIZGF_;
      ParGridFunction * SRCGF_;
      ParGridFunction * SCXGF_;
      ParGridFunction * SGF_;
      ParGridFunction * QiGF_;
      ParGridFunction * totEnergyGF_;

   public:
      IonTotalEnergyOp(const MPI_Session & mpi, const DGParams & dg,
                       const PlasmaParams & plasma,
                       ParFiniteElementSpace & vfes,
                       ParFiniteElementSpace & h1_fes,
                       ParGridFunctionArray & yGF,
                       ParGridFunctionArray & kGF,
                       const AdvectionDiffusionBC & bcs,
                       const CoupledBCs & cbcs,
                       const ITECoefs & espcoefs,
                       const CmnCoefs & cmncoefs,
                       VectorCoefficient & B3Coef,
                       double ChiPerp,
                       int term_flag, int vis_flag,
                       int logging,
                       const std::string & log_prefix);

      virtual ~IonTotalEnergyOp();

      virtual void SetTime(double t);

      void Update();

      virtual int GetDefaultVisFlag() { return 255; }

      virtual void RegisterDataFields(DataCollection & dc);

      virtual void PrepareDataFields();
   };

   class ElectronTotalEnergyOp : public TotalEnergyOp
   {
   private:
      enum TermFlag {DIFFUSION_TERM = 0,
                     ADVECTION_TERM, KE_ADVECTION_TERM,
                     IONIZATION_SINK_TERM, RECOMBINATION_SINK_TERM,
                     EQUIPARTITION_SOURCE_TERM, SOURCE_TERM
                    };

      enum VisField {DIFFUSION_PARA_COEF = 0, DIFFUSION_PERP_COEF,
                     ADVECTION_COEF, IONIZATION_SINK_COEF,
                     RECOMBINATION_SINK_COEF, EQUIPARTITION_SOURCE_COEF,
                     SOURCE_COEF, ELECTRON_TOTAL_ENERGY
                    };

      const ElectronTotalEnergyCoefs & etecoefs_;

      double ChiPerpConst_;

      TotalEnergyCoef                  totEnergyCoef_;
      KineticEnergyCoef                kinEnergyCoef_;
      TotalEnergyAdvectionCoef         advFluxCoef_;
      StaticPressureAdvectionCoef      aneViCoef_;
      ElectronThermalParaDiffusionCoef ChiParaCoef_;
      StateVariableConstantCoef        ChiPerpCoef_;
      StateVariableCoef *              ChiParaCoefPtr_;
      StateVariableCoef *              ChiPerpCoefPtr_;
      Aniso2DDiffusionCoef             ChiCoef_;
      ProductCoefficient               nChiParaCoef_;
      ProductCoefficient               nChiPerpCoef_;
      ProductCoefficient               nkChiParaCoef_;
      ProductCoefficient               nkChiPerpCoef_;
      StateVariableScalarMatrixProductCoef nChiCoef_;
      StateVariableScalarMatrixProductCoef nkChiCoef_;
      StateVariableScalarVectorProductCoef keVCoef_;

      ElectronEnergyIonizationCoef     SIZCoef_;
      ElectronEnergyRecombinationCoef  SRCCoef_;

      ParGridFunction * ChiParaGF_;
      ParGridFunction * ChiPerpGF_;
      ParGridFunction * AdvGF_;
      ParGridFunction * SIZGF_;
      ParGridFunction * SRCGF_;
      ParGridFunction * SGF_;
      ParGridFunction * QiGF_;
      ParGridFunction * totEnergyGF_;

   public:
      ElectronTotalEnergyOp(const MPI_Session & mpi, const DGParams & dg,
                            const PlasmaParams & plasma,
                            ParFiniteElementSpace & vfes,
                            ParFiniteElementSpace & h1_fes,
                            ParGridFunctionArray & yGF,
                            ParGridFunctionArray & kGF,
                            const AdvectionDiffusionBC & bcs,
                            const CoupledBCs & cbcs,
                            const ETECoefs & espcoefs,
                            const CmnCoefs & cmncoefs,
                            VectorCoefficient & B3Coef,
                            double ChiPerp,
                            int term_flag, int vis_flag,
                            int logging,
                            const std::string & log_prefix);

      virtual ~ElectronTotalEnergyOp();

      virtual void SetTime(double t);

      void Update();

      virtual int GetDefaultVisFlag() { return 15; }

      virtual void RegisterDataFields(DataCollection & dc);

      virtual void PrepareDataFields();
   };

   class DummyOp : public TransportOp
   {
   public:
      DummyOp(const MPI_Session & mpi, const DGParams & dg,
              const PlasmaParams & plasma,
              ParGridFunctionArray & yGF,
              ParGridFunctionArray & kGF,
              const AdvectionDiffusionBC & bcs,
              const CoupledBCs & cbcs,
              const CommonCoefs & cmncoefs,
              VectorCoefficient & B3Coef,
              int index,
              const std::string &eqn_name,
              const std::string &field_name,
              int term_flag, int vis_flag,
              int logging, const std::string & log_prefix);

      virtual void SetTimeStep(double dt)
      {
         if (mpi_.Root() && logging_ > 1)
         {
            std::cout << "Setting time step: " << dt << " in DummyOp\n";
         }
         TransportOp::SetTimeStep(dt);
      }

      void Update();

      virtual int GetDefaultVisFlag() { return 0; }
   };

   class VisualizationOp;

   class CombinedOp : public Operator
   {
   private:
      const MPI_Session & mpi_;
      int neq_;
      int logging_;

      ParFiniteElementSpace &fes_;
      // ParGridFunctionArray  &yGF_;
      ParGridFunctionArray  &kGF_;

      Array<TransportOp*> op_;

      const Vector &wgts_;

      Array<int> & offsets_;
      mutable BlockOperator *grad_;

      void updateOffsets();

   public:
      CombinedOp(const MPI_Session & mpi, const DGParams & dg,
                 const PlasmaParams & plasma, const Vector &eqn_weights,
                 ParFiniteElementSpace & vfes,
                 ParFiniteElementSpace & h1_fes,
                 ParGridFunctionArray & yGF, ParGridFunctionArray & kGF,
                 const TransportBCs & bcs,
                 const TransportCoefs & coefs,
                 Array<int> & offsets,
                 double DiPerp, double XiPerp, double XePerp,
                 const Array<int> & term_flags,
                 const Array<int> & vis_flags,
                 unsigned int op_flag = 31, int logging = 0);

      ~CombinedOp();

      void SetTime(double t);
      void SetTimeStep(double dt);
      void SetLogging(int logging);

      inline Coefficient * GetDnCoef()
      { return op_[0]->GetDiffusionCoef(); }
      inline MatrixCoefficient * GetDiCoef()
      { return op_[1]->GetDiffusionMatrixCoef(); }
      inline MatrixCoefficient * GetEtaCoef()
      { return op_[2]->GetDiffusionMatrixCoef(); }
      inline MatrixCoefficient * GetnXiCoef()
      { return op_[3]->GetDiffusionMatrixCoef(); }
      inline MatrixCoefficient * GetnXeCoef()
      { return op_[4]->GetDiffusionMatrixCoef(); }

      inline Solver* GetPreconditionerBlock(int i) const
      {
         return op_[i]->GetPreconditioner();
      }

      void Update();

      void Mult(const Vector &k, Vector &y) const;

      void UpdateGradient(const Vector &x) const;

      Operator &GetGradient(const Vector &x) const
      { UpdateGradient(x); return *grad_; }

      void RegisterDataFields(DataCollection & dc);

      void PrepareDataFields();

      void InitializeGLVis();

      void DisplayToGLVis();
   };

   class TransportLeftPrec : public BlockDiagonalPreconditioner
   {
   private:
      Array<Operator*> diag_prec_;
#ifdef MFEM_USE_SUPERLU
      Array<SuperLURowLocMatrix*> slu_mat_;
#endif
      DGTransportTDO::CombinedOp & comb_op_;

      TransPrecParams p_;

   public:
      TransportLeftPrec(const Array<int> &offsets, const TransPrecParams &p,
                        CombinedOp &combOp);
      ~TransportLeftPrec();

      virtual void SetOperator(const Operator &op);
   };

   class RightPreconditioner
   {
   public:
      virtual void Mult(const Vector&x, Vector &y) = 0;
      virtual void InverseMult(const Vector&y, Vector &x) = 0;
   };

   class RightBlockDiagonalPreconditioner : public RightPreconditioner
   {
   protected:
      const Array<int> &offsets_;
      const Vector &scale_factors_;

   public:
      RightBlockDiagonalPreconditioner(const Array<int> &offsets,
                                       const Vector &scale_factors)
         :offsets_(offsets), scale_factors_(scale_factors)
      {
         MFEM_VERIFY(offsets.Size() - 1 == scale_factors.Size(),
                     "RightBlockDiagonalPreconditioner: "
                     "Incompatible numbers of offsets and scale factors.");
      }

      virtual void Mult(const Vector&x, Vector &y)
      {
         for (int i=0; i<offsets_.Size() - 1; i++)
         {
            for (int j=offsets_[i]; j<offsets_[i+1]; j++)
            {
               y[j] = scale_factors_[i] * x[j];
            }
         }
      }

      virtual void InverseMult(const Vector&y, Vector &x)
      {
         for (int i=0; i<offsets_.Size() - 1; i++)
         {
            for (int j=offsets_[i]; j<offsets_[i+1]; j++)
            {
               x[j] = y[j] / scale_factors_[i];
            }
         }
      }
   };

   class TransportRightPrec : public RightBlockDiagonalPreconditioner
   {
   private:


      // DGTransportTDO::CombinedOp & comb_op_;

      // TransPrecParams p_;

   public:
      TransportRightPrec(const Array<int> &offsets,
                         const Vector &scale_factors,
                         const TransPrecParams &p,
                         CombinedOp &combOp)
         : RightBlockDiagonalPreconditioner(offsets, scale_factors)//, p_(p)
      {}

      ~TransportRightPrec() {}

      // virtual void SetOperator(const Operator &op) {}
   };

   /// GMRES method with right preconditioner
   class GMRESRPCSolver : public IterativeSolver
   {
   protected:
      int m; // see SetKDim()

      RightPreconditioner *r_prec;

      static inline void GeneratePlaneRotation(double &dx, double &dy,
                                               double &cs, double &sn);

      static inline void ApplyPlaneRotation(double &dx, double &dy,
                                            double &cs, double &sn);

      static inline void Update(Vector &x, int k, DenseMatrix &h, Vector &s,
                                Array<Vector*> &v);

   public:
      GMRESRPCSolver() { m = 50; r_prec = NULL; }

#ifdef MFEM_USE_MPI
      GMRESRPCSolver(MPI_Comm comm_)
         : IterativeSolver(comm_) { m = 50; r_prec = NULL; }
#endif

      /// Set the number of iteration to perform between restarts, default is 50.
      void SetKDim(int dim) { m = dim; }

      virtual void SetRightPreconditioner(RightPreconditioner &pr)
      { r_prec = &pr; }

      virtual void Mult(const Vector &b, Vector &x) const;
   };

   CombinedOp op_;

   TransportLeftPrec  newton_op_l_prec_;
   TransportRightPrec newton_op_r_prec_;
   GMRESRPCSolver     newton_op_solver_;
   NewtonSolver       newton_solver_;

   mutable Vector x_;
   mutable Vector y_;
   Vector u_;
   Vector dudt_;

public:
   DGTransportTDO(const MPI_Session & mpi, const DGParams & dg,
                  const PlasmaParams & plasma,
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
                  const Array<int> & term_flags,
                  const Array<int> & vis_flags,
                  bool imex = true,
                  unsigned int op_flag = 31,
                  int logging = 0);

   ~DGTransportTDO();

   void SetTime(const double _t);
   void SetLogging(int logging);

   double CheckGradient();

   bool CheckForSteadyState();

   void RegisterDataFields(DataCollection & dc);

   void PrepareDataFields();

   void InitializeGLVis();

   void DisplayToGLVis();

   inline Coefficient * GetDnCoefficient() { return op_.GetDnCoef(); }
   inline MatrixCoefficient * GetDiCoefficient() { return op_.GetDiCoef(); }
   inline MatrixCoefficient * GetEtaCoefficient() { return op_.GetEtaCoef(); }
   inline MatrixCoefficient * GetnXiCoefficient() { return op_.GetnXiCoef(); }
   inline MatrixCoefficient * GetnXeCoefficient() { return op_.GetnXeCoef(); }

   virtual void ImplicitSolve(const double dt, const Vector &y, Vector &k);

   void Update();
};

class DGTransportTDO::VisualizationOp : public TransportOp
{
private:
   enum VisField {B_POLOIDAL = 0,
                  B_TOROIDAL,
                  COULOMB_LOG,
                  TAU_I,
                  TAU_E,
                  IONIZATION_RATE,
                  RECOMBINATION_RATE,
                  CHARGE_EXCHANGE_RATE,
                  ION_SOUND_SPEED
                 };

   // VectorCoefficient & B3Coef_;
   VectorXYCoefficient BxyCoef_;
   VectorZCoefficient  BzCoef_;

   CoulombLogEICoef          lnLambdaCoef_;
   IonCollisionTimeCoef      TauICoef_;
   ElectronCollisionTimeCoef TauECoef_;
   ApproxIonizationRate      SigmaIZCoef_;
   ApproxRecombinationRate   SigmaRCCoef_;
   ApproxChargeExchangeRate  SigmaCXCoef_;
   SoundSpeedCoef            CsCoef_;

   ParGridFunction * BxyGF_;
   ParGridFunction * BzGF_;
   ParGridFunction * lnLambdaGF_;
   ParGridFunction * TauIGF_;
   ParGridFunction * TauEGF_;
   ParGridFunction * SigmaIZGF_;
   ParGridFunction * SigmaRCGF_;
   ParGridFunction * SigmaCXGF_;
   ParGridFunction * CsGF_;

public:
   VisualizationOp(const MPI_Session & mpi, const DGParams & dg,
                   const PlasmaParams & plasma,
                   ParFiniteElementSpace &vfes,
                   ParGridFunctionArray & yGF,
                   ParGridFunctionArray & kGF,
                   const AdvectionDiffusionBC & bcs,
                   const CoupledBCs & cbcs,
                   const CommonCoefs & cmncoefs,
                   VectorCoefficient & B3Coef,
                   int vis_flag,
                   int logging,
                   const std::string & log_prefix);

   ~VisualizationOp();

   virtual void SetTimeStep(double dt)
   {
      if (mpi_.Root() && logging_ > 1)
      {
         std::cout << "Setting time step: " << dt << " in VisualizationOp\n";
      }
      TransportOp::SetTimeStep(dt);
   }

   void Update();

   inline bool CheckVisFlag(int flag) { return (vis_flag_>> flag) & 1; }

   virtual int GetDefaultVisFlag() { return 3; }

   virtual void RegisterDataFields(DataCollection & dc);

   virtual void PrepareDataFields();
};

class MultiSpeciesDiffusion;
class MultiSpeciesAdvection;
/*
class TransportSolver : public ODESolver
{
private:
   ODESolver * impSolver_;
   ODESolver * expSolver_;

   ParFiniteElementSpace & sfes_; // Scalar fields
   ParFiniteElementSpace & vfes_; // Vector fields
   ParFiniteElementSpace & ffes_; // Full system

   BlockVector & nBV_;

   ParGridFunction & B_;

   Array<int> & charges_;
   Vector & masses_;

   MultiSpeciesDiffusion * msDiff_;

   void initDiffusion();

public:
   TransportSolver(ODESolver * implicitSolver, ODESolver * explicitSolver,
                   ParFiniteElementSpace & sfes,
                   ParFiniteElementSpace & vfes,
                   ParFiniteElementSpace & ffes,
                   BlockVector & nBV,
                   ParGridFunction & B,
                   Array<int> & charges,
                   Vector & masses);
   ~TransportSolver();

   void Update();

   void Step(Vector &x, double &t, double &dt);
};
*/
/*
class ChiParaCoefficient : public Coefficient
{
private:
 BlockVector & nBV_;
 ParGridFunction nGF_;
 GridFunctionCoefficient nCoef_;
 GridFunctionCoefficient TCoef_;

 int ion_;
 Array<int> & z_;
 Vector     * m_;
 Vector       n_;

public:
 ChiParaCoefficient(BlockVector & nBV, Array<int> & charges);
 ChiParaCoefficient(BlockVector & nBV, int ion_species,
                    Array<int> & charges, Vector & masses);
 void SetT(ParGridFunction & T);

 double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

class ChiPerpCoefficient : public Coefficient
{
private:
 int ion_;

public:
 ChiPerpCoefficient(BlockVector & nBV, Array<int> & charges);
 ChiPerpCoefficient(BlockVector & nBV, int ion_species,
                    Array<int> & charges, Vector & masses);

 double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

class ChiCrossCoefficient : public Coefficient
{
private:
 int ion_;

public:
 ChiCrossCoefficient(BlockVector & nBV, Array<int> & charges);
 ChiCrossCoefficient(BlockVector & nBV, int ion_species,
                     Array<int> & charges, Vector & masses);

 double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

class ChiCoefficient : public MatrixCoefficient
{
private:
 ChiParaCoefficient  chiParaCoef_;
 ChiPerpCoefficient  chiPerpCoef_;
 ChiCrossCoefficient chiCrossCoef_;
 VectorGridFunctionCoefficient BCoef_;

 Vector bHat_;

public:
 ChiCoefficient(int dim, BlockVector & nBV, Array<int> & charges);
 ChiCoefficient(int dim, BlockVector & nBV, int ion_species,
                Array<int> & charges, Vector & masses);

 void SetT(ParGridFunction & T);
 void SetB(ParGridFunction & B);

 void Eval(DenseMatrix &K, ElementTransformation &T,
           const IntegrationPoint &ip);
};
*/
/*
class EtaParaCoefficient : public Coefficient
{
private:
 BlockVector & nBV_;
 ParGridFunction nGF_;
 GridFunctionCoefficient nCoef_;
 GridFunctionCoefficient TCoef_;

 int ion_;
 Array<int> & z_;
 Vector     * m_;
 Vector       n_;

public:
 EtaParaCoefficient(BlockVector & nBV, Array<int> & charges);
 EtaParaCoefficient(BlockVector & nBV, int ion_species,
                    Array<int> & charges, Vector & masses);

 void SetT(ParGridFunction & T);

 double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};
*/
/*
class MultiSpeciesDiffusion : public TimeDependentOperator
{
private:
   ParFiniteElementSpace &sfes_;
   ParFiniteElementSpace &vfes_;

   BlockVector & nBV_;

   Array<int> & charges_;
   Vector & masses_;

   void initCoefficients();
   void initBilinearForms();

public:
   MultiSpeciesDiffusion(ParFiniteElementSpace & sfes,
                         ParFiniteElementSpace & vfes,
                         BlockVector & nBV,
                         Array<int> & charges,
                         Vector & masses);

   ~MultiSpeciesDiffusion();

   void Assemble();

   void Update();

   void ImplicitSolve(const double dt, const Vector &x, Vector &y);
};
*/

// Time-dependent operator for the right-hand side of the ODE representing the
// DG weak form for the diffusion term. (modified from ex14p)
class DiffusionTDO : public TimeDependentOperator
{
private:
   const int dim_;
   double dt_;
   double dg_sigma_;
   double dg_kappa_;

   ParFiniteElementSpace &fes_;
   // ParFiniteElementSpace &dfes_;
   ParFiniteElementSpace &vfes_;

   ParBilinearForm m_;
   ParBilinearForm d_;

   ParLinearForm rhs_;
   ParGridFunction x_;

   HypreParMatrix * M_;
   HypreParMatrix * D_;

   Vector RHS_;
   Vector X_;

   HypreSolver * solver_;
   HypreSolver * amg_;

   MatrixCoefficient &nuCoef_;
   ScalarMatrixProductCoefficient dtNuCoef_;

   void initSolver(double dt);

public:
   DiffusionTDO(ParFiniteElementSpace &fes,
                ParFiniteElementSpace &dfes,
                ParFiniteElementSpace &_vfes,
                MatrixCoefficient & nuCoef,
                double dg_sigma,
                double dg_kappa);

   // virtual void Mult(const Vector &x, Vector &y) const;

   virtual void ImplicitSolve(const double dt, const Vector &x, Vector &y);

   virtual ~DiffusionTDO() { }
};

// Time-dependent operator for the right-hand side of the ODE representing the
// DG weak form for the advection term.
class AdvectionTDO : public TimeDependentOperator
{
private:
   const int dim_;
   const int num_equation_;
   const double specific_heat_ratio_;

   mutable double max_char_speed_;

   ParFiniteElementSpace &vfes_;
   Operator &A_;
   SparseMatrix &Aflux_;
   DenseTensor Me_inv_;

   mutable Vector state_;
   mutable DenseMatrix f_;
   mutable DenseTensor flux_;
   mutable Vector z_;

   void GetFlux(const DenseMatrix &state, DenseTensor &flux) const;

public:
   AdvectionTDO(ParFiniteElementSpace &_vfes,
                Operator &A, SparseMatrix &Aflux, int num_equation,
                double specific_heat_ratio);

   virtual void Mult(const Vector &x, Vector &y) const;

   virtual ~AdvectionTDO() { }
};

// Implements a simple Rusanov flux
class RiemannSolver
{
private:
   int num_equation_;
   double specific_heat_ratio_;
   Vector flux1_;
   Vector flux2_;

public:
   RiemannSolver(int num_equation, double specific_heat_ratio);
   double Eval(const Vector &state1, const Vector &state2,
               const Vector &nor, Vector &flux);
};


// Constant (in time) mixed bilinear form multiplying the flux grid function.
// The form is (vec(v), grad(w)) where the trial space = vector L2 space (mesh
// dim) and test space = scalar L2 space.
class DomainIntegrator : public BilinearFormIntegrator
{
private:
   Vector shape_;
   DenseMatrix flux_;
   DenseMatrix dshapedr_;
   DenseMatrix dshapedx_;

public:
   DomainIntegrator(const int dim, const int num_equation);

   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Tr,
                                       DenseMatrix &elmat);
};

// Interior face term: <F.n(u),[w]>
class FaceIntegrator : public NonlinearFormIntegrator
{
private:
   int num_equation_;
   double max_char_speed_;
   RiemannSolver rsolver_;
   Vector shape1_;
   Vector shape2_;
   Vector funval1_;
   Vector funval2_;
   Vector nor_;
   Vector fluxN_;
   IntegrationPoint eip1_;
   IntegrationPoint eip2_;

public:
   FaceIntegrator(RiemannSolver &rsolver_, const int dim,
                  const int num_equation);

   virtual void AssembleFaceVector(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Tr,
                                   const Vector &elfun, Vector &elvect);
};

/** Integrator for the DG form:

    - < {(Q grad(u)).n}, [v] > + sigma < [u], {(Q grad(v)).n} >
    + kappa < {h^{-1} cot(theta_T) Q_max^2/Q_min} [u], [v] >,

    where Q is a matrix diffusion coefficient and u, v are the trial
    and test spaces, respectively. Q_max and Q_min are the global
    approximations of the maximum and minimum eigenvalues of Q. The
    function cot(theta_T) is a measure of the distortion of the mesh
    with theta_T approximating the minimum interior angle of each
    element. The parameters sigma and kappa determine the DG method to
    be used (when this integrator is added to the "broken"
    DiffusionIntegrator):
    * sigma = -1, kappa >= kappa0: symm. interior penalty (IP or SIPG) method,
    * sigma = +1, kappa > 0: non-symmetric interior penalty (NIPG) method,
    * sigma = +1, kappa = 0: the method of Baumann and Oden. */
/*
class DGAnisoDiffusionIntegrator : public BilinearFormIntegrator
{
protected:
   MatrixCoefficient *MQ;
   Coefficient *CotTheta;
   double q0, q1;
   double sigma, kappa;

   // these are not thread-safe!
   Vector shape1, shape2, dshape1dn, dshape2dn, nor, nh, ni;
   DenseMatrix jmat, dshape1, dshape2, mq, adjJ;

public:
   DGAnisoDiffusionIntegrator(MatrixCoefficient &q,
                              Coefficient &cotTheta,
                              const double qMin, const double qMax,
                              const double s, const double k)
      : MQ(&q), CotTheta(&cotTheta), q0(qMin), q1(qMax), sigma(s), kappa(k) { }
   using BilinearFormIntegrator::AssembleFaceMatrix;
   virtual void AssembleFaceMatrix(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Trans,
                                   DenseMatrix &elmat);
};
*/

class DGAdvDiffBaseIntegrator
{
protected:
   Coefficient *Q;
   MatrixCoefficient *MQ;
   VectorCoefficient *beta;
   Coefficient *QPara;
   Coefficient *QPerp;
   double lambda, sigma, kappa1, kappa2;

   DGAdvDiffBaseIntegrator(Coefficient & q, VectorCoefficient & b,
                           double l, double s, double k1, double k2)
      :
      Q(&q),
      MQ(NULL),
      beta(&b),
      QPara(NULL),
      QPerp(NULL),
      lambda(l),
      sigma(s),
      kappa1(k1),
      kappa2(k2)
   { }

   DGAdvDiffBaseIntegrator(MatrixCoefficient & q, VectorCoefficient & b,
                           Coefficient *qPara, Coefficient *qPerp,
                           double l, double s, double k1, double k2)
      :
      Q(NULL),
      MQ(&q),
      beta(&b),
      QPara(qPara),
      QPerp(qPerp),
      lambda(l),
      sigma(s),
      kappa1(k1),
      kappa2(k2)
   { }

};

/** Integrator for the DG form:

    < {- Q grad(u) + beta u}_alpha, [v] >
    - sigma < [u], {- Q grad(v) + beta v}_alpha >
    + sigma < [u], {beta v} >
    + kappa < [u], [v] >

    Where:
       {Psi}_alpha = alpha_1 Psi_1 + alpha_2 Psi_2 and alpha_2 = 1 - alpha_1
       {Psi} = (Psi_1 + Psi_2) / 2
       [phi] = n_1 phi_1 + n_2 phi_2
    The diffusion coefficient is the matrix Q, the advection coefficient is
    the vector beta.  The parameter sigma determines the DG method to be used
    (when this integrator is added to the "broken" DiffusionIntegrator and
    the ConservativeConvectionIntegrator):
    * sigma = -1: symm. interior penalty (IP or SIPG) method,
    * sigma = +1: non-symmetric interior penalty (NIPG) method,

    The alpha parameters are determined using a continuous scalar field tau
    according to:
       alpha = (0.5, 0.5) + 0.5 tau (sign(beta.n_1), sign(beta.n_2))
    When tau = 0 this leads to an equal weighting across interelement
    boundaries. When tau = 1 this leads to classical upwinding. Values between
    these extremes can be used to control the degree of upwinding between each
    pair of elements.

    The parameter kappa is a penalty parameter which encourages continuity of
    the solution. See the 2007 paper "Estimation of penalty parameters for
    symmetric interior penalty Galerkin methods" by Y. Epshteyn and B. Riviere
    for advice on selecting kappa. Crudely the rule is:
       kappa > p (p+1) f(Q) g(T_h) in 2D
       kappa > p (p+2) f(Q) g(T_h) in 3D
    Where g(T_h) is a function of the mesh spacing and distortion, and f(Q)
    is (Q_max)^2/Q_min with Q_max and Q_min being the maximum and minimum
    eigenvalues of the diffusion coefficient. It's likely that the advection
    velocity should also contribute to kappa but exactly how is unclear.

    Finally it should be noted that this formulation is borowed from the 2009
    paper "Discontinuous Galerkin methods for advection-diffusion-reaction
    problems" by B. Ayuso and D. Marini where it appears as equation 3.8 (with
    slight modifications). Note in particular that our sigma parameter is the
    negative of the theta parameter from the paper.
*/
class DGAdvDiffIntegrator : public BilinearFormIntegrator,
   DGAdvDiffBaseIntegrator
{
private:
#ifndef MFEM_THREAD_SAFE
   Vector shape1, shape2, nQdshape1, nQdshape2;
   DenseMatrix dshape1, dshape2;
#endif

   double ComputeUpwindingParam(double epsilon, double betaMag);

public:
   DGAdvDiffIntegrator(Coefficient & q, VectorCoefficient & b,
                       double l, double s, double k1, double k2)
      : DGAdvDiffBaseIntegrator(q, b, l, s, k1, k2) {}

   DGAdvDiffIntegrator(MatrixCoefficient & q, VectorCoefficient & b,
                       Coefficient *qPara, Coefficient *qPerp,
                       double l, double s, double k1, double k2)
      : DGAdvDiffBaseIntegrator(q, b, qPara, qPerp, l, s, k1, k2) {}

   using BilinearFormIntegrator::AssembleFaceMatrix;
   virtual void AssembleFaceMatrix(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Trans,
                                   DenseMatrix &elmat);
};

class DGAdvDiffBdrIntegrator : public BilinearFormIntegrator,
   DGAdvDiffBaseIntegrator
{
private:
#ifndef MFEM_THREAD_SAFE
   Vector shape1, nQdshape1;
   DenseMatrix dshape1;
#endif

public:
   DGAdvDiffBdrIntegrator(Coefficient & q, VectorCoefficient & b,
                          double l, double s, double k1, double k2)
      : DGAdvDiffBaseIntegrator(q, b, l, s, k1, k2) {}

   DGAdvDiffBdrIntegrator(MatrixCoefficient & q, VectorCoefficient & b,
                          Coefficient *qPara, Coefficient *qPerp,
                          double l, double s, double k1, double k2)
      : DGAdvDiffBaseIntegrator(q, b, qPara, qPerp, l, s, k1, k2) {}

   using BilinearFormIntegrator::AssembleFaceMatrix;
   virtual void AssembleFaceMatrix(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Trans,
                                   DenseMatrix &elmat);
};

/** Boundary linear integrator for imposing non-zero Dirichlet boundary
    conditions, to be used in conjunction with DGDiffusionIntegrator.
    Specifically, given the Dirichlet data u_D, the linear form assembles the
    following integrals on the boundary:

    sigma < u_D, (Q grad(v)).n > + kappa < {h^{-1} Q} u_D, v >,

    where Q is a scalar or matrix diffusion coefficient and v is the test
    function. The parameters sigma and kappa should be the same as the ones
    used in the DGDiffusionIntegrator. */
class DGAdvDiffDirichletLFIntegrator : public LinearFormIntegrator,
   DGAdvDiffBaseIntegrator
{
private:
   Coefficient *uD;

   // these are not thread-safe!
   Vector shape, dshape_dn, nor, nh, ni, vb;
   DenseMatrix dshape, mq, adjJ;

public:
   DGAdvDiffDirichletLFIntegrator(Coefficient &u,
                                  Coefficient &q,
                                  VectorCoefficient & b,
                                  double l,
                                  double s,
                                  double k1,
                                  double k2)
      : DGAdvDiffBaseIntegrator(q, b, l, s, k1, k2),
        uD(&u)
   { }

   DGAdvDiffDirichletLFIntegrator(Coefficient &u,
                                  MatrixCoefficient &q,
                                  VectorCoefficient & b,
                                  Coefficient *qPara,
                                  Coefficient *qPerp,
                                  double l,
                                  double s,
                                  double k1,
                                  double k2)
      : DGAdvDiffBaseIntegrator(q, b, qPara, qPerp, l, s, k1, k2),
        uD(&u)
   { }

   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect)
   {
      mfem_error("DGAdvDiffDirichletLFIntegrator::AssembleRHSElementVect");
   }
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       FaceElementTransformations &Tr,
                                       Vector &elvect);
};

/** Integrator for the DG form:

    < {- Q grad(u)}_alpha, [v] >
    - sigma < [u], {- Q grad(v)}_alpha >
    + kappa < [u], [v] >

    Where:
       {Psi}_alpha = alpha_1 Psi_1 + alpha_2 Psi_2 and alpha_2 = 1 - alpha_1
       {Psi} = (Psi_1 + Psi_2) / 2
       [phi] = n_1 phi_1 + n_2 phi_2
    The diffusion coefficient is the matrix Q.  The parameter sigma determines
    the DG method to be used
    (when this integrator is added to the "broken" DiffusionIntegrator:
    * sigma = -1: symm. interior penalty (IP or SIPG) method,
    * sigma = +1: non-symmetric interior penalty (NIPG) method,

    The alpha parameters are determined using a continuous scalar field tau
    according to:
       alpha = (0.5, 0.5) + 0.5 tau (sign(beta.n_1), sign(beta.n_2))

    The parameter kappa is a penalty parameter which encourages continuity of
    the solution. See the 2007 paper "Estimation of penalty parameters for
    symmetric interior penalty Galerkin methods" by Y. Epshteyn and B. Riviere
    for advice on selecting kappa. Crudely the rule is:
       kappa > p (p+1) f(Q) g(T_h) in 2D
       kappa > p (p+2) f(Q) g(T_h) in 3D
    Where g(T_h) is a function of the mesh spacing and distortion, and f(Q)
    is (Q_max)^2/Q_min with Q_max and Q_min being the maximum and minimum
    eigenvalues of the diffusion coefficient. It's likely that the advection
    velocity should also contribute to kappa but exactly how is unclear.

    Finally it should be noted that this formulation is borowed from the 2009
    paper "Discontinuous Galerkin methods for advection-diffusion-reaction
    problems" by B. Ayuso and D. Marini where it appears as equation 3.8 (with
    slight modifications). Note in particular that our sigma parameter is the
    negative of the theta parameter from the paper.
*/
class DGAnisoDiffBaseIntegrator
{
protected:
   MatrixCoefficient *MQ;
   Coefficient *QPara;
   Coefficient *QPerp;
   double sigma, kappa;

   DGAnisoDiffBaseIntegrator(MatrixCoefficient & q,
                             Coefficient *qPara, Coefficient *qPerp,
                             double s, double k)
      :
      MQ(&q),
      QPara(qPara),
      QPerp(qPerp),
      sigma(s),
      kappa(k)
   { }

};

class DGAnisoDiffIntegrator : public BilinearFormIntegrator,
   DGAnisoDiffBaseIntegrator
{
private:
#ifndef MFEM_THREAD_SAFE
   Vector shape1, shape2, nQdshape1, nQdshape2;
   DenseMatrix dshape1, dshape2;
#endif

public:
   DGAnisoDiffIntegrator(MatrixCoefficient & q,
                         Coefficient *qPara, Coefficient *qPerp,
                         double s, double k)
      : DGAnisoDiffBaseIntegrator(q, qPara, qPerp, s, k)
   {}

   using BilinearFormIntegrator::AssembleFaceMatrix;
   virtual void AssembleFaceMatrix(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Trans,
                                   DenseMatrix &elmat);
};

class DGAnisoDiffBdrIntegrator : public BilinearFormIntegrator,
   DGAnisoDiffBaseIntegrator
{
private:
#ifndef MFEM_THREAD_SAFE
   Vector shape1, nQdshape1;
   DenseMatrix dshape1;
#endif

public:
   DGAnisoDiffBdrIntegrator(MatrixCoefficient & q,
                            Coefficient *qPara, Coefficient *qPerp,
                            double s, double k)
      : DGAnisoDiffBaseIntegrator(q, qPara, qPerp, s, k) {}

   using BilinearFormIntegrator::AssembleFaceMatrix;
   virtual void AssembleFaceMatrix(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Trans,
                                   DenseMatrix &elmat);
};

/** Boundary linear integrator for imposing non-zero Dirichlet boundary
    conditions, to be used in conjunction with DGDiffusionIntegrator.
    Specifically, given the Dirichlet data u_D, the linear form assembles the
    following integrals on the boundary:

    sigma < u_D, (Q grad(v)).n > + kappa < {h^{-1} Q} u_D, v >,

    where Q is a scalar or matrix diffusion coefficient and v is the test
    function. The parameters sigma and kappa should be the same as the ones
    used in the DGDiffusionIntegrator. */
class DGAnisoDiffDirichletLFIntegrator : public LinearFormIntegrator,
   DGAnisoDiffBaseIntegrator
{
private:
   Coefficient *uD;

   // these are not thread-safe!
   Vector shape, dshape_dn, nor, nh, ni, vb;
   DenseMatrix dshape, mq, adjJ;

public:
   DGAnisoDiffDirichletLFIntegrator(Coefficient &u,
                                    MatrixCoefficient &q,
                                    Coefficient *qPara,
                                    Coefficient *qPerp,
                                    double s,
                                    double k)
      : DGAnisoDiffBaseIntegrator(q, qPara, qPerp, s, k),
        uD(&u)
   { }

   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect)
   {
      mfem_error("DGAnisoDiffDirichletLFIntegrator::AssembleRHSElementVect");
   }
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       FaceElementTransformations &Tr,
                                       Vector &elvect);
};

} // namespace transport

} // namespace plasma

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_TRANSPORT_SOLVER
