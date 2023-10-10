#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

#ifndef PARPROBLEM_DEFS
#define PARPROBLEM_DEFS

// abstract ParGeneralOptProblem class
// of the form
// min_(u,m) f(u,m) s.t. c(u,m)=0 and m>=ml
// the primal variable (u, m) is represented as a BlockVector
// think about supporting general lower and upper bounds (see HiOP user manual) 

class ParGeneralOptProblem
{
protected:
    int dimU, dimM, dimC;
    ParFiniteElementSpace * fesU = nullptr;
    ParFiniteElementSpace * fesM = nullptr;
    Array<int> block_offsetsx;
    Vector ml;
public:
    ParGeneralOptProblem(ParFiniteElementSpace * fesU_, ParFiniteElementSpace * fesM_); // constructor
    virtual double CalcObjective(const BlockVector &) const = 0;
    virtual void Duf(const BlockVector &, Vector &) const = 0;
    virtual void Dmf(const BlockVector &, Vector &) const = 0;
    void CalcObjectiveGrad(const BlockVector &, BlockVector &) const;
    virtual HypreParMatrix * Duuf(const BlockVector &) = 0;
    virtual HypreParMatrix * Dumf(const BlockVector &) = 0;
    virtual HypreParMatrix * Dmuf(const BlockVector &) = 0;
    virtual HypreParMatrix * Dmmf(const BlockVector &) = 0;
    virtual HypreParMatrix * Duc(const BlockVector &) = 0;
    virtual HypreParMatrix * Dmc(const BlockVector &) = 0;
    // TO DO: include Hessian terms of constraint c
    virtual void c(const BlockVector &, Vector &) const = 0;
    int GetDimU() const { return dimU; };
    int GetDimM() const { return dimM; }; 
    int GetDimC() const { return dimC; };
    ParFiniteElementSpace * GetfesU() {return fesU;}
    ParFiniteElementSpace * GetfesM() {return fesM;}
    Vector Getml() const { return ml; };
    ~ParGeneralOptProblem(); // destructor
};


// abstract ContactProblem class
// of the form
// min_d e(d) s.t. g(d) >= 0
class ParOptProblem : public ParGeneralOptProblem
{
protected:
    Array<int> block_offsetsx;
    HypreParMatrix * Ih;
public:
    ParOptProblem(ParFiniteElementSpace * fesU_, ParFiniteElementSpace * fesM_);        // constructor
    double CalcObjective(const BlockVector &) const; // objective e
    void Duf(const BlockVector &, Vector &) const;
    void Dmf(const BlockVector &, Vector &) const;

    HypreParMatrix * Duuf(const BlockVector &);
    HypreParMatrix * Dumf(const BlockVector &);
    HypreParMatrix * Dmuf(const BlockVector &);
    HypreParMatrix * Dmmf(const BlockVector &);
    HypreParMatrix * Duc(const BlockVector &);
    HypreParMatrix * Dmc(const BlockVector &);

    void c(const BlockVector &, Vector &) const;
    virtual double E(const Vector &) const = 0;               // objective e(d) (energy function)
    virtual void DdE(const Vector &, Vector &) const = 0;      // gradient of objective De / Dd
    virtual HypreParMatrix * DddE(const Vector &) = 0;
    // Hessian of objective D^2 e / D d^2
    virtual HypreParMatrix * Ddg(const Vector &) = 0;
    // Jacobian of inequality constraint Dg / Dd
    virtual void g(const Vector &, Vector &) const = 0;       // inequality constraint g(d) >= 0 (gap function)
    int GetDimD() const { return fesU->GetTrueVSize(); };
    int GetDimS() const { return fesM->GetTrueVSize(); };
    virtual ~ParOptProblem();
};

class ParObstacleProblem : public ParOptProblem
{
protected:
   // data to define energy objective function e(d) = 0.5 d^T K d - f^T d, g(d) = d >= \psi
   // stiffness matrix used to define objective
   ParBilinearForm *Kform;
   ParLinearForm   *fform;
   Array<int> ess_tdof_list; // needed for calls to FormSystemMatrix
   HypreParMatrix  K;
   HypreParMatrix  *J;
   ParFiniteElementSpace *Vh;
   Vector f;
   Vector psi;
public :
   ParObstacleProblem(ParFiniteElementSpace*, ParFiniteElementSpace*, double (*fSource)(const Vector &));
   ParObstacleProblem(ParFiniteElementSpace*, ParFiniteElementSpace*, double (*fSource)(const Vector &), double (*obstacleSource)(const Vector &), Array<int> tdof_list, Vector &);
   double E(const Vector &) const;
   void DdE(const Vector &, Vector &) const;
   HypreParMatrix* DddE(const Vector &);
   void g(const Vector &, Vector &) const;
   HypreParMatrix* Ddg(const Vector &);
   virtual ~ParObstacleProblem();
};

#endif
