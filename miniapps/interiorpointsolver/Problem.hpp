#ifndef PARPROBLEM_DEFS
#define PARPROBLEM_DEFS

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "utilities.hpp"

// using namespace std;
// using namespace mfem;

namespace mfem {

// abstract ParGeneralOptProblem class
// of the form
// min_(u,m) f(u,m) s.t. c(u,m)=0 and m>=ml
// the primal variable (u, m) is represented as a BlockVector
class ParGeneralOptProblem
{
protected:
    int dimU, dimM, dimC;
    int dimUglb, dimMglb;
    HYPRE_BigInt * dofOffsetsU;
    HYPRE_BigInt * dofOffsetsM;
    Array<int> block_offsetsx;
    Vector ml;
public:
    ParGeneralOptProblem();
    virtual void Init(HYPRE_BigInt * dofOffsetsU_, HYPRE_BigInt * dofOffsetsM_);
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
    virtual void c(const BlockVector &, Vector &) const = 0;
    int GetDimU() const { return dimU; };
    int GetDimM() const { return dimM; }; 
    int GetDimC() const { return dimC; };
    int GetDimUGlb() const { return dimUglb; };
    int GetDimMGlb() const { return dimMglb; };
    HYPRE_BigInt * GetDofOffsetsU() const { return dofOffsetsU; };
    HYPRE_BigInt * GetDofOffsetsM() const { return dofOffsetsM; }; 
    Vector Getml() const { return ml; };
    ~ParGeneralOptProblem();
};


// abstract ContactProblem class
// of the form
// min_d e(d) s.t. g(d) >= 0
class ParOptProblem : public ParGeneralOptProblem
{
protected:
    HypreParMatrix * Ih;
public:
    ParOptProblem();
    void Init(HYPRE_BigInt *, HYPRE_BigInt *);
    
    // ParGeneralOptProblem methods are defined in terms of
    // ParOptProblem specific methods: E, DdE, DddE, g, Ddg
    double CalcObjective(const BlockVector &) const; 
    void Duf(const BlockVector &, Vector &) const;
    void Dmf(const BlockVector &, Vector &) const;
    HypreParMatrix * Duuf(const BlockVector &);
    HypreParMatrix * Dumf(const BlockVector &);
    HypreParMatrix * Dmuf(const BlockVector &);
    HypreParMatrix * Dmmf(const BlockVector &);
    void c(const BlockVector &, Vector &) const;
    HypreParMatrix * Duc(const BlockVector &);
    HypreParMatrix * Dmc(const BlockVector &);
    
    // ParOptProblem specific methods:
    
    // energy objective function e(d)
    // input: d an mfem::Vector
    // output: e(d) a double
    virtual double E(const Vector &d) const = 0;

    // gradient of energy objective De / Dd
    // input: d an mfem::Vector,
    //        gradE an mfem::Vector, which will be the gradient of E at d
    // output: none    
    virtual void DdE(const Vector &d, Vector &gradE) const = 0;

    // Hessian of energy objective D^2 e / Dd^2
    // input:  d, an mfem::Vector
    // output: The Hessian of the energy objective at d, a pointer to a HypreParMatrix
    virtual HypreParMatrix * DddE(const Vector &d) = 0;

    // Constraint function g(d) >= 0, e.g., gap function
    // input: d, an mfem::Vector,
    //       gd, an mfem::Vector, which upon successfully calling the g method will be
    //                            the evaluation of the function g at d
    // output: none
    virtual void g(const Vector &d, Vector &gd) const = 0;

    // Jacobian of constraint function Dg / Dd, e.g., gap function Jacobian
    // input:  d, an mfem::Vector,
    // output: The Jacobain of the constraint function g at d, a pointer to a HypreParMatrix
    virtual HypreParMatrix * Ddg(const Vector &) = 0;
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
   ParObstacleProblem(ParFiniteElementSpace*, double (*fSource)(const Vector &), double (*obstacleSource)(const Vector &));
   ParObstacleProblem(ParFiniteElementSpace*, double (*fSource)(const Vector &), double (*obstacleSource)(const Vector &), Array<int> tdof_list, Vector &);
   double E(const Vector &) const;
   void DdE(const Vector &, Vector &) const;
   HypreParMatrix* DddE(const Vector &);
   void g(const Vector &, Vector &) const;
   HypreParMatrix* Ddg(const Vector &);
   virtual ~ParObstacleProblem();
};



class ReducedProblem : public ParOptProblem
{
protected:
  HypreParMatrix *J;
  HypreParMatrix *P; // projector
  ParOptProblem  *problem;
public:
  ReducedProblem(ParOptProblem *problem, HYPRE_Int * constraintMask);
  ReducedProblem(ParOptProblem *problem, HypreParVector & constraintMask);
  double E(const Vector &) const;
  void DdE(const Vector &, Vector &) const;
  HypreParMatrix * DddE(const Vector &);
  void g(const Vector &, Vector &) const;
  HypreParMatrix * Ddg(const Vector &);
  virtual ~ReducedProblem();
};


}

#endif
