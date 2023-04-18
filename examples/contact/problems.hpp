#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

#ifndef PROBLEM_DEFS
#define PROBLEM_DEFS


// abstract OptProblem class
// of the form
// min_(u,m) f(u,m) s.t. c(u,m)=0 and m>=ml
// the primal variable (u, m) is represented as a BlockVector
// think about supporting general lower and upper bounds (see HiOP user manual) 

class OptProblem
{
protected:
    int dimU, dimM, dimC;
    Array<int> block_offsetsx;
    Vector ml;
public:
    OptProblem(); // constructor
    virtual double CalcObjective(const BlockVector &) const = 0;
    virtual void Duf(const BlockVector &, Vector &) const = 0;
    virtual void Dmf(const BlockVector &, Vector &) const = 0;
    void CalcObjectiveGrad(const BlockVector &, BlockVector &) const;
    virtual void Duuf(const BlockVector &, SparseMatrix *&) = 0;
    virtual void Dumf(const BlockVector &, SparseMatrix *&) = 0;
    virtual void Dmuf(const BlockVector &, SparseMatrix *&) = 0;
    virtual void Dmmf(const BlockVector &, SparseMatrix *&) = 0;
    //void Dxxf(const BlockVector &, BlockOperator *&);
    virtual void c(const BlockVector &, Vector &) const = 0;
    virtual void Duc(const BlockVector &, SparseMatrix *&) = 0;
    virtual void Dmc(const BlockVector &, SparseMatrix *&) = 0;
    // TO DO: include Hessian terms of constraint c
    int GetDimU() const { return dimU; };
    int GetDimM() const { return dimM; }; 
    int GetDimC() const { return dimC; };
    Vector Getml() const { return ml; };
    ~OptProblem(); // destructor
};


// abstract ContactProblem class
// of the form
// min_d e(d) s.t. g(d) >= 0
// TO DO: add functionality for gap function Hessian apply 
class ContactProblem : public OptProblem
{
protected:
    int dimD;
    int dimS;
    Array<int> block_offsetsx;
public:
    ContactProblem(int);        // constructor
    double CalcObjective(const BlockVector &) const; // objective e
    void Duf(const BlockVector &, Vector &) const;
    void Dmf(const BlockVector &, Vector &) const;
    void Duuf(const BlockVector &, SparseMatrix *&);
    void Dumf(const BlockVector &, SparseMatrix *&);
    void Dmuf(const BlockVector &, SparseMatrix *&);
    void Dmmf(const BlockVector &, SparseMatrix *&);
    void c(const BlockVector &, Vector &) const;
    void Duc(const BlockVector &, SparseMatrix *&);
    void Dmc(const BlockVector &, SparseMatrix *&);
    virtual double E(const Vector &) const = 0;               // objective e(d) (energy function)
    virtual void DdE(const Vector &, Vector &) const = 0;      // gradient of objective De / Dd
    virtual void DddE(const Vector &, SparseMatrix *&) = 0;  // Hessian of objective D^2 e / D d^2
    virtual void g(const Vector &, Vector &) const = 0;       // inequality constraint g(d) >= 0 (gap function)
    virtual void Ddg(const Vector &, SparseMatrix *&) = 0;   // Jacobian of inequality constraint Dg / Dd
    int GetDimD() const { return dimD; };
    int GetDimS() const { return dimS; };
    virtual ~ContactProblem();
};


class ObstacleProblem : public ContactProblem
{
protected:
   // data to define energy objective function e(d) = 0.5 d^T K d - f^T d, g(d) = d >= 0
   // stiffness matrix used to define objective
   BilinearForm *Kform;
   LinearForm   *fform;
   Array<int> empty_tdof_list; // needed for calls to FormSystemMatrix
   SparseMatrix  K;
   Vector f;
   FiniteElementSpace *Vh;
public : 
   ObstacleProblem(FiniteElementSpace* );
   double E(const Vector &) const;
   void DdE(const Vector &, Vector &) const;
   void DddE(const Vector &, SparseMatrix *&);
   void g(const Vector &, Vector &) const;
   void Ddg(const Vector &, SparseMatrix *&);
   static double fRhs(const Vector &);
   virtual ~ObstacleProblem();
};


class QPContactExample : public ContactProblem
{
protected :
    // data to define energy objective function e(d) = 0.5 d^T K d + f^T d, g(d) = J d >= 0
    SparseMatrix *K;
    SparseMatrix *J;
    Vector *f;
public : 
   QPContactExample(SparseMatrix *, SparseMatrix *, Vector *);
   double E(const Vector &) const;
   void DdE(const Vector &, Vector &) const;
   void DddE(const Vector &, SparseMatrix *&);
   void g(const Vector &, Vector &) const;
   void Ddg(const Vector &, SparseMatrix *&);
   virtual ~QPContactExample();
};



#endif
