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

class OptProblem
{
protected:
    int dimU, dimM, dimC;
    Array<int> block_offsetsx;
    Vector ml;
public:
    OptProblem();
    virtual double CalcObjective(const BlockVector &) const = 0;
    virtual void Duf(const BlockVector &, Vector &) const = 0;
    virtual void Dmf(const BlockVector &, Vector &) const = 0;
    void CalcObjectiveGrad(const BlockVector &, BlockVector &) const;
    virtual SparseMatrix* Duuf(const BlockVector &) = 0;
    virtual SparseMatrix* Dumf(const BlockVector &) = 0;
    virtual SparseMatrix* Dmuf(const BlockVector &) = 0;
    virtual SparseMatrix* Dmmf(const BlockVector &) = 0;
    virtual void c(const BlockVector &, Vector &) const = 0;
    virtual SparseMatrix* Duc(const BlockVector &) = 0;
    virtual SparseMatrix* Dmc(const BlockVector &) = 0;
    // TO DO: include Hessian terms of constraint c
    // TO DO: include log-barrier lumped-mass and pass that
    // to the optimizer
    //virtual SparseMatrix* GetLogBarrierLumpedMass() = 0;
    int GetDimU() const { return dimU; };
    int GetDimM() const { return dimM; }; 
    int GetDimC() const { return dimC; };
    Vector Getml() const { return ml; };
    ~OptProblem();
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
    //ContactProblem(int, int);        // constructor
    ContactProblem();
    void InitializeParentData(int, int);
    double CalcObjective(const BlockVector &) const; // objective e
    void Duf(const BlockVector &, Vector &) const;
    void Dmf(const BlockVector &, Vector &) const;
    SparseMatrix* Duuf(const BlockVector &);
    SparseMatrix* Dumf(const BlockVector &);
    SparseMatrix* Dmuf(const BlockVector &);
    SparseMatrix* Dmmf(const BlockVector &);
    void c(const BlockVector &, Vector &) const;
    SparseMatrix* Duc(const BlockVector &);
    SparseMatrix* Dmc(const BlockVector &);
    virtual double E(const Vector &) const = 0;           // objective e(d) (energy function)
    virtual void DdE(const Vector &, Vector &) const = 0; // gradient of objective De / Dd
    virtual SparseMatrix* DddE(const Vector &) = 0;       // Hessian of objective D^2 e / D d^2
    virtual void g(const Vector &, Vector &) const = 0;   // inequality constraint g(d) >= 0 (gap function)
    virtual SparseMatrix* Ddg(const Vector &) = 0;        // Jacobian of inequality constraint Dg / Dd
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
   SparseMatrix  *J;
   FiniteElementSpace *Vh;
   Vector f;
public : 
   ObstacleProblem(FiniteElementSpace* , double (*fSource)(const Vector &));
   double E(const Vector &) const;
   void DdE(const Vector &, Vector &) const;
   SparseMatrix* DddE(const Vector &);
   void g(const Vector &, Vector &) const;
   SparseMatrix* Ddg(const Vector &);
   // TO DO: include lumped-mass for the log-barrier term
   //SparseMatrix* GetLogBarrierLumpedMass();
   virtual ~ObstacleProblem();
};

class DirichletObstacleProblem : public ContactProblem
{
protected:
   // data to define energy objective function e(d) = 0.5 d^T K d - f^T d, g(d) = d + \psi >= 0
   // stiffness matrix used to define objective
   BilinearForm *Kform;
   LinearForm   *fform;
   Array<int> ess_tdof_list; // needed for calls to FormSystemMatrix
   SparseMatrix  *K;
   SparseMatrix  *J;
   FiniteElementSpace *Vh;
   Vector f;
   Vector psi;
   Vector xDC;
public : 
   DirichletObstacleProblem(FiniteElementSpace*, Vector&,  double (*fSource)(const Vector &), double (*obstacleSource)(const Vector &), Array<int> tdof_list);
   double E(const Vector &) const;
   void DdE(const Vector &, Vector &) const;
   SparseMatrix* DddE(const Vector &);
   void g(const Vector &, Vector &) const;
   SparseMatrix* Ddg(const Vector &);
   virtual ~DirichletObstacleProblem();
};

#endif
