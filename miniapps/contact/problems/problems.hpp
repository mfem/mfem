#include "mfem.hpp"
#include <fstream>
#include <iostream>


using namespace std;
using namespace mfem;

#ifndef PROBLEM_DEFS
#define PROBLEM_DEFS

// abstract GeneralOptProblem class
// of the form
// min_(u,m) f(u,m) s.t. c(u,m)=0 and m>=ml
// the primal variable (u, m) is represented as a BlockVector
class GeneralOptProblem
{
protected:
    int dimU, dimM, dimC;
    int dimUGlb, dimMGlb, dimCGlb;
    #ifdef MFEM_USE_MPI
      HYPRE_BigInt * dofOffsetsU;
      HYPRE_BigInt * dofOffsetsM;
    #endif
    Array<int> block_offsetsx;
    Vector ml;
public:
    GeneralOptProblem();
    virtual double CalcObjective(const BlockVector &) const = 0;
    virtual void Duf(const BlockVector &, Vector &) const = 0;
    virtual void Dmf(const BlockVector &, Vector &) const = 0;
    void CalcObjectiveGrad(const BlockVector &, BlockVector &) const;
    #ifdef MFEM_USE_MPI
       void InitGeneral(HYPRE_BigInt * dofOffsetsU_, HYPRE_BigInt * dofOffsetsM_);
       virtual HypreParMatrix * Duuf(const BlockVector &) = 0;
       virtual HypreParMatrix * Dumf(const BlockVector &) = 0;
       virtual HypreParMatrix * Dmuf(const BlockVector &) = 0;
       virtual HypreParMatrix * Dmmf(const BlockVector &) = 0;
       virtual HypreParMatrix * Duc(const BlockVector &) = 0;
       virtual HypreParMatrix * Dmc(const BlockVector &) = 0;
       HYPRE_BigInt * GetDofOffsetsU() const { return dofOffsetsU; };
       HYPRE_BigInt * GetDofOffsetsM() const { return dofOffsetsM; };
    #else
       virtual void InitGeneral(int dimU, int dimM);
       virtual SparseMatrix * Duuf(const BlockVector &) = 0;
       virtual SparseMatrix * Dumf(const BlockVector &) = 0;
       virtual SparseMatrix * Dmuf(const BlockVector &) = 0;
       virtual SparseMatrix * Dmmf(const BlockVector &) = 0;
       virtual SparseMatrix * Duc(const BlockVector &) = 0;
       virtual SparseMatrix * Dmc(const BlockVector &) = 0;
    #endif    
    virtual void c(const BlockVector &, Vector &) const = 0;
    int GetDimU() const { return dimU; };
    int GetDimM() const { return dimM; }; 
    int GetDimC() const { return dimC; };
    int GetDimUGlb() const { return dimUGlb; };
    int GetDimMGlb() const { return dimMGlb; };
    int GetDimCGlb() const { return dimCGlb; };
    Vector Getml() const { return ml; };
    ~GeneralOptProblem();
};


// Specialized optimization problem
// of the form
// min_d e(d) s.t. g(d) >= 0
// suited for contact mechanics problems that can be formualted
// as an optimization problem
class OptProblem : public GeneralOptProblem
{
protected:
    #ifdef MFEM_USE_MPI
       HypreParMatrix * Ih;
    #else
       SparseMatrix * Ih;
    #endif
public:
    OptProblem();
    
    // GeneralOptProblem methods are defined in terms of
    // OptProblem specific methods: E, DdE, DddE, g, Ddg
    double CalcObjective(const BlockVector &) const; 
    void Duf(const BlockVector &, Vector &) const;
    void Dmf(const BlockVector &, Vector &) const;
    #ifdef MFEM_USE_MPI
       void Init(HYPRE_BigInt *, HYPRE_BigInt *);
       HypreParMatrix * Duuf(const BlockVector &);
       HypreParMatrix * Dumf(const BlockVector &);
       HypreParMatrix * Dmuf(const BlockVector &);
       HypreParMatrix * Dmmf(const BlockVector &);
       HypreParMatrix * Duc(const BlockVector &);
       HypreParMatrix * Dmc(const BlockVector &);
    #else
       void Init(int, int);
       SparseMatrix * Duuf(const BlockVector &);
       SparseMatrix * Dumf(const BlockVector &);
       SparseMatrix * Dmuf(const BlockVector &);
       SparseMatrix * Dmmf(const BlockVector &);
       SparseMatrix * Duc(const BlockVector &);
       SparseMatrix * Dmc(const BlockVector &);
    #endif

    void c(const BlockVector &, Vector &) const;
    
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
    // output: The Hessian of the energy objective at d, a pointer to a (HyprePar or Sparse) Matrix
    #ifdef MFEM_USE_MPI
        virtual HypreParMatrix * DddE(const Vector &d) = 0;
    #else
	virtual SparseMatrix   * DddE(const Vector &d) = 0;
    #endif

    // Constraint function g(d) >= 0, e.g., gap function
    // input: d, an mfem::Vector,
    //       gd, an mfem::Vector, which upon successfully calling the g method will be
    //                            the evaluation of the function g at d
    // output: none
    virtual void g(const Vector &d, Vector &gd) const = 0;

    // Jacobian of constraint function Dg / Dd, e.g., gap function Jacobian
    // input:  d, an mfem::Vector,
    // output: The Jacobain of the constraint function g at d, a pointer to a (HyprePar or Sparse) Matrix
    #ifdef MFEM_USE_MPI
        virtual HypreParMatrix * Ddg(const Vector &) = 0;
    #else
	virtual SparseMatrix   * Ddg(const Vector &) = 0;
    #endif
    virtual ~OptProblem();
};






class ObstacleProblem : public OptProblem
{
protected:
   // data to define energy objective function e(d) = 0.5 d^T K d - f^T d, g(d) = d >= \psi
   // stiffness matrix used to define objective
   #ifdef MFEM_USE_MPI
      ParFiniteElementSpace * Vh;
      ParBilinearForm * Kform;
      ParLinearForm   * fform;
      HypreParMatrix    K;
      HypreParMatrix  * J;
   #else
      FiniteElementSpace * Vh;
      BilinearForm * Kform;
      LinearForm   * fform;
      SparseMatrix K;
      SparseMatrix * J;
   #endif
   Array<int> ess_tdof_list; 
   Vector f;
   Vector psi;
public :
   double E(const Vector &) const;
   void DdE(const Vector &, Vector &) const;
   void g(const Vector &, Vector &) const;
   #ifdef MFEM_USE_MPI
      ObstacleProblem(ParFiniteElementSpace*, double (*fSource)(const Vector &), double (*obstacleSource)(const Vector &));
      ObstacleProblem(ParFiniteElementSpace*, double (*fSource)(const Vector &), double (*obstacleSource)(const Vector &), Array<int> tdof_list, Vector &);
      HypreParMatrix* DddE(const Vector &);
      HypreParMatrix* Ddg(const Vector &);
   #else
      ObstacleProblem(FiniteElementSpace*, double (*fSource)(const Vector &), double (*obstacleSource)(const Vector &));
      SparseMatrix* DddE(const Vector &);
      SparseMatrix* Ddg(const Vector &);
   #endif      
   virtual ~ObstacleProblem();
};

//
//
//
//class ReducedProblem : public ParOptProblem
//{
//protected:
//  HypreParMatrix *J;
//  HypreParMatrix *P; // projector
//  ParOptProblem  *problem;
//public:
//  ReducedProblem(ParOptProblem *problem, HYPRE_Int * constraintMask);
//  ReducedProblem(ParOptProblem *problem, HypreParVector & constraintMask);
//  double E(const Vector &) const;
//  void DdE(const Vector &, Vector &) const;
//  HypreParMatrix * DddE(const Vector &);
//  void g(const Vector &, Vector &) const;
//  HypreParMatrix * Ddg(const Vector &);
//  virtual ~ReducedProblem();
//};




#endif
