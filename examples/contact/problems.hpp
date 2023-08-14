#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <set>
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
   DirichletObstacleProblem(FiniteElementSpace*, Vector&,  double (*fSource)(const Vector &), double (*obstacleSource)(const Vector &), Array<int> tdof_list, bool);
   double E(const Vector &) const;
   void DdE(const Vector &, Vector &) const;
   SparseMatrix* DddE(const Vector &);
   void g(const Vector &, Vector &) const;
   SparseMatrix* Ddg(const Vector &);
   virtual ~DirichletObstacleProblem();
};


// abstract out technology for removing null rows of the Jacobian from an existing contact problem
class ReducedContactProblem : public ContactProblem
{
protected:
   Array<int> activeConstraints;
   Array<int> fixedDofs;
   ContactProblem * contact;
   int dimSin;
public:
   ReducedContactProblem(ContactProblem * contact, Array<int> activeConstraints, Array<int> fixedDofs);
   double E(const Vector &) const;
   void DdE(const Vector &, Vector &) const;
   SparseMatrix* DddE(const Vector &);
   void g(const Vector &, Vector &) const;
   SparseMatrix* Ddg(const Vector &);
   virtual ~ReducedContactProblem();
};


class QPContactProblem : public ContactProblem
{
protected:
  SparseMatrix *K;
  SparseMatrix *J;
  Vector f;
  Vector g0;
public:
  QPContactProblem(const SparseMatrix, const SparseMatrix, const Vector, const Vector);
  double E(const Vector &) const;
  void DdE(const Vector &, Vector &) const;
  SparseMatrix* DddE(const Vector &);
  void g(const Vector &, Vector &) const;
  SparseMatrix* Ddg(const Vector &);
  virtual ~QPContactProblem();
};


typedef int Index;
typedef double Number;

class ExContactBlockTL : public ContactProblem
{
public:
   double E(const Vector &) const;
   void DdE(const Vector &, Vector &) const;
   SparseMatrix* DddE(const Vector &);
   void g(const Vector &, Vector &) const;
   SparseMatrix* Ddg(const Vector &);
   FiniteElementSpace GetVh1();
   FiniteElementSpace GetVh2();

public:
   /** default constructor */
   ExContactBlockTL(int );
   

   /** default destructor */
   virtual ~ExContactBlockTL();

   ///**@name Overloaded from TNLP */
   ///** Method to return some info about the nlp */
   //virtual bool get_nlp_info(
   //   Index&          n,
   //   Index&          m,
   //   Index&          nnz_jac_g,
   //   Index&          nnz_h_lag,
   //   IndexStyleEnum& index_style
   //);

   ///** Method to return the bounds for my problem */
   //virtual bool get_bounds_info(
   //   Index   n,
   //   Number* x_l,
   //   Number* x_u,
   //   Index   m,
   //   Number* g_l,
   //   Number* g_u
   //);

   ///** Method to return the starting point for the algorithm */
   //virtual bool get_starting_point(
   //   Index   n,
   //   bool    init_x,
   //   Number* x,
   //   bool    init_z,
   //   Number* z_L,
   //   Number* z_U,
   //   Index   m,
   //   bool    init_lambda,
   //   Number* lambda
   //);

   /* Method to return the objective value */
   virtual bool eval_f(
      Index         n,
      const Number* x,
      bool          new_x,
      Number&       obj_value
   ) const;

   /* Method to return the gradient of the objective */
   virtual bool eval_grad_f(
      Index         n,
      const Number* x,
      bool          new_x,
      Number*       grad_f
   ) const;

   /* Method to return the constraint residuals */
   virtual bool eval_g(
      Index         n,
      const Number* x,
      bool          new_x,
      Index         m,
      Number*       cons
   ) const;

   /* Method to return:
      1) The structure of the Jacobian (if "values" is NULL)
      2) The values of the Jacobian (if "values" is not NULL)
   */
   virtual bool eval_jac_g(
      Index         n,
      const Number* x,
      bool          new_x,
      Index         m,
      Index         nele_jac,
      Index*        iRow,
      Index*        jCol,
      Number*       values
   ) const;

   /* Method to return:
    *   1) The structure of the Hessian of the Lagrangian (if "values" is NULL)
    *   2) The values of the Hessian of the Lagrangian (if "values" is not NULL)
   */
   virtual bool eval_h(
      Index         n,
      const Number* x,
      bool          new_x,
      Number        obj_factor,
      Index         m,
      const Number* lambda,
      bool          new_lambda,
      Index         nele_hess,
      Index*        iRow,
      Index*        jCol,
      Number*       values
   );

   ///** This method is called when the algorithm is complete so the TNLP can store/write the solution */
   //virtual void finalize_solution(
   //   SolverReturn               status,
   //   Index                      n,
   //   const Number*              x,
   //   const Number*              z_L,
   //   const Number*              z_U,
   //   Index                      m,
   //   const Number*              g,
   //   const Number*              lambda,
   //   Number                     obj_value,
   //   const IpoptData*           ip_data,
   //   IpoptCalculatedQuantities* ip_cq
   //);

private:
   void update_g() const;
   void update_jac();
   void update_hess();

private:
   /**@name Methods to block default compiler methods.
    *
    * The compiler automatically generates the following three methods.
    *  Since the default compiler implementation is generally not what
    *  you want (for all but the most simple classes), we usually
    *  put the declarations of these methods in the private section
    *  and never implement them. This prevents the compiler from
    *  implementing an incorrect "default" behavior without us
    *  knowing. (See Scott Meyers book, "Effective C++")
    */
   ExContactBlockTL(
      const ExContactBlockTL&
   );

   ExContactBlockTL& operator=(
      const ExContactBlockTL&
   );

   Array<int> attr;
   Array<int> m_attr;
   Array<int> s_conn; // connectivity of the second/slave mesh
   std::string mesh_file1;
   std::string mesh_file2;
   Mesh* mesh1;
   Mesh* mesh2;
   FiniteElementCollection* fec1;
   FiniteElementCollection* fec2;
   FiniteElementSpace* fespace1;
   FiniteElementSpace* fespace2;
   Array<int> ess_tdof_list1;
   Array<int> ess_tdof_list2;
   GridFunction nodes0;
   GridFunction* nodes1;
   GridFunction* nodes2;
   mutable GridFunction* x1;
   mutable GridFunction* x2;
   LinearForm* b1;
   LinearForm* b2;
   PWConstCoefficient* lambda1_func;
   PWConstCoefficient* lambda2_func;
   PWConstCoefficient* mu1_func;
   PWConstCoefficient* mu2_func;
   BilinearForm* a1;
   BilinearForm* a2;

   mfem::Vector lambda1;
   mfem::Vector lambda2;
   mfem::Vector mu1;
   mfem::Vector mu2;
   mutable mfem::Vector xyz;

   std::set<int> bdryVerts2;

   int dim;
   // degrees of freedom of both meshes
   int ndof_1;
   int ndof_2;
   int ndofs;
   // number of nodes for each mesh
   int nnd_1;
   int nnd_2;
   int nnd;

   int npoints;

   SparseMatrix A1;
   mfem::Vector B1, X1;
   SparseMatrix A2;
   mfem::Vector B2, X2;

   SparseMatrix* K;
   mutable mfem::Vector gapv;
   mutable mfem::Vector m_xi;
   mutable mfem::Vector xs;

   mutable Array<int> m_conn; // only works for linear elements that have 4 vertices!
   mutable DenseMatrix* coordsm;
   mutable SparseMatrix* M;

   mutable std::vector<SparseMatrix>* dM;

   Array<int> Dirichlet_dof;
   Array<double> Dirichlet_val;

public: 
   Mesh * GetMesh1() {return mesh1;}
   Mesh * GetMesh2() {return mesh2;}
   Array<int> GetDirichletDofs() {return Dirichlet_dof;}
   Array<double> GetDirichletVals() {return Dirichlet_val;}

};

#endif
