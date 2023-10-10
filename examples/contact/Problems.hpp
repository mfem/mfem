#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <set>

using namespace std;
using namespace mfem;

#ifndef PROBLEM_DEFS
#define PROBLEM_DEFS



// abstract GeneralOptProblem class
// for the problem
// min_(u,m) f(u,m) 
// such that c(u,m)=0 and m >= ml
class GeneralOptProblem
{
protected:
    int dimU, dimM, dimC;
    Array<int> block_offsetsx;
    Vector ml;
public:
    GeneralOptProblem();
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
    virtual SparseMatrix* lDuuc(const BlockVector &, const Vector &) = 0;
    virtual SparseMatrix* lDumc(const BlockVector &, const Vector &) = 0;
    virtual SparseMatrix* lDmuc(const BlockVector &, const Vector &) = 0;
    virtual SparseMatrix* lDmmc(const BlockVector &, const Vector &) = 0;
    // TO DO: include log-barrier lumped-mass and pass that
    // to the optimizer
    //virtual SparseMatrix* GetLogBarrierLumpedMass() = 0;
    int GetDimU() const { return dimU; };
    int GetDimM() const { return dimM; }; 
    int GetDimC() const { return dimC; };
    Vector Getml() const { return ml; };
    ~GeneralOptProblem();
};


// abstract OptProblem class
// of the form
// min_d e(d) s.t. g(d) >= 0
class OptProblem : public GeneralOptProblem
{
protected:
    int dimD;
    int dimS;
    Array<int> block_offsetsx;
    SparseMatrix * negIdentity;
    SparseMatrix * zeroMatum;
    SparseMatrix * zeroMatmu;
    SparseMatrix * zeroMatmm;
public:
    //OptProblem(int, int);        // constructor
    OptProblem();
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
    SparseMatrix* lDuuc(const BlockVector &, const Vector &);
    SparseMatrix* lDumc(const BlockVector &, const Vector &);
    SparseMatrix* lDmuc(const BlockVector &, const Vector &);
    SparseMatrix* lDmmc(const BlockVector &, const Vector &);
    virtual double E(const Vector &) const = 0;           // objective e(d) (energy function)
    virtual void DdE(const Vector &, Vector &) const = 0; // gradient of objective De / Dd
    virtual SparseMatrix* DddE(const Vector &) = 0;       // Hessian of objective D^2 e / D d^2
    virtual void g(const Vector &, Vector &) const = 0;   // inequality constraint g(d) >= 0 (gap function)
    virtual SparseMatrix* Ddg(const Vector &) = 0;        // Jacobian of inequality constraint Dg / Dd
    virtual SparseMatrix* lDddg(const Vector &, const Vector &) = 0;
    int GetDimD() const { return dimD; };
    int GetDimS() const { return dimS; };
    virtual ~OptProblem();
};


class ObstacleProblem : public OptProblem
{
protected:
   // data to define energy objective function e(d) = 0.5 d^T K d - f^T d, g(d) = d + \psi >= 0
   // stiffness matrix used to define objective
   BilinearForm *Kform;
   LinearForm   *fform;
   Array<int> ess_tdof_list;
   SparseMatrix  *K;
   SparseMatrix  *J;
   SparseMatrix  *Hcl;
   FiniteElementSpace *Vh;
   Vector f;
   Vector psil;
   Vector psiu;
   bool twoBounds;
   Vector xDC;
   double Ce;
public : 
   ObstacleProblem(FiniteElementSpace*, double (*fSource)(const Vector &), double (*obstacleSource)(const Vector &));
   ObstacleProblem(FiniteElementSpace*, Vector&,  double (*fSource)(const Vector &), double (*obstacleSource)(const Vector &), Array<int> tdof_list);
   ObstacleProblem(FiniteElementSpace*, Vector &, double (*fSource)(const Vector &), double (*obstacleSourcel)(const Vector &), double (*obstacleSourceu)(const Vector &), Array<int> tdof_list);
   double E(const Vector &) const;
   void DdE(const Vector &, Vector &) const;
   SparseMatrix* DddE(const Vector &);
   void g(const Vector &, Vector &) const;
   SparseMatrix* Ddg(const Vector &);
   SparseMatrix * lDddg(const Vector &, const Vector &);
   virtual ~ObstacleProblem();
};


class QPOptProblem : public OptProblem
{
protected:
  SparseMatrix *K;
  SparseMatrix *J;
  SparseMatrix *zeroMatdd;
  Vector f;
  Vector g0;
public:
  QPOptProblem(const SparseMatrix, const SparseMatrix, const Vector, const Vector);
  double E(const Vector &) const;
  void DdE(const Vector &, Vector &) const;
  SparseMatrix* DddE(const Vector &);
  void g(const Vector &, Vector &) const;
  SparseMatrix* Ddg(const Vector &);
  SparseMatrix * lDddg(const Vector &, const Vector &);
  virtual ~QPOptProblem();
};


class ExContactBlockTL : public OptProblem
{
public:
   double E(const Vector &) const;
   void DdE(const Vector &, Vector &) const;
   SparseMatrix* DddE(const Vector &);
   void g(const Vector &, Vector &) const;
   SparseMatrix* Ddg(const Vector &);
   SparseMatrix * lDddg(const Vector &, const Vector &);
   FiniteElementSpace GetVh1();
   FiniteElementSpace GetVh2();
   SparseMatrix *zeroMatdd;
public:
   /** default constructor */
   ExContactBlockTL(Mesh *, Mesh *, int);
   

   /** default destructor */
   virtual ~ExContactBlockTL();

private:
   void update_g() const;

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
   BlockVector *B;
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
   Array<int> block_offsets;
public: 
   Mesh * GetMesh1() {return mesh1;}
   Mesh * GetMesh2() {return mesh2;}
   GridFunction & GetMesh1GridFunction() {return *x1;}
   GridFunction & GetMesh2GridFunction() {return *x2;}
   Array<int> & GetMesh1DirichletDofs() {return ess_tdof_list1;}
   Array<int> & GetMesh2DirichletDofs() {return ess_tdof_list2;}
};



#endif
