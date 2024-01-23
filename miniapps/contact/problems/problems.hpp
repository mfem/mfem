#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "problems_util.hpp"

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
    bool parallel;
public:
    GeneralOptProblem();
    virtual double CalcObjective(const BlockVector &) = 0;
    virtual void Duf(const BlockVector &, Vector &) = 0;
    virtual void Dmf(const BlockVector &, Vector &) = 0;
    void CalcObjectiveGrad(const BlockVector &, BlockVector &);
    #ifdef MFEM_USE_MPI
       void InitGeneral(HYPRE_BigInt * dofOffsetsU_, HYPRE_BigInt * dofOffsetsM_);
       HYPRE_BigInt * GetDofOffsetsU() const { return dofOffsetsU; };
       HYPRE_BigInt * GetDofOffsetsM() const { return dofOffsetsM; };
    #endif
    virtual void InitGeneral(int dimU, int dimM);
    virtual Operator * Duuf(const BlockVector &) = 0;
    virtual Operator * Dumf(const BlockVector &) = 0;
    virtual Operator * Dmuf(const BlockVector &) = 0;
    virtual Operator * Dmmf(const BlockVector &) = 0;
    virtual Operator * Duc(const BlockVector &) = 0;
    virtual Operator * Dmc(const BlockVector &) = 0;
    virtual void c(const BlockVector &, Vector &) = 0;
    int GetDimU() const { return dimU; };
    int GetDimM() const { return dimM; }; 
    int GetDimC() const { return dimC; };
    int GetDimUGlb() const { return dimUGlb; };
    int GetDimMGlb() const { return dimMGlb; };
    int GetDimCGlb() const { return dimCGlb; };
    bool IsParallel() const { return parallel; };
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
#endif
   SparseMatrix * Isparse;
public:
   OptProblem();
    
   // GeneralOptProblem methods are defined in terms of
   // OptProblem specific methods: E, DdE, DddE, g, Ddg
   double CalcObjective(const BlockVector &); 
   void Duf(const BlockVector &, Vector &);
   void Dmf(const BlockVector &, Vector &);
#ifdef MFEM_USE_MPI
   void Init(HYPRE_BigInt *, HYPRE_BigInt *);
#endif
    void Init(int, int);
    Operator * Duuf(const BlockVector &);
    Operator * Dumf(const BlockVector &);
    Operator * Dmuf(const BlockVector &);
    Operator * Dmmf(const BlockVector &);
    Operator * Duc(const BlockVector &);
    Operator * Dmc(const BlockVector &);

    void c(const BlockVector &, Vector &);
    
    // ParOptProblem specific methods:
    
    // energy objective function e(d)
    // input: d an mfem::Vector
    // output: e(d) a double
    virtual double E(const Vector &d) = 0;

    // gradient of energy objective De / Dd
    // input: d an mfem::Vector,
    //        gradE an mfem::Vector, which will be the gradient of E at d
    // output: none    
    virtual void DdE(const Vector &d, Vector &gradE) = 0;

    // Hessian of energy objective D^2 e / Dd^2
    // input:  d, an mfem::Vector
    // output: The Hessian of the energy objective at d, a pointer to a (HyprePar or Sparse) Matrix
    virtual Operator * DddE(const Vector &d) = 0;

    // Constraint function g(d) >= 0, e.g., gap function
    // input: d, an mfem::Vector,
    //       gd, an mfem::Vector, which upon successfully calling the g method will be
    //                            the evaluation of the function g at d
    // output: none
    virtual void g(const Vector &d, Vector &gd) = 0;

    // Jacobian of constraint function Dg / Dd, e.g., gap function Jacobian
    // input:  d, an mfem::Vector,
    // output: The Jacobain of the constraint function g at d, a pointer to a (HyprePar or Sparse) Matrix
    virtual Operator * Ddg(const Vector &) = 0;
    virtual ~OptProblem();
};






class ObstacleProblem : public OptProblem
{
protected:
   // data to define energy objective function e(d) = 0.5 d^T K d - f^T d, g(d) = d >= \psi
   // stiffness matrix used to define objective
#ifdef MFEM_USE_MPI
   ParFiniteElementSpace * Vhp;
   ParBilinearForm * Kformp;
   ParLinearForm   * fformp;
   HypreParMatrix    Kh;
   HypreParMatrix  * Jh;
#endif
   FiniteElementSpace * Vh;
   BilinearForm * Kform;
   LinearForm   * fform;
   SparseMatrix K;
   SparseMatrix * J;
   Array<int> ess_tdof_list; 
   Vector f;
   Vector psi;
public :
   double E(const Vector & d);
   void DdE(const Vector & d, Vector & dE);
   void   g(const Vector & d, Vector & gd);
   Operator * DddE(const Vector & d);
   Operator * Ddg (const Vector & d);
   ObstacleProblem(FiniteElementSpace*, double (*fSource)(const Vector &), double (*obstacleSource)(const Vector &));
   virtual ~ObstacleProblem();
};


class ElasticityProblem
{
   private:
      bool formsystem = false;
      bool own_mesh;
      bool parallel;
      Mesh * mesh = nullptr;
      int order;
      int ndofs;
      int ntdofs;
      int gndofs;
      FiniteElementCollection * fec = nullptr;
      FiniteElementSpace      * fes = nullptr;
      Vector lambda, mu;
      PWConstCoefficient lambda_cf, mu_cf;
      Array<int> ess_bdr, ess_tdof_list;
      BilinearForm *a = nullptr;
      LinearForm    b;
      GridFunction  x;
      SparseMatrix A;
      Vector X, B;
      
#ifdef MFEM_USE_MPI 
      MPI_Comm comm;
      ParMesh * pmesh = nullptr;
      ParFiniteElementSpace * fesp = nullptr;
      ParBilinearForm * ap = nullptr;
      ParLinearForm bp;
      ParGridFunction xp;
      HypreParMatrix Ap;
#endif
      void Init();
      
   public:
      ElasticityProblem(const char * mesh_file, int ref, int order_);
#ifdef MFEM_USE_MPI
      ElasticityProblem(MPI_Comm comm_, const char *mesh_file , int sref, int pref, int order_);
      ElasticityProblem(ParMesh * pmesh_, int order_);
#endif
      Mesh * GetMesh();
      FiniteElementCollection * GetFECol() { return fec; };
      FiniteElementSpace * GetFESpace();
      int GetNumDofs() { return ndofs; };
      int GetNumTDofs() { return ntdofs; };
      int GetGlobalNumDofs() { return gndofs; };
      Operator & GetOperator();
      Vector & GetRHS();
      void SetLambda(const Vector & lambda_);
      void SetMu(const Vector & mu_);
      void FormLinearSystem();
      void UpdateLinearSystem();
      void SetDisplacementDirichletData(const Vector & delta);
      GridFunction & GetDisplacementGridFunction();
#ifdef MFEM_USE_MPI
      ParGridFunction & GetDisplacementParGridFunction();
#endif
      Array<int> & GetEssentialDofs() { return ess_tdof_list; };
      bool IsParallel() const { return parallel; };
      ~ElasticityProblem();
};


class ContactProblem : public OptProblem
{
private:
   int numprocs;
   int myid;
   ElasticityProblem * prob1 = nullptr;
   ElasticityProblem * prob2 = nullptr;
   FiniteElementSpace * vfes1 = nullptr;
   FiniteElementSpace * vfes2 = nullptr;
#ifdef MFEM_USE_MPI
   MPI_Comm comm;
   ParFiniteElementSpace * vfes1p = nullptr;
   ParFiniteElementSpace * vfes2p = nullptr;
#endif
   int dim;
   GridFunction nodes0;
   GridFunction *nodes1 = nullptr;
   std::set<int> contact_vertices;
   bool recompute = true;
   bool compute_hessians = true;
   std::vector<int> dof_offsets;
   std::vector<int> vertex_offsets; 
   std::vector<int> constraints_offsets; 
   Array<int> tdof_offsets;
   Array<int> constraints_starts;
   Array<int> globalvertices1;
   Array<int> globalvertices2;
   Array<int> vertices2;
   Array<int> vertices1;
   
   protected:
      bool parallel;
      int npoints=0;
      int gnpoints=0;
      int nv, gnv;
      SparseMatrix * K = nullptr;
      BlockVector *B = nullptr;
      Vector gapv;
      SparseMatrix * M = nullptr;
      Array<SparseMatrix *> dM;
#ifdef MFEM_USE_MPI
      HypreParMatrix * Kp = nullptr;
      HypreParMatrix * Mp = nullptr;
      Array<HypreParMatrix*> dMp;
      void ParComputeContactVertices();
#endif
      void ComputeContactVertices();
   
   public:
      ContactProblem(ElasticityProblem * prob1_, ElasticityProblem * prob2_);
#ifdef MFEM_USE_MPI
      MPI_Comm GetComm() { return comm; };
      void ParComputeGapFunctionAndDerivatives(const Vector & displ1, const Vector & displ2);
#endif
      void ComputeGapFunctionAndDerivatives(const Vector & displ1, const Vector & displ2);
      int GetNumDofs();
      int GetGlobalNumDofs();
      int GetNumConstraints() { return npoints; };
      int GetGlobalNumConstraints() { return gnpoints; };
      std::vector<int> & GetDofOffets() { return dof_offsets; }
      std::vector<int> & GetVertexOffsets() { return vertex_offsets; }
      std::vector<int> & GetConstraintsOffsets() { return constraints_offsets; }
      Array<int> & GetConstraintsStarts() { return constraints_starts; }
      Vector & GetGapFunction() { return gapv; }
      Operator * GetJacobian();
      Array<Operator *> GetHessian();
      double E(const Vector & d);
      void DdE(const Vector &d, Vector &gradE);
      Operator * DddE(const Vector &d);
      void g(const Vector &d, Vector &gd); // todo : add in cpp file
      Operator * Ddg(const Vector &d);
};


#endif
