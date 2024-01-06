#include "mfem.hpp"
#include "parproblems_util.hpp"
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
    virtual double CalcObjective(const BlockVector &) = 0;
    virtual void Duf(const BlockVector &, Vector &) = 0;
    virtual void Dmf(const BlockVector &, Vector &) = 0;
    void CalcObjectiveGrad(const BlockVector &, BlockVector &);
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
    virtual void c(const BlockVector &, Vector &) = 0;
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
    double CalcObjective(const BlockVector &); 
    void Duf(const BlockVector &, Vector &);
    void Dmf(const BlockVector &, Vector &);
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
    virtual void g(const Vector &d, Vector &gd) = 0;

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
   double E(const Vector &);
   void DdE(const Vector &, Vector &);
   void g(const Vector &, Vector &);
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

#ifdef MFEM_USE_MPI
   class ParElasticityProblem
   {
   private:
      MPI_Comm comm;
      bool formsystem = false;
      ParMesh * pmesh = nullptr;
      int order;
      int ndofs;
      int ntdofs;
      int gndofs;
      FiniteElementCollection * fec = nullptr;
      ParFiniteElementSpace * fes = nullptr;
      Vector lambda, mu;
      PWConstCoefficient lambda_cf, mu_cf;
      Array<int> ess_bdr, ess_tdof_list;
      ParBilinearForm *a=nullptr;
      ParLinearForm b;
      ParGridFunction x;
      HypreParMatrix A;
      Vector B,X;
      void Init();
      bool own_mesh;
   public:
      ParElasticityProblem(MPI_Comm comm_, const char *mesh_file , int sref, int pref, int order_ = 1) : comm(comm_), order(order_) 
      {
         own_mesh = true;
         Mesh * mesh = new Mesh(mesh_file,1,1);
         for (int i = 0; i<sref; i++)
         {
            mesh->UniformRefinement();
         }
         pmesh = new ParMesh(comm,*mesh);
         MFEM_VERIFY(pmesh->GetNE(), "ParElasticityProblem::Empty partition");
         delete mesh;
         for (int i = 0; i<pref; i++)
         {
            pmesh->UniformRefinement();
         }
         Init();
      }
   
      ParElasticityProblem(ParMesh * pmesh_, int order_ = 1) :  pmesh(pmesh_), order(order_)
      {
         own_mesh = false;
         comm = pmesh->GetComm();
         Init();
      }
   
      ParMesh * GetMesh() { return pmesh; }
      ParFiniteElementSpace * GetFESpace() { return fes; }
      FiniteElementCollection * GetFECol() { return fec; }
      int GetNumDofs() { return ndofs; }
      int GetNumTDofs() { return ntdofs; }
      int GetGlobalNumDofs() { return gndofs; }
      HypreParMatrix & GetOperator() 
      { 
         MFEM_VERIFY(formsystem, "System not formed yet. Call FormLinearSystem()"); 
         return A; 
      }
      Vector & GetRHS() 
      { 
         MFEM_VERIFY(formsystem, "System not formed yet. Call FormLinearSystem()"); 
         return B; 
      }
   
      void SetLambda(const Vector & lambda_) 
      { 
         lambda = lambda_; 
         lambda_cf.UpdateConstants(lambda);
      }
      void SetMu(const Vector & mu_) 
      { 
         mu = mu_; 
         mu_cf.UpdateConstants(mu);
      }
   
      void FormLinearSystem();
      void UpdateLinearSystem();
   
      void SetDisplacementDirichletData(const Vector & delta) 
      {
         VectorConstantCoefficient delta_cf(delta);
         x.ProjectBdrCoefficient(delta_cf,ess_bdr);
      };
   
      ParGridFunction & GetDisplacementGridFunction() {return x;};
      Array<int> & GetEssentialDofs() {return ess_tdof_list;};
   
      ~ParElasticityProblem()
      {
         delete a;
         delete fes;
         delete fec;
         if (own_mesh)
         {
            delete pmesh;
         }
      }
   };


   class ParContactProblem : public OptProblem
   {
   private:
      MPI_Comm comm;
      int numprocs;
      int myid;
      ParElasticityProblem * prob1 = nullptr;
      ParElasticityProblem * prob2 = nullptr;
      ParFiniteElementSpace * vfes1 = nullptr;
      ParFiniteElementSpace * vfes2 = nullptr;
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
      int npoints=0;
      int gnpoints=0;
      int nv, gnv;
      HypreParMatrix * K = nullptr;
      BlockVector *B = nullptr;
      Vector gapv;
      HypreParMatrix * M=nullptr;
      Array<HypreParMatrix*> dM;
      void ComputeContactVertices();
   
   public:
      ParContactProblem(ParElasticityProblem * prob1_, ParElasticityProblem * prob2_);
   
      ParElasticityProblem * GetElasticityProblem1() {return prob1;}
      ParElasticityProblem * GetElasticityProblem2() {return prob2;}
      MPI_Comm GetComm() {return comm;}
      int GetNumDofs() {return K->Height();}
      int GetGlobalNumDofs() {return K->GetGlobalNumRows();}
      int GetNumContraints() {return npoints;}
      int GetGlobalNumConstraints() {return gnpoints;}
      
      std::vector<int> & GetDofOffets() { return dof_offsets; }
      std::vector<int> & GetVertexOffsets() { return vertex_offsets; }
      std::vector<int> & GetConstraintsOffsets() { return constraints_offsets; }
      Array<int> & GetConstraintsStarts() { return constraints_starts; }
      
      Vector & GetGapFunction() {return gapv;}
   
      HypreParMatrix * GetJacobian() {return M;}
      Array<HypreParMatrix*> & GetHessian() {return dM;}
      void ComputeGapFunctionAndDerivatives(const Vector & displ1, const Vector &displ2);
   
      double E(const Vector & d);
      void DdE(const Vector &d, Vector &gradE);
      HypreParMatrix* DddE(const Vector &d);
      //void g(const Vector &d, Vector &gd, bool compute_hessians_ = true);
      void g(const Vector &d, Vector &gd);
      HypreParMatrix* Ddg(const Vector &d);
      HypreParMatrix* lDddg(const Vector &d, const Vector &l);
   
      ~ParContactProblem()
      {
         delete B;
         delete K;
         delete M;
         for (int i = 0; i<dM.Size(); i++)
         {
            delete dM[i];
         }
         delete vfes1;
         delete vfes2;
      }
   };




#endif


#endif
