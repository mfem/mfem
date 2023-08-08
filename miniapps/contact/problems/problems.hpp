#include "problems_util.hpp"


class ElasticityProblem
{
private:
   bool formsystem = false;
   Mesh * mesh = nullptr;
   int order;
   int ndofs;
   FiniteElementCollection * fec = nullptr;
   FiniteElementSpace * fes = nullptr;
   Vector lambda, mu;
   PWConstCoefficient lambda_cf, mu_cf;
   Array<int> ess_bdr, ess_tdof_list;
   BilinearForm *a=nullptr;
   LinearForm b;
   GridFunction x;
   SparseMatrix A;
   Vector B,X;
   void Init();
public:
   ElasticityProblem(const char *mesh_file , int ref, int order_ = 1) : order(order_) 
   {
      mesh = new Mesh(mesh_file,1,1);
      for (int i = 0; i<ref; i++)
      {
         mesh->UniformRefinement();
      }
      Init();
   }

   Mesh * GetMesh() { return mesh; }
   FiniteElementSpace * GetFESpace() { return fes; }
   int GetNumDofs() { return ndofs; }
   SparseMatrix & GetOperator() 
   {
      MFEM_VERIFY(formsystem, "System not formed yet. Call FormLinearSystem()"); 
      return A; 
   }

   Vector & GetRHS() 
   {
      MFEM_VERIFY(formsystem, "System not formed yet. Call FormLinearSystem()"); 
      return B; 
   }

   void FormLinearSystem();
   void UpdateLinearSystem();

   void SetDisplacementDirichletData(const Vector & delta) 
   {
      VectorConstantCoefficient delta_cf(delta);
      x.ProjectBdrCoefficient(delta_cf,ess_bdr);
   };

   void UpdateDisplacement(const Vector & x_) 
   {
      // x = x_;
      // mesh->MoveVertices(x);
      // mesh->NodesUpdated();
   };

   GridFunction & GetDisplacementGridFunction() {return x;};
   Array<int> & GetEssentialDofs() {return ess_tdof_list;};

   ~ElasticityProblem()
   {
      delete a;
      delete fes;
      delete fec;
      delete mesh;
   }
};


class ContactProblem
{
private:
   ElasticityProblem * prob1 = nullptr;
   ElasticityProblem * prob2 = nullptr;
   GridFunction nodes0;
   GridFunction *nodes1 = nullptr;
   std::set<int> contact_vertices;
   bool recompute = true;

protected:
   int npoints=0;
   SparseMatrix *K =nullptr;
   BlockVector *B = nullptr;
   Vector gapv;
   Array<SparseMatrix*> dM;
   SparseMatrix * M=nullptr;
   void ComputeContactVertrices();
public:
   ContactProblem(ElasticityProblem * prob1_, ElasticityProblem * prob2_);

   ElasticityProblem * GetElasticityProblem1() {return prob1;}
   ElasticityProblem * GetElasticityProblem2() {return prob2;}

   int GetNumDofs() {return K->Height();}
   int GetNumContraints() {return npoints;}
   Vector & GetGapFunction() {return gapv;}
   SparseMatrix * GetJacobian() {return M;}
   Array<SparseMatrix*> & GetHessian() {return dM;}
   void ComputeGapFunctionAndDerivatives(const Vector & displ1, const Vector &displ2);

   virtual double E(const Vector & d);
   virtual void DdE(const Vector &d, Vector &gradE);
   virtual SparseMatrix* DddE(const Vector &d);
   void g(const Vector &d, Vector &gd);
   virtual SparseMatrix* Ddg(const Vector &d);
   virtual SparseMatrix* lDddg(const Vector &d, const Vector &l);

   ~ContactProblem()
   {
      delete B;
      delete K;
      delete M;
      for (int i = 0; i<dM.Size(); i++)
      {
         delete dM[i];
      }
   }
};


class QPContactProblem : public ContactProblem
{
private:
   int dimD, dimS;
public:
   QPContactProblem(ElasticityProblem * prob1_, ElasticityProblem * prob2_);

   double E(const Vector & d);
   void DdE(const Vector &d, Vector &gradE);
   SparseMatrix* DddE(const Vector &d);
   void g(const Vector &d, Vector &gd);
   SparseMatrix* Ddg(const Vector &d);
   SparseMatrix* lDddg(const Vector &d, const Vector &l);
};


class QPOptContactProblem
{
private:
   ContactProblem * problem = nullptr;
   int dimU, dimM, dimC;
   Array<int> block_offsets;
   Vector ml;
   SparseMatrix * NegId = nullptr;
public:
   QPOptContactProblem(ContactProblem * problem_);
   int GetDimU();
   int GetDimM();
   int GetDimC();
   Vector & Getml();
   SparseMatrix * Duuf(const BlockVector &);
   SparseMatrix * Dumf(const BlockVector &);
   SparseMatrix * Dmuf(const BlockVector &);
   SparseMatrix * Dmmf(const BlockVector &);
   SparseMatrix * Duc(const BlockVector &);
   SparseMatrix * Dmc(const BlockVector &);
   SparseMatrix * lDuuc(const BlockVector &, const Vector &);
   void c(const BlockVector &, Vector &);
   double CalcObjective(const BlockVector &);
   void CalcObjectiveGrad(const BlockVector &, BlockVector &);
   ~QPOptContactProblem();
};
