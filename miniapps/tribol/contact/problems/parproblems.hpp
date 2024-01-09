
#include "parproblems_util.hpp"

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
      bool vis = false;
      if (vis)
      {
         int myid, num_procs;
         MPI_Comm_rank(comm, &myid);
         MPI_Comm_size(comm, &num_procs);
         char vishost[] = "localhost";
         int  visport   = 19916;
         socketstream sol_sock(vishost, visport);
         sol_sock << "parallel " << num_procs << " " << myid << "\n";
         sol_sock.precision(8);
         sol_sock << "solution\n" << *pmesh << x << std::flush;
         MFEM_ABORT("");
      }
   };

   void SetDisplacementDirichletData(const Vector & delta, Array<int> essbdr) 
   {
      VectorConstantCoefficient delta_cf(delta);
      x.ProjectBdrCoefficient(delta_cf,essbdr);
      bool vis = false;
      if (vis)
      {
         int myid, num_procs;
         MPI_Comm_rank(comm, &myid);
         MPI_Comm_size(comm, &num_procs);
         char vishost[] = "localhost";
         int  visport   = 19916;
         socketstream sol_sock(vishost, visport);
         sol_sock << "parallel " << num_procs << " " << myid << "\n";
         sol_sock.precision(8);
         sol_sock << "solution\n" << *pmesh << x << std::flush;
         MFEM_ABORT("");
      }
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


class ParContactProblem
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
   void g(const Vector &d, Vector &gd, bool compute_hessians_ = true);
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


class QPOptParContactProblem
{
private:
   ParContactProblem * problem = nullptr;
   int dimU, dimM, dimC;
   // Array<int> block_offsets;
   Vector ml;
   HypreParMatrix * NegId = nullptr;
public:
   QPOptParContactProblem(ParContactProblem * problem_);
   int GetDimU();
   int GetDimM();
   int GetDimC();
   Vector & Getml();
   MPI_Comm GetComm() {return problem->GetComm();}
   int * GetConstraintsStarts() {return problem->GetConstraintsStarts().GetData();}
   int GetGlobalNumConstraints() {return problem->GetGlobalNumConstraints();}

   ParElasticityProblem * GetElasticityProblem1() {return problem->GetElasticityProblem1();}
   ParElasticityProblem * GetElasticityProblem2() {return problem->GetElasticityProblem2();}

   HypreParMatrix * Duuf(const BlockVector &);
   HypreParMatrix * Dumf(const BlockVector &);
   HypreParMatrix * Dmuf(const BlockVector &);
   HypreParMatrix * Dmmf(const BlockVector &);
   HypreParMatrix * Duc(const BlockVector &);
   HypreParMatrix * Dmc(const BlockVector &);
   HypreParMatrix * lDuuc(const BlockVector &, const Vector &);
   void c(const BlockVector &, Vector &);
   double CalcObjective(const BlockVector &);
   void CalcObjectiveGrad(const BlockVector &, BlockVector &);
   ~QPOptParContactProblem();
};



#ifdef MFEM_USE_TRIBOL
class ParContactProblemTribol
{
private:
   MPI_Comm comm;
   int numprocs;
   int myid;
   ParElasticityProblem * prob = nullptr;
   ParFiniteElementSpace * vfes = nullptr;
   ParMesh * pmesh = nullptr;
   int dim;
   GridFunction nodes0;
   GridFunction *nodes1 = nullptr;
   std::vector<int> dof_offsets;
   std::vector<int> constraints_offsets; 
   Array<int> tdof_offsets;
   Array<int> constraints_starts;

   void SetupTribol();

protected:
   HypreParMatrix * K = nullptr;
   BlockVector *B = nullptr;
   HypreParMatrix * M=nullptr;

public:
// for now we work on 1 (merged mesh).
// TODO work with 2 meshes independently 
   ParContactProblemTribol(ParElasticityProblem * prob_);

   ~ParContactProblemTribol();
};


#endif