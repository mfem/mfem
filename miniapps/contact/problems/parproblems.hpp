
#include "parproblems_util.hpp"

class ParElasticityProblem
{
private:
   MPI_Comm comm;
   bool formsystem = false;
   ParMesh * pmesh = nullptr;
   Array<int> ess_bdr_attr, ess_bdr_attr_comp;
   int order;
   int ndofs;
   int ntdofs;
   int gndofs;
   FiniteElementCollection * fec = nullptr;
   ParFiniteElementSpace * fes = nullptr;
   Vector lambda, mu;
   PWConstCoefficient lambda_cf, mu_cf;
   Array<int> ess_bdr, ess_tdof_list;
   ParBilinearForm * a = nullptr;
   ParLinearForm * b = nullptr;
   ParGridFunction x;
   HypreParMatrix A;
   Vector B,X;
   ConstantCoefficient pressure_cf;
   VectorArrayCoefficient * bf = nullptr;
   void Init();
   bool own_mesh;
public:
   ParElasticityProblem(MPI_Comm comm_, const char *mesh_file , int sref, int pref, 
                        Array<int> & ess_bdr_attr_, Array<int> & ess_bdr_attr_comp_, 
                        int order_ = 1 ) 
   : comm(comm_), ess_bdr_attr(ess_bdr_attr_),ess_bdr_attr_comp(ess_bdr_attr_comp_), order(order_)
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

   ParElasticityProblem(ParMesh * pmesh_, Array<int> & ess_bdr_attr_, Array<int> & ess_bdr_attr_comp_,  int order_ = 1) 
   :  pmesh(pmesh_), ess_bdr_attr(ess_bdr_attr_), ess_bdr_attr_comp(ess_bdr_attr_comp_), order(order_)
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

   void SetNeumanPressureData(ConstantCoefficient &f, Array<int> & bdr_marker)
   { 
      pressure_cf.constant = f.constant;
      b->AddBoundaryIntegrator(new VectorBoundaryFluxLFIntegrator(pressure_cf),bdr_marker);
   }

   void SetNeumanData(int comp, int bdrattr, double value)
   { 
      int dim = pmesh->Dimension();
      bf = new VectorArrayCoefficient(dim);
      for (int i = 0; i < dim; i++)
      {
         if (i == comp)
         {
            Vector pull_force(pmesh->bdr_attributes.Max());
            pull_force = 0.0;
            pull_force(bdrattr-1) = value;
            bf->Set(i, new PWConstCoefficient(pull_force));
         }
         else
         {
            bf->Set(i, new ConstantCoefficient(0.0));
         }
      }
      b->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(*bf));
   }

   void UpdateEssentialBC(Array<int> & ess_bdr_attr_, Array<int> & ess_bdr_attr_comp_)
   {
      ess_bdr_attr = ess_bdr_attr_;
      ess_bdr_attr_comp = ess_bdr_attr_comp_;
      ess_tdof_list.SetSize(0);
      if (pmesh->bdr_attributes.Size())
      {
         ess_bdr.SetSize(pmesh->bdr_attributes.Max());
      }
      ess_bdr = 0; 
      Array<int> ess_tdof_list_temp;
      for (int i = 0; i < ess_bdr_attr.Size(); i++ )
      {
         ess_bdr[ess_bdr_attr[i]-1] = 1;
         fes->GetEssentialTrueDofs(ess_bdr,ess_tdof_list_temp,ess_bdr_attr_comp[i]);
         ess_tdof_list.Append(ess_tdof_list_temp);
         ess_bdr[ess_bdr_attr[i]-1] = 0;
      }
   }

   void UpdateStep()
   {
      if (formsystem)
      {
         delete b;
         b = new ParLinearForm(fes);
         delete a;
         a = new ParBilinearForm(fes);
         a->AddDomainIntegrator(new ElasticityIntegrator(lambda_cf,mu_cf));

         // a->Update();
         formsystem = false;
      }
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

   void ResetDisplacementDirichletData()
   {
      x = 0.0;
   }

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
      delete b;
      delete fes;
      delete fec;
      if (own_mesh)
      {
         delete pmesh;
      }
      delete bf;
   }
};


// #ifdef MFEM_USE_TRIBOL

class ParContactProblem
{
private:
   MPI_Comm comm;
   int numprocs;
   int myid;
   ParElasticityProblem * prob = nullptr;
   ParFiniteElementSpace * vfes = nullptr;
   int dim;
   std::set<int> contact_vertices;
   std::vector<int> dof_offsets;
   std::vector<int> vertex_offsets; 
   std::vector<int> constraints_offsets; 
   Array<int> tdof_offsets;
   Array<int> constraints_starts;
   Array<int> globalvertices;
   Array<int> vertices;
   ParGridFunction * coords = nullptr;

protected:
   int npoints=0;
   int gnpoints=0;
   int nv, gnv;
   HypreParMatrix * K = nullptr;
   HypreParMatrix * Pi = nullptr;
   HypreParMatrix * Pb = nullptr;
   Vector *B = nullptr;
   Vector gapv;
   HypreParMatrix * M=nullptr;
   void SetupTribol();
   void SetupTribolDoublePass();
   std::set<int> mortar_attrs;
   // plane of top block
   std::set<int> nonmortar_attrs;
   bool doublepass = false;
   bool compute_dof_restrictions = false;

public:
   ParContactProblem(ParElasticityProblem * prob_, 
                     const std::set<int> & mortar_attrs_, const std::set<int> & nonmortar_attrs_,
                     ParGridFunction * coords_,
                      bool doublepass = false,
                      bool compute_dof_restrictions = false);

   ParElasticityProblem * GetElasticityProblem() {return prob;}
   MPI_Comm GetComm() {return comm;}
   int GetNumDofs() {return K->Height();}
   int GetGlobalNumDofs() {return K->GetGlobalNumRows();}
   int GetNumContraints() {return M->Height();}
   int GetGlobalNumConstraints() { return M->GetGlobalNumRows(); }
   
   std::vector<int> & GetDofOffets() { return dof_offsets; }
   std::vector<int> & GetVertexOffsets() { return vertex_offsets; }
   std::vector<int> & GetConstraintsOffsets() { return constraints_offsets; }
   Array<int> & GetConstraintsStarts() { return constraints_starts; }
   
   Vector & GetGapFunction() {return gapv;}

   HypreParMatrix * GetJacobian() {return M;}

   double E(const Vector & d);
   void DdE(const Vector &d, Vector &gradE);
   HypreParMatrix* DddE(const Vector &d);
   void g(const Vector &d, Vector &gd);
   HypreParMatrix* Ddg(const Vector &d);
   HypreParMatrix* lDddg(const Vector &d, const Vector &l);

   HypreParMatrix * GetRestrictionToInteriorDofs() {return Pi;}
   HypreParMatrix * GetRestrictionToContactDofs() {return Pb;}

   ~ParContactProblem()
   {
      delete B;
      delete K;
      delete M;
      delete Pi;
      delete Pb;
   }
};


class QPOptParContactProblem
{
private:
   ParContactProblem * problem = nullptr;
   int dimU, dimM, dimC;
   Vector ml;
   HypreParMatrix * NegId = nullptr;
   const Vector xref;
public:
   QPOptParContactProblem(ParContactProblem * problem_, const Vector & xref_);
   int GetDimU();
   int GetDimM();
   int GetDimC();
   Vector & Getml();
   MPI_Comm GetComm() {return problem->GetComm();}
   int * GetConstraintsStarts() {return problem->GetConstraintsStarts().GetData();}
   int GetGlobalNumConstraints() {return problem->GetGlobalNumConstraints();}

   ParElasticityProblem * GetElasticityProblem() {return problem->GetElasticityProblem();}

   HypreParMatrix * Duuf(const BlockVector &);
   HypreParMatrix * Dumf(const BlockVector &);
   HypreParMatrix * Dmuf(const BlockVector &);
   HypreParMatrix * Dmmf(const BlockVector &);
   HypreParMatrix * Duc(const BlockVector &);
   HypreParMatrix * Dmc(const BlockVector &);
   HypreParMatrix * lDuuc(const BlockVector &, const Vector &);

   HypreParMatrix * GetRestrictionToInteriorDofs() {return problem->GetRestrictionToInteriorDofs();}
   HypreParMatrix * GetRestrictionToContactDofs() {return problem->GetRestrictionToContactDofs();}

   void c(const BlockVector &, Vector &);
   double CalcObjective(const BlockVector &);
   void CalcObjectiveGrad(const BlockVector &, BlockVector &);
   ~QPOptParContactProblem();
};

// #endif
