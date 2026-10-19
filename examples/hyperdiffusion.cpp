/**
 * Sample run: ./hyperdiffusion
 * 
 * 
 * Error scaling tests:
 *   mpirun -np 4 ./hyperdiffusion -dt 1e-4 -nt 1000 -no-vis
 *   mpirun -np 4 ./hyperdiffusion -dt 1e-4 -nt 1000 -no-vis -ne 15
 *   mpirun -np 4 ./hyperdiffusion -dt 1e-4 -nt 1000 -no-vis -r 1
 *   mpirun -np 4 ./hyperdiffusion -dt 1e-4 -nt 1000 -no-vis -ne 30
 *   mpirun -np 4 ./hyperdiffusion -dt 1e-4 -nt 1000 -no-vis -r 2
 * 
 *   mpirun -np 4 ./hyperdiffusion -dt 1e-4 -nt 1000 -no-vis -o 3 -eta 10
 *   mpirun -np 4 ./hyperdiffusion -dt 1e-4 -nt 1000 -no-vis -o 3 -eta 10 -r 1
 *   mpirun -np 8 ./hyperdiffusion -dt 1e-4 -nt 1000 -no-vis -o 3 -eta 10 -r 2
 */
#include <mfem.hpp>

using namespace mfem;

/**
 * @brief Sample exact solution devised by AI + verified by human.
 * 
 * @details Solution is for a periodic domain [0,2π) x [0,2π) solving
 * with κ=1 (ie: du/dt + ∇⁴u = 0):
 * 
 *    u(x,y,t)=e⁻⁴ᵗcos(x)cos(y).
 * 
 */
real_t ExactSolution(const Vector &x, real_t t);

/**
 * @brief Gradient of exact solution above, for computing H1-error.
 */
void GradExactSolution(const Vector &x, real_t t, Vector &grad_u);

/**
 * @brief Hyper-diffusion operator.
 * 
 * @details The semi-discrete form can be written as:
 *
 *                       M (du/dt) + Bu = 0,
 *
 * where M is a mass matrix, B is the biharmonic/C0-IP matrix, and u is the 
 * solution vector.
 *
 * For explicit time integration in TimeDependentOperator::Mult, the
 * equation to solve for (du/dt) for time tₙ is simply
 *
 *                (du/dt) = M⁻¹(-Buₙ).
 *
 * For implicit time integration in TimeDependentOperator::ImplicitSolve, the
 * equation to solve for k=du/dt can be written as
 *
 *                (M + Δt B)(du/dt) = -Buₙ.
 *
 * Writing A = M + Δt K, the equation is then
 *
 *                (du/dt) = A⁻¹(-Buₙ).
 * 
 * @warning Periodicity is assumed -- no essential BCs applied.
 */
class HyperDiffusionOperator : public TimeDependentOperator
{
protected:
   CGSolver M_solver;
   CGSolver A_solver;
   HypreBoomerAMG M_prec;
   HypreBoomerAMG A_prec;

   ParBilinearForm M;
   std::unique_ptr<HypreParMatrix> M_mat;

   ParBilinearForm B;
   std::unique_ptr<HypreParMatrix> B_mat;

   std::unique_ptr<HypreParMatrix> A_mat;

   mutable Vector rhs;

   void EvalRHS(const Vector &u, Vector &out) const;

public:
  HyperDiffusionOperator(ParFiniteElementSpace &f, real_t eta);

  void Mult(const Vector &u, Vector &du_dt) const override;

  void ImplicitSolve(const real_t dt, const Vector &u, Vector &k) override;
};

/**
 * @brief Compute the physical Hessian for affine elements.
 * 
 * @details This is exactly the `FiniteElement::CalcPhysHessian` implementation
 * copied and pasted, but assuming that mesh element Hessian = 0. **This is 
 * true for straight-edged simplices and parallelograms/parallelopipeds.**
 */
void CalcPhysHessianAffine(const FiniteElement &el,
                           ElementTransformation &Trans,
                           DenseMatrix &Hessian);

/**
 * @brief Single integrator for biharmonic w/ C0-IP.
 * 
 * @details Joined from ex42 PR. Removed coefficient **and affine elements are 
 * assumed!**
 */
class C0IPBiharmonicIntegrator : public BilinearFormIntegrator
{
private:
   // Biharmonic element matrix:
   inline static const Vector factors_2D{1.0, 2.0, 1.0};
   mutable DenseMatrix hessian;
   mutable Vector factors;

   // C0-IP face matrix:
   const double eta;
   mutable Vector normal[2], dnshape[2], nv[2], nd2nshape[2];
   mutable DenseMatrix dshape[2], hshape[2], blockJ[2][2], blockC[2][2], 
                       elmatJ_p, elmatC_p;
public:
   C0IPBiharmonicIntegrator(real_t eta_) : eta(eta_) {}

   void AssembleElementMatrix(const FiniteElement &el,
                              ElementTransformation &Trans, 
                              DenseMatrix &elmat) override;

   void AssembleFaceMatrix(const FiniteElement &el1, 
                           const FiniteElement &el2,
                           FaceElementTransformations &Trans, 
                           DenseMatrix &elmat) override;
};

/**
 * @brief Copied directly from miniapps/common/pfem-extras but adapted to
 * include commands.
 */
void VisualizeField(socketstream &sock, const char *vishost, int visport,
                    const ParGridFunction &gf, const char *title,
                    int x = 0, int y = 0, int w = 400, int h = 400,
                    const char *keys = NULL, bool vec = false,
                    const char *commands = NULL);

int main(int argc, char *argv[])
{
   // Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int size = Mpi::WorldSize();
   int rank = Mpi::WorldRank();
   Hypre::Init();

   // Parse command line args
   int order = 2;
   int num_elems = 10;
   int ref_levels = 0;
   bool quad = true;
   int ode_solver_type = 23; // SDIRK33Solver
   real_t dt = 0.01;
   int nt = 100;
   real_t eta = 1;
   int pvdc_freq = 0;
   bool visualization = true;
   int visport = 19916;
   char vishost[] = "localhost";

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&num_elems, "-ne", "--num-elems",
                  "Number of elements in x- and y-directions.");
   args.AddOption(&ref_levels, "-r", "--ref-levels",
                  "Number of refinements of mesh to complete.");
   args.AddOption(&quad, "-q", "--quads", "-t", "--tris", 
                  "Use quads or tris.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                 ODESolver::Types.c_str());
   args.AddOption(&dt, "-dt", "--time-step",
                 "Time step.");
   args.AddOption(&nt, "-nt", "--num-timesteps", 
                  "Number of time steps.");
   args.AddOption(&eta, "-eta", "--penalty-coeff",
                  "Penalty coefficient.");
   args.AddOption(&pvdc_freq, "-pv", "--paraview-freq",
                 "ParaView data collection write frequency. 0 to disable.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visport, "-p", "--send-port", "Socket for GLVis.");
   args.Parse();
   if (!args.Good())
   {
      if (rank == 0)
      {
         args.PrintUsage(mfem::out);
      }
      return 1;
   }
   if (rank == 0)
   {
      args.PrintOptions(mfem::out);
   }
   MFEM_VERIFY(order >= 2, "Order must be >= 2.");

   // Create the mesh
   Mesh mesh;
   {
      Element::Type el_type = quad ? Element::QUADRILATERAL : Element::TRIANGLE;
      Mesh tmp = Mesh::MakeCartesian2D(num_elems, num_elems, el_type,
                                     true, 2*M_PI, 2*M_PI);
      for (int i = 0; i < ref_levels; i++)
      {
         tmp.UniformRefinement();
      }
      mesh = Mesh::MakePeriodic(tmp, 
               tmp.CreatePeriodicVertexMapping({Vector({2*M_PI, 0.0}),
                                               Vector({0.0, 2*M_PI})}));
   }

   // Partition the mesh
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   // Initialize the FE collection and FiniteElementSpace
   H1_FECollection fe_coll(order, 2);
   ParFiniteElementSpace fespace(&pmesh, &fe_coll, 1);

   // Initialize the solution
   ParGridFunction u_gf(&fespace);
   FunctionCoefficient u_exact(ExactSolution);
   u_exact.SetTime(0.0);
   u_gf.ProjectCoefficient(u_exact);

   // Initialize time-dependent operator
   HyperDiffusionOperator oper(fespace, eta);

   // Initialize ODE Solver
   std::unique_ptr<ODESolver> ode_solver = ODESolver::Select(ode_solver_type);
   ode_solver->Init(oper);

   // Initialize ParaViewDataCollection for output.
   std::unique_ptr<ParaViewDataCollection> pvdc;
   if (pvdc_freq > 0)
   {
      pvdc = std::make_unique<ParaViewDataCollection>("ParaView", &pmesh);
      pvdc->SetHighOrderOutput(true);
      pvdc->SetLevelsOfDetail(order);
      pvdc->SetDataFormat(VTKFormat::BINARY);
      pvdc->RegisterField("Numerical", &u_gf);
      pvdc->RegisterCoeffField("Exact", &u_exact);
      pvdc->SetCycle(0);
      pvdc->SetTime(0.0);
      pvdc->Save();
   }

   // GLVis parameters.
   socketstream sout;
   const auto WriteGLVis = 
   [&]()
   {
      VisualizeField(sout, vishost, visport, u_gf,
                     "Hyperdiffusion 2D Test", 10, 10, 600, 300,
                     "]", false,
                     "autoscale off\n");
   };

   if (visualization)
   {
      WriteGLVis();
   }

   real_t t = 0.0;
   Array<double> it_times(nt);
   StopWatch tic_toc;
   for (int ti = 0; ti < nt; ti++)
   {
      tic_toc.Start();
      ode_solver->Step(u_gf.GetTrueVector(), t, dt);
      tic_toc.Stop();
      u_gf.SetFromTrueVector();
      
      it_times[ti] = tic_toc.RealTime();
      tic_toc.Clear();

      if (rank == 0)
      {
         mfem::out << "step " << ti << " | time " << t << " | iter time " 
         << it_times[ti] << std::endl;
      }

      if (visualization)
      {
         WriteGLVis();
      }

      if (pvdc && ti % pvdc_freq == 0)
      {
         pvdc->SetTime(t);
         pvdc->SetCycle(ti);
         pvdc->Save();
      }
   }

   // Compute and write errors.
   const real_t h = (2*M_PI/num_elems)/std::pow(2, ref_levels);
   u_exact.SetTime(t);
   const real_t l2_error = u_gf.ComputeL2Error(u_exact);

   VectorFunctionCoefficient grad_u_exact(2, GradExactSolution);
   grad_u_exact.SetTime(t);
   const real_t h1_error = u_gf.ComputeH1Error(&u_exact, &grad_u_exact);
   if (rank == 0)
   {
      mfem::out << "h = " << h << std::endl;
      mfem::out << "L2-error = " << l2_error << std::endl;
      mfem::out << "H1-error = " << h1_error << std::endl;
   }

   // Write average iteration time
   double ave_iter = it_times.Sum()/nt;
   if (rank == 0)
   {
      mfem::out << "Average iteration time: " << ave_iter << std::endl;
   }

   return 0;
}

real_t ExactSolution(const Vector &x, real_t t)
{
   return std::exp(-4*t)*std::cos(x[0])*std::cos(x[1]);
}

void GradExactSolution(const Vector &x, real_t t, Vector &grad_u)
{
   grad_u[0] = -std::exp(-4*t)*std::sin(x[0])*std::cos(x[1]);
   grad_u[1] = -std::exp(-4*t)*std::cos(x[0])*std::sin(x[1]);
}

HyperDiffusionOperator::HyperDiffusionOperator(ParFiniteElementSpace &f,
                                               real_t eta)
:  TimeDependentOperator(f.GetTrueVSize()),
   M_solver(f.GetComm()),
   A_solver(f.GetComm()),
   M(&f),
   B(&f), 
   rhs(f.GetTrueVSize())
{
   // Double-precision relative tolerance:
   const real_t rel_tol = 1e-12;

   const int max_iter = 1000;

   M.AddDomainIntegrator(new MassIntegrator());
   M.Assemble(0);
   M.Finalize(0);
   M_mat = std::unique_ptr<HypreParMatrix>(M.ParallelAssemble());

   B.AddDomainIntegrator(new C0IPBiharmonicIntegrator(eta));
   B.AddInteriorFaceIntegrator(new C0IPBiharmonicIntegrator(eta));
   B.Assemble(0);
   B.Finalize(0);
   B_mat = std::unique_ptr<HypreParMatrix>(B.ParallelAssemble());

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(rel_tol);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(max_iter);
   M_solver.SetPrintLevel(IterativeSolver::PrintLevel().Warnings().Errors());
   M_prec.SetOperator(*M_mat);
   M_prec.SetPrintLevel(0);
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(*M_mat);

   A_solver.iterative_mode = false;
   A_solver.SetRelTol(rel_tol);
   A_solver.SetAbsTol(0.0);
   A_solver.SetMaxIter(max_iter);
   A_solver.SetPrintLevel(IterativeSolver::PrintLevel().Warnings().Errors());
   A_prec.SetPrintLevel(0);
}

void HyperDiffusionOperator::EvalRHS(const Vector &u, Vector &out) const
{
   out = 0.0;
   B_mat->Mult(u, out);
   out.Neg();
}

void HyperDiffusionOperator::Mult(const Vector &u, Vector &du_dt) const
{
   EvalRHS(u, rhs);

   M_solver.Mult(rhs, du_dt);
}

void HyperDiffusionOperator::ImplicitSolve(const real_t dt, const Vector &u,
                                           Vector &k)
{
   A_mat.reset();
   A_mat = std::unique_ptr<HypreParMatrix>(Add(1.0, *M_mat, dt, *B_mat));

   EvalRHS(u, rhs);

   A_prec.SetOperator(*A_mat);
   A_solver.SetPreconditioner(A_prec);
   A_solver.SetOperator(*A_mat);
   A_solver.Mult(rhs, k);
}

void CalcPhysHessianAffine(const FiniteElement &el,
                           ElementTransformation &Trans,
                           DenseMatrix &Hessian)
{
   int dim = el.GetDim();
   int dof = el.GetDof();

   // Roll 2-Tensors in vectors and 4-Tensor in Matrix, exploiting symmetry
   Array<int> map(dim*dim);
   if (dim == 3)
   {
      map[0] = 0;
      map[1] = 1;
      map[2] = 2;

      map[3] = 1;
      map[4] = 3;
      map[5] = 4;

      map[6] = 2;
      map[7] = 4;
      map[8] = 5;
   }
   else if (dim == 2)
   {
      map[0] = 0;
      map[1] = 1;

      map[2] = 1;
      map[3] = 2;
   }
   else
   {
      map[0] = 0;
   }

   // Hessian in ref coords
   int size = (dim*(dim+1))/2;
   DenseMatrix hess(dof, size);
   el.CalcHessian(Trans.GetIntPoint(), hess);

   // Gradient in physical coords
   // if (Trans.Hessian().FNorm2() > 1e-10)
   // {
   //    DenseMatrix grad(dof, dim);
   //    CalcPhysDShape(Trans, grad);
   //    DenseMatrix gmap(dof, size);
   //    Mult(grad,Trans.Hessian(),gmap);
   //    hess -= gmap;
   // }

   // LHM
   DenseMatrix lhm(size,size);
   DenseMatrix invJ = Trans.Jacobian();
   lhm = 0.0;
   for (int i = 0; i < dim; i++)
   {
      for (int j = 0; j < dim; j++)
      {
         for (int k = 0; k < dim; k++)
         {
            for (int l = 0; l < dim; l++)
            {
               lhm(map[i*dim+j],map[k*dim+l]) += invJ(i,k)*invJ(j,l);
            }
         }
      }
   }
   // Correct multiplicity
   Vector mult(size);
   mult = 0.0;
   for (int i = 0; i < dim*dim; i++) { mult[map[i]]++; }
   lhm.InvRightScaling(mult);

   // Hessian in physical coords
   lhm.Invert();
   Mult(hess, lhm, Hessian);
}

void C0IPBiharmonicIntegrator::AssembleElementMatrix(const FiniteElement &el,
                                                     ElementTransformation &Trans, 
                                                     DenseMatrix &elmat)
{
   int ndof = el.GetDof();
   int dim = el.GetDim();

   MFEM_ASSERT(dim == 2, "Dimension must be 2.");

   hessian.SetSize(ndof, dim * (dim + 1) / 2);
   elmat.SetSize(ndof);
   factors.SetSize(dim * (dim + 1) / 2);

   elmat = 0.0;

   const IntegrationRule *ir = GetIntegrationRule(el, Trans);
   if (ir == NULL)
   {
      int order = 2*el.GetOrder();
      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const mfem::IntegrationPoint &ip = ir->IntPoint(i);
      Trans.SetIntPoint(&ip);

      // el.CalcPhysHessian(Trans, hessian);
      CalcPhysHessianAffine(el, Trans, hessian);
      factors = factors_2D;
      factors *= ip.weight * Trans.Weight();

      AddMultADAt(hessian, factors, elmat);
   }
}

void C0IPBiharmonicIntegrator::AssembleFaceMatrix(
   const FiniteElement &el1,
   const FiniteElement &el2, 
   FaceElementTransformations &Trans, 
   DenseMatrix &elmat)
{
   int dim = el1.GetDim();
   MFEM_ASSERT(dim == 2, "Dimension must be 2.");

   int ndof[2] = {el1.GetDof(), 0};
   int num_elems = 1;
   if (Trans.Elem2No >= 0)
   {
      ndof[1] = el2.GetDof();
      num_elems++;
   }

   for (int i = 0; i < num_elems; i++)
   {
      normal[i].SetSize(dim);
      dshape[i].SetSize(ndof[i], dim);
      hshape[i].SetSize(ndof[i], dim * (dim + 1) / 2);
      nv[i].SetSize(dim * (dim + 1) / 2);
      dnshape[i].SetSize(ndof[i]);
      nd2nshape[i].SetSize(ndof[i]);
   }

   for (int i = 0; i < num_elems; i++)
   {
      for (int j = 0; j < num_elems; j++)
      {
         blockJ[i][j].SetSize(ndof[i], ndof[j]);
         blockC[i][j].SetSize(ndof[i], ndof[j]);
      }
   }

   elmatJ_p.SetSize(ndof[0] + ndof[1]);
   elmatC_p.SetSize(ndof[0] + ndof[1]);
   elmat.SetSize(ndof[0] + ndof[1]);
   elmat = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order = 2 * std::max(el1.GetOrder(), ndof[1] ? el2.GetOrder() : 0);
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
   }

   // Compute edge length
   double h_e  = 0.0;
   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      Trans.SetAllIntPoints(&ip);
      h_e += ip.weight * Trans.Weight();
   }

   const FiniteElement *els[2] = {&el1, &el2};
   ElementTransformation *el_trans[2] = {Trans.Elem1, Trans.Elem2};

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      elmatJ_p = 0.0;
      elmatC_p = 0.0;

      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);

      // Compute normal gradients + Hessians
      for (int i = 0; i < num_elems; i++)
      {
         if (i == 0)
         {
            CalcOrtho(Trans.Jacobian(), normal[i]);
            normal[i] /= normal[i].Norml2();
         }
         else
         {
            normal[i] = normal[0];
            normal[i] *= -1;
         }
         els[i]->CalcPhysDShape(*el_trans[i], dshape[i]);
         //els[i]->CalcPhysHessian(*el_trans[i], hshape[i]);
         CalcPhysHessianAffine(*els[i], *el_trans[i], hshape[i]);
         dshape[i].Mult(normal[i], dnshape[i]);
         nv[i][0] = normal[i][0]*normal[i][0];
         nv[i][1] = 2*normal[i][0]*normal[i][1];
         nv[i][2] = normal[i][1]*normal[i][1];
         hshape[i].Mult(nv[i], nd2nshape[i]);
      }

      // Compute blocks
      for (int i = 0; i < num_elems; i++)
      {
         for (int j = 0; j < num_elems; j++)
         {
            blockJ[i][j] = 0.0;
            blockC[i][j] = 0.0;
            AddMult_a_VWt(-1.0, dnshape[i], nd2nshape[j], blockJ[i][j]);
            elmatJ_p.SetSubMatrix(i*ndof[0], j*ndof[0], blockJ[i][j]);

            AddMult_a_VWt(eta/h_e, dnshape[i], dnshape[j], blockC[i][j]);
            elmatC_p.SetSubMatrix(i*ndof[0], j*ndof[0], blockC[i][j]);
         }
      }

      // Symmetrize the jump term
      elmatJ_p.Symmetrize();
      if (!ndof[1])
      {
         elmatJ_p *= 2;
      }

      // Add penalty term
      elmatJ_p += elmatC_p;
      elmatJ_p *= ip.weight * Trans.Weight();
      elmat += elmatJ_p;
   }
}

void VisualizeField(socketstream &sock, const char *vishost, int visport,
                    const ParGridFunction &gf, const char *title,
                    int x, int y, int w, int h, const char *keys, bool vec,
                    const char *commands)
{
  ParMesh &pmesh = *gf.ParFESpace()->GetParMesh();
  MPI_Comm comm  = pmesh.GetComm();

  int num_procs, myid;
  MPI_Comm_size(comm, &num_procs);
  MPI_Comm_rank(comm, &myid);

  bool newly_opened = false;
  int  connection_failed;

  do {
    if (myid == 0) {
      if (!sock.is_open() || !sock) {
        sock.open(vishost, visport);
        sock.precision(8);
        newly_opened = true;
      }
      sock << "solution\n";
    }

    pmesh.PrintAsOne(sock);
    gf.SaveAsOne(sock);

    if (myid == 0 && newly_opened) {
      if (commands) {
        sock << commands << "\n";
      }
      sock << "window_title '" << title << "'\n"
           << "window_geometry "
           << x << " " << y << " " << w << " " << h << "\n";
      if (keys) {
        sock << "keys " << keys << "\n";
      } else {
        sock << "keys maaAc";
      }
      if (vec) {
        sock << "vvv";
      }
      sock << std::endl;
    }

    if (myid == 0) {
      connection_failed = !sock && !newly_opened;
    }
    MPI_Bcast(&connection_failed, 1, MPI_INT, 0, comm);
  } while (connection_failed);
}
