//                       MFEM Example 1 - Parallel Version
//
// Compile with: make ex1p
//
// Sample runs:  mpirun -np 4 ex1p -m ../data/square-disc.mesh
//               mpirun -np 4 ex1p -m ../data/star.mesh
//               mpirun -np 4 ex1p -m ../data/escher.mesh
//               mpirun -np 4 ex1p -m ../data/fichera.mesh
//               mpirun -np 4 ex1p -m ../data/square-disc-p2.vtk -o 2
//               mpirun -np 4 ex1p -m ../data/square-disc-p3.mesh -o 3
//               mpirun -np 4 ex1p -m ../data/square-disc-nurbs.mesh -o -1
//               mpirun -np 4 ex1p -m ../data/disc-nurbs.mesh -o -1
//               mpirun -np 4 ex1p -m ../data/pipe-nurbs.mesh -o -1
//               mpirun -np 4 ex1p -m ../data/ball-nurbs.mesh -o 2
//               mpirun -np 4 ex1p -m ../data/star-surf.mesh
//               mpirun -np 4 ex1p -m ../data/square-disc-surf.mesh
//               mpirun -np 4 ex1p -m ../data/inline-segment.mesh
//               mpirun -np 4 ex1p -m ../data/amr-quad.mesh
//               mpirun -np 4 ex1p -m ../data/amr-hex.mesh
//               mpirun -np 4 ex1p -m ../data/mobius-strip.mesh
//               mpirun -np 4 ex1p -m ../data/mobius-strip.mesh -o -1 -sc
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Laplace problem
//               -Delta u = 1 with homogeneous Dirichlet boundary conditions.
//               Specifically, we discretize using a FE space of the specified
//               order, or if order < 1 using an isoparametric/isogeometric
//               space (i.e. quadratic for quadratic curvilinear mesh, NURBS for
//               NURBS mesh, etc.)
//
//               The example highlights the use of mesh refinement, finite
//               element grid functions, as well as linear and bilinear forms
//               corresponding to the left-hand side and right-hand side of the
//               discrete linear system. We also cover the explicit elimination
//               of essential boundary conditions, static condensation, and the
//               optional connection to the GLVis tool for visualization.

#include "mfem.hpp"
#include "../common/pfem_extras.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

static double nl_exp_       = 2.5;
static double theta_        = 0.0;
static double chi_perp_     = 1.0;
static double chi_min_para_ = 100.0;
static double chi_max_para_ = 1000.0;

double uFunc(const Vector &x)
{
   return sin(M_PI * x[0]) * sin(M_PI * x[1]);
}

double QFunc(const Vector &x)
{
  double chi_ratio = (nl_exp_ > 0.0) ?
    pow(chi_min_para_ / chi_max_para_, 1.0 / nl_exp_) : 1.0;
  double u = uFunc(x);
  double T = chi_ratio + (1.0 - chi_ratio) * u;
  double cx = cos(M_PI * x[0]);
  double sx = sin(M_PI * x[0]);
  double cy = cos(M_PI * x[1]);
  double sy = sin(M_PI * x[1]);
  double ct = cos(theta_);
  double st = sin(theta_);
  double s2t = sin(2.0 * theta_);
  return M_PI * M_PI * (chi_perp_ * (u + cx * cy * s2t) +
			chi_max_para_ * (u - cx * cy * s2t) * pow(T, nl_exp_) +
			chi_max_para_ * nl_exp_ * (1.0 - chi_ratio) *
			(u * u - sx * sx * st * st - sy * sy * ct * ct -
			 u * cx * cy * s2t) * pow(T, nl_exp_ - 1.0) );
}

void unitVectorField(const Vector &, Vector &u)
{
  u.SetSize(2);
  u[0] = cos(theta_);
  u[1] = sin(theta_);
}

class ChiParaCoef : public MatrixCoefficient
{
private:
   MatrixCoefficient * bbT_;
   GridFunctionCoefficient * T_;
   double nl_exp_;
   double chi_min_;
   double chi_max_;
   double gamma_;

public:
  ChiParaCoef(MatrixCoefficient &bbT, GridFunctionCoefficient &T,
	      double nl_exp, double chi_min, double chi_max)
     : MatrixCoefficient(2), bbT_(&bbT), T_(&T), nl_exp_(nl_exp),
        chi_min_(chi_min), chi_max_(chi_max),
       gamma_(pow(chi_min/chi_max, nl_exp_))
   {}

   void SetTemp(GridFunction & T) { T_->SetGridFunction(&T); }

   void Eval(DenseMatrix &K, ElementTransformation &T,
             const IntegrationPoint &ip)
   {
      bbT_->Eval(K, T, ip);
      
      if ( nl_exp_ == 0.0)
      {
	 K *= chi_max_;
      }
      else
      {
 	 double Tval = T_->Eval(T, ip);
 	 K *= chi_max_ * pow(gamma_ + (1.0 - gamma_) * Tval, nl_exp_);
      }  
   }
};

class ChiCoef : public MatrixSumCoefficient
{
private:
   ChiParaCoef * chiParaCoef_;

public:
   ChiCoef(MatrixCoefficient & chiPerp, ChiParaCoef & chiPara)
      : MatrixSumCoefficient(chiPerp, chiPara), chiParaCoef_(&chiPara) {}

   void SetTemp(GridFunction & T) { chiParaCoef_->SetTemp(T); }
};

class dChiCoef : public MatrixCoefficient
{
private:
   MatrixCoefficient * bbT_;
   GridFunctionCoefficient * T_;
   double nl_exp_;
   double chi_min_;
   double chi_max_;
   double gamma_;

public:
   dChiCoef(MatrixCoefficient &bbT, GridFunctionCoefficient &T,
            double nl_exp, double chi_min, double chi_max)
     : MatrixCoefficient(2), bbT_(&bbT), T_(&T), nl_exp_(nl_exp),
        chi_min_(chi_min), chi_max_(chi_max),
        gamma_(pow(chi_max/chi_min, 0.4) - 1.0)
   {}

   void SetTemp(GridFunction & T) { T_->SetGridFunction(&T); }

   void Eval(DenseMatrix &K, ElementTransformation &T,
             const IntegrationPoint &ip)
   {
      bbT_->Eval(K, T, ip);
      double Tval = T_->Eval(T, ip);
      K *= nl_exp_ * chi_max_ * (1.0 - gamma_) *
	pow(gamma_ + (1.0 - gamma_) * Tval, nl_exp_ - 1.0);
   }
};

class ImplicitDiffOp : public Operator
{
public:
   ImplicitDiffOp(ParFiniteElementSpace & H1_FESpace,
                  Coefficient & TBdr,
                  Array<int> & bdr_attr,
                  Coefficient & heatCap,
                  ChiCoef & chi,
                  dChiCoef & dchi,
                  Coefficient & heatSource,
                  bool nonlinear = false);
   ~ImplicitDiffOp();

   void SetState(ParGridFunction & T);

   void Mult(const Vector &x, Vector &y) const;

   Operator & GetGradient(const Vector &x) const;

   Solver & GetGradientSolver() const;

   const Vector & GetRHS() const { return RHS_; }

private:

   bool first_;
   bool nonLinear_;

   Array<int> & ess_bdr_attr_;
   Array<int>   ess_bdr_tdofs_;

   Coefficient * bdrCoef_;
   Coefficient * cpCoef_;
   ChiCoef     * chiCoef_;
   dChiCoef    * dChiCoef_;
   Coefficient * QCoef_;
   ScalarMatrixProductCoefficient dtChiCoef_;

   mutable ParGridFunction T0_;
   mutable ParGridFunction T1_;
   mutable ParGridFunction dT_;

   mutable GradientGridFunctionCoefficient gradTCoef_;
   ScalarVectorProductCoefficient dtGradTCoef_;
   MatVecCoefficient dtdChiGradTCoef_;

   ParBilinearForm m0cp_;
   mutable ParBilinearForm s0chi_;
   mutable ParBilinearForm a0_;

   mutable HypreParMatrix A_;
   mutable ParGridFunction dTdt_;
   mutable ParLinearForm Q_;
   mutable ParLinearForm Qs_;
   mutable ParLinearForm rhs_;

   mutable  Vector SOL_;
   mutable  Vector RHS_;
   // Vector RHS0_; // Dummy RHS vector which hase length zero

   mutable Solver         * AInv_;
   mutable HypreBoomerAMG * APrecond_;
};

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   int n = 1;
   int el_type = Element::QUADRILATERAL;
   int order = 1;
   int ser_ref_levels = 0;
   int par_ref_levels = 0;
   bool static_cond = false;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&n, "-n", "--num-elems-1d",
                  "Number of elements in x and y directions.  "
                  "Total number of elements is n^2.");
   args.AddOption(&el_type, "-e", "--element-type",
                  "Element type: 2-Triangle, 3-Quadrilateral.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&theta_, "-t", "--theta",
                  "Angle of strong diffusion in degrees.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   theta_ *= M_PI / 180.0;
   
   // 3. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh = new Mesh(n, n, (Element::Type)el_type, 1);
   int dim = mesh->Dimension();

   // 4. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 10,000 elements.
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh->UniformRefinement();
   }

   // 6. Define a parallel finite element space on the parallel mesh. Here we
   //    use continuous Lagrange finite elements of the specified order. If
   //    order < 1, we instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim);
   }
   else if (pmesh->GetNodes())
   {
      fec = pmesh->GetNodes()->OwnFEC();
      if (myid == 0)
      {
         cout << "Using isoparametric FEs: " << fec->Name() << endl;
      }
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim);
   }
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
   HYPRE_Int size = fespace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   L2_FECollection L2FEC0(0, dim);
   ParFiniteElementSpace   L2FESpace0(pmesh, &L2FEC0);

   // 7. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs. In this example, the boundary conditions are defined
   //    by marking all the boundary attributes from the mesh as essential
   //    (Dirichlet) and converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
   if (pmesh->bdr_attributes.Size())
   {
      ess_bdr.SetSize(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 8. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system, which in this case is
   //    (1,phi_i) where phi_i are the basis functions in fespace.
   ConstantCoefficient zeroCoef(0.0);
   ConstantCoefficient oneCoef(1.0);
   FunctionCoefficient uCoef(uFunc);
   FunctionCoefficient QCoef(QFunc);
   ParLinearForm *Q = new ParLinearForm(fespace);
   Q->AddDomainIntegrator(new DomainLFIntegrator(QCoef));
   Q->Assemble();

   // 9. Define the solution vector x as a parallel finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   ParGridFunction u(fespace);
   ParGridFunction u_error(&L2FESpace0);
   ParGridFunction Q_gf(fespace);
   //u = 0.0;
   u.ProjectCoefficient(uCoef);
   Q_gf.ProjectCoefficient(QCoef);
   
   // 10. Set up the parallel bilinear form a(.,.) on the finite element space
   //     corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //     domain integrator.
   VectorFunctionCoefficient vCoef(2, unitVectorField);
   OuterProductCoefficient vvTCoef(vCoef, vCoef);
   IdentityMatrixCoefficient ICoef(2);
   GridFunctionCoefficient uGFCoef(&u);
   
   ChiParaCoef chiPara(vvTCoef, uGFCoef, nl_exp_, chi_min_para_, chi_max_para_);
   MatrixSumCoefficient chiPerp(ICoef, vvTCoef, chi_perp_, -chi_perp_);
   ChiCoef chiCoef(chiPerp, chiPara);   
   dChiCoef dchiCoef(vvTCoef, uGFCoef, nl_exp_, chi_min_para_, chi_max_para_);

   ImplicitDiffOp ido(*fespace, zeroCoef, ess_bdr, oneCoef,
		      chiCoef, dchiCoef, QCoef, chi_min_para_ != chi_max_para_);
   
   ParBilinearForm *a = new ParBilinearForm(fespace);
   a->AddDomainIntegrator(new DiffusionIntegrator(chiCoef));

   // 11. Assemble the parallel bilinear form and the corresponding linear
   //     system, applying any necessary transformations such as: parallel
   //     assembly, eliminating boundary conditions, applying conforming
   //     constraints for non-conforming AMR, static condensation, etc.
   if (static_cond) { a->EnableStaticCondensation(); }
   a->Assemble();

   HypreParMatrix A;
   Vector Q_dof, u_dof;
   a->FormLinearSystem(ess_tdof_list, u, *Q, A, u_dof, Q_dof);

   if (myid == 0)
   {
      cout << "Size of linear system: " << A.GetGlobalNumRows() << endl;
   }

   // 12. Define and apply a parallel PCG solver for AX=B with the BoomerAMG
   //     preconditioner from hypre.
   HypreSolver *amg = new HypreBoomerAMG(A);
   HyprePCG *pcg = new HyprePCG(A);
   pcg->SetTol(1e-12);
   pcg->SetMaxIter(200);
   pcg->SetPrintLevel(2);
   pcg->SetPreconditioner(*amg);
   pcg->Mult(Q_dof, u_dof);

   // 13. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   a->RecoverFEMSolution(u_dof, *Q, u);

   u.GridFunction::ComputeElementL2Errors(uCoef, u_error);

   double err = u.ComputeL2Error(uCoef);
      if (myid == 0)
   {
      cout << "L2 Error of Solution: " << err << endl;
   }
 
   // 14. Save the refined mesh and the solution in parallel. This output can
   //     be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
   {
      ostringstream mesh_name, sol_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      sol_name << "sol." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      u.Save(sol_ofs);
   }

   // 15. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      socketstream vis_T, vis_Q, vis_errT;
      char vishost[] = "localhost";
      int  visport   = 19916;

      // Make sure all ranks have sent their 'v' solution before initiating
      // another set of GLVis connections (one from each rank):
      MPI_Barrier(pmesh->GetComm());

      vis_T.precision(8);
      vis_Q.precision(8);
      vis_errT.precision(8);

      int Wx = 0, Wy = 0; // window position
      int Ww = 350, Wh = 350; // window size
      int offx = Ww+10;//, offy = Wh+45; // window offsets

      miniapps::VisualizeField(vis_Q, vishost, visport,
                               Q_gf, "Heat Soruce", Wx, Wy, Ww, Wh);

      Wx += offx;
      miniapps::VisualizeField(vis_T, vishost, visport,
                               u, "Temperature", Wx, Wy, Ww, Wh);

      Wx += offx;
      miniapps::VisualizeField(vis_errT, vishost, visport,
			       u_error, "Error in T", Wx, Wy, Ww, Wh);
   }

   // 16. Free the used memory.
   delete pcg;
   delete amg;
   delete a;
   delete Q;
   delete fespace;
   if (order > 0) { delete fec; }
   delete pmesh;

   MPI_Finalize();

   return 0;
}

ImplicitDiffOp::ImplicitDiffOp(ParFiniteElementSpace & H1_FESpace,
                               Coefficient & TBdr,
                               Array<int> & bdr_attr,
                               Coefficient & heatCap,
                               ChiCoef & chi,
                               dChiCoef & dchi,
                               Coefficient & heatSource,
                               bool nonlinear)
   : Operator(H1_FESpace.GetTrueVSize()),
     first_(true),
     nonLinear_(nonlinear),
     ess_bdr_attr_(bdr_attr),
     bdrCoef_(&TBdr),
     cpCoef_(&heatCap),
     chiCoef_(&chi),
     dChiCoef_(&dchi),
     QCoef_(&heatSource),
     dtChiCoef_(1.0, *chiCoef_),
     T0_(&H1_FESpace),
     T1_(&H1_FESpace),
     dT_(&H1_FESpace),
     gradTCoef_(&T0_),
     dtGradTCoef_(-1.0, gradTCoef_),
     dtdChiGradTCoef_(*dChiCoef_, dtGradTCoef_),
     m0cp_(&H1_FESpace),
     s0chi_(&H1_FESpace),
     a0_(&H1_FESpace),
     dTdt_(&H1_FESpace),
     Q_(&H1_FESpace),
     Qs_(&H1_FESpace),
     rhs_(&H1_FESpace),
     RHS_(H1_FESpace.GetTrueVSize()),
     // RHS0_(0),
     AInv_(NULL),
     APrecond_(NULL)
{
   H1_FESpace.GetEssentialTrueDofs(ess_bdr_attr_, ess_bdr_tdofs_);

   m0cp_.AddDomainIntegrator(new MassIntegrator(*cpCoef_));
   s0chi_.AddDomainIntegrator(new DiffusionIntegrator(*chiCoef_));

   a0_.AddDomainIntegrator(new MassIntegrator(*cpCoef_));
   a0_.AddDomainIntegrator(new DiffusionIntegrator(dtChiCoef_));
   if (nonLinear_)
   {
      a0_.AddDomainIntegrator(new MixedScalarWeakDivergenceIntegrator(
                                 dtdChiGradTCoef_));
   }

   Qs_.AddDomainIntegrator(new DomainLFIntegrator(*QCoef_));
   if (!tdQ_) { Qs_.Assemble(); }
}

ImplicitDiffOp::~ImplicitDiffOp()
{
   delete AInv_;
   delete APrecond_;
}

void ImplicitDiffOp::SetState(ParGridFunction & T)
{
   T0_ = T;

   newTime_ = fabs(t - t_) > 0.0;
   newTimeStep_= (fabs(1.0-dt/dt_)>1e-6);

   t_  = newTime_     ?  t :  t_;
   dt_ = newTimeStep_ ? dt : dt_;

   if (tdBdr_ && (newTime_ || newTimeStep_))
   {
      bdrCoef_->SetTime(t_ + dt_);
   }

   if (newTimeStep_ || first_)
   {
      dtChiCoef_.SetAConst(dt_);
      dtGradTCoef_.SetAConst(-dt_);
   }

   if ((tdCp_ && newTime_) || first_)
   {
      m0cp_.Update();
      m0cp_.Assemble();
      m0cp_.Finalize();
   }

   if (!tdChi_ && first_)
   {
      s0chi_.Assemble();
      s0chi_.Finalize();

      ofstream ofsS0("s0_const_initial.mat");
      s0chi_.SpMat().Print(ofsS0);

      a0_.Assemble();
      a0_.Finalize();
   }
   else if (tdChi_ && newTime_ && !nonLinear_)
   {
      chiCoef_->SetTemp(T0_);
      s0chi_.Update();
      s0chi_.Assemble(0);
      s0chi_.Finalize(0);

      ofstream ofsS0("s0_lin_initial.mat");
      s0chi_.SpMat().Print(ofsS0);

      a0_.Update();
      a0_.Assemble(0);
      a0_.Finalize(0);
   }

   if ((tdQ_ && newTime_) || first_)
   {
      cout << "Assembling Q" << endl;
      QCoef_->SetTime(t_ + dt_);
      Qs_.Assemble();
      Qs_.ParallelAssemble(RHS_);
      cout << "Norm of Q: " << Qs_.Norml2() << endl;
   }

   first_       = false;
}

void ImplicitDiffOp::Mult(const Vector &dT, Vector &Q) const
{
   dT_.Distribute(dT);

   add(T0_, dt_, dT_, T1_);

   if (tdChi_ && nonLinear_)
   {
      chiCoef_->SetTemp(T1_);
      s0chi_.Update();
      s0chi_.Assemble(0);
      s0chi_.Finalize(0);
   }
   else
   {
      cout << "Well this is a surprise..." << endl;
   }
   m0cp_.Mult(dT_, Q_);
   s0chi_.AddMult(T1_, Q_);

   Q_.ParallelAssemble(Q);
   Q.SetSubVector(ess_bdr_tdofs_, 0.0);
}

Operator & ImplicitDiffOp::GetGradient(const Vector &dT) const
{
   if (tdChi_)
   {
      if (!nonLinear_)
      {
         chiCoef_->SetTemp(T0_);
      }
      else
      {
         dT_.Distribute(dT);
         add(T0_, dt_, dT_, T1_);

         chiCoef_->SetTemp(T1_);
         dChiCoef_->SetTemp(T1_);
         gradTCoef_.SetGridFunction(&T1_);
      }
      s0chi_.Update();
      s0chi_.Assemble(0);
      s0chi_.Finalize(0);

      a0_.Update();
      a0_.Assemble(0);
      a0_.Finalize(0);
   }

   if (!nonLinear_)
   {
      s0chi_.Mult(T0_, rhs_);

      rhs_ -= Qs_;
      rhs_ *= -1.0;
   }
   else
   {
      rhs_ = Qs_;
   }

   dTdt_.ProjectBdrCoefficient(*bdrCoef_, ess_bdr_attr_);

   a0_.FormLinearSystem(ess_bdr_tdofs_, dTdt_, rhs_, A_, SOL_, RHS_);

   return A_;
}

Solver & ImplicitDiffOp::GetGradientSolver() const
{
   if (!nonLinear_)
   {
      Operator & A_op = this->GetGradient(T0_); // T0_ will be ignored
      HypreParMatrix & A_hyp = dynamic_cast<HypreParMatrix &>(A_op);

      if (tdChi_)
      {
         delete AInv_;     AInv_     = NULL;
         delete APrecond_; APrecond_ = NULL;
      }

      if ( AInv_ == NULL )
      {
         // A_hyp.Print("A.mat");

         HyprePCG * AInv_pcg = NULL;

         cout << "Building PCG" << endl;
         AInv_pcg = new HyprePCG(A_hyp);
         AInv_pcg->SetTol(1e-10);
         AInv_pcg->SetMaxIter(200);
         AInv_pcg->SetPrintLevel(0);
         if ( APrecond_ == NULL )
         {
            cout << "Building AMG" << endl;
            APrecond_ = new HypreBoomerAMG(A_hyp);
            APrecond_->SetPrintLevel(0);
            AInv_pcg->SetPreconditioner(*APrecond_);
         }
         AInv_ = AInv_pcg;
      }
   }
   else
   {
      if (AInv_ == NULL)
      {
         /*
              HypreSmoother *J_hypreSmoother = new HypreSmoother;
         J_hypreSmoother->SetType(HypreSmoother::l1Jacobi);
         J_hypreSmoother->SetPositiveDiagonal(true);
         JPrecond_ = J_hypreSmoother;

              GMRESSolver * AInv_gmres = NULL;

              cout << "Building GMRES" << endl;
              AInv_gmres = new GMRESSolver(T0_.ParFESpace()->GetComm());
              AInv_gmres->SetRelTol(1e-12);
              AInv_gmres->SetAbsTol(0.0);
              AInv_gmres->SetMaxIter(20000);
              AInv_gmres->SetPrintLevel(2);
         AInv_gmres->SetPreconditioner(*JPrecond_);
         AInv_ = AInv_gmres;
         */
         HypreGMRES * AInv_gmres = NULL;

         cout << "Building HypreGMRES" << endl;
         AInv_gmres = new HypreGMRES(T0_.ParFESpace()->GetComm());
         AInv_gmres->SetTol(1e-12);
         AInv_gmres->SetMaxIter(200);
         AInv_gmres->SetPrintLevel(2);
         if ( APrecond_ == NULL )
         {
            cout << "Building AMG" << endl;
            APrecond_ = new HypreBoomerAMG();
            APrecond_->SetPrintLevel(0);
            AInv_gmres->SetPreconditioner(*APrecond_);
         }
         AInv_ = AInv_gmres;
      }
   }

   return *AInv_;
}
