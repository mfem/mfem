//            MFEM Distance Function Solver - Parallel Version
//
// Compile with: make distance
//
// Sample runs:
//   Problem 0: exact boundary alignment:
//     mpirun -np 4 distance -m ../data/inline-segment.mesh -rs 3 -t 0.5
//     mpirun -np 4 distance -m ../data/inline-quad.mesh -rs 1 -t 0.1
//     mpirun -np 4 distance -m ./cir.msh -t 0.01
//     mpirun -np 4 distance -m ../data/star.mesh
//
//    Problem 1: level set enclosing a volume:
//     mpirun -np 4 distance -m ../data/inline-quad.mesh -p 1 -rs 3
//
//

// Sample runs:  mpirun -np 4 ex1p -m ../data/square-disc.mesh
//               mpirun -np 4 ex1p -m ../data/star.mesh
//               mpirun -np 4 ex1p -m ../data/star-mixed.mesh
//               mpirun -np 4 ex1p -m ../data/escher.mesh
//               mpirun -np 4 ex1p -m ../data/fichera.mesh
//               mpirun -np 4 ex1p -m ../data/fichera-mixed.mesh
//               mpirun -np 4 ex1p -m ../data/toroid-wedge.mesh
//               mpirun -np 4 ex1p -m ../data/periodic-annulus-sector.msh
//               mpirun -np 4 ex1p -m ../data/periodic-torus-sector.msh
//               mpirun -np 4 ex1p -m ../data/square-disc-p2.vtk -o 2
//               mpirun -np 4 ex1p -m ../data/square-disc-p3.mesh -o 3
//               mpirun -np 4 ex1p -m ../data/square-disc-nurbs.mesh -o -1
//               mpirun -np 4 ex1p -m ../data/star-mixed-p2.mesh -o 2
//               mpirun -np 4 ex1p -m ../data/disc-nurbs.mesh -o -1
//               mpirun -np 4 ex1p -m ../data/pipe-nurbs.mesh -o -1
//               mpirun -np 4 ex1p -m ../data/ball-nurbs.mesh -o 2
//               mpirun -np 4 ex1p -m ../data/fichera-mixed-p2.mesh -o 2
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

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

/** A time-dependent operator for the right-hand side of the ODE. The DG weak
 form of du/dt = v.grad(u) is M du/dt = K u + b, where M and K are the mass
 and advection matrices, and b describes the flow on the boundary. This can
 be written as a general ODE, du/dt = M^{-1} (K u + b), and this class is
 used to evaluate the right-hand side. */
class FE_Evolution : public TimeDependentOperator
{
private:
    SparseMatrix &M, &K;
    const Vector &b;
    DSmoother M_prec;
    CGSolver M_solver;

    mutable Vector z;

public:
    FE_Evolution(SparseMatrix &_M, SparseMatrix &_K, const Vector &_b)
       : TimeDependentOperator(_M.Size()), M(_M), K(_K), b(_b), z(_M.Size())
    {
        M_solver.SetPreconditioner(M_prec);
        M_solver.SetOperator(M);

        M_solver.iterative_mode = false;
        M_solver.SetRelTol(1e-9);
        M_solver.SetAbsTol(0.0);
        M_solver.SetMaxIter(100);
        M_solver.SetPrintLevel(0);
    }

    virtual void Mult(const Vector &x, Vector &y) const
    {
        // y = M^{-1} (K x + b)
        K.Mult(x, z);
        z += b;
        M_solver.Mult(z, y);
    }
};

class ExactSegmentDistCoeff : public Coefficient
{
   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      Vector x(3);
      T.Transform(ip, x);
      return min(x(0), 1.0 - x(0));
   }
};

class ExactQuadDistCoeff : public Coefficient
{
   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      Vector x(3);
      T.Transform(ip, x);
      return min( min(x(0), 1.0 - x(0)), min(x(1), 1.0 - x(1)) );
   }
};

class ExactCircleDistCoeff : public Coefficient
{
   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      Vector x(3);
      T.Transform(ip, x);
      const double rad = sqrt( x(0) * x(0) + x(1) * x(1) );
      return 1.0 - rad;
   }
};

class GradientCoefficient : public VectorCoefficient
{
private:
   GridFunction &u;
public:
   GradientCoefficient(GridFunction &u_, int dim)
      : VectorCoefficient(dim), u(u_) { }
   void Eval(Vector &V, ElementTransformation &T, const IntegrationPoint &ip)
   {
      T.SetIntPoint(&ip);
      u.GetGradient(T, V);
   }
};

// Coefficient for redistancing: S(w) * grad_w / |grad_w|.
class RedistVelocityCoefficient : public VectorCoefficient
{
private:
    GridFunction &w;
    double max_velocity, dx;

public:
    RedistVelocityCoefficient(GridFunction &w_, int dim, double delta_x)
       : VectorCoefficient(dim), w(w_), max_velocity(0.0), dx(delta_x) { }

    void Eval(Vector &V, ElementTransformation &T, const IntegrationPoint &ip)
    {
        T.SetIntPoint(&ip);

        Vector grad_w;
        w.GetGradient(T, grad_w);

        const double gnorm = grad_w.Norml2();
        if (gnorm > 0.0) { grad_w /= gnorm; }

        const double val = w.GetValue(T.ElementNo, ip);
        grad_w *= val / sqrt(val * val + gnorm * gnorm * dx * dx);

        if (gnorm > max_velocity) { max_velocity = grad_w.Norml2(); }

        V = grad_w;
    }

    double GetMagVelocity() const { return max_velocity; }
};

class BoundaryCoefficient : public Coefficient
{
private:
   GridFunction &w;

public:
    BoundaryCoefficient(GridFunction &w_): w(w_){ }
    double Eval(ElementTransformation &T, const IntegrationPoint &ip)
    {
       // The incoming boundary should just decrease, nothing is coming in.
       return 0.0;
       //return u.GetValue(T.ElementNo, ip);
    }
};

// Computes S(w) at a point.
class SCoefficient : public Coefficient
{
private:
   GridFunction &w;
   double dx;

public:
   SCoefficient(GridFunction &w_, double delta_x) : w(w_), dx(delta_x) { }

   double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      T.SetIntPoint(&ip);
      Vector grad_w;
      w.GetGradient(T, grad_w);

      const double gnorm = grad_w.Norml2();
      const double val = w.GetValue(T.ElementNo, ip);

      return val / sqrt(val * val + gnorm * gnorm * dx * dx);
   }
};

double surface_level_set(const Vector &x)
{
   const double sine = 0.25 * std::sin(4 * M_PI * x(0));
   return (x(1) >= sine + 0.5) ? 1.0 : -1.0;
}

void redistance_level_set(ParGridFunction &w, int num_steps, double dx)
{
   const int order = w.ParFESpace()->GetFE(0)->GetOrder();
   const int dim = w.ParFESpace()->GetMesh()->Dimension();
   const int myid = w.ParFESpace()->GetMyRank();
   const bool skip_zeros = false;
   const double dt = 0.001;
   ParMesh *pmesh = w.ParFESpace()->GetParMesh();

   DG_FECollection L2fec(order, dim);
   ParFiniteElementSpace l2_fes(pmesh, &L2fec);
   ParFiniteElementSpace l2_fes_vec(pmesh, &L2fec, dim);
   ParGridFunction grad_w_l2(&l2_fes_vec);

   ParGridFunction w_dg(&l2_fes);
   w_dg.ProjectGridFunction(w);

   ParLinearForm b(&l2_fes);

   BoundaryCoefficient bc(w_dg);
   GradientCoefficient grad_w_coef(w_dg, dim);
   RedistVelocityCoefficient vel_l2_rd_coef(w_dg, dim, dx);

   // Redistancing mass.
   ParBilinearForm m(&l2_fes);
   m.AddDomainIntegrator(new MassIntegrator);
   m.Assemble();
   m.Finalize();

   // Redistancing advection.
   ParBilinearForm k(&l2_fes);
   k.AddDomainIntegrator(new ConvectionIntegrator(vel_l2_rd_coef, -1.0));
   k.AddInteriorFaceIntegrator(
      new TransposeIntegrator(new DGTraceIntegrator(vel_l2_rd_coef, 1.0, -0.5)));
   k.AddBdrFaceIntegrator(
      new TransposeIntegrator(new DGTraceIntegrator(vel_l2_rd_coef, 1.0, -0.5)));
   k.Assemble(skip_zeros);
   k.Finalize(skip_zeros);

   // Redistancing RHS.
   SCoefficient rhs_l2_coef(w_dg, dx);
   b.AddDomainIntegrator(new DomainLFIntegrator(rhs_l2_coef));
   b.AddBdrFaceIntegrator(
      new BoundaryFlowIntegrator(bc, vel_l2_rd_coef, -1.0, -0.5));
   b.Assemble();

   FE_Evolution redist(m.SpMat(), k.SpMat(), b);

   RK2Solver ode_solver(1.0);
   ode_solver.Init(redist);

   double norm_old = 0.0;
   for (int r = 0; r < num_steps; r++)
   {
      grad_w_l2.ProjectCoefficient(grad_w_coef);
      const double norm = grad_w_l2.Normlinf();

      if (myid == 0) { std::cout << "Step " << r << ": "<< norm << std::endl; }

      if ((norm < 1.1 && norm > .9) || fabs(norm_old - norm) < 1e-5)
      {
         cout << "Redistancing converged in " << r << " steps." << endl;
         break;
      }

      k.Update();
      k.Assemble(skip_zeros);
      k.Finalize(skip_zeros);
      b.Update();
      b.Assemble();

      double t_red = dt * num_steps; // Shouldn't matter.
      double dt_rd = 5.0 * dt;
      ode_solver.Step(w_dg, t_red, dt_rd);
      norm_old = norm;
   }
   w.ProjectGridFunction(w_dg);
}

void DiffuseField(ParGridFunction &field, int smooth_steps)
{
   // Setup the Laplacian operator.
   ParBilinearForm *Lap = new ParBilinearForm(field.ParFESpace());
   Lap->AddDomainIntegrator(new DiffusionIntegrator());
   Lap->Assemble();
   Lap->Finalize();
   HypreParMatrix *A = Lap->ParallelAssemble();

   HypreSmoother *S = new HypreSmoother(*A,0,smooth_steps);
   S->iterative_mode = true;

   Vector tmp(A->Width());
   field.SetTrueVector();
   Vector fieldtrue = field.GetTrueVector();
   tmp = 0.0;
   S->Mult(tmp, fieldtrue);

   field.SetFromTrueDofs(fieldtrue);

   delete S;
   delete Lap;
}

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int problem = 0;
   int rs_levels = 0;
   int order = 1;
   double t_param = 1.0;
   const char *device_config = "cpu";
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem type:\n\t"
                  "0: exact alignment with the mesh boundary\n\t"
                  "1: zero level set enclosing a volume");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&t_param, "-t", "--t-param", "Varadhan's t constant");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
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
   if (myid == 0) { args.PrintOptions(cout); }

   // 3. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (myid == 0) { device.Print(); }

   // Refine the mesh.
   Mesh mesh(mesh_file, 1, 1);
   const int dim = mesh.Dimension();
   for (int lev = 0; lev < rs_levels; lev++) { mesh.UniformRefinement(); }

   // Compute average mesh size (assumes similar cells).
   double area = 0.0, dx;
   const int zones_cnt = mesh.GetNE();
   for (int i = 0; i < zones_cnt; i++) { area += mesh.GetElementVolume(i); }
   switch (mesh.GetElementBaseGeometry(0))
   {
      case Geometry::SQUARE:
         dx = sqrt(area / zones_cnt); break;
      case Geometry::TRIANGLE:
         dx = sqrt(2.0 * area / zones_cnt); break;
      case Geometry::CUBE:
         dx = pow(area / zones_cnt, 1.0/3.0); break;
      case Geometry::TETRAHEDRON:
         dx = pow(6.0 * area / zones_cnt, 1.0/3.0); break;
      default: MFEM_ABORT("Unknown zone type!");
   }
   dx /= order;

   // MPI distribution.
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   // 7. Define a parallel finite element space on the parallel mesh. Here we
   //    use continuous Lagrange finite elements of the specified order. If
   //    order < 1, we instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   bool delete_fec;
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim);
      delete_fec = true;
   }
   else if (pmesh.GetNodes())
   {
      fec = pmesh.GetNodes()->OwnFEC();
      delete_fec = false;
      if (myid == 0)
      {
         cout << "Using isoparametric FEs: " << fec->Name() << endl;
      }
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim);
      delete_fec = true;
   }
   ParFiniteElementSpace fespace(&pmesh, fec);
   HYPRE_Int size = fespace.GlobalTrueVSize();
   if (myid == 0) { cout << "Number of FE unknowns: " << size << endl; }

   // 8. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs. In this example, the boundary conditions are defined
   //    by marking all the boundary attributes from the mesh as essential
   //    (Dirichlet) and converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   if (pmesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh.bdr_attributes.Max());
      ess_bdr = 1;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // Solution x with correct Dirichlet BC.
   ParGridFunction w(&fespace);
   w = 0.0;
   Array<int> bdr(pmesh.bdr_attributes.Max()); bdr = 1;
   ConstantCoefficient one(1.0);
   w.ProjectBdrCoefficient(one, bdr);

   // Set up RHS.
   ParLinearForm b(&fespace);
   b = 0.0;

   // Diffusion and mass terms in the LHS.
   ParBilinearForm a(&fespace);
   a.AddDomainIntegrator(new MassIntegrator(one));
   ConstantCoefficient t_coeff(t_param);
   a.AddDomainIntegrator(new DiffusionIntegrator(t_coeff));
   a.Assemble();

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, w, b, A, X, B);

   // Solve the linear system A X = B; CG + BoomerAMG.
   Solver *prec = new HypreBoomerAMG;

   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(5000);
   cg.SetPrintLevel(1);
   if (prec) { cg.SetPreconditioner(*prec); }
   cg.SetOperator(*A);
   cg.Mult(B, X);
   delete prec;

   // 14. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   a.RecoverFEMSolution(X, b, w);

   ParGridFunction u(&fespace);
   if (problem == 1)
   {
      FunctionCoefficient c(surface_level_set);
      w.ProjectCoefficient(c);
      DiffuseField(w, 1);

      u = w;
      redistance_level_set(u, 500, dx);

      /*
      // Transform so that the peak is at 0.
      for (int i = 0; i < w.Size(); i++)
      {
         double x = w(i);
         x += 1.0;
         x /= 2.0;
         w(i) = 4.0 * x * (1.0 - x);
      } */
   }

   // Varadhan transformation.
   if (problem == 0)
   {
      for (int i = 0; i < u.Size(); i++)
      {
         if (problem == 0) { u(i) = - sqrt(t_param) * log(w(i)); }
      }
   }

   Coefficient *exact_dist = NULL;
   ParGridFunction u_error(&fespace);
   if (problem == 0 && strcmp(mesh_file, "../data/inline-segment.mesh") == 0)
   {
      exact_dist = new ExactSegmentDistCoeff;
   }
   if (problem == 0 && strcmp(mesh_file, "../data/inline-quad.mesh") == 0)
   {
      exact_dist = new ExactQuadDistCoeff;
   }
   if (problem == 0 && strcmp(mesh_file, "./cir.msh") == 0)
   {
      exact_dist = new ExactCircleDistCoeff;
   }
   if (exact_dist)
   {
      const double l1 = u.ComputeL1Error(*exact_dist),
                   linf = u.ComputeMaxError(*exact_dist);
      if (myid == 0)
      {
         std::cout << "L1   error: " << l1 << endl
                   << "Linf error: " << linf << endl;
      }
      // Visualize the error.
      u_error.ProjectCoefficient(*exact_dist);
      for (int i = 0; i < u.Size(); i++)
      {
         u_error(i) = fabs(u_error(i) - u(i));
      }
   }

   // 15. Save the refined mesh and the solution in parallel. This output can
   //     be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
   {
      ostringstream mesh_name, sol_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      sol_name << "sol." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh.Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      w.Save(sol_ofs);
   }

   // 16. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;

      socketstream sol_sock_w(vishost, visport);
      sol_sock_w << "parallel " << num_procs << " " << myid << "\n";
      sol_sock_w.precision(8);
      sol_sock_w << "solution\n" << pmesh << w;
      sol_sock_w << "window_geometry " << 0 << " " << 0 << " "
                                       << 600 << " " << 600 << "\n"
                 << "window_title '" << "w" << "'\n" << flush;

      socketstream sol_sock_u(vishost, visport);
      sol_sock_u << "parallel " << num_procs << " " << myid << "\n";
      sol_sock_u.precision(8);
      sol_sock_u << "solution\n" << pmesh << u;
      sol_sock_u << "window_geometry " << 600 << " " << 0 << " "
                                       << 600 << " " << 600 << "\n"
                 << "window_title '" << "u" << "'\n" << flush;

      if (exact_dist)
      {
         socketstream sol_sock_e(vishost, visport);
         sol_sock_e << "parallel " << num_procs << " " << myid << "\n";
         sol_sock_e.precision(8);
         sol_sock_e << "solution\n" << pmesh << u_error;
         sol_sock_e << "window_geometry " << 1200 << " " << 0 << " "
                                          << 600 << " " << 600 << "\n"
                    << "window_title '" << "|u - d|" << "'\n" << flush;
      }
   }

   ParaViewDataCollection paraview_dc("Dist", &pmesh);
   paraview_dc.SetPrefixPath("ParaView");
   paraview_dc.SetLevelsOfDetail(order);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);
   paraview_dc.SetCycle(0);
   paraview_dc.SetTime(0.0);
   paraview_dc.RegisterField("w",&w);
   paraview_dc.RegisterField("u",&u);
   paraview_dc.Save();

   if (delete_fec) { delete fec; }

   MPI_Finalize();
   return 0;
}
