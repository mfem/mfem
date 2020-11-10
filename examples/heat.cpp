//            MFEM Distance Function Solver - Parallel Version
//
// Compile with: make distance
//
// Sample runs:
//   Problem 0: point sources
//     mpirun -np 4 heat -m ../data/inline-quad.mesh -rs 2 -t 1.0
//
//   Problem 1: level sets
//      mpirun -np 4 heat -m ../data/inline-quad.mesh -rs 3 -o 2 -t 1.0 -p 1
//      mpirun -np 4 heat -m ../data/periodic-square.mesh -rs 5 -o 2 -t 1.0 -p 2
//      mpirun -np 4 heat -m ../data/periodic-cube.mesh -rs 3 -o 2 -t 1.0 -p 2
//
//
//    K. Crane et al:
//    Geodesics in Heat: A New Approach to Computing Distance Based on Heat Flow

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

class GradientCoefficient : public VectorCoefficient
{
private:
   const GridFunction &u;

public:
   GradientCoefficient(const GridFunction &u_gf, int dim)
      : VectorCoefficient(dim), u(u_gf) { }

   void Eval(Vector &V, ElementTransformation &T, const IntegrationPoint &ip)
   {
      T.SetIntPoint(&ip);

      u.GetGradient(T, V);
      const double norm = V.Norml2() + 1e-12;
      V /= -norm;
   }
};

double surface_level_set(const Vector &x)
{
   const double sine = 0.25 * std::sin(4 * M_PI * x(0));
   return (x(1) >= sine + 0.5) ? 0.0 : 1.0;
}

double Gyroid(const Vector & xx)
{
   const double period = 4.0 * M_PI;
   double x=xx[0]*period;
   double y=xx[1]*period;
   double z=0.0;
   if(xx.Size()==3)
   {
      z=xx[2]*period;
   }
   return std::sin(x)*std::cos(y) +
          std::sin(y)*std::cos(z) +
          std::sin(z)*std::cos(x);
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

class DistanceFunction
{
private:
   // Collection and space for the distance function.
   H1_FECollection fec;
   ParFiniteElementSpace pfes;
   ParGridFunction distance, source, diffused_source;

   // Diffusion coefficient.
   double t_param;
   // Length scale of the mesh.
   double dx;
   // List of true essential boundary dofs.
   Array<int> ess_tdof_list;

public:
   DistanceFunction(ParMesh &pmesh, int order, double diff_coeff)
      : fec(order, pmesh.Dimension()),
        pfes(&pmesh, &fec),
        distance(&pfes), source(&pfes), diffused_source(&pfes),
        t_param(diff_coeff)
   {
      // Compute average mesh size (assumes similar cells).
      double loc_area = 0.0;
      for (int i = 0; i < pmesh.GetNE(); i++)
      {
         loc_area += pmesh.GetElementVolume(i);
      }
      double glob_area;
      MPI_Allreduce(&loc_area, &glob_area, 1, MPI_DOUBLE,
                    MPI_SUM, pfes.GetComm());

      const int glob_zones = pmesh.GetGlobalNE();
      switch (pmesh.GetElementBaseGeometry(0))
      {
         case Geometry::SEGMENT:
            dx = glob_area / glob_zones; break;
         case Geometry::SQUARE:
            dx = sqrt(glob_area / glob_zones); break;
         case Geometry::TRIANGLE:
            dx = sqrt(2.0 * glob_area / glob_zones); break;
         case Geometry::CUBE:
            dx = pow(glob_area / glob_zones, 1.0/3.0); break;
         case Geometry::TETRAHEDRON:
            dx = pow(6.0 * glob_area / glob_zones, 1.0/3.0); break;
         default: MFEM_ABORT("Unknown zone type!");
      }
      dx /= order;


      // List of true essential boundary dofs.
      if (pmesh.bdr_attributes.Size())
      {
         Array<int> ess_bdr(pmesh.bdr_attributes.Max());
         ess_bdr = 1;
         pfes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }
   }

   ParGridFunction &ComputeDistance(Coefficient &level_set,
                                    int smooth_steps = 0, bool transform = true)
   {
      source.ProjectCoefficient(level_set);

      // Optional smoothing of the initial level set.
      if (smooth_steps > 0) { DiffuseField(source, smooth_steps); }

      // Transform so that the peak is at 0.
      // Assumes range [0, 1].
      if (transform)
      {
         for (int i = 0; i < source.Size(); i++)
         {
            const double x = source(i);
            source(i) = (x < 0.0 || x > 1.0) ? 0.0 : 4.0 * x * (1.0 - x);
         }
      }

      // Solver.
      CGSolver cg(MPI_COMM_WORLD);
      cg.SetRelTol(1e-12);
      cg.SetMaxIter(100);
      cg.SetPrintLevel(1);
      OperatorPtr A;
      Vector B, X;

      // Step 1 - diffuse.
      {
         // Set up RHS.
         ParLinearForm b1(&pfes);
         GridFunctionCoefficient src_coeff(&source);
         b1.AddDomainIntegrator(new DomainLFIntegrator(src_coeff));
         b1.Assemble();

         // Diffusion and mass terms in the LHS.
         ParBilinearForm a1(&pfes);
         a1.AddDomainIntegrator(new MassIntegrator);
         const double dt = t_param * dx * dx;
         ConstantCoefficient t_coeff(dt);
         a1.AddDomainIntegrator(new DiffusionIntegrator(t_coeff));
         a1.Assemble();

         // Solve with Dirichlet BC.
         ParGridFunction u_dirichlet(&pfes);
         u_dirichlet = 0.0;
         a1.FormLinearSystem(ess_tdof_list, u_dirichlet, b1, A, X, B);
         Solver *prec = new HypreBoomerAMG;
         cg.SetPreconditioner(*prec);
         cg.SetOperator(*A);
         cg.Mult(B, X);
         a1.RecoverFEMSolution(X, b1, u_dirichlet);
         delete prec;

         // Diffusion and mass terms in the LHS.
         ParBilinearForm a_n(&pfes);
         a_n.AddDomainIntegrator(new MassIntegrator);
         a_n.AddDomainIntegrator(new DiffusionIntegrator(t_coeff));
         a_n.Assemble();

         // Solve with Neumann BC.
         ParGridFunction u_neumann(&pfes);
         ess_tdof_list.DeleteAll();
         a_n.FormLinearSystem(ess_tdof_list, u_neumann, b1, A, X, B);
         Solver *prec2 = new HypreBoomerAMG;
         cg.SetPreconditioner(*prec2);
         cg.SetOperator(*A);
         cg.Mult(B, X);
         a_n.RecoverFEMSolution(X, b1, u_neumann);
         delete prec2;

         for (int i = 0; i < diffused_source.Size(); i++)
         {
            diffused_source(i) = 0.5 * (u_neumann(i) + u_dirichlet(i));
         }
      }

      // Step 2 - solve for the distance using the normalized gradient.
      {
         // RHS - normalized gradient.
         ParLinearForm b2(&pfes);
         GradientCoefficient grad_u(diffused_source,
                                    pfes.GetMesh()->Dimension());
         b2.AddDomainIntegrator(new DomainLFGradIntegrator(grad_u));
         b2.Assemble();

         // LHS - diffusion.
         ParBilinearForm a2(&pfes);
         a2.AddDomainIntegrator(new DiffusionIntegrator);
         a2.Assemble();

         // No BC.
         Array<int> no_ess_tdofs;

         a2.FormLinearSystem(no_ess_tdofs, distance, b2, A, X, B);

         Solver *prec2 = new HypreBoomerAMG;
         cg.SetPreconditioner(*prec2);
         cg.SetOperator(*A);
         cg.Mult(B, X);
         a2.RecoverFEMSolution(X, b2, distance);
         delete prec2;
      }

      // Rescale the distance to have minimum at zero.
      double d_min_loc = distance.Min();
      double d_min_glob;
      MPI_Allreduce(&d_min_loc, &d_min_glob, 1, MPI_DOUBLE,
                    MPI_MIN, pfes.GetComm());
      distance -= d_min_glob;

      return distance;
   }

   const ParGridFunction &GetLastSourceGF() const
   { return source; }
   const ParGridFunction &GetLastDiffusedSourceGF() const
   { return diffused_source; }
};

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
   int order = 2;
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
   args.AddOption(&t_param, "-t", "--t-param", "Diffusion time step");
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

   // MPI distribution.
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   Coefficient *ls_coeff;
   int smooth_steps; bool transform;
   if (problem == 0)
   {
      ls_coeff = new DeltaCoefficient(0.75, 0.625, 1.0);
      smooth_steps = 0;
      transform = false;
   }
   else if (problem == 1)
   {
      ls_coeff = new FunctionCoefficient(surface_level_set);
      smooth_steps = 5;
      transform = true;
   }
   else
   {
      ls_coeff = new FunctionCoefficient(Gyroid);
      smooth_steps = 0;
      transform = true;
   }
   DistanceFunction dist_func(pmesh, order, t_param);
   ParGridFunction &distance = dist_func.ComputeDistance(*ls_coeff,
                                                         smooth_steps, transform);
   const ParGridFunction &src = dist_func.GetLastSourceGF(),
                         &diff_src = dist_func.GetLastDiffusedSourceGF();
   delete ls_coeff;

   H1_FECollection fec(order, dim);
   ParFiniteElementSpace fespace_vec(&pmesh, &fec, dim);
   GradientCoefficient grad_u(dist_func.GetLastDiffusedSourceGF(), dim);
   ParGridFunction x(&fespace_vec);
   x.ProjectCoefficient(grad_u);

   // Send the solution by socket to a GLVis server.
   if (visualization)
   {
      int size = 500;
      char vishost[] = "localhost";
      int  visport   = 19916;

      socketstream sol_sock_w(vishost, visport);
      sol_sock_w << "parallel " << num_procs << " " << myid << "\n";
      sol_sock_w.precision(8);
      sol_sock_w << "solution\n" << pmesh << src;
      sol_sock_w << "window_geometry " << 0 << " " << 0 << " "
                                       << size << " " << size << "\n"
                 << "window_title '" << "u0" << "'\n" << flush;

      socketstream sol_sock_u(vishost, visport);
      sol_sock_u << "parallel " << num_procs << " " << myid << "\n";
      sol_sock_u.precision(8);
      sol_sock_u << "solution\n" << pmesh << diff_src;
      sol_sock_u << "window_geometry " << size << " " << 0 << " "
                                       << size << " " << size << "\n"
                 << "window_title '" << "u" << "'\n" << flush;

      socketstream sol_sock_x(vishost, visport);
      sol_sock_x << "parallel " << num_procs << " " << myid << "\n";
      sol_sock_x.precision(8);
      sol_sock_x << "solution\n" << pmesh << x;
      sol_sock_x << "window_geometry " << 2*size << " " << 0 << " "
                                       << size << " " << size << "\n"
                 << "window_title '" << "X" << "'\n"
                 << "keys evvRj*******A\n" << flush;

      socketstream sol_sock_d(vishost, visport);
      sol_sock_d << "parallel " << num_procs << " " << myid << "\n";
      sol_sock_d.precision(8);
      sol_sock_d << "solution\n" << pmesh << distance;
      sol_sock_d << "window_geometry " << size << " " << size << " "
                                       << size << " " << size << "\n"
                 << "window_title '" << "Distance" << "'\n"
                 << "keys rRjmm*****\n" << flush;
   }

   /*
   ParaViewDataCollection paraview_dc("Dist", &pmesh);
   paraview_dc.SetPrefixPath("ParaView");
   paraview_dc.SetLevelsOfDetail(order);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);
   paraview_dc.SetCycle(0);
   paraview_dc.SetTime(0.0);
   paraview_dc.RegisterField("w", &src);
   paraview_dc.RegisterField("u", &diff_src);
   paraview_dc.Save();
   */

   ConstantCoefficient zero(0.0);
   const double u0_norm = src.ComputeL2Error(zero),
                u_norm  = diff_src.ComputeL2Error(zero),
                d_norm  = distance.ComputeL2Error(zero);
   if (myid == 0)
   {
     std::cout <<  u0_norm << " "<< u_norm << " " << d_norm << std::endl;
   }

   MPI_Finalize();
   return 0;
}
