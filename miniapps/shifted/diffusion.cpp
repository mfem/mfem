//            MFEM Shifted boundary method solver - Parallel Version
//
// Compile with: make diffusion
//
// Sample runs:
//   Problem 1: Circular hole of radius 0.2 at the center of the domain.
//              -nabla^u = 1 with homogeneous boundary conditions
//   mpirun -np 4 diffusion -m ../../data/inline-quad.mesh -rs 3 -o 1 -vis -lst 1
//   mpirun -np 4 diffusion -m ../../data/inline-hex.mesh -rs 2 -o 2 -vis -lst 1 -ho 1 -alpha 10
//
//   Problem 2: Circular hole of radius 0.2 at the center of the domain.
//              -nabla^u = f with inhomogeneous boundary conditions, f is setup
//              such that u = x^p + y^p, where p = 2 by default.
//   mpirun -np 4 diffusion -m ../../data/inline-quad.mesh -rs 2 -o 2 -vis -lst 2
//
//   Problem 3: Domain is y = [0, 1] but mesh is shifted to [-1.e-4, 1].
//              -nabla^u = f with inhomogeneous boundary conditions, f is setup
//              such that u = sin(pi*x*y)
//   mpirun -np 4 diffusion -m ../../data/inline-quad.mesh -rs 2 -o 1 -vis -lst 3
//
//   Problem 4: Same as Problem 3 but now we also treat a domain boundary as
//   shifted boundary
#include "../../mfem.hpp"
#include <fstream>
#include <iostream>
#include "sbm-aux.hpp"
#include "../common/mfem-common.hpp"
#include "sbm_solver.hpp"
#include "marking.hpp"

using namespace mfem;
using namespace std;

int main(int argc, char *argv[])
{
   // Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // Parse command-line options.
   const char *mesh_file = "../../data/inline-quad.mesh";
   int order = 2;
   bool visualization = true;
   int ser_ref_levels = 0;
   bool exact = true;
   int level_set_type = 1;
   int ho_terms = 0;
   double alpha = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   //   args.AddOption(&exact, "-ex", "--exact", "-no-ex",
   //                  "--no-exact",
   //                  "Use exact representaion of distance vector function.");
   args.AddOption(&level_set_type, "-lst", "--level-set-type",
                  "level-set-type:");
   args.AddOption(&ho_terms, "-ho", "--high-order",
                  "Additional high-order terms to include");
   args.AddOption(&alpha, "-alpha", "--alpha",
                  "Nitsche penalty parameter (~1 for 2D, ~10 for 3D).");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // Enable hardware devices such as GPUs, and programming models such as
   // CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device("cpu");
   device.Print();

   // Refine the mesh.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();
   for (int lev = 0; lev < ser_ref_levels; lev++) { mesh.UniformRefinement(); }

   // MPI distribution.
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   // Define a finite element space on the mesh. Here we use continuous
   // Lagrange finite elements of the specified order. If order < 1, we
   // instead use an isoparametric/isogeometric space.
   if (order <= 0) { order = 1; }
   H1_FECollection fec(order, dim);
   ParFiniteElementSpace pfespace(&pmesh, &fec);

   Vector vxyz;

   ParFiniteElementSpace pfespace_mesh(&pmesh, &fec, dim);
   pmesh.SetNodalFESpace(&pfespace_mesh);
   ParGridFunction x_mesh(&pfespace_mesh);
   pmesh.SetNodalGridFunction(&x_mesh);
   vxyz = *pmesh.GetNodes();
   int nodes_cnt = vxyz.Size()/dim;
   if (level_set_type == 3)   //stretch quadmesh from [0, 1] to [-1.e-4, 1]
   {
      for (int i = 0; i < nodes_cnt; i++)
      {
         vxyz(i+nodes_cnt) = (1.+1.e-4)*vxyz(i+nodes_cnt)-1.e-4;
      }
   }
   else if (level_set_type ==
            4)   //stretch quadmesh from [0, 1] to [1.e-4, 1+2e-4]
   {
      for (int i = 0; i < nodes_cnt; i++)
      {
         vxyz(i+nodes_cnt) = (1.+1.e-4)*vxyz(i+nodes_cnt)+1.e-4/std::pow(2.,
                                                                         ser_ref_levels);
      }
   }
   pmesh.SetNodes(vxyz);
   pfespace.ExchangeFaceNbrData();
   cout << "Number of finite element unknowns: "
        << pfespace.GetTrueVSize() << endl;

   // Define the solution vector x as a finite element grid function
   // corresponding to fespace. Initialize x with initial guess of zero,
   // which satisfies the boundary conditions.
   ParGridFunction x(&pfespace);
   // ParGridFunction for level_set_value.
   ParGridFunction level_set_val(&pfespace);

   // Determine if each element in the ParMesh is inside the actual domain,
   // partially cut by the actual domain boundary, or completely outside
   // the domain.
   Dist_Level_Set_Coefficient dist_fun_level_coef(level_set_type);
   level_set_val.ProjectCoefficient(dist_fun_level_coef);
   level_set_val.ExchangeFaceNbrData();

   ShiftedFaceMarker marker(pmesh, level_set_val, pfespace);
   Array<int> elem_marker;
   marker.MarkElements(elem_marker);

   // Visualize the element markers.
   if (visualization)
   {
      L2_FECollection fecl2 = L2_FECollection(0, dim);
      ParFiniteElementSpace pfesl2(&pmesh, &fecl2);
      ParGridFunction elem_marker_gf(&pfesl2);
      for (int i = 0; i < elem_marker_gf.Size(); i++)
      {
         elem_marker_gf(i) = (double)elem_marker[i];
      }
      char vishost[] = "localhost";
      int  visport   = 19916, s = 350;
      socketstream sol_sock;
      common::VisualizeField(sol_sock, vishost, visport, elem_marker_gf,
                             "Element Flags", 0, 0, s, s, "Rjmpc");
   }

   // Get a list of dofs associated with shifted boundary faces.
   Array<int> sbm_dofs; // Array of dofs on SBM faces
   marker.ListShiftedFaceDofs(elem_marker, sbm_dofs);

   if (visualization)
   {
      ParGridFunction face_dofs(&pfespace);
      face_dofs = 0.0;
      for (int i = 0; i < sbm_dofs.Size(); i++)
      {
         face_dofs(sbm_dofs[i]) = 1.0;
      }
      char vishost[] = "localhost";
      int  visport   = 19916, s = 350;
      socketstream sol_sock;
      common::VisualizeField(sol_sock, vishost, visport, face_dofs,
                             "Shifted Face Dofs", 0, s, s, s, "Rjmpc");
   }

   // Make a list of inactive tdofs that will be eleminated from the system.
   Array<int> ess_tdof_list;
   marker.ListEssentialTDofs(elem_marker, sbm_dofs, ess_tdof_list);

   Array<int> ess_shift_bdr(pmesh.bdr_attributes.Max());
   if (pmesh.bdr_attributes.Size())
   {
      ess_shift_bdr = 0;
      if (level_set_type == 4)
      {
         ess_shift_bdr[0] = 1;
      }
   }

   // Compute Distance Vector - Use analytic distance vectors for now.
   auto distance_vec_space = new ParFiniteElementSpace(pfespace.GetParMesh(),
                                                       pfespace.FEColl(), dim);
   ParGridFunction distance(distance_vec_space);

   // Get the Distance from the level set either using a numerical approach
   // or project an exact analytic function.
   //   HeatDistanceSolver dist_func(1.0);
   //   dist_func.smooth_steps = 1;
   //   dist_func.ComputeVectorDistance(dist_fun_level_coef, distance);

   VectorCoefficient *dist_vec = NULL;
   if (true)
   {
      Dist_Vector_Coefficient *dist_vec_fcoeff =
         new Dist_Vector_Coefficient(dim, level_set_type);
      dist_vec = dist_vec_fcoeff;
      distance.ProjectDiscCoefficient(*dist_vec);
   }

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916, s = 350;
      socketstream sol_sock;
      common::VisualizeField(sol_sock, vishost, visport, distance,
                             "Distance Vector", s, s, s, s, "Rjmpcvv", 1);
   }

   // Set up a list to indicate element attributes to be included in assembly,
   // so that inactive elements are excluded.
   const int max_elem_attr = pmesh.attributes.Max();
   Array<int> ess_elem(max_elem_attr);
   ess_elem = 1;
   ess_elem.Append(0);
   for (int i = 0; i < pmesh.GetNE(); i++)
   {
      if (elem_marker[i] >= 1) { pmesh.SetAttribute(i, max_elem_attr+1); }
   }
   pmesh.SetAttributes();

   // Set up the linear form b(.) which corresponds to the right-hand side of
   // the FEM linear system.
   ParLinearForm b(&pfespace);
   FunctionCoefficient *rhs_f = NULL;
   if (level_set_type == 1)
   {
      rhs_f = new FunctionCoefficient(rhs_fun_circle);
   }
   else if (level_set_type == 2)
   {
      rhs_f = new FunctionCoefficient(rhs_fun_xy_exponent);
   }
   else if (level_set_type == 3 || level_set_type == 4)
   {
      rhs_f = new FunctionCoefficient(rhs_fun_xy_sinusoidal);
   }
   else
   {
      MFEM_ABORT("Dirichlet velocity function not set for level set type.\n");
   }
   b.AddDomainIntegrator(new DomainLFIntegrator(*rhs_f), ess_elem);

   // Dirichlet BC that must be imposed on the true boundary.
   ShiftedFunctionCoefficient *dbcCoef = NULL;
   if (level_set_type == 1)
   {
      dbcCoef = new ShiftedFunctionCoefficient(dirichlet_velocity_circle);
   }
   else if (level_set_type == 2)
   {
      dbcCoef = new ShiftedFunctionCoefficient(dirichlet_velocity_xy_exponent);
   }
   else if (level_set_type == 3 || level_set_type == 4)
   {
      dbcCoef = new ShiftedFunctionCoefficient(dirichlet_velocity_xy_sinusoidal);
   }
   else
   {
      MFEM_ABORT("Dirichlet velocity function not set for level set type.\n");
   }
   b.AddInteriorFaceIntegrator(new SBM2DirichletLFIntegrator(&pmesh, *dbcCoef,
                                                              alpha, *dist_vec,
                                                              elem_marker,
                                                              ho_terms));
   b.AddBdrFaceIntegrator(new SBM2DirichletLFIntegrator(&pmesh, *dbcCoef,
                                                         alpha, *dist_vec,
                                                         elem_marker,
                                                         ho_terms), ess_shift_bdr);
   b.Assemble();

   // Set up the bilinear form a(.,.) on the finite element space
   // corresponding to the Laplacian operator -Delta, by adding the Diffusion
   // domain integrator and SBM integrator.
   ParBilinearForm a(&pfespace);
   ConstantCoefficient one(1.);

   a.AddDomainIntegrator(new DiffusionIntegrator(one), ess_elem);
   a.AddInteriorFaceIntegrator(new SBM2DirichletIntegrator(&pmesh, alpha,
                                                           *dist_vec, elem_marker, ho_terms));
   a.AddBdrFaceIntegrator(new SBM2DirichletIntegrator(&pmesh, alpha, *dist_vec,
                                                      elem_marker, ho_terms), ess_shift_bdr);

   // Assemble the bilinear form and the corresponding linear system,
   // applying any necessary transformations.
   a.KeepNbrBlock();
   a.Assemble();

   // Project the exact solution as an initial condition for dirichlet boundaries.
   x.ProjectCoefficient(*dbcCoef);

   // Form the linear system and solve it.
   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   cout << "Size of linear system: " << A->Height() << endl;

   Solver *S = NULL;
   Solver *prec = NULL;
   prec = new HypreBoomerAMG;
   BiCGSTABSolver *bicg = new BiCGSTABSolver(MPI_COMM_WORLD);
   bicg->SetRelTol(1e-12);
   bicg->SetMaxIter(2000);
   bicg->SetPrintLevel(1);
   bicg->SetPreconditioner(*prec);
   bicg->SetOperator(*A);
   S = bicg;
   S->Mult(B, X);

   // Recover the solution as a finite element grid function.
   a.RecoverFEMSolution(X, b, x);

   // Save the mesh and the solution.
   ofstream mesh_ofs("diffusion.mesh");
   mesh_ofs.precision(8);
   pmesh.PrintAsOne(mesh_ofs);
   ofstream sol_ofs("diffusion.gf");
   sol_ofs.precision(8);
   x.SaveAsOne(sol_ofs);

   // Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916, s = 350;
      socketstream sol_sock;
      common::VisualizeField(sol_sock, vishost, visport, x,
                             "Solution", s, 0, s, s, "Rj");
   }

   // Construct an error gridfunction if the exact solution is known.
   ParGridFunction err(x);
   Vector pxyz(dim);
   pxyz(0) = 0.;
   for (int i = 0; i < nodes_cnt; i++)
   {
      pxyz(0) = vxyz(i);;
      pxyz(1) = vxyz(i+nodes_cnt);
      double exact_val = 0.;
      if (level_set_type == 1)
      {
         exact_val = dirichlet_velocity_circle(pxyz);
      }
      else if (level_set_type == 2)
      {
         exact_val = dirichlet_velocity_xy_exponent(pxyz);
      }
      else if (level_set_type == 3 || level_set_type == 4)
      {
         exact_val = dirichlet_velocity_xy_sinusoidal(pxyz);
      }
      err(i) = std::fabs(x(i) - exact_val);
   }

   double global_error = err.Norml2();
   if (myid == 0 && level_set_type != 1)
   {
      std::cout << global_error << " Global error - L2 norm.\n";
   }

   if (visualization && level_set_type >= 3 && level_set_type <= 4)
   {
      char vishost[] = "localhost";
      int  visport   = 19916, s = 350;
      socketstream sol_sock;
      common::VisualizeField(sol_sock, vishost, visport, err,
                             "Error", 2*s, 0, s, s, "Rj");
   }

   int NEglob = pmesh.GetGlobalNE();
   double errnorm = x.ComputeL2Error(*dbcCoef);
   if (level_set_type >= 3 && level_set_type <= 5 && myid == 0)
   {
      ofstream myfile;
      myfile.open ("error.txt", ios::app);
      double h_factor = pow(1./2, ser_ref_levels*1.);
      cout << order << " " <<
           ho_terms << " " <<
           h_factor << " " <<
           errnorm << " " <<
           NEglob << " " <<
           "k10-analytic-L2Error\n";
      myfile.close();
   }

   // Free the used memory.
   delete prec;
   delete S;
   delete dbcCoef;
   delete rhs_f;
   delete dist_vec;
   delete distance_vec_space;

   MPI_Finalize();

   return 0;
}
