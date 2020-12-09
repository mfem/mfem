//                                MFEM Example 1
//
// Compile with: make ex1p
//
// Sample runs:  mpirun -np 4 ex1p -nm 3 -np1 2 -np2 1 -np3 1
//               mpirun -np 4 ex1p -nm 2 -np1 2 -np2 2

// Description:  Overlapping grids with MFEM:
//               This example code demonstrates use of MFEM to solve the
//               Poisson problem:
//                              -nabla^2 u = 1 \in [0, 1]^2, u_b = 0 \in \dO
//               with homogeneous boundary conditions on the domain boundary
//               modeled using an arbitrary number of overlapping grids.
//               Using simultaneous Schwarz iterations, the Poisson equation is
//               solved iteratively, with boundary data interpolated between
//               the overlapping grids at each iteration. The overlapping
//               Schwarz method was introduced by H. A. Schwarz in 1870, and a
//               concise description of the simultaneous Schwarz iterations for
//               this problem is given in Section 2.2 of [1]:
//
//           [1] Mittal, K., Dutta, S., & Fischer, P. (2020). Stability analysis
//               of a singlerate and multirate predictor-corrector scheme for
//               overlapping grids. arXiv preprint arXiv:2010.00118.

#include "../../mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Method to use FindPointsGSLIB to determine the boundary points of a mesh
// that are interior to another mesh.
void GetInterdomainBoundaryPoints(OversetFindPointsGSLIB &finder,
                                  Vector &vxyz, int color,
                                  Array<int> ess_tdof_list,
                                  Array<int> &ess_tdof_list_int, int dim)
{
   int nb1 = ess_tdof_list.Size(),
       nt1 = vxyz.Size()/dim;

   Vector bnd1(nb1*dim);
   Array<unsigned int> colorv(nb1);
   for (int i = 0; i < nb1; i++)
   {
      int idx = ess_tdof_list[i];
      for (int d = 0; d < dim; d++) { bnd1(i+d*nb1) = vxyz(idx + d*nt1); }
      colorv[i] = (unsigned int)color;
   }

   finder.FindPoints(bnd1, colorv );

   const Array<unsigned int> &code_out = finder.GetCode();

   //Setup ess_tdof_list_int
   for (int i = 0; i < nb1; i++)
   {
      int idx = ess_tdof_list[i];
      if (code_out[i] != 2) { ess_tdof_list_int.Append(idx); }
   }
}

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   int lim_meshes = 3; //should be greater than nmeshes
   Array <const char *> mesh_file_list(lim_meshes);
   Array <int> np_list(lim_meshes), rs_levels(lim_meshes),
         rp_levels(lim_meshes);
   mesh_file_list[0]         = "../../data/square-disc.mesh";
   mesh_file_list[1]         = "../../data/inline-quad.mesh";
   mesh_file_list[2]         = "../../data/inline-quad.mesh";
   int order                 = 2;
   const char *device_config = "cpu";
   bool visualization        = true;
   rs_levels                 = 0;
   rp_levels                 = 0;
   np_list                   = 0;
   double rel_tol            = 1.e-8;
   int nmeshes               = 2;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file_list[0], "-m1", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&mesh_file_list[1], "-m2", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&mesh_file_list[2], "-m3", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&rs_levels[0], "-r1", "--refine-serial",
                  "Number of times to refine the mesh 1 uniformly in serial.");
   args.AddOption(&rs_levels[1], "-r2", "--refine-serial",
                  "Number of times to refine the mesh 2 uniformly in serial.");
   args.AddOption(&rs_levels[2], "-r3", "--refine-serial",
                  "Number of times to refine the mesh 3 uniformly in serial.");
   args.AddOption(&rp_levels[0], "-rp1", "--refine-parallel",
                  "Number of times to refine the mesh 1 uniformly in parallel.");
   args.AddOption(&rp_levels[1], "-rp2", "--refine-parallel",
                  "Number of times to refine the mesh 2 uniformly in parallel.");
   args.AddOption(&rp_levels[2], "-rp3", "--refine-parallel",
                  "Number of times to refine the mesh 3 uniformly in parallel.");
   args.AddOption(&np_list[0], "-np1", "--np1",
                  "number of MPI ranks for mesh 1");
   args.AddOption(&np_list[1], "-np2", "--np2",
                  "number of MPI ranks for mesh 2");
   args.AddOption(&np_list[2], "-np3", "--np3",
                  "number of MPI ranks for mesh 3");
   args.AddOption(&nmeshes, "-nm", "--nm",
                  "number of meshes");
   args.AddOption(&rel_tol, "-rt", "--relative tolerance",
                  "Tolerance for Schwarz iteration convergence criterion.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // 3. Setup MPI communicator for each mesh
   MPI_Comm *comml = new MPI_Comm;
   int color = 0;
   int npsum = 0;
   for (int i = 0; i < nmeshes; i++)
   {
      npsum += np_list[i];
      if (myid < npsum) { color = i; break; }
   }

   MPI_Comm_split(MPI_COMM_WORLD, color, myid, comml);
   int myidlocal, numproclocal;
   MPI_Comm_rank(*comml, &myidlocal);
   MPI_Comm_size(*comml, &numproclocal);

   // 4. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (myid == 0) { device.Print(); }

   // 5. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh *mesh = new Mesh(mesh_file_list[color], 1, 1);
   int dim = mesh->Dimension();

   // 6. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 50,000
   //    elements.
   for (int lev = 0; lev < rs_levels[color]; lev++) { mesh->UniformRefinement(); }

   // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh;
   pmesh = new ParMesh(*comml, *mesh);
   for (int l = 0; l < rp_levels[color]; l++)
   {
      pmesh->UniformRefinement();
   }
   delete mesh;

   // 5. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order. If order < 1, we
   //    instead use an isoparametric/isogeometric space.
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

   // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking all
   //    the boundary attributes from the mesh as essential (Dirichlet) and
   //    converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   if (pmesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 7. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
   //    the basis functions in the finite element fespace1.
   ParLinearForm *b = new ParLinearForm(fespace);
   ConstantCoefficient one(1.0);
   b->AddDomainIntegrator(new DomainLFIntegrator(one));
   b->Assemble();

   // 8. Define the solution vector x as a finite element grid function
   //    corresponding to fespace1. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   ParGridFunction x(fespace);
   x = 0.0;
   x.SetTrueVector();

   // 9. Setup FindPointsGSLIB and determine points on each mesh's boundary that
   //    are interior to another mesh.
   pmesh->SetCurvature(order, false, dim, Ordering::byNODES);
   {
      Vector vxyz = *pmesh->GetNodes();

      // Modify the inline-quad.mesh such that it does not cover the entire
      // domain.
      if (nmeshes == 2)
      {
         if (color == 1)   // rescale from [0, 1]^2 to [0.25, 0.75]^2
         {
            for (int i = 0; i < vxyz.Size(); i++)
            {
               vxyz(i) = 0.5 + 0.5*(vxyz(i)-0.5);
            }
         }
      }
      else if (nmeshes == 3)
      {
         if (color == 1)
         {
            // rescale from [0, 1]^2 to [0.21, 0.61] in x and [0.25, 0.75] in y
            const int pts_cnt = vxyz.Size()/dim;
            for (int i = 0; i < pts_cnt; i++)
            {
               vxyz(i) = 0.41 + 0.4*(vxyz(i)-0.5);
            }
            for (int i = 0; i < pts_cnt; i++)
            {
               vxyz(i+pts_cnt) = 0.5 + 0.5*(vxyz(i+pts_cnt)-0.5);
            }
         }
         else if (color == 2)
         {
            // rescale from [0, 1]^2 to [0.4, 0.8] in x and [0.2, 0.8] in y
            const int pts_cnt = vxyz.Size()/dim;
            for (int i = 0; i < pts_cnt; i++)
            {
               vxyz(i) = 0.6 + 0.4*(vxyz(i)-0.5);
            }
            for (int i = 0; i < pts_cnt; i++)
            {
               vxyz(i+pts_cnt) = 0.5 + 0.6*(vxyz(i+pts_cnt)-0.5);
            }
         }
      }
      pmesh->SetNodes(vxyz);
   }

   pmesh->GetNodes()->SetTrueVector();
   Vector vxyz = pmesh->GetNodes()->GetTrueVector();


   OversetFindPointsGSLIB finder(MPI_COMM_WORLD);
   finder.Setup(*pmesh, color);

   Array<int> ess_tdof_list_int;
   GetInterdomainBoundaryPoints(finder, vxyz, color,
                                ess_tdof_list, ess_tdof_list_int, dim);

   // 10. Use FindPointsGSLIB to interpolate the solution at interdomain boundary
   //     points.
   const int nb1 = ess_tdof_list_int.Size(),
             nt1 = vxyz.Size()/dim;

   MFEM_VERIFY(nb1!=0, " Please use overlapping grids.");

   Array<unsigned int> colorv;
   colorv.SetSize(nb1);

   MPI_Barrier(MPI_COMM_WORLD);
   Vector bnd1(nb1*dim);
   for (int i = 0; i < nb1; i++)
   {
      int idx = ess_tdof_list_int[i];
      for (int d = 0; d < dim; d++) { bnd1(i+d*nb1) = vxyz(idx + d*nt1); }
      colorv[i] = (unsigned int)color;
   }
   Vector interp_vals1(nb1);
   finder.Interpolate(bnd1, colorv, x, interp_vals1);

   // 11. Set up the bilinear form a(.,.) on the finite element space
   //     corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //     domain integrator.
   ParBilinearForm *a = new ParBilinearForm(fespace);
   a->AddDomainIntegrator(new DiffusionIntegrator(one));

   // 12. Assemble the bilinear form and the corresponding linear system,
   //     applying any necessary transformations such as: eliminating boundary
   //     conditions, applying conforming constraints for non-conforming AMR,
   //     static condensation, etc.
   a->Assemble();

   delete b;

   // 13. Use simultaneous Schwarz iterations to iteratively solve the PDE
   //     and interpolate interdomain boundary data to impose Dirichlet
   //     boundary conditions.

   int NiterSchwarz = 100;
   for (int schwarz = 0; schwarz < NiterSchwarz; schwarz++)
   {
      ParLinearForm *b = new ParLinearForm(fespace);
      b->AddDomainIntegrator(new DomainLFIntegrator(one));
      b->Assemble();

      OperatorPtr A;
      Vector B, X;
      a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

      // 11. Solve the linear system A X = B.
      // Use a simple symmetric Gauss-Seidel preconditioner with PCG.
      Solver *prec = NULL;
      //prec = new HypreBoomerAMG;
      CGSolver cg(MPI_COMM_WORLD);
      cg.SetRelTol(1e-12);
      cg.SetMaxIter(2000);
      cg.SetPrintLevel(0);
      if (prec) { cg.SetPreconditioner(*prec); }
      cg.SetOperator(*A);
      cg.Mult(B, X);
      delete prec;

      // 12. Recover the solution as a finite element grid function.
      a->RecoverFEMSolution(X, *b, x);

      // Interpolate boundary condition
      finder.Interpolate(x, interp_vals1);

      double dxmax = std::numeric_limits<float>::min();
      double xinf = x.Normlinf();
      double xinfg = xinf;
      MPI_Allreduce(&xinf, &xinfg, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      x.SetTrueVector();
      Vector xt = x.GetTrueVector();
      for (int i = 0; i < nb1; i++)
      {
         int idx = ess_tdof_list_int[i];
         double dx = std::abs(xt(idx)-interp_vals1(i))/xinfg;
         if (dx > dxmax) { dxmax = dx; }
         xt(idx) = interp_vals1(i);
      }
      x.SetFromTrueDofs(xt);
      double dxmaxg = dxmax;
      MPI_Allreduce(&dxmax, &dxmaxg, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

      delete b;

      if (myid == 0)
      {
         std::cout << std::setprecision(8)    <<
                   "Iteration: "           << schwarz <<
                   ", Relative residual: " << dxmaxg   << endl;
      }

      if (dxmaxg < rel_tol) { break; }
   }

   {
      // output files
      ostringstream mesh_name, sol_name;
      mesh_name << "og_mesh." << setfill('0') << setw(6) << myid;
      sol_name << "og_sol." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }

   {
      //visit visualization
      string namefile;
      namefile = "og" + std::to_string(color);
      VisItDataCollection visit_dc(namefile, pmesh);
      visit_dc.RegisterField("velocity", &x);
      visit_dc.SetFormat(true ?
                         DataCollection::SERIAL_FORMAT :
                         DataCollection::PARALLEL_FORMAT);
      visit_dc.Save();
   }

   // 14. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      //sol_sock << "parallel " << numproclocal << " " << myidlocal << "\n";
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *pmesh << x << flush;
   }

   // 15. Free the used memory.
   finder.FreeData();
   delete a;
   delete fespace;
   if (order > 0) { delete fec; }
   delete pmesh;
   delete comml;


   MPI_Finalize();

   return 0;
}
