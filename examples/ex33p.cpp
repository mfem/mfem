//                       MFEM Example 33 - Parallel Version
//
// Compile with: make ex33p
//
// Sample runs:  mpirun -np 4 ex33p -m ../data/square-disc.mesh -alpha 0.33 -o 2
//               mpirun -np 4 ex33p -m ../data/star.mesh -alpha 0.99 -o 3
//               mpirun -np 4 ex33p -m ../data/inline-quad.mesh -alpha 0.5 -o 3
//               mpirun -np 4 ex33p -m ../data/disc-nurbs.mesh -alpha 0.33 -o 3
//               mpirun -np 4 ex33p -m ../data/l-shape.mesh -alpha 0.33 -o 3 -r 4
//
// Description:
//
//  In this example we solve the following fractional PDE with MFEM:
//
//    ( - Δ )^α u = f  in Ω,      u = 0  on ∂Ω,      0 < α < 1,
//
//  To solve this FPDE, we rely on a rational approximation [2] of the normal
//  linear operator A^{-α}, where A = - Δ (with associated homogeneous
//  boundary conditions). Namely, we first approximate the operator
//
//    A^{-α} ≈ Σ_{i=0}^N c_i (A + d_i I)^{-1},      d_0 = 0,   d_i > 0,
//
//  where I is the L2-identity operator and the coefficients c_i and d_i
//  are generated offline to a prescribed accuracy in a pre-processing step.
//  We use the triple-A algorithm [1] to generate the rational approximation
//  that this partial fractional expansion derives from. We then solve N+1
//  independent integer-order PDEs,
//
//    A u_i + d_i u_i = c_i f  in Ω,      u_i = 0  on ∂Ω,      i=0,...,N,
//
//  using MFEM and sum u_i to arrive at an approximate solution of the FPDE
//
//    u ≈ Σ_{i=0}^N u_i.
//
// References:
//
// [1] Nakatsukasa, Y., Sète, O., & Trefethen, L. N. (2018). The AAA algorithm
//     for rational approximation. SIAM Journal on Scientific Computing, 40(3),
//     A1494-A1522.
//
// [2] Harizanov, S., Lazarov, R., Margenov, S., Marinov, P., & Pasciak, J.
//     (2020). Analysis of numerical methods for spectral fractional elliptic
//     equations based on the best uniform rational approximation. Journal of
//     Computational Physics, 408, 109285.
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "ex33.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 0. Initialize MPI.
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   int num_refs = 3;
   bool visualization = true;
   bool visualize_x = false;
   double alpha = 0.5;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&num_refs, "-r", "--refs",
                  "Number of uniform refinements");
   args.AddOption(&alpha, "-alpha", "--alpha",
                  "Fractional exponent");
   args.AddOption(&visualize_x, "-vis_x", "--visualize_x", "-no-vis_x",
                  "--no-visualization_x",
                  "Enable or disable GLVis visualization of each integer-order PDE solution.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization of the fractional PDE solution.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   if (Mpi::Root())
   {
      args.PrintOptions(cout);
   }

   Array<double> coeffs, poles;

   // 2. Compute the coefficients that define the integer-order PDEs.
   ComputePartialFractionApproximation(alpha,coeffs,poles);

   int num_par_solves;
   int max_par_solves = max(1,num_procs/2);
   for (num_par_solves=max_par_solves; num_par_solves>0; num_par_solves--)
   {
      if (num_procs%num_par_solves==0 && num_par_solves<coeffs.Size())
      {
         break;
      }
   }
   if (num_par_solves == 1) {num_par_solves = num_procs;}

   int solver_ranks = num_procs/num_par_solves;

   // 3. Split the MPI communicator:
   //    row_comm is used for parallel partition of the mesh
   //    col_comm is used for independent integer-order solves
   int row_color = myid / solver_ranks; // Determine color based on row
   int col_color = myid % solver_ranks; // Determine color based on col

   MPI_Comm row_comm, col_comm;
   MPI_Comm_split(MPI_COMM_WORLD, row_color, myid, &row_comm);
   MPI_Comm_split(MPI_COMM_WORLD, col_color, myid, &col_comm);

   int row_rank, row_size, col_rank, col_size;
   MPI_Comm_rank(row_comm, &row_rank);
   MPI_Comm_size(row_comm, &row_size);
   MPI_Comm_rank(col_comm, &col_rank);
   MPI_Comm_size(col_comm, &col_size);

   if (Mpi::Root())
   {
      mfem::out << "\nTotal number of MPI ranks = " << num_procs << endl;
      mfem::out << "Number of independent parallel solves = " << col_size << endl;
      mfem::out << "Number of MPI ranks within each solve = " << row_size
                <<"\n" << endl;
   }

   // 4. Read the mesh from the given mesh file.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   // 5. Refine the mesh to increase the resolution.
   for (int i = 0; i < num_refs; i++)
   {
      mesh.UniformRefinement();
   }

   ParMesh pmesh(row_comm, mesh);
   mesh.Clear();

   // 6. Define a finite element space on the mesh.
   H1_FECollection fec(order, dim);
   ParFiniteElementSpace fespace(&pmesh, &fec);
   if (Mpi::Root())
   {
      cout << "Number of finite element unknowns: "
           << fespace.GetTrueVSize() << endl;
   }

   // 7. Determine the list of true (i.e. conforming) essential boundary dofs.
   Array<int> ess_tdof_list;
   if (pmesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh.bdr_attributes.Max());
      ess_bdr = 1;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 8. Define diffusion coefficient, load, and solution GridFunction.
   ConstantCoefficient f(1.0);
   ConstantCoefficient one(1.0);
   ParGridFunction u(&fespace);
   ParGridFunction x(&fespace);
   u = 0.0;

   // 9. Set up the linear form b(.) for integer-order PDE solves.
   ParLinearForm b(&fespace);
   b.AddDomainIntegrator(new DomainLFIntegrator(f));
   b.Assemble();

   int my_coeff_size = max(coeffs.Size()/col_size,1);
   int ibeg = col_rank*my_coeff_size;
   if (ibeg + 2*my_coeff_size > coeffs.Size())
   {
      my_coeff_size = coeffs.Size()-col_rank*my_coeff_size;
   }
   else if (ibeg > coeffs.Size() - 1)
   {
      my_coeff_size = 0;
   }

   int iend = ibeg+my_coeff_size;


   for (int i = ibeg; i < iend; i++)
   {
      // 10. Reset GridFunction for integer-order PDE solve.
      x = 0.0;

      // 11. Set up the bilinear form a(.,.) for integer-order PDE solve.
      ParBilinearForm a(&fespace);
      a.AddDomainIntegrator(new DiffusionIntegrator(one));
      ConstantCoefficient d_i(-poles[i]);
      a.AddDomainIntegrator(new MassIntegrator(d_i));
      a.Assemble();

      // 12. Assemble the bilinear form and the corresponding linear system.
      OperatorPtr A;
      Vector B, X;
      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

      // 13. Solve the linear system A X = B.
      HypreBoomerAMG * prec = new HypreBoomerAMG;
      prec->SetPrintLevel(-1);

      int print_level = (col_rank==0) ? 3 : 0;
      if (Mpi::Root())
      {
         mfem::out << "\nMPI rank " <<  myid
                   << ": Solving PDE -Δ u + " << -poles[i]
                   << " u = " << coeffs[i] << " f " << endl;
      }
      CGSolver cg(row_comm);
      cg.SetRelTol(1e-12);
      cg.SetMaxIter(2000);
      cg.SetPrintLevel(print_level);
      cg.SetPreconditioner(*prec);
      cg.SetOperator(*A);
      cg.Mult(B, X);
      delete prec;

      // 14. Recover the solution as a finite element grid function.
      a.RecoverFEMSolution(X, b, x);

      // 15. Accumulate integer-order PDE solutions.
      x *= coeffs[i];
      u += x;

      // 16. Send integer-order PDE solutions to a GLVis server.
      if (visualize_x)
      {
         if (col_rank > 0 && i < iend-1)
         {
            MPI_Status status;
            MPI_Recv(nullptr,0,MPI_INT, col_rank-1,0,col_comm,&status);
         }
         char vishost[] = "localhost";
         int  visport   = 19916;
         socketstream xout(vishost, visport);
         xout.precision(8);
         ostringstream oss;
         oss << "Solution of PDE -Δ u + " << -poles[i]
             << " u = " << coeffs[i] << " f" ;
         xout << "parallel " << row_size << " " << row_rank << "\n";
         xout << "solution\n" << pmesh << x
              << "window_title '" << oss.str() << "'" << flush;
         if (col_rank < col_size-1)
         {
            MPI_Send(nullptr,0,MPI_INT,col_rank+1,0,col_comm);
         }
      }
   }

   // 17. Accumulate for the fractional PDE solution
   MPI_Allreduce(MPI_IN_PLACE, u.GetData(), u.Size(),
                 MPI_DOUBLE, MPI_SUM,col_comm);

   // 18. Send fractional PDE solution to a GLVis server.
   if (visualization)
   {
      if (col_rank == 0)
      {
         char vishost[] = "localhost";
         int  visport   = 19916;
         socketstream uout(vishost, visport);
         uout.precision(8);
         ostringstream oss;
         oss << "Solution of fractional PDE -Δ^" << alpha
             << " u = f" ;
         uout << "parallel " << row_size << " " << row_rank << "\n";
         uout << "solution\n" << pmesh << u
              << "window_title '" << oss.str() << "'" << flush;
      }
   }

   return 0;
}
