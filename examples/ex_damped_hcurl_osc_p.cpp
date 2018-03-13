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
//               of essential boundary conditions and the optional connection
//               to the GLVis tool for visualization.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

static double muInv_   = 1.0;
static double epsilon_ = 1.0;
static double sigma_   = 1.5;
//static double sigma0_  = 1.0;
static double omega_   = 1.0;
//static double rad_     = 0.2;

static double At_ = M_SQRT2;
static double Ax_ = M_SQRT2;

void x_real_exact(const Vector &, Vector&);
void x_imag_exact(const Vector &, Vector&);
void f_real_exact(const Vector &, Vector&);
void f_imag_exact(const Vector &, Vector&);

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   double field_type = 0.5;
   bool visualization = 1;
   bool herm_conv = true;
   
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&field_type, "-f", "--field-type",
                  "Field Type Mixture: 0 - Curl Free, 1 - Divergence Free");
   args.AddOption(&herm_conv, "-herm", "--hermitian", "-no-herm",
                  "--no-hermitian", "Use convention for Hermitian operators.");
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

   ComplexOperator::Convention conv =
     herm_conv ? ComplexOperator::HERMITIAN : ComplexOperator::BLOCK_SYMMETRIC;

   if ( field_type != 0.5 )
     {
       At_ = cos(0.5 * M_PI * field_type);
       Ax_ = sin(0.5 * M_PI * field_type);
     }
   
   // 3. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim  = mesh->Dimension();
   int sdim = mesh->SpaceDimension();
   
   // 4. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 10,000 elements.
   {
     int ref_levels = 1;
     //        (int)floor(log(10000./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   {
      int par_ref_levels = 1;
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
   }

   // 6. Define a parallel finite element space on the parallel mesh. Here we
   //    use continuous Lagrange finite elements of the specified order. If
   //    order < 1, we instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec = new ND_FECollection(order, dim);
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
   HYPRE_Int size = fespace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   // 7. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs. In this example, the boundary conditions are defined
   //    by marking all the boundary attributes from the mesh as essential
   //    (Dirichlet) and converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   if (pmesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 8. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system.
   ParComplexLinearForm *b = new ParComplexLinearForm(fespace, conv);
   VectorFunctionCoefficient f_r(sdim, f_real_exact);
   VectorFunctionCoefficient f_i(sdim, f_imag_exact);
   b->AddDomainIntegrator(new VectorFEDomainLFIntegrator(f_r),
                          new VectorFEDomainLFIntegrator(f_i));
   b->Assemble();

   // 9. Define the solution vector x as a parallel finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   ParComplexGridFunction x(fespace);
   ParComplexGridFunction x_exact(fespace);
   VectorFunctionCoefficient x_r(sdim, x_real_exact);
   VectorFunctionCoefficient x_i(sdim, x_imag_exact);
   x.ProjectCoefficient(x_r, x_i);
   x_exact.ProjectCoefficient(x_r, x_i);

   if (visualization&&false)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock_r(vishost, visport);
      socketstream sol_sock_i(vishost, visport);
      sol_sock_r << "parallel " << num_procs << " " << myid << "\n";
      sol_sock_i << "parallel " << num_procs << " " << myid << "\n";
      sol_sock_r.precision(8);
      sol_sock_i.precision(8);
      sol_sock_r << "solution\n" << *pmesh << x_exact.real()
                 << "window_title 'Exact Real Part'" << flush;
      sol_sock_i << "solution\n" << *pmesh << x_exact.imag()
                 << "window_title 'Exact Imaginary Part'" << flush;
   }

   // 10. Set up the parallel sesquilinear form a(.,.) on the finite element
   //     space corresponding to the damped harmonic oscillator operator
   //     -Div(tau Grad) + omega^2 rho - i omega sigma, by adding the
   //     Diffusion domain integrator and appropriate Mass domain integrators.

   ConstantCoefficient muInvCoef(muInv_);
   // ConstantCoefficient negMuInvCoef(-muInv_);
   ConstantCoefficient epsilonCoef(omega_ * omega_ * epsilon_);
   ConstantCoefficient negEpsilonCoef(-omega_ * omega_ * epsilon_);
   ConstantCoefficient sigmaCoef(omega_ * sigma_);
   ConstantCoefficient negSigmaCoef(-omega_ * sigma_);

   ParSesquilinearForm *a = new ParSesquilinearForm(fespace, conv);
   /*
   a->AddDomainIntegrator(new CurlCurlIntegrator(muInvCoef),
                          new VectorFEMassIntegrator(negSigmaCoef));
   */
   a->AddDomainIntegrator(new CurlCurlIntegrator(muInvCoef),
                          NULL);
   a->AddDomainIntegrator(new VectorFEMassIntegrator(negEpsilonCoef),
                          new VectorFEMassIntegrator(sigmaCoef));

   // 10a. Set up the parallel bilinear form for the preconditioner
   //      corresponding to the operator
   //      -Div(tau Grad) + omega^2 rho + omega sigma
   ParBilinearForm *pcOp = new ParBilinearForm(fespace);
   pcOp->AddDomainIntegrator(new CurlCurlIntegrator(muInvCoef));
   pcOp->AddDomainIntegrator(new VectorFEMassIntegrator(epsilonCoef));
   pcOp->AddDomainIntegrator(new VectorFEMassIntegrator(sigmaCoef));
   // 11. Assemble the parallel bilinear form and the corresponding linear
   //     system, applying any necessary transformations such as: parallel
   //     assembly, eliminating boundary conditions, applying conforming
   //     constraints for non-conforming AMR, etc.
   a->Assemble();
   pcOp->Assemble();

   OperatorHandle A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
   x = 0.0;
   X = 0.0;

   OperatorHandle PCOp;
   pcOp->FormSystemMatrix(ess_tdof_list, PCOp);

   if (myid == 0)
   {
     ComplexHypreParMatrix * Ahyp =
       dynamic_cast<ComplexHypreParMatrix*>(A.Ptr());

     cout << "Size of linear system: "
	  << 2 * Ahyp->real().GetGlobalNumRows() << endl << endl;
   }

   // 12. Define and apply a parallel FGMRES solver for AX=B with the BoomerAMG
   //     preconditioner from hypre.
   {
     Array<HYPRE_Int> blockTrueOffsets;
     blockTrueOffsets.SetSize(3);
     blockTrueOffsets[0] = 0;
     blockTrueOffsets[1] = PCOp.Ptr()->Height();
     blockTrueOffsets[2] = PCOp.Ptr()->Height();
     blockTrueOffsets.PartialSum();

     BlockDiagonalPreconditioner BDP(blockTrueOffsets);

     HypreAMS amsr(dynamic_cast<HypreParMatrix&>(*PCOp.Ptr()), fespace);
     ScaledOperator amsi(&amsr,
			 (conv == ComplexOperator::HERMITIAN)?1.0:-1.0);
     
     BDP.SetDiagonalBlock(0,&amsr);
     BDP.SetDiagonalBlock(1,&amsi);
     BDP.owns_blocks = 0;
     
     FGMRESSolver fgmres(MPI_COMM_WORLD);
     fgmres.SetPreconditioner(BDP);
     fgmres.SetOperator(*A.Ptr());
     fgmres.SetRelTol(1e-12);
     fgmres.SetMaxIter(1000);
     fgmres.SetPrintLevel(1);
     fgmres.Mult(B, X);
   }
   
   // 13. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   a->RecoverFEMSolution(X, *b, x);

   double err_r = x.real().ComputeL2Error(x_r);
   double err_i = x.imag().ComputeL2Error(x_i);
   
   if ( myid == 0 )
   {
     cout << endl;
     cout << "|| Re (x_h - x) ||_{L^2} = " << err_r << endl;
     cout << "|| Im (x_h - x) ||_{L^2} = " << err_i << endl;
     cout << endl;
   }
   
   // 14. Save the refined mesh and the solution in parallel. This output can
   //     be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
   {
      ostringstream mesh_name, sol_r_name, sol_i_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      sol_r_name << "sol_r." << setfill('0') << setw(6) << myid;
      sol_i_name << "sol_i." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);

      ofstream sol_r_ofs(sol_r_name.str().c_str());
      ofstream sol_i_ofs(sol_i_name.str().c_str());
      sol_r_ofs.precision(8);
      sol_i_ofs.precision(8);
      x.real().Save(sol_r_ofs);
      x.imag().Save(sol_i_ofs);
   }

   // 15. Send the solution by socket to a GLVis server.
   if (visualization&&false)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock_r(vishost, visport);
      socketstream sol_sock_i(vishost, visport);
      sol_sock_r << "parallel " << num_procs << " " << myid << "\n";
      sol_sock_i << "parallel " << num_procs << " " << myid << "\n";
      sol_sock_r.precision(8);
      sol_sock_i.precision(8);
      sol_sock_r << "solution\n" << *pmesh << x.real()
                 << "window_title 'Comp Real Part'" << flush;
      sol_sock_i << "solution\n" << *pmesh << x.imag()
                 << "window_title 'Comp Imaginary Part'" << flush;
   }
   if (visualization&&false)
   {
      x_exact -= x;

      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock_r(vishost, visport);
      socketstream sol_sock_i(vishost, visport);
      sol_sock_r << "parallel " << num_procs << " " << myid << "\n";
      sol_sock_i << "parallel " << num_procs << " " << myid << "\n";
      sol_sock_r.precision(8);
      sol_sock_i.precision(8);
      sol_sock_r << "solution\n" << *pmesh << x_exact.real()
                 << "window_title 'Exact-Comp Real Part'" << flush;
      sol_sock_i << "solution\n" << *pmesh << x_exact.imag()
                 << "window_title 'Exact-Comp Imaginary Part'" << flush;
   }
   if (visualization)
   {
      ParGridFunction x_t(fespace);
      x_t = x.real();
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *pmesh << x_t
	       << "window_title 'Harmonic Solution (t = 0.0 T)'"
	       << "pause\n" << flush;
      if (myid == 0)
	cout << "GLVis visualization paused."
	     << " Press space (in the GLVis window) to resume it.\n";
      int num_frames = 32;
      int i = 0;
      while(sol_sock)
      {
	double t = (double)(i % num_frames) / num_frames;
	ostringstream oss;
	oss << "Harmonic Solution (t = " << t << " T)";
	
	add(cos(2.0 * M_PI * t), x.real(),
	    sin(2.0 * M_PI * t), x.imag(), x_t);
	sol_sock << "parallel " << num_procs << " " << myid << "\n";
	sol_sock << "solution\n" << *pmesh << x_t
		 << "window_title '" << oss.str() << "'" << flush;
	i++;
      }
   }

   // 16. Free the used memory.
   delete a;
   delete b;
   delete pcOp;
   delete fespace;
   delete fec;
   delete pmesh;

   MPI_Finalize();

   return 0;
}

void x_real_exact(const Vector &p, Vector &x)
{
   int dim = p.Size();
   x.SetSize(dim); x = 0.0;

   double cx = cos(M_PI * p(0));
   double cy = (dim>1)?cos(M_PI * p(1)):1.0;
   double cz = (dim>2)?cos(M_PI * p(2)):1.0;

   double sx = sin(M_PI * p(0));
   double sy = (dim>1)?sin(M_PI * p(1)):1.0;
   double sz = (dim>2)?sin(M_PI * p(2)):1.0;

   double at = (dim>2)?sqrt(1.0/6.0):0.5;
   double ax = (dim>2)?sqrt(1.0/12.0):0.5;
   
   x(0) = -at * At_ * cx * sy * sz;
   x(1) = -at * At_ * sx * cy * sz;
   if (dim>2) x(2) = -at * At_ * sx * sy * cz;

   if (dim == 2)
   {
     x(0) += ax * Ax_ * sx * cy;
     x(1) -= ax * Ax_ * cx * sy;
   }
   else
   {
     x(0) += ax * Ax_ * sx * (cy * sz - sy * cz);
     x(1) += ax * Ax_ * sy * (cz * sx - sz * cx);
     x(2) += ax * Ax_ * sz * (cx * sy - sx * cy);
   }
}

void x_imag_exact(const Vector &p, Vector &x)
{
   int dim = p.Size();
   x.SetSize(dim); x = 0.0;

   double cx = cos(M_PI * p(0));
   double cy = (dim>1)?cos(M_PI * p(1)):1.0;
   double cz = (dim>2)?cos(M_PI * p(2)):1.0;

   double sx = sin(M_PI * p(0));
   double sy = (dim>1)?sin(M_PI * p(1)):1.0;
   double sz = (dim>2)?sin(M_PI * p(2)):1.0;

   double at = (dim>2)?sqrt(1.0/6.0):0.5;
   double ax = (dim>2)?sqrt(1.0/12.0):0.5;
   
   x(0) = at * At_ * sx * cy * cz;
   x(1) = at * At_ * cx * sy * cz;
   if (dim>2) x(2) = at * At_ * cx * cy * sz;

   if (dim == 2)
   {
     x(0) -= ax * Ax_ * cx * sy;
     x(1) += ax * Ax_ * sx * cy;
   }
   else
   {
     x(0) += ax * Ax_ * cx * (cy * sz - sy * cz);
     x(1) += ax * Ax_ * cy * (cz * sx - sz * cx);
     x(2) += ax * Ax_ * cz * (cx * sy - sx * cy);
   }
}
/*
complex<double> f_exact(const Vector &x)
{
   int dim = x.Size();

   double cx = cos(M_PI * x(0));
   double cy = (dim>1)?cos(M_PI * x(1)):1.0;
   double cz = (dim>2)?cos(M_PI * x(2)):1.0;

   double sx = sin(M_PI * x(0));
   double sy = (dim>1)?sin(M_PI * x(1)):1.0;
   double sz = (dim>2)?sin(M_PI * x(2)):1.0;

   complex<double> a(omega_ * sigma_,
                     rho_ * pow(omega_, 2) - pow(M_PI, 2) * tau_ * dim);
   complex<double> z(cx * cy * cz, - sx * sy * sz);

   return a * z;
}
*/
/*
void E0_exact(const Vector &x, Vector &E0r, Vector &E0i)
{
     int dim = x.Size();

     E0r.SetSize(dim); E0r = 0.0;
     E0i.SetSize(dim); E0i = 0.0;

     double kx = sqrt(0.5 * omega_ * sigma0_ / muInv_);
     double expkx = exp(-kx * x(0));
     
     E0r(1) =   expkx * cos(kx * x(0));
     E0i(1) = - expkx * sin(kx * x(0));
}
*/
void f_real_exact(const Vector &p, Vector &f)
{
     int dim = p.Size();
     f.SetSize(dim);

   double cx = cos(M_PI * p(0));
   double cy = (dim>1)?cos(M_PI * p(1)):1.0;
   double cz = (dim>2)?cos(M_PI * p(2)):1.0;

   double sx = sin(M_PI * p(0));
   double sy = (dim>1)?sin(M_PI * p(1)):1.0;
   double sz = (dim>2)?sin(M_PI * p(2)):1.0;

   double k2 = (double)dim * M_PI * M_PI * muInv_ - epsilon_ * omega_* omega_;

   if (dim == 2)
     {
       f(0) = 0.5 * At_ * omega_ *
	 (omega_ * epsilon_ * cx * sy - sigma_ * sx * cy)
	 + 0.5 * Ax_ *
	 (omega_ * sigma_ * cx * sy + k2 * sx * cy);
       f(1) = -0.5 * At_ * omega_ *
	 (sigma_ * cx * sy - omega_ * epsilon_ * sx * cy)
	 - 0.5 * Ax_ *
	 (k2 * cx * sy + omega_ * sigma_ * sx * cy);
     }
     else
     {
       double syz = sin(M_PI * (p[1] - p[2]));
       double szx = sin(M_PI * (p[2] - p[0]));
       double sxy = sin(M_PI * (p[0] - p[1]));
       
       double at = sqrt(1.0 / 6.0);
       double ax = sqrt(1.0 / 12.0);
       
       f(0) = at * At_ * omega_ *
	 (omega_ * epsilon_ * cx * sy * sz - sigma_ * sx * cy * cz)
	 -ax * Ax_ * (k2 * sx - omega_ * sigma_ * cx) * syz;
       f(1) = at * At_ * omega_ *
	 (omega_ * epsilon_ * sx * cy * sz - sigma_ * cx * sy * cz)
	 -ax * Ax_ * (k2 * sy - omega_ * sigma_ * cy) * szx;
       f(2) = at * At_ * omega_ *
	 (omega_ * epsilon_ * sx * sy * cz - sigma_ * cx * cy * sz)
	 -ax * Ax_ * (k2 * sz - omega_ * sigma_ * cz) * sxy;
     }
}

void f_imag_exact(const Vector &p, Vector &f)
{
   int dim = p.Size();
   f.SetSize(dim);

   double cx = cos(M_PI * p(0));
   double cy = (dim>1)?cos(M_PI * p(1)):1.0;
   double cz = (dim>2)?cos(M_PI * p(2)):1.0;

   double sx = sin(M_PI * p(0));
   double sy = (dim>1)?sin(M_PI * p(1)):1.0;
   double sz = (dim>2)?sin(M_PI * p(2)):1.0;

   double k2 = (double)dim * M_PI * M_PI * muInv_ - epsilon_ * omega_* omega_;

   if (dim == 2)
   {
      f(0) = -0.5 * At_ * omega_ *
	(sigma_ * cx * sy + omega_ * epsilon_ * sx * cy)
	- 0.5 * Ax_ *
	(k2 * cx * sy - omega_ * sigma_ * sx * cy);
      f(1) = -0.5 * At_ * omega_ *
	(omega_ * epsilon_ * cx * sy + sigma_ * sx * cy)
	- 0.5 * Ax_ *
	(omega_ * sigma_ * cx * sy - k2 * sx * cy);
   }
   else
   {
       double syz = sin(M_PI * (p[1] - p[2]));
       double szx = sin(M_PI * (p[2] - p[0]));
       double sxy = sin(M_PI * (p[0] - p[1]));
       
       double at = sqrt(1.0 / 6.0);
       double ax = sqrt(1.0 / 12.0);
       
       f(0) = -at * At_ * omega_ *
	 (omega_ * epsilon_ * sx * cy * cz + sigma_ * cx * sy * sz)
	 -ax * Ax_ * (k2 * cx + omega_ * sigma_ * sx) * syz;
       f(1) = -at * At_ * omega_ *
	 (omega_ * epsilon_ * cx * sy * cz + sigma_ * sx * cy * sz)
	 -ax * Ax_ * (k2 * cy + omega_ * sigma_ * sy) * szx;
       f(2) = -at * At_ * omega_ *
	 (omega_ * epsilon_ * cx * cy * sz + sigma_ * sx * sy * cz)
	 -ax * Ax_ * (k2 * cz + omega_ * sigma_ * sz) * sxy;
   }
}
