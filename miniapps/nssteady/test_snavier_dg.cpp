#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <random>
#include "snavier_dg.hpp"

using namespace std;
using namespace mfem;

struct s_NavierContext
{
   // Mesh
   int dim = 2;
   int elem= 1;
   int n   = 1;
   int ser_ref_levels = 0;
   int par_ref_levels = 0;
   int total_ref_levels= 0; // for convergence study

   // Finite Element Spaces
   int  order = 1;
   bool reduce_order_pressure = true;
   bool reduce_order_sig = true;
   double kappa_0=1.0;  // stabilization parameter

   // Problems
   int example=1;
   double kinvis = 1.0 / 1.0;
   double reference_pressure = 0.0;
   double Re	 = 1.0 / kinvis;
   bool dirichlet=false; // assume using neumann boundary.

   // Linear solver
   bool iterative=false;
   bool petsc =true;
   const char *petscrc_file = "rc_direct";
   bool use_ksp_solver = false;
   int max_lin_it = 1000;
   double lin_it_rtol = 1e-6;
   double lin_it_atol = 1e-8;
   int setPrintLevel_Lin = 2;

   // Picard iteration
   bool Picard=false;
   int max_picard_it = 20;
   double picard_it_rtol = 1e-7;
   double picard_it_atol = 1e-10;
   int setPrintLevel_Picard = 1;

   // Post-processing
   bool visualization = false;

   // Print
   bool verbose=true;
} ctx;

// analytical solution
double p_exact(const Vector &xvec);
void u_exact(const Vector &xvec, Vector &u);
void w_exact(const Vector &xvec, Vector &u);
void sig_exact(const Vector &xvec, Vector &sig);
void pseudo_traction_exact(const Vector &xvec, Vector &pseudo_traction);
void f_source(const Vector &xvec, Vector &u);

// initial guess
void u_init(const Vector &xvec, Vector &u);

int main(int argc, char *argv[])
{
   //
   /// Initialize MPI.
   //
   int nprocs, myrank;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

   //
   /// Parse command-line options.
   //
   mfem::OptionsParser args(argc, argv);
   args.AddOption(&ctx.example, "-e", "--example",
                  "Example to use.");
   args.AddOption(&ctx.dim,
                     "-d",
                     "--dimension",
                     "Dimension of the problem (2 = 2d, 3 = 3d)");
   args.AddOption(&ctx.elem,
                     "-e",
                     "--element-type",
                     "Type of elements used (0: Quad/Hex, 1: Tri/Tet)");
   args.AddOption(&ctx.n,
                     "-n",
                     "--num-elements",
                     "Number of elements in uniform mesh.");
   args.AddOption(&ctx.ser_ref_levels,
                     "-rs",
                     "--refine-serial",
                     "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&ctx.par_ref_levels,
                     "-rp",
                     "--refine-parallel",
                     "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&ctx.order, "-o", "--order",
                     "Finite element order (polynomial degree)");
   args.AddOption(&ctx.reduce_order_pressure, "-reduce-p", "--reduce-order-p", "-full-p",
				  	 "--full-order-p",
					 "Using k-1 order in approximation of pressure.");
   args.AddOption(&ctx.reduce_order_sig, "-reduce-s", "--reduce-order-s", "-full-s",
				  	 "--full-order-s",
					 "Using k-1 order in approximation of gradient of velocity.");
   args.AddOption(&ctx.kappa_0, "-k_0", "--kappa_0",
                  	 "Setting up the stabilization parameter.");
   args.AddOption(&ctx.kinvis,
                     "-kv",
                     "--kin-viscosity",
                     "Kinematic viscosity");
   args.AddOption(&ctx.petsc, "-petsc", "--use-petsc",
                  	 "-no-petsc", "--no-use-petsc",
					 "Enable or disable SC solver.");
   args.AddOption(&ctx.petscrc_file, "-petscopts", "--petscopts",
                  	 "PetscOptions file to use.");
   args.AddOption(&ctx.use_ksp_solver, "-ksp", "--ksp_solver", "-lu",
                  	 "--lu_solver", "Iterative solver or direct solver setup.");
   args.AddOption(&ctx.lin_it_atol, "-lin_atol", "--lin_abs_tolerance",
                  	 "Absolute tolerance in the iteration of the linear solver.");
   args.AddOption(&ctx.max_lin_it, "-lin_maxit", "--lin_max_nonlin_it",
                  	 "Maximum number of iterations of the linear solver.");
   args.AddOption(&ctx.setPrintLevel_Lin, "-printl", "--print_level",
                  	 "Setting the printlevel.");
   args.AddOption(&ctx.Picard,
                     "-Picard",
					 "--solve-Picard",
					 "-Oseen",
					 "--solve-Oseen",
					 "Solve full stationary incompressible flow system (with Picard iteration) or the Oseen equations.");
   args.AddOption(&ctx.picard_it_atol,
                  	 "-pic-atol",
					 "--pictard-abs-tolerance",
					 "Absolute tolerance for the Picard solve.");
   args.AddOption(&ctx.max_picard_it,
                  	 "-pic-it",
					 "--picard-iterations",
					 "Maximum iterations for the linear solve.");
   args.AddOption(&ctx.visualization,
                     "-vis",
					 "--visualization",
					 "-no-vis",
					 "--no-visualization",
					 "Enable or disable GLVis visualization.");
   args.AddOption(&ctx.verbose, "-v", "--verbose",
                  	 "-s", "--silent",
					 "Enable or disable printing out detailed information.");

   args.Parse();
   if (!args.Good())
   {
      if (ctx.verbose && (myrank==0))
      {
         args.PrintUsage(std::cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (ctx.verbose && (myrank==0))
   {
       args.PrintOptions(std::cout);
   }

   //
   /// Sanity check for the inputs
   //
   if(ctx.kappa_0 < 1e-14)
	   mfem_error("The stabilization parameter has to be greater than zero!");
   if(!(ctx.Picard))
	   ctx.max_picard_it = 1;
   //
   /// Read the (serial) mesh from the given mesh file on all processors.
   //
   Element::Type type;
   switch (ctx.elem)
   {
      case 0: // quad
         type = (ctx.dim == 2) ? Element::QUADRILATERAL: Element::HEXAHEDRON;
         break;
      case 1: // tri
         type = (ctx.dim == 2) ? Element::TRIANGLE: Element::TETRAHEDRON;
         break;
   }

   Mesh mesh;
   switch (ctx.dim)
   {
   	  // Create a (0,1)x(0,1) square or (0,1)x(0,1)x(0,1) cube
      case 2: // 2d
         mesh = Mesh::MakeCartesian2D(ctx.n,ctx.n,type,true);
         break;
      case 3: // 3d
         mesh = Mesh::MakeCartesian3D(ctx.n,ctx.n,ctx.n,type,true);
         break;
   }

   //
   /// Set up boundary attribute (assume the domain is (0,1)^d)
   //
   /*            attr=1
    *          * - - - *
    * attr==1  |       |
    *          |       |  attr=2
    *          * - - - *
    *            attr=1
    *
    *   attr=1: Dirichlet
    *   attr=2: Neumann
    */
   if (!ctx.dirichlet){
	   for (int i = 0; i < mesh.GetNBE(); i++){
		   Array< int > bdr_v;
		   mesh.GetBdrElementVertices(i,bdr_v);
		   double *coord;
		   double right_bdr= 0.0;
		   // loop over all nodes that form a boundary element (2d-edge, 3d-face)
		   for(int d=0; d<bdr_v.Size(); d++){
			   coord = mesh.GetVertex(bdr_v[d]);
			   right_bdr += coord[0]; // sum up x-coordinate
		   }
		   if (ctx.example == 2){
			   // do nothing now
		   }else{
			   // only works for square doamin (0,1)x(0,1) (or (0,1)x(0,1)x(0,1))
			   if (std::fabs(right_bdr-ctx.dim) < 1e-14){
				   mesh.SetBdrAttribute(i, 2); // Neumann bouundary
			   }else{
				   mesh.SetBdrAttribute(i, 1); // Dirichlet bouundary
			   }
		   }
	   }
   }else{
	   for (int i = 0; i < mesh.GetNBE(); i++)
		   mesh.SetBdrAttribute(i, 1); // Dirichlet bouundary
   }



   for (int l = 0; l < ctx.ser_ref_levels; l++)
   {
       mesh.UniformRefinement();
   }

   //
   /// Initialize errors (used for convergence study)
   //
   Vector u_l2errors(ctx.total_ref_levels+1), p_l2errors(ctx.total_ref_levels+1);
   u_l2errors = 0.0;
   p_l2errors = 0.0;

   //
   /// Define a parallel mesh by a partitioning of the serial mesh.
   // Refine this mesh further in parallel to increase the resolution. Once the
   // parallel mesh is defined, the serial mesh can be deleted.
   //
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   {
       for (int l = 0; l < ctx.par_ref_levels; l++)
       {
          pmesh->UniformRefinement();
       }
   }

   //
   /// Define the coefficients (e.g. parameters, analytical solution/s).
   //
   FunctionCoefficient p_ex(p_exact);
   VectorFunctionCoefficient u_ex(ctx.dim, u_exact);
   VectorFunctionCoefficient w_field(ctx.dim, w_exact);
   VectorFunctionCoefficient f_coeff(ctx.dim, f_source);
   VectorFunctionCoefficient pseudo_traction_coeff((ctx.dim)*(ctx.dim),pseudo_traction_exact);
   VectorFunctionCoefficient u_ini(ctx.dim, u_init);

   // Create solver
   int porder=ctx.order;
   if(ctx.reduce_order_pressure)
	   porder -= 1;

   int sigorder=ctx.order;
   if(ctx.reduce_order_sig)
	   sigorder -= 1;
   SNavierPicardDGSolver* NSSolver = new SNavierPicardDGSolver(pmesh, sigorder, ctx.order, porder, ctx.kappa_0, ctx.kinvis, ctx.verbose);


   // Set parameters of the Fixed Point Solver
   SolverParams sFP = {ctx.picard_it_rtol, ctx.picard_it_atol, ctx.max_picard_it, ctx.setPrintLevel_Picard};   // rtol, atol, maxIter, print level
   NSSolver->SetFixedPointSolver(sFP);

   // Set parameters of the Linear Solvers
   SolverParams sL = {ctx.lin_it_rtol, ctx.lin_it_atol, ctx.max_lin_it, ctx.setPrintLevel_Lin, ctx.petsc, ctx.petscrc_file};
   NSSolver->SetLinearSolvers(sL);


   //
   /// Add boundary conditions (Velocity-Dirichlet, Traction) and forcing term/s
   //
   // Acceleration term
   Array<int> domain_attr(pmesh->attributes.Max());
   domain_attr = 1;
   NSSolver->AddAccelTerm(&f_coeff,domain_attr);

   // Essential velocity bcs
   Array<int> ess_attr(pmesh->bdr_attributes.Max());
   ess_attr = 0; ess_attr[0] = 1; // Dirichlet boundary (attribute: 0+1=1)
   NSSolver->AddVelDirichletBC(&u_ex, ess_attr);

   // Traction (neumann) bcs
   Array<int> trac_attr(pmesh->bdr_attributes.Max());
   trac_attr = 0; trac_attr[1] = 1; // Neumann boundary (attribute: 1+1=2)
   NSSolver->AddTractionBC(&pseudo_traction_coeff,trac_attr);

   //
   /// Set initial guess
   //
   if(ctx.Picard){
	   // Picard (N-S) solve
	   NSSolver->SetInitialConditionVel(u_ini, u_ini);
   }else{
	   // Oseen solve
	   NSSolver->SetInitialConditionVel(u_ex, w_field);
   }
   //
   /// Finalize Setup of solver
   //
   NSSolver->Setup();

   //
   /// Solve the forward problem
   //
   NSSolver->FSolve();

   //
   /// Extract the numerical solution
   //
   ParGridFunction* velocityPtr = NSSolver->GetVSol();
   ParGridFunction* pressurePtr = NSSolver->GetPSol();

   //
   /// Evaluate errors
   //
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
       irs[i] = &(IntRules.Get(i, std::max(2, 2*ctx.order+3)));
   }
   double err_u = velocityPtr->ComputeL2Error(u_ex, irs);
   double err_p = pressurePtr->ComputeL2Error(p_ex, irs);

   int ref_levels=0; //TODO
   u_l2errors(ref_levels) = fabs(err_u);
   p_l2errors(ref_levels) = fabs(err_p);

   if (ctx.verbose && (myrank==0)){
		cout <<"\n\n"<<endl;
		cout << "Re: "<< ctx.Re << endl;
		std::cout << "-----------------------\n";
		printf("        sigma (psuedo stress)  u (velocity)   p (pressure)\n");
		printf("order        %d                    %d             %d      \n",sigorder,ctx.order,porder);
		std::cout << "-----------------------\n";
		std::cout <<
				 "level  u_l2errors  order   p_l2errors  order\n";
		std::cout << "-----------------------\n";
		for (int ref_levels = 0;
				ref_levels <= ctx.total_ref_levels; ref_levels++)
		{
		  if (ref_levels == 0)
		  {
			 std::cout << "  " << ref_levels << "    "
					   << std::setprecision(2) << std::scientific << u_l2errors(ref_levels)
					   << "    " << " -       "
					   << std::setprecision(2) << std::scientific << p_l2errors(ref_levels)
					   << "    " << " -       "<< std::endl;
		  }
		  else
		  {
			 double u_order   = log(u_l2errors(ref_levels)/u_l2errors(ref_levels-1))/log(0.5);
			 double p_order   = log(p_l2errors(ref_levels)/p_l2errors(ref_levels-1))/log(0.5);
			 std::cout << "  " << ref_levels << "    "
					   << "    " << std::setprecision(2) << std::scientific << u_l2errors(ref_levels)
					   << "   " << std::setprecision(4) << std::fixed << u_order
					   << "    " << std::setprecision(2) << std::scientific << p_l2errors(ref_levels)
					   << "   " << std::setprecision(4) << std::fixed << p_order << std::endl;
		  }
		}
   }


   delete pmesh;
   MPI_Finalize();
   return 0;
}

double p_exact(const Vector &xvec)
{
   double x = xvec(0);
   double y = xvec(1);
   double z = 0.0;
   int dim = xvec.Size();
   if (dim == 3) z = xvec(2);

   double p=0.0;
   if(dim == 2){
	   switch (ctx.example){
		   case 1:
		   {
			   p = 1-2*x;
			   break;
		   }
		   case 2:
		   {
			   break;
		   }
		   case 0:
		   default:
		   {
			   break;
		   }
	   }
   }else{
	   switch (ctx.example){
		   case 1:
		   {
			   p=1-2*x;
			   break;
		   }
		   case 2:
		   {
			   break;
		   }
		   case 0:
		   default:
		   {
			   break;
		   }
	   }
   }
   return p;
}

void u_exact(const Vector &xvec, Vector &u)
{
   double x = xvec(0);
   double y = xvec(1);
   double z = 0.0;
   int dim = xvec.Size();
   if (dim == 3) z = xvec(2);

   u=0.0;
   if (dim == 2){
	   switch (ctx.example){
		   case 0:
		   {
			   u(0) = 1.0;
			   u(1) = 1.0;
			   break;
		   }
		   case 1:
		   {
			   u(0) = x+pow(y,2);
			   u(1) = -y+pow(x,2);
			   break;
		   }
		   case 2:
		   {
			   break;
		   }
		   default:
		   {
			   break;
		   }
	   }
   }else{
	   switch (ctx.example){
		   case 0:
		   {
			   u(0) = 1.0;
			   u(1) = 1.0;
			   u(2) = 1.0;
			   break;
		   }
		   case 1:
		   {
			   u(0) = 2*x+pow(y,2)+pow(z,2);
			   u(1) = -y+pow(x,2)-2*z;
			   u(2) = -z+pow(x,2)+pow(y,2);
			   break;
		   }
		   case 2:
		   {
			   break;
		   }
		   default:
		   {
			   break;
		   }
	   }
   }
}

void w_exact(const Vector &xvec, Vector &w)
{
   double x = xvec(0);
   double y = xvec(1);
   double z = 0.0;
   int dim = xvec.Size();
   if (dim == 3) z = xvec(2);

   w=0.0;
   if (dim == 2){
	   switch (ctx.example){
		   case 0:
		   case 1:
		   {
			   w(0) = 1.0;
			   w(1) = 0.0;
			   break;
		   }
		   case 2:
		   {
			   u_exact(xvec, w);
			   break;
		   }
		   default:
		   {
			   break;
		   }
	   }
   }else{
	   switch (ctx.example){
		   case 0:
		   case 1:
		   {
			   w(0) = 1.0;
			   w(1) = 0.0;
			   w(2) = 0.0;
			   break;
		   }
		   case 2:
		   {
			   break;
		   }
		   default:
		   {
			   break;
		   }
	   }
   }
}

void sig_exact(const Vector &xvec, Vector &sig)
{
   double x = xvec(0);
   double y = xvec(1);
   double z = 0.0;
   int dim = xvec.Size();
   if (dim == 3) z = xvec(2);

   sig = 0.0;
   if(dim == 2){
	   switch (ctx.example){
		   case 1:
		   {
			   sig(0) = 1.0;
			   sig(1) = 2*y;
			   sig(2) = 2*x;
			   sig(3) = -1.0;
			   break;
		   }
		   case 2:
		   {
			   break;
		   }
		   case 0:
		   default:
		   {
			   break;
		   }
	   }
   }else{
	   switch (ctx.example){
		   case 1:
		   {
			  sig(0)= 2.0;
			  sig(1)= 2*y;
			  sig(2)= 2*z;
			  sig(3)= 2*x;
			  sig(4)= -1.0;
			  sig(5)= -2.0;
			  sig(6)= 2*x;
			  sig(7)= 2*y;
			  sig(8)= -1.0;
			   break;
		   }
		   case 2:
		   {
			   break;
		   }
		   case 0:
		   default:
		   {
			   break;
		   }
	   }
   }
   sig *= 1.0/ctx.Re;
}

void pseudo_traction_exact(const Vector &xvec, Vector &pseudo_traction)
{
	pseudo_traction = 0.0;
	// sigma - pI
	int dim = xvec.Size();
	Vector sig(dim*dim), pI(dim*dim);
	sig = 0.0; pI=0.0;
	sig_exact(xvec, sig);
	double p = p_exact(xvec);

	for (int d=0;d<dim;d++)
		pI(d*(dim+1)) = p; // diagonally assign

	pseudo_traction.Add(1.0,sig);
	pseudo_traction.Add(-1.0,pI);
}

void f_source(const Vector &xvec, Vector &f)
{
	double x = xvec(0);
	double y = xvec(1);
	double z = 0.0;
	int dim = xvec.Size();
	if (dim == 3) z = xvec(2);

	Vector sig(dim*dim);
	sig_exact(xvec, sig);
	Vector w(dim);
	w_exact(xvec, w);

	double Lap_u1 = 0.0;
	double Lap_u2 = 0.0;
	double Lap_u3 = 0.0;

	double dx_u1 = 0.0;
	double dy_u1 = 0.0;
	double dz_u1 = 0.0;

	double dx_u2 = 0.0;
	double dy_u2 = 0.0;
	double dz_u2 = 0.0;

	double dx_u3 = 0.0;
	double dy_u3 = 0.0;
	double dz_u3 = 0.0;

	double dx_p = 0.0;
	double dy_p = 0.0;
	double dz_p = 0.0;

	f = 0.0;
	if(dim == 2){
		dx_u1 = ctx.Re*sig(0);
		dy_u1 = ctx.Re*sig(1);
		dx_u2 = ctx.Re*sig(2);
		dy_u2 = ctx.Re*sig(3);

		switch (ctx.example){
		   case 1:
		   {
				Lap_u1 = 2.0;
				Lap_u2 = 2.0;

				dx_p = -2.0;
				dy_p = 0.0;
			   break;
		   }
		   case 2:
		   {
			   break;
		   }
		   case 0:
		   default:
		   {
			   break;
		   }
		}
	}else{
		dx_u1 = ctx.Re*sig(0);
		dy_u1 = ctx.Re*sig(1);
		dz_u1 = ctx.Re*sig(2);

		dx_u2 = ctx.Re*sig(3);
		dy_u2 = ctx.Re*sig(4);
		dz_u2 = ctx.Re*sig(5);

		dx_u3 = ctx.Re*sig(6);
		dy_u3 = ctx.Re*sig(7);
		dz_u3 = ctx.Re*sig(8);

		switch (ctx.example){
		   case 1:
		   {
				Lap_u1 = 4.0;
				Lap_u2 = 2.0;
				Lap_u3 = 4.0;

				dx_p = -2.0;
				dy_p = 0.0;
				dz_p = 0.0;
			   break;
		   }
		   case 2:
		   {
			   break;
		   }
		   case 0:
		   default:
		   {
			   break;
		   }
		}
	}

	if(dim == 2){
		f(0) = -(1.0/double(ctx.Re))*Lap_u1 + dx_p + (w(0) * dx_u1 + w(1) * dy_u1);
		f(1) = -(1.0/double(ctx.Re))*Lap_u2 + dy_p + (w(0) * dx_u2 + w(1) * dy_u2);
	}else{
		f(0) = -(1.0/double(ctx.Re))*Lap_u1 + dx_p + (w(0) * dx_u1 + w(1) * dy_u1 + w(2)*dz_u1);
		f(1) = -(1.0/double(ctx.Re))*Lap_u2 + dy_p + (w(0) * dx_u2 + w(1) * dy_u2 + w(2)*dz_u2);
		f(2) = -(1.0/double(ctx.Re))*Lap_u3 + dz_p + (w(0) * dx_u3 + w(1) * dy_u3 + w(2)*dz_u3);
	}
}

void u_init(const Vector &xvec, Vector &u)
{
	// start with the uniform flow with value of 1.
	u = 1.0;
}
