#include "petsc.h"
#include "petscvec.h"
#include "petscmat.h"
#include "petscmath.h"
#include "mfem.hpp"
#include <cstdio>
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

static  char help[]=
"This example demonstrates use of the TAO package to \n\
solve an obstacle problem defined using MFEM. Discretization \n\
is based on serial ex1.cpp from MFEM examples.\n";

// Define a ring-shaped obstacle function centered at the origin with radius 0.4
double RingObstacle(const Vector &x) 
{
  double t = 0.05;
  double r = pow(x(0), 2) + pow(x(1), 2);
  double rad = 0.4;
  double ul = pow(rad, 2);
  double ll = pow(rad-t, 2);
  if ((ll <= r) && (r <= ul)) {
    return 1.0;
  } else {
    return PETSC_NINFINITY;
  }
}

// Define a square shaped obstacle function centered at the origin with side length 0.8
double SquareObstacle(const Vector &x)
{
  double t = 0.05;
  double lim = 0.4;
  if (((x(0) <= lim) && (x(0) >= lim-t) && (x(1) <= lim) && (x(1) >= -lim)) ||
      ((x(0) >= -lim) && (x(0) <= -lim+t) && (x(1) <= lim) && (x(1) >= -lim)) ||
      ((x(1) <= lim) && (x(1) >= lim-t) && (x(0) <= lim) && (x(0) >= -lim)) ||
      ((x(1) >= -lim) && (x(1) <= -lim+t) && (x(0) <= lim) && (x(0) >= -lim))) {
    return 1.0;
  } else {
    return PETSC_NINFINITY;
  }
}

// Context that carries the necessary MFEM data structures inside TAO
typedef struct {
  SparseMatrix A;
  Vector U, LB, UB, work;
  vector <Vector> hist;
  int size;
  Mat H;
} AppCtx;

// TAO function call-back for computing the objective value and its gradient vector
PetscErrorCode FormFunctionGradient(Tao tao, Vec X, PetscReal *fcn,Vec G,void *ptr)
{
  AppCtx *user = (AppCtx*)ptr;
  PetscErrorCode ierr;
  double *data;
  const PetscReal *xx;
  PetscReal *gg;
  PetscReal gnorm;
  
  ierr = VecGetArrayRead(X, &xx);
  data = user->U.GetData();
  for (int i=0; i<user->size; ++i) data[i] = xx[i];
  ierr = VecRestoreArrayRead(X, &xx);
  
  *fcn = 0.5 * user->A.InnerProduct(user->U, user->U);
  
  user->A.Mult(user->U, user->work);
  ierr = VecGetArray(G, &gg);
  for (int i=0; i<user->size; ++i) gg[i] = user->work(i);
  ierr = VecRestoreArray(G, &gg);
  ierr = VecNorm(G, NORM_2, &gnorm);
  
  return 0;
}

// TAO Hessian call-back does nothing because we use a "matrix-free" Hessian shell
PetscErrorCode FormHessian(Tao tao,Vec X,Mat hes, Mat Hpre, void *ptr)
{
  return 0;
}

// User-defined TAO monitor that stores copies of the solution vector at each iteration
PetscErrorCode Monitor(Tao tao, void *ctx)
{
  AppCtx             *user = (AppCtx*) ctx;
  PetscErrorCode     ierr;
  PetscInt           its;
  PetscReal          f, gnorm, cnorm, xdiff;
  TaoConvergedReason reason;
  
  ierr = TaoGetSolutionStatus(tao, &its, &f, &gnorm, &cnorm, &xdiff, &reason);CHKERRQ(ierr);
  
  // store the history of the solutution
  Vector new_sol(user->U);
  user->hist.push_back(new_sol);
  
  return 0;
}

// "Matrix-free" Hessian-vector product function for the MFEM stiffness matrix
PetscErrorCode StiffMult(Mat A, Vec X, Vec Y)
{
  AppCtx *user;
  PetscErrorCode ierr;
  const double *xx;
  double *data, *yy;
  
  ierr = MatShellGetContext(A, &user);CHKERRQ(ierr);
  
  ierr = VecGetArrayRead(X, &xx);CHKERRQ(ierr);
  Vector xvec(user->work);
  data = xvec.GetData();
  for (int i=0; i<user->size; ++i) data[i] = xx[i];
  ierr = VecRestoreArrayRead(X, &xx);CHKERRQ(ierr);
  
  user->A.Mult(xvec, user->work);
  ierr = VecGetArray(Y, &yy);CHKERRQ(ierr);
  data = user->work.GetData();
  for (int i=0; i<user->size; ++i) yy[i] = data[i];
  ierr = VecRestoreArray(Y, &yy);CHKERRQ(ierr);
  
  return 0;
}

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../../../data/star.mesh";
   int order = 1;
   bool static_cond = false;
   bool visualization = 1;
   double fcn, prod_norm;
   PetscErrorCode ierr;
   AppCtx user;
   Tao tao;
   Vec X, XL, XU;
   PetscReal *bounds;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }   
   args.PrintOptions(cout);


   // 2. Read the mesh from the given mesh file.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 3. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 50,000
   //    elements.
   {
      int ref_levels =
         (int)floor(log(50000./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // 4. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order. If order < 1, we
   //    instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim);
   }
   else if (mesh->GetNodes())
   {
      fec = mesh->GetNodes()->OwnFEC();
      cout << "Using isoparametric FEs: " << fec->Name() << endl;
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim);
   }
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
   cout << "Number of finite element unknowns: "
        << fespace->GetTrueVSize() << endl;

   // 5. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking all
   //    the boundary attributes from the mesh as essential (Dirichlet) and
   //    converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   if (mesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 6. Define lower and upper bounds for the optimization problem as GridFunctions.
   //    Both bounds are zero at the boundary of the mesh. Lower bound includes 
   //    the obstacle function, while the upper bound is "infinity".
   ConstantCoefficient zero(0.0);
   FunctionCoefficient obs(RingObstacle);
   ConstantCoefficient inf(PETSC_INFINITY);
   
   GridFunction lb(fespace);
   lb.ProjectCoefficient(obs);
   lb.ProjectBdrCoefficient(zero, ess_tdof_list);
   user.LB = lb;
   
   GridFunction ub(fespace);
   ub.ProjectCoefficient(inf);
   ub.ProjectBdrCoefficient(zero, ess_tdof_list);
   user.UB = ub;

   // 7. Define the solution vector u as a finite element grid function
   //    corresponding to fespace. Also create a work vector here.
   GridFunction u(fespace);
   u = 0.0;
   user.U = u;
   GridFunction work(fespace);
   work = 0.0;
   user.work = work;

   // 8. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator Delta, by adding the Diffusion
   //    domain integrator.
   ConstantCoefficient one(1.0);
   BilinearForm *a = new BilinearForm(fespace);
   a->AddDomainIntegrator(new DiffusionIntegrator(one));
   a->Assemble();

   // 9. Extract the stiffness matrix from the bilinear form.
   a->FormSystemMatrix(ess_tdof_list, user.A);
   user.size = user.A.Height();
   cout << "Size of linear system: " << user.size << endl;
   
   // 10. Initialize PETSc and prepare the PETSc data structures
   ierr = PetscInitialize( &argc, &argv,(char *)0,help );if (ierr) return ierr;
   
   ierr = VecCreateSeq(PETSC_COMM_SELF, user.size, &X);CHKERRQ(ierr);
   ierr = VecSet(X, 1.0);CHKERRQ(ierr);
   
   ierr = VecDuplicate(X, &XL);CHKERRQ(ierr);
   ierr = VecGetArray(XL, &bounds);
   for (int i=0; i<user.size; ++i) bounds[i] = user.LB(i);
   ierr = VecGetArray(XL, &bounds);
   
   ierr = VecDuplicate(X, &XU);CHKERRQ(ierr);
   ierr = VecGetArray(XU, &bounds);
   for (int i=0; i<user.size; ++i) bounds[i] = user.UB(i);
   ierr = VecRestoreArray(XU, &bounds);
   
   // 11. Create the shell matrix that performs the "matrix-free" 
   //     Hessian-vector product required by TAO.
   ierr = MatCreateShell(PETSC_COMM_SELF, user.size, user.size, PETSC_DETERMINE, PETSC_DETERMINE, (void*) &user, &user.H);CHKERRQ(ierr);
   ierr = MatShellSetOperation(user.H, MATOP_MULT, (void(*)(void))StiffMult);CHKERRQ(ierr);
   
   if (visualization) {
     Vector init_sol(user.work);
     init_sol = 1.0;
     user.hist.push_back(init_sol);
   }
   
   // 12. Create the TAO optimization algorithm, configure it and start the solution
   ierr = TaoCreate(PETSC_COMM_WORLD, &tao);CHKERRQ(ierr);
   ierr = TaoSetType(tao, TAOBNLS);CHKERRQ(ierr);
   ierr = TaoSetInitialVector(tao, X);CHKERRQ(ierr);
   ierr = TaoSetObjectiveAndGradientRoutine(tao, FormFunctionGradient, (void*) &user);CHKERRQ(ierr);
   ierr = TaoSetHessianRoutine(tao, user.H, user.H, FormHessian, (void*) &user);CHKERRQ(ierr);
   ierr = TaoSetVariableBounds(tao, XL, XU);CHKERRQ(ierr);
   ierr = TaoSetMonitor(tao, Monitor, &user, NULL);CHKERRQ(ierr);
   ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);
   ierr = TaoSolve(tao);CHKERRQ(ierr);
   
   // 13. Recover the solution as a finite element grid function.
   if (visualization) {
     GridFunction xc(fespace);
     VisItDataCollection visit_dc("obstacle", mesh);
     visit_dc.RegisterField("solution", &xc);
     int vis_cycle = 0;
     
     int hist_len = user.hist.size();
     for (int i=0; i<hist_len; ++i) {
       LinearForm *rc = new LinearForm(fespace);
       rc->AddBoundaryIntegrator(new BoundaryLFIntegrator(zero));
       rc->Assemble();

       BilinearForm *c = new BilinearForm(fespace);
       c->AddDomainIntegrator(new DiffusionIntegrator(one));
       c->Assemble();

       SparseMatrix Cmat;
       Vector Xvec, Rvec;
       c->FormLinearSystem(ess_tdof_list, xc, *rc, Cmat, Xvec, Rvec);
       Xvec = user.hist[i];
       c->RecoverFEMSolution(Xvec, *rc, xc);
       
       visit_dc.SetCycle(vis_cycle++);
       visit_dc.SetTime(i);
       visit_dc.Save();
       
       delete c;
       delete rc;
     }
   }
   
   // 14. Clean up PETSc memory
   ierr = VecDestroy(&X);CHKERRQ(ierr);
   ierr = VecDestroy(&XU);CHKERRQ(ierr);
   ierr = VecDestroy(&XL);CHKERRQ(ierr);
   ierr = MatDestroy(&user.H);CHKERRQ(ierr);
   ierr = TaoDestroy(&tao);CHKERRQ(ierr);
   
   ierr = PetscFinalize();

   // 15. Clean up MFEM memory
   delete a;
   delete fespace;
   if (order > 0) { delete fec; }
   delete mesh;

   return 0;
}
