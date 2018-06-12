//                                MFEM Example 20
//
// Compile with: make ex20
//
// Sample runs:  ex20
//               ex20 -m ../data/inline-tri.mesh
//               ex20 -m ../data/disc-nurbs.mesh -r 3 -o 2 -tf 2
//
// Description:  This example solves the wave equation
//               problem of the form d^2u/dt^2 = c^2 \Delta u.
//
//               The example demonstrates the use of 2nd order time integration.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <dlfcn.h>

using namespace std;
using namespace mfem;

/** After spatial discretization, the conduction model can be written as:
 *
 *     d^2u/dt^2 = M^{-1}(-Ku)
 *
 *  where u is the vector representing the temperature, M is the mass matrix,
 *  and K is the diffusion operator with diffusivity depending on u:
 *  (\kappa + \alpha u).
 *
 *  Class WaveOperator represents the right-hand side of the above ODE.
 */
class WaveOperator : public TimeDependent2Operator
{
protected:
   FiniteElementSpace &fespace;


   LinearForm   *b;
   BilinearForm *M;
   BilinearForm *K;
   SparseMatrix *T; // T = M + dt K
   double current_fac0;

   CGSolver M_solver; // Krylov solver for inverting the mass matrix M
   DSmoother M_prec;  // Preconditioner for the mass matrix M

   CGSolver T_solver; // Implicit solver for T = M + fac0*K
   DSmoother T_prec;  // Preconditioner for the implicit solver

   Coefficient *oog;
   mutable Vector z; // auxiliary vector

public:
   WaveOperator(FiniteElementSpace &f, Array<int> &fs_bdr, Coefficient *motion);
   virtual void ExplicitSolve(const Vector &phi, const Vector &dphidt,
                              Vector &d2phidt2) const;
   /** Solve the Backward-Euler equation:
       d2udt2 = f(u + fac0*d2udt2,dudt + fac1*d2udt2, t), for the unknown d2udt2.*/
   virtual void ImplicitSolve(const double fac0, const double fac1,
                              const Vector &phi, const Vector &dphidt, Vector &d2phidt2);

   ///
   void SetParameters(const Vector &phi);

   virtual ~WaveOperator();
};


WaveOperator::WaveOperator(FiniteElementSpace &f, Array<int> &fs_bdr, Coefficient *motion)
   : TimeDependent2Operator(f.GetTrueVSize(), 0.0), fespace(f), M(NULL), K(NULL),
     T(NULL), current_fac0(0.0), z(height)
{

   // Laplace
   K = new BilinearForm(&fespace);
   K->AddDomainIntegrator(new DiffusionIntegrator());
   K->Assemble();
   K->Finalize();

   // Boundary mass --> at free-surface
   oog = new ConstantCoefficient(1.0/9.81);
   M = new BilinearForm(&fespace);
   M->AddBdrFaceIntegrator(new BoundaryMassIntegrator(*oog), fs_bdr);
   M->Assemble();
   M->Finalize();

  // b = new LinearForm(&fespace);
  // b->AddBoundaryIntegrator(new BoundaryLFIntegrator(*motion));

   T_solver.iterative_mode = false;
   T_solver.SetRelTol(1e-18);
   T_solver.SetAbsTol(0.0);
   T_solver.SetMaxIter(200);
   T_solver.SetPrintLevel(0);
   T_solver.SetPreconditioner(T_prec);

}

void WaveOperator::ExplicitSolve(const Vector &phi, const Vector &dphidt,
                                 Vector &d2phidt2)  const
{
   mfem_error("This wave formulation only works for implicit time integrators");
}

void WaveOperator::ImplicitSolve(const double fac0, const double fac1,
                                 const Vector &phi, const Vector &dphidt, Vector &d2phidt2)
{
   // Solve the equation:
   //  M*d2udt2 + K(u + fac0*d2udt2) = 0
   //  [ M + fac0*K ] d2udt2 = - K*u
   // for d2udt2

   //      motion.SetTime(t);


   if (fac0 != current_fac0)
   {
      if (!T) delete T;
      current_fac0 = fac0;
      T = Add(1.0, M->SpMat(), fac0, K->SpMat());
      T_solver.SetOperator(*T);
   }
  // b->Assemble();
   K->Mult(phi, z);
  // z += *b;
   z.Neg();
   T_solver.Mult(z, d2phidt2);
}

void WaveOperator::SetParameters(const Vector &phi)
{
}

WaveOperator::~WaveOperator()
{
   if (!T) delete T;
   delete M;
   delete K;
   delete oog;
}
//----------------------------------------------
class LibFunctionCoefficient : public Coefficient
{
protected:
   typedef double (*TDFunPtr)(double *, int, double);
   TDFunPtr TDFunction;
   void *libHandle;

public:
   /// Define a time-independent coefficient from a C-library
   LibFunctionCoefficient(std::string libName, std::string funName)
   {
      libHandle = dlopen (libName.c_str(), RTLD_LAZY);
      if (!libHandle)
      {
         std::cout <<libName<<"  "<<funName<<std::endl;
         mfem_error("Lib not found.\n");
      }

      TDFunction = (TDFunPtr)dlsym(libHandle, funName.c_str());

      if (!TDFunction) { mfem_error("Function not found.\n"); }
   };

   /// Evaluate coefficient
   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
   {
      double x[3];
      Vector transip(x, 3);
      T.Transform(ip, transip);
      return ((*TDFunction)(transip.GetData(),transip.Size(),GetTime()));
   };

   /// Destructor
   ~LibFunctionCoefficient() { dlclose(libHandle); };
};

//----------------------------------------------
void Project(FiniteElementSpace &fespace, GridFunction &gf, Coefficient &coeff)
{
   LinearForm *b = new LinearForm(&fespace);
   b->AddDomainIntegrator(new DomainLFIntegrator(coeff));
   b->Assemble();

   BilinearForm *M = new BilinearForm(&fespace);
   M->AddDomainIntegrator(new MassIntegrator());
   M->Assemble();
   M->Finalize();

   CGSolver M_solver; // Krylov solver for inverting the mass matrix M
   DSmoother M_prec;  // Preconditioner for the mass matrix M

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-17);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(130);
   M_solver.SetPrintLevel(0);
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(M->SpMat());
   M_solver.Mult(*b, gf);
}

double Phi0(const Vector &x)
{
   return 0.001*cos(2*M_PI*x[0])*cosh(2*M_PI*x[1]) ;
}

//----------------------------------------------
int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../../data/inline-quad.mesh";
   int ref_levels = 2;
   int order = 1;
   int ode_solver_type = 10;
   double t_final = 0.5;
   double dt = 1.0e-2;
   double speed = 1.0/9.81;
   bool visit = true;
   int vis_steps = 5;
   Array<int> fs_idx;
   const char *lib = "NONE";

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Backward Euler, 2 - SDIRK2, 3 - SDIRK3,\n\t"
                  "\t   11 - Forward Euler, 12 - RK2, 13 - RK3 SSP, 14 - RK4, \n"
                  "\t   99 - Generalized alpha");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&fs_idx, "-f", "--free-surface",
                  "Free-surface mesh index.");
   args.AddOption(&lib, "-l", "--lib",
                  "Library.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral and hexahedral meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   if (dim == 1)
   {
      cout << "Not for 1 dimensional meshes." << endl;
      return 2;
   }

   // 3. Define the ODE solver used for time integration. Several implicit
   //    singly diagonal implicit Runge-Kutta (SDIRK) methods, as well as
   //    explicit Runge-Kutta methods are available.
   ODE2Solver *ode_solver;
   switch (ode_solver_type)
   {
      // Implicit methods
      case 0: ode_solver = new GeneralizedAlpha2Solver(0.0); break;
      case 1: ode_solver = new GeneralizedAlpha2Solver(0.1); break;
      case 2: ode_solver = new GeneralizedAlpha2Solver(0.2); break;
      case 3: ode_solver = new GeneralizedAlpha2Solver(0.3); break;
      case 4: ode_solver = new GeneralizedAlpha2Solver(0.4); break;
      case 5: ode_solver = new GeneralizedAlpha2Solver(0.5); break;
      case 6: ode_solver = new GeneralizedAlpha2Solver(0.6); break;
      case 7: ode_solver = new GeneralizedAlpha2Solver(0.7); break;
      case 8: ode_solver = new GeneralizedAlpha2Solver(0.8); break;
      case 9: ode_solver = new GeneralizedAlpha2Solver(0.9); break;
      case 10: ode_solver = new GeneralizedAlpha2Solver(1.0); break;

      case 11: ode_solver = new AverageAccelerationSolver(); break;
      case 12: ode_solver = new LinearAccelerationSolver(); break;
      case 13: ode_solver = new CentralDifferenceSolver(); break;
      case 14: ode_solver = new FoxGoodwinSolver(); break;

      default:
         cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         delete mesh;
         return 3;
   }

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement, where 'ref_levels' is a
   //    command-line parameter.
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   // 5. Define the vector finite element space representing the current and the
   //    initial temperature, u_ref.
   FiniteElementCollection *fec;
   NURBSExtension *NURBSext = NULL;
   int own_fec = 0;

   if (order == -1) // Isoparametric
   {
      if (mesh->GetNodes())
      {
         fec = mesh->GetNodes()->OwnFEC();
         own_fec = 0;
         cout << "Using isoparametric FEs: " << fec->Name() << endl;
      }
      else
      {
         cout <<"Mesh does not have FEs --> Assume order 1.\n";
         fec = new H1_FECollection(1, dim);
         own_fec = 1;
      }
   }
   else if (mesh->NURBSext && (order > 0) )  // Subparametric NURBS
   {
      fec = new NURBSFECollection(order);
      own_fec = 1;
      int nkv = mesh->NURBSext->GetNKV();

      Array<int> edgeOrder(nkv);
      edgeOrder = order;

      NURBSext = new NURBSExtension(mesh->NURBSext, edgeOrder);
   }
   else
   {
      fec = new H1_FECollection(order, dim);
      own_fec = 1;
   }
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, NURBSext, fec);
   cout << "Number of finite element unknowns: "
        << fespace->GetTrueVSize() << endl;

   // 7. Set the boundary conditions.

   if (mesh->bdr_attributes.Size() == 0)
   {
      cout << "No boundary data in mesh!"<<endl;
      return 4;
   }

   if (fs_idx.Size() == 0)
   {
      cout << "No free-surface index in input"<<endl;
      args.PrintUsage(cout);
      return 5;
   }

   Array<int> fs_bdr(mesh->bdr_attributes.Max());
   fs_bdr = 0;

   for (int i = 0; i < fs_idx.Size(); i++)
   {
      int idx = mesh->bdr_attributes.Find(fs_idx[i]);

      if (idx ==  -1)
      {
         cout << "\nIncorrect Free-surface index: "<<fs_idx[i]<<endl;
         cout << "\nPossible options are: "<<endl;
         mesh->bdr_attributes.Print(cout);
         return 6;
      }

      fs_bdr[idx] = 1;
   }

   // 8a. Initialize the conduction operator and the visualization.
   Coefficient *motion,*phi_0,*dphidt_0;

   if (strcmp(lib,"NONE") == 0)
   {
      motion = new ConstantCoefficient(0.0);
      phi_0 = new FunctionCoefficient(Phi0);
      dphidt_0 = new ConstantCoefficient(0.0);
   }
   else
   {
      motion = new LibFunctionCoefficient(lib ,"motion");
      phi_0 = new LibFunctionCoefficient(lib ,"phi");
      dphidt_0 = new LibFunctionCoefficient(lib ,"dphidt");
   }

   // 8b.
   GridFunction phi_gf(fespace);
   GridFunction dphidt_gf(fespace);

   Project(*fespace,phi_gf,*phi_0);
   Project(*fespace,dphidt_gf,*dphidt_0);

   // 8c. 
   Vector phi, dphidt;

   dphidt_gf.GetTrueDofs(dphidt);
   phi_gf.GetTrueDofs(phi);
   phi_gf.SetFromTrueDofs(phi);

   // 8d. Visualisation
   VisItDataCollection visit_dc("solution/lin-wave1", mesh);
   visit_dc.RegisterField("phi", &phi_gf);
   visit_dc.RegisterField("dphidt", &dphidt_gf);

   visit_dc.SetCycle(0);
   visit_dc.SetTime(0.0);
   visit_dc.Save();

   // 9. Perform time-integration (looping over the time iterations, ti, with a
   //    time-step dt). 
   WaveOperator oper(*fespace, fs_bdr, motion);
   ode_solver->Init(oper);
   double t = 0.0;

   bool last_step = false;
   for (int ti = 1; !last_step; ti++)
   {
      if (t + dt >= t_final - dt/2)
      {
         last_step = true;
      }
      motion->SetTime(t);
      ode_solver->Step(phi, dphidt, t, dt);

      if (last_step || (ti % vis_steps) == 0)
      {
         cout << "step " << ti << ", t = " << t << endl;

         phi_gf.SetFromTrueDofs(phi);
         dphidt_gf.SetFromTrueDofs(dphidt);

         visit_dc.SetCycle(ti);
         visit_dc.SetTime(t);
         visit_dc.Save();

      }
      oper.SetParameters(phi); // dudt???
   }

   // 10. Free the used memory.
   delete ode_solver;
   delete mesh;
   delete phi_0,dphidt_0,motion;

   return 0;
}

