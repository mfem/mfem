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
class WaveOperator : public TimeDependentOperator
{
protected:
   FiniteElementSpace &fespace;
   Array<int> block_offsets;

   SparseMatrix Mm, Km_phi, Km_eta;
   BlockMatrix *Kb,  *Tb;

   GMRESSolver T_solver;
   BlockDiagonalPreconditioner *T_prec;
   Solver *PC0, *PC1;

   double c1,c2;
   double g,current_dt;
   mutable Vector z; // auxiliary vector

public:
   WaveOperator(FiniteElementSpace &f, Array<int> &fs_bdr, Coefficient *motion, double c1_ = 0.5, double c2_ = 0.5);
   virtual void Mult(const Vector &u, const Vector &dudt) const;
   /** Solve the Backward-Euler equation: k = f(u + dt*k, t), for the unknown k.
   This is the only requirement for high-order SDIRK implicit integration.*/
   virtual void ImplicitSolve(const double dt,const Vector &u, Vector &dudt);

   ///
   void SetParameters(const Vector &phi);

   virtual ~WaveOperator();
};

WaveOperator::WaveOperator(FiniteElementSpace &f, Array<int> &fs_bdr, Coefficient *motion, double c1_, double c2_)
   : TimeDependentOperator(2*f.GetTrueVSize(), 0.0), fespace(f),
     Tb(NULL),Kb(NULL),PC0(NULL),PC1(NULL),current_dt(0.0),z(height),g(9.81)
{
   c1 = c1_;
   c2 = c2_;

   if (c1+c2 < 1.0) mfem_error("Formulation coefficients to small");

   // Offset
   block_offsets.SetSize(3);
   block_offsets[0] = 0;
   block_offsets[1] = fespace.GetVSize();
   block_offsets[2] = 2 * fespace.GetVSize();

   // Laplace
   BilinearForm K(&fespace);
   K.AddDomainIntegrator(new DiffusionIntegrator());
   K.Assemble();
   K.Finalize();
   Km_phi = K.SpMat();
   Km_eta = K.SpMat();

   // Eliminate free-surface rows
   Array<int> tdof_list_fs;
   fespace.GetEssentialTrueDofs(fs_bdr, tdof_list_fs);
   for (int i = 0; i<tdof_list_fs.Size(); i++)
   {
      Km_eta.EliminateRow(tdof_list_fs[i]);
   }

   // Boundary mass --> at free-surface
   ConstantCoefficient one(1.0);
   BilinearForm M(&fespace);
   M.AddBdrFaceIntegrator(new BoundaryMassIntegrator(one), fs_bdr);
   M.Assemble();
   M.Finalize();
   Mm = M.SpMat();

   // Preconditioner
   T_prec = new BlockDiagonalPreconditioner(block_offsets);

   // Solver
   T_solver.iterative_mode = false;
   T_solver.SetRelTol(1e-18);
   T_solver.SetAbsTol(0.0);
   T_solver.SetMaxIter(200);
   T_solver.SetPrintLevel(0);
}

void WaveOperator::Mult(const Vector &u, const Vector &dudt)  const
{
   mfem_error("This wave formulation only works for implicit time integrators");
}

void WaveOperator::ImplicitSolve(const double dt, const Vector &u, Vector &dudt)
{
   if (dt != current_dt)
   {
      if (!Tb) delete Tb;
      if (!Kb) delete Kb;
      if (!PC0) delete PC0;
      if (!PC1) delete PC1;

      current_dt = dt;
      double alpha = 1.0/dt;
      Kb = new BlockMatrix(block_offsets);
      Kb->SetBlock(0, 0, &Km_phi);
      //Kb->SetBlock(0, 1, Scale(0.5*alpha, Mm));
      //Kb->SetBlock(1, 1, Scale(0.5*g, Mm)); 
      Kb->SetBlock(0, 1, Add(c2*alpha, Mm, 0.0, Mm));
      Kb->SetBlock(1, 1, Add(c1*g, Mm, 0.0, Mm));

      Tb = new BlockMatrix(block_offsets);
      Tb->SetBlock(0, 0, Add(dt, Km_phi, c2*alpha / g, Mm));
      Tb->SetBlock(0, 1, Add(-1.0, Mm, c2*dt*alpha, Mm));
      //Tb->SetBlock(1, 0, Scale(0.5, Mm));
      Tb->SetBlock(1, 0, Add(c1, Mm, 0.0, Mm));
      Tb->SetBlock(1, 1, Add(c1*dt*g, Mm, 1.0, Km_eta));

      PC0 = new DSmoother(Tb->GetBlock(0, 0));
      PC1 = new DSmoother(Tb->GetBlock(1, 1));
      T_prec->SetDiagonalBlock(0, PC0);
      T_prec->SetDiagonalBlock(1, PC1);

      T_solver.SetPreconditioner(*T_prec);
      T_solver.SetOperator(*Tb);
   }

   Kb->Mult(u, z);
   z.Neg();
   T_solver.Mult(z, dudt);
}

void WaveOperator::SetParameters(const Vector &u)
{

}

WaveOperator::~WaveOperator()
{
   if (!Tb) delete Tb;
   if (!Kb) delete Kb;
   if (!PC0) delete PC0;
   if (!PC1) delete PC1;
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

double Eta0(const Vector &x)
{
   return 0.0;
}

//----------------------------------------------
int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../../data/inline-quad.mesh";
   int ref_levels = 2;
   int order = 1;
   int ode_solver_type = 22;
   double t_final = 0.5;
   double dt = 1.0e-2;
   double c1 = 0.5;
   double c2 = 0.5;
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
                  "\t   22 - Midpoint, 23 - SDIRK23, 24 - SDIRK34, \n"
                  "\t   99 - Generalized alpha");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&c1, "-c1", "--c1",
                  "c1.");
   args.AddOption(&c2, "-c2", "--c2",
                  "c2.");
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
    ODESolver *ode_solver;
    switch (ode_solver_type)
    {
        // Implicit L-stable methods
        case 1:  ode_solver = new BackwardEulerSolver; break;
        case 2:  ode_solver = new SDIRK23Solver(2); break;
        case 3:  ode_solver = new SDIRK33Solver; break;

        // Implicit A-stable methods (not L-stable)
        case 22: ode_solver = new ImplicitMidpointSolver; break;
        case 23: ode_solver = new SDIRK23Solver; break;
        case 24: ode_solver = new SDIRK34Solver; break;
        default:
           cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
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

   // 8a. Initial conditions
   Coefficient *motion,*phi_0,*eta_0;
   
   if (strcmp(lib,"NONE") == 0)
   {
      motion = new ConstantCoefficient(0.0);
      phi_0 = new FunctionCoefficient(Phi0);
      eta_0 = new FunctionCoefficient(Eta0);
   }
   else
   {
      motion = new LibFunctionCoefficient(lib,"motion");
      phi_0 = new LibFunctionCoefficient(lib,"phi");
      eta_0 = new LibFunctionCoefficient(lib,"eta");
   }

   // 8b. Gridfunctions
   GridFunction phi_gf, eta_gf;
   Array<int> block_offsets(3);
   block_offsets[0] = 0;
   block_offsets[1] = fespace->GetVSize();
   block_offsets[2] = 2 * fespace->GetVSize();

   BlockVector uBlock(block_offsets);

   phi_gf.MakeRef(fespace, uBlock.GetBlock(0), 0);
   eta_gf.MakeRef(fespace, uBlock.GetBlock(1), 0);

   Project(*fespace,phi_gf,*phi_0);
   Project(*fespace,eta_gf,*eta_0);

   // 8c. Vectors
   Vector phi, eta;

   phi_gf.GetTrueDofs(phi);
   eta_gf.GetTrueDofs(eta);
   phi_gf.SetFromTrueDofs(phi);
   eta_gf.SetFromTrueDofs(eta);

   // 8d. Visualisation
   VisItDataCollection visit_dc("solution/lin-wave2", mesh);
   visit_dc.RegisterField("phi", &phi_gf);
   visit_dc.RegisterField("eta", &eta_gf);

   visit_dc.SetCycle(0);
   visit_dc.SetTime(0.0);
   visit_dc.Save();

   // 9. Perform time-integration (looping over the time iterations, ti, with a
   //    time-step dt). 
   WaveOperator oper(*fespace, fs_bdr, motion, c1,c2);
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
      ode_solver->Step(uBlock, t, dt);

      if (last_step || (ti % vis_steps) == 0)
      {
         cout << "step " << ti << ", t = " << t << endl;

         phi_gf.SetFromTrueDofs(phi);
         eta_gf.SetFromTrueDofs(eta);

         visit_dc.SetCycle(ti);
         visit_dc.SetTime(t);
         visit_dc.Save();
      }
      oper.SetParameters(uBlock); 
   }

   // 10. Free the used memory.
   delete ode_solver;
   delete mesh;
   delete phi_0,eta_0,motion;

   return 0;
}

