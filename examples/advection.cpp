
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;


void Prefine(FiniteElementSpace & fes_old,
             GridFunction &u, Coefficient &gf_ex, GridFunction &orders_gf,
             double min_thresh, double max_thresh);


// Velocity coefficient
void velocity_function(const Vector &x, Vector &v);

// Initial condition
double u0_function(const Vector &x, double);

// Inflow boundary condition
double inflow_function(const Vector &x);

// Mesh bounding box
Vector bb_min, bb_max;

class FE_Evolution : public TimeDependentOperator
{
private:
   BilinearForm &M, &K;
   const Vector &b;
   Solver *M_prec;
   CGSolver M_solver;

   mutable Vector z;

public:
   FE_Evolution(BilinearForm &M_, BilinearForm &K_, const Vector &b_);
   void Update();
   virtual void Mult(const Vector &x, Vector &y) const;

   virtual ~FE_Evolution();
};


int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/periodic-hexagon.mesh";
   int ref_levels = 2;
   int order = 1;
   double t_final = 10.0;
   double dt = 0.0005;
   bool visualization = true;
   int vis_steps = 5;

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   Mesh mesh0 = Mesh::MakeCartesian2D(64, 1, mfem::Element::QUADRILATERAL,false, 2,
                                      1);


   std::vector<Vector> translations = {Vector({2.0,0.0}), };


   Mesh mesh = Mesh::MakePeriodic(mesh0,
                                  mesh0.CreatePeriodicVertexMapping(translations));


   mesh.EnsureNCMesh();

   int dim = mesh.Dimension();

   ODESolver *ode_solver = new RK4Solver;


   mesh.GetBoundingBox(bb_min, bb_max, max(order, 1));

   // 5. Define the discontinuous DG finite element space of the given
   //    polynomial order on the refined mesh.
   DG_FECollection fec(order, dim, BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec);
   FiniteElementSpace fes_old(&mesh, &fec);

   cout << "Number of unknowns: " << fes.GetVSize() << endl;

   VectorFunctionCoefficient velocity(dim, velocity_function);
   FunctionCoefficient inflow(inflow_function);
   FunctionCoefficient u0(u0_function);

   BilinearForm m(&fes);
   BilinearForm k(&fes);
   m.AddDomainIntegrator(new MassIntegrator);
   constexpr double alpha = -1.0;
   k.AddDomainIntegrator(new ConvectionIntegrator(velocity, alpha));
   k.AddInteriorFaceIntegrator(
      new NonconservativeDGTraceIntegrator(velocity, alpha));
   k.AddBdrFaceIntegrator(
      new NonconservativeDGTraceIntegrator(velocity, alpha));

   LinearForm b(&fes);
   b.AddBdrFaceIntegrator(
      new BoundaryFlowIntegrator(inflow, velocity, alpha,-0.5));

   m.Assemble();
   int skip_zeros = 0;
   k.Assemble(skip_zeros);
   b.Assemble();
   m.Finalize();
   k.Finalize(skip_zeros);

   // 7. Define the initial conditions, save the corresponding grid function to
   //    a file and (optionally) save data in the VisIt format and initialize
   //    GLVis visualization.
   GridFunction u(&fes);
   u0.SetTime(0.);
   u.ProjectCoefficient(u0);

   L2_FECollection orders_fec(0,dim);
   FiniteElementSpace orders_fes(&mesh,&orders_fec);
   GridFunction orders_gf(&orders_fes);
   for (int i = 0; i<mesh.GetNE(); i++) { orders_gf(i) = order; }

   socketstream sout;
   socketstream meshout;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      sout.open(vishost, visport);
      meshout.open(vishost, visport);
      if (!sout)
      {
         cout << "Unable to connect to GLVis server at "
              << vishost << ':' << visport << endl;
         visualization = false;
         cout << "GLVis visualization disabled.\n";
      }
      else
      {
         sout.precision(precision);
         sout << "solution\n" << mesh << u;
         sout << flush;
         meshout.precision(precision);
         meshout << "solution\n" << mesh << orders_gf;
         meshout << flush;
      }
   }


   // 8. Define the time-dependent evolution operator describing the ODE
   //    right-hand side, and perform time-integration (looping over the time
   //    iterations, ti, with a time-step dt).
   FE_Evolution adv(m, k, b);

   double t = 0.0;
   adv.SetTime(t);

   GridFunction gf_ex(&fes);
   FunctionCoefficient u_ex(u0_function);

   ode_solver->Init(adv);

   bool done = false;
   for (int ti = 0; !done; )
   {
      double dt_real = min(dt, t_final - t);

      ode_solver->Step(u, t, dt_real);
      ti++;

      done = (t >= t_final - 1e-8*dt);

      if (done || ti % vis_steps == 0)
      {
         cout << "time step: " << ti << ", time: " << t << endl;
         u_ex.SetTime(t);
         Prefine(fes_old,u,u_ex, orders_gf, 5e-5, 5e-4);
         m.Update();
         m.Assemble();
         m.Finalize();
         k.Update();
         k.Assemble(skip_zeros);
         k.Finalize(skip_zeros);
         b.Update();
         b.Assemble();
         adv.Update();
         ode_solver->Init(adv);
         if (visualization)
         {
            GridFunction * pr_u = ProlongToMaxOrder(&u);
            sout << "solution\n" << mesh << *pr_u << flush;
            meshout << "solution\n" << mesh << orders_gf << flush;
         }
      }
   }

   // 10. Free the used memory.
   delete ode_solver;

   return 0;
}


// Implementation of class FE_Evolution
FE_Evolution::FE_Evolution(BilinearForm &M_, BilinearForm &K_, const Vector &b_)
   : TimeDependentOperator(M_.Height()), M(M_), K(K_), b(b_), z(M_.Height())
{
   Array<int> ess_tdof_list;
   M_prec = new OperatorJacobiSmoother(M, ess_tdof_list);
   M_solver.SetOperator(M);
   M_solver.SetPreconditioner(*M_prec);
   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-9);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(100);
   M_solver.SetPrintLevel(0);
}

void FE_Evolution::Update()
{
   height = M.Height();
   width = M.Width();
   z.SetSize(M.Height());

   Array<int> ess_tdof_list;
   delete M_prec;
   M_prec = new OperatorJacobiSmoother(M, ess_tdof_list);
   M_solver.SetOperator(M);
   M_solver.SetPreconditioner(*M_prec);
   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-9);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(100);
   M_solver.SetPrintLevel(0);
}

void FE_Evolution::Mult(const Vector &x, Vector &y) const
{
   // y = M^{-1} (K x + b)
   K.Mult(x, z);
   z += b;
   M_solver.Mult(z, y);
}

FE_Evolution::~FE_Evolution()
{
   delete M_prec;
}

// Velocity coefficient
void velocity_function(const Vector &x, Vector &v)
{
   v.SetSize(2);
   v(0) = 1.;
   v(1) = 0.;
}

// Initial condition
double u0_function(const Vector &x, double t)
{
   // give x0, y0;
   double x0 = 0.5;
   // double y0 = 0.5;
   double w = 100.;
   double c = 1.;
   double ds = c*t;

   double xx = x(0) - ds;
   double yy = x(1) - ds;

   double tol = 1e-6;
   if (xx>= 2.0+tol || xx<= 0.0-tol)
   {
      xx -= (int)xx;
   }
   if (yy>= 1.0+tol || yy<= 0.0-tol)
   {
      yy -= (int)yy;
   }

   double dr2 = (xx-x0)*(xx-x0);
   return 1. + exp(-w*dr2);
}

// Inflow boundary condition (zero for the problems considered in this example)
double inflow_function(const Vector &x)
{
   return 1.0;
}


void Prefine(FiniteElementSpace & fes_old,
             GridFunction &u, Coefficient &ex, GridFunction &orders_gf,
             double min_thresh, double max_thresh)
{
   // get element errors
   FiniteElementSpace * fes = u.FESpace();
   int ne = fes->GetMesh()->GetNE();
   Vector errors(ne);
   u.ComputeElementL2Errors(ex,errors);
   for (int i = 0; i<ne; i++)
   {
      double error = errors(i);
      int order = fes->GetElementOrder(i);
      if (error < min_thresh && order > 1)
      {
         fes->SetElementOrder(i,order-1);
      }
      else if (error > max_thresh && order < 2)
      {
         fes->SetElementOrder(i, order+1);
      }
      else
      {
         // do nothing
      }
   }

   fes->Update(false);

   PRefinementTransferOperator * T = new PRefinementTransferOperator(fes_old,*fes);

   GridFunction u_fine(fes);
   T->Mult(u,u_fine);

   // copy the orders to the old space
   for (int i = 0; i<ne; i++)
   {
      int order = fes->GetElementOrder(i);
      fes_old.SetElementOrder(i,order);
      orders_gf(i) = order;
   }
   fes_old.Update(false);

   delete T;

   // update old gridfuntion;
   u = u_fine;

}
