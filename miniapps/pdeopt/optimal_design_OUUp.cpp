//                                Solution of distributed control problem
//
// Compile with: make optimal_design_OUU
//
// Sample runs:
//    optimal_design_OUU -r 3
//    optimal_design_OUU -m ../../data/star.mesh -r 3
//    optimal_design_OUU -sl 1 -m ../../data/mobius-strip.mesh -r 4
//    optimal_design_OUU -m ../../data/star.mesh -sl 5 -r 3 -mf 0.5 -o 5 -max 0.75
//
// Description:  This examples solves the following PDE-constrained
//               optimization problem:
//
//         max J(K) = E[ (f,u) ]
//
//         subject to   - div( K\nabla u ) = f    in \Omega a.s.
//                                       u = 0    on \partial\Omega
//         and            \int_\Omega K dx <= V vol(\Omega)
//         and            a <= K(x) <= b
//
//   Joachim Peterson 1999 for proof

#include "mfem.hpp"
#include <memory>
#include <iostream>
#include <fstream>
#include <random>
#include <time.h>

using namespace std;
using namespace mfem;

/** The Lagrangian for this problem is
 *    
 *    L(u,K,p) = E [ (f,u) ] - E [ (K \nabla u, \nabla p) + (f,p) ]
 * 
 *      u, p \in H^1_0(\Omega)
 *      K \in L^\infty(\Omega)
 * 
 *  Note that
 * 
 *    \partial_p L = 0        (1)
 *  
 *  delivers the state equation
 *    
 *    (\nabla u, \nabla v) = (f,v)  for all v in H^1_0(\Omega)
 * 
 *  and
 *  
 *    \partial_u L = 0        (2)
 * 
 *  delivers the adjoint equation (same as the state eqn)
 * 
 *    (\nabla p, \nabla v) = (f,v)  for all v in H^1_0(\Omega)
 *    
 *  and at the solutions u=p of (1) and (2), respectively,
 * 
 *  D_K J = D_K L = \partial_u L \partial_K u + \partial_p L \partial_K p
 *                + \partial_K L
*                = \partial_K L
 *                = (|\nabla u|^2, \cdot)
 * 
 * We update the control f_k with projected gradient descent via
 * 
 *  K_{k+1} = P_2 ( P_1 ( K_k - \gamma |\nabla u|^2 ) )
 * 
 * where P_1 is the projection operator enforcing (K,1) <= V, P_2 is the
 * projection operator enforcing a <= u(x) <= b, and \gamma is a specified
 * step length.
 * 
 */

double load(const Vector & x)
{
   double x1 = x(0);
   double x2 = x(1);
   double r = sqrt(x1*x1 + x2*x2);
   if (r <= 0.5)
   {
      return 1.0;
   }
   else
   {
      return 0.0;
   }
   // if (x1 < 0)
   // {
   //    return 1.0;
   // }
   // else
   // {
   //    return 0.0;
   // }
}

double damage_function(const Vector & x, double x1, double y1)
{
    double r1 = x(0) - x1;
    double r2 = x(1) - y1;
    double r = sqrt(r1*r1 + r2*r2);
    if (r <= 0.1)
    {
        return 1e-1;
      //   return 1.0;
    }
    else
    {
        return 1.0;
    }
}

class RandomFunctionCoefficient : public Coefficient
{
private:
   double a = 0.0;
   double b = 1.0;
   double x1, y1;
   double (*Function)(const Vector &, double, double);
public:
   RandomFunctionCoefficient(double (*F)(const Vector &, double, double), int seed = 0) 
   : Function(F)
   {
      srand((unsigned)seed);
      const double max = (double)(RAND_MAX) + 1.;
      rand();
      x1 = std::abs(rand()/max) * (b-a) + a;
      y1 = std::abs(rand()/max) * (b-a) + a;
   }
   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
   {
      Vector transip(3);
      T.Transform(ip, transip);
      return ((*Function)(transip, x1, y1));
   }
   void resample(int seed = 0)
   {
      srand((unsigned)seed);
      const double max = (double)(RAND_MAX) + 1.;
      rand();
      x1 = std::abs(rand()/max) * (b-a) + a;
      y1 = std::abs(rand()/max) * (b-a) + a;
      cout << "x1, y1 = " << x1 << ", " << y1 << endl; 
   }
};


class PoissonSolver
{
private:
   int myid;
   Mesh * mesh = nullptr;
   int order = 1;
   Array<int> ess_tdof_list;
   BilinearForm * a = nullptr;
   LinearForm *b = nullptr;
   OperatorPtr A;
   Vector B, C, X;
   RandomFunctionCoefficient * damage_coeff = nullptr;
   FiniteElementCollection * state_fec = nullptr;
   FiniteElementSpace * state_fes = nullptr;
   FiniteElementCollection * control_fec = nullptr;
   FiniteElementSpace * control_fes = nullptr;
   GridFunction * u = nullptr;
   GridFunction * p = nullptr;
   GridFunction * K = nullptr;
   GridFunction * K_old = nullptr;
   GridFunction * grad = nullptr;
   FunctionCoefficient * f = nullptr;
   ConstantCoefficient *zero = nullptr;
   ConstantCoefficient *one = nullptr;
   double norm = 0.;
   double compliance = 0.;
public:
   PoissonSolver(int myid_,Mesh * mesh_, int order_, RandomFunctionCoefficient * damage_coeff_) :
   myid(myid_),mesh(mesh_), order(order_), damage_coeff(damage_coeff_) { }

   void Setup()
   {
      int dim = mesh->Dimension();
      state_fec = new H1_FECollection(order,dim);
      state_fes = new FiniteElementSpace(mesh,state_fec);
      control_fec = new L2_FECollection(order-1,dim, BasisType::Positive);
      control_fes = new FiniteElementSpace(mesh,control_fec);
      f = new FunctionCoefficient(load);
      zero = new ConstantCoefficient(0.0);
      one = new ConstantCoefficient(1.0);
      K = new GridFunction(control_fes);     *K = 1.0;
      K_old = new GridFunction(control_fes); *K_old = 0.0;
      u = new GridFunction(state_fes); *u = 0.0;
      p = new GridFunction(state_fes); *p = 0.0;

      b = new LinearForm(state_fes);
      b->AddDomainIntegrator(new DomainLFIntegrator(*f));
      b->Assemble();

      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 1;
      state_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   void Solve()
   {

      damage_coeff->resample((int)time(0) + myid);
      delete a;
      a = new BilinearForm(state_fes);
      GridFunctionCoefficient diffusion_coeff(K);
      ProductCoefficient damaged_diffusion_coeff(diffusion_coeff,*damage_coeff);
      a->AddDomainIntegrator(new DiffusionIntegrator(damaged_diffusion_coeff));
      a->Assemble();
      a->FormLinearSystem(ess_tdof_list, *u, *b, A, X, B);

      GSSmoother M((SparseMatrix&)(*A));
      PCG(*A, M, B, X, 0, 200, 1e-12, 0.0);
      a->RecoverFEMSolution(X, *b, *u);
   }

   const GridFunction * GetGrad()
   {
      delete grad; grad = nullptr;
      GradientGridFunctionCoefficient grad_u(u);
      InnerProductCoefficient norm2_grad_u(grad_u,grad_u);
      grad = new GridFunction(control_fes);
      grad->ProjectCoefficient(norm2_grad_u);
      return grad;
   }
   const GridFunction * GetStateSol()
   {
      return u;
   }

   GridFunction * GetDiffusionCoeff()
   {
      return K;
   }

   double ComputeNormAndUpdate(double step_length)
   {
      *K_old -= *K;
      norm = K_old->ComputeL2Error(*zero)/step_length;
      *K_old = *K;
      return norm;
   }

   double GetCompliance()
   {
      compliance = (*b)(*u);
      return compliance;
   }

   ~PoissonSolver()
   {
      delete a;
      delete b;
      delete state_fec;
      delete state_fes;
      delete control_fec;
      delete control_fes;
      delete u;
      delete p;
      delete K;
      delete K_old;
      delete grad;
      delete f;
      delete zero;
      delete one;
   }
};


int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   MPI_Session mpi;
   int num_procs = mpi.WorldSize();
   int myid = mpi.WorldRank();

   srand((unsigned) time(NULL) + myid);


   const char *mesh_file = "../../data/inline-quad.mesh";
   int ref_levels = 2;
   int order = 2;
   bool visualization = true;
   double step_length = 1.0;
   double mass_fraction = 0.5;
   int max_it = 1e3;
   double tol = 1e-6;
   double K_max = 0.9;
   double K_min = 1e-3;
   int batch_size = 5;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&step_length, "-sl", "--step-length",
                  "Step length for gradient descent.");
   args.AddOption(&batch_size, "-bs", "--batch-size",
                  "batch size for stochastic gradient descent.");               
   args.AddOption(&max_it, "-mi", "--max-it",
                  "Maximum number of gradient descent iterations.");
   args.AddOption(&mass_fraction, "-mf", "--mass-fraction",
                  "Mass fraction for diffusion coefficient.");
   args.AddOption(&K_max, "-max", "--K-max",
                  "Maximum of diffusion diffusion coefficient.");
   args.AddOption(&K_min, "-min", "--K-min",
                  "Minimum of diffusion diffusion coefficient.");
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
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral and hexahedral meshes with the same code.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh.UniformRefinement();
   }

   H1_FECollection state_fec(order, dim);
   L2_FECollection control_fec(order-1, dim, BasisType::Positive);
   FiniteElementSpace state_fes(&mesh, &state_fec);
   FiniteElementSpace control_fes(&mesh, &control_fec);

   int state_size = state_fes.GetTrueVSize();
   int control_size = control_fes.GetTrueVSize();
   if (myid == 0)
   {
      cout << "Number of state unknowns: " << state_size << endl;
      cout << "Number of control unknowns: " << control_size << endl;
   }

   
   // 10. Define some tools for later
   ConstantCoefficient zero(0.0);
   ConstantCoefficient one(1.0);
   GridFunction onegf(&control_fes);
   onegf = 1.0;
   LinearForm vol_form(&control_fes);
   vol_form.AddDomainIntegrator(new DomainLFIntegrator(one));
   vol_form.Assemble();
   double domain_volume = vol_form(onegf);

   // 11. Connect to GLVis. Prepare for VisIt output.
   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sout_u,sout_p,sout_K;
   if (visualization)
   {
      if (myid == 0)
      {
         sout_u.open(vishost, visport);
         sout_K.open(vishost, visport);
         sout_u.precision(8);
         sout_K.precision(8);
      }
   }



   int seed = (int)time(0) + myid;
   RandomFunctionCoefficient damage_coeff(damage_function, seed);
   GridFunction avg_grad(&control_fes);

   int global_adaptive_batch_size = batch_size;
   int adaptive_batch_size = global_adaptive_batch_size/num_procs;
   // for convinience keep the batch size the same on each proc;
   global_adaptive_batch_size = num_procs * adaptive_batch_size;
   double theta = 2.5;

   PoissonSolver * psolver = new PoissonSolver(myid,&mesh, order, &damage_coeff);
   psolver->Setup();
   for (int k = 1; k <= max_it; k++)
   {
      // cout << "Step = " << k << endl;
      // cout << "batch_size = " << adaptive_batch_size << endl;
      avg_grad = 0.0;
      double grad_norm = 0.;

      for (int ib = 0; ib<adaptive_batch_size; ib++)
      {
         psolver->Solve();
         if (visualization)
         {
            if (myid == 0)
            {
               sout_u << "solution\n" << mesh << *psolver->GetStateSol()
                   << "window_title 'State u'" << flush;
            }
         }

         avg_grad += *psolver->GetGrad();
         grad_norm += pow(psolver->GetGrad()->ComputeL2Error(zero),2);  
      }

      // MPI reduction
      MPI_Allreduce(MPI_IN_PLACE, avg_grad.GetData(), avg_grad.Size(), 
                    MPI_DOUBLE, MPI_SUM,MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &grad_norm, 1, 
                    MPI_DOUBLE, MPI_SUM,MPI_COMM_WORLD);      


      grad_norm /= (double)global_adaptive_batch_size;  
      avg_grad /= (double)global_adaptive_batch_size;

      double avg_grad_norm = pow(avg_grad.ComputeL2Error(zero),2);

      cout << "avg_grad_norm = " << avg_grad_norm << endl;
      cout << "grad_norm = " << grad_norm << endl;

      double variance = (grad_norm - avg_grad_norm)/(global_adaptive_batch_size - 1);  

      // J. Update control.
      avg_grad *= step_length;

      GridFunction & K = *psolver->GetDiffusionCoeff();

      K += avg_grad;

      // K. Project onto constraint set (optimality criteria)
      double mass = vol_form(K);
      while ( true )
      {
         // Project to \int K = mass_fraction * vol
         // double scale = mass_fraction * domain_volume / mass;
         // K *= scale;
         K += mass_fraction - ( mass / domain_volume );

         // Project to [K_min,K_max]
         for (int i = 0; i < K.Size(); i++)
         {
            if (K[i] > K_max) 
            {
               K[i] = K_max;
            }
            else if (K[i] < K_min)
            {
               K[i] = K_min;
            }
            else
            { // do nothing
            }
         }

         mass = vol_form(K);
         if ( abs( mass / domain_volume - mass_fraction ) < 1e-4 )
         {
            break;
         }
      }

      // I. Compute norm of update.
      double norm = psolver->ComputeNormAndUpdate(step_length);
      double compliance = psolver->GetCompliance();

      // L. Exit if norm of grad is small enough.
      if (myid == 0)
      {
         mfem::out << "norm of reduced gradient = " << norm << endl;
         mfem::out << "compliance = " << compliance << endl;
         mfem::out << "mass_fraction = " << vol_form(K) / domain_volume << endl;
      }

      if (norm < tol)
      {
         break;
      }

      if (visualization)
      {
         if (myid == 0)
         {
            sout_K << "solution\n" << mesh << K
                  << "window_title 'Control K'" << flush;
         }
      }

      if (myid == 0)
      {
         cout << "variance " << variance << endl;
         cout << "norm " << norm << endl;
      }
      double ratio = sqrt(variance) / norm ;

      MFEM_VERIFY(IsFinite(ratio), "ratio not finite");

      if (ratio > theta)
      {
         global_adaptive_batch_size = (int)(pow(ratio / theta,2.) * global_adaptive_batch_size); 
      }
      adaptive_batch_size = global_adaptive_batch_size/num_procs;
      // for convinience keep the batch size the same on each proc;
      global_adaptive_batch_size = num_procs * adaptive_batch_size;
   }

   delete psolver;

   return 0;
}