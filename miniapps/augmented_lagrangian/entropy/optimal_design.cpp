//                                Solution of optimal design problem
//
// Compile with: make optimal_design
//
// Sample runs:
//    optimal_design -r 3
//    optimal_design -m ../../data/star.mesh -r 3
//    optimal_design -sl 1 -m ../../data/mobius-strip.mesh -r 4
//    optimal_design -m ../../data/star.mesh -sl 5 -r 3 -mf 0.5 -o 5 -max 0.75
//
// Description:  This examples solves the following PDE-constrained
//               optimization problem:
//
//         min J(ρ) = β (f,u) + B(ρ)
//
//         subject to   - Div( r(ρ) C ∇u ) = 0    in Ω
//                                       u = 0    on Γ_d
//                                (C ∇u) n = g    on Γ_t
//
//         and            0 <= ρ <= 1,  (ρ,1) <= θ vol(Ω)
//
// Here, B(ρ) = -(ρ,ln(ρ)) - (1-ρ,ln(1-ρ)) and r(ρ) = r_0 (1-ρ) + ρ

#include "mfem.hpp"
#include <memory>
#include <iostream>
#include <fstream>

using namespace std;
using namespace mfem;

/** The fixed point method is
 * 
 *  ρ_{k+1} = projit(expit( -β ||∇u(ρ_k)||_C ))
 * 
 * where projit is defined
 * 
 *  projit(τ) = expit(lnit(τ) +  λ)
 * 
 * where λ \in R is found by solving 
 * 
 *  (expit(lnit(τ) +  λ),1) = θ vol(Ω)
 * 
 */

double lnit(double x)
{
   double tol = 1e-12;
   x = min(max(tol,x),1.0-tol);
   // MFEM_ASSERT(x>0.0, "Argument must be > 0");
   // MFEM_ASSERT(x<1.0, "Argument must be < 1");
   return log(x/(1.0-x));
}

double expit(double x)
{
   if (x >= 0)
   {
      return 1.0/(1.0+exp(-x));
   }
   else
   {
      return exp(x)/(1.0+exp(x));
   }
}

double dexpitdx(double x)
{
   double tmp = expit(-x);
   return tmp - pow(tmp,2);
}

/**
 * @brief Compute the root of the function
 *            f(c) = (expit(lnit(tau) + c),1) - θ vol(Ω)
 */
void projit(GridFunction &tau, double &c, LinearForm &vol_form, double volume_fraction, double tol=1e-12, int max_its=10)
{
   GridFunction ftmp(tau.FESpace());
   GridFunction dftmp(tau.FESpace());
   for (int k=0; k<max_its; k++)
   {
      // Compute f(c) and dfdc(c)
      for (int i=0; i<tau.Size(); i++)
      {
         ftmp[i]  = expit(lnit(tau[i]) + c) - volume_fraction;
         dftmp[i] = dexpitdx(lnit(tau[i]) + c);
      }
      double f = vol_form(ftmp);
      double df = vol_form(dftmp);
      // std::cout << "c  = " << c << endl;
      // std::cout << "f  = " << f << endl;
      // std::cout << "df = " << df << endl;

      double dc = -f/df;
      c += dc;
      if (abs(dc) < tol) { break; }
   }
   tau = ftmp;
   tau += volume_fraction;
}

class VolForce:public VectorCoefficient
{
public:
    VolForce(double r_,double x,double y, double fx, double fy):VectorCoefficient(2)
    {
        r=r_;
        cntr.SetSize(2);
        frce.SetSize(2);
        cntr[0]=x;
        cntr[1]=y;
        frce[0]=fx;
        frce[1]=fy;
    }

    VolForce(double r_,double x, double y, double z, double fx, double fy, double fz):VectorCoefficient(3)
    {
        r=r_;
        cntr.SetSize(3);
        frce.SetSize(3);
        cntr[0]=x;
        cntr[1]=y;
        cntr[2]=z;
        frce[0]=fx;
        frce[1]=fy;
        frce[2]=fz;
    }

    void Eval (Vector &V, ElementTransformation &T, const IntegrationPoint &ip)
    {
        //evaluate the true coordinate of the ip
        Vector xx; xx.SetSize(T.GetDimension());
        T.Transform(ip,xx);

        for(int i=0;i<xx.Size();i++){
            xx[i]=xx[i]-cntr[i];
        }
        double cr=std::sqrt(xx*xx);

        if(cr<=r)
        {
            for(int i=0;i<T.GetDimension();i++)
            {
                V[i]=frce[i];
            }
        }else{
            V=0.0;
        }
    }

    void Set(double r_,double x,double y, double fx, double fy)
    {
        r=r_;
        cntr[0]=x;
        cntr[1]=y;
        frce[0]=fx;
        frce[1]=fy;
    }

    void Set(double r_,double x, double y, double z, double fx, double fy, double fz)
    {
        r=r_;
        cntr[0]=x;
        cntr[1]=y;
        cntr[2]=z;
        frce[0]=fx;
        frce[1]=fy;
        frce[2]=fz;
    }


private:
    double r;
    mfem::Vector cntr;
    mfem::Vector frce;
};

// A Coefficient for computing the components of the stress.
class StrainEnergyDensityCoefficient : public Coefficient
{
protected:
   Coefficient &lambda, &mu;
   GridFunction *u; // displacement
   DenseMatrix grad; // auxiliary matrix, used in Eval
   double r_min;

public:
   StrainEnergyDensityCoefficient(Coefficient &lambda_, Coefficient &mu_, double r_min_=0.0)
      : lambda(lambda_), mu(mu_), u(NULL), r_min(r_min_)
   {
      MFEM_ASSERT(r_min_ >= 0.0, "r_min must be >= 0");
      MFEM_ASSERT(r_min_ < 1.0,  "r_min must be > 1");
   }

   void SetDisplacement(GridFunction &u_) { u = &u_; }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../bar2d.msh";
   int ref_levels = 2;
   int order = 2;
   double beta = 1.0;
   double volume_fraction = 0.5;
   double r_min = 1e-2;
   int max_it = 1e3;
   double tol = 1e-8;
   bool visualization = true;
   double alpha = 1.0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file for optimization domain.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly before optimization.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite element spaces.");
   args.AddOption(&alpha, "-alpha", "--alpha",
                  "Step length.");
   args.AddOption(&beta, "-beta", "--beta",
                  "Entropy weight.");
   args.AddOption(&volume_fraction, "-vf", "--volume-fraction",
                  "Volume fraction.");
   args.AddOption(&r_min, "-rmin", "--r-min",
                  "Minimum volume density.");
   args.AddOption(&max_it, "-mi", "--max-it",
                  "Maximum number of fixed point iterations.");
   args.AddOption(&tol, "-tol", "--tol",
                  "Convergence tolerance.");
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

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral and hexahedral meshes with the same code.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   // 3. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement, where 'ref_levels' is a
   //    command-line parameter.
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh.UniformRefinement();
   }

   // 4. Define the vector finite element spaces representing the state variable u,
   //    adjoint variable p, and the control variable f.
   H1_FECollection state_fec(order, dim);
   // L2_FECollection control_fec(order, dim);
   // H1_FECollection control_fec(order-1, dim, BasisType::Positive);
   L2_FECollection control_fec(order-1, dim, BasisType::Positive);
   FiniteElementSpace state_fes(&mesh, &state_fec, dim);
   FiniteElementSpace control_fes(&mesh, &control_fec);

   int state_size = state_fes.GetTrueVSize();
   int control_size = control_fes.GetTrueVSize();
   cout << "Number of state unknowns: " << state_size << endl;
   cout << "Number of control unknowns: " << control_size << endl;

   // 5. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking only
   //    boundary attribute 1 from the mesh as essential and converting it to a
   //    list of true dofs.
   MFEM_VERIFY(mesh.bdr_attributes.Size() > 0,
            "Boundary attributes required in the mesh.");
   Array<int> ess_tdof_list, ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[0] = 1;
   // ess_bdr[2] = 1;
   state_fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // 6. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system. In this case, b_i equals the boundary integral
   //    of f*phi_i where f represents a "pull down" force on the Neumann part
   //    of the boundary and phi_i are the basis functions in the finite element
   //    fespace. The force is defined by the VectorArrayCoefficient object f,
   //    which is a vector of Coefficient objects. The fact that f is non-zero
   //    on boundary attribute 2 is indicated by the use of piece-wise constants
   //    coefficient for its last component.

   VolForce f(0.05,2.90,0.5,0.0,-1.0);
   // VectorArrayCoefficient f(dim);
   // f.Set(dim-1, new ConstantCoefficient(-1.0e-2));
   // for (int i = 0; i < dim-1; i++)
   // {
   //    f.Set(i, new ConstantCoefficient(0.0));
   // }
   // {
   //    Vector pull_force(mesh.bdr_attributes.Max());
   //    pull_force = 0.0;
   //    pull_force(1) = -1.0e-2;
   //    f.Set(dim-1, new PWConstCoefficient(pull_force));
   // }
   LinearForm b(&state_fes);
   b.AddDomainIntegrator(new VectorDomainLFIntegrator(f));
   // b.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(f));
   b.Assemble();
   OperatorPtr A, M;
   Vector B, C, X;

   // 7. Set the initial guess for rho and the boundary conditions for u and p.
   GridFunction u(&state_fes);
   GridFunction rho(&control_fes);
   GridFunction rho_old(&control_fes);
   GridFunction r(&control_fes);
   GridFunction grad(&control_fes);
   u = 0.0;
   rho = 0.5;
   rho_old = 0.0;
   r = 0.0;
   grad = 0.0;

   // 8. Define the Lame parameters
   ConstantCoefficient Lame_lambda(1.0);
   ConstantCoefficient Lame_mu(100.0);
   StrainEnergyDensityCoefficient EnergyDensity(Lame_lambda, Lame_mu, r_min);

   // 9. Define H1-mass matrix
   BilinearForm m(&state_fes);
   m.AddDomainIntegrator(new VectorMassIntegrator());
   m.Assemble();
   Array<int> empty_list;
   m.FormSystemMatrix(empty_list, M);
   GSSmoother SM((SparseMatrix&)(*M));

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
   socketstream sout_u,sout_rho;
   if (visualization)
   {
      sout_u.open(vishost, visport);
      sout_rho.open(vishost, visport);
      sout_u.precision(8);
      sout_rho.precision(8);
   }

   // 12. Perform Picard iterations
   double lambda = 0.0;
   for (int k = 1; k <= max_it; k++)
   {
      // A. Form state equation
      BilinearForm a(&state_fes);
      r = 1.0;
      r -= rho;
      r *= r_min;
      r += rho;
      GridFunctionCoefficient density_coeff(&r);
      ProductCoefficient density_lambda(Lame_lambda,density_coeff);
      ProductCoefficient density_mu(Lame_mu,density_coeff);
      a.AddDomainIntegrator(new ElasticityIntegrator(density_lambda,density_mu));
      a.Assemble();
      a.FormLinearSystem(ess_tdof_list, u, b, A, X, B);

      // B. Solve state equation
      GSSmoother S((SparseMatrix&)(*A));
      PCG(*A, S, B, X, 0, 800, 1e-12, 0.0);

      // C. Recover state variable
      a.RecoverFEMSolution(X, b, u);

      // D. Send the solution by socket to a GLVis server.
      if (visualization)
      {
         sout_u << "solution\n" << mesh << u
                << "window_title 'State u'" << flush;
      }

      // E. Compute strain energy density (i.e.,  grad F(ρ) = -(1-ρ_min)||\nabla u||_C )
      // 
      // DivergenceGridFunctionCoefficient divu(&u);
      // ProductCoefficient norm2divu(divu,divu);
      // ProductCoefficient EnergyDensity(norm2divu,Lame_lambda);
      EnergyDensity.SetDisplacement(u);
      grad.ProjectCoefficient(EnergyDensity);
      grad *= -1.0;

      // LinearForm d(&control_fes);
      // d.AddDomainIntegrator(new DomainLFIntegrator(EnergyDensity));
      // d.Assemble();
      // BilinearForm L2proj(&control_fes);
      // InverseIntegrator * m = new InverseIntegrator(new MassIntegrator());
      // L2proj.AddDomainIntegrator(m);
      // L2proj.Assemble();
      // Array<int> empty_list;
      // OperatorPtr invM;
      // L2proj.FormSystemMatrix(empty_list,invM);   
      // invM->Mult(d,grad);
      
      // F. Update design.
      for (int i = 0; i < rho.Size(); i++)
      {
         rho[i] = expit((1+alpha/beta)*lnit(rho[i]) - alpha*grad[i]);
         // rho[i] = expit(beta*grad[i]);
      }
      projit(rho, lambda, vol_form, volume_fraction);

      // G. Exit if norm of update is small enough.
      GridFunctionCoefficient tmp(&rho_old);
      double norm = rho.ComputeL2Error(tmp);
      rho_old = rho;
      double compliance = b(u);
      double volume = vol_form(rho);
      mfem::out << "norm of update = " << norm << endl;
      mfem::out << "compliance = " << compliance << endl;
      mfem::out << "volume fraction = " << volume / domain_volume << endl;
      mfem::out << "lambda = " << lambda << endl;
      // if (norm < tol)
      // {
      //    break;
      // }

      if (visualization)
      {
          sout_rho << "solution\n" << mesh << rho
              << "window_title 'Density rho'" << flush;
      }

   }

   return 0;
}

double StrainEnergyDensityCoefficient::Eval(ElementTransformation &T,
                               const IntegrationPoint &ip)
{
   MFEM_ASSERT(u != NULL, "displacement field is not set");

   double L = lambda.Eval(T, ip);
   double M = mu.Eval(T, ip);
   u->GetVectorGradient(T, grad);
   double div_u = grad.Trace();
   double density = L*div_u*div_u;
   int dim = T.GetSpaceDim();
   for (int i=0; i<dim; i++)
   {
      for (int j=0; j<dim; j++)
      {
         density += M*grad(i,j)*(grad(i,j)+grad(j,i));
      }
   }
   return (1-r_min)*density/2.0;
}