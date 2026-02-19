/*
 * Proximal Galerkin Projection
 * min || u - u0 ||^2 / 2
 * subject to u >= 0 a.e.
 * Here, || . || is the L2 norm.
 * The constraint is enforced by the Bregman divergence associated with the Shannon entropy
 * R(u) = u ln(u) - u
 * D_R(u, v) = R(u) - R(v) - grad R(v) (u - v)
 *
 * Since R is singular, we weakly enforce the constraint
 * min || u - u0 ||^2 + (1/t) D_R(u, u_k)
 *
 * The optimality condition is
 * u - u0 + (1/alpha) (grad R(u) - grad R(u_k)) = 0
 * Letting psi = grad R(u), we have
 *
 * u - u0 + (1/alpha) (psi - psi_k) = 0
 * u - grad R^*(psi) = 0
 *
 * Introducing lambda = (psi_k - psi) / alpha, we have
 * psi = psi_k - alpha lambda
 * u - u0 - lambda = 0
 * u - grad R^*(psi_k - alpha lambda) = 0
 *
 * Then, we have
 * Mu - M lambda = U0
 * Mu - grad R^*(psi_k - alpha lambda) = 0
 *
 * The Newton system is
 * Mu - M lambda = U0
 * Mu + alpha*hess R^*(psi_k - alpha lambda_prev) lambda = grad R^*(psi_k - alpha lambda_prev) + alpha*hess R^*(psi_k - alpha lambda_prev) lambda_prev
 *
*/
#include "legendre.hpp"
#include <cstdlib>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command line options.
   int primal_order = 2;
   int latent_order = 0;
   int ref_levels = 2;

   OptionsParser args(argc, argv);
   args.AddOption(&primal_order, "-o", "--order",
                  "Finite element polynomial degree");
   args.AddOption(&latent_order, "-lo", "--latent-order",
                  "Latent finite element polynomial degree");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.ParseCheck();

   // 2. Read the mesh from the given mesh file, and refine once uniformly.
   Mesh mesh = Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL);
   for (int l=0; l<ref_levels; l++)
   {
      mesh.UniformRefinement();
   }
   const int dim = mesh.Dimension();

   // 3. Define a finite element space on the mesh. Here we use H1 continuous
   //    high-order Lagrange finite elements of the given order.
   L2_FECollection primal_fec(primal_order, mesh.Dimension());
   FiniteElementSpace primal_fes(&mesh, &primal_fec);
   cout << "Number of unknowns: " << primal_fes.GetTrueVSize() << endl;
   PLBound plb(&primal_fes);
   Vector points(plb.GetControlPoints());
   IntegrationRule ir_1D(points.Size());
   for (int i=0; i<points.Size(); i++)
   {
      ir_1D.IntPoint(i).x = points[i];
      ir_1D.IntPoint(i).weight = 1.0;
   }
   IntegrationRule ir(ir_1D, ir_1D);

   QuadratureSpace latent_qs(mesh, ir);

   // 5. Define the solution x as a finite element grid function in fespace. Set
   //    the initial guess to zero, which also sets the boundary conditions.
   GridFunction u(&primal_fes);
   QuadratureFunction psi(&latent_qs);
   u = 0.0;
   psi = 0.0;

   // 6. Setup target function.
   // This function is not positive, so the projection will be clipped version of this function.
   FunctionCoefficient u_targ([](const Vector &x) { return std::sin(2*M_PI * x[0]) * std::sin(2*M_PI * x[1]); });

   // 7, Define entropy function
   Shannon entropy;
   PrimalCoefficient u_cf(psi, entropy);
   PrimalGradientCoefficient hess_cf(psi, entropy); // grad (grad R^*(psi))

   // 7. Element Loop
   // Since our space is L2, we can do element-wise projection.
   MassIntegrator mass;
   DomainLFIntegrator targ(u_targ, &ir);
   DomainLFIntegrator int_u(u_cf, &ir);
   DenseMatrix M, M_mixed, H;
   Vector b;

   DenseMatrix L = plb.GetLowerBoundMatrix(dim);
   DenseMatrix neg_Lt = L;
   neg_Lt.Transpose();
   neg_Lt *= -1.0;

   DenseMatrix pg_mat;
   Array<int> offsets(3);
   BlockVector pg_rhs, pg_sol, pg_prev, newt_prev;
   Vector psi_prev, psi_curr;

   Array<int> primal_glb_idx;
   Array<int> latent_glb_idx;
   DenseMatrixInverse pg_inv;
   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock.precision(8);
   sol_sock << "solution\n" << mesh << u << flush;

   for (int e_idx=0; e_idx<mesh.GetNE(); e_idx++)
   {
      ElementTransformation *Tr = mesh.GetElementTransformation(e_idx);
      const FiniteElement *primal_fe = primal_fes.GetFE(e_idx);
      primal_fes.GetElementDofs(e_idx, primal_glb_idx);
      int primal_dof = primal_fe->GetDof();
      int latent_dof = ir.GetNPoints();
      pg_mat.SetSize(primal_dof + latent_dof);
      H.SetSize(latent_dof);
      H = 0.0;

      offsets[0] = 0;
      offsets[1] = primal_dof;
      offsets[2] = primal_dof + latent_dof;

      psi_prev.SetSize(latent_dof);
      psi_prev = 0.0;
      psi_curr.SetSize(latent_dof);
      psi_curr = 0.0;

      pg_rhs.Update(offsets);
      pg_rhs = 0.0;
      pg_sol.Update(offsets);
      pg_sol = 0.0;
      pg_prev.Update(offsets);
      newt_prev.Update(offsets);

      Vector &u_e = pg_sol.GetBlock(0);
      Vector &lam_e = pg_sol.GetBlock(1);

      Vector &primal_res = pg_rhs.GetBlock(0);
      Vector &dual_res = pg_rhs.GetBlock(1);

      mass.AssembleElementMatrix(*primal_fe, *Tr, M);
      // M *= 1.0 / mesh.GetElementVolume(e_idx); // normalize
      targ.AssembleRHSElementVect(*primal_fe, *Tr, primal_res);

      pg_mat.SetSubMatrix(0, 0, M);
      pg_mat.SetSubMatrix(primal_dof, 0, L);
      pg_mat.SetSubMatrix(0, primal_dof, neg_Lt);

      real_t alpha = 0.01; // proximal step size
      out << "Element " << e_idx << std::endl;
      bool prox_converged = false;
      for (int prox_it = 0; prox_it < 1000; prox_it++)
      {
         out << "  Proximal iteration " << prox_it << std::endl;
         pg_prev = pg_sol;
         psi_prev = psi_curr;

         // TODO: Wrap this into an Operator and use NewtonSolver
         bool newt_converged = false;
         for (int newt_it = 0; newt_it < 20; newt_it++)
         {
            newt_prev = pg_sol;
            add(psi_prev, -alpha, lam_e, psi_curr);

            for (int j=0; j<latent_dof; j++)
            {
               const real_t primval = entropy.gradinv(psi_curr[j]);
               const real_t hessval = entropy.hessinv(psi_curr[j]);

               dual_res[j] = primval;
               H(j, j) = hessval;
            }

            H *= alpha;
            H.AddMult(lam_e, dual_res);
            pg_mat.SetSubMatrix(primal_dof, primal_dof, H);

            pg_inv.Factor(pg_mat);
            pg_inv.Mult(pg_rhs, pg_sol);
            // if (pg_sol.GetBlock(0).DistanceTo(newt_prev.GetBlock(0)) < 1e-08)
            if (pg_sol.DistanceTo(newt_prev) < 1e-08)
            {
               newt_converged = true;
               break;
            }
         }
         MFEM_VERIFY(newt_converged, "Newton solver did not converge");
         // if (pg_sol.GetBlock(0).DistanceTo(pg_prev.GetBlock(0)) < 1e-08)
         if (pg_sol.DistanceTo(pg_prev) < 1e-08)
         {
            prox_converged = true;
            break;
         }
         alpha *= 1.1;
      }
      MFEM_VERIFY(prox_converged, "Proximal solver did not converge");
      u.SetSubVector(primal_glb_idx, u_e);
      sol_sock << "solution\n" << mesh << u << flush;
   }

   return EXIT_SUCCESS;
}
