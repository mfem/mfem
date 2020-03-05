//                        MFEM Example Hybridizable DG
//
// Compile with: make hdg_poisson
//
// Sample runs:  hdg_poisson -o 1 -r 1 -tr 4 -no-vis
//            hdg_poisson -o 5 -r 1 -tr 4 -no-vis
//            hdg_poisson -o 1 -r 4 -tr 1
//            hdg_poisson -o 5 -r 4 -tr 1
//            hdg_poisson -o 1 -r 1 -tr 4 -no-vis -m ../data/inline-tri.mesh
//            hdg_poisson -o 5 -r 1 -tr 4 -no-vis -m ../data/inline-tri.mesh
//            hdg_poisson -o 1 -r 5 -tr 1 -m ../data/inline-tri.mesh
//            hdg_poisson -o 5 -r 5 -tr 1 -m ../data/inline-tri.mesh
//
// Description:  This example code solves the 2D/3D diffusion problem
//                     -\nu Delta u = f
//               with Dirichlet boundary conditions, using HDG discretization.
//
// The methods approximates the solution u, the diffusive flux q = -\nu \nabla u, 
// and the restriction of u to the faces, denoted by lambda.
//
// The weak form is: seek (q,u,\lambda) such that for all (v, w, \mu)
//
// -\nu^{-1}(q, v)       + (u, div(v))       - <\lambda, v \cdot n>      = 0
//  (div(q), w)          + <\tau u, w>       - <\tau \lambda, w>         = (f, w)
// -<[[q \cdot n]], \mu> - <[[\tau u]], \mu> + <[[(\tau \lambda]], \mu>  = 0
//
// where [[.]] is the jump operator, (.,.) is the d-dimensional L2 product,
// <.,.> is the d-1 dimensional L2 product.
//
// The discretization is based on the paper:
//
// N.C. Nguyen, J. Peraire, B. Cockburn, An implicit high-order hybridizable
// discontinuous Galerkin method for linear convection–diffusion equations,
// J. Comput. Phys., 2009, 228:9, 3232--3254.
//
// Contributed by: T. Horvath, S. Rhebergen, A. Sivas
//                 University of Waterloo

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>


using namespace std;
using namespace mfem;

// Define the analytical solution and forcing terms / boundary conditions
double uFun_ex(const Vector & x);
void qFun_ex(const Vector & x, Vector & q);
double fFun(const Vector & x);
double diff;

// We can minimize the expression |\nu \nabla u_h^* + q_h |^2 over a single element K,
// for p+1 degree u_h^*, with the constraint \int_K u_h^* = \int_K u_h, so the mean
// of u_h^* is the same as the one of u_h.
//
// This results in the problem
//
// (nabla w_h, \nu \nabla u_h^*) = -(nabla w_h, q_h)
// (1, u_h^*)                    = (1, u_h)
//
// Since the fist equation on its own would generate a singular problem
// the last line of the system is rewritten by the second equation.
//
// This elementwise operation will provide a superconvergent solution
// \|u-u_h\|_{L^2} < C h^{p+2} |u|_{p+1}
class HDGPostProcessing
{
private:
   GridFunction *q, *u;

   FiniteElementSpace *fes;

   Coefficient *diffcoeff;

protected:
   const IntegrationRule *IntRule;

public:
   HDGPostProcessing(FiniteElementSpace *f, GridFunction &_q, GridFunction &_u,
                     Coefficient &_diffcoeff)
      : q(&_q), u(&_u), fes(f), diffcoeff(&_diffcoeff) {IntRule = NULL; }

   void Postprocessing(GridFunction &u_postprocessed) ;
};

int main(int argc, char *argv[])
{
   StopWatch chrono;

   // 1. Parse command-line options.
   const char *mesh_file = "../data/inline-tri.mesh";
   int order = 1;
   int initial_ref_levels = 0;
   int total_ref_levels = 1;
   bool visualization = true;
   bool post = true;
   bool save = true;
   double memA = 0.0;
   double memB = 0.0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&initial_ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly for the initial calculation.");
   args.AddOption(&total_ref_levels, "-tr", "--totalrefine",
                  "Number of times to refine the mesh uniformly to get the convergence rates.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&post, "-post", "--postprocessing",
                  "-no-post", "--no-postprocessing",
                  "Enable or disable postprocessing.");
   args.AddOption(&save, "-save", "--save-files", "-no-save",
                  "--no-save-files",
                  "Enable or disable file saving.");
   args.AddOption(&memA, "-memA", "--memoryA",
                  "Storage of A.");
   args.AddOption(&memB, "-memB", "--memoryB",
                  "Storage of B.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // memA, memB \in [0,1], memB <= memA
   if (memB > memA)
   {
      std::cout << "memB cannot be more than memA. Resetting to be equal" << std::endl
                << std::flush;
      memA = memB;
   }
   if (memA > 1.0)
   {
      std::cout << "memA cannot be more than 1. Resetting to 1" << std::endl <<
                std::flush;
      memA = 1.0;
   }
   else if (memA < 0.0)
   {
      std::cout << "memA cannot be less than 0. Resetting to 0." << std::endl <<
                std::flush;
      memA = 0.0;
   }
   if (memB > 1.0)
   {
      std::cout << "memB cannot be more than 1. Resetting to 1" << std::endl <<
                std::flush;
      memB = 1.0;
   }
   else if (memB < 0.0)
   {
      std::cout << "memB cannot be less than 0. Resetting to 0." << std::endl <<
                std::flush;
      memB = 0.0;
   }

   // 2. Read the mesh from the given mesh file. Refine it up to the initial_ref_levels.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   for (int ii=0; ii<initial_ref_levels; ii++)
   {
      mesh->UniformRefinement();
   }

   // 3. Vectors for the different discretization errors
   Vector u_l2errors(total_ref_levels), q_l2errors(total_ref_levels),
          mean_l2errors(total_ref_levels), u_star_l2errors(total_ref_levels);

   // 4. Define a finite element collections and spaces on the mesh.
   FiniteElementCollection *dg_coll(new DG_FECollection(order, dim));
   FiniteElementCollection *face(new DG_Interface_FECollection(order, dim));

   // Finite element spaces:
   // V_space is the vector valued DG space on elements for q_h
   // W_space is the scalar DG space on elements for u_h
   // M_space is the DG space on faces for lambda_h
   FiniteElementSpace *V_space = new FiniteElementSpace(mesh, dg_coll, dim);
   FiniteElementSpace *W_space = new FiniteElementSpace(mesh, dg_coll);
   FiniteElementSpace *M_space = new FiniteElementSpace(mesh, face);

   // 5. Define the coefficients, the exact solutions, the right hand side and the diffusion coefficient along with the diffusion penalty parameter.
   FunctionCoefficient fcoeff(fFun);

   FunctionCoefficient ucoeff(uFun_ex);
   VectorFunctionCoefficient qcoeff(dim, qFun_ex);

   diff = 1.;
   ConstantCoefficient diffusion(diff); // diffusion constant
   double tau_D = 5.0;

   // 6. Define the different forms and gridfunctions.
   HDGBilinearForm *AVarf(new HDGBilinearForm(V_space, W_space, M_space));
   AVarf->AddHDGDomainIntegrator(new HDGDomainIntegratorDiffusion(diffusion));
   AVarf->AddHDGFaceIntegrator(new HDGFaceIntegratorDiffusion(tau_D));

   GridFunction lambda_variable(M_space);
   GridFunction q_variable(V_space), u_variable(W_space);

   LinearForm *fform(new LinearForm);
   fform->AddDomainIntegrator(new DomainLFIntegrator(fcoeff));

   for (int ref_levels = initial_ref_levels;
        ref_levels < (initial_ref_levels + total_ref_levels); ref_levels++)
   {
      // 7. Compute the problem size and define the right hand side vectors
      int dimV = V_space->GetVSize();
      int dimW = W_space->GetVSize();
      int dimM = M_space->GetVSize();

      std::cout << "***********************************************************\n";
      std::cout << "dim(W) = " << dimV << "\n";
      std::cout << "dim(V) = " << dimW << "\n";
      std::cout << "dim(M) = " << dimM << "\n";
      std::cout << "dim(W+V+M) = " << dimV + dimW + dimM << "\n";
      std::cout << "***********************************************************\n";

      Vector rhs_R(dimV);
      Vector rhs_F(dimW);
      Vector V_aux(dimV);
      Vector W_aux(dimW);

      V_aux = 0.0;
      W_aux = 0.0;
      rhs_R = 0.0;

      // 8. To eliminate the boundary conditions we project the BC to a grid function
      // defined for the facet unknowns.
      FunctionCoefficient lambda_coeff(uFun_ex);
      lambda_variable.ProjectCoefficientSkeletonDG(lambda_coeff);

      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 1;

      // 9. Assemble the RHS and the Schur complement
      fform->Update(W_space, rhs_F, 0);
      fform->Assemble();

      GridFunction *R = new GridFunction(V_space, rhs_R);
      GridFunction *F = new GridFunction(W_space, rhs_F);
      AVarf->AssembleSC(R, F, ess_bdr, lambda_variable, memA, memB);
      AVarf->Finalize();

      SparseMatrix* SC = AVarf->SpMatSC();
      Vector* SC_RHS = AVarf->VectorSC();
      // AVarf->VectorSC() provides -C*A^{-1} RF, the RHS for the
      // Schur complement is  L - C*A^{-1} RF, but L is zero for this case.

      // 10. Solve the Schur complement system
      int maxIter(4000);
      double rtol(1.e-13);
      double atol(0.0);
      GSSmoother M(*SC);
      BiCGSTABSolver solver;
      solver.SetAbsTol(atol);
      solver.SetRelTol(rtol);
      solver.SetMaxIter(maxIter);
      solver.SetOperator(*SC);
      solver.SetPrintLevel(-1);
      solver.SetPreconditioner(M);
      chrono.Clear();
      chrono.Start();
      solver.Mult(*SC_RHS, lambda_variable);
      chrono.Stop();

      if (solver.GetConverged())
         std::cout << "Iterative method converged in " << solver.GetNumIterations()
                   << " iterations with a residual norm of " << solver.GetFinalNorm() << ".\n";
      else
         std::cout << "Iterative method did not converge in " <<
                   solver.GetNumIterations()
                   << " iterations. Residual norm is " << solver.GetFinalNorm() << ".\n";
      std::cout << "Iterative method solver took " << chrono.RealTime() << "s. \n";

      // Delete the SC matrix to save memory
      SC = NULL;

      // 11. Reconstruction
      // Reconstruct the solution u and q from the facet solution lambda
      AVarf->Reconstruct(R, F, &lambda_variable, &q_variable, &u_variable);

      // 12. Compute the discretization error
      int order_quad = max(2, 2*order+2);
      const IntegrationRule *irs[Geometry::NumGeom];
      for (int i=0; i < Geometry::NumGeom; ++i)
      {
         irs[i] = &(IntRules.Get(i, order_quad));
      }
      double err_u    = u_variable.ComputeL2Error(ucoeff, irs);
      double err_q    = q_variable.ComputeL2Error(qcoeff, irs);
      double err_mean  = u_variable.ComputeMeanLpError(2.0, ucoeff, irs);

      u_l2errors(ref_levels-initial_ref_levels) = fabs(err_u);
      q_l2errors(ref_levels-initial_ref_levels) = fabs(err_q);
      mean_l2errors(ref_levels-initial_ref_levels) = fabs(err_mean);

      std::cout << "|| u_h - u_ex || = " << err_u << "\n";
      std::cout << "|| q_h - q_ex || = " << err_q << "\n";
      std::cout << "|| mean(u_h) - mean(u_ex) || = " << err_mean << "\n";

      // 13. Save the mesh and the solution.
      if (save)
      {
         ofstream mesh_ofs("ex_hdg.mesh");
         mesh_ofs.precision(8);
         mesh->Print(mesh_ofs);

         ofstream q_variable_ofs("sol_q.gf");
         q_variable_ofs.precision(8);
         q_variable.Save(q_variable_ofs);

         ofstream u_variable_ofs("sol_u.gf");
         u_variable_ofs.precision(8);
         u_variable.Save(u_variable_ofs);

         ofstream lambda_variable_ofs("sol_lambda.gf");
         lambda_variable_ofs.precision(8);
         lambda_variable.Save(lambda_variable_ofs);
      }

      // 14. Send the solution by socket to a GLVis server.
      if (visualization)
      {
         char vishost[] = "localhost";
         int  visport   = 19916;
         socketstream u_sock(vishost, visport);
         u_sock.precision(8);
         u_sock << "solution\n" << *mesh << u_variable << "window_title 'Solution u'" <<
                endl;

         socketstream q_sock(vishost, visport);
         q_sock.precision(8);
         q_sock << "solution\n" << *mesh << q_variable << "window_title 'Solution q'" <<
                endl;
      }

      // 15. Postprocessing
      if (post)
      {
         FiniteElementCollection *dg_coll_pstar(new DG_FECollection(order+1, dim));
         FiniteElementSpace *Vstar_space = new FiniteElementSpace(mesh, dg_coll_pstar);

         GridFunction u_post(Vstar_space);

         HDGPostProcessing *hdgpost(new HDGPostProcessing(Vstar_space, q_variable,
                                                          u_variable, diffusion));

         hdgpost->Postprocessing(u_post);

         order_quad = max(2, 2*order+5);
         for (int i=0; i < Geometry::NumGeom; ++i)
         {
            irs[i] = &(IntRules.Get(i, order_quad));
         }
         double err_u_post   = u_post.ComputeL2Error(ucoeff, irs);

         u_star_l2errors(ref_levels-initial_ref_levels) = fabs(err_u_post);

         std::cout << "|| u^*_h - u_ex || = " << err_u_post << "\n";

         if (save)
         {
            ofstream u_post_ofs("sol_u_star.gf");
            u_post_ofs.precision(8);
            u_post.Save(u_post_ofs);
         }

         if (visualization)
         {
            char vishost[] = "localhost";
            int  visport   = 19916;
            socketstream u_star_sock(vishost, visport);
            u_star_sock.precision(8);
            u_star_sock << "solution\n" << *mesh << u_post <<
                        "window_title 'Solution u_star'" << endl;
         }
      }

      // 16. Refine the mesh to increase the resolution and update the spaces and the forms.
      mesh->UniformRefinement();

      V_space->Update(0);
      W_space->Update(0);
      M_space->Update(0);

      AVarf->Update();
      q_variable.Update();
      u_variable.Update();
      lambda_variable.Update();
   }

   // 17. Print the results
   std::cout << "\n\n-----------------------\n";
   std::cout <<
             "level  u_l2errors  order   q_l2errors  order   mean_l2errors  order u_star_l2errors   order\n";
   std::cout << "-----------------------\n";
   for (int ref_levels = 0; ref_levels < total_ref_levels; ref_levels++)
   {
      if (ref_levels == 0)
      {
         std::cout << "  " << ref_levels << "    "
                   << std::setprecision(2) << std::scientific << u_l2errors(ref_levels)
                   << "   " << " -       "
                   << std::setprecision(2) << std::scientific << q_l2errors(ref_levels)
                   << "    " << " -       "
                   << std::setprecision(2) << std::scientific << mean_l2errors(ref_levels)
                   << "    " << " -       "
                   << std::setprecision(2) << std::scientific << u_star_l2errors(ref_levels)
                   << "    " << " -       " << std::endl;
      }
      else
      {
         double u_order     = log(u_l2errors(ref_levels)/u_l2errors(ref_levels-1))/log(
                                 0.5);
         double q_order     = log(q_l2errors(ref_levels)/q_l2errors(ref_levels-1))/log(
                                 0.5);
         double mean_order   = log(mean_l2errors(ref_levels)/mean_l2errors(
                                      ref_levels-1))/log(0.5);
         double u_star_order = log(u_star_l2errors(ref_levels)/u_star_l2errors(
                                      ref_levels-1))/log(0.5);
         std::cout << "  " << ref_levels << "    "
                   << std::setprecision(2) << std::scientific << u_l2errors(ref_levels)
                   << "  " << std::setprecision(4) << std::fixed << u_order
                   << "    " << std::setprecision(2) << std::scientific << q_l2errors(ref_levels)
                   << "   " << std::setprecision(4) << std::fixed << q_order
                   << "    " << std::setprecision(2) << std::scientific << mean_l2errors(
                      ref_levels)
                   << "   " << std::setprecision(4) << std::fixed << mean_order
                   << "    " << std::setprecision(2) << std::scientific << u_star_l2errors(
                      ref_levels)
                   << "   " << std::setprecision(4) << std::fixed << u_star_order << std::endl;
      }
   }
   std::cout << "\n\n";

   // 18. Free the used memory.
   delete mesh;
   delete V_space;
   delete W_space;
   delete M_space;
   delete AVarf;
   delete fform;
   delete dg_coll;
   delete face;

   std::cout << "Done." << std::endl ;

   return 0;
}


double uFun_ex(const Vector & x)
{
   double xi(x(0));
   double yi(x(1));

   int dim = x.Size();

   switch (dim)
   {
      case 2:
      {
         return 1.0 + xi + sin(2.0*M_PI*xi)*sin(2.0*M_PI*yi);
         break;
      }
      case 3:
      {
         double zi(x(2));
         return xi + sin(2.0*M_PI*xi)*sin(2.0*M_PI*yi)*sin(2.0*M_PI*zi);
         break;
      }
   }

   return 0;
}

void qFun_ex(const Vector & x, Vector & q)
{
   double xi(x(0));
   double yi(x(1));
   int dim = x.Size();

   switch (dim)
   {
      case 2:
      {
         q(0) = -diff*1.0 - diff*2.0*M_PI*cos(2.0*M_PI*xi)*sin(2.0*M_PI*yi);
         q(1) =  0.0 - diff*2.0*M_PI*sin(2.0*M_PI*xi)*cos(2.0*M_PI*yi);
         break;
      }
      case 3:
      {
         double zi(x(2));
         q(0) = -diff*1.0 - diff*2.0*M_PI*cos(2.0*M_PI*xi)*sin(2.0*M_PI*yi)*sin(
                   2.0*M_PI*zi);
         q(1) =  0.0 - diff*2.0*M_PI*sin(2.0*M_PI*xi)*cos(2.0*M_PI*yi)*sin(2.0*M_PI*zi);
         q(2) =  0.0 - diff*2.0*M_PI*sin(2.0*M_PI*xi)*sin(2.0*M_PI*yi)*cos(2.0*M_PI*zi);
         break;
      }
   }
}


double fFun(const Vector & x)
{
   double xi(x(0));
   double yi(x(1));
   int dim = x.Size();

   switch (dim)
   {
      case 2:
      {
         return diff*8.0*M_PI*M_PI*sin(2.0*M_PI*xi)*sin(2.0*M_PI*yi);
         break;
      }
      case 3:
      {
         double zi(x(2));
         return diff*12.0*M_PI*M_PI*sin(2.0*M_PI*xi)*sin(2.0*M_PI*yi)*sin(2.0*M_PI*zi);
         break;
      }
   }

   return 0;

}

// Postprocessing
void HDGPostProcessing::Postprocessing(GridFunction &u_postprocessed)
{
   Mesh *mesh = fes->GetMesh();
   Array<int>  vdofs;
   Vector      elmat2, shape, RHS, to_RHS, vals, uvals;
   double      RHS2;
   DenseMatrix elmat, invdfdx, dshape, dshapedxt, qvals;

   int  ndofs;
   const FiniteElement *fe_elem;
   ElementTransformation *Trans;

   for (int i = 0; i < fes->GetNE(); i++)
   {
      fes->GetElementVDofs(i, vdofs);
      ndofs = vdofs.Size();
      vals.SetSize(ndofs);
      // elmat is the matrix for the -(nabla w_h, q_h) term
      elmat.SetSize(ndofs);
      // elmat 1 is the vector for the (1, u_h^*) term
      elmat2.SetSize(ndofs);
      shape.SetSize(ndofs);

      RHS.SetSize(ndofs);
      to_RHS.SetSize(ndofs);

      elmat = 0.0;
      elmat2 = 0.0;
      RHS = 0.0;
      RHS2 = 0.0;

      fe_elem = fes->GetFE(i);
      int dim = fe_elem->GetDim();
      int spaceDim = dim;
      invdfdx.SetSize(dim, spaceDim);
      dshape.SetSize(ndofs, spaceDim);
      dshapedxt.SetSize(ndofs, spaceDim);

      Trans = mesh->GetElementTransformation(i);

      const IntegrationRule *ir = IntRule;
      if (ir == NULL)
      {
         int order = 3*fe_elem->GetOrder() + 3;
         ir = &IntRules.Get(fe_elem->GetGeomType(), order);
      }

      // Get the values of u_h and q_h
      u->GetValues(i, *ir, uvals);
      q->GetVectorValues(*Trans, *ir, qvals);

      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);

         fe_elem->CalcDShape(ip, dshape);
         fe_elem->CalcShape(ip, shape);

         Trans->SetIntPoint(&ip);
         // Compute invdfdx = / adj(J),       if J is square
         //               \ adj(J^t.J).J^t, otherwise
         CalcAdjugate(Trans->Jacobian(), invdfdx);
         double w = Trans->Weight();
         w = ip.weight / w;
         w *= diffcoeff->Eval(*Trans, ip);
         Mult(dshape, invdfdx, dshapedxt);

         // compute the (nabla w_h, \nu \nabla u_h^*) term
         AddMult_a_AAt(w, dshapedxt, elmat);

         dshapedxt *= ip.weight ;

         Vector qval_col;
         qvals.GetColumn(j, qval_col);

         // compute (nabla w_h, q_h)
         dshapedxt.Mult(qval_col, to_RHS);

         // subtract it from the rhs
         RHS -= to_RHS;

         // compute (1, u_h^*)
         shape *= (Trans->Weight() * ip.weight);
         elmat2 += shape;

         // compute (1, u_h)
         double rhs_weight = (Trans->Weight() * ip.weight);
         RHS2  += (uvals(j)*rhs_weight);

      }

      // changing the last row and the last entry
      for (int j = 0; j < ndofs; j++)
      {
         elmat(ndofs-1,j) = elmat2(j);
      }
      RHS(ndofs-1) = RHS2;

      // solve the local problem
      elmat.Invert();
      elmat.Mult(RHS, vals);
      u_postprocessed.SetSubVector(vdofs, vals);

   }
}
