#include "mfem.hpp"
#include <fstream>
#include <iostream>

// make elasticity -j && ./elasticity -rs 5 -o 2 -m lshaped.mesh
// make elasticity -j && ./elasticity -rs 3 -o 2 -m platewhole.mesh

using namespace mfem;
using namespace std;

//----------------------------------------------------------------
// Forcing function: Zero body force.
void ZeroForce(const Vector &x, Vector &f)
{
   f.SetSize(x.Size());
   f = 0.0;
}

//----------------------------------------------------------------
// Traction function: Apply a constant (nonzero) traction.
// For example, here we apply a constant upward traction.
// (You can adjust the constant vector as needed.)
void ConstantTractionY(const Vector &x, Vector &g)
{
   g.SetSize(x.Size());
   g = 0.0;
   // Let the vertical component be nonzero.
   // For a 2D problem, we set g = [0, 1] (traction in the y-direction).
   g[1] = 1.0;
}

void ConstantTractionX(const Vector &x, Vector &g)
{
   g.SetSize(x.Size());
   g = 0.0;
   // Let the vertical component be nonzero.
   // For a 2D problem, we set g = [0, 1] (traction in the y-direction).
   g[0] = 1.0;
}

void ConstantTractionNegX(const Vector &x, Vector &g)
{
   g.SetSize(x.Size());
   g = 0.0;
   // Let the vertical component be nonzero.
   // For a 2D problem, we set g = [0, 1] (traction in the y-direction).
   g[0] = -1.0;
}

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   OptionsParser args(argc, argv);
   const char *mesh_file = "lshaped.mesh"; // Your L-shaped mesh file.
   int order = 2;            // Finite element order
   int ref_levels = 3;       // Number of uniform refinements
   bool visualization = true;

   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Polynomial degree of mesh finite element space.");
   args.AddOption(&ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Read the mesh.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 3. Uniform mesh refinements.
   for (int l = 0; l < ref_levels; l++)
   {
      mesh->UniformRefinement();
   }

   // 4. Define a vector finite element space (each displacement component is in H1).
   FiniteElementCollection *fec = new H1_FECollection(order, dim);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec, dim);

   // 5. Mark essential (Dirichlet) boundaries.
   // Here we assume boundary attribute 1 is for Dirichlet conditions (clamped boundaries).
   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 0;
   // ess_bdr[3] = 1;
   if (strcmp(mesh_file, "lshaped.mesh") == 0)
   {
      ess_bdr[3] = 1;
   }
   Array<int> ess_tdof_list;
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // 6. Define the GridFunction (solution) and initialize it to zero.
   GridFunction x(fespace);
   x = 0.0;

   // 7. Set up the right-hand side linear form.
   LinearForm *b = new LinearForm(fespace);

   // Domain integrator for the body force: here, zero.
   VectorFunctionCoefficient f_coeff(dim, ZeroForce);
   b->AddDomainIntegrator(new VectorDomainLFIntegrator(f_coeff));

   // Neumann boundary: apply a nonzero (constant) traction on attribute 2.
   Array<int> neumann_bdr(mesh->bdr_attributes.Max());
   neumann_bdr = 0;
   neumann_bdr[1] = 1;
   VectorFunctionCoefficient g_coeff(dim, ConstantTractionX);
   b->AddBdrFaceIntegrator(new VectorBoundaryLFIntegrator(g_coeff), neumann_bdr);

   Array<int> neumann_bdr_negx(mesh->bdr_attributes.Max());
   neumann_bdr_negx = 0;
   neumann_bdr_negx[3] = 1;
   VectorFunctionCoefficient g_coeff_negx(dim, ConstantTractionNegX);
   if (strcmp(mesh_file, "platewhole.mesh") == 0)
   {
      b->AddBdrFaceIntegrator(new VectorBoundaryLFIntegrator(g_coeff_negx), neumann_bdr_negx);
   }

   b->Assemble();

   // 8. Set up the bilinear form corresponding to the linear elasticity operator.
   BilinearForm *a = new BilinearForm(fespace);

   // Define the material properties.
   // For a given Young's modulus E and Poisson's ratio nu:
   double E = 1.0;    // Young's modulus (you can adjust as needed)
   double nu = 0.3;   // Poisson's ratio
   double mu = E / (2.0 * (1.0 + nu));
   double lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0*nu));
   ConstantCoefficient lc(lambda);
   ConstantCoefficient muc(mu);

   // MFEM's ElasticityIntegrator implements:
   //   a(u,v) = ∫ [λ (div u)(div v) + 2μ ε(u):ε(v)] dx.
   a->AddDomainIntegrator(new ElasticityIntegrator(lc, muc));
   a->Assemble();

   // 9. Form the linear system A X = B, eliminating the Dirichlet DOFs.
   SparseMatrix A;
   Vector X, Bvec;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, Bvec);

   // 10. Solve the linear system.
   // Here we use the Conjugate Gradient (CG) method with a Gauss-Seidel smoother.
   GSSmoother M(A);
   PCG(A, M, Bvec, X, 1, 1000, 1e-12, 0.0);

   // 11. Recover the finite element solution.
   a->RecoverFEMSolution(X, *b, x);

   // 12. Save the solution for visualization.
   {
      ofstream sol_ofs("solution.gf");
      x.Save(sol_ofs);
   }

   // Optionally, visualize the solution in GLVis.
   if (visualization)
   {
      osockstream sock(19916, "localhost");
      sock << "solution\n";
      mesh->Print(sock);
      x.Save(sock);
      sock.send();
      sock << "window_title 'Elasticity: Displacement'\n"
           << "window_geometry "
           << 0 << " " << 0 << " " << 400 << " " << 400 << "\n"
           << "keys jRmclA" << endl;
   }

   FiniteElementCollection *fec_dc = new L2_FECollection(order-1, dim);
   FiniteElementSpace *fespace_dc = new FiniteElementSpace(mesh, fec_dc);
   GridFunction stress_gf(fespace_dc);

   FiniteElementCollection *fec_dc2 = new L2_FECollection(0, dim);
   FiniteElementSpace *fespace_dc2 = new FiniteElementSpace(mesh, fec_dc2);
   GridFunction stress_gf2(fespace_dc2);
   stress_gf = 0.0;

   // Loop over elements.
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      const FiniteElement *fe = fespace->GetFE(i);
      const FiniteElement *fe2 = fespace_dc->GetFE(i);
      ElementTransformation *T = mesh->GetElementTransformation(i);
      int nd = fe->GetDof();
      int nd2 = fe2->GetDof();
      int order = fe->GetOrder();
      int int_order = 2*order + 3;
      const IntegrationRule *ir = &IntRules.Get(fe->GetGeomType(), int_order);
      Vector shape(nd2);
      Array<int> dofs;
      Vector vals;
      fespace->GetElementVDofs(i, dofs);
      x.GetSubVector(dofs, vals);
      DenseMatrix dshape(nd, dim);
      Vector elvec(nd2);
      elvec = 0.0;
      for (int q = 0; q < ir->GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir->IntPoint(q);
         T->SetIntPoint(&ip);
         double w = ip.weight;
         fe->CalcDShape(ip, dshape);
         DenseMatrix grad(dim, dim);
         grad = 0.0;
         for (int a = 0; a < nd; a++)
         {
            for (int j = 0; j < dim; j++)
            {
               for (int k = 0; k < dim; k++)
               {
                  // In the vector FE, the a-th node has "dim" components stored consecutively.
                  // du_j/dr_k
                  grad(j,k) += vals[a + j*nd] * dshape(a,k);
               }
            }
         }

         DenseMatrix Jac(dim);
         Jac = T->Jacobian();
         DenseMatrix invJac(dim);
         invJac = Jac;
         invJac.Invert();
         DenseMatrix grad_phys(dim, dim);
         Mult(grad, invJac, grad_phys);

         DenseMatrix strain(dim, dim);
         strain = 0.0;
         for (int j = 0; j < dim; j++)
         {
            for (int k = 0; k < dim; k++)
            {
               strain(j,k) = 0.5*(grad_phys(j,k) + grad_phys(k,j));
            }
         }

         // Compute the stress tensor: σ = λ (tr ε) I + 2 μ ε.
         DenseMatrix stress(dim, dim);
         double trace = strain(0,0) + strain(1,1);
         stress(0,0) = lambda * trace + 2 * mu * strain(0,0);
         stress(1,1) = lambda * trace + 2 * mu * strain(1,1);
         stress(0,1) = 2 * mu * strain(0,1);
         stress(1,0) = stress(0,1);

         // Compute the von Mises stress (for plane stress/strain in 2D)
         double sigma_vm = sqrt( 0.5*stress(0,0)*stress(0,0)
                               + 0.5*stress(1,1)*stress(1,1)
                     + 0.5*(stress(0,0)-stress(1,1))*(stress(0,0)-stress(1,1))
                           + 3.0*stress(0,1)*stress(0,1) );
         // Store the computed stress in the DG GridFunction.
         fe2->CalcShape(ip, shape);
         elvec.Add(ip.weight*sigma_vm, shape);
      }
      fespace_dc->GetElementDofs(i, dofs);
      stress_gf.AddElementVector(dofs, elvec);
      stress_gf2(i) = elvec.Sum();
   }

   // 14. Save the stress field.
   {
      ofstream stress_ofs("stress.gf");
      stress_gf.Save(stress_ofs);
   }

   // 15. Visualize the displacement solution and the stress field using GLVis.
   if (visualization)
   {
      osockstream sock(19916, "localhost");
      sock << "solution\n";
      mesh->Print(sock);
      stress_gf.Save(sock);
      sock.send();
      sock << "window_title 'Elasticity: Von Mises Stress'\n"
           << "window_geometry "
           << 470 << " " << 0 << " " << 400 << " " << 400 << "\n"
           << "keys jRmclA" << endl;
   }

   if (visualization)
   {
      osockstream sock(19916, "localhost");
      sock << "solution\n";
      mesh->Print(sock);
      stress_gf2.Save(sock);
      sock.send();
      sock << "window_title 'Elasticity: Von Mises Stress (element-total)'\n"
           << "window_geometry "
           << 940 << " " << 0 << " " << 400 << " " << 400 << "\n"
           << "keys jRmclA" << endl;
   }

   // 13. Free the allocated memory.
   delete a;
   delete b;
   delete fespace;
   delete fec;
   delete mesh;

   return 0;
}

