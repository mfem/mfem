//                       MFEM Example ?? - Serial Version
//
// Compile with: make amr-battery
//
// Sample runs:  amr-battery -rs 7
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the thermal battery
//               problem described in Mitchell 2013.
//
//               The boundary conditions are defined as (where u is the solution
//               field):
//
//                  Dirichlet: u = d
//                  Neumann:   n.Grad(u) = g
//                  Robin:     n.Grad(u) + a u = b
//
//               We recommend viewing Examples 1 and 27 before viewing this
//               example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

static double cx[5] = {0.0, 6.1, 6.5, 8.0, 8.4};
static double cy[8] = {0.0, 0.8, 1.6, 3.6, 18.8, 21.2, 23.2, 24.0};

static double cp[5] = {25.0, 7.0, 5.0, 0.2, 0.05};
static double cq[5] = {25.0, 0.8, 0.0001, 0.2, 0.05};
static double cf[5] = {0.0, 1.0, 1.0, 0.0, 0.0};

enum BoundarySegment
  {
    top = 1,  // 1-based for MFEM attributes
    right = 2,
    bottom = 3,
    left = 4
  };

Mesh * GenerateSerialMesh(const double sx, const double sy);

void diffCoeff(const Vector &x, Vector &d);
double fcoeff(const Vector & x);

// Compute the average value of alpha*n.Grad(sol) + beta*sol over the boundary
// attributes marked in bdr_marker. Also computes the L2 norm of
// alpha*n.Grad(sol) + beta*sol - gamma over the same boundary.
double IntegrateBC(const GridFunction &sol, const Array<int> &bdr_marker,
                   double alpha, double beta, double gamma,
                   double &err);

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   int ser_ref_levels = 2;
   int order = 1;
   bool visualization = true;
   double nbc_val = 0.0;
   int max_dofs = 50000;
   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&max_dofs, "-md", "--max-dofs",
                  "Stop after reaching this many degrees of freedom.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(mfem::out);
      return 1;
   }
   args.PrintOptions(mfem::out);

   // 2. Construct the (serial) mesh and refine it if requested.
   Mesh *mesh = GenerateSerialMesh(cx[4], cy[7]);
   for (int l = 0; l < ser_ref_levels; l++)
    {
      mesh->UniformRefinement();
    }

   int dim = mesh->Dimension();

   // 3. Define a finite element space on the serial mesh. Here we use either
   //    continuous Lagrange finite elements or discontinuous Galerkin finite
   //    elements of the specified order.
   H1_FECollection fec(order, dim);
   FiniteElementSpace fespace(mesh, &fec);
   int size = fespace.GetTrueVSize();
   mfem::out << "Number of finite element unknowns: " << size << endl;

   // 4. Create "marker arrays" to define the portions of boundary associated
   //    with each type of boundary condition. These arrays have an entry
   //    corresponding to each boundary attribute.  Placing a '1' in entry i
   //    marks attribute i+1 as being active, '0' is inactive.
   Array<int> nbc_bdr(mesh->bdr_attributes.Max());
   Array<int> rbc_bdr_top(mesh->bdr_attributes.Max());
   Array<int> rbc_bdr_right(mesh->bdr_attributes.Max());
   Array<int> rbc_bdr_bottom(mesh->bdr_attributes.Max());
   //Array<int> dbc_bdr(mesh->bdr_attributes.Max());

   nbc_bdr = 0; nbc_bdr[BoundarySegment::left - 1] = 1;
   rbc_bdr_top = 0; rbc_bdr_top[BoundarySegment::top - 1] = 1;
   rbc_bdr_right = 0; rbc_bdr_right[BoundarySegment::right - 1] = 1;
   rbc_bdr_bottom = 0; rbc_bdr_bottom[BoundarySegment::bottom - 1] = 1;

   Array<int> ess_tdof_list(0);

   // 5. Setup the various coefficients needed for the Laplace operator and the
   //    various boundary conditions.

   const double c_top = 1.0;
   const double c_right = 2.0;
   const double c_bottom = 3.0;

   const double gn_top = 3.0;
   const double gn_right = 2.0;
   const double gn_bottom = 1.0;

   VectorFunctionCoefficient matCoef(2, diffCoeff);

   ConstantCoefficient nbcCoef(nbc_val);

   ConstantCoefficient rbcACoef_top(c_top);
   ConstantCoefficient rbcACoef_right(c_right);
   ConstantCoefficient rbcACoef_bottom(c_bottom);

   ConstantCoefficient rbcBCoef_top(gn_top);
   ConstantCoefficient rbcBCoef_right(gn_right);
   ConstantCoefficient rbcBCoef_bottom(gn_bottom);

   // 6. Define the solution vector u as a finite element grid function
   //    corresponding to fespace. Initialize u with initial guess of zero.
   GridFunction u(&fespace);
   u = 0.0;

   // 7. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator.
   BilinearForm a(&fespace);
   BilinearFormIntegrator *integ = new DiffusionIntegrator(matCoef);
   a.AddDomainIntegrator(integ);

   // Add a mass integrator on the Robin boundary
   a.AddBoundaryIntegrator(new MassIntegrator(rbcACoef_top), rbc_bdr_top);
   a.AddBoundaryIntegrator(new MassIntegrator(rbcACoef_right), rbc_bdr_right);
   a.AddBoundaryIntegrator(new MassIntegrator(rbcACoef_bottom), rbc_bdr_bottom);

   // 8. Set up an error estimator. Here we use the Zienkiewicz-Zhu estimator
   //    that uses the ComputeElementFlux method of the DiffusionIntegrator to
   //    recover a smoothed flux (gradient) that is subtracted from the element
   //    flux to get an error indicator. We need to supply the space for the
   //    smoothed flux: an (H1)^sdim (i.e., vector-valued) space is used here.
   FiniteElementSpace flux_fespace(mesh, &fec, dim);
   ZienkiewiczZhuEstimator estimator(*integ, u, flux_fespace);
   estimator.SetAnisotropic();

   // 9. A refiner selects and refines elements based on a refinement strategy.
   //    The strategy here is to refine elements with errors larger than a
   //    fraction of the maximum element error. Other strategies are possible.
   //    The refiner will call the given error estimator.
   ThresholdRefiner refiner(estimator);
   refiner.SetTotalErrorFraction(0.7);
   
   // 10. Assemble the linear form for the right hand side vector.
   LinearForm b(&fespace);

   FunctionCoefficient rhs(fcoeff);
   b.AddDomainIntegrator(new DomainLFIntegrator(rhs));

   // Add the desired value for n.Grad(u) on the Neumann boundary
   b.AddBoundaryIntegrator(new BoundaryLFIntegrator(nbcCoef), nbc_bdr);

   // Add the desired value for n.Grad(u) + a*u on the Robin boundary
   b.AddBoundaryIntegrator(new BoundaryLFIntegrator(rbcBCoef_top), rbc_bdr_top);
   b.AddBoundaryIntegrator(new BoundaryLFIntegrator(rbcBCoef_right), rbc_bdr_right);
   b.AddBoundaryIntegrator(new BoundaryLFIntegrator(rbcBCoef_bottom), rbc_bdr_bottom);

   // 11. The main AMR loop. In each iteration we solve the problem on the
   //     current mesh, visualize the solution, and refine the mesh.
   for (int it = 0; ; it++)
     {
       int cdofs = fespace.GetTrueVSize();
       cout << "\nAMR iteration " << it << endl;
       cout << "Number of unknowns: " << cdofs << endl;

       a.Assemble();

       b.Assemble();

       // 12. Construct the linear system.
       OperatorPtr A;
       Vector B, X;
       a.FormLinearSystem(ess_tdof_list, u, b, A, X, B);

#ifndef MFEM_USE_SUITESPARSE
       // 13. Define a simple symmetric Gauss-Seidel preconditioner and use it to
       //     solve the system AX=B with PCG in the symmetric case, and GMRES in the
       //     non-symmetric one.
       {
	 GSSmoother M((SparseMatrix&)(*A));
	 //DSmoother M((SparseMatrix&)(*A));
	 PCG(*A, M, B, X, 1, 5000, 1e-12, 0.0);
       }
#else
       // 13. If MFEM was compiled with SuiteSparse, use UMFPACK to solve the
       //     system.
       UMFPackSolver umf_solver;
       umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
       umf_solver.SetOperator(*A);
       umf_solver.Mult(B, X);
#endif

       // 14. Recover the grid function corresponding to U. This is the local finite
       //     element solution.
       a.RecoverFEMSolution(X, b, u);

       // 15. Send the solution by socket to a GLVis server.
       if (visualization)
	 {
	   string title_str = "H1";
	   char vishost[] = "localhost";
	   int  visport   = 19916;
	   socketstream sol_sock(vishost, visport);
	   sol_sock.precision(8);
	   sol_sock << "solution\n" << *mesh << u
		    << "window_title '" << title_str << " Solution'"
		    << " keys 'mmc'" << flush;
	 }
       
       // 16. Compute the various boundary integrals.
       mfem::out << endl
		 << "Verifying boundary conditions" << endl
		 << "=============================" << endl;
       {
	 // Integrate n.Grad(u) on the homogeneous Neumann boundary and compare to
	 // the expected value of zero.
	 double err, avg = IntegrateBC(u, nbc_bdr, 1.0, 0.0, nbc_val, err);

	 bool hom_nbc = true;
	 mfem::out << "Average of n.Grad(u) on Gamma_nbc:\t"
		   << avg << ", \t"
		   << (hom_nbc ? "absolute" : "relative")
		   << " error " << err << endl;
       }
       {
	 // Integrate n.Grad(u) + a * u on the Robin boundary and compare to the
	 // expected value.
	 double err, avg = IntegrateBC(u, rbc_bdr_top, cq[0], c_top, gn_top, err);

	 bool hom_rbc = (gn_top == 0.0);
	 err /=  hom_rbc ? 1.0 : fabs(gn_top);
	 mfem::out << "Average of q n.Grad(u)+c*u on Gamma_rbc_top:\t"
		   << avg << ", \t"
		   << (hom_rbc ? "absolute" : "relative")
		   << " error " << err << endl;
       }
       {
	 // Integrate n.Grad(u) + a * u on the Robin boundary and compare to the
	 // expected value.
	 double err, avg = IntegrateBC(u, rbc_bdr_right, cq[0], c_right, gn_right, err);

	 bool hom_rbc = (gn_right == 0.0);
	 err /=  hom_rbc ? 1.0 : fabs(gn_right);
	 mfem::out << "Average of q n.Grad(u)+c*u on Gamma_rbc_right:\t"
		   << avg << ", \t"
		   << (hom_rbc ? "absolute" : "relative")
		   << " error " << err << endl;
       }
       {
	 // Integrate n.Grad(u) + a * u on the Robin boundary and compare to the
	 // expected value.
	 double err, avg = IntegrateBC(u, rbc_bdr_bottom, cq[0], c_bottom, gn_bottom, err);

	 bool hom_rbc = (gn_bottom == 0.0);
	 err /=  hom_rbc ? 1.0 : fabs(gn_bottom);
	 mfem::out << "Average of q n.Grad(u)+c*u on Gamma_rbc_bottom:\t"
		   << avg << ", \t"
		   << (hom_rbc ? "absolute" : "relative")
		   << " error " << err << endl;
       }

       if (cdofs > max_dofs)
	 {
	   cout << "Reached the maximum number of dofs. Stop." << endl;
	   break;
	 }

       // 17. Call the refiner to modify the mesh. The refiner calls the error
       //     estimator to obtain element errors, then it selects elements to be
       //     refined and finally it modifies the mesh. The Stop() method can be
       //     used to determine if a stopping criterion was met.
       refiner.Apply(*mesh);
       if (refiner.Stop())
	 {
	   cout << "Stopping criterion satisfied. Stop." << endl;
	   break;
	 }

       // 18. Update the space to reflect the new state of the mesh. Also,
       //     interpolate the solution x so that it lies in the new space but
       //     represents the same function. This saves solver iterations later
       //     since we'll have a good initial guess of x in the next step.
       //     Internally, FiniteElementSpace::Update() calculates an
       //     interpolation matrix which is then used by GridFunction::Update().
       fespace.Update();
       u.Update();

       // 19. Inform also the bilinear and linear forms that the space has
       //     changed.
       a.Update();
       b.Update();
     } // end of AMR loop

   // 20. Save the refined mesh and the solution. This output can be viewed
   //     later using GLVis: "glvis -m refined.mesh -g sol.gf".
   {
      ofstream mesh_ofs("refined.mesh");
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);
      ofstream sol_ofs("sol.gf");
      sol_ofs.precision(8);
      u.Save(sol_ofs);
   }

   // 21. Free the used memory.
   delete mesh;

   return 0;
}

Mesh * GenerateSerialMesh(const double sx, const double sy)
{
  Mesh mesh = Mesh::MakeCartesian2D(1, 1, Element::Type::QUADRILATERAL, true, sx, sy);

  for (int i=0; i<mesh.GetNBE(); ++i)
    {
      Element *bdryElem = mesh.GetBdrElement(i);
      Array<int> v;
      bdryElem->GetVertices(v);
      MFEM_VERIFY(v.Size() == 2, "");

      const double mx = 0.5 * (mesh.GetVertex(v[0])[0] + mesh.GetVertex(v[1])[0]);
      const double my = 0.5 * (mesh.GetVertex(v[0])[1] + mesh.GetVertex(v[1])[1]);

      if (mx == 0.0)
	mesh.SetBdrAttribute(i, BoundarySegment::left);
      else if (my == 0.0)
	mesh.SetBdrAttribute(i, BoundarySegment::bottom);
      else if (mx == sx)
	mesh.SetBdrAttribute(i, BoundarySegment::right);
      else if (my == sy)
	mesh.SetBdrAttribute(i, BoundarySegment::top);
      else
	{
	  MFEM_ABORT("attribute not set");
	}
    }

  MFEM_VERIFY(mesh.bdr_attributes.Max() == 4, "");

  return new Mesh(mesh);
}

double IntegrateBC(const GridFunction &x, const Array<int> &bdr,
                   double alpha, double beta, double gamma,
                   double &err)
{
   double nrm = 0.0;
   double avg = 0.0;
   err = 0.0;

   const bool a_is_zero = alpha == 0.0;
   const bool b_is_zero = beta == 0.0;

   const FiniteElementSpace &fes = *x.FESpace();
   MFEM_ASSERT(fes.GetVDim() == 1, "");
   Mesh &mesh = *fes.GetMesh();
   Vector shape, loc_dofs, w_nor;
   DenseMatrix dshape;
   Array<int> dof_ids;
   for (int i = 0; i < mesh.GetNBE(); i++)
   {
      if (bdr[mesh.GetBdrAttribute(i)-1] == 0) { continue; }

      FaceElementTransformations *FTr = mesh.GetBdrFaceTransformations(i);
      if (FTr == nullptr) { continue; }

      const FiniteElement &fe = *fes.GetFE(FTr->Elem1No);
      MFEM_ASSERT(fe.GetMapType() == FiniteElement::VALUE, "");
      const int int_order = 2*fe.GetOrder() + 3;
      const IntegrationRule &ir = IntRules.Get(FTr->FaceGeom, int_order);

      fes.GetElementDofs(FTr->Elem1No, dof_ids);
      x.GetSubVector(dof_ids, loc_dofs);
      if (!a_is_zero)
      {
         const int sdim = FTr->Face->GetSpaceDim();
         w_nor.SetSize(sdim);
         dshape.SetSize(fe.GetDof(), sdim);
      }
      if (!b_is_zero)
      {
         shape.SetSize(fe.GetDof());
      }
      for (int j = 0; j < ir.GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir.IntPoint(j);
         IntegrationPoint eip;
         FTr->Loc1.Transform(ip, eip);
         FTr->Face->SetIntPoint(&ip);
         double face_weight = FTr->Face->Weight();
         double val = 0.0;
         if (!a_is_zero)
         {
            FTr->Elem1->SetIntPoint(&eip);
            fe.CalcPhysDShape(*FTr->Elem1, dshape);
            CalcOrtho(FTr->Face->Jacobian(), w_nor);
            val += alpha * dshape.InnerProduct(w_nor, loc_dofs) / face_weight;
         }
         if (!b_is_zero)
         {
            fe.CalcShape(eip, shape);
            val += beta * (shape * loc_dofs);
         }

         // Measure the length of the boundary
         nrm += ip.weight * face_weight;

         // Integrate alpha * n.Grad(x) + beta * x
         avg += val * ip.weight * face_weight;

         // Integrate |alpha * n.Grad(x) + beta * x - gamma|^2
         val -= gamma;
         err += (val*val) * ip.weight * face_weight;
      }
   }

   // Normalize by the length of the boundary
   if (std::abs(nrm) > 0.0)
   {
      err /= nrm;
      avg /= nrm;
   }

   // Compute l2 norm of the error in the boundary condition (negative
   // quadrature weights may produce negative 'err')
   err = (err >= 0.0) ? sqrt(err) : -sqrt(-err);

   // Return the average value of alpha * n.Grad(x) + beta * x
   return avg;
}

int GetSubdomain(const Vector & x)
{
  int ix{0};
  int iy{0};

  for (int i=1; i<4; ++i)
    {
      if (x[0] >= cx[i])
	ix = i;
      else
	break;
    }

  for (int i=1; i<7; ++i)
    {
      if (x[1] >= cy[i])
	iy = i;
      else
	break;
    }

  int k = 5;
  if (iy == 0 || iy == 6 || ix == 3)
    k = 1;
  else if (ix == 1 && iy > 0 && iy < 5)
    k = 4;
  else if (ix == 0 && iy == 3)
    k = 3;
  else if (ix == 0 && (iy == 2 || iy == 4))
    k = 2;

  return k;
}

double pcoeff(const Vector & x)
{
  return cp[GetSubdomain(x) - 1];
}

double qcoeff(const Vector & x)
{
  return cq[GetSubdomain(x) - 1];
}

void diffCoeff(const Vector &x, Vector &d)
{
  d = 0.0;
  d(0) = pcoeff(x);
  d(1) = qcoeff(x);
}

double fcoeff(const Vector & x)
{
  return cf[GetSubdomain(x) - 1];
}
