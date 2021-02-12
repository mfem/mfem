//                       MFEM Example 1 Ortho - Parallel Version
//
// Compile with: make ex1p
//
// Sample runs:  mpirun -np 4 ex1p-orth
//               mpirun -np 4 ex1p-orth -c 2
//               mpirun -np 4 ex1p-orth -c 3
//               mpirun -np 4 ex1p-orth -c 4 -rs 4
//               mpirun -np 4 ex1p-orth -c 6
//               mpirun -np 4 ex1p-orth -c 7
//               mpirun -np 4 ex1p-orth -c 8
//               mpirun -np 4 ex1p-orth -c 9
//               mpirun -np 4 ex1p-orth -c 10 -rs 4
//               mpirun -np 4 ex1p-orth -c 11 -n2 2 -rs 3
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Laplace problem
//               -Delta u = 1 with homogeneous Dirichlet boundary conditions in
//               a variety of orthogonal coordinate systems.  The discretization
//               is identical to that used in example 1 but here we use
//               non-trivial coefficients in the Laplace operator and the right-
//               hand-side vector to mimic a curvilinear coordinate system.  We
//               also transform the mesh and solve the standard Laplace problem
//               on the transformed mesh to compare the solutions.
//
//               The example highlights the use of standard differential
//               operators to mimic the behavior of more exotic operators
//               derived from coordinate transformations.
//
//               We recommend viewing Example 1 and Example 11-cyl before
//               viewing this example.
//
//               Note: the notation used in this code comes from the Wikipedia
//               page https://en.wikipedia.org/wiki/Orthogonal_coordinates.
//               There are, however, minor differences made to ensure that
//               we use right-handed coordinate systems in all cases.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Enumeration listing all supported 2D orthogonal coordinate systems
enum CoordSys {POLAR = 1, PARABOLIC_CYL, ELLIPTIC, BIPOLAR,
               CYLINDRICAL, SPHERICAL, PARABOLIC, PROLATE_SPHEROIDAL,
               OBLATE_SPHEROIDAL, TOROIDAL, BISPHERICAL
              };

static CoordSys coords_ = (CoordSys)1;
static double q1_min_ = NAN;
static double q1_max_ = NAN;
static double q2_min_ = NAN;
static double q2_max_ = NAN;
static double a_ = 1.0;

// Set default values for coordinate ranges q1_min_, q1_max_, q2_min_,
// and q2_max_ based on the selected coordinate system, coords_.
void SetRanges();

// Shift the mesh so that the origin is at (q1_min_, q2_min_)
void trans1(const Vector &u, Vector &x)
{
   x.SetSize(2);
   x[0] = u[0] + q1_min_;
   x[1] = u[1] + q2_min_;
}

// Apply conformal mapping from cartesian coordinates to the
// orthogonal coordinate system specified by coords_.
void trans(const Vector &u, Vector &x);

// Returns one of the three coordinate scale factors h_i describing
// the orthogonal coordinate system.
class OrthoCoef : public Coefficient
{
private:
   int ind_;

public:
   OrthoCoef(int index) : ind_(index) {}

   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip);
};

// Integration weight coefficient h_1 * h_2 * h_3
class OrthoWeightCoef : public Coefficient
{
private:
   Coefficient &h1Coef_;
   Coefficient &h2Coef_;
   Coefficient &h3Coef_;

public:
   OrthoWeightCoef(Coefficient &h1Coef,
                   Coefficient &h2Coef,
                   Coefficient &h3Coef)
      : h1Coef_(h1Coef),
        h2Coef_(h2Coef),
        h3Coef_(h3Coef) {}

   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip);
};

// Matrix-valued coefficient appearing in the weak form of the
// Laplacian operator.
class OrthoMatrixCoef : public MatrixCoefficient
{
private:
   Coefficient &h1Coef_;
   Coefficient &h2Coef_;
   Coefficient &h3Coef_;

public:
   OrthoMatrixCoef(Coefficient &h1Coef,
                   Coefficient &h2Coef,
                   Coefficient &h3Coef)
      : MatrixCoefficient(2),
        h1Coef_(h1Coef),
        h2Coef_(h2Coef),
        h3Coef_(h3Coef)
   {}

   virtual void Eval(DenseMatrix &K, ElementTransformation &T,
                     const IntegrationPoint &ip);

};

// Radial weight factor to distinguish volumes of revolution from
// extruded volumes
class RhoCoef : public Coefficient
{
public:
   RhoCoef() {}

   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
   {
      double u_data[2];
      Vector u(u_data, 2);
      T.Transform(ip, u);
      return u[0];
   }

};

// Matrix-valued radial weight factor to distinguish volumes of
// revolution from extruded volumes within the Laplacian operator
class RhoMatrixCoef : public MatrixCoefficient
{
public:
   RhoMatrixCoef()
      : MatrixCoefficient(2)
   {}

   virtual void Eval(DenseMatrix &K, ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      double u_data[2];
      Vector u(u_data, 2);
      T.Transform(ip, u);

      K.SetSize(2);
      K(0,0) = u[0];
      K(0,1) = 0.0;
      K(1,0) = 0.0;
      K(1,1) = u[0];
   }

};

static bool static_cond_ = false;
static bool pa_ = false;

// Ensure that m >= 3 if a periodic mesh has been selected
void AdjustDimensions(int &m, int &n, int & rs, int & rp);

// Setup and solve the Poisson problem with boundary conditions
// appropriate to the selected coordinate system.
void Poisson(ParMesh &pmesh, ParFiniteElementSpace &fespace,
             MatrixCoefficient &LCoef, Coefficient &MCoef,
             ParGridFunction &x);

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   int coords = 1;
   int n1 = 1;
   int n2 = 1;
   int el_type_flag = 1;
   Element::Type el_type;
   int ser_ref_levels = 2;
   int par_ref_levels = 1;
   int morder = 2;
   int order = 2;
   const char *device_config = "cpu";
   bool comp = true;
   bool discont = false;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&coords, "-c", "--coord-sys",
                  "Coordinate system: 1 - POLAR, 2 - PARABOLIC_CYL, "
                  "3 - ELLIPTIC, 4 - BIPOLAR, 5 - CYLINDRICAL, 6 - SPHERICAL, "
                  "7 - PARABOLIC, 8 - PROLATE_SPHEROIDAL, "
                  "9 - OBLATE_SPHEROIDAL, 10 - TOROIDAL, 11 - BISPHERICAL");
   args.AddOption(&n1, "-n1", "--num-elements-1",
                  "Number of elements in q1-direction.");
   args.AddOption(&n2, "-n2", "--num-elements-2",
                  "Number of elements in q2-direction.");
   args.AddOption(&q1_min_, "-q1-min", "--q1-minimum-1",
                  "Minimum value of q1 coordinate.");
   args.AddOption(&q1_max_, "-q1-max", "--q1-maximum-1",
                  "Maximum value of q1 coordinate.");
   args.AddOption(&q2_min_, "-q2-min", "--q2-minimum-1",
                  "Minimum value of q2 coordinate.");
   args.AddOption(&q2_max_, "-q2-max", "--q2-maximum-1",
                  "Maximum value of q2 coordinate.");
   args.AddOption(&a_, "-a", "--scale-parameter",
                  "Scale paramter appearing in some of the transformations.");
   args.AddOption(&el_type_flag, "-e", "--element-type",
                  "Element type: 0 - Triangle, 1 - Quadrilateral.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&morder, "-mo", "--mesh-order",
                  "Order (polynomial degree) for the mesh geometry.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&comp, "-comp", "--compare", "-no-comp",
                  "--no-compare", "Compare to standard curved mesh solution.");
   args.AddOption(&static_cond_, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&pa_, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
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
      MPI_Finalize();
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // Cast the user input to the enumerated type and set the appropriate
   // coordinate ranges.
   coords_ = (CoordSys)coords;
   SetRanges();

   // The output mesh could be quadrilaterals or triangles
   el_type = (el_type_flag == 0) ? Element::TRIANGLE : Element::QUADRILATERAL;
   if (el_type != Element::TRIANGLE && el_type != Element::QUADRILATERAL)
   {
      cout << "Unsupported element type" << endl;
      exit(1);
   }

   if (coords_ == BIPOLAR || coords_ == TOROIDAL)
   {
      AdjustDimensions(n1, n2, ser_ref_levels, par_ref_levels);
   }
   else if (coords_ == POLAR || coords_ == ELLIPTIC)
   {
      AdjustDimensions(n2, n1, ser_ref_levels, par_ref_levels);
   }

   // 3. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (myid == 0) { device.Print(); }

   // 3. Prepare a rectangular mesh with the desired dimensions and element
   //    type.
   Mesh *mesh = new Mesh(n1, n2, el_type, false,
                         q1_max_ - q1_min_, q2_max_ - q2_min_);
   mesh->Transform(trans1);
   int dim = mesh->Dimension();

   if (coords_ == POLAR || coords_ == ELLIPTIC || coords_ == BIPOLAR ||
       coords_ == TOROIDAL)
   {
      // 4. Stitch the ends of the mesh together
      discont = true;
      mesh->SetCurvature(1, discont, 2, Ordering::byVDIM);
      Array<int> v2v(mesh->GetNV());
      for (int i = 0; i < v2v.Size(); i++)
      {
         v2v[i] = i;
      }

      if (coords_ == POLAR || coords_ == ELLIPTIC)
      {
         // identify vertices at the extremes of the mesh in the q2 direction
         for (int i=0; i<n1 + 1; i++)
         {
            v2v[v2v.Size() - n1 - 1 + i] = i;
         }
      }
      else if (coords_ == BIPOLAR || coords_ == TOROIDAL)
      {
         // identify vertices at the extremes of the mesh in the q1 direction
         for (int i=0; i<n2 + 1; i++)
         {
            v2v[(n1 + 1) * i + n1] = (n1 + 1) * i;
         }
      }
      // renumber elements
      for (int i = 0; i < mesh->GetNE(); i++)
      {
         Element *el = mesh->GetElement(i);
         int *v = el->GetVertices();
         int nv = el->GetNVertices();
         for (int j = 0; j < nv; j++)
         {
            v[j] = v2v[v[j]];
         }
      }
      // renumber boundary elements
      for (int i = 0; i < mesh->GetNBE(); i++)
      {
         Element *el = mesh->GetBdrElement(i);
         int *v = el->GetVertices();
         int nv = el->GetNVertices();
         for (int j = 0; j < nv; j++)
         {
            v[j] = v2v[v[j]];
         }
      }
      mesh->RemoveUnusedVertices();
      mesh->RemoveInternalBoundaries();
      mesh->FinalizeTopology();
   }

   // 5. Refine the serial mesh on all processors to increase the resolution.
   {
      for (int l = 0; l < ser_ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh_ortho = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   {
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh_ortho->UniformRefinement();
      }
   }

   // 7. Create a standard curved mesh to describe the same geometry
   ParMesh *pmesh_curved = new ParMesh(*pmesh_ortho);
   pmesh_curved->SetCurvature(morder, discont);
   pmesh_curved->Transform(trans);

   // 8. Define a parallel finite element space on the parallel mesh. Here we
   //    use continuous Lagrange finite elements of the specified order.
   H1_FECollection fec(order, dim);
   ParFiniteElementSpace fespace_ortho(pmesh_ortho, &fec);
   HYPRE_Int size = fespace_ortho.GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   // 9. Declare the coordinate scaling factors and the coefficients
   //    needed to form the mass matrix and Laplacian.
   OrthoCoef h1Coef(0);
   OrthoCoef h2Coef(1);
   OrthoCoef h3Coef(2);
   OrthoWeightCoef WCoef(h1Coef, h2Coef, h3Coef);
   OrthoMatrixCoef LCoef(h1Coef, h2Coef, h3Coef);

   // 10. Setup and solve the Poisson problem on the cartesian mesh
   ParGridFunction x_ortho(&fespace_ortho); x_ortho = 0.0;
   Poisson(*pmesh_ortho, fespace_ortho, LCoef, WCoef, x_ortho);

   // 11. Save the refined mesh and the solution in parallel. This output can
   //     be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
   {
      ostringstream mesh_name, sol_name;
      mesh_name << "mesh_ortho." << setfill('0') << setw(6) << myid;
      sol_name << "sol_ortho." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh_ortho->Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      x_ortho.Save(sol_ofs);
   }

   // 12. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *pmesh_ortho << x_ortho << flush
               << "window_title 'Straight Mesh'"
               << "keys m\n";

      socketstream mix_sol_sock(vishost, visport);
      mix_sol_sock << "parallel " << num_procs << " " << myid << "\n";
      mix_sol_sock.precision(8);
      mix_sol_sock << "solution\n" << *pmesh_curved << x_ortho << flush
                   << "window_title 'Straight Solution - Curved Mesh' "
                   << "window_geometry 400 0 400 350"
                   << "keys m\n";
   }

   // 13. Compare to a solution computed on the corresponding curved mesh.
   if (comp)
   {
      ParFiniteElementSpace fespace_curved(pmesh_curved, &fec);
      ParGridFunction x_curved(&fespace_curved); x_curved = 0.0;

      double err = -1.0;
      GridFunctionCoefficient xCoef(&x_ortho);

      // 14. Setup and solve the Poisson problem on the cartesian mesh
      if (coords_ == POLAR || coords_ == PARABOLIC_CYL ||
          coords_ == ELLIPTIC || coords_ == BIPOLAR)
      {
         // These coordinate systems can be viewed as truly two-dimensional
         // or simply extruded into the third dimension and so they require
         // no special coefficients.
         DenseMatrix OneMat(2);
         OneMat = 0.0; OneMat(0,0) = 1.0; OneMat(1,1) = 1.0;
         MatrixConstantCoefficient OneCoef(OneMat);
         ConstantCoefficient oneCoef(1.0);
         Poisson(*pmesh_curved, fespace_curved, OneCoef, oneCoef, x_curved);

         // 15a. Measure the difference in the two solutions using an L2 norm.
         err = x_curved.ComputeL2Error(xCoef);
      }
      else
      {
         // The remaining coordinate systems are truly three-dimensional and
         // involve rotation about the second coordinate axis. Consequently,
         // they require a radial scale factor both in the mass matrix and the
         // Laplacian operator.
         RhoCoef rhoCoef;
         RhoMatrixCoef RhoCoef;
         Poisson(*pmesh_curved, fespace_curved, RhoCoef, rhoCoef, x_curved);

         // 15b. Measure the difference in the two solutions using an L2 norm.
         PowerCoefficient sqrtRhoCoef(rhoCoef, 0.5);
         ProductCoefficient rxoCoef(sqrtRhoCoef, xCoef);
         GridFunctionCoefficient xcCoef(&x_curved);
         ProductCoefficient rxcCoef(sqrtRhoCoef, xcCoef);
         ParGridFunction rx_curved(&fespace_curved);
         rx_curved.ProjectCoefficient(rxcCoef);
         err = rx_curved.ComputeL2Error(rxoCoef);
      }

      if (myid == 0)
      {
         cout << "\n|| u_curved - u_ortho ||_{L^2} = " << err << '\n' << endl;
      }

      // 15. Save the refined mesh and the solution in parallel. This output can
      //     be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
      {
         ostringstream mesh_name, sol_name;
         mesh_name << "mesh_std." << setfill('0') << setw(6) << myid;
         sol_name << "sol_std." << setfill('0') << setw(6) << myid;

         ofstream mesh_ofs(mesh_name.str().c_str());
         mesh_ofs.precision(8);
         pmesh_curved->Print(mesh_ofs);

         ofstream sol_ofs(sol_name.str().c_str());
         sol_ofs.precision(8);
         x_curved.Save(sol_ofs);
      }

      // 16. Send the solution by socket to a GLVis server.
      if (visualization)
      {
         char vishost[] = "localhost";
         int  visport   = 19916;

         socketstream cart_sol_sock(vishost, visport);
         cart_sol_sock << "parallel " << num_procs << " " << myid << "\n";
         cart_sol_sock.precision(8);
         cart_sol_sock << "solution\n" << *pmesh_curved << x_curved << flush
                       << "window_title 'Curved Mesh' "
                       << "window_geometry 800 0 400 350"
                       << "keys m\n";
      }

      delete pmesh_curved;
   }

   // 17. Free the used memory.
   delete pmesh_ortho;

   MPI_Finalize();

   return 0;
}

void AdjustDimensions(int &m, int &n, int & rs, int & rp)
{
   while (m < 3 && rs + rp > 0)
   {
      m *= 2;
      n *= 2;
      (rs > 0) ? rs-- : rp--;
   }
   if (m < 3) { m = 3; }
}

void Poisson(ParMesh &pmesh, ParFiniteElementSpace &fespace,
             MatrixCoefficient &LCoef, Coefficient &MCoef,
             ParGridFunction &x)
{
   // 8. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs. In this example, the boundary conditions are defined
   //    by marking all the boundary attributes from the mesh as essential
   //    (Dirichlet) and converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   if (pmesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh.bdr_attributes.Max());
      ess_bdr = 1;
      if (coords_ == CYLINDRICAL)
      {
         ess_bdr[3] = 0;
      }
      else if (coords_ == SPHERICAL || coords_ == PROLATE_SPHEROIDAL ||
               coords_ == OBLATE_SPHEROIDAL)
      {
         ess_bdr[0] = 0;
         ess_bdr[2] = 0;
      }
      else if (coords_ == PARABOLIC)
      {
         ess_bdr[0] = 0;
         ess_bdr[3] = 0;
      }
      else if (coords_ == BISPHERICAL)
      {
         ess_bdr[1] = 0;
         ess_bdr[3] = 0;
      }
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 9. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system, which in this case is
   //    (1,phi_i) where phi_i are the basis functions in fespace.
   ParLinearForm b(&fespace);
   b.AddDomainIntegrator(new DomainLFIntegrator(MCoef));
   b.Assemble();

   // 10. Define the solution vector x as a parallel finite element grid function
   //     corresponding to fespace. Initialize x with initial guess of zero,
   //     which satisfies the boundary conditions.
   // ParGridFunction x(fespace);
   // x = 0.0;

   // 11. Set up the parallel bilinear form a(.,.) on the finite element space
   //     corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //     domain integrator.
   ParBilinearForm a(&fespace);
   if (pa_) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a.AddDomainIntegrator(new DiffusionIntegrator(LCoef));

   // 12. Assemble the parallel bilinear form and the corresponding linear
   //     system, applying any necessary transformations such as: parallel
   //     assembly, eliminating boundary conditions, applying conforming
   //     constraints for non-conforming AMR, static condensation, etc.
   if (static_cond_) { a.EnableStaticCondensation(); }
   a.Assemble();

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   // 13. Solve the linear system A X = B.
   //     * With full assembly, use the BoomerAMG preconditioner from hypre.
   //     * With partial assembly, use Jacobi smoothing, for now.
   Solver *prec = NULL;
   if (pa_)
   {
      if (UsesTensorBasis(fespace))
      {
         prec = new OperatorJacobiSmoother(a, ess_tdof_list);
      }
   }
   else
   {
      prec = new HypreBoomerAMG;
   }
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(1);
   if (prec) { cg.SetPreconditioner(*prec); }
   cg.SetOperator(*A);
   cg.Mult(B, X);
   delete prec;

   // 14. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   a.RecoverFEMSolution(X, b, x);
}

void SetRanges()
{
   switch (coords_)
   {
      case POLAR:
         if (isnan(q1_min_)) { q1_min_ = 0.5; }
         if (isnan(q1_max_)) { q1_max_ = 4.0; }
         if (isnan(q2_min_)) { q2_min_ = -M_PI; }
         if (isnan(q2_max_)) { q2_max_ =  M_PI; }
         break;
      case PARABOLIC_CYL:
         if (isnan(q1_min_)) { q1_min_ = -4.0; }
         if (isnan(q1_max_)) { q1_max_ = 4.0; }
         if (isnan(q2_min_)) { q2_min_ = 0.5; }
         if (isnan(q2_max_)) { q2_max_ = 4.0; }
         break;
      case ELLIPTIC:
         if (isnan(q1_min_)) { q1_min_ = 0.5; }
         if (isnan(q1_max_)) { q1_max_ = 2.0; }
         if (isnan(q2_min_)) { q2_min_ = -M_PI; }
         if (isnan(q2_max_)) { q2_max_ =  M_PI; }
         break;
      case BIPOLAR:
         if (isnan(q1_min_)) { q1_min_ = -M_PI; }
         if (isnan(q1_max_)) { q1_max_ =  M_PI; }
         if (isnan(q2_min_)) { q2_min_ = 0.5; }
         if (isnan(q2_max_)) { q2_max_ = 4.0; }
         break;
      case CYLINDRICAL:
         if (isnan(q1_min_)) { q1_min_ = 0.0; }
         if (isnan(q1_max_)) { q1_max_ = 4.0; }
         if (isnan(q2_min_)) { q2_min_ = 0.0; }
         if (isnan(q2_max_)) { q2_max_ = 4.0; }
         break;
      case SPHERICAL:
         if (isnan(q1_min_)) { q1_min_ = 0.5; }
         if (isnan(q1_max_)) { q1_max_ = 4.0; }
         if (isnan(q2_min_)) { q2_min_ = 0.0; }
         if (isnan(q2_max_)) { q2_max_ = M_PI; }
         break;
      case PARABOLIC:
         if (isnan(q1_min_)) { q1_min_ = 0.0; }
         if (isnan(q1_max_)) { q1_max_ = 4.0; }
         if (isnan(q2_min_)) { q2_min_ = 0.0; }
         if (isnan(q2_max_)) { q2_max_ = 4.0; }
         break;
      case PROLATE_SPHEROIDAL:
         if (isnan(q1_min_)) { q1_min_ = 0.5; }
         if (isnan(q1_max_)) { q1_max_ = 2.0; }
         if (isnan(q2_min_)) { q2_min_ = 0.0; }
         if (isnan(q2_max_)) { q2_max_ = M_PI; }
         break;
      case OBLATE_SPHEROIDAL:
         if (isnan(q1_min_)) { q1_min_ = 0.5; }
         if (isnan(q1_max_)) { q1_max_ = 2.0; }
         if (isnan(q2_min_)) { q2_min_ = -0.5 * M_PI; }
         if (isnan(q2_max_)) { q2_max_ =  0.5 * M_PI; }
         break;
      case TOROIDAL:
         if (isnan(q1_min_)) { q1_min_ = -M_PI; }
         if (isnan(q1_max_)) { q1_max_ =  M_PI; }
         if (isnan(q2_min_)) { q2_min_ = 0.5; }
         if (isnan(q2_max_)) { q2_max_ = 4.0; }
         break;
      case BISPHERICAL:
         if (isnan(q1_min_)) { q1_min_ = 0.0; }
         if (isnan(q1_max_)) { q1_max_ = M_PI; }
         if (isnan(q2_min_)) { q2_min_ = 0.5; }
         if (isnan(q2_max_)) { q2_max_ = 4.0; }
         break;
   }
}

void trans(const Vector &u, Vector &x)
{
   x.SetSize(2);

   switch (coords_)
   {
      case POLAR:
         x[0] = u[0] * cos(u[1]);
         x[1] = u[0] * sin(u[1]);
         break;
      case PARABOLIC_CYL:
         x[0] = 0.5 * (u[0] * u[0] - u[1] * u[1]);
         x[1] = u[0] * u[1];
         break;
      case ELLIPTIC:
         x[0] = a_ * cosh(u[0]) * cos(u[1]);
         x[1] = a_ * sinh(u[0]) * sin(u[1]);
         break;
      case BIPOLAR:
      {
         double den = (cosh(u[1]) - cos(u[0]));
         x[0] = a_ * sinh(u[1]) / den;
         x[1] = a_ * sin(u[0]) / den;
      }
      break;
      case CYLINDRICAL:
      {
         x[0] = u[0];
         x[1] = u[1];
      }
      break;
      case SPHERICAL:
      {
         x[0] = u[0] * sin(u[1]);
         x[1] = -u[0] * cos(u[1]);
      }
      break;
      case PARABOLIC:
      {
         x[0] = u[0] * u[1];
         x[1] = -0.5 * (u[0] * u[0] - u[1] * u[1]);
      }
      break;
      case PROLATE_SPHEROIDAL:
      {
         x[0] = a_ * sinh(u[0]) * sin(u[1]);
         x[1] = -a_ * cosh(u[0]) * cos(u[1]);
      }
      break;
      case OBLATE_SPHEROIDAL:
      {
         x[0] = a_ * cosh(u[0]) * cos(u[1]);
         x[1] = a_ * sinh(u[0]) * sin(u[1]);
      }
      break;
      case TOROIDAL:
      {
         double den = (cosh(u[1]) - cos(u[0]));
         x[0] = a_ * sinh(u[1]) / den;
         x[1] = a_ * sin(u[0]) / den;
      }
      break;
      case BISPHERICAL:
      {
         double den = (cosh(u[1]) + cos(u[0]));
         x[0] = a_ * sin(u[0]) / den;
         x[1] = a_ * sinh(u[1]) / den;
      }
      break;
   }
}

double OrthoCoef::Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
{
   double u_data[2];
   Vector u(u_data, 2);
   T.Transform(ip, u);

   switch (coords_)
   {
      case POLAR:
         switch (ind_)
         {
            case 0:
               return 1.0;
            case 1:
               return u[0];
            case 2:
               return 1.0;
            default:
               return 0.0;
         }
         break;
      case PARABOLIC_CYL:
         switch (ind_)
         {
            case 0:
               return sqrt(u[0] * u[0] + u[1] * u[1]);
            case 1:
               return sqrt(u[0] * u[0] + u[1] * u[1]);
            case 2:
               return 1.0;
            default:
               return 0.0;
         }
         break;
      case ELLIPTIC:
         switch (ind_)
         {
            case 0:
               return a_ * sqrt(pow(sinh(u[0]), 2) + pow(sin(u[1]), 2));
            case 1:
               return a_ * sqrt(pow(sinh(u[0]), 2) + pow(sin(u[1]), 2));
            case 2:
               return 1.0;
            default:
               return 0.0;
         }
         break;
      case BIPOLAR:
      {
         double den = (cosh(u[1]) - cos(u[0]));

         switch (ind_)
         {
            case 0:
               return a_ / den;
            case 1:
               return a_ / den;
            case 2:
               return 1.0;
            default:
               return 0.0;
         }
         break;
      }
      case CYLINDRICAL:
         switch (ind_)
         {
            case 0:
               return 1.0;
            case 1:
               return 1.0;
            case 2:
               return u[0];
            default:
               return 0.0;
         }
         break;
      case SPHERICAL:
         switch (ind_)
         {
            case 0:
               return 1.0;
            case 1:
               return u[0];
            case 2:
               return u[0] * sin(u[1]);
            default:
               return 0.0;
         }
         break;
      case PARABOLIC:
         switch (ind_)
         {
            case 0:
               return sqrt(u[0] * u[0] + u[1] * u[1]);
            case 1:
               return sqrt(u[0] * u[0] + u[1] * u[1]);
            case 2:
               return u[0] * u[1];
            default:
               return 0.0;
         }
         break;
      case PROLATE_SPHEROIDAL:
         switch (ind_)
         {
            case 0:
               return a_ * sqrt(pow(sinh(u[0]), 2) + pow(sin(u[1]), 2));
            case 1:
               return a_ * sqrt(pow(sinh(u[0]), 2) + pow(sin(u[1]), 2));
            case 2:
               return a_ * sinh(u[0]) * sin(u[1]);
            default:
               return 0.0;
         }
         break;
      case OBLATE_SPHEROIDAL:
         switch (ind_)
         {
            case 0:
               return a_ * sqrt(pow(sinh(u[0]), 2) + pow(sin(u[1]), 2));
            case 1:
               return a_ * sqrt(pow(sinh(u[0]), 2) + pow(sin(u[1]), 2));
            case 2:
               return a_ * cosh(u[0]) * cos(u[1]);
            default:
               return 0.0;
         }
         break;
      case TOROIDAL:
      {
         double den = (cosh(u[1]) - cos(u[0]));

         switch (ind_)
         {
            case 0:
               return a_ / den;
            case 1:
               return a_ / den;
            case 2:
               return a_ * sinh(u[1]) / den;
            default:
               return 0.0;
         }
         break;
      }
      case BISPHERICAL:
      {
         double den = (cosh(u[1]) + cos(u[0]));

         switch (ind_)
         {
            case 0:
               return a_ / den;
            case 1:
               return a_ / den;
            case 2:
               return a_ * sin(u[0]) / den;
            default:
               return 0.0;
         }
         break;
      }
   }
   return 0.0;
}

double OrthoWeightCoef::Eval(ElementTransformation &T,
                             const IntegrationPoint &ip)
{
   double h1 = h1Coef_.Eval(T, ip);
   double h2 = h2Coef_.Eval(T, ip);
   double h3 = h3Coef_.Eval(T, ip);

   return h1 * h2 * h3;
}

void OrthoMatrixCoef::Eval(DenseMatrix &K, ElementTransformation &T,
                           const IntegrationPoint &ip)
{
   double h1 = h1Coef_.Eval(T, ip);
   double h2 = h2Coef_.Eval(T, ip);
   double h3 = h3Coef_.Eval(T, ip);

   K.SetSize(2);
   K(0,0) = h2 * h3 / h1;
   K(0,1) = 0.0;
   K(1,0) = 0.0;
   K(1,1) = h1 * h3 / h2;
}
