//                       MFEM
//
// Compile with: make ex1p
//
// Sample runs:
// mpirun -np 1 ex1p -m ../../data/inline-quad.mesh  -rs 1 -vis -o 2

#include "../../mfem.hpp"
#include <fstream>
#include <iostream>
#include "distfunction.hpp"

using namespace std;
using namespace mfem;

double dist_fun(const Vector &x);
double dist_fun_level_set(const Vector &x);

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   /// 1. Parse command-line options.
   const char *mesh_file = "../data/square-disc.mesh";
   int order = 2;
   bool static_cond = false;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = true;
   int ser_ref_levels = 0;
   double dbc_val = 0.0;
   bool smooth = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();
   for (int lev = 0; lev < ser_ref_levels; lev++) { mesh.UniformRefinement(); }

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   {
      int par_ref_levels = 0;
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh.UniformRefinement();
      }
   }

   // 5. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order. If order < 1, we
   //    instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec_mesh;
   if (order > 0)
   {
      fec_mesh = new H1_FECollection(order, dim);
   }
   else
   {
      fec_mesh = new H1_FECollection(order = 1, dim);
   }
   ParFiniteElementSpace *pfespace_mesh = new ParFiniteElementSpace(&pmesh, fec_mesh, dim);
   pmesh.SetNodalFESpace(pfespace_mesh);
   ParGridFunction x_mesh(pfespace_mesh);
   pmesh.SetNodalGridFunction(&x_mesh);

   // 5. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order. If order < 1, we
   //    instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim);
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim);
   }
   ParFiniteElementSpace pfespace(&pmesh, fec);
   L2_FECollection fecl2 = L2_FECollection(0, dim);
   ParFiniteElementSpace pfesl2(&pmesh, &fecl2);
   cout << "Number of finite element unknowns: "
        << pfespace.GetTrueVSize() << endl;

   // 8. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   ParGridFunction x(&pfespace), dist(&pfespace);
   x = 0.0;
   ParGridFunction gfl2(&pfesl2);

   FunctionCoefficient dist_fun_coef(dist_fun);

   FunctionCoefficient dist_fun_level_coef(dist_fun_level_set);
   DistanceFunction dist_func(pmesh, order, 1.0);
   ParGridFunction &distance = dist_func.ComputeDistance(dist_fun_level_coef,
                                                         10, true);
   const ParGridFunction &src = dist_func.GetLastSourceGF(),
                         &diff_src = dist_func.GetLastDiffusedSourceGF();

   GradientCoefficient grad_u(dist_func.GetLastDiffusedSourceGF(), dim);

   dist.ProjectCoefficient(dist_fun_level_coef);
   distance.ProjectCoefficient(dist_fun_coef);

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock << "solution\n" << pmesh << distance << flush;
      sol_sock << "window_title 'Distance function'\n"
               << "window_geometry "
               << 0 << " " << 0 << " " << 350 << " " << 350 << "\n"
               << "keys Rjmpc" << endl;
   }

   int max_attr     = pmesh.attributes.Max();
   IntegrationRules IntRulesLo(0, Quadrature1D::GaussLobatto);

   // Set trim flag based on the distance field
   // 0 if completely in the domain
   // 1 if completely outside the domain
   // 2 if partially inside the domain
   Array<int> trim_flag(pmesh.GetNE());
   trim_flag = 0;
   Vector vals;
   for (int i = 0; i < pmesh.GetNE(); i++)
   {
      ElementTransformation *Tr = pmesh.GetElementTransformation(i);
      const IntegrationRule &ir =
         IntRulesLo.Get(pmesh.GetElementBaseGeometry(i), 4*Tr->OrderJ());
      dist.GetValues(i, ir, vals);

      int count = 0;
      for (int j = 0; j < ir.GetNPoints(); j++)
      {
         double val = vals(j);
         if (val <= 0.) { count++; }
      }
      if (count == ir.GetNPoints())
      {
         pmesh.SetAttribute(i, max_attr+1);
         trim_flag[i] = 1;
      }
      else if (count > 0)
      {
         trim_flag[i] = 2;
      }
   }

   Array<int> sbm_int_face;
   Array<int> sbm_int_face_el;
   Array<int> sbm_int_flag; // 1 if int face, 2 if bdr face
   // Get SBM faces
   for (int i = 0; i < pmesh.GetNumFaces(); i++)
   {
      FaceElementTransformations *tr;
      tr = pmesh.GetInteriorFaceTransformations (i);
      if (tr != NULL)
      {
         int ne1 = tr->Elem1No;
         int ne2 = tr->Elem2No;
         int te1 = trim_flag[ne1], te2 = trim_flag[ne2];
         if (te1 == 1 && te2 != 1)
         {
            sbm_int_face.Append(i);
            sbm_int_face_el.Append(ne2);
            sbm_int_flag.Append(1);
         }
         if (te2 == 1 && te1 != 1)
         {
            sbm_int_face.Append(i);
            sbm_int_face_el.Append(ne1);
            sbm_int_flag.Append(1);
         }
      }
   }

   for (int i = 0; i < pmesh.GetNBE(); i++)
   {
      FaceElementTransformations *tr;
      tr = pmesh.GetBdrFaceTransformations (i);
      if (tr != NULL)
      {
         int ne1 = tr->Elem1No;
         int te1 = trim_flag[ne1];
         if (te1 == 2)
         {
            sbm_int_face.Append(i);
            sbm_int_face_el.Append(ne1);
            sbm_int_flag.Append(2);
         }
      }
   }

   for (int i = 0; i < gfl2.Size(); i++)
   {
      gfl2(i) = trim_flag[i]*1.;
   }

   if (visualization && false)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock << "solution\n" << pmesh << gfl2 << flush;
      sol_sock << "window_title 'Element flags'\n"
               << "window_geometry "
               << 0 << " " << 0 << " " << 350 << " " << 350 << "\n"
               << "keys Rjmpc" << endl;
   }

   x = 0;


   // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking all
   //    the boundary attributes from the mesh as essential (Dirichlet) and
   //    converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   Array<int> ess_bdr(pmesh.bdr_attributes.Max());
   if (pmesh.bdr_attributes.Size())
   {
      ess_bdr = 1;
   }
   Array<int> dofs;
   // approach 2 - get dofs of all elements that are not marked
   // First get all essential vdofs at boundary
   Array<int> ess_vdofs_bdr;
   pfespace.GetEssentialVDofs(ess_bdr, ess_vdofs_bdr);

   // now get all dofs that are not part of the untrimmed elements
   Array<int> ess_vdofs_hole(ess_vdofs_bdr.Size());
   ess_vdofs_hole = -1;
   for (int e = 0; e < trim_flag.Size(); e++)
   {
      if (trim_flag[e] != 1)
      {
         pfespace.GetElementVDofs(e, dofs);
         for (int i = 0; i < dofs.Size(); i++) {
             ess_vdofs_hole[dofs[i]] = 0;
         }
      }
   }

   // now combine the two lists
   for (int i = 0; i < ess_vdofs_hole.Size(); i++)
   {
      if (ess_vdofs_bdr[i] == -1) { ess_vdofs_hole[i] = -1; }
   }

   for (int i = 0; i < ess_vdofs_hole.Size() ; i++) {
       ess_vdofs_hole[i] += 1;
   }

   pfespace.Synchronize(ess_vdofs_hole);

   for (int i = 0; i < ess_vdofs_hole.Size() ; i++) {
       ess_vdofs_hole[i] -= 1;
   }

   // convert to tdofs
   Array<int> ess_tdofs;
   pfespace.GetRestrictionMatrix()->BooleanMult(ess_vdofs_hole,
                                                ess_tdofs);
   pfespace.MarkerToList(ess_tdofs, ess_tdof_list);

   ParGridFunction x_dx(&pfespace), x_dy(&pfespace),
                   x_dx_dy(pfespace_mesh);
   distance.GetDerivative(1, 0, x_dx);
   distance.GetDerivative(1, 1, x_dy);
   // set vector magnitude
   for (int i = 0; i < x_dx.Size(); i++)
   {
      double dxv = x_dx(i),
             dyv = x_dy(i);
      double mag = dxv*dxv + dyv*dyv;
      if (mag > 0) { mag = pow(mag, 0.5); }
      x_dx(i) *= distance(i)/mag;
      x_dy(i) *= distance(i)/mag;
   }

   // copy to vector GridFunction
   for (int i = 0; i < x_dx_dy.Size()/dim; i++)
   {
      x_dx_dy(i) = x_dx(i);
      x_dx_dy(i + x_dx_dy.Size()/dim) = x_dy(i);
   }
   x_dx_dy *= -1; // true = surrogate + d
   VectorGridFunctionCoefficient dist_vec(&x_dx_dy);

   if (visualization && false)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock << "solution\n" << pmesh << x_dx_dy << flush;
      sol_sock << "window_title 'DDDerivative distfun'\n"
               << "window_geometry "
               << 350 << " " << 350 << " " << 350 << " " << 350 << "\n"
               << "keys Rjmpc" << endl;
   }

   // 7. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
   //    the basis functions in the finite element fespace.
   double alpha = 10.;
   ParLinearForm b(&pfespace);
   ConstantCoefficient one(1.0);
   b.AddDomainIntegrator(new DomainLFIntegrator(one), trim_flag);
   ConstantCoefficient dbcCoef(dbc_val);
   b.AddShiftedBdrFaceIntegrator(new SBM2LFIntegrator(dbcCoef, alpha, dist_vec),
                                 sbm_int_face, sbm_int_face_el, sbm_int_flag);
   b.Assemble();

   // 9. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator.
   ParBilinearForm a(&pfespace);
   //(nabla u, nabla w) - Term 1
   a.AddDomainIntegrator(new DiffusionIntegrator(one), trim_flag);
   a.AddShiftedBdrFaceIntegrator(new SBM2Integrator(alpha, dist_vec),
                                 sbm_int_face, sbm_int_face_el, sbm_int_flag);
   x = 0;

   // 10. Assemble the bilinear form and the corresponding linear system,
   //     applying any necessary transformations such as: eliminating boundary
   //     conditions, applying conforming constraints for non-conforming AMR,
   //     static condensation, etc.
   if (static_cond) { a.EnableStaticCondensation(); }
   std::cout << " abuot to assemble\n";

   a.Assemble();

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   cout << "Size of linear system: " << A->Height() << endl;

   Solver *prec = NULL;
   prec = new HypreBoomerAMG;

   GMRESSolver gmres(MPI_COMM_WORLD);
   gmres.SetRelTol(1e-12);
   gmres.SetMaxIter(10000);
   gmres.SetKDim(50);
   gmres.SetPrintLevel(1);
   gmres.SetPreconditioner(*prec);
   gmres.SetOperator(*A);
   gmres.Mult(B, X);
   delete prec;

   // 12. Recover the solution as a finite element grid function.
   a.RecoverFEMSolution(X, b, x);

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << pmesh << x << flush;
   }


   // 13. Save the refined mesh and the solution. This output can be viewed later
   //     using GLVis: "glvis -m refined.mesh -g sol.gf".
   ofstream mesh_ofs("ex1-sbm.mesh");
   mesh_ofs.precision(8);
   pmesh.PrintAsOne(mesh_ofs);
   ofstream sol_ofs("ex1-sbm.gf");
   sol_ofs.precision(8);
   x.SaveAsOne(sol_ofs);

   // 14. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << mesh << x << flush;
      sol_sock << "window_title 'Solution'\n"
               << "window_geometry "
               << 350 << " " << 0 << " " << 350 << " " << 350 << "\n"
               << "keys Rj" << endl;
   }

   // 15. Free the used memory.
   delete fec;
   delete fec_mesh;

   MPI_Finalize();

   return 0;
}

#define ring_radius 0.2
#define level_set_type 1
// use 0.2 ring for dist_fun because square_disc has the
// hole with this radius
double dist_fun(const Vector &x)
{
   if (level_set_type == 1) { // circle of radius 0.2 - centered at 0.5, 0.5
       double dx = x(0) - 0.5,
          dy = x(1) - 0.5,
          rv = dx*dx + dy*dy;
       rv = rv > 0 ? pow(rv, 0.5) : 0;
       double dist0 = rv - ring_radius; // +ve is the domain
       return dist0;
   }
   else { // circle of radius 0.2 at 0.25, 0.25 and 0.75, 0.75
       double xc1 = 0.3, xc2 = 0.7,
              yc1 = 0.3, yc2 = 0.7;
       double dx = x(0) - xc1,
          dy = x(1) - yc1,
          rv = dx*dx + dy*dy;
       rv = rv > 0 ? pow(rv, 0.5) : 0;
       double dist1 = rv - ring_radius; // +ve is the domain
       dx = x(0) - xc2;
       dy = x(1) - yc2;
       rv = dx*dx + dy*dy;
       rv = rv > 0 ? pow(rv, 0.5) : 0;
       double dist2 = rv - ring_radius;
       return std::min(dist1, dist2);
   }
}

double dist_fun_level_set(const Vector &x)
{
   double dist = dist_fun(x);
   if (dist > 0.) { return 1; }
   //else if (dist == 0.) { return 0.5; }
   else { return 0.; }
}
#undef level_set_type
#undef ring_radius
