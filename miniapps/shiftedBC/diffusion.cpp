//                       MFEM
//
// Compile with: make ex1p
//
// Sample runs:
// mpirun -np 1 ex1p -m ../../data/inline-quad.mesh  -rs 0 -vis -o 2
// mpirun -np 1 ex1p -m quad.mesh -rs 0 -o 2 -st 1 -ex -lst 1
// mpirun -np 1 ex1p -m quad.mesh -rs 0 -o 2 -st 1 -ex -lst 3
// mpirun -np 1 ex1p -m quad.mesh -rs 0 -o 2 -st 1 -ex -lst 4
#include "../../mfem.hpp"
#include <fstream>
#include <iostream>
#include "distfunction.hpp"
#include "sbm-aux.hpp"

using namespace mfem;
using namespace std;

double rhs_fun_xy(const Vector &x);

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
   bool exact = true;
   int solver_type = 1;
   int level_set_type = 1;
   int ho_terms = 0;

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
   args.AddOption(&exact, "-ex", "--exact", "-no-ex",
                  "--no-exact",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&solver_type, "-st", "--solver-type",
                  "Solver type: 0- CG, 1 - BiCG.");
   args.AddOption(&level_set_type, "-lst", "--level-set-type",
                  "level-set-type:");
   args.AddOption(&ho_terms, "-ho", "--high-order",
                  "Additional high-order terms to include");
   /// IMPORTANT: Level set type notes
   /// 1 - circlular hole of radius 0.2 at the center of domain [0, 1].
   ///     -nabla^2 = 1. Exact solution is generated using a very fine mesh
   /// that is body fitted.
   /// 3 - circular hole of radius 0.2 at the center of domain [0, 1]. Solution
   /// is linear from y = 0 to 1. We use this to make sure we get exact solution.
   /// 4. - Walls are at y = 0 to 1. In this case we stretch the mesh from
   /// [0,1] to [-0.1, 1.1] to get shifted faces. The solution is analytic
   /// sinusoidal function.

   /// Use level set (1) and (4) for convergence study.
   /// Use (3) to make sure we get exact solution with SBM.

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

   // 4. Define a finite element space on the mesh. Here we use continuous
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

   Vector vxyz;

   ParFiniteElementSpace pfespace_mesh(&pmesh, fec, dim);
   pmesh.SetNodalFESpace(&pfespace_mesh);
   ParGridFunction x_mesh(&pfespace_mesh);
   pmesh.SetNodalGridFunction(&x_mesh);
   vxyz = *pmesh.GetNodes();
   int nodes_cnt = vxyz.Size()/dim;
   if (level_set_type == 4) { //stretch quadmesh from [0, 1] to -[0.01, 1.01]
       for (int i = 0; i < nodes_cnt; i++) {
           vxyz(i+nodes_cnt) = -0.001 + 1.002*vxyz(i+nodes_cnt);
       }
   }
   pmesh.SetNodes(vxyz);
   pfespace.ExchangeFaceNbrData();

   // Setup FESpace for L2 function (used to mark element flags etc.)
   L2_FECollection fecl2 = L2_FECollection(0, dim);
   L2_FECollection fecl2ho = L2_FECollection(order, dim);
   ParFiniteElementSpace pfesl2(&pmesh, &fecl2);
   ParFiniteElementSpace pfesl2ho(&pmesh, &fecl2ho);
   cout << "Number of finite element unknowns: "
        << pfespace.GetTrueVSize() << endl;

   // 5. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   ParGridFunction x(&pfespace);
   // ParGridFunction for level_set_value.
   ParGridFunction level_set_val(&pfespace);

   // 6. Determine if each element in the ParMesh is inside the actual domain,
   //    partially cut by the actual domain boundary, or completely outside
   //    the domain.
   Dist_Level_Set_Coefficient dist_fun_level_coef(level_set_type);
   level_set_val.ProjectCoefficient(dist_fun_level_coef);
   level_set_val.ExchangeFaceNbrData();

   IntegrationRules IntRulesLo(0, Quadrature1D::GaussLobatto);
   FaceElementTransformations *tr = NULL;

   // Set trim flag based on the distance field
   // 0 if completely in the domain
   // 1 if completely outside the domain
   // 2 if partially inside the domain
   ParGridFunction elem_flags(&pfesl2);
   Array<int> trim_flag(pmesh.GetNE()+pmesh.GetNSharedFaces());
   trim_flag = 0;
   Vector vals;
   // Check elements on the current MPI rank
   for (int i = 0; i < pmesh.GetNE(); i++)
   {
      ElementTransformation *Tr = pmesh.GetElementTransformation(i);
      const IntegrationRule &ir =
         IntRulesLo.Get(pmesh.GetElementBaseGeometry(i), 4*Tr->OrderJ());
      level_set_val.GetValues(i, ir, vals);

      int count = 0;
      for (int j = 0; j < ir.GetNPoints(); j++)
      {
         double val = vals(j);
         if (val <= 0.) { count++; }
      }
      if (count == ir.GetNPoints()) // completely outside
      {
         trim_flag[i] = 1;
      }
      else if (count > 0) // partially outside
      {
         trim_flag[i] = 2;
      }
   }

   // Check neighbors on the adjacent MPI rank
   for (int i = pmesh.GetNE(); i < pmesh.GetNE()+pmesh.GetNSharedFaces(); i++)
   {
       int shared_fnum = i-pmesh.GetNE();
       tr = pmesh.GetSharedFaceTransformations(shared_fnum);
       int Elem2NbrNo = tr->Elem2No - pmesh.GetNE();

       ElementTransformation *eltr =
               pfespace.GetFaceNbrElementTransformation(Elem2NbrNo);
       const IntegrationRule &ir =
         IntRulesLo.Get(pfespace.GetFaceNbrFE(Elem2NbrNo)->GetGeomType(),
                        4*eltr->OrderJ());

       const int nip = ir.GetNPoints();
       vals.SetSize(nip);
       int count = 0;
       for (int j = 0; j < nip; j++) {
          const IntegrationPoint &ip = ir.IntPoint(j);
          vals[j] = level_set_val.GetValue(tr->Elem2No, ip);
          if (vals[j] <= 0.) { count++; }
       }

      if (count == ir.GetNPoints())
      {
         trim_flag[i] = 1;
      }
      else if (count > 0)
      {
         trim_flag[i] = 2;
      }
   }

   for (int i = 0; i < elem_flags.Size(); i++)
   {
      elem_flags(i) = trim_flag[i]*1.;
   }

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock << "solution\n" << pmesh << elem_flags << flush;
      sol_sock << "window_title 'Element flags'\n"
               << "window_geometry "
               << 0 << " " << 0 << " " << 350 << " " << 350 << "\n"
               << "keys Rjmpc" << endl;
   }

   // 7. Determine Shifted boundary faces of the mesh. This requires us to check
   //    interior faces of the mesh, boundary faces of the mesh, and interior
   //    faces of the mesh that are on processor boundaries.
   Array<int> sbm_face_num; // store the face number
   Array<int> sbm_face_el_num; // store the element number adjacent to the face
   // that is actually part of the domain. We need this because for an internal
   // face there are two options, and we need to integrate on the face that is
   // part of the pseudo-domain.
   Array<int> sbm_int_flag; // flag indicating the face type
   // 1 if int face, 2 if bdr face, 3 if int face on processor boundary
   // Get SBM faces
   Array<int> sbm_dofs; // Array of dofs on sbm faces
   Array<int> dofs; // work array

   // First we check interior faces of the mesh (excluding interior faces that
   // are on the processor boundaries)
   double count1 = 0;
   for (int i = 0; i < pmesh.GetNumFaces(); i++)
   {
      FaceElementTransformations *tr = NULL;
      tr = pmesh.GetInteriorFaceTransformations (i);
      const int faceno = i;
      if (tr != NULL)
      {
         count1 += 1;
         int ne1 = tr->Elem1No;
         int ne2 = tr->Elem2No;
         int te1 = trim_flag[ne1], te2 = trim_flag[ne2];
         if (te1 == 2 && te2 == 0)
         {
             sbm_face_num.Append(i);
             sbm_face_el_num.Append(ne2);
             sbm_int_flag.Append(1);
             pfespace.GetFaceDofs(faceno, dofs);
             sbm_dofs.Append(dofs);
         }
         if (te1 == 0 && te2 == 2)
         {
             sbm_face_num.Append(i);
             sbm_face_el_num.Append(ne1);
             sbm_int_flag.Append(1);
             pfespace.GetFaceDofs(faceno, dofs);
             sbm_dofs.Append(dofs);
         }
      }
   }


   // Here we add boundary faces that we want to model as SBM faces.
   // For the method where we clip inside the domain, a boundary face
   // has to be set as SBM face using its attribute.
   double count2 = 0;
   for (int i = 0; i < pmesh.GetNBE(); i++)
   {
      int attr = pmesh.GetBdrAttribute(i);
      FaceElementTransformations *tr;
      tr = pmesh.GetBdrFaceTransformations (i);
      if (tr != NULL) {
          if (attr == 100) { // add all boundary faces with attr=100 as SBM faces
              count2 += 1;
              int ne1 = tr->Elem1No;
              int te1 = trim_flag[ne1];
              const int faceno = pmesh.GetBdrFace(i);
              if (te1 == 0)
              {
                 sbm_face_num.Append(i);
                 sbm_face_el_num.Append(ne1);
                 sbm_int_flag.Append(2);
                 pfespace.GetFaceDofs(faceno, dofs);
                 sbm_dofs.Append(dofs);
              }
          }
      }
   }

   // Now we add interior faces that are on processor boundaries. This does not
   // work yet so we can only really run on 1 MPI rank.
   double count3 = 0;
   double count3b = 0;
   for (int i = 0; i < pmesh.GetNSharedFaces(); i++)
   {
      tr = pmesh.GetSharedFaceTransformations(i);
      if (tr != NULL)
      {
          count3b += 1;
         int ne1 = tr->Elem1No;
         int te1 = trim_flag[ne1];
         int te2 = trim_flag[i+pmesh.GetNE()];
         const int faceno = pmesh.GetSharedFace(i);
         // Add if the element on this proc is completely inside the domain
         // and the the element on other proc is not
         if (te2 == 2 && te1 == 0)
         {
            count3 += 1;
            sbm_face_num.Append(i);
            sbm_face_el_num.Append(ne1);
            sbm_int_flag.Append(3);
            pfespace.GetFaceDofs(faceno, dofs);
            sbm_dofs.Append(dofs);
         }
      }
   }


   // 8. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    To do this, we first make a list of all dofs that are on the real boundary
   //    of the mesh, then add all the dofs of the elements that are completely
   //    outside our shifted boundary. Then we remove the dofs from sbm faces.
   Array<int> ess_tdof_list;
   Array<int> ess_bdr(pmesh.bdr_attributes.Max());
   if (pmesh.bdr_attributes.Size())
   {
      ess_bdr = 1;
   }
   Array<int> ess_vdofs_bdr;
   pfespace.GetEssentialVDofs(ess_bdr, ess_vdofs_bdr);

   // now get all dofs that are part of the elements not in the domain
   Array<int> ess_vdofs_hole(ess_vdofs_bdr.Size());
   ess_vdofs_hole = 0;
   for (int e = 0; e < pmesh.GetNE(); e++)
   {
      if (trim_flag[e] > 0)
      {
         pfespace.GetElementVDofs(e, dofs);
         for (int i = 0; i < dofs.Size(); i++) {
             ess_vdofs_hole[dofs[i]] = -1;
         }
      }
   }

   // now combine the two lists
   for (int i = 0; i < ess_vdofs_hole.Size(); i++)
   {
      if (ess_vdofs_bdr[i] == -1) { ess_vdofs_hole[i] = -1; }
   }

   // Now unmark dofs that are on SBM faces
   for (int i = 0; i < sbm_dofs.Size(); i++) {
       ess_vdofs_hole[sbm_dofs[i]] = 0;
   }

   // remark dofs that are on dirichlet faces
   if (level_set_type == 4) {
       for (int i = 0; i < ess_vdofs_bdr.Size(); i++) {
           if (ess_vdofs_bdr[i] == -1) { ess_vdofs_hole[i] = -1; }
       }
   }

   // Now synchronize
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

   // 9. Get the Distance from the level set either using a numerical approach
   //    or project an exact analytic function.
   DistanceFunction dist_func(pmesh, order, 1.0);
   ParGridFunction &distance = dist_func.ComputeDistance(dist_fun_level_coef,
                                                         1, true);
   Dist_Value_Coefficient dist_fun_coef(level_set_type);
   if (exact) {
       distance.ProjectCoefficient(dist_fun_coef); // analytic projection
   }
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

   // 10. Construct the distance vector using numerical distances or an
   //     exact analytic function.
   ParFiniteElementSpace *distance_vec_space = new ParFiniteElementSpace(
                                 distance.ParFESpace()->GetParMesh(),
                                 distance.ParFESpace()->FEColl(), dim);
   ParGridFunction x_dx_dy(distance_vec_space);
   VectorCoefficient *dist_vec = NULL;

   if (!exact) {
       ParGridFunction x_dx(distance.ParFESpace()), x_dy(distance.ParFESpace());
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

       VectorGridFunctionCoefficient *dist_vec_gfcoeff =
               new VectorGridFunctionCoefficient(&x_dx_dy);
       dist_vec = dist_vec_gfcoeff;
   }
   else {
       Dist_Vector_Coefficient *dist_vec_fcoeff =
               new Dist_Vector_Coefficient(dim, level_set_type);
       dist_vec = dist_vec_fcoeff;
       x_dx_dy.ProjectDiscCoefficient(*dist_vec);
   }

   if (visualization)
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

   // 12. Set up SBM integration parameter - alpha
   double pfactor = std::max(ser_ref_levels*1., order*1.);
   //double alpha = 100/pow(2., pfactor);
   double alpha = 1.;

   if (ho_terms == 0) { alpha = 10; }
   if (ho_terms == 1) { alpha = 1; }
   if (ho_terms == 2) {
       alpha = 1.0;
       if (order == 5) { alpha = 0.1; }
       if (order == 6) { alpha = 0.05; }
   }
   if (ho_terms == 3) {
       alpha = 1.;
       if (order >=5) {
           alpha = 0.05;
           if (ser_ref_levels >= 3) { alpha = 0.0675; }
       }
   }
   if (ho_terms == 4) {
       alpha = 0.06;// 0.065; //0.05;
       if (ser_ref_levels >= 3) { alpha = 0.0675; }
       if (order <= 4) { alpha = 1.; }
   }


   // 13. Set up the linear form b(.) which corresponds to the right-hand side of
   //     the FEM linear system.
   ParLinearForm b(&pfespace);
   double fval = 1.;
   if (level_set_type == 3 || level_set_type == 5) {
      fval = 0.;
   }
   ConstantCoefficient rhs_f(fval);
   FunctionCoefficient rhs_fxy(rhs_fun_xy);
   if (level_set_type == 4) {
       b.AddDomainIntegrator(new DomainLFIntegrator(rhs_fxy), trim_flag);
   }
   else {
       b.AddDomainIntegrator(new DomainLFIntegrator(rhs_f), trim_flag);
   }

   SBMFunctionCoefficient dbcCoef(dirichlet_velocity, level_set_type);
   b.AddShiftedBdrFaceIntegrator(new SBM2LFIntegrator(dbcCoef, alpha, *dist_vec, ho_terms),
                                 sbm_face_num, sbm_face_el_num, sbm_int_flag);
   b.Assemble();

   // 14. Set up the bilinear form a(.,.) on the finite element space
   //     corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //     domain integrator and SBM integrator.
   ParBilinearForm a(&pfespace);
   ConstantCoefficient one(1.);

   a.AddDomainIntegrator(new DiffusionIntegrator(one), trim_flag);
   a.AddShiftedBdrFaceIntegrator(new SBM2Integrator(alpha, *dist_vec, ho_terms),
                                 sbm_face_num, sbm_face_el_num, sbm_int_flag);

   // 15. Assemble the bilinear form and the corresponding linear system,
   //     applying any necessary transformations such as: eliminating boundary
   //     conditions, applying conforming constraints for non-conforming AMR,
   //     static condensation, etc.
   if (static_cond) { a.EnableStaticCondensation(); }
   a.Assemble();

   // Project the exact solution as an initial condition for dirichlet boundaries.
   x = 0;
   x.ProjectCoefficient(dbcCoef);

   // 16. Form the linear system and solve it.
   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   cout << "Size of linear system: " << A->Height() << endl;

   Solver *S = NULL;
   Solver *prec = NULL;
   prec = new HypreBoomerAMG;
   if (solver_type == 0) {
       CGSolver *cg = new CGSolver(MPI_COMM_WORLD);
       cg->SetRelTol(1e-12);
       cg->SetMaxIter(1000);
       cg->SetPrintLevel(1);
       cg->SetPreconditioner(*prec);
       cg->SetOperator(*A);
       S = cg;
   }
   else {
       BiCGSTABSolver *bicg = new BiCGSTABSolver(MPI_COMM_WORLD);
       //GMRESSolver *bicg = new GMRESSolver(MPI_COMM_WORLD);
       bicg->SetRelTol(1e-12);
       bicg->SetMaxIter(2000);
       bicg->SetPrintLevel(1);
       bicg->SetPreconditioner(*prec);
       bicg->SetOperator(*A);
       S = bicg;
   }
   S->Mult(B, X);

   // 17. Recover the solution as a finite element grid function.
   a.RecoverFEMSolution(X, b, x);

   // 18. Save the refined mesh and the solution. This output can be viewed later
   //     using GLVis: "glvis -m refined.mesh -g sol.gf".
   ofstream mesh_ofs("ex1-sbm.mesh");
   mesh_ofs.precision(8);
   pmesh.PrintAsOne(mesh_ofs);
   ofstream sol_ofs("ex1-sbm.gf");
   sol_ofs.precision(8);
   x.SaveAsOne(sol_ofs);

   // 19. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock << "solution\n" << pmesh << x << flush;
      sol_sock << "window_title 'Solution'\n"
               << "window_geometry "
               << 350 << " " << 0 << " " << 350 << " " << 350 << "\n"
               << "keys Rj" << endl;
   }

   // 20. Construct an error gridfunction if the exact solution is known.
   ParGridFunction err(x);
   Vector pxyz(dim);
   pxyz(0) = 0.;
   for (int i = 0; i < nodes_cnt; i++) {
       double yv = vxyz(i+nodes_cnt);
       pxyz(0) = vxyz(i);;
       pxyz(1) = yv;
       double exact_val = dirichlet_velocity(pxyz, level_set_type);
       if (level_set_type == 3) {
           if (yv < 0.1 || yv > 0.9) { err(i) = 0.; }
           else { err(i) = std::fabs(x(i) - exact_val); }
       }
       else if (level_set_type == 4) {
           if (yv < 0. || yv > 1.0) { err(i) = 0.; }
           else { err(i) = std::fabs(x(i) - exact_val); }
           err(i) = std::fabs(x(i) - exact_val);
       }
       else if (level_set_type == 5) {
           if (yv < 0. || yv > 1.0) { err(i) = 0.; }
           else { err(i) = std::fabs(x(i) - exact_val); }
       }
   }

   if (visualization && level_set_type >= 3 && level_set_type <= 5)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock << "solution\n" << pmesh << err << flush;
      sol_sock << "window_title 'Error'\n"
               << "window_geometry "
               << 700 << " " << 0 << " " << 350 << " " << 350 << "\n"
               << "keys Rj" << endl;
   }



   int NEglob = pmesh.GetGlobalNE();
   double errnorm = x.ComputeL2Error(dbcCoef);
   if (level_set_type >= 3 && level_set_type <= 5 && myid == 0)
   {
      ofstream myfile;
      myfile.open ("error.txt", ios::app);
      double h_factor = pow(1./2, ser_ref_levels*1.);
      cout << order << " " <<
                ho_terms << " " <<
                h_factor << " " <<
                errnorm << " " <<
                NEglob << " " <<
                "k10-analytic-L2Error\n";
      myfile.close();
   }


   // 15. Free the used memory.
   delete prec;
   delete S;
   delete dist_vec;
   delete distance_vec_space;
   delete fec;

   MPI_Finalize();

   return 0;
}

double rhs_fun_xy(const Vector &x)
{
    return std::sin(M_PI*x(0))*sin(M_PI*x(1));
}
