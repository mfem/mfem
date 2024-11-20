// make pmesh-opt-sliding -j4 && ./pmesh-opt-sliding -vl 2 -ni 1 -vis -qo 6 -qt 2 -mid 80 -tid 2 -o 2 -pre
// make pmesh-opt-sliding -j4 && ./pmesh-opt-sliding -m aleuntangled.mesh -vl 2 -ni 50 -vis -qo 8 -qt 1 -mid 80 -o 2 -tid 2 -no-pre
#include "mfem.hpp"
#include "../common/mfem-common.hpp"
#include <iostream>
#include <fstream>
#include "mesh-optimizer.hpp"

using namespace mfem;
using namespace std;

// void NodalTransform(const Vector &x, Vector &p)
// {
//    double xv = x(0);
//    double yv = x(1);
//    double dxmin = -0.2;
//    double dxmax = 0.3;
//    double dy = yv;
//    double dx = std::fabs(xv-0.5);
//    double dxx = dxmin + yv*yv*(dxmax-dxmin);
//    dxx = (dx-0.5)/(0.5)*dxx;
//    p(0) = x(0) + dxx;
//    p(1) = x(1);
// }

void ModifyBoundaryAttributesForNodeMovement(ParMesh *pmesh, ParGridFunction &x)
{
   const int dim = pmesh->Dimension();
   for (int i = 0; i < pmesh->GetNBE(); i++)
   {
      mfem::Array<int> dofs;
      pmesh->GetNodalFESpace()->GetBdrElementDofs(i, dofs);
      mfem::Vector bdr_xy_data;
      mfem::Vector dof_xyz(dim);
      mfem::Vector dof_xyz_compare;
      mfem::Array<int> xyz_check(dim);
      for (int j = 0; j < dofs.Size(); j++)
      {
         for (int d = 0; d < dim; d++)
         {
            dof_xyz(d) = x(pmesh->GetNodalFESpace()->DofToVDof(dofs[j], d));
         }
         if (j == 0)
         {
            dof_xyz_compare = dof_xyz;
            xyz_check = 1;
         }
         else
         {
            for (int d = 0; d < dim; d++)
            {
               if (std::abs(dof_xyz(d)-dof_xyz_compare(d)) < 1.e-10)
               {
                  xyz_check[d] += 1;
               }
            }
         }
      }
      if (dim == 2)
      {
         if (xyz_check[0] == dofs.Size())
         {
            pmesh->GetNodalFESpace()->GetMesh()->SetBdrAttribute(i, 1);
         }
         else if (xyz_check[1] == dofs.Size())
         {
            pmesh->GetNodalFESpace()->GetMesh()->SetBdrAttribute(i, 2);
         }
         else
         {
            pmesh->GetNodalFESpace()->GetMesh()->SetBdrAttribute(i, 4);
         }
      }
      else if (dim == 3)
      {
         if (xyz_check[0] == dofs.Size())
         {
            pmesh->GetNodalFESpace()->GetMesh()->SetBdrAttribute(i, 1);
         }
         else if (xyz_check[1] == dofs.Size())
         {
            pmesh->GetNodalFESpace()->GetMesh()->SetBdrAttribute(i, 2);
         }
         else if (xyz_check[2] == dofs.Size())
         {
            pmesh->GetNodalFESpace()->GetMesh()->SetBdrAttribute(i, 3);
         }
         else
         {
            pmesh->GetNodalFESpace()->GetMesh()->SetBdrAttribute(i, 4);
         }
      }
   }
}

void BlendDisplacement(ParMesh *pmesh, ParGridFunction &u, double alpha, double beta=1.0)
{
   int order = pmesh->GetNodalFESpace()->GetMaxElementOrder();
   int dim = pmesh->Dimension();

   for (int i = 0; i < dim; i++)
   {
      H1_FECollection fec(order, dim);
      ParFiniteElementSpace fespace(pmesh, &fec, 1);

      ParLinearForm b(&fespace);
      b = 0.0;

      ParBilinearForm a(&fespace);
      a.SetAssemblyLevel(AssemblyLevel::FULL);

      ConstantCoefficient one(beta);
      ConstantCoefficient alphac(alpha);
      a.AddDomainIntegrator(new DiffusionIntegrator(one)); // For ∇^2
      // a.AddDomainIntegrator(new MassIntegrator(alphac));   // For -α^2 u
      a.Assemble();

      Array<int> ess_tdof_list(0);
      Array<int> bdr(pmesh->bdr_attributes.Max());
      bdr = 1;
      fespace.GetEssentialTrueDofs(bdr, ess_tdof_list);

      OperatorPtr A;
      Vector B, X;

      int nnodes = u.Size()/dim;
      Vector ucomp(u.GetData()+i*nnodes, nnodes);
      a.FormLinearSystem(ess_tdof_list, ucomp, b, A, X, B);

      HypreBoomerAMG *amg = new HypreBoomerAMG;
      amg->SetSystemsOptions(dim);
      HyprePCG pcg(MPI_COMM_WORLD);
      pcg.SetTol(1e-12);
      pcg.SetMaxIter(200);
      pcg.SetPrintLevel(2);
      pcg.SetPreconditioner(*amg);
      pcg.SetOperator(*A);
      pcg.SetPrintLevel(0);
      pcg.Mult(B, X);
      delete amg;

      a.RecoverFEMSolution(X, b, ucomp);
   }
}


int main (int argc, char *argv[])
{
   // 0. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 1. Set the method's default parameters.
   const char *mesh_file = "aletangled.mesh";
   int mesh_poly_deg     = 2;
   int rs_levels         = 0;
   int rp_levels         = 0;
   real_t jitter         = 0.0;
   int metric_id         = 2;
   int target_id         = 1;
   real_t lim_const      = 0.0;
   real_t adapt_lim_const   = 0.0;
   int quad_type         = 1;
   int quad_order        = 8;
   int solver_type       = 0;
   int solver_iter       = 100;
#ifdef MFEM_USE_SINGLE
   real_t solver_rtol    = 1e-4;
#else
   real_t solver_rtol    = 1e-10;
#endif
   int solver_art_type   = 0;
   int lin_solver        = 2;
   int max_lin_iter      = 100;
   bool move_bnd         = true;
   bool visualization    = true;
   int verbosity_level   = 0;
   int adapt_eval        = 0;
   bool exactaction      = false;
   bool integ_over_targ  = true;
   const char *devopt    = "cpu";
   bool pa               = false;
   int mesh_node_ordering = 0;
   bool presolve = true;

   // 2. Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&mesh_poly_deg, "-o", "--order",
                  "Polynomial degree of mesh finite element space.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&rp_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&jitter, "-ji", "--jitter",
                  "Random perturbation scaling factor.");
   args.AddOption(&metric_id, "-mid", "--metric-id",
                  "Mesh optimization metric:\n\t"
                  "T-metrics\n\t"
                  "1  : |T|^2                          -- 2D no type\n\t"
                  "2  : 0.5|T|^2/tau-1                 -- 2D shape (condition number)\n\t"
                  "7  : |T-T^-t|^2                     -- 2D shape+size\n\t"
                  "9  : tau*|T-T^-t|^2                 -- 2D shape+size\n\t"
                  "14 : |T-I|^2                        -- 2D shape+size+orientation\n\t"
                  "22 : 0.5(|T|^2-2*tau)/(tau-tau_0)   -- 2D untangling\n\t"
                  "50 : 0.5|T^tT|^2/tau^2-1            -- 2D shape\n\t"
                  "55 : (tau-1)^2                      -- 2D size\n\t"
                  "56 : 0.5(sqrt(tau)-1/sqrt(tau))^2   -- 2D size\n\t"
                  "58 : |T^tT|^2/(tau^2)-2*|T|^2/tau+2 -- 2D shape\n\t"
                  "77 : 0.5(tau-1/tau)^2               -- 2D size\n\t"
                  "80 : (1-gamma)mu_2 + gamma mu_77    -- 2D shape+size\n\t"
                  "85 : |T-|T|/sqrt(2)I|^2             -- 2D shape+orientation\n\t"
                  "90 : balanced combo mu_50 & mu_77   -- 2D shape+size\n\t"
                  "94 : balanced combo mu_2 & mu_56    -- 2D shape+size\n\t"
                  "98 : (1/tau)|T-I|^2                 -- 2D shape+size+orientation\n\t"
                  // "211: (tau-1)^2-tau+sqrt(tau^2+eps)  -- 2D untangling\n\t"
                  // "252: 0.5(tau-1)^2/(tau-tau_0)       -- 2D untangling\n\t"
                  "301: (|T||T^-1|)/3-1              -- 3D shape\n\t"
                  "302: (|T|^2|T^-1|^2)/9-1          -- 3D shape\n\t"
                  "303: (|T|^2)/3/tau^(2/3)-1        -- 3D shape\n\t"
                  "304: (|T|^3)/3^{3/2}/tau-1        -- 3D shape\n\t"
                  // "311: (tau-1)^2-tau+sqrt(tau^2+eps)-- 3D untangling\n\t"
                  "313: (|T|^2)(tau-tau0)^(-2/3)/3   -- 3D untangling\n\t"
                  "315: (tau-1)^2                    -- 3D no type\n\t"
                  "316: 0.5(sqrt(tau)-1/sqrt(tau))^2 -- 3D no type\n\t"
                  "321: |T-T^-t|^2                   -- 3D shape+size\n\t"
                  "322: |T-adjT^-t|^2                -- 3D shape+size\n\t"
                  "323: |J|^3-3sqrt(3)ln(det(J))-3sqrt(3)  -- 3D shape+size\n\t"
                  "328: balanced combo mu_301 & mu_316   -- 3D shape+size\n\t"
                  "332: (1-gamma) mu_302 + gamma mu_315  -- 3D shape+size\n\t"
                  "333: (1-gamma) mu_302 + gamma mu_316  -- 3D shape+size\n\t"
                  "334: (1-gamma) mu_303 + gamma mu_316  -- 3D shape+size\n\t"
                  "328: balanced combo mu_302 & mu_318   -- 3D shape+size\n\t"
                  "347: (1-gamma) mu_304 + gamma mu_316  -- 3D shape+size\n\t"
                  // "352: 0.5(tau-1)^2/(tau-tau_0)     -- 3D untangling\n\t"
                  "360: (|T|^3)/3^{3/2}-tau              -- 3D shape\n\t"
                  "A-metrics\n\t"
                  "11 : (1/4*alpha)|A-(adjA)^T(W^TW)/omega|^2 -- 2D shape\n\t"
                  "36 : (1/alpha)|A-W|^2                      -- 2D shape+size+orientation\n\t"
                  "107: (1/2*alpha)|A-|A|/|W|W|^2             -- 2D shape+orientation\n\t"
                  "126: (1-gamma)nu_11 + gamma*nu_14a         -- 2D shape+size\n\t"
                 );
   args.AddOption(&target_id, "-tid", "--target-id",
                  "Target (ideal element) type:\n\t"
                  "1: Ideal shape, unit size\n\t"
                  "2: Ideal shape, equal size\n\t"
                  "3: Ideal shape, initial size\n\t"
                  "4: Given full analytic Jacobian (in physical space)\n\t"
                  "5: Ideal shape, given size (in physical space)");
   args.AddOption(&lim_const, "-lc", "--limit-const", "Limiting constant.");
   args.AddOption(&adapt_lim_const, "-alc", "--adapt-limit-const",
                  "Adaptive limiting coefficient constant.");
   args.AddOption(&quad_type, "-qt", "--quad-type",
                  "Quadrature rule type:\n\t"
                  "1: Gauss-Lobatto\n\t"
                  "2: Gauss-Legendre\n\t"
                  "3: Closed uniform points");
   args.AddOption(&quad_order, "-qo", "--quad_order",
                  "Order of the quadrature rule.");
   args.AddOption(&solver_type, "-st", "--solver-type",
                  " Type of solver: (default) 0: Newton, 1: LBFGS");
   args.AddOption(&solver_iter, "-ni", "--newton-iters",
                  "Maximum number of Newton iterations.");
   args.AddOption(&solver_rtol, "-rtol", "--newton-rel-tolerance",
                  "Relative tolerance for the Newton solver.");
   args.AddOption(&solver_art_type, "-art", "--adaptive-rel-tol",
                  "Type of adaptive relative linear solver tolerance:\n\t"
                  "0: None (default)\n\t"
                  "1: Eisenstat-Walker type 1\n\t"
                  "2: Eisenstat-Walker type 2");
   args.AddOption(&lin_solver, "-ls", "--lin-solver",
                  "Linear solver:\n\t"
                  "0: l1-Jacobi\n\t"
                  "1: CG\n\t"
                  "2: MINRES\n\t"
                  "3: MINRES + Jacobi preconditioner\n\t"
                  "4: MINRES + l1-Jacobi preconditioner");
   args.AddOption(&max_lin_iter, "-li", "--lin-iter",
                  "Maximum number of iterations in the linear solve.");
   args.AddOption(&move_bnd, "-bnd", "--move-boundary", "-fix-bnd",
                  "--fix-boundary",
                  "Enable motion along horizontal and vertical boundaries.");
   args.AddOption(&exactaction, "-ex", "--exact_action",
                  "-no-ex", "--no-exact-action",
                  "Enable exact action of TMOP_Integrator.");
   args.AddOption(&integ_over_targ, "-it", "--integrate-target",
                  "-ir", "--integrate-reference",
                  "Integrate over target (-it) or reference (-ir) element.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&verbosity_level, "-vl", "--verbosity-level",
                  "Verbosity level for the involved iterative solvers:\n\t"
                  "0: no output\n\t"
                  "1: Newton iterations\n\t"
                  "2: Newton iterations + linear solver summaries\n\t"
                  "3: newton iterations + linear solver iterations");
   args.AddOption(&adapt_eval, "-ae", "--adaptivity-evaluator",
                  "0 - Advection based (DEFAULT), 1 - GSLIB.");
   args.AddOption(&devopt, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&presolve, "-pre", "--pre", "-no-pre",
                  "--no-pre",
                  "Enable or disable pre solve.");

   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

   bool visit = true;

   // 3. Initialize and refine the starting mesh.
   Mesh *mesh = new Mesh(mesh_file, 1, 1, false);
   for (int lev = 0; lev < rs_levels; lev++)
   {
      mesh->UniformRefinement();
   }
   const int dim = mesh->Dimension();

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < rp_levels; lev++)
   {
      pmesh->UniformRefinement();
   }

   // 4. Define a finite element space on the mesh. Here we use vector finite
   //    elements which are tensor products of quadratic finite elements. The
   //    number of components in the vector finite element space is specified by
   //    the last parameter of the FiniteElementSpace constructor.
   FiniteElementCollection *fec;
   if (mesh_poly_deg <= 0)
   {
      fec = new QuadraticPosFECollection;
      mesh_poly_deg = 2;
   }
   else { fec = new H1_FECollection(mesh_poly_deg, dim); }
   ParFiniteElementSpace *pfespace = new ParFiniteElementSpace(pmesh, fec, dim,
                                                               mesh_node_ordering);

   // 5. Make the mesh curved based on the above finite element space. This
   //    means that we define the mesh elements through a fespace-based
   //    transformation of the reference element.
   pmesh->SetNodalFESpace(pfespace);

   // 7. Get the mesh nodes (vertices and other degrees of freedom in the finite
   //    element space) as a finite element grid function in fespace. Note that
   //    changing x automatically changes the shapes of the mesh elements.
   ParGridFunction x(pfespace);
   pmesh->SetNodalGridFunction(&x);

   ParGridFunction dx = x;

   Vector center(dim);
   for (int e = 0; e < pmesh->GetNE(); e++)
   {
      pmesh->GetElementCenter(e, center);
      if (center(0) < 0.5)
      {
         pmesh->SetAttribute(e, 1);
      }
      else {
         pmesh->SetAttribute(e, 2);
      }
   }
   pmesh->SetAttributes();

   // 1 - top, 2 -> right, 3->bottom, 4->left
   if (presolve)
   {
      for (int i = 0; i < pmesh->GetNBE(); i++)
      {
         int attr = pmesh->GetBdrAttribute(i);
         if (attr == 1)
         {
            pmesh->SetBdrAttribute(i, 3);
         }
         else if (attr == 2)
         {
            pmesh->SetBdrAttribute(i, 4);
         }
         else if (attr == 3)
         {
            pmesh->SetBdrAttribute(i, 2);
         }
         else if (attr == 4)
         {
            pmesh->SetBdrAttribute(i, 1);
         }
      }
   }
   pmesh->SetAttributes();
   {
      ostringstream mesh_name;
      mesh_name << "inputbdrmod.mesh";
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->PrintAsSerial(mesh_ofs);
   }

   // VectorFunctionCoefficient nfc(dim, NodalTransform);
   // pmesh->GetNodes()->ProjectCoefficient(nfc);
   // pmesh->Transform(nfc);

   VisItDataCollection dc("pmeshslide", pmesh);
   dc.SetFormat(DataCollection::SERIAL_FORMAT);
   dc.RegisterField("displacement", &dx);
   if (presolve)
   {
      VisItDataCollection dco("pmeshslideorig", pmesh);
      dco.SetFormat(DataCollection::SERIAL_FORMAT);
      dco.RegisterField("displacement", &dx);
      dco.SetCycle(0);
      dco.SetTime(0.0);
      dco.Save();
      dc.SetCycle(0);
      dc.SetTime(0.0);
      dc.Save();
   }
   else
   {
      dc.SetCycle(1);
      dc.SetTime(1.0);
      dc.Save();
   }
   int vcount = 2;

   int vcount2 = 0;
   VisItDataCollection dc2("pslidedisp", pmesh);
   dc2.SetFormat(DataCollection::SERIAL_FORMAT);
   dc2.SetCycle(vcount2);
   dc2.SetTime((vcount2++)*1.0);
   dc2.Save();


   // 8. Define a vector representing the minimal local mesh size in the mesh
   //    nodes. We index the nodes using the scalar version of the degrees of
   //    freedom in pfespace. Note: this is partition-dependent.
   //
   //    In addition, compute average mesh size and total volume.
   Vector h0(pfespace->GetNDofs());
   h0 = infinity();
   real_t vol_loc = 0.0;
   Array<int> dofs;
   for (int i = 0; i < pmesh->GetNE(); i++)
   {
      // Get the local scalar element degrees of freedom in dofs.
      pfespace->GetElementDofs(i, dofs);
      // Adjust the value of h0 in dofs based on the local mesh size.
      const real_t hi = pmesh->GetElementSize(i);
      for (int j = 0; j < dofs.Size(); j++)
      {
         h0(dofs[j]) = min(h0(dofs[j]), hi);
      }
      vol_loc += pmesh->GetElementVolume(i);
   }
   real_t vol_glb;
   MPI_Allreduce(&vol_loc, &vol_glb, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_SUM, MPI_COMM_WORLD);
   const real_t small_phys_size = pow(vol_glb, 1.0 / dim) / 100.0;

   // 9. Add a random perturbation to the nodes in the interior of the domain.
   //    We define a random grid function of fespace and make sure that it is
   //    zero on the boundary and its values are locally of the order of h0.
   //    The latter is based on the DofToVDof() method which maps the scalar to
   //    the vector degrees of freedom in pfespace.
   ParGridFunction rdm(pfespace);
   rdm.Randomize();
   rdm -= 0.25; // Shift to random values in [-0.5,0.5].
   rdm *= jitter;
   rdm.HostReadWrite();
   // Scale the random values to be of order of the local mesh size.
   for (int i = 0; i < pfespace->GetNDofs(); i++)
   {
      for (int d = 0; d < dim; d++)
      {
         rdm(pfespace->DofToVDof(i,d)) *= h0(i);
      }
   }
   Array<int> vdofs;
   for (int i = 0; i < pfespace->GetNBE(); i++)
   {
      // Get the vector degrees of freedom in the boundary element.
      pfespace->GetBdrElementVDofs(i, vdofs);
      // Set the boundary values to zero.
      for (int j = 0; j < vdofs.Size(); j++) { rdm(vdofs[j]) = 0.0; }
   }
   x -= rdm;
   // Set the perturbation of all nodes from the true nodes.
   x.SetTrueVector();
   x.SetFromTrueVector();

   Array<int> facedofs, facedofs2;
   //
   // Construct a mesh from the interface elements
   int bdrattr = 4;
   int nfaces = 0;
   Array<int> fdofs;
   for (int f = 0; f < pmesh->GetNBE(); f++)
   {
      int attrib = pmesh->GetBdrAttribute(f);
      if (attrib == bdrattr)
      {
         nfaces += 1;
         pfespace->GetBdrElementDofs(f, fdofs);
         facedofs.Append(fdofs);
      }
   }

   int bdrattr2 = 3;
   int nfaces2 = 0;
   for (int f = 0; f < pmesh->GetNBE(); f++)
   {
      int attrib = pmesh->GetBdrAttribute(f);
      if (attrib == bdrattr2)
      {
         nfaces2 += 1;
         pfespace->GetBdrElementDofs(f, fdofs);
         facedofs2.Append(fdofs);
      }
   }
   // std::cout << nfaces << " k10numofelements\n";

   Array<int> attrd1, attrd2, attrd3, attrd4;
   for (int f = 0; f < pmesh->GetNBE(); f++)
   {
      int attrib = pmesh->GetBdrAttribute(f);
      int fnum = pmesh->GetBdrElementFaceIndex(f);
      pfespace->GetFaceDofs(fnum, dofs);
      if (attrib == 1)
      {
         attrd1.Append(dofs);
      }
      if (attrib == 2)
      {
         attrd2.Append(dofs);
      }
      if (attrib == 3)
      {
         attrd3.Append(dofs);
      }
      if (attrib == 4)
      {
         attrd4.Append(dofs);
      }
   }

   Array<int> cornerdofs;

   attrd1.Sort(); attrd1.Unique();
   attrd2.Sort(); attrd2.Unique();
   attrd3.Sort(); attrd3.Unique();
   attrd4.Sort(); attrd4.Unique();

   for (int i = 0; i < attrd4.Size(); i++)
   {
      int dof = attrd4[i];
      if (attrd2.Find(dof) != -1 || attrd3.Find(dof) != -1)
      {
         cornerdofs.Append(dof);
         cornerdofs.Append(dof+x.Size()/dim);
      }
   }
   for (int i = 0; i < attrd3.Size(); i++)
   {
      int dof = attrd3[i];
      if (attrd1.Find(dof) != -1 || attrd4.Find(dof) != -1)
      {
         cornerdofs.Append(dof);
         cornerdofs.Append(dof+x.Size()/dim);
      }
   }

   // cornerdofs.Print();
   // MFEM_ABORT(" ");

   // Make a line mesh with these many elements
   Mesh *intmesh = new Mesh(1, nfaces*2, nfaces, 0, 2);
   {
      for (int i = 0; i < nfaces; i++)
      {
         for (int j = 0; j < 2; j++)
         {
            Vector vert(dim);
            vert = 0.5;
            intmesh->AddVertex(vert.GetData());
         }
         Array<int> verts(dim);
         verts[0] = i*2+0;
         verts[1] = i*2+1;
         intmesh->AddSegment(verts, 1);
      }
      intmesh->Finalize(true, true);
      intmesh->FinalizeTopology();
      intmesh->SetCurvature(mesh_poly_deg, false);
      std::cout << intmesh->GetNE() << " " << intmesh->GetNumGeometries(1) << " k102\n";

      const FiniteElementSpace *intnodespace = intmesh->GetNodalFESpace();
      GridFunction *intnodes = intmesh->GetNodes();

      FaceElementTransformations *face_elem_transf;
      int count = 0;
      Vector vect;
      Vector nodeval(dim);
      for (int f = 0; f < pmesh->GetNBE(); f++)
      {
         int attrib = pmesh->GetBdrAttribute(f);
         int fnum = pmesh->GetBdrElementFaceIndex(f);
         if (attrib == bdrattr)
         {
            intnodespace->GetElementVDofs(count, dofs);
            const FiniteElement *fe = intnodespace->GetFE(count);
            face_elem_transf = pmesh->GetFaceElementTransformations(fnum);
            IntegrationRule irule = fe->GetNodes();
            int npts = irule.GetNPoints();
            vect.SetSize(npts*2);
            for (int q = 0; q < npts; q++)
            {
               IntegrationPoint &ip = irule.IntPoint(q);
               IntegrationPoint eip;
               face_elem_transf->Loc1.Transform(ip, eip);
               x.GetVectorValue(face_elem_transf->Elem1No, eip, nodeval);
               vect(q + 0) = nodeval(0);
               vect(q + npts) = nodeval(1);
               // std::cout << f  << " " << q << " " << ip.x << " " << ip.y << " k10info\n";
            }
            intnodes->SetSubVector(dofs, vect);
            count++;
         }
      }
   }

   Mesh *intmesh2 = new Mesh(1, nfaces2*2, nfaces2, 0, 2);
   {
      for (int i = 0; i < nfaces2; i++)
      {
         for (int j = 0; j < 2; j++)
         {
            Vector vert(dim);
            vert = 0.5;
            intmesh2->AddVertex(vert.GetData());
         }
         Array<int> verts(dim);
         verts[0] = i*2+0;
         verts[1] = i*2+1;
         intmesh2->AddSegment(verts, 1);
      }
      intmesh2->Finalize(true, true);
      intmesh2->FinalizeTopology();
      intmesh2->SetCurvature(mesh_poly_deg, false);
      std::cout << intmesh2->GetNE() << " " << intmesh2->GetNumGeometries(1) << " k102\n";

      const FiniteElementSpace *intnodespace = intmesh2->GetNodalFESpace();
      GridFunction *intnodes = intmesh2->GetNodes();

      FaceElementTransformations *face_elem_transf;
      int count = 0;
      Vector vect;
      Vector nodeval(dim);
      for (int f = 0; f < pmesh->GetNBE(); f++)
      {
         int attrib = pmesh->GetBdrAttribute(f);
         int fnum = pmesh->GetBdrElementFaceIndex(f);
         if (attrib == bdrattr2)
         {
            intnodespace->GetElementVDofs(count, dofs);
            const FiniteElement *fe = intnodespace->GetFE(count);
            face_elem_transf = pmesh->GetFaceElementTransformations(fnum);
            IntegrationRule irule = fe->GetNodes();
            int npts = irule.GetNPoints();
            vect.SetSize(npts*2);
            for (int q = 0; q < npts; q++)
            {
               IntegrationPoint &ip = irule.IntPoint(q);
               IntegrationPoint eip;
               face_elem_transf->Loc1.Transform(ip, eip);
               x.GetVectorValue(face_elem_transf->Elem1No, eip, nodeval);
               vect(q + 0) = nodeval(0);
               vect(q + npts) = nodeval(1);
               // std::cout << f  << " " << q << " " << ip.x << " " << ip.y << " k10info\n";
            }
            intnodes->SetSubVector(dofs, vect);
            count++;
         }
      }
   }

   facedofs.Sort();
   facedofs.Unique();


   facedofs2.Sort();
   facedofs2.Unique();

   // facedofs.Print();

   // facedofs2.Print();
   // MFEM_ABORT(" ");

   VisItDataCollection dcs("pmeshsurf", intmesh);
   dcs.SetFormat(DataCollection::SERIAL_FORMAT);
   dcs.SetCycle(1);
   dcs.SetTime(1.0);
   dcs.Save();

   VisItDataCollection dcs2("pmeshsurf2", intmesh2);
   dcs2.SetFormat(DataCollection::SERIAL_FORMAT);
   dcs2.SetCycle(1);
   dcs2.SetTime(1.0);
   dcs2.Save();

   {
      ostringstream mesh_name;
      mesh_name << "interface.mesh";
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      intmesh->Print(mesh_ofs);
   }
   {
      ostringstream mesh_name;
      mesh_name << "interface2.mesh";
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      intmesh2->Print(mesh_ofs);
   }
   // facedofs.Print();
   // MFEM_ABORT(" ");

   FindPointsGSLIB finder(MPI_COMM_WORLD);
   finder.SetupSurf(*intmesh, 0.8);
   finder.SetDistanceToleranceForPointsFoundOnBoundary(10);

   FindPointsGSLIB finder2(MPI_COMM_WORLD);
   finder2.SetupSurf(*intmesh2, 0.8);
   finder2.SetDistanceToleranceForPointsFoundOnBoundary(10);

   // Output bounding boxes
   Mesh *mesh_abb, *mesh_obb;
   if (true)
   {
      mesh_abb  = finder.GetBoundingBoxMeshSurf(0);  // Axis aligned bounding box
      mesh_obb  = finder.GetBoundingBoxMeshSurf(1);  // Oriented bounding box
      if (myid==0)
      {
         VisItDataCollection dc0("findersurfabb", mesh_abb);
         dc0.SetFormat(DataCollection::SERIAL_FORMAT);
         dc0.Save();

         VisItDataCollection dc1("findersurfobb", mesh_obb);
         dc1.SetFormat(DataCollection::SERIAL_FORMAT);
         dc1.Save();

         VisItDataCollection dc3("findersurf", intmesh);
         dc3.SetFormat(DataCollection::SERIAL_FORMAT);
         dc3.Save();
      }
   }
   MPI_Barrier(MPI_COMM_WORLD);

   Mesh *mesh_abb2, *mesh_obb2;
   if (true)
   {
      mesh_abb2  = finder2.GetBoundingBoxMeshSurf(0);  // Axis aligned bounding box
      mesh_obb2  = finder2.GetBoundingBoxMeshSurf(1);  // Oriented bounding box
      if (myid==0)
      {
         VisItDataCollection dc0("findersurfabb2", mesh_abb2);
         dc0.SetFormat(DataCollection::SERIAL_FORMAT);
         dc0.Save();

         VisItDataCollection dc1("findersurfobb2", mesh_obb2);
         dc1.SetFormat(DataCollection::SERIAL_FORMAT);
         dc1.Save();

         VisItDataCollection dc3("findersurf2", intmesh2);
         dc3.SetFormat(DataCollection::SERIAL_FORMAT);
         dc3.Save();
      }
   }

   // 10. Save the starting (prior to the optimization) mesh to a file. This
   //     output can be viewed later using GLVis: "glvis -m perturbed -np
   //     num_mpi_tasks".
   {
      ostringstream mesh_name;
      mesh_name << "perturbed.mesh";
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->PrintAsOne(mesh_ofs);
   }

   // 11. Store the starting (prior to the optimization) positions.
   ParGridFunction x0(pfespace);
   x0 = x;

   // 12. Form the integrator that uses the chosen metric and target.
   real_t min_detJ = -0.1;
   TMOP_QualityMetric *metric = NULL;
   switch (metric_id)
   {
      // T-metrics
      case 1: metric = new TMOP_Metric_001; break;
      case 2: metric = new TMOP_Metric_002; break;
      case 4: metric = new TMOP_Metric_004; break;
      case 7: metric = new TMOP_Metric_007; break;
      case 9: metric = new TMOP_Metric_009; break;
      case 50: metric = new TMOP_Metric_050; break;
      case 80: metric = new TMOP_Metric_080(0.5); break;
      case 22: metric = new TMOP_Metric_022(min_detJ); break;
      default:
         if (myid == 0) { cout << "Unknown metric_id: " << metric_id << endl; }
         return 3;
   }

   if (metric_id < 300)
   {
      MFEM_VERIFY(dim == 2, "Incompatible metric for 3D meshes");
   }
   if (metric_id >= 300)
   {
      MFEM_VERIFY(dim == 3, "Incompatible metric for 2D meshes");
   }

   TargetConstructor::TargetType target_t;
   TargetConstructor *target_c = NULL;
   HessianCoefficient *adapt_coeff = NULL;
   HRHessianCoefficient *hr_adapt_coeff = NULL;
   H1_FECollection ind_fec(mesh_poly_deg, dim);
   ParFiniteElementSpace ind_fes(pmesh, &ind_fec);
   ParFiniteElementSpace ind_fesv(pmesh, &ind_fec, dim);
   ParGridFunction size(&ind_fes), aspr(&ind_fes), ori(&ind_fes);
   ParGridFunction aspr3d(&ind_fesv);
   ParGridFunction disc(&ind_fes);
   ParGridFunction disccopy(disc);

   const AssemblyLevel al =
      pa ? AssemblyLevel::PARTIAL : AssemblyLevel::LEGACY;

   switch (target_id)
   {
      case 1: target_t = TargetConstructor::IDEAL_SHAPE_UNIT_SIZE; break;
      case 2: target_t = TargetConstructor::IDEAL_SHAPE_EQUAL_SIZE; break;
      case 3: target_t = TargetConstructor::IDEAL_SHAPE_GIVEN_SIZE; break;
      case 4:
      {
         target_t = TargetConstructor::GIVEN_FULL;
         AnalyticAdaptTC *tc = new AnalyticAdaptTC(target_t);
         adapt_coeff = new HessianCoefficient(dim, metric_id);
         tc->SetAnalyticTargetSpec(NULL, NULL, adapt_coeff);
         target_c = tc;
         break;
      }
      default:
         if (myid == 0) { cout << "Unknown target_id: " << target_id << endl; }
         return 3;
   }
   if (target_c == NULL)
   {
      target_c = new TargetConstructor(target_t, MPI_COMM_WORLD);
   }
   target_c->SetNodes(x0);

   TMOP_QualityMetric *metric_to_use = metric;
   auto tmop_integ = new TMOP_Integrator(metric_to_use, target_c);
   tmop_integ->IntegrateOverTarget(integ_over_targ);

   tmop_integ->SetExactActionFlag(exactaction);

   // Setup the quadrature rules for the TMOP integrator.
   IntegrationRules *irules = NULL;
   switch (quad_type)
   {
      case 1: irules = &IntRulesLo; break;
      case 2: irules = &IntRules; break;
      case 3: irules = &IntRulesCU; break;
      default:
         if (myid == 0) { cout << "Unknown quad_type: " << quad_type << endl; }
         return 3;
   }
   tmop_integ->SetIntegrationRules(*irules, quad_order);
   if (myid == 0 && dim == 2)
   {
      cout << "Triangle quadrature points: "
           << irules->Get(Geometry::TRIANGLE, quad_order).GetNPoints()
           << "\nQuadrilateral quadrature points: "
           << irules->Get(Geometry::SQUARE, quad_order).GetNPoints() << endl;
   }
   if (myid == 0 && dim == 3)
   {
      cout << "Tetrahedron quadrature points: "
           << irules->Get(Geometry::TETRAHEDRON, quad_order).GetNPoints()
           << "\nHexahedron quadrature points: "
           << irules->Get(Geometry::CUBE, quad_order).GetNPoints()
           << "\nPrism quadrature points: "
           << irules->Get(Geometry::PRISM, quad_order).GetNPoints() << endl;
   }

   // 13. Setup the final NonlinearForm (which defines the integral of interest,
   //     its first and second derivatives). Here we can use a combination of
   //     metrics, i.e., optimize the sum of two integrals, where both are
   //     scaled by used-defined space-dependent weights.  Note that there are
   //     no command-line options for the weights and the type of the second
   //     metric; one should update those in the code.
   ParNonlinearForm a(pfespace);
   a.AddDomainIntegrator(tmop_integ);

   // Compute the minimum det(J) of the starting mesh.
   min_detJ = infinity();
   const int NE = pmesh->GetNE();
   for (int i = 0; i < NE; i++)
   {
      const IntegrationRule &ir =
         irules->Get(pfespace->GetFE(i)->GetGeomType(), quad_order);
      ElementTransformation *transf = pmesh->GetElementTransformation(i);
      for (int j = 0; j < ir.GetNPoints(); j++)
      {
         transf->SetIntPoint(&ir.IntPoint(j));
         min_detJ = min(min_detJ, transf->Jacobian().Det());
      }
   }
   real_t minJ0;
   MPI_Allreduce(&min_detJ, &minJ0, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_MIN, MPI_COMM_WORLD);
   min_detJ = minJ0;
   if (myid == 0)
   { cout << "Minimum det(J) of the original mesh is " << min_detJ << endl; }

   if (min_detJ < 0.0)
   {
      // MFEM_ABORT("The input mesh is inverted! Try an untangling metric.");
   }
   if (min_detJ < 0.0)
   {
      MFEM_VERIFY(target_t == TargetConstructor::IDEAL_SHAPE_UNIT_SIZE,
                  "Untangling is supported only for ideal targets.");

      const DenseMatrix &Wideal =
         Geometries.GetGeomToPerfGeomJac(pfespace->GetFE(0)->GetGeomType());
      min_detJ /= Wideal.Det();

      real_t h0min = h0.Min(), h0min_all;
      MPI_Allreduce(&h0min, &h0min_all, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_MIN, MPI_COMM_WORLD);
      // Slightly below minJ0 to avoid div by 0.
      min_detJ -= 0.01 * h0min_all;
   }

   // For HR tests, the energy is normalized by the number of elements.
   const real_t init_energy = a.GetParGridFunctionEnergy(x);
   real_t init_metric_energy = init_energy;
   std::cout << init_metric_energy << " k101\n";

   // Visualize the starting mesh and metric values.
   // Note that for combinations of metrics, this only shows the first metric.
   if (visualization)
   {
      char title[] = "Initial metric values";
      vis_tmop_metric_p(mesh_poly_deg, *metric, *target_c, *pmesh, title, 0);
   }

   // 14. Fix all boundary nodes, or fix only a given component depending on the
   //     boundary attributes of the given mesh.  Attributes 1/2/3 correspond to
   //     fixed x/y/z components of the node.  Attribute dim+1 corresponds to
   //     an entirely fixed node.
    Array<int> ess_vdofs, ess_vdofs_orig;
   if (move_bnd == false)
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      a.SetEssentialBC(ess_bdr);
   }
   else
   {
      int n = 0;
      for (int i = 0; i < pmesh->GetNBE(); i++)
      {
         const int nd = pfespace->GetBE(i)->GetDof();
         const int attr = pmesh->GetBdrElement(i)->GetAttribute();
         if (attr == 1 || attr == 2) { n += nd; }
         // if (attr == 3) { n += nd * dim; }
         // if (attr == 4) { n += nd * dim; }
      }
      ess_vdofs.SetSize(n);
      n = 0;
      for (int i = 0; i < pmesh->GetNBE(); i++)
      {
         const int nd = pfespace->GetBE(i)->GetDof();
         const int attr = pmesh->GetBdrElement(i)->GetAttribute();
         pfespace->GetBdrElementVDofs(i, vdofs);
         if (attr == 1) // Fix x components.
         {
            for (int j = 0; j < nd; j++)
            { ess_vdofs[n++] = vdofs[j]; }
         }
         else if (attr == 2) // Fix y components.
         {
            for (int j = 0; j < nd; j++)
            { ess_vdofs[n++] = vdofs[j+nd]; }
         }
         // else if (attr == 3) // Fix all components.
         // {
         //    for (int j = 0; j < vdofs.Size(); j++)
         //    { ess_vdofs[n++] = vdofs[j]; }
         // }
         // else if (attr == 3 || attr == 4) // Fix all components.
         // {
         //    for (int j = 0; j < vdofs.Size(); j++)
         //    { ess_vdofs[n++] = vdofs[j]; }
         // }
      }
      // ess_vdofs.Append(cornerdofs);
      // ess_vdofs.Sort();
      // ess_vdofs.Unique();
      a.SetEssentialVDofs(ess_vdofs);
   }
   ess_vdofs_orig = ess_vdofs;

   // As we use the inexact Newton method to solve the resulting nonlinear
   // system, here we setup the linear solver for the system's Jacobian.
   Solver *S = NULL, *S_prec = NULL;
#ifdef MFEM_USE_SINGLE
   const real_t linsol_rtol = 1e-5;
#else
   const real_t linsol_rtol = 1e-12;
#endif
   // Level of output.
   IterativeSolver::PrintLevel linsolver_print;
   if (verbosity_level == 2)
   { linsolver_print.Errors().Warnings().FirstAndLast(); }
   if (verbosity_level > 2)
   { linsolver_print.Errors().Warnings().Iterations(); }
   if (lin_solver == 0)
   {
      S = new DSmoother(1, 1.0, max_lin_iter);
   }
   else if (lin_solver == 1)
   {
      CGSolver *cg = new CGSolver(MPI_COMM_WORLD);
      cg->SetMaxIter(max_lin_iter);
      cg->SetRelTol(linsol_rtol);
      cg->SetAbsTol(0.0);
      cg->SetPrintLevel(linsolver_print);
      S = cg;
   }
   else
   {
      MINRESSolver *minres = new MINRESSolver(MPI_COMM_WORLD);
      minres->SetMaxIter(max_lin_iter);
      minres->SetRelTol(linsol_rtol);
      minres->SetAbsTol(0.0);
      minres->SetPrintLevel(linsolver_print);
      if (lin_solver == 3 || lin_solver == 4)
      {
         if (pa)
         {
            MFEM_VERIFY(lin_solver != 4, "PA l1-Jacobi is not implemented");
            auto js = new OperatorJacobiSmoother;
            js->SetPositiveDiagonal(true);
            S_prec = js;
         }
         else
         {
            auto hs = new HypreSmoother;
            hs->SetType((lin_solver == 3) ? HypreSmoother::Jacobi
                        /* */             : HypreSmoother::l1Jacobi, 1);
            hs->SetPositiveDiagonal(true);
            S_prec = hs;
         }
         minres->SetPreconditioner(*S_prec);
      }
      S = minres;
   }

   //
   // Perform the nonlinear optimization.
   //
   const IntegrationRule &ir =
      irules->Get(pfespace->GetFE(0)->GetGeomType(), quad_order);
   TMOPNewtonSolver solver(pfespace->GetComm(), ir, solver_type);
   if (solver_type == 1)
   {
      LBFGSSolver *lbfgs = dynamic_cast<LBFGSSolver *>(&solver);
      lbfgs->SetHistorySize(1);
   }
   // Provide all integration rules in case of a mixed mesh.
   solver.SetIntegrationRules(*irules, quad_order);
   // Specify linear solver when we use a Newton-based solver.
   if (solver_type == 0) { solver.SetPreconditioner(*S); }
   // For untangling, the solver will update the min det(T) values.
   solver.SetMinDetPtr(&min_detJ);
   solver.SetRelTol(solver_rtol);
   solver.SetAbsTol(0.0);
   if (solver_art_type > 0)
   {
      solver.SetAdaptiveLinRtol(solver_art_type, 0.5, 0.9);
   }
   // Level of output.
   IterativeSolver::PrintLevel newton_print;
   if (verbosity_level > 0)
   { newton_print.Errors().Warnings().Iterations(); }
   solver.SetPrintLevel(newton_print);

   solver.SetOperator(a);
   Vector b(0);
   int nnodes = x.Size()/2;
   for (int iter = 0; iter < solver_iter; iter++)
   {
      if (iter == 0 && presolve)
      {
         ess_vdofs = ess_vdofs_orig;
         ess_vdofs.Append(facedofs);
         for (int i = 0; i < facedofs.Size(); i++)
         {
            int dof = facedofs[i] + nnodes;
            ess_vdofs.Append(dof);
         }

         ess_vdofs.Append(facedofs2);
         for (int i = 0; i < facedofs2.Size(); i++)
         {
            int dof = facedofs2[i] + nnodes;
            ess_vdofs.Append(dof);
         }
         ess_vdofs.Sort();
         ess_vdofs.Unique();
         a.SetEssentialVDofs(ess_vdofs);
         solver.SetOperator(a);
         solver.SetMaxIter(100);
      }
      else
      {
         ess_vdofs = ess_vdofs_orig;
         ess_vdofs.Append(cornerdofs);
         ess_vdofs.Sort();
         ess_vdofs.Unique();
         a.SetEssentialVDofs(ess_vdofs);
         solver.SetOperator(a);
         solver.SetMaxIter(1);
      }
      Vector xorig = x;
      solver.Mult(b, x.GetTrueVector());
      x.SetFromTrueVector();
      if (iter == 0 && presolve)
      {
         dc.SetCycle(vcount);
         dc.SetTime((vcount++)*1.0);
         dc.Save();
         continue;
      }
      // continue;
      // continue;

      dx = x;
      dx -= xorig;

      // Find closest surface point
      double alpha = 1.0;
      double foundclosest = false;
      int count = 0;
      const FiniteElementSpace *intnodespace = intmesh->GetNodalFESpace();
      GridFunction *intnodes = intmesh->GetNodes();

      const FiniteElementSpace *intnodespace2 = intmesh2->GetNodalFESpace();
      GridFunction *intnodes2 = intmesh2->GetNodes();
      while (!foundclosest && count < 20)
      {
         if (count > 0) { alpha *= 0.2; }

         count++;

         x = xorig;
         x.Add(alpha, dx);

         dc2.SetCycle(vcount2);
         dc2.SetTime((vcount2++)*1.0);
         dc2.Save();

         Vector fnodes(facedofs.Size()*dim);
         int nnodes = x.Size()/dim;
         for (int i = 0; i < facedofs.Size(); i++)
         {
            fnodes(i) = x(facedofs[i]);
            fnodes(i+facedofs.Size()) = x(facedofs[i]+nnodes);
         }

         finder.FindPointsSurf(fnodes);
         unsigned int maxcode = 0;
         Array<unsigned int> code = finder.GetCode();
         Array<unsigned int> elems = finder.GetElem();
         for (int i = 0; i < code.Size(); i++)
         {
            maxcode = std::max(maxcode, code[i]);
         }
         std::cout << count << " " <<  maxcode << " k10maxcode1\n";
         if (maxcode == 2) {  continue; }

         Vector fnodes2(facedofs2.Size()*dim);
         for (int i = 0; i < facedofs2.Size(); i++)
         {
            fnodes2(i) = x(facedofs2[i]);
            fnodes2(i+facedofs2.Size()) = x(facedofs2[i]+nnodes);
         }
         // fnodes2.Print();
         finder2.FindPointsSurf(fnodes2);

         Array<unsigned int> code2 = finder2.GetCode();
         Array<unsigned int> elems2 = finder2.GetElem();
         maxcode = 0;
         for (int i = 0; i < code2.Size(); i++)
         {
            std::cout << i << " " << code2[i] << " " << elems2[i] << " k10code2\n";
            maxcode = std::max(maxcode, code2[i]);
         }
         std::cout << count << " " <<  maxcode << " k10maxcode2\n";
         // MFEM_ABORT(" ");
         if (maxcode == 2) {  continue; }


         foundclosest = true;

         dx = 0.0;

         Vector rst = finder.GetReferencePosition();
         Vector vals(dim);
         for (int i = 0; i < elems.Size(); i++)
         {
            int dof_id = facedofs[i];
            IntegrationPoint ip;
            ip.x = rst[i];
            intnodes->GetVectorValue(elems[i], ip, vals);
            dx(dof_id) = -(x(dof_id) - vals(0));
            dx(dof_id+nnodes) = -(x(dof_id+nnodes) - vals(1));
            // std::cout << i << " " << dof_id << " " << dx(dof_id) << " " <<
            // dx(dof_id+nnodes) << " k10dx1\n";
         }

         Vector rst2 = finder2.GetReferencePosition();
         for (int i = 0; i < elems2.Size(); i++)
         {
            int dof_id = facedofs2[i];
            IntegrationPoint ip;
            ip.x = rst2[i];
            intnodes2->GetVectorValue(elems2[i], ip, vals);
            dx(dof_id) = -(x(dof_id) - vals(0));
            dx(dof_id+nnodes) = -(x(dof_id+nnodes) - vals(1));
            // std::cout << i << " " << dof_id << " " << dx(dof_id) << " " <<
            // dx(dof_id+nnodes) << " k10dx2\n";
         }


         std::cout << dx.Size() << " Blend displacement\n";
         BlendDisplacement(pmesh, dx, 0.0, 1.0);
         std::cout << "Blend displacement done\n";

         // dc.SetCycle(vcount);
         // dc.SetTime((vcount++)*1.0);
         // dc.Save();

         x += dx;

         dc.SetCycle(vcount);
         dc.SetTime((vcount++)*1.0);
         dc.Save();

         std::cout << iter << " " << alpha << " " << count << " k10alphacount\n";
         // MFEM_ABORT(" ");
      }
      MFEM_VERIFY(foundclosest, "Could not find closest point on surface");
   }

   // 16. Save the optimized mesh to a file. This output can be viewed later
   //     using GLVis: "glvis -m optimized -np num_mpi_tasks".
   {
      ostringstream mesh_name;
      if (presolve)
      {
         mesh_name << "aleuntangled.mesh";
      }
      else
      {
         mesh_name << "optimized.mesh";
      }
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->PrintAsSerial(mesh_ofs);
   }

   // Report the final energy of the functional.
   const real_t fin_energy = a.GetParGridFunctionEnergy(x);
   real_t fin_metric_energy = fin_energy;
   if (myid == 0)
   {
      std::cout << std::scientific << std::setprecision(4);
      cout << "Initial strain energy: " << init_energy
           << " = metrics: " << init_metric_energy
           << " + extra terms: " << init_energy - init_metric_energy << endl;
      cout << "  Final strain energy: " << fin_energy
           << " = metrics: " << fin_metric_energy
           << " + extra terms: " << fin_energy - fin_metric_energy << endl;
      cout << "The strain energy decreased by: "
           << (init_energy - fin_energy) * 100.0 / init_energy << " %." << endl;
   }

   // Visualize the final mesh and metric values.
   if (visualization)
   {
      char title[] = "Final metric values";
      vis_tmop_metric_p(mesh_poly_deg, *metric, *target_c, *pmesh, title, 600);
   }

   // Visualize the mesh displacement.
   if (visualization)
   {
      x0 -= x;
      socketstream sock;
      if (myid == 0)
      {
         sock.open("localhost", 19916);
         sock << "solution\n";
      }
      pmesh->PrintAsOne(sock);
      x0.SaveAsOne(sock);
      if (myid == 0)
      {
         sock << "window_title 'Displacements'\n"
              << "window_geometry "
              << 1200 << " " << 0 << " " << 600 << " " << 600 << "\n"
              << "keys jRmclA" << endl;
      }
   }

   FunctionCoefficient mat_coeff(material_indicator_2d_sharp);
   disc.ProjectCoefficient(mat_coeff);
   if (visualization)
   {
      socketstream sock;
      if (myid == 0)
      {
         sock.open("localhost", 19916);
         sock << "solution\n";
      }
      pmesh->PrintAsOne(sock);
      disc.SaveAsOne(sock);
      if (myid == 0)
      {
         sock << "window_title 'Displacements'\n"
            << "window_geometry "
            << 600 << " " << 600 << " " << 600 << " " << 600 << "\n"
            << "keys jRmclA" << endl;
      }
   }

   if (presolve)
   {
      VisItDataCollection dco("pmeshslidefixbnd", pmesh);
      dco.SetFormat(DataCollection::SERIAL_FORMAT);
      dco.RegisterField("displacement", &dx);
      dco.SetCycle(0);
      dco.SetTime(0.0);
      dco.Save();
   }

   // dc.SetCycle(2);
   // dc.SetTime(2.0);
   // dc.Save();


   delete S;
   delete S_prec;
   delete target_c;
   delete hr_adapt_coeff;
   delete adapt_coeff;
   delete metric;
   delete pfespace;
   delete fec;
   delete pmesh;

   return 0;
}
