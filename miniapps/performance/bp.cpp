// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project
// (17-SC-20-SC), a collaborative effort of two U.S. Department of Energy
// organizations (Office of Science and the National Nuclear Security
// Administration) responsible for the planning and preparation of a capable
// exascale ecosystem, including software, applications, hardware, advanced
// system engineering and early testbed platforms, in support of the nation's
// exascale computing imperative.


//==============================================================================
//                  MFEM Bake-off Problems 1, 2, 3, and 4
//                                Version 1
//
// Compile with: see README.md
//
// Sample runs:  see README.md
//
// Description:  These benchmarks (CEED Bake-off Problems BP1 and BP3) test the
//               performance of high-order mass (BP1) and stiffness (BP3) matrix
//               operator evaluation with "partial assembly" algorithms.
//
//               Code is based on MFEM's HPC ex1, http://mfem.org/performance.
//
//               More details about CEED's bake-off problems can be found at
//               http://ceed.exascaleproject.org/bps.
//==============================================================================

#include <mfem.hpp>

using namespace mfem;

#ifndef GEOM
#define GEOM Geometry::CUBE
#endif

#ifndef MESH_P
#define MESH_P 3
#endif

#ifndef SOL_P
#define SOL_P 3
#endif

#ifndef IR_ORDER
#define IR_ORDER 0
#endif

#ifndef IR_TYPE
// 0 - Gauss quadrature, 1 - Gauss-Lobatto quadrature
#define IR_TYPE 0
#endif

#ifndef PROBLEM
// 0- Diffusion, else TMassKernel
#define PROBLEM 0
#endif

#ifndef VDIM
#define VDIM 1
#endif

#ifndef MESH_FILE
#define MESH_FILE "../../data/fichera.mesh"
#endif

// This vector layout is used for the solution space only.
#ifndef VEC_LAYOUT
#define VEC_LAYOUT Ordering::byVDIM
#endif

// Define template parameters for optimized build.
const Geometry::Type geom     = GEOM;      // mesh elements  (default: hex)
const int            mesh_p   = MESH_P;    // mesh curvature (default: 3)
const int            sol_p    = SOL_P;     // solution order (default: 3)
const int            ir_q     = IR_TYPE ? sol_p+1 : sol_p+2;
const int            ir_order = IR_ORDER ? IR_ORDER :
                                (IR_TYPE ? 2*ir_q-3 : 2*ir_q-1);


// Workaround for a bug in XL C++ on BG/Q version 12.01.0000.0014
#if defined(__xlC__) && (__xlC__ < 0x0d00)
#include <../mfem/linalg/tlayout.hpp>
namespace mfem
{
const int mesh_dim = Geometry::Constants<geom>::Dimension;
template class StridedLayout1D<mesh_dim*VDIM,1>;
}
#endif // defined(__xlC__) && (__xlC__ < 0x0d00)


#include <mfem-performance.hpp>
#include <fstream>
#include <iostream>

using namespace std;

IntegrationRules GaussLobattoRules(0, Quadrature1D::GaussLobatto);

template <int Dim, int Q, typename real_t>
class GaussLobattoIntegrationRule
   : public TProductIntegrationRule<Dim, Q, 2*Q-3, real_t>
{
public:
   typedef TProductIntegrationRule<Dim,Q,2*Q-3,real_t> base_class;

   using base_class::geom;
   using base_class::order;
   using base_class::qpts_1d;

protected:
   using base_class::weights_1d;

public:
   GaussLobattoIntegrationRule()
   {
      const IntegrationRule &ir_1d = Get1DIntRule();
      MFEM_ASSERT(ir_1d.GetNPoints() == qpts_1d, "quadrature rule mismatch");
      for (int j = 0; j < qpts_1d; j++)
      {
         weights_1d.data[j] = ir_1d.IntPoint(j).weight;
      }
   }

   static const IntegrationRule &Get1DIntRule()
   {
      return GaussLobattoRules.Get(Geometry::SEGMENT, order);
   }
   static const IntegrationRule &GetIntRule()
   {
      return GaussLobattoRules.Get(geom, order);
   }
};


// Static mesh type
typedef H1_FiniteElement<geom,mesh_p>         mesh_fe_t;
typedef H1_FiniteElementSpace<mesh_fe_t>      mesh_fes_t;
typedef TMesh<mesh_fes_t>                     mesh_t;

// Static solution finite element space type
typedef H1_FiniteElement<geom,sol_p>          sol_fe_t;
typedef H1_FiniteElementSpace<sol_fe_t>       sol_fes_t;

// Static quadrature, coefficient and integrator types
#if (IR_TYPE == 0)
typedef TIntegrationRule<geom,ir_order>       int_rule_t;
#else
const int rdim = Geometry::Constants<geom>::Dimension;
typedef GaussLobattoIntegrationRule<rdim,ir_order/2+2,double>
                                              int_rule_t;
#endif
typedef TConstantCoefficient<>                coeff_t;
#if (PROBLEM == 0)
typedef TIntegrator<coeff_t,TDiffusionKernel> integ_t;
#else
typedef TIntegrator<coeff_t,TMassKernel> integ_t;
#endif
#if (VDIM == 1)
typedef ScalarLayout                          vec_layout_t;
#else
typedef VectorLayout<VEC_LAYOUT,VDIM>         vec_layout_t;
#endif

// Static bilinear form type, combining the above types
typedef TBilinearForm<mesh_t,sol_fes_t,int_rule_t,integ_t,
        vec_layout_t> HPCBilinearForm;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   const int            vdim     = VDIM;
   const Ordering::Type ordering = VEC_LAYOUT; // for solution space only

   // 2. Parse command-line options.
   const char *mesh_file = MESH_FILE;
   int ser_ref_levels = -1;
   int par_ref_levels = +1;
   Array<int> nxyz;
   int order = sol_p;
   const char *basis_type = "G"; // Gauss-Lobatto
   bool static_cond = false;
   const char *pc = "none";
   bool perf = true;
   bool matrix_free = true;
   int max_iter = 50;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&nxyz, "-c", "--cartesian-partitioning",
                  "Use Cartesian partitioning.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&basis_type, "-b", "--basis-type",
                  "Basis: G - Gauss-Lobatto, P - Positive, U - Uniform");
   args.AddOption(&perf, "-perf", "--hpc-version", "-std", "--standard-version",
                  "Enable high-performance, tensor-based, assembly/evaluation.");
   args.AddOption(&matrix_free, "-mf", "--matrix-free", "-asm", "--assembly",
                  "Use matrix-free evaluation or efficient matrix assembly in "
                  "the high-performance version.");
   args.AddOption(&pc, "-pc", "--preconditioner",
                  "Preconditioner: lor - low-order-refined (matrix-free) AMG, "
                  "ho - high-order (assembled) AMG, none.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&max_iter, "-mi", "--max-iter",
                  "Maximum number of iterations.");
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
   if (static_cond && perf && matrix_free)
   {
      if (myid == 0)
      {
         cout << "\nStatic condensation can not be used with matrix-free"
              " evaluation!\n" << endl;
      }
      MPI_Finalize();
      return 2;
   }
   MFEM_VERIFY(perf || !matrix_free,
               "--standard-version is not compatible with --matrix-free");
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   enum PCType { NONE, LOR, HO };
   PCType pc_choice;
   if (!strcmp(pc, "ho")) { pc_choice = HO; }
   else if (!strcmp(pc, "lor")) { pc_choice = LOR; }
   else if (!strcmp(pc, "none")) { pc_choice = NONE; }
   else
   {
      mfem_error("Invalid Preconditioner specified");
      return 3;
   }

   // See class BasisType in fem/fe_coll.hpp for available basis types
   int basis = BasisType::GetType(basis_type[0]);
   if (myid == 0)
   {
      cout << "Using " << BasisType::Name(basis) << " basis ..." << endl;
   }

   // 3. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 4. Check if the optimized version matches the given mesh
   if (perf)
   {
      if (myid == 0)
      {
         cout << "High-performance version using integration rule with "
              << int_rule_t::qpts << " points ..." << endl;
         cout << "Quadrature rule type: "
              << (IR_TYPE == 0 ? "Gauss" : "Gauss-Lobatto") << endl;
      }
      if (!mesh_t::MatchesGeometry(*mesh))
      {
         if (myid == 0)
         {
            cout << "The given mesh does not match the optimized 'geom' parameter.\n"
                 << "Recompile with suitable 'geom' value." << endl;
         }
         delete mesh;
         MPI_Finalize();
         return 4;
      }
      else if (!mesh_t::MatchesNodes(*mesh))
      {
         if (myid == 0)
         {
            cout << "Switching the mesh curvature to match the "
                 << "optimized value (order " << mesh_p << ") ..." << endl;
         }
         mesh->SetCurvature(mesh_p, false, -1, Ordering::byNODES);
      }
   }

   // 5. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 10,000 elements.
   {
      int ref_levels =
         (int)floor(log(10000./mesh->GetNE())/log(2.)/dim);
      ref_levels = (ser_ref_levels != -1) ? ser_ref_levels : ref_levels;
      for (int l = 0; l < ref_levels; l++)
      {
         if (myid == 0)
         {
            cout << "Serial refinement: level " << l << " -> level " << l+1
                 << " ..." << flush;
         }
         mesh->UniformRefinement();
         MPI_Barrier(MPI_COMM_WORLD);
         if (myid == 0)
         {
            cout << " done." << endl;
         }
      }
   }
   if (!perf && mesh->NURBSext)
   {
      const int new_mesh_p = std::min(sol_p, mesh_p);
      if (myid == 0)
      {
         cout << "NURBS mesh: switching the mesh curvature to be "
              << "min(sol_p, mesh_p) = " << new_mesh_p << " ..." << endl;
      }
      mesh->SetCurvature(new_mesh_p, false, -1, Ordering::byNODES);
   }

   // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   MFEM_VERIFY(nxyz.Size() == 0 || nxyz.Size() == mesh->SpaceDimension(),
               "Expected " << mesh->SpaceDimension() << " integers with the "
               "option --cartesian-partitioning.");
   int *partitioning = nxyz.Size() ? mesh->CartesianPartitioning(nxyz) : NULL;
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh, partitioning);
   delete [] partitioning;
   delete mesh;
   {
      for (int l = 0; l < par_ref_levels; l++)
      {
         if (myid == 0)
         {
            cout << "Parallel refinement: level " << l << " -> level " << l+1
                 << " ..." << flush;
         }
         pmesh->UniformRefinement();
         MPI_Barrier(MPI_COMM_WORLD);
         if (myid == 0)
         {
            cout << " done." << endl;
         }
      }
   }
   if (pmesh->MeshGenerator() & 1) // simplex mesh
   {
      MFEM_VERIFY(pc_choice != LOR, "triangle and tet meshes do not support"
                  " the LOR preconditioner yet");
   }

   pmesh->PrintInfo(cout);

   // 7. Define a parallel finite element space on the parallel mesh. Here we
   //    use continuous Lagrange finite elements of the specified order. If
   //    order < 1, we instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim, basis);
   }
   else if (pmesh->GetNodes())
   {
      fec = pmesh->GetNodes()->OwnFEC();
      if (myid == 0)
      {
         cout << "Using isoparametric FEs: " << fec->Name() << endl;
      }
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim, basis);
   }
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec,
                                                              vdim, ordering);
   HYPRE_Int size = fespace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   ParMesh *pmesh_lor = NULL;
   FiniteElementCollection *fec_lor = NULL;
   ParFiniteElementSpace *fespace_lor = NULL;
   if (pc_choice == LOR)
   {
      int basis_lor = basis;
      if (basis == BasisType::Positive) { basis_lor=BasisType::ClosedUniform; }
      pmesh_lor = new ParMesh(pmesh, order, basis_lor);
      fec_lor = new H1_FECollection(1, dim);
      fespace_lor = new ParFiniteElementSpace(pmesh_lor, fec_lor,
                                              vdim, ordering);
   }

   // 8. Check if the optimized version matches the given space
   if (perf && !sol_fes_t::Matches(*fespace))
   {
      if (myid == 0)
      {
         cout << "The given order does not match the optimized parameter.\n"
              << "Recompile with suitable 'sol_p' value." << endl;
      }
      delete fespace;
      delete fec;
      delete mesh;
      MPI_Finalize();
      return 5;
   }

   // 9. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs. In this example, the boundary conditions are defined
   //    by marking all the boundary attributes from the mesh as essential
   //    (Dirichlet) and converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   if (pmesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 10. Set up the parallel linear form b(.) which corresponds to the
   //     right-hand side of the FEM linear system, which in this case is
   //     (1,phi_i) where phi_i are the basis functions in fespace.
   ParLinearForm *b = new ParLinearForm(fespace);
   ConstantCoefficient one(1.0);
   Vector uvec(vdim);
   for (int i = 0; i < vdim; i++)
   {
      uvec(i) = i + 1.0;
   }
   uvec /= uvec.Norml2();
   VectorConstantCoefficient unit_vec(uvec);
   if (vdim == 1)
   {
      b->AddDomainIntegrator(new DomainLFIntegrator(one));
   }
   else
   {
      b->AddDomainIntegrator(new VectorDomainLFIntegrator(unit_vec));
   }
   b->Assemble();

   // 11. Define the solution vector x as a parallel finite element grid
   //     function corresponding to fespace. Initialize x with initial guess of
   //     zero, which satisfies the boundary conditions.
   ParGridFunction x(fespace);
   x = 0.0;

   // 12. Set up the parallel bilinear form a(.,.) on the finite element space
   //     that will hold the matrix corresponding to the Laplacian operator.
   ParBilinearForm *a = new ParBilinearForm(fespace);
   ParBilinearForm *a_pc = NULL;
   if (pc_choice == LOR) { a_pc = new ParBilinearForm(fespace_lor); }
   if (pc_choice == HO)  { a_pc = new ParBilinearForm(fespace); }

   // 13. Assemble the parallel bilinear form and the corresponding linear
   //     system, applying any necessary transformations such as: parallel
   //     assembly, eliminating boundary conditions, applying conforming
   //     constraints for non-conforming AMR, static condensation, etc.
   if (static_cond)
   {
      a->EnableStaticCondensation();
      MFEM_VERIFY(pc_choice != LOR,
                  "cannot use LOR preconditioner with static condensation");
   }

   if (myid == 0)
   {
      cout << "Assembling the local matrix ..." << flush;
   }
#ifdef USE_MPI_WTIME
   double my_rt_start = MPI_Wtime();
#else
   tic_toc.Clear();
   tic_toc.Start();
#endif
   // Pre-allocate sparsity assuming dense element matrices; the actual memory
   // allocation happens when a->Assemble() is called.
   a->UsePrecomputedSparsity();

   HPCBilinearForm *a_hpc = NULL;
   Operator *a_oper = NULL;

   if (!perf)
   {
      // Standard assembly using a diffusion domain integrator
      if (vdim == 1)
      {
         a->AddDomainIntegrator(new DiffusionIntegrator(one));
      }
      else
      {
         a->AddDomainIntegrator(new VectorDiffusionIntegrator(one));
      }
      a->Assemble();
   }
   else
   {
      // High-performance assembly/evaluation using the templated operator type
      a_hpc = new HPCBilinearForm(integ_t(coeff_t(1.0)), *fespace);
      if (matrix_free)
      {
         a_hpc->Assemble(); // partial assembly
      }
      else
      {
         a_hpc->AssembleBilinearForm(*a); // full matrix assembly
      }
   }
#ifdef USE_MPI_WTIME
   double rt_min, rt_max, my_rt;
   my_rt = MPI_Wtime() - my_rt_start;
#else
   tic_toc.Stop();
   double rt_min, rt_max, my_rt;
   my_rt = tic_toc.RealTime();
#endif
   MPI_Reduce(&my_rt, &rt_min, 1, MPI_DOUBLE, MPI_MIN, 0, pmesh->GetComm());
   MPI_Reduce(&my_rt, &rt_max, 1, MPI_DOUBLE, MPI_MAX, 0, pmesh->GetComm());
   if (myid == 0)
   {
      cout << " done, " << rt_max << " (" << rt_min << ") s." << endl;
      cout << "\n\"DOFs/sec\" in assembly: "
           << 1e-6*size/rt_max << " ("
           << 1e-6*size/rt_min << ") million.\n" << endl;
   }

   // 14. Define and apply a parallel PCG solver for AX=B with the BoomerAMG
   //     preconditioner from hypre.

   // Setup the operator matrix (if applicable)
   HypreParMatrix A;
   Vector B, X;
   if (myid == 0)
   {
      cout << "FormLinearSystem() ..." << endl;
   }
#ifdef USE_MPI_WTIME
   my_rt_start = MPI_Wtime();
#else
   tic_toc.Clear();
   tic_toc.Start();
#endif
   if (perf && matrix_free)
   {
      a_hpc->FormLinearSystem(ess_tdof_list, x, *b, a_oper, X, B);
      if (myid == 0)
      {
         cout << "Size of linear system: " << size << endl;
      }
   }
   else
   {
      a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
      HYPRE_Int glob_size = A.GetGlobalNumRows();
      HYPRE_Int glob_nnz = A.NNZ();
      if (myid == 0)
      {
         cout << "Size of linear system: " << glob_size << endl;
         cout << "Average nonzero entries per row: "
              << 1.0*glob_nnz/glob_size << endl;
      }
      a_oper = &A;
   }
#ifdef USE_MPI_WTIME
   my_rt = MPI_Wtime() - my_rt_start;
#else
   tic_toc.Stop();
   my_rt = tic_toc.RealTime();
#endif
   MPI_Reduce(&my_rt, &rt_min, 1, MPI_DOUBLE, MPI_MIN, 0, pmesh->GetComm());
   MPI_Reduce(&my_rt, &rt_max, 1, MPI_DOUBLE, MPI_MAX, 0, pmesh->GetComm());
   if (myid == 0)
   {
      cout << "FormLinearSystem() ... done, " << rt_max << " (" << rt_min
           << ") s." << endl;
      cout << "\n\"DOFs/sec\" in FormLinearSystem(): "
           << 1e-6*size/rt_max << " ("
           << 1e-6*size/rt_min << ") million.\n" << endl;
   }

   // Setup the matrix used for preconditioning
   if (myid == 0)
   {
      cout << "Assembling the preconditioning matrix ..." << flush;
   }
#ifdef USE_MPI_WTIME
   my_rt_start = MPI_Wtime();
#else
   tic_toc.Clear();
   tic_toc.Start();
#endif

   HypreParMatrix A_pc;
   if (pc_choice == LOR)
   {
      // TODO: assemble the LOR matrix using the performance code
      if (vdim == 1)
      {
         a_pc->AddDomainIntegrator(new DiffusionIntegrator(one));
      }
      else
      {
         a_pc->AddDomainIntegrator(new VectorDiffusionIntegrator(one));
      }
      a_pc->UsePrecomputedSparsity();
      a_pc->Assemble();
      a_pc->FormSystemMatrix(ess_tdof_list, A_pc);
   }
   else if (pc_choice == HO)
   {
      if (!matrix_free)
      {
         A_pc.MakeRef(A); // matrix already assembled, reuse it
      }
      else
      {
         a_pc->UsePrecomputedSparsity();
         a_hpc->AssembleBilinearForm(*a_pc);
         a_pc->FormSystemMatrix(ess_tdof_list, A_pc);
      }
   }
#ifdef USE_MPI_WTIME
   my_rt = MPI_Wtime() - my_rt_start;
#else
   tic_toc.Stop();
   my_rt = tic_toc.RealTime();
#endif
   MPI_Reduce(&my_rt, &rt_min, 1, MPI_DOUBLE, MPI_MIN, 0, pmesh->GetComm());
   MPI_Reduce(&my_rt, &rt_max, 1, MPI_DOUBLE, MPI_MAX, 0, pmesh->GetComm());
   if (myid == 0)
   {
      cout << " done, " << rt_max << "s." << endl;
   }

   // Solve with CG or PCG, depending if the matrix A_pc is available
   CGSolver *pcg;
   pcg = new CGSolver(MPI_COMM_WORLD);
   pcg->SetRelTol(1e-6);
   pcg->SetMaxIter(max_iter);
   pcg->SetPrintLevel(3);

   HypreSolver *amg = NULL;

   pcg->SetOperator(*a_oper);
   if (pc_choice != NONE)
   {
      HypreBoomerAMG *bamg = new HypreBoomerAMG(A_pc);
      if (vdim > 1 && ordering == Ordering::byVDIM)
      {
         bamg->SetSystemsOptions(vdim);
      }
      amg = bamg;
      pcg->SetPreconditioner(*amg);
   }

#ifdef USE_MPI_WTIME
   my_rt_start = MPI_Wtime();
#else
   tic_toc.Clear();
   tic_toc.Start();
#endif

   pcg->Mult(B, X);

#ifdef USE_MPI_WTIME
   my_rt = MPI_Wtime() - my_rt_start;
#else
   tic_toc.Stop();
   my_rt = tic_toc.RealTime();
#endif
   delete amg;

   MPI_Reduce(&my_rt, &rt_min, 1, MPI_DOUBLE, MPI_MIN, 0, pmesh->GetComm());
   MPI_Reduce(&my_rt, &rt_max, 1, MPI_DOUBLE, MPI_MAX, 0, pmesh->GetComm());
   if (myid == 0)
   {
      // Note: In the pcg algorithm, the number of operator Mult() calls is
      //       N_iter and the number of preconditioner Mult() calls is N_iter+1.
      cout << "Total CG time:    " << rt_max << " (" << rt_min << ") sec."
           << endl;
      cout << "Time per CG step: "
           << rt_max / pcg->GetNumIterations() << " ("
           << rt_min / pcg->GetNumIterations() << ") sec." << endl;
      cout << "\n\"DOFs/sec\" in CG: "
           << 1e-6*size*pcg->GetNumIterations()/rt_max << " ("
           << 1e-6*size*pcg->GetNumIterations()/rt_min << ") million.\n"
           << endl;
   }

   // 15. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   if (perf && matrix_free)
   {
      a_hpc->RecoverFEMSolution(X, *b, x);
   }
   else
   {
      a->RecoverFEMSolution(X, *b, x);
   }

   // 16. Save the refined mesh and the solution in parallel. This output can
   //     be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
   if (false)
   {
      ostringstream mesh_name, sol_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      sol_name << "sol." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }

   // 17. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *pmesh << x << flush;
   }

   // 18. Free the used memory.
   delete a;
   delete a_hpc;
   if (a_oper != &A) { delete a_oper; }
   delete a_pc;
   delete b;
   delete fespace;
   delete fespace_lor;
   delete fec_lor;
   delete pmesh_lor;
   if (order > 0) { delete fec; }
   delete pmesh;
   delete pcg;

   MPI_Finalize();

   return 0;
}
