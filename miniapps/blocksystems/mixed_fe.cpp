//                       MFEM Example 5 - Parallel Version
//
// Compile with: make ex5p
//
// Sample runs:  mpirun -np 4 ex5p -m ../data/square-disc.mesh
//               mpirun -np 4 ex5p -m ../data/star.mesh
//               mpirun -np 4 ex5p -m ../data/beam-tet.mesh
//               mpirun -np 4 ex5p -m ../data/beam-hex.mesh
//               mpirun -np 4 ex5p -m ../data/escher.mesh
//               mpirun -np 4 ex5p -m ../data/fichera.mesh
//
// Description:  This example code solves a simple 2D/3D mixed Darcy problem
//               corresponding to the saddle point system
//                                 k*u + grad p = f
//                                 - div u      = g
//               with natural boundary condition -p = <given pressure>.
//               Here, we use a given exact solution (u,p) and compute the
//               corresponding r.h.s. (f,g).  We discretize with Raviart-Thomas
//               finite elements (velocity u) and piecewise discontinuous
//               polynomials (pressure p).
//
//               The example demonstrates the use of the BlockMatrix class, as
//               well as the collective saving of several grid functions in a
//               VisIt (visit.llnl.gov) visualization format.
//
//               We recommend viewing examples 1-4 before viewing this example.

#include "mfem.hpp"
#include "mixed_fe_solvers.hpp"
#include <fstream>
#include <iostream>
#include <assert.h>
#include <memory>

using namespace std;
using namespace mfem;

// Define the analytical solution and forcing terms / boundary conditions
void uFun_ex(const Vector & x, Vector & u);
double pFun_ex(const Vector & x);
void fFun(const Vector & x, Vector & f);
double gFun(const Vector & x);
double f_natural(const Vector & x);

int main(int argc, char *argv[])
{
    StopWatch chrono;
    auto ResetTimer = [&chrono]() { chrono.Clear(); chrono.Start(); };

    // 1. Initialize MPI.
    int num_procs, myid;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    bool verbose = (myid == 0);

    // 2. Parse command-line options.
    int order = 0;
    bool visualization = true;
    bool divfree = false;
    bool GMG = 0;
    int num_refines = 2;
    bool ML_particular = false;

    OptionsParser args(argc, argv);
    args.AddOption(&order, "-o", "--order",
                   "Finite element order (polynomial degree).");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                   "--no-visualization",
                   "Enable or disable GLVis visualization.");
    args.AddOption(&divfree, "-df", "--divfree", "-no-df",
                   "--no-divfree",
                   "whether to use the divergence free solver or not.");
    args.AddOption(&ML_particular, "-ml-part", "--multilevel-particular", "-no-ml-part",
                   "--no-multilevel-particular",
                   "whether to use the divergence free solver or not.");
    args.AddOption(&GMG, "-GMG", "--GeometricMG", "-AMG",
                   "--AlgebraicMG",
                   "whether to use goemetric or algebraic multigrid solver.");
    args.AddOption(&num_refines, "-r", "--ref",
                   "Number of parallel refinement steps.");
    args.Parse();
    if (!args.Good())
    {
        if (verbose)
        {
            args.PrintUsage(cout);
        }
        MPI_Finalize();
        return 1;
    }
    if (verbose)
    {
        args.PrintOptions(cout);
    }

    int max_iter(500);
    double rtol(1.e-12);
    double atol(1.e-15);

    // 3. Read the (serial) mesh from the given mesh file on all processors.  We
    //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
    //    and volume meshes with the same code.
    Mesh *mesh = new Mesh(2, 2, 2, mfem::Element::TETRAHEDRON, true);

    int dim = mesh->Dimension();

    // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
    //    this mesh further in parallel to increase the resolution. Once the
    //    parallel mesh is defined, the serial mesh can be deleted.
    ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
    delete mesh;

    Array<int> ess_bdr(pmesh->bdr_attributes.Max());
    ess_bdr = 0;
    ess_bdr[1] = 1;

    // 6. Define a parallel finite element space on the parallel mesh. Here we
    //    use the Raviart-Thomas finite elements of the specified order.
    DFSDataCollector* collector;

    ResetTimer();

    if (divfree)
    {
        DFSParameters param;
        param.verbose = verbose;
        param.ml_particular = ML_particular;
        param.MG_type = GMG ? GeometricMG : AlgebraicMG;
        param.B_has_nullity_one = (ess_bdr.Sum() == ess_bdr.Size());
        param.CTMC_solve_param.max_iter = param.BBT_solve_param.max_iter = max_iter;
        param.CTMC_solve_param.rel_tol = param.BBT_solve_param.rel_tol = rtol;
        param.CTMC_solve_param.abs_tol = param.BBT_solve_param.abs_tol = atol;

        collector = new DFSDataCollector(order, num_refines, pmesh, ess_bdr, param);
    }

    for (int l = 0; l < num_refines; l++)
    {
        pmesh->UniformRefinement();
        if (divfree) collector->CollectData(pmesh);
    }

    RT_FECollection hdiv_coll(order, dim);
    L2_FECollection l2_coll(order, dim);

    auto R_space = divfree ? collector->hdiv_fes_.get() :
                             new ParFiniteElementSpace(pmesh, &hdiv_coll);

    auto W_space = divfree ? collector->l2_fes_.get() :
                             new ParFiniteElementSpace(pmesh, &l2_coll);

    if (verbose) cout << "FE spaces constructed in " << chrono.RealTime() << "\n";

    HYPRE_Int dimR = R_space->GlobalTrueVSize();
    HYPRE_Int dimW = W_space->GlobalTrueVSize();
    HYPRE_Int dimN = divfree ? collector->hcurl_fes_->GlobalTrueVSize() : 0;

    if (verbose)
    {
        cout << "***********************************************************\n";
        cout << "dim(R) = " << dimR << "\n";
        cout << "dim(W) = " << dimW << "\n";
        cout << "dim(R+W) = " << dimR + dimW << "\n";
        if (divfree) cout << "dim(N) = " << dimN << "\n";
        cout << "***********************************************************\n";
    }

    ResetTimer();

    // 7. Define the two BlockStructure of the problem.  block_offsets is used
    //    for Vector based on dof (like ParGridFunction or ParLinearForm),
    //    block_trueOffstes is used for Vector based on trueDof (HypreParVector
    //    for the rhs and solution of the linear system).  The offsets computed
    //    here are local to the processor.
    Array<int> block_offsets(3); // number of variables + 1
    block_offsets[0] = 0;
    block_offsets[1] = R_space->GetVSize();
    block_offsets[2] = W_space->GetVSize();
    block_offsets.PartialSum();

    Array<int> block_trueOffsets(3); // number of variables + 1
    block_trueOffsets[0] = 0;
    block_trueOffsets[1] = R_space->TrueVSize();
    block_trueOffsets[2] = W_space->TrueVSize();
    block_trueOffsets.PartialSum();

    // 8. Define the coefficients, analytical solution, and rhs of the PDE.
    ConstantCoefficient k(1.0);

    VectorFunctionCoefficient fcoeff(dim, fFun);
    FunctionCoefficient fnatcoeff(f_natural);
    FunctionCoefficient gcoeff(gFun);

    VectorFunctionCoefficient ucoeff(dim, uFun_ex);
    FunctionCoefficient pcoeff(pFun_ex);

    // 9. Define the parallel grid function and parallel linear forms, solution
    //    vector and rhs.
    BlockVector x(block_offsets), rhs(block_offsets);
    BlockVector trueX(block_trueOffsets), trueRhs(block_trueOffsets);

    ParGridFunction *u(new ParGridFunction);
    ParGridFunction *p(new ParGridFunction);
    u->MakeRef(R_space, x.GetBlock(0), 0);
    p->MakeRef(W_space, x.GetBlock(1), 0);
    *u = 0.0;
    u->ProjectBdrCoefficientNormal(ucoeff, ess_bdr);
    u->ParallelProject(trueX.GetBlock(0));

    ParLinearForm *fform(new ParLinearForm);
    fform->Update(R_space, rhs.GetBlock(0), 0);
    fform->AddDomainIntegrator(new VectorFEDomainLFIntegrator(fcoeff));
    fform->AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(fnatcoeff));
    fform->Assemble();

    ParLinearForm *gform(new ParLinearForm);
    gform->Update(W_space, rhs.GetBlock(1), 0);
    gform->AddDomainIntegrator(new DomainLFIntegrator(gcoeff));
    gform->Assemble();

    // 10. Assemble the finite element matrices for the Darcy operator
    //
    //                            D = [ M  B^T ]
    //                                [ B   0  ]
    //     where:
    //
    //     M = \int_\Omega k u_h \cdot v_h d\Omega   u_h, v_h \in R_h
    //     B   = -\int_\Omega \div u_h q_h d\Omega   u_h \in R_h, q_h \in W_h
    ParBilinearForm *mVarf(new ParBilinearForm(R_space));
    ParMixedBilinearForm *bVarf(new ParMixedBilinearForm(R_space, W_space));

    HypreParMatrix *M, *B;

    mVarf->AddDomainIntegrator(new VectorFEMassIntegrator(k));
    mVarf->Assemble();
    mVarf->EliminateEssentialBC(ess_bdr, *u, *fform);
    mVarf->Finalize();
    M = mVarf->ParallelAssemble();

    bVarf->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
    bVarf->Assemble();
    bVarf->SpMat() *= -1.0;
    bVarf->EliminateTrialDofs(ess_bdr, *u, *gform);
    bVarf->Finalize();
    B = bVarf->ParallelAssemble();

    fform->ParallelAssemble(trueRhs.GetBlock(0));
    gform->ParallelAssemble(trueRhs.GetBlock(1));

    if (verbose) cout << "Algebraic system assembled in " << chrono.RealTime() << "s.\n";
    ResetTimer();

    if (divfree)
    {
        DivFreeSolver ml_df_solver(*M, *B, collector->hcurl_fes_.get(), collector->GetData());

        if (verbose) cout << "Div free solver constructed in " << chrono.RealTime() << "s.\n";
        ResetTimer();

        ml_df_solver.Mult(trueRhs, trueX);
    }
    else
    {
        OperatorHandle BT(B->Transpose());
        BlockOperator darcyOp(block_trueOffsets);
        darcyOp.SetBlock(0,0, M);
        darcyOp.SetBlock(0,1, BT.As<HypreParMatrix>());
        darcyOp.SetBlock(1,0, B);

        L2H1Preconditioner darcyPr(*M, *B, block_trueOffsets);

        MINRESSolver solver(MPI_COMM_WORLD);
        SetOptions(solver, 0, max_iter, atol, rtol, false);
        solver.SetOperator(darcyOp);
        solver.SetPreconditioner(darcyPr);

        if (verbose) cout << "Div free solver constructed in " << chrono.RealTime() << "s.\n";
        ResetTimer();

        solver.Mult(trueRhs, trueX);

        PrintConvergence(solver, verbose);
    }
    string solver_name = divfree ? "Divergence free" : "Block-diagonal-preconditioned MINRES";
    if (verbose) cout << solver_name << " solver took " << chrono.RealTime() << "s.\n";

    // 13. Extract the parallel grid function corresponding to the finite element
    //     approximation X. This is the local solution on each processor. Compute
    //     L2 error norms.
    u->Distribute(&(trueX.GetBlock(0)));
    p->Distribute(&(trueX.GetBlock(1)));

    int order_quad = max(2, 2*order+1);
    const IntegrationRule *irs[Geometry::NumGeom];
    for (int i=0; i < Geometry::NumGeom; ++i)
    {
        irs[i] = &(IntRules.Get(i, order_quad));
    }

    double err_u  = u->ComputeL2Error(ucoeff, irs);
    double norm_u = ComputeGlobalLpNorm(2, ucoeff, *pmesh, irs);
    double err_p  = p->ComputeL2Error(pcoeff, irs);
    double norm_p = ComputeGlobalLpNorm(2, pcoeff, *pmesh, irs);

    if (verbose)
    {
        cout << "|| u_h - u_ex || / || u_ex || = " << err_u / norm_u << "\n";
        cout << "|| p_h - p_ex || / || p_ex || = " << err_p / norm_p << "\n";
    }

    // 17. Free the used memory.
    delete fform;
    delete gform;
    delete u;
    delete p;
    delete B;
    delete M;
    delete mVarf;
    delete bVarf;
    if (!divfree) delete W_space;
    if (!divfree) delete R_space;
    if (divfree) delete collector;
    delete pmesh;

    MPI_Finalize();

    return 0;
}


void uFun_ex(const Vector & x, Vector & u)
{
    double xi(x(0));
    double yi(x(1));
    double zi(0.0);
    if (x.Size() == 3)
    {
        zi = x(2);
    }

    u(0) = - exp(xi)*sin(yi)*cos(zi);
    u(1) = - exp(xi)*cos(yi)*cos(zi);

    if (x.Size() == 3)
    {
        u(2) = exp(xi)*sin(yi)*sin(zi);
    }
}

// Change if needed
double pFun_ex(const Vector & x)
{
    double xi(x(0));
    double yi(x(1));
    double zi(0.0);

    if (x.Size() == 3)
    {
        zi = x(2);
    }

    return exp(xi)*sin(yi)*cos(zi);
}

void fFun(const Vector & x, Vector & f)
{
    f = 0.0;
}

double gFun(const Vector & x)
{
    if (x.Size() == 3)
    {
        return -pFun_ex(x);
    }
    else
    {
        return 0;
    }
}

double f_natural(const Vector & x)
{
    return (-pFun_ex(x));
}
