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
    int par_ref_levels = 2;
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
    args.AddOption(&par_ref_levels, "-r", "--ref",
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

    // 3. Read the (serial) mesh from the given mesh file on all processors.  We
    //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
    //    and volume meshes with the same code.
    Mesh *mesh = new Mesh(2, 2, 2, mfem::Element::TETRAHEDRON, true);

    int dim = mesh->Dimension();

    // 4. Refine the serial mesh on all processors to increase the resolution. In
    //    this example we do 'ref_levels' of uniform refinement. We choose
    //    'ref_levels' to be the largest number that gives a final mesh with no
    //    more than 10,000 elements.

    // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
    //    this mesh further in parallel to increase the resolution. Once the
    //    parallel mesh is defined, the serial mesh can be deleted.
    ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
    delete mesh;

    Array<int> ess_bdr(pmesh->bdr_attributes.Max());
    ess_bdr = 0;

    // 6. Define a parallel finite element space on the parallel mesh. Here we
    //    use the Raviart-Thomas finite elements of the specified order.
    ND_FECollection hcurl_coll(order+1, dim);
    RT_FECollection hdiv_coll(order, dim);
    L2_FECollection l2_coll(order, dim);

    ParFiniteElementSpace* N_space;
    ParFiniteElementSpace* R_space = new ParFiniteElementSpace(pmesh, &hdiv_coll);
    ParFiniteElementSpace* W_space = new ParFiniteElementSpace(pmesh, &l2_coll);

    // Constructing multigrid hierarchy while refining the mesh (if GMG is true)
    InterpolationCollector* P_N;
    HdivL2Hierarchy* hdiv_l2_hierarchy;

    chrono.Clear();
    chrono.Start();

    if (divfree && ML_particular)
    {
        hdiv_l2_hierarchy = new HdivL2Hierarchy(*R_space, *W_space, par_ref_levels, ess_bdr);
    }

    if (divfree)
    {
        N_space = new ParFiniteElementSpace(pmesh, &hcurl_coll);
        P_N = new InterpolationCollector(*N_space, par_ref_levels);
    }

    for (int l = 1; l < par_ref_levels+1; l++)
    {
        pmesh->UniformRefinement();

        R_space->Update();
        W_space->Update();

        if (divfree && ML_particular)
        {
            hdiv_l2_hierarchy->CollectData(*R_space, *W_space);
        }

        if (divfree)
        {
            N_space->Update();
            if (GMG)
            {
                P_N->CollectData(*N_space);
            }
        }
    }
    if (verbose)
        cout << "Divergence free hierarchy constructed in " << chrono.RealTime() << "\n";

    HYPRE_Int dimR = R_space->GlobalTrueVSize();
    HYPRE_Int dimW = W_space->GlobalTrueVSize();
    HYPRE_Int dimN = divfree ? N_space->GlobalTrueVSize() : 0;

    if (verbose)
    {
        cout << "***********************************************************\n";
        cout << "dim(R) = " << dimR << "\n";
        cout << "dim(W) = " << dimW << "\n";
        cout << "dim(R+W) = " << dimR + dimW << "\n";
        if (divfree)
            cout << "dim(N) = " << dimN << "\n";
        cout << "***********************************************************\n";
    }

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

    ParLinearForm *fform(new ParLinearForm);
    fform->Update(R_space, rhs.GetBlock(0), 0);
    fform->AddDomainIntegrator(new VectorFEDomainLFIntegrator(fcoeff));
    fform->AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(fnatcoeff));
    fform->Assemble();
    fform->ParallelAssemble(trueRhs.GetBlock(0));

    ParLinearForm *gform(new ParLinearForm);
    gform->Update(W_space, rhs.GetBlock(1), 0);
    gform->AddDomainIntegrator(new DomainLFIntegrator(gcoeff));
    gform->Assemble();
    gform->ParallelAssemble(trueRhs.GetBlock(1));

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
    mVarf->Finalize();
    M = mVarf->ParallelAssemble();

    bVarf->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
    bVarf->Assemble();
    bVarf->Finalize();
    bVarf->SpMat() *= -1.0;
    B = bVarf->ParallelAssemble();

    HypreParMatrix *BT = B->Transpose();

    ParDiscreteLinearOperator *DiscreteCurl;
    if (divfree)
    {
        DiscreteCurl = new ParDiscreteLinearOperator(N_space, R_space);
        DiscreteCurl->AddDomainInterpolator(new CurlInterpolator);
        DiscreteCurl->Assemble();
        DiscreteCurl->Finalize();
    }

    int max_iter(500);
    double rtol(1.e-9);
    double atol(1.e-12);

    chrono.Clear();
    chrono.Start();

    BlockOperator darcyOp(block_trueOffsets);
    if (divfree)
    {
        unique_ptr<HypreParMatrix> C(DiscreteCurl->ParallelAssemble());

        MLDivFreeSolveParameters param;
        param.verbose = verbose;
        param.ml_part = ML_particular;
        MLDivFreeSolver ml_df_solver(*hdiv_l2_hierarchy, *M, *B, *BT, *C, param);
        if (GMG) ml_df_solver.SetupMG(*P_N); else ml_df_solver.SetupAMS(*N_space);
        ml_df_solver.Mult(trueRhs, trueX);
    }
    else
    {
        darcyOp.SetBlock(0,0, M);
        darcyOp.SetBlock(0,1, BT);
        darcyOp.SetBlock(1,0, B);

        L2H1Preconditioner darcyPr(*M, *B, block_trueOffsets);

        MINRESSolver solver(MPI_COMM_WORLD);
        SetOptions(solver, 0, max_iter, atol, rtol);
        solver.SetOperator(darcyOp);
        solver.SetPreconditioner(darcyPr);
        trueX = 0.0;
        solver.Mult(trueRhs, trueX);

        PrintConvergence(solver, verbose);
    }
    string solver_name = divfree ? "Divergence free" : "Block-diagonal-preconditioned MINRES";
    if (verbose) cout << solver_name << " solver took " << chrono.RealTime() << "s.\n";

    // 13. Extract the parallel grid function corresponding to the finite element
    //     approximation X. This is the local solution on each processor. Compute
    //     L2 error norms.
    ParGridFunction *u(new ParGridFunction);
    ParGridFunction *p(new ParGridFunction);
    u->MakeRef(R_space, x.GetBlock(0), 0);
    p->MakeRef(W_space, x.GetBlock(1), 0);
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

    // 14. Save the refined mesh and the solution in parallel. This output can be
    //     viewed later using GLVis: "glvis -np <np> -m mesh -g sol_*".
    {
        ostringstream mesh_name, u_name, p_name;
        mesh_name << "mesh." << setfill('0') << setw(6) << myid;
        u_name << "sol_u." << setfill('0') << setw(6) << myid;
        p_name << "sol_p." << setfill('0') << setw(6) << myid;

        ofstream mesh_ofs(mesh_name.str().c_str());
        mesh_ofs.precision(8);
        pmesh->Print(mesh_ofs);

        ofstream u_ofs(u_name.str().c_str());
        u_ofs.precision(8);
        u->Save(u_ofs);

        ofstream p_ofs(p_name.str().c_str());
        p_ofs.precision(8);
        p->Save(p_ofs);
    }

    // 15. Save data in the VisIt format
    VisItDataCollection visit_dc("Mixed-FE-Parallel", pmesh);
    visit_dc.RegisterField("velocity", u);
    visit_dc.RegisterField("pressure", p);
    visit_dc.Save();

    // 16. Send the solution by socket to a GLVis server.
    if (visualization)
    {
        char vishost[] = "localhost";
        int  visport   = 19916;
        socketstream u_sock(vishost, visport);
        u_sock << "parallel " << num_procs << " " << myid << "\n";
        u_sock.precision(8);
        u_sock << "solution\n" << *pmesh << *u << "window_title 'Velocity'"
               << endl;
        // Make sure all ranks have sent their 'u' solution before initiating
        // another set of GLVis connections (one from each rank):
        MPI_Barrier(pmesh->GetComm());
        socketstream p_sock(vishost, visport);
        p_sock << "parallel " << num_procs << " " << myid << "\n";
        p_sock.precision(8);
        p_sock << "solution\n" << *pmesh << *p << "window_title 'Pressure'"
               << endl;
    }

    // 17. Free the used memory.
    delete fform;
    delete gform;
    delete u;
    delete p;
    delete BT;
    delete B;
    delete M;
    delete mVarf;
    delete bVarf;
    delete W_space;
    delete R_space;
    if (divfree)
    {
        if (ML_particular) delete hdiv_l2_hierarchy;
        delete DiscreteCurl;
        delete N_space;
        delete P_N;
    }

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
