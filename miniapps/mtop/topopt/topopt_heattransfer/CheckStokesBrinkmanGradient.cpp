#include "mfem.hpp"
#include "diffusion_mass_solver.hpp"
#include "HeatTransferLinForms.hpp"
#include "HeatTransferTopOpt.hpp"
#include <iostream>
#include <iomanip>
#include <memory>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
    // 1. Initialize MPI and HYPRE
    Mpi::Init();
    int num_procs = Mpi::WorldSize();
    int myid = Mpi::WorldRank();
    Hypre::Init();

    // 2. Options and Parameters  
    int order = 2;
    int ser_ref_levels = 2;
    real_t b_val = 0.001;
    real_t a_val = 100.0;

    OptionsParser args(argc, argv);
    args.AddOption(&order, "-o", "--order", "Finite element order.");
    args.AddOption(&ser_ref_levels, "-rs", "--refine-serial", "Number of serial refinements.");
    args.Parse(); 

    if (!args.Good())
    {
        if (Mpi::Root()) args.PrintUsage(cout);
        return 1;
    }
    if (Mpi::Root()) args.PrintOptions(cout);

    // 3. Setup Mesh
    Mesh *mesh = new Mesh(Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL));
    for (int i = 0; i < ser_ref_levels; i++)
    { 
        mesh->UniformRefinement();
    }
    ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
    delete mesh; 
 
    // 4. Finite Element Spaces
    // Velocity (H1, dim), Pressure (H1, order-1), Design (L2, order-1)
    H1_FECollection v_coll(order, pmesh->Dimension());
    H1_FECollection p_coll(order - 1, pmesh->Dimension());
    H1_FECollection d_coll(order - 1, pmesh->Dimension());

    ParFiniteElementSpace V_space(pmesh, &v_coll, pmesh->Dimension());
    ParFiniteElementSpace P_space(pmesh, &p_coll);
    ParFiniteElementSpace D_space(pmesh, &d_coll);

    // 5. Grid Functions
    ParGridFunction rho(&D_space);
    ParGridFunction h_gf(&D_space);
    ParGridFunction u_gf(&V_space);
    ParGridFunction p_gf(&P_space);
    ParGridFunction v_gf(&V_space);
    ParGridFunction q_gf(&P_space);

    // Randomize states with specific seeds for reproducibility
    rho.Randomize(1);
    h_gf.Randomize(2);
    u_gf.Randomize(3);
    p_gf.Randomize(4);
    v_gf.Randomize(5);
    q_gf.Randomize(6);

    // We restrict rho to [0.2, 0.8] so that `rho +/- scale * h` strictly stays
    // within [0,1]. This avoids tripping the bounds-clamping logic in 
    // BrinkmanCoefficient which would otherwise ruin the finite difference derivatives.
    rho *= 0.6;
    rho += 0.2;

    // Scale h_gf so the maximum perturbation is 0.1
    double h_max = h_gf.Normlinf();
    h_gf /= (h_max * 10.0);

    // Save baseline design
    ParGridFunction rho_0 = rho;

    // 6. Block Vectors for U = [u, p] and V_test = [v, q]
    Array<int> block_offsets(3);
    block_offsets[0] = 0;
    block_offsets[1] = V_space.GetTrueVSize();
    block_offsets[2] = P_space.GetTrueVSize();
    block_offsets.PartialSum();

    BlockVector U_block(block_offsets), V_block(block_offsets);
    u_gf.GetTrueDofs(U_block.GetBlock(0));
    p_gf.GetTrueDofs(U_block.GetBlock(1));
    v_gf.GetTrueDofs(V_block.GetBlock(0));
    q_gf.GetTrueDofs(V_block.GetBlock(1));

    // 7. Exact Gradient Setup using your Integrator
    // 1. Create your Brinkman Coefficient as usual
    BrinkmanCoefficient b_cf(&rho, b_val, a_val);

    // 2. Create a scalar H1 space matching the velocity order
    ParFiniteElementSpace scalar_V_space(pmesh, &v_coll, 1);

    // 3. Project the coefficient onto this scalar space
    ParGridFunction b_gf(&scalar_V_space);
    b_gf.ProjectCoefficient(b_cf);

    // 4. Configure the solver with the GridFunction, NOT the raw coefficient
    ConstantCoefficient visc(1.0); 
    BrinkmanStokesSolver solver(V_space, P_space);
    solver.SetRelTol(1e-13);
    solver.SetAbsTol(0.0);
    solver.SetMaxIter(500);
    solver.SetPrintLevel(0);
    solver.SetViscosity(visc);
    
    // Pass the projected grid function
    solver.SetBrinkmanPenalization(b_gf);

    ParLinearForm dJ(&D_space);
    dJ.AddDomainIntegrator(new StokesMassBrinkmanDesignLFIntegrator(rho, u_gf, v_gf, b_cf));
    dJ.Assemble();
    
    Vector dJ_tv;
    dJ.ParallelAssemble(dJ_tv);

    Vector h_tv;
    h_gf.GetTrueDofs(h_tv);   

    // Exact Directional Derivative
    double exact_grad = InnerProduct(MPI_COMM_WORLD, dJ_tv, h_tv);

    if (Mpi::Root())
    {
        cout << "\n======================================================\n";
        cout << setprecision(10) << "Exact Gradient (dJ/drho * h): " << exact_grad << "\n";
        cout << "======================================================\n";
        cout << left << setw(15) << "Scale"
             << setw(20) << "FD Approx"
             << setw(20) << "Rel Error"
             << setw(20) << "Abs Error" << "\n";
        cout << "------------------------------------------------------\n";
    }

    // 8. Finite Difference Evaluation Function
    //ConstantCoefficient visc(1.0); // Required by StokesSolver
    //BrinkmanStokesSolver solver(V_space, P_space);
    // solver.SetViscosity(visc);
    // solver.SetBrinkmanPenalization(b_cf);

    // Lambda to evaluate J = V_block^T * S(rho) * U_block
    auto ComputeJ = [&](double eps) {
        // Apply perturbation
        rho = rho_0;
        rho.Add(eps, h_gf);
        b_gf.ProjectCoefficient(b_cf);

        // Tell the solver the coefficient changed and re-assemble
        solver.SetNeedsAssembly(true);
        solver.Assemble();

        const Operator *S = solver.GetOperator();
        Vector Y(S->Height());
        S->Mult(U_block, Y);

        // Compute dot product globally across all true DOFs
        return mfem::InnerProduct(MPI_COMM_WORLD, V_block, Y);
    };

    // 9. Sweep through scales to verify quadratic convergence of the error
    double scale = 1.0;
    double best_rel_err = 1.0;

    for (int i = 0; i < 7; i++)
    {
        double Jp = ComputeJ(scale);
        double Jm = ComputeJ(-scale);
        
        // Centered difference
        double fd = (Jp - Jm) / (2.0 * scale);

        double abs_err = abs(fd - exact_grad);
        double rel_err = abs_err / max(abs(exact_grad), 1e-12);
        best_rel_err = min(best_rel_err, rel_err); 

        if (Mpi::Root())
        {
            cout << left << setw(15) << scientific << setprecision(3) << scale
                 << setw(20) << setprecision(10) << fd
                 << setw(20) << setprecision(4) << rel_err
                 << setw(20) << setprecision(4) << abs_err << "\n";
        }

        scale *= 0.1;
    }

    if (Mpi::Root())
    {
        // Expecting convergence to at least 1e-5 relative error or better. 
        if (best_rel_err < 1e-5)
            cout << "\nSUCCESS: Gradient matches finite differences!\n";
        else
            cout << "\nWARNING: Gradient and FD differ significantly.\n";
    }

    delete pmesh;
    return 0;
}