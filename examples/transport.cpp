#include "mfem.hpp"
#include <cstdio>
#include <fstream>
#include <vector>
#include <iostream>
using namespace mfem;


class CoefficientWithState : public Coefficient
{
protected:
    double (*Function)(const Vector &, const Vector &);

public:
    /// Define a time-independent coefficient from a C-function
    CoefficientWithState(double (*f)(const Vector &, const Vector &))
    {
      Function = f;
    }

    /// Evaluate coefficient
    virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip) {
        double x[3];
        Vector transip(x, 3);
        T.Transform(ip, transip);
        return ((*Function)(state_, transip));
    }
    
    void SetState(Vector state) { 
        state_.SetSize(state.Size());
        state_ = state;
    }

private:
   Vector state_;
};


// freq used in definition of psi_function2(omega,x)
#define PI 3.14159265358979323846
double freq = 1.52;
double sigma_t_function(const Vector &x);
double sigma_s_function(const Vector &x);
double psi_function2(const Vector &omega, const Vector &x);
double Q_function2(const Vector &omega, const Vector &x);
double inflow_function2(const Vector &omega, const Vector &x);

struct AIR_parameters {
   int distance;
   std::string prerelax;
   std::string postrelax;
   int interp_type;
   int relax_type;
   int coarsen_type;
   double strength_tolC;
   double strength_tolR;
   double filter_tolR;
   double filterA_tol;
};


int main(int argc, char *argv[])
{
    /* Initialize MPI; if using more than one thread, need version of 
       MPI that supports MPI_THREAD_MULTIPLE */
    int num_procs, myid;
    int provided; 
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid); 
    
    /* Simulation parameters */
    int use_gmres        = 0;
    int max_iter         = 100;
    int dim              = 2;
    int meshOrder        = 4;
    int feOrder          = 4;
    int ser_ref_levels   = 2;
    int par_ref_levels   = 0;
    double theta         = 3*PI/16.0;
    int blocksize;
    int print_level      = 2;
    double solve_tol     = 1e-12;
    int basis_type       = 1;
    int trisolve_precond = 0;
    int cycle_type       = 1;
    int is_triangular    = 0;
    int gmres_switch     = -1;

    // Mesh parameters
    int num_coarse_els;

    AIR_parameters AIR = {15, "", "FA", 100, 6, 6, 0.1, 0.01, 0.0, 1e-4};
    const char* temp_prerelax = "";
    const char* temp_postrelax = "FFC";

    OptionsParser args(argc, argv);
    args.AddOption(&feOrder, "-o", "--order",
                  "Finite element order.");
    args.AddOption(&print_level, "-p", "--print-level",
                  "Hypre print level.");
    args.AddOption(&solve_tol, "-tol", "--solve-tol",
                  "Tolerance to solve linear system.");
    args.AddOption(&cycle_type, "-c", "--cycle-type",
                  "Cycle type; 0=F, 1=V, 2=W.");
    args.AddOption(&ser_ref_levels, "-rs", "--level",
                  "Number levels serial mesh refinement.");
    args.AddOption(&par_ref_levels, "-rp", "--level",
                  "Number levels parallel mesh refinement.");
    args.AddOption(&dim, "-d", "--dim",
                  "Problem dimension.");
    args.AddOption(&basis_type, "-b", "--basis-type",
                  "DG finite element basis type. 0 for G-Leg, 1 for G-Lob.");
    args.AddOption(&use_gmres, "-gmres", "--use-gmres",
                  "Boolean to use GMRES as solver (default with AIR preconditioning).");
    args.AddOption(&trisolve_precond, "-trisolve", "--precond-trisolve",
                  "Precondition GMRES with an on-processor triangular solve.");
    args.AddOption(&is_triangular, "-tri", "--is-triangular",
                  "Assume matrix is triangular for AIR.");
    args.AddOption(&gmres_switch, "-gswitch", "--gmres-switch",
                  "Size to solve AIR systems using gmres vs. direct.");
    args.AddOption(&(AIR.distance), "-Ad", "--AIR-distance",
                  "Distance restriction neighborhood for AIR.");
    args.AddOption(&(AIR.interp_type), "-Ai", "--AIR-interpolation",
                  "Index for hypre interpolation routine.");
    args.AddOption(&(AIR.coarsen_type), "-Ac", "--AIR-coarsen_type",
                  "Index for hypre coarsening routine.");
    args.AddOption(&(AIR.strength_tolC), "-AsC", "--AIR-strengthC",
                   "Theta value determining strong connections for AIR (coarsen_type).");
    args.AddOption(&(AIR.strength_tolR), "-AsR", "--AIR-strengthR",
                   "Theta value determining strong connections for AIR (restriction).");
    args.AddOption(&(AIR.filter_tolR), "-AfR", "--AIR-filterR",
                   "Theta value eliminating small entries in restriction (after building).");
    args.AddOption(&(AIR.filterA_tol), "-Af", "--AIR-filter",
                  "Theta value to eliminate small connections in AIR hierarchy. Use -1 to specify O(h).");
    args.AddOption(&(AIR.relax_type), "-Ar", "--AIR-relaxation",
                  "Index for hypre relaxation routine.");
    args.AddOption(&temp_prerelax, "-Ar1", "--AIR-prerelax",
                  "String denoting prerelaxation scheme; e.g., FCC.");
    args.AddOption(&temp_postrelax, "-Ar2", "--AIR-postrelax",
                  "String denoting postrelaxation scheme; e.g., FFC.");
    args.Parse();
    AIR.prerelax = std::string(temp_prerelax);
    AIR.postrelax = std::string(temp_postrelax);
    if (trisolve_precond) use_gmres = 1;
    if (!args.Good()) {
        if (myid == 0) {
            args.PrintUsage(std::cout);
        }
        MPI_Finalize();
        return 1;
    }
    if (myid == 0) {
        args.PrintOptions(std::cout);
    }
    
    meshOrder = feOrder;
    blocksize = (feOrder+1)*(feOrder+1);

    // Set up serial coarse mesh
    const char *mesh_file = "../data/UnsQuad.0.mesh";
    Mesh *coarse_mesh = new Mesh(mesh_file, 1, 1);
    dim = coarse_mesh->Dimension();
    for (int lev = 0; lev<ser_ref_levels; lev++) {
        coarse_mesh->UniformRefinement();
    }
    num_coarse_els = coarse_mesh->GetNE();

    // Define a parallel mesh by partitioning the serial mesh.
    // TODO ; might not need this eventually
    ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *coarse_mesh);
    for (int lev = 0; lev<par_ref_levels; lev++) {
        pmesh->UniformRefinement();
    }

    // Get parallel element partitioning (this array has the processor
    // ID on which a given element is in the "home" domain).
    Mesh *serial_mesh = new Mesh(*coarse_mesh);
    std::vector<int> my_coarse_inds;
    int *partitioning;
    partitioning = coarse_mesh->GeneratePartitioning(num_procs);

    // Mark elements stored on other processors as part of coarse-grid, save indices
    int coarse_ind = 0;
    for (int el=0; el<serial_mesh->GetNE(); el++) {
        if (partitioning[el] != myid) {
            serial_mesh->GetElement(el)->loc_coarse_el = coarse_ind;
            serial_mesh->GetElement(el)->glob_coarse_el = el;
            my_coarse_inds.push_back(el);
            coarse_ind++;
        }
    }

    // Refine "fine" elements par_ref_levels times to match parallel mesh
    int non_conforming = 1;     // Bool to use nonconforming refinement on all elements
    Array marked_elements;
    for (int lev = 0; lev<par_ref_levels; lev++) {

        // Mark all fine elements (not indicated previously as coarse) for refinement
        marked_elements.SetSize(0);
        for (int el = 0; el < serial_mesh->GetNE(); el++) {
            if (serial_mesh->GetElement(i)->loc_coarse_el < 0) {            
                marked_elements.Append(el);
            }
        }

        // Apply non-conforming refinement on marked elements
        serial_mesh->GeneralRefinement(marked_elements, non_conforming);
    }

    // Construct ghost domain around home domain

        // -->  TODO (not necessary for initial implementation, probably later)

    // Mark fine elements in serial mesh and construct mapping operators
    // between parallel and serial spaces
    int fine_el = 0;
    std::vector<int> par_to_ser;
    std::map<int, int> ser_to_par;
    for (int el=0; el<serial_mesh->GetNE(); el++) {
        if (serial_mesh->GetElement(el)->loc_coarse_el < 0) {
            serial_mesh->GetElement(el)->fine_el = fine_el;
            my_coarse_inds.push_back(el);
            par_to_ser.push_back(el);
            ser_to_par[el] = fine_el;
            fine_el++;
        }
    }

    // Define finite element spaces on parallel mesh, serial mesh, and original
    // coarse serial mesh
    DG_FECollection fec(feOrder, dim, basis_type);
    FiniteElementSpace coarse_fes(coarse_mesh, &fec);
    FiniteElementSpace serial_fes(serial_mesh, &fec);
    ParFiniteElementSpace parallel_fes(pmesh, &fec);

    // Define angle of flow, coefficients and integrators
    std::vector<double> omega0 {cos(theta), sin(theta)};
    Vector omega(&omega0[0],2);

    CoefficientWithState inflow_coeff(inflow_function2);
    CoefficientWithState Q_coeff(Q_function2);  
    inflow_coeff.SetState(omega);
    Q_coeff.SetState(omega);
    FunctionCoefficient sigma_t_coeff(sigma_t_function);
    VectorConstantCoefficient *direction = new VectorConstantCoefficient(omega); 

    // TODO : Construct bilinear form on serial mesh




    // Construct parallel bilinear form on parallel mesh
    ParBilinearForm *bl_form = new ParBilinearForm(&pfes);
    bl_form -> AddDomainIntegrator(new MassIntegrator(sigma_t_coeff));
    bl_form -> AddDomainIntegrator(new TransposeIntegrator(
                new ConvectionIntegrator(*direction, -1.0)));
    bl_form -> AddInteriorFaceIntegrator(new DGTraceIntegrator(*direction, 1.0, 0.5));  // Interior face integrators
    bl_form -> AddBdrFaceIntegrator(new DGTraceIntegrator(*direction, 1.0, 0.5));       // Boundary face integrators
    bl_form -> Assemble();
    bl_form -> Finalize();

    // Form the right-hand side
    ParLinearForm *l_form = new ParLinearForm(&pfes);
    l_form -> AddBdrFaceIntegrator(new BoundaryFlowIntegrator(inflow_coeff, *direction, -1.0, -0.5));
    l_form -> AddDomainIntegrator(new DomainLFIntegrator(Q_coeff));
    l_form -> Assemble();

    // Build sparse matrices and scale system by block-diagonal inverse
    HypreParMatrix *A = bl_form -> ParallelAssemble();
    HypreParVector *B = l_form -> ParallelAssemble();
    Vector X(pfes.GetVSize());

    HypreParMatrix A_s;
    HypreParVector B_s;
    delete bl_form;
    delete l_form;

    if (myid == 0) {
        std::cout << "Local n: " << A->GetNumRows() << "\nGlobal n = " << A->M() \
              << "\nNNZ = " << A->NNZ() << "\n";
    }
    BlockInverseScale(A, &A_s, B, &B_s, blocksize, 1);

    delete A;
    delete B;

    // Build Hypre solver and preconditioner; solve linear system
    HypreBoomerAMG *AMG_solver = NULL;
    HypreGMRES *GMRES_solver = NULL;
    HypreTriSolve *preconditioner = NULL;
    
    if (!trisolve_precond) {
        AMG_solver = new HypreBoomerAMG(A_s);
        AMG_solver->SetLAIROptions(AIR.distance, AIR.prerelax, AIR.postrelax,
                                  AIR.strength_tolC, AIR.strength_tolR, AIR.filter_tolR,
                                  AIR.interp_type, AIR.relax_type, AIR.filterA_tol,
                                  AIR.coarsen_type, -1, 1);  
        AMG_solver->SetCycleType(cycle_type);
        AMG_solver->SetMaxLevels(50);
        if (is_triangular) {
            AMG_solver->SetTriangular();
        }
        if (gmres_switch > 0) {
            AMG_solver->SetGMRESSwitchR(gmres_switch);
        }
        if (use_gmres) {
            GMRES_solver = new HypreGMRES(A_s);
            GMRES_solver->SetTol(solve_tol);
            GMRES_solver->SetMaxIter(max_iter);
            GMRES_solver->SetPrintLevel(print_level);
            GMRES_solver->SetPreconditioner(*AMG_solver);
            GMRES_solver->SetZeroInintialIterate();
            GMRES_solver->iterative_mode = false;
        }
        else {
            AMG_solver->SetPrintLevel(print_level);
            AMG_solver->SetTol(solve_tol);
            AMG_solver->SetMaxIter(max_iter);
        }
    }
    else {
        preconditioner = new HypreTriSolve();
        GMRES_solver = new HypreGMRES(A_s);
        GMRES_solver->SetTol(solve_tol);
        GMRES_solver->SetMaxIter(max_iter);
        GMRES_solver->SetPrintLevel(print_level);
        GMRES_solver->SetPreconditioner(*preconditioner);
        GMRES_solver->SetZeroInintialIterate();
        GMRES_solver->iterative_mode = false;
    }

    // scale the rhs and solve system
    if (use_gmres) {
        GMRES_solver->Mult(B_s, X);
        delete GMRES_solver;
        if (trisolve_precond) delete preconditioner;
        else delete AMG_solver;
    }
    else {
        AMG_solver->Mult(B_s, X);
        delete AMG_solver; 
    }

    delete partitioning;
    delete mesh;
    delete pmesh;
    MPI_Finalize();
    return 0;
}


////////////////////////////////////////////////////////////////////////////////

double psi_function2(const Vector &omega, const Vector &x) {
    double x1 = x(0);
    double x2 = x(1);
    double psi = .5 * (x1*x1 + x2*x2 + 1.) + std::cos(freq*(x1+x2));
    psi = psi * (omega(0)*omega(0) + omega(1));
    return psi;
}


double sigma_s_function(const Vector &x) {
    return 0.;
}


double sigma_t_function(const Vector &x) {
    double x1 = x(0);
    double x2 = x(1);
    // double val_abs = x1*x2 + x1*x1 + 1.;
    double val_abs = x1*x2 + x1*x1;
    double sig_s = sigma_s_function(x);
    return val_abs + sig_s;
}


double Q_function2(const Vector &omega, const Vector &x) {
    double x1 = x(0);
    double x2 = x(1);
    double sig = sigma_t_function(x);
    double val_sin = freq * std::sin(freq*(x1+x2));
    double psi_dx_dot_v = omega(0)*(x1-val_sin) + omega(1)*(x2-val_sin);
    psi_dx_dot_v = psi_dx_dot_v * (omega(0)*omega(0) + omega(1));
    double psi = psi_function2(omega, x);
    return psi_dx_dot_v + sig * psi;
}


double inflow_function2(const Vector &omega, const Vector &x) {
    double psi = psi_function2(omega, x);
    return psi;
}




