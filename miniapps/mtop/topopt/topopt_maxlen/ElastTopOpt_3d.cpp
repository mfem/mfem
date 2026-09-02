// Linear elasticity topology optimization with a max thickness constraint, 3D.
// Thickness measure is calculated by solving an advection pde, one advection
// direction per ray, giving one thickness constraint per direction.
//
// The elasticity/filter/optimizer all run on the full mesh (Option C): elements
// outside domain_attr are frozen, either fixed solid (solid_attr, rho = 1) or
// fixed void (rho = 0).  The volume budget covers only domain_attr.
//
// With no -m the built-in 3x1x1 Cartesian hex beam is used.  The
// circular_plate_hex_sleeves_embedded_cylinder.msh setup is selected by name.
//
// Before the optimization every solver (filter, one elasticity solve per load
// case, one advection solve per ray) is run once on the initial design and
// timed (with its Krylov / pseudo-time iteration count).  -no-opt stops there
// (initial timed evaluation only); -pv also writes the initial fields to
// ParaView/<run_tag>_init.  -spl controls the live iterative-solver report
// (default on; -spl 0 silences it, -spl 2 is verbose).
//
// The advection (thickness) solve dominates the cost.  Its evaluation space is
// low-order DG (-dgo, default 1) and its operator is full/sparse-assembled by
// default (-adv-fa; -adv-pa for matrix-free partial assembly, better at high
// -dgo).  The pseudo-transient march is tuned with -cfl / -atf / -atol.
//
// Sample run:  mpirun -np 8 ./ElastTopOpt_3d -r 2 -rf 0.05 -vf 0.4
// Sample run:  mpirun -np 8 ./ElastTopOpt_3d -m circular_plate_hex_sleeves_embedded_cylinder.msh -vf 0.3 -pv
// Sample run:  mpirun -np 8 ./ElastTopOpt_3d -m circular_plate_hex_sleeves_embedded_cylinder.msh -no-opt -pv

#include "mfem.hpp"
#include "ElastTopOpt.hpp"
#include "qoi.hpp"
#include "pseudo_transient_solver.hpp"
#include "../../diffusion_mass_solver.hpp"
#include "../../mma/MMA_MFEM.hpp"
#include "../../mtop_solvers.hpp"
#include "../../linear_elasticity_solver.hpp"
#include "checkpoint.hpp"
#include <memory>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>

using namespace std;
using namespace mfem;

// Boundary/body data for one elasticity solve.
struct LoadCase
{
    Array<int>    clamp_attrs;
    Array<int>    load_attrs;                          // surface tractions
    Array<real_t> fx, fy, fz;

    // Body-force-density loads.  Each entry loads a group of element attributes
    // with either a constant vector (value) or a spatially varying field (fn);
    // fn takes precedence when set.  One entry per attribute group lets each
    // group carry its own amplitude.
    struct VolumeLoad
    {
        Array<int> attrs;
        Vector     value;                                    // constant body force
        void      (*fn)(const Vector &, Vector &) = nullptr; // varying body force
    };
    std::vector<VolumeLoad> vol_loads;
};

// Full problem definition for one mesh
// one elasticity solve per load case.
struct MeshProblem
{
    Array<int>            domain_attr;       // designable subdomain(s)
    Array<int>            solid_attr;        // fixed-solid volumes: design rho pinned
                                             // to 1 (E_max stiffness), not optimized
    Array<int>            outer_bdr_attrs;   // free surfaces: rho~ = 0 in the filter
    Array<int>            neumann_bdr_attrs; // filter natural (zero-flux) BC: no
                                             // Dirichlet value imposed there
    Array<int>            ray_bdr_attrs;     // outflow candidate surfaces for the
                                             // advection rays (thickness measure)
    Array<int>            cut_faces;         // serial-mesh face ids on the design/void
                                             // interface; each gets a boundary element
                                             // so the filter pins rho~ = 1 there

    // advection ray directions for the max-thickness measure (one QoI per ray);
    // empty -> the thickness constraint is off (n_dir = 0)
    std::vector<std::unique_ptr<VectorCoefficient>> rays;

    std::vector<LoadCase> cases;
};

MeshProblem loadMesh(int myid, const char *mesh_file, Mesh &mesh);

// Extract Mesh with non-zero density
void SaveSolidSubmesh(ParMesh &pmesh, ParGridFunction &desi_density,
                      ParGridFunction &phys_density, const std::string &run_tag, 
                      int order, real_t threshold = 0.1);

int main(int argc, char *argv[])
{
    Mpi::Init();
    int num_procs = Mpi::WorldSize();
    int myid = Mpi::WorldRank();
    Hypre::Init();

    // Timer
    double init_time = MPI_Wtime();

    // Progress instrumentation: prints "[<elapsed>s] <stage>" on rank 0, flushed,
    // so a long-running or stuck phase is always identifiable on screen.
    auto stage = [&](const std::string &msg)
    {
        if (myid == 0)
        {
            mfem::out << "[" << fixed << setprecision(2)
                      << (MPI_Wtime() - init_time) << "s] " << msg
                      << defaultfloat << setprecision(6) << std::endl;
        }
    };

    // 1. Options.
    const char *mesh_file = "";       // empty: use the built-in Cartesian beam
    int    ref_levels   = 0;
    int    order        = 2;
    real_t r_f          = 3.0;        // min filter length (mesh units; the
                                     // circular plate spans ~120, so O(few))
    real_t alpha_min    = 1e-6;       // thickness variable lower bound
    real_t alpha_max    = 1.0;        // thickness variable upper bound
    real_t beta         = 2.0;        // Heaviside beta
    real_t eta          = 0.2;        // Heaviside eta
    real_t vol_fraction = 0.4;
    int    max_it       = 300;
    real_t tol          = 1e-4;       // stopping tol on iteration error
    real_t move         = 0.1;        // MMA move limit
    real_t epsilon      = 1e-2;       // thickness residual tolerance

    // advection (ray) thickness-solve controls -- this solve dominates the cost
    int    dg_order     = 1;          // DG order of the advection eval space
    bool   adv_pa       = false;      // advection operator: partial vs full assembly
                                     // (PA wins at high order; full/sparse at p=1)
    real_t adv_cfl      = 0.5;        // CFL number -> pseudo-time step
    real_t adv_tfinal   = 100.0;     // absolute pseudo-time cap (steps = t_final/dt)
    real_t adv_tol      = 1e-6;       // steady-state relative-rate tolerance
    bool   minv_fa      = true;       // DG mass inverse: exact assembled block-
                                     // diagonal (true) vs matrix-free per-elem CG
    real_t minv_tol     = 1e-8;       // matrix-free DG mass-inverse CG rel tol

    int  cp          = 0;       // 0 = off, 1 = rho only, 2 = full state
    int  restart     = 0;       // 0 = off, 1 = rho only, 2 = full state
    int  pc_type     = 2;       // 0 = Jacobi, 1 = LOR diagonal AMG, 2 = LOR monolithic AMG
    bool lor_by_vdim = true;    // monolithic LOR (-pc 2) ordering: byVDIM vs byNODES
    const int seed   = 0;

    bool visualization = true;
    bool paraview      = false;
    bool optimize      = true;   // run the optimization loop after the initial eval
    int  solver_print  = 1;      // iterative-solver report: 0 off, 1 on
                                 // (CG history / PT summary), 2 verbose
                                 // (+ AMG, + every pseudo-time step)

    const real_t E_min    = 1e-3;     // SIMP void stiffness
    const real_t E_max    = 1.0;      // SIMP E max
    const real_t exponent = 1.0;      // SIMP exponent (applied to the projection)
    // --- PLAIN SIMP ---  use p = 3 when SIMP acts directly on rho~
    // const real_t exponent = 3.0;

    int    init_it   = 25;
    real_t decay     = 0.5;
    real_t eps_floor = 1e-10;
    int    decay_int = 50;

    int    beta_steps = 100;           // Heaviside beta continuation steps
    real_t beta_max   = 2.0;          // Heaviside beta max value

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
                    "mesh file to use; omit for the built-in Cartesian beam");
    args.AddOption(&ref_levels, "-r", "--refine", "uniform refinement levels");
    args.AddOption(&order, "-o", "--order", "finite element order");
    args.AddOption(&vol_fraction, "-vf", "--volume-fraction", "volume fraction");
    args.AddOption(&r_f, "-rf", "--r_fwidth", "min filter width");
    args.AddOption(&alpha_min, "-amin", "--alpha_min", "lower bound for the thickness variables");
    args.AddOption(&alpha_max, "-amax", "--alpha_max", "upper bound for the thickness variables");
    args.AddOption(&beta, "-b", "--beta", "Heaviside beta");
    args.AddOption(&beta_steps, "-bs", "--beta-steps", "Heaviside beta continuation steps");
    args.AddOption(&beta_max, "-bm", "--beta-max", "Heaviside beta max value");
    args.AddOption(&eta, "-eta", "--eta", "Heaviside eta");
    args.AddOption(&epsilon, "-e", "--epsilon", "thickness residual tolerance (initial)");
    args.AddOption(&decay, "-d", "--decay", "decay rate of epsilon");
    args.AddOption(&decay_int, "-di", "--decay_int", "decay interval of epsilon");
    args.AddOption(&init_it, "-ii", "--init_it", "initial iterations before decay");
    args.AddOption(&max_it, "-mi", "--max-it", "max optimization iterations");
    args.AddOption(&tol, "-tol", "--tol", "stopping tol on max design change");
    args.AddOption(&move, "-mv", "--move", "MMA move limit");
    args.AddOption(&pc_type, "-pc", "--elast-precond", "elasticity preconditioner: "
                    "0 = Jacobi, 1 = LOR diagonal AMG,  2 = LOR monolithic AMG");
    args.AddOption(&lor_by_vdim, "-vdim", "--by-vdim", "-nodes", "--by-nodes",
                    "monolithic LOR ordering: byVDIM / byNODES");
    args.AddOption(&cp, "-cp", "--checkpoint",
                    "checkpointing: 0 = off, 1 = rho only, 2 = rho + alpha + MMA state");
    args.AddOption(&restart, "-restart", "--restart",
                    "restart: 0 = off, 1 = load rho only, 2 = load full state and resume");
    args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                    "store solution in paraview");
    args.AddOption(&visualization, "-vis", "--visualization",
                    "-no-vis", "--no-visualization", "enable GLVis visualization");
    args.AddOption(&optimize, "-opt", "--optimize", "-no-opt", "--no-optimize",
                    "run the optimization loop (off: initial timed evaluation only)");
    args.AddOption(&solver_print, "-spl", "--solver-print-level",
                    "iterative-solver report (filter / elasticity / advection): "
                    "0 = off, 1 = on, 2 = verbose");
    args.AddOption(&dg_order, "-dgo", "--dg-order",
                    "DG order of the advection (thickness) evaluation space");
    args.AddOption(&adv_pa, "-adv-pa", "--advection-partial-assembly",
                    "-adv-fa", "--advection-full-assembly",
                    "advection operator assembly: partial (matrix-free) or full (sparse)");
    args.AddOption(&adv_cfl, "-cfl", "--adv-cfl",
                    "advection pseudo-transient CFL number (larger = bigger time step)");
    args.AddOption(&adv_tfinal, "-atf", "--adv-terminal-time",
                    "advection pseudo-transient absolute time cap");
    args.AddOption(&adv_tol, "-atol", "--adv-tol",
                    "advection pseudo-transient steady-state rate tolerance");
    args.AddOption(&minv_fa, "-minv-fa", "--minv-full-assembly",
                    "-minv-mf", "--minv-matrix-free",
                    "DG mass inverse: exact assembled block-diagonal vs per-element CG");
    args.AddOption(&minv_tol, "-mit", "--minv-tol",
                    "matrix-free DG mass-inverse per-element CG relative tolerance");
    args.Parse();
    if (!args.Good())
    {
        if (myid == 0) { args.PrintUsage(cout); }
        return 1;
    }
    if (myid == 0) { args.PrintOptions(cout); }

    // initial (uniform) design density -- depends on the parsed options
    const real_t domain_init = alpha_max * vol_fraction;

    // 2. Load the mesh and the problem description (domain, loads).
    stage(std::string("loading mesh: ") +
          (mesh_file[0] ? mesh_file : "<built-in Cartesian beam>"));
    Mesh mesh;
    MeshProblem prob = loadMesh(myid, mesh_file, mesh);
    stage("mesh loaded (serial: " + std::to_string(mesh.GetNE()) + " elements, "
          + std::to_string(mesh.GetNBE()) + " bdr elements)");

    const int n_elast_solve = static_cast<int>(prob.cases.size());
    Array<int> &domain_attr       = prob.domain_attr;
    Array<int> &solid_attr        = prob.solid_attr;
    Array<int> &outer_bdr_attrs   = prob.outer_bdr_attrs;
    Array<int> &neumann_bdr_attrs = prob.neumann_bdr_attrs;
    Array<int> &ray_bdr_attrs     = prob.ray_bdr_attrs;

    // Give every listed design/void interface face a boundary element so the
    // PDE filter can impose rho~ = 1 there.  Done on the serial mesh, before
    // partitioning and refinement, so the ids refer to one global face
    // numbering (mesh.GetFace) and the new boundary faces are distributed and
    // refined with the rest of the mesh.  The interface faces get a fresh
    // boundary attribute (max + 1); the filter BC loop below then treats it
    // as solid (rho~ = 1) like any non-free-surface boundary.
    if (prob.cut_faces.Size() > 0)
    {
        stage("tagging design/void interface faces");
        const int cut_attr =
            (mesh.bdr_attributes.Size() ? mesh.bdr_attributes.Max() : 0) + 1;
        Array<Element *> new_be(prob.cut_faces.Size());
        for (int k = 0; k < prob.cut_faces.Size(); k++)
        {
            const int f = prob.cut_faces[k];
            MFEM_VERIFY(f >= 0 && f < mesh.GetNFaces(),
                        "cut_faces: face id " << f << " out of range [0,"
                        << mesh.GetNFaces() << ")");
            new_be[k] = mesh.GetFace(f)->Duplicate(&mesh);
            new_be[k]->SetAttribute(cut_attr);
        }
        mesh.AddBdrElements(new_be, prob.cut_faces);
        mesh.SetAttributes();
        if (myid == 0)
        {
            mfem::out << "interface: " << prob.cut_faces.Size()
                      << " faces tagged bdr attribute " << cut_attr
                      << " (filter rho~ = 1)\n";
        }
    }

    // 3. Refine the mesh and construct pmesh / the design subdomain.
    const int dim = mesh.Dimension();

    vector<unique_ptr<VectorCoefficient>> &ray_cf = prob.rays;
    const int n_dir = static_cast<int>(ray_cf.size());

    stage("partitioning mesh across " + std::to_string(num_procs) + " ranks");
    ParMesh pmesh(MPI_COMM_WORLD, mesh);
    mesh.Clear();

    if (ref_levels > 0)
    {
        stage("uniform refinement x " + std::to_string(ref_levels));
        for (int l = 0; l < ref_levels; l++) { pmesh.UniformRefinement(); }
    }

    // Solve on the full mesh (Option C).  Elements whose attribute is listed in
    // solid_attr are fixed solid (design rho pinned to 1); every other element
    // outside domain_attr is fixed void (rho pinned to 0).  Both carry their
    // pinned stiffness through SIMP and are excluded from the optimizer.

    // 3b. Mark the outflow boundary for each ray direction.
    Array<int> candidate_be;        // ray outflow candidate boundary elements
    Array<int> candidate_attr;
    for (int i = 0; i < pmesh.GetNBE(); i++)
    {
        const int el_attr = pmesh.GetBdrAttribute(i);

        if (ray_bdr_attrs.Find(el_attr) < 0) continue;
        candidate_be.Append(i);
        candidate_attr.Append(el_attr);
    }

    vector<unique_ptr<ParSubMesh>> outflow(n_dir);
    if (n_dir > 0) { stage("building outflow submeshes for " +
                           std::to_string(n_dir) + " ray direction(s)"); }

    for (int r = 0; r < n_dir; r++)
    {
        stage("  outflow submesh for ray " + std::to_string(r));
        // mark outflow (v . n > 0) on the candidate boundary elements
        const int outflow_attr = 100 + r;
        for (int k = 0; k < candidate_be.Size(); k++)
        {
            const int idx = candidate_be[k];
            ElementTransformation *trans = pmesh.GetBdrElementTransformation(idx);
            const IntegrationPoint &ip = Geometries.GetCenter(
                                            pmesh.GetBdrElementGeometry(idx));
            trans->SetIntPoint(&ip);

            Vector vv(dim);
            ray_cf[r]->Eval(vv, *trans, ip);

            Vector normal(dim);
            CalcOrtho(trans->Jacobian(), normal);

            bool is_outflow = (vv * normal > 0);
            if (is_outflow) { pmesh.SetBdrAttribute(idx, outflow_attr); }
        }
        pmesh.SetAttributes();

        Array<int> submesh_attr;  submesh_attr.Append(outflow_attr);
        outflow[r] = make_unique<ParSubMesh>(ParSubMesh::CreateFromBoundary(pmesh, submesh_attr));

        // restore original attributes before marking the next ray
        for (int k = 0; k < candidate_be.Size(); k++)
        {
            const int original_attr = candidate_attr[k];
            pmesh.SetBdrAttribute(candidate_be[k], original_attr);
        }
        pmesh.SetAttributes();
    }

    // 4. Define finite element collections and spaces.
    stage("building finite element spaces");
    // The design/control space is normally one order below the state and filter
    // spaces; clamp it at 1 so -o 1 (linear elasticity) still has a valid H1
    // control space (then control == filter order).
    const int control_order = std::max(order - 1, 1);
    H1_FECollection state_fec(order, dim);
    H1_FECollection filter_fec(order, dim);
    H1_FECollection control_fec(control_order, dim, BasisType::GaussLobatto);
    ParFiniteElementSpace state_fes(&pmesh, &state_fec, dim, Ordering::byNODES);
    ParFiniteElementSpace filter_fes(&pmesh, &filter_fec);
    ParFiniteElementSpace control_fes(&pmesh, &control_fec);

    // Printing all true dofs.
    HYPRE_BigInt state_size   = state_fes.GlobalTrueVSize();
    HYPRE_BigInt filter_size  = filter_fes.GlobalTrueVSize();
    HYPRE_BigInt control_size = control_fes.GlobalTrueVSize();
    const long long global_ne = pmesh.GetGlobalNE();     // collective
    long long global_nbe = pmesh.GetNBE();
    MPI_Allreduce(MPI_IN_PLACE, &global_nbe, 1, MPI_LONG_LONG, MPI_SUM,
                  MPI_COMM_WORLD);
    if (myid == 0)
    {
        cout << "\nmesh elements = " << global_ne
             << ",  boundary elements = " << global_nbe << "\n"
             << "state dofs = "   << state_size
             << ",  filter dofs = "  << filter_size
             << ",  design dofs = " << control_size << endl;
    }

    stage("marking passive (fixed solid / void) regions");
    // --- Option C: passive (non-design) regions ---------------------------
    // Elements outside domain_attr are either fixed solid (attr in solid_attr,
    // rho pinned to 1) or fixed void (rho pinned to 0).  Collect the control
    // true dofs that carry a pinned value, together with that value.  Marking
    // is done on gridfunctions and reduced through ParallelAssemble so it is
    // independent of the mesh partition; a dof shared with a designable element
    // stays free (design wins), and among the remainder solid wins over void.
    Array<int>  passive_ctrl_tdofs;
    Vector      passive_ctrl_vals;
    {
        ParGridFunction design_mark(&control_fes), solid_mark(&control_fes);
        design_mark = 0.0;
        solid_mark  = 0.0;
        Array<int> edofs;
        for (int e = 0; e < pmesh.GetNE(); e++)
        {
            const int attr = pmesh.GetAttribute(e);
            ParGridFunction *mk = nullptr;
            if      (domain_attr.Find(attr) >= 0) { mk = &design_mark; }
            else if (solid_attr.Find(attr)  >= 0) { mk = &solid_mark;  }
            else                                  { continue; }   // fixed void
            control_fes.GetElementDofs(e, edofs);
            for (int k = 0; k < edofs.Size(); k++)
            {
                const int d = edofs[k] < 0 ? -1 - edofs[k] : edofs[k];
                (*mk)(d) = 1.0;
            }
        }
        std::unique_ptr<HypreParVector> design_tv(design_mark.ParallelAssemble());
        std::unique_ptr<HypreParVector> solid_tv(solid_mark.ParallelAssemble());
        for (int t = 0; t < design_tv->Size(); t++)
        {
            if ((*design_tv)(t) < 0.5) { passive_ctrl_tdofs.Append(t); } // not designable
        }
        passive_ctrl_vals.SetSize(passive_ctrl_tdofs.Size());
        for (int i = 0; i < passive_ctrl_tdofs.Size(); i++)
        {
            passive_ctrl_vals(i) =
                (*solid_tv)(passive_ctrl_tdofs[i]) >= 0.5 ? 1.0 : 0.0;
        }
    }

    // The volume budget is measured against the designable region only, so the
    // fixed solid and void neither spend nor dilute it.  VolumeResidual is fed
    // an element-attribute marker so its integrals (and Vstar) cover only
    // domain_attr:  Vstar = vol_fraction * |design region|.
    Array<int> design_elem_marker(pmesh.attributes.Size() ? pmesh.attributes.Max()
                                                          : 0);
    design_elem_marker = 0;
    for (int a = 0; a < domain_attr.Size(); a++)
    {
        const int attr = domain_attr[a];
        if (attr >= 1 && attr <= design_elem_marker.Size())
        {
            design_elem_marker[attr - 1] = 1;
        }
    }

    real_t vol_design = 0.0, vol_full = 0.0;
    for (int e = 0; e < pmesh.GetNE(); e++)
    {
        const real_t ve = pmesh.GetElementVolume(e);
        vol_full += ve;
        if (domain_attr.Find(pmesh.GetAttribute(e)) >= 0) { vol_design += ve; }
    }
    MPI_Allreduce(MPI_IN_PLACE, &vol_design, 1, MPITypeMap<real_t>::mpi_type,
                  MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &vol_full, 1, MPITypeMap<real_t>::mpi_type,
                  MPI_SUM, MPI_COMM_WORLD);

    if (myid == 0)
    {
        cout << "design / full mesh volume = " << vol_design << " / " << vol_full
             << ",  passive control dofs = " << passive_ctrl_tdofs.Size() << endl;
    }

    // 5. Initialize all the grid functions and coefficients
    stage("initializing design field and coefficients");
    ParGridFunction rho(&control_fes);
    ParGridFunction rho_filter(&filter_fes);

    // Spatially varying initial design: a smooth field oscillating about
    // domain_init (in xy radius and in z), clamped to (0,1), plus a small random
    // perturbation to break symmetry.  The passive regions are overwritten with
    // their pinned values further below, so this only seeds the design domain.
    {
        const real_t two_pi = 8.0 * std::atan(real_t(1));
        FunctionCoefficient rho0_cf([=](const Vector &x) -> real_t
        {
            const real_t rr = std::sqrt(x[0]*x[0] + x[1]*x[1]);
            const real_t z  = (x.Size() == 3) ? x[2] : real_t(0);
            real_t s = domain_init
                     + 0.25 * domain_init * std::sin(two_pi * rr / real_t(30))
                     + 0.15 * domain_init * std::cos(two_pi * z  / real_t(20));
            return std::min(real_t(1) - real_t(1e-3),
                            std::max(real_t(1e-3), s));
        });
        rho.ProjectCoefficient(rho0_cf);

        ParGridFunction rho_noise(&control_fes);
        rho_noise.Randomize(seed);
        rho_noise -= 0.5;
        rho_noise *= 0.02;
        rho += rho_noise;
    }
    rho_filter = domain_init;

    GridFunctionCoefficient rho_cf(&rho);

    // 5b. Thickness design variables live on the outflow submeshes, one set per
    //     ray direction.  Solving for rho_a requires DG, so alpha also lives in
    //     DG.  The thickness measure is a line integral of density, so a low DG
    //     order is plenty and much cheaper (see -dgo).
    DG_FECollection dgfec(dg_order, dim, BasisType::GaussLobatto);
    ParFiniteElementSpace dgfes(&pmesh, &dgfec);

    vector<unique_ptr<DG_FECollection>> sub_dg_fec(n_dir);
    vector<unique_ptr<ParFiniteElementSpace>> sub_dg_fes(n_dir);
    vector<unique_ptr<ParGridFunction>> alpha(n_dir);

    for (int r = 0; r < n_dir; r++)
    {
        const int sub_dim = outflow[r]->Dimension();
        sub_dg_fec[r] = make_unique<DG_FECollection>(dg_order, sub_dim, BasisType::Positive);
        sub_dg_fes[r] = make_unique<ParFiniteElementSpace>(outflow[r].get(), sub_dg_fec[r].get());

        alpha[r] = make_unique<ParGridFunction>(sub_dg_fes[r].get());
        *alpha[r] = domain_init;
    }

    // Lame constants and SIMP material coefficients
    ConstantCoefficient one_cf(1.0);
    ConstantCoefficient E_cf(1.0), nu_cf(0.3);
    IsoElasticyLambdaCoeff lambda_cf(&E_cf, &nu_cf);
    IsoElasticySchearCoeff mu_cf(&E_cf, &nu_cf);

    // Heaviside projections of rho~:  eroded -> stiffness, dilated -> volume,
    // intermediate -> the design that is actually reported.
    HeavisideCoefficient rho_erod_cf(&rho_filter, beta, 1-eta);
    HeavisideGradCoefficient rho_erod_grad_cf(&rho_filter, beta, 1-eta);

    HeavisideCoefficient rho_dila_cf(&rho_filter, beta, eta);
    HeavisideGradCoefficient rho_dila_grad_cf(&rho_filter, beta, eta);

    HeavisideCoefficient rho_inter_cf(&rho_filter, beta, 0.5);

    // SIMP on the eroded projection: r(rho_e) = E_min + rho_e^p (E_max - E_min)
    SIMPCoefficient simp_cf(rho_erod_cf, E_min, E_max, exponent);                // r(rho_e)
    SIMPGradCoefficient simp_grad_cf(rho_erod_cf, E_min, E_max, exponent);       // r'(rho_e)

    // --- PLAIN SIMP ---  SIMP directly on rho~, no projection
    // SIMPCoefficient simp_cf(&rho_filter, E_min, E_max, exponent);             // r(rho~)
    // SIMPGradCoefficient simp_grad_cf(&rho_filter, E_min, E_max, exponent);    // r'(rho~)

    // 6. Construct the solvers.
    // 6a. Linear elasticity solvers, one per load case.
    ProductCoefficient lambda_simp_cf(simp_cf, lambda_cf);   // r(rho_e) * lambda0
    ProductCoefficient mu_simp_cf(simp_cf, mu_cf);           // r(rho_e) * mu0

    // setting up preconditioner
    LinearElasticitySolver::PreconditionerType elast_pc;
    if (pc_type == 0)
    {
        elast_pc = LinearElasticitySolver::PreconditionerType::Jacobi;
    }
    else if (pc_type == 1)
    {
        elast_pc = LinearElasticitySolver::PreconditionerType::LORDiagonalAMG;
    }
    else if (pc_type == 2)
    {
        elast_pc = LinearElasticitySolver::PreconditionerType::LORMonolithicAMG;
    }
    else
    {
        MFEM_ABORT("Unknown preconditioner! Elasticity preconditioner: "
                    "0 = Jacobi, 1 = LOR diagonal AMG, 2 = LOR monolithic AMG");
    }

    stage("configuring " + std::to_string(n_elast_solve) +
          " elasticity solver(s) (loads / BCs / preconditioner)");
    vector<unique_ptr<LinearElasticitySolver>> elast(n_elast_solve);
    // owns every body-force coefficient for the whole run (referenced by the
    // elasticity linear forms)
    vector<unique_ptr<VectorCoefficient>> vol_force;
    for (int i = 0; i < n_elast_solve; i++)
    {
        // apply boundary loads
        elast[i] = make_unique<LinearElasticitySolver>(state_fes);
        const LoadCase &lc = prob.cases[i];
        for (int j = 0; j < lc.load_attrs.Size(); j++)
        {
            Vector f(dim);
            f(0) = lc.fx[j];
            f(1) = lc.fy[j];
            if (dim == 3) { f(2) = lc.fz[j]; }

            auto load_cf = make_shared<VectorConstantCoefficient>(f);
            elast[i]->AddBoundaryLoad(lc.load_attrs[j], load_cf);
        }

        // apply body-force-density loads (one coefficient per attribute group)
        for (const LoadCase::VolumeLoad &vl : lc.vol_loads)
        {
            unique_ptr<VectorCoefficient> cf;
            if (vl.fn)
            {
                cf = make_unique<VectorFunctionCoefficient>(dim, vl.fn);
            }
            else
            {
                MFEM_VERIFY(vl.value.Size() == dim, "VolumeLoad.value size != dim");
                cf = make_unique<VectorConstantCoefficient>(vl.value);
            }
            for (int a = 0; a < vl.attrs.Size(); a++)
            {
                elast[i]->AddVolumeLoad(vl.attrs[a], *cf);
            }
            vol_force.push_back(std::move(cf));
        }

        // add fixed boundaries
        for (int j = 0; j < lc.clamp_attrs.Size(); j++)
        {
            elast[i]->AddBoundaryID(lc.clamp_attrs[j]);
        }

        // configurate the elast solver
        elast[i]->SetLambda(lambda_simp_cf);
        elast[i]->SetMu(mu_simp_cf);
        elast[i]->SetPreconditionerType(elast_pc);
        elast[i]->SetMonolithicLOROrdering(
            lor_by_vdim ? Ordering::byVDIM : Ordering::byNODES);
        elast[i]->SetPrintLevel(solver_print);
        elast[i]->SetRelTol(1e-7);
        elast[i]->SetAbsTol(1e-14);
        elast[i]->SetMaxIter(1000);
    }

    ParGridFunction u(&state_fes);
    StrainEnergyDensityCoefficient energy_cf(&lambda_cf, &mu_cf, &u);

    // dc/drho~ = - (r'(rho_e) * H_e'(rho~)) * psi0(u)
    ProductCoefficient drdrho_cf(simp_grad_cf, rho_erod_grad_cf);
    ProductCoefficient prod(energy_cf, drdrho_cf);
    ProductCoefficient dcdrho_tilde_cf(-1.0, prod);

    // --- PLAIN SIMP ---  dc/drho~ = -r'(rho~) * psi0
    // ProductCoefficient prod(energy_cf, simp_grad_cf);
    // ProductCoefficient dcdrho_tilde_cf(-1.0, prod);

    // 6b. Min length scale filter solver (diffusion-mass PDE filter).
    stage("setting up PDE filter (r_f = " + std::to_string(r_f) + ")");
    PDEFilter filter(control_fes, filter_fes);
    filter.SetFilterRadius(r_f);
    DiffusionMassSolver &filter_solver = filter.GetSolver();
    filter_solver.SetPrintLevel(solver_print);   // before Assemble (marks dirty)
    // Boundary attributes on the full mesh: the free surfaces in outer_bdr_attrs
    // get rho~ = 0; attributes in neumann_bdr_attrs get a natural (zero-flux) BC
    // and are left out of the Dirichlet set entirely; every other boundary
    // attribute is solid (rho~ = 1).  That covers the design/void interface
    // faces tagged above and any exterior boundary the mesh setup did not list.
    // (pmesh.bdr_attributes is globally reduced, so every attribute is present
    // on every rank and this loop stays collective even where no such face is
    // owned.)
    Array<int> filter_bdr_marker(pmesh.bdr_attributes.Size()
                                 ? pmesh.bdr_attributes.Max() : 0);
    filter_bdr_marker = 0;
    for (int i = 0; i < pmesh.bdr_attributes.Size(); i++)
    {
        const int a = pmesh.bdr_attributes[i];
        if (neumann_bdr_attrs.Find(a) >= 0) { continue; }       // natural BC
        const bool is_outer = (outer_bdr_attrs.Find(a) >= 0);
        filter_solver.Boundary().Add(a, is_outer ? 0.0 : 1.0);  // free surface / solid
        filter_bdr_marker[a - 1] = 1;
    }
    stage("assembling PDE filter operator + preconditioner");
    filter.Assemble();

    // lifting for dirichlet bc
    Array<int> filter_ess_tdofs;
    filter_fes.GetEssentialTrueDofs(filter_bdr_marker, filter_ess_tdofs);

    Vector rho_filter_lift_tv;
    {
        stage("filter Dirichlet-lift solve");
        ParGridFunction lift(&filter_fes);
        filter_solver.Solve(lift);
        lift.GetTrueDofs(rho_filter_lift_tv);
        rho_filter_lift_tv.SetSubVector(filter_ess_tdofs, real_t(0));
    }

    // 6c. Advection solvers for the thickness measure, one per ray direction,
    //     with the pseudo-transient time step set from the CFL condition.
    //
    // DG mass inverse, applied 3x per RK step for thousands of pseudo-time steps.
    // The DG mass is block-diagonal (no inter-element coupling), so with -minv-fa
    // we assemble the exact block inverse once (InverseIntegrator: elmat = M_e,
    // then M_e^{-1}) and each apply is a single sparse matvec -- much cheaper
    // than the matrix-free per-element CG, especially at low order.  -minv-mf
    // keeps the DGMassInverse (better at high order / on GPU); its per-element
    // CG default tol (1e-12) far over-solves relative to -atol, so loosen it.
    // (Minv only affects the pseudo-time path, not the converged rho_a.)
    // Declared before `advect` so they outlive the solvers that borrow them.
    if (n_dir > 0) { stage("setting up advection (ray) solvers + DG mass inverse"); }
    std::unique_ptr<HypreParMatrix> minv_fa_mat;
    std::unique_ptr<DGMassInverse>  minv_mf;
    if (minv_fa)
    {
        if (n_dir > 0) { stage("assembling exact block-diagonal DG mass inverse"); }
        ParBilinearForm minv_form(&dgfes);
        minv_form.AddDomainIntegrator(new InverseIntegrator(new MassIntegrator));
        minv_form.Assemble();
        minv_form.Finalize();
        minv_fa_mat.reset(minv_form.ParallelAssemble());
    }
    else
    {
        minv_mf = make_unique<DGMassInverse>(dgfes);
        minv_mf->SetRelTol(minv_tol);
        minv_mf->SetAbsTol(minv_tol * real_t(1e-2));
        minv_mf->SetMaxIter(30);
    }

    vector<unique_ptr<MaterialThicknessSolver>> advect(n_dir);

    real_t hmin = infinity();
    for (int i = 0; i < pmesh.GetNE(); i++)
    {
        hmin = min(pmesh.GetElementSize(i, 1), hmin);
    }
    MPI_Allreduce(MPI_IN_PLACE, &hmin, 1,  MPITypeMap<real_t>::mpi_type, MPI_MIN,
                    pmesh.GetComm());
    const real_t dt = adv_cfl * hmin / (2 * dg_order + 1);   // DG-CFL: degree dg_order
    if (myid == 0 && n_dir > 0)
    {
        mfem::out << "advection: DG order " << dg_order << ", K "
                  << (adv_pa ? "partial" : "full") << " assembly, Minv "
                  << (minv_fa ? "exact block-diagonal" : "matrix-free CG")
                  << "; pseudo-transient dt = " << dt << ", t_final = " << adv_tfinal
                  << " (<= " << (int)std::ceil(adv_tfinal / dt) << " steps), tol = "
                  << adv_tol << defaultfloat << setprecision(6) << std::endl;
    }

    for (int r = 0; r < n_dir; r++)
    {
        advect[r] = make_unique<MaterialThicknessSolver>(filter_fes, dgfes, *ray_cf[r],
                                                         adv_pa);
        if (minv_fa) { advect[r]->GetSolver().SetMinv(*minv_fa_mat); }
        else         { advect[r]->SetMinv(*minv_mf); }
        advect[r]->GetSolver().SetTimeStep(dt);          // pseudo-transient time step
        advect[r]->GetSolver().SetTerminalTime(adv_tfinal); // absolute stopping condition
        advect[r]->GetSolver().SetTol(adv_tol);          // steady-state rate tolerance
        advect[r]->GetSolver().SetPrintLevel(solver_print);
    }

    // 7. Construct the quantity of interest objects
    stage("constructing quantity-of-interest objects (compliance / volume / thickness)");
    Compliance comp(MPI_COMM_WORLD, &filter_fes, simp_cf, energy_cf);

    // volume of the dilated field over domain_attr only, measured against
    // V* = vol_fraction |design region|
    VolumeResidual vol_qoi(MPI_COMM_WORLD, &filter_fes, &rho_dila_cf,
                           &rho_dila_grad_cf, vol_fraction, &design_elem_marker);

    // Max-thickness constraint residual per ray:  1/2 ∫_Gamma_out,r (rho_a − α_r)²
    vector<unique_ptr<AdvectThicknessResidual>> adv_res(n_dir);
    for (int r = 0; r < n_dir; r++)
    {
        adv_res[r] = make_unique<AdvectThicknessResidual>(*outflow[r],
                                                          advect[r]->GetRhoA(),
                                                          *alpha[r]);
    }

    // --- PLAIN SIMP ---  linear volume constraint on the filtered density,
    //                     g(rho) = (1, rho~)/Vstar - 1.  The Dirichlet filter BCs
    //                     make the filter non-volume-preserving, so (1,rho) is
    //                     not equivalent.
    // ParLinearForm vol_form(&filter_fes);
    // vol_form.AddDomainIntegrator(new DomainLFIntegrator(one_cf));
    // vol_form.Assemble();
    // std::unique_ptr<HypreParVector> vol_w(vol_form.ParallelAssemble());
    //
    // real_t domain_volume;
    // real_t loc = vol_w->Sum();
    // MPI_Allreduce(&loc, &domain_volume, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM, MPI_COMM_WORLD);
    // const real_t Vstar = vol_fraction * domain_volume;
    //
    // // d(1,rho~)/drho = L^T w is constant: the filter operator never changes,
    // // and the Dirichlet lifting contributes a constant offset with zero derivative.
    // Vector dvol_drho(control_fes.GetTrueVSize());
    // filter.MultTranspose(*vol_w, dvol_drho);

    // 8. MMA optimizer and its per-iteration work vectors.
    // stacked design  x = [ rho ; alpha_0 ; alpha_1 ; ... ; alpha_{n_dir-1} ]
    const int n  = control_fes.GetTrueVSize();      // local rho design variables
    const int nf = filter_fes.GetTrueVSize();
    vector<int> m(n_dir);
    for (int r = 0; r < n_dir; r++) { m[r] = sub_dg_fes[r]->GetTrueVSize(); }

    Array<int> toffsets(n_dir + 2);
    toffsets[0] = 0;
    toffsets[1] = n;
    for (int r = 0; r < n_dir; r++) { toffsets[2 + r] = m[r]; }
    toffsets.PartialSum();

    const int num_con = 1 + n_dir;                  // constraints: volume + one thickness per ray

    // 8a. stacked design  x = [ rho ; alpha_0 ; ... ]
    Vector rho_tv(n), rho_old(n);
    vector<Vector> alpha_tv(n_dir);

    // initialize for normalization
    real_t init_comp = 1.0;

    rho.GetTrueDofs(rho_tv);
    for (int r = 0; r < n_dir; r++) { alpha[r]->GetTrueDofs(alpha_tv[r]); }

    // Initialize checkpoint system
    Checkpoint checkpoint("checkpoints", MPI_COMM_WORLD);
    int start_iteration = 1;
    const int cp_interval = 5;

    MFEM_VERIFY(cp >= 0 && cp <= 2, "-cp must be 0, 1 or 2.");
    MFEM_VERIFY(restart >= 0 && restart <= 2, "-restart must be 0, 1 or 2.");

    // restart = 1: load rho only
    if (restart == 1)
    {
        MFEM_VERIFY(checkpoint.RhoExists(),
                    "Restart from rho requested but no rho file found.");
        MFEM_VERIFY(checkpoint.LoadRho(rho_tv), "Failed to load rho.");

        if (myid == 0)
        {
            mfem::out << "\nWarm start: rho loaded, running from iteration 1\n";
        }
    }
    // restart = 2: full state, resume where the previous run stopped
    else if (restart == 2)
    {
        MFEM_VERIFY(checkpoint.Exists(),
                    "Restart requested but no checkpoint found.");
        MFEM_VERIFY(checkpoint.ValidateCompatibility(ref_levels, order, n_dir),
                    "Checkpoint incompatible with current run parameters.");
        MFEM_VERIFY(checkpoint.Load(rho_tv, alpha_tv),
                    "Failed to load checkpoint data.");

        start_iteration = checkpoint.GetIteration() + 1;   // `it` is 1-indexed
        epsilon = checkpoint.GetEpsilon();
        init_comp = checkpoint.GetInitComp();

        if (myid == 0)
        {
            mfem::out << "\nRestarting from iteration " << start_iteration
                      << " with epsilon = " << epsilon << "\n";
        }
    }

    // Option C: pin the passive regions (fixed solid -> 1, fixed void -> 0)
    // before they enter the optimizer (also overrides a restart file).
    rho_tv.SetSubVector(passive_ctrl_tdofs, passive_ctrl_vals);

    rho.SetFromTrueDofs(rho_tv);
    for (int r = 0; r < n_dir; r++) { alpha[r]->SetFromTrueDofs(alpha_tv[r]); }

    BlockVector tx_local(toffsets);
    tx_local.GetBlock(0) = rho_tv;
    for (int r = 0; r < n_dir; r++) { tx_local.GetBlock(1 + r) = alpha_tv[r]; }

    Vector a(num_con), c(num_con), d(num_con);
    a = 0.0; c = 1000.0; d = 0.0;
    stage("initializing MMA optimizer (" + std::to_string(num_con) + " constraints)");
    mfem_mma::MMAOptimizerParallel mma(MPI_COMM_WORLD, toffsets.Last(), num_con, tx_local, a, c, d);

    // Restore MMA state if restarting
    if (restart == 2 && start_iteration > 1 && checkpoint.GetXOld1().Size() > 0)
    {
        mma.RestoreState(checkpoint.GetXOld1(), checkpoint.GetXOld2(),
                         checkpoint.GetLowerAsymptotes(), checkpoint.GetUpperAsymptotes());
        if (myid == 0)
        {
            mfem::out << "MMA state restored from checkpoint.\n";
        }
    }

    // 8b. objective initialization
    BlockVector df0dx(toffsets);                    // objective gradient  df0/dx = [ dc/drho ; 0 ; ... ]
    Vector dcdrho(n);                               // compliance gradient  dc/drho

    // 8c. local constraints
    Vector fival(num_con);
    vector<Vector> dfidx(num_con);

    BlockVector dvol(toffsets);                     // volume gradient  [ dg/drho ; 0 ; ... ]
    dvol = 0.0; dfidx[0] = dvol;
    Vector dvol_tilde(nf);                          // dV_d/drho~

    // one full-size gradient BlockVector per ray-thickness constraint; only
    // block(0) (drho) and block(1+r) (dalpha_r) are ever nonzero.
    vector<BlockVector> dthick(n_dir, BlockVector(toffsets));

    // --- PLAIN SIMP ---  the linear volume gradient is constant:  [ L^T w/Vstar ; 0 ; ... ]
    // dvol.GetBlock(0) = dvol_drho;
    // dvol.GetBlock(0) /= Vstar;
    // dfidx[0] = dvol;

    // 8d. mma upper and lower bounds
    BlockVector tx_min(toffsets), tx_max(toffsets);

    // 9. Visualizations
    // 9a. GLVis
    char vishost[] = "localhost";  int visport = 19916;
    socketstream sout;

    if (visualization) {
        stage("opening GLVis connection to " + std::string(vishost) + ":"
              + std::to_string(visport));
        sout.open(vishost, visport);
        sout.precision(8);

        sout << "parallel " << num_procs << " " << myid << "\n"
            << "solution\n" << pmesh << rho_filter
            << "window_title 'Projected density'\n"
            << "window_geometry 0 0 800 600\n"
            << "colorbar_numberformat '%.2f'\n"
            << "keys c\n" << flush;
    }

    // 9b. Paraview
    ParGridFunction phys_density(&filter_fes);
    std::ostringstream run_tag;
    run_tag << "3dbeam_amax" << alpha_max << "_vf" << vol_fraction;
    ParaViewDataCollection paraview_dc(run_tag.str(), &pmesh);

    if (paraview) {
        paraview_dc.SetPrefixPath("ParaView");
        paraview_dc.SetLevelsOfDetail(order);
        paraview_dc.SetDataFormat(VTKFormat::BINARY);
        paraview_dc.SetHighOrderOutput(true);
        paraview_dc.RegisterField("density", &phys_density);
        paraview_dc.RegisterField("rho_filter", &rho_filter);
    }

    // 9c. Initialization block runtime.
    double init_block_time = MPI_Wtime() - init_time;
    if (myid == 0)
    {
        mfem::out << "\nInitialization block runtime: " << fixed << setprecision(4)
                   << init_block_time << " s\n"
                   << defaultfloat << setprecision(6);   // restore stream format
    }

    // 9d. CSV convergence log (rank 0 only).
    std::ofstream csv;
    if (myid == 0)
    {
        csv.open("convergence.csv");
        csv << "it,c,volume,max_rho_a,max_alpha,fival_max,eps,beta,iterErr,iter_time\n";
    }

    stage("setup complete");

    // 9e. Pre-optimization evaluation of every physics on the initial design:
    //     one PDE-filter solve, one linear-elasticity solve per load case, and
    //     one advection (ray) solve per direction.  Each solver is timed and the
    //     timings are reported.  With -pv the fields are also written to a
    //     ParaView archive ("<run_tag>_init").  Runs whenever an archive is
    //     wanted or the optimization is skipped (-no-opt).
    if (paraview || !optimize)
    {
        stage("=== initial evaluation on the starting design ===");
        rho.GetTrueDofs(rho_tv);
        rho_tv.SetSubVector(passive_ctrl_tdofs, passive_ctrl_vals);

        double t_filter = 0.0;
        std::vector<double> t_elast(n_elast_solve, 0.0);   // full Solve() wall time
        std::vector<double> s_elast(n_elast_solve, 0.0);   // of which: setup (assembly + preconditioner)
        std::vector<double> t_advect(n_dir, 0.0);
        int it_filter = 0;
        std::vector<int> it_elast(n_elast_solve, 0);
        std::vector<int> it_advect(n_dir, 0);

        // (a) PDE filter
        stage("initial eval: PDE filter solve");
        Vector rho_filter_tv(nf);
        double t0 = MPI_Wtime();
        filter.Mult(rho_tv, rho_filter_tv);
        t_filter = MPI_Wtime() - t0;
        it_filter = filter.GetSolver().GetNumIterations();
        rho_filter_tv += rho_filter_lift_tv;
        rho_filter.SetFromTrueDofs(rho_filter_tv);
        phys_density.ProjectCoefficient(rho_inter_cf);

        ParGridFunction rho_dila_gf(&filter_fes);
        rho_dila_gf.ProjectCoefficient(rho_dila_cf);
        Vector rho_dila_tv(nf);
        rho_dila_gf.GetTrueDofs(rho_dila_tv);

        // raw design lifted to the filter space so every archived field is the
        // same order
        ParGridFunction rho_vis(&filter_fes);
        rho_vis.ProjectCoefficient(rho_cf);

        // one displacement field per load case, one accumulated rho_a per ray,
        // one ray direction field per ray; declared before init_dc so they
        // outlive it
        std::vector<std::unique_ptr<ParGridFunction>> u_init(n_elast_solve);
        std::vector<std::unique_ptr<ParGridFunction>> rho_a_init(n_dir);
        std::vector<std::unique_ptr<ParGridFunction>> ray_init(n_dir);

        ParaViewDataCollection init_dc(run_tag.str() + "_init", &pmesh);
        init_dc.SetPrefixPath("ParaView");
        init_dc.SetLevelsOfDetail(order);
        init_dc.SetDataFormat(VTKFormat::BINARY);
        init_dc.SetHighOrderOutput(true);
        init_dc.RegisterField("rho", &rho_vis);
        init_dc.RegisterField("rho_filter", &rho_filter);
        init_dc.RegisterField("phys_density", &phys_density);

        // (b) linear elasticity
        for (int i = 0; i < n_elast_solve; i++)
        {
            stage("initial eval: elasticity solve, load case " + std::to_string(i)
                  + "/" + std::to_string(n_elast_solve - 1));
            elast[i]->SetNeedsAssembly();
            t0 = MPI_Wtime();
            elast[i]->Solve(u);
            t_elast[i] = MPI_Wtime() - t0;
            s_elast[i] = elast[i]->GetAssemblyTime()
                       + elast[i]->GetPrecAssemblyTime();
            it_elast[i] = elast[i]->GetNumIterations();
            u_init[i] = make_unique<ParGridFunction>(&state_fes);
            *u_init[i] = u;
            init_dc.RegisterField("u_" + std::to_string(i), u_init[i].get());
        }

        // (c) advection thickness measure (+ the ray direction field)
        for (int r = 0; r < n_dir; r++)
        {
            stage("initial eval: advection solve, ray " + std::to_string(r));

            ray_init[r] = make_unique<ParGridFunction>(&state_fes);   // H1 vector, vdim = dim
            ray_init[r]->ProjectCoefficient(*ray_cf[r]);
            init_dc.RegisterField("ray_" + std::to_string(r), ray_init[r].get());

            advect[r]->SetRhs(rho_dila_tv);
            t0 = MPI_Wtime();
            advect[r]->FSolve();
            t_advect[r] = MPI_Wtime() - t0;
            it_advect[r] = advect[r]->GetSolver().GetIterCount();
            rho_a_init[r] = make_unique<ParGridFunction>(&dgfes);
            *rho_a_init[r] = advect[r]->GetRhoA();
            init_dc.RegisterField("rho_a_" + std::to_string(r), rho_a_init[r].get());
        }

        if (paraview)
        {
            stage("initial eval: writing ParaView archive");
            init_dc.SetCycle(0);
            init_dc.SetTime(0.0);
            init_dc.Save();
        }

        if (myid == 0)
        {
            double t_total  = t_filter;
            double s_total  = 0.0;
            int    it_total = it_filter;
            auto row = [](const std::string &name, double t, int nit,
                          double setup = -1.0)
            {
                mfem::out << left << setw(30) << name
                          << right << fixed << setprecision(4) << setw(10) << t
                          << " s" << setw(8) << nit << " it";
                if (setup >= 0.0)
                {
                    mfem::out << "   setup " << setprecision(4) << setup << " s";
                }
                mfem::out << '\n';
            };
            mfem::out << "\n--- initial evaluation: solver time / iterations ---\n";
            row("  PDE filter", t_filter, it_filter);
            for (int i = 0; i < n_elast_solve; i++)
            {
                t_total  += t_elast[i];
                s_total  += s_elast[i];
                it_total += it_elast[i];
                row("  elasticity load case " + std::to_string(i),
                    t_elast[i], it_elast[i], s_elast[i]);
            }
            for (int r = 0; r < n_dir; r++)
            {
                t_total  += t_advect[r];
                it_total += it_advect[r];
                row("  advection ray " + std::to_string(r),
                    t_advect[r], it_advect[r]);
            }
            row("  total", t_total, it_total);
            if (n_elast_solve > 0)
            {
                mfem::out << left << setw(30) << "  elasticity setup total"
                          << right << fixed << setprecision(4) << setw(10)
                          << s_total << " s   (assembly + preconditioner, "
                          << "already included in the elasticity rows above)\n";
            }
            if (paraview)
            {
                mfem::out << "wrote initial-state ParaView archive: ParaView/"
                          << run_tag.str() << "_init\n";
            }
            mfem::out << defaultfloat << setprecision(6)   // restore stream format
                      << std::flush;
        }
    }

    if (!optimize)
    {
        if (myid == 0)
        {
            if (csv.is_open()) { csv.close(); }
            mfem::out << "\n-no-opt: skipping the optimization loop" << std::endl;
        }
        return 0;
    }

    // 10. Optimization loop.
    real_t iterationError = 1.0;

    // Track next iteration for epsilon decay and beta doubling
    int next_epsilon_decay = init_it;
    int next_beta_double = init_it + beta_steps;

    // fast-forward the schedule counters 
    if (restart == 2)
    {
        if (decay_int > 0)
        {
            while (next_epsilon_decay < start_iteration) { next_epsilon_decay += decay_int; }
        }
        if (beta_steps > 0)
        {
            while (next_beta_double < start_iteration && beta < beta_max)
            {
                beta *= 2;
                next_beta_double += beta_steps;
            }
        }
        rho_erod_cf.SetBeta(beta);  rho_erod_grad_cf.SetBeta(beta);
        rho_dila_cf.SetBeta(beta);  rho_dila_grad_cf.SetBeta(beta);
        rho_inter_cf.SetBeta(beta);
    }

    double opt_start_time = MPI_Wtime();

    stage("=== entering optimization loop (start iteration "
          + std::to_string(start_iteration) + ") ===");

    int it = start_iteration;
    for (; (it <= init_it) || (it <= max_it && iterationError > tol); it++)
    {
        double iter_start_time = MPI_Wtime() - opt_start_time;
        stage("[iteration " + std::to_string(it) + "] solving ...");
        // finer sub-stage markers only on the first pass, to localize a hang
        // without flooding every iteration:
        const bool trace = (it == start_iteration);

        // (1) forward filter:  (r_f^2 K + M) ρ~ = M_fc ρ  (+ Dirichlet lifting)
        if (trace) { stage("  it 1: forward filter"); }
        rho.GetTrueDofs(rho_tv);
        rho_tv.SetSubVector(passive_ctrl_tdofs, passive_ctrl_vals); // Option C: pinned regions
        Vector rho_filter_tv(nf);
        filter.Mult(rho_tv, rho_filter_tv);
        rho_filter_tv += rho_filter_lift_tv;
        rho_filter.SetFromTrueDofs(rho_filter_tv);

        // construct dialated desgin coefficients
        ParGridFunction rho_dila_gf(&filter_fes);
        rho_dila_gf.ProjectCoefficient(rho_dila_cf);
        Vector rho_dila_tv(nf);
        rho_dila_gf.GetTrueDofs(rho_dila_tv);

        ParGridFunction rho_dila_grad_gf(&filter_fes);
        rho_dila_grad_gf.ProjectCoefficient(rho_dila_grad_cf);
        Vector rho_dila_grad_tv(nf);
        rho_dila_grad_gf.GetTrueDofs(rho_dila_grad_tv);

        // (2) state solves:  K(ρ~) u = f   (self-adjoint compliance), averaged
        //     over the load cases, together with the adjoint filter rhs
        real_t compliance = 0.0;
        Vector adj_rhs_tv(nf);
        adj_rhs_tv = 0.0;
        double elast_runtime = MPI_Wtime();
        for (int i = 0; i < n_elast_solve; i++)
        {
            if (trace) { stage("  it 1: elasticity solve " + std::to_string(i)); }
            elast[i]->SetNeedsAssembly();   // reset assembly flag
            elast[i]->Solve(u);
            compliance += comp.Eval();

            ParLinearForm adj_rhs(&filter_fes);
            adj_rhs.AddDomainIntegrator(new DomainLFIntegrator(dcdrho_tilde_cf));
            adj_rhs.Assemble();
            std::unique_ptr<HypreParVector> adj_rhs_e_tv(adj_rhs.ParallelAssemble());
            adj_rhs_tv += *adj_rhs_e_tv;
        }
        elast_runtime = MPI_Wtime() - elast_runtime;
        compliance  /= n_elast_solve;
        adj_rhs_tv  /= n_elast_solve;

        // (3) adjoint filter + objective gradient:
        //     w~  = (r_f^2 K + M)^{-1} ∫ (-r'(ρ~) psi_0) φ_i
        //     dc/drho = M_fc^T w~
        if (trace) { stage("  it 1: adjoint filter + volume QoI"); }
        filter.MultTranspose(adj_rhs_tv, dcdrho);
        df0dx.GetBlock(0) = dcdrho;                     // objective gradient
        for (int r = 0; r < n_dir; r++) { df0dx.GetBlock(1 + r) = 0.0; }

        // (4) volume constraint and gradient on the dilated field:
        //       g        = V_d / V* - 1
        //       dg/drho~ = (H_d'(ρ~), φ_i) / V*
        fival(0) = vol_qoi.Eval() - 1.0;                // update constraint value
        real_t vol = (fival(0) + 1.0) * vol_fraction;   // current volume fraction
        vol_qoi.GetGrad(dvol_tilde);
        filter.MultTranspose(dvol_tilde, dvol.GetBlock(0));
        dfidx[0] = dvol;                                // update constraint gradient

        // --- PLAIN SIMP ---  linear volume constraint on rho~ (gradient is
        //                     constant, set once outside the loop)
        // const real_t vol_int = InnerProduct(MPI_COMM_WORLD, *vol_w, rho_filter_tv);
        // real_t vol = vol_int / domain_volume;
        // fival(0) = vol_int / Vstar - 1.0;

        // (5) advect rho~ along each ray to get the thickness measure rho_a, then the
        //     max-thickness constraint and gradient per direction:
        //       1/2 ∫(rho_a−α_r)² − ε ≤ 0
        //       dR/dalpha_r = (α_r − rho_a) on Gamma_out,r
        //       dR/drho     = M_fc^T N^T (rho_a − α_r)  via the adjoint advection solve
        real_t fi_thick  = -infinity();
        real_t max_rho_a = -infinity();
        real_t max_alpha = -infinity();
        double adv_runtime = MPI_Wtime();
        for (int r = 0; r < n_dir; r++)
        {
            if (trace) { stage("  it 1: advection fwd+adj, ray " + std::to_string(r)); }
            // forward
            advect[r]->SetRhs(rho_dila_tv);
            advect[r]->FSolve();
            const real_t thickres = adv_res[r]->Eval();

            // record max value of rho_a and alpha
            real_t local_max = advect[r]->GetRhoA().Max();
            real_t global_max = local_max;
            MPI_Allreduce(&local_max, &global_max, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX,
                        advect[r]->GetRhoA().ParFESpace()->GetComm());

            max_rho_a = std::max(max_rho_a, global_max);

            local_max = alpha[r]->Max();
            MPI_Allreduce(&local_max, &global_max, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX,
                        alpha[r]->ParFESpace()->GetComm());

            max_alpha = std::max(max_alpha, global_max);

            // evaluate adjoint gradient
            dthick[r] = 0.0;

            Vector dGdrhoa;
            adv_res[r]->GetGrad(dGdrhoa, dthick[r].GetBlock(1 + r));

            // transfer dGdrhoa back to the full-domain dgfes
            ParGridFunction g_sub(sub_dg_fes[r].get());  g_sub.SetFromTrueDofs(dGdrhoa);
            ParGridFunction g_full(&dgfes);              g_full = 0.0;
            outflow[r]->Transfer(g_sub, g_full);
            Vector rhs_full;  g_full.GetTrueDofs(rhs_full);

            // chain rule adjoint solve: dG/drho = M_fc^T N^T g
            advect[r]->SetAdjointRhs(rhs_full);
            advect[r]->ASolve();

            Vector dGdrho_tilde(advect[r]->GetSensitivity());
            dGdrho_tilde *= rho_dila_grad_tv;
            filter.MultTranspose(dGdrho_tilde, dthick[r].GetBlock(0));

            fival(1 + r) = thickres - epsilon;     // update constraint value
            // dthick[r] /= epsilon;
            dfidx[1 + r] = dthick[r];                    // update constraint gradient

            fi_thick = std::max(fi_thick, fival(1 + r));
        }
        adv_runtime = MPI_Wtime() - adv_runtime;

        // (6) box constraints:  rho ∈ [0,1],  α_r ∈ [alpha_min, alpha_max]  (move limits)
        for (int r = 0; r < n_dir; r++) { alpha[r]->GetTrueDofs(alpha_tv[r]); }
        for (int i = 0; i < n; i++)
        {
            tx_min[i] = std::max(real_t(0), rho_tv[i] - move);
            tx_max[i] = std::min(real_t(1), rho_tv[i] + move);
        }
        // Option C: freeze the passive regions (xmin = xmax = pinned value).
        for (int k = 0; k < passive_ctrl_tdofs.Size(); k++)
        {
            tx_min[passive_ctrl_tdofs[k]] = passive_ctrl_vals(k);
            tx_max[passive_ctrl_tdofs[k]] = passive_ctrl_vals(k);
        }
        for (int r = 0; r < n_dir; r++)
        {
            for (int i = 0; i < m[r]; i++)
            {
                tx_min[toffsets[1 + r] + i] = std::max(alpha_min, alpha_tv[r][i] - move);
                tx_max[toffsets[1 + r] + i] = std::min(alpha_max, alpha_tv[r][i] + move);
            }
        }

        // (7) MMA update on the stacked design  x = [ ρ ; α_0 ; ... ; α_{n_dir-1} ]
        tx_local.GetBlock(0) = rho_tv;
        for (int r = 0; r < n_dir; r++) { tx_local.GetBlock(1 + r) = alpha_tv[r]; }
        rho_old = rho_tv;

        // Normalize compliance and gradient by initial value
        if (it == 1) { init_comp = compliance; }
        compliance /= init_comp;
        df0dx /= init_comp;

        if (trace) { stage("  it 1: MMA update"); }
        mma.Update(tx_local, df0dx, compliance, fival, dfidx.data(), tx_min, tx_max);
        rho.SetFromTrueDofs(tx_local.GetBlock(0));
        for (int r = 0; r < n_dir; r++) { alpha[r]->SetFromTrueDofs(tx_local.GetBlock(1 + r)); }

        // measure iteration error
        ParGridFunction rho_old_gf(&control_fes);
        rho_old_gf.SetFromTrueDofs(rho_old);
        iterationError = rho_old_gf.ComputeL1Error(rho_cf);

        double iter_end_time = MPI_Wtime() - opt_start_time;
        double iter_runtime  = iter_end_time - iter_start_time;
        double elapsed_time  = MPI_Wtime() - init_time;   // includes the setup block

        // (8) reporting
        if (myid == 0)
        {
            const int w = 12;               // column width
            mfem::out << "\nIteration " << it << '\n' << string(8*w, '=') << "\n\n" << left
                    << setw(w) << "c"
                    << setw(w) << "volume"
                    << setw(w) << "max_rho_a"
                    << setw(w) << "max_alpha"
                    << setw(w) << "fival_max"
                    << setw(w) << "eps"
                    << setw(w) << "beta"
                    << setw(w) << "iterErr" << '\n'
                    << string(8*w, '-') << '\n'
                    << fixed      << setprecision(6) << setw(w) << compliance
                    <<               setprecision(4) << setw(w) << vol
                    <<               setprecision(4) << setw(w) << max_rho_a
                    <<               setprecision(4) << setw(w) << max_alpha
                    << scientific << setprecision(2) << setw(w) << fi_thick
                    <<               setprecision(2) << setw(w) << epsilon
                    << fixed      << setprecision(0) << setw(w) << beta
                    << scientific << setprecision(2) << setw(w) << iterationError << "\n\n";

            // runtime outputs
            const int lw = 18;              // label width
            mfem::out << fixed << setprecision(2) << left
                    << setw(lw) << "elast solve"     << right << setw(8) << elast_runtime << " s\n" << left
                    << setw(lw) << "advection solve" << right << setw(8) << adv_runtime   << " s\n" << left
                    << setw(lw) << "iteration"       << right << setw(8) << iter_runtime  << " s\n" << left
                    << setw(lw) << "total elapsed"   << right << setw(8) << elapsed_time  << " s   "
                    << setprecision(0) << floor(elapsed_time/3600) << "h "
                    << fmod(floor(elapsed_time/60), 60) << "m" << endl;

            csv << it << ','
                << scientific << setprecision(8) << compliance << ','
                << vol << ','
                << max_rho_a << ','
                << max_alpha << ','
                << fi_thick << ','
                << epsilon << ','
                << fixed << setprecision(0) << beta << ','
                << scientific << setprecision(8) << iterationError << ','
                << fixed << setprecision(4) << iter_runtime << '\n';
            csv.flush();
        }

        // (9) tighten the max-thickness tolerance and update beta
        // Epsilon decay: starts at init_it, then every decay_int iterations
        if (it == next_epsilon_decay)
        {
            epsilon = std::max(epsilon * decay, eps_floor);
            next_epsilon_decay += decay_int;
        }

        // Beta doubling: starts at init_it + beta_steps, then every beta_steps iterations
        if (it == next_beta_double && beta < beta_max)
        {
            beta *= 2;
            rho_erod_cf.SetBeta(beta);  rho_erod_grad_cf.SetBeta(beta);
            rho_dila_cf.SetBeta(beta);  rho_dila_grad_cf.SetBeta(beta);
            rho_inter_cf.SetBeta(beta);
            next_beta_double += beta_steps;
        }

        // Checkpoint every cp_interval iterations
        if (cp > 0 && it % cp_interval == 0)
        {
            rho.GetTrueDofs(rho_tv);

            if (cp == 1)
            {
                // rho only
                checkpoint.SaveRho(rho_tv, it);
            }
            else
            {
                for (int r = 0; r < n_dir; r++) { alpha[r]->GetTrueDofs(alpha_tv[r]); }

                // Save with MMA state for proper restart
                checkpoint.Save(rho_tv, alpha_tv,
                                mma.GetXOld1(), mma.GetXOld2(),
                                mma.GetLowerAsymptotes(), mma.GetUpperAsymptotes(),
                                it, n_dir, ref_levels, order, epsilon, init_comp);
            }
        }

        // physical density for both GLVis and the ParaView archive
        phys_density.ProjectCoefficient(rho_inter_cf);
        // --- PLAIN SIMP ---
        // phys_density.ProjectCoefficient(simp_cf);

        if (visualization)
        {
            sout << "parallel " << num_procs << " " << myid << "\n"
                << "solution\n" << pmesh << phys_density << flush;
        }

        // save every 50 iterations
        // if (paraview && it % 50 == 0)
        // {
        //     paraview_dc.SetCycle(it);
        //     paraview_dc.SetTime(it);
        //     paraview_dc.Save();
        // }
    }

    stage("optimization loop finished");
    double total_runtime = MPI_Wtime() - init_time;

    if (myid == 0)
    {
        csv.close();
        mfem::out << "\nfinished after " << (it - 1) << " iterations"
                  << "\ntotal runtime is " << total_runtime << " s\n";
    }

    // save the final solution
    // if (paraview)
    // {
    //     paraview_dc.SetCycle(it - 1);
    //     paraview_dc.SetTime(it - 1);
    //     paraview_dc.Save();
    // }

    // 11. Post process the solution mesh.  Everything lives on the full mesh
    // now; SaveSolidSubmesh thresholds by element so the passive void (rho~ ~ 0)
    // drops out on its own.
    if (paraview)
    {
        stage("post-processing: extracting + saving solid submesh");
        SaveSolidSubmesh(pmesh, rho_filter, phys_density, run_tag.str(), order, 0.4);
    }

    stage("done");
    return 0;
}

// Unit vector field pointing away from the z-axis: v = (x, y, 0) / |(x, y)|.
// The max-thickness measure advects the density along these straight radial
// lines; rho_a accumulates from the central hole outward and is read on the
// outer free surface where v.n > 0.
static void RadialOutwardRay(const Vector &x, Vector &v)
{
    v.SetSize(x.Size());
    v = 0.0;
    const real_t r = std::sqrt(x[0]*x[0] + x[1]*x[1]);
    if (r > 1e-12) { v[0] = x[0]/r; v[1] = x[1]/r; }
}

// save the thresholded design by clipping from the max value
void SaveSolidSubmesh(ParMesh &pmesh, ParGridFunction &desi_density,
                      ParGridFunction &phys_density, const std::string &run_tag, 
                      int order, real_t threshold)
{
    const int sol_attr = 1000;

    for (int i = 0; i < pmesh.GetNE(); i++)
    {
        real_t elem_max = -infinity();

        ElementTransformation *T = pmesh.GetElementTransformation(i);
        const FiniteElement *fe = desi_density.FESpace()->GetFE(i);
        const IntegrationRule &ir = fe->GetNodes();

        for (int j = 0; j < ir.GetNPoints(); j++)
        {
            const IntegrationPoint &ip = ir.IntPoint(j);
            T->SetIntPoint(&ip);
            real_t val = desi_density.GetValue(*T, ip);
            elem_max = max(elem_max, val);
        }

        if (elem_max > threshold)
        {
            pmesh.SetAttribute(i, sol_attr);
        }
    }
    pmesh.SetAttributes();

    Array<int> sol_mesh_attrs(1);
    sol_mesh_attrs[0] = sol_attr;

    ParSubMesh sol_submesh = ParSubMesh::CreateFromDomain(pmesh, sol_mesh_attrs);
    ParFiniteElementSpace filter_subfes(&sol_submesh, desi_density.ParFESpace()->FEColl());

    ParGridFunction desi_density_sub(&filter_subfes);
    ParGridFunction phys_density_sub(&filter_subfes);

    ParSubMesh::Transfer(desi_density, desi_density_sub);
    ParSubMesh::Transfer(phys_density, phys_density_sub);

    // save in separate paraview
    ParaViewDataCollection dc(run_tag, &sol_submesh);
    dc.SetPrefixPath("ParaView_fsol");
    dc.SetLevelsOfDetail(order);
    dc.SetDataFormat(VTKFormat::BINARY);
    dc.SetHighOrderOutput(true);
    dc.RegisterField("density", &phys_density_sub);
    dc.RegisterField("rho_filter", &desi_density_sub);
    dc.Save();
}

// 3x1x1 cantilever beam on a built-in hex grid, the default when no -m is given.
static MeshProblem SetupCartesianBeam(Mesh &mesh)
{
    mesh = Mesh::MakeCartesian3D(12, 4, 4, Element::HEXAHEDRON, 3.0, 1.0, 1.0);

    MeshProblem p;
    p.domain_attr = Array<int>({ 1 });          // MakeCartesian3D tags every element 1

    // attrs: 1 z=0, 2 y=0, 3 x=lx, 4 y=ly, 5 x=0 (clamped), 6 z=lz.
    p.outer_bdr_attrs = Array<int>({1, 2, 3, 4, 6});
    p.ray_bdr_attrs   = p.outer_bdr_attrs;      // p.rays left empty -> n_dir = 0

    p.cases.resize(1);
    LoadCase &lc = p.cases[0];
    lc.clamp_attrs = Array<int>({ 5 });

    // body force pushing down (-Z) on a small patch near the bottom right corner.
    LoadCase::VolumeLoad vl;
    vl.attrs = Array<int>({ 1 });
    vl.fn = [](const Vector &x, Vector &f)
    {
        const int dim = x.Size();

        f.SetSize(dim);
        f = 0.0;

        real_t radius = 0.05;
        real_t center_x = 2.9;
        real_t center_z = 0.1;

        bool x_in_range = (x[0] < center_x + radius) && (x[0] > center_x - radius);
        bool z_in_range = (x[2] < center_z + radius) && (x[2] > center_z - radius);

        if (x_in_range && z_in_range) f(2) = -1.0;
    };
    lc.vol_loads.push_back(vl);

    return p;
}

// Circular plate with an embedded cylinder and perimeter/top hex sleeves.
// gmsh physical groups (first tag -> MFEM attribute):
//   volumes  : 1  Plate                 -> design
//              2  CentralHollowCylinder -> design
//              3  EmbeddingCylinder     -> fixed void
//              11-22  PerimeterSleeve_1..12 -> fixed solid
//              31-36  TopSleeve_1..6        -> fixed solid + loaded
//   surfaces : 1  NonSleeveBoundary               -> filter rho~ = 0
//              2  EmbeddingCylinderOuterBoundary  -> filter natural BC, ray surface
//              11-22  PerimeterSleeveSurface_1..12 -> u = 0
//              31-36  TopSleeveSurface_1..6        -> filter rho~ = 1 (default)
// The mesh is centred on the z-axis, so "radial" == outward from (0,0).
static MeshProblem SetupCircularPlate(Mesh &mesh, const char *mesh_file)
{
    mesh = Mesh(mesh_file);

    MeshProblem p;

    // --- volumes ---------------------------------------------------------
    p.domain_attr = Array<int>({ 1, 2 });
    for (int a = 11; a <= 22; a++) { p.solid_attr.Append(a); }   // perimeter sleeves
    for (int a = 31; a <= 36; a++) { p.solid_attr.Append(a); }   // top sleeves
    // attribute 3 (embedding cylinder): not listed -> fixed void

    // --- filter boundary conditions ------------------------------------
    p.outer_bdr_attrs   = Array<int>({ 1 });   // rho~ = 0
    p.neumann_bdr_attrs = Array<int>({ 2 });   // zero-flux (no Dirichlet)
    p.ray_bdr_attrs     = Array<int>({ 1 });   // advection outflow surface (outer rim)
    // every other surface -> rho~ = 1

    // --- max-thickness rays ------------------------------------------------
    // One radial-outward field: the advection accumulates from the central hole
    // outward, and rho_a is read on the outer free surface (surface 1) where
    // v.n > 0, giving the radial material span from hub to rim.
    const int dim = mesh.Dimension();
    p.rays.push_back(
        std::make_unique<VectorFunctionCoefficient>(dim, RadialOutwardRay));

    // --- load cases ----------------------------------------------------
    Array<int> clamp_bdr;                       // u = 0 on the perimeter sleeves
    for (int a = 11; a <= 22; a++) { clamp_bdr.Append(a); }

    const int first_sleeve_attr = 31;
    const int n_sleeve          = 6;            // TopSleeve_1..6 -> attr 31..36
    Array<int> all_sleeves;
    for (int k = 0; k < n_sleeve; k++) { all_sleeves.Append(first_sleeve_attr + k); }

    p.cases.resize(3);

    // LC1: outward radial body force, unit magnitude, on every top sleeve.
    {
        LoadCase &lc = p.cases[0];
        lc.clamp_attrs = clamp_bdr;

        LoadCase::VolumeLoad vl;
        vl.attrs = all_sleeves;
        vl.fn = [](const Vector &x, Vector &f)
        {
            f.SetSize(x.Size());
            f = 0.0;
            const real_t r = std::sqrt(x[0]*x[0] + x[1]*x[1]);
            if (r > 1e-12) { f[0] = x[0]/r; f[1] = x[1]/r; }
        };
        lc.vol_loads.push_back(vl);
    }

    // LC2 / LC3: unit z body force on alternating top sleeves.
    //   LC2 amplitudes over sleeves 1..6:  1 0 1 0 1 0   (odd sleeves)
    //   LC3 amplitudes over sleeves 1..6:  0 1 0 1 0 1   (even sleeves)
    for (int lc_idx = 1; lc_idx <= 2; lc_idx++)
    {
        LoadCase &lc = p.cases[lc_idx];
        lc.clamp_attrs = clamp_bdr;

        const int loaded_parity = (lc_idx == 1) ? 0 : 1;   // sleeve index k parity
        for (int k = 0; k < n_sleeve; k++)
        {
            if (k % 2 != loaded_parity) { continue; }

            LoadCase::VolumeLoad vl;
            vl.attrs = Array<int>({ first_sleeve_attr + k });
            vl.value.SetSize(3);
            vl.value = 0.0;
            vl.value(2) = 1.0;
            lc.vol_loads.push_back(vl);
        }
    }

    return p;
}

// select the per-mesh setup from the mesh file name
MeshProblem loadMesh(int myid, const char *mesh_file, Mesh &mesh)
{
    // no -m: fall back to the built-in Cartesian beam
    if (!mesh_file || mesh_file[0] == '\0')
    {
        return SetupCartesianBeam(mesh);
    }

    if (strstr(mesh_file, "circular_plate_hex_sleeves_embedded_cylinder") != NULL)
    {
        return SetupCircularPlate(mesh, mesh_file);
    }

    if (myid == 0) { mfem::out << "invalid mesh file" << endl; }
    return MeshProblem();
}
