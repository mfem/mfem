// Post-processing driver for the max-thickness topology optimization.
// Loads a saved design (rho, and alpha when present) from a checkpoint
// directory, filters it, runs the thickness advection once per ray direction
// with the GMRES/BlockILU linear solver, and writes the thickness fields to
// ParaView.  No elasticity solve, no MMA, no optimization loop -- the mesh,
// filter, ray and outflow setup mirror ElastTopOpt_ct.cpp so the fields match
// what that driver would have produced at the checkpointed iteration.
//
// The checkpoint stores bare true-dof vectors, so the run must use the same
// mesh, -r, -o, -rt and MPI rank count that wrote it.
//
// Sample run:  mpirun -np 8 ./ElastTopOpt_post -m "./data/disk_6_holes.msh" -cd checkpoints_ct -vf 0.4 -amax 1.0

#include "mfem.hpp"
#include "ElastTopOpt.hpp"
#include "qoi.hpp"
#include "pseudo_transient_solver.hpp"
#include "../../diffusion_mass_solver.hpp"
#include "checkpoint.hpp"
#include <memory>
#include <iomanip>
#include <sstream>
#include <vector>

using namespace std;
using namespace mfem;

// Boundary data for one elasticity solve.  Unused here, but loadMesh() is
// shared verbatim with ElastTopOpt_ct.cpp, which fills it.
struct LoadCase
{
    Array<int>    clamp_attrs;
    Array<int>    load_attrs;
    Array<real_t> fx, fy;
};

// n_dir directions evenly spaced over [start, start+span).
struct RaySet
{
    int    n_dir = 4;
    real_t span  = M_PI;
    real_t start = 0.0;
};

// Ray directions per mode.
struct RaySpec
{
    RaySet parallel;
    RaySet cone;
    real_t half_ang = M_PI / 12;    // cone half-opening angle (15 deg -> 30 deg cone)
    real_t dom_h = 1.05;            // cone half angle height
};

// Full problem definition for one mesh.
struct MeshProblem
{
    Array<int>            domain_attr;
    Array<int>            outer_bdr_attrs;
    std::vector<LoadCase> cases;
    RaySpec               rays;
};

enum class RayMode { Parallel = 1, Cone = 2 };

std::vector<Vector> BuildRays(RayMode mode, const RaySpec &spec, int dim);
std::unique_ptr<VectorCoefficient> MakeRayCoeff(RayMode mode, const RaySpec &spec,
                                                const Vector &axis, int dim);
MeshProblem loadMesh(int myid, const char *mesh_file, Mesh &mesh);

int main(int argc, char *argv[])
{
    Mpi::Init();
    int myid = Mpi::WorldRank();
    Hypre::Init();

    // 1. Options.  The discretization options must match the run that wrote
    //    the checkpoint; beta/eta only affect the reported density field.
    const char *mesh_file = nullptr;
    const char *cp_dir    = "checkpoints_ct";
    const char *pv_name   = "";       // empty -> derive the name from the options
    int    ref_levels   = 0;
    int    order        = 2;
    real_t r_f          = 0.03;       // min filter length
    real_t alpha_max    = 1.0;        // thickness variable upper bound
    real_t vol_fraction = 0.4;
    real_t beta         = 2.0;        // Heaviside beta (reported density only)
    real_t eta          = 0.2;        // Heaviside eta (reported density only)
    int    ray_type     = 1;          // thickness ray strategy: 1=parallel, 2=cone
    real_t ray_angle    = -1.0;       // single ray at this angle; negative = full ray set
    bool   rho_only     = false;      // ignore metadata/alpha, load rho alone

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh", "mesh file to use", true);
    args.AddOption(&cp_dir, "-cd", "--checkpoint-dir",
                   "directory holding the checkpoint to post-process");
    args.AddOption(&pv_name, "-pvn", "--paraview-name",
                   "ParaView collection name (default: derived from the options)");
    args.AddOption(&ref_levels, "-r", "--refine", "uniform refinement levels");
    args.AddOption(&order, "-o", "--order", "finite element order");
    args.AddOption(&vol_fraction, "-vf", "--volume-fraction", "volume fraction");
    args.AddOption(&r_f, "-rf", "--r_fwidth", "min filter width");
    args.AddOption(&alpha_max, "-amax", "--alpha_max", "upper bound for the thickness variables");
    args.AddOption(&beta, "-b", "--beta", "Heaviside beta (reported density only)");
    args.AddOption(&eta, "-eta", "--eta", "Heaviside eta (reported density only)");
    args.AddOption(&ray_type, "-rt", "--ray-type", "ray type: 1 = parallel, 2 = cone");
    args.AddOption(&ray_angle, "-ang", "--ray-angle",
                   "solve a single direction at this angle in degrees, instead of the "
                   "mesh's ray set; negative uses the full set (angles wrap, so use 315 "
                   "rather than -45)");
    args.AddOption(&rho_only, "-ro", "--rho-only", "-full", "--full-checkpoint",
                   "load rho alone, ignoring metadata.txt and alpha; needed when a "
                   "SaveRho() write left stale metadata from an earlier run behind");
    args.Parse();
    if (!args.Good())
    {
        if (myid == 0) { args.PrintUsage(cout); }
        return 1;
    }
    if (!mesh_file)
    {
        if (myid == 0) { cout << "Error: -m <mesh file> is required." << endl; }
        return 1;
    }
    // the checkpointed alpha is indexed by the ray set it was optimized with,
    // so it does not carry over to an arbitrary single angle
    if (ray_angle >= 0.0) { rho_only = true; }

    if (myid == 0) { args.PrintOptions(cout); }

    // fallback alpha when the checkpoint only carries rho
    const real_t domain_init = alpha_max * vol_fraction;
    const RayMode ray_mode = (ray_type == 2) ? RayMode::Cone : RayMode::Parallel;

    // 2. Load the mesh and the problem description (domain, ray spec).
    Mesh mesh;
    MeshProblem prob = loadMesh(myid, mesh_file, mesh);

    Array<int> &domain_attr     = prob.domain_attr;
    Array<int> &outer_bdr_attrs = prob.outer_bdr_attrs;

    // 3. Refine the mesh and construct pmesh / the design subdomain.
    const int dim = mesh.Dimension();

    // -ang replaces the mesh's ray set with the single direction of interest
    // (the diagonals, say, on the square domain).
    const bool single_ray = (ray_angle >= 0.0);
    vector<Vector> ray_dirs;
    if (single_ray)
    {
        const real_t ang = ray_angle * M_PI / 180.0;
        Vector v(dim); v = 0.0;
        v(0) = cos(ang);
        v(1) = sin(ang);
        ray_dirs.push_back(v);
    }
    else
    {
        ray_dirs = BuildRays(ray_mode, prob.rays, dim);
    }
    const int n_dir = static_cast<int>(ray_dirs.size());

    for (int l = 0; l < ref_levels; l++)
    {
        mesh.UniformRefinement();
    }

    ParMesh pmesh(MPI_COMM_WORLD, mesh);
    mesh.Clear();

    ParSubMesh design_domain = ParSubMesh::CreateFromDomain(pmesh, domain_attr);

    // 3b. Build ray fields and mark the outflow boundary for each direction.
    Array<int> candidate_be;        // extract only the outer boundary elements
    Array<int> candidate_attr;
    for (int i = 0; i < pmesh.GetNBE(); i++)
    {
        const int el_attr = pmesh.GetBdrAttribute(i);

        if (outer_bdr_attrs.Find(el_attr) < 0) continue;
        candidate_be.Append(i);
        candidate_attr.Append(el_attr);
    }

    vector<unique_ptr<VectorCoefficient>> ray_cf(n_dir);
    vector<unique_ptr<ParSubMesh>> outflow(n_dir);

    for (int r = 0; r < n_dir; r++)
    {
        ray_cf[r] = MakeRayCoeff(ray_mode, prob.rays, ray_dirs[r], dim);

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
    H1_FECollection filter_fec(order, dim);
    H1_FECollection control_fec(order-1, dim, BasisType::GaussLobatto);
    ParFiniteElementSpace filter_fes(&design_domain, &filter_fec);
    ParFiniteElementSpace control_fes(&design_domain, &control_fec);

    // parent-mesh H1 space, only used to lift the design fields for output
    ParFiniteElementSpace parent_filter_fes(&pmesh, &filter_fec);

    HYPRE_BigInt filter_size  = filter_fes.GlobalTrueVSize();
    HYPRE_BigInt control_size = control_fes.GlobalTrueVSize();
    if (myid == 0)
    {
        cout << "\nfilter dofs = " << filter_size
             << ",  design dofs = " << control_size
             << ",  rays = " << n_dir << endl;
    }

    // 5. Design fields.
    ParGridFunction rho(&control_fes);
    ParGridFunction rho_filter(&filter_fes);
    rho = domain_init;
    rho_filter = domain_init;

    // 5b. Thickness variables live on the outflow submeshes, one set per ray.
    DG_FECollection dgfec(order, dim, BasisType::GaussLobatto);
    ParFiniteElementSpace dgfes(&pmesh, &dgfec);

    vector<unique_ptr<DG_FECollection>> sub_dg_fec(n_dir);
    vector<unique_ptr<ParFiniteElementSpace>> sub_dg_fes(n_dir);
    vector<unique_ptr<ParGridFunction>> alpha(n_dir);

    for (int r = 0; r < n_dir; r++)
    {
        const int sub_dim = outflow[r]->Dimension();
        sub_dg_fec[r] = make_unique<DG_FECollection>(order, sub_dim, BasisType::Positive);
        sub_dg_fes[r] = make_unique<ParFiniteElementSpace>(outflow[r].get(), sub_dg_fec[r].get());

        alpha[r] = make_unique<ParGridFunction>(sub_dg_fes[r].get());
        *alpha[r] = domain_init;
    }

    // reported design density: the eta = 0.5 projection of rho~
    HeavisideCoefficient rho_inter_cf(&rho_filter, beta, 0.5);

    // 6. Min length scale filter solver (diffusion-mass PDE filter).
    PDEFilter filter(control_fes, filter_fes);
    filter.SetFilterRadius(r_f);
    DiffusionMassSolver &filter_solver = filter.GetSolver();
    for (int a = 1; a <= design_domain.bdr_attributes.Max(); a++)
    {
        bool is_outer = (outer_bdr_attrs.Find(a) >= 0);
        if (is_outer)
        {
            filter_solver.Boundary().Add(a, 0.0);  // Outer boundaries: rho~ = 0
        }
        else
        {
            filter_solver.Boundary().Add(a, 1.0);  // Holes: rho~ = 1
        }
    }
    filter.Assemble();

    // PDEFilter::Mult solves the constrained system without eliminating the
    // essential columns, so nonzero Dirichlet data never reaches the interior.
    // The filter is linear, so recover the missing A_fe*g lifting once and add
    // it back, exactly as the optimization driver does.
    Array<int> filter_bdr_marker(design_domain.bdr_attributes.Max());
    filter_bdr_marker = 1;
    Array<int> filter_ess_tdofs;
    filter_fes.GetEssentialTrueDofs(filter_bdr_marker, filter_ess_tdofs);

    Vector rho_filter_lift_tv;
    {
        ParGridFunction lift(&filter_fes);
        filter_solver.Solve(lift);
        lift.GetTrueDofs(rho_filter_lift_tv);
        rho_filter_lift_tv.SetSubVector(filter_ess_tdofs, real_t(0));
    }

    // 6b. Advection solvers for the thickness measure, one per ray direction.
    vector<unique_ptr<MaterialThicknessSolver>> advect(n_dir);
    for (int r = 0; r < n_dir; r++)
    {
        advect[r] = make_unique<MaterialThicknessSolver>(filter_fes, dgfes, *ray_cf[r]);
        advect[r]->AssembleLinearSolver();
    }

    // 6c. Max-thickness residual per ray:  1/2 int_Gamma_out,r (rho_a - alpha_r)^2
    vector<unique_ptr<AdvectThicknessResidual>> adv_res(n_dir);
    for (int r = 0; r < n_dir; r++)
    {
        adv_res[r] = make_unique<AdvectThicknessResidual>(*outflow[r],
                                                          advect[r]->GetRhoA(),
                                                          *alpha[r]);
    }

    // 7. Load the checkpointed design.  A full checkpoint carries rho + alpha
    //    and is validated against the current discretization; a rho-only
    //    checkpoint (SaveRho) leaves alpha at its uniform initial value.
    Checkpoint checkpoint(cp_dir, MPI_COMM_WORLD);

    Vector rho_tv(control_fes.GetTrueVSize());
    rho.GetTrueDofs(rho_tv);

    vector<Vector> alpha_tv(n_dir);
    for (int r = 0; r < n_dir; r++) { alpha[r]->GetTrueDofs(alpha_tv[r]); }

    bool have_alpha = false;
    if (checkpoint.Exists() && !rho_only)
    {
        MFEM_VERIFY(checkpoint.ValidateCompatibility(ref_levels, order, n_dir),
                    "Checkpoint incompatible with the current run parameters.");
        MFEM_VERIFY(checkpoint.Load(rho_tv, alpha_tv),
                    "Failed to load checkpoint data.");
        have_alpha = true;
        for (int r = 0; r < n_dir; r++) { alpha[r]->SetFromTrueDofs(alpha_tv[r]); }

        if (myid == 0)
        {
            mfem::out << "\nLoaded checkpoint from '" << cp_dir << "':"
                      << "\n  iteration = " << checkpoint.GetIteration()
                      << "\n  epsilon   = " << checkpoint.GetEpsilon()
                      << "\n  init_comp = " << checkpoint.GetInitComp() << "\n";
        }
    }
    else
    {
        MFEM_VERIFY(checkpoint.RhoExists(),
                    "No checkpoint found in the directory given by -cd.");
        MFEM_VERIFY(checkpoint.LoadRho(rho_tv), "Failed to load rho.");

        if (myid == 0)
        {
            mfem::out << "\nLoaded rho-only checkpoint from '" << cp_dir
                      << "'; alpha stays at the uniform value " << domain_init
                      << ", so the residuals below are not meaningful.\n";
        }
    }
    rho.SetFromTrueDofs(rho_tv);

    // 8. Forward pass: filter rho, then advect rho~ along each ray.
    const int nf = filter_fes.GetTrueVSize();
    Vector rho_filter_tv(nf);
    filter.Mult(rho_tv, rho_filter_tv);
    rho_filter_tv += rho_filter_lift_tv;
    rho_filter.SetFromTrueDofs(rho_filter_tv);

    // one stored copy of rho_a per ray: the solvers each own theirs, but all
    // directions are solved before anything is written out
    vector<unique_ptr<ParGridFunction>> rho_a(n_dir);
    vector<unique_ptr<ParGridFunction>> rho_a_sub(n_dir);
    ParGridFunction rho_a_max(&dgfes);
    rho_a_max = 0.0;

    Vector ray_max(n_dir), ray_alpha_max(n_dir), ray_res(n_dir);

    double adv_runtime = MPI_Wtime();
    for (int r = 0; r < n_dir; r++)
    {
        advect[r]->SetRhs(rho_filter_tv);
        advect[r]->LinearFSolve();

        rho_a[r] = make_unique<ParGridFunction>(&dgfes);
        *rho_a[r] = advect[r]->GetRhoA();

        // outflow trace: where the thickness measure actually lives
        rho_a_sub[r] = make_unique<ParGridFunction>(sub_dg_fes[r].get());
        *rho_a_sub[r] = 0.0;
        ParSubMesh::Transfer(*rho_a[r], *rho_a_sub[r]);

        // DG: no shared dofs, so the envelope is a plain entrywise max
        for (int i = 0; i < rho_a_max.Size(); i++)
        {
            rho_a_max(i) = max(rho_a_max(i), (*rho_a[r])(i));
        }

        ray_max(r)       = rho_a[r]->Max();     // local max, reduced below
        ray_alpha_max(r) = alpha[r]->Max();
        ray_res(r)       = adv_res[r]->Eval();  // already global
    }
    adv_runtime = MPI_Wtime() - adv_runtime;

    MPI_Allreduce(MPI_IN_PLACE, ray_max.GetData(), n_dir,
                  MPITypeMap<real_t>::mpi_type, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, ray_alpha_max.GetData(), n_dir,
                  MPITypeMap<real_t>::mpi_type, MPI_MAX, MPI_COMM_WORLD);

    if (myid == 0)
    {
        mfem::out << "\nAdvection runtime: " << fixed << setprecision(4)
                  << adv_runtime << " s\n\n"
                  << "  ray    max rho_a    max alpha      residual\n";
        for (int r = 0; r < n_dir; r++)
        {
            mfem::out << setw(5) << r << scientific << setprecision(4)
                      << setw(13) << ray_max(r)
                      << setw(13) << ray_alpha_max(r)
                      << setw(14) << ray_res(r) << "\n";
        }
        mfem::out << "  max over rays: rho_a = " << ray_max.Max()
                  << ",  residual = " << ray_res.Max() << "\n";
        if (have_alpha)
        {
            mfem::out << "  checkpoint epsilon = " << checkpoint.GetEpsilon()
                      << " (residual - epsilon is the constraint value)\n";
        }
        mfem::out << defaultfloat;
    }

    // 9. ParaView output.
    std::ostringstream tag;
    if (pv_name[0] != '\0') { tag << pv_name; }
    else
    {
        tag << "post_rt" << ray_type;
        if (single_ray) { tag << "_ang" << ray_angle; }
        tag << "_amax" << alpha_max << "_vf" << vol_fraction;
    }

    const int cycle = have_alpha ? checkpoint.GetIteration() : 0;

    // 9a. Volume fields on the parent mesh: the per-ray thickness fields, their
    //     envelope, and the design lifted from the design subdomain.
    ParGridFunction rho_filter_parent(&parent_filter_fes);
    ParGridFunction density_parent(&parent_filter_fes);
    rho_filter_parent = 0.0;
    density_parent    = 0.0;
    {
        ParGridFunction density(&filter_fes);
        density.ProjectCoefficient(rho_inter_cf);
        ParSubMesh::Transfer(rho_filter, rho_filter_parent);
        ParSubMesh::Transfer(density, density_parent);
    }

    ParaViewDataCollection pv_vol(tag.str(), &pmesh);
    pv_vol.SetPrefixPath("ParaView_post");
    pv_vol.SetLevelsOfDetail(order);
    pv_vol.SetDataFormat(VTKFormat::BINARY);
    pv_vol.SetHighOrderOutput(true);
    pv_vol.RegisterField("rho_filter", &rho_filter_parent);
    pv_vol.RegisterField("density", &density_parent);
    pv_vol.RegisterField("rho_a_max", &rho_a_max);
    vector<string> field_names(n_dir);
    for (int r = 0; r < n_dir; r++)
    {
        field_names[r] = "rho_a_" + to_string(r);
        pv_vol.RegisterField(field_names[r], rho_a[r].get());
    }
    pv_vol.SetCycle(cycle);
    pv_vol.SetTime(cycle);
    pv_vol.Save();

    // 9b. Outflow traces, one collection per ray: the thickness measure and the
    //     thickness variable it is constrained against.
    for (int r = 0; r < n_dir; r++)
    {
        ParaViewDataCollection pv_out(tag.str() + "_out" + to_string(r),
                                      outflow[r].get());
        pv_out.SetPrefixPath("ParaView_post");
        pv_out.SetLevelsOfDetail(order);
        pv_out.SetDataFormat(VTKFormat::BINARY);
        pv_out.SetHighOrderOutput(true);
        pv_out.RegisterField("rho_a", rho_a_sub[r].get());
        pv_out.RegisterField("alpha", alpha[r].get());
        pv_out.SetCycle(cycle);
        pv_out.SetTime(cycle);
        pv_out.Save();
    }

    if (myid == 0)
    {
        mfem::out << "\nParaView output written to ParaView_post/" << tag.str()
                  << " (+ _out<r> for the outflow traces)\n";
    }

    return 0;
}

// ray axis directions for the selected mode, evenly spaced over its ray set
std::vector<Vector> BuildRays(RayMode mode, const RaySpec &spec, int dim)
{
    const RaySet &s = (mode == RayMode::Cone) ? spec.cone : spec.parallel;

    std::vector<Vector> dirs;
    for (int i = 0; i < s.n_dir; i++)
    {
        const real_t ang = s.start + s.span * i / s.n_dir;
        Vector v(dim); v = 0.0;
        v(0) = cos(ang);
        v(1) = sin(ang);
        dirs.push_back(v);
    }
    return dirs;
}

// advection direction field for one ray: parallel is a constant axis, cone is a
// unit field diverging from a source 1/tan(half_ang) behind the origin
std::unique_ptr<VectorCoefficient> MakeRayCoeff(RayMode mode, const RaySpec &spec,
                                                const Vector &axis, int dim)
{
    if (mode == RayMode::Parallel)
    {
        return std::make_unique<VectorConstantCoefficient>(axis);
    }

    const real_t R = spec.dom_h / sin(spec.half_ang);
    Vector src(axis); src *= -R;
    auto field = [src, dim](const Vector &x, Vector &d)
    {
        d.SetSize(dim);
        subtract(x, src, d);
        const real_t n = d.Norml2();
        if (n > 0) { d /= n; }
    };
    return std::make_unique<VectorFunctionCoefficient>(dim, field);
}

// disk mesh: one load case, rays swept over the full circle
static MeshProblem SetupDisk6Holes(Mesh &mesh, const char *mesh_file)
{
    mesh = Mesh(mesh_file);

    MeshProblem p;
    p.domain_attr.Append(1);
    p.outer_bdr_attrs = Array<int>({1});
    p.rays.parallel = { 5, 2 * M_PI, 0.0 };     // aligned with pentagon edges
    p.rays.cone     = { 5, 2 * M_PI, 0.0 };

    p.cases.resize(1);
    LoadCase &lc = p.cases[0];
    lc.clamp_attrs = Array<int>({ 7 });

    const real_t angles_deg_forces[] = {90.0, 162.0, 234.0, 306.0, 18.0};
    for (int j = 0; j < 5; j++)
    {
        const real_t ang = angles_deg_forces[j] * M_PI / 180.0;
        const real_t fdx = -sin(ang), fdy = cos(ang);
        lc.load_attrs.Append(2 + j);
        lc.fx.Append(-fdx);
        lc.fy.Append(-fdy);
    }
    return p;
}

// square mesh: two load cases with swapped clamp/load, 4 parallel or 8 cone rays
static MeshProblem SetupSquare4Holes(Mesh &mesh, const char *mesh_file)
{
    mesh = Mesh(mesh_file);

    MeshProblem p;
    p.domain_attr.Append(1);
    p.outer_bdr_attrs = Array<int>({1, 2, 3, 4});
    p.rays.parallel = { 4, M_PI,     0.0 };
    p.rays.cone     = { 8, 2 * M_PI, 0.0 };
    p.rays.half_ang = M_PI / 4;
    p.rays.dom_h    = 0.8;

    p.cases.resize(2);

    // first elast solve
    p.cases[0].clamp_attrs = Array<int>({ 6, 7});
    p.cases[0].load_attrs  = Array<int>({ 5, 8});
    p.cases[0].fx          = Array<real_t>({ 1,-1});
    p.cases[0].fy          = Array<real_t>({-1, 1});

    // second elast solve
    p.cases[1].clamp_attrs = Array<int>({ 5, 8});
    p.cases[1].load_attrs  = Array<int>({ 6, 7});
    p.cases[1].fx          = Array<real_t>({-1, 1});
    p.cases[1].fy          = Array<real_t>({-1, 1});

    return p;
}

// pentagon mesh: five load cases, rays aligned with the edges
static MeshProblem SetupPentagon(Mesh &mesh, const char *mesh_file)
{
    mesh = Mesh(mesh_file);

    MeshProblem p;
    p.domain_attr.Append(1);
    p.outer_bdr_attrs = Array<int>({1});
    p.rays.parallel = { 5, 2 * M_PI, 0.0 };     // aligned with pentagon edges
    p.rays.cone     = { 5, 2 * M_PI, 0.0 };

    const int n_case = 5;
    const real_t angles_deg_forces[] = {18.0, 90.0, 162.0, 234.0, 306.0};
    real_t fdx[5], fdy[5];
    for (int k = 0; k < n_case; k++)
    {
        const real_t ang = angles_deg_forces[k] * M_PI / 180.0;
        fdx[k] = cos(ang);
        fdy[k] = sin(ang);
    }

    const int first_attr = 2;
    p.cases.resize(n_case);
    for (int i = 0; i < n_case; i++)
    {
        const int clamped_attr = first_attr + i;
        LoadCase &lc = p.cases[i];
        lc.clamp_attrs = Array<int>({ clamped_attr });

        for (int j = 0; j < n_case; j++)
        {
            const int attr = first_attr + j;
            if (attr == clamped_attr) { continue; }
            lc.load_attrs.Append(attr);
            lc.fx.Append(-fdx[j]);
            lc.fy.Append(-fdy[j]);
        }
    }
    return p;
}

// select the per-mesh setup from the mesh file name
MeshProblem loadMesh(int myid, const char *mesh_file, Mesh &mesh)
{
    if (strstr(mesh_file, "disk_6_holes.msh") != NULL)
    {
        return SetupDisk6Holes(mesh, mesh_file);
    }
    else if (strstr(mesh_file, "d_square_4_holes.msh") != NULL)
    {
        return SetupSquare4Holes(mesh, mesh_file);
    }
    else if (strstr(mesh_file, "circular_5_holes_pentagon.msh") != NULL)
    {
        return SetupPentagon(mesh, mesh_file);
    }

    if (myid == 0) { mfem::out << "invalid mesh file" << endl; }
    return MeshProblem();
}
