/// Solve the steady isentropic vortex problem on cut-cell domain
// set this const expression to true in order to use entropy variables for state
constexpr bool entvar = false;
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <random>
#include "euler_integ_cut.hpp"
#include "evolver_GD.hpp"
#include "cut_quad.hpp"
#include "gd_def_cut.hpp"
#include "gd_cut.hpp"
using namespace std;
using namespace mfem;
std::default_random_engine gen(std::random_device{}());
std::uniform_real_distribution<double> normal_rand(-1.0, 1.0);
static std::uniform_real_distribution<double> uniform_rand(0.0, 1.0);
const double rho = 0.9856566615165173;
const double rhoe = 2.061597236955558;
const double rhou[3] = {0.09595562550099601, -0.030658751626551423, -0.13471469906596886};

/// \brief Defines the exact solution for the steady isentropic vortex
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - state variables stored as a 4-vector
void uexact(const Vector &x, Vector &u);

template <int dim>
void randBaselinePert(const mfem::Vector &x, mfem::Vector &u)
{

    const double scale = 0.01;
    u(0) = rho * (1.0 + scale * uniform_rand(gen));
    u(dim + 1) = rhoe * (1.0 + scale * uniform_rand(gen));
    for (int di = 0; di < dim; ++di)
    {
        u(di + 1) = rhou[di] * (1.0 + scale * uniform_rand(gen));
    }
}

/// function to calculate conservative variables l2error
template <int dim, bool entvar>
double calcCutConservativeVarsL2Error(
    void (*u_exact)(const mfem::Vector &, mfem::Vector &), GridFunction *u, mfem::FiniteElementSpace *fes,
    std::vector<bool> &EmbeddedElems, std::map<int, IntegrationRule *> &CutSquareIntRules,
    int num_state, int entry)
{
    // This lambda function computes the error at a node
    // Beware: this is not particularly efficient, given the conditionals
    // Also **NOT thread safe!**
    Vector qdiscrete(dim + 2), qexact(dim + 2); // define here to avoid reallocation
    auto node_error = [&](const Vector &discrete, const Vector &exact) -> double {
        if (entvar)
        {
            calcConservativeVars<dim>(discrete.GetData(),
                                      qdiscrete.GetData());
            calcConservativeVars<dim>(exact.GetData(), qexact.GetData());
        }
        else
        {
            qdiscrete = discrete;
            qexact = exact;
        }
        double err = 0.0;
        if (entry < 0)
        {
            for (int i = 0; i < dim + 2; ++i)
            {
                double dq = qdiscrete(i) - qexact(i);
                err += dq * dq;
            }
        }
        else
        {
            err = qdiscrete(entry) - qexact(entry);
            err = err * err;
        }
        return err;
    };

    VectorFunctionCoefficient exsol(num_state, u_exact);
    DenseMatrix vals, exact_vals;
    Vector u_j, exsol_j;
    double loc_norm = 0.0;
    for (int i = 0; i < fes->GetNE(); i++)
    {
        if (EmbeddedElems.at(i) == true)
        {
            loc_norm += 0.0;
        }
        else
        {
            const FiniteElement *fe = fes->GetFE(i);
            const IntegrationRule *ir;
            ir = CutSquareIntRules[i];
            if (ir == NULL)
            {
                int intorder = 2 * fe->GetOrder() + 3;
                ir = &(IntRules.Get(fe->GetGeomType(), intorder));
            }
            ElementTransformation *T = fes->GetElementTransformation(i);
            u->GetVectorValues(*T, *ir, vals);
            exsol.Eval(exact_vals, *T, *ir);
            for (int j = 0; j < ir->GetNPoints(); j++)
            {
                const IntegrationPoint &ip = ir->IntPoint(j);
                T->SetIntPoint(&ip);
                vals.GetColumnReference(j, u_j);
                exact_vals.GetColumnReference(j, exsol_j);
                loc_norm += ip.weight * T->Weight() * node_error(u_j, exsol_j);
            }
        }
    }

    double norm = loc_norm;
    if (norm < 0.0) // This was copied from mfem...should not happen for us
    {
        return -sqrt(-norm);
    }
    return sqrt(norm);
}

double CutComputeL2Error(GridFunction &x, FiniteElementSpace *fes,
                         Coefficient &exsol, const std::vector<bool> &EmbeddedElems,
                         std::map<int, IntegrationRule *> &CutSquareIntRules);
void randState(const mfem::Vector &x, mfem::Vector &u)
{
    for (int i = 0; i < u.Size(); ++i)
    {
        u(i) = 2.0 * uniform_rand(gen) - 1.0;
    }
}

double calcResidualNorm(NonlinearForm *res, FiniteElementSpace *fes, CentGridFunction &uc)
{
    CentGridFunction residual(fes);
    residual = 0.0;
    res->Mult(uc, residual);
    return residual.Norml2();
}

/// get freestream state values for the far-field bcs
template <int dim, bool entvar>
void getFreeStreamState(mfem::Vector &q_ref)
{
    double mach_fs = 0.3;
    q_ref = 0.0;
    q_ref(0) = 1.0;
    q_ref(1) = q_ref(0) * mach_fs; // ignore angle of attack
    q_ref(2) = 0.0;
    q_ref(dim + 1) = 1 / (euler::gamma * euler::gami) + 0.5 * mach_fs * mach_fs;
}

/// \brief Defines the random function for the jabocian check
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - conservative variables stored as a 4-vector
void pert(const Vector &x, Vector &p);

/// \brief Returns the value of the integrated math entropy over the domain
double calcEntropyTotalExact();

/// \brief Defines the exact solution for the steady isentropic vortex
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - state variables stored as a 4-vector
void uexact(const Vector &x, Vector &u);

int main(int argc, char *argv[])
{
    // 1. Parse command-line options.
    int order = 1;
    int N = 5;
    int ref_levels = -1;
    int ncr = 2;
    double radius = 1.0;
    /// number of state variables
    int num_state = 4;
    double alpha = 1.0;
    double cutsize;
    OptionsParser args(argc, argv);
    args.AddOption(&order, "-o", "--order",
                   "Finite element order (polynomial degree) or -1 for"
                   " isoparametric space.");
    args.AddOption(&N, "-n", "--#elements",
                   "number of mesh elements.");
    args.AddOption(&radius, "-r", "--radius",
                   "radius of circle.");
    args.AddOption(&ref_levels, "-ref", "--refine",
                   "refine levels");
    args.Parse();
    if (!args.Good())
    {
        args.PrintUsage(cout);
        return 1;
    }
    args.PrintOptions(cout);

    Mesh *mesh = new Mesh(N, N, Element::QUADRILATERAL, true,
                          3, 3, true);
    ofstream sol_ofv("square_mesh_vortex.vtk");
    sol_ofv.precision(14);
    mesh->PrintVTK(sol_ofv, 0);
    int dim = mesh->Dimension();
    /// find the elements to refine
    for (int k = 0; k < 1; ++k)
    {
        Array<int> marked_elements;
        for (int i = 0; i < mesh->GetNE(); ++i)
        {
            if ((cutByGeom<1>(mesh, i) == true))
            {
                marked_elements.Append(i);
            }
        }
        mesh->GeneralRefinement(marked_elements, 1);
    }
    ofstream wmesh("square_mesh_vortex_nc.vtk");
    wmesh.precision(14);
    mesh->PrintVTK(wmesh, 0);

    cout << "#elements after refinement " << mesh->GetNE() <<  endl;
    for (int l = 0; l < ref_levels; l++)
    {
        mesh->UniformRefinement();
    }

    //find the elements cut by inner circle boundary
    vector<int> cutelems_inner;
    vector<int> embeddedelems_inner;
    vector<int> cutFaces_inner;
    vector<int> cutBdrFaces_inner;
    for (int i = 0; i < mesh->GetNE(); ++i)
    {
        if (cutByGeom<1>(mesh, i) == true)
        {
            cutelems_inner.push_back(i);
        }
        if (insideBoundary<1>(mesh, i) == true)
        {
            embeddedelems_inner.push_back(i);
        }
    }
    cout << "elements cut by inner circle:  " << endl;
    for (int i = 0; i < cutelems_inner.size(); ++i)
    {
        cout << cutelems_inner.at(i) << endl;
    }
    cout << "elements completely inside inner circle:  " << endl;
    for (int i = 0; i < embeddedelems_inner.size(); ++i)
    {
        cout << embeddedelems_inner.at(i) << endl;
    }
    /// find faces cut by inner circle
    for (int i = 0; i < mesh->GetNumFaces(); ++i)
    {
        FaceElementTransformations *tr;
        // tr = mesh->GetInteriorFaceTransformations(i);
        tr = mesh->GetFaceElementTransformations(i);
        if (tr->Elem2No >= 0)
        {
            if ((find(cutelems_inner.begin(), cutelems_inner.end(), tr->Elem1No) != cutelems_inner.end()) &&
                (find(cutelems_inner.begin(), cutelems_inner.end(), tr->Elem2No) != cutelems_inner.end()))
            {
                // cout << "interior face is " << tr->Face->ElementNo << endl;
                // cout << tr->Elem1No << " , " << tr->Elem2No << endl;
                cutFaces_inner.push_back(tr->Face->ElementNo);
            }
        }
        if (tr->Elem2No < 0)
        {
            if (find(cutelems_inner.begin(), cutelems_inner.end(), tr->Elem1No) != cutelems_inner.end())
            {
                cutBdrFaces_inner.push_back(tr->Face->ElementNo);
                // cout << "boundary face is " << tr->Face->ElementNo << endl;
                // cout << tr->Elem1No << endl;
            }
        }
    }
    // cout << "faces cut by inner circle " << endl;
    for (int k = 0; k < cutFaces_inner.size(); ++k)
    {
        // cout << cutFaces_inner.at(k) << endl;
    }

    //find the elements cut by outer circle boundary
    vector<int> cutelems_outer;
    vector<int> embeddedelems_outer;
    vector<int> cutinteriorFaces;
    vector<int> cutFaces_outer;
    for (int i = 0; i < mesh->GetNE(); ++i)
    {
        if (cutByGeom<2>(mesh, i) == true)
        {
            cutelems_outer.push_back(i);
        }
        if (insideBoundary<2>(mesh, i) == true)
        {
            embeddedelems_outer.push_back(i);
        }
    }
    cout << "elements cut by outer circle:  " << endl;
    for (int i = 0; i < cutelems_outer.size(); ++i)
    {
        cout << cutelems_outer.at(i) << endl;
    }
    cout << "elements completely outside outer circle:  " << endl;
    for (int i = 0; i < embeddedelems_outer.size(); ++i)
    {
        cout << embeddedelems_outer.at(i) << endl;
    }

    /// find faces cut by outer circle
    for (int i = 0; i < mesh->GetNumFaces(); ++i)
    {
        FaceElementTransformations *tr;
        // tr = mesh->GetInteriorFaceTransformations(i);
        tr = mesh->GetFaceElementTransformations(i);
        if (tr->Elem2No >= 0)
        {
            if ((find(cutelems_outer.begin(), cutelems_outer.end(), tr->Elem1No) != cutelems_outer.end()) &&
                (find(cutelems_outer.begin(), cutelems_outer.end(), tr->Elem2No) != cutelems_outer.end()))
            {
                // cout << "interior face is " << tr->Face->ElementNo << endl;
                // cout << tr->Elem1No << " , " << tr->Elem2No << endl;
                cutFaces_outer.push_back(tr->Face->ElementNo);
            }
        }
    }

    /// find the elements for which we don't need to solve
    std::vector<bool> EmbeddedElems;
    for (int i = 0; i < mesh->GetNE(); ++i)
    {
        if (insideBoundary<1>(mesh, i) == true)
        {
            EmbeddedElems.push_back(true);
        }
        else if (insideBoundary<2>(mesh, i) == true)
        {
            EmbeddedElems.push_back(true);
        }
        else
        {
            EmbeddedElems.push_back(false);
        }
    }
    vector<int> solidElems;
    solidElems.insert(solidElems.end(), embeddedelems_inner.begin(), embeddedelems_inner.end());
    solidElems.insert(solidElems.end(), embeddedelems_outer.begin(), embeddedelems_outer.end());

    cout << "solid elements " << endl;
    for (int k = 0; k < solidElems.size(); ++k)
    {
        cout << solidElems.at(k) << endl;
    }

    std::map<int, bool> immersedFaces;
    //cout << "immersed faces " << endl;
    for (int i = 0; i < mesh->GetNumFaces(); ++i)
    {
        FaceElementTransformations *tr;
        tr = mesh->GetInteriorFaceTransformations(i);
        if (tr != NULL)
        {
            if ((EmbeddedElems.at(tr->Elem1No) == true) && (EmbeddedElems.at(tr->Elem2No)) == false)
            {
                immersedFaces[tr->Face->ElementNo] = true;
                //cout << "face is " << tr->Face->ElementNo << " with elements " << tr->Elem1No << " , " << tr->Elem2No << endl;
            }
            else if ((EmbeddedElems.at(tr->Elem2No) == true) && (EmbeddedElems.at(tr->Elem1No)) == false)
            {
                immersedFaces[tr->Face->ElementNo] = true;
                // cout << "face is " << tr->Face->ElementNo << " with elements " << tr->Elem1No << " , " << tr->Elem2No << endl;
            }
            else if ((EmbeddedElems.at(tr->Elem2No) == true) && (EmbeddedElems.at(tr->Elem1No)) == true)
            {
                immersedFaces[tr->Face->ElementNo] = true;
                //cout << "face is " << tr->Face->ElementNo << " with elements " << tr->Elem1No << " , " << tr->Elem2No << endl;
            }
        }
    }
    /// Integration rule maps

    // for cut element
    std::map<int, IntegrationRule *> cutSquareIntRules;
    // for embedded boundary face elements
    std::map<int, IntegrationRule *> cutSegmentIntRules;
    // for cut interior/boundary faces
    std::map<int, IntegrationRule *> cutFaceIntRules;

    // for inner circle
    std::map<int, IntegrationRule *> cutSegmentIntRules_inner;
    // for cut interior faces
    std::map<int, IntegrationRule *> cutFaceIntRules_inner;
    // for cut boundary faces
    std::map<int, IntegrationRule *> cutBdrFaceIntRules_inner;

    // for outer circle, later to be appended with above maps accordingly
    std::map<int, IntegrationRule *> cutSquareIntRules_outer;
    // for embedded boundary face elements
    std::map<int, IntegrationRule *> cutSegmentIntRules_outer;
    // for cut interior/boundary faces
    std::map<int, IntegrationRule *> cutFaceIntRules_outer;

    int deg = min((order + 2) * (order + 2), 10);
    double inner_radius = 1.0;
    double outer_radius = 3.0;

    // int rule for inner circle elements
    GetCutElementIntRule<2, 1>(mesh, cutelems_inner, deg, inner_radius, cutSquareIntRules);
    GetCutSegmentIntRule<2, 1>(mesh, cutelems_inner, cutFaces_inner, deg, inner_radius, cutSegmentIntRules_inner,
                               cutFaceIntRules);
    GetCutBdrSegmentIntRule<2, 1>(mesh, cutelems_inner, cutBdrFaces_inner, deg, inner_radius,
                                  cutBdrFaceIntRules_inner);
    // int rule for outer circle elements
    GetCutElementIntRule<2, 2>(mesh, cutelems_outer, deg, outer_radius, cutSquareIntRules_outer);
    GetCutSegmentIntRule<2, 2>(mesh, cutelems_outer, cutFaces_outer, deg, outer_radius, cutSegmentIntRules_outer,
                               cutFaceIntRules_outer);

    cutSegmentIntRules.insert(cutSegmentIntRules_inner.begin(), cutSegmentIntRules_inner.end());
    cutSquareIntRules.insert(cutSquareIntRules_outer.begin(), cutSquareIntRules_outer.end());
    cutSegmentIntRules.insert(cutSegmentIntRules_outer.begin(), cutSegmentIntRules_outer.end());
    cutFaceIntRules.insert(cutFaceIntRules_outer.begin(), cutFaceIntRules_outer.end());

    // finite element collection
    FiniteElementCollection *fec = new DG_FECollection(order, dim);

    // finite element space
    FiniteElementSpace *fes = new FiniteElementSpace(mesh, fec, num_state,
                                                     Ordering::byVDIM);
    cout << "Number of finite element unknowns: "
         << fes->GetTrueVSize() << endl;
    FiniteElementSpace *fes_GD = new GalerkinDifference(mesh, fec, EmbeddedElems,
                                                        num_state, Ordering::byVDIM, order);
    cout << "Number of finite element unknowns in GD: "
         << fes_GD->GetTrueVSize() << endl;

    Vector qfs(dim + 2);
    getFreeStreamState<2, 0>(qfs);

    Array<int> bndry_marker_isentropic;
    bndry_marker_isentropic.Append(1);
    bndry_marker_isentropic.Append(0);
    bndry_marker_isentropic.Append(0);
    bndry_marker_isentropic.Append(1);


    /// nonlinearform
    NonlinearForm *res = new NonlinearForm(fes_GD);
    res->AddDomainIntegrator(new CutEulerDomainIntegrator<2>(num_state, cutSquareIntRules, EmbeddedElems, alpha));
    res->AddDomainIntegrator(new CutEulerVortexBoundaryIntegrator<2, 1, 0>(fec, num_state, qfs, cutSegmentIntRules_inner, alpha));
    res->AddDomainIntegrator(new CutEulerVortexBoundaryIntegrator<2, 2, 0>(fec, num_state, qfs, cutSegmentIntRules_outer, alpha));
    res->AddInteriorFaceIntegrator(new CutEulerFaceIntegrator<2>(fec, immersedFaces, cutFaceIntRules, 1.0, num_state, alpha));
    res->AddBdrFaceIntegrator(new CutEulerBoundaryIntegrator<2, 1, 0>(fec, cutBdrFaceIntRules_inner, EmbeddedElems,
                                                                      num_state, qfs, alpha),
                              bndry_marker_isentropic);

    // bilinear form
    BilinearForm *mass = new BilinearForm(fes);

    // set up the mass matrix
    mass->AddDomainIntegrator(new CutEulerMassIntegrator(cutSquareIntRules, EmbeddedElems, num_state));
    mass->Assemble();
    mass->Finalize();
    SparseMatrix &mass_old = mass->SpMat();
    SparseMatrix *cp = dynamic_cast<GalerkinDifference *>(fes_GD)->GetCP();
    SparseMatrix *p = RAP(*cp, mass_old, *cp);
    SparseMatrix &M = *p;

    /// grid function
    GridFunction u(fes);
    VectorFunctionCoefficient u0(num_state, uexact);
    GridFunction u_test(fes);
    u_test.ProjectCoefficient(u0);

    /// GD grid function
    CentGridFunction uc(fes_GD);
    uc.ProjectCoefficient(u0);
    fes_GD->GetProlongationMatrix()->Mult(uc, u);
    ofstream projection("initial_projection.vtk");
    projection.precision(14);
    mesh->PrintVTK(projection, 0);
    u.SaveVTK(projection, "projection", 0);
    projection.close();
    u_test -= u;
    cout << "After projection, the difference norm is " << u_test.Norml2() << '\n';
    ofstream proj_ofs("projection_error_cutGD.vtk");
    proj_ofs.precision(14);
    mesh->PrintVTK(proj_ofs, 0);
    u.SaveVTK(proj_ofs, "project_error", 0);
    proj_ofs.close();

    CentGridFunction residual(fes_GD);
    residual = 0.0;
    res->Mult(uc, residual);
    cout << "sum of residual " << residual.Sum() << endl;

    //residual.Print();
    // check the residual
    ofstream res_ofs("residual_cut_GD.vtk");
    res_ofs.precision(14);
    mesh->PrintVTK(res_ofs, 1);
    residual.SaveVTK(res_ofs, "Residual", 1);
    res_ofs.close();
    /// time-marching method
    std::unique_ptr<mfem::ODESolver> ode_solver;
    //ode_solver.reset(new RK4Solver);
    ode_solver.reset(new BackwardEulerSolver);
    cout << "ode_solver set " << endl;

    /// TimeDependentOperator
    unique_ptr<mfem::TimeDependentOperator> evolver(new mfem::EulerEvolver(M, res,
                                                                           0.0, TimeDependentOperator::Type::IMPLICIT));
    /// set up the evolver
    auto t = 0.0;
    evolver->SetTime(t);
    ode_solver->Init(*evolver);

    /// solve the ode problem
    double res_norm0 = calcResidualNorm(res, fes_GD, uc);
    double t_final = 1000;
    std::cout << "initial residual norm: " << res_norm0 << "\n";
    double dt_init = 1000.0;
    double dt_old;
    // initial l2_err
    double l2_err_init = calcCutConservativeVarsL2Error<2, 0>(uexact, &u, fes, EmbeddedElems,
                                                              cutSquareIntRules, num_state, 0);
    cout << "l2_err_init " << l2_err_init << endl;

    double dt = 0.0;
    double res_norm;
    int exponent = 2;
    res_norm = res_norm0;
    for (auto ti = 0; ti < 30000; ++ti)
    {
        /// calculate timestep
        dt_old = dt;
        dt = dt_init * pow(res_norm0 / res_norm, exponent);
        dt = max(dt, dt_old);
        // print iterations
        std::cout << "iter " << ti << ": time = " << t << ": dt = " << dt << endl;
        //   std::cout << " (" << round(100 * t / t_final) << "% complete)";
        std::cout << "residual norm: " << res_norm << "\n";

        if (res_norm <= 1e-11)
            break;

        if (isnan(res_norm))
            break;
        ode_solver->Step(uc, t, dt);
        res_norm = calcResidualNorm(res, fes_GD, uc);
    }
    fes_GD->GetProlongationMatrix()->Mult(uc, u);
    ofstream finalsol_ofs("final_sol_cut_GD.vtk");
    finalsol_ofs.precision(14);
    mesh->PrintVTK(finalsol_ofs, 1);
    u.SaveVTK(finalsol_ofs, "Solution", 1);
    finalsol_ofs.close();

    //calculate final solution error
    double l2_err_final = calcCutConservativeVarsL2Error<2, 0>(uexact, &u, fes, EmbeddedElems,
                                                               cutSquareIntRules, num_state, 0);
    cout << "l2_err_final " << l2_err_final << endl;
    // Free the used memory.
    delete res;
    delete mass;
    delete fes;
    delete fes_GD;
    delete fec;
    delete mesh;
    return 0;
} // main ends

// Exact solution; note that I reversed the flow direction to be clockwise, so
// the problem and mesh are consistent with the LPS paper (that is, because the
// triangles are subdivided from the quads using the opposite diagonal)
void uexact(const Vector &x, Vector &q)
{
    q.SetSize(4);
    Vector u(4);
    double ri = 1.0;
    double Mai = 0.5; //0.95
    double rhoi = 2.0;
    double prsi = 1.0 / euler::gamma;
    double rinv = ri / sqrt(x(0) * x(0) + x(1) * x(1));
    double rho = rhoi * pow(1.0 + 0.5 * euler::gami * Mai * Mai * (1.0 - rinv * rinv),
                            1.0 / euler::gami);
    //cout << "rho " << rho << endl;
    double Ma = sqrt((2.0 / euler::gami) * ((pow(rhoi / rho, euler::gami)) *
                                                (1.0 + 0.5 * euler::gami * Mai * Mai) -
                                            1.0));
    //cout << "Ma " << Ma << endl;
    double theta;
    if (x(0) > 1e-15)
    {
        theta = atan(x(1) / x(0));
    }
    else
    {
        theta = M_PI / 2.0;
    }
    double press = prsi * pow((1.0 + 0.5 * euler::gami * Mai * Mai) /
                                  (1.0 + 0.5 * euler::gami * Ma * Ma),
                              euler::gamma / euler::gami);
    double a = sqrt(euler::gamma * press / rho);

    u(0) = rho;
    u(1) = -rho * a * Ma * sin(theta);
    u(2) = rho * a * Ma * cos(theta);
    u(3) = press / euler::gami + 0.5 * rho * a * a * Ma * Ma;

    q = u;

    //    double mach_fs = 0.3;
    //    q(0) = 1.0;
    //    q(1) = q(0) * mach_fs; // ignore angle of attack
    //    q(2) = 0.0;
    //    q(3) = 1 / (euler::gamma * euler::gami) + 0.5 * mach_fs * mach_fs;
    //    if (((x(0) * x(0)) + (x(1) * x(1)) < 1.0) /*|| ((x(0) * x(0)) + (x(1) * x(1)) > 9.0)*/)
    //    {
    //        double mach_fs = 0.3;
    //        q(0) = 1.0;
    //        q(1) = q(0) * mach_fs; // ignore angle of attack
    //        q(2) = 0.0;
    //        q(3) = 1 / (euler::gamma * euler::gami) + 0.5 * mach_fs * mach_fs;
    //    }
}