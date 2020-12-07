/// Solve the steady isentropic vortex problem on cut-cell domain
// set this const expression to true in order to use entropy variables for state
constexpr bool entvar = false;
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <random>
#include "euler_integ_cut.hpp"
#include "evolver.hpp"
#include "cut_quad.hpp"
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

double getlsvalue(double ax, Vector &cent)
{

    double r = 1.0;
    double xc = 20.0;
    double yc = 20.0;
    double ay = ax / 1.0;
    return 1 * ((((cent(0) - xc) * (cent(0) - xc)) / (ax * ax)) + (((cent(1) - yc) * (cent(1) - yc)) / (ay * ay)) - (r * r));
}

double getlsvalue2(double ax, Vector &cent)
{

    double r = 1.0;
    double xc = 20.0;
    double yc = 20.0;
    double ay = ax / 5.0;
    return 1 * ((((cent(0) - xc) * (cent(0) - xc)) / (ax * ax)) + (((cent(1) - yc) * (cent(1) - yc)) / (ay * ay)) - (r * r));
}
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
/// function to calculate drag
double calcDrag(mfem::FiniteElementSpace *fes, mfem::GridFunction u,
                int num_state,
                std::map<int, IntegrationRule *> cutSegmentIntRules,
                double alpha)
{
    /// check initial drag value
    mfem::Vector drag_dir(2);

    drag_dir = 0.0;
    int iroll = 0;
    int ipitch = 1;
    double aoa_fs = 0.0;
    double mach_fs = 1.0;

    drag_dir(iroll) = cos(aoa_fs);
    drag_dir(ipitch) = sin(aoa_fs);
    drag_dir *= 1.0 / pow(mach_fs, 2.0); // to get non-dimensional Cd

    NonlinearForm *dragf = new NonlinearForm(fes);

    dragf->AddDomainIntegrator(
        new PressureForce<2, 1, entvar>(drag_dir, num_state,
                                        cutSegmentIntRules, alpha));

    double drag = dragf->GetEnergy(u);
    return drag;
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
    cout << "#elements in fes " << fes->GetNE() << endl;
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

double calcResidualNorm(NonlinearForm *res, FiniteElementSpace *fes, GridFunction &u)
{
    GridFunction residual(fes);
    residual = 0.0;
    res->Mult(u, residual);
    return residual.Norml2();
}

/// get freestream state values for the far-field bcs
template <int dim, bool entvar>
void getFreeStreamState(mfem::Vector &q_ref)
{
    double mach_fs = 0.5;
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
    double radius = 1.0;
    int ref_levels = -1;
    int ncr1 = -1;
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
                          40, 40, true);
    ofstream sol_ofv("square_mesh_ellipse.vtk");
    sol_ofv.precision(14);
    mesh->PrintVTK(sol_ofv, 0);
    sol_ofv.close();
    int dim = mesh->Dimension();

    /// find the elements to refine
    for (int k = 0; k < ncr1; ++k)
    {
        Array<int> marked_elements1;
        for (int i = 0; i < mesh->GetNE(); ++i)
        {
            Vector cent;
            GetElementCenter(mesh, i, cent);
            double lsv = getlsvalue(15.0, cent);
            if (lsv < 0.0)
            {
                marked_elements1.Append(i);
            }
        }

        mesh->GeneralRefinement(marked_elements1, 1);
    }

    for (int k = 0; k < ncr1 + 1; ++k)
    {
        Array<int> marked_elements1;
        for (int i = 0; i < mesh->GetNE(); ++i)
        {
            Vector cent;
            GetElementCenter(mesh, i, cent);
            double lsv = getlsvalue(8.0, cent);
            if (lsv < 0.0)
            {
                marked_elements1.Append(i);
            }
        }

        mesh->GeneralRefinement(marked_elements1, 1);
    }

    for (int k = 0; k < ncr1 + 3; ++k)
    {
        Array<int> marked_elements1;
        for (int i = 0; i < mesh->GetNE(); ++i)
        {
            Vector cent;
            GetElementCenter(mesh, i, cent);
            double rad = 0.5;
            double r = sqrt(((cent(0) - 19.5) * (cent(0) - 19.5)) + ((cent(1) - 20.0) * (cent(1) - 20.0)));
            //double lsv = 0.5 * (1.0 - cos(r));

            double lsvle = ((cent(0) - 19.5) * (cent(0) - 19.5)) + ((cent(1) - 20.0) * (cent(1) - 20.0)) - (rad * rad);
            double lsvte = ((cent(0) - 20.5) * (cent(0) - 20.5)) + ((cent(1) - 20.0) * (cent(1) - 20.0)) - (rad * rad);

            if (lsvle < 0.0 || lsvte < 0.0)
            {
                marked_elements1.Append(i);
            }
        }

        mesh->GeneralRefinement(marked_elements1, 1);
    }

    for (int k = 0; k < ncr1 + 1; ++k)
    {
        Array<int> marked_elements1;
        for (int i = 0; i < mesh->GetNE(); ++i)
        {
            Vector cent;
            GetElementCenter(mesh, i, cent);
            double rad = 0.2;
            double r = sqrt(((cent(0) - 19.5) * (cent(0) - 19.5)) + ((cent(1) - 20.0) * (cent(1) - 20.0)));
            double lsv = (0.5 * (1.0 - cos(M_PI * r)));
            double lsvle = ((cent(0) - 19.5) * (cent(0) - 19.5)) + ((cent(1) - 20.0) * (cent(1) - 20.0)) - (rad * rad);
            double lsvte = ((cent(0) - 20.5) * (cent(0) - 20.5)) + ((cent(1) - 20.0) * (cent(1) - 20.0)) - (rad * rad);

            if (lsvle < 0.0 || lsvte < 0.0)
            {
                marked_elements1.Append(i);
            }

            // if (lsv < 0.0 )
            // {
            //     marked_elements1.Append(i);
            // }
        }

        mesh->GeneralRefinement(marked_elements1, 1);
    }

    for (int l = 0; l < ref_levels; l++)
    {
        mesh->UniformRefinement();
    }

    mesh->Finalize();
    cout << "#elements after refinement " << mesh->GetNE() << endl;
    ofstream wmesh("square_mesh_vortex_nc.vtk");
    wmesh.precision(14);
    mesh->PrintVTK(wmesh, 0);
    wmesh.close();
    //find the elements cut by outer circle boundary
    vector<int> cutelems;
    vector<int> cutinteriorFaces;
    vector<int> cutFaces;

    /// find the elements for which we don't need to solve
    std::vector<bool> EmbeddedElems;
    vector<int> solidElems;
    for (int i = 0; i < mesh->GetNE(); ++i)
    {
        if (cutByGeom<3>(mesh, i) == true)
        {
            cutelems.push_back(i);
        }

        if (insideBoundary<3>(mesh, i) == true)
        {
            EmbeddedElems.push_back(true);
            solidElems.push_back(i);
        }
        else
        {
            EmbeddedElems.push_back(false);
        }
    }
    cout << "elements cut by  ellipse:  " << cutelems.size() << endl;
    for (int i = 0; i < cutelems.size(); ++i)
    {
        cout << cutelems.at(i) << endl;
    }

    cout << "solid elements: " << solidElems.size() << endl;
    for (int k = 0; k < solidElems.size(); ++k)
    {
        cout << solidElems.at(k) << endl;
    }
    /// find faces cut by inner circle
    for (int i = 0; i < mesh->GetNumFaces(); ++i)
    {
        FaceElementTransformations *tr;
        // tr = mesh->GetInteriorFaceTransformations(i);
        tr = mesh->GetFaceElementTransformations(i);
        if (tr->Elem2No >= 0)
        {
            if ((find(cutelems.begin(), cutelems.end(), tr->Elem1No) != cutelems.end()) &&
                (find(cutelems.begin(), cutelems.end(), tr->Elem2No) != cutelems.end()))
            {
                // cout << "interior face is " << tr->Face->ElementNo << endl;
                // cout << tr->Elem1No << " , " << tr->Elem2No << endl;
                cutFaces.push_back(tr->Face->ElementNo);
            }
        }
    }
    cout << "faces cut by ellipse " << cutFaces.size() << endl;
    // for (int k = 0; k < cutFaces.size(); ++k)
    // {
    //     cout << cutFaces.at(k) << endl;
    // }

    std::map<int, bool> immersedFaces;
    //cout << "immersed faces" << endl;
    for (int i = 0; i < mesh->GetNumFaces(); ++i)
    {
        FaceElementTransformations *tr;
        tr = mesh->GetInteriorFaceTransformations(i);
        if (tr != NULL)
        {
            if ((EmbeddedElems.at(tr->Elem1No) == true) && (EmbeddedElems.at(tr->Elem2No)) == false)
            {
                immersedFaces[tr->Face->ElementNo] = true;
                // cout << "face is " << tr->Face->ElementNo << " with elements " << tr->Elem1No << " , " << tr->Elem2No << endl;
            }
            else if ((EmbeddedElems.at(tr->Elem2No) == true) && (EmbeddedElems.at(tr->Elem1No)) == false)
            {
                immersedFaces[tr->Face->ElementNo] = true;
                //cout << "face is " << tr->Face->ElementNo << " with elements " << tr->Elem1No << " , " << tr->Elem2No << endl;
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

    int deg = min((order + 2) * (order + 2), 10);

    // int rule for cut elements
    GetCutElementIntRule<2, 3>(mesh, cutelems, deg, radius, cutSquareIntRules);
    GetCutSegmentIntRule<2, 3>(mesh, cutelems, cutFaces, deg, radius, cutSegmentIntRules,
                               cutFaceIntRules);

    // finite element collection
    FiniteElementCollection *fec = new DG_FECollection(order, dim);

    // finite element space
    FiniteElementSpace *fes = new FiniteElementSpace(mesh, fec, num_state,
                                                     Ordering::byVDIM);
    cout << "Number of finite element unknowns: "
         << fes->GetTrueVSize() << endl;

    Vector qfs(dim + 2);
    getFreeStreamState<2, 0>(qfs);

    /// check area
    cout << "--------- area and perimeter test --------- " << endl;
    GridFunction x(fes);
    NonlinearForm *a = new NonlinearForm(fes);
    a->AddDomainIntegrator(new CutEulerDomainIntegrator<2>(num_state, cutSquareIntRules, EmbeddedElems, alpha));
    double area;
    area = a->GetEnergy(x);
    double ar = 1600 - (M_PI * 0.5 * 0.05);
    cout << "correct area: " << ar << endl;
    cout << "calculated area: " << area << endl;
    NonlinearForm *po = new NonlinearForm(fes);

    po->AddDomainIntegrator(new CutEulerVortexBoundaryIntegrator<2, 3, 0>(fec, num_state, qfs, cutSegmentIntRules, alpha));

    double perim;
    perim = po->GetEnergy(x);

    double peri = 2.031987090050448;

    cout << "correct perimeter for ellipse: " << peri << endl;

    cout << "calculated perimeter : " << perim << endl;

    cout << "area err " << endl;
    cout << abs(area - ar) << endl;
    cout << "perimeter err " << endl;
    cout << abs(peri - perim) << endl;
    cout << "---------test done--------- " << endl;

    delete po;

    /// nonlinearform
    NonlinearForm *res = new NonlinearForm(fes);
    res->AddDomainIntegrator(new CutEulerDomainIntegrator<2>(num_state, cutSquareIntRules, EmbeddedElems, alpha));
    res->AddDomainIntegrator(new CutEulerVortexBoundaryIntegrator<2, 3, 0>(fec, num_state, qfs, cutSegmentIntRules, alpha));
    res->AddInteriorFaceIntegrator(new CutEulerFaceIntegrator<2>(fec, immersedFaces, cutFaceIntRules, 1.0, num_state, alpha));
    res->AddBdrFaceIntegrator(new EulerBoundaryIntegrator<2, 0>(fec, num_state, qfs, alpha));

    // check if the integrators are correct
    // double delta = 1e-5;

    // // initialize state; here we randomly perturb a constant state
    // GridFunction q(fes);
    // VectorFunctionCoefficient pert(num_state, randBaselinePert<2>);
    // q.ProjectCoefficient(pert);

    // // initialize the vector that the Jacobian multiplies
    // GridFunction v(fes);
    // VectorFunctionCoefficient v_rand(num_state, randState);
    // v.ProjectCoefficient(v_rand);

    // // evaluate the Jacobian and compute its product with v
    // Operator &Jac = res->GetGradient(q);
    // GridFunction jac_v(fes);
    // Jac.Mult(v, jac_v);

    // // now compute the finite-difference approximation...
    // GridFunction q_pert(q), r(fes), jac_v_fd(fes);
    // q_pert.Add(-delta, v);
    // res->Mult(q_pert, r);
    // q_pert.Add(2 * delta, v);
    // res->Mult(q_pert, jac_v_fd);
    // jac_v_fd -= r;
    // jac_v_fd /= (2 * delta);

    // for (int i = 0; i < jac_v.Size(); ++i)
    // {
    //     std::cout << std::abs(jac_v(i) - (jac_v_fd(i))) << "\n";
    //     MFEM_ASSERT(abs(jac_v(i) - (jac_v_fd(i))) < 1e-09, "jacobian is incorrect");
    // }

    /// bilinear form
    BilinearForm *mass = new BilinearForm(fes);

    // set up the mass matrix
    mass->AddDomainIntegrator(new CutEulerMassIntegrator(cutSquareIntRules, EmbeddedElems, num_state));
    mass->Assemble();
    mass->Finalize();

    /// grid function
    GridFunction u(fes);
    VectorFunctionCoefficient u0(num_state, uexact);
    u.ProjectCoefficient(u0);
    // cout << "exact solution size " <<  u.Size() << endl;
    // u.Print();
    GridFunction residual(fes);
    residual = 0.0;
    res->Mult(u, residual);
    cout << "sum of residual " << residual.Sum() << endl;
    std::cout << "initial residual norm: " << residual.Norml2() << "\n";
    // cout << "residual " << endl;
    // residual.Print();
    //residual.Print();
    // check the residual
    ofstream res_ofs("residual_cut.vtk");
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
    unique_ptr<mfem::TimeDependentOperator> evolver(new mfem::EulerEvolver(mass, res,
                                                                           0.0, TimeDependentOperator::Type::IMPLICIT));
    /// set up the evolver
    auto t = 0.0;
    evolver->SetTime(t);
    ode_solver->Init(*evolver);

    /// solve the ode problem
    double res_norm0 = calcResidualNorm(res, fes, u);
    double t_final = 1000;
    std::cout << "initial residual norm: " << res_norm0 << "\n";
    double dt_init = 1.0;
    double dt_old;
    // initial l2_err
    double l2_err_init = calcCutConservativeVarsL2Error<2, 0>(uexact, &u, fes, EmbeddedElems,
                                                              cutSquareIntRules, num_state, 0);
    cout << "l2_err_init " << l2_err_init << endl;

    double dt = 0.0;
    double res_norm;
    int exponent = 2;
    res_norm = res_norm0;
    for (auto ti = 0; ti < 40000; ++ti)
    {
        /// calculate timestep
        dt_old = dt;
        dt = dt_init * pow(res_norm0 / res_norm, exponent);
        dt = max(dt, dt_old);
        //dt = dt_init;
        // print iterations
        std::cout << "iter " << ti << ": time = " << t << ": dt = " << dt << endl;
        //   std::cout << " (" << round(100 * t / t_final) << "% complete)";
        if (res_norm <= 1e-11)
            break;

        if (isnan(res_norm))
            break;
        ode_solver->Step(u, t, dt);
        res_norm = calcResidualNorm(res, fes, u);
    }
    cout << "=========================================" << endl;
    std::cout << "final residual norm: " << res_norm << "\n";
    double drag = calcDrag(fes, u, num_state, cutSegmentIntRules, alpha);
    double drag_err = abs(drag - (-1 / 1.4));
    cout << "drag: " << drag << endl;
    cout << "drag_error: " << drag_err << endl;
    ofstream finalsol_ofs("final_sol_euler_vortex_cut.vtk");
    finalsol_ofs.precision(14);
    mesh->PrintVTK(finalsol_ofs, 3);
    u.SaveVTK(finalsol_ofs, "Solution", 3);
    finalsol_ofs.close();
    cout << "=========================================" << endl;

    // Free the used memory.
    delete res;
    delete mass;
    delete fes;
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
    double mach_fs = 0.5;
    q(0) = 1.0;
    q(1) = q(0) * mach_fs; // ignore angle of attack
    q(2) = 0.0;
    q(3) = 1 / (euler::gamma * euler::gami) + 0.5 * mach_fs * mach_fs;
}
