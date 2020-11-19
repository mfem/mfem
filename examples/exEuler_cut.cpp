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

template <int N>
struct circle
{
    double xscale;
    double yscale;
    double xmin;
    double ymin;
    double radius;
    template <typename T>
    T operator()(const blitz::TinyVector<T, N> &x) const
    {
        // level-set function to work in physical space
        // return -1 * (((x[0] - 5) * (x[0] - 5)) +
        //               ((x[1]- 5) * (x[1] - 5)) - (0.5 * 0.5));
        // level-set function for reference elements
        return -1 * ((((x[0] * xscale) + xmin - 0.5) * ((x[0] * xscale) + xmin - 0.5)) +
                     (((x[1] * yscale) + ymin - 0.5) * ((x[1] * yscale) + ymin - 0.5)) - (radius * radius));
    }
    template <typename T>
    blitz::TinyVector<T, N> grad(const blitz::TinyVector<T, N> &x) const
    {
        // return blitz::TinyVector<T, N>(-1 * (2.0 * (x(0) - 5)), -1 * (2.0 * (x(1) - 5)));
        return blitz::TinyVector<T, N>(-1 * (2.0 * xscale * ((x(0) * xscale) + xmin - 0.5)),
                                       -1 * (2.0 * yscale * ((x(1) * yscale) + ymin - 0.5)));
    }
};

int main(int argc, char *argv[])
{
    // 1. Parse command-line options.
    int order = 1;
    int N = 5;
    bool static_cond = false;
    bool pa = false;
    const char *device_config = "cpu";
    bool visualization = true;
    double sigma = -1.0;
    double kappa = 50.0;
    double cutsize;
    double radius = 1.0;
    /// number of state variables
    int num_state = 4;
    double alpha = 1.0;
    OptionsParser args(argc, argv);
    args.AddOption(&order, "-o", "--order",
                   "Finite element order (polynomial degree) or -1 for"
                   " isoparametric space.");
    args.AddOption(&N, "-n", "--#elements",
                   "number of mesh elements.");
    args.AddOption(&radius, "-r", "--radius",
                   "radius of circle.");
    args.Parse();
    if (!args.Good())
    {
        args.PrintUsage(cout);
        return 1;
    }
    if (kappa < 0)
    {
        kappa = (order + 1) * (order + 1);
    }
    args.PrintOptions(cout);

    Mesh *mesh = new Mesh(N, N, Element::QUADRILATERAL, true,
                          3, 3, true);
    ofstream sol_ofv("square_mesh_vortex.vtk");
    sol_ofv.precision(14);
    mesh->PrintVTK(sol_ofv, 0);

    //find the elements cut by inner circle boundary
    vector<int> cutelems_inner;
    vector<int> embeddedelems_inner;
    vector<int> cutFaces_inner;

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
                cout << "interior face is " << tr->Face->ElementNo << endl;
                cout << tr->Elem1No << " , " << tr->Elem2No << endl;
                cutFaces_inner.push_back(tr->Face->ElementNo);
            }
        }
        if (tr->Elem2No < 0)
        {
            if (find(cutelems_inner.begin(), cutelems_inner.end(), tr->Elem1No) != cutelems_inner.end())
            {
                cutFaces_inner.push_back(tr->Face->ElementNo);
                cout << "boundary face is " << tr->Face->ElementNo << endl;
                cout << tr->Elem1No << endl;
            }
        }
    }
    cout << "faces cut by inner circle " << endl;
    for (int k = 0; k < cutFaces_inner.size(); ++k)
    {
        cout << cutFaces_inner.at(k) << endl;
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

    int dim = mesh->Dimension();

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
                cout << "interior face is " << tr->Face->ElementNo << endl;
                cout << tr->Elem1No << " , " << tr->Elem2No << endl;
                cutFaces_outer.push_back(tr->Face->ElementNo);
            }
        }
// the boundary faces are not required for outer circle
#if 0
       if (tr->Elem2No < 0)
       {
           if (find(cutelems_outer.begin(), cutelems_outer.end(), tr->Elem1No) != cutelems_outer.end())
           {
               cutFaces_outer.push_back(tr->Face->ElementNo);
               cout << "boundary face is " << tr->Face->ElementNo << endl;
               cout << tr->Elem1No << endl;
           }
       }
#endif
    }

    cout << "faces cut by outer circle " << cutFaces_outer.size() << endl;
    for (int k = 0; k < cutFaces_outer.size(); ++k)
    {
        cout << cutFaces_outer.at(k) << endl;
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

    cout << "cut faces " << endl;
    for (int k = 0; k < cutFaces_outer.size(); ++k)
    {
        cout << cutFaces_outer.at(k) << endl;
    }

    std::map<int, bool> immersedFaces;
    cout << "immersed faces " << endl;
    for (int i = 0; i < mesh->GetNumFaces(); ++i)
    {
        FaceElementTransformations *tr;
        tr = mesh->GetInteriorFaceTransformations(i);
        if (tr != NULL)
        {
            if ((EmbeddedElems.at(tr->Elem1No) == true) && (EmbeddedElems.at(tr->Elem2No)) == false)
            {
                immersedFaces[tr->Face->ElementNo] = true;
                cout << "face is " << tr->Face->ElementNo << " with elements " << tr->Elem1No << " , " << tr->Elem2No << endl;
            }
            else if ((EmbeddedElems.at(tr->Elem2No) == true) && (EmbeddedElems.at(tr->Elem1No)) == false)
            {
                immersedFaces[tr->Face->ElementNo] = true;
                cout << "face is " << tr->Face->ElementNo << " with elements " << tr->Elem1No << " , " << tr->Elem2No << endl;
            }
            else if ((EmbeddedElems.at(tr->Elem2No) == true) && (EmbeddedElems.at(tr->Elem1No)) == true)
            {
                immersedFaces[tr->Face->ElementNo] = true;
                cout << "face is " << tr->Face->ElementNo << " with elements " << tr->Elem1No << " , " << tr->Elem2No << endl;
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
    // for cut interior/boundary faces
    std::map<int, IntegrationRule *> cutFaceIntRules_inner;

    // for outer circle, later to be appended with above maps accordingly
    std::map<int, IntegrationRule *> cutSquareIntRules_outer;
    // for embedded boundary face elements
    std::map<int, IntegrationRule *> cutSegmentIntRules_outer;
    // for cut interior/boundary faces
    std::map<int, IntegrationRule *> cutFaceIntRules_outer;

    int deg = order + 1;
    double inner_radius = 1.0;
    double outer_radius = 3.0;

    // int rule for inner circle elements
    GetCutElementIntRule<2, 1>(mesh, cutelems_inner, deg, inner_radius, cutSquareIntRules);
    GetCutSegmentIntRule<2, 1>(mesh, cutelems_inner, cutFaces_inner, deg, inner_radius, cutSegmentIntRules_inner,
                               cutFaceIntRules);

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
    
    Vector qfs(dim + 2);
    getFreeStreamState<2, 0>(qfs);
    /// nonlinearform
    NonlinearForm *res = new NonlinearForm(fes);
    res->AddDomainIntegrator(new CutEulerDomainIntegrator<2>(num_state, cutSquareIntRules, EmbeddedElems, alpha));
    res->AddDomainIntegrator(new CutEulerBoundaryIntegrator<2, 1, 0>(fec, num_state, qfs, cutSegmentIntRules_inner, alpha));
    res->AddDomainIntegrator(new CutEulerBoundaryIntegrator<2, 2, 0>(fec, num_state, qfs, cutSegmentIntRules_outer, alpha));
    res->AddInteriorFaceIntegrator(new CutEulerFaceIntegrator<2>(fec, immersedFaces, cutFaceIntRules, 1.0, num_state, alpha));
    /// check if the integrators are correct
    double delta = 1e-5;

   // initialize state; here we randomly perturb a constant state
   GridFunction q(fes);
   VectorFunctionCoefficient pert(num_state, randBaselinePert<2>);
   q.ProjectCoefficient(pert);

   // initialize the vector that the Jacobian multiplies
   GridFunction v(fes);
   VectorFunctionCoefficient v_rand(num_state, randState);
   v.ProjectCoefficient(v_rand);

   // evaluate the Jacobian and compute its product with v
   Operator &Jac = res->GetGradient(q);
   GridFunction jac_v(fes);
   Jac.Mult(v, jac_v);

   // now compute the finite-difference approximation...
   GridFunction q_pert(q), r(fes), jac_v_fd(fes);
   q_pert.Add(-delta, v);
   res->Mult(q_pert, r);
   q_pert.Add(2 * delta, v);
   res->Mult(q_pert, jac_v_fd);
   jac_v_fd -= r;
   jac_v_fd /= (2 * delta);

   for (int i = 0; i < jac_v.Size(); ++i)
   {
      std::cout << std::abs(jac_v(i) - (jac_v_fd(i))) << "\n";
      MFEM_ASSERT(abs(jac_v(i) - (jac_v_fd(i))) < 1e-09, "jacobian is incorrect");
   }

} // main ends

// back-up code
#if 0
    cout << "boundary faces " << endl;
    /// find boundary faces cut by vortex 
    for (int ib=0; ib<mesh->GetNBE(); ++ib)
    {
       FaceElementTransformations *trans;
       trans = mesh->GetFaceElementTransformations(ib);
       if (trans->Elem2No < 0)
       {
           if (find(cutelems_inner.begin(), cutelems_inner.end(), trans->Elem1No) != cutelems_inner.end())
           {
               cutFaces.push_back(trans->Face->ElementNo);
               cout << "face is " << trans->Face->ElementNo << endl;
               cout << trans->Elem1No  << endl;
           }
       }
    }
   // cut faces
   cout << "faces cut by inner circle:  " << endl;
   for (int i = 0; i < cutFaces.size(); ++i)
   {
       FaceElementTransformations *tr, *trans;
       int fid = cutFaces.at(i);
       tr = mesh->GetInteriorFaceTransformations(fid);
       if (tr != NULL)
       {
           cout << "interior face elements for Face " << tr->Face->ElementNo << " : " << tr->Elem1No << " , " << tr->Elem2No << endl;
       }
       else
       {
           trans = mesh->GetBdrFaceTransformations(fid);
           cout << "boundary face element for Face " << trans->Face->ElementNo << " : " << trans->Elem1No << endl;
       }
   }
   cout << "boundary faces cut by inner circle:  " << endl;
   for (int i = 0; i < cutboundaryFaces.size(); ++i)
   { 
       FaceElementTransformations *tr;
       int fid = cutboundaryFaces.at(i);
       tr = mesh->GetBdrFaceTransformations(fid);
       if (tr !=NULL)
       {
           cout << "boundary face element for Face "  << tr->Face->ElementNo << " : " << tr->Elem1No  << endl;
       }
   }
#endif
