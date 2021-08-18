#include "imex.hpp"

// Lagrange interpolating polynomials
// q=2
// (1-1/2(2+a))y1 + 1/2(2+a)y2
// q=3
// (-1-a+1/2 (1+a) (2+a)) y1 + (2+a+(-1-a) (2+a)) y2 + 1/2 (1+a) (2+a) y3
// q=4
// (1-(3 (2+a))/2+9/8 (4/3+a) (2+a)-9/16 (2/3+a) (4/3+a) (2+a)) y1 + 
//     ((3 (2+a))/2-9/4 (4/3+a) (2+a)+27/16 (2/3+a) (4/3+a) (2+a)) y2 +
//     (9/8 (4/3+a) (2+a)-27/16 (2/3+a) (4/3+a) (2+a)) y3+9/16 (2/3+a) (4/3+a) (2+a) y4
void InterpolateBDF(int q, double alpha, std::vector< Vector*> sols, Vector &x)
{
    double &a = alpha;
    if (q==2)
    {
        std::cout << "Not implemented for linear\n";
    }
    if (q==2)
    {
        Vector &y1 = *sols[0];
        Vector &y2 = *sols[1];
        double c1 = 1.-0.5*(2+a);
        double c2 = 0.5*(2+a);
        for (int i=0; i<x.Size(); i++)
        {
            x(i) = c1*y1(i) + c2*y2(i);
        }
    }
    else if (q==3)
    {
        Vector &y1 = *sols[0];
        Vector &y2 = *sols[1];
        Vector &y3 = *sols[2];
        double c1 = 0.5*(1+a)*(2+a) - (1+a);
        double c2 = (2+a)-(1+a)*(2+a);
        double c3 = 0.5*(1+a)*(2+a);
        for (int i=0; i<x.Size(); i++)
        {
            x(i) = c1*y1(i) + c2*y2(i) + c3*y3(i);
        }
    }
    else if (q==4)
    {
        Vector &y1 = *sols[0];
        Vector &y2 = *sols[1];
        Vector &y3 = *sols[2];
        Vector &y4 = *sols[3];
        double d43a = a + 4.0/3.0;
        double d23a = a + 2.0/3.0;
        double ap2 = a + 2;
        double c1 = (1.0 - 1.5*ap2 + 1.125*d43a*ap2 - 0.5625*d23a*d43a*ap2);
        double c2 = (1.5*ap2 - 2.25*d43a*ap2 + 1.6875*d23a*d43a*ap2);
        double c3 = (1.125*d43a*ap2 - 1.6875*d23a*d43a*ap2);
        double c4 = 0.5625*d23a*d43a*ap2;
        for (int i=0; i<x.Size(); i++)
        {
            x(i) = c1*y1(i) + c2*y2(i) + c3*y3(i) + c4*y4(i);
        }
    }
    else
    {
        mfem_warning("Only implemented up to 4th order.\n");
    }
}

IMEXRK::~IMEXRK()
{
    for (int i = 0; i < tableaux.s; i++)
    {
        if (imp_stages[i])
        {
            delete imp_stages[i];
        }
        if (exp_stages[i])
        {
            delete exp_stages[i];
        }
    }
}

void IMEXRK::Init(IMEXTimeDependentOperator &_imex)
{
    ODESolver::Init(_imex);
    imex = &_imex;
    exp_stages.resize(tableaux.s);
    imp_stages.resize(tableaux.s);

    // Only allocate first implicit stage vector if ESDIRK
    if (tableaux.esdirk)
    {
        imp_stages[0] = new Vector(imex->Width());
        *imp_stages[0] = 0.0;
    }
    else
    {
        imp_stages[0] = NULL;
    }
    exp_stages[0] = new Vector(imex->Width());
    *exp_stages[0] = 0.0;

    // Allocate interior implicit/explicit stages
    for (int i=1; i<(tableaux.s-1); i++)
    {
        imp_stages[i] = new Vector(imex->Width());
        *imp_stages[i] = 0.0;
        exp_stages[i] = new Vector(imex->Width());
        *exp_stages[i] = 0.0;
    }

    // Only allocate last explicit stage if necessary
    imp_stages[tableaux.s-1] = new Vector(imex->Width());
    *imp_stages[tableaux.s-1] = 0.0;
    if (!tableaux.stiffly_accurate)
    {
        exp_stages[tableaux.s-1] = new Vector(imex->Width());
        *exp_stages[tableaux.s-1] = 0.0;
    }
}

void IMEXRK::Step(Vector &x, double &t, double &dt)
{
    int s = tableaux.s;
    bool debug = false;

    // Apply first explicit stage
    imex->SetTime(t);
    Vector temp(x.Size());
    imex->ExplicitMult(x, temp);
    imex->MassInv(temp, (*exp_stages[0]) );

    // Apply first explicit stage to ESDIRK implicit schemes
    if (tableaux.esdirk)
    {
        imex->ImplicitMult(x, temp);
        imex->MassInv(temp, (*imp_stages[0]) );
    }

    // Loop over stages
    for (int i = 1; i < s; i++)
    {
        // Set time for this stage.
        // NOTE : assume same abscissa {c} for both schemes, so time is
        // the same for the ImplicitSolve() and ExplicitMult() in this stage.
        imex->SetTime(t + dt * tableaux.c0(i));

        // Add implicit stages to \hat{x}; first correct stage
        // vectors that were added during the previous stage,
        // then add new stage vector (after loop)
        for (int j = 0; j < i; j++)
        {
            // Add constant from this stage, subtract constant
            // from previous stage
            double c0 = dt * (tableaux.Ai(i, j) - tableaux.Ai(i - 1, j));

            // x += dt*(Ai_{i,j} - Ai_{i-1,j})*k_j, j<(i-1)
            if (std::abs(c0) > 1e-15)
            {
                x.Add(c0, (*imp_stages[j]) );
            }
        }

        // Add explicit stages to \hat{x}; first correct stage
        // vectors that were added during the previous stage,
        // then add new stage vector (after loop)
        for (int j = 0; j < (i - 1); j++)
        {
            // Add constant from this stage, subtract constant
            // from previous stage
            double c0 = dt * (tableaux.Ae(i, j) - tableaux.Ae(i - 1, j));

            // x += dt*(Ae_{i,j} - Ae_{i-1,j})*\hat{k}_j
            if (std::abs(c0) > 1e-15)
            {
                x.Add(c0, (*exp_stages[j]) );
            }
        }
        // x += dt*Ae_{i,i-1}*\hat{k}_{i}
        if (std::abs(tableaux.Ae(i, i - 1)) > 1e-15)
        {
            x.Add(dt * tableaux.Ae(i, i - 1), (*exp_stages[i - 1]) );
        }

        // Solve implicit equation for k_i, k_i = N_I(\hat{x}_i + dt*A_ii*k_i)
        if (std::abs(tableaux.Ai(i, i)) > 1e-15)
        {
            imex->ImplicitSolve(dt * tableaux.Ai(i, i), x, (*imp_stages[i]));
            x.Add(dt * tableaux.Ai(i, i), (*imp_stages[i]) );
        }

        // Solve explicit stage vector \hat{k}_{i+1} for next RK
        // stage unless this is last stage and we do not need
        // \hat{k}_{i+1} to form solution
        if ( i < (s - 1) || std::abs(tableaux.be(s - 1)) > 1e-15)
        {
            imex->ExplicitMult(x, temp );
            imex->MassInv(temp, (*exp_stages[i]) );
        }
    }

    if (!tableaux.stiffly_accurate)
    {
        // ----------------- Form solution -----------------
        // Add implicit stages to solution x; first correct stage
        // vectors that were added during the previous stage,
        // then add latest stage vector (after loop)
        for (int j = 0; j < s; j++)
        {
            // x += dt*(bi_{j} - Ai_{i-1,j})*k_j, j<(s-1)
            double c0 = dt * (tableaux.bi(j) - tableaux.Ai(s - 1, j));
            if (std::abs(c0) > 1e-15)
            {
                x.Add(c0, (*imp_stages[j]) );
            }
        }

        // Add explicit stages to solution x; first correct stage
        // vectors that were added during the previous stage,
        // then add new stage vector (after loop)
        for (int j = 0; j < (s - 1); j++)
        {
            double c0 = dt * (tableaux.be(j) - tableaux.Ae(s - 1, j));
            // x += dt*(be_{j} - Ae_{i,j})*\hat{k}_j
            if (std::abs(c0) > 1e-15)
            {
                x.Add(c0, (*exp_stages[j]) );
            }
        }
        if (std::abs(tableaux.be(s - 1)) > 1e-15)
        {
            x.Add(dt * tableaux.be(s - 1), (*exp_stages[s - 1]) );
        }
    }
    t += dt;
}

IMEXBDF::~IMEXBDF()
{
    for (int i = 0; i < sols.size(); i++)
    {
        delete sols[i];
    }
    for (int i = 0; i < exp_sols.size(); i++)
    {
        delete exp_sols[i];
    }
}

void IMEXBDF::Init(IMEXTimeDependentOperator &_imex)
{
    exp_nodes.resize(data.q);
    ODESolver::Init(_imex);
    imex = &_imex;
    if (data.GetID() < 10)
    {
        mfem_error("Invalid IMEXBDF data type.\n");
    }

    // Polynomial BDF
    if (data.shifted_nodes)
    {
        sols.resize(data.q);
        exp_sols.resize(data.q);
        for (int i = 0; i < (data.q); i++)
        {
            sols[i] = new Vector(imex->Width());
            exp_sols[i] = new Vector(imex->Width());
        }
    }
    // Classical BDF
    else
    {
        sols.resize(data.q - 1);
        exp_sols.resize(data.q - 1);
        for (int i = 0; i < (data.q - 1); i++)
        {
            sols[i] = new Vector(imex->Width());
            exp_sols[i] = new Vector(imex->Width());
        }
    }

    // Set zero previous solutions as initialized, construct
    // RK method to initialize additional steps at high-order
    // accuracy if necessary.
    initialized = 0;
    if (data.q == 2)
    {
        RKsolver = new IMEXRK(IMEXRKData::Type::IMEX222);
        RKsolver->Init(*imex);
    }
    else if (data.q == 3)
    {
        RKsolver = new IMEXRK(IMEXRKData::Type::IMEX443);
        RKsolver->Init(*imex);
    }
    else if (data.q > 3)
    {
        mfem_error("Currently only support up to third order IMEX-BDF.\n");
    }
}

void IMEXBDF::Step(Vector &x, double &t, double &dt)
{
    double r = dt / data.alpha;

    // Store initial condition from first time step and apply explicit
    // operator (not relevant for standard first order IMEX BDF)
    if (initialized == 0)
    {
        if (data.shifted_nodes || data.q > 1)
        {
            (*sols[0]) = x;
            // For shifted nodes, do not need to compute ExplicitMult here
            if(!data.shifted_nodes)
            {
                imex->SetTime(t);
                imex->ExplicitMult(x, *exp_sols[0]);
            }
            // Store quadrature node in time
            exp_nodes[0] = t;
        }
        initialized++;
        dt_prev = dt;
    }

    // Make sure use the same time step as previously
    if (std::abs(dt_prev - dt) > 1e-15)
    {
        std::cout << "dt = " << dt << ", dtold = " << dt_prev << "\n";
        mfem_error("Must use same dt for all time steps! "
            "Restart with new dt not implemented.\n");
    }

    // Initialize high order starting values using RK
    if (initialized < data.q)
    {
        // Take RK time step of appropriate order *on polynomial quadrature
        // nodes*. dt for this initialiation step is 
        //      dt0 = r * (z(i) - z(i-1))
        // for quadrature nodes z = {z1,...,zq}.
        double dt0 = r * (data.z0(initialized) - data.z0(initialized-1) );
        RKsolver->Step(x, t, dt0);

        // Store solution and apply explicit operator (need to store
        // one less solution for classical BDF (!data.shifted_nodes))
        if ( initialized < (data.q-1) || data.shifted_nodes )
        {
            *(sols[initialized]) = x;
            // For shifted nodes, do not need to compute ExplicitMult here
            if (!data.shifted_nodes) 
            {
                imex->SetTime(t);
                imex->ExplicitMult(x, *exp_sols[initialized]);
            }
        }
        // Store quadrature node in time
        exp_nodes[initialized] = t;
        
        initialized++;
        // Delete RK time stepper when we are done with it
        if (initialized == data.q)
        {
            delete RKsolver;
            RKsolver = NULL;
        }
    }
    // Typical BDF time steps once we have sufficient starting values
    else
    {
        // Separate Step function depending on whether the BDF interpolatory
        // points are shifted (Polynomial BDF) or overlap (classical BDF).
        if (data.shifted_nodes)
        {
            AlphaStep(x, t, r);
        }
        else
        {
            // Option to recompute explicit components for less memory use
            // This is done automatically for q=1 in ClassicalStep().
            if (!recompute_exp || data.q==1) 
            {
                ClassicalStep(x, t, r);
            }
            else
            {
                ClassicalStepNoStore(x, t, r);
            }
        }

        // Store quadrature nodes for next time step; all solutions
        // are simply shifted by += dt = r*alpha
        for (int i=0; i<data.q; i++)
        {
            exp_nodes[i] += dt;
        }
        t += dt;
    }
}

// Here, sols[.] stores solution values at previous time steps
void IMEXBDF::AlphaStep(Vector &x, double &t, double &r)
{
    // For IMEX schemes, compute explicit evaluations of stored solution
    for (int i = 0; i < data.q; i++)
    {
        imex->SetTime(exp_nodes[i]);
        imex->ExplicitMult(*(sols[i]), *(exp_sols[i]));
    }

    // Interpolate initial guess to x using previous solutions
    // TODO : check this is correct, see if can use extra previous solution
    //      before we replace with x.
    //      - Also, does this account for non uniform quadrature nodes if
    //      they are nonoverlapping? I.e., our nodes are separated by r,
    //      but the input/output by r*alpha.
    if (interpolate)
    {
        InterpolateBDF(data.q, data.alpha, sols, x);
    }

    // Apply mass matrix to stored solution values for construction
    // of right hand side
    {
        Vector temp(imex->Width());
        for (int i = 0; i < data.q; i++)
        {
            imex->MassMult(*(sols[i]), temp);
            *(sols[i]) = temp;
        }
    }

    // Construct right-hand side for implicit solve as linear
    // combination of mass matrix times previous solution values
    // and explicit part of operator evaluated at previous solution
    // values
    Vector *rhs = new Vector(imex->Width());
    (*rhs) = 0.0;
    for (int i = 0; i < data.q; i++)
    {
        if (std::abs(data.A(data.q - 1, i)) > 1e-15)
        {
            rhs->Add(data.A(data.q - 1, i), *(sols[i]));
        }
        // Add explicit components
        if (std::abs(r * data.Be(data.q - 1, i)) > 1e-15)
        {
            rhs->Add(r * data.Be(data.q - 1, i), *(exp_sols[i]));
        }
    }

    // Solve implicit equation for new solution
    imex->SetTime(t + data.alpha*r);
    double gamma = r * data.Bi(data.q - 1);
    imex->ImplicitSolve2(gamma, *rhs, x);

    // Evaluate implicit part of operator on new solution. We solved for
    // x such that
    //    M*x - gamma f(x) = rhs
    // Then, f(x) = (M*x - rhs) / gamma. f(x) is stored in place of rhs.
    {
        Vector x0(imex->Width());
        imex->MassMult(x, x0);
        (*rhs) -= x0;
        double scale = -1.0/gamma;
        (*rhs) *= scale;
    }

    // -------- Solve for solution at modified quadrature points -------- //
    // Copy previously stored solutions
    std::vector< Vector*> temp_sol(data.q);
    for (int i = 0; i < (data.q - 1); i++)
    {
        temp_sol[i] = new Vector(*(sols[i]));  // Copy constructor
    }
    temp_sol[data.q - 1] = sols[data.q - 1];

    // First apply diagonal scaling of A matrix and add implicit
    // part of new solution (doing this separate allows us to avoid
    // constructing an additional vector)
    for (int i = 0; i < (data.q - 1); i++)
    {
        (*sols[i]) *= data.A(i, i); // Diagonal component of A
        // Implicit part of updated solution
        if (std::abs(data.Bi(i))*r > 1e-15)
        {
            sols[i]->Add(r * data.Bi(i), *rhs);
        }
    }

    // Solve for updated solution at all other quadrature nodes.
    // Requires forming linear combination of currently stored
    // solutions and explicit componenets, then peforming mass inverse
    for (int i = 0; i < (data.q - 1); i++)
    {
        // Form linear combination of stored mass matrix times previous
        // solution values
        for (int j = 0; j < data.q; j++)
        {
            if (i == j)
            {
                continue;   // Already accounted for this one
            }
            else if (std::abs(data.A(i, j)) > 1e-15)
            {
                sols[i]->Add(data.A(i, j), *(temp_sol[j]) );
            }
        }

        // Add linear combination of explicit part of operator
        // evaluated at previous points
        for (int j = 0; j < data.q; j++)
        {
            if (std::abs(r * data.Be(i, j)) > 1e-15)
            {
                sols[i]->Add(r * data.Be(i, j), *(exp_sols[j]) );
            }
        }

        // Apply mass inverse, swap pointers so result is stored in sols[i]
        imex->MassInv(*sols[i], *rhs);
        Vector *temp_vec = sols[i];
        sols[i] = rhs;
        rhs = temp_vec;
        temp_vec = NULL;
    }

    // Update sols[q-1] with stored implicit solution
    (*sols[data.q - 1]) = x;

    // Clean up temp vectors and pointers
    for (int i = 0; i < (data.q - 1); i++)
    {
        delete temp_sol[i];
    }
    delete rhs;
}

// Classical IMEX-BDF step where we store the explicit component of
// previous solution values as well. Here, sols[.] stores mass matrix
// times solution values at previous time steps, for easier construction
// of right-hand side.
void IMEXBDF::ClassicalStep(Vector &x, double &t, double &r)
{
    // Construct right-hand side for implicit solve as linear
    // combination of previous solution values and explicit part
    // of operator evaluated at previous solution values. This
    // does not yet account for most recent solution, x.
    Vector rhs(imex->Width());
    rhs = 0.0;
    for (int i = 0; i < data.q - 1; i++)
    {
        double temp = data.A(data.q - 1, i);
        if (std::abs(temp) > 1e-15)
        {
            rhs.Add(temp, *(sols[i]));
        }
        // Add explicit component
        temp = data.Be(data.q - 1, i) * r;
        if (std::abs(temp) > 1e-15)
        {
            rhs.Add(temp, *(exp_sols[i]));
        }
    }

    if (data.q > 1)
    {
        // Shift q stored solution pointers back by one to add current
        // solution, x, in array entry [q-1] corresponding to most
        // recent time step.
        Vector *temp_vec = sols[0];
        for (int i = 0; i < (data.q - 2); i++)
        {
            sols[i] = sols[i + 1];
        }
        // Store mass-matrix times most recent solution value, x
        sols[data.q - 2] = temp_vec;
        temp_vec = NULL;
        imex->MassMult(x, *(sols[data.q - 2]) );

        // Shift q stored explicit component pointers back by one to
        // add current solution, x, in array entry [q-1] corresponding
        // to most recent time step.
        temp_vec = exp_sols[0];
        for (int i = 0; i < (data.q - 2); i++)
        {
            exp_sols[i] = exp_sols[i + 1];
        }
        // Compute and store explicit component of most recent
        // solution value, x
        exp_sols[data.q - 2] = temp_vec;
        temp_vec = NULL;
        imex->SetTime(t);
        imex->ExplicitMult(x, *(exp_sols[data.q - 2]));

        // Update RHS with most recent solution information
        double temp = data.A(data.q - 1, data.q - 1);
        if (std::abs(temp) > 1e-15)
        {
            rhs.Add(temp, *(sols[data.q - 2]));
        }
        // Add explicit component
        temp = data.Be(data.q - 1, data.q - 1) * r;
        if (std::abs(temp) > 1e-15)
        {
            rhs.Add(temp, *(exp_sols[data.q - 2]));
        }
    }
    // Special case for q=1
    else
    {
        Vector temp_vec(x.Size());
        imex->MassMult(x, temp_vec);

        // Update RHS with most recent solution information
        double temp = data.A(data.q - 1, data.q - 1);
        if (std::abs(temp) > 1e-15)
        {
            rhs.Add(temp, temp_vec);
        }

        // Compute and add explicit component
        imex->SetTime(t);
        imex->ExplicitMult(x, temp_vec);
        temp = data.Be(data.q - 1, data.q - 1) * r;
        if (std::abs(temp) > 1e-15)
        {
            rhs.Add(temp, temp_vec);
        }
    }

    // Interpolate initial guess to x using previous solutions
    // TODO : check that this is correct.
    if (interpolate)
    {
        Vector temp_vec(x.Size());
        InterpolateBDF(data.q, data.alpha, sols, temp_vec);
        imex->MassInv(temp_vec, x);
    }

    // Solve implicit equation for new solution
    double gamma = r * data.Bi(data.q - 1);
    imex->SetTime(t + data.alpha*r);
    imex->ImplicitSolve2(gamma, rhs, x);
}

// Classical IMEX-BDF step where we do not store the explicit component
// of previous solution values
void IMEXBDF::ClassicalStepNoStore(Vector &x, double &t, double &r)
{
    // Construct right-hand side for implicit solve as linear
    // combination of previous solution values and explicit part
    // of operator evaluated at previous solution values. This
    // does not yet account for most recent solution, x.
    Vector rhs(imex->Width());
    Vector temp_exp(imex->Width());
    rhs = 0.0;
    for (int i = 0; i < data.q - 1; i++)
    {
        double temp = data.A(data.q - 1, i);
        if (std::abs(temp) > 1e-15)
        {
            rhs.Add(temp, *(sols[i]));
        }
        // Add explicit component
        temp = data.Be(data.q - 1, i) * r;
        if (std::abs(temp) > 1e-15)
        {
            imex->SetTime(exp_nodes[i]);
            imex->ExplicitMult(*(sols[i]), temp_exp);
            rhs.Add(temp, temp_exp);
        }
    }

    // Shift q stored solution pointers back by one to add current
    // solution, x, in array entry [q-1] corresponding to most
    // recent time step.
    Vector *temp_vec = sols[0];
    for (int i = 0; i < (data.q - 2); i++)
    {
        sols[i] = sols[i + 1];
    }
    // Store mass-matrix times most recent solution value, x
    sols[data.q - 2] = temp_vec;
    temp_vec = NULL;
    imex->MassMult(x, *(sols[data.q - 2]) );

    // Update RHS with most recent solution information
    double temp = data.A(data.q - 1, data.q - 1);
    if (std::abs(temp) > 1e-15)
    {
        rhs.Add(temp, *(sols[data.q - 2]));
    }

    // Compute explicit component of most recent solution value,
    // x, and add to rhs
    imex->SetTime(t);
    imex->ExplicitMult(x, temp_exp);
    temp = data.Be(data.q - 1, data.q - 1) * r;
    if (std::abs(temp) > 1e-15)
    {
        rhs.Add(temp, temp_exp);
    }

    // TODO : add optional function to provide initial guess

    // Solve implicit equation for new solution
    double gamma = r * data.Bi(data.q - 1);
    imex->SetTime(t + data.alpha*r);
    imex->ImplicitSolve2(gamma, rhs, x);
}


/// Set data required by solvers
void IMEXRKData::SetExplicitData(DenseMatrix Ae_, Vector be_, Vector ce_)
{
    Ae = Ae_;
    be = be_;
    if (s > 0 && s != Ae.Size())
    {
        mfem_error("Implicit and Explicit Butcher tableaux must have same dimensions!\n");
    }
    s = Ae.Size();
    if (be.Size() != s || ce_.Size() != s)
    {
        mfem_error("Explicit Butcher tableaux dimensions inconsistent!\n");
    }
    // Ensure same abscissae for implicit and explicit schemes
    if (c0.Size() > 0)
    {
        bool same = true;
        for (int i = 0; i < c0.Size(); i++)
        {
            if (std::abs(c0(i) - ce_(i)) > 1e-15)
            {
                same = false;
                break;
            }
        }
        if (!same)
        {
            mfem_error("Must use same abscissae for implicit and explicit schemes!\n");
        }
    }
    c0 = ce_;
}

/// Set data required by solvers
void IMEXRKData::SetImplicitData(DenseMatrix Ai_, Vector bi_, Vector ci_, bool esdirk_)
{
    esdirk = esdirk_;
    Ai = Ai_;
    bi = bi_;
    if (s > 0 && s != Ai.Size())
    {
        mfem_error("Implicit and Explicit Butcher tableaux must have same dimensions!\n");
    }
    s = Ai.Size();
    if (bi.Size() != s || ci_.Size() != s)
    {
        mfem_error("Explicit Butcher tableaux dimensions inconsistent!\n");
    }
    // Ensure same abscissae for implicit and explicit schemes
    if (c0.Size() > 0)
    {
        bool same = true;
        for (int i = 0; i < c0.Size(); i++)
        {
            if (std::abs(c0(i) - ci_(i)) > 1e-15)
            {
                same = false;
                break;
            }
        }
        if (!same)
        {
            mfem_error("Must use same abscissae for implicit and explicit schemes!\n");
        }
    }
    c0 = ci_;
}

/// Initialize butcher arrays to size s and with zero entries
void IMEXRKData::InitData()
{
    Ai.SetSize(s);
    Ae.SetSize(s);
    bi.SetSize(s);
    be.SetSize(s);
    c0.SetSize(s);
    Ai = 0.0;
    Ae = 0.0;
    bi = 0.0;
    be = 0.0;
    c0 = 0.0;
}

void IMEXRKData::SetData()
{
    switch (ID)
    {
    case Type::IMEX111:
    {
        s = 2;
        esdirk = false;
        stiffly_accurate = true;
        InitData();
        Ae(1, 0) = 1.0;
        be(0) = 1.0;
        Ai(1, 1) = 1.0;
        bi(1) = 1.0;
        c0(1) = 1.0;
        break;
    }
    case Type::IMEX121:
    {
        s = 2;
        esdirk = false;
        stiffly_accurate = false;
        InitData();
        Ae(1, 0) = 1.0;
        be(1) = 1.0;
        Ai(1, 1) = 1.0;
        bi(1) = 1.0;
        c0(1) = 1.0;
        break;
    }
    case Type::IMEX122:
    {
        s = 2;
        esdirk = false;
        stiffly_accurate = false;
        InitData();
        Ae(1, 0) = 0.5;
        be(1) = 1.0;
        Ai(1, 1) = 0.5;
        bi(1) = 1.0;
        c0(1) = 0.5;
        break;
    }
    case Type::IMEX222:
    {
        s = 3;
        esdirk = false;
        stiffly_accurate = true;
        InitData();
        double gamma = (2.0 - std::sqrt(2.0)) / 2.0;
        double delta = 1 - 1.0 / (2 * gamma);
        Ae(1, 0) = gamma;
        Ae(2, 0) = delta;
        Ae(2, 1) = 1 - delta;
        be(0) = delta;
        be(1) = 1 - delta;
        Ai(1, 1) = gamma;
        Ai(2, 1) = 1 - gamma;
        Ai(2, 2) = gamma;
        bi(1) = 1 - gamma;
        bi(2) = gamma;
        c0(1) = gamma;
        c0(2) = 1;
        break;
    }
    case Type::IMEX232:
    {
        s = 3;
        esdirk = false;
        stiffly_accurate = false;
        InitData();
        double gamma = (2.0 - std::sqrt(2.0)) / 2.0;
        double delta = -2 * std::sqrt(2.0) / 3.0;
        Ae(1, 0) = gamma;
        Ae(2, 0) = delta;
        Ae(2, 1) = 1 - delta;
        be(1) = 1 - gamma;
        be(2) = gamma;
        Ai(1, 1) = gamma;
        Ai(2, 1) = 1 - gamma;
        Ai(2, 2) = gamma;
        bi(1) = 1 - gamma;
        bi(2) = gamma;
        c0(1) = gamma;
        c0(2) = 1;
        break;
    }
    case Type::IMEX233:
    {
        s = 3;
        esdirk = false;
        stiffly_accurate = false;
        InitData();
        double gamma = (3.0 + std::sqrt(3.0)) / 6.0;
        Ae(1, 0) = gamma;
        Ae(2, 0) = gamma - 1;
        Ae(2, 1) = 2 * (1 - gamma);
        be(1) = 0.5;
        be(2) = 0.5;
        Ai(1, 1) = gamma;
        Ai(2, 1) = 1 - 2 * gamma;
        Ai(2, 2) = gamma;
        bi(1) = 0.5;
        bi(2) = 0.5;
        c0(1) = gamma;
        c0(2) = 1 - gamma;
        break;
    }
    case Type::IMEX443:
    {
        s = 5;
        esdirk = false;
        stiffly_accurate = true;
        InitData();
        //
        Ae(1, 0) = 0.5;
        Ae(2, 0) = 11.0/18.0;
        Ae(2, 1) = 1.0/18.0;
        Ae(3, 0) = 5.0/6.0;
        Ae(3, 1) = -5.0/6.0;
        Ae(3, 2) = 0.5;
        Ae(4, 0) = 0.25;
        Ae(4, 1) = 1.75;
        Ae(4, 2) = 0.75;
        Ae(4, 3) = -1.75;
        //
        be(0) = 0.25;
        be(1) = 1.75;
        be(2) = 0.75;
        be(3) = -1.75;
        //
        Ai(1, 1) = 0.5;
        Ai(2, 1) = 1.0/6.0;
        Ai(2, 2) = 0.5;
        Ai(3, 1) = -0.5;
        Ai(3, 2) = 0.5;
        Ai(3, 3) = 0.5;
        Ai(4, 1) = 1.5;
        Ai(4, 2) = -1.5;
        Ai(4, 3) = 0.5;
        Ai(4, 4) = 0.5;
        //
        bi(1) = 1.5;
        bi(2) = -1.5;
        bi(3) = 0.5;
        bi(4) = 0.5;
        //
        c0(1) = 0.5;
        c0(2) = 2.0/3.0;
        c0(3) = 0.5;
        c0(4) = 1.0;
        break;
    }
    case Type::ARK43:
    {
        s = 4;
        esdirk = true;
        stiffly_accurate = false;
        InitData();
        Ae(1, 0) = 1767732205903. / 2027836641118.;
        Ae(2, 0) = 5535828885825. / 10492691773673.;
        Ae(2, 1) = 788022342437. / 10882634858940.;
        Ae(3, 0) = 6485989280629. / 16251701735622.;
        Ae(3, 1) = -4246266847089. / 9704473918619.;
        Ae(3, 2) = 10755448449292. / 10357097424841.;
        be(0) = 1471266399579. / 7840856788654.;
        be(1) = -4482444167858. / 7529755066697.;
        be(2) = 11266239266428. / 11593286722821.;
        be(3) = 1767732205903. / 4055673282236;

        Ai(1, 0) = 1767732205903. / 4055673282236.;
        Ai(1, 1) = 1767732205903. / 4055673282236.;
        Ai(2, 0) = 2746238789719. / 10658868560708.;
        Ai(2, 1) = -640167445237. / 6845629431997.;
        Ai(2, 2) = 1767732205903. / 4055673282236.;
        Ai(3, 0) = 1471266399579. / 7840856788654.;
        Ai(3, 1) = -4482444167858. / 7529759066697.;
        Ai(3, 2) = 11266239266428. / 11593286722821.;
        Ai(3, 3) = 1767732205903. / 4055673282236.;
        bi(0) = 1471266399579. / 7840856788654.;
        bi(1) = -4482444167858. / 7529755066697.;
        bi(2) = 11266239266428. / 11593286722821.;
        bi(3) = 1767732205903. / 4055673282236;

        c0(1) = 1767732205903. / 2027836641118.;
        c0(2) = 3. / 5.;
        c0(3) = 1.;
        break;
    }
    default: {
        mfem_error("RKData:: Invalid Runge Kutta type.\n");
    }
    }
}

/// Initialize arrays to size q and with zero entries
void BDFData::InitData()
{
    A.SetSize(q);
    Be.SetSize(q);
    Bi.SetSize(q);
    z0.SetSize(q);
    A = 0.0;
    Be = 0.0;
    Bi = 0.0;
}

void BDFData::SetData()
{
    switch (ID)
    {
    case Type::IMEX_BDF1:
    {
        q = 1;
        alpha = 1.0;
        InitData();
        A(0, 0) = 1.0;
        Bi(0) = 1.0;
        Be(0, 0) = 1.0;
        break;
    }
    case Type::IMEX_BDF2:
    {
        q = 2;
        if (alpha < 0)
        {
            alpha = 2.0 / (q - 1);
        }
        InitData();
        double &z = alpha;
        double denom = (4 * z + 4);
        A(0, 0) = (4 - z * z) / denom;
        A(0, 1) = z * (4 + z) / denom;
        A(1, 0) = -z * z / denom;
        A(1, 1) = (z + 2) * (z + 2) / denom;
        Bi(0) = 2 * z * (z - 2) / denom;
        Bi(1) = 2 * z * (z + 2) / denom;
        Be(0, 0) = -z * z * (z - 2) / denom;
        Be(0, 1) = z * (z * z - 4) / denom;
        Be(1, 0) = -z * z * (z + 2) / denom;
        Be(1, 1) = z * (z + 2) * (z + 2) / denom;
        break;
    }
    case Type::IMEX_BDF3:
    {
        q = 3;
        if (alpha < 0)
        {
            alpha = 2.0 / (q - 1);
        }
        InitData();
        double &z = alpha;
        double z2 = z * z;
        double z3 = z * z2;
        double z4 = z2 * z2;
        double denom = (6 * z2 + 12 * z + 4);
        A(0, 0) = (z4 + 2 * z3 - 11 * z2 + 4 * z + 4) / denom;
        A(0, 1) = -2 * (z * (z3 + 4 * z2 - 8 * z - 8)) / denom;
        A(0, 2) = (z * (z3 + 6 * z2 + z - 8)) / denom;
        A(1, 0) = (z * (z3 + 2 * z2 - 2 * z - 1)) / denom;
        A(1, 1) = -2 * (z4 + 4 * z3 + z2 - 4 * z - 2) / denom;
        A(1, 2) = (z * (z3 + 6 * z2 + 10 * z + 5)) / denom;
        A(2, 0) = (z2 * (z + 1) * (z + 1)) / denom;
        A(2, 1) = -2 * (z2 * (z + 2) * (z + 2)) / denom;
        A(2, 2) = ((z2 + 3 * z + 2) * (z2 + 3 * z + 2)) / denom;

        Bi(0) = 2 * (z * (z2 - 3 * z + 2)) / denom;
        Bi(1) = 2 * (z * (z2 - 1)) / denom;
        Bi(2) = 2 * (z * (z2 + 3 * z + 2)) / denom;

        Be(0, 0) = (z2 * (z3 - 2 * z2 - z + 2)) / denom;
        Be(0, 1) = 2 * (z2 * (-z3 + z2 + 4 * z - 4)) / denom;
        Be(0, 2) = (z * (z4 - 5 * z2 + 4)) / denom;
        Be(1, 0) = ((z - 1) * z2 * (z + 1) * (z + 1)) / denom;
        Be(1, 1) = 2 * (z2 * (-z3 - 2 * z2 + z + 2)) / denom;
        Be(1, 2) = (z * (z + 1) * (z + 1) * (z2 + z - 2)) / denom;
        Be(2, 0) = (z2 * (z + 1) * (z + 1) * (z + 2)) / denom;
        Be(2, 1) = -2 * (z2 * (z + 1) * (z + 2) * (z + 2)) / denom;
        Be(2, 2) = (z * (z2 + 3 * z + 2) * (z2 + 3 * z + 2)) / denom;
        break;
    }
    case Type::IMEX_BDF4:
    {
        q = 4;
        if (alpha < 0) 
        {
            alpha = 2.0 / (q - 1);
        }
        InitData();
        double &z = alpha;
        double z2 = alpha * alpha;
        double z3 = alpha * z2;
        double z4 = z2 * z2;
        double z5 = z3 * z2;
        double z6 = z3 * z3;
        double denom = (64 * (9 * z3 + 27 * z2 + 22 * z + 4));
        double c1 = (9 * z3 + 36 * z2 + 44 * z + 16);
        double c2 = (9 * z2 + 18 * z + 8);
        double c3 = (3 * z2 + 8 * z + 4);
        double c4 = (3 * z2 + 10 * z + 8);

        A(0, 0) = (-81 * z6 - 324 * z5 + 1476 * z4 - 288 * z3 - 1792 * z2 + 576 * z + 256) / denom;
        A(0, 1) = (27 * z * (9 * z5 + 48 * z4 - 128 * z3 - 128 * z2 + 208 * z + 64)) / denom;
        A(0, 2) = -(27 * z * (9 * z5 + 60 * z4 - 68 * z3 - 224 * z2 + 64 * z + 64)) / denom;
        A(0, 3) = (z * (81 * z5 + 648 * z4 + 144 * z3 - 1728 * z2 - 368 * z + 832)) / denom;
        A(1, 0) = -(z * (243 * z5 + 972 * z4 - 1188 * z3 - 1440 * z2 + 768 * z + 256)) / (3 * denom);
        A(1, 1) = (243 * z6 + 1296 * z5 - 216 * z4 - 2880 * z3 - 528 * z2 + 1024 * z + 256) / denom;
        A(1, 2) = -(3 * z * (81 * z5 + 540 * z4 + 468 * z3 - 864 * z2 - 1024 * z - 256)) / denom;
        A(1, 3) = (z * (243 * z5 + 1944 * z4 + 3672 * z3 + 1152 * z2 - 1680 * z - 896)) / (3 * denom);
        A(2, 0) = (z * (-243 * z5 - 972 * z4 - 756 * z3 + 288 * z2 + 384 * z + 64)) / (3 * denom);
        A(2, 1) = (3 * z * (81 * z5 + 432 * z4 + 576 * z3 - 304 * z - 64)) / denom;
        A(2, 2) = (-243 * z6 - 1620 * z5 - 3348 * z4 - 2016 * z3 + 960 * z2 + 1216 * z + 256) / denom;
        A(2, 3) = (z * (243 * z5 + 1944 * z4 + 5616 * z3 + 7488 * z2 + 4656 * z + 1088)) / (3 * denom);
        A(3, 0) = -(z2 * c2 * c2) / denom;
        A(3, 1) = (27 * z2 * c3 * c3) / denom;
        A(3, 2) = -(27 * z2 * c4 * c4) / denom;
        A(3, 3) = (c1 * c1) / denom;

        Bi(0) = 16 * (z * (9 * z3 - 36 * z2 + 44 * z - 16)) / denom;
        Bi(1) = 16 * (z * (27 * z3 - 36 * z2 - 12 * z + 16)) / (3 * denom);
        Bi(2) = 16 * (z * (27 * z3 + 36 * z2 - 12 * z - 16)) / (3 * denom);
        Bi(3) = 16 * (z * (9 * z3 + 36 * z2 + 44 * z + 16)) / denom;

        Be(0, 0) = (z2 * (-81 * z5 + 162 * z4 + 180 * z3 - 360 * z2 - 64 * z + 128)) / denom;
        Be(0, 1) = (9 * z2 * (27 * z5 - 36 * z4 - 120 * z3 + 160 * z2 + 48 * z - 64)) / denom;
        Be(0, 2) = -(9 * z2 * (27 * z5 - 18 * z4 - 156 * z3 + 104 * z2 + 192 * z - 128)) / denom;
        Be(0, 3) = (z * (81 * z6 - 504 * z4 + 784 * z2 - 256)) / denom;
        Be(1, 0) = -(z2 * (3 * z + 2) * (3 * z + 2) * (27 * z3 - 18 * z2 - 48 * z + 32)) / (3 * denom);
        Be(1, 1) = (3 * z2 * (3 * z + 2) * (3 * z + 2) * (9 * z3 - 28 * z + 16)) / denom;
        Be(1, 2) = -(3 * z2 * (81 * z5 + 162 * z4 - 180 * z3 - 360 * z2 + 64 * z + 128)) / denom;
        Be(1, 3) = (z * (3 * z + 2) * (3 * z + 2) * (27 * z4 + 36 * z3 - 84 * z2 - 64 * z + 64)) / (3 * denom);
        Be(2, 0) = -(z2 * (3 * z - 2) * c2 * c2) / (3 * denom);
        Be(2, 1) = (3 * z2 * (3 * z + 2) * (3 * z + 2) * (9 * z3 + 24 * z2 + 4 * z - 16)) / denom;
        Be(2, 2) = -(3 * z2 * (3 * z + 4) * (3 * z + 4) * (9 * z3 + 18 * z2 - 4 * z - 8)) / denom;
        Be(2, 3) = (z * (3 * z2 + 4 * z - 4) * c2 * c2) / (3 * denom);
        Be(3, 0) = -(z2 * (z + 2) * c2 * c2) / denom;
        Be(3, 1) = (9 * z2 * (3 * z + 4) * c3 * c3) / denom;
        Be(3, 2) = -(9 * z2 * (3 * z + 2) * c4 * c4) / denom;
        Be(3, 3) = (z * c1 * c1) / denom;
    }
    default: {
        mfem_error("RKData:: Invalid BDF type.\n");
    }
    }
    // Standard BDF method
    if (std::abs(2 - alpha * (q - 1)) < 1e-15 || q == 1)
    {
        shifted_nodes = false;
    }
    // Modified polynomial BDF
    else
    {
        shifted_nodes = true;
    }

    // Define quadrature nodes
    if (q != 1)
    {
        for (int i = 1; i <= q; i++)
        {
            z0(i - 1) = 2 * (i - 1) / (q - 1);
        }
    }
    else
    {
        z0(0) = 1;
    }
}
