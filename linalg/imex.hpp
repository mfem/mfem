#ifndef MFEM_IMEX
#define MFEM_IMEX

#include "mfem.hpp"

using namespace mfem;
using namespace std;

/** Class for spatial discretizations of a PDE resulting in the time-dependent, 
    nonlinear set of ODEs with implicit-explicit additive partition
        M*du/dt = N_E(u,t) + N_I(u,t).
    MFEM typically treats time integration as 
        du/dt = F^{-1} G(u),
    Here F represents what MFEM calls the “implicit” part, and G represents the
    “explicit” part; in simpler terms, F is typically just a mass matrix. 

    For BDF schemes, the ImplicitSolve function is a bit different, and it is
    more natural to apply M and M^{-1} separate from the Mult functions, so we
    include MassMult and MassInv as functions to be provided, and do not include
    such actions in the Mult functions. */
class IMEXTimeDependentOperator : public TimeDependentOperator
{
protected:
    mutable Vector temp;    // Auxillary vector
    
public:
    // Sets linearly implicit to false by default
    IMEXTimeDependentOperator(int n, double t=0.0, Type type=EXPLICIT)
        : TimeDependentOperator(n, t, type) { };
    ~IMEXTimeDependentOperator() { };

    /** Apply action of implicit part of operator y <- N_I(x,y). For fully
        implicit schemes, this just corresponds to applying the time-dependent
        (nonlinear) operator.
        PREVIOUSLY CALLED ExplicitMult */
    virtual void ImplicitMult(const Vector &x, Vector &y) const = 0;

    /** Apply action of explicit part of operator y <- N_E(x,y) */
    virtual void ExplicitMult(const Vector &x, Vector &y) const { y = 0.0; };

    /** Solve k = f(x+dt*k) for stage k, where f() is the implicit part of
        the operator. Used in Runge-Kutta methods. */
    virtual void ImplicitSolve(const double dt, const Vector &x, Vector &k)
    { mfem_error("IMEXTimeDependentOperator::ImplicitSolve() is not overridden!"); };

    /** Solve M*x - dtf(x, t) = b for solution x, where f() is the implicit
        part of the operator. Used in BDF methods. */
    virtual void ImplicitSolve2(const double dt, const Vector &b, Vector &x)
    { mfem_error("IMEXTimeDependentOperator::ImplicitSolve2() is not overridden!"); };

    /** Apply action mass matrix, y = M*x. 
        If not re-implemented, this method simply generates an error.
        PREVIOUSLY CALLED ImplictMult */
    virtual void MassMult(const Vector &x, Vector &y) const = 0;

    /** Apply action of inverse of mass matrix, y = M^{-1}*x. 
        If not re-implemented, this method simply generates an error.
        NOTE : only necessary for PolyIMEX methods. */
    virtual void MassInv(const Vector &x, Vector &y) const = 0;
};

/** Class holding RK Butcher tableau, and associated data required by
    implicit and explicit splitting. */
class IMEXRKData 
{
public:
    // Implicit Runge Kutta type. Enumeration (s, \sigma, p):
    // - s = number of implicit stages
    // - \sigma = number of explicit stages
    // - p = order
    // In this notation, when s = \sigma, we satisfy (2.3)/(2.4) in
    // Ascher et al., and do not need to compute the final explicit
    // stage. This is represented in the stiffly_accurate boolean.
    enum Type { 
        IMEX111 = 111,
        IMEX121 = 121,
        IMEX122 = 122,
        IMEX222 = 222,
        IMEX232 = 232,
        IMEX233 = 233,
        IMEX443 = 443,
    // ARK ESDIRK-ERK schemes: enumeration (s,p), for total number of
    // stages s.
        ARK43 = -43
    };

    IMEXRKData() : s(-1) { };
    IMEXRKData(Type ID_) : ID(ID_) { SetData(); };
    ~IMEXRKData() { };
    
    /// Set explicit RK data
    void SetExplicitData(DenseMatrix Ae_, Vector be_, Vector ce_);
    /// Set implicit RK data
    void SetImplicitData(DenseMatrix Ai_, Vector bi_, Vector ci_, bool esdirk_=false);
    void SetID(Type ID_) { ID=ID_; SetData(); };

    bool esdirk;
    bool stiffly_accurate;
    bool use_final_exp_stage;
    int s;

    DenseMatrix Ai;     // Implicit Butcher matrix
    Vector bi;          // Implicit Butcher tableau weights
    DenseMatrix Ae;     // Explicit Butcher matrix
    Vector be;          // Explicit Butcher tableau weights
    Vector c0;          // Butcher tableau nodes (same for implicit and explicit!)

private:    
    Type ID;
    void SetData();
    void InitData();
};

/** Class for two-part additive IMEX RK method, where explicit and implicit
    stage vectors are stored. Assume same abscissae, {c}, for both schemes.
    Butcher Data must be provided either in a custom IMEXRKData object, or
    using the IMEXRKData::Type for predefined tableaux. */
class IMEXRK : public ODESolver
{
protected:
    IMEXRKData tableaux;
    std::vector< Vector *> exp_stages;
    std::vector< Vector *> imp_stages;
    IMEXTimeDependentOperator *imex;    // Spatial discretization. 

public:
    IMEXRK(IMEXRKData tableaux_) : ODESolver(), tableaux(tableaux_) { };
    IMEXRK(IMEXRKData::Type type_) : ODESolver(), tableaux(type_) { };
    ~IMEXRK();
    void Init(IMEXTimeDependentOperator &_imex);
    void Step(Vector &x, double &t, double &dt) override;
};

/** Class holding BDF integrator data. Setting alpha < 0 (the default
    constructor) defines alpha = 2/(q-1), corresponding to classical BDF
    of order q. */
class BDFData 
{
public:

    enum Type {
        BDF1 = 01, BDF2 = 02, BDF3 = 03, BDF4 = 04,
        IMEX_BDF1 = 11, IMEX_BDF2 = 12, IMEX_BDF3 = 13,
        IMEX_BDF4 = 14
    };

    BDFData() { };
    BDFData(Type ID_, double alpha_=-1) : ID(ID_), alpha(alpha_) { SetData(); };
    ~BDFData() { };

    int GetID() { return static_cast<int>(ID); };
    void SetID(Type ID_, double alpha_=-1) {
        ID=ID_;
        alpha = alpha_;
        SetData();
    };
    void Print() {
        std::cout << "q     = " << q << "\n";
        std::cout << "alpha = " << alpha << "\n";
        std::cout << "A:\n";
        A.PrintMatlab();
        std::cout << "Be:\n";
        Be.PrintMatlab();
        std::cout << "Bi:\n";
        Bi.Print();
        std::cout << "z:\n";
        z0.Print();
    };

    double alpha;
    int q;              // Number of previous values stored
    bool shifted_nodes; // false = clssical BDF, true = Polynomial BDF w/ shifted nodes
    DenseMatrix A;      // Previous solution coefficients
    Vector Bi;          // Implicit coefficients
    DenseMatrix Be;     // Explicit coefficients
    Vector z0;


private:    
    Type ID;
    void SetData();
    void InitData();
};


/** Class for IMEX-BDF methods, including classical IMEX-BDF and IMEX-
    Polynomial-BDF (IMEX-PBDF). IMEX-PBDF methods have an additional 
    alpha parameter, where larger alpha leads to smaller stability
    regions and a smaller leading accuracy constant, while smaller
    alpha leads to larger stabiltiy regions and a larger accuracy
    constant. For classical methods, there are two implementations:
        - ClassicalStep() stores previous solutions and the explicit
        part of the operator evaluated on the solution, and
        - ClassicalStepNoStore() does not store the explicit 
        component, but must re-evaluate q times during each time
        step.
    This option can be set via the recompute_exp input. The type of
    scheme must be set through the BDFData structure or BDFData::Type.
        There is also an option to use pointwise Lagrange interpolating 
    polynomials to provide an initial guess for the ImplicitSolve. This
    is set via InterpolateGuess(). This option is only implemented for
    PBDF. */
class IMEXBDF : public ODESolver
{
private:
    BDFData data;
    bool recompute_exp;
    bool interpolate;
    int initialized;
    double dt_prev;
    std::vector< Vector*> sols;
    std::vector< Vector*> exp_sols;
    IMEXTimeDependentOperator *imex;    // Spatial discretization
    IMEXRK *RKsolver;
    std::vector<double> exp_nodes;

    void AlphaStep(Vector &x, double &t, double &dt);
    void ClassicalStep(Vector &x, double &t, double &dt);
    void ClassicalStepNoStore(Vector &x, double &t, double &dt);

public:
    IMEXBDF(BDFData data_, bool recompute_exp_=false) :
      ODESolver(), data(data_), recompute_exp(recompute_exp_),
      interpolate(false) { };
    IMEXBDF(BDFData::Type scheme, bool recompute_exp_=false) :
      ODESolver(), recompute_exp(recompute_exp_), interpolate(false)
      { data.SetID(scheme); };
    IMEXBDF(BDFData::Type scheme, double alpha) :
      ODESolver(), interpolate(false), recompute_exp(false)
      { data.SetID(scheme, alpha); };
    ~IMEXBDF();

    void Init(IMEXTimeDependentOperator &_imex);
    void Step(Vector &x, double &t, double &dt);
    void InterpolateGuess() {interpolate = true; };
};

#endif