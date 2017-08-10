//                               ETHOS Example - Mesh optimizer
//
// Compile with: make exsm
//
// Sample runs:  exsm -m blade.mesh     with Ideal target and metric 2
//               exsm -m tipton.mesh    with Ideal equal size target and metric 9
//
// Description: This example code performs mesh optimization using Target-matrix
//              optimization paradigm coupled with Variational Minimization. The
//              smoother is based on a combination of target element and quality metric
//              that the users chooses, which in-turn determines which features of the
//              mesh - size, quality and/or orientation - are optimized.


#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <limits>
#include <unistd.h>

using namespace mfem;

using namespace std;

class RelaxedNewtonSolver : public NewtonSolver
{
private:
    // Quadrature points that are checked for negative Jacobians etc.
    const IntegrationRule &ir;
    
public:
    RelaxedNewtonSolver(const IntegrationRule &irule) : ir(irule) { }
    
#ifdef MFEM_USE_MPI
    RelaxedNewtonSolver(const IntegrationRule &irule, MPI_Comm _comm)
    : NewtonSolver(_comm), ir(irule) { }
#endif
    
    virtual double ComputeScalingFactor(const Vector &x, const Vector &c) const;
};


double RelaxedNewtonSolver::ComputeScalingFactor(const Vector &x,
                                                 const Vector &c) const
{
    
    const NonlinearForm *nlf = dynamic_cast<const NonlinearForm *>(oper);
    
    const int NE = nlf->FESpace()->GetMesh()->GetNE();
    
    Vector xsav = x; //create a copy of x
    int passchk = 0;
    int iters = 0;
    double alpha = 1.;
    int jachk = 1;
    double initenergy = 0.0;
    double finenergy = 0.0;
    double nanchk;
    
    
    initenergy =nlf->GetEnergy(x);
    
    while (passchk !=1 && iters < 20)
    {
        iters += 1;
        jachk = 1;
        add (x,-alpha,c,xsav);
        
        GridFunction x_gf(nlf->FESpace());
        x_gf = xsav;
        
        finenergy = nlf->GetEnergy(xsav);
        nanchk = isnan(finenergy);
        for (int i = 0; i < NE; i++)
        {
            const FiniteElement &fe = *x_gf.FESpace()->GetFE(i);
            const int dim = fe.GetDim(), nsp = ir.GetNPoints(),
            dof = fe.GetDof();
            
            DenseMatrix Jtr(dim);
            
            DenseMatrix dshape(dof, dim), pos(dof, dim);
            Array<int> xdofs(dof * dim);
            Vector posV(pos.Data(), dof * dim);
            
            x_gf.FESpace()->GetElementVDofs(i, xdofs);
            x_gf.GetSubVector(xdofs, posV);
            for (int j = 0; j < nsp; j++)
            {
                fe.CalcDShape(ir.IntPoint(j), dshape);
                MultAtB(pos, dshape, Jtr);
                double det = Jtr.Det();
                if (det<=0.)
                {
                    jachk *= 0;
                }
            }
        }
        
        if (finenergy>1.0*initenergy || nanchk!=0 || jachk==0)
        {
            passchk = 0;
            alpha *= 0.5;
        }
        else
        {
            passchk = 1;
        }
    }
    
    cout << initenergy << " " << finenergy <<  " energy value before and after newton iteration\n";
    cout << "alpha value is " << alpha << " \n";
    
    if (passchk==0) { alpha =0; }
    return alpha;
}

class DescentNewtonSolver : public NewtonSolver
{
private:
    // Quadrature points that are checked for negative Jacobians etc.
    const IntegrationRule &ir;
    
public:
    DescentNewtonSolver(const IntegrationRule &irule) : ir(irule) { }
    
#ifdef MFEM_USE_MPI
    DescentNewtonSolver(const IntegrationRule &irule, MPI_Comm _comm)
    : NewtonSolver(_comm), ir(irule) { }
#endif
    
    virtual double ComputeScalingFactor(const Vector &x, const Vector &c) const;
};

double DescentNewtonSolver::ComputeScalingFactor(const Vector &x,
                                                 const Vector &c) const
{
    const NonlinearForm *nlf = dynamic_cast<const NonlinearForm *>(oper);
    
    const int NE = nlf->FESpace()->GetMesh()->GetNE();
    
    Vector xsav = x;
    int passchk = 0;
    int iters = 0;
    double alpha = 1.;
    int jachk = 1;
    double initenergy = 0.0;
    double finenergy = 0.0;
    int nanchk = 0;
    
    GridFunction x_gf(nlf->FESpace());
    x_gf = xsav;
    
    double tauval = 1e+6;
    for (int i = 0; i < NE; i++)
    {
        const FiniteElement &fe = *x_gf.FESpace()->GetFE(i);
        const int dim = fe.GetDim(), nsp = ir.GetNPoints(),
        dof = fe.GetDof();
        DenseMatrix Jtr(dim);
        const GridFunction *nds;
        DenseMatrix dshape(dof, dim), pos(dof, dim);
        Array<int> xdofs(dof * dim);
        Vector posV(pos.Data(), dof * dim);
        x_gf.FESpace()->GetElementVDofs(i, xdofs);
        x_gf.GetSubVector(xdofs, posV);
        for (int j = 0; j < nsp; j++)
        {
            fe.CalcDShape(ir.IntPoint(j), dshape);
            MultAtB(pos, dshape, Jtr);
            double det = Jtr.Det();
            tauval = min(tauval,det);
        }
    }
    if (tauval>0)
    {
        tauval = 1e-4;
    }
    else
    {
        tauval -= 1e-2;
    }
    cout << "the determine tauval is " << tauval << "\n";
    
    initenergy =nlf->GetEnergy(x);
    cout << "energy level is " << initenergy << " \n";
    const int nsp = ir.GetNPoints();
    
    while (passchk !=1 && iters < 15)
    {
        iters += 1;
        jachk = 1;
        add (x,-alpha,c,xsav);
        x_gf = xsav;
        
        finenergy = nlf->GetEnergy(xsav);
        nanchk = isnan(finenergy);
        
        if (finenergy>1.0*initenergy || nanchk!=0)
        {
            passchk = 0;
            alpha *= 0.5;
        }
        else
        {
            passchk = 1;
        }
    }
    
    if (passchk==0) { alpha =0; }
    return alpha;
}


#define BIG_NUMBER 1e+100 // Used when a matrix is outside the metric domain.
#define NBINS 25          // Number of intervals in the metric histogram.
#define GAMMA 0.9         // Used for composite metrics 73, 79, 80.
#define BETA0 0.01        // Used for adaptive pseudo-barrier metrics.
#define TAU0_EPS 0.001    // Used for adaptive shifted-barrier metrics.

int main (int argc, char *argv[])
{
    Mesh *mesh;
    char vishost[] = "localhost";
    int  visport   = 19916;
    int  ans;
    vector<double> logvec (100);
    double weight_fun(const Vector &x);
    double tstart_s=clock();
    
    
    bool dump_iterations = false;
    
    // 3. Read the mesh from the given mesh file. We can handle triangular,
    //    quadrilateral, tetrahedral or hexahedral elements with the same code.
    const char *mesh_file = "../data/tipton.mesh";
    int order = 1;
    bool static_cond = false;
    bool visualization = 1;
    int rs_levels = 0;
    int combomet = 0;
    
    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
                   "Mesh file to use.");
    args.AddOption(&order, "-o", "--order",
                   "Finite element order (polynomial degree) or -1 for"
                   " isoparametric space.");
    args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                   "--no-static-condensation", "Enable static condensation.");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                   "--no-visualization",
                   "Enable or disable GLVis visualization.");
    args.AddOption(&rs_levels, "-rs", "--refine-serial",
                   "Number of times to refine the mesh uniformly in serial.");
    args.AddOption(&combomet, "-cmb", "--combination-of-metrics",
                   "Metric combination");
    
    
    args.Parse();
    if (!args.Good())
    {
        args.PrintUsage(cout);
        return 1;
    }
    args.PrintOptions(cout);
    cout << mesh_file << " about to read a mesh file\n";
    mesh = new Mesh(mesh_file, 1, 1,false);
    cout << "read a mesh file\n";
    
    int dim = mesh->Dimension();
    
    // 4. Refine the mesh to increase the resolution. In this example we do
    //    'rs_levels' of uniform refinement.
    {
        for (int lev = 0; lev < rs_levels; lev++)
        {
            mesh->UniformRefinement();
        }
        
        logvec[0]=rs_levels;
    }
    cout << rs_levels << " serial refinements specified\n";
    
    // 5. Define a finite element space on the mesh. Here we use vector finite
    //    elements which are tensor products of quadratic finite elements. The
    //    dimensionality of the vector finite element space is specified by the
    //    last parameter of the FiniteElementSpace constructor.
    cout << "Mesh curvature: ";
    if (mesh->GetNodes())
    {
        cout << mesh->GetNodes()->OwnFEC()->Name();
    }
    else
    {
        cout << "(NONE)";
    }
    cout << endl;
    
    int mesh_poly_deg = 1;
    cout <<
    "Enter polynomial degree of mesh finite element space:\n"
    "0) QuadraticPos (quads only)\n"
    "p) Degree p >= 1\n"
    " --> " << flush;
    cin >> mesh_poly_deg;
    FiniteElementCollection *fec;
    if (mesh_poly_deg <= 0)
    {
        fec = new QuadraticPosFECollection;
        mesh_poly_deg = 2;
    }
    else
    {
        fec = new H1_FECollection(mesh_poly_deg, dim);
    }
    FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec, dim);
    logvec[1]=mesh_poly_deg;
    
    // 6. Make the mesh curved based on the above finite element space. This
    //    means that we define the mesh elements through a fespace-based
    //    transformation of the reference element.
    mesh->SetNodalFESpace(fespace);
    
    // 7. Set up the right-hand side vector b. In this case we do not need to use
    //    a LinearForm object because b=0.
    Vector b(fespace->GetVSize());
    b = 0.0;
    
    // 8. Get the mesh nodes (vertices and other quadratic degrees of freedom in
    //    the finite element space) as a finite element grid function in fespace.
    GridFunction *x;
    x = mesh->GetNodes();
    
    
    // 9. Define a vector representing the minimal local mesh size in the mesh
    //    nodes. We index the nodes using the scalar version of the degrees of
    //    freedom in fespace.
    Vector h0(fespace->GetNDofs());
    h0 = numeric_limits<double>::infinity();

    {
        Array<int> dofs;
        // loop over the mesh elements
        for (int i = 0; i < fespace->GetNE(); i++)
        {
            // get the local scalar element degrees of freedom in dofs
            fespace->GetElementDofs(i, dofs);
            // adjust the value of h0 in dofs based on the local mesh size
            for (int j = 0; j < dofs.Size(); j++)
            {
                h0(dofs[j]) = min(h0(dofs[j]), mesh->GetElementSize(i));
            }
        }
    }
    
    // 10. Add a random perturbation of the nodes in the interior of the domain.
    //     We define a random grid function of fespace and make sure that it is
    //     zero on the boundary and its values are locally of the order of h0.
    //     The latter is based on the DofToVDof() method which maps the scalar to
    //     the vector degrees of freedom in fespace.
    GridFunction rdm(fespace);
    double jitter = 0.0; // perturbation scaling factor
    rdm.Randomize();
    rdm -= 0.25; // shift to random values in [-0.5,0.5]
    rdm *= jitter;
    {
        for (int i = 0; i < fespace->GetNDofs(); i++)
            for (int d = 0; d < dim; d++)
            {
                rdm(fespace->DofToVDof(i,d)) *= h0(i);
            }
        
        Array<int> vdofs;
        // loop over the boundary elements
        for (int i = 0; i < fespace->GetNBE(); i++)
        {
            // get the vector degrees of freedom in the boundary element
            fespace->GetBdrElementVDofs(i, vdofs);
            // set the boundary values to zero
            for (int j = 0; j < vdofs.Size(); j++)
            {
                rdm(vdofs[j]) = 0.0;
            }
        }
    }
    *x -= rdm;
    
    // 11. Save the perturbed mesh to a file. This output can be viewed later
    //     using GLVis: "glvis -m perturbed.mesh".
    {
        ofstream mesh_ofs("perturbed.mesh");
        mesh->Print(mesh_ofs);
    }
    
    // 14. Simple mesh smoothing can be performed by relaxing the node coordinate
    //     grid function x with the matrix A and right-hand side b. This process
    //     converges to the solution of Ax=b, which we solve below with PCG. Note
    //     that the computed x is the A-harmonic extension of its boundary values
    //     (the coordinates of the boundary vertices). Furthermore, note that
    //     changing x automatically changes the shapes of the elements in the
    //     mesh. The vector field that gives the displacements to the perturbed
    //     positions is saved in the grid function x0.
    GridFunction x0(fespace);
    x0 = *x;
    
    L2_FECollection mfec(mesh_poly_deg, mesh->Dimension(), BasisType::GaussLobatto); //this for vis
    FiniteElementSpace mfes(mesh, &mfec, 1);
    GridFunction metric(&mfes);
    
    
    HyperelasticModel *model;
    NonlinearFormIntegrator *nf_integ;
    
    Coefficient *c = NULL;
    TargetJacobian *tj = NULL;
    
    HyperelasticModel *model2; //this is for combo
    Coefficient *c2 = NULL;     //this is for combo
    TargetJacobian *tj2 = NULL; //this is for combo
    
    //{IDEAL, IDEAL_EQ_SIZE, IDEAL_INIT_SIZE,
    // IDEAL_CUSTOM_SIZE, TARGET_MESH, ALIGNED}
    int tjtype;
    int modeltype;
    cout <<
    "Specify Target Jacobian type:\n"
    "1) IDEAL\n"
    "2) IDEAL_EQ_SIZE\n"
    "3) IDEAL_INIT_SIZE\n"
    "4) IDEAL_EQ_SCALE_SIZE\n"
    " --> " << flush;
    cin >> tjtype;
    
    if (tjtype == 1)
    {
        tj    = new TargetJacobian(TargetJacobian::IDEAL);
        cout << " you chose Target Jacobian - Ideal \n";
    }
    else if (tjtype == 2)
    {
        tj    = new TargetJacobian(TargetJacobian::IDEAL_EQ_SIZE);
        cout << " you chose Target Jacobian - Ideal_eq_size \n";
    }
    else if (tjtype == 3)
    {
        tj    = new TargetJacobian(TargetJacobian::IDEAL_INIT_SIZE);
        cout << " you chose Target Jacobian - Ideal_init_size \n";
    }
    else if (tjtype == 4)
    {
        tj    = new TargetJacobian(TargetJacobian::IDEAL_EQ_SCALE_SIZE);
        cout << " you chose Target Jacobian - Ideal_eq_scale_size \n";
    }
    else
    {
        cout << tjtype;
        cout << "You did not choose a valid option\n";
        cout << "Target Jacobian will default to IDEAL\n";
        tj    = new TargetJacobian(TargetJacobian::IDEAL);
    }
    
    //Metrics for 2D & 3D
    double tauval = -0.1;
    if (dim==2)
    {
        cout << "Choose optimization metric:\n"
        << "1  : |T|^2 \n"
        << "     shape.\n"
        << "2  : 0.5 |T|^2 / tau  - 1 \n"
        << "     shape, condition number.\n"
        << "7  : |T - T^-t|^2 \n"
        << "     shape+size.\n"
        << "9  : tau*|T - T^-t|^2 \n"
        << "     shape+size.\n"
        << "22  : |T|^2 - 2*tau / (2*tau - 2*tau_0)\n"
        << "     untangling.\n"
        << "50  : |T^tT|^2/(2tau^2) - 1\n"
        << "     shape.\n"
        << "52  : (tau-1)^2/ (2*tau - 2*tau_0)\n"
        << "     untangling.\n"
        << "55  : (tau-1)^2\n"
        << "     size.\n"
        << "56  : 0.5*(sqrt(tau) - 1/sqrt(tau))^2\n"
        << "     size.\n"
        << "58  : (|T^tT|^2/(tau^2) - 2*|T|^2/tau + 2)\n"
        << "     shape.\n"
        << "77  : 0.5*(tau - 1/tau)^2\n"
        << "     size metric\n"
        << "211  : (tau-1)^2 - tau + sqrt(tau^2+beta)\n"
        << "      untangling.\n"
        " --> " << flush;
        cin >> modeltype;
        model    = new TMOPHyperelasticModel001;
        if (modeltype == 1)
        {
            model = new TMOPHyperelasticModel001;
            cout << " you chose metric 1 \n";
        }
        else if (modeltype == 2)
        {
            model = new TMOPHyperelasticModel002;
            cout << " you chose metric 2 \n";
        }
        else if (modeltype == 7)
        {
            model = new TMOPHyperelasticModel007;
            cout << " you chose metric 7 \n";
        }
        else if (modeltype == 9)
        {
            model = new TMOPHyperelasticModel009;
        }
        else if (modeltype == 22)
        {
            model = new TMOPHyperelasticModel022(tauval);
            cout << " you chose metric 22\n";
        }
        else if (modeltype == 50)
        {
            model = new TMOPHyperelasticModel050;
            cout << " you chose metric 50\n";
        }
        else if (modeltype == 52)
        {
            model = new TMOPHyperelasticModel252(tauval);
            cout << " you chose metric 52 \n";
        }
        else if (modeltype == 55)
        {
            model = new TMOPHyperelasticModel055;
            cout << " you chose metric 55 \n";
        }
        else if (modeltype == 56)
        {
            model = new TMOPHyperelasticModel056;
            cout << " you chose metric 56\n";
        }
        else if (modeltype == 58)
        {
            model = new TMOPHyperelasticModel058;
            cout << " you chose metric 58\n";
        }
        else if (modeltype == 77)
        {
            model = new TMOPHyperelasticModel077;
            cout << " you chose metric 77\n";
        }
        else if (modeltype == 211)
        {
            model = new TMOPHyperelasticModel211;
            cout << " you chose metric 211 \n";
        }
        else
        {
            cout << "You did not choose a valid option\n";
            cout << "Model type will default to 1\n";
            cout << modeltype;
        }
    }
    else {  //Metrics for 3D
        cout << "Choose optimization metric:\n"
        << "1  : (|T||T^-1|)/3 - 1 \n"
        << "     shape.\n"
        << "2  : (|T|^2|T^-1|^2)/9 - 1 \n"
        << "     shape.\n"
        << "3  : (|T|^2)/3*tau^(2/3) - 1 \n"
        << "     shape.\n"
        << "15  : (tau-1)^2\n"
        << "     size.\n"
        << "16  : 1/2 ( sqrt(tau) - 1/sqrt(tau))^2\n"
        << "     size.\n"
        << "21  : (|T-T^-t|^2)\n"
        << "     shape+size.\n"
        << "52  : (tau-1)^2/ (2*tau - 2*tau_0)\n"
        << "     untangling.\n"
        " --> " << flush;
        double tauval = -0.1;
        cin >> modeltype;
        model    = new TMOPHyperelasticModel302;
        if (modeltype == 1)
        {
            model = new TMOPHyperelasticModel301;
            cout << " you chose metric 1 \n";
        }
        if (modeltype == 2)
        {
            model = new TMOPHyperelasticModel302;
            cout << " you chose metric 2 \n";
        }
        else if (modeltype == 3)
        {
            model = new TMOPHyperelasticModel303;
            cout << " you chose metric 3\n";
        }
        else if (modeltype == 15)
        {
            model = new TMOPHyperelasticModel315;
            cout << " you chose metric 15\n";
        }
        else if (modeltype == 16)
        {
            model = new TMOPHyperelasticModel316;
            cout << " you chose metric 16\n";
        }
        else if (modeltype == 21)
        {
            model = new TMOPHyperelasticModel321;
            cout << " you chose metric 21\n";
        }
        else if (modeltype == 52)
        {
            model = new TMOPHyperelasticModel352(tauval);
            cout << " you chose metric 52\n";
        }
        else
        {
            cout << "You did not choose a valid option\n";
            cout << "Model type will default to 2\n";
            cout << modeltype;
        }
    }
    
    logvec[2]=tjtype;
    logvec[3]=modeltype;
    
    tj->SetNodes(*x);
    tj->SetInitialNodes(x0);
    HyperelasticNLFIntegrator *he_nlf_integ;
    he_nlf_integ = new HyperelasticNLFIntegrator(model, tj);
    
    int ptflag = 1; //if 1 - GLL, else uniform
    int nptdir = 8; //number of sample points in each direction
    const IntegrationRule *ir = NULL;
    if (ptflag == 1)
    {
        // Gauss-Lobatto points.
        ir = &IntRulesLo.Get(fespace->GetFE(0)->GetGeomType(),nptdir);
        cout << "Sample point distribution is GLL based\n";
    }
    else
    {
        // Closed uniform points.
        ir = &IntRulesCU.Get(fespace->GetFE(0)->GetGeomType(),nptdir);
        cout << "Sample point distribution is uniformly spaced\n";
    }
    he_nlf_integ->SetIntegrationRule(*ir);
    //he_nlf_integ->SetLimited(0.05, x0);
    
    //
    nf_integ = he_nlf_integ;
    NonlinearForm a(fespace);
    
    // This is for trying a combo of two integrators
    FunctionCoefficient rhs_coef (weight_fun);
    Coefficient *combo_coeff = NULL;
    
    if (combomet==1) {
        c = new ConstantCoefficient(1.25);  //weight of original metric
        he_nlf_integ->SetCoefficient(*c);
        nf_integ = he_nlf_integ;
        a.AddDomainIntegrator(nf_integ);
        
        
        tj2    = new TargetJacobian(TargetJacobian::IDEAL_EQ_SCALE_SIZE);
        //tj2    = new TargetJacobian(TargetJacobian::IDEAL_INIT_SIZE);
        //tj2    = new TargetJacobian(TargetJacobian::IDEAL_EQ_SIZE);
        model2 = new TMOPHyperelasticModel077;
        tj2->SetNodes(*x);
        tj2->SetInitialNodes(x0);
        HyperelasticNLFIntegrator *he_nlf_integ2;
        he_nlf_integ2 = new HyperelasticNLFIntegrator(model2, tj2);
        he_nlf_integ2->SetIntegrationRule(*ir);
        
        
        //c2 = new ConstantCoefficient(1.00);
        //he_nlf_integ2->SetCoefficient(*c2);     //weight of new metric
        
        he_nlf_integ2->SetCoefficient(rhs_coef);     //weight of new metric as function
        cout << "You have added a combo metric \n";
        a.AddDomainIntegrator(he_nlf_integ2);
    }
    else{
        a.AddDomainIntegrator(nf_integ);
    }
    //
    
    
    InterpolateHyperElasticModel(*model, *tj, *mesh, metric);
    osockstream sol_sock2(visport, vishost);
    sol_sock2 << "solution\n";
    mesh->Print(sol_sock2);
    metric.Save(sol_sock2);
    sol_sock2.send();
    sol_sock2 << "keys " << "JRem" << endl;
    
    // Set essential vdofs by hand for x = 0 and y = 0 (2D). These are
    // attributes 1 and 2.
    const int nd  = x->FESpace()->GetBE(0)->GetDof();
    int n = 0;
    for (int i = 0; i < mesh->GetNBE(); i++)
    {
        const int attr = mesh->GetBdrElement(i)->GetAttribute();
        if (attr == 1 || attr == 2) { n += nd; }
        if (attr == 3) { n += nd * dim; }
    }
    Array<int> ess_vdofs(n), vdofs;
    n = 0;
    for (int i = 0; i < mesh->GetNBE(); i++)
    {
        const int attr = mesh->GetBdrElement(i)->GetAttribute();
        
        
        x->FESpace()->GetBdrElementVDofs(i, vdofs);
        if (attr == 1) // y = 0; fix y components.
        {
            for (int j = 0; j < nd; j++)
            { ess_vdofs[n++] = vdofs[j+nd]; }
        }
        else if (attr == 2) // x = 0; fix x components.
        {
            for (int j = 0; j < nd; j++)
            { ess_vdofs[n++] = vdofs[j]; }
        }
        else if (attr == 3) // vdofs on the other boundary.
        {
            for (int j = 0; j < vdofs.Size(); j++)
            { ess_vdofs[n++] = vdofs[j]; }
        }
    }
    a.SetEssentialVDofs(ess_vdofs);
    
    
    // Fix all boundary nodes.
    int bndrflag;
    cout <<
    "Allow boundary node movement:\n"
    "1 - yes\n"
    "2 - no\n"
    " --> " << flush;
    cin >> bndrflag;
    if (bndrflag != 1)
    {
        //k10partend - the three lines below used to be uncommented
        Array<int> ess_bdr(mesh->bdr_attributes.Max());
        ess_bdr = 1;
        a.SetEssentialBC(ess_bdr);
    }
    logvec[4]=bndrflag;
    
    cout << "Choose linear smoother:\n"
    "0) l1-Jacobi\n"
    "1) CG\n"
    "2) MINRES\n" << " --> \n" << flush;
    cin >> ans;
    Solver *S;
    logvec[5]=ans;
    const double rtol = 1e-12;
    if (ans == 0)
    {
        cout << "Enter number of linear smoothing iterations -->\n " << flush;
        cin >> ans;
        S = new DSmoother(1, 1., ans);
    }
    else if (ans == 1)
    {
        cout << "Enter number of CG smoothing iterations -->\n " << flush;
        cin >> ans;
        CGSolver *cg = new CGSolver;
        cg->SetMaxIter(ans);
        cg->SetRelTol(rtol);
        cg->SetAbsTol(0.0);
        cg->SetPrintLevel(3);
        S = cg;
    }
    else
    {
        cout << "Enter number of MINRES smoothing iterations -->\n " << flush;
        cin >> ans;
        MINRESSolver *minres = new MINRESSolver;
        minres->SetMaxIter(ans);
        minres->SetRelTol(rtol);
        minres->SetAbsTol(0.0);
        minres->SetPrintLevel(3);
        S = minres;
        
    }
    logvec[6]=ans;
    cout << "Enter number of Newton iterations -->\n " << flush;
    cin >> ans;
    logvec[7]=ans;
    
    logvec[8]=a.GetEnergy(*x);
    cout.precision(4);
    cout << "Initial strain energy : " << setprecision(16)
    << logvec[8] << endl;
    
    // save original
    Vector xsav = *x;
    //set value of tau_0 for metric 22 and get min jacobian for mesh statistic
    //
    Array<int> dofs;
    tauval = 1.e+6;
    double minjaco;
    const int NE = mesh->GetNE();
    const GridFunction &nodes = *mesh->GetNodes();
    for (int i = 0; i < NE; i++)
    {
        const FiniteElement &fe = *nodes.FESpace()->GetFE(i);
        const int dim = fe.GetDim();
        int nsp = ir->GetNPoints();
        int dof = fe.GetDof();
        DenseTensor Jtr(dim, dim, nsp);
        const GridFunction *nds;
        nds = &nodes;
        DenseMatrix dshape(dof, dim), pos(dof, dim);
        Array<int> xdofs(dof * dim);
        Vector posV(pos.Data(), dof * dim);
        nds->FESpace()->GetElementVDofs(i, xdofs);
        nds->GetSubVector(xdofs, posV);
        for (int j = 0; j < nsp; j++)
        {
            fe.CalcDShape(ir->IntPoint(j), dshape);
            MultAtB(pos, dshape, Jtr(j));
            double det = Jtr(j).Det();
            tauval = min(tauval,det);
        }
    }
    minjaco = tauval;
    cout << "minimum jacobian in the original mesh is " << minjaco << " \n";
    int newtonits = 0;
    if (tauval>0)
    {
        tauval = 1e-1;
        RelaxedNewtonSolver *newt= new RelaxedNewtonSolver(*ir);
        newt->SetPreconditioner(*S);
        newt->SetMaxIter(ans);
        newt->SetRelTol(rtol);
        newt->SetAbsTol(0.0);
        newt->SetPrintLevel(1);
        newt->SetOperator(a);
        Vector b(0);
        cout << " Relaxed newton solver will be used \n";
        newt->Mult(b, *x);
        if (!newt->GetConverged())
            cout << "NewtonIteration : rtol = " << rtol << " not achieved."
            << endl;
    }
    else
    {
        if (dim==2 && modeltype!=52) {
            model = new TMOPHyperelasticModel022(tauval);
            cout << "model 22 will be used since mesh has negative jacobians\n";
        }
        else if (dim==3)
        {
            model = new TMOPHyperelasticModel352(tauval);
            cout << "model 52 will be used since mesh has negative jacobians\n";
        }
        tauval -= 0.01;
        DescentNewtonSolver *newt= new DescentNewtonSolver(*ir);
        newt->SetPreconditioner(*S);
        newt->SetMaxIter(ans);
        newt->SetRelTol(rtol);
        newt->SetAbsTol(0.0);
        newt->SetPrintLevel(1);
        newt->SetOperator(a);
        Vector b(0);
        cout << " There are inverted elements in the mesh \n";
        cout << " Descent newton solver will be used \n";
        newt->Mult(b, *x);
        if (!newt->GetConverged())
            cout << "NewtonIteration : rtol = " << rtol << " not achieved."
            << endl;
    }
    
    logvec[9]=a.GetEnergy(*x);
    cout << "Final strain energy   : " << a.GetEnergy(*x) << endl;
    cout << "Initial strain energy was  : " << logvec[8] << endl;
    cout << "% change is  : " << (logvec[8]-logvec[9])*100/logvec[8] << endl;
    logvec[10] = (logvec[8]-logvec[9])*100/logvec[8];
    logvec[11] = newtonits;
    
    if (tj)
    {
        InterpolateHyperElasticModel(*model, *tj, *mesh, metric);
        osockstream sol_sock(visport, vishost);
        sol_sock << "solution\n";
        mesh->Print(sol_sock);
        metric.Save(sol_sock);
        sol_sock.send();
        sol_sock << "keys " << "JRmm" << endl;
    }
    
    // 17. Get some mesh statistics
    double minjacs = 1.e+100;
    for (int i = 0; i < NE; i++)
    {
        const FiniteElement &fe = *nodes.FESpace()->GetFE(i);
        const int dim = fe.GetDim();
        int nsp = ir->GetNPoints();
        int dof = fe.GetDof();
        DenseTensor Jtr(dim, dim, nsp);
        const GridFunction *nds;
        nds = &nodes;
        DenseMatrix dshape(dof, dim), pos(dof, dim);
        Array<int> xdofs(dof * dim);
        Vector posV(pos.Data(), dof * dim);
        nds->FESpace()->GetElementVDofs(i, xdofs);
        nds->GetSubVector(xdofs, posV);
        for (int j = 0; j < nsp; j++)
        {
            //cout << "point number " << j << "\n";
            fe.CalcDShape(ir->IntPoint(j), dshape);
            MultAtB(pos, dshape, Jtr(j));
            double det = Jtr(j).Det();
            minjacs = min(minjacs,det);
        }
    }
    
    cout << "minimum jacobian before and after smoothing " << minjaco << " " << minjacs << " \n";
    
    delete S;
    delete model;
    delete c;
    
    delete model2;
    delete combo_coeff;
    
    // Define mesh displacement
    x0 -= *x;
    
    // 15. Save the smoothed mesh to a file. This output can be viewed later
    //     using GLVis: "glvis -m smoothed.mesh".
    {
        ofstream mesh_ofs("smoothed.mesh");
        mesh_ofs.precision(14);
        mesh->Print(mesh_ofs);
    }
    
    // 17. Free the used memory.
    delete fespace;
    delete fec;
    delete mesh;
    
    // Execution time
    double tstop_s=clock();
    cout << "The total time it took for this example's execution is: " << (tstop_s-tstart_s)/1000000. << " seconds\n";
    return 0;
}
double weight_fun(const Vector &x)
{
    double r2 = x(0)*x(0) + x(1)*x(1);
    double l2;
    if (r2>0)
    {
        r2 = sqrt(r2);
    }
    double den = 0.002;
    l2 = 0.2 +     0.5*std::tanh((r2-0.16)/den)-(0.5*std::tanh((r2-0.17)/den))
    +0.5*std::tanh((r2-0.23)/den)-(0.5*std::tanh((r2-0.24)/den));
    return l2;
}

