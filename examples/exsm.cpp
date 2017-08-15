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

using namespace mfem;
using namespace std;

class RelaxedNewtonSolver : public NewtonSolver
{
private:
   // Quadrature points that are checked for negative Jacobians etc.
   const IntegrationRule &ir;

public:
   RelaxedNewtonSolver(const IntegrationRule &irule) : ir(irule) { }

   virtual double ComputeScalingFactor(const Vector &x, const Vector &c) const;
};

double RelaxedNewtonSolver::ComputeScalingFactor(const Vector &x,
                                                 const Vector &c) const
{
    const NonlinearForm *nlf = dynamic_cast<const NonlinearForm *>(oper);
    
    const int NE = nlf->FESpace()->GetMesh()->GetNE();
    
    Vector xsav = x; //create a copy of x
    int passchk = 0;
    int iter_cnt = 0;
    double alpha = 1.;
    int jachk = 1;
    double initenergy = 0.0;
    double finenergy = 0.0;
    double nanchk;
    
    initenergy = nlf->GetEnergy(x);
    
    while (passchk !=1 && iter_cnt < 20)
    {
        ++iter_cnt;
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
    
    cout << initenergy << " " << finenergy
         << " energy value before and after the Newton iteration" << endl;
    cout << "alpha value is " << alpha << endl;
    
    if (passchk==0) { alpha = 0.0; }
    return alpha;
}

class DescentNewtonSolver : public NewtonSolver
{
private:
   // Quadrature points that are checked for negative Jacobians etc.
   const IntegrationRule &ir;

public:
   DescentNewtonSolver(const IntegrationRule &irule) : ir(irule) { }

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
    
    while (passchk !=1 && iters < 15)
    {
        iters += 1;
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
    
    if (passchk==0) { alpha = 0.0; }
    return alpha;
}

double weight_fun(const Vector &x);

int main (int argc, char *argv[])
{
    char vishost[] = "localhost";
    int  visport   = 19916;
    
    // 1. Parse command-line options.
    const char *mesh_file = "../data/tipton.mesh";
    int mesh_poly_deg = 1;
    int rs_levels = 0;
    int metric_id = 1;
    int target_id = 1;
    int newton_iter = 10;
    int lin_solver = 2;
    bool move_bnd = false;
    bool visualization = true;
    int combomet = 0;
    
    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
                   "Mesh file to use.");
    args.AddOption(&mesh_poly_deg, "-o", "--order",
                   "Polynomial degree of mesh finite element space.");
    args.AddOption(&rs_levels, "-rs", "--refine-serial",
                   "Number of times to refine the mesh uniformly in serial.");
    args.AddOption(&metric_id, "-mid", "--metric-id",
       "Mesh optimization metric:\n\t"
       "1  : |T|^2                          -- 2D shape\n\t"
       "2  : 0.5|T|^2/tau-1                 -- 2D shape (condition number)\n\t"
       "7  : |T-T^-t|^2                     -- 2D shape+size\n\t"
       "9  : tau*|T-T^-t|^2                 -- 2D shape+size\n\t"
       "22 : 0.5(|T|^2-2*tau)/(tau-tau_0)   -- 2D untangling\n\t"
       "50 : 0.5|T^tT|^2/tau^2-1            -- 2D shape\n\t"
       "52 : 0.5(tau-1)^2/(tau-tau_0)       -- 2D untangling\n\t"
       "55 : (tau-1)^2                      -- 2D size\n\t"
       "56 : 0.5(sqrt(tau)-1/sqrt(tau))^2   -- 2D size\n\t"
       "58 : |T^tT|^2/(tau^2)-2*|T|^2/tau+2 -- 2D shape\n\t"
       "77 : 0.5(tau-1/tau)^2               -- 2D size\n\t"
       "211: (tau-1)^2-tau+sqrt(tau^2+beta) -- 2D untangling\n\t"
       "301: (|T||T^-1|)/3-1              -- 3D shape\n\t"
       "302: (|T|^2|T^-1|^2)/9-1          -- 3D shape\n\t"
       "303: (|T|^2)/3*tau^(2/3)-1        -- 3D shape\n\t"
       "315: (tau-1)^2                    -- 3D size\n\t"
       "316: 0.5(sqrt(tau)-1/sqrt(tau))^2 -- 3D size\n\t"
       "321: |T-T^-t|^2                   -- 3D shape+size\n\t"
       "352: 0.5(tau-1)^2/(tau-tau_0)     -- 3D untangling");
    args.AddOption(&target_id, "-tid", "--target-id",
       "Target (ideal element) type:\n\t"
       "1: IDEAL\n\t"
       "2: IDEAL_EQ_SIZE\n\t"
       "3: IDEAL_INIT_SIZE\n\t"
       "4: IDEAL_EQ_SCALE_SIZE");
    args.AddOption(&newton_iter, "-ni", "--newton-iters",
                   "Maximum number of Newton iterations.");
    args.AddOption(&lin_solver, "-ls", "--lin-solver",
                   "ODE solver: 0 - l1-Jacobi, 1 - CG, 2 - MINRES.");
    args.AddOption(&move_bnd, "-bnd", "--move-boundary", "-fix-bnd",
                   "--fix-boundary",
                   "Enable motion along horizontal and vertical boundaries.");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                   "--no-visualization",
                   "Enable or disable GLVis visualization.");
    args.AddOption(&combomet, "-cmb", "--combination-of-metrics",
                   "Metric combination");
    args.Parse();
    if (!args.Good())
    {
        args.PrintUsage(cout);
        return 1;
    }
    args.PrintOptions(cout);

    // 2. Initialize and refine the starting mesh.
    Mesh *mesh = new Mesh(mesh_file, 1, 1, false);
    for (int lev = 0; lev < rs_levels; lev++)
    {
       mesh->UniformRefinement();
    }
    const int dim = mesh->Dimension();
    cout << "Mesh curvature: ";
    if (mesh->GetNodes())
    {
        cout << mesh->GetNodes()->OwnFEC()->Name();
    }
    else { cout << "(NONE)"; }
    cout << endl;
    
    // 3. Define a finite element space on the mesh. Here we use vector finite
    //    elements which are tensor products of quadratic finite elements. The
    //    dimensionality of the vector finite element space is specified by the
    //    last parameter of the FiniteElementSpace constructor.
    FiniteElementCollection *fec;
    if (mesh_poly_deg <= 0)
    {
        fec = new QuadraticPosFECollection;
        mesh_poly_deg = 2;
    }
    else { fec = new H1_FECollection(mesh_poly_deg, dim); }
    FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec, dim);
    
    // 4. Make the mesh curved based on the above finite element space. This
    //    means that we define the mesh elements through a fespace-based
    //    transformation of the reference element.
    mesh->SetNodalFESpace(fespace);
    
    // 5. Set up the right-hand side vector b. In this case we do not need to use
    //    a LinearForm object because b=0.
    Vector b(fespace->GetVSize());
    b = 0.0;
    
    // 6. Get the mesh nodes (vertices and other quadratic degrees of freedom in
    //    the finite element space) as a finite element grid function in fespace.
    //    Furthermore, note that changing x automatically changes the shapes of
    //    the elements in the mesh.
    GridFunction *x = mesh->GetNodes();
    
    // 7. Define a vector representing the minimal local mesh size in the mesh
    //    nodes. We index the nodes using the scalar version of the degrees of
    //    freedom in fespace.
    Vector h0(fespace->GetNDofs());
    h0 = numeric_limits<double>::infinity();
    {
        Array<int> dofs;
        // Loop over the mesh elements.
        for (int i = 0; i < fespace->GetNE(); i++)
        {
            // Get the local scalar element degrees of freedom in dofs.
            fespace->GetElementDofs(i, dofs);
            // Adjust the value of h0 in dofs based on the local mesh size.
            for (int j = 0; j < dofs.Size(); j++)
            {
                h0(dofs[j]) = min(h0(dofs[j]), mesh->GetElementSize(i));
            }
        }
    }
    
    // 8. Add a random perturbation of the nodes in the interior of the domain.
    //    We define a random grid function of fespace and make sure that it is
    //    zero on the boundary and its values are locally of the order of h0.
    //    The latter is based on the DofToVDof() method which maps the scalar to
    //    the vector degrees of freedom in fespace.
    GridFunction rdm(fespace);
    const double jitter = 0.0; // perturbation scaling factor
    rdm.Randomize();
    rdm -= 0.25; // shift to random values in [-0.5,0.5]
    rdm *= jitter;
    {
        for (int i = 0; i < fespace->GetNDofs(); i++)
        {
            for (int d = 0; d < dim; d++)
            {
                rdm(fespace->DofToVDof(i,d)) *= h0(i);
            }
        }
        
        Array<int> vdofs;
        // Loop over the boundary elements.
        for (int i = 0; i < fespace->GetNBE(); i++)
        {
            // Get the vector degrees of freedom in the boundary element.
            fespace->GetBdrElementVDofs(i, vdofs);
            // Set the boundary values to zero.
            for (int j = 0; j < vdofs.Size(); j++) { rdm(vdofs[j]) = 0.0; }
        }
    }
    *x -= rdm;
    
    // 9. Save the perturbed mesh to a file. This output can be viewed later
    //    using GLVis: "glvis -m perturbed.mesh".
    {
        ofstream mesh_ofs("perturbed.mesh");
        mesh->Print(mesh_ofs);
    }
    
    // 10. The vector field that gives the displacements to the perturbed
    //     positions is saved in the grid function x0.
    GridFunction x0(fespace);
    x0 = *x;
    
    // Used for visualization of the metric values.
    L2_FECollection mfec(mesh_poly_deg, mesh->Dimension(), BasisType::GaussLobatto);
    FiniteElementSpace mfes(mesh, &mfec, 1);
    GridFunction metric(&mfes);
    
    NonlinearFormIntegrator *nf_integ;
    
    Coefficient *c = NULL;
    
    // Used for combinations of metrics.
    HyperelasticModel *model2 = NULL;
    Coefficient *c2 = NULL;
    TargetJacobian *tj2 = NULL;

    double tauval = -0.1;
    HyperelasticModel *model = NULL;
    switch (metric_id)
    {
       case 1: model = new TMOPHyperelasticModel001; break;
       case 2: model = new TMOPHyperelasticModel002; break;
       case 7: model = new TMOPHyperelasticModel007; break;
       case 9: model = new TMOPHyperelasticModel009; break;
       case 22: model = new TMOPHyperelasticModel022(tauval); break;
       case 50: model = new TMOPHyperelasticModel050; break;
       case 52: model = new TMOPHyperelasticModel252(tauval); break;
       case 55: model = new TMOPHyperelasticModel055; break;
       case 56: model = new TMOPHyperelasticModel056; break;
       case 58: model = new TMOPHyperelasticModel058; break;
       case 77: model = new TMOPHyperelasticModel077; break;
       case 211: model = new TMOPHyperelasticModel211; break;
       case 301: model = new TMOPHyperelasticModel301; break;
       case 302: model = new TMOPHyperelasticModel302; break;
       case 303: model = new TMOPHyperelasticModel303; break;
       case 315: model = new TMOPHyperelasticModel315; break;
       case 316: model = new TMOPHyperelasticModel316; break;
       case 321: model = new TMOPHyperelasticModel321; break;
       case 352: model = new TMOPHyperelasticModel352(tauval); break;
       default: cout << "Unknown metric_id: " << metric_id << endl;
                return 3;
    }

    TargetJacobian *tj = NULL;
    switch (target_id)
    {
       case 1: tj = new TargetJacobian(TargetJacobian::IDEAL); break;
       case 2: tj = new TargetJacobian(TargetJacobian::IDEAL_EQ_SIZE); break;
       case 3: tj = new TargetJacobian(TargetJacobian::IDEAL_INIT_SIZE); break;
       case 4: tj = new TargetJacobian(TargetJacobian::IDEAL_EQ_SCALE_SIZE); break;
       default: cout << "Unknown target_id: " << target_id << endl;
                return 3;
    }
    
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
    
    nf_integ = he_nlf_integ;
    NonlinearForm a(fespace);
    
    // This is for trying a combo of two integrators
    FunctionCoefficient rhs_coef (weight_fun);
    Coefficient *combo_coeff = NULL;
    
    if (combomet==1)
    {
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
    else { a.AddDomainIntegrator(nf_integ); }
    
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
    
    if (move_bnd == false)
    {
        Array<int> ess_bdr(mesh->bdr_attributes.Max());
        ess_bdr = 1;
        a.SetEssentialBC(ess_bdr);
    }
    
    Solver *S = NULL;
    const double rtol  = 1e-12;
    const int max_iter = 100;
    if (lin_solver == 0)
    {
        S = new DSmoother(1, 1., max_iter);
    }
    else if (lin_solver == 1)
    {
        CGSolver *cg = new CGSolver;
        cg->SetMaxIter(max_iter);
        cg->SetRelTol(rtol);
        cg->SetAbsTol(0.0);
        cg->SetPrintLevel(3);
        S = cg;
    }
    else
    {
        MINRESSolver *minres = new MINRESSolver;
        minres->SetMaxIter(max_iter);
        minres->SetRelTol(rtol);
        minres->SetAbsTol(0.0);
        minres->SetPrintLevel(3);
        S = minres;
    }
    
    const double init_en = a.GetEnergy(*x);
    cout.precision(4);
    cout << "Initial strain energy : " << setprecision(16) << init_en << endl;
    
    // save original
    Vector xsav = *x;

    //set value of tau_0 for metric 22 and get min jacobian for mesh statistic
    tauval = 1.e+6;
    double minJ0;
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
    minJ0 = tauval;
    cout << "minimum jacobian in the original mesh is " << minJ0 << " \n";
    if (tauval>0)
    {
        tauval = 1e-1;
        RelaxedNewtonSolver *newt= new RelaxedNewtonSolver(*ir);
        newt->SetPreconditioner(*S);
        newt->SetMaxIter(newton_iter);
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
        if (dim==2 && metric_id!=52) {
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
        newt->SetMaxIter(newton_iter);
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
    
    const double fin_en = a.GetEnergy(*x);
    cout << "Final strain energy   : " << fin_en << endl;
    cout << "Initial strain energy was  : " << init_en << endl;
    cout << "% change is  : " << (init_en - fin_en) * 100.0 / init_en << endl;
    
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
    double minJn = 1.e+100;
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
            minJn = min(minJn,det);
        }
    }
    
    cout << "min|J| before / after smoothing: "
         << minJ0 << " / " << minJn << endl;
    
    delete S;
    delete model;
    delete c;
    
    delete model2;
    delete c2;
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
    
    return 0;
}

// Used for the 2D Tipton mesh.
double weight_fun(const Vector &x)
{
   const double r = sqrt(x(0)*x(0) + x(1)*x(1) + 1e-12);
   const double den = 0.002;
   double l2 = 0.2 + 0.5*std::tanh((r-0.16)/den) - 0.5*std::tanh((r-0.17)/den)
                   + 0.5*std::tanh((r-0.23)/den) - 0.5*std::tanh((r-0.24)/den);
   return l2;
}
