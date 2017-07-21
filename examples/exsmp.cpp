//                               ETHOS Example 1
//
// Compile with: make exsm
//
// Sample runs:  exsm -m tipton.mesh
//
// Description: This example code performs a simple mesh smoothing based on a
//              topologically defined "mesh Laplacian" matrix.
//
//              The example highlights meshes with curved elements, the
//              assembling of a custom finite element matrix, the use of vector
//              finite element spaces, the definition of different spaces and
//              grid functions on the same mesh, and the setting of values by
//              iterating over the interior and the boundary elements.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <limits>
#include <unistd.h>

using namespace mfem;

using namespace std;

// This is where K10 methods start
// Implement Relaxed Newton solver to avoid mesh becoming invalid during the
// optimization process
class RelaxedNewtonSolver : public IterativeSolver
{
protected:
    mutable Vector r, c;
public:
    RelaxedNewtonSolver() { }
    
#ifdef MFEM_USE_MPI
    RelaxedNewtonSolver(MPI_Comm _comm) : IterativeSolver(_comm) { }
#endif
        virtual void SetOperator(const Operator &op);
    
        virtual void SetSolver(Solver &solver) { prec = &solver; }
    
        virtual void Mult(const Vector &b, Vector &x) const;
    
        virtual void Mult2(const Vector &b, Vector &x, ParMesh &pmesh,
                           const IntegrationRule &ir, int *itnums,
                           const ParNonlinearForm &nlf) const;
};

void RelaxedNewtonSolver::SetOperator(const Operator &op)
{
    oper = &op;
    height = op.Height();
    width = op.Width();
    MFEM_ASSERT(height == width, "square Operator is required.");
    
    r.SetSize(width);
    c.SetSize(width);
}

void RelaxedNewtonSolver::Mult(const Vector &b, Vector &x) const
{
    return;
}

void RelaxedNewtonSolver::Mult2(const Vector &b, Vector &x,
                                ParMesh &pmesh, const IntegrationRule &ir,
                                int *itnums, const ParNonlinearForm &nlf) const
{
    int myid;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    
    MFEM_ASSERT(oper != NULL, "the Operator is not set (use SetOperator).");
    MFEM_ASSERT(prec != NULL, "the Solver is not set (use SetSolver).");
    
    int it;
    double norm, norm_goal;
    bool have_b = (b.Size() == Height());
    
    if (!iterative_mode)
    {
        x = 0.0;
    }
    
    oper->Mult(x, r);
    if (have_b)
    {
        r -= b;
    }
    
    norm = Norm(r);
    norm_goal = std::max(rel_tol*norm, abs_tol);
    
    prec->iterative_mode = false;
    
    // x_{i+1} = x_i - [DF(x_i)]^{-1} [F(x_i)-b]
    for (it = 0; true; it++)
    {
        MFEM_ASSERT(IsFinite(norm), "norm = " << norm);
        if (print_level >= 0)
        {
            cout << "Newton iteration " << setw(2) << it
                 << " : ||r|| = " << norm << '\n';
        }
        
        if (norm <= norm_goal)
        {
            converged = 1;
            break;
        }
        
        if (it >= max_iter)
        {
            converged = 0;
            break;
        }
        
        prec->SetOperator(oper->GetGradient(x));
        prec->Mult(r, c);  // c = [DF(x_i)]^{-1} [F(x_i)-b]
        
        //k10 some changes
        const int NE = pmesh.GetNE();
    
        Vector xsav = x; //create a copy of x
        Vector csav = c;
        int tchk = 0;
        int iters = 0;
        double alpha = 1.;
        int jachk = 1;
        double initenergy = 0.0;
        double finenergy = 0.0;
        int nanchk;
        
        initenergy = nlf.GetEnergy(x);
        
        while (tchk !=1 && iters < 20)
        {
            iters += 1;
            //cout << "number of iters is " << iters << "\n";
            jachk = 1;
            c = csav;
            x = xsav;
            add (xsav,-alpha,csav,x);

            ParGridFunction x_gf(nlf.ParFESpace());
            x_gf.Distribute(x);

            finenergy = nlf.GetEnergy(x);
            nanchk = isnan(finenergy);
            for (int i = 0; i < NE; i++)
            {
                const FiniteElement &fe = *x_gf.ParFESpace()->GetFE(i);
                const int dim = fe.GetDim(), nsp = ir.GetNPoints(),
                dof = fe.GetDof();
                
                DenseMatrix Jtr(dim);
                
                DenseMatrix dshape(dof, dim), pos(dof, dim);
                Array<int> xdofs(dof * dim);
                Vector posV(pos.Data(), dof * dim);
                
                x_gf.ParFESpace()->GetElementVDofs(i, xdofs);
                x_gf.GetSubVector(xdofs, posV);
                for (int j = 0; j < nsp; j++)
                {
                    fe.CalcDShape(ir.IntPoint(j), dshape);
                    MultAtB(pos, dshape, Jtr);
                    double det = Jtr.Det();
                }
            }
            int jachkglob;
            MPI_Allreduce(&jachk, &jachkglob, 1, MPI_INT, MPI_MIN,
                          pmesh.GetComm());
            int nanchkglob;
            MPI_Allreduce(&nanchk, &nanchkglob, 1, MPI_INT, MPI_MIN,
                          pmesh.GetComm());
            
            //finenergy = initenergy - 1; //WARNING: THIS IS JUST FOR TIPTONUNI.MESH
            if (finenergy>1.0*initenergy || nanchkglob!=0 || jachkglob==0)
            {
                tchk = 0;
                alpha *= 0.5;
            }
            else
            {
                tchk = 1;
            }
        }
        if (print_level >= 0)
        {
           cout << initenergy << " " << finenergy
                <<  " energy value before and after newton iteration\n";
           cout << "alpha value is " << alpha << " \n";
        }
        
        if (tchk == 0) { alpha = 0; }
        
        add (xsav, -alpha, csav, x);
        
        oper->Mult(x, r);
        if (have_b)
        {
            r -= b;
        }
        norm = Norm(r);
        
        if (alpha == 0)
        {
            converged = 0;
            if (print_level >= 0)
            {
               cout << "Line search was not successfull.. stopping Newton\n";
            }
            break;
        }
    }
    
    final_iter = it;
    final_norm = norm;
    *itnums = final_iter;
}

// Relaxed newton iterations which is kind of like descent.. it can handle inverted
// elements but the metric value must decrease :)
class DescentNewtonSolver : public IterativeSolver
{
protected:
    mutable Vector r, c;
public:
    DescentNewtonSolver() { }
    
#ifdef MFEM_USE_MPI
    DescentNewtonSolver(MPI_Comm _comm) : IterativeSolver(_comm) { }
#endif
    virtual void SetOperator(const Operator &op);
    
    virtual void SetSolver(Solver &solver) { prec = &solver; }
    
    virtual void Mult(const Vector &b, Vector &x) const;
    
    virtual void Mult2(const Vector &b, ParGridFunction &x,
                       const ParMesh &pmesh, const IntegrationRule &ir,
                       int *itnums, const ParNonlinearForm &nlf,
                       double &tauval) const;
};
void DescentNewtonSolver::SetOperator(const Operator &op)
{
    oper = &op;
    height = op.Height();
    width = op.Width();
    MFEM_ASSERT(height == width, "square Operator is required.");
    
    r.SetSize(width);
    c.SetSize(width);
}

void DescentNewtonSolver::Mult(const Vector &b, Vector &x) const
{
    return;
}

void DescentNewtonSolver::Mult2(const Vector &b, ParGridFunction &x,
                                const ParMesh &pmesh, const IntegrationRule &ir,
                                int *itnums, const ParNonlinearForm &nlf,
                                double &tauval) const

{
    MFEM_ASSERT(oper != NULL, "the Operator is not set (use SetOperator).");
    MFEM_ASSERT(prec != NULL, "the Solver is not set (use SetSolver).");
    
    int it;
    double norm, norm_goal;
    bool have_b = (b.Size() == Height());
    
    if (!iterative_mode)
    {
        x = 0.0;
    }
    
    oper->Mult(x, r);
    if (have_b)
    {
        r -= b;
    }
    
    norm = Norm(r);
    norm_goal = std::max(rel_tol*norm, abs_tol);
    
    prec->iterative_mode = false;
    // x_{i+1} = x_i - [DF(x_i)]^{-1} [F(x_i)-b]
    for (it = 0; true; it++)
    {
        MFEM_ASSERT(IsFinite(norm), "norm = " << norm);
        if (print_level >= 0)
            cout << "Newton iteration " << setw(2) << it
            << " : ||r|| = " << norm << '\n';
        
        if (norm <= norm_goal)
        {
            converged = 1;
            break;
        }
        
        
        if (it >= max_iter)
        {
            converged = 0;
            break;
        }
        
        const int NE = pmesh.GetNE();
        const GridFunction &nodes = *pmesh.GetNodes();
        
        tauval = 1e+6;
        int nelinvorig= 0;
        for (int i = 0; i < NE; i++)
        {
            const FiniteElement &fe = *nodes.FESpace()->GetFE(i);
            const int dim = fe.GetDim(), nsp = ir.GetNPoints(),
            dof = fe.GetDof();
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
                fe.CalcDShape(ir.IntPoint(j), dshape);
                MultAtB(pos, dshape, Jtr(j));
                double det = Jtr(j).Det();
                tauval = min(tauval,det);
                if (det<=0.)
                {
                    nelinvorig += 1;
                }
            }
        }

        double tauvalglob;
        MPI_Allreduce(&tauval, &tauvalglob, 1, MPI_DOUBLE, MPI_MIN,
                      pmesh.GetComm());

        
        if (tauvalglob>0)
        {
            tauval = 1e-4;
        }
        else
        {
            tauval = tauvalglob-1e-2;
        }
        cout << "the determine tauval is " << tauval << "\n";

        prec->SetOperator(oper->GetGradient(x));
        prec->Mult(r, c);  // c = [DF(x_i)]^{-1} [F(x_i)-b]
        
        Vector xsav = x; //create a copy of x
        int tchk = 0;
        int iters = 0;
        double alpha = 1;
        double initenergy = 0.0;
        double finenergy = 0.0;
        double nanchk;
        int nelinvnew;
        nelinvnew = 0;
        
        initenergy = nlf.GetEnergy(x);
        cout << "energy level is " << initenergy << " \n";
        const int nsp = ir.GetNPoints();
        
        while (tchk !=1 && iters <  15)
        {
            iters += 1;
            add (xsav,-alpha,c,x);
            finenergy = nlf.GetEnergy(x);
            nanchk = isnan(finenergy);
            
            int nanchkglob;
            MPI_Allreduce(&nanchk, &nanchkglob, 1, MPI_INT, MPI_MIN,
                          pmesh.GetComm());
            
            if (finenergy>initenergy || nanchkglob!=0)
            {
                alpha *= 0.1;
            }
                else
            {
                tchk = 1;
            }
        }
        
        if (tchk==0)
        {
            alpha =0;
        }
        add (xsav,-alpha,c,x);
        
        oper->Mult(x, r);
        if (have_b)
        {
            r -= b;
        }
        norm = Norm(r);
        
        if (alpha == 0)
        {
            converged = 0;
            cout << "Line search was not successfull.. stopping Newton\n";
            break;
        }
        
    }
    
    final_iter = it;
    final_norm = norm;
    *itnums = final_iter;
        
}

#define BIG_NUMBER 1e+100 // Used when a matrix is outside the metric domain.
#define NBINS 25          // Number of intervals in the metric histogram.
#define GAMMA 0.9         // Used for composite metrics 73, 79, 80.
#define BETA0 0.01        // Used for adaptive pseudo-barrier metrics.
#define TAU0_EPS 0.001    // Used for adaptive shifted-barrier metrics.

//IntegrationRules IntRulesCU(0, Quadrature1D::ClosedUniform);

int main (int argc, char *argv[])
{
    // 1. Initialize MPI.
    int num_procs, myid;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    
    Mesh *mesh;
    char vishost[] = "localhost";
    int  visport   = 19916;
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
    args.Parse();
    if (!args.Good())
    {
        if (myid == 0)
        {
            args.PrintUsage(cout);
        }
        MPI_Finalize();
        return 1;
    }
    if (myid == 0)
    {
        args.PrintOptions(cout);
    }
    mesh = new Mesh(mesh_file, 1, 1,false);
    
    int dim = mesh->Dimension();
    
    // 4. Refine the mesh to increase the resolution. In this example we do
    //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
    //    largest number that gives a final mesh with no more than 1000
    //    elements.
    {
        int ref_levels = (int)floor(log(1000./mesh->GetNE())/log(2.)/dim);
        if (myid == 0)
        {
           cout << "enter refinement levels [" << ref_levels << "] --> " << flush;
           cin >> ref_levels;
        }
        int ref_levels_g;
        MPI_Allreduce(&ref_levels, &ref_levels_g, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD);
        
        for (int l = 0; l < ref_levels_g; l++)
        {
            mesh->UniformRefinement();
        }
        
        logvec[0]=ref_levels;
    }
    
    // 5. Define a finite element space on the mesh. Here we use vector finite
    //    elements which are tensor products of quadratic finite elements. The
    //    dimensionality of the vector finite element space is specified by the
    //    last parameter of the FiniteElementSpace constructor.
    if (myid == 0)
    {
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
    }
    
    int mesh_poly_deg;
    if (myid == 0)
    {
       cout << "Enter polynomial degree of mesh finite element space:\n"
               "0) QuadraticPos (quads only)\n"
               "p) Degree p >= 1\n"
               " --> " << flush;
       cin >> mesh_poly_deg;
    }
    MPI_Bcast(&mesh_poly_deg, num_procs, MPI_INT, 0, MPI_COMM_WORLD);

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
    ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
    delete mesh;
    ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec, dim);
    logvec[1] = mesh_poly_deg;
    
    // 6. Make the mesh curved based on the above finite element space. This
    //    means that we define the mesh elements through a fespace-based
    //    transformation of the reference element.
    pmesh->SetNodalFESpace(fespace);
    
    // 7. Set up the right-hand side vector b. In this case we do not need to use
    //    a LinearForm object because b=0.
    Vector b(fespace->TrueVSize());
    b = 0.0;
    
    // 8. Get the mesh nodes (vertices and other quadratic degrees of freedom in
    //    the finite element space) as a finite element grid function in fespace.
    ParGridFunction x(fespace);
    pmesh->SetNodalGridFunction(&x);
    
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
                h0(dofs[j]) = min(h0(dofs[j]), pmesh->GetElementSize(i));
            }
        }
    }
    
    // 10. Add a random perturbation of the nodes in the interior of the domain.
    //     We define a random grid function of fespace and make sure that it is
    //     zero on the boundary and its values are locally of the order of h0.
    //     The latter is based on the DofToVDof() method which maps the scalar to
    //     the vector degrees of freedom in fespace.
    ParGridFunction rdm(fespace);
    double jitter = 0.0; // perturbation scaling factor
    rdm.Randomize();
    rdm -= 0.2; // shift to random values in [-0.5,0.5]
    rdm *= jitter;
    {
        // scale the random values to be of order of the local mesh size
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
    HypreParVector *trueF = rdm.ParallelAverage();
    rdm = *trueF;
    x -= rdm;
    
    // 11. Save the perturbed mesh to a file. This output can be viewed later
    //     using GLVis: "glvis -m perturbed.mesh".
    {
        //ofstream mesh_ofs("perturbed.mesh");
        //pmesh->Print(mesh_ofs);
        
        ostringstream mesh_name, velo_name, ee_name;
        mesh_name << "perturbed." << setfill('0') << setw(6) << myid;
        ofstream mesh_ofs(mesh_name.str().c_str());
        mesh_ofs.precision(8);
        pmesh->Print(mesh_ofs);
        
    }
    
    // 14. Simple mesh smoothing can be performed by relaxing the node coordinate
    //     grid function x with the matrix A and right-hand side b. This process
    //     converges to the solution of Ax=b, which we solve below with PCG. Note
    //     that the computed x is the A-harmonic extension of its boundary values
    //     (the coordinates of the boundary vertices). Furthermore, note that
    //     changing x automatically changes the shapes of the elements in the
    //     mesh. The vector field that gives the displacements to the perturbed
    //     positions is saved in the grid function x0.
    ParGridFunction x0(fespace);
    x0 = *x;
    
    // Visualization structures.
    L2_FECollection mfec(mesh_poly_deg, pmesh->Dimension(),
                         BasisType::GaussLobatto);
    ParFiniteElementSpace mfes(pmesh, &mfec, 1);
    ParGridFunction metric(&mfes);
    
    HyperelasticModel *model;
    NonlinearFormIntegrator *nf_integ;
    HyperelasticNLFIntegrator *he_nlf_integ2;
        
    Coefficient *c = NULL;
    TargetJacobian *tj = NULL;

    // Used for combos.
    HyperelasticModel *model2;
    Coefficient *c2 = NULL;
    TargetJacobian *tj2 = NULL;

    // {IDEAL, IDEAL_EQ_SIZE, IDEAL_INIT_SIZE,
    // IDEAL_CUSTOM_SIZE, TARGET_MESH, ALIGNED}
    int tjtype;
    int modeltype;
    if (myid==0)
    {
        cout <<
            "Specify Target Jacobian type:\n"
            "1) IDEAL\n"
            "2) IDEAL_EQ_SIZE\n"
            "3) IDEAL_INIT_SIZE\n"
            "4) IDEAL_EQ_SCALE_SIZE\n"
            " --> " << flush;
        cin >> tjtype;
    }
    MPI_Bcast(&tjtype, num_procs, MPI_INT, 0, MPI_COMM_WORLD);
    tj    = new TargetJacobian(TargetJacobian::IDEAL, MPI_COMM_WORLD);
    if (tjtype == 1)
    {
       tj    = new TargetJacobian(TargetJacobian::IDEAL, MPI_COMM_WORLD);
    }
    else if (tjtype == 2)
    {
       tj    = new TargetJacobian(TargetJacobian::IDEAL_EQ_SIZE, MPI_COMM_WORLD);
    }
    else if (tjtype == 3)
    {
       tj    = new TargetJacobian(TargetJacobian::IDEAL_INIT_SIZE, MPI_COMM_WORLD);
    }
    else if (tjtype == 4)
    {
       tj    = new TargetJacobian(TargetJacobian::IDEAL_EQ_SCALE_SIZE, MPI_COMM_WORLD);
    }
    else
    {
        if (myid==0)
        {
            cout << tjtype;
            cout << "You did not choose a valid option\n";
            cout << "Target Jacobian will default to IDEAL\n";
        }
    }

    //Metrics for 2D & 3D
    double tauval = -0.1;
    if (dim==2)
    {
        if (myid==0)
        {
            cout << "Choose optimization metric:\n"
            << "1  : |T|^2 \n"
            << "     shape.\n"
            << "2  : 0.5 |T|^2 / tau  - 1 \n"
            << "     shape, condition number.\n"
            << "7  : |T - T^-t|^2 \n"
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
        }
       double tauval = -0.5;
       MPI_Bcast(&modeltype, num_procs, MPI_INT, 0, MPI_COMM_WORLD);
       model    = new TMOPHyperelasticModel001;
       if (modeltype == 1)
       {
          model = new TMOPHyperelasticModel001;
       }
       else if (modeltype == 2)
       {
          model = new TMOPHyperelasticModel002;
       }
       else if (modeltype == 7)
       {
          model = new TMOPHyperelasticModel007;
       }
       else if (modeltype == 22)
       {
          model = new TMOPHyperelasticModel022(tauval);
       }
       else if (modeltype == 50)
       {
           model = new TMOPHyperelasticModel050;
       }
       else if (modeltype == 52)
       {
          model = new TMOPHyperelasticModel252(tauval);
       }
       else if (modeltype == 55)
       {
           model = new TMOPHyperelasticModel055;
       }
       else if (modeltype == 56)
       {
           model = new TMOPHyperelasticModel056;
       }
       else if (modeltype == 58)
       {
           model = new TMOPHyperelasticModel058;
       }
       else if (modeltype == 77)
       {
           model = new TMOPHyperelasticModel077;
       }
       else if (modeltype == 211)
       {
           model = new TMOPHyperelasticModel211;
       }
       else
       {
           if (myid==0)
           {
               cout << "You did not choose a valid option\n";
               cout << "Model type will default to 1\n";
               cout << modeltype;
           }
       }
    }
    else
    {  //Metrics for 3D
        if (myid==0)
        {
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
            cin >> modeltype;
        }
       double tauval = -0.1;
       MPI_Bcast(&modeltype, num_procs, MPI_INT, 0, MPI_COMM_WORLD);
       model    = new TMOPHyperelasticModel302;
       if (modeltype == 1)
       {
          model = new TMOPHyperelasticModel301;
       }
       if (modeltype == 2)
       {
          model = new TMOPHyperelasticModel302;
       }
       else if (modeltype == 3)
       {
          model = new TMOPHyperelasticModel303;
       }
       else if (modeltype == 15)
       {
          model = new TMOPHyperelasticModel315;
       }
       else if (modeltype == 16)
       {
          model = new TMOPHyperelasticModel316;
       }
       else if (modeltype == 21)
       {
          model = new TMOPHyperelasticModel321;
       }
       else if (modeltype == 52)
       {
          model = new TMOPHyperelasticModel352(tauval);
       }
       else
       {
           if (myid==0)
           {
               cout << "You did not choose a valid option\n";
               cout << "Model type will default to 2\n";
               cout << modeltype;
           }
       }
    }
        
    logvec[2]=tjtype;
    logvec[3]=modeltype;

    tj->SetNodes(x);
    tj->SetInitialNodes(x0);
    HyperelasticNLFIntegrator *he_nlf_integ;
    he_nlf_integ = new HyperelasticNLFIntegrator(model, tj);
        
    int ptflag = 1; //if 1 - GLL, else uniform
    int nptdir = 8; //number of sample points in each direction
    const IntegrationRule *ir =
          &IntRulesLo.Get(fespace->GetFE(0)->GetGeomType(),nptdir); //this for GLL points "LO"
    he_nlf_integ->SetIntegrationRule(*ir);
    if (ptflag==1) {
        if (myid==0)
        {
            cout << "Sample point distribution is GLL based\n";
        }
    }
    else {
       const IntegrationRule *ir =
             &IntRulesCU.Get(fespace->GetFE(0)->GetGeomType(),nptdir); //this for uniform points "CU"
       he_nlf_integ->SetIntegrationRule(*ir);
        if (myid==0)
        {
            cout << "Sample point distribution is uniformly spaced\n";
        }
    }
        
    nf_integ = he_nlf_integ;
    ParNonlinearForm a(fespace);
        
    // This is for trying a combo of two integrators.
    FunctionCoefficient rhs_coef(weight_fun);
    const int combomet = 1;
    if (combomet==1)
    {
       c = new ConstantCoefficient(0.5);  //weight of original metric
       he_nlf_integ->SetCoefficient(*c);
       nf_integ = he_nlf_integ;
       a.AddDomainIntegrator(nf_integ);

       tj2    = new TargetJacobian(TargetJacobian::IDEAL_EQ_SCALE_SIZE, MPI_COMM_WORLD);
       model2 = new TMOPHyperelasticModel077;
       tj2->SetNodes(x);
       tj2->SetInitialNodes(x0);
       he_nlf_integ2 = new HyperelasticNLFIntegrator(model2, tj2);
       he_nlf_integ2->SetIntegrationRule(*ir);

       //c2 = new ConstantCoefficient(2.5);
       //he_nlf_integ2->SetCoefficient(*c2);     //weight of new metric

       he_nlf_integ2->SetCoefficient(rhs_coef); //weight of new metric as function
       cout << "You have added a combo metric \n";
       a.AddDomainIntegrator(he_nlf_integ2);
    }
    else
    {
       a.AddDomainIntegrator(nf_integ);
    }
        
    InterpolateHyperElasticModel(*model, *tj, *pmesh, metric);
    osockstream sol_sock2(visport, vishost);
    sol_sock2 << "solution\n";
    pmesh->PrintAsOne(sol_sock2);
    metric.SaveAsOne(sol_sock2);
    sol_sock2.send();
    sol_sock2 << "keys " << "JRem" << endl;
    
    // Set essential vdofs by hand for x = 0 and y = 0 (2D). These are
    // attributes 1 and 2.
    const int nd  = x.ParFESpace()->GetBE(0)->GetDof();
    int n = 0;
    for (int i = 0; i < pmesh->GetNBE(); i++)
    {
       const int attr = pmesh->GetBdrElement(i)->GetAttribute();
       if (attr == 1 || attr == 2) { n += nd; }
       if (attr == 3) { n += nd * dim; }
    }
    Array<int> ess_vdofs(n), vdofs;
    n = 0;
    for (int i = 0; i < pmesh->GetNBE(); i++)
    {
       const int attr = pmesh->GetBdrElement(i)->GetAttribute();

       x.FESpace()->GetBdrElementVDofs(i, vdofs);
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
    if (myid==0)
    {
        cout <<
            "Allow boundary node movement:\n"
            "1 - yes\n"
            "2 - no\n"
            " --> " << flush;
        cin >> bndrflag;
    }
    MPI_Bcast(&bndrflag, num_procs, MPI_INT, 0, MPI_COMM_WORLD);
    tj    = new TargetJacobian(TargetJacobian::IDEAL);
    if (bndrflag != 1)
    {
       //k10partend - the three lines below used to be uncommented
       Array<int> ess_bdr(pmesh->bdr_attributes.Max());
       ess_bdr = 1;
       a.SetEssentialBC(ess_bdr);
    }
    logvec[4]=bndrflag;
    int lsmtype;
    if (myid==0)
    {
        cout << "Choose linear smoother:\n"
            "0) l1-Jacobi\n"
            "1) CG\n"
            "2) MINRES\n" << " --> \n" << flush;
        cin >> lsmtype;
    }
    MPI_Bcast(&lsmtype, num_procs, MPI_INT, 0, MPI_COMM_WORLD);
    Solver *S;
    logvec[5]=lsmtype;
    int lsmits;
    const double rtol = 1e-12;
    if (lsmtype == 0)
    {
        if (myid==0)
        {
            cout << "Enter number of linear smoothing iterations -->\n " << flush;
            cin >> lsmits;
        }
       MPI_Bcast(&lsmits, num_procs, MPI_INT, 0, MPI_COMM_WORLD);
       S = new DSmoother(1, 1., lsmits);
    }
    else if (lsmtype == 1)
    {
        if (myid==0)
        {
            cout << "Enter number of CG smoothing iterations -->\n " << flush;
            cin >> lsmits;
        }
       MPI_Bcast(&lsmits, num_procs, MPI_INT, 0, MPI_COMM_WORLD);
       CGSolver *cg = new CGSolver;
       cg->SetMaxIter(lsmits);
       cg->SetRelTol(rtol);
       cg->SetAbsTol(0.0);
       cg->SetPrintLevel(3);
       S = cg;
    }
    else
    {
        if (myid==0)
        {
            cout << "Enter number of MINRES smoothing iterations -->\n " << flush;
            cin >> lsmits;
        }
       MPI_Bcast(&lsmits, num_procs, MPI_INT, 0, MPI_COMM_WORLD);
       MINRESSolver *minres = new MINRESSolver(MPI_COMM_WORLD);
       minres->SetMaxIter(lsmits);
       minres->SetRelTol(rtol);
       minres->SetAbsTol(0.0);
       minres->SetPrintLevel(3);
       S = minres;
    }
    logvec[6]=lsmits;
    int newits;
    if (myid==0)
    {
        cout << "Enter number of Newton iterations -->\n " << flush;
        cin >> newits;
    }
    MPI_Bcast(&newits, num_procs, MPI_INT, 0, MPI_COMM_WORLD);
    logvec[7]=newits;
    logvec[8]=a.GetEnergy(x);
    if (myid==0)
    {
        cout << "Initial strain energy : " << logvec[8] << endl;
    }
    
        
    // save original
    Vector xsav = x;
    //set value of tau_0 for metric 22 and get min jacobian for mesh statistic
    tauval = 1.e+6;
    double minjaco;
    const int NE = pmesh->GetNE();
    for (int i = 0; i < NE; i++)
    {
       const FiniteElement &fe = *x.ParFESpace()->GetFE(i);
       const int dim = fe.GetDim();
       int nsp = ir->GetNPoints();
       int dof = fe.GetDof();
       DenseTensor Jtr(dim, dim, nsp);
       DenseMatrix dshape(dof, dim), pos(dof, dim);
       Array<int> xdofs(dof * dim);
       Vector posV(pos.Data(), dof * dim);
       x.ParFESpace()->GetElementVDofs(i, xdofs);
       x.GetSubVector(xdofs, posV);
       for (int j = 0; j < nsp; j++)
       {
          //cout << "point number " << j << "\n";
          fe.CalcDShape(ir->IntPoint(j), dshape);
          MultAtB(pos, dshape, Jtr(j));
          double det = Jtr(j).Det();
          tauval = min(tauval,det);
       }
    }
    minjaco = tauval;
    MPI_Allreduce(&tauval, &minjaco, 1, MPI_DOUBLE, MPI_MIN,MPI_COMM_WORLD);
    
    if (myid==0)
    {
        cout << "minimum jacobian in the original mesh is " << minjaco << " \n";
    }
    int newtonits = 0;
    if (tauval>0)
    {
       tauval = 1e-1;
       RelaxedNewtonSolver *newt= new RelaxedNewtonSolver(MPI_COMM_WORLD);
       newt->SetPreconditioner(*S);
       newt->SetMaxIter(newits);
       newt->SetRelTol(rtol);
       newt->SetAbsTol(0.0);
       newt->SetPrintLevel(1);
       newt->SetOperator(a);
       Vector b;

       Vector X(fespace->TrueVSize());
       fespace->GetRestrictionMatrix()->Mult(x, X);

        if (myid==0)
        {
            cout << " Relaxed newton solver will be used \n";
        }
       newt->Mult2(b, X, *pmesh, *ir, &newtonits, a);
       if (!newt->GetConverged())
       {
           if (myid==0)
           {
               cout << "NewtonIteration : rtol = " << rtol << " not achieved."
               << endl;
           }
       }
       fespace->Dof_TrueDof_Matrix()->Mult(X, x);
    }
    else
    {
       if (dim==2 && modeltype!=52) {
          model = new TMOPHyperelasticModel022(tauval);
           if (myid==0)
           {
               cout << "model 22 will be used since mesh has negative jacobians\n";
           }
       }
       else if (dim==3)
       {
          model = new TMOPHyperelasticModel352(tauval);
           if (myid==0)
           {
               cout << "model 52 will be used since mesh has negative jacobians\n";
           }
       }
       tauval -= 0.01;
       DescentNewtonSolver *newt= new DescentNewtonSolver;
       newt->SetPreconditioner(*S);
       newt->SetMaxIter(newits);
       newt->SetRelTol(rtol);
       newt->SetAbsTol(0.0);
       newt->SetPrintLevel(1);
       newt->SetOperator(a);
       Vector b;
        if (myid==0)
        {
            cout << " There are inverted elements in the mesh \n";
            cout << " Descent newton solver will be used \n";
        }
       newt->Mult2(b, x, *pmesh, *ir, &newtonits, a, tauval);
       if (!newt->GetConverged())
           if (myid==0)
           {
               cout << "NewtonIteration : rtol = " << rtol << " not achieved."
               << endl;
           }
    }

    logvec[9] = a.GetEnergy(x);
    if (myid == 0)
    {
       cout << "Final strain energy   : "      << logvec[9] << endl;
       cout << "Initial strain energy was  : " << logvec[8] << endl;
       cout << "% change is  : " << (logvec[8]-logvec[9])*100/logvec[8] << endl;
    }
    logvec[10] = (logvec[8]-logvec[9])*100/logvec[8];
    logvec[11] = newtonits;

    if (tj)
    {
       InterpolateHyperElasticModel(*model, *tj, *pmesh, metric);
       osockstream sol_sock(visport, vishost);
       sol_sock << "solution\n";
       pmesh->PrintAsOne(sol_sock);
       metric.SaveAsOne(sol_sock);
       sol_sock.send();
       sol_sock << "keys " << "JRem" << endl;
    }

    // 17. Get some mesh statistics
    double minjacs = 1.e+100;
    for (int i = 0; i < NE; i++)
    {
       const FiniteElement &fe = *x.ParFESpace()->GetFE(i);
       const int dim = fe.GetDim();
       int nsp = ir->GetNPoints();
       int dof = fe.GetDof();
       DenseTensor Jtr(dim, dim, nsp);
       DenseMatrix dshape(dof, dim), pos(dof, dim);
       Array<int> xdofs(dof * dim);
       Vector posV(pos.Data(), dof * dim);
       x.ParFESpace()->GetElementVDofs(i, xdofs);
       x.GetSubVector(xdofs, posV);
       for (int j = 0; j < nsp; j++)
       {
          //cout << "point number " << j << "\n";
          fe.CalcDShape(ir->IntPoint(j), dshape);
          MultAtB(pos, dshape, Jtr(j));
          double det = Jtr(j).Det();
          minjacs = min(minjacs,det);
       }
    }
    tauval = minjacs;
    MPI_Allreduce(&tauval, &minjacs, 1, MPI_DOUBLE, MPI_MIN,
                  MPI_COMM_WORLD);
    if (myid==0)
    {
        cout << "minimum jacobian before and after smoothing " << minjaco << " " << minjacs << " \n";
    }
    

    delete S;
    delete model;
    delete c;
    
    // Define mesh displacement
    x0 -= *x;
    
    // 15. Save the smoothed mesh to a file. This output can be viewed later
    //     using GLVis: "glvis -m smoothed.mesh".
    {
        //ofstream mesh_ofs("smoothed.mesh");
        //mesh_ofs.precision(14);
        //pmesh->Print(mesh_ofs);
        
        ostringstream mesh_name, velo_name, ee_name;
        mesh_name << "smoothed." << setfill('0') << setw(6) << myid;
        ofstream mesh_ofs(mesh_name.str().c_str());
        mesh_ofs.precision(8);
        pmesh->Print(mesh_ofs);
        
    }

    
    // 17. Free the used memory.
    delete fespace;
    delete fec;
    delete pmesh;
    
    // Execution time
    if (myid == 0)
    {
       double tstop_s = clock();
       cout << "The total time ist took for this example's execution is: "
            << (tstop_s-tstart_s)/1000000. << " seconds\n";
    }

    // write log to text file-k10
    int logflag = 100;
    if (myid == 0)
    {
       cout << "How do you want to write log to a new file:\n"
               "0) New file\n"
               "1) Append\n" << " --> " << flush;
       cin >> logflag;

       ofstream outputFile;
       if (logflag==0)
       {
           outputFile.open("logfile.txt");
           outputFile << mesh_file << " ";
       }
       else
       {
           outputFile.open("logfile.txt",fstream::app);
           outputFile << "\n" << mesh_file << " ";
       }

       for (int i=0; i<12; i++)
       {
           outputFile << logvec[i] << " ";
       }
       outputFile.close();
       // pause 1 second.. this is because X11 cannot restart if you push stuff too soon
    }
    int logflagg;
    MPI_Allreduce(&logflag, &logflagg, 1, MPI_DOUBLE, MPI_MIN,
                  MPI_COMM_WORLD);
    
    MPI_Finalize();
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
    l2 = 0;
    //This is for tipton
    if (r2 >= 0.10 && r2 <= 0.15 )
    {
        l2 = 1;
    }
    //l2 = 0.01+0.5*std::tanh((r2-0.13)/0.01)-(0.5*std::tanh((r2-0.14)/0.01))
    //        +0.5*std::tanh((r2-0.21)/0.01)-(0.5*std::tanh((r2-0.22)/0.01));
    l2 = 0.1+0.5*std::tanh((r2-0.12)/0.005)-(0.5*std::tanh((r2-0.13)/0.005))
    +0.5*std::tanh((r2-0.18)/0.005)-(0.5*std::tanh((r2-0.19)/0.005));
    //l2 = 10*r2;
    /*
    //This is for blade
    int l4 = 0, l3 = 0;
    double xmin, xmax, ymin, ymax,dx ,dy;
    xmin = 0.9; xmax = 1.;
    ymin = -0.2; ymax = 0.;
    dx = (xmax-xmin)/2;dy = (ymax-ymin)/2;
    
    if (abs(x(0)-xmin)<dx && abs(x(0)-xmax)<dx) {
        l4 = 1;
    }
    if (abs(x(1)-ymin)<dy && abs(x(1)-ymax)<dy) {
        l3 = 1;
    }
    l2 = l4*l3;
     */
    
    // This is for square perturbed
    /*
     if (r2 < 0.5) {
        l2 = 1;
    }
     */
    
    return l2;
}
