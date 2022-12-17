
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "coefficients.hpp"
#include "elplast.hpp"

using namespace std;
using namespace mfem;


class MyVectorCoeff:public VectorCoefficient
{
public:
    MyVectorCoeff():VectorCoefficient(3)
    {
        SetTime(1.0);
    }

    virtual
    ~MyVectorCoeff(){}


    void Eval(Vector &V, ElementTransformation &T,
               const IntegrationPoint &ip)
    {
        Vector xx(3);
        T.Transform(ip,xx);
        V=xx; V*=0.0;
        V[0]=xx[0]*GetTime();
        //V[1]=xx[0]*GetTime();
        //V[2]=xx[2]*GetTime();
    }

private:
};

class MyForceCoeff:public VectorCoefficient
{
public:
    MyForceCoeff():VectorCoefficient(3)
    {
        SetTime(1.0);
    }

    virtual
    ~MyForceCoeff(){}


    void Eval(Vector &V, ElementTransformation &T,
               const IntegrationPoint &ip)
    {
        Vector xx(3);
        T.Transform(ip,xx);
        V=xx; V*=0.0;
        if(xx[0]>1.9){V[0]=GetTime();}
        //V[1]=xx[1]*GetTime();
        //V[2]=xx[2]*GetTime();
    }

private:
};


class MyBlockPrec:public BlockDiagonalPreconditioner
{
public:
    MyBlockPrec(const Array<int> & offsets, ParFiniteElementSpace* fesel):BlockDiagonalPreconditioner(offsets)
    {
        pr1=new HypreBoomerAMG();
        pr1->SetPrintLevel(0);
        pr1->SetElasticityOptions(fesel);
        pr2=new HypreBoomerAMG();
        pr2->SetPrintLevel(0);

    }

    virtual void SetOperator(const Operator &op)
    {
        const BlockOperator& blo=static_cast<const BlockOperator&>(op);
        pr1->SetOperator(blo.GetBlock(0,0));
        pr2->SetOperator(blo.GetBlock(1,1));
        SetDiagonalBlock(0,pr1);
        SetDiagonalBlock(1,pr2);
    }

    virtual ~MyBlockPrec()
    {
        delete pr1;
        delete pr2;
    }

private:
    HypreBoomerAMG* pr1;
    HypreBoomerAMG* pr2;

};




int main(int argc, char *argv[])
{
    // 0. Initialize MPI and HYPRE.
    Mpi::Init(argc, argv);
    int num_procs = Mpi::WorldSize();
    int myid = Mpi::WorldRank();
    Hypre::Init();

    // 1. Parse command-line options.
    const char *mesh_file = "../data/beam-tri.mesh";
    int serial_ref_levels = 0;
    int order = 2;
    //bool static_cond = false;
    bool static_cond = true;
    bool visualization = 1;

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
                   "Mesh file to use.");
    args.AddOption(&serial_ref_levels, "-rs", "--refine-serial",
                   "Number of uniform serial refinements (before parallel"
                   " partitioning)");
    args.AddOption(&order, "-o", "--order",
                   "Finite element order (polynomial degree).");
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
       return 1;
    }
    if (myid == 0)
    {
       args.PrintOptions(cout);
    }

    // 2. Read the mesh from the given mesh file. We can handle triangular,
    //    quadrilateral, tetrahedral, and hexahedral meshes with the same code.
    Mesh mesh(mesh_file, 1, 1);
    int dim = mesh.Dimension();
    MFEM_VERIFY(mesh.SpaceDimension() == dim, "invalid mesh");

    // 3. Refine the mesh before parallel partitioning. Since a NURBS mesh can
    //    currently only be refined uniformly, we need to convert it to a
    //    piecewise-polynomial curved mesh. First we refine the NURBS mesh a bit
    //    more and then project the curvature to quadratic Nodes.
    if (mesh.NURBSext && serial_ref_levels == 0)
    {
       serial_ref_levels = 2;
    }
    for (int i = 0; i < serial_ref_levels; i++)
    {
       mesh.UniformRefinement();
    }
    if (mesh.NURBSext)
    {
       mesh.SetCurvature(2);
    }
    mesh.EnsureNCMesh();

    ParMesh pmesh(MPI_COMM_WORLD, mesh);
    mesh.Clear();

    // 4. Define a finite element space on the mesh. The polynomial order is
    //    one (linear) by default, but this can be changed on the command line.
    H1_FECollection fec(order, dim);
    ParFiniteElementSpace ufespace(&pmesh, &fec, dim);
    ParFiniteElementSpace efespace(&pmesh, &fec, 1);

    QuadratureSpace  qfes(&pmesh,3*order);
    QuadratureFunction kappa;
    QuadratureFunction eep;

    kappa.SetSpace(&qfes,1); kappa=0.0;
    eep.SetSpace(&qfes,6);   eep=0.0;

    ParGridFunction u(&ufespace); u=0.0;
    ParGridFunction ep(&efespace); ep=0.0;
    MyVectorCoeff co; co.SetTime(0.5);
    MyForceCoeff  fo; fo.SetTime(0.5);

    u.ProjectCoefficient(co);

    Array<ParFiniteElementSpace *> pfes(2);
    pfes[0]=&ufespace;
    pfes[1]=&efespace;
    ParBlockNonlinearForm* bnl=new ParBlockNonlinearForm();
    bnl->SetParSpaces(pfes);

    //add the integrator
    NLElPlastIntegrator* itgr=new NLElPlastIntegrator(1,0.2,1.0,0.01);
    itgr->SetPlasticStrains(eep,kappa);
    //itgr->SetForce(fo);
    bnl->AddDomainIntegrator(itgr);

    Array<int> offset;
    BlockVector bv; bv.Update(bnl->GetBlockTrueOffsets()); bv=0.0;
    BlockVector rv; rv.Update(bnl->GetBlockTrueOffsets()); rv=0.0;

    u.GetTrueDofs(bv.GetBlock(0));

    bnl->Mult(bv,rv);

    BlockOperator& op=bnl->GetGradient(bv);


    // Define the essential boundary attributes
    Array<int> ess_bdr_u(pmesh.bdr_attributes.Max());
    Array<int> ess_bdr_e(pmesh.bdr_attributes.Max());
    Array<Array<int>*> ess_bdr(2);
    ess_bdr[0]=&ess_bdr_u;
    ess_bdr[1]=&ess_bdr_e;

    ess_bdr_u = 1; ess_bdr_u[0] = 1;
    ess_bdr_e = 0;

    Array<Vector*> nrhs(2);
    nrhs[0]=nullptr;
    nrhs[1]=nullptr;

    bnl->SetEssentialBC(ess_bdr,nrhs);


    ParGridFunction du(u);
    du.SetFromTrueDofs(bv.GetBlock(0));
    ep.SetFromTrueDofs(bv.GetBlock(1));

    // ParaView output.
    ParaViewDataCollection dacol("ParaView", &pmesh);
    dacol.SetLevelsOfDetail(order);
    dacol.SetDataFormat(VTKFormat::ASCII);
    dacol.RegisterField("odisp", &u);
    dacol.RegisterField("ddisp", &du);
    dacol.RegisterField("ep",&ep);

    /*
    dacol.SetTime(1.0);
    dacol.SetCycle(1);
    dacol.Save();
    */




    /*
    BlockDiagonalPreconditioner* blprec=new BlockDiagonalPreconditioner(bnl->GetBlockTrueOffsets());
    HypreBoomerAMG* pr1=new HypreBoomerAMG();
    pr1->SetPrintLevel(1);
    HypreBoomerAMG* pr2=new HypreBoomerAMG();
    pr2->SetPrintLevel(1);
    */


    MyBlockPrec* blprec=new MyBlockPrec(bnl->GetBlockTrueOffsets(), &ufespace);

    /*
    GMRESSolver* gmres = new GMRESSolver(MPI_COMM_WORLD);
    gmres->SetAbsTol(1e-12);
    gmres->SetRelTol(1e-9);
    gmres->SetMaxIter(500);
    gmres->SetPrintLevel(2);
    gmres->SetPreconditioner(*blprec);
    */

    CGSolver* cg = new CGSolver(MPI_COMM_WORLD);
    cg->SetAbsTol(1e-12);
    cg->SetRelTol(1e-9);
    cg->SetMaxIter(500);
    cg->SetPrintLevel(2);
    cg->SetPreconditioner(*blprec);


    NewtonSolver* ns=new  NewtonSolver(MPI_COMM_WORLD);

    ns->iterative_mode = true;
    ns->SetSolver(*cg);
    ns->SetOperator(*bnl);
    ns->SetPrintLevel(1);
    ns->SetRelTol(1e-5);
    ns->SetAbsTol(1e-12);
    ns->SetMaxIter(8);

    u.GetTrueDofs(bv.GetBlock(0));
    //bv=0.0;
    Vector b;
    //ns->Mult(b,bv);

    for(int bi=0;bi<10;bi++){
        co.SetTime(0.4+bi*0.1);
        u.ProjectCoefficient(co);
        u.GetTrueDofs(bv.GetBlock(0));
        ns->Mult(b,bv);

        //update the plasticity vars
        itgr->SetUpdateFlag(true);
        bnl->Mult(bv,rv);
        itgr->SetUpdateFlag(false);

        u.SetFromTrueDofs(bv.GetBlock(0));
        ep.SetFromTrueDofs(bv.GetBlock(1));

        dacol.SetTime(1.0+bi);
        dacol.SetCycle(1+bi);
        dacol.Save();

    }


    delete ns;
    delete cg;
    delete blprec;

    MPI_Barrier(MPI_COMM_WORLD);



    delete bnl;




    return 0;
}
