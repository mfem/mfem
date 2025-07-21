#include "mfem.hpp"
#include "mtop_solvers.hpp"

#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


class DensCoeff:public Coefficient
{
public:
    DensCoeff(real_t ll=1):len(ll)
    {}

    virtual
        ~DensCoeff(){};

    virtual
        real_t Eval(ElementTransformation &T,
                        const IntegrationPoint &ip) override
    {
        T.SetIntPoint(&ip);
        real_t x[3];
        Vector transip(x, 3); transip=0.0;
        T.Transform(ip, transip);

        real_t l=transip.Norml2();
        return ((sin(len*transip(0))*cos(len*transip(1)))>real_t(-0.1));
    }
private:
    real_t len;
};


int main(int argc, char *argv[])
{
    // 1. Initialize MPI and HYPRE.
    Mpi::Init();
    int num_procs = Mpi::WorldSize();
    int myid = Mpi::WorldRank();
    Hypre::Init();
    // 2. Parse command-line options.
    const char *mesh_file = "canti_2D_6.msh";
    int order = 2;
    bool static_cond = false;
    bool pa = false;
    bool fa = false;
    const char *device_config = "cpu";
    bool visualization = true;
    bool algebraic_ceed = false;
    real_t filter_radius=real_t(0.1);

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
                   "Mesh file to use.");
    args.AddOption(&order, "-o", "--order",
                   "Finite element order (polynomial degree) or -1 for"
                   " isoparametric space.");
    args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                   "--no-static-condensation", "Enable static condensation.");
    args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                   "--no-partial-assembly", "Enable Partial Assembly.");
    args.AddOption(&fa, "-fa", "--full-assembly", "-no-fa",
                   "--no-full-assembly", "Enable Full Assembly.");
    args.AddOption(&device_config, "-d", "--device",
                   "Device configuration string, see Device::Configure().");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                   "--no-visualization",
                   "Enable or disable GLVis visualization.");
    args.AddOption(&filter_radius, "-fr", "--filter",
                   "Set the filter radius.");
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

    // 3. Enable hardware devices such as GPUs, and programming models such as
    //    CUDA, OCCA, RAJA and OpenMP based on command line options.
    Device device(device_config);
    if (myid == 0) { device.Print(); }

    // 4. Read the (serial) mesh from the given mesh file on all processors.  We
    //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
    //    and volume meshes with the same code.
    Mesh mesh(mesh_file, 1, 1);
    int dim = mesh.Dimension();

    // 5. Refine the serial mesh on all processors to increase the resolution. In
    //    this example we do 'ref_levels' of uniform refinement. We choose
    //    'ref_levels' to be the largest number that gives a final mesh with no
    //    more than 10,000 elements.
    {
        int ref_levels =
            (int)floor(log(1000./mesh.GetNE())/log(2.)/dim);
        for (int l = 0; l < ref_levels; l++)
        {
            mesh.UniformRefinement();
        }
    }


    // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
    //    this mesh further in parallel to increase the resolution. Once the
    //    parallel mesh is defined, the serial mesh can be deleted.
    ParMesh pmesh(MPI_COMM_WORLD, mesh);
    mesh.Clear();
    {
        int par_ref_levels = 1;
        for (int l = 0; l < par_ref_levels; l++)
        {
            pmesh.UniformRefinement();
        }
    }

    //allocate the fiter
    FilterOperator* filt=new FilterOperator(filter_radius,&pmesh);
    //set the boundary conditions
    filt->AddBC(1,1.0);
    filt->AddBC(2,1.0);
    filt->AddBC(3,1.0);
    filt->AddBC(4,1.0);
    filt->AddBC(5,1.0);
    filt->AddBC(6,1.0);
    filt->AddBC(7,1.0);
    filt->AddBC(8,0.0);
    //allocate the slover after setting the BC and before applying the filter
    filt->Assemble();




    //define the control field
    ParGridFunction odens(filt->GetDesignFES());
    //define the filtered field
    ParGridFunction fdens(filt->GetFilterFES());



    //create initial density distribution
    DensCoeff odc(2*M_PI); //define the coefficient
    odens.ProjectCoefficient(odc); odens=1.0;
    odens.SetTrueVector();
    Vector odv(filt->GetDesignFES()->TrueVSize()); odens.GetTrueDofs(odv);


    Vector fdv(filt->GetFilterFES()->TrueVSize()); fdv=0.0; //define the true vector of the filtered field
    //filter the density field
    filt->Mult(odv,fdv);
    //set the filtered pargrid function using the filtered true vector
    fdens.SetFromTrueDofs(fdv);

    //set the elasticity solver
    IsoLinElasticSolver* elsolver=new IsoLinElasticSolver(&pmesh,order);
    //set the boundary conditions [1,2,..,7]
    for(int i=2;i<8;i++){
        elsolver->AddDispBC(i,4,0.0);
    }

    //set surface load
    elsolver->AddSurfLoad(1,0.0,1.0);

    //define material and interpolation parameters coefficient factory
    IsoComplCoef icc;
    //set density, solution, and interpolation parameters for the compliance coefficient factory
    icc.SetGridFunctions(&fdens,&(elsolver->GetDisplacements()));
    icc.SetMaterial(1e-3,1.0,0.2);

    //set the material to the elasticity solver
    elsolver->SetMaterial(*(icc.GetE()),*(icc.GetNu()));
    //solve the discrete elastic system
    elsolver->Assemble();
    elsolver->FSolve();

    //extract the solution
    ParGridFunction& sol=elsolver->GetDisplacements();

    //project the coefficients
    ParGridFunction egf(filt->GetFilterFES()); egf.ProjectCoefficient(*(icc.GetE()));
    ParGridFunction ngf(filt->GetFilterFES()); ngf.ProjectCoefficient(*(icc.GetNu()));


    //compute the gradients
    ParGridFunction ggf(filt->GetFilterFES()); ggf.ProjectCoefficient(*(icc.GetGradIsoComp()));
    ParGridFunction cgf(filt->GetFilterFES()); cgf.ProjectCoefficient(icc);

    //unit grid function
    ParGridFunction onegf(filt->GetFilterFES());
    onegf = 1.0;

    //compute compliance
    double vcompl=0.0;
    {
        ParLinearForm compl_form(filt->GetFilterFES());
        compl_form.AddDomainIntegrator(new DomainLFIntegrator(icc));
        compl_form.Assemble();
        vcompl=compl_form(onegf);
    }
    if(myid==0){
        std::cout<<"Reference Compliance = "<<vcompl<<std::endl;
    }

    //true gradient vector
    Vector fgrad(filt->GetFilterFES()->GetTrueVSize()); fgrad=0.0;
    //compute gradient
    {
        ParLinearForm compl_grad(filt->GetFilterFES());
        compl_grad.AddDomainIntegrator(new DomainLFIntegrator(*(icc.GetGradIsoComp())));
        compl_grad.Assemble();
        compl_grad.ParallelAssemble(fgrad);

    }


    //FD check on the filtered field
    {

        mfem::Vector prtv; prtv.SetSize(fdens.GetTrueVector().Size());
        mfem::Vector tmpv; tmpv.SetSize(fdens.GetTrueVector().Size());

        prtv.Randomize();

        double nd=mfem::InnerProduct(pmesh.GetComm(),prtv,prtv);
        double td=mfem::InnerProduct(pmesh.GetComm(),prtv,fgrad);


        td=td/nd;
        double lsc=1.0;
        double lqoi;

        fdens.GetTrueDofs(fdv);

        //perturb the density
        add(prtv,fdv,tmpv);
        //set the perturbed density
        fdens.SetFromTrueDofs(tmpv);
        //solve the linear elastic system with the perturbed density
        elsolver->FSolve();
        //compute again the compliance
        {
            ParLinearForm compl_form(filt->GetFilterFES());
            compl_form.AddDomainIntegrator(new DomainLFIntegrator(icc));
            compl_form.Assemble();
            lqoi=compl_form(onegf);
            double ld=(lqoi-vcompl)/lsc;
            if(myid==0){
                std::cout << "dx=" << lsc <<" FD approximation=" << ld/nd
                          << " adjoint gradient=" << td
                          << " err=" << std::fabs(ld/nd-td) << std::endl;
            }
        }

        for(int l=0;l<2;l++){
            lsc/=10.0;
            prtv/=10.0;

            //perturb the density
            add(prtv,fdv,tmpv);
            //set the perturbed density
            fdens.SetFromTrueDofs(tmpv);
            //solve the linear elastic system with the perturbed density
            elsolver->FSolve();
            //compute again the compliance
            {
                ParLinearForm compl_form(filt->GetFilterFES());
                compl_form.AddDomainIntegrator(new DomainLFIntegrator(icc));
                compl_form.Assemble();
                lqoi=compl_form(onegf);
                double ld=(lqoi-vcompl)/lsc;
                if(myid==0){
                    std::cout << "dx=" << lsc <<" FD approximation=" << ld/nd
                              << " adjoint gradient=" << td
                              << " err=" << std::fabs(ld/nd-td) << std::endl;
                }
            }
        }

    }


    //true gradient vector
    Vector ograd(odens.GetTrueVector().Size()); ograd=0.0;
    //gradient grid function
    ParGridFunction ogf(filt->GetDesignFES());
    //Apply the adjoint filter operation
    filt->AFilter(icc.GetGradIsoComp(),ogf);
    //get the true vector of the gradient
    ogf.GetTrueDofs(ograd);

    //alternative way to compute the gradient with repsect to ograd
    //filt->MultTranspose(fgrad,ograd);

    //FD check on the original density field
    {
        mfem::Vector prtv; prtv.SetSize(odens.GetTrueVector().Size());
        mfem::Vector tmpv; tmpv.SetSize(odens.GetTrueVector().Size());

        prtv.Randomize(); //prtv*=((1e-3)-(1e-9));

        double nd=mfem::InnerProduct(pmesh.GetComm(),prtv,prtv);
        double td=mfem::InnerProduct(pmesh.GetComm(),prtv,ograd);


        td=td/nd;
        double lsc=1.0;
        double lqoi;

        odens.GetTrueDofs(odv);

        //perturb the density
        add(prtv,odv,tmpv);
        //filter the density
        filt->Mult(tmpv,fdv);
        //set the filtered density
        fdens.SetFromTrueDofs(fdv);
        //solve the linear elastic system with the perturbed density
        elsolver->FSolve();
        //compute again the compliance
        {
            ParLinearForm compl_form(filt->GetFilterFES());
            compl_form.AddDomainIntegrator(new DomainLFIntegrator(icc));
            compl_form.Assemble();
            lqoi=compl_form(onegf);
            double ld=(lqoi-vcompl)/lsc;
            if(myid==0){
                std::cout << "dx=" << lsc <<" FD approximation=" << ld/nd
                          << " adjoint gradient=" << td
                          << " err=" << std::fabs(ld/nd-td) << std::endl;
            }
        }

        for(int l=0;l<2;l++){
            lsc/=10.0;
            prtv/=10.0;

            //perturb the density
            add(prtv,odv,tmpv);
            //filter the density
            filt->Mult(tmpv,fdv);
            //set the filtered density
            fdens.SetFromTrueDofs(fdv);
            //solve the linear elastic system with the perturbed density
            elsolver->FSolve();
            //compute again the compliance
            {
                ParLinearForm compl_form(filt->GetFilterFES());
                compl_form.AddDomainIntegrator(new DomainLFIntegrator(icc));
                compl_form.Assemble();
                lqoi=compl_form(onegf);
                double ld=(lqoi-vcompl)/lsc;
                if(myid==0){
                    std::cout << "dx=" << lsc <<" FD approximation=" << ld/nd
                              << " adjoint gradient=" << td
                              << " err=" << std::fabs(ld/nd-td) << std::endl;
                }
            }
        }
    }




    {
        ParaViewDataCollection paraview_dc("filt", &pmesh);
        paraview_dc.SetPrefixPath("ParaView");
        paraview_dc.SetLevelsOfDetail(order);
        paraview_dc.SetDataFormat(VTKFormat::BINARY);
        paraview_dc.SetHighOrderOutput(true);
        paraview_dc.SetCycle(0);
        paraview_dc.SetTime(0.0);
        paraview_dc.RegisterField("odens",&odens);
        paraview_dc.RegisterField("fdens",&fdens);
        paraview_dc.RegisterField("sol",&sol);
        paraview_dc.RegisterField("E",&egf);
        paraview_dc.RegisterField("nu",&ngf);
        paraview_dc.RegisterField("grad",&ggf);
        paraview_dc.RegisterField("energy",&cgf);
        //paraview_dc.RegisterField("cdens",&cdens);
        paraview_dc.Save();
    }

    delete elsolver;
    delete filt;

    Mpi::Finalize();
    return 0;
}
