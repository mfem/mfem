#include "mfem.hpp"
#include "marking.hpp"
#include "cut_integrators.hpp"

using namespace mfem;

//Level set function for sphere in 3D and circle in 2D
double sphere_ls(const Vector &x)
{
   double r2= x*x;
   return -sqrt(r2)+1.0;//the radius is 1.0
}


int main(int argc, char *argv[])
{
    // Initialize MPI and HYPRE.
    Mpi::Init(argc, argv);
    int myrank = Mpi::WorldRank();
    Hypre::Init();

    // Parse command-line options.
    const char *mesh_file = "../../data/star-q3.mesh";
    int order = 2;
    bool visualization = true;
    int ser_ref_levels = 0;
    int par_ref_levels = 0;

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
                   "Mesh file to use.");
    args.AddOption(&order, "-o", "--order",
                   "Finite element order (polynomial degree) or -1 for"
                   " isoparametric space.");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                   "--no-visualization",
                   "Enable or disable GLVis visualization.");
    args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                   "Number of times to refine the mesh uniformly in serial.");
    args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                   "Number of times to refine the mesh uniformly in parallel.");
    args.Parse();
    if (!args.Good())
    {
       if (myrank == 0)
       {
          args.PrintUsage(std::cout);
       }
       return 1;
    }
    if (myrank == 0) { args.PrintOptions(std::cout); }



    // Enable hardware devices such as GPUs, and programming models such as CUDA,
    // OCCA, RAJA and OpenMP based on command line options.
    Device device("cpu");
    if (myrank == 0) { device.Print(); }

    // Refine the mesh.
    Mesh mesh(mesh_file, 1, 1);
    int dim = mesh.Dimension();
    for (int lev = 0; lev < ser_ref_levels; lev++) { mesh.UniformRefinement(); }
    if (myrank == 0)
    {
       std::cout << "Number of elements: " << mesh.GetNE() << std::endl;
    }

    Coefficient *ls_coeff = nullptr;
    ls_coeff=new FunctionCoefficient(sphere_ls);

    // MPI distribution.
    ParMesh pmesh(MPI_COMM_WORLD, mesh);
    for (int lev = 0; lev < par_ref_levels; lev++) { pmesh.UniformRefinement(); }
    mesh.Clear();

    // Define a finite element space on the mesh. Here we use continuous Lagrange
    // finite elements of the specified order. If order < 1, we fix it to 1.
    if (order < 1) { order = 1; }
    H1_FECollection fec(order, dim);
    ParFiniteElementSpace sfespace(&pmesh, &fec, 1); //LS fe space
    ParGridFunction lsf(&sfespace);
    lsf.ProjectCoefficient(*ls_coeff);

    ParFiniteElementSpace pfespace(&pmesh, &fec, dim);
    pmesh.SetNodalFESpace(&pfespace);
    ParGridFunction nod(&pfespace);
    pmesh.SetNodalGridFunction(&nod);
    ParGridFunction ggf(&pfespace);

    ParNonlinearForm* nf=new ParNonlinearForm(&pfespace);

    ElementMarker elmark(pmesh,true,true,3);
    elmark.SetLevelSetFunction(lsf);
    Array<int> el_markers;
    elmark.MarkElements(el_markers);
    CutIntegrationRules* cint=new CutIntegrationRules(3,lsf,el_markers);

    CutVolLagrangeIntegrator* intgr=new CutVolLagrangeIntegrator(3);
    intgr->SetCutIntegration(cint);

    nf->AddDomainIntegrator(intgr);

    double vol=nf->GetEnergy(nod.GetTrueVector());
    if(myrank==0){ std::cout<<"vol="<<vol<<" err="<<vol-M_PI<<std::endl;}

    Vector grd; grd.SetSize(pfespace.TrueVSize()); grd=0.0;
    nf->Mult(nod.GetTrueVector(),grd); ggf.SetFromTrueDofs(grd);

    //check the gradients by FD
    {
        Vector vtmp;
        Vector prtv;
        Vector tmpv;

        prtv.SetSize(pfespace.TrueVSize());
        tmpv.SetSize(pfespace.TrueVSize());
        vtmp.SetSize(pfespace.TrueVSize());
        nod.GetTrueDofs(vtmp);


        prtv.Randomize();

        double nd=mfem::InnerProduct(pmesh.GetComm(), prtv,prtv);
        double td=mfem::InnerProduct(pmesh.GetComm(),prtv,grd);

        td=td/nd;
        double lsc=1.0;
        double lqoi;

        for(int l=0;l<10;l++){
            lsc/=10.0;
            prtv/=10.0;
            add(prtv,vtmp,tmpv);

            //calculate new integration rule
            nod.SetFromTrueDofs(tmpv);
            pmesh.DeleteGeometricFactors();
            lqoi=nf->GetEnergy(tmpv);
            double ld=(lqoi-vol)/lsc;
            if(myrank==0){
                std::cout << "dx=" << lsc <<" FD approximation=" << ld/nd
                                     << " true gradient=" << td<<" r="<< ld/(td*nd)
                                     << " err=" << std::fabs(ld/nd-td) << std::endl;
            }
        }

    }

    delete cint;
    delete nf;
    delete ls_coeff;

    //dump out the data for ParaView
    if(visualization)
    {
        mfem::ParaViewDataCollection paraview_dc("TopOpt", &pmesh);
        paraview_dc.SetPrefixPath("ParaView");
        paraview_dc.SetLevelsOfDetail(order);
        paraview_dc.SetDataFormat(mfem::VTKFormat::BINARY);
        paraview_dc.SetHighOrderOutput(true);
        paraview_dc.SetCycle(0);
        paraview_dc.SetTime(0.0);
        paraview_dc.RegisterField("lsf",&lsf);
        paraview_dc.RegisterField("grd",&ggf);
        paraview_dc.RegisterField("nod",&nod);
        paraview_dc.Save();
    }

}
