#include "mfem.hpp"
#include "mtop_solvers.hpp"

#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


class VectorForceCoeff:public VectorCoefficient
{
public:
    VectorForceCoeff(Vector& A_, Vector& cc, real_t r_)
        : VectorCoefficient(A_.Size()), a(A_), c(cc), r(r_)
    {
    }

    virtual void Eval(Vector &V, ElementTransformation &T,
                      const IntegrationPoint &ip)
    {
        V.SetSize(a.Size());
        Vector transip(a.Size()); transip=0.0;
        T.Transform(ip, transip);


        transip.Add(-1.0,c);
        real_t dist=transip.Norml2();

        if(dist<r){
            V.Set(1.0,a);
        }else{
            V=0.0;
        }

    }

private:
    Vector a;
    Vector c;
    real_t r;
};


int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 2. Parse command-line options.
   const char *mesh_file = "../../data/star.mesh";
   int order = 1;
   bool static_cond = false;
   bool pa = false;
   bool fa = false;
   const char *device_config = "cpu";
   bool visualization = true;
   bool algebraic_ceed = false;

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
#ifdef MFEM_USE_CEED
   args.AddOption(&algebraic_ceed, "-a", "--algebraic",
                  "-no-a", "--no-algebraic",
                  "Use algebraic Ceed solver");
#endif
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
   if(Mpi::WorldRank()==0){
       std::cout<<pmesh.GetNE()<<std::endl;
   }

   H1_FECollection* vfec=new H1_FECollection(order,dim);
   //ParFiniteElementSpace* vfes=new ParFiniteElementSpace(&pmesh,vfec,dim, mfem::Ordering::byVDIM);
   ParFiniteElementSpace* vfes=new ParFiniteElementSpace(&pmesh,vfec,dim, mfem::Ordering::byNODES);

   //extract the BC dofs
   Array<int> ess_dofs; //true dofs
   {
       Array<int> ess_bdr(pmesh.bdr_attributes.Max());
       ess_bdr = 0;
       ess_bdr[0] = 1;
       ess_bdr[1] = 1;
       ess_bdr[2] = 1;
       ess_bdr[3] = 1;
       ess_bdr[4] = 0;
       vfes->GetEssentialTrueDofs(ess_bdr, ess_dofs);
   }


   std::unique_ptr<ParBilinearForm> mf; //mass bilinear form
   std::unique_ptr<ParBilinearForm> cf; //damping bilinear form
   std::unique_ptr<ParBilinearForm> kf; //stiffness bilinear form
   std::unique_ptr<ParBilinearForm> wf;

   ConstantCoefficient rho(1.0);
   ConstantCoefficient damp(0.02);
   ConstantCoefficient E(1.0);
   ConstantCoefficient nu(0.2);

   IsoElasticyLambdaCoeff lambda(&E,&nu);
   IsoElasticySchearCoeff mu(&E,&nu);

   //stiffness operator
   kf.reset(new ParBilinearForm(vfes));
   //mass operator
   mf.reset(new ParBilinearForm(vfes));
   //damping operator
   cf.reset(new ParBilinearForm(vfes));
   //Helmholtz operator
   wf.reset(new ParBilinearForm(vfes));

   if(pa){
       kf->SetAssemblyLevel(mfem::AssemblyLevel::PARTIAL);
       mf->SetAssemblyLevel(mfem::AssemblyLevel::PARTIAL);
       cf->SetAssemblyLevel(mfem::AssemblyLevel::PARTIAL);
       wf->SetAssemblyLevel(mfem::AssemblyLevel::PARTIAL);
   }else{
       kf->SetAssemblyLevel(mfem::AssemblyLevel::LEGACY);
       mf->SetAssemblyLevel(mfem::AssemblyLevel::LEGACY);
       cf->SetAssemblyLevel(mfem::AssemblyLevel::LEGACY);
       wf->SetAssemblyLevel(mfem::AssemblyLevel::LEGACY);
   }

   std::cout<<"Add Integrators"<<std::endl;

   kf->AddDomainIntegrator(new ElasticityIntegrator(lambda,mu));
   mf->AddDomainIntegrator(new VectorMassIntegrator(rho));
   cf->AddDomainIntegrator(new VectorMassIntegrator(damp));
   //Helmholtz operator
   wf->AddDomainIntegrator(new ElasticityIntegrator(lambda,mu));
   real_t freq=0.01;
   //ProductCoefficient pc(-freq*freq,rho);
   ConstantCoefficient pc(-freq*freq*1.0);
   wf->AddDomainIntegrator(new VectorMassIntegrator(pc));


   std::cout<<"Start Assemble"<<std::endl;
   kf->Assemble();
   kf->Finalize();
   std::cout<<"K ready"<<std::endl;
   mf->Assemble();
   mf->Finalize();
   std::cout<<"M ready"<<std::endl;
   cf->Assemble();
   cf->Finalize();
   std::cout<<"C ready"<<std::endl;
   wf->Assemble();
   wf->Finalize();
   std::cout<<"W ready"<<std::endl;

   std::cout<<"Cons"<<std::endl;

   const Operator *P = kf->GetProlongation();


   ConstrainedOperator ckf(new RAPOperator(*P,*kf,*P),ess_dofs,true, Operator::DiagonalPolicy::DIAG_ONE);
   ConstrainedOperator cmf(new RAPOperator(*P,*mf,*P),ess_dofs,true, Operator::DiagonalPolicy::DIAG_ZERO);
   ConstrainedOperator ccf(new RAPOperator(*P,*cf,*P),ess_dofs,true, Operator::DiagonalPolicy::DIAG_ZERO);
   ConstrainedOperator cwf(new RAPOperator(*P,*wf,*P),ess_dofs,true, Operator::DiagonalPolicy::DIAG_ONE);

   OperatorHandle hkf;
   OperatorHandle hmf;
   OperatorHandle hcf;
   OperatorHandle hwf;
   kf->FormSystemMatrix(ess_dofs, hkf);
   mf->FormSystemMatrix(ess_dofs, hmf);
   cf->FormSystemMatrix(ess_dofs, hcf);
   wf->FormSystemMatrix(ess_dofs, hwf);


   std::cout<<"Forcing!"<<std::endl;

   //force coefficient
   Vector A(dim); A=0.0; A[1]=1.0; //firce amplitude
   Vector cc(dim); cc[0]=3.9; cc[1]=0.5; if(dim==3){cc[2]=0.0;} //force possition
   VectorForceCoeff fc(A,cc,0.1);

   ParLinearForm lf(vfes);
   lf.AddDomainIntegrator(new VectorDomainLFIntegrator(fc));
   lf.Assemble();

   std::cout<<"LF is assmebled!"<<std::endl;

   Array<int> block_true_offsets;
   block_true_offsets.SetSize(3);
   block_true_offsets[0] = 0;
   block_true_offsets[1] = vfes->GetTrueVSize();
   block_true_offsets[2] = vfes->GetTrueVSize();
   block_true_offsets.PartialSum();

   // solution
   BlockVector x; x.Update(block_true_offsets); x=0.0;
   // RHS
   BlockVector f; f.Update(block_true_offsets);
   std::cout<<pmesh.GetMyRank()<<" f0.size="<<f.GetBlock(0).Size()<<std::endl;
   std::cout<<pmesh.GetMyRank()<<" f1.size="<<f.GetBlock(1).Size()<<std::endl;
   std::cout<<pmesh.GetMyRank()<<" cwf.size="<<cwf.Width()<<std::endl;
   std::cout<<pmesh.GetMyRank()<<" vfes.size="<<vfes->GetTrueVSize()<<std::endl;
   lf.ParallelAssemble(f.GetBlock(0));
   lf.ParallelAssemble(f.GetBlock(1));

   std::cout<<"RHS is ready!"<<std::endl;

   //cwf.EliminateRHS(x.GetBlock(0), f.GetBlock(0));
   //cwf.EliminateRHS(x.GetBlock(1), f.GetBlock(1));

   BlockOperator bop(block_true_offsets);
   bop.SetBlock(0,0,hwf.Ptr(),1.0);
   bop.SetBlock(0,1,hcf.Ptr(),-1.0);
   bop.SetBlock(1,0,hcf.Ptr(),1.0);
   bop.SetBlock(1,1,hwf.Ptr(),1.0);


   SumOperator W(hkf.Ptr(),1.0,hmf.Ptr(),-freq*freq,false,false);

   std::cout<<"Allocate PRESB"<<std::endl;
   PRESBPrec* prec=new PRESBPrec(pmesh.GetComm(),1);
   prec->SetOperators(hwf.Ptr(),hmf.Ptr(),1.0,freq*freq,1);
   prec->SetAbsTol(1e-12);
   prec->SetRelTol(1e-10);
   prec->SetMaxIter(10);

   //prec->Mult(f,x);

   //set the linear solver
   GMRESSolver* ls=new GMRESSolver(pmesh.GetComm());
   ls->SetOperator(bop);
   ls->SetAbsTol(1e-12);
   ls->SetRelTol(1e-12);
   ls->SetMaxIter(1000);
   ls->SetPrintLevel(1);

   ls->SetOperator(bop);
   //ls->SetPreconditioner(*prec);
   ls->Mult(f,x);
   delete ls;

   delete prec;

   //check the solutions
   ParGridFunction rx(vfes); rx.SetFromTrueDofs(x.GetBlock(0));
   ParGridFunction ix(vfes); ix.SetFromTrueDofs(x.GetBlock(1));
   {
        ParaViewDataCollection paraview_dc("stokes", &pmesh);
        paraview_dc.SetPrefixPath("ParaView");
        paraview_dc.SetLevelsOfDetail(order);
        paraview_dc.SetDataFormat(VTKFormat::BINARY);
        paraview_dc.SetHighOrderOutput(true);
        paraview_dc.SetCycle(0);
        paraview_dc.SetTime(0.0);
        paraview_dc.RegisterField("re",&rx);
        paraview_dc.RegisterField("im",&ix);
        paraview_dc.Save();
   }

   BlockVector r(f);

   bop.Mult(x,r);
   r.Add(-1,f);
   real_t rr=mfem::InnerProduct(pmesh.GetComm(), r,r);
   if(pmesh.GetMyRank()==0){
       std::cout<<"Residual="<<sqrt(rr)<<std::endl;
   }

   delete vfes;
   delete vfec;
   Mpi::Finalize();
   return 0;
}
