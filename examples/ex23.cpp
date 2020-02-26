//                                MFEM Example 1
//
// Compile with: make ex1
//
// Sample runs:  ex1 -m ../data/square-disc.mesh
//               ex1 -m ../data/star.mesh
//               ex1 -m ../data/star-mixed.mesh
//               ex1 -m ../data/escher.mesh
//               ex1 -m ../data/fichera.mesh
//               ex1 -m ../data/fichera-mixed.mesh
//               ex1 -m ../data/toroid-wedge.mesh
//               ex1 -m ../data/square-disc-p2.vtk -o 2
//               ex1 -m ../data/square-disc-p3.mesh -o 3
//               ex1 -m ../data/square-disc-nurbs.mesh -o -1
//               ex1 -m ../data/star-mixed-p2.mesh -o 2
//               ex1 -m ../data/disc-nurbs.mesh -o -1
//               ex1 -m ../data/pipe-nurbs.mesh -o -1
//               ex1 -m ../data/fichera-mixed-p2.mesh -o 2
//               ex1 -m ../data/star-surf.mesh
//               ex1 -m ../data/square-disc-surf.mesh
//               ex1 -m ../data/inline-segment.mesh
//               ex1 -m ../data/amr-quad.mesh
//               ex1 -m ../data/amr-hex.mesh
//               ex1 -m ../data/fichera-amr.mesh
//               ex1 -m ../data/mobius-strip.mesh
//               ex1 -m ../data/mobius-strip.mesh -o -1 -sc
//
// Device sample runs:
//               ex1 -pa -d cuda
//               ex1 -pa -d raja-cuda
//               ex1 -pa -d occa-cuda
//               ex1 -pa -d raja-omp
//               ex1 -pa -d occa-omp
//               ex1 -pa -d ceed-cpu
//               ex1 -pa -d ceed-cuda
//               ex1 -m ../data/beam-hex.mesh -pa -d cuda
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Laplace problem
//               -Delta u = 1 with homogeneous Dirichlet boundary conditions.
//               Specifically, we discretize using a FE space of the specified
//               order, or if order < 1 using an isoparametric/isogeometric
//               space (i.e. quadratic for quadratic curvilinear mesh, NURBS for
//               NURBS mesh, etc.)
//
//               The example highlights the use of mesh refinement, finite
//               element grid functions, as well as linear and bilinear forms
//               corresponding to the left-hand side and right-hand side of the
//               discrete linear system. We also cover the explicit elimination
//               of essential boundary conditions, static condensation, and the
//               optional connection to the GLVis tool for visualization.

#include "mfem.hpp"
#include <fstream>
#include <iostream>


#include "../fem/adnonlininteg.hpp"


using namespace std;

namespace mfem{
    
class VolNonlinearForm: public NonlinearFormIntegrator
{
protected:
    double eta;
    double beta;
public:
    VolNonlinearForm(double eta_, double beta_){
        eta=eta_;
        beta=beta_;}
    virtual ~VolNonlinearForm(){ }
    
    double Project(double inp)
    {
        // tanh projection - Wang&Lazarov&Sigmund2011 
        double a=std::tanh(eta*beta);
        double b=std::tanh(beta*(1.0-eta));
        double c=std::tanh(beta*(inp-eta));
        double rez=(a+c)/(a+b);
        return rez;
    }

    double ProjGrad(double inp)
    {
        double c=std::tanh(beta*(inp-eta));
        double a=std::tanh(eta*beta);
        double b=std::tanh(beta*(1.0-eta));
        double rez=beta*(1.0-c*c)/(a+b);
        return rez;
    }

    
    double ProjSec(double inp)
    {
        double c=std::tanh(beta*(inp-eta));
        double a=std::tanh(eta*beta);
        double b=std::tanh(beta*(1.0-eta));
        double rez=-2.0*beta*beta*c*(1.0-c*c)/(a+b);
        return rez;
    }
    
    
    virtual double GetElementEnergy(const mfem::FiniteElement & el,
                                           mfem::ElementTransformation & trans,
                                           const mfem::Vector & elfun) override
    {
        double energy=0.0;
        int ndof = el.GetDof();
        int ndim = el.GetDim();
        
        const mfem::IntegrationRule *ir = NULL;
        int order = 2 * trans.OrderGrad(&el) - 1; // correct order?
        ir = &mfem::IntRules.Get(el.GetGeomType(), order);
        
        mfem::Vector shapef(ndof);
        
        double w;
        for (int i = 0; i < ir -> GetNPoints(); i++)
        {
            const mfem::IntegrationPoint &ip = ir->IntPoint(i);
            trans.SetIntPoint(&ip);
            el.CalcShape(ip,shapef);
            w= Project(shapef*elfun);
            w= ip.weight * trans.Weight() * w;
            energy = energy + w;
        }
        return energy;
    }
    
    virtual void AssembleElementVector(const mfem::FiniteElement & el,
                                       mfem::ElementTransformation & trans,
                                       const mfem::Vector & elfun,
                                       mfem::Vector & elvect) override
    {
        
        
        int ndof = el.GetDof();
    
        const mfem::IntegrationRule *ir = NULL;
        int order = 2 * trans.OrderGrad(&el) - 1; // correct order?
        ir = &mfem::IntRules.Get(el.GetGeomType(), order);
        
        elvect.SetSize(ndof);
        elvect=0.0;
        
        mfem::Vector shapef(ndof);
        double w;
        for (int i = 0; i < ir -> GetNPoints(); i++)
        {
            const mfem::IntegrationPoint &ip = ir->IntPoint(i);
            trans.SetIntPoint(&ip);
            el.CalcShape(ip,shapef);
            w= ProjGrad(shapef*elfun);
            w= ip.weight * trans.Weight() * w;
            elvect.Add(w,shapef);
        }
        
    }
    
    
    virtual void AssembleElementGrad(const mfem::FiniteElement & el,
                                     mfem::ElementTransformation & trans,
                                     const mfem::Vector & elfun,
                                     mfem::DenseMatrix & elmat) override
    {
        int ndof = el.GetDof();
    
        const mfem::IntegrationRule *ir = NULL;
        int order = 2 * trans.OrderGrad(&el) - 1; // correct order?
        ir = &mfem::IntRules.Get(el.GetGeomType(), order);
        
        elmat.SetSize(ndof);
        elmat=0.0;
        
        mfem::Vector shapef(ndof);
        double w;
        for (int i = 0; i < ir -> GetNPoints(); i++)
        {
            const mfem::IntegrationPoint &ip = ir->IntPoint(i);
            trans.SetIntPoint(&ip);
            el.CalcShape(ip,shapef);
            w= ProjSec(shapef*elfun);
            w= ip.weight * trans.Weight() * w;
            AddMult_a_VVt(w, shapef, elmat);
        }
        
   }
    
};


class VolNonlinearFormFADH:public FADNonlinearFormIntegratorH
{
private:
    double eta;
    double beta;
    
    template<typename DType>
    DType Project(DType inp)
    {
        // tanh projection - Wang&Lazarov&Sigmund2011 
        double a=std::tanh(eta*beta);
        double b=std::tanh(beta*(1.0-eta));
        DType c=tanh(beta*(inp-eta));
        DType rez=(a+c)/(a+b);
        return rez;
    }
    
public:
    
    VolNonlinearFormFADH(double eta_, double beta_){
        eta=eta_;
        beta=beta_;
    }
    
    
    virtual FADType ElementEnergy(const mfem::FiniteElement & el,
                                       mfem::ElementTransformation & trans,
                                       const FADVector & elfun) override
    {
        FADType rez=MyElementEnergy<FADType,FADVector>(el,trans,elfun);
        return rez;
    }
    
    virtual SADType ElementEnergy(const mfem::FiniteElement & el,
                                       mfem::ElementTransformation & trans,
                                       const SADVector & elfun) override
    {
        return MyElementEnergy<SADType,SADVector>(el,trans,elfun);
    }
    
    template<typename MDType, typename MVType>
    MDType MyElementEnergy(const mfem::FiniteElement & el,
                            mfem::ElementTransformation & trans,
                            const MVType & elfun)
    {
        MDType energy=MDType();
        int ndof = el.GetDof();
        
        const mfem::IntegrationRule *ir = NULL;
        int order = 2 * trans.OrderGrad(&el) - 1; // correct order?
        ir = &mfem::IntRules.Get(el.GetGeomType(), order);
        
        mfem::Vector shapef(ndof);
        
        MDType w;
        for (int i = 0; i < ir -> GetNPoints(); i++)
        {
            const mfem::IntegrationPoint &ip = ir->IntPoint(i);
            trans.SetIntPoint(&ip);
            el.CalcShape(ip,shapef);
            w= Project(elfun*shapef);
            w= ip.weight * trans.Weight() * w;
            energy = energy + w;
        }
        return energy;
    }
    
    
    virtual double ElementEnergy(const mfem::FiniteElement & el, 
                                 mfem::ElementTransformation & Tr, 
                                 const mfem::Vector & elfun) override
    {
        return GetElementEnergy(el,Tr,elfun);
    }
    
    virtual double GetElementEnergy(const mfem::FiniteElement & el,
                                           mfem::ElementTransformation & trans,
                                           const mfem::Vector & elfun) override
    {
        double rez;
        rez=MyElementEnergy<double,mfem::Vector>(el,trans,elfun);
        return rez;
    
    }
};




}



double TFunc(const mfem::Vector& a){
    double sca=4.0;
    double rez=(std::sin(sca*a[0])*std::sin(sca*a[1])*std::sin(sca*a[2]))*0.5+0.5;
    return rez;
}


int main(int argc, char *argv[])
{
   
    // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   bool static_cond = false;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = true;

   mfem::OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   mfem::Device device(device_config);
   device.Print();

   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   mfem::Mesh *mesh = new mfem::Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();
   
   
   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 50,000
   //    elements.
   
   {
      int ref_levels =
         (int)floor(log(50000./mesh->GetNE())/log(2.)/dim);
      ref_levels=1;
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }
   

   // 5. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order. If order < 1, we
   //    instead use an isoparametric/isogeometric space.
   mfem::FiniteElementCollection *fec;
   if (order > 0)
   {
      fec = new mfem::H1_FECollection(order, dim);
   }
   else if (mesh->GetNodes())
   {
      fec = mesh->GetNodes()->OwnFEC();
      cout << "Using isoparametric FEs: " << fec->Name() << endl;
   }
   else
   {
      fec = new mfem::H1_FECollection(order = 1, dim);
   }
   mfem::FiniteElementSpace *fespace = new mfem::FiniteElementSpace(mesh, fec);
   cout << "Number of finite element unknowns: "
        << fespace->GetTrueVSize() << endl;
   
   mfem::NonlinearForm* nf0=new mfem::NonlinearForm(fespace);
   mfem::NonlinearForm* nf1=new mfem::NonlinearForm(fespace);

   mfem::FunctionCoefficient ifun(TFunc);
   //create an input for the NonlinearForm
   mfem::GridFunction* igf = new mfem::GridFunction(fespace);
   igf->ProjectCoefficient(ifun);
   
   std::cout << "Size of the grid function igf:"<<igf->Size()<<std::endl;
   
   
   mfem::Vector* resv0=new mfem::Vector(fespace->GetTrueVSize());
   mfem::Vector* resv1=new mfem::Vector(fespace->GetTrueVSize());
   mfem::Vector* stat=new mfem::Vector(fespace->GetTrueVSize());
   
   igf->GetTrueDofs(*stat);
   
   
   //compute the energy - the total volume above 0.5
   nf0->AddDomainIntegrator(new mfem::VolNonlinearForm(0.5,8.0));
   nf1->AddDomainIntegrator(new mfem::VolNonlinearFormFADH(0.5,8.0));
   
   double vol0=nf0->GetEnergy(*stat);
   double vol1=nf1->GetEnergy(*stat);
   std::cout<<"The total volume is:("<<vol0<<","<<vol1<<")"<<std::endl;
   nf0->Mult(*stat,*resv0);
   nf1->Mult(*stat,*resv1);
   //project back the gradients to a grid function
   mfem::GridFunction* ggf0=new mfem::GridFunction(fespace);
   ggf0->SetFromTrueDofs(*resv0);
   mfem::GridFunction* ggf1=new mfem::GridFunction(fespace);
   ggf1->SetFromTrueDofs(*resv1);
   
   
   resv0->Add(-1.0,*resv1);
   std::cout<<"Norm|v_1-v_0|="<<resv0->Norml2()<<std::endl;
   
   mfem::Operator& grad0(nf0->GetGradient(*stat));
   mfem::SparseMatrix* spmat0=dynamic_cast<mfem::SparseMatrix*>(&grad0);
   mfem::Operator& grad1(nf1->GetGradient(*stat));
   mfem::SparseMatrix* spmat1=dynamic_cast<mfem::SparseMatrix*>(&grad1);
   std::cout<<"Norm mat1="<<spmat0->MaxNorm()<<" mat2="<<spmat1->MaxNorm()<<std::endl;
   spmat0->Add(-1.0,*spmat1);
   std::cout<<"Norm diff"<<spmat0->MaxNorm()<<std::endl;
   {
       std::fstream mstr; 
       mstr.open("mat.dat",std::ios::out);
       spmat0->PrintMatlab(mstr);
       mstr.close();
   }
   
   mfem::ParaViewDataCollection *dacol=new mfem::ParaViewDataCollection("IGF_OUT",mesh);
   dacol->SetLevelsOfDetail(2);
   dacol->SetCycle(1);
   dacol->SetTime(0.0); // set the time
   dacol->RegisterField("density",igf);
   dacol->RegisterField("grads0",ggf0);
   dacol->RegisterField("grads1",ggf1);
   dacol->Save();
   delete dacol;
   
   delete ggf0;
   delete ggf1;
   delete stat; 
   delete resv0;
   delete resv1;
   delete igf;     
   delete nf0;
   delete nf1;
   delete fespace;
   delete fec;
   delete mesh;
   
   return 0;
}
