//                                MFEM Example 26
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <random>
#include "element-smoother.hpp"
using namespace std;
using namespace mfem;

class DiffusionMultigrid : public GeometricMultigrid
{
private:
   Coefficient * cf = nullptr;
   int smoother_kind = 0;
   // 0: Jacobi, 1:Chebychev, 2: element-smoother(matrix-free)
   HypreBoomerAMG* amg;

public:
   // Constructs a diffusion multigrid for the ParFiniteElementSpaceHierarchy
   // and the array of essential boundaries
   DiffusionMultigrid(ParFiniteElementSpaceHierarchy& fespaces,
                      Array<int>& ess_bdr, Coefficient * cf_,int smoother_ = 0)
      : GeometricMultigrid(fespaces), cf(cf_), smoother_kind(smoother_)
   {
      ConstructCoarseOperatorAndSolver(fespaces.GetFESpaceAtLevel(0), ess_bdr,cf);

      for (int level = 1; level < fespaces.GetNumLevels(); ++level)
      {
         ConstructOperatorAndSmoother(fespaces.GetFESpaceAtLevel(level), ess_bdr,cf);
      }
   }

   virtual ~DiffusionMultigrid()
   {
      delete amg;
   }

private:
   void ConstructBilinearForm(ParFiniteElementSpace& fespace, Array<int>& ess_bdr,
                              bool partial_assembly, Coefficient * cf)
   {
      ParBilinearForm* form = new ParBilinearForm(&fespace);
      if (partial_assembly)
      {
         form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      }
      form->AddDomainIntegrator(new DiffusionIntegrator(*cf));
      form->Assemble();
      bfs.Append(form);

      essentialTrueDofs.Append(new Array<int>());
      fespace.GetEssentialTrueDofs(ess_bdr, *essentialTrueDofs.Last());
   }

   void ConstructCoarseOperatorAndSolver(ParFiniteElementSpace& coarse_fespace,
                                         Array<int>& ess_bdr, 
                                         Coefficient * cf)
   {
      ConstructBilinearForm(coarse_fespace, ess_bdr, false, cf);

      HypreParMatrix* hypreCoarseMat = new HypreParMatrix();
      bfs.Last()->FormSystemMatrix(*essentialTrueDofs.Last(), *hypreCoarseMat);

      amg = new HypreBoomerAMG(*hypreCoarseMat);
      amg->SetPrintLevel(-1);

      CGSolver* pcg = new CGSolver(MPI_COMM_WORLD);
      pcg->SetPrintLevel(-1);
      pcg->SetMaxIter(10);
      pcg->SetRelTol(sqrt(1e-4));
      pcg->SetAbsTol(0.0);
      pcg->SetOperator(*hypreCoarseMat);
      pcg->SetPreconditioner(*amg);

      AddLevel(hypreCoarseMat, pcg, true, true);
   }

   void ConstructOperatorAndSmoother(ParFiniteElementSpace& fespace,
                                     Array<int>& ess_bdr, Coefficient *cf)
   {
      ConstructBilinearForm(fespace, ess_bdr, true, cf);

      OperatorPtr opr;
      opr.SetType(Operator::ANY_TYPE);
      bfs.Last()->FormSystemMatrix(*essentialTrueDofs.Last(), opr);
      opr.SetOperatorOwner(false);
      Solver * smoother = nullptr;
      Vector diag;
      if (smoother_kind < 2 )
      {
         diag.SetSize(fespace.GetTrueVSize());
         bfs.Last()->AssembleDiagonal(diag);
      }

      switch (smoother_kind)
      {
      case 0:
         smoother = new OperatorJacobiSmoother(diag,*essentialTrueDofs.Last(),0.6667);
         break;
      case 1:
         smoother = new OperatorChebyshevSmoother(opr.Ptr(), diag,
                                       *essentialTrueDofs.Last(), 5);
         break;   
      case 2:
         smoother = new ElementSmoother(&fespace,ess_bdr,cf);
         break;      
      case 3:
         {
            ElementSmoother * sm = new ElementSmoother(&fespace,ess_bdr,cf);
            smoother = new OperatorChebyshevSmoother(*opr,*sm,5,fespace.GetComm());
            break;      
         }
      default:
         MFEM_ABORT("Wrong smoother choice");
         break;
      }
      AddLevel(opr.Ptr(), smoother, true, true);
   }
};

int dim;
int  exact   = 0;
bool tpcoeff = true;

double f_exact(const Vector & x);
double u_exact(const Vector & x);
void usol(const Vector & x, double &u, Vector & Grad, double & d2u);

double DiffusionCoeff(const Vector & x);
double TPDiffusionCoeff(const Vector & x, int coord);
void DiffusionCoeffGrad(const Vector & x, Vector & Grad);


int main(int argc, char *argv[])
{
   // 0. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 1. Parse command-line options.
   const char *mesh_file = "../../data/inline-quad.mesh";
   int init_geometric_refinements = 0;
   int pinit_geometric_refinements = 0;
   int geometric_refinements = 0;
   int order_refinements = 2;
   const char *device_config = "cpu";
   bool visualization = true;
   int order = 1;
   double skew_factor = 0.0;
   double scale_factor = 1.0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element order.");
   args.AddOption(&init_geometric_refinements, "-ref", "--initial-geometric-refinements",
                  "Number of serial geometric refinements defining the coarse mesh.");
   args.AddOption(&pinit_geometric_refinements, "-pref", "--initial-geometric-refinements",
                  "Number of parallel geometric refinements defining the coarse mesh.");                  
   args.AddOption(&geometric_refinements, "-gr", "--geometric-refinements",
                  "Number of geometric refinements done prior to order refinements.");
   args.AddOption(&order_refinements, "-or", "--order-refinements",
                  "Number of order refinements. Finest level in the hierarchy has order 2^{or}.");
   args.AddOption(&tpcoeff, "-tpcoeff", "--tp-coefficient", "-no-tpcoeff",
               "--no-tp-coefficient", "Tensor product diffusion coefficient or not");
   args.AddOption(&exact, "-exact", "--exact", "Exact Solution flag: 0: unknown");   
   args.AddOption(&skew_factor, "-c", "--skew_factor", "Skew_factor");   
   args.AddOption(&scale_factor, "-s", "--scale_factor", "Scale_factor");   
   
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
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

   // 2. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (myid == 0) { device.Print(); }


   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   // int nx = pow(2,init_geometric_refinements);
   // int ny = pow(2,init_geometric_refinements);
   // Mesh *mesh = new Mesh(1,1,1,mfem::Element::HEXAHEDRON,true,1.0,2.0,3.0,false);
   // Mesh *mesh = new Mesh(1,1,mfem::Element::QUADRILATERAL,true,1.0,1.0,false);
   // Mesh *mesh = new Mesh(1,4.0);
   // move nodes
   dim = mesh->Dimension();
   mesh->EnsureNodes();
   
   mesh->SetCurvature(3);
   GridFunction * nodes = mesh->GetNodes();
   // *nodes +=1.0;
   // *nodes *=0.5;
   double c = skew_factor;
   double s = scale_factor;
   if (dim == 2)
   {
      for (int i=0; i<nodes->Size()/2; i++)
      {
      // double temp = (*nodes)(2*i);
      // (*nodes)(2*i) += (*nodes)(2*i)*(*nodes)(2*i) + c*pow((*nodes)(2*i+1),2);
      // (*nodes)(2*i) +=  c*pow((*nodes)(2*i+1),2);
         (*nodes)(2*i) +=  c*pow((*nodes)(2*i+1),2);
      // (*nodes)(2*i+1) = (*nodes)(2*i+1)*(*nodes)(2*i+1) + c*pow(temp,2);
      }
   }
   else
   {
      for (int i=0; i<nodes->Size()/3; i++)
      {
         // (*nodes)(3*i) +=  c*pow((*nodes)(3*i+1),2);
         // (*nodes)(3*i+1) +=  c*pow((*nodes)(3*i+2),3);
         (*nodes)(3*i+2) +=  c*pow((*nodes)(3*i),2);
      }
   }

   for (int i=0; i<nodes->Size(); i++)
   {
      (*nodes)(i) *=  s;
   }

   dim = mesh->Dimension();
   // mesh->EnsureNCMesh();

   // 4. Refine the mesh to increase the resolution and order
   {
      for (int l = 0; l < init_geometric_refinements; l++)
      {
         mesh->UniformRefinement();
      }
   }

   ParMesh * pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   mesh->Clear();
   {
      for (int l = 0; l < pinit_geometric_refinements; l++)
      {
         pmesh->UniformRefinement();
      }
   }

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream mesh_sock(vishost, visport);
      mesh_sock << "parallel " << num_procs << " " << myid << "\n";
      mesh_sock.precision(8);
      mesh_sock << "mesh\n" << *pmesh << flush;
   }

   FiniteElementCollection *fec = new H1_FECollection(order, dim);
   ParFiniteElementSpace *coarse_fespace = new ParFiniteElementSpace(pmesh, fec);
   ParFiniteElementSpaceHierarchy fespaces(pmesh, coarse_fespace, true, true);


   Coefficient * cf = nullptr;
   if (exact)
   {
      cf = new FunctionCoefficient(DiffusionCoeff);
   }
   else
   {
      cf = new ConstantCoefficient(1.0);
   }

   Array<FiniteElementCollection*> collections;
   collections.Append(fec);
   for (int level = 0; level < geometric_refinements; ++level)
   {
      fespaces.AddUniformlyRefinedLevel();
   }
   for (int level = 0; level < order_refinements; ++level)
   {
      // order++;
      order *=2;
      // collections.Append(new H1_FECollection(std::pow(2, level+1), dim));
      collections.Append(new H1_FECollection(order, dim));
      fespaces.AddOrderRefinedLevel(collections.Last());
   }

   HYPRE_Int size = fespaces.GetFinestFESpace().GlobalTrueVSize();

   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
      cout << "Order = " << order << endl;
   }

   // 6. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
   //    the basis functions in the finite element fespace.
   FunctionCoefficient f(f_exact);
   ConstantCoefficient one(1.0);
     
   
   Array<int> ess_bdr;
   if(pmesh->bdr_attributes.Size())
   {   
      ess_bdr.SetSize(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
   }
   ParGridFunction x(&fespaces.GetFinestFESpace());
   // GridFunction gf_coeff(&fespaces.GetFinestFESpace());
   ParMesh * ref_mesh = fespaces.GetFinestFESpace().GetParMesh();
   L2_FECollection * l2fec = new L2_FECollection(order,dim);
   ParFiniteElementSpace * l2fes = new ParFiniteElementSpace(ref_mesh,l2fec);
   ParGridFunction gf_coeff(l2fes);
   // gf_coeff.ProjectCoefficient(*cf);
   gf_coeff.ProjectDiscCoefficient(*cf,mfem::GridFunction::AvgType::ARITHMETIC);
   
   int print_level = 3;
   int max_iter = 2000;
   double rtol = 1e-8;
   StopWatch chrono;
   // for (int i = 0; i<=6; i++)
   for (int i = 0; i<=6; i++)
   {
      OperatorPtr A;
      Vector B, X;
      Solver * prec = nullptr;
      CGSolver pcg(MPI_COMM_WORLD);
      pcg.SetPrintLevel(print_level);
      pcg.SetMaxIter(max_iter);
      pcg.SetRelTol(rtol);
      // i=1; Chebychev-Jacobi-MG
      // i=2; Chebychev-Element-MG
      // i=3; Chebychev-Jacobi-Smoother
      // i=4; Element-Smoother 
      // i=5; Chebychev-Element-Smoother 
      ParLinearForm *b = new ParLinearForm(&fespaces.GetFinestFESpace());
      if (exact)
      {
         b->AddDomainIntegrator(new DomainLFIntegrator(f));
      }
      else
      {
         b->AddDomainIntegrator(new DomainLFIntegrator(one));
      }
      b->Assemble();
      FunctionCoefficient u_ex(u_exact);
      x = 0.0;
      if (exact) x.ProjectCoefficient(u_ex);

      if (i<4)
      {
         prec = new DiffusionMultigrid(fespaces, ess_bdr, cf,i);
         dynamic_cast<DiffusionMultigrid *>(prec)->
                  SetCycleType(Multigrid::CycleType::VCYCLE, 1, 1);
         dynamic_cast<DiffusionMultigrid *>(prec)->
                  FormFineLinearSystem(x, *b, A, X, B);
         if (i == 0)
         {
            if (myid == 0)
               cout << "\nJacobi-MG " << endl;
         }         
         else if (i == 1)
         {
            if (myid == 0)
               cout << "\nJacobi-Chebychev-MG " << endl;
         }
         else if (i == 2)
         {
            if (myid == 0)
               cout << "\nElement-Smoother-MG " << endl;
         }
         else 
         {
            if (myid == 0)
               cout << "\nElement-Chebychev-MG " << endl;
         }
         pcg.SetOperator(*A);
         if (prec) { pcg.SetPreconditioner(*prec); }
         chrono.Clear();
         chrono.Start();         
         pcg.Mult(B,X);
         chrono.Stop();      
         if (myid == 0)
            cout<< "PCG::mult time = " << chrono.RealTime() << endl;       
        // Recover the solution as a finite element grid function.
        dynamic_cast<DiffusionMultigrid *>(prec)->RecoverFineFEMSolution(X, *b, x);
        delete prec;
      }
      else
      {
         ParBilinearForm a(&fespaces.GetFinestFESpace());
         a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
         DiffusionIntegrator * aa = new DiffusionIntegrator(*cf);
         int order1 = fespaces.GetFinestFESpace().GetOrder(0);
         IntegrationRule *irs = TensorIntegrationRule(fespaces.GetFinestFESpace(),order1); 
         aa->SetIntegrationRule(*irs);
         a.AddDomainIntegrator(aa);
         a.Assemble();
         
         Array<int> ess_tdof_list;
         fespaces.GetFinestFESpace().GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
         a.FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
         if (i==4)
         {
            Vector diag(fespaces.GetFinestFESpace().GetTrueVSize());
            a.AssembleDiagonal(diag);
            prec = new OperatorChebyshevSmoother(A.Ptr(), diag,ess_tdof_list, 1, MPI_COMM_WORLD);
            if (myid == 0)
               cout << "\nJacobi-Chebychev " << endl;
         }
         else if (i==5)
         {
            prec = new ElementSmoother(&fespaces.GetFinestFESpace(),ess_bdr, cf); 
            if (myid == 0)
               cout << "\nElementSmoother " << endl;
         }
         else
         {
            ElementSmoother *S = new ElementSmoother(&fespaces.GetFinestFESpace(),ess_bdr, cf); 
            prec = new OperatorChebyshevSmoother(*A, *S, 1, MPI_COMM_WORLD);
            if (myid == 0)
               cout << "\nElement-Chebychev " << endl;
         }
         pcg.SetOperator(*A);
         if (prec) { pcg.SetPreconditioner(*prec); }
         chrono.Clear();
         chrono.Start();         
         pcg.Mult(B,X);
         chrono.Stop();
         if (myid == 0)
            cout<< "PCG::mult time = " << chrono.RealTime() << endl;        
         delete prec;

         a.RecoverFEMSolution(X,*b,x);
      }
      delete b;
   }

   // 12. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *fespaces.GetFinestFESpace().GetMesh() << x <<
               flush;
      socketstream coeff_sock(vishost, visport);
      coeff_sock << "parallel " << num_procs << " " << myid << "\n";
      coeff_sock.precision(8);
      coeff_sock << "solution\n" << *fespaces.GetFinestFESpace().GetMesh() << gf_coeff <<
               flush;         
   }

   // 13. Free the used memory.
   for (int level = 0; level < collections.Size(); ++level)
   {
      delete collections[level];
   }

   MPI_Finalize();
   return 0;
}


double f_exact(const Vector & x)
{
   // -div (f * grad(u)) = (f * gradu(0))_x + (f*gradu(1))_y + + (f*gradu(2))_z
   // = f_x*gradu(0) + f * gradu(0)_x + f_y * gradu(1) + f* gradu(1)_y + f_z * gradu(2) + f* gradu(2)_z
   // = f_x*gradu(0) + f_y * gradu(1) + f_z * gradu(2) + f*d2u
   double u;
   double d2u;
   Vector gradu;
   usol(x,u,gradu,d2u);
   Vector gradf;
   DiffusionCoeffGrad(x,gradf);
   double f = DiffusionCoeff(x);
   double val = gradf * gradu + f*d2u;
   return -val;
}

double u_exact(const Vector & x)
{
   double u;
   Vector gradu;
   double d2u;
   usol(x,u,gradu,d2u);
   return u;
}

void usol(const Vector & x, double &u, Vector & Grad, double & d2u)
{
   Grad.SetSize(dim);
   if (exact == 1)
   {
      Vector alpha(dim); alpha = 5.0;
      // Vector alpha(dim); alpha = 0.5;
      double s = alpha * x; // dot product
      u = sin(M_PI*s);
      d2u = 0.0;
      for (int i = 0; i<dim; i++)
      {
         Grad[i] = alpha(i) * M_PI * cos(M_PI*s);
         d2u += alpha(i)*alpha(i); 
      }
      d2u = - M_PI*M_PI * d2u * u;
   }
   else if (exact == 2)
   {
      double c_0 = 1.2;
      double k_0 = 3.0;
      double c_1 = 2.3;
      double k_1 = 5.0;
      double c_2 = 1.3;
      double k_2 = 1.0;

      double alpha = c_0 + k_0 * x(0);
      double beta  = c_1 + k_1 * x(1);
      double gamma = 1.0;
      
      if (dim == 2)
      {
         u = sin(M_PI * alpha) *  sin(M_PI * beta);
         Grad[0] = M_PI*k_0 * cos(alpha) * sin(M_PI*beta);
         Grad[1] = M_PI*k_1 * cos(beta) * sin(M_PI*alpha);
      }
      else if (dim == 3)
      {
         gamma  = c_2 + k_2 * x(2);
         u = sin(M_PI * alpha) *  sin(M_PI * beta) * sin(M_PI*gamma);
         Grad[0] = M_PI*k_0 * cos(alpha) * sin(M_PI*beta) * sin(M_PI*gamma);
         Grad[1] = M_PI*k_1 * cos(beta) * sin(M_PI*alpha) * sin(M_PI*gamma);
         Grad[2] = M_PI*k_2 * cos(gamma) * sin(M_PI*alpha) * sin(M_PI*beta);
      }

      double u_xx = - M_PI * M_PI * k_0 * k_0 * u;
      double u_yy = - M_PI * M_PI * k_1 * k_1 * u;
      double u_zz = - M_PI * M_PI * k_2 * k_2 * u;
      d2u = u_xx + u_yy; 
      if (dim == 3 ) d2u += u_zz;
   }
}

double TPDiffusionCoeff(const Vector & x, int coord)
{
   double val;
   switch (coord)
   {
      case  0: val = 4.+3.*x(0); break;
      case  1: val = 0.5+7.*x(1)*x(1); break;
      case  2: val = (0.1+2.*x(2)); break;
      default: 
         val = (4.+3.*x(0))*(0.5+7.*x(1)*x(1)); 
         if (dim == 3 ) val *= (0.1+2.*x(2));
      break;
      // case  0: val = x(0); break;
      // case  1: val = 1.0; break;
      // case  2: val = 1.0; break;
      // default: val = x(0); break;
      // case  0: val = 3.0; break;
      // case  1: val = 2.0; break;
      // case  2: val = 1.0; break;
      // default: val = 6.0; break;
   }
   return val;
   // return 2.0;

}

double DiffusionCoeff(const Vector & x)
{
   double val;
   if (tpcoeff)
   {
      val = (4.+3.*x(0))*(0.5+7.*x(1)*x(1));
      if (dim == 3) val *= (0.1+2.*x(2));
   }
   else
   {
      // val = 2.0+cos(x.Sum());
      Vector cf(dim);
      // cf(0) = 0.1; cf(1) = 3.; 
      cf(0) = 1.0; cf(1) = 2.0; 
      // if (dim == 3) cf(2) = -7.8;
      if (dim == 3) cf(2) = +1.8;
      // double dd = x * cf + 1.5* x(1)*x(1);
      double dd = x * cf;
      // // double dd = x * cf;
      // // val = exp(cos(dd));
      val = exp(dd);

      // Vector alpha(dim); alpha = 5.0;
      // double s = alpha * x; // dot product
      // val = 2.0+sin(M_PI*s);

   }   
   return val;
}


void DiffusionCoeffGrad(const Vector & x, Vector & Grad)
{
   Grad.SetSize(dim);
   if (tpcoeff)
   {
      if (dim == 2)
      {
         Grad[0] = 3.* (0.5+7.*x(1)*x(1));
         Grad[1] = 14.* x(1) * (4.+3.*x(0));
      }
      else
      {
         Grad[0] = 3.* (0.5+7.*x(1)*x(1))*(0.1+2.*x(2));
         Grad[1] = 14.* x(1) * (4.+3.*x(0))*(0.1+2.*x(2));
         Grad[2] = 2.*(4.+3.*x(0))*(0.5+7.*x(1)*x(1));
      }
   }
   else
   {
      Vector cf(dim);
      // cf(0) = 0.1; cf(1) = 3.; 
      cf(0) = 1.0; cf(1) = 2.0; 
      // if (dim == 3) cf(2) = -7.8;
      if (dim == 3) cf(2) = 1.8;
      // double dd = x * cf  + 1.5* x(1)*x(1);
      double dd = x * cf;
      Vector alpha(dim); alpha = 5.0;

      // for (int d = 0; d<dim; d++)
      // {
      // //    // Grad[d] = -sin(x.Sum());
      // //    // Grad[d] = -cf(d) * exp(cos(dd))*sin(dd);
      // //    Grad[d] = cf(d) * exp(dd);
      //    Grad[d] = alpha(d) * M_PI * cos(M_PI*s);
      // }
      if (dim == 2)
      {
         Grad[0] = (cf(0) )*exp(dd);
         Grad[1] = (cf(1) + 3.0*x(1))*exp(dd);
      }
      else
      {
         // Grad[0] = (cf(0) + 1.5 * x(1))*exp(dd);
         Grad[0] = cf(0)*exp(dd);
         Grad[1] = cf(1)*exp(dd);
         Grad[2] = cf(2)*exp(dd);
      }
   }
}


