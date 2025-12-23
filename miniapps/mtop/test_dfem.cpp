#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "mtop_solvers.hpp"

using namespace mfem;

using mfem::future::dual;
using mfem::future::tuple;
using mfem::future::tensor;

using mfem::future::Weight;
using mfem::future::Gradient;
using mfem::future::Identity;



///////////////////////////////////////////////////////////////////////////////
/// \brief The SQFunction struct defining the Stokes operator at
/// integration points which is valid in 2D and 3D
template <int DIM, typename scalar_t=real_t> struct SQFunction
{
   using mati_t = tensor<scalar_t, DIM, DIM>;

   struct Stokes
   {
      MFEM_HOST_DEVICE inline auto operator()(const mati_t &dudxi,
                                              const real_t &M, // viscosity
                                              const mati_t &J,
                                              const real_t &w) const
      {
         /*
         mati_t invJ = mfem::future::inv<scalar_t>(J);
         const mati_t JxW = transpose(invJ) * det(J) * w;
         const auto eps = sym(dudxi * invJ);
         return tuple{(2.0 * M * eps) * JxW};
         */

         const mati_t invJ=mfem::future::inv<scalar_t>(J);
         const mati_t JxW = transpose(invJ) * det(J) * w;
         const auto eps = mfem::future::sym(dudxi * invJ);
         return tuple{(2.0 * M * eps) * JxW};

      }
   };
};

///////////////////////////////////////////////////////////////////////////////
/// \brief The Mass MQFunction struct defining Vector Mass operator
/// at integration points which is valid in 2D and 3D
template <int DIM,typename scalar_t=real_t> struct MQFunction
{
   using mati_t = tensor<scalar_t, DIM, DIM>;
   using veci_t = tensor<scalar_t, DIM>;
   struct Mass
   {
      MFEM_HOST_DEVICE inline auto operator()(const veci_t &u,
                                              const scalar_t &M, // mass coefficient
                                              const mati_t &J,
                                              const real_t &w) const
      {
         return tuple{(M * u) * det(J) * w};
      }
   };
};


///////////////////////////////////////////////////////////////////////////////
/// \brief The Anisotropic AEQFunction struct defining the Elasticity operator
/// at integration points which is valid in 3D
template <int NMAT,typename scalar_t=real_t> struct AEQFunction3D
{
   // Dimension
   static constexpr int DIM = 3;

   // Number of independent components in a symmetric DIM×DIM matrix
   static constexpr int NVOIGT = DIM * (DIM + 1) / 2;

   // Number of entries in the material matrix
   static constexpr int NMAT_ENTRIES = NVOIGT * (NVOIGT+1) / 2;

   // Total size of the flat array
   static constexpr int SIZE = NMAT * NMAT_ENTRIES;

   // The actual packed data: [NMAT][NVOIGT] in row-major order
   real_t data[SIZE];

   // ------------------------------------------------------------------
   // Mapping (i,j) → Voigt index for order: 11,22,33, 23,13,12
   // ------------------------------------------------------------------
   //  (0,0)→0   (1,1)→1   (2,2)→2
   //  (1,2)→3   (0,2)→4   (0,1)→5   (and symmetric)
   MFEM_HOST_DEVICE inline
   static constexpr int voigt_index(int i, int j)
   {
      if (i > j) { int tmp = i; i = j; j = tmp; }  // ensure i <= j

      if (i == 0 && j == 0) { return 0; }
      if (i == 1 && j == 1) { return 1; }
      if (i == 2 && j == 2) { return 2; }
      if (i == 1 && j == 2) { return 3; }
      if (i == 0 && j == 2) { return 4; }
      if (i == 0 && j == 1) { return 5; }

      return -1; // unreachable
   }

   // ------------------------------------------------------------------
   // Accessors
   // ------------------------------------------------------------------
   MFEM_HOST_DEVICE inline
   static constexpr int mat_index(int i, int j, int k, int l)
   {
      int ij = voigt_index(i,j);
      int kl = voigt_index(k,l);
      if (ij > kl) { int tmp = ij; ij = kl; kl = tmp; }  // ensure ij <= kl
      return (kl * (kl + 1)) / 2 + ij;
   }

   MFEM_HOST_DEVICE inline
   static constexpr int mat_index(int i, int j)
   {
      if (i>j) { int tmp = i; i = j; j = tmp; }
      {
         return (j * (j + 1)) / 2 + i;
      }
   }

   using mati_t = tensor<scalar_t, DIM, DIM>;
   using veci_t = tensor<scalar_t, NMAT>;

   struct Elasticity
   {
      MFEM_HOST_DEVICE inline auto operator()(const mati_t &dudxi,
                                              const veci_t &rhoi,
                                              const mati_t &J,
                                              const real_t &w) const
      {
         const mati_t invJ = mfem::future::inv<scalar_t>(J);
         const mati_t JxW = transpose(invJ) * det(J) * w;

         const auto eps = mfem::future::sym(dudxi * invJ);
         auto str= 0.0 * eps; // initialize to zero the stress tensor
         for (int im=0; im<NMAT; im++)
         {
            for (int i=0; i<DIM; i++)
            {
               for (int j=0; j<DIM; j++)
               {
                  for (int k=0; k<DIM; k++)
                  {
                     for (int l=0; l<DIM; l++)
                     {
                        // Voigt notation mapping
                        str[i][j] += rhoi[im] * data[im*NMAT_ENTRIES + mat_index(i,j,k,l)] * eps[k][l];
                     }
                  }
               }
            }
         }

         return tuple{str*JxW};
      }
   };
};





class NqptUniformParameterSpace : public
   mfem::future::UniformParameterSpace
{
public:
   NqptUniformParameterSpace(mfem::ParMesh &mesh,
                             const mfem::IntegrationRule &ir,
                             int vdim) :
      mfem::future::UniformParameterSpace(mesh, ir, vdim, false)
   {
      dtq.nqpt = ir.GetNPoints();
   }
};


int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   mfem::Mpi::Init(argc, argv);
   int myrank = mfem::Mpi::WorldRank();
   mfem::Hypre::Init();

   // Parse command-line options.
   const char *mesh_file = "./dfg_bench_flow_tri.msh";
   int order = 2;
   bool static_cond = false;
   int ser_ref_levels = 1;
   int par_ref_levels = 1;
   real_t newton_rel_tol = 1e-7;
   real_t newton_abs_tol = 1e-12;
   int newton_iter = 10;
   int print_level = 1;
   bool visualization = false;

   mfem::OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels,
                  "-rs",
                  "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels,
                  "-rp",
                  "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&visualization,
                  "-vis",
                  "--visualization",
                  "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&newton_rel_tol,
                  "-rel",
                  "--relative-tolerance",
                  "Relative tolerance for the Newton solve.");
   args.AddOption(&newton_abs_tol,
                  "-abs",
                  "--absolute-tolerance",
                  "Absolute tolerance for the Newton solve.");
   args.AddOption(&newton_iter,
                  "-it",
                  "--newton-iterations",
                  "Maximum iterations for the Newton solve.");
   args.Parse();
   if (!args.Good())
   {
      if (myrank == 0)
      {
         args.PrintUsage(std::cout);
      }
      return 1;
   }

   if (myrank == 0)
   {
      args.PrintOptions(std::cout);
   }

   // Read the (serial) mesh from the given mesh file on all processors.  We
   // can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   // and volume meshes with the same code.
   mfem::Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();
   int spaceDim = mesh.SpaceDimension();


   // Refine the serial mesh on all processors to increase the resolution. In
   // this example we do 'ref_levels' of uniform refinement. We choose
   // 'ref_levels' to be the largest number that gives a final mesh with no
   // more than 10,000 elements.
   {
      int ref_levels =
         (int)floor(log(1000./mesh.GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
   }

   // Define a parallel mesh by a partitioning of the serial mesh. Refine
   // this mesh further in parallel to increase the resolution. Once the
   // parallel mesh is defined, the serial mesh can be deleted.
   mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   {
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh.UniformRefinement();
      }
   }

   std::cout<<"My rank="<<pmesh.GetMyRank()<<std::endl;
   pmesh.PrintInfo(std::cout);

   H1_FECollection* vfec=new H1_FECollection(order, dim);
   H1_FECollection* pfec=new H1_FECollection(order-1, dim);


   //construct the FEM spaces
   ParFiniteElementSpace* vfes= new ParFiniteElementSpace(&pmesh, vfec, dim,
                                                          Ordering::byNODES);
   ParFiniteElementSpace* pfes=new ParFiniteElementSpace(&pmesh, pfec);

   Vector y; y.SetSize(vfes->TrueVSize()); y.Randomize();
   Vector x; x.SetSize(vfes->TrueVSize()); x.Randomize();
   ParGridFunction xgf(vfes);
   vfes->GetProlongationMatrix()->Mult(x, xgf);


   ConstantCoefficient zerocoef(0.0);
   ConstantCoefficient visc(1.0);

   Array<int> ess_tdofv;

   ParBilinearForm* stokes_bf=new ParBilinearForm(vfes);
   stokes_bf->AddDomainIntegrator(new ElasticityIntegrator(zerocoef,visc));
   stokes_bf->Assemble();
   stokes_bf->Finalize();
   std::unique_ptr<HypreParMatrix> A(stokes_bf->ParallelAssemble());
   delete stokes_bf;

   //partial assembly
   stokes_bf=new ParBilinearForm(vfes);
   stokes_bf->AddDomainIntegrator(new ElasticityIntegrator(zerocoef,visc));
   stokes_bf->SetAssemblyLevel(mfem::AssemblyLevel::PARTIAL);
   stokes_bf->Assemble();
   mfem::ConstrainedOperator *Kc;
   std::unique_ptr<mfem::OperatorHandle> Kh;
   {
      Operator *Kop;
      stokes_bf->FormSystemOperator(ess_tdofv, Kop);
      Kh = std::make_unique<OperatorHandle>(Kop);
      Kc = dynamic_cast<mfem::ConstrainedOperator*>(Kop);
   }

   //dfem operator
   static constexpr int U = 0, Coords = 1, LCoeff = 2, MuCoeff = 3;
   std::unique_ptr<mfem::CoefficientVector> Mu_cv;
   std::unique_ptr<mfem::future::DifferentiableOperator> dop;

   pmesh.EnsureNodes();
   ParGridFunction* nodes(static_cast<ParGridFunction *>(pmesh.GetNodes()));
   ParFiniteElementSpace* mfes(static_cast<ParFiniteElementSpace*>
                               (nodes->ParFESpace()));

   const mfem::FiniteElement *fe(vfes->GetFE(0));
   //const mfem::IntegrationRule &ir(IntRules.Get(fe->GetGeomType(),
   //                fe->GetOrder() + fe->GetOrder() + fe->GetDim() - 1 + 5));
   const mfem::IntegrationRule &ir(IntRules.Get(fe->GetGeomType(),
                                                fe->GetOrder() + fe->GetOrder()));
   mfem::QuadratureSpace qs(pmesh, ir);
   NqptUniformParameterSpace Mu_ps(pmesh, ir, 1) ;

   std::cout<<"rank="<<myrank<<" qspace.size="<<qs.GetSize()<<std::endl;

   std::cout<<"rank="<<myrank<<" mu_ps.size="<<Mu_ps.GetTrueVSize()<<std::endl;

   Array<int> domain_attributes;
   if (pmesh.attributes.Size() > 0)
   {
      domain_attributes.SetSize(pmesh.attributes.Max());
      domain_attributes = 1;
   }

   // sample mu on the integration points
   Mu_cv = std::make_unique<CoefficientVector>(visc, qs);

   std::cout<<"rank="<<myrank<<" m_cv.size="<<Mu_cv->Size()<<std::endl;


   // define the differentiable operator
   dop = std::make_unique<mfem::future::DifferentiableOperator>(
   std::vector<mfem::future::FieldDescriptor> {{ U, vfes }},
   std::vector<mfem::future::FieldDescriptor>
   {
      { MuCoeff, &Mu_ps},
      { Coords, mfes }
   },
   pmesh);

   dop->SetParameters({ Mu_cv.get(), nodes });

   const auto inputs =
      mfem::future::tuple{ Gradient<U>{},
                           Identity<MuCoeff>{},
                           Gradient<Coords>{},
                           Weight{} };

   const auto output = mfem::future::tuple{ Gradient<U>{} };

   //const auto output = mfem::future::tuple{ Identity<MuCoeff>{} };

   //define the q-function
   if (2 == spaceDim)
   {
      typename SQFunction<2>::Stokes s2qf;
      dop->AddDomainIntegrator(s2qf, inputs, output, ir, domain_attributes);
   }
   else if (3 == spaceDim)
   {
      typename SQFunction<3>::Stokes s3qf;
      dop->AddDomainIntegrator(s3qf, inputs, output, ir, domain_attributes);
   }
   else { MFEM_ABORT("Space dimension not supported"); }

   mfem::ConstrainedOperator *Kdc;
   std::unique_ptr<mfem::OperatorHandle> Kdh;
   {
      Operator *Kdop;
      dop->FormSystemOperator(ess_tdofv, Kdop);
      Kdh = std::make_unique<OperatorHandle>(Kdop);
      Kdc = dynamic_cast<mfem::ConstrainedOperator*>(Kdop);
   }


   //differentiable operator
   std::unique_ptr<mfem::future::DifferentiableOperator> dopd;
   //define the differentiable operator
   dopd = std::make_unique<mfem::future::DifferentiableOperator>(
   std::vector<mfem::future::FieldDescriptor> {{ U, vfes }},
   std::vector<mfem::future::FieldDescriptor>
   {
      { MuCoeff, &Mu_ps},
      { Coords, mfes }
   },
   pmesh);

   //dopd->SetParameters({ Mu_cv.get(), nodes });
   //define the q-function
   if (2 == spaceDim)
   {
      using mfem::future::dual;
      using dual_t = dual<real_t, real_t>;
      typename SQFunction<2,dual_t>::Stokes s2qf;
      auto derivatives = std::integer_sequence<size_t, U, Coords> {};
      dopd->AddDomainIntegrator(s2qf, inputs, output, ir, domain_attributes,
                                derivatives);

   }
   else if (3 == spaceDim)
   {
      using mfem::future::dual;
      using dual_t = dual<real_t, real_t>;
      typename SQFunction<3,dual_t>::Stokes s3qf;
      auto derivatives = std::integer_sequence<size_t, U, Coords> {};
      dopd->AddDomainIntegrator(s3qf, inputs, output, ir, domain_attributes,
                                derivatives);
   }
   else { MFEM_ABORT("Space dimension not supported"); }



   std::shared_ptr<mfem::future::DerivativeOperator> dres_du;
   //the parameters should be set from grid functions
   dres_du=dopd->GetDerivative(U, {&xgf}, {Mu_cv.get(), nodes});


   std::shared_ptr<mfem::future::DerivativeOperator> dres_dcoor;
   dres_dcoor=dopd->GetDerivative(Coords, {&xgf}, {Mu_cv.get(), nodes});


   double dt;
   int maxnit=2;
   //test the full assembly matrix-vector product
   tic();
   for (int i=0; i<maxnit; i++) {  A->Mult(x, y);   }
   dt=toc();
   real_t normy1=InnerProduct(pmesh.GetComm(), y, y);
   if (0==myrank)
   {
      std::cout << "y = Af*x computed: " << normy1 << " dt=" << dt <<std::endl;
   }

   //test partial matrix-vector product
   tic();
   for (int i=0; i<maxnit; i++) {   Kc->Mult(x, y);}
   dt=toc();
   real_t normy2=InnerProduct(pmesh.GetComm(), y, y);
   if (0==myrank)
   {
      std::cout << "y = Ap*x computed: " << normy2 << " dt=" << dt <<std::endl;
      //  std::cout << "Difference norm = " << fabs(normy1 - normy2) <<std::endl;
   }

   //test dfem matrix-vector product
   tic();
   for (int i=0; i<maxnit; i++) {   Kdc->Mult(x, y); }
   dt=toc();
   real_t normy3=InnerProduct(pmesh.GetComm(), y, y);
   if (0==myrank)
   {
      std::cout << "y = Ad*x computed: " << normy3 << " dt=" << dt <<std::endl;
      // std::cout << "Difference norm = " << fabs(normy1 - normy3) <<std::endl;
   }

   //test dfem derivative operator matrix-vector product
   tic();
   for (int i=0; i<maxnit; i++) {   dres_du->Mult(x, y);}
   dt=toc();
   real_t normy4=InnerProduct(pmesh.GetComm(), y, y);
   if (0==myrank)
   {
      std::cout << "y = Af*x computed: " << normy4 << " dt=" << dt <<std::endl;
      //  std::cout << "Difference norm = " << fabs(normy1 - normy4) <<std::endl;
   }

   Vector resc; resc.SetSize(mfes->TrueVSize());
   Vector inpv; inpv.SetSize(mfes->TrueVSize()); inpv.Randomize();
   dres_dcoor->Mult(inpv, resc);
   real_t normc=InnerProduct(pmesh.GetComm(), resc, resc);
   real_t normi=InnerProduct(pmesh.GetComm(), inpv, inpv);
   if (0==myrank)
   {
      std::cout << "y = Adcoor*x computed: " << normc <<" inp=" <<normi <<std::endl;
   }


   delete stokes_bf;

   //vector mass tests
   ConstantCoefficient mass_coef(1.0);
   const auto mass_inputs =
      mfem::future::tuple{ mfem::future::Value<U>{},
                           mfem::future::Identity<MuCoeff>{},
                           mfem::future::Gradient<Coords>{},
                           mfem::future::Weight{} };

   const auto mass_output = mfem::future::tuple{ mfem::future::Value<U>{} };
   // sample \rho on the integration points
   Mu_cv = std::make_unique<CoefficientVector>(mass_coef, qs);
   //differentiable operator
   std::unique_ptr<mfem::future::DifferentiableOperator> mopd;
   //define the differentiable operator
   mopd = std::make_unique<mfem::future::DifferentiableOperator>(
   std::vector<mfem::future::FieldDescriptor> {{ U, vfes }},
   std::vector<mfem::future::FieldDescriptor>
   {
      { MuCoeff, &Mu_ps}, //same dimmension as viscoity for Stokes
      { Coords, mfes }
   },
   pmesh);

   //define the q-function
   mopd->SetParameters({ Mu_cv.get(), nodes });
   if (2 == spaceDim)
   {
      using mfem::future::dual;
      using dual_t = dual<real_t, real_t>;
      typename MQFunction<2,dual_t>::Mass m2qf;
      auto derivatives = std::integer_sequence<size_t, U, Coords> {};
      mopd->AddDomainIntegrator(m2qf, mass_inputs, mass_output, ir, domain_attributes,
                                derivatives);

   }
   else if (3 == spaceDim)
   {
      using mfem::future::dual;
      using dual_t = dual<real_t, real_t>;
      typename MQFunction<3,dual_t>::Mass m3qf;
      auto derivatives = std::integer_sequence<size_t, U, Coords> {};
      mopd->AddDomainIntegrator(m3qf, mass_inputs, mass_output, ir, domain_attributes,
                                derivatives);
   }
   else { MFEM_ABORT("Space dimension not supported"); }

   std::shared_ptr<mfem::future::DerivativeOperator> dmass_du;
   //the parameters should be set from grid functions
   dmass_du=mopd->GetDerivative(U, {&xgf}, {Mu_cv.get(), nodes});
   std::shared_ptr<mfem::future::DerivativeOperator> dmass_dcoor;
   dmass_dcoor=mopd->GetDerivative(Coords, {&xgf}, {Mu_cv.get(), nodes});

   //test dfem derivative operator matrix-vector product
   tic();
   for (int i=0; i<maxnit; i++) {   dmass_du->Mult(x, y); }
   dt=toc();
   real_t normm1=InnerProduct(pmesh.GetComm(), y, y);
   if (0==myrank)
   {
      std::cout << "y = Mf*x computed: " << normm1 << " dt=" << dt <<std::endl;
   }

   mfem::ConstrainedOperator *Mdc;
   std::unique_ptr<mfem::OperatorHandle> Mdh;
   {
      Operator *Mdop;
      mopd->FormSystemOperator(ess_tdofv, Mdop);
      Mdh = std::make_unique<OperatorHandle>(Mdop);
      Mdc = dynamic_cast<mfem::ConstrainedOperator*>(Mdop);
   }

   //test dfem derivative operator matrix-vector product
   tic();
   for (int i=0; i<maxnit; i++) {   Mdc->Mult(x, y); }
   dt=toc();
   normm1=InnerProduct(pmesh.GetComm(), y, y);
   if (0==myrank)
   {
      std::cout << "y = Mf*x computed: " << normm1 << " dt=" << dt <<std::endl;
   }

   stokes_bf=new ParBilinearForm(vfes);
   stokes_bf->AddDomainIntegrator(new VectorMassIntegrator(mass_coef));
   stokes_bf->Assemble();
   {
      Operator *Kop;
      stokes_bf->FormSystemOperator(ess_tdofv, Kop);
      Kh = std::make_unique<OperatorHandle>(Kop);
      Kc = dynamic_cast<mfem::ConstrainedOperator*>(Kop);
   }

   //test partial matrix-vector product
   tic();
   for (int i=0; i<maxnit; i++) {   Kc->Mult(x, y);}
   dt=toc();
   real_t normm2=InnerProduct(pmesh.GetComm(), y, y);
   if (0==myrank)
   {
      std::cout << "y = Mp*x computed: " << normm2 << " dt=" << dt <<std::endl;
      std::cout << "Difference norm = " << fabs(normm1 - normm2) <<std::endl;
   }

   mfem::PowerMethod pm(pmesh.GetComm());
   Vector v0(x.Size()); v0.Randomize();
   double eigenvalue=pm.EstimateLargestEigenvalue(*Kdc, v0, 20, 1e-12);
   if (0==myrank)
   {
      std::cout << "Largest eigenvalue of dfem mass matrix: " << eigenvalue
                <<std::endl;
   }






   delete stokes_bf;


   delete pfes;
   delete vfes;
   delete pfec;
   delete vfec;

   MPI::Finalize();
   return 0;
}
