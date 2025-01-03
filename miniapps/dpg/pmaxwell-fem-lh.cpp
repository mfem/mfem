//                   MFEM Ultraweak DPG Maxwell parallel example
//
// the "ultraweak" (UW) DPG formulation for the Maxwell problem

//      ∇×(1/μ ∇×E) - ω² ϵ E = Ĵ ,   in Ω
//                       E×n = E₀ , on ∂Ω

#include "mfem.hpp"
#include "util/pcomplexweakform.hpp"
#include "../common/mfem-common.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

class AzimuthalECoefficient : public Coefficient
{
private:
   const GridFunction * vgf;
public:
   AzimuthalECoefficient(const GridFunction * vgf_)
      : Coefficient(), vgf(vgf_) {}
   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
   {
      Vector X, E;
      vgf->GetVectorValue(T,ip,E);
      T.Transform(ip, X);
      real_t x = X(0);
      real_t y = X(1);
      real_t r = sqrt(x*x + y*y);

      real_t val = -x*E[1] + y*E[0];
      return val/r;
   }
};


class EpsilonMatrixCoefficient : public MatrixArrayCoefficient
{
private:
   Mesh * mesh = nullptr;
   ParMesh * pmesh = nullptr;
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Array<ParGridFunction * > pgfs;
   Array<GridFunctionCoefficient * > gf_cfs;
   GridFunction * vgf = nullptr;
   int dim;
   int sdim;
   bool vis=false;
public:
   EpsilonMatrixCoefficient(const char * filename, Mesh * mesh_, ParMesh * pmesh_,
                            double scale = 1.0, double vis_=false)
      : MatrixArrayCoefficient(mesh_->Dimension()), mesh(mesh_), pmesh(pmesh_),
        dim(mesh->Dimension()), vis(vis_)
   {
      std::filebuf fb;
      fb.open(filename,std::ios::in);
      std::istream is(&fb);
      vgf = new GridFunction(mesh,is);
      fb.close();
      FiniteElementSpace * vfes = vgf->FESpace();
      int vdim = vfes->GetVDim();
      const FiniteElementCollection * fec = vfes->FEColl();
      FiniteElementSpace * fes = new FiniteElementSpace(mesh, fec);
      int * partitioning = mesh->GeneratePartitioning(num_procs);
      double *data = vgf->GetData();
      GridFunction gf;
      pgfs.SetSize(vdim);
      gf_cfs.SetSize(vdim);
      sdim = sqrt(vdim);
      for (int i = 0; i<sdim; i++)
      {
         for (int j = 0; j<sdim; j++)
         {
            int k = i*sdim+j;
            gf.MakeRef(fes,&data[k*fes->GetVSize()]);
            pgfs[k] = new ParGridFunction(pmesh,&gf,partitioning);
            (*pgfs[k])*=scale;
            gf_cfs[k] = new GridFunctionCoefficient(pgfs[k]);
            // skip if i or j > dim
            if (i<dim && j<dim)
            {
               Set(i,j,gf_cfs[k], true);
            }
         }
      }
   }

   // Visualize the components of the matrix coefficient
   // in separate GLVis windows for each component
   void VisualizeMatrixCoefficient()
   {
      Array<socketstream *> sol_sock(pgfs.Size());
      for (int k = 0; k<pgfs.Size(); k++)
      {
         if (Mpi::Root()) { mfem::out << "Visualizing component " << k << endl; }
         char vishost[] = "localhost";
         int visport = 19916;
         sol_sock[k] = new socketstream(vishost, visport);
         sol_sock[k]->precision(8);
         *sol_sock[k] << "parallel " << num_procs << " " << myid << "\n";
         int i = k/sdim;
         int j = k%sdim;
         // plot with the title "Epsilon Matrix Coefficient Component (i,j)"
         *sol_sock[k] << "solution\n" << *pmesh << *pgfs[k]
                      << "window_title 'Epsilon Matrix Coefficient Component (" << i << "," << j <<
                      ")'" << flush;
      }
   }
   void Update()
   {
      pgfs[0]->ParFESpace()->Update();
      for (int k = 0; k<pgfs.Size(); k++)
      {
         pgfs[k]->Update();
      }
   }

   ~EpsilonMatrixCoefficient()
   {
      for (int i = 0; i<pgfs.Size(); i++)
      {
         delete pgfs[i];
      }
      pgfs.DeleteAll();
   }

};


int main(int argc, char *argv[])
{
   Mpi::Init();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   const char *mesh_file = "data/mesh2D.mesh";
   const char * eps_r_file = "data/eps2D_r.gf";
   const char * eps_i_file = "data/eps2D_i.gf";

   int order = 2;
   int par_ref_levels = 0;
   bool visualization = false;
   double rnum=4.6e9;
   bool paraview = false;
   double factor = 1.0;
   double mu = 1.257e-6/factor;
   double epsilon_scale = 8.8541878128e-12*factor;
   bool mumps_solver = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree)");
   args.AddOption(&par_ref_levels, "-pr", "--parallel-refinement_levels",
                  "Number of parallel refinement levels.");
   args.AddOption(&rnum, "-rnum", "--number_of_wavelenths",
                  "Number of wavelengths");
   args.AddOption(&mu, "-mu", "--permeability",
                  "Permeability of free space (or 1/(spring constant)).");
#ifdef MFEM_USE_MUMPS
   args.AddOption(&mumps_solver, "-mumps", "--mumps-solver", "-no-mumps",
                  "--no-mumps-solver", "Use the MUMPS Solver.");
#endif
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&paraview, "-paraview", "--paraview", "-no-paraview",
                  "--no-paraview",
                  "Enable or disable ParaView visualization.");
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

   double omega = 2.*M_PI*rnum;

   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   Array<int> int_bdr_attr;
   for (int i = 0; i < mesh.GetNBE(); i++)
   {
      if (mesh.FaceIsInterior(mesh.GetBdrElementFaceIndex(i)))
      {
         int_bdr_attr.Append(mesh.GetBdrAttribute(i));
      }
   }
   int_bdr_attr.Sort();
   int_bdr_attr.Unique();

   // mesh.RemoveInternalBoundaries();
   ParMesh pmesh(MPI_COMM_WORLD, mesh);

   EpsilonMatrixCoefficient eps_r_cf(eps_r_file,&mesh,&pmesh, epsilon_scale);
   EpsilonMatrixCoefficient eps_i_cf(eps_i_file,&mesh,&pmesh, epsilon_scale);

   for (int i = 0; i<par_ref_levels; i++)
   {
      pmesh.UniformRefinement();
      eps_r_cf.Update();
      eps_i_cf.Update();
   }

   FiniteElementCollection *fec = new ND_FECollection(order, dim);
   ParFiniteElementSpace *E_fes = new ParFiniteElementSpace(&pmesh, fec);

   // Bilinear form coefficients
   ConstantCoefficient one(1.0);
   ConstantCoefficient muinv(1./mu);

   ScalarMatrixProductCoefficient m_cf_r(-omega*omega, eps_r_cf);
   ScalarMatrixProductCoefficient m_cf_i(-omega*omega, eps_i_cf);

   ParComplexLinearForm *b = new ParComplexLinearForm(E_fes);
   b->Vector::operator=(0.0);

   ParSesquilinearForm *a = new ParSesquilinearForm(E_fes);
   a->AddDomainIntegrator(new CurlCurlIntegrator(muinv),nullptr);
   a->AddDomainIntegrator(new VectorFEMassIntegrator(m_cf_r),
                          new VectorFEMassIntegrator(m_cf_i));

   socketstream E_out_r;
   socketstream E_theta_out_r;
   socketstream E_theta_out_i;

   ParComplexGridFunction E_gf(E_fes);
   E_gf.real() = 0.0;
   E_gf.imag() = 0.0;

   L2_FECollection L2fec(order, dim);
   ParFiniteElementSpace L2_fes(&pmesh, &L2fec);

   ParGridFunction E_theta_r(&L2_fes);
   ParGridFunction E_theta_i(&L2_fes);
   ParGridFunction E_theta(&L2_fes);
   E_theta = 0.0;

   ParaViewDataCollection * paraview_dc = nullptr;
   ParaViewDataCollection * paraview_tdc = nullptr;

   if (paraview)
   {
      paraview_dc = new ParaViewDataCollection(mesh_file, &pmesh);
      paraview_dc->SetPrefixPath("ParaViewFEM2D");
      paraview_dc->SetLevelsOfDetail(order);
      paraview_dc->SetCycle(0);
      paraview_dc->SetDataFormat(VTKFormat::BINARY);
      paraview_dc->SetHighOrderOutput(true);
      paraview_dc->SetTime(0.0); // set the time
      paraview_dc->RegisterField("E_r",&E_gf.real());
      paraview_dc->RegisterField("E_i",&E_gf.imag());
      paraview_dc->RegisterField("E_theta_r",&E_theta_r);
      paraview_dc->RegisterField("E_theta_i",&E_theta_i);

      paraview_tdc = new ParaViewDataCollection(mesh_file, &pmesh);
      paraview_tdc->SetPrefixPath("ParaViewFEM2D/TimeHarmonic");
      paraview_tdc->SetLevelsOfDetail(order);
      paraview_tdc->SetCycle(0);
      paraview_tdc->SetDataFormat(VTKFormat::BINARY);
      paraview_tdc->SetHighOrderOutput(true);
      paraview_tdc->SetTime(0.0); // set the time
      paraview_tdc->RegisterField("E_theta_t",&E_theta);
   }

   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
   Array<int> one_r_bdr;
   Array<int> one_i_bdr;
   Array<int> negone_r_bdr;
   Array<int> negone_i_bdr;

   if (pmesh.bdr_attributes.Size())
   {
      ess_bdr.SetSize(pmesh.bdr_attributes.Max());
      one_r_bdr.SetSize(pmesh.bdr_attributes.Max());
      one_i_bdr.SetSize(pmesh.bdr_attributes.Max());
      negone_r_bdr.SetSize(pmesh.bdr_attributes.Max());
      negone_i_bdr.SetSize(pmesh.bdr_attributes.Max());
      ess_bdr = 1;
      int_bdr_attr.Print();
      // remove internal boundaries
      for (int i = 0; i<int_bdr_attr.Size(); i++)
      {
         ess_bdr[int_bdr_attr[i]-1] = 0;
      }

      E_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      one_r_bdr = 0;  one_i_bdr = 0;
      negone_r_bdr = 0;  negone_i_bdr = 0;

      // attr = 30,2 (real)
      one_r_bdr[30-1] = 1;  one_r_bdr[2-1] = 1;
      // attr = 26,6 (imag)
      one_i_bdr[26-1] = 1;  one_i_bdr[6-1] = 1;
      // attr = 22,10 (real)
      negone_r_bdr[22-1] = 1; negone_r_bdr[10-1] = 1;
      // attr = 18,14 (imag)
      negone_i_bdr[18-1] = 1; negone_i_bdr[14-1] = 1;
   }

   Vector zero(dim); zero = 0.0;
   Vector one_x(dim); one_x = 0.0; one_x(0) = 1.0;
   Vector negone_x(dim); negone_x = 0.0; negone_x(0) = -1.0;
   VectorConstantCoefficient zero_cf(zero);
   VectorConstantCoefficient one_x_cf(one_x);
   VectorConstantCoefficient negone_x_cf(negone_x);

   E_gf.ProjectBdrCoefficientTangent(one_x_cf,zero_cf, one_r_bdr);
   E_gf.ProjectBdrCoefficientTangent(negone_x_cf,zero_cf, negone_r_bdr);
   E_gf.ProjectBdrCoefficientTangent(zero_cf,one_x_cf, one_i_bdr);
   E_gf.ProjectBdrCoefficientTangent(zero_cf,negone_x_cf, negone_i_bdr);

   b->Assemble();
   a->Assemble();

   OperatorPtr Ah;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, E_gf, *b, Ah, X, B);

#ifdef MFEM_USE_MUMPS
   HypreParMatrix *A = Ah.As<ComplexHypreParMatrix>()->GetSystemMatrix();
   // auto cpardiso = new CPardisoSolver(A->GetComm());
   auto solver = new MUMPSSolver(MPI_COMM_WORLD);
   solver->SetMatrixSymType(MUMPSSolver::MatType::UNSYMMETRIC);
   solver->SetPrintLevel(1);
   solver->SetOperator(*A);
   solver->Mult(B,X);
   delete A;
   delete solver;
#else
   MFEM_ABORT("MFEM compiled without mumps");
#endif

   a->RecoverFEMSolution(X, *b, E_gf);

   AzimuthalECoefficient az_e_r(&E_gf.real());
   AzimuthalECoefficient az_e_i(&E_gf.imag());

   E_theta_r.ProjectCoefficient(az_e_r);
   E_theta_i.ProjectCoefficient(az_e_i);

   if (visualization)
   {
      const char * keys = nullptr;
      char vishost[] = "localhost";
      int  visport   = 19916;
      common::VisualizeField(E_out_r,vishost, visport, E_gf.real(),
                             "Numerical Electric field (real part)", 0, 0, 500, 500, keys);
      common::VisualizeField(E_theta_out_r,vishost, visport, E_theta_r,
                             "Numerical Electric field (azimuthal)", 0, 0, 500, 500, keys);
   }

   if (paraview)
   {
      paraview_dc->SetCycle(0);
      paraview_dc->SetTime((double)0);
      paraview_dc->Save();
      delete paraview_dc;

      int num_frames = 32;
      for (int i = 0; i<num_frames; i++)
      {
         real_t t = (real_t)(i % num_frames) / num_frames;
         add(cos(real_t(2.0*M_PI)*t), E_theta_r,
             sin(real_t(2.0*M_PI)*t), E_theta_i, E_theta);
         paraview_tdc->SetCycle(i);
         paraview_tdc->SetTime(t);
         paraview_tdc->Save();
      }
      delete paraview_tdc;
   }


   delete a;
   delete b;
   delete E_fes;
   delete fec;

   return 0;

}