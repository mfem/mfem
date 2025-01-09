#include "maxwell_utils.hpp"

real_t AzimuthalECoefficient::Eval(ElementTransformation &T,
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

EpsilonMatrixCoefficient::EpsilonMatrixCoefficient(const char * filename,
                                                   Mesh * mesh_, ParMesh * pmesh_,
                                                   real_t scale)
   : MatrixArrayCoefficient(mesh_->Dimension()), mesh(mesh_), pmesh(pmesh_),
     dim(mesh->Dimension())
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
         if (i<dim && j<dim)
         {
            Set(i,j,gf_cfs[k], true);
         }
      }
   }
}

void EpsilonMatrixCoefficient::VisualizeMatrixCoefficient()
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
      *sol_sock[k] << "solution\n" << *pmesh << *pgfs[k]
                   << "window_title 'Epsilon Matrix Coefficient Component (" << i << "," << j <<
                   ")'" << flush;
   }
}

void EpsilonMatrixCoefficient::Update()
{
   pgfs[0]->ParFESpace()->Update();
   for (int k = 0; k<pgfs.Size(); k++)
   {
      pgfs[k]->Update();
   }
}

EpsilonMatrixCoefficient::~EpsilonMatrixCoefficient()
{
   for (int i = 0; i<pgfs.Size(); i++)
   {
      delete pgfs[i];
   }
   pgfs.DeleteAll();
}