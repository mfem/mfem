
#include "mfem.hpp"

#include <fstream>
#include <numeric>

using namespace std;
using namespace mfem;


std::vector<double> computeDiscontinutyMagnitudes(ParMesh* pmesh, 
               ParGridFunction* gf)
{
   const ParFiniteElementSpace* fespace = gf->ParFESpace();

   vector<double> discontinuities(pmesh->GetNumFaces()+pmesh->GetNSharedFaces());
   fill(discontinuities.begin(),discontinuities.end(), 0.0);

   for (int f = 0; f < pmesh->GetNumFaces(); f++)
   {
      auto face_element = fespace->GetFaceElement(f);
      auto FT           = pmesh->GetFaceElementTransformations(f);

      ///@TODO how to obtain the "correct" rule?
      auto int_rules = IntegrationRules();
      const auto int_rule = int_rules.Get(FT->FaceGeom, 2 * FT->Face->Order() - 1);
      const auto nip = int_rule.GetNPoints();

      // We just check for continuity in the interior of the field.
      if (pmesh->FaceIsInterior(f))
      {
         Vector jumps(nip*fespace->GetVDim());
         jumps = 0.0;

         // Contribution of half face on the side of e₁.
         for (int i = 0; i < nip; i++)
         {
            // Evaluate gf at IP
            auto fip = int_rule.IntPoint(i);
            IntegrationPoint ip;
            FT->Loc1.Transform(fip, ip);

            Vector val(fespace->GetVDim());
            gf->GetVectorValue(FT->Elem1No, ip, val);

            for(int dim=0;dim<fespace->GetVDim();dim++)
            {
               jumps(fespace->GetVDim()*i + dim) = val(dim)*fip.weight;
            } 
         }

         // Contribution of half face of e₂.
         for (int i = 0; i < nip; i++)
         {
            // Evaluate flux vector at IP
            auto fip = int_rule.IntPoint(i);
            IntegrationPoint ip;
            FT->Loc2.Transform(fip, ip);

            Vector val(fespace->GetVDim());
            gf->GetVectorValue(FT->Elem2No, ip, val);

            for(int dim=0;dim<fespace->GetVDim();dim++)
            {
               jumps(fespace->GetVDim()*i + dim) -= val(dim)*fip.weight;
            } 
         }

         // 
         for(int i=0;i<jumps.Size();i++)
         {
            jumps(i) = abs(jumps(i));
         }

         discontinuities[f] = jumps.Sum();
      }
   }

   pmesh->ExchangeFaceNbrData();
   gf->ExchangeFaceNbrData();

   for (int sf = 0; sf < pmesh->GetNSharedFaces(); sf++)
   {
      auto FT = pmesh->GetSharedFaceTransformations(sf, true);

      ///@TODO how to obtain the "correct" rule?
      auto int_rules = IntegrationRules();
      auto face_element = fespace->GetFaceElement(0);
      const auto int_rule =
            int_rules.Get(FT->FaceGeom, 2 * FT->Face->Order() - 1);
      const auto nip = int_rule.GetNPoints();

      Vector jumps(nip*fespace->GetVDim());
      jumps = 0.0;

      // Contribution of half face on the side of e₁.
      for (int i = 0; i < nip; i++)
      {
            // Evaluate flux vector at integration point
            auto fip = int_rule.IntPoint(i);
            IntegrationPoint ip;
            FT->Loc1.Transform(fip, ip);

            Vector val(fespace->GetVDim());
            gf->GetVectorValue(FT->Elem1No, ip, val);

            for(int dim=0;dim<fespace->GetVDim();dim++)
            {
               jumps(fespace->GetVDim()*i + dim) = val(dim)*fip.weight;
            } 
      }

      // Contribution of half face of e₂.
      for (int i = 0; i < nip; i++)
      {
         // Evaluate flux vector at integration point
         auto fip = int_rule.IntPoint(i);
         IntegrationPoint ip;
         FT->Loc2.Transform(fip, ip);

         Vector val(fespace->GetVDim());
         // @TODO Not working. This seems to be a bug.
         gf->GetVectorValue(FT->Elem2No, ip, val);

         for(int dim=0;dim<fespace->GetVDim();dim++)
         {
            jumps(fespace->GetVDim()*i + dim) -= val(dim)*fip.weight;
         } 
      }

      // 
      for(int i=0;i<jumps.Size();i++)
      {
         jumps(i) = abs(jumps(i));
      }

      discontinuities[pmesh->GetNumFaces()+sf] = jumps.Sum();
   }
   
   return discontinuities;
}



int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   MPI_Session mpi;
   if (!mpi.Root()) { mfem::out.Disable(); mfem::err.Disable(); }

   // 2. Parse command-line options.
   const char *mesh_filename = "";
   const char *gf_filename = "";
   bool visualization = true;
   double tolerance = 1e-14; 
   bool partitioned_mesh = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_filename, "-m", "--mesh",
                  "The mesh on which the solulution to check is defined.", true);
   args.AddOption(&gf_filename, "-gf", "--grid-function", 
                  "The solulution for which continuity will be checked.", true);
   args.AddOption(&tolerance, "-tol", "--tolerance", "Tolerance for the continuity check.");
   args.AddOption(&partitioned_mesh, "-p", "--partitioned", "", "", 
                  "Is the mesh already partitioned.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Visualize discontinuities via GLVis.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(mfem::out);
      return 1;
   }
   args.PrintOptions(mfem::out);

   // 3. Load mesh.
   ParMesh* pmesh = [&mpi](const char *filename, bool partitioned_mesh)
   {
      if(partitioned_mesh)
      {
         string par_filename = MakeParFilename(filename, mpi.WorldRank());
         ifstream par_file(par_filename);
         return new ParMesh(MPI_COMM_WORLD, par_file);
      }
      else 
      {
         ifstream file(filename);
         Mesh mesh(file);
         return new ParMesh(MPI_COMM_WORLD, mesh);
      }
   }(mesh_filename, partitioned_mesh);

   // 4. Load the field to check.
   ParGridFunction* gf = [&mpi](ParMesh* pmesh, const char *filename)
   {
      string par_filename = MakeParFilename(filename, mpi.WorldRank());
      ifstream file(par_filename);
      return new ParGridFunction(pmesh, file);
   }(pmesh, gf_filename);

   auto local_discontinuities = computeDiscontinutyMagnitudes(pmesh, gf);
   bool is_locally_discontinuous = any_of(local_discontinuities.begin(), local_discontinuities.end(), [&tolerance](double val){
      return val > tolerance;
   });
   bool is_discontinuous;
   std::cout << is_locally_discontinuous << endl;
   MPI_Allreduce(&is_locally_discontinuous, &is_discontinuous, 1,
               MPI_CXX_BOOL, MPI_LOR, MPI_COMM_WORLD);
   if(!is_discontinuous)
   {
      if(mpi.Root())
      {
         cout << "Field is continuous." << endl;
      }
   }
   else 
   {
      if(mpi.Root())
      {
         cout << "Field is discontinuous." << endl;
      }

      if(visualization)
      {
         // Jumps will be visualized as piece-wise constants on the corresponding elements
         L2_FECollection disc_fec(0, pmesh->SpaceDimension());
         ParFiniteElementSpace disc_fespace(pmesh, &disc_fec);
         ParGridFunction dgf(&disc_fespace);
         dgf = 0.0;

         // Map jumps
         for (int fi = 0; fi < pmesh->GetNumFaces(); fi++)
         {
            auto FT           = pmesh->GetFaceElementTransformations(fi);
            if(FT->Elem1No >= 0)
            {
               dgf(FT->Elem1No) += local_discontinuities[fi];
               if(FT->Elem2No >= 0)
               {
                  dgf(FT->Elem2No) += local_discontinuities[fi];
               }
            }
         }

         for (int sfi = 0; sfi < pmesh->GetNSharedFaces(); sfi++)
         {
            auto FT = pmesh->GetSharedFaceTransformations(sfi, true);
            if(FT->Elem1No >= 0)
            {
               dgf(FT->Elem1No) += local_discontinuities[pmesh->GetNumFaces()+sfi];
               if(FT->Elem2No >= 0)
               {
                  dgf(FT->Elem2No) += local_discontinuities[pmesh->GetNumFaces()+sfi];
               }
            }
         }

         // GLVis server to visualize to
         char vishost[] = "localhost";
         int visport    = 19916;
         {
            socketstream sol_sock(vishost, visport);

            bool succeeded = sol_sock.good();
      
            bool all_succeeded;
            MPI_Allreduce(&succeeded, &all_succeeded, 1,
                        MPI_CXX_BOOL, MPI_LAND, MPI_COMM_WORLD);
            succeeded = all_succeeded;
      
            if (!succeeded)
            {
               mfem::out << "Connection to " << vishost << ':' << visport
                        << " failed." << endl;
               return 1;
            }

            sol_sock.precision(8);

            // Visualization
            sol_sock << "parallel " << mpi.WorldSize() << " "
               << mpi.WorldRank() << "\n"
               << "solution\n" << *pmesh << dgf
               << "window_title 'Discontinuities'\n" << flush;
         }

         {
            socketstream sol_sock(vishost, visport);

            bool succeeded = sol_sock.good();
      
            bool all_succeeded;
            MPI_Allreduce(&succeeded, &all_succeeded, 1,
                        MPI_CXX_BOOL, MPI_LAND, MPI_COMM_WORLD);
            succeeded = all_succeeded;
      
            if (!succeeded)
            {
               mfem::out << "Connection to " << vishost << ':' << visport
                        << " failed." << endl;
               return 1;
            }

            sol_sock.precision(8);

            // Visualization
            sol_sock << "parallel " << mpi.WorldSize() << " "
               << mpi.WorldRank() << "\n"
               << "solution\n" << *pmesh << *gf
               << "window_title 'Solution'\n" << flush;
         }
      }
   }

   return 0;
}
