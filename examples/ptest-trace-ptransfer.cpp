#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

real_t sin_func(const Vector &x)
{
   real_t val = sin(M_PI * x.Sum());
   return val;
   // return 1;
}

void sin_vfunc(const Vector &x, Vector &y)
{
   y.SetSize(x.Size());
   for (int i = 0; i < y.Size(); i++)
   {
      y(i) = sin(M_PI * x[i]);
      // y(i) = i;
   }
}

int main(int argc, char *argv[])
{
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();
   // 1. Parse command line options.
   string mesh_file = "../data/star.mesh";
   int order = 1;

   bool visualization = false;

   char vishost[] = "localhost";
   int  visport   = 19916;
   bool assembleP = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&assembleP, "-ap", "--assemble-P", "-no-ap", "--no-assemble-P",
                  "Assemble the P-refinement transfer operator as a matrix or not.");
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

   Mesh mesh(mesh_file);
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   int dim = pmesh.Dimension();
   int sdim = pmesh.SpaceDimension();

   Array<FiniteElementCollection*> fecs, trace_fecs;
   Array<FiniteElementCollection*> preffecs, preftrace_fecs;
   Array<ParFiniteElementSpace*> fespaces, trace_fespaces;
   Array<ParFiniteElementSpace*> preffespaces, preftrace_fespaces;
   Array<ParGridFunction*> gf, trace_gf, trace_gf_mapped;
   Array<ParGridFunction*> prefgf, preftrace_gf, preftrace_gf_mapped;
   Array<socketstream *> sockets, prefsockets;
   Array<socketstream *> tracesockets, preftracesockets;
   Array<TrueTransferOperator*> Ptransfers;
   Array<TrueTransferOperator*> trace_Ptransfers;
   Array<PRefinementTransferOperator*> Ptransfers_assembled;
   Array<PRefinementTransferOperator*> trace_Ptransfers_assembled;

   fecs.Append(new H1_FECollection(order, dim));
   fecs.Append(new RT_FECollection(order, dim));
   fecs.Append(new ND_FECollection(order, dim));

   preffecs.Append(new H1_FECollection(order+1, dim));
   preffecs.Append(new RT_FECollection(order+1, dim));
   preffecs.Append(new ND_FECollection(order+1, dim));


   fespaces.Append(new ParFiniteElementSpace(&pmesh, fecs[0]));
   fespaces.Append(new ParFiniteElementSpace(&pmesh, fecs[0], dim,
                                             Ordering::byNODES));
   fespaces.Append(new ParFiniteElementSpace(&pmesh, fecs[0], dim,
                                             Ordering::byVDIM));
   fespaces.Append(new ParFiniteElementSpace(&pmesh, fecs[1]));
   fespaces.Append(new ParFiniteElementSpace(&pmesh, fecs[2]));

   preffespaces.Append(new ParFiniteElementSpace(&pmesh, preffecs[0]));
   preffespaces.Append(new ParFiniteElementSpace(&pmesh, preffecs[0], dim,
                                                 Ordering::byNODES));
   preffespaces.Append(new ParFiniteElementSpace(&pmesh, preffecs[0], dim,
                                                 Ordering::byVDIM));
   preffespaces.Append(new ParFiniteElementSpace(&pmesh, preffecs[1]));
   preffespaces.Append(new ParFiniteElementSpace(&pmesh, preffecs[2]));


   trace_fecs.Append(new H1_Trace_FECollection(order, dim));
   trace_fecs.Append(new RT_Trace_FECollection(order, dim));
   trace_fecs.Append(new ND_Trace_FECollection(order, dim));

   preftrace_fecs.Append(new H1_Trace_FECollection(order+1, dim));
   preftrace_fecs.Append(new RT_Trace_FECollection(order+1, dim));
   preftrace_fecs.Append(new ND_Trace_FECollection(order+1, dim));

   trace_fespaces.Append(new ParFiniteElementSpace(&pmesh, trace_fecs[0]));
   trace_fespaces.Append(new ParFiniteElementSpace(&pmesh, trace_fecs[0], dim,
                                                   Ordering::byNODES));
   trace_fespaces.Append(new ParFiniteElementSpace(&pmesh, trace_fecs[0], dim,
                                                   Ordering::byVDIM));
   trace_fespaces.Append(new ParFiniteElementSpace(&pmesh, trace_fecs[1]));
   trace_fespaces.Append(new ParFiniteElementSpace(&pmesh, trace_fecs[2]));

   preftrace_fespaces.Append(new ParFiniteElementSpace(&pmesh, preftrace_fecs[0]));
   preftrace_fespaces.Append(new ParFiniteElementSpace(&pmesh, preftrace_fecs[0],
                                                       dim, Ordering::byNODES));
   preftrace_fespaces.Append(new ParFiniteElementSpace(&pmesh, preftrace_fecs[0],
                                                       dim, Ordering::byVDIM));
   preftrace_fespaces.Append(new ParFiniteElementSpace(&pmesh, preftrace_fecs[1]));
   preftrace_fespaces.Append(new ParFiniteElementSpace(&pmesh, preftrace_fecs[2]));

   FunctionCoefficient cf(sin_func);
   VectorFunctionCoefficient vec_cf(sdim, sin_vfunc);

   for (int i = 0; i < fespaces.Size(); i++)
   {
      gf.Append(new ParGridFunction(fespaces[i]));
      *gf[i] = 0.0;
      trace_gf.Append(new ParGridFunction(trace_fespaces[i]));
      *trace_gf[i] = 0.0;
      trace_gf_mapped.Append(new ParGridFunction(fespaces[i]));
      *trace_gf_mapped[i] = 0.0;
      prefgf.Append(new ParGridFunction(preffespaces[i]));
      *prefgf[i] = 0.0;
      preftrace_gf.Append(new ParGridFunction(preftrace_fespaces[i]));
      *preftrace_gf[i] = 0.0;
      preftrace_gf_mapped.Append(new ParGridFunction(preffespaces[i]));
      *preftrace_gf_mapped[i] = 0.0;
      if (visualization)
      {
         sockets.Append(new socketstream(vishost, visport));
         tracesockets.Append(new socketstream(vishost, visport));
         prefsockets.Append(new socketstream(vishost, visport));
         preftracesockets.Append(new socketstream(vishost, visport));
      }
      Ptransfers.Append(new TrueTransferOperator(
                           *(fespaces[i]), *(preffespaces[i])));

      Ptransfers_assembled.Append(
         new PRefinementTransferOperator(
            *(fespaces[i]), *(preffespaces[i]), true));

      trace_Ptransfers.Append(new TrueTransferOperator(
                                 *(trace_fespaces[i]), *(preftrace_fespaces[i])));
      trace_Ptransfers_assembled.Append(
         new PRefinementTransferOperator(
            *(trace_fespaces[i]), *(preftrace_fespaces[i]), true));
   }

   gf[0]->ProjectCoefficient(cf);
   for (int i = 1; i < gf.Size(); i++)
   {
      gf[i]->ProjectCoefficient(vec_cf);
   }

   trace_gf[0]->ProjectTraceCoefficient(cf);
   trace_gf[1]->ProjectTraceCoefficient(vec_cf);
   trace_gf[2]->ProjectTraceCoefficient(vec_cf);
   trace_gf[3]->ProjectTraceCoefficientNormal(vec_cf);
   trace_gf[4]->ProjectTraceCoefficientTangent(vec_cf);

   Array<int> vdofs, trace_vdofs;
   Vector values;
   // prolongate to higher order spaces
   Vector trueX, truepX;
   for (int i = 0; i < fespaces.Size(); i++)
   {
      gf[i]->GetTrueDofs(trueX);
      truepX.SetSize(preffespaces[i]->GetTrueVSize());
      Ptransfers_assembled[i]->GetPrefinementTrueTransferOperator()->Mult(trueX,
                                                                          truepX);
      // Ptransfers[i]->Mult(trueX, truepX);
      prefgf[i]->SetFromTrueDofs(truepX);

      trace_gf[i]->GetTrueDofs(trueX);
      truepX.SetSize(preftrace_fespaces[i]->GetTrueVSize());
      trace_Ptransfers_assembled[i]->GetPrefinementTrueTransferOperator()->Mult(
         trueX, truepX);
      // trace_Ptransfers[i]->Mult(trueX, truepX);
      preftrace_gf[i]->SetFromTrueDofs(truepX);
   }

   // zero out bubble dofs before and after p-ref to compare with trace
   for (int i = 0; i<pmesh.GetNE(); i++)
   {
      for (int j = 0; j < fespaces.Size(); j++)
      {
         fespaces[j]->GetElementInteriorVDofs(i, vdofs);
         gf[j]->SetSubVector(vdofs, 0.0);
         preffespaces[j]->GetElementInteriorVDofs(i, vdofs);
         prefgf[j]->SetSubVector(vdofs, 0.0);
      }
   }

   // embed the trace dofs to a volume GridFunction for comparison
   Array<int> face_vdofs;
   for (int i = 0; i<pmesh.GetNumFaces(); i++)
   {
      for (int j = 0; j < fespaces.Size(); j++)
      {
         values.SetSize(0);
         trace_vdofs.SetSize(0);
         face_vdofs.SetSize(0);
         trace_fespaces[j]->GetFaceVDofs(i, trace_vdofs);
         trace_gf[j]->GetSubVector(trace_vdofs, values);
         fespaces[j]->GetFaceVDofs(i, face_vdofs);
         trace_gf_mapped[j]->SetSubVector(face_vdofs,values);


         values.SetSize(0);
         trace_vdofs.SetSize(0);
         face_vdofs.SetSize(0);
         preftrace_fespaces[j]->GetFaceVDofs(i, trace_vdofs);
         preftrace_gf[j]->GetSubVector(trace_vdofs, values);
         preffespaces[j]->GetFaceVDofs(i, face_vdofs);
         preftrace_gf_mapped[j]->SetSubVector(face_vdofs,values);
      }
   }

   std::vector<std::string> titles =
   {
      "H1 Field",
      "Vec H1 Field (byNODES)",
      "Vec H1 Field (byVDIM)",
      "RT Field",
      "ND Field"
   };

   std::vector<std::string> trace_titles =
   {
      "H1 Trace",
      "Vec H1 Trace (byNODES)",
      "Vec H1 Trace (byVDIM)",
      "RT Trace (normal)",
      "ND Trace (tangent)"
   };

   std::vector<std::string> preftitles =
   {
      "Pref H1 Field",
      "Pref Vec H1 Field (byNODES)",
      "Pref Vec H1 Field (byVDIM)",
      "Pref RT Field",
      "Pref ND Field"
   };

   std::vector<std::string> preftrace_titles =
   {
      "Pref H1 Trace",
      "Pref Vec H1 Trace (byNODES)",
      "Pref Vec H1 Trace (byVDIM)",
      "Pref RT Trace (normal)",
      "Pref ND Trace (tangent)"
   };

   if (visualization)
   {
      for (int i = 0; i < fespaces.Size(); i++)
      {
         {
            *(sockets[i]) << "parallel " << num_procs << " " << myid << "\n";
            sockets[i]->precision(8);
            *(sockets[i]) << "solution\n" << pmesh << *(gf[i])
                          << "window_title '" << titles[i] << "'" << flush;
            MPI_Barrier(MPI_COMM_WORLD); // try to prevent streams from mixing
         }
         {
            *(tracesockets[i]) << "parallel " << num_procs << " " << myid << "\n";
            tracesockets[i]->precision(8);
            *(tracesockets[i]) << "solution\n" << pmesh << *(trace_gf_mapped[i])
                               << "window_title '" << trace_titles[i] << "'"
                               << flush;
            MPI_Barrier(MPI_COMM_WORLD); // try to prevent streams from mixing
         }

         {
            *(prefsockets[i]) << "parallel " << num_procs << " " << myid << "\n";
            prefsockets[i]->precision(8);
            *(prefsockets[i]) << "solution\n" << pmesh << *(prefgf[i])
                              << "window_title '" << preftitles[i] << "'" << flush;
            MPI_Barrier(MPI_COMM_WORLD); // try to prevent streams from mixing
         }

         {
            *(preftracesockets[i]) << "parallel " << num_procs << " " << myid << "\n";
            preftracesockets[i]->precision(8);
            *(preftracesockets[i]) << "solution\n" << pmesh << *(preftrace_gf_mapped[i])
                                   << "window_title '" << preftrace_titles[i] << "'"
                                   << flush;
            MPI_Barrier(MPI_COMM_WORLD); // try to prevent streams from mixing
         }
      }
   }

   // Compute Error norms between trace and volume
   for (int i = 0; i < fespaces.Size(); i++)
   {
      *trace_gf_mapped[i] -= *gf[i];
      real_t err = std::sqrt(InnerProduct(MPI_COMM_WORLD, *trace_gf_mapped[i],
                                          *trace_gf_mapped[i]));

      if (Mpi::Root())
      {
         cout << "\n||" << titles[i] << " - " << trace_titles[i]
              << "||₂" << std::string(10, ' ') << " = "  << err << endl;
      }

      *preftrace_gf_mapped[i] -= *prefgf[i];
      real_t tr_err = std::sqrt(InnerProduct(MPI_COMM_WORLD, *preftrace_gf_mapped[i],
                                             *preftrace_gf_mapped[i]));

      if (Mpi::Root())
      {
         cout << "||" << preftitles[i] << " - " << preftrace_titles[i]
              << "||₂ = " << tr_err << endl;
      }
   }

   return 0;
}