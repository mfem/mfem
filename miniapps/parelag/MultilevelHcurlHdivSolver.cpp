// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// We recommend viewing MFEM's examples 3 and 4 before viewing this miniapp.

#include <fstream>
#include <sstream>
#include <ostream>
#include <string>
#include <vector>
#include <memory>

#include <mpi.h>

#include "elag.hpp"
#include "utilities/MPIDataTypes.hpp"

using namespace mfem;
using namespace parelag;
using namespace std;

void bdrfunc(const Vector &, Vector &);
void rhsfunc(const Vector &, Vector &);

int main(int argc, char *argv[])
{
   // Initialize MPI.
   mpi_session session(argc, argv);
   MPI_Comm comm = MPI_COMM_WORLD;
   int num_ranks, myid;
   MPI_Comm_size(comm, &num_ranks);
   MPI_Comm_rank(comm, &myid);

   Timer total_timer = TimeManager::AddTimer("Program Execution -- Total");
   Timer init_timer = TimeManager::AddTimer("Initial Setup");

   if (!myid)
      cout << "-- This is an example of using a geometric-like multilevel "
           "hierarchy, constructed by ParElag,\n"
           "-- to solve respective finite element H(curl) and H(div) forms: \n"
           "(alpha curl u, curl v) + (beta u, v);\n"
           "(alpha div u, div v) + (beta u, v).\n\n";

   // Get basic parameters from command line.
   const char *xml_file_c = NULL;
   bool hcurl = true;
   bool visualize = false;
   double tolSVD = 1e-3;
   OptionsParser args(argc, argv);
   args.AddOption(&xml_file_c, "-f", "--xml-file",
                  "XML parameter list (an XML file with detailed parameters).");
   args.AddOption(&hcurl, "-curl", "--hcurl", "-div", "--hdiv",
                  "Whether the H(curl) or H(div) form is being solved.");
   args.AddOption(&visualize, "-v", "--visualize", "-nv", "--no-visualize",
                  "Use GLVis to visualize the final solution and the "
                  "agglomerates.");
   args.AddOption(&tolSVD, "-s", "--svd-tol",
                  "SVD tolerance. It is used for filtering out local linear "
                  "dependencies in the basis construction and extension "
                  "process in ParElag. Namely, right singular vectors with "
                  "singular values smaller than this tolerance are removed.");
   args.Parse();
   if (!args.Good())
   {
      if (!myid)
      {
         args.PrintUsage(cout);
      }
      return EXIT_FAILURE;
   }
   if (!xml_file_c)
   {
      if (hcurl)
      {
         xml_file_c = "MultilevelHcurlSolver_cube_example_parameters.xml";
      }
      else
      {
         xml_file_c = "MultilevelHdivSolver_cube_example_parameters.xml";
      }
      if (!myid)
      {
         cout << "No XML parameter list provided! Using default "
              << xml_file_c << "." << endl;
      }
   }
   if (!myid)
   {
      args.PrintOptions(cout);
   }
   string xml_file(xml_file_c);

   // Read and parse the detailed parameter list from file.
   unique_ptr<ParameterList> master_list;
   ifstream xml_in(xml_file);
   if (!xml_in.good())
   {
      if (!myid)
      {
         cerr << "ERROR: Cannot read from input file: " << xml_file << ".\n";
      }
      return EXIT_FAILURE;
   }
   SimpleXMLParameterListReader reader;
   master_list = reader.GetParameterList(xml_in);
   xml_in.close();

   // General parameters for the problem.
   ParameterList& prob_list = master_list->Sublist("Problem parameters", true);

   // The file from which to read the mesh.
   const string meshfile = prob_list.Get("Mesh file", "");

   // The number of times to refine in serial.
   // Negative means refine until mesh is big enough to distribute, i.e.,
   // until the number of elements is 6 times the number of processes.
   int ser_ref_levels = prob_list.Get("Serial refinement levels", -1);

   // The number of times to refine in parallel. This determines the
   // number of levels in the AMGe hierarchy, if amge_levels is set to 0.
   const int par_ref_levels = prob_list.Get("Parallel refinement levels", 2);

   // Number of levels in the AMGe hierarchy. Should not be larger than
   // par_ref_levels + 1. If set to 0, it will be interpreted as equal to
   // par_ref_levels + 1.
   const int amge_levels = prob_list.Get("AMGe levels", 0);

   // The order of the finite elements on the finest level.
   const int feorder = prob_list.Get("Finite element order", 0);

   // The order of the polynomials to include in the coarse spaces
   // (after interpolating them onto the fine space).
   const int upscalingOrder = prob_list.Get("Upscaling order", 0);

   // A list of 1s and 0s stating which boundary attribute is appointed as
   // essential. If only a single entry is given, it is applied to the whole
   // boundary. That is, if a single 0 is given the whole boundary is "natural",
   // while a single 1 means that the whole boundary is essential.
   vector<int> par_ess_attr = prob_list.Get("Essential attributes",
                                            vector<int> {1});

   // A list of (piecewise) constant values for the coefficient 'alpha', in
   // accordance with the mesh attributes. If only a single entry is given, it
   // is applied to the whole mesh/domain.
   vector<double> alpha_vals = prob_list.Get("alpha values",
                                             vector<double> {1.0});

   // A list of (piecewise) constant values for the coefficient 'beta', in
   // accordance with the mesh attributes. If only a single entry is given, it
   // is applied to the whole mesh/domain.
   vector<double> beta_vals = prob_list.Get("beta values",
                                            vector<double> {1.0});

   // The list of solvers to invoke.
   auto list_of_solvers = prob_list.Get<list<string>>("List of linear solvers");

   ostringstream mesh_msg;
   if (!myid)
   {
      mesh_msg << '\n' << string(50, '*') << '\n'
               << "*                     Mesh: " << meshfile << "\n*\n"
               << "*                 FE order: " << feorder << '\n'
               << "*          Upscaling order: " << upscalingOrder << "\n*\n";
   }

   // Read the (serial) mesh from the given mesh file and uniformly refine it.
   shared_ptr<ParMesh> pmesh;
   {
      if (!myid)
      {
         cout << "\nReading and refining serial mesh...\n";
         cout << "Times to refine mesh in serial: " << ser_ref_levels << ".\n";
      }

      ifstream imesh(meshfile);
      if (!imesh)
      {
         if (!myid)
         {
            cerr << "ERROR: Cannot open mesh file: " << meshfile << ".\n";
         }
         return EXIT_FAILURE;
      }

      auto mesh = make_unique<Mesh>(imesh, true, true);
      imesh.close();

      for (int l = 0; l < ser_ref_levels; ++l)
      {
         if (!myid)
         {
            cout << "Refining mesh in serial: " << l + 1 << "...\n";
         }
         mesh->UniformRefinement();
      }

      if (ser_ref_levels < 0)
      {
         ser_ref_levels = 0;
         for (; mesh->GetNE() < 6 * num_ranks; ++ser_ref_levels)
         {
            if (!myid)
            {
               cout << "Refining mesh in serial: " << ser_ref_levels + 1
                    << "...\n";
            }
            mesh->UniformRefinement();
         }
      }

      if (!myid)
      {
         cout << "Times refined mesh in serial: " << ser_ref_levels << ".\n";
         cout << "Building and refining parallel mesh...\n";
         cout << "Times to refine mesh in parallel: " << par_ref_levels
              << ".\n";
         mesh_msg << "*    Serial refinements: " << ser_ref_levels << '\n'
                  << "*      Coarse mesh size: " << mesh->GetNE() << "\n*\n";
      }

      pmesh = make_shared<ParMesh>(comm, *mesh);
   }

   // Mark essential boundary attributes.
   MFEM_VERIFY(par_ess_attr.size() <= 1 ||
               par_ess_attr.size() == (unsigned) pmesh->bdr_attributes.Max(),
               "Incorrect size of the essential attributes vector in parameters"
               " input.");
   vector<Array<int>> ess_attr(1);
   ess_attr[0].SetSize(pmesh->bdr_attributes.Max());
   if (par_ess_attr.size() == 0)
   {
      ess_attr[0] = 1;
   }
   else if (par_ess_attr.size() == 1)
   {
      ess_attr[0] = par_ess_attr[0];
   }
   else
   {
      for (unsigned i = 0; i < par_ess_attr.size(); ++i)
      {
         ess_attr[0][i] = par_ess_attr[i];
      }
   }

   // Initialize piecewise constant coefficients in the form.
   MFEM_VERIFY(alpha_vals.size() <= 1 ||
               alpha_vals.size() == (unsigned) pmesh->attributes.Max(),
               "Incorrect size of the 'alpha' local values vector in parameters"
               " input.");
   MFEM_VERIFY(alpha_vals.size() <= 1 ||
               alpha_vals.size() == (unsigned) pmesh->attributes.Max(),
               "Incorrect size of the 'alpha' local values vector in parameters"
               " input.");
   PWConstCoefficient alpha(pmesh->attributes.Max());
   PWConstCoefficient beta(pmesh->attributes.Max());

   if (alpha_vals.size() == 0)
   {
      alpha = 1.0;
   }
   else if (alpha_vals.size() == 1)
   {
      alpha = alpha_vals[0];
   }
   else
   {
      for (unsigned i = 0; i < alpha_vals.size(); ++i)
      {
         alpha(i+1) = alpha_vals[i];
      }
   }

   if (beta_vals.size() == 0)
   {
      beta = 1.0;
   }
   else if (beta_vals.size() == 1)
   {
      beta = beta_vals[0];
   }
   else
   {
      for (unsigned i = 0; i < beta_vals.size(); ++i)
      {
         beta(i+1) = beta_vals[i];
      }
   }

   // Refine the mesh in parallel.
   const int nDimensions = pmesh->Dimension();

   // This is mainly because AMS and ADS (at least the way ParElag uses them)
   // are bound to be used in 3D. Note that, for the purpose of demonstration,
   // some of the code below is still constructed in a way that is applicable in
   // 2D as well, taking into account that case as well. Also, in 2D, ParElag
   // defaults to H(div) interpretation of form 1.
   MFEM_VERIFY(nDimensions == 3, "Only 3D problems are currently supported.");

   const int nLevels = amge_levels <= 0 ? par_ref_levels + 1 : amge_levels;
   MFEM_VERIFY(nLevels <= par_ref_levels + 1,
               "Number of AMGe levels too high relative to parallel"
               " refinements.");
   vector<int> level_nElements(nLevels);
   for (int l = 0; l < par_ref_levels; ++l)
   {
      if (!myid)
      {
         cout << "Refining mesh in parallel: " << l + 1
              << (par_ref_levels - l > nLevels ? " (not in hierarchy)"
                  : " (in hierarchy)")
              << "...\n";
      }
      if (par_ref_levels - l < nLevels)
      {
         level_nElements[par_ref_levels - l] = pmesh->GetNE();
      }
      pmesh->UniformRefinement();
   }
   level_nElements[0] = pmesh->GetNE();

   if (!myid)
   {
      cout << "Times refined mesh in parallel: " << par_ref_levels << ".\n";
   }

   {
      size_t local_num_elmts = pmesh->GetNE(), global_num_elmts;
      MPI_Reduce(&local_num_elmts, &global_num_elmts, 1, GetMPIType<size_t>(0),
                 MPI_SUM, 0, comm);
      if (!myid)
      {
         mesh_msg << "*  Parallel refinements: " << par_ref_levels << '\n'
                  << "*        Fine mesh size: " << global_num_elmts << '\n'
                  << "*          Total levels: " << nLevels << '\n'
                  << string(50, '*') << "\n\n";
      }
   }

   if (!myid)
   {
      cout << mesh_msg.str();
   }
   pmesh->ReorientTetMesh();
   init_timer.Stop();

   // Obtain the hierarchy of agglomerate topologies.
   Timer agg_timer = TimeManager::AddTimer("Mesh Agglomeration -- Total");
   Timer agg0_timer = TimeManager::AddTimer("Mesh Agglomeration -- Level 0");
   if (!myid)
   {
      cout << "Agglomerating topology for " << nLevels - 1
           << " coarse levels...\n";
   }

   constexpr auto AT_elem = AgglomeratedTopology::ELEMENT;
   // This partitioner simply geometrically coarsens the mesh by recovering the
   // geometric coarse elements as agglomerate elements. That is, it reverts the
   // MFEM uniform refinement procedure to provide agglomeration.
   MFEMRefinedMeshPartitioner partitioner(nDimensions);
   vector<shared_ptr<AgglomeratedTopology>> topology(nLevels);

   if (!myid)
   {
      cout << "Agglomerating level: 0...\n";
   }

   topology[0] = make_shared<AgglomeratedTopology>(pmesh, nDimensions);

   if (!myid)
   {
      cout << "Level 0 global number of mesh entities: "
           << topology[0]->
           GetNumberGlobalTrueEntities((AgglomeratedTopology::Entity)0);
      for (int j = 1; j <= nDimensions; ++j)
         cout << ", " << topology[0]->
              GetNumberGlobalTrueEntities((AgglomeratedTopology::Entity)j);
      cout << endl;
   }

   agg0_timer.Stop();

   for (int l = 0; l < nLevels - 1; ++l)
   {
      Timer aggl_timer = TimeManager::AddTimer(std::string("Mesh "
                                                           "Agglomeration -- Level ").
                                               append(std::to_string(l+1)));
      Array<int> partitioning(topology[l]->GetNumberLocalEntities(AT_elem));
      partitioner.Partition(topology[l]->GetNumberLocalEntities(AT_elem),
                            level_nElements[l + 1], partitioning);

      if (!myid)
      {
         cout << "Agglomerating level: " << l + 1 << "...\n";
      }

      topology[l + 1] = topology[l]->CoarsenLocalPartitioning(partitioning,
                                                              false, false, 2);
      if (!myid)
      {
         cout << "Level " << l + 1 << " global number of mesh entities: "
              << topology[l + 1]->
              GetNumberGlobalTrueEntities((AgglomeratedTopology::Entity)0);
         for (int j = 1; j <= nDimensions; ++j)
         {
            cout << ", " << topology[l + 1]->
                 GetNumberGlobalTrueEntities((AgglomeratedTopology::Entity)j);
         }
         cout << endl;
      }
   }

   agg_timer.Stop();

   if (visualize && nDimensions <= 3)
   {
      for (int l = 1; l < nLevels; ++l)
      {
         ShowTopologyAgglomeratedElements(topology[l].get(), pmesh.get());
      }
   }

   // Construct the hierarchy of spaces, thus forming a hierarchy of (partial)
   // de Rham sequences.
   Timer derham_timer = TimeManager::AddTimer("DeRhamSequence Construction -- "
                                              "Total");
   Timer derham0_timer = TimeManager::AddTimer("DeRhamSequence Construction -- "
                                               "Level 0");
   if (!myid)
   {
      cout << "Building the fine-level de Rham sequence...\n";
   }

   vector<shared_ptr<DeRhamSequence>> sequence(topology.size());

   const int jform = DeRhamSequence::GetForm(nDimensions,
                                             hcurl ? DeRhamSequence::HCURL :
                                             DeRhamSequence::HDIV);
   if (nDimensions == 3)
   {
      sequence[0] = make_shared<DeRhamSequence3D_FE>(topology[0], pmesh.get(),
                                                     feorder, true, false);
   }
   else
   {
      MFEM_VERIFY(nDimensions == 2, "Only 2D or 3D problems are supported "
                  "by the utilized ParElag.");
      if (hcurl)
      {
         MFEM_ABORT("No H(curl) 2D interpretation of form 1 is implemented.");
      }
      sequence[0] = make_shared<DeRhamSequence2D_Hdiv_FE>(topology[0],
                                                          pmesh.get(), feorder,
                                                          true, false);
   }

   // To build H(curl) (form 1 in 3D), it is needed to obtain all forms and
   // spaces with larger indices. To use the so called "Hiptmair smoothers", a
   // one form lower is needed (H1, form 0). Anyway, to use AMS all forms and
   // spaces to H1 (0 form) are needed. Therefore, the entire de Rham complex is
   // constructed.
   // To build H(div) (form 2 in 3D), it is needed to obtain all forms and
   // spaces with larger indices. To use the so called "Hiptmair smoothers", a
   // one form lower is needed (H(curl), form 1, in 3D). To use AMS and ADS, all
   // forms and spaces to H1 (0 form) are needed. Therefore, the entire de Rham
   // complex is constructed.
   sequence[0]->SetjformStart(0);

   DeRhamSequenceFE *DRSequence_FE = sequence[0]->FemSequence();
   MFEM_VERIFY(DRSequence_FE,
               "Failed to obtain the fine-level de Rham sequence.");

   if (!myid)
   {
      cout << "Level 0 global number of dofs: "
           << DRSequence_FE->GetDofHandler(0)->GetDofTrueDof().
           GetTrueGlobalSize();
      for (int j = 1; j <= nDimensions; ++j)
      {
         cout << ", " << DRSequence_FE->GetDofHandler(j)->GetDofTrueDof().
              GetTrueGlobalSize();
      }
      cout << endl;
   }

   if (!myid)
   {
      cout << "Setting coefficients and computing fine-level local "
           << "matrices...\n";
   }

   DRSequence_FE->ReplaceMassIntegrator(AT_elem, jform,
                                        make_unique<VectorFEMassIntegrator>(beta), false);
   if (hcurl && nDimensions == 3)
   {
      DRSequence_FE->ReplaceMassIntegrator(AT_elem, jform + 1,
                                           make_unique<VectorFEMassIntegrator>(alpha), true);
   }
   else
   {
      DRSequence_FE->ReplaceMassIntegrator(AT_elem, jform + 1,
                                           make_unique<MassIntegrator>(alpha), true);
   }

   if (!myid)
   {
      cout << "Interpolating and setting polynomial targets...\n";
   }

   DRSequence_FE->SetUpscalingTargets(nDimensions, upscalingOrder);
   derham0_timer.Stop();

   if (!myid)
   {
      cout << "Building the coarse-level de Rham sequences...\n";
   }

   for (int l = 0; l < nLevels - 1; ++l)
   {
      Timer derhaml_timer = TimeManager::AddTimer(std::string("DeRhamSequence "
                                                              "Construction -- Level ").
                                                  append(std::to_string(l+1)));
      if (!myid)
      {
         cout << "Building the level " << l + 1 << " de Rham sequences...\n";
      }

      sequence[l]->SetSVDTol(tolSVD);
      sequence[l + 1] = sequence[l]->Coarsen();

      if (!myid)
      {
         auto DRSequence = sequence[l + 1];
         cout << "Level " << l + 1 << " global number of dofs: "
              << DRSequence->GetDofHandler(0)->GetDofTrueDof().
              GetTrueGlobalSize();
         for (int j = 1; j <= nDimensions; ++j)
         {
            cout << ", " << DRSequence->GetDofHandler(j)->GetDofTrueDof().
                 GetTrueGlobalSize();
         }
         cout << endl;
      }
   }
   derham_timer.Stop();

   Timer assemble_timer = TimeManager::AddTimer("Fine Matrix Assembly");
   if (!myid)
   {
      cout << "Assembling the fine-level system...\n";
   }

   VectorFunctionCoefficient rhscoeff(nDimensions, rhsfunc);
   VectorFunctionCoefficient solcoeff(nDimensions, bdrfunc);

   // Take the vector FE space and construct a RHS linear form on it. Then, move
   // the linear form to a vector. This is local, i.e. on all known dofs for the
   // process.
   FiniteElementSpace *fespace = DRSequence_FE->GetFeSpace(jform);
   auto rhsform = make_unique<LinearForm>(fespace);
   rhsform->AddDomainIntegrator(new VectorFEDomainLFIntegrator(rhscoeff));
   rhsform->Assemble();
   unique_ptr<Vector> rhs = move(rhsform);

   // Obtain the boundary data. This is local, i.e. on all known dofs for the
   // process.
   auto solgf = make_unique<GridFunction>(fespace);
   solgf->ProjectCoefficient(solcoeff);
   unique_ptr<Vector> sol = move(solgf);

   // Create the parallel linear system.
   const SharingMap& hcurlhdiv_dofTrueDof =
      sequence[0]->GetDofHandler(jform)->GetDofTrueDof();

   // System RHS, B. It is defined on the true dofs owned by the process.
   Vector B(hcurlhdiv_dofTrueDof.GetTrueLocalSize());

   // System matrix, A.
   shared_ptr<HypreParMatrix> A;
   {
      // Get the mass and derivative operators.
      // For H(curl):
      // M1 represents the form (beta u, v) on H(curl) vector fields.
      // M2 represents the form (alpha u, v) on H(div) vector fields, in 3D.
      // D1 is the curl operator from H(curl) vector fields to H(div) vector
      // fields, in 3D.
      // In 2D, instead of considering H(div) vector fields, L2 scalar fields
      // are to be considered.
      // Thus, D1^T * M2 * D1 represents the form (alpha curl u, curl v) on
      // H(curl) vector fields.
      // For H(div):
      // M1 represents the form (beta u, v) on H(div) vector fields.
      // M2 represents the form (alpha u, v) on L2 scalar fields.
      // D1 is the divergence operator from H(div) vector fields to L2 scalar
      // fields.
      // Thus, D1^T * M2 * D1 represents the form (alpha div u, div v) on H(div)
      // vector fields.
      auto M1 = sequence[0]->ComputeMassOperator(jform),
           M2 = sequence[0]->ComputeMassOperator(jform + 1);
      auto D1 = sequence[0]->GetDerivativeOperator(jform);

      // spA = D1^T * M2 * D1 + M1 represents the respective H(curl) or H(div)
      // form:
      //    (alpha curl u, curl v) + (beta u, v), on H(curl) vector fields;
      //    (alpha div u, div v) + (beta u, v), on H(div) vector fields.
      // This is local, i.e. on all known dofs for the process.
      auto spA = ToUnique(Add(*M1, *ToUnique(RAP(*D1, *M2, *D1))));

      // Eliminate the boundary conditions
      Array<int> marker(spA->Height());
      marker = 0;
      sequence[0]->GetDofHandler(jform)->MarkDofsOnSelectedBndr(ess_attr[0],
                                                                marker);

      for (int i = 0; i < spA->Height(); ++i)
      {
         if (marker[i])
         {
            spA->EliminateRowCol(i, sol->Elem(i), *rhs);
         }
      }

      A = Assemble(hcurlhdiv_dofTrueDof, *spA, hcurlhdiv_dofTrueDof);
      hcurlhdiv_dofTrueDof.Assemble(*rhs, B);
   }
   if (!myid)
   {
      cout << "A size: " << A->GetGlobalNumRows() << 'x'
           << A->GetGlobalNumCols() << '\n' << " A NNZ: " << A->NNZ() << '\n';
   }
   MFEM_VERIFY(B.Size() == A->Height(),
               "Matrix and vector size are incompatible.");
   assemble_timer.Stop();

   // Perform the solves.
   Timer solvers_timer = TimeManager::AddTimer("Solvers -- Total");
   if (!myid)
   {
      cout << "\nRunning fine-level solvers...\n\n";
   }

   // Create the solver library.
   auto lib = SolverLibrary::CreateLibrary(
                 master_list->Sublist("Preconditioner Library"));

   // Loop through the solvers.
   for (const auto& solver_name : list_of_solvers)
   {
      Timer solver_timer = TimeManager::AddTimer(std::string("Solver \"").
                                                 append(solver_name).
                                                 append("\" -- Total"));
      // Get the solver factory.
      auto solver_factory = lib->GetSolverFactory(solver_name);
      auto solver_state = solver_factory->GetDefaultState();
      solver_state->SetDeRhamSequence(sequence[0]);
      solver_state->SetBoundaryLabels(ess_attr);
      solver_state->SetForms({jform});

      // Build the solver.
      Timer build_timer = TimeManager::AddTimer(std::string("Solver \"").
                                                append(solver_name).
                                                append("\" -- Build"));
      if (!myid)
      {
         cout << "Building solver \"" << solver_name << "\"...\n";
      }
      unique_ptr<Solver> solver = solver_factory->BuildSolver(A, *solver_state);
      build_timer.Stop();

      // Run the solver.
      Timer pre_timer = TimeManager::AddTimer(std::string("Solver \"").
                                              append(solver_name).
                                              append("\" -- Pre-solve"));
      if (!myid)
      {
         cout << "Solving system with \"" << solver_name << "\"...\n";
      }

      // Note that X is on true dofs owned by the process, while x is on local
      // dofs that are known to the process.
      Vector X(A->Width()), x(sequence[0]->GetNumberOfDofs(jform));
      X=0.0;

      {
         Vector tmp(A->Height());
         A->Mult(X, tmp);
         tmp *= -1.0;
         tmp += B;

         double local_norm = tmp.Norml2();
         local_norm *= local_norm;
         double global_norm;
         MPI_Reduce(&local_norm, &global_norm, 1, GetMPIType(local_norm),
                    MPI_SUM, 0, comm);
         if (!myid)
         {
            cout << "Initial residual l2 norm: " << sqrt(global_norm) << '\n';
         }
      }
      pre_timer.Stop();

      // Perform the solve.
      Timer solve_timer = TimeManager::AddTimer(std::string("Solver \"").
                                                append(solver_name).
                                                append("\" -- Solve"));
      solver->Mult(B, X);
      solve_timer.Stop();

      Timer post_timer = TimeManager::AddTimer(std::string("Solver \"").
                                               append(solver_name).
                                               append("\" -- Post-solve"));
      {
         Vector tmp(A->Height());
         A->Mult(X, tmp);
         tmp *= -1.0;
         tmp += B;

         double local_norm = tmp.Norml2();
         local_norm *= local_norm;
         double global_norm;
         MPI_Reduce(&local_norm, &global_norm, 1, GetMPIType(local_norm),
                    MPI_SUM, 0, comm);
         if (!myid)
         {
            cout << "Final residual l2 norm: " << sqrt(global_norm) << '\n';
         }
      }

      if (!myid)
      {
         cout << "Solver \"" << solver_name << "\" finished.\n";
      }

      // Visualize the solution.
      if (visualize)
      {
         hcurlhdiv_dofTrueDof.Distribute(X, x);
         MultiVector tmp(x.GetData(), 1, x.Size());
         sequence[0]->show(jform, tmp);
      }
      post_timer.Stop();
   }
   solvers_timer.Stop();

   total_timer.Stop();
   TimeManager::Print();

   if (!myid)
   {
      cout << "\nFinished.\n";
   }

   return EXIT_SUCCESS;
}

// A vector field, used for setting boundary conditions.
void bdrfunc(const Vector &p, Vector &F)
{
   F = 0.0;
}

// The right hand side.
void rhsfunc(const Vector &p, Vector &f)
{
   f = 1.0;
}
