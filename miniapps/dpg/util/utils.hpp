// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "mfem.hpp"
#include <ctime>
#include <string>
#include <sstream>
using namespace mfem;

std::string GetTimestamp();
void WriteParametersToFile(const mfem::OptionsParser& args,
                           const std::string& output_dir);
void CreateParaViewPath(const char* mesh_file, std::string& output_dir);

std::string GetFilename(const std::string& filePath);

class ElementTdofs
{
private:
   const ParFiniteElementSpace* pfes;
   std::vector<int> tdof_offsets;
   MPI_Comm comm;
   int num_procs;
   bool boundary = false;
   // Compute offsets for True DOFs across all processors
   void ComputeTdofOffsets();
   // Determine which rank owns a given True DOF (tdof)
   int GetRank(int tdof);
   // Distribute indices using MPI_Alltoallv
   void DistributeIndices(Array<int>& indices, Array<int>& processors,
                          Array<int>& recv_idx);
public:
   ElementTdofs(const ParFiniteElementSpace* pfes_);
   void EnableBoundary() {boundary = true;}


   // Extract and distribute True DOFs based on the element attribute
   Array<int> GetTrueDOFs(int element_attribute);
   HypreParMatrix * GetProlongationMatrix(int element_attribute);
};


class SolverWithFiltering : public Solver
{
private:
   MPI_Comm comm;
   int numProcs, myid;
   const Operator * Op;
   const Operator * P;
   const Solver * solver;
   const Solver * filter_solver;
   void Init(MPI_Comm comm_);
public:
   SolverWithFiltering(MPI_Comm comm_);
   void SetOperator(const Operator & Op_);
   void SetSubspaceProlongationMap(const Operator & P_);
   void SetSolver(const Solver * solver_);
   void SetFilterSolver(const Solver * filter_solver_);
   virtual void Mult(const Vector & b, Vector & x) const;
};