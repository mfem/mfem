// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "mesh_operators.hpp"
#include "pmesh.hpp"

namespace mfem
{

MeshOperatorSequence::~MeshOperatorSequence()
{
   // delete in reverse order
   for (int i = sequence.Size()-1; i >= 0; i--)
   {
      delete sequence[i];
   }
}

int MeshOperatorSequence::ApplyImpl(Mesh &mesh)
{
   if (sequence.Size() == 0) { return NONE; }
next_step:
   step = (step + 1) % sequence.Size();
   bool last = (step == sequence.Size() - 1);
   int mod = sequence[step]->ApplyImpl(mesh);
   switch (mod & MASK_ACTION)
   {
      case NONE:     if (last) { return NONE; } goto next_step;
      case CONTINUE: return last ? mod : (REPEAT | (mod & MASK_INFO));
      case STOP:     return STOP;
      case REPEAT:    --step; return mod;
   }
   return NONE;
}

void MeshOperatorSequence::Reset()
{
   for (int i = 0; i < sequence.Size(); i++)
   {
      sequence[i]->Reset();
   }
   step = 0;
}


ThresholdRefiner::ThresholdRefiner(ErrorEstimator &est)
   : estimator(est)
{
   aniso_estimator = dynamic_cast<AnisotropicErrorEstimator*>(&estimator);
   total_norm_p = infinity();
   total_err_goal = 0.0;
   total_fraction = 0.5;
   local_err_goal = 0.0;
   max_elements = std::numeric_limits<long>::max();

   threshold = 0.0;
   num_marked_elements = 0L;
   current_sequence = -1;

   non_conforming = -1;
   nc_limit = 0;
}

double ThresholdRefiner::GetNorm(const Vector &local_err, Mesh &mesh) const
{
#ifdef MFEM_USE_MPI
   ParMesh *pmesh = dynamic_cast<ParMesh*>(&mesh);
   if (pmesh)
   {
      return ParNormlp(local_err, total_norm_p, pmesh->GetComm());
   }
#endif
   return local_err.Normlp(total_norm_p);
}

int ThresholdRefiner::ApplyImpl(Mesh &mesh)
{
   threshold = 0.0;
   num_marked_elements = 0;
   marked_elements.SetSize(0);
   current_sequence = mesh.GetSequence();

   const long num_elements = mesh.GetGlobalNE();
   if (num_elements >= max_elements) { return STOP; }

   const int NE = mesh.GetNE();
   const Vector &local_err = estimator.GetLocalErrors();
   MFEM_ASSERT(local_err.Size() == NE, "invalid size of local_err");

   const double total_err = GetNorm(local_err, mesh);
   if (total_err <= total_err_goal) { return STOP; }

   if (total_norm_p < infinity())
   {
      threshold = std::max(total_err * total_fraction *
                           std::pow(num_elements, -1.0/total_norm_p),
                           local_err_goal);
   }
   else
   {
      threshold = std::max(total_err * total_fraction, local_err_goal);
   }

   for (int el = 0; el < NE; el++)
   {
      if (local_err(el) > threshold)
      {
         marked_elements.Append(Refinement(el));
      }
   }

   if (aniso_estimator)
   {
      const Array<int> &aniso_flags = aniso_estimator->GetAnisotropicFlags();
      if (aniso_flags.Size() > 0)
      {
         for (int i = 0; i < marked_elements.Size(); i++)
         {
            Refinement &ref = marked_elements[i];
            ref.ref_type = aniso_flags[ref.index];
         }
      }
   }

   num_marked_elements = mesh.ReduceInt(marked_elements.Size());
   if (num_marked_elements == 0) { return STOP; }

   mesh.GeneralRefinement(marked_elements, non_conforming, nc_limit);
   return CONTINUE + REFINED;
}

void ThresholdRefiner::Reset()
{
   estimator.Reset();
   current_sequence = -1;
   num_marked_elements = 0;
   // marked_elements.SetSize(0); // not necessary
}

DRLRefiner::DRLRefiner(GridFunction& u_) : u(u_)
{
   int ret = _import_array();
   if (ret < 0) {
      printf("problem with import_array\n");
   }

   obs_x = 42;
   obs_y = 42;

   PyRun_SimpleString("import sys");
   PyRun_SimpleString("sys.path.append('.')");

   // This is a workaround for something in tensorflow that dies without it.
   PyRun_SimpleString("if not hasattr(sys, 'argv'):\n"
                      "  sys.argv  = ['']");

   printf("importing rllib_eval python module... ");
   PyObject* eval_mod = PyImport_ImportModule("rllib_eval");
   if (eval_mod == 0) {
      PyErr_Print();
      exit(1);
   }
   printf("ok.\n");

   printf("getting evaluator class... ");
   PyObject* eval_class = PyObject_GetAttrString(eval_mod, "Evaluator");
   if (eval_class == 0) {
      PyErr_Print();
      exit(1);
   }
   printf("ok.\n");
   Py_DECREF(eval_mod);

   printf("making empty arg list... ");
   PyObject* args = Py_BuildValue("()");
   if (args == 0) {
      PyErr_Print();
      exit(1);
   }
   printf("ok.\n");

   printf("instantiating eval object... ");
   PyObject* eval_obj = PyEval_CallObject(eval_class, args);
   if (eval_obj == NULL) {
      PyErr_Print();
      exit(1);
   }
   printf("ok.\n");

   printf("getting eval method from eval object...\n");
   eval_method = PyObject_GetAttrString(eval_obj, "eval");
   if (eval_method == 0) {
      PyErr_Print();
      exit(1);
   }
   printf("ok.\n");
}

int DRLRefiner::ApplyImpl(Mesh &mesh)
{
   printf("starting drl refiner... \n");
   marked_elements.SetSize(0);

   double ref_w = 2.0;
   assert(obs_x == obs_y);
   int ref_n = obs_x;
   double ref_lo = 0.0 -ref_w;
   double ref_hi = 1.0 +ref_w;
   double ref_dx = (ref_hi -ref_lo)/ref_n;

   for (int k = 0; k < mesh.GetNE(); k++) {

      // assemble matrix of sample points, ref_space -> phys_space
      ElementTransformation* trk = mesh.GetElementTransformation(k);
      IntegrationPoint ipk;
      Vector xk(2);
      DenseMatrix m(2,42*42);
      int c = 0;
      for (int j = 0; j < obs_y; ++j) {
         ipk.y = ref_lo +(j+0.5)*ref_dx;
         for (int i = 0; i < obs_x; ++i) {
            ipk.x = ref_lo +(i+0.5)*ref_dx;
            trk->Transform(ipk, xk);
            m.SetCol(c++,xk);
         }
      }

      // phys_space -> elements, ips
      Array<int> elems;
      Array<IntegrationPoint> ips;
      int n = mesh.FindPoints(m, elems, ips, false);

      // printf("n = %d\n",n);
      // printf("# elems %d\n",elems.Size());
      // printf("# ips %d\n",ips.Size());

      // Build observation from GridFunction using elements, ips
      double* obs = new double[42*42];
      n = 0;
      bool complete = true;
      for (int j = 0; j < obs_y; ++j) {
         for (int i = 0; i < obs_x; ++i) {
            int el = elems[n];
            if (el == -1) {
               obs[i+42*j] = 0.0;
               complete = false;
            }
            else {
               IntegrationPoint& ip = ips[n];
               obs[i+42*j] = u.GetValue(el, ip);
            }
            n++;
         }
      }
      //printf("complete = %d\n",complete);

      // apply policy: state -> action
      bool refine = false;
      if (complete) {

         if (k == 81) {
            printf("element %d",k);
            for (int j = 0; j < 42; j++) {
               for (int i = 0; i < 42; i++) {
                  printf("(%d,%d) %f\n",i,j,obs[i+42*j]);
               }
            }
         }

         // convert to numpy array
         npy_intp dims[3];
         dims[0] = 42;
         dims[1] = 42;
         dims[2] = 1;
         PyObject *pArray = PyArray_SimpleNewFromData(
            3, dims, NPY_DOUBLE, reinterpret_cast<void*>(obs));
         if (pArray == NULL) {
            printf("pArray NULL!\n");
         }
         PyObject* action = PyObject_CallFunctionObjArgs(eval_method,pArray,NULL);
         if (action == 0) {
            PyErr_Print();
            exit(1);
         }

         // parse integer return value
         int action_val;
         PyArg_Parse(action, "i", &action_val);
         printf("action for element %d is %d\n", k, action_val);
         refine = bool(action_val);
      }

      if (refine) {
         marked_elements.Append(Refinement(k));
      }
   }

   long int num_marked_elements = mesh.ReduceInt(marked_elements.Size());
   if (num_marked_elements == 0) { return STOP; }

   bool nonconforming = true;
   mesh.GeneralRefinement(marked_elements, nonconforming, nc_limit);
   return CONTINUE + REFINED;
}

void DRLRefiner::Reset()
{
   current_sequence = -1;
}

int ThresholdDerefiner::ApplyImpl(Mesh &mesh)
{
   if (mesh.Conforming()) { return NONE; }

   const Vector &local_err = estimator.GetLocalErrors();
   bool derefs = mesh.DerefineByError(local_err, threshold, nc_limit, op);

   return derefs ? CONTINUE + DEREFINED : NONE;
}


int Rebalancer::ApplyImpl(Mesh &mesh)
{
#ifdef MFEM_USE_MPI
   ParMesh *pmesh = dynamic_cast<ParMesh*>(&mesh);
   if (pmesh && pmesh->Nonconforming())
   {
      pmesh->Rebalance();
      return CONTINUE + REBALANCED;
   }
#endif
   return NONE;
}


} // namespace mfem
