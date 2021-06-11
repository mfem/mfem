#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace mfem;
using namespace std;

class MAL_DRLRefiner : public MeshOperator
{
protected:
   GridFunction& u;

   Array<Refinement> marked_elements;
   long current_sequence;

   int imgsz;
   int local_sample;
   int local_context;
   bool observe_jacobian;
   bool observe_error;

   int nc_limit;

   PyObject* eval_method;
   PyObject* get_local_sample_method;
   PyObject* get_local_context_method;
   PyObject* get_observe_error_method;
   PyObject* get_observe_jacobian_method;

   /** @brief Apply the operator to the mesh.
       @return STOP if a stopping criterion is satisfied or no elements were
       marked for refinement; REFINED + CONTINUE otherwise. */
   virtual int ApplyImpl(Mesh &mesh);

public:

   /// Construct a MAL_DRLRefiner that will operate on u.
   MAL_DRLRefiner(GridFunction &u);

   // default destructor (virtual)

   /** @brief Set the maximum ratio of refinement levels of adjacent elements
       (0 = unlimited). */
   void SetNCLimit(int nc_limit)
   {
      MFEM_ASSERT(nc_limit >= 0, "Invalid NC limit");
      this->nc_limit = nc_limit;
   }

   virtual void Reset();
};

MAL_DRLRefiner::MAL_DRLRefiner(GridFunction& u_) : u(u_)
{
   int ret = _import_array();
   if (ret < 0) {
      printf("problem with import_array\n");
   }

   PyRun_SimpleString("import sys");
   PyRun_SimpleString("sys.path.append('.')");

   // This is a workaround for something in tensorflow that dies without it.
   PyRun_SimpleString("if not hasattr(sys, 'argv'):\n"
                      "  sys.argv  = ['']");

   PyObject* eval_mod = PyImport_ImportModule("mal_rllib_eval");
   if (eval_mod == 0) {
      PyErr_Print();
      exit(1);
   }

   PyObject* eval_class = PyObject_GetAttrString(eval_mod, "Evaluator");
   if (eval_class == 0) {
      PyErr_Print();
      exit(1);
   }
   Py_DECREF(eval_mod);

   PyObject* args = Py_BuildValue("()");
   if (args == 0) {
      PyErr_Print();
      exit(1);
   }

   PyObject* eval_obj = PyEval_CallObject(eval_class, args);
   if (eval_obj == NULL) {
      PyErr_Print();
      exit(1);
   }

   eval_method = PyObject_GetAttrString(eval_obj, "eval");
   if (eval_method == 0) {
      PyErr_Print();
      exit(1);
   }

   get_local_sample_method = PyObject_GetAttrString(eval_obj, "get_local_sample");
   get_local_context_method = PyObject_GetAttrString(eval_obj, "get_local_context");
   get_observe_error_method = PyObject_GetAttrString(eval_obj, "get_observe_error");
   get_observe_jacobian_method = PyObject_GetAttrString(eval_obj, "get_observe_jacobian");

   PyObject* local_sample_p = PyObject_CallFunctionObjArgs(
                              get_local_sample_method, nullptr);
   PyObject* local_context_p = PyObject_CallFunctionObjArgs(
                              get_local_context_method, nullptr);
   PyObject* observe_jacobian_p = PyObject_CallFunctionObjArgs(
                              get_observe_jacobian_method, nullptr);
   PyObject* observe_error_p = PyObject_CallFunctionObjArgs(
                              get_observe_error_method, nullptr);
                              
   PyArg_Parse(local_sample_p, "i", &local_sample);
   PyArg_Parse(local_context_p, "i", &local_context);
   int observe_jacobian_i, observe_error_i;
   PyArg_Parse(observe_jacobian_p, "i", &observe_jacobian_i);
   PyArg_Parse(observe_error_p, "i", &observe_error_i);
   observe_jacobian = bool(observe_jacobian_i);
   observe_error = bool(observe_error_i);
}

int MAL_DRLRefiner::ApplyImpl(Mesh &mesh)
{

   marked_elements.SetSize(0);
   imgsz = local_sample + 2 * local_context;
      
   for (int k = 0; k < mesh.GetNE(); k++) {

      // get scalar info
      const int scalar_size = 1 + observe_error + observe_jacobian;
      double* scalar_obs1 = new double[scalar_size];
      scalar_obs1[0] = 1;
      bool boundary = false;
      Array<int> fcs, cor;
      int e1, e2, inf1, inf2, ncf;
      mesh.GetElementEdges(k, fcs, cor);
      for (int f = 0; f < fcs.Size() && boundary == false; f++) {
         mesh.GetFaceElements(fcs[f], &e1, &e2);
         mesh.GetFaceInfos(fcs[f], &inf1, &inf2, &ncf);
         if (e2 < 0 && inf2 < 0 && ncf == -1) {
            boundary = true;
         }
      }
      if (!boundary) {
         scalar_obs1[0] = 2;
      }


      double error_threshold = 1.0e-5;
      if (observe_jacobian && observe_error) {
         scalar_obs1[1] = mesh.GetElementVolume(k);
         scalar_obs1[2] = error_threshold;
      }
      else if (observe_jacobian) {
         scalar_obs1[1] = mesh.GetElementVolume(k);
      }
      else if (observe_error) {
         scalar_obs1[1] = error_threshold;
      }

      // assemble matrix of sample points, ref_space -> phys_space
      ElementTransformation* trk = mesh.GetElementTransformation(k);
      IntegrationPoint ipk;
      Vector xk(2);
      DenseMatrix m(2,imgsz*imgsz);
      int c = 0;
      double r_init = 0.001,
             r_final = 1.0-r_init;
      for (int j = 0; j < imgsz; ++j) {
         for (int i = 0; i < imgsz; ++i) {
            if (!boundary) {
               ipk.y = (j - local_context + 0.5)/local_sample;
               ipk.x = (i - local_context + 0.5)/local_sample;
            }
            else {
               ipk.y = (r_init + j*(r_final-r_init))/(imgsz-1);
               ipk.x = (r_init + i*(r_final-r_init))/(imgsz-1);
            }

            trk->Transform(ipk, xk);
            m.SetCol(c++,xk);
         }
      } 

      // phys_space -> elements, ips
      Array<int> elems(imgsz*imgsz);
      Array<IntegrationPoint> ips(imgsz*imgsz);
      int n = mesh.FindPoints(m, elems, ips, false);

      // Build observation from GridFunction using elements, ips
      double* obs1 = new double[imgsz*imgsz];
      n = 0;
      bool complete = true;
      for (int j = 0; j < imgsz; ++j) {
         for (int i = 0; i < imgsz; ++i) {
            int el = elems[n];
            if (el == -1) {
               obs1[i*imgsz+j] = 0.0;
               complete = false;
            }
            else {
               IntegrationPoint& ip = ips[n];
               obs1[i*imgsz+j] = u.GetValue(el, ip);
            }
            n++;
         }
      }

      // TODO: More efficient way to do the below mirroring
      // invert i
      double* obs2 = new double[imgsz*imgsz];
      n = 0;
      complete = true;
      for (int j = 0; j < imgsz; ++j) {
         for (int i = imgsz-1; i >= 0; --i) {
            // int el = elems[n];
            // if (el == -1) {
            //    obs2[i*imgsz+j] = 0.0;
            //    complete = false;
            // }
            // else {
            //    IntegrationPoint& ip = ips[n];
            //    obs2[i*imgsz+j] = u.GetValue(el, ip);
            // }
            obs2[i*imgsz+j] = obs1[n];
            n++;
         }
      }

      // invert j
      double* obs3 = new double[imgsz*imgsz];
      n = 0;
      complete = true;
      for (int j = imgsz-1; j >= 0; --j) {
         for (int i = 0; i < imgsz; ++i) {
            // int el = elems[n];
            // if (el == -1) {
            //    obs3[i*imgsz+j] = 0.0;
            //    complete = false;
            // }
            // else {
            //    IntegrationPoint& ip = ips[n];
            //    obs3[i*imgsz+j] = u.GetValue(el, ip);
            // }
            obs3[i*imgsz+j] = obs1[n];
            n++;
         }
      }

      // invert i and j
      double* obs4 = new double[imgsz*imgsz];
      n = 0;
      complete = true;
      for (int j = imgsz-1; j >= 0; --j) {
         for (int i = imgsz-1; i >= 0; --i) {
            // int el = elems[n];
            // if (el == -1) {
            //    obs4[i*imgsz+j] = 0.0;
            //    complete = false;
            // }
            // else {
            //    IntegrationPoint& ip = ips[n];
            //    obs4[i*imgsz+j] = u.GetValue(el, ip);
            // }
            obs4[i*imgsz+j] = obs1[n];
            n++;
         }
      }


      // apply policy: state -> action
      bool refine = false;
      if (complete) {

         // convert to numpy array
         npy_intp dims[3];
         dims[0] = imgsz;
         dims[1] = imgsz;
         dims[2] = 1;

         npy_intp scalar_dims[1];
         scalar_dims[0] = scalar_size;

         PyObject *pArray1 = PyArray_SimpleNewFromData(
            3, dims, NPY_DOUBLE, reinterpret_cast<void*>(obs1));
         PyObject *pArray2 = PyArray_SimpleNewFromData(
            3, dims, NPY_DOUBLE, reinterpret_cast<void*>(obs2));
         PyObject *pArray3 = PyArray_SimpleNewFromData(
            3, dims, NPY_DOUBLE, reinterpret_cast<void*>(obs3));
         PyObject *pArray4 = PyArray_SimpleNewFromData(
            3, dims, NPY_DOUBLE, reinterpret_cast<void*>(obs4));   
         PyObject *sArray1 = PyArray_SimpleNewFromData(
            1, scalar_dims, NPY_DOUBLE, reinterpret_cast<void*>(scalar_obs1));
         if (pArray1 == NULL) printf("pArray1 NULL!\n");

         PyObject* action1 = PyObject_CallFunctionObjArgs(
                     eval_method, pArray1, sArray1, nullptr);
         PyObject* action2 = PyObject_CallFunctionObjArgs(
            eval_method, pArray2, sArray1, nullptr);
         PyObject* action3 = PyObject_CallFunctionObjArgs(
            eval_method, pArray3, sArray1, nullptr);
         PyObject* action4 = PyObject_CallFunctionObjArgs(
            eval_method, pArray4, sArray1, nullptr);

         if (action1 == 0 || action2 == 0) {
            PyErr_Print();
            exit(1);
         }

         // parse integer return value
         int action1_val;
         PyArg_Parse(action1, "i", &action1_val);
         int action2_val;
         PyArg_Parse(action2, "i", &action2_val);
         int action3_val;
         PyArg_Parse(action3, "i", &action3_val);
         int action4_val;
         PyArg_Parse(action4, "i", &action4_val);
         refine =
            bool(action1_val) ||
            bool(action2_val) ||
            bool(action3_val) ||
            bool(action4_val);
      }

      if (refine) {
         marked_elements.Append(Refinement(k));
      }
   }

   long int num_marked_elements = mesh.ReduceInt(marked_elements.Size());
   printf("marked %d elements\n",num_marked_elements);
   if (num_marked_elements == 0) { return STOP; }

   bool nonconforming = true;
   mesh.GeneralRefinement(marked_elements, nonconforming, nc_limit);
   return CONTINUE + REFINED;
}

void MAL_DRLRefiner::Reset()
{
   current_sequence = -1;
}
