/// Templated AD (block) nonlinear form integrators implementations
#pragma once
#include "_ad_intg.hpp"

namespace mfem
{

template <ADEval mode>
inline int ADNonlinearFormIntegrator<mode>::InitInputShapes(
   const FiniteElement &el,
   ElementTransformation &Tr,
   DenseMatrix &shapes)
{
   const int sdim = Tr.GetSpaceDim();
   const int dim = el.GetDim();
   int idx[static_cast<int>(ADEval::NUMOPT)];
   idx[0] = 0;
   idx[1] = idx[0] + (hasFlag(mode, ADEval::QVALUE) ? 1 : 0);
   idx[2] = idx[1] +  (hasFlag(mode, ADEval::VALUE)
                       ? hasFlag(mode, ADEval::VECFE)
                       ? dim // if vector-FE
                       : 1 // if scalar-FE
                       : 0); // no value
   idx[3] = idx[2] + (hasFlag(mode, ADEval::GRAD) ? sdim : 0);
   idx[4] = idx[3] + (hasFlag(mode, ADEval::DIV) ? 1 : 0);
   idx[5] = idx[4] + (hasFlag(mode, ADEval::CURL) ? el.GetCurlDim() : 0);
   const int shapedim = idx[5];
   const int dof = el.GetDof();
   shapes.SetSize(dof, shapedim);

   if constexpr (hasFlag(mode, ADEval::QVALUE)) { shapes.SetCol(idx[0], 0.0); }

   if constexpr (hasFlag(mode, ADEval::VALUE))
   {
      if constexpr (hasFlag(mode, ADEval::VECFE)) { vshape.UseExternalData(shapes.GetData() + dof*idx[1], dof, dim); }
      else { shapes.GetColumnReference(idx[1], shape); }
   }

   if constexpr (hasFlag(mode, ADEval::GRAD))
   {
      gshape.UseExternalData(shapes.GetData() + dof*idx[2],
                             dof, sdim);
   }
   if constexpr (hasFlag(mode, ADEval::DIV))
   {
      shapes.GetColumnReference(idx[3], divshape);
   }

   if constexpr (hasFlag(mode, ADEval::CURL))
   {
      curlshape.UseExternalData(shapes.GetData() + dof*idx[4],
                                dof, el.GetCurlDim());
   }

   return shapedim;
}

template <ADEval mode>
inline void ADNonlinearFormIntegrator<mode>::CalcInputShapes(
   const FiniteElement &el,
   ElementTransformation &Tr,
   const IntegrationPoint &ip,
   DenseMatrix &allshapes)
{
   // Get quadrature value
   // ip should be from the same integration rule with base quadrature
   if constexpr (hasFlag(mode, ADEval::QVALUE)) { allshapes.SetCol(0, 0.0); allshapes(ip.index, 0) = 1.0; }

   // Get value shape
   if constexpr (hasFlag(mode, ADEval::VALUE))
   {
      if constexpr (hasFlag(mode, ADEval::VECFE)) { el.CalcPhysVShape(Tr, vshape); }
      else { el.CalcPhysShape(Tr, shape); }
   }

   // Get gradient shape
   if constexpr (hasFlag(mode, ADEval::GRAD)) { el.CalcPhysDShape(Tr, gshape); }

   // Get divergence shape
   if constexpr (hasFlag(mode, ADEval::DIV))
   {
      if constexpr (hasFlag(mode, ADEval::GRAD))
      {
         gshape.GetRowSums(divshape);
      }
      else
      {
         el.CalcPhysDivShape(Tr, divshape);
      }
   }

   // Get divergence shape
   if constexpr (hasFlag(mode, ADEval::CURL)) { el.CalcPhysCurlShape(Tr, curlshape); }
}

/// Perform the local action of the NonlinearFormIntegrator
template <ADEval mode>
real_t ADNonlinearFormIntegrator<mode>::GetElementEnergy(
   const FiniteElement &el,
   ElementTransformation &Tr,
   const Vector &elfun)
{
   const int dof = el.GetDof();
   const int order = el.GetOrder();

   const int sdim = Tr.GetSpaceDim();
   const int dim = el.GetDim();

   real_t energy = 0.0;

   int shapedim = InitInputShapes(el, Tr, allshapes);
   x.SetSize(f.n_input);
   if constexpr (hasFlag(mode, ADEval::VECTOR))
   {
      elfun_matview.UseExternalData(const_cast<real_t*>(elfun.GetData()),
                                    dof, vdim);
      xmat.UseExternalData(x.GetData(), shapedim, vdim);
   }

   const IntegrationRule * ir = GetIntegrationRule(el, Tr);
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Tr.SetIntPoint(&ip);

      CalcInputShapes(el, Tr, ip, allshapes);

      if constexpr (hasFlag(mode, ADEval::VECTOR))
      {
         MultAtB(allshapes, elfun_matview, xmat);
      }
      else
      {
         allshapes.MultTranspose(elfun, x);
      }
      energy += f(x, Tr, ip)*Tr.Weight()*ip.weight;
   }
   return energy;
}

/// Compute the local <grad f, v>
template <ADEval mode>
void ADNonlinearFormIntegrator<mode>::AssembleElementVector(
   const FiniteElement &el,
   ElementTransformation &Tr,
   const Vector &elfun, Vector &elvect)
{
   const int dof = el.GetDof();
   const int order = el.GetOrder();

   const int sdim = Tr.GetSpaceDim();
   const int dim = el.GetDim();

   real_t w;
   elvect.SetSize(dof*vdim);
   elvect = 0.0;

   x.SetSize(f.n_input);
   jac.SetSize(f.n_input);

   int shapedim = InitInputShapes(el, Tr, allshapes);

   if constexpr (hasFlag(mode, ADEval::VECTOR))
   {
      elfun_matview.UseExternalData(const_cast<real_t*>(elfun.GetData()),
                                    dof, vdim);
      elvectmat.UseExternalData(elvect.GetData(), dof, vdim);
      xmat.UseExternalData(x.GetData(), shapedim, vdim);
      jacMat.UseExternalData(jac.GetData(), shapedim, vdim);
   }

   const IntegrationRule * ir = GetIntegrationRule(el, Tr);
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Tr.SetIntPoint(&ip);
      w = ip.weight * Tr.Weight();

      CalcInputShapes(el, Tr, ip, allshapes);

      // Convert dof to x = [[value, grad], [value, grad], ...]
      if constexpr (hasFlag(mode, ADEval::VECTOR)) { MultAtB(allshapes, elfun_matview, xmat); }
      else { allshapes.MultTranspose(elfun, x); }

      f.Gradient(x, Tr, ip, jac);
      jac *= w;

      if constexpr (hasFlag(mode, ADEval::VECTOR))
      {
         AddMult(allshapes, jacMat, elvectmat);
      }
      else
      {
         allshapes.AddMult(jac, elvect);
      }
   }
}

/// Assemble the local <H_f(x)(u), v>
template <ADEval mode>
void ADNonlinearFormIntegrator<mode>::AssembleElementGrad(
   const FiniteElement &el,
   ElementTransformation &Tr,
   const Vector &elfun, DenseMatrix &elmat)
{
   const int dof = el.GetDof();
   const int order = el.GetOrder();

   const int sdim = Tr.GetSpaceDim();
   const int dim = el.GetDim();

   real_t w;
   elmat.SetSize(dof*vdim);
   elmat = 0.0;

   int shapedim = InitInputShapes(el, Tr, allshapes);
   MFEM_ASSERT(shapedim*vdim == f.n_input,
               "ADNonlinearFormIntegrator: "
               "shapedim*vdim must match n_input");

   x.SetSize(f.n_input);
   H.SetSize(f.n_input);
   Hx.SetSize(dof, shapedim*vdim*vdim);


   if constexpr (hasFlag(mode, ADEval::VECTOR))
   {
      elfun_matview.UseExternalData(const_cast<real_t*>(elfun.GetData()),
                                    dof, vdim);
      xmat.UseExternalData(x.GetData(), shapedim, vdim);
      partelmat.SetSize(dof, dof);
      Hs.UseExternalData(H.GetData(), shapedim, vdim*shapedim*vdim);
   }

   const IntegrationRule * ir = GetIntegrationRule(el, Tr);
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Tr.SetIntPoint(&ip);
      w = ip.weight * Tr.Weight();
      CalcInputShapes(el, Tr, ip, allshapes);

      // Convert dof to x = [[value, grad], [value, grad], ...]
      if constexpr (hasFlag(mode, ADEval::VECTOR)) { MultAtB(allshapes, elfun_matview, xmat); }
      else { allshapes.MultTranspose(elfun, x); }

      f.Hessian(x, Tr, ip, H);
      H *= w;

      if constexpr (hasFlag(mode, ADEval::VECTOR))
      {
         Mult(allshapes, Hs, Hx);
         const int nel = shapedim*dof;
         for (int c=0; c<vdim; c++)
         {
            for (int r=0; r<=c; r++)
            {
               Hxsub.UseExternalData(Hx.GetData() + (c*vdim + r)*nel, dof, shapedim);
               MultABt(allshapes, Hxsub, partelmat);
               elmat.AddSubMatrix(c*dof, r*dof, partelmat);
               if (c != r)
               {
                  elmat.AddSubMatrix(r*dof, c*dof, partelmat);
               }
            }
         }
      }
      else
      {
         Mult(allshapes, H, Hx);
         AddMultABt(allshapes, Hx, elmat);
      }
   }
}

/// @brief Perform the local action of the NonlinearFormIntegrator resulting
/// from a face integral term.
template <ADEval mode>
void ADNonlinearFormIntegrator<mode>::AssembleFaceVector(
   const FiniteElement &el1,
   const FiniteElement &el2,
   FaceElementTransformations &Tr,
   const Vector &elfun, Vector &elvect)
{
   MFEM_ABORT("ADNonlinearFormIntegrator::AssembleFaceVector: "
              "This method is not implemented.");
}


/// @brief Assemble the local action of the gradient of the
/// NonlinearFormIntegrator resulting from a face integral term.
template <ADEval mode>
void ADNonlinearFormIntegrator<mode>::AssembleFaceGrad(
   const FiniteElement &el1,
   const FiniteElement &el2,
   FaceElementTransformations &Tr,
   const Vector &elfun, DenseMatrix &elmat)
{
   MFEM_ABORT("ADNonlinearFormIntegrator::AssembleFaceVector: "
              "This method is not implemented.");
}

template <ADEval... modes>
inline std::array<int, sizeof...(modes)>
                              ADBlockNonlinearFormIntegrator<modes...>::InitInputShapes(
                                 const Array<const FiniteElement *>& els,
                                 ElementTransformation &Tr,
                                 std::vector<DenseMatrix> &shapes)
{
   MFEM_ASSERT(els.Size() == numSpaces,
               "ADBlockNonlinearFormIntegrator: "
               "el.Size()=" << els.Size() << " must match numSpaces=" << numSpaces);
   const int sdim = Tr.GetSpaceDim();
   std::array<int, sizeof...(modes)> shapedims{};

   _constexpr_for([&](auto i)
   {
      constexpr auto mode = modes_arr[i];
      const FiniteElement &el = *els[i];
      const int sdim = Tr.GetSpaceDim();
      const int dim = el.GetDim();
      int idx[static_cast<int>(ADEval::NUMOPT)];
      idx[0] = 0;
      idx[1] = idx[0] + (hasFlag(modes_arr[i], ADEval::QVALUE) ? 1 : 0);
      idx[2] = idx[1] +  (hasFlag(modes_arr[i], ADEval::VALUE)
                          ? hasFlag(modes_arr[i], ADEval::VECFE)
                          ? dim // if vector-FE
                          : 1 // if scalar-FE
                          : 0); // no value
      idx[3] = idx[2] + (hasFlag(modes_arr[i], ADEval::GRAD) ? sdim : 0);
      idx[4] = idx[3] + (hasFlag(modes_arr[i], ADEval::DIV) ? 1 : 0);
      idx[5] = idx[4] + (hasFlag(modes_arr[i], ADEval::CURL) ? el.GetCurlDim() : 0);
      const int shapedim = idx[5];
      const int dof = el.GetDof();
      shapes[i].SetSize(dof, shapedim);

      if constexpr (hasFlag(mode, ADEval::QVALUE)) { shapes[i].SetCol(idx[0], 0.0); }

      if constexpr (hasFlag(mode, ADEval::VALUE))
      {
         if constexpr (hasFlag(mode, ADEval::VECFE)) { vshape[i].UseExternalData(shapes[i].GetData() + dof*idx[1], dof, dim); }
         else { shapes[i].GetColumnReference(idx[1], shape[i]); }
      }

      if constexpr (hasFlag(mode, ADEval::GRAD))
      {
         gshape[i].UseExternalData(shapes[i].GetData() + dof*idx[2],
                                   dof, sdim);
      }
      if constexpr (hasFlag(mode, ADEval::DIV))
      {
         shapes[i].GetColumnReference(idx[3], divshape[i]);
      }

      if constexpr (hasFlag(mode, ADEval::CURL))
      {
         curlshape[i].UseExternalData(shapes[i].GetData() + dof*idx[4],
                                      dof, el.GetCurlDim());
      }
      shapedims[i] = shapedim;
   }, std::make_index_sequence<sizeof...(modes)> {});
   return shapedims;
}
template <ADEval... modes>
inline void
ADBlockNonlinearFormIntegrator<modes...>::CalcInputShapes(
   const Array<const FiniteElement *>& els,
   ElementTransformation &Tr,
   const IntegrationPoint &ip,
   std::vector<DenseMatrix> &allshapes)
{
   _constexpr_for([&](auto i)
   {
      const auto&el = *els[i];
      constexpr auto mode = modes_arr[i];
      // Get quadrature value
      // ip should be from the same integration rule with base quadrature
      if constexpr (hasFlag(mode, ADEval::QVALUE)) { allshapes[i].SetCol(0, 0.0); allshapes[i](ip.index, 0) = 1.0; }

      // Get value shape
      if constexpr (hasFlag(mode, ADEval::VALUE))
      {
         if constexpr (hasFlag(mode, ADEval::VECFE)) { el.CalcPhysVShape(Tr, vshape[i]); }
         else { el.CalcPhysShape(Tr, shape[i]); }
      }

      // Get gradient shape
      if constexpr (hasFlag(mode, ADEval::GRAD)) { el.CalcPhysDShape(Tr, gshape[i]); }

      // Get divergence shape
      if constexpr (hasFlag(mode, ADEval::DIV))
      {
         if constexpr (hasFlag(mode, ADEval::GRAD))
         {
            gshape[i].GetRowSums(divshape[i]);
         }
         else
         {
            el.CalcPhysDivShape(Tr, divshape[i]);
         }
      }

      // Get divergence shape
      if constexpr (hasFlag(mode, ADEval::CURL)) { el.CalcPhysCurlShape(Tr, curlshape[i]); }
   }, std::make_index_sequence<sizeof...(modes)> {});
}

/// Compute the local energy
template <ADEval... modes>
real_t ADBlockNonlinearFormIntegrator<modes...>::GetElementEnergy(
   const Array<const FiniteElement *> &el,
   ElementTransformation &Tr,
   const Array<const Vector*> &elfun)
{
   MFEM_ASSERT(el.Size() == numSpaces,
               "ADBlockNonlinearFormIntegrator: "
               "el.Size()=" << el.Size() << " must match numSpaces=" << numSpaces);
   std::array<int, numSpaces> dof{};
   std::array<int, numSpaces> order{};
   for (int i=0; i<numSpaces; i++)
   {
      dof[i] = el[i]->GetDof();
      order[i] = el[i]->GetOrder();
   }

   const int sdim = Tr.GetSpaceDim();
   const int dim = el[0]->GetDim();

   real_t energy = 0.0;

   std::array<int, numSpaces> shapedim(InitInputShapes(el, Tr, allshapes));
   x.SetSize(f.n_input);
   int x_idx = 0;
   _constexpr_for([&](auto vi)
   {
      xvar[vi].MakeRef(x, x_idx, shapedim[vi]*vdim[vi]);
      x_idx += shapedim[vi]*vdim[vi];
      if constexpr (hasFlag(modes_arr[vi], ADEval::VECTOR))
      {
         elfun_matview[vi].UseExternalData(const_cast<real_t*>(elfun[vi]->GetData()),
                                           dof[vi], vdim[vi]);
         xmat[vi].UseExternalData(xvar[vi].GetData(), shapedim[vi], vdim[vi]);
      }
   }, std::make_index_sequence<sizeof...(modes)> {});

   const IntegrationRule * ir = GetIntegrationRule(el, Tr);
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Tr.SetIntPoint(&ip);

      CalcInputShapes(el, Tr, ip, allshapes);

      _constexpr_for([&](auto vi)
      {
         if constexpr (hasFlag(modes_arr[vi], ADEval::VECTOR))
         {
            MultAtB(allshapes[vi], elfun_matview[vi], xmat[vi]);
         }
         else
         {
            allshapes[vi].MultTranspose(*elfun[vi], xvar[vi]);
         }
      }, std::make_index_sequence<sizeof...(modes)> {});
      energy += f(x, Tr, ip)*Tr.Weight()*ip.weight;
   }
   return energy;
}

/// Perform the local action of the NonlinearFormIntegrator
template <ADEval... modes>
void ADBlockNonlinearFormIntegrator<modes...>::AssembleElementVector(
   const Array<const FiniteElement *>&el,
   ElementTransformation &Tr,
   const Array<const Vector *>&elfun,
   const Array<Vector *>&elvect)
{
   MFEM_ASSERT(el.Size() == numSpaces,
               "ADBlockNonlinearFormIntegrator: "
               "el.Size()=" << el.Size() << " must match numSpaces=" << numSpaces);
   std::array<int, numSpaces> dof{};
   std::array<int, numSpaces> order{};
   for (int i=0; i<numSpaces; i++)
   {
      dof[i] = el[i]->GetDof();
      order[i] = el[i]->GetOrder();
   }

   const int sdim = Tr.GetSpaceDim();
   const int dim = el[0]->GetDim();
   for (int i=0; i<numSpaces; i++)
   {
      elvect[i]->SetSize(dof[i]*vdim[i]);
      *elvect[i] = 0.0;
   }

   std::array<int, numSpaces> shapedim(InitInputShapes(el, Tr, allshapes));
   Array<int> x_idx(numSpaces+1);
   x_idx[0] = 0;
   for (int i=0; i<numSpaces; i++)
   {
      x_idx[i+1] = shapedim[i]*vdim[i];
   }
   x_idx.PartialSum();
   x.SetSize(f.n_input);
   jac.SetSize(f.n_input);
   _constexpr_for([&](auto vi)
   {
      xvar[vi].MakeRef(x, x_idx[vi], shapedim[vi]*vdim[vi]);
      jacVar[vi].MakeRef(jac, x_idx[vi], shapedim[vi]*vdim[vi]);
      if constexpr (hasFlag(modes_arr[vi], ADEval::VECTOR))
      {
         elfun_matview[vi].UseExternalData(const_cast<real_t*>(elfun[vi]->GetData()),
                                           dof[vi], vdim[vi]);
         xmat[vi].UseExternalData(xvar[vi].GetData(), shapedim[vi], vdim[vi]);
         jacVarMat[vi].UseExternalData(jacVar[vi].GetData(), shapedim[vi], vdim[vi]);
      }
   }, std::make_index_sequence<sizeof...(modes)> {});

   const IntegrationRule * ir = GetIntegrationRule(el, Tr);
   real_t w;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Tr.SetIntPoint(&ip);
      w = Tr.Weight()*ip.weight;

      CalcInputShapes(el, Tr, ip, allshapes);

      _constexpr_for([&](auto vi)
      {
         if constexpr (hasFlag(modes_arr[vi], ADEval::VECTOR))
         {
            MultAtB(allshapes[vi], elfun_matview[vi], xmat[vi]);
         }
         else
         {
            allshapes[vi].MultTranspose(*elfun[vi], xvar[vi]);
         }
      }, std::make_index_sequence<sizeof...(modes)> {});
      f.Gradient(x, Tr, ip, jac);
      jac *= w;

      _constexpr_for([&](auto vi)
      {
         if constexpr (hasFlag(modes_arr[vi], ADEval::VECTOR))
         {
            AddMult(allshapes[vi], jacVarMat[vi], elvectmat[vi]);
         }
         else
         {
            allshapes[vi].AddMult(jacVar[vi], *elvect[vi]);
         }
      }, std::make_index_sequence<sizeof...(modes)> {});
   }
}

/// Perform the local action of the NonlinearFormIntegrator
template <ADEval... modes>
void ADBlockNonlinearFormIntegrator<modes...>::AssembleElementGrad(
   const Array<const FiniteElement *>&el,
   ElementTransformation &Tr,
   const Array<const Vector *>&elfun,
   const Array2D<DenseMatrix *>&elmat)
{
   MFEM_ASSERT(el.Size() == numSpaces,
               "ADBlockNonlinearFormIntegrator: "
               "el.Size()=" << el.Size() << " must match numSpaces=" << numSpaces);
   Array<int> dof(numSpaces);
   Array<int> order(numSpaces);
   for (int i=0; i<numSpaces; i++)
   {
      dof[i] = el[i]->GetDof();
      order[i] = el[i]->GetOrder();
   }

   const int sdim = Tr.GetSpaceDim();
   const int dim = el[0]->GetDim();
   for (int j=0; j<numSpaces; j++)
   {
      for (int i=0; i<numSpaces; i++)
      {
         elmat(i,j)->SetSize(dof[i]*vdim[i], dof[j]*vdim[j]);
         *elmat(i,j) = 0.0;
      }
   }

   std::array<int, numSpaces> shapedim(InitInputShapes(el, Tr, allshapes));
   Array<int> x_idx(numSpaces+1), H_idx(numSpaces+1);
   x_idx[0] = 0;
   H_idx[0] = 0;
   for (int i=0; i<numSpaces; i++)
   {
      x_idx[i+1] = shapedim[i]*vdim[i];
      H_idx[i+1] = shapedim[i]*vdim[i]*vdim[i];
   }
   x_idx.PartialSum();
   x.SetSize(f.n_input);
   H.SetSize(f.n_input);

   _constexpr_for([&](auto vi)
   {
      xvar[vi].MakeRef(x, x_idx[vi], shapedim[vi]*vdim[vi]);
      if constexpr (hasFlag(modes_arr[vi], ADEval::VECTOR))
      {
         elfun_matview[vi].UseExternalData(const_cast<real_t*>(elfun[vi]->GetData()),
                                           dof[vi], vdim[vi]);
         xmat[vi].UseExternalData(xvar[vi].GetData(), shapedim[vi], vdim[vi]);
      }
   }, std::make_index_sequence<sizeof...(modes)> {});

   const IntegrationRule * ir = GetIntegrationRule(el, Tr);
   real_t w;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Tr.SetIntPoint(&ip);
      w = Tr.Weight()*ip.weight;

      CalcInputShapes(el, Tr, ip, allshapes);

      _constexpr_for([&](auto vi)
      {
         if constexpr (hasFlag(modes_arr[vi], ADEval::VECTOR))
         {
            MultAtB(allshapes[vi], elfun_matview[vi], xmat[vi]);
         }
         else
         {
            allshapes[vi].MultTranspose(*elfun[vi], xvar[vi]);
         }
      }, std::make_index_sequence<sizeof...(modes)> {});
      f.Hessian(x, Tr, ip, H);
      H *= w;
      _constexpr_for([&](auto trial_i)
      {
         _constexpr_for([&](auto test_i)
         {
            H.GetSubMatrix(x_idx[test_i], x_idx[test_i+1], x_idx[trial_i], x_idx[trial_i+1],
                           Hsub);
            Hsub.SetSize(shapedim[test_i], vdim[test_i]*vdim[trial_i]*shapedim[trial_i]);
            Hx.SetSize(dof[test_i], vdim[test_i]*vdim[trial_i]*shapedim[trial_i]);
            Mult(allshapes[test_i], Hsub, Hx);
            if constexpr (hasFlag(modes_arr[test_i], ADEval::VECTOR))
            {
               MFEM_ABORT("NOT YET IMPLEMENTED");
            }
            else
            {
               AddMultABt(Hx, allshapes[trial_i], *elmat(test_i, trial_i));
            }
         }, std::make_index_sequence<sizeof...(modes)> {});
      }, std::make_index_sequence<sizeof...(modes)> {});
   }
}

/// @brief Perform the local action of the NonlinearFormIntegrator resulting
/// from a face integral term.
template <ADEval... modes>
void ADBlockNonlinearFormIntegrator<modes...>::AssembleFaceVector(
   const Array<const FiniteElement *>&el1,
   const Array<const FiniteElement *>&el2,
   FaceElementTransformations &Tr,
   const Array<const Vector *>&elfun,
   const Array<Vector *>&elvect)
{
   MFEM_ABORT("ADBlockNonlinearFormIntegrator::AssembleFaceVector: "
              "This method is not implemented.");
}


/// @brief Assemble the local action of the gradient of the
/// NonlinearFormIntegrator resulting from a face integral term.
template <ADEval... modes>
void ADBlockNonlinearFormIntegrator<modes...>::AssembleFaceGrad(
   const Array<const FiniteElement *>&el1,
   const Array<const FiniteElement *>&el2,
   FaceElementTransformations &Tr,
   const Array<const Vector *>&elfun,
   const Array2D<DenseMatrix *>&elmat)
{
   MFEM_ABORT("ADBlockNonlinearFormIntegrator::AssembleFaceGrad: "
              "This method is not implemented.");
}
} // namespace mfem
