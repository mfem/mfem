

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include "AcroTensor.hpp"

using namespace mfem;
using namespace acrobatic;

double GetAvgPostCachedAsmTime(TensorEngine &TE, FiniteElementSpace *fes, bool ongpu, int num_samples);
void TensorAssemble(TensorEngine &TE, Tensor &M, Tensor &D, Tensor &W, Tensor &T, Tensor &C, Tensor &E);
void GlobalAssemble(FiniteElementSpace *fes, Tensor &M);

int main(int argc, char *argv[])
{
   const char *mesh_file = "../data/star.mesh";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(std::cout);
      return 1;
   }
   args.PrintOptions(std::cout);

   Mesh *mesh;
   std::ifstream imesh(mesh_file);
   mesh = new Mesh(imesh, 1, 1);
   imesh.close();
   int dim = mesh->Dimension();
   int ref_levels = (int)floor(log(50000./mesh->GetNE())/log(2.)/dim);
   for (int l = 0; l < ref_levels; l++)
   {
      mesh->UniformRefinement();
   }
   
   TensorEngine TE1("Interpreted");
   TensorEngine TE2("IndexCached");
   TensorEngine TE3("Thrust");
   FiniteElementCollection *fec;
   FiniteElementSpace *fes;
   std::cout << mesh_file << std::endl;
   std::cout << "Order\tTE1cpu\t\tTE1gpu\t\tTE2cpu\t\tTE2gpu\t\tTE3cpu\t\tTE3gpu" << std::endl;
   for (int order = 1; order <= 5; ++order)
   {
      fec = new H1_FECollection(order, dim);
      fes = new FiniteElementSpace(mesh, fec);

      std::cout << order << std::setprecision(2) << std::scientific;
      std::cout << "\t" << GetAvgPostCachedAsmTime(TE1, fes, false, 10);
      std::cout << "\t" << GetAvgPostCachedAsmTime(TE1, fes, true, 10);
      std::cout << "\t" << GetAvgPostCachedAsmTime(TE2, fes, false, 10);
      std::cout << "\t" << GetAvgPostCachedAsmTime(TE2, fes, true, 10);
      //std::cout << "\t" << GetAvgPostCachedAsmTime(TE3, fes, false, 10);
      //std::cout << "\t" << GetAvgPostCachedAsmTime(TE3, fes, true, 10);
      std::cout << std::endl;

      delete fes;
      delete fec;
   }


   return 0;
}


double GetAvgPostCachedAsmTime(TensorEngine &TE, FiniteElementSpace *fes, bool ongpu, int num_samples)
{
   tic();
   const FiniteElement *el = fes->GetFE(0);
   IsoparametricTransformation trans;
   fes->GetElementTransformation(0, &trans);
   int num_elements = fes->GetNE();
   int dim = el->GetDim();
   int p = el->GetOrder();

   Vector shape;

   //Set up our 1D/ND intrules
   int irorder = 2 * p + trans.OrderW();
   const IntegrationRule *ir1d = &IntRules.Get(Geometry::SEGMENT, irorder);
   const IntegrationRule *irfull;
   if (dim == 1)
   {
      irfull = ir1d;
   }
   else if (dim == 2)
   {
      irfull = &IntRules.Get(Geometry::SQUARE, irorder);
   }
   else if (dim == 3)
   {
      irfull = &IntRules.Get(Geometry::CUBE, irorder);
   }

   //Create the tensors we are going to work with 
   //M_e_i1_i2_i3_j1_j2_j3 (in 3d)
   //B1d_k_i
   //D_e_k1_k2_k3 (in 3d)
   std::vector<int> mdims(1+2*dim, p+1);
   mdims[0] = num_elements;
   std::vector<int> ddims(1+dim, ir1d->Size());
   ddims[0] = num_elements;
   std::vector<int> wdims(dim, ir1d->Size());
   Tensor M(mdims); 
   Tensor B(ir1d->Size(), p+1);                   //This is B1d from the notes
   Tensor C(ir1d->Size(), p+1, p+1);
   Tensor D(ddims);
   Tensor W(wdims);
   Tensor T(num_elements, irfull->Size());
   Tensor E(num_elements, p+1, p+1, p+1); //Temp space for 2D

   //Fill B, and compute C
   shape.SetSize(B.GetDim(1));
   for (int k = 0; k < B.GetDim(0); ++k)
   {
      const IntegrationPoint &ip = ir1d->IntPoint(k);
      el->CalcShape1D(ip, shape);
      for (int i = 0; i < B.GetDim(1); ++i)
      {
         B(k,i) = shape[i];
      }
   }
   TE["C_k_i_j = B_k_i B_k_j"](C, B, B);


   //Fill W
   for (int k = 0; k < irfull->Size(); ++k)
   {
      const IntegrationPoint &ip = irfull->IntPoint(k);
      trans.SetIntPoint(&ip);
      double w = ip.weight;   
      W[k] = w;  //linear access into the tensor
   }

   //Fill T (This could be improved by filling a J_e_k1_k2_m_n and computing the det in the tensor lib )
   T.Reshape(num_elements, irfull->Size());
   for (int e = 0; e < num_elements; ++e)
   {
      fes->GetElementTransformation(e, &trans);
      for (int k = 0; k < irfull->Size(); ++k)
      {  
         const IntegrationPoint &ip = irfull->IntPoint(k);
         trans.SetIntPoint(&ip);
         T(e, k) = trans.Weight();
      }
   }
   std::vector<int> tdims(dim+1, ir1d->Size());
   tdims[0] = num_elements;
   T.Reshape(tdims);

   //Now use the tensor library to assemble the element matrices all at once
   if (ongpu)
   {
      M.MapToGPU();
      D.MapToGPU();
      W.MapToGPU();
      T.MapToGPU();
      C.MapToGPU();
      E.MapToGPU();
      M.MoveToGPU();
      D.MoveToGPU();
      W.MoveToGPU();
      T.MoveToGPU();
      C.MoveToGPU();
      E.MoveToGPU();      
   }
   TensorAssemble(TE, M, D, W, T, C, E);
   if (ongpu)
   {
      M.MoveFromGPU();     //Assuming we need to transfer back
      M.SwitchToGPU();
   }

   //GlobalAssemble(fes, M);
   double asmtime1 = toc();

   //Now pretend the mesh changed and we are going to assemble again
   double avgtime = 0.0;
   for (int i = 0; i < num_samples; ++i)
   {
      tic();

      //Fill W
      if (ongpu)
      {
         W.SwitchFromGPU();      
      }
      for (int k = 0; k < irfull->Size(); ++k)
      {
         const IntegrationPoint &ip = irfull->IntPoint(k);
         trans.SetIntPoint(&ip);
         double w = ip.weight;   
         W[k] = w;  //linear access into the tensor
      }

      //Fill T (This could be improved by filling a J_e_k1_k2_m_n and computing the det in the tensor lib )
      if (ongpu)
      {
         T.SwitchFromGPU();
      }
      T.Reshape(num_elements, irfull->Size());
      for (int e = 0; e < num_elements; ++e)
      {
         fes->GetElementTransformation(e, &trans);
         for (int k = 0; k < irfull->Size(); ++k)
         {  
            const IntegrationPoint &ip = irfull->IntPoint(k);
            trans.SetIntPoint(&ip);
            T(e, k) = trans.Weight();
         }
      }
      T.Reshape(tdims);

      if (ongpu)
      {
         W.MoveToGPU();
         T.MoveToGPU();
      }
      TensorAssemble(TE, M, D, W, T, C, E);
      if (ongpu)
      {
         M.MoveFromGPU();     //Assuming we need to transfer back
         M.SwitchToGPU();
      }

      //GlobalAssemble(fes, M);
      avgtime += toc();
   }

   TE.Clear();
   return avgtime / double(num_samples);
}

void TensorAssemble(TensorEngine &TE, Tensor &M, Tensor &D, Tensor &W, Tensor &T, Tensor &C, Tensor &E)
{
   int dim = W.GetRank();
   int num_elements = M.GetDim(0);
   int p = M.GetDim(1) - 1;
   if (W.GetRank() == 1)
   {
      TE["D_e_k = W_k T_e_k"](D, W, T);
      TE["M_e_i_j = C_k_i_j D_e_k"](M, C, D);
   }
   else if (W.GetRank() == 2)
   {
      //M_e_i1_i2_j1_j2 = C_k1_i1_j1 C_k2_i2_j2 D_e_k1_k2
      TE["D_e_k1_k2 = W_k1_k2 T_e_k1_k2"](D, W, T);
      TE["E_e_i2_j2_k1 = C_k2_i2_j2 D_e_k1_k2"](E, C, D);
      TE["M_e_i1_i2_j1_j2 = C_k1_i1_j1 E_e_i2_j2_k1"](M, C, E);
   }
   else if (W.GetRank() == 3)
   {
      TE["D_e_k1_k2_k3 = W_k1_k2_k3 T_e_k1_k2_k3"](D, W, T);
      TE["M_e_i1_i2_i3_j1_j2_j3 = C_k1_i1_j1 C_k2_i2_j2 C_k3_i3_j3 D_e_k1_k2_k3"](M, C, C, C, D);
   }


}

void GlobalAssemble(FiniteElementSpace *fes, Tensor &M)
{
   int num_elements = fes->GetNE();
   int num_dofs_per_el = fes->GetFE(0)->GetDof() * fes->GetVDim();
   DenseTensor *element_matrices = new DenseTensor();
   DenseMatrix *elmat_p;
   element_matrices->UseExternalData(M.GetData(), num_dofs_per_el, num_dofs_per_el, num_elements);
   SparseMatrix *mat = new SparseMatrix(fes->GetVSize());
   Array<int>  vdofs;
   int skip_zeros = 1;
   //Do a global assembly operation
   for (int i = 0; i < num_elements; i++)
   {
      fes->GetElementVDofs(i, vdofs);
      elmat_p = &(*element_matrices)(i);
      mat->AddSubMatrix(vdofs, vdofs, *elmat_p, skip_zeros);
   }

   delete mat;
   delete element_matrices;
}