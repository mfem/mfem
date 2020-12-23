
#include "DofMaps.hpp"
#include "MeshPart.hpp"

void FindPtsGetCommonElements(Mesh & mesh0, Mesh & mesh1, 
                              Array<int> & elems0, Array<int> & elems1)
{
   int dim = mesh0.Dimension();
   const int ne0 = mesh0.GetNE();
   Vector centers(ne0*dim);
   elems0.SetSize(0);
   elems1.SetSize(0);
   for (int i = 0; i < ne0; i++)
   {
      Vector center(dim);
      mesh0.GetElementCenter(i,center);
      for (int d=0; d<dim; d++)
      {
         centers[ne0*d + i] = center[d];
      }
   }
   // Evaluate mesh 1 grid function.
   FindPointsGSLIB finder;
   finder.Setup(mesh1);
   finder.FindPoints(centers);
   Array<int> elem_map = finder.GetElem();
   Array<int> code = finder.GetCode();
   finder.FreeData();

   for (int i = 0; i<code.Size(); i++)
   {
      if (!code[i]) 
      {  // element is found
         elems0.Append(i);
         elems1.Append(elem_map[i]);
      }
   }
}                              

// Assuming there are no dublicated indices in the lists
void GetCommonIndices(const Array<int> & list0, const Array<int> & list1, Array<int> & idx0, Array<int> & idx1)
{
   unordered_map<int, int> map0, map1;
   int i = 0, j = 0;
   for (auto k : list0) map0[k] = i++;
   for (auto k : list1) map1[k] = j++;
   
   for (auto k : map0)
   {
      if (map1.find(k.first) != map1.end())
      {
         idx0.Append(k.second);
         idx1.Append(map1[k.first]);
      }
   }
}

// Array2D<int> DofMap (row0: the fes0 index (Domain))
//                     (row1: the fes1 index (Range))
Array2D<int> * GetDofMap(const FiniteElementSpace &fes0, const FiniteElementSpace &fes1, 
               const Array<int> * elems0_, const Array<int> * elems1_)
{
   Array<int> elems0, elems1;
   if (!elems0_ || !elems1_)
   {  // construct the element lists using gslib
      FindPtsGetCommonElements(*fes0.GetMesh(), *fes1.GetMesh(), elems0, elems1);
   }
   else
   {
      GetCommonIndices(*elems0_, *elems1_, elems0, elems1);   
   }


   cout << "Elems0: " ; elems0.Print(cout,10);
   cout << "Elems1: " ; elems1.Print(cout,10);

   return nullptr;
}


void PartitionFE(const FiniteElementSpace * fes, int nrsubmeshes, double ovlp, 
                 Array<FiniteElementSpace*> & fespaces, 
                 Array<Array<int> * > ElemMaps,
                 Array<Array<int> * > DofMaps)
{
   Mesh * mesh = fes->GetMesh();
   Array<Mesh *> meshes;
   Array<Array<int> * > elems;


   PartitionMesh(mesh,nrsubmeshes,ovlp,meshes,elems);

   
}