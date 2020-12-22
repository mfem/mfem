
#include "DofMaps.hpp"

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

   }
   else
   {
      elems0 = *elems0_;
      elems1 = *elems1_;
   }


   cout << "Elems0: " ; elems0.Print(cout,10);
   cout << "Elems1: " ; elems1.Print(cout,10);

   Array<int> idx0, idx1;
   GetCommonIndices(elems0, elems1, idx0, idx1);

   // cout << "idx0: " ; idx0.Print(cout, 10);
   // cout << "idx1: " ; idx1.Print(cout, 10);


   return nullptr;
}