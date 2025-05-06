#include "partition.hpp"

#ifdef MFEM_USE_METIS
#include "metis.h"
#else
#error "METIS is required
#endif

namespace mfem
{

Array<idx_t> PartitionMesh(Mesh &mesh, const int npart,
                           OptRef<Array<int>> subset)
{
   const int part_method = 1;

   const int ne = subset ? subset->Size() : mesh.GetNE();
   idx_t mpart = npart;
   Array<idx_t> p(ne);

   // Early return special cases for one partition requested, or more partitions
   // then elements.
   if (npart == 1)
   {
      for (int i = 0; i < ne; i++)
      {
         p[i] = 0;
      }
      return p;
   }
   else if (ne <= npart)
   {
      for (int i = 0; i < ne; i++)
      {
         p[i] = i;
      }
      return p;
   }

   const Table &e2e = mesh.ElementToElementTable();
   Array<idx_t> I, J;

   {
      const int *iI = e2e.HostReadI();
      const int *iJ = e2e.HostReadJ();
      const int m = iI[ne];
      I.SetSize(ne+1);
      J.SetSize(m);
      for (int k = 0; k < ne + 1; k++) { I[k] = iI[k]; }
      for (int k = 0; k < m; k++) { J[k] = iJ[k]; }
   }

   idx_t options[40];
   METIS_SetDefaultOptions(options);
   options[METIS_OPTION_CONTIG] = 1; // set METIS_OPTION_CONTIG

   // If the mesh is disconnected, disable METIS_OPTION_CONTIG.
   // {
   //    Array<int> part(partitioning, ne);
   //    part = 0; // single part for the whole mesh
   //    Array<int> component; // size will be set to num. elem.
   //    Array<int> num_comp;  // size will be set to num. parts (1)
   //    mesh.FindPartitioningComponents(*el_to_el, part, component, num_comp);
   //    if (num_comp[0] > 1) { options[METIS_OPTION_CONTIG] = 0; }
   // }

   // Sort the neighbor lists
   if (part_method >= 0 && part_method <= 2)
   {
      for (int i = 0; i < ne; i++)
      {
         // Sort in increasing order.
         // std::sort(J+I[i], J+I[i+1]);

         // Sort in decreasing order, as in previous versions of MFEM.
         std::sort(J+I[i], J+I[i+1], std::greater<idx_t>());
      }
   }

   // This function should be used to partition a graph into a small
   // number of partitions (less than 8).
   if (part_method == 0 || part_method == 3)
   {
      idx_t n = ne;
      idx_t ncon = 1;
      idx_t edgecut;
      const idx_t err = METIS_PartGraphRecursive(
                           &n, &ncon, I, J, NULL, NULL, NULL, &mpart, NULL,
                           NULL, options, &edgecut, p.HostWrite());
      MFEM_VERIFY(err == 1, "Error in METIS_PartGraphRecursive");
   }

   // This function should be used to partition a graph into a large
   // number of partitions (greater than 8).
   if (part_method == 1 || part_method == 4)
   {
      idx_t n = ne;
      idx_t ncon = 1;
      idx_t edgecut;
      const idx_t err = METIS_PartGraphKway(
                           &n, &ncon, I, J, NULL, NULL, NULL, &mpart, NULL,
                           NULL, options, &edgecut, p.HostWrite());
      MFEM_VERIFY(err == 1, "Error in METIS_PartGraphKway");
   }

   // Check for empty partitionings (a "feature" in METIS)
   // if (npart > 1 && ne > npart)
   // {
   //    Array< Pair<int,int> > psize(npart);
   //    int empty_parts;

   //    // Count how many elements are in each partition, and store the result in
   //    // psize, where psize[i].one is the number of elements, and psize[i].two
   //    // is partition index. Keep track of the number of empty parts.
   //    auto count_partition_elements = [&]()
   //    {
   //       for (int i = 0; i < npart; i++)
   //       {
   //          psize[i].one = 0;
   //          psize[i].two = i;
   //       }

   //       for (int i = 0; i < ne; i++)
   //       {
   //          psize[partitioning[i]].one++;
   //       }

   //       empty_parts = 0;
   //       for (int i = 0; i < npart; i++)
   //       {
   //          if (psize[i].one == 0) { empty_parts++; }
   //       }
   //    };

   //    count_partition_elements();

   //    // This code just split the largest partitionings in two.
   //    // Do we need to replace it with something better?
   //    while (empty_parts)
   //    {
   //       if (print_messages)
   //       {
   //          mfem::err << "Mesh::GeneratePartitioning(...): METIS returned "
   //                    << empty_parts << " empty parts!"
   //                    << " Applying a simple fix ..." << endl;
   //       }

   //       SortPairs<int,int>(psize, npart);

   //       for (int i = npart-1; i > npart-1-empty_parts; i--)
   //       {
   //          psize[i].one /= 2;
   //       }

   //       for (int j = 0; j < ne; j++)
   //       {
   //          for (int i = npart-1; i > npart-1-empty_parts; i--)
   //          {
   //             if (psize[i].one == 0 || partitioning[j] != psize[i].two)
   //             {
   //                continue;
   //             }
   //             else
   //             {
   //                partitioning[j] = psize[npart-1-i].two;
   //                psize[i].one--;
   //             }
   //          }
   //       }

   //       // Check for empty partitionings again
   //       count_partition_elements();
   //    }
   // }

   return p;
}

}
