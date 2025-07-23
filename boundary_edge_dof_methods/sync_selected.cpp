#include "sync_selected.hpp"

#ifdef MFEM_USE_MPI

#include <vector>
#include "../fem/pgridfunc.hpp"

namespace mfem
{

void SynchronizeMarkedDoFs(ParFiniteElementSpace *pfes, Array<double> &values, 
                          const Array<int> &ldof_marker)
{
   MFEM_VERIFY(values.Size() == ldof_marker.Size(), 
               "values and ldof_marker arrays must be the same size");
   
   // First, synchronize the marker itself to ensure all processors agree on which DoFs to sync
   Array<int> global_marker(ldof_marker);
   pfes->Synchronize(global_marker);
   
   // Create a temporary array with only the values we want to synchronize
   Array<double> marked_values(values.Size());
   marked_values = 0.0;
   
   // Copy only the marked values
   for (int i = 0; i < values.Size(); i++)
   {
      if (global_marker[i])
      {
         marked_values[i] = values[i];
      }
   }
   
   // Synchronize the marked values
   pfes->SynchronizeBC(marked_values);
   
   // Merge the synchronized values back into the original array
   for (int i = 0; i < values.Size(); i++)
   {
      if (global_marker[i])
      {
         values[i] = marked_values[i];
      }
   }
}

void SynchronizeMarkedDoFs(ParFiniteElementSpace *pfes, ParGridFunction &gf, 
                          const Array<int> &ldof_marker)
{
   // Access the underlying Vector data in the ParGridFunction
   Array<double> values(gf.GetData(), gf.Size());
   
   // Call the Array<double> version
   SynchronizeMarkedDoFs(pfes, values, ldof_marker);
}

} // namespace mfem

#endif // MFEM_USE_MPI