#include "mfem.hpp"

using namespace mfem;

double computeError(GridFunction& u, GridFunction& uRef, int p);

int main(int argc, char** argv)
{
   Mpi::Init(argc, argv);

   const char *fieldName = "";
   const char *comparisonName = "";
   const char *referenceName = "";
   int index = -1;
   int p = 2;


   OptionsParser args(argc, argv);
   args.AddOption(&fieldName, "-f", "--field",
                  "Name of field for error computation.");
   args.AddOption(&comparisonName, "-c" , "--comparison",
                  "Name of 'comparison' DataCollection (can include path).");
   args.AddOption(&referenceName, "-r", "--reference",
                  "Name of 'reference' DataCollection (can include path).");
   args.AddOption(&index, "-i", "--index",
                  "Index in DataCollections for error computation.");
   args.AddOption(&p, "-p", "--norm",
                  "The order of the Lp norm to apply.");
   args.Parse();
   int errorCode = 0;
   if (!args.Good())
   {
      errorCode = 1;
   }
   else if (strcmp(comparisonName, "") == 0)
   {
      if (Mpi::Root())
         mfem::out << "Need to specify a comparison DataCollection." << std::endl;
      errorCode = 2;
   }
   else if (strcmp(referenceName, "") == 0)
   {
      if (Mpi::Root())
         mfem::out << "Need to specify a reference DataCollection." << std::endl;
      errorCode = 3;
   }
   else if (strcmp(fieldName, "") == 0)
   {
      if (Mpi::Root())
         mfem::out << "Need to specify a field to compute error of." << std::endl;
      errorCode = 4;
   }
   else if (index < 0)
   {
      if (Mpi::Root())
         mfem::out << "Need to specify an index in the DataCollections." << std::endl;
      errorCode = 5;
   }

   if (errorCode > 0)
   {
      if (Mpi::Root())
         args.PrintUsage(mfem::out);
      Mpi::Finalize();
      return errorCode;
   }
   if (Mpi::Root())
      args.PrintOptions(mfem::out);

   VisItDataCollection data(MPI_COMM_WORLD, comparisonName);
   VisItDataCollection dataRef(MPI_COMM_WORLD, referenceName);

   data.Load(index);
   dataRef.Load(index);

   if (!data.HasField(fieldName) || !dataRef.HasField(fieldName))
   {
      if (Mpi::Root())
         mfem::out << "Field " << fieldName << " not found in DataCollections" << std::endl;
      Mpi::Finalize();
      return 5;
   }

   GridFunction& u = *data.GetField(fieldName);
   GridFunction& uRef = *dataRef.GetField(fieldName);

   double error = computeError(u, uRef, p);

   if (Mpi::Root())
      std::cout << "Error: " << error << std::endl;

   return 0;
}


double computeError(GridFunction& u, GridFunction& uRef, int p)
{
   const MPI_Comm& communicator = MPI_COMM_WORLD;
   const FiniteElementSpace& fespace = *(u.FESpace());
   const int dim = fespace.GetMesh()->Dimension();
   const int X_OFFSET = 0;
   const int WEIGHT_OFFSET = X_OFFSET + dim;
   const int VALUE_OFFSET = WEIGHT_OFFSET + 1;
   const int TOTAL_OFFSET = VALUE_OFFSET + 1;
   const int RESIZE_INCREMENT = 1000 * TOTAL_OFFSET;

   // loop through all elements and integration points, storing the physical
   // locations, weights, and u values
   std::vector<double> localInfo(RESIZE_INCREMENT);
   int index = 0;
   for (int id=0; id < fespace.GetMesh()->GetNE(); id++)
   {
      const FiniteElement& element = *fespace.GetFE(id);
      ElementTransformation& T = *fespace.GetElementTransformation(id);
      const int quadratureOrder = 2*element.GetOrder() + 1;
      const IntegrationRule quadratureRule =
         IntRules.Get(element.GetGeomType(), quadratureOrder);
      int numQuadraturePoints = quadratureRule.GetNPoints();
      for (int i=0; i < numQuadraturePoints; i++)
      {
         // get u value at and physical location of current integration point
         const IntegrationPoint ip = quadratureRule.IntPoint(i);
         Vector x;
         T.SetIntPoint(&ip);
         T.Transform(ip, x);
         for (int j=0; j < x.Size(); j++)
            localInfo[index + X_OFFSET + j] = x(j);
         localInfo[index + WEIGHT_OFFSET] = ip.weight * T.Weight();
         localInfo[index + VALUE_OFFSET] = u.GetValue(id, ip);
         index += TOTAL_OFFSET;
         if (index >= localInfo.size())
            localInfo.resize(localInfo.size() + RESIZE_INCREMENT);
      }
   }

   // gather size of local info vectors from all ranks
   localInfo.resize(index);
   std::vector<int> sizes(Mpi::WorldSize());
   MPI_Allgather(&index, 1, MPI_INT, sizes.data(), 1, MPI_INT, communicator);

   // gather info vectors from all ranks
   std::vector<int> displacements(Mpi::WorldSize()+1);
   displacements[0] = 0;
   for (int i=0; i < Mpi::WorldSize(); i++)
      displacements[i+1] = displacements[i] + sizes[i];
   std::vector<double> globalInfo(displacements.back());
   MPI_Allgatherv(localInfo.data(), localInfo.size(), MPI_DOUBLE,
      globalInfo.data(), sizes.data(), displacements.data(), MPI_DOUBLE,
      communicator);

   // unpack locations and determine which ones are in the local reference mesh
   int numPoints = globalInfo.size() / TOTAL_OFFSET;
   MFEM_ASSERT(globalInfo.size() % TOTAL_OFFSET == 0, "Global info size not correct");
   DenseMatrix points(dim, numPoints);
   for (int j=0; j < numPoints; j++)
   {
      for (int i=0; i < dim; i++)
        points(i, j) = globalInfo[j*TOTAL_OFFSET + X_OFFSET + i];
   }
   Array<int> ids;
   Array<IntegrationPoint> ips;
   int found = uRef.FESpace()->GetMesh()->FindPoints(points, ids, ips, false);

   // check to make sure all points were found
   const int root = 0;
   int globalFound;
   MPI_Reduce(&found, &globalFound, 1, MPI_INT, MPI_SUM, root, communicator);
   if (Mpi::WorldRank() == root)
      MFEM_VERIFY(globalFound == numPoints,
         "Some points in comparison mesh not found in reference mesh.");

   // sum the contributions to the errors for the locations in the local
   // reference mesh
   double localSum = 0.0;
   for (int j=0; j < numPoints; j++)
   {
      if (ids[j] >= 0)
      {
         const double value = globalInfo[j*TOTAL_OFFSET + VALUE_OFFSET];
         const double valueRef = uRef.GetValue(ids[j], ips[j]);
         localSum += globalInfo[j*TOTAL_OFFSET + WEIGHT_OFFSET] *
            std::pow(std::abs(value - valueRef), p);
      }
   }

   // obtain global sum and apply scaling to get the global error
   double globalSum;
   MPI_Allreduce(&localSum, &globalSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   return std::pow(globalSum, 1.0/p);
}
