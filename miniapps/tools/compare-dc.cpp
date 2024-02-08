#include "mfem.hpp"

using namespace mfem;

class LpErrorCalculator
{
   const int DIM;
   const int ROOT_RANK;
   const int X_OFFSET;
   const int WEIGHT_OFFSET;
   const int VALUE_OFFSET;
   const int TOTAL_OFFSET;

   std::vector<double> info;

public:

   LpErrorCalculator(const int dim) : DIM(dim), ROOT_RANK(0), X_OFFSET(0),
      WEIGHT_OFFSET(X_OFFSET + DIM),
      VALUE_OFFSET(WEIGHT_OFFSET + 1),
      TOTAL_OFFSET(VALUE_OFFSET + 1) {}

   void setComparisonInfo(GridFunction& u, MPI_Comm communicator)
   {
      const FiniteElementSpace& fespace = *(u.FESpace());
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

      // gather size of local info vectors from all ranks to root
      localInfo.resize(index);
      int numRanks;
      MPI_Comm_size(communicator, &numRanks);
      std::vector<int> sizes(numRanks);
      MPI_Gather(&index, 1, MPI_INT, sizes.data(), 1, MPI_INT, ROOT_RANK,
         communicator);

      // gather info vectors from all ranks to root
      std::vector<int> displacements(numRanks+1);
      displacements[0] = 0;
      for (int i=0; i < numRanks; i++)
         displacements[i+1] = displacements[i] + sizes[i];
      info.resize(displacements.back());
      MPI_Gatherv(localInfo.data(), localInfo.size(), MPI_DOUBLE,
         info.data(), sizes.data(), displacements.data(), MPI_DOUBLE,
         ROOT_RANK, communicator);

   }

   void broadcastComparisonInfo()
   {
      int count = info.size();
      MPI_Bcast(&count, 1, MPI_INT, ROOT_RANK, MPI_COMM_WORLD);
      if (info.empty()) // i.e., not root
         info.resize(count);
      MPI_Bcast(info.data(), count, MPI_DOUBLE, ROOT_RANK, MPI_COMM_WORLD);
   }

   double computeError(GridFunction& uRef, int p)
   {
      MFEM_VERIFY(!info.empty(),
         "Comparison info is empty, was broadcastComparisonInfo called?");

      // unpack locations and determine which ones are in the local reference mesh
      const int numPoints = info.size() / TOTAL_OFFSET;
      MFEM_ASSERT(info.size() % TOTAL_OFFSET == 0,
         "Global info size not correct");
      DenseMatrix points(DIM, numPoints);
      for (int j=0; j < numPoints; j++)
      {
         for (int i=0; i < DIM; i++)
         points(i, j) = info[j*TOTAL_OFFSET + X_OFFSET + i];
      }
      Array<int> ids;
      Array<IntegrationPoint> ips;
      int found = uRef.FESpace()->GetMesh()->FindPoints(points, ids, ips, false);

      // check to make sure all points were found
      const int root = 0;
      int globalFound;
      MPI_Reduce(&found, &globalFound, 1, MPI_INT, MPI_SUM, root, MPI_COMM_WORLD);
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
            const double value = info[j*TOTAL_OFFSET + VALUE_OFFSET];
            const double valueRef = uRef.GetValue(ids[j], ips[j]);
            localSum += info[j*TOTAL_OFFSET + WEIGHT_OFFSET] *
               std::pow(std::abs(value - valueRef), p);
         }
      }

      // obtain global sum and apply scaling to get the global error
      double globalSum;
      MPI_Allreduce(&localSum, &globalSum, 1, MPI_DOUBLE, MPI_SUM,
         MPI_COMM_WORLD);
      return std::pow(globalSum, 1.0/p);
   }
};


int main(int argc, char** argv)
{
   Mpi::Init(argc, argv);

   const char *fieldName = "";
   const char *comparisonName = "";
   const char *referenceName = "";
   int index = -1;
   int p = 2;
   int numComparisonRanks = Mpi::WorldSize();


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
   args.AddOption(&numComparisonRanks, "-nc", "--num-comparison-ranks",
                  "The number of ranks used to produce 'comparison' DataCollection.");
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

   VisItDataCollection dataRef(MPI_COMM_WORLD, referenceName);
   dataRef.Load(index);
   MFEM_VERIFY(dataRef.HasField(fieldName),
      "Field " << fieldName << " not found in reference DataCollection");
   GridFunction& uRef = *dataRef.GetField(fieldName);

   LpErrorCalculator errorCalculator(uRef.FESpace()->GetMesh()->Dimension());

   MPI_Comm comparisonComm;
   const int color = Mpi::WorldRank() / numComparisonRanks;
   MPI_Comm_split(MPI_COMM_WORLD, color, Mpi::WorldRank(), &comparisonComm);
   if (color == 0)
   {
      VisItDataCollection data(comparisonComm, comparisonName);
      data.Load(index);
      MFEM_VERIFY(data.HasField(fieldName),
         "Field " << fieldName << " not found in comparision DataCollection");
      GridFunction& u = *data.GetField(fieldName);
      errorCalculator.setComparisonInfo(u, comparisonComm);
   }

   errorCalculator.broadcastComparisonInfo();

   double error = errorCalculator.computeError(uRef, p);

   if (Mpi::Root())
      mfem::out << "Error: " << error << std::endl;

   Mpi::Finalize();
   return 0;
}
