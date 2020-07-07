//                       MFEM Example 1 - Parallel Version
//
// Compile with: make ex1p
//
// Sample runs:  mpirun -np 4 ex1p -m ../data/square-disc.mesh
//               mpirun -np 4 ex1p -m ../data/star.mesh
//               mpirun -np 4 ex1p -m ../data/star-mixed.mesh
//               mpirun -np 4 ex1p -m ../data/escher.mesh
//               mpirun -np 4 ex1p -m ../data/fichera.mesh
//               mpirun -np 4 ex1p -m ../data/fichera-mixed.mesh
//               mpirun -np 4 ex1p -m ../data/toroid-wedge.mesh
//               mpirun -np 4 ex1p -m ../data/periodic-annulus-sector.msh
//               mpirun -np 4 ex1p -m ../data/periodic-torus-sector.msh
//               mpirun -np 4 ex1p -m ../data/square-disc-p2.vtk -o 2
//               mpirun -np 4 ex1p -m ../data/square-disc-p3.mesh -o 3
//               mpirun -np 4 ex1p -m ../data/square-disc-nurbs.mesh -o -1
//               mpirun -np 4 ex1p -m ../data/star-mixed-p2.mesh -o 2
//               mpirun -np 4 ex1p -m ../data/disc-nurbs.mesh -o -1
//               mpirun -np 4 ex1p -m ../data/pipe-nurbs.mesh -o -1
//               mpirun -np 4 ex1p -m ../data/ball-nurbs.mesh -o 2
//               mpirun -np 4 ex1p -m ../data/fichera-mixed-p2.mesh -o 2
//               mpirun -np 4 ex1p -m ../data/star-surf.mesh
//               mpirun -np 4 ex1p -m ../data/square-disc-surf.mesh
//               mpirun -np 4 ex1p -m ../data/inline-segment.mesh
//               mpirun -np 4 ex1p -m ../data/amr-quad.mesh
//               mpirun -np 4 ex1p -m ../data/amr-hex.mesh
//               mpirun -np 4 ex1p -m ../data/mobius-strip.mesh
//               mpirun -np 4 ex1p -m ../data/mobius-strip.mesh -o -1 -sc
//
// Device sample runs:
//               mpirun -np 4 ex1p -pa -d cuda
//               mpirun -np 4 ex1p -pa -d occa-cuda
//               mpirun -np 4 ex1p -pa -d raja-omp
//               mpirun -np 4 ex1p -pa -d ceed-cpu
//               mpirun -np 4 ex1p -pa -d ceed-cuda
//               mpirun -np 4 ex1p -m ../data/beam-tet.mesh -pa -d ceed-cpu
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Laplace problem
//               -Delta u = 1 with homogeneous Dirichlet boundary conditions.
//               Specifically, we discretize using a FE space of the specified
//               order, or if order < 1 using an isoparametric/isogeometric
//               space (i.e. quadratic for quadratic curvilinear mesh, NURBS for
//               NURBS mesh, etc.)
//
//               The example highlights the use of mesh refinement, finite
//               element grid functions, as well as linear and bilinear forms
//               corresponding to the left-hand side and right-hand side of the
//               discrete linear system. We also cover the explicit elimination
//               of essential boundary conditions, static condensation, and the
//               optional connection to the GLVis tool for visualization.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

void myGetLocalA(const HypreParMatrix &in_A,
                 Array<HYPRE_Int> &I, Array<int64_t> &J, Array<double> &Data)
{

  mfem::SparseMatrix Diag, Offd;
  HYPRE_Int* cmap; //column map

  in_A.GetDiag(Diag); Diag.SortColumnIndices();
  in_A.GetOffd(Offd, cmap); Offd.SortColumnIndices();

  //Number of rows in this partition
  int row_len = std::abs(in_A.RowPart()[1] -
			 in_A.RowPart()[0]); //end of row partition

  //Note Amgx requires 64 bit integers for column array
  //So we promote in this routine
  int *DiagI = Diag.GetI();
  int *DiagJ = Diag.GetJ();
  double *DiagA = Diag.GetData();

  int *OffI = Offd.GetI();
  int *OffJ = Offd.GetJ();
  double *OffA = Offd.GetData();

  I.SetSize(row_len+1);

  //Enumerate the local rows [0, num rows in proc)
  I[0]=0;
  for (int i=0; i<row_len; i++)
    {
      I[i+1] = I[i] + (DiagI[i+1] - DiagI[i]) + (OffI[i+1] - OffI[i]);
    }

  const HYPRE_Int *colPart = in_A.ColPart();
  J.SetSize(I[row_len]); //J = -777;
  Data.SetSize(I[row_len]); //Data = -777;

  int cstart = colPart[0];

  int k    = 0;
  for (int i=0; i<row_len; i++)
    {

      int jo, icol;
      int ncols_o = OffI[i+1] - OffI[i];
      int ncols_d = DiagI[i+1] - DiagI[i];

      //OffDiagonal
      for (jo=0; jo<ncols_o; jo++)
	{
	  icol = cmap[*OffJ];
	  if (icol >= cstart) { break; }
	  J[k]   = icol; OffJ++;
	  Data[k++] = *OffA++;
	}

      //Diagonal matrix
      for (int j=0; j<ncols_d; j++)
	{
	  J[k]   = cstart + *DiagJ++;
	  Data[k++] = *DiagA++;
	}

      //OffDiagonal
      for (int j=jo; j<ncols_o; j++)
	{
	  J[k]   = cmap[*OffJ++];
	  Data[k++] = *OffA++;
	}
    }

}

void GatherArray(Array<double> &inArr, Array<double> &outArr,
                 int MPI_SZ, MPI_Comm &cpuWorld)
{
  //Calculate number of elements to be collected from each process
  mfem::Array<int> Apart(MPI_SZ);
  int locAsz = inArr.Size();
  MPI_Allgather(&locAsz, 1, MPI_INT,
		Apart.GetData(),1, MPI_INT,cpuWorld);

  MPI_Barrier(cpuWorld);

  //Determine stride for process
  mfem::Array<int> Adisp(MPI_SZ);
  Adisp[0] = 0;
  for(int i=1; i<MPI_SZ; ++i){
    Adisp[i] = Adisp[i-1] + Apart[i-1];
  }

  MPI_Gatherv(inArr.HostReadWrite(), inArr.Size(), MPI_DOUBLE,
	      outArr.HostWrite(), Apart.HostRead(), Adisp.HostRead(),
	      MPI_DOUBLE, 0, cpuWorld);
}

void GatherArray(Vector &inArr, Vector &outArr,
                 int MPI_SZ, MPI_Comm &cpuWorld,Array<int> &Apart, Array<int> &Adisp)
{
  //Calculate number of elements to be collected from each process
  //mfem::Array<int> Apart(MPI_SZ);
  int locAsz = inArr.Size();
  MPI_Allgather(&locAsz, 1, MPI_INT,
		Apart.GetData(),1, MPI_INT,cpuWorld);

  MPI_Barrier(cpuWorld);

  //Determine stride for process
  //mfem::Array<int> Adisp(MPI_SZ);
  Adisp[0] = 0;
  for(int i=1; i<MPI_SZ; ++i){
    Adisp[i] = Adisp[i-1] + Apart[i-1];
  }

  MPI_Gatherv(inArr.HostReadWrite(), inArr.Size(), MPI_DOUBLE,
	      outArr.HostWrite(), Apart.HostRead(), Adisp.HostRead(),
	      MPI_DOUBLE, 0, cpuWorld);
}

void ScatterArray(Vector &inArr, Vector &outArr,
                 int MPI_SZ, MPI_Comm &cpuWorld, Array<int> &Apart, Array<int> &Adisp)
{

  MPI_Scatterv(inArr.HostReadWrite(),Apart.HostRead(),Adisp.HostRead(),
              MPI_DOUBLE,outArr.HostWrite(),inArr.Size(),
	            MPI_DOUBLE, 0, cpuWorld);
}

void GatherArray(Array<int> &inArr, Array<int> &outArr,
                 int MPI_SZ, MPI_Comm &cpuWorld)
{
  //Calculate number of elements to be collected from each process
  mfem::Array<int> Apart(MPI_SZ);
  int locAsz = inArr.Size();
  MPI_Allgather(&locAsz, 1, MPI_INT,
		Apart.GetData(),1, MPI_INT,cpuWorld);

  //Apart.Print();
  MPI_Barrier(cpuWorld);

  //Determine stride for process
  mfem::Array<int> Adisp(MPI_SZ);
  Adisp[0] = 0;
  for(int i=1; i<MPI_SZ; ++i){
    Adisp[i] = Adisp[i-1] + Apart[i-1];
  }
  //Adisp.Print();
  MPI_Gatherv(inArr.HostReadWrite(), inArr.Size(), MPI_INT,
	      outArr.HostWrite(), Apart.HostRead(), Adisp.HostRead(),
	      MPI_INT, 0, cpuWorld);
}

void GatherArray(Array<int64_t> &inArr, Array<int64_t> &outArr,
                 int MPI_SZ, MPI_Comm &cpuWorld)
{
  //Calculate number of elements to be collected from each process
  mfem::Array<int> Apart(MPI_SZ);
  int locAsz = inArr.Size();
  MPI_Allgather(&locAsz, 1, MPI_INT,
		Apart.GetData(),1, MPI_INT,cpuWorld);

  MPI_Barrier(cpuWorld);

  //Determine stride for process
  mfem::Array<int> Adisp(MPI_SZ);
  Adisp[0] = 0;
  for(int i=1; i<MPI_SZ; ++i){
    Adisp[i] = Adisp[i-1] + Apart[i-1];
  }

  MPI_Gatherv(inArr.HostReadWrite(), inArr.Size(), MPI_INT64_T,
	      outArr.HostWrite(), Apart.HostRead(), Adisp.HostRead(),
	      MPI_INT64_T, 0, cpuWorld);

  MPI_Barrier(cpuWorld);
}

int main(int argc, char *argv[])
{
  // 1. Initialize MPI.
  int num_procs, myid;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  // 2. Parse command-line options.
  //const char *mesh_file = "../data/star.mesh";
  const char *mesh_file = "../data/inline-quad.mesh";

  int order = 1;
  bool static_cond = false;
  bool pa = false;
  bool amgx = true;
  const char *device_config = "cpu";
  bool visualization = true;
  const char *amgx_cfg = 0;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh",
		 "Mesh file to use.");
  args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
		 " isoparametric space.");
  args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
		 "--no-static-condensation", "Enable static condensation.");
  args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
		 "--no-partial-assembly", "Enable Partial Assembly.");
  args.AddOption(&device_config, "-d", "--device",
		 "Device configuration string, see Device::Configure().");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
		 "--no-visualization",
		 "Enable or disable GLVis visualization.");
  args.AddOption(&amgx_cfg, "-c","--c","AMGX solver file");
  args.AddOption(&amgx, "-amgx","--amgx","-no-amgx",
		 "--no-amgx","Use AMGX");
  args.Parse();
  if (!args.Good())
    {
      if (myid == 0)
	{
	  args.PrintUsage(cout);
	}
      MPI_Finalize();
      return 1;
    }
  if (myid == 0)
    {
      args.PrintOptions(cout);
    }

  // 3. Enable hardware devices such as GPUs, and programming models such as
  //    CUDA, OCCA, RAJA and OpenMP based on command line options.
  Device device(device_config);
  if (myid == 0) { device.Print(); }

  // 4. Read the (serial) mesh from the given mesh file on all processors.  We
  //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
  //    and volume meshes with the same code.
  Mesh *mesh = new Mesh(mesh_file, 1, 1);
  int dim = mesh->Dimension();

  // 5. Refine the serial mesh on all processors to increase the resolution. In
  //    this example we do 'ref_levels' of uniform refinement. We choose
  //    'ref_levels' to be the largest number that gives a final mesh with no
  //    more than 10,000 elements.
  {
    int ref_levels = 0;
    //(int)floor(log(10000./mesh->GetNE())/log(2.)/dim);
    for (int l = 0; l < ref_levels; l++)
      {
	mesh->UniformRefinement();
      }
  }

  // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
  //    this mesh further in parallel to increase the resolution. Once the
  //    parallel mesh is defined, the serial mesh can be deleted.
  ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
  delete mesh;
  {
    int par_ref_levels = 0;
    for (int l = 0; l < par_ref_levels; l++)
      {
	pmesh->UniformRefinement();
      }
  }

  // 7. Define a parallel finite element space on the parallel mesh. Here we
  //    use continuous Lagrange finite elements of the specified order. If
  //    order < 1, we instead use an isoparametric/isogeometric space.
  FiniteElementCollection *fec;
  if (order > 0)
    {
      fec = new H1_FECollection(order, dim);
    }
  else if (pmesh->GetNodes())
    {
      fec = pmesh->GetNodes()->OwnFEC();
      if (myid == 0)
	{
	  cout << "Using isoparametric FEs: " << fec->Name() << endl;
	}
    }
  else
    {
      fec = new H1_FECollection(order = 1, dim);
    }
  ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
  HYPRE_Int size = fespace->GlobalTrueVSize();
  if (myid == 0)
    {
      cout << "Number of finite element unknowns: " << size << endl;
    }

  // 8. Determine the list of true (i.e. parallel conforming) essential
  //    boundary dofs. In this example, the boundary conditions are defined
  //    by marking all the boundary attributes from the mesh as essential
  //    (Dirichlet) and converting them to a list of true dofs.
  Array<int> ess_tdof_list;
  if (pmesh->bdr_attributes.Size())
    {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    }

  // 9. Set up the parallel linear form b(.) which corresponds to the
  //    right-hand side of the FEM linear system, which in this case is
  //    (1,phi_i) where phi_i are the basis functions in fespace.
  ParLinearForm *b = new ParLinearForm(fespace);
  ConstantCoefficient one(1.0);
  b->AddDomainIntegrator(new DomainLFIntegrator(one));
  b->Assemble();

  // 10. Define the solution vector x as a parallel finite element grid function
  //     corresponding to fespace. Initialize x with initial guess of zero,
  //     which satisfies the boundary conditions.
  ParGridFunction x(fespace);
  x = 0.0;

  // 11. Set up the parallel bilinear form a(.,.) on the finite element space
  //     corresponding to the Laplacian operator -Delta, by adding the Diffusion
  //     domain integrator.
  ParBilinearForm *a = new ParBilinearForm(fespace);
  if (pa) { a->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
  a->AddDomainIntegrator(new DiffusionIntegrator(one));

  // 12. Assemble the parallel bilinear form and the corresponding linear
  //     system, applying any necessary transformations such as: parallel
  //     assembly, eliminating boundary conditions, applying conforming
  //     constraints for non-conforming AMR, static condensation, etc.
  if (static_cond) { a->EnableStaticCondensation(); }
  a->Assemble();

  HypreParMatrix A;
  Vector B, X;
  a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

  //AMGX
  std::string amgx_str;
  amgx_str = amgx_cfg;
  //NvidiaAMGX amgx;
  //amgx.Init(MPI_COMM_WORLD, "dDDI", amgx_str);
  //amgx.SetA(A);
  //X = 0.0; //set to zero
  //amgx.Solve(X, B);
  //Mini AMGX Parallel Example
  {
    int MPI_SZ, MPI_RANK;

    AMGX_matrix_handle      A_amgx;
    AMGX_vector_handle x_amgx, b_amgx;
    AMGX_solver_handle solver_amgx;

    AMGX_Mode amgx_mode = AMGX_mode_dDDI;

    int ring;
    AMGX_config_handle  cfg;
    static AMGX_resources_handle   rsrc;

    //Local processor
    Array<int> loc_I;
    Array<int64_t> loc_J;
    Array<double> loc_A;

    //MPI procs that will talk to gpus
    MPI_Comm cpuWorld;
    MPI_Comm amgx_comm=MPI_COMM_NULL;

    MPI_Comm_dup(MPI_COMM_WORLD, &cpuWorld);
    MPI_Comm_size(cpuWorld, &MPI_SZ);
    MPI_Comm_rank(cpuWorld, &MPI_RANK);

    //Has a gpu
    int gpuProc = MPI_UNDEFINED;
    int nDevs, deviceId;
    cudaGetDeviceCount(&nDevs);
    cudaGetDevice(&deviceId);
    //printf("nDevs = %d MPI_SZ = %d \n",nDevs,MPI_SZ);

    if (nDevs == MPI_SZ) // # of the devices and local precosses are the same
    {
        deviceId = MPI_RANK;
        gpuProc = 0;
    }
    else if (nDevs > MPI_SZ) // there are more devices than processes
    {
        deviceId = MPI_RANK;
        gpuProc = 0;
    }
    else // there more processes than devices
    {
        int     nBasic = MPI_SZ / nDevs,
                nRemain = MPI_SZ % nDevs;

        if (MPI_RANK < (nBasic+1)*nRemain)
        {
            deviceId = MPI_RANK / (nBasic + 1);
            if (MPI_RANK % (nBasic + 1) == 0)  gpuProc = 0;
        }
        else
        {
            deviceId = (MPI_RANK - (nBasic+1)*nRemain) / nBasic + nRemain;
            if ((MPI_RANK - (nBasic+1)*nRemain) % nBasic == 0) gpuProc = 0;
        }
    }

    //printf("MPI RANK = %d gpuProc = %d deviceId = %d\n",MPI_RANK,gpuProc,deviceId);
    MPI_Comm_split(cpuWorld, gpuProc, 0, &amgx_comm);



    //printf("No of devices %d deviceId %d \n", nDevs, deviceId);

    //If using GPU init!
    if(gpuProc == 0) {

      if(MPI_RANK == 0){
      printf("setting up AMGX \n");
      AMGX_SAFE_CALL(AMGX_initialize());

      AMGX_SAFE_CALL(AMGX_initialize_plugins());

      AMGX_SAFE_CALL(AMGX_install_signal_handler());
      }

      AMGX_SAFE_CALL(AMGX_config_create_from_file(&cfg, amgx_str.c_str()));

      //Number of devices is set to 1

      if (MPI_RANK == 0) {AMGX_resources_create(&rsrc, cfg, &amgx_comm, 1, &deviceId);}

      AMGX_vector_create(&x_amgx, rsrc, amgx_mode);
      AMGX_vector_create(&b_amgx, rsrc, amgx_mode);

      AMGX_matrix_create(&A_amgx, rsrc, amgx_mode);

      AMGX_solver_create(&solver_amgx, rsrc, amgx_mode, cfg);

      // obtain the default number of rings based on current configuration
      AMGX_config_get_default_number_of_rings(cfg, &ring);
    }

    int nLocalRows;
    int nGlobalRows = A.M();
    int globalNNZ = A.NNZ();

    //Step 1.
    //Merge Diagonal and OffDiagonal into a single CSR matrix
    myGetLocalA(A, loc_I, loc_J, loc_A);
    /*
    printf("\n Look Here \n\n");
    for(int i = 0; i < loc_A.Size(); i++){
      printf("%g \n", loc_A[i]);
    };
    */
    //Get J and count all NNZ
    int J_allsz(0), all_NNZ(0);
    const int loc_Jz_sz = loc_J.Size();
    const int loc_A_sz = loc_A.Size();

    MPI_Allreduce(&loc_Jz_sz, &J_allsz, 1, MPI_INT, MPI_SUM, cpuWorld);
    MPI_Allreduce(&loc_A_sz, &all_NNZ, 1, MPI_INT, MPI_SUM, cpuWorld);
    MPI_Barrier(cpuWorld);


  //  printf("all_Jz %d all_NNZ %d global_NNZ %d \n", J_allsz, all_NNZ, globalNNZ);

  //  printf("loc_I.Size() %d \n", loc_I.Size());
  //  printf("nGlobalRows+MPI_SZ %d \n", nGlobalRows+MPI_SZ);

    //Consolidate to rank 0
    Array<int> all_I(nGlobalRows+MPI_SZ);
    Array<int64_t> all_J(J_allsz); all_J = 0.0;
    Array<double> all_A(all_NNZ);


    GatherArray(loc_I, all_I, MPI_SZ, cpuWorld);
    GatherArray(loc_J, all_J, MPI_SZ, cpuWorld);
    GatherArray(loc_A, all_A, MPI_SZ, cpuWorld);
    MPI_Barrier(cpuWorld);
    Array<int> z_ind(MPI_SZ+1);

    int local_nnz=0;
    if(gpuProc==0) {

      int iter = 1;

      while(iter < MPI_SZ-1){
        //Determine the indices of zeros in global all_I array
        int counter = 0;
        z_ind[counter] = counter;
        counter++;
        for(int idx=1; idx<all_I.Size()-1; idx++){
          if(all_I[idx]==0){
            z_ind[counter] = idx-1;
            counter++;
          }
        }
        z_ind[MPI_SZ] = all_I.Size()-1;
        //End of determining indices of zeros in global all_I Array

        //Bump all_I
        for(int idx=z_ind[1]+1; idx < z_ind[2]; idx++){
	        all_I[idx] = all_I[idx-1] + (all_I[idx+1] - all_I[idx]);
        }

        //Shift array after bump to remove uncesssary values in middle of array
        for(int idx=z_ind[2]; idx < all_I.Size()-1; ++idx){
          all_I[idx] = all_I[idx+1];
        }
        iter++;
    }

    // LAST TIME THROUGH ARRAY
    //Determine the indices of zeros in global row_ptr array
    int counter = 0;
    z_ind[counter] = counter;
    counter++;
    for(int idx=1; idx<all_I.Size()-1; idx++){
      if(all_I[idx]==0){
        z_ind[counter] = idx-1;
        counter++;
      }
    }
    z_ind[MPI_SZ] = all_I.Size()-1;
    //End of determining indices of zeros in global all_I Array\

    //BUMP all_I one last time
    for(int idx=z_ind[1]+1; idx < all_I.Size()-1; idx++){
      all_I[idx] = all_I[idx-1] + (all_I[idx+1] - all_I[idx]);
    }

      local_nnz = all_I[all_I.Size()-MPI_SZ];
      nLocalRows = A.M();

    }else{
      nLocalRows = 0;
      local_nnz = 0;
    }

  //  printf("MPI_RANK %d : NNZ %d \n", MPI_RANK, local_nnz);

    //Step 2.
    //Create a vector of offsets describing matrix row partitions
    mfem::Array<int64_t> rowPart(2);
    rowPart[0] = 0;
    rowPart[1] = A.M();

    //printf("uses rowPart size %d \n", rowPart.Size());
    for(int i=0; i<rowPart.Size(); ++i){
    //  printf("%ld ",rowPart[i]);
  }//printf("\n");
    //const int m_I_nLocalRows = m_I.HostRead()[nLocalRows];

   //printf("id %d nGlobalRows %d nLocalRows %d nnz %d \n",
	   //MPI_RANK, nGlobalRows, nLocalRows, local_nnz);


    //if(MPI_RANK ==0) {
    if(amgx_comm != MPI_COMM_NULL) {

      MPI_Barrier(amgx_comm);
      AMGX_distribution_handle dist;
      AMGX_distribution_create(&dist, cfg);
      AMGX_distribution_set_partition_data(dist, AMGX_DIST_PARTITION_OFFSETS,
					   rowPart.GetData());

    //  printf("my rank %d in amgx \n", MPI_RANK);
      printf("all sizes I = %d J = %d A = %d \n",all_I.Size(), all_J.Size(), all_A.Size());


      AMGX_matrix_upload_distributed(A_amgx, nGlobalRows, nLocalRows,
				     local_nnz,
				     1, 1, all_I.HostReadWrite(),
				     all_J.HostReadWrite(), all_A.HostReadWrite(),
				     nullptr, dist);
      AMGX_distribution_destroy(dist);

      MPI_Barrier(amgx_comm);

      AMGX_solver_setup(solver_amgx, A_amgx);

      //Step 4. Bind vectors to A
      AMGX_vector_bind(x_amgx, A_amgx);
      AMGX_vector_bind(b_amgx, A_amgx);
    }

    //Gather vectors
    Vector all_X(nGlobalRows);
    Vector all_B(nGlobalRows);
    Array<int> Apart_X(MPI_SZ);
    Array<int> Adisp_X(MPI_SZ);
    Array<int> Apart_B(MPI_SZ);
    Array<int> Adisp_B(MPI_SZ);

    GatherArray(X, all_X, MPI_SZ, cpuWorld, Apart_X, Adisp_X);
    GatherArray(B, all_B, MPI_SZ, cpuWorld, Apart_B, Adisp_B);
    MPI_Barrier(cpuWorld);
    Apart_X.Print();
    Adisp_B.Print();
    if(amgx_comm != MPI_COMM_NULL) {

      AMGX_vector_upload(x_amgx, all_X.Size(), 1, all_X.HostReadWrite());
      AMGX_vector_upload(b_amgx, all_B.Size(), 1, all_B.HostReadWrite());

      MPI_Barrier(amgx_comm);

      AMGX_solver_solve(solver_amgx,b_amgx, x_amgx);

      AMGX_SOLVE_STATUS   status;
      AMGX_solver_get_status(solver_amgx, &status);
      if (status != AMGX_SOLVE_SUCCESS)
	{
	  printf("Amgx failed to solve system, error code %d. \n", status);
	}


      AMGX_vector_download(x_amgx, all_X.HostWrite());



    }


    //Clean up
    if(gpuProc == 0)
      {
	AMGX_solver_destroy(solver_amgx);
	AMGX_matrix_destroy(A_amgx);

	AMGX_vector_destroy(x_amgx);
	AMGX_vector_destroy(b_amgx);


	AMGX_SAFE_CALL(AMGX_config_destroy(cfg));

	AMGX_SAFE_CALL(AMGX_finalize_plugins());
	AMGX_SAFE_CALL(AMGX_finalize());
	MPI_Comm_free(&amgx_comm);
      }
      ScatterArray(all_X, X, MPI_SZ, cpuWorld,Apart_X,Adisp_X);


  }//Mini example end


   // 14. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
  a->RecoverFEMSolution(X, *b, x);




  // 15. Save the refined mesh and the solution in parallel. This output can
  //     be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
  {
    ostringstream mesh_name, sol_name;
    mesh_name << "mesh." << setfill('0') << setw(6) << myid;
    sol_name << "sol." << setfill('0') << setw(6) << myid;

    ofstream mesh_ofs(mesh_name.str().c_str());
    mesh_ofs.precision(8);
    pmesh->Print(mesh_ofs);

    ofstream sol_ofs(sol_name.str().c_str());
    sol_ofs.precision(8);
    x.Save(sol_ofs);
  }

  // 16. Send the solution by socket to a GLVis server.
  if (visualization)
    {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *pmesh << x << flush;
    }

  // 17. Free the used memory.
  delete a;
  delete b;
  delete fespace;
  if (order > 0) { delete fec; }
  delete pmesh;

  MPI_Finalize();

  return 0;
}
