#include "mfem.hpp"

#ifdef MFEM_USE_PETSC
#include "exodusII.h"
#endif

int main(int argc, char *argv[])
{
    // Parse command-line options.
    const char *mesh_file = "input.exo";
    const char *gmsh_file = "gmsh.out";


    mfem::OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
                   "Mesh file to use.");

    args.AddOption(&gmsh_file, "-o", "--output",
                   "GMSH output file.");


    args.Parse();
    if (!args.Good())
    {
       args.PrintUsage(std::cout);
       return 1;
    }
    args.PrintOptions(std::cout);
#ifdef MFEM_USE_PETSC

    std::fstream gmsh;
    gmsh.open(gmsh_file,std::ios::out);
    gmsh<<"$MeshFormat\n2 0 8\n$EndMeshFormat"<<std::endl;

    int comp_ws=sizeof(double);
    int io_ws=0;
    float version;

    int exo_id=ex_open(mesh_file,EX_READ,&comp_ws,&io_ws,&version);

    int num_dim, num_nodes, num_elems;
    int num_elem_blk, num_node_sets, num_side_sets;
    char title[256];


    ex_get_init(exo_id, title, &num_dim, &num_nodes, &num_elems, &num_elem_blk, &num_node_sets,
                      &num_side_sets);


    std::cout<<"num_dim="<<num_dim<<std::endl;
    std::cout<<"num_nodes="<<num_nodes<<std::endl;
    std::cout<<"num_elems="<<num_elems<<std::endl;
    std::cout<<"num_elem_blk="<<num_elem_blk<<std::endl;
    std::cout<<"num_node_sets="<<num_node_sets<<std::endl;
    std::cout<<"num_side_sets="<<num_side_sets<<std::endl;

    //read the nodes and dump them to a file
    {
        int numn=num_nodes;
        //ex_inquire(exo_id,EX_INQ_NODES,&numn ,NULL,NULL);
        std::cout<<"Num nodes ="<<numn<<std::endl;

        double* x=new double[numn];
        double* y=new double[numn];
        double* z=new double[numn];

        ex_get_coord(exo_id,x,y,z);
        gmsh<<"$Nodes"<<std::endl;
        gmsh<<numn<<std::endl;

        for(int i=0;i<numn;i++)
        {
            gmsh<<i+1<<" "<<x[i]<<" "<<y[i]<<" "<<z[i]<<std::endl;
        }

        gmsh<<"$EndNodes"<<std::endl;

        delete [] x;
        delete [] y;
        delete [] z;
    }

    
    //do the element processing
    {
        std::cout<<"Num blocks="<<num_elem_blk<<std::endl;
        int ids[num_elem_blk];
        ex_get_ids(exo_id, EX_ELEM_BLOCK, ids);
        char el_type[MAX_STR_LENGTH+1];
        int numel;
        int numel_nod;
        int numel_attr;

        gmsh<<"$Elements"<<std::endl;
        gmsh<<num_elems<<std::endl;

        int el_id=0;

        for(int bl=0;bl<num_elem_blk;bl++)
        {
            ex_get_block(exo_id,EX_ELEM_BLOCK,ids[bl],
                         el_type,&numel,&numel_nod,nullptr,nullptr,&numel_attr);

            std::cout<<"bl="<<ids[bl]<<" nel="<<numel;
            std::cout<<" nod="<<numel_nod<<" natt="<<numel_attr;
            std::cout<<std::endl;
            int* conn=new int[numel*numel_nod];

            ex_get_conn(exo_id,EX_ELEM_BLOCK,ids[bl],conn,nullptr,nullptr);

            for(int i=0;i<numel;i++)
            {
                el_id++;
                gmsh<<el_id<<" ";
                //element type
                if(strcasecmp(el_type,"TETRA4")==0){
                    gmsh<<"4 ";
                }else if(strcasecmp(el_type,"TETRA10")==0){
                    gmsh<<"11 ";
                }
                else if(strcasecmp(el_type,"HEX8")==0){
                    gmsh<<"5 ";
                }else if(strcasecmp(el_type,"BAR2")==0){
                    gmsh<<"1 ";
                }else if(strcasecmp(el_type,"TRI3")==0){
                    gmsh<<"2 ";
                }else if(strcasecmp(el_type,"SHELL4")==0){
                    gmsh<<"3 ";
                }

                //dump the rest of the element data
                gmsh<<"3 "<<ids[bl]<<" "<<ids[bl];
                gmsh<<" 0"; //num tags, geom domain, volume

                for(int ii=0;ii<numel_nod;ii++)
                {
                    gmsh<<" "<<conn[i*numel_nod+ii];
                }
                gmsh<<std::endl;
              }

              delete [] conn;
        }
        gmsh<<"$EndElements"<<std::endl;
    }
    

    gmsh.close();
    ex_close(exo_id);
#endif
}
