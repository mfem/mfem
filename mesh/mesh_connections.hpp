class MeshConnections
{
   private:
   Mesh *mesh;
   Array<Array<Table*>> T;

   public:
   struct EntityID
   {
      int dim;
      int id;
      bool boundary;
   }

   struct EntityIDs
   {
      int dim;
      Array<int> ids;
      bool boundary;
   }

   //Dimension independent interface
   bool        IsChild(EntityID parent, EntityID child);
   Array<int> &Neighbors(EntityID entity, int shared_dim);
   Array<int> &ChildrenOfEntity(EntityID parent, int child_dim);
   Array<int> &ChildrenOfEntities(EntityIDs parents, int child_dim);
   Array<int> &ParentsOfEntity(EntityID child, int parent_dim);
   Array<int> &ParentsOfAnyEntities(EntityIDs children, int parent_dim);
   Array<int> &ParentsCoveredByEntities(EntityIDs children, int parent_dim);

   private:
   //Raw table access
   Table *GetTable(int dim_m, int dim_n);
};