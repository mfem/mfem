class MeshConnections
{
   private:
   Mesh *mesh;
   Table *elem_to_elem, *elem_to_face, *elem_to_edge, *elem_to vert;
   Table *face_to_elem, *face_to_face, *face_to_edge, *face_to vert;
   Table *edge_to_elem, *edge_to_face, *edge_to_edge, *edge_to vert;
   Table *vert_to_elem, *vert_to_face, *vert_to_edge, *vert_to vert;

   public:
   //Queries
   bool IsFaceInElem(int face, int elem);
   bool IsEdgeInElem(int edge, int elem);
   bool IsVertInElem(int vert, int elem);
   bool IsEdgeInFace(int edge, int face);
   bool IsVertInFace(int face, int elem);
   bool IsVertInEdge(int edge, int elem);
   
   //Neighbor mappings
   void NeighborsToElem(int elem, Array<int> &elems);
   void NeighborsToFace(int face, Array<int> &faces);
   void NeighborsToEdge(int edge, Array<int> &edges);
   void NeighborsToVert(int vert, Array<int> &verts);

   //Singular forward mappings
   void FacesInElem(Array<int> &faces, int elem);
   void EdgesInElem(Array<int> &edges, int elem);
   void VertsInElem(Array<int> &verts, int elem);
   void EdgesInFace(Array<int> &edges, int face);
   void VertsInFace(Array<int> &verts, int face);
   void VertsInEdge(Array<int> &verts, int edge);

   //Mutiple forward mappings
   void FacesInElems(Array<int> &faces, const Array<int> &elems);  
   void EdgesInElems(Array<int> &edges, const Array<int> &elems);
   void VertsInElems(Array<int> &verts, const Array<int> &elems);
   void EdgesInFaces(Array<int> &edges, const Array<int> &faces);
   void VertsInFaces(Array<int> &verts, const Array<int> &edges);
   void VertsInEdges(Array<int> &verts, const Array<int> &edges);
      
   //Singular reverse mappings
   void ElemsWithFace(Array<int> &elems, int face);
   void ElemsWithEdge(Array<int> &elems, int edge);
   void ElemsWithVert(Array<int> &elems, int vert);
   void FacesWithEdge(Array<int> &faces, int edge);
   void FacesWithVert(Array<int> &faces, int vert);
   void EdgesWithVert(Array<int> &edges, int vert);

   //Multiple Reverse mappings for the bigger containing
   //ANY instances in the smaller
   //E.G.  ElemsWithAnyFaces will return the elements
   //      that have ANY of their faces denoted in the "faces" array
   void ElemsTouchedByFaces(Array<int> &elems, const Array<int> &faces);
   void ElemsTouchedByEdges(Array<int> &elems, const Array<int> &edges);
   void ElemsTouchedByVerts(Array<int> &elems, const Array<int> &verts);
   void FacesTouchedByEdges(Array<int> &faces, const Array<int> &edges);
   void FacesTouchedByVerts(Array<int> &faces, const Array<int> &verts);
   void EdgesTouchedByVerts(Array<int> &edges, const Array<int> &verts);   

   //Multiple Reverse mappings for the bigger containing
   //ANY instances in the smaller
   //E.G.  ElemsWithAllFaces will return the elements
   //      that have ALL of their faces denoted in the "faces" array
   void ElemsContainedByFaces(Array<int> &elems, const Array<int> &faces);
   void ElemsContainedByEdges(Array<int> &elems, const Array<int> &edges);
   void ElemsContainedByVerts(Array<int> &elems, const Array<int> &verts);
   void FacesContainedByEdges(Array<int> &faces, const Array<int> &edges);
   void FacesContainedByVerts(Array<int> &faces, const Array<int> &verts);
   void EdgesContainedByVerts(Array<int> &edges, const Array<int> &verts);

   //Raw table access
   Table *ElemToVertTable();
   Table *ElemToFaceTable();
   Table *ElemToEdgeTable();
   Table *ElemToVertTable();
   Table *FaceToVertTable();
   Table *FaceToFaceTable();
   Table *FaceToEdgeTable();
   Table *FaceToVertTable();
   Table *EdgeToVertTable();
   Table *EdgeToFaceTable();
   Table *EdgeToEdgeTable();
   Table *EdgeToVertTable();
   Table *VertToVertTable();
   Table *VertToFaceTable();
   Table *VertToEdgeTable();
   Table *VertToVertTable();
};