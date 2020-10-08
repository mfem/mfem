#include <fstream>
#include "xfl.hpp"

// *****************************************************************************
extern FILE *yyin;
int yyparse(Node**);
int yylex_destroy(void);

// ****************************************************************************
// * Node constructor/destructor
// ****************************************************************************
Node::Node(const int sn):
   keep(false),
   next(nullptr), children(nullptr), parent(nullptr),
   sn(sn),  id(0), nb_siblings(0) { } //DBG("\n\033[32m[Node] %s",name); }

// ****************************************************************************
Node::~Node() { } //DBG("\n\033[31m[~Node] %s", name.c_str()); }

// ****************************************************************************
// * astNewNode, astNewToken & astNewRule
// ****************************************************************************
Node *astAddNode(Node_ptr n_ptr)
{
   assert(n_ptr);
   static std::list<Node_ptr> node_list_to_be_destructed;
   node_list_to_be_destructed.push_back(n_ptr);
   return n_ptr.get();
}

// ****************************************************************************
// * astAddChild
// ****************************************************************************
Node* astAddChild(Node* root, Node* n)
{
   assert(n);
   assert(root);
   n->parent = root;
   if (!root->children) { return root->children = n; }
   Node* child = root->children;
   for (; child->next; child = child->next);
   return child->next = n;
}

// ****************************************************************************
// * astAddNext
// ****************************************************************************
Node* astAddNext(Node* root, Node* n)
{
   assert(n);
   assert(root);
   n->parent = root->parent;
   if (!root->next) { return root->next = n; }
   Node* next = root;
   for (; next->next; next = next->next);
   return next->next = n;
}

// ****************************************************************************
// * tr \" to \'
// ****************************************************************************
static const char* strKillQuote(const std::string &str)
{
   char *p = const_cast<char*>(str.c_str());
   for (; *p != 0; p++) if (*p == '\"') { *p = '\''; }
   return str.c_str();
}

// *****************************************************************************
static bool skip(const Node *n, const bool simplify)
{
   if (!simplify) { return false; }
   if (yy::is_token(n)) { return false; }
   if (n->nb_siblings == 0) { return false; }
   if (n->keep) { return false; }
   const bool is_child_a_rule = yy::is_rule(n->children);
   return is_child_a_rule && n->children->nb_siblings == 1;
}

// *****************************************************************************
static int astTreeSaveNodes(FILE* fTreeOutput, Node *n,
                            const bool simplify, int id)
{
   for (; n not_eq NULL; n = n->next)
   {
      if (yy::is_rule(n))
      {
         if (!skip(n, simplify))
            fprintf(fTreeOutput,
                    "\n\tNode_%d [label=\"%s\" color=\"#%s\"]",
                    n->id = id++,
                    strKillQuote(n->Name()),
                    skip(n, true) ? "FFDDCC" : "CCDDCC");
      }
      else
      {
         fprintf(fTreeOutput,
                 "\n\tNode_%d [label=\"%s\" color=\"#CCCCDD\"]",
                 n->id = id++,
                 strKillQuote(n->Name()));
      }
      id = astTreeSaveNodes(fTreeOutput, n->children, simplify, id);
   }
   return id;
}

// *****************************************************************************
static int astTreeSaveEdges(FILE* fTreeOutput,
                            const Node *n,
                            const Node *father,
                            const bool simplify)
{
   for (; n; astTreeSaveEdges(fTreeOutput, n->children, n, simplify),
        n = n->next)
   {
      if (skip(n, simplify)) { continue; }
      const Node *from = father;
      while (skip(from, simplify)) { from = from->parent; }
      fprintf(fTreeOutput, "\n\tNode_%d -> Node_%d;", from->id, n->id);
   }
   return 0;
}

// *****************************************************************************
static void setNbSiblings(const Node *n)
{
   int nb_siblings = 0;
   for (Node *c = n->children; c; c = c->next) { nb_siblings += 1; }
   for (Node *c = n->children; c; c = c->next) { c->nb_siblings = nb_siblings; }
}

// *****************************************************************************
static void astNbSiblings(Node *n)
{
   if (!n) { return; }
   if (n->parent && n->nb_siblings == 0) { setNbSiblings(n->parent); }
   astNbSiblings(n->children);
   astNbSiblings(n->next);
}

// *****************************************************************************
// * astTreeSave
// *****************************************************************************
int astTreeSave(const char* file_name, Node *root, const bool simplify)
{
   FILE *file;
   char fName[FILENAME_MAX];
   // ***************************************************************************
   astNbSiblings(root);
   // ***************************************************************************
   sprintf(fName, "%s.dot", file_name);
   // Saving tree file
   if ((file = fopen(fName, "w")) == 0)
   {
      return -1 | printf("[astTreeSave] fopen ERROR");
   }
   fprintf(file,
           "digraph {\nordering=out;\n\tNode [style = filled, shape = circle];");

   astTreeSaveNodes(file, root, simplify, 0);
   if (astTreeSaveEdges(file, root->children, root, simplify) not_eq 0)
   {
      return -1 | printf("[astTreeSave] ERROR");
   }
   fprintf(file, "\n}\n");
   fclose(file);
   return 0;
}

// *****************************************************************************
#define XFL_man "[1;36m[XFL] version %.1f[m\n"
#define XFL_version 0.01

// *****************************************************************************
int main(const int argc, char* argv[])
{
   int c;
   Node *root = nullptr;
   int longindex = 0;
   bool simplify = false;
   bool dump = false;
   bool gpu = false;
   std::string i_filename, o_filename;

   const struct option longopts[] =
   {
      {"help", no_argument, NULL, OPTION_HELP},
      {NULL, 0, NULL, 0}
   };

   if (argc == 1) { exit(~0 | fprintf(stderr, XFL_man, XFL_version)); }

   while ((c = getopt_long(argc,argv,"gtTedi:o:h",longopts,&longindex)) != -1)
   {
      switch (c)
      {
         case 'g': gpu = true; break;
         case 't': simplify = dump = true; break;
         case 'T': dump = true; break;
         case 'e': yyecho = true; break;
         case 'd':
#if YYDEBUG
            yydebug = 1;
#endif
            break;
         case 'i': i_filename.assign(optarg); break;
         case 'o': o_filename.assign(optarg); break;
         case 'h':
         case OPTION_HELP: printf("[xfl] OPTION_HELP\n"); return 0;
         case '?':
            if ((optopt > (int)'A') && (optopt < (int)'z'))
            {
               fprintf (stderr, "[xfl] Unknown option `-%c'.\n", optopt);
            }
            else { fprintf (stderr, "[xfl] Unknown option character `\\%d'.\n", optopt); }
            exit(~0);
         default: exit(~0 | fprintf(stderr, "[xfl] Error in command line\n"));
      }
   }
   const bool is_i_file = !i_filename.empty();
   const bool is_o_file = !o_filename.empty();
   assert(is_i_file == is_o_file);
   std::ifstream i_file(i_filename.c_str(), std::ios::in | std::ios::binary);
   if (is_i_file)
   {
      assert(!i_file.fail());
      assert(i_file.is_open());
   }
   std::ofstream o_file(o_filename.c_str(),
                        std::ios::out | std::ios::binary | std::ios::trunc);
   if (is_o_file)
   {
      assert(o_file.is_open());
   }
   //std::istream &in = is_i_file ? i_file : std::cin;
   std::ostream &out = is_o_file ? o_file : std::cout;

   // point to next command line input
   if (!is_i_file)
   {
      if (optind >= argc) { exit(~0 | fprintf(stderr, "[31m[xfl] no input file[m\n")); }
      if (yyecho) { printf("[32m[xfl] argv: '%s'[m\n", argv[optind]); }
      optarg = argv[optind++];
      i_filename.assign(optarg);
      assert(optind == argc); // only one input file for now
   }

   if (!(yyin = fopen(i_filename.c_str(), "r")))
   {
      return printf("[xfl] Could not open '%s' file\n", i_filename.c_str());
   }
   if (yyecho) { printf("[32m[xfl] parse: %s[m\n", i_filename.c_str()); }

   if (yyparse(&root))
   {
      return fclose(yyin) | printf("[xfl] Error while parsing!\n");
   }
   assert(root);

   if (gpu)
   {
      struct gpu dev(out);
      dfs(root, dev);
   }
   else
   {
      struct cpu dev(out);
      dfs(root, dev);
   }

   if (dump)
   {
      assert(root);
      astTreeSave("ast", root, simplify);
   }

   fclose(yyin);
   fflush(NULL);
   yylex_destroy();
   return 0;
}

