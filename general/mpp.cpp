#include <list>
#include <string>
#include <cstring>
#include <ciso646>
#include <cassert>
#include <fstream>
#include <iostream>
using namespace std;
#define STR(X) #X
#define STRINGIFY(X) STR(X)

// *****************************************************************************
#define trk(...) {printf("\n%s>",__func__);fflush(0);}
#define dbg(...) {printf(__VA_ARGS__);fflush(0);}

// *****************************************************************************
// * STRUCTS: context, error & args
// *****************************************************************************
struct argument {
  string type, name;
  bool star, is_const, restrict;
  argument(): star(false), is_const(false), restrict(false) {}
};

// *****************************************************************************
struct kernel{
   bool jit;
   string xcc;
   string dirname;
   string name;
   string static_format;
   string static_args;
   string static_tmplt;
   string any_pointer_params;
   string any_pointer_params_;
   string any_pointer_args;
   string d2u;
   string u2d;
};

// *****************************************************************************
struct context {
#ifdef MFEM_USE_GPU
   const bool mm = true;
#else
   const bool mm = false;
#endif
#ifdef MFEM_USE_JIT
   const bool jit = true;
#else
   const bool jit = false;
#endif
   int line;
   int block;
   string& file;
   istream& in;
   ostream& out;
   std::list<argument*> args;
   kernel k;
public:
   context(istream& i, ostream& o, string &f)
      : line(1), block(-2), file(f), in(i), out(o){}
};

// *****************************************************************************
struct error {
   int line;
   string file;
   const char *msg;
   error(int l, string f, const char *m): line(l), file(f), msg(m) {}
};

// *****************************************************************************
static const char* strrnc(const char* s, const unsigned char c, int n=1) {
   size_t len = strlen(s);
   char* p = (char*)s+len-1;
   for (; n; n--,p--,len--) {
      for (; len; p--,len--)
         if (*p==c) break;
      if (! len) return NULL;
      if (n==1) return p;
   }
   return NULL;
}

// *****************************************************************************
static inline void check(context &pp, const bool test, const char *msg = NULL){
   if (not test) throw error(pp.line,pp.file,msg);
}

// *****************************************************************************
static inline int help(char* argv[]) {
   cout << "MFEM preprocessor:";
   cout << argv[0] << " -o output input" << endl;
   return ~0;
}

// *****************************************************************************
static inline bool is_newline(const int ch) {
   return static_cast<unsigned char>(ch) == '\n';
}

// *****************************************************************************
static inline int get(context &pp) {
   return pp.in.get();
}

// *****************************************************************************
static inline int put(context &pp) {
   const int c = get(pp);
   assert(c != EOF);
   pp.out.put((char) c);
   return c;
}

// *****************************************************************************
static inline void skip_space(context &pp) {
   while (isspace(pp.in.peek())) {
      const int c = pp.in.peek();
      check(pp,c!='\v',"Vertical tab detected!");
      if (is_newline(c)) pp.line++;
      put(pp);
   }
}

// *****************************************************************************
static inline void drop_space(context &pp) {
   while (isspace(pp.in.peek())) {
      if (is_newline(pp.in.peek())) pp.line++;
      pp.in.get();
   }
}

// *****************************************************************************
static inline bool is_comment(context &pp) {
   if (pp.in.peek() != '/') return false;
   pp.in.get();
   assert(!pp.in.eof());
   const int c = pp.in.peek();
   //assert(c!=EOF);
   pp.in.unget();
   if (c == '/' || c == '*') return true;
   return false;
}

// *****************************************************************************
static inline void singleLineComment(context &pp) {
   while (/*pp.in.peek()!=EOF &&*/not is_newline(pp.in.peek())) put(pp);
   pp.line++;
}

// *****************************************************************************
static inline void blockComment(context &pp) {
   char c;
   while (pp.in.get(c)) {
      pp.out.put(c);
      if (is_newline(c)) pp.line++;
      if (c == '*' && pp.in.peek() == '/') {
         put(pp);
         skip_space(pp);
         return;
      }
   }
}

// *****************************************************************************
static inline void comments(context &pp) {
   const int c1 = put(pp); assert(c1 != EOF);
   check(pp,c1=='/',"Comments w/o 1st char");
   const int c2 = put(pp); assert(c2 != EOF);
   check(pp,c2=='/' || c2=='*',"Comment w/o 2nd char");
   if (c2 == '/') return singleLineComment(pp);
   return blockComment(pp);
}

// *****************************************************************************
static inline bool is_alnum(context &pp) {
   const int c = pp.in.peek();
   assert(c != EOF);
   return isalnum(c) || c == '_';
}

// *****************************************************************************
static inline string get_name(context &pp) {
   string str;
   check(pp,is_alnum(pp),"Name w/o alnum 1st letter");
   while (pp.in.peek()!=EOF && is_alnum(pp))
      str += pp.in.get();
   return str;
}

// *****************************************************************************
static inline string get_directive(context &pp) {
   string str;
   check(pp,pp.in.peek()=='#',"Directive w/o 1st '#'");
   while (pp.in.peek()!=EOF &&
          (is_alnum(pp) || pp.in.peek()=='#'))
      str += pp.in.get();
   return str;
}

// *****************************************************************************
static inline string peekn(context &pp, const int n) {
   int k = 0;
   assert(n<64);
   static char c[64];
   for (k=0;k<=n;k+=1) c[k] = 0;
   for (k=0; k<n; k+=1) {
      if (pp.in.peek()==EOF) break;
      c[k] = pp.in.get();
   }
   string rtn = c;
   for (int l=0; l<k; l+=1) pp.in.unget();
   return rtn;
}

// *****************************************************************************
static inline string peekID(context &pp) {
   int k = 0;
   const int n = 64;
   static char c[64];
   for (k=0;k<n;k+=1) c[k] = 0;
   for (k=0; k<n; k+=1) {
      if (pp.in.peek()==EOF) break;
      if (! is_alnum(pp)) break;
      c[k]=pp.in.get();
   }
   string rtn(c);
   for (int l=0; l<k; l+=1) pp.in.unget();
   return rtn;
}

// *****************************************************************************
static inline void drop_name(context &pp) {
   while (pp.in.peek()!=EOF && is_alnum(pp))
      pp.in.get();
}

// *****************************************************************************
static inline bool isvoid(context &pp) {
   skip_space(pp);
   const string void_peek = peekn(pp,4);
   if (void_peek == "void") return true;
   return false;
}

// *****************************************************************************
static inline bool isstatic(context &pp) {
   skip_space(pp);
   const string void_peek = peekn(pp,6);
   if (void_peek == "static") return true;
   return false;
}

// *****************************************************************************
static inline bool is_star(context &pp) {
   skip_space(pp);
   if (pp.in.peek() == '*') return true;
   return false;
}

// *****************************************************************************
static inline bool is_coma(context &pp) {
   skip_space(pp);
   if (pp.in.peek() == ',') return true;
   return false;
}

// *****************************************************************************
static inline bool get_args(context &pp) {
   trk();
   bool empty = true;
   argument *arg = new argument();
   for (int p=0; pp.in.peek() != EOF; empty=false) {
      if (is_star(pp)){
         arg->star = true;
         put(pp);
         continue;
      }
      if (is_coma(pp)){
         put(pp);
         continue;
      }
      const string &id = peekID(pp);
      dbg(" id:%s",id.c_str());
      drop_name(pp);
      // Qualifiers
      if (id=="const") { pp.out << id; arg->is_const = true; continue; }
      if (id=="__restrict") { pp.out << id; arg->restrict = true; continue; }
      // Types
      if (id=="char") { pp.out << id; arg->type = id; continue; }
      if (id=="int") { pp.out << id; arg->type = id; continue; }
      if (id=="short") { pp.out << id; arg->type = id; continue; }
      if (id=="unsigned") { pp.out << id; arg->type = id; continue; }
      if (id=="long") { pp.out << id; arg->type = id; continue; }
      if (id=="bool") { pp.out << id; arg->type = id; continue; }
      if (id=="float") { pp.out << id; arg->type = id; continue; }
      if (id=="double") { pp.out << id; arg->type = id; continue; }
      if (id=="size_t") { pp.out << id; arg->type = id; continue; }
      const bool jit = pp.k.jit;
      pp.out << ((jit || ! pp.mm)?"":"_") << id;
      // focus on the name, we should have qual & type
      arg->name = id;
      pp.args.push_back(arg);
      arg = new argument();
      const int c = pp.in.peek();
      check(pp,c != EOF,"EOF while in args");
      if (c == '(') p+=1;
      if (c == ')') p-=1;
      if (is_newline(c)) pp.line++;
      if (p<0) { return empty; }
      skip_space(pp);
      check(pp,pp.in.peek()==',',"No coma while in args");
      put(pp);
   }
   assert(false);
   dbg("eof return");
   return empty;
}

// *****************************************************************************
static inline void rtcKernelRefresh(context &pp){
   trk();
   pp.k.xcc = STRINGIFY(MFEM_CXX) " " \
      STRINGIFY(MFEM_BUILD_FLAGS) " " \
      "-O3 -std=c++11 -Wall";
   pp.k.dirname = STRINGIFY(MFEM_SRC);
   pp.k.static_args.clear();
   pp.k.static_tmplt.clear();
   pp.k.static_format.clear();
   pp.k.any_pointer_args.clear();
   pp.k.any_pointer_params.clear();
   pp.k.any_pointer_params_.clear();
   pp.k.d2u.clear();
   pp.k.u2d.clear();
   
   for(std::list<argument*>::iterator ia = pp.args.begin();
       ia != pp.args.end() ; ia++) {
      const argument *a = *ia;
      const bool is_const = a->is_const;
      //const bool is_restrict = a->restrict;
      const bool is_pointer = a->star;
      const char *type = a->type.c_str();
      const char *name = a->name.c_str();
      if (is_const && ! is_pointer){
         const bool dbl = strcmp(type,"double")==0;
         if (! pp.k.static_format.empty()) pp.k.static_format += ",";
         pp.k.static_format += dbl?"0x%lx":"%ld";
         if (! pp.k.static_args.empty()) pp.k.static_args += ",";
         pp.k.static_args += dbl?"u":"";
         pp.k.static_args += name;
         if (! pp.k.static_tmplt.empty()) pp.k.static_tmplt += ",";
         pp.k.static_tmplt += "const ";
         pp.k.static_tmplt += dbl?"uint64_t":type;
         pp.k.static_tmplt += " ";
         pp.k.static_tmplt += dbl?"t":"";
         pp.k.static_tmplt += name;
         if (dbl){
            {
               pp.k.d2u += "const double ";
               pp.k.d2u += name;
               pp.k.d2u += " = (union_du){u:t";
               pp.k.d2u += name;
               pp.k.d2u += "}.d;";
            }
            {
               pp.k.u2d += "const uint64_t u";
               pp.k.u2d += name;
               pp.k.u2d += " = (union_du){";
               pp.k.u2d += name;
               pp.k.u2d += "}.u;";
            }
         }
      }
      if (is_const && is_pointer){
         if (! pp.k.any_pointer_args.empty()) pp.k.any_pointer_args += ",";
         pp.k.any_pointer_args += name;
         if (! pp.k.any_pointer_params.empty()) {
            pp.k.any_pointer_params += ",";
            pp.k.any_pointer_params_ += ",";
         }
         {
            pp.k.any_pointer_params += "const ";
            pp.k.any_pointer_params += type;
            pp.k.any_pointer_params += " *";
            pp.k.any_pointer_params += name;
         }
         {
            pp.k.any_pointer_params_ += "const ";
            pp.k.any_pointer_params_ += type;
            pp.k.any_pointer_params_ += " *_";
            pp.k.any_pointer_params_ += name;
         }
      }
      if (! is_const && is_pointer){
         if (! pp.k.any_pointer_args.empty()) pp.k.any_pointer_args += ",";
         pp.k.any_pointer_args += name;
         if (! pp.k.any_pointer_params.empty()){
            pp.k.any_pointer_params += ",";
            pp.k.any_pointer_params_ += ",";
         }
         {
            pp.k.any_pointer_params += type;
            pp.k.any_pointer_params += " *";
            pp.k.any_pointer_params += name;
         }
         {
            pp.k.any_pointer_params_ += type;
            pp.k.any_pointer_params_ += " *_";
            pp.k.any_pointer_params_ += name;
         }
      }
   }
}

// *****************************************************************************
static inline void rtcKernelPrefix(const context &pp){     
   trk();
   pp.out << "\n\ttypedef void (*kernel_t)("<<pp.k.any_pointer_params<<");";
   pp.out << "\n\tstatic std::unordered_map<size_t,ok::okrtc<kernel_t>*> __kernels;";
   pp.out << "\n\t" << pp.k.u2d;
   pp.out << "\n\tconst char *src=R\"_(";
   pp.out << "#include <cstdint>";
   pp.out << "\n#include <cstring>";
   pp.out << "\n#include <stdbool.h>";
   pp.out << "\n#include \"general/okina.hpp\"";
   pp.out << "\ntypedef union {double d; uint64_t u;} union_du;";
   pp.out << "\ntemplate<" << pp.k.static_tmplt << ">";
   pp.out << "\nvoid rtc_" << pp.k.name << "(";
   pp.out << (pp.mm?pp.k.any_pointer_params_:pp.k.any_pointer_params) << "){";
   pp.out << "\n\t" << pp.k.d2u;
}

// *****************************************************************************
static inline void rtcKernelPostfix(context &pp){
   trk();
   pp.out << "\nextern \"C\" void k%016lx(" << pp.k.any_pointer_params << "){";
	pp.out << "\n\trtc_"<<pp.k.name
          <<"<" << pp.k.static_format<<">(" << pp.k.any_pointer_args << ");";
   pp.out << "\n})_\";";
   pp.out << "\n\tconst char *xcc = \"" << pp.k.xcc << "\";";
   pp.out << "\n\tconst size_t args_seed = std::hash<size_t>()(0);";
   pp.out << "\n\tconst size_t args_hash = ok::hash_args(args_seed,"
          << pp.k.static_args << ");";
   pp.out << "\n\tif (!__kernels[args_hash]){";
   pp.out << "\n\t\t__kernels[args_hash] = new ok::okrtc<kernel_t>"
          << "(xcc,src," << "\"-I" << pp.k.dirname << "\","
          << pp.k.static_args << ");";
   pp.out << "}\n\t(__kernels[args_hash]->operator_void("
          << pp.k.any_pointer_args << "));\n}";
   pp.block--;
   pp.k.jit = false;
}

// *****************************************************************************
/*static inline void goto_first_left_paren(context &pp) {
   for (; pp.in.peek()!=EOF; put(pp)) {
      const char c = pp.in.peek();
      check(pp,c != EOF);
      if (c == '\n') pp.line++;
      if (c == '(') return;
   }
}

// *****************************************************************************
static inline void goto_last_right_paren(context &pp) {
   for (; pp.in.peek()!=EOF; put(pp)) {
      const char c = pp.in.peek();
      check(pp,c != EOF);
      if (c == '\n') pp.line++;
      if (c == ')') return;
   }
}*/

// *****************************************************************************
static inline void __kernel(context &pp) {
   trk();
   //        "__kernel "
   pp.out << "         ";
   drop_space(pp);
   check(pp,isvoid(pp) || isstatic(pp),"Kernel w/o void or static");
   dbg("isstatic?");
   if (isstatic(pp)) {
      pp.out << get_name(pp);
      skip_space(pp);
   }
   dbg("void_return_type?");
   const string void_return_type = get_name(pp);
   pp.out << void_return_type;
   // Get kernel's name
   dbg("skip_space?");
   skip_space(pp);
   dbg("get_name?");
   const string name = get_name(pp);
   pp.out << name;
   pp.k.name = name;
   dbg("kernel:'%s'",name.c_str());
   dbg("skip_space?");
   skip_space(pp);
   if (! pp.mm) return;
   // check we are at the left parenthesis
   dbg("check?");
   check(pp,pp.in.peek()=='(',"No 1st '(' in kernel");
   put(pp); 
   // Go to first possible argument
   dbg("skip_space?");
   skip_space(pp);
   if (isvoid(pp)) { // if it is 'void' don't add any coma
      drop_name(pp);
   } else {
      pp.args.clear();
      dbg("get_args?");
      get_args(pp);
      if (pp.k.jit) rtcKernelRefresh(pp);
   }
}

// *****************************************************************************
// * '__' was hit, now fetch its 'id'
// *****************************************************************************
static inline void __id(context &pp, string id = "") {
   trk();
   if (id.empty()) id = get_name(pp);
   if (id=="__jit"){
      skip_space(pp);
      pp.k.jit = true;
      id = get_name(pp);
      check(pp,id=="__kernel","No 'kernel' keyword after 'jit' qualifier");
   }
   if (id=="__kernel"){
      // Get arguments of this kernel
      __kernel(pp);
      if (! pp.mm) return;
      check(pp,pp.in.peek()==')',"No last ')' in kernel");
      put(pp);
      skip_space(pp);
      // Make sure we are about to start a statement block
      check(pp,pp.in.peek()=='{',"No statement block found");
      put(pp);
      // Generate the RTC prefix for this kernel
      if (pp.k.jit) rtcKernelPrefix(pp);
      pp.block = 0;
      if (! pp.mm) return;
      // Generate the GET_* code
      for(std::list<argument*>::iterator ia = pp.args.begin();
          ia != pp.args.end() ; ia++) {
         const argument *a = *ia;
         const bool is_const = a->is_const;
         //const bool is_restrict = a->restrict;
         const bool is_pointer = a->star;
         const char *type = a->type.c_str();
         const char *name = a->name.c_str();
         if (is_const && ! is_pointer){
            if (!pp.k.jit){
               pp.out << "\n\tconst " << type << " " << name
                      << " = (const " << type << ")"
                      << " (_" << name << ");";
            }
         }
         if (is_const && is_pointer){
            pp.out << "\n\tconst " << type << "* " << name
                   << " = (const " << type << "*)"
                   << " mfem::mm::adrs(_" << name << ");";
         }
         if (! is_const && is_pointer){
            pp.out << "\n\t" << type << "* " << name
                   << " = (" << type << "*)"
                   << " mfem::mm::adrs(_" << name << ");";
         }
      }
      return;
   }
   pp.out << id;
}

// *****************************************************************************
static inline void sharpId(context &pp) {
   trk();
   string id = get_directive(pp);
   if (id=="#jit"){
      skip_space(pp);
      pp.k.jit = true;
      id = get_directive(pp);
      check(pp,id=="#kernel","No 'kernel' token found after the 'jit' one");
      __id(pp,"__kernel");
      return;
   }
   if (id=="#kernel"){
      __id(pp,"__kernel");
      return;
   }
   pp.out << id;
}

// *****************************************************************************
static inline int process(context &pp) {
   trk();
   char ch;
   pp.k.jit = false;
   if (pp.jit) pp.out << "#include \"../../general/okrtc.hpp\"\n";
   while (true){
      if (is_comment(pp)) comments(pp);
      if (peekn(pp,2) == "__") __id(pp);
      if (pp.in.peek() == '#') sharpId(pp);
      if (pp.block==-1) { if (pp.k.jit) rtcKernelPostfix(pp); }
      if (pp.block>=0 && pp.in.peek() == '{') { pp.block++; }
      if (pp.block>=0 && pp.in.peek() == '}') { pp.block--; }
      if (is_newline(pp.in.peek())) { pp.line++;}
      pp.in.get(ch);
      if (pp.in.eof()) break;
      pp.out << (char) ch;
      std::cout << (char) ch;
      fflush(0);
   }
   return 0;
}

// *****************************************************************************
int main(const int argc, char* argv[]) {
   trk();
   string input, output, file;   
   if (argc<=1) return help(argv);
   for (int i=1; i<argc; i+=1) {
      // -h lauches help
      if (argv[i] == string("-h"))
         return help(argv);
      // -o fills output
      if (argv[i] == string("-o")) {
         output = argv[i+1];
         i+=1;
         continue;
      }
      // should give input file
      const char* last_dot = strrnc(argv[i],'.');
      const size_t ext_size = last_dot?strlen(last_dot):0;
      if (last_dot && ext_size>0) {
         assert(file.size()==0);
         file = input = argv[i];
      }
   }
   assert(! input.empty());
   const bool output_file = ! output.empty();
   ifstream in(input.c_str(), ios::in | std::ios::binary);
   ofstream out(output.c_str(), ios::out | std::ios::binary | ios::trunc);
   assert(!in.fail());
   assert(in.is_open());
   if (output_file) {assert(out.is_open());}
   context pp(in,output_file?out:cout,file);
   try {
      process(pp);
   } catch (error err) {
      cerr << endl
	   << err.file << ":" << err.line << ":"
           << " mpp error"
           << (err.msg?": ":"")
           << (err.msg?err.msg:"")
           << endl;
      remove(output.c_str());
      return ~0;
   }
   in.close();
   out.close();
   return 0;
}
