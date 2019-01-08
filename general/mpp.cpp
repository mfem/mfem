#include <list>
#include <string>
#include <ciso646>
#include <cassert>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <string.h>
using namespace std;

// *****************************************************************************
// * STRUCTS: context, error & args
// *****************************************************************************
struct context {
   int line;
   string& file;
   istream& in;
   ostream& out;
public:
   context(istream& i, ostream& o, string &f): line(1), file(f), in(i), out(o){}
};

// *****************************************************************************
struct error {
   int line;
   string file;
   error(int l, string f): line(l), file(f) {}
};

// *****************************************************************************
struct argument {
  string type, name;
  bool star, is_const, restrict;
  argument(): star(false), is_const(false), restrict(false) {}
};

// *****************************************************************************
const char* strrnchr(const char* s, const unsigned char c, int n=1) {
   size_t len = strlen(s);
   char* p = (char*)s+len-1;
   for (; n; n--,p--,len--) {
      for (; len; p--,len--)
         if (*p==c) break;
      if (not len) return NULL;
      if (n==1) return p;
   }
   return NULL;
}

// *****************************************************************************
static inline void check(context &pp, const bool test){
   if (not test) throw error(pp.line,pp.file);
}

// *****************************************************************************
static inline int help(char* argv[]) {
   cout << "MFEM preprocessor:";
   cout << argv[0] << " -o output input" << endl;
   return ~0;
}

// *****************************************************************************
static inline bool is_newline(const char ch) {
   return static_cast<unsigned char>(ch) == '\n';
}

// *****************************************************************************
static inline char get(context &pp) {
   char ch;
   pp.in.get(ch);
   return ch;
}

// *****************************************************************************
static inline char put(context &pp) {
   const char c = get(pp);
   pp.out << c;
   return c;
}

// *****************************************************************************
static inline void put_space(context &pp) {
   while (isspace(pp.in.peek())) {
      if (pp.in.peek() == '\n') pp.line++;
      put(pp);
   }
}

// *****************************************************************************
static inline void drop_space(context &pp) {
   while (isspace(pp.in.peek())) {
      if (pp.in.peek() == '\n') pp.line++;
      pp.in.get();
   }
}

// *****************************************************************************
static inline bool is_comment(context &pp) {
   if (pp.in.peek() != '/') return false;
   pp.in.get();
   const char c = pp.in.peek();
   pp.in.unget();
   if (c == '/' or c == '*') return true;
   return false;
}

// *****************************************************************************
static inline void singleLineComment(context &pp) {
   while (pp.in.peek()!=EOF and pp.in.peek()!='\n') put(pp);
   pp.line++;
}

// *****************************************************************************
static inline void blockComment(context &pp) {
   while (not pp.in.eof()) {
      const char c = put(pp);
      if (c == '\n') pp.line++;
      if (c == '*' and pp.in.peek() == '/') {
         put(pp);
         put_space(pp);
         return;
      }
   }
}

// *****************************************************************************
static inline void comments(context &pp) {
   const char c1 = put(pp); check(pp,c1=='/');
   const char c2 = put(pp); check(pp,c2=='/' or c2=='*');
   if (c2 == '/') return singleLineComment(pp);
   return blockComment(pp);
}

// *****************************************************************************
static inline bool is_alpha(context &pp) {
   const int c = pp.in.peek();
   return isalpha(c) || c == '_';
}

// *****************************************************************************
static inline string get_name(context &pp) {
   string str;
   check(pp,is_alpha(pp));
   while ((not pp.in.eof()) and (pp.in.peek()!=EOF) and
          (isalnum(pp.in.peek()) or pp.in.peek()=='_'))
      str += pp.in.get();
   return str;
}

// *****************************************************************************
static inline string peekn(context &pp, const int n) {
   char c[n+1];
   for (int k=0;k<=n;k+=1) c[k] = 0;
   int k = 0;
   for (; k<n; k+=1) {
      if (pp.in.peek()==EOF) break;
      c[k] = pp.in.get();
   }
   string rtn = c;
   for (int l=0; l<k; l+=1) pp.in.unget();
   return rtn;
}

// *****************************************************************************
static inline string peekID(context &pp) {
   const int n = 128;
   char c[n] = {0};
   int k = 0;
   for (; k<n; k+=1) {
      const char p = pp.in.peek();
      if (p==EOF) break;
      if (not is_alpha(pp)) break;
      c[k]=pp.in.get();
   }
   string rtn = c;
   for (int l=0; l<k; l+=1) pp.in.unget();
   return rtn;
}

// *****************************************************************************
static inline void drop_name(context &pp) {
   while ((not pp.in.eof()) and (pp.in.peek()!=EOF) and
          (isalnum(pp.in.peek()) or pp.in.peek()=='_'))
      pp.in.get();
}

// *****************************************************************************
static inline bool isvoid(context &pp) {
   put_space(pp);
   const string void_peek = peekn(pp,4);
   if (void_peek == "void") return true;
   return false;
}

// *****************************************************************************
static inline bool is_star(context &pp) {
   put_space(pp);
   if (pp.in.peek() == '*') return true;
   return false;
}

// *****************************************************************************
static inline bool is_coma(context &pp) {
   put_space(pp);
   if (pp.in.peek() == ',') return true;
   return false;
}

// *****************************************************************************
static inline void goto_start_of_left_paren(context &pp) {
   for (; not pp.in.eof(); put(pp)) {
      const char c = pp.in.peek();
      check(pp,c != EOF);
      if (c == '\n') pp.line++;
      if (c == '(') return;
   }
}

// *****************************************************************************
static inline bool get_args(context &pp, std::list<argument*> &args) {
   bool empty = true;
   argument *arg = new argument();
   for (int p=0; not pp.in.eof(); empty=false) {
      if (is_star(pp)){
         arg->star = true;
         put(pp);
         continue;
      }
      if (is_coma(pp)){
         put(pp);
         continue;
      }
      const string id = peekID(pp);
      drop_name(pp);
      // char short unsigned long signed float
      if (id=="const") { pp.out << id; arg->is_const = true; continue; }
      if (id=="__restrict") { pp.out << id; arg->restrict = true; continue; }
      if (id=="int") { pp.out << id; arg->type = id; continue; }
      if (id=="bool") { pp.out << id; arg->type = id; continue; }
      if (id=="double") { pp.out << id; arg->type = id; continue; }
      if (id=="size_t") { pp.out << id; arg->type = id; continue; }
      pp.out << "_" << id;
      // focus on the name, we should have qual & type
      arg->name = id;
      args.push_back(arg);
      arg = new argument();
      const char c = pp.in.peek();
      check(pp,c != EOF);
      if (c == '(') p+=1;
      if (c == ')') p-=1;
      if (c == '\n') pp.line++;
      if (p<0) { return empty; }
      drop_space(pp);
      check(pp,pp.in.peek()==',');
      put(pp);
   }
   return empty;
}

// *****************************************************************************
static inline void __kernel(context &pp, std::list<argument*> &args) {
   //        "__kernel "
   pp.out << "         ";
   drop_space(pp);
   goto_start_of_left_paren(pp);
   // check we are at the left parenthesis
   check(pp,pp.in.peek()=='(');
   put(pp); // put '('
   // Go to first possible argument
   put_space(pp);
   if (isvoid(pp)) { // if it is 'void' don't add any coma
      drop_name(pp);
   } else {
      const bool empty = get_args(pp,args);
      check(pp,pp.in.peek()==')');
      if (not empty) pp.out << ", ";
   }
   // __kernel((CPU, GPU & JIT)) will add more options than the '0'
   pp.out << "const unsigned int __kernel =0";
}

// *****************************************************************************
// * '__' was hit, now fetch its 'id'
// *****************************************************************************
static inline void __id(context &pp) {
   const string id = get_name(pp);
   if (id=="__kernel"){      
      std::list<argument*> args;
      __kernel(pp,args);
      check(pp,pp.in.peek()==')');
      put(pp);
      put_space(pp);
      check(pp,pp.in.peek()=='{');
      put(pp);
      for(std::list<argument*>::iterator ia = args.begin();
          ia != args.end() ; ia++) {
         const argument *a = *ia;
         const bool is_const = a->is_const;
         //const bool is_restrict = a->restrict;
         const bool is_pointer = a->star;
         const char *type = a->type.c_str();
         const char *name = a->name.c_str();
         if (is_const and not is_pointer){
            pp.out << "\n   // Could JIT " << name;
         }
         if (is_const and is_pointer){
            pp.out << "\n   GET_CONST_ADRS_T_("<<name<<", "<<type<<");";
         }
         if (not is_const and is_pointer){
            pp.out << "\n   GET_ADRS_T_("<<name<<", "<<type<<");";
         }/*
            dbg("%s%s %s%s%s",
            is_const?"const ":"",
            type,
            is_restrict?" __restrict ":"",
            is_pointer?"*":"",
            name);*/
      }
      return;
   }
   pp.out << id;
}

// *****************************************************************************
static inline int process(context &pp) {
   while (not pp.in.eof()) {
      if (is_comment(pp)) comments(pp);
      if (pp.in.peek() != EOF) put(pp);
      if (peekn(pp,2) == "__") __id(pp);
      if (is_newline(pp.in.peek())) { pp.line++;}
   }
   return 0;
}

// *****************************************************************************
int main(const int argc, char* argv[]) {
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
      const char* last_dot = strrnchr(argv[i],'.');
      const size_t ext_size = last_dot?strlen(last_dot):0;
      if (last_dot && ext_size>0) {
         assert(file.size()==0);
         file = input = argv[i];
      }
   }
   assert(not input.empty());
   const bool output_file = not output.empty();
   ifstream in(input.c_str(), ios::in);
   ofstream out(output.c_str(),ios::out);
   assert(in.is_open());
   if (output_file) {assert(out.is_open());}
   context pp(in,output_file?out:cout,file);
   try {
      process(pp);
   } catch (error err) {
      cerr << err.file << ":" << err.line << ":"
           << " parser error" << endl;
      unlink(output.c_str());
      exit(-1);
   }
   return 0;
}
