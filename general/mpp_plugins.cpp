#if 0

// *****************************************************************************
void genPtrOkina(context &pp)
{
   // Generate the GET_* code
   for (argument_it ia = pp.args.begin();
        ia != pp.args.end() ; ia++)
   {
      const argument a = *ia;
      const bool is_const = a.is_const;
      //const bool is_restrict = a.restrict;
      const bool is_pointer = a.is_ptr;
      const char *type = a.type.c_str();
      const char *name = a.name.c_str();
      if (is_const && ! is_pointer)
      {
         pp.out << "\n\tconst " << type << " " << name
                << " = (" << type << ")"
                << " (_" << name << ");";

      }
      if (is_const && is_pointer)
      {
         pp.out << "\n\tconst " << type << "* " << name
                << " = (" << type << "*)"
                << " mfem::mm::ptr(_" << name << ");";
      }
      if (!is_const && is_pointer)
      {
         pp.out << "\n\t" << type << "* " << name
                << " = (" << type << "*)"
                << " mfem::mm::ptr(_" << name << ");";
      }
   }
}

// *****************************************************************************
// * Template generation
// *****************************************************************************
void __range(context &pp, argument &arg)
{
   char c;
   bool dash = false;
   // Verify and eat '('
   check(pp,get(pp)=='(',"__template should declare its range");
   do
   {
      const int n = get_digit(pp);
      if (dash)
      {
         for (int i=arg.range.back()+1; i<n; i++)
         {
            arg.range.push_back(i);
         }
      }
      dash = false;
      arg.range.push_back(n);
      c = get(pp);
      assert(not pp.in.eof());
      check(pp, c==',' or c=='-' or  c==')', "Unknown __template range");
      if (c=='-')
      {
         dash = true;
      }
   }
   while (c!=')');
}

// *****************************************************************************
void get_targs(context &pp)
{
   int nargs = 0;
   int targs = 0;
   argument arg;
   pp.args.clear();
   // Go to first possible argument
   drop_space(pp);
   if (isvoid(pp)) { assert(false); }
   string current_arg;
   for (int p=0; true;)
   {
      skip_space(pp,current_arg);
      comments(pp);
      if (is_star(pp))
      {
         arg.is_ptr = true;
         current_arg += get(pp);
         continue;
      }
      skip_space(pp,current_arg);
      comments(pp);
      if (is_coma(pp))
      {
         current_arg += get(pp);
         continue;
      }
      const string &id = peekID(pp);
      drop_name(pp);
      // Qualifiers
      if (id=="__range") { __range(pp,arg); arg.is_tpl = true; continue; }
      if (id=="const") { current_arg += id; arg.is_const = true; continue; }
      if (id=="__restrict") { current_arg += id; arg.is_restrict = true; continue; }
      // Types
      if (id=="char") { current_arg += id; arg.type = id; continue; }
      if (id=="int") { current_arg += id; arg.type = id; continue; }
      if (id=="short") { current_arg += id; arg.type = id; continue; }
      if (id=="unsigned") { current_arg += id; arg.type = id; continue; }
      if (id=="long") { current_arg += id; arg.type = id; continue; }
      if (id=="bool") { current_arg += id; arg.type = id; continue; }
      if (id=="float") { current_arg += id; arg.type = id; continue; }
      if (id=="double") { current_arg += id; arg.type = id; continue; }
      if (id=="size_t") { current_arg += id; arg.type = id; continue; }
      // focus on the name, we should have qual & type
      arg.name = id;
      if (not arg.is_tpl)
      {
         pp.args.push_back(arg);
         pp.ker.tpl.signature += current_arg + (not pp.mm?"":"_") + id;
         {
            pp.ker.tpl.std_args += (nargs==0)?"":", ";
            pp.ker.tpl.std_args +=  arg.name;
         }
         nargs += 1;
      }
      else
      {
         pp.ker.tpl.template_parameters += (targs==0)?"":", ";
         pp.ker.tpl.template_parameters += "const " + arg.type + " " + arg.name;
         pp.ker.tpl.ranges.push_back(arg.range);
         //pp.ker.tpl.template_parameters += "/*";
         //for (int n : arg.range) { pp.ker.tpl.template_parameters += std::to_string(n) + " "; }
         //pp.ker.tpl.template_parameters += "*/";
         {
            pp.ker.tpl.template_args += (targs==0)?"":", ";
            pp.ker.tpl.template_args += arg.name;
         }
         targs += 1;
      }
      pp.ker.tpl.std_parameters += current_arg + id + (nargs==0 and targs>0?",":"");
      arg = argument();
      current_arg = string();
      const int c = pp.in.peek();
      assert(not pp.in.eof());
      if (c == '(') { p+=1; }
      if (c == ')') { p-=1; }
      if (p<0) { break; }
      skip_space(pp,current_arg);
      comments(pp);
      check(pp,pp.in.peek()==',',"<__template> No coma while in args");
      get(pp);
      if (nargs>0) { current_arg += ","; }
   }
}

// *****************************************************************************
void __template(context &pp)
{
   pp.ker.T = true;
   pp.ker.tpl = tpl_t();
   drop_space(pp);
   comments(pp);
   string id = get_id(pp);
   check(pp,id=="__kernel","No 'kernel' keyword after 'template' qualifier");
   check(pp,isvoid(pp) or isstatic(pp),"Templated kernel w/o void or static");
   if (isstatic(pp))
   {
      pp.ker.tpl.return_type += get_id(pp);
      skip_space(pp,pp.ker.tpl.return_type);
   }
   const string void_return_type = get_id(pp);
   pp.ker.tpl.return_type += void_return_type;
   // Get kernel's name
   skip_space(pp,pp.ker.tpl.return_type);
   const string name = get_id(pp);
   pp.ker.name = name;
   skip_space(pp, pp.ker.tpl.return_type);
   // check we are at the left parenthesis
   check(pp,pp.in.peek()=='(',"No 1st '(' in kernel");
   get(pp);
   // Get the arguments
   get_targs(pp);
   // Make sure we have hit the last ')' of the arguments
   check(pp,pp.in.peek()==')',"No last ')' in kernel");
   pp.ker.tpl.signature += get(pp);
   // Now dump the templated kernel needs before the body
   pp.out << "template<";
   pp.out << pp.ker.tpl.template_parameters;
   pp.out << ">\n";
   pp.out << pp.ker.tpl.return_type;
   pp.out << "__" << pp.ker.name;
   pp.out << "(" << pp.ker.tpl.signature;
   // Std body dump to pp.out
   skip_space(pp);
   // Make sure we are about to start a compound statement
   check(pp,pp.in.peek()=='{',"<>No compound statement found");
   put(pp);
   // If we are using the memory manager, generate the calls
   if (pp.mm) { genPtrOkina(pp); }
   // Starts counting the compound statements
   pp.compound_statements = 0;
}

// *****************************************************************************
static list<list<int> > outer_product(const list<list<int> > &v)
{
   list<list<int> > s = {{}};
   for (const auto &u : v)
   {
      list<list<int> > r;
      for (const auto &x:s)
      {
         for (const auto y:u)
         {
            r.push_back(x);
            r.back().push_back(y);
         }
      }
      s = std::move(r);
   }
   return s;
}*

// *****************************************************************************
void tplPostfix(context &pp)
{
   if (not pp.ker.T) { return; }
   if (pp.compound_statements>=0 && pp.in.peek() == '{') { pp.compound_statements++; }
   if (pp.compound_statements>=0 && pp.in.peek() == '}') { pp.compound_statements--; }
   if (pp.compound_statements!=-1) { return; }
   check(pp,pp.in.peek()=='}',"<>No compound statements found");
   put(pp);
   // Stop counting the compound statements and flush the T status
   pp.compound_statements--;
   pp.ker.T = false;
   // Now push template kernel launcher
   pp.out << "\n// *****************************************************************************\n";
   pp.out << pp.ker.tpl.return_type;
   pp.out << pp.ker.name;
   pp.out << "(" << pp.ker.tpl.std_parameters;
   pp.out << "){";
   pp.out << "\n\ttypedef ";
   pp.out << pp.ker.tpl.return_type << "(*__T" << pp.ker.name << ")";
   pp.out << "(" << pp.ker.tpl.signature << ";";
   pp.out << "\n\tconst size_t id = hash_args(std::hash<size_t>()(0), "
          << pp.ker.tpl.template_args
          << ");";
   pp.out << "\n\tstatic std::unordered_map<size_t, "
          << "__T" << pp.ker.name << "> call = {";
   for (list<int> range : outer_product(pp.ker.tpl.ranges))
   {
      pp.out << "\n\t\t{";
      int i=1;
      const int n = range.size();
      size_t hash = 0;
      for (int r : range) { hash = hash_args(hash,r); }
      pp.out << std::hex << "0x" << hash ;// <<"ul";
      pp.out << ",&__"<<pp.ker.name<<"<";
      for (int r : range)
      {
         pp.out << to_string(r) << (i==n?"":",");
         i+=1;
      }
      pp.out << ">},";
   }
   pp.out << "\n\t};";
   pp.out << "\n\tassert(call[id]);";
   pp.out << "\n\tcall[id](";
   pp.out << pp.ker.tpl.std_args;
   pp.out << ");";
   pp.out << "\n}";
}

// *****************************************************************************
void __embed(context &pp)
{
   // Skip "__embed"
   pp.out << "       ";
   pp.ker.embedding = true;
   // Goto first '{'
   while ('{' != put(pp));
   // Starts counting the compound statements
   pp.compound_statements = 0;
}
// *****************************************************************************
void embedPostfix(context &pp)
{
   if (not pp.ker.embedding) { return; }
   if (pp.compound_statements>=0 && pp.in.peek() == '{') { pp.compound_statements++; }
   if (pp.compound_statements>=0 && pp.in.peek() == '}') { pp.compound_statements--; }
   if (pp.compound_statements!=-1) { return; }
   check(pp,pp.in.peek()=='}',"<>No compound statements found");
   put(pp);
   pp.compound_statements--;
   pp.ker.embedding = false;
   pp.ker.embed += "\n";
}
#endif
