double elliptic_ea ( double a );
void elliptic_ea_values ( int &n_data, double &x, double &fx );
double elliptic_ek ( double k );
void elliptic_ek_values ( int &n_data, double &x, double &fx );
double elliptic_em ( double m );
void elliptic_em_values ( int &n_data, double &x, double &fx );
double elliptic_fa ( double a );
void elliptic_fa_values ( int &n_data, double &x, double &fx );
double elliptic_fk ( double k );
void elliptic_fk_values ( int &n_data, double &x, double &fx );
double elliptic_fm ( double m );
void elliptic_fm_values ( int &n_data, double &x, double &fx );
double elliptic_pia ( double n, double a );
double elliptic_inc_ea ( double phi, double a );
void elliptic_inc_ea_values ( int &n_data, double &phi, double &a, double &ea );
double elliptic_inc_ek ( double phi, double k );
void elliptic_inc_ek_values ( int &n_data, double &phi, double &k, double &ek );
double elliptic_inc_em ( double phi, double m );
void elliptic_inc_em_values ( int &n_data, double &phi, double &m, double &em );
double elliptic_inc_fa ( double phi, double a );
void elliptic_inc_fa_values ( int &n_data, double &phi, double &a, double &fa );
double elliptic_inc_fk ( double phi, double k );
void elliptic_inc_fk_values ( int &n_data, double &phi, double &k, double &fk );
double elliptic_inc_fm ( double phi, double m );
void elliptic_inc_fm_values ( int &n_data, double &phi, double &m, double &fm );
double elliptic_inc_pia ( double phi, double n, double a );
void elliptic_inc_pia_values ( int &n_data, double &phi, double &n, double &a, 
  double &pia );
double elliptic_inc_pik ( double phi, double n, double k );
void elliptic_inc_pik_values ( int &n_data, double &phi, double &n, double &k, 
  double &pik );
double elliptic_inc_pim ( double phi, double n, double m );
void elliptic_inc_pim_values ( int &n_data, double &phi, double &n, double &m, 
  double &pim );
void elliptic_pia_values ( int &n_data, double &n, double &a, double &pia );
double elliptic_pik ( double n, double k );
void elliptic_pik_values ( int &n_data, double &n, double &k, double &pik );
double elliptic_pim ( double n, double m );
void elliptic_pim_values ( int &n_data, double &n, double &m, double &pim );
double jacobi_cn ( double u, double m );
void jacobi_cn_values ( int &n_data, double &u, double &a, double &k,
  double &m, double &fx );
double jacobi_dn ( double u, double m );
void jacobi_dn_values ( int &n_data, double &u, double &a, double &k,
  double &m, double &fx );
double jacobi_sn ( double u, double m );
void jacobi_sn_values ( int &n_data, double &u, double &a, double &k,
  double &m, double &fx );
double rc ( double x, double y, double errtol, int &ierr );
double rd ( double x, double y, double z, double errtol, int &ierr );
double rf ( double x, double y, double z, double errtol, int &ierr );
double rj ( double x, double y, double z, double p, double errtol, int &ierr );
void sncndn ( double u, double m, double &sn, double &cn, double &dn );
void timestamp ( );

