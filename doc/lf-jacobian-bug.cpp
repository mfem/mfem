// Dial the nonlinearity and watch two things together: does GetGradient stay the
// derivative of Mult, and does Newton converge? If the Jacobian error tracks the
// breakdown, that is the cause; if it does not, it is something else.
#define BOOST_TEST_MODULE Dial
#include <boost/test/unit_test.hpp>
#include "ConvergenceHarness.hpp"
#include "meq/GradShafranov.hpp"
#include "meq/Source.hpp"
#include <cstdio>
#include <vector>
#include <cmath>

namespace
{
	meq::tests::Rectangle box() { return { 0.6, 1.4, -0.6, 0.6 }; }
	using O = meq::GradShafranovSolver::NonlinearOrdering;

	/// F = c psi^2, so dF/dpsi = 2 c psi. One knob, and it vanishes at psi = 0,
	/// so the datum has to be non-homogeneous -- as it is for the GS-2 sources.
	class Dial : public meq::Source
	{
		public:
			explicit Dial( double cIn ) : c( cIn ) {}
			double f( double, double, double psi ) const override
			{ return c*psi*psi; }
			double dFdPsi( double, double, double psi ) const override
			{ return 2.0*c*psi; }
		private:
			double c;
	};

	double datum( double, double z ) { return 0.3*z/box().zMax; }

	struct Result { double jacRel; bool converged; int iters; };

	Result run( double c, O ordering, double hStep = 1.0e-5 )
	{
		Dial source( c );
		mfem::Mesh mesh = meq::tests::makeMesh( box(), 8 );
		mfem::FunctionCoefficient d( []( mfem::Vector const &x )
			{ return datum( x( 0 ), x( 1 ) ); } );

		Result out{ -1.0, false, -1 };

		{   // Jacobian against a central difference, essential rows masked.
			meq::GradShafranovSolver s( mesh, 1 );
			s.setSource( source );
			s.setBoundaryData( d );
			s.setNonlinearOrdering( ordering );
			s.prepare();

			mfem::Operator &R = s.reducedOperator();
			int const n = R.Height();
			std::vector<bool> ess( n, false );
			mfem::Array<int> const &e = s.essentialTraceDofs();
			for ( int i = 0; i < e.Size(); ++i ) ess[ e[ i ] ] = true;

			mfem::Vector x( n ), v( n );
			x.Randomize( 3 ); x *= 0.05;
			v.Randomize( 7 );
			for ( int i = 0; i < n; ++i ) if ( ess[ i ] ) { x( i ) = 0.0; v( i ) = 0.0; }
			v *= 1.0/v.Norml2();

			double const h = hStep;
			mfem::Vector xp( x ), xm( x ), rp( n ), rm( n ), Jv( n );
			xp.Add( h, v ); xm.Add( -h, v );
			R.Mult( xp, rp ); R.Mult( xm, rm );
			mfem::Vector fd( rp ); fd -= rm; fd *= 1.0/( 2.0*h );
			R.GetGradient( x ).Mult( v, Jv );

			double num = 0.0, den = 0.0;
			for ( int i = 0; i < n; ++i )
				if ( !ess[ i ] )
				{ double const dd = Jv( i ) - fd( i ); num += dd*dd; den += fd( i )*fd( i ); }
			out.jacRel = std::sqrt( num )/std::max( 1.0e-300, std::sqrt( den ) );
		}

		{   // and an actual solve
			meq::GradShafranovSolver s( mesh, 1 );
			s.setSource( source );
			s.setBoundaryData( d );
			s.setNonlinearOrdering( ordering );
			s.setNewtonControl( 1.0e-10, 1.0e-14, 40 );
			try { s.solve(); out.converged = true; }
			catch ( std::exception const & ) { out.converged = false; }
			out.iters = s.newtonIterations();
		}
		return out;
	}
}

BOOST_AUTO_TEST_CASE( dial )
{
	std::printf( "\n  F = c psi^2 on the potential mass, k=1, 8x8\n" );
	std::printf( "  %-8s | %-34s | %-34s\n", "c",
	             "condense-then-linearise", "LINEARISE-then-condense" );
	std::printf( "\n  h-independence at c = 100 (a real Jacobian error does not move with h)\n" );
	for ( double h : { 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7 } )
		std::printf( "    h=%.0e : condense %.3e   LINEARISE %.3e\n", h,
		             run( 100.0, O::CondenseThenLinearise, h ).jacRel,
		             run( 100.0, O::LineariseThenCondense, h ).jacRel );
	std::printf( "\n" );
	for ( double c : { 1.0, 10.0, 100.0, 1000.0, 1.0e4 } )
	{
		Result const a = run( c, O::CondenseThenLinearise );
		Result const b = run( c, O::LineariseThenCondense );
		std::printf( "  %-8g | dJ %.2e  %s in %2d | dJ %.2e  %s in %2d\n", c,
		             a.jacRel, a.converged ? "ok  " : "FAIL", a.iters,
		             b.jacRel, b.converged ? "ok  " : "FAIL", b.iters );
		std::fflush( stdout );
	}
}
