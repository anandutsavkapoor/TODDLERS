/* This file is part of Cloudy and is copyright (C)1978-2025 by Gary J. Ferland and
 * others.  For conditions of distribution and use see copyright notice in license.txt */
/*RT_continuum attenuation of diffuse and beamed continua */
/*pnegopc save negative opacities on io unit, iff 'set negopc' command was given */
#include "cddefines.h"
#include "rt.h"
#include "rfield.h"
#include "opacity.h"
#include "dense.h"
#include "geometry.h"
#include "trace.h"
#include "radius.h"
#include "iso.h"
#include "hextra.h"
#include "mole.h"
#include "freebound.h"
#include "cosmology.h"
#include "vectorize.h"
#include "grainvar.h"

/*cmshft - so Compton scattering shift of spectrum 
 * this code is a placeholder */
STATIC void cmshft(void)
{
	DEBUG_ENTRY( "cmshft()" );

	/* first check whether Compton scattering is in as heat/cool */
	if( !rfield.lgComptonOn )
	{ 
		return;
	}

	if( rfield.lgComptonOn )
	{ 
		return;
	}

	/* do reshuffle */
	for( long i=0; i < rfield.nflux; i++ )
	{
		continue;
	}
	return;
}


#if !defined(NDEBUG)
/*pnegopc save negative opacities on io unit, iff 'set negopc' command was given */
STATIC void pnegopc(void)
{
	FILE *ioFile;

	DEBUG_ENTRY( "pnegopc()" );

	if( opac.lgNegOpacIO )
	{
		/* option to save negative opacities */
		ioFile = open_data( "negopc.txt", "w" );
		for( long i=0; i < rfield.nflux; i++ )
		{
			fprintf( ioFile, "%10.2e %10.2e \n", rfield.anu(i), 
				opac.opacity_abs[i] );
		}
		fclose( ioFile);
	}
	return;
}
#endif

/*RT_continuum attenuation of diffuse and beamed continua */
void RT_continuum(void)
{
	DEBUG_ENTRY( "RT_continuum()" );

	if( trace.lgTrace && trace.lgConBug )
	{
		fprintf( ioQQQ, " Energy, flux, OTS:\n" );
		for( long i=0; i < rfield.nflux; i++ )
		{
			fprintf( ioQQQ, "%6ld%10.2e%10.2e%10.2e", i, rfield.anu(i), 
				rfield.flux[0][i] + rfield.outlin[0][i] + rfield.ConInterOut[i], 
				rfield.otscon[i] + rfield.otslin[i] + rfield.outlin_noplot[i]);
		}
		fprintf( ioQQQ, "\n" );
	}

	/* begin sanity check in debug mode */
#	if !defined(NDEBUG)
	bool lgFlxNeg = false;
	for( long i=0; i < rfield.nflux; i++ )
	{
		if( rfield.flux[0][i] < 0. )
		{
			fprintf( ioQQQ, " radius_increment finds negative intensity in flux.\n" );
			fprintf( ioQQQ, " Intensity, frequency, pointer=%11.3e%11.3e%6ld\n", 
				rfield.flux[0][i], rfield.anu(i), i );
			lgFlxNeg = true;
		}
		if( rfield.otscon[i] < 0. )
		{
			fprintf( ioQQQ, " radius_increment finds negative intensity in otscon.\n" );
			fprintf( ioQQQ, " Intensity, frequency, pointer=%11.3e%11.3e%6ld\n", 
				rfield.otscon[i], rfield.anu(i), i );
			lgFlxNeg = true;
		}
		if( opac.tmn[i] < 0. )
		{
			fprintf( ioQQQ, " radius_increment finds negative tmn.\n" );
			fprintf( ioQQQ, " value, frequency, pointer=%11.3e%11.3e%6ld %4.4s\n", 
				opac.tmn[i], rfield.anu(i), i, rfield.chLineLabel[i].c_str() );
			lgFlxNeg = true;
		}
		if( rfield.otslin[i] < 0. )
		{
			fprintf( ioQQQ, " radius_increment finds negative intensity in otslin.\n" );
			fprintf( ioQQQ, " Intensity, frequency, pointer=%11.3e%11.3e%6ld %4.4s\n", 
				rfield.otslin[i], rfield.anu(i), i, rfield.chLineLabel[i].c_str()  );
			lgFlxNeg = true;
		}
		if( rfield.outlin[0][i] < 0. )
		{
			fprintf( ioQQQ, " radius_increment finds negative intensity in outlin.\n" );
			fprintf( ioQQQ, " Intensity, frequency, pointer=%11.3e%11.3e%6ld %4.4s\n", 
				rfield.outlin[0][i], rfield.anu(i), i, rfield.chLineLabel[i].c_str()  );
			lgFlxNeg = true;
		}
		if( rfield.ConInterOut[i] < 0. )
		{
			fprintf( ioQQQ, " radius_increment finds negative intensity in ConInterOut.\n" );
			fprintf( ioQQQ, " Intensity, frequency, pointer=%11.3e%11.3e%6ld %4.4s\n", 
				rfield.ConInterOut[i], rfield.anu(i), i, rfield.chContLabel[i].c_str()  );
			lgFlxNeg = true;
		}
		if( opac.opacity_abs[i] < 0. )
		{
			opac.lgOpacNeg = true;
			/* this sub will save negative opacities on io unit,
			 * iff 'set negopc' command was given */
			pnegopc();
		}
	}
	if( lgFlxNeg )
	{
		fprintf( ioQQQ, " Insanity has occurred, this is zone%4ld\n", 
			nzone );
		ShowMe();
		cdEXIT(EXIT_FAILURE);
	}
	/*end sanity check*/
#	endif

	// ratio of inner to outer radii, at this point
	// radius is the outer radius of this zone 
	double DilutionHere = POW2((radius.Radius - radius.drad*radius.dRadSign)/
		radius.Radius);

	rfield.EnergyIncidCont = 0.;
	rfield.EnergyDiffCont = 0.;

	// attenuation of flux by optical depths IN THIS ZONE 
	// DirectionalCosin is 1/COS(theta), is usually 1, reset with illumination command,
	// option for illumination of slab at an angle 
	double fac = radius.drad_x_fillfac*geometry.DirectionalCosin;
	for( long i=0; i < rfield.nflux; i++ )
		rfield.vexp_arg[i] = -opac.opacity_abs[i]*fac;
	vexp( rfield.vexp_arg.data(), opac.ExpZone.data(), 0, rfield.nflux );
	// this loop should not be to <= nflux since highest cell is for
	// continuum unit integration
	// scattering opacities are included in energy exchange here in the
	// sphere case, since photons diffuse out of the closed sphere.
	// scattering opacities are not included as extinction sources in the
	// sphere case
	for( long i=0; i < rfield.nflux; i++ )
	{
		if( cosmology.lgDo )
		{
			opac.TauAbsGeo[0][i] = 0.;
			opac.TauScatGeo[0][i] = 0.;
			opac.TauAbsFace[i] = 0.;
			opac.TauScatFace[i] = 0.;
		}

		double dTau_abs = opac.opacity_abs[i]*radius.drad_x_fillfac;
		double dTau_sct = opac.opacity_sct[i]*radius.drad_x_fillfac;

		// sum total continuous optical depths
		opac.TauAbsGeo[0][i] += (realnum)(dTau_abs);
		opac.TauScatGeo[0][i] += (realnum)(dTau_sct);

		// following only optical depth to illuminated face
		opac.TauAbsFace[i] += (realnum)(dTau_abs);
		opac.TauScatFace[i] += (realnum)(dTau_sct);

		// these are total in inward direction, large if spherical
		opac.TauTotalGeo[0][i] = opac.TauAbsGeo[0][i] + opac.TauScatGeo[0][i];

		// e(-tau) in inward direction, up to illuminated face
		opac.ExpmTau[i] *= (realnum)opac.ExpZone[i];

		// e2(tau) in inward direction, up to illuminated face 
		opac.E2TauAbsFace[i] = (realnum)e2(opac.TauAbsFace[i]);
		ASSERT( opac.E2TauAbsFace[i] <= 1. && opac.E2TauAbsFace[i] >= 0. );

		// on second and later iterations define outward E2
		if( iteration > 1 )
		{
			// e2 from current position to outer edge of shell 
			realnum tau = MAX2(SMALLFLOAT , opac.TauAbsTotal[i] - opac.TauAbsFace[i] );
			opac.E2TauAbsOut[i] = (realnum)e2( tau );
			ASSERT( opac.E2TauAbsOut[i]>=0. && opac.E2TauAbsOut[i]<=1. );
		}

		// DilutionHere is square of ratio of inner to outer radius
		double AttenuationDilutionFactor = opac.ExpZone[i]*DilutionHere;
		ASSERT( AttenuationDilutionFactor <= 1.0 );

		// continuum has three parts 
		rfield.flux_beam_const[i] *= (realnum)AttenuationDilutionFactor;
		rfield.flux_beam_time[i] *= (realnum)AttenuationDilutionFactor;
		rfield.flux_isotropic[i] *= (realnum)AttenuationDilutionFactor;
		rfield.flux[0][i] = rfield.flux_beam_const[i] + rfield.flux_beam_time[i] +
			rfield.flux_isotropic[i];

		// update SummedCon here since flux changed
		rfield.SummedCon[i] = rfield.flux[0][i] + rfield.SummedDif[i];
		rfield.SummedOcc[i] = rfield.SummedCon[i]*rfield.convoc[i];

		// outward lines and diffuse continua 
		rfield.outlin[0][i] *= (realnum)AttenuationDilutionFactor;
		rfield.outlin_noplot[i] *= (realnum)AttenuationDilutionFactor;

		// interactive outward diffuse continuum
		// This fixit() originally was a call to TestCode() and was merged from the rt branch.
		// The rt branch was subsequently modified, apparantly to fix the problem mentioned below.
		// These changes on the rt branch seem to have never been merged to the trunk.
		// See also the mail on cloudy-dev "!Test code is in place." dated 2016-02-10 09:34 UTC
		fixit("moved from rt_diffuse; this preserves the original order of the next 2 lines and is incorrect");
		rfield.ConInterOut[i] += rfield.DiffuseEscape[i]*(realnum)radius.dVolOutwrd;
		rfield.ConInterOut[i] *= (realnum)AttenuationDilutionFactor;

		// this is not the interacting continuum
		rfield.ConEmitOut[0][i] *= (realnum)AttenuationDilutionFactor;
		rfield.ConEmitOut[0][i] += rfield.ConEmitLocal[nzone][i]*(realnum)radius.dVolOutwrd*opac.tmn[i]/**/;

		// gas-phase nebular continuum with DUST extinction removed (SKIRT applies dust
		// downstream) but GAS extinction RETAINED. Gas continuous opacity matters here -- in
		// particular the on-the-spot reabsorption of the Lyman continuum -- and removing it
		// would resurrect an unphysical LyC bump, so we keep it (the gas within-zone escape
		// opac.tmn is also kept). opac.ExpZone = sexp(opacity_abs*dr/2) carries gas+dust;
		// rescale it by the gas fraction of the absorption opacity to get the gas-only
		// inter-zone transmission exp(-(opacity_abs - dust)*dr/2). Subtract grain thermal
		// emission so the source term is gas-only. This is the emergent ConEmitOut above with
		// only the dust contribution (extinction + thermal emission) taken out.
		double dustOpacAbs = ( gv.lgDustOn() && gv.lgGrainPhysicsOn ) ?
			gv.dstab[i]*dense.gas_phase[ipHYDROGEN] : 0.;
		double ExpZoneGas = opac.ExpZone[i];
		if( dustOpacAbs > 0. && opac.opacity_abs[i] > dustOpacAbs )
			ExpZoneGas = pow( (double)opac.ExpZone[i],
				(opac.opacity_abs[i] - dustOpacAbs)/opac.opacity_abs[i] );
		rfield.ConEmitOutUnatt[0][i] *= (realnum)(ExpZoneGas*DilutionHere);
		realnum gasEmit = rfield.ConEmitLocal[nzone][i];
		if( gv.lgDustOn() && gv.lgGrainPhysicsOn )
			gasEmit -= gv.GrainEmission[i];
		rfield.ConEmitOutUnatt[0][i] += gasEmit*(realnum)radius.dVolOutwrd*opac.tmn[i];

		// set occupation numbers, first attenuated incident continuum 
		rfield.OccNumbIncidCont[i] = rfield.flux[0][i]*rfield.convoc[i];

		// local diffuse continua 
		rfield.OccNumbDiffCont[i] = rfield.ConEmitLocal[nzone][i]*rfield.convoc[i];

		// outward diffuse continuum 
		rfield.OccNumbContEmitOut[i] = rfield.ConEmitOut[0][i]*rfield.convoc[i];

		// integrated energy flux, ergs s^-1 cm^-2 
		rfield.EnergyIncidCont += rfield.flux[0][i]*rfield.anu(i);
		rfield.EnergyDiffCont += (rfield.outlin[0][i] + rfield.outlin_noplot[i] +
			rfield.ConInterOut[i])* rfield.anu(i);
	}

	rfield.setTrimming();

	// convert Ryd to erg 
	rfield.EnergyIncidCont *= (realnum)EN1RYD;
	rfield.EnergyDiffCont *= (realnum)EN1RYD;

	// sanity check, compare total Lyman continuum optical depth 
	// with amount of extinction there
	// this is amount continuum attenuated to illuminated face, 
	// but only do test if flux positive, not counting scattering opacity,
	// and correction for spherical dilution not important 
	if( rfield.flux[0][iso_sp[ipH_LIKE][ipHYDROGEN].fb[0].ipIsoLevNIonCon-1]>SMALLFLOAT &&
		(rfield.flux[0][iso_sp[ipH_LIKE][ipHYDROGEN].fb[0].ipIsoLevNIonCon-1]/
		SDIV(rfield.flux_total_incident[0][iso_sp[ipH_LIKE][ipHYDROGEN].fb[0].ipIsoLevNIonCon-1]) ) > SMALLFLOAT && 
		!opac.lgScatON &&
		radius.depth/radius.Radius < 1e-4 )
	{
		// ratio of current to incident continuum, converted to optical depth
		/* >>chng 99 apr 23, this crashed on alpha due to underflow to zero in argy
		* to log.  defended two ways - above checks that ratio of fluxes is large enough, 
		* and here convert to double.
		* error found by Peter van Hoof */
		double tau_effec = -log((double)rfield.flux[0][iso_sp[ipH_LIKE][ipHYDROGEN].fb[0].ipIsoLevNIonCon-1]/
			(double)opac.tmn[iso_sp[ipH_LIKE][ipHYDROGEN].fb[0].ipIsoLevNIonCon-1]/
			(double)rfield.flux_total_incident[0][iso_sp[ipH_LIKE][ipHYDROGEN].fb[0].ipIsoLevNIonCon-1]);

		// this is computed absorption optical depth to illuminated face
		double tau_true = opac.TauAbsFace[iso_sp[ipH_LIKE][ipHYDROGEN].fb[0].ipIsoLevNIonCon-1]*geometry.DirectionalCosin;

		// first test is relative error, second is to absolute error and comes
		// in for very small tau, where differences are in the round-off 
		if( fabs( tau_effec - tau_true ) / MAX2(tau_effec , tau_true) > 0.01 && 
			// for very small inner optical depths, the tmn correction is major,
			// and this test is not precise
			fabs(tau_effec-tau_true)>MAX2(0.001,1.-opac.tmn[iso_sp[ipH_LIKE][ipHYDROGEN].fb[0].ipIsoLevNIonCon-1]) )
		{
			// in print below must add extra HI col den since this will later be
			// incremented in RT_tau_inc 
			fprintf( ioQQQ,
				" PROBLEM radius_increment Lyman continuum insanity zone %li, effective tau=%g, atomic tau=%g simple tau=%g\n",
				nzone, 
				tau_effec , 
				tau_true ,
				6.327e-18*(dense.xIonDense[ipHYDROGEN][0]*radius.drad_x_fillfac+findspecieslocal("H")->column) );
			TotalInsanity();
		}
	}

	// do scattering opacity, not included when sphere is set
	if( opac.lgScatON )
	{
		for( long i=0; i < rfield.nflux; i++ )
		{
			// Lightman and White equation 11 in small epsilon limit,
			// >>refer	continuum	RT	Lightman, A.P., & White, T.R. 1988, ApJ, 335, 57 */
			double AttenuationScatteringFactor = 1./(1. + radius.drad_x_fillfac*opac.opacity_sct[i]);
			ASSERT( AttenuationScatteringFactor <= 1.0 );
			rfield.flux_beam_const[i] *= (realnum)AttenuationScatteringFactor;
			rfield.flux_beam_time[i] *= (realnum)AttenuationScatteringFactor;
			rfield.flux_isotropic[i] *= (realnum)AttenuationScatteringFactor;
			rfield.flux[0][i] = rfield.flux_beam_const[i] + rfield.flux_beam_time[i] +
				rfield.flux_isotropic[i];

			rfield.ConInterOut[i] *= (realnum)AttenuationScatteringFactor;
			rfield.ConEmitOut[0][i] *= (realnum)AttenuationScatteringFactor;
			rfield.outlin[0][i] *= (realnum)AttenuationScatteringFactor;
			rfield.outlin_noplot[i] *= (realnum)AttenuationScatteringFactor;
		}
	}

	// this dilution is needed to conserve volume in spherical models.  tests such
	// as parispn.in will fault if this is removed
	realnum Dilution = realnum(pow2(radius.rinner)/(radius.Radius*(radius.Radius-radius.drad)));

	// this is a unit of energy that will be passed through the code as a test
	// that all integrations are carried out.  A similar test is set in lineset1
	// and verified in PrtFinal.  The opacity at this cell is zero so only
	// geometrical dilution will affect the integral
	// Radius is currently outer edge of zone, so radius-drad/2 is radius
	// of center of zone 
	rfield.ConEmitLocal[nzone][rfield.nflux] = 1.e-10f * Dilution;
	rfield.DiffuseEscape[rfield.nflux] = 1.e-10f * Dilution;
	// must do unit integration somewhere 
	rfield.ConInterOut[rfield.nflux] += 
		rfield.DiffuseEscape[rfield.nflux]*(realnum)radius.dVolOutwrd;

	// opacity should be zero at this energy so J not changed elsewhere
	ASSERT( opac.opacity_abs[rfield.nflux] == 0. );

	// placeholder code for Compton scattering 
	cmshft();

	// attenuate neutrons if they are present 
	if( hextra.lgNeutrnHeatOn )
	{
		// correct for optical depth effects 
		hextra.totneu *= (realnum)sexp(hextra.CrsSecNeutron*
			dense.gas_phase[ipHYDROGEN]*radius.drad_x_fillfac*geometry.DirectionalCosin);
		// correct for spherical effects 
		hextra.totneu *= (realnum)DilutionHere;
	}

	// following radiation factors are extinguished by 1/r**2ilution, electron
	// scattering by free and bound electrons

	// do all emergent spectrum from illuminated face if model is NOT spherical
	if( !geometry.lgSphere )
	{
		double Reflec_Diffuse_Cont;

		// emission starting at the the plasma frequency
		for( long i=rfield.ipPlasma-1; i < rfield.nflux; i++ )
		{
			if( opac.TauAbsGeo[0][i] < 30. )
			{
				// ConEmitLocal is diffuse emission per unit vol, fill factor
				// the 1/2 comes from isotropic emission 
				Reflec_Diffuse_Cont = rfield.ConEmitLocal[nzone][i]/2.*
					radius.drad_x_fillfac * opac.E2TauAbsFace[i]*radius.r1r0sq;

				// ConEmitReflec - reflected diffuse continuum
				rfield.ConEmitReflec[0][i] += (realnum)(Reflec_Diffuse_Cont);

				// the reflected part of the incident continuum
				rfield.ConRefIncid[0][i] += (realnum)(rfield.flux[0][i]*opac.opacity_sct[i]*
					radius.drad_x_fillfac*opac.E2TauAbsFace[i]*radius.r1r0sq);

				// reflected line emission 
				rfield.reflin[0][i] += (realnum)(rfield.outlin[0][i]*opac.opacity_sct[i]*
					radius.drad_x_fillfac*opac.E2TauAbsFace[i]*radius.r1r0sq);
			}
		}
	}
	return;
}
