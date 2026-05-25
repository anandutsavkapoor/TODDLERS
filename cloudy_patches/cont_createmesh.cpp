/* This file is part of Cloudy and is copyright (C)1978-2025 by Gary J. Ferland and
 * others.  For conditions of distribution and use see copyright notice in license.txt */
/*rfield_opac_alloc allocate space for opacity arrays */
#include "cddefines.h"
#include "rfield.h"
#include "iterations.h"
#include "dense.h"
#include "trace.h"
#include "opacity.h"
#include "geometry.h"
#include "continuum.h"

/*rfield_opac_alloc allocate space for opacity arrays */
STATIC void rfield_opac_alloc();

void ContCreateMesh()
{
	long int i;

	/* flag to say whether pointers have ever been evaluated */
	static bool lgPntEval = false;

	DEBUG_ENTRY( "ContCreateMesh()" );

	/* lgPntEval is local static variable defined false when defined. 
	 * it is set true below, so that pointers only created one time in the
	 * history of this coreload. */
	if( lgPntEval )
	{
		if( trace.lgTrace )
		{
			fprintf( ioQQQ, " ContCreateMesh called, not evaluating.\n" );
		}
		vzero(opac.TauAbsFace);
		return;
	}
	else
	{
		if( trace.lgTrace )
		{
			fprintf( ioQQQ, " ContCreateMesh called first time.\n" );
		}
		lgPntEval = true;
	}

	/* ================================================================ */

	// the frequency mesh should already have been set up at this point in cdDrive()
	ASSERT( rfield.lgMeshSetUp() );
	ASSERT( rfield.nflux_with_check == rfield.ncells() );
	ASSERT( rfield.nflux == rfield.nflux_with_check-1 );
	ASSERT( rfield.nPositive == rfield.nflux );

	/* allocate space for continuum arrays within rfield.h and opacity arrays in opacity.h
	 * sets lgRfieldAllocated true */
	rfield_opac_alloc();

	/* geometry.nend_max is largest number of zones needed on any iteration,
	 * will use it to allocate arrays that save source function as function of zone */
	/* now change all limits, for all iterations, to this value */
	geometry.nend_max = iterations.nend[0];
	for( i=1; i < iterations.iter_alloc; i++ )
	{
		geometry.nend_max = MAX2( geometry.nend_max , iterations.nend[i] );
	}
	/* nend_max+1 because search phase is zone 0, first zone at illumin face is 1 */
	rfield.ConEmitLocal.alloc(geometry.nend_max+1, rfield.nflux_with_check);
	rfield.ConSourceFcnLocal.alloc(geometry.nend_max+1, rfield.nflux_with_check);
	rfield.ConSourceFcnLocal = 1_r;

	/* this is done here when the space is first allocated, 
	 * then done on every subsequent initialization in zero.c */
	rfield_opac_zero( 0, rfield.nflux_with_check );

	double scaledThomson = (TE1RYD/ELECTRON_MASS/SPEEDLIGHT/SPEEDLIGHT)*EN1RYD*BOLTZMANN*SIGMA_THOMSON*1e15;
	long ipnt = 0;
	/* now save current form of array, and define some quantities related to it */
	for( i=0; i < rfield.nflux_with_check; i++ )
	{
		double alf , bet;

		/* following are Compton exchange factors from Tarter */
		/* this code also appears in highen, but coef needed before that routine called. */
		alf = 1./(1. + rfield.anu(i)*(1.1792e-4 + 7.084e-10*rfield.anu(i)));
		bet = 1. - alf*rfield.anu(i)*(1.1792e-4 + 2.*7.084e-10*rfield.anu(i))/4.;
		rfield.csigh[i] = (realnum)(alf*rfield.anu(i)*rfield.anu(i)*scaledThomson);
		rfield.csigc[i] = (realnum)(alf*bet*rfield.anu(i)*scaledThomson);

		/* >>chng 05 feb 28, add transmission and mapping coef */
		/* map these coarse continua into fine continuum grid */
		if( rfield.anu(i) < rfield.fine_ener_lo || rfield.anu(i) > rfield.fine_ener_hi )
		{
			/* 0 (false) says not defined */
			rfield.ipnt_coarse_2_fine[i] = 0;
		}
		else
		{
			if( ipnt==0 )
			{
				/* this is the first one that maps onto the fine array */
				rfield.ipnt_coarse_2_fine[i] = 0;
				ipnt = 1;
			}
			else
			{
				/* find first fine frequency that is greater than this coarse value */
				while( ipnt < rfield.nfine && rfield.fine_anu[ipnt] < rfield.anu(i) )
				{
					++ipnt;
				}
				rfield.ipnt_coarse_2_fine[i] = ipnt;
			}
		}
		/*fprintf(ioQQQ," coarse %li nu= %.3e points to fine %li nu=%.3e\n",
			i, rfield.anu(i) , rfield.ipnt_coarse_2_fine[i] , rfield.fine_anu[rfield.ipnt_coarse_2_fine[i]] );*/
	}
	rfield.resetCoarseTransCoef();
	return;
}

/* allocate arrays within rfield */
STATIC void rfield_opac_alloc()
{
	long i;

	DEBUG_ENTRY( "rfield_opac_alloc()" );

	/* allocate one more than we use for the unit integration,
	 * will back up at end of routine */
	++rfield.nflux_with_check;

	/* >>chng 03 feb 12, add fine mesh fine grid fine opacity array to keep track of line overlap */
	/** \todo	3	consider making the fine opacity array a double.  with a float, the opacity 
	 * itself often becomes a denormalized number, it then becomes significant when multiplied
	 * by dr - can cause numerical noise.  this is why the coarse opacity array is a double */

	/* frequency range in Rydberg needed for all resonance lines */
	rfield.fine_ener_lo = rfield.emm();
	rfield.fine_ener_hi = 1500.f;

	/* set resolution of fine continuum mesh. 
	 * rfield.fine_opac_velocity_width is width per cell, cm/s 
	 * choose width so that most massive species (usually Fe) is well resolved
	 * 
	 * rfield.fine_opac_nelem is the most massive (hence sharpest line)
	 * we will worry about.  By default this is iron but can be changed
	 * with SET FINE CONTINUUM command 
	 * 
	 * TeLowestFineOpacity of 1e4 K is temperature where line width is
	 * evaluated.  Tests were done using the stop temperature in its place
	 * Te below 1e4 K made fine opacity grid huge 
	 * do not let temp get higher than 1e4 either - code run with stop temp 10 set
	 * stop temp of 1e10K and assert thrown at line 204 of cont_createpointers.c 
	 * simply use 1e4 K as a characteristic temperature */
	/** \todo	1	set temp of 1e4K will be too coarse a line for PDRs where
	 * H2 line overlap is very important */
	double TeLowestFineOpacity = 1e4;
	rfield.fine_opac_velocity_width = 
		(realnum)sqrt(2.*BOLTZMANN/ATOMIC_MASS_UNIT*TeLowestFineOpacity/
		dense.AtomicWeight[rfield.fine_opac_nelem] ) / 
		/* we want fine_opac_nresolv continuum elements across this line
		 * default is 1, changed with SET FINE CONTINUUM command */
		rfield.fine_opac_nresolv;

	/* we are at first zone so velocity shift is zero */
	rfield.ipFineConVelShift = 0;

	/* dimensionless resolution, dE/E, this is used in ipoint to get offset in fine mesh */
	rfield.fine_resol = rfield.fine_opac_velocity_width / SPEEDLIGHT;

	/* the number of cells needed */
	rfield.nfine = (long)(log10( rfield.fine_ener_hi / rfield.fine_ener_lo ) / log10( 1. + rfield.fine_resol ) );
	if( rfield.nfine <= 0 )
		TotalInsanity();

	/* this is the fine opacity array to ghost the main low-resolution array */
	rfield.fine_opac_zone.resize(rfield.nfine);
	vzero(rfield.fine_opac_zone);

	/* this is the fine total optical array to ghost the main low-resolution array */
	rfield.fine_opt_depth.resize(rfield.nfine);
	vzero(rfield.fine_opt_depth);

	rfield.fine_anu.resize(rfield.nfine);

	/* now fill in energy array */
	ASSERT( rfield.fine_ener_lo > 0. && rfield.fine_resol > 0 );
	{
		double bbb = 1.+rfield.fine_resol, 
			aaa = 1.;
		for( i=0;i<rfield.nfine; ++i )
		{
			aaa *= bbb;
			rfield.fine_anu[i] = rfield.fine_ener_lo * (realnum) aaa;
		}
	}
	/* done with fine array */

	/* used to count number of lines per cell */
	rfield.line_count.resize(rfield.nflux_with_check);
	vzero(rfield.line_count);
	rfield.flux_beam_time.resize(rfield.nflux_with_check);
	rfield.flux_isotropic.resize(rfield.nflux_with_check);
	rfield.flux_beam_const.resize(rfield.nflux_with_check);
	rfield.flux_accum.resize(rfield.nflux_with_check);
	rfield.ExtinguishFactor.resize(rfield.nflux_with_check);
	rfield.convoc.resize(rfield.nflux_with_check);
	rfield.OccNumbIncidCont.resize(rfield.nflux_with_check);
	rfield.OccNumbDiffCont.resize(rfield.nflux_with_check);
	rfield.OccNumbContEmitOut.resize(rfield.nflux_with_check);
	rfield.ConInterOut.resize(rfield.nflux_with_check);
	rfield.SummedCon.resize(rfield.nflux_with_check);
	rfield.SummedDif.resize(rfield.nflux_with_check);
	rfield.SummedDifSave.resize(rfield.nflux_with_check);
	rfield.SummedOcc.resize(rfield.nflux_with_check);
	rfield.ConOTS_local_photons.resize(rfield.nflux_with_check);
	rfield.DiffuseEscape.resize(rfield.nflux_with_check);
	rfield.TotDiff2Pht.resize(rfield.nflux_with_check);
	rfield.ConOTS_local_OTS_rate.resize(rfield.nflux_with_check);
	rfield.otslin.resize(rfield.nflux_with_check);
	rfield.otscon.resize(rfield.nflux_with_check);
	rfield.outlin_noplot.resize(rfield.nflux_with_check);
	rfield.flux_beam_const_save.resize(rfield.nflux_with_check);
	rfield.flux_time_beam_save.resize(rfield.nflux_with_check);
	rfield.flux_isotropic_save.resize(rfield.nflux_with_check);
	rfield.setCoarseTransCoefPtr(rfield.nflux_with_check);
	rfield.DiffuseLineEmission.resize(rfield.nflux_with_check);
	rfield.ipnt_coarse_2_fine.resize(rfield.nflux_with_check);

	/* possibly save cumulative flux */
	rfield.flux.resize( rfield.nflux_with_check );
	rfield.ConEmitReflec.resize( rfield.nflux_with_check );
	rfield.ConEmitOut.resize( rfield.nflux_with_check );
	rfield.ConEmitOutUnatt.resize( rfield.nflux_with_check );
	rfield.ConRefIncid.resize( rfield.nflux_with_check );
	rfield.flux_total_incident.resize( rfield.nflux_with_check );
	rfield.reflin.resize( rfield.nflux_with_check );
	rfield.outlin.resize( rfield.nflux_with_check );

	rfield.csigh.resize(rfield.nflux_with_check);
	rfield.csigc.resize(rfield.nflux_with_check);
	rfield.eeBremsDif.resize(rfield.nflux_with_check);

	rfield.comdn.resize(rfield.nflux_with_check);
	rfield.comup.resize(rfield.nflux_with_check);
	rfield.ContBoltz.resize(rfield.nflux_with_check);
	rfield.ContBoltzHelp1.resize(rfield.nflux_with_check);
	rfield.ContBoltzHelp2.resize(rfield.nflux_with_check);
	rfield.vexp_arg.resize(rfield.nflux_with_check);
	rfield.ContBoltzAvg.resize(rfield.nflux_with_check);

	rfield.otssav.alloc(rfield.nflux_with_check, 2);

	/* char rfield.chLineLabel[NLINES][5];*/
	rfield.chLineLabel.resize(rfield.nflux_with_check);
	rfield.chContLabel.resize(rfield.nflux_with_check);

	opac.TauAbsFace.resize(rfield.nflux_with_check);
	vzero(opac.TauAbsFace);

	opac.TauScatFace.resize(rfield.nflux_with_check);
	opac.E2TauAbsFace.resize(rfield.nflux_with_check);
	opac.E2TauAbsTotal.resize(rfield.nflux_with_check);
	opac.TauAbsTotal.resize(rfield.nflux_with_check);
	opac.E2TauAbsOut.resize(rfield.nflux_with_check);
	opac.ExpmTau.resize(rfield.nflux_with_check);
	opac.tmn.resize(rfield.nflux_with_check);

	opac.opacity_abs.resize(rfield.nflux_with_check);
	opac.opacity_abs_savzon1.resize(rfield.nflux_with_check);
	opac.OldOpacSave.resize(rfield.nflux_with_check);
	opac.opacity_sct.resize(rfield.nflux_with_check);
	opac.albedo.resize(rfield.nflux_with_check);
	opac.opacity_sct_savzon1.resize(rfield.nflux_with_check);
	opac.OpacStatic.resize(rfield.nflux_with_check);
	opac.FreeFreeOpacity.resize(rfield.nflux_with_check);
	opac.eeFreeFreeOpacity.resize(rfield.nflux_with_check);
	opac.ExpZone.resize(rfield.nflux_with_check);

	for( i=0; i<2; ++i)
	{
		opac.TauAbsGeo[i].resize(rfield.nflux_with_check);
		opac.TauScatGeo[i].resize(rfield.nflux_with_check);
		opac.TauTotalGeo[i].resize(rfield.nflux_with_check);
	}

	/* fix allocate trick for one more than we use for the unit integration */
	--rfield.nflux_with_check;

	/* say that space exists */
	lgRfieldAllocated = true;
	return;
}

/*rfield_opac_zero zero out rfield arrays between certain limits */
void rfield_opac_zero( 
					  /* index for first element in arrays to be set to zero */
					  long lo , 
					  /* array index for highest element to be set */
					  long ihi )
{
	long int i;

	/* >>chng 01 aug 19, space not allocated yet,
	* following code must also be present in contcreatemesh where
	* space allocated for the first time */
	if( lgRfieldAllocated )
	{
		unsigned long n=(unsigned long)(ihi-lo+1);
		memset(&rfield.OccNumbDiffCont[lo]      , 0 , n*sizeof(realnum) );
		memset(&rfield.OccNumbContEmitOut[lo]   , 0 , n*sizeof(realnum) );
		memset(&rfield.ContBoltz[lo]            , 0 , n*sizeof(double) );
		memset(&rfield.ContBoltzHelp1[lo]       , 0 , n*sizeof(double) );
		memset(&rfield.ContBoltzHelp2[lo]       , 0 , n*sizeof(double) );
		memset(&rfield.ContBoltzAvg[lo]         , 0 , n*sizeof(double) );
		/*>>chng 06 aug 15, this is now 2D array, saving diffuse continuum
		* over all zones for use in exact RT */
		/*memset(&rfield.ConEmitLocal[lo]       , 0 , n*sizeof(realnum) );*/
		memset(&rfield.ConEmitReflec[0][lo]     , 0 , n*sizeof(realnum) );
		memset(&rfield.ConEmitOut[0][lo]        , 0 , n*sizeof(realnum) );
		memset(&rfield.reflin[0][lo]            , 0 , n*sizeof(realnum) );
		memset(&rfield.ConRefIncid[0][lo]       , 0 , n*sizeof(realnum) );
		memset(&rfield.SummedCon[lo]            , 0 , n*sizeof(double) );
		memset(&rfield.convoc[lo]               , 0 , n*sizeof(realnum) );
		memset(&rfield.flux[0][lo]              , 0 , n*sizeof(realnum) );
		memset(&rfield.flux_total_incident[0][lo], 0 , n*sizeof(realnum) );
		memset(&rfield.flux_beam_const_save[lo] , 0 , n*sizeof(realnum) );
		memset(&rfield.flux_time_beam_save[lo]  , 0 , n*sizeof(realnum) );
		memset(&rfield.flux_isotropic_save[lo]  , 0 , n*sizeof(realnum) );
		memset(&rfield.SummedOcc[lo]            , 0 , n*sizeof(realnum) );
		memset(&rfield.SummedDif[lo]            , 0 , n*sizeof(realnum) );
		memset(&rfield.flux_accum[lo]           , 0 , n*sizeof(realnum) );
		memset(&rfield.otslin[lo]               , 0 , n*sizeof(realnum) );
		memset(&rfield.otscon[lo]               , 0 , n*sizeof(realnum) );
		memset(&rfield.ConInterOut[lo]          , 0 , n*sizeof(realnum) );
		memset(&rfield.outlin[0][lo]            , 0 , n*sizeof(realnum) );
		memset(&rfield.outlin_noplot[lo]        , 0 , n*sizeof(realnum) );
		memset(&rfield.ConOTS_local_OTS_rate[lo], 0 , n*sizeof(realnum) );
		memset(&rfield.ConOTS_local_photons[lo] , 0 , n*sizeof(realnum) );
		memset(&opac.OldOpacSave[lo]            , 0 , n*sizeof(double) );
		memset(&opac.opacity_abs[lo]            , 0 , n*sizeof(double) );
		memset(&opac.opacity_sct[lo]            , 0 , n*sizeof(double) );
		memset(&opac.albedo[lo]                 , 0 , n*sizeof(double) );
		memset(&opac.FreeFreeOpacity[lo]        , 0 , n*sizeof(double) );

		/* these are not defined on first iteration */
		memset( &opac.E2TauAbsTotal[lo]        , 0 , n*sizeof(realnum) );
		memset( &opac.E2TauAbsOut[lo]          , 0 , n*sizeof(realnum) );
		memset( &opac.TauAbsTotal[lo]          , 0 , n*sizeof(realnum) );

		for( i=lo; i <= ihi; i++ )
		{
			opac.TauTotalGeo[0][i] = TAUMIN;
			opac.TauAbsGeo[0][i] = TAUMIN;
			opac.TauScatGeo[0][i] = TAUMIN;
			opac.tmn[i] = 1.;
			opac.ExpZone[i] = 1.;
			opac.E2TauAbsFace[i] = 1.;
			opac.ExpmTau[i] = 1.;
			opac.OpacStatic[i] = 1.;
		}
		/* also zero out fine opacity fine grid fine mesh array */
		vzero(rfield.fine_opac_zone);
		/* also zero out fine opacity array */
		vzero(rfield.fine_opt_depth);
	}
	return;
}
