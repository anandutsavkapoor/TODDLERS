/* This file is part of Cloudy and is copyright (C)1978-2025 by Gary J. Ferland and
 * others.  For conditions of distribution and use see copyright notice in license.txt */
/*IterStart set and save values of many variables at start of iteration after initial temperature set*/
/*IterRestart restart iteration */
#include "cddefines.h"
#include "cddrive.h"
#include "iso.h"
#include "taulines.h"
#include "hydrogenic.h"
#include "struc.h"
#include "dynamics.h"
#include "prt.h"
#include "hyperfine.h"
#include "magnetic.h"
#include "continuum.h"
#include "geometry.h"
#include "h2.h"
#include "co.h"
#include "he.h"
#include "grains.h"
#include "pressure.h"
#include "stopcalc.h"
#include "conv.h"
#include "mean.h"
#include "thermal.h"
#include "atoms.h"
#include "wind.h"
#include "opacity.h"
#include "timesc.h"
#include "trace.h"
#include "colden.h"
#include "secondaries.h"
#include "hmi.h"
#include "radius.h"
#include "phycon.h"
#include "called.h"
#include "ionbal.h"
#include "atmdat.h"
#include "lines.h"
#include "molcol.h"
#include "input.h"
#include "rt.h"
#include "iterations.h"
#include "cosmology.h"
#include "deuterium.h"
#include "mole.h"
#include "rfield.h"
#include "freebound.h"
#include "two_photon.h"
#include "dense.h"

/* these are the static saved variables, set in IterStart and reset in IterRestart */

static double h2plus_heat_save, HeatH2Dish_used_save, HeatH2Dexc_used_save,
	hmihet_save, hmitot_save, H2_Solomon_dissoc_rate_used_H2g_save,deriv_HeatH2Dexc_used_save,
	H2_Solomon_dissoc_rate_used_H2s_save, H2_H2g_to_H2s_rate_used_save, H2_photodissoc_used_H2s_save,
	H2_photodissoc_used_H2g_save, 
	UV_Cont_rel2_Draine_DB96_face,
	UV_Cont_rel2_Draine_DB96_depth , UV_Cont_rel2_Habing_TH85_face;
static multi_arr<double,2> saveMoleSource, saveMoleSink;
static multi_arr<realnum,3> SaveMoleChTrRate;

/* arguments are atomic number, ionization stage*/
static realnum xIonFsave[LIMELM][LIMELM+1];
static realnum deutDenseSave0, deutDenseSave1;
static double HeatSave[LIMELM][LIMELM];
static realnum supsav[LIMELM][LIMELM],p2nit,d5200r;
/* save values of dr and drNext */
static double drSave , drNextSave;

/* also places to save them between iterations */
static long int IonLowSave[LIMELM], 
	IonHighSave[LIMELM];

/* these are used to reset the level populations of the h and he model atoms */
/*static realnum hnsav[LIMELM][LMHLVL+1], */
static bool lgHNSAV = false;

static multi_arr<double,3> HOpacRatSav;

static double ortho_save, para_save, edsav;
static multi_arr<double,3> hnsav;

static realnum gas_phase_save[LIMELM];
static vector<double> den_save;

/*IterStart set and save values of many variables at start of iteration after initial temperature set*/
void IterStart()
{
	long int i, 
	  ion, 
	  ion2,
	  ipISO,
	  n ,
	  nelem;

	DEBUG_ENTRY( "IterStart()" );

	/* allocate two matrices if first time in this core load 
	 * these variables are allocated here because they are file static - 
	 * used to save values at start of first zone, and reset to this value at
	 * start of new iteration */
	if( !lgHNSAV )
	{
		/* set flag so we never do this again */
		lgHNSAV = true;

		HOpacRatSav.reserve(NISO);
		for( ipISO=ipH_LIKE; ipISO < NISO; ++ipISO )
		{
			HOpacRatSav.reserve(ipISO,LIMELM);
			for( nelem=ipISO; nelem<LIMELM; ++nelem )
			{
				HOpacRatSav.reserve(ipISO, nelem, iso_sp[ipISO][nelem].numLevels_max+1);
			}
		}
		HOpacRatSav.alloc();
		hnsav.alloc(HOpacRatSav.clone());

		saveMoleSource.reserve(LIMELM);
		SaveMoleChTrRate.reserve(LIMELM);
		for( nelem=0; nelem < LIMELM; ++nelem )
		{
			/* chemistry source and sink terms for ionization ladders */
			saveMoleSource.reserve(nelem, nelem+2);
			SaveMoleChTrRate.reserve(nelem, nelem+2);
			for( ion=0; ion < nelem+2; ++ion )
			{
				SaveMoleChTrRate.reserve(nelem, ion, nelem+2);
			}
		}
		saveMoleSource.alloc();
		saveMoleSink.alloc(saveMoleSource.clone());
		SaveMoleChTrRate.alloc();

		den_save.resize(mole_global.num_calc);
	}

	/* IterStart is called at the start of EVERY iteration */
	if( trace.lgTrace )
	{
		fprintf( ioQQQ, " IterStart called.\n" );
	}

	/* check if this is the last iteration */
	if( iteration > iterations.itermx )
		iterations.lgLastIt = true;

	/* flag set true in RT_line_one_tauinc if maser ever capped */
	rt.lgMaserCapHit = false;
	/* flag for remembering if maser ever set dr in any zone */
	rt.lgMaserSetDR = false;

	LineSave.lgIsoContSubSignif = false;

	/* zero out charge transfer heating and cooling fractions */
	atmdat.HCharHeatMax = 0.;
	atmdat.HCharCoolMax = 0.;

	/* these are set false if limits of Case B tables is ever exceeded */
	for(nelem=0; nelem<HS_NZ; ++nelem )
	{
		atmdat.lgHCaseBOK[0][nelem] = true;
		atmdat.lgHCaseBOK[1][nelem] = true;
	}

	/* reset the largest number of levels in the database species */
	for( long ipSpecies=0; ipSpecies<nSpecies; ++ipSpecies )
	{
		dBaseSpecies[ipSpecies].numLevels_local = dBaseSpecies[ipSpecies].numLevels_max;
		dBaseSpecies[ipSpecies].CoolTotal = 0.;
	}

	/* the reason this calculation stops */
	strncpy( StopCalc.chReasonStop, "reason not specified.", 
		sizeof(StopCalc.chReasonStop) );

	/* zero fractions of He0 destruction due to 23S */
	he.nzone = 0;
	he.frac_he0dest_23S = 0.;
	he.frac_he0dest_23S_photo = 0.;

	dense.EdenMax = 0.;
	dense.EdenMin = BIGFLOAT;

	timesc.time_H2_Dest_longest = 0.;
	timesc.time_H2_Form_longest = 0.;
	/* remains neg if not evaluated */
	timesc.time_H2_Dest_here = -1.;
	timesc.time_H2_Form_here = 0.;

	timesc.BigCOMoleForm = 0.;
	timesc.TimeH21cm = 0.;
	thermal.CoolHeatMax = 0.;
	hydro.HCollIonMax = 0.;

	atmdat.HIonFracMax = 0.;

	/* will save highest n=2p/1s ratio for hydrogen atom*/
	hydro.pop2mx = 0.;
	hydro.lgHiPop2 = false;

	hydro.nLyaHot = 0;
	hydro.TLyaMax = 0.;

	/* evaluate the gas and radiation pressure */
	/* this sets values of pressure.PresTotlCurr */
	pressure.PresInteg = 0.;
	pressure.pinzon = 0.;
	pressure.PresIntegElecThin = 0.;
	PresTotCurrent();

	dynamics.HeatMax = 0.;
	dynamics.CoolMax = 0.;
	if( dynamics.lgAdvection || dynamics.lgTimeDependentStatic )
		DynaIterStart();

	/* reset these since we now have a valid solution */
	pressure.pbeta = 0.;
	pressure.RadBetaMax = 0.;
	pressure.lgPradCap = false;
	pressure.lgPradDen = false;
	/* this flag will say we hit the sonic point */
	pressure.lgSonicPoint = false;
	pressure.lgStrongDLimbo = false;

	/* define two timescales for equilibrium, Compton and thermal */
	timesc.tcmptn = 0;
	if( rfield.qtot>SMALLFLOAT )
		timesc.tcmptn = 1.e0/(rfield.qtot*6.65e-25*dense.eden);
	timesc.time_therm_long = 1.5*dense.pden*BOLTZMANN*phycon.te/thermal.ctot;

	/* this will be total mass in computed structure */
	dense.xMassTotal = 0.;

	for( nelem=0; nelem < LIMELM; nelem++ )
	{

		if( dense.lgElmtOn[nelem] )
		{

			/* now zero out the ionic fractions */
			for( ion=0; ion < (nelem + 2); ion++ )
			{
				xIonFsave[nelem][ion] = dense.xIonDense[nelem][ion];
			}

			for( ion=0; ion < LIMELM; ion++ )
			{
				HeatSave[nelem][ion] = thermal.heating(nelem,ion);
			}
		}
	}

	deutDenseSave0 = deut.xIonDense[0];
	deutDenseSave1 = deut.xIonDense[1];

	/* >>chng 02 jan 28, add ipISO loop */
	for( ipISO=ipH_LIKE; ipISO<NISO; ++ipISO )
	{
		/* >>chng 03 apr 11, from 0 to ipISO */
		for( nelem=ipISO; nelem < LIMELM; nelem++ )
		{
			if( dense.lgElmtOn[nelem] )
			{
				for( n=0; n < iso_sp[ipISO][nelem].numLevels_max; n++ )
				{
					HOpacRatSav[ipISO][nelem][n] = iso_sp[ipISO][nelem].fb[n].ConOpacRatio;
					hnsav[ipISO][nelem][n] = iso_sp[ipISO][nelem].st[n].Pop();
				}
			}
			for( vector<two_photon>::iterator tnu = iso_sp[ipISO][nelem].TwoNu.begin(); tnu != iso_sp[ipISO][nelem].TwoNu.end(); ++tnu )
				tnu->induc_dn_max = 0.;
		}
	}

	rfield.ipEnergyBremsThin = 0;
	rfield.EnergyBremsThin = 0.;

	p2nit = atoms.p2nit;
	d5200r = atoms.d5200r;

	/* save molecular fractions and ionization range */
	for( nelem=ipHYDROGEN; nelem < LIMELM; nelem++ )
	{
		/* save molecular densities */
		gas_phase_save[nelem] = dense.gas_phase[nelem];
		IonLowSave[nelem] = dense.IonLow[nelem];
		IonHighSave[nelem] = dense.IonHigh[nelem];
	}
	/*fprintf(ioQQQ,"DEBUG IterStart save set to gas_phase hden %.4e \n",
		dense.gas_phase[0]);*/

	edsav = dense.eden;

	/* save molecular densities */
	for( i=0; i<mole_global.num_calc; i++) 
	{
		den_save[i] = mole.species[i].den;
	}
	ortho_save = h2.ortho_density;
	para_save = h2.para_density;

	for( nelem=ipHYDROGEN; nelem < LIMELM; ++nelem )
	{
		/* these have one more ion than above */
		for( ion=0; ion<nelem+2; ++ion )
		{
			/* zero out the source and sink arrays */
			saveMoleSource[nelem][ion] = mole.source[nelem][ion];
			saveMoleSink[nelem][ion] = mole.sink[nelem][ion];
			for( ion2=0; ion2<nelem+2; ++ion2 )
			{
				SaveMoleChTrRate[nelem][ion][ion2] = mole.xMoleChTrRate[nelem][ion][ion2];
			}
		}
	}

	hmihet_save = hmi.hmihet;
	hmitot_save = hmi.hmitot;

	h2plus_heat_save = hmi.h2plus_heat;

	HeatH2Dish_used_save = hmi.HeatH2Dish_used;
	HeatH2Dexc_used_save = hmi.HeatH2Dexc_used;

	deriv_HeatH2Dexc_used_save = hmi.deriv_HeatH2Dexc_used;
	H2_Solomon_dissoc_rate_used_H2g_save = hmi.H2_Solomon_dissoc_rate_used_H2g;
	H2_Solomon_dissoc_rate_used_H2s_save = hmi.H2_Solomon_dissoc_rate_used_H2s;

	H2_H2g_to_H2s_rate_used_save = hmi.H2_H2g_to_H2s_rate_used;

	H2_photodissoc_used_H2s_save = hmi.H2_photodissoc_used_H2s;
	H2_photodissoc_used_H2g_save = hmi.H2_photodissoc_used_H2g;

	UV_Cont_rel2_Draine_DB96_face = hmi.UV_Cont_rel2_Draine_DB96_face;
	UV_Cont_rel2_Draine_DB96_depth = hmi.UV_Cont_rel2_Draine_DB96_depth;
	UV_Cont_rel2_Habing_TH85_face = hmi.UV_Cont_rel2_Habing_TH85_face;

	/* save zone thickness, and next zone thickness */
	drSave = radius.drad;
	drNextSave = radius.drNext;
	/* remember largest and smallest dr over iteration */
	radius.dr_min_last_iter = BIGFLOAT;
	radius.dr_max_last_iter = 0.;

	geometry.nprint = iterations.IterPrnt[iteration-1];

	colden.TotMassColl = 0.;
	colden.tmas = 0.;
	colden.wmas = 0.;
	colden.rjnmin = FLT_MAX;
	colden.ajmmin = FLT_MAX;

	thermal.nUnstable = 0;
	thermal.lgUnstable = false;

	rfield.extin_mag_B_point = 0.;
	rfield.extin_mag_V_point = 0.;
	rfield.extin_mag_B_extended = 0.;
	rfield.extin_mag_V_extended = 0.;

	/* plus 1 is to zero out point where unit integration occurs */
	for( i=0; i < rfield.nflux_with_check+1; i++ )
	{
		/* diffuse fields continuum */
		/* >>chng 03 nov 08, recover SummedDif */
		rfield.SummedDifSave[i] = rfield.SummedDif[i];
		rfield.ConEmitReflec[0][i] = 0.;
		rfield.ConEmitOut[0][i] = 0.;
		rfield.ConEmitOutUnatt[0][i] = 0.;
		rfield.ConInterOut[i] = 0.;
		/* save OTS continuum for next iteration */
		rfield.otssav[i][0] = rfield.otscon[i];
		rfield.otssav[i][1] = rfield.otslin[i];
		rfield.outlin[0][i] = 0.;
		rfield.outlin_noplot[i] = 0.;
		rfield.ConOTS_local_OTS_rate[i] = 0.;
		rfield.ConOTS_local_photons[i] = 0.;
		rfield.DiffuseEscape[i] = 0.;

		/* save initial absorption, scattering opacities for next iteration */
		opac.opacity_abs_savzon1[i] = opac.opacity_abs[i];
		opac.opacity_sct_savzon1[i] = opac.opacity_sct[i];

		/* will accumulate optical depths through cloud */
		opac.TauAbsFace[i] = TAUMIN;
		opac.TauScatFace[i] = TAUMIN;
		/* >>chng 99 dec 04, having exactly 1 zone thickness for first zone caused discontinuity
		 * for heating in very high T models in func_map.in.  zone 1 and 2 were 20% different,
		 * since tau in is 1e-20, e2 is 0.9999, and so some H ots
		 * these were not here at all - changed to dr/2 */
		/* attenuation of flux by optical depths IN THIS ZONE 
		 * DirectionalCosin is 1/COS(theta), is usually 1, reset with illumination command,
		 * option for illumination of slab at an angle */
		/* >>chng 04 oct 09, from drad to radius.drad_x_fillfac - include fill fac, PvH */
		opac.ExpZone[i] = sexp(opac.opacity_abs[i]*radius.drad_x_fillfac/2.*geometry.DirectionalCosin);

		/* e(-tau) in inward direction, up to illuminated face */
		opac.ExpmTau[i] = (realnum)opac.ExpZone[i];

		/* e2(tau) in inward direction, up to illuminated face, this is used to get the
		 * recombination escape probability */
		opac.E2TauAbsFace[i] = (realnum)e2(opac.TauAbsFace[i] );
	}

	/* this zeros out arrays to hold mean ionization fractions
	 * later entered by mean
	 * read out by setlim */
	mean.zero();

	/* zero out the column densities */
	for( i=0; i < NCOLD; i++ )
	{
		colden.colden[i] = 0.;
	}
	colden.H0_ov_Tspin = 0.;
	colden.OH_ov_Tspin = 0.;
	colden.coldenH2_ov_vel = 0.;

	/* upper and lower levels of H0 1s */
	colden.H0_21cm_upper =0;
	colden.H0_21cm_lower =0;

	for( diatom_iter diatom = diatoms.begin(); diatom != diatoms.end(); ++diatom )
	{
		(*diatom)->ortho_colden = 0.;
		(*diatom)->para_colden = 0.;
	}

	for( i=0; i < mole_global.num_calc; i++ )
	{
		mole.species[i].column = 0.;
	}

	/* these are some line of sight emission measures */
	colden.dlnenp = 0.;
	colden.dlnenHep = 0.;
	colden.dlnenHepp = 0.;
	colden.dlnenCp = 0.;
	colden.H0_ov_Tspin = 0.;
	colden.OH_ov_Tspin = 0.;

	// zero column densities of all states
	for( unsigned i = 0; i < mole.species.size(); ++i )
	{
		if( mole.species[i].levels != NULL )
		{
			for( qList::iterator st = mole.species[i].levels->begin(); st != mole.species[i].levels->end(); ++st )
			{
				(*st).ColDen() = 0.;	
			}
		}
	}

	ipISO = ipH_LIKE;
	for( long nelem=ipISO; nelem < LIMELM; ++nelem )
	{
		if( dense.lgElmtOn[nelem] )
		{
			for( long nHi=2; nHi < iso_ctrl.nLymanHLike[nelem]; nHi++ )
			{
				ExtraLymanLinesJ05[nelem][nHi].Lo()->ColDen() = 0.;
				ExtraLymanLinesJ15[nelem][nHi].Lo()->ColDen() = 0.;
			}
		}
	}

	ipISO = ipHE_LIKE;
	for( long nelem=ipISO; nelem < LIMELM; ++nelem )
	{
		if( dense.lgElmtOn[nelem] )
		{
			for(long ipHi = iso_sp[ipISO][nelem].st[iso_sp[ipISO][nelem].numLevels_local-1].n()+1; ipHi < iso_ctrl.nLyman[ipISO]; ipHi++ )
			{
				ExtraLymanLinesHeLike[nelem][ipExtraLymanLinesHeLike[nelem][ipHi]].Lo()->ColDen() = 0.;
			}
		}
	}

	/* now zero heavy element molecules */
	molcol("ZERO",ioQQQ);

	/* this will be sum of all free free heating over model */
	thermal.FreeFreeTotHeat = 0.;

	thermal.thist = 0.;
	thermal.tlowst = 1e20f;

	wind.AccelAver = 0.;
	wind.acldr = 0.;
	ionbal.ilt = 0;
	ionbal.iltln = 0;
	ionbal.ilthn = 0;
	ionbal.ihthn = 0;
	ionbal.ifail = 0;

	secondaries.SecHIonMax = 0.;
	for( nelem=ipHYDROGEN; nelem<LIMELM; ++nelem )
	{
		for( ion=0; ion<nelem+1; ++ion )
		{
			supsav[nelem][ion] = secondaries.csupra[nelem][ion];
		}
	}
	secondaries.x12sav = secondaries.x12tot;
	secondaries.hetsav = secondaries.HeatEfficPrimary;
	secondaries.savefi = secondaries.SecIon2PrimaryErg;
	for( nelem=ipHYDROGEN; nelem<LIMELM; ++nelem)
	{
		for( ion=0; ion<nelem+1; ++ion )
		{
			ionbal.CompRecoilHeatRateSave[nelem][ion] = ionbal.CompRecoilHeatRate[nelem][ion];
			ionbal.CompRecoilIonRateSave[nelem][ion] = ionbal.CompRecoilIonRate[nelem][ion];
		}
	}

	/* these will keep track of the number of convergence failures that occur */
	conv.nTeFail = 0;
	conv.nTotalFailures = 0;
	conv.nPreFail = 0;
	conv.failmx = 0.;
	conv.nIonFail = 0;
	conv.nPopFail = 0;
	conv.nNeFail = 0;
	conv.nGrainFail = 0;
	conv.nChemFail = 0;
	conv.dCmHdT = 0.;

	GrainStartIter();

	rfield.comtot = 0.;

	co.codfrc = 0.;
	co.codtot = 0.;

	hmi.HeatH2DexcMax = 0.;
	hmi.CoolH2DexcMax = 0.;
	hmi.h2dfrc = 0.;
	hmi.h2line_cool_frac = 0.;
	hmi.h2dtot = 0.;
	thermal.HeatLineMax = 0.;
	thermal.GBarMax = 0.;
	hyperfine.cooling_max = 0.;
	hydro.cintot = 0.;
	geometry.lgZoneTrp = false;

	hmi.h2pmax = 0.;

	/************************************************************************
	 *
	 * allocate space for lines arrays 
	 *
	 ************************************************************************/



	/* this was set in call to lines above */
	ASSERT( LineSave.nsum > 0 );
	ASSERT( LineSave.size() >= (size_t) LineSave.nsum );

	/* zero emission line arrays - this has to be done on every iteration */
	for( i=0; i < LineSave.nsum; i++ )
	{
		LineSave.lines[i].SumLineZero();
		LineSave.lines[i].emslinZero();
	}

	/* now zero out some variables set by last call to LINES */
	hydro.cintot = 0.;
	thermal.totcol = 0.;
	rfield.comtot = 0.;
	thermal.FreeFreeTotHeat = 0.;
	thermal.power = 0.;
	thermal.HeatLineMax = 0.;
	thermal.GBarMax = 0.;
	hyperfine.cooling_max = 0.;

	hmi.h2pmax = 0.;

	co.codfrc = 0.;
	co.codtot = 0.;

	hmi.h2dfrc = 0.;
	hmi.h2line_cool_frac = 0.;
	hmi.h2dtot = 0.;
	timesc.sound = 0.;

	LineSave.ipNormLine = LineSave.findline(LineSave.NormLine);
	if( LineSave.ipNormLine < 0 )
	{
		/* did not find the line if return is negative */
		fprintf( ioQQQ, "PROBLEM could not find the normalisation line.\n");
		fprintf( ioQQQ, "IterStart could not find the line: %s", LineSave.NormLine.str().c_str() );
		fprintf( ioQQQ, "Please check the emission line output to find the correct line identification.\n");
		fprintf( ioQQQ, "Sorry.\n");
		LineSave.ipNormLine = 0;
		fprintf( ioQQQ, "Setting normalisation line to first line in stack, and proceeding.\n");
	}

	/* set up stop line command on first iteration 
	 * find index for lines and save for future iterations */
	if( iteration == 1 )
	{
		for( auto& p : StopCalc.sle )
		{
			double relint, absint ;

			/* returns array index for line in array stack if we found the line, 
			 * return negative of total number of lines as debugging aid if line not found */
			p.ipStopLine1 = cdLine( p.line1, &relint, &absint );

			if( p.ipStopLine1 < 0 )
			{
				fprintf( ioQQQ, 
					" IterStart could not find first line in STOP LINE command, line number %ld: ",
						 p.ipStopLine1 );
				prt_line_err( ioQQQ, p.line1 );
				cdEXIT(EXIT_FAILURE);
			}

			p.ipStopLine2 = cdLine( p.line2, &relint, &absint );

			if( p.ipStopLine2 < 0 )
			{
				fprintf( ioQQQ, 
					" IterStart could not find second line in STOP LINE command, line number %ld: ", 
						 p.ipStopLine2 );
				prt_line_err( ioQQQ, p.line2 );
				cdEXIT(EXIT_FAILURE);
			}

			if( trace.lgTrace )
			{
				fprintf( ioQQQ, " stop line 1 is number %5ld label is %s\n", 
						 p.ipStopLine1, LineSave.lines[p.ipStopLine1].label().c_str() );
				fprintf( ioQQQ, " stop line 2 is number %5ld label is %s\n", 
						 p.ipStopLine2, LineSave.lines[p.ipStopLine2].label().c_str() );
			}
		}
	}

	/* option to only print last iteration */
	if( prt.lgPrtLastIt )
	{
		if( iteration == 1 )
		{
			called.lgTalk = false;
		}

		/* initial condition of TALK may be off if optimization used or not master rank
		 * sec part is for print last command 
		 * lgTalkForcedOff is set true when cdTalk is called with false
		 * to turn off printing */
		if( iterations.lgLastIt && !prt.lgPrtStart && !called.lgTalkForcedOff )
		{
			called.lgTalk = called.lgTalkSave;
		}
	}

	if( opac.lgCaseB )
	{
		if( trace.lgTrace )
		{
			fprintf( ioQQQ, " IterStart does not change mid-zone optical depth since CASE B\n" );
		}
	}

	/* check if induced recombination can be ignored */
	hydro.FracInd = 0.;
	hydro.fbul = 0.;

	/* remember some things about effects of induced rec on H only
	 * don't do ground state since SPHERE turns it off */
	for( i=ipH2p; i < (iso_sp[ipH_LIKE][ipHYDROGEN].numLevels_max - 1); i++ )
	{
		if( iso_sp[ipH_LIKE][ipHYDROGEN].trans(i+1,i).Emis().Aul() <= iso_ctrl.SmallA )
			continue;

		double ratio = iso_sp[ipH_LIKE][ipHYDROGEN].fb[i].RecomInducRate*iso_sp[ipH_LIKE][ipHYDROGEN].fb[i].PopLTE/
			SDIV(iso_sp[ipH_LIKE][ipHYDROGEN].fb[i].RecomInducRate*iso_sp[ipH_LIKE][ipHYDROGEN].fb[i].PopLTE + 
			iso_sp[ipH_LIKE][ipHYDROGEN].fb[i].RadRecomb[ipRecRad]*
			iso_sp[ipH_LIKE][ipHYDROGEN].fb[i].RadRecomb[ipRecNetEsc]);
		if( ratio > hydro.FracInd )
		{
			hydro.FracInd = (realnum)ratio;
			hydro.ndclev = i;
		}

		ratio = iso_sp[ipH_LIKE][ipHYDROGEN].trans(i+1,i).Emis().pump()/
			(iso_sp[ipH_LIKE][ipHYDROGEN].trans(i+1,i).Emis().pump() + 
			iso_sp[ipH_LIKE][ipHYDROGEN].trans(i+1,i).Emis().Aul());

		if( ratio > hydro.fbul )
		{
			hydro.fbul = (realnum)ratio;
			hydro.nbul = i;
		}
	}

	if( trace.lgTrace )
		fprintf( ioQQQ, " IterStart returns.\n" );
	return;
}

/*IterRestart reset many variables at the start of a new iteration 
 * called by cloudy after calculation is completed, when more iterations 
 * are needed - the iteration has been incremented when this routine is
 * called so iteration == 2 after first iteration, we are starting
 * the second iteration */
void IterRestart(void)
{
	long int i, 
	  ion, 
		ion2,
	  ipISO ,
	  n, 
	  nelem;
	double SumOTS;

	DEBUG_ENTRY( "IterRestart()" );

	/* this is case where temperature floor has been set, if it was hit
	 * then we did a constant temperature calculation, and must go back to 
	 * a thermal solution 
	 * test on thermal.lgTemperatureConstantCommandParsed distinguishes
	 * from temperature floor option, so not reset if constant temperature
	 * was actually set */
	if( StopCalc.TeFloor > 0. && !thermal.lgTemperatureConstantCommandParsed )
	{
		thermal.lgTemperatureConstant = false;
		thermal.ConstTemp = 0.;
	}

	/* reset some parameters needed for magnetic field */
	Magnetic_reinit();

	opac.stimax[0] = 0.;
	opac.stimax[1] = 0.;

	for( diatom_iter diatom = diatoms.begin(); diatom != diatoms.end(); ++diatom )
		(*diatom)->H2_Reset();

	for( nelem=ipHYDROGEN; nelem < LIMELM; ++nelem )
	{
		/* these have one more ion than above */
		for( ion=0; ion<nelem+2; ++ion )
		{
			/* zero out the source and sink arrays */
			mole.source[nelem][ion] = saveMoleSource[nelem][ion];
			mole.sink[nelem][ion] = saveMoleSink[nelem][ion];
			for( ion2=0; ion2<nelem+2; ++ion2 )
			{
				mole.xMoleChTrRate[nelem][ion][ion2] = SaveMoleChTrRate[nelem][ion][ion2];
			}
		}
	}

	/* reset molecular abundances */
	for( i=0; i<mole_global.num_calc; i++) 
	{
		mole.species[i].den = den_save[i];
	}
	dense.updateXMolecules();
	deut.updateXMolecules();
	hmi.H2_total = findspecieslocal("H2")->den + findspecieslocal("H2*")->den;
	hmi.HD_total = findspecieslocal("HD")->den + findspecieslocal("HD*")->den;
	/*fprintf(ioQQQ," IterRestar sets H2 total to %.2e\n",hmi.H2_total );*/
	h2.ortho_density = ortho_save;
	h2.para_density = para_save;
	{
		hmi.H2_total_f = (realnum)hmi.H2_total;
		h2.ortho_density_f = (realnum)h2.ortho_density;
		h2.para_density_f = (realnum)h2.para_density;
	}

	/* zero out the column densities */
	for( i=0; i < NCOLD; i++ )
	{
		colden.colden[i] = 0.;
	}
	colden.coldenH2_ov_vel = 0.;

	for( i=0; i < mole_global.num_calc; i++ )
	{
		/* column densities */
		mole.species[i].column = 0.;
		/* largest fraction of atoms in molecules */
		mole.species[i].xFracLim = 0.;
		mole.species[i].atomLim = null_nuclide;
	}


	/* close out this iteration if dynamics or time dependence is enabled */
	if( dynamics.lgAdvection || dynamics.lgTimeDependentStatic )
		DynaIterEnd();

	rfield.extin_mag_B_point = 0.;
	rfield.extin_mag_V_point = 0.;
	rfield.extin_mag_B_extended = 0.;
	rfield.extin_mag_V_extended = 0.;

	hmi.hmihet = hmihet_save;
	hmi.hmitot = hmitot_save;

	hmi.h2plus_heat = h2plus_heat_save;
	hmi.HeatH2Dish_used = HeatH2Dish_used_save;
	hmi.HeatH2Dexc_used = HeatH2Dexc_used_save;

	hmi.deriv_HeatH2Dexc_used = (realnum)deriv_HeatH2Dexc_used_save;
	hmi.H2_Solomon_dissoc_rate_used_H2g = H2_Solomon_dissoc_rate_used_H2g_save;
	hmi.H2_Solomon_dissoc_rate_used_H2s = H2_Solomon_dissoc_rate_used_H2s_save;
	hmi.H2_H2g_to_H2s_rate_used = H2_H2g_to_H2s_rate_used_save;
	hmi.H2_photodissoc_used_H2s = H2_photodissoc_used_H2s_save;
	hmi.H2_photodissoc_used_H2g = H2_photodissoc_used_H2g_save;

	hmi.UV_Cont_rel2_Draine_DB96_face = (realnum)UV_Cont_rel2_Draine_DB96_face;
	hmi.UV_Cont_rel2_Draine_DB96_depth = (realnum)UV_Cont_rel2_Draine_DB96_depth;
	hmi.UV_Cont_rel2_Habing_TH85_face = (realnum)UV_Cont_rel2_Habing_TH85_face;

	rfield.ipEnergyBremsThin = 0;
	rfield.EnergyBremsThin = 0.;
	rfield.lgUSphON = false;

	radius.glbdst = 0.;
	/* zone thickness, and next zone thickness */
	radius.drad = drSave;
	radius.drad_mid_zone = drSave/2.;
	radius.drNext = drNextSave;

	/* was min dr ever used? */
	radius.lgDrMinUsed = false;

	continuum.cn4861 = continuum.sv4861;
	continuum.cn1367 = continuum.sv1367;
	continuum.cn2066 = continuum.sv2066;
	continuum.cn1216 = continuum.sv1216;

	/* total luminosity */
	continuum.totlsv = continuum.TotalLumin;

	/* set debugger on now if NZONE desired is 0 */
	if( (trace.nznbug == 0 && iteration >= trace.npsbug) && trace.lgTrOvrd )
	{
		if( trace.nTrConvg==0 )
		{
			trace.lgTrace = true;
		}
		else
			/* trace convergence entered = but with negative flag = make positive,
			 * abs and not mult by -1 since may trigger more than one time */
			trace.nTrConvg = abs( trace.nTrConvg );

		fprintf( ioQQQ, " IterRestart called.\n" );
	}
	else
	{
		trace.lgTrace = false;
	}

	/* zero secondary suprathermals variables for first ionization attem */
	for( nelem=ipHYDROGEN; nelem<LIMELM; ++nelem )
	{
		for( ion=0; ion<nelem+1; ++ion )
		{
			secondaries.csupra[nelem][ion] = supsav[nelem][ion];
		}
	}
	secondaries.x12tot = secondaries.x12sav;
	secondaries.HeatEfficPrimary = secondaries.hetsav;
	secondaries.SecIon2PrimaryErg = secondaries.savefi;
	for( nelem=0; nelem<LIMELM; ++nelem)
	{
		for( ion=0; ion<nelem+1; ++ion )
		{
			ionbal.CompRecoilHeatRate[nelem][ion] = ionbal.CompRecoilHeatRateSave[nelem][ion];
			ionbal.CompRecoilIonRate[nelem][ion] = ionbal.CompRecoilIonRateSave[nelem][ion];
		}
	}

	wind.lgVelPos = true;
	wind.AccelMax = 0.;
	wind.AccelAver = 0.;
	wind.acldr = 0.;
	wind.windv = wind.windv0;

	thermal.nUnstable = 0;
	thermal.lgUnstable = false;

	pressure.pbeta = 0.;
	pressure.RadBetaMax = 0.;
	pressure.lgPradCap = false;
	pressure.lgPradDen = false;
	/* this flag will say we hit the sonic point */
	pressure.lgSonicPoint = false;
	pressure.lgStrongDLimbo = false;

	pressure.PresInteg = 0.;
	pressure.pinzon = 0.;
	pressure.PresIntegElecThin = 0.;

	pressure.RhoGravity_dark = 0.;
	pressure.RhoGravity_self = 0.;
	pressure.RhoGravity_external = 0.;
	pressure.RhoGravity = 0.;
	pressure.IntegRhoGravity = 0.;

	EdenChange( edsav );
	dense.EdenHCorr = edsav;
	dense.EdenHCorr_f = (realnum)dense.EdenHCorr;
	dense.EdenTrue = edsav;

	for( nelem=ipHYDROGEN; nelem < LIMELM; nelem++ )
	{
		/* reset molecular densities */
		dense.SetGasPhaseDensity( nelem, gas_phase_save[nelem] );
		dense.IonLow[nelem] = IonLowSave[nelem];
		dense.IonHigh[nelem] = IonHighSave[nelem];
	}
	/*fprintf(ioQQQ,"DEBUG IterRestart gas_phase set to save  hden %.4e\n",
		dense.gas_phase[0]);*/

	for( long ipISO=ipH_LIKE; ipISO<NISO; ++ipISO )
	{
		for( long nelem=ipISO; nelem < LIMELM; ++nelem )
		{
			iso_sp[ipISO][nelem].Reset();
		}
	}

	for( ipISO=ipH_LIKE; ipISO<NISO; ++ipISO )
	{
		for( nelem=ipISO; nelem < LIMELM; nelem++ )
		{
			if( dense.lgElmtOn[nelem] )
			{
				for( n=ipH1s; n < iso_sp[ipISO][nelem].numLevels_max; n++ )
				{
					iso_sp[ipISO][nelem].fb[n].ConOpacRatio = HOpacRatSav[ipISO][nelem][n];
					iso_sp[ipISO][nelem].st[n].Pop() = hnsav[ipISO][nelem][n];
				}
			}
		}
	}

	if( trace.lgTrace && trace.lgNeBug )
	{
		fprintf( ioQQQ, "     EDEN set to%12.4e by IterRestart.\n", 
		  dense.eden );
	}

	for( nelem=ipHYDROGEN; nelem < LIMELM; nelem++ )
	{
		for( ion=0; ion < (nelem + 2); ion++ )
		{
			dense.xIonDense[nelem][ion] = xIonFsave[nelem][ion];
		}
		for( i=0; i < LIMELM; i++ )
		{
			thermal.setHeating(nelem,i, HeatSave[nelem][i]);
		}
	}

	deut.xIonDense[0] = deutDenseSave0;
	deut.xIonDense[1] = deutDenseSave1;

	GrainRestartIter();

	rfield.resetCoarseTransCoef();

	/* continuum was saved in flux_total_incident */
	for( i=0; i < rfield.nflux_with_check; i++ )
	{
		/* time-constant part of beamed continuum */
		rfield.flux_beam_const[i] = rfield.flux_beam_const_save[i];
		/* continuum flux_time_beam_save has the initial value of the
		 * time-dependent beamed continuum */
		rfield.flux_beam_time[i] = rfield.flux_time_beam_save[i]*
			rfield.time_continuum_scale;

		if( cosmology.lgDo && cosmology.redshift_current < cosmology.redshift_start )
		{
			double slope_ratio;
			double fac_old = TE1RYD*rfield.anu(i)/(CMB_TEMP*(1. + cosmology.redshift_current - cosmology.redshift_step ));
			double fac_new = TE1RYD*rfield.anu(i)/(CMB_TEMP*(1. + cosmology.redshift_current));

			if( fac_old > log10(DBL_MAX) )
			{
				slope_ratio = 0.;
			}
			else if( fac_new > log10(DBL_MAX) )
			{
				slope_ratio = 0.;
			}
			else if( fac_old > 1.e-5 )
			{
				slope_ratio = (exp(fac_old) - 1.)/(exp(fac_new) - 1.);
			}
			else
			{
				slope_ratio = (fac_old*(1. + fac_old/2.))/(fac_new*(1. + fac_new/2.));
			}

			rfield.flux_isotropic_save[i] *= (realnum)slope_ratio; 
		}
		
		rfield.flux_isotropic[i] = rfield.flux_isotropic_save[i];

		rfield.flux[0][i] = rfield.flux_isotropic[i] + rfield.flux_beam_time[i] +
			rfield.flux_beam_const[i];
		rfield.flux_total_incident[0][i] = rfield.flux[0][i];

		rfield.SummedDif[i] = rfield.SummedDifSave[i];
		rfield.SummedCon[i] = rfield.flux[0][i] + rfield.SummedDif[i];
		rfield.SummedOcc[i] = rfield.SummedCon[i]*rfield.convoc[i];

		rfield.OccNumbIncidCont[i] = rfield.flux[0][i]*rfield.convoc[i];
		rfield.otscon[i] = rfield.otssav[i][0];
		rfield.otslin[i] = rfield.otssav[i][1];
		rfield.ConInterOut[i] = 0.;
		rfield.OccNumbDiffCont[i] = 0.;
		rfield.OccNumbContEmitOut[i] = 0.;
		rfield.outlin[0][i] = 0.;
		rfield.outlin_noplot[i] = 0.;
		rfield.ConOTS_local_OTS_rate[i] = 0.;
		rfield.ConOTS_local_photons[i] = 0.;
		rfield.DiffuseEscape[i] = 0.;

		opac.opacity_abs[i] = opac.opacity_abs_savzon1[i];
		opac.OldOpacSave[i] = opac.opacity_abs_savzon1[i];
		opac.opacity_sct[i] = opac.opacity_sct_savzon1[i];
		opac.albedo[i] = 
			opac.opacity_sct[i]/MAX2(1e-30,opac.opacity_sct[i] + opac.opacity_abs[i]);
		opac.tmn[i] = 1.;
		/* >>chng 99 dec 04, having exactly 1 for first zone caused discontinuity
		 * for heating in very high T models in func_map.in.  zone 1 and 2 were 20% different,
		 * since tau in is 1e-20, e2 is 0.9999, and so some H ots
		opac.ExpmTau[i] = 1.;
		opac.E2TauAbsFace[i] = 1.;*/
		/* >>chng 99 dec 04, having exactly 1 for first zone caused discontinuity
		 * for heating in very high T models in func_map.in.  zone 1 and 2 were 20% different,
		 * since tau in is 1e-20, e2 is 0.9999, and so some H ots
		 * these were not here at all*/
		/* attenuation of flux by optical depths IN THIS ZONE 
		 * DirectionalCosin is 1/COS(theta), is usually 1, reset with illumination command,
		 * option for illumination of slab at an angle */
		opac.ExpZone[i] = sexp(opac.opacity_abs[i]*radius.drad/2.*geometry.DirectionalCosin);

		/* e(-tau) in inward direction, up to illuminated face */
		opac.ExpmTau[i] = (realnum)opac.ExpZone[i];

		/* e2(tau) in inward direction, up to illuminated face */
		opac.E2TauAbsFace[i] = (realnum)e2(opac.TauAbsFace[i]);
		rfield.reflin[0][i] = 0.;
		rfield.ConEmitReflec[0][i] = 0.;
		rfield.ConEmitOut[0][i] = 0.;
		rfield.ConEmitOutUnatt[0][i] = 0.;
		rfield.ConRefIncid[0][i] = 0.;

		/* escape in the outward direction
		 * on second and later iterations define outward E2 */
		if( iteration > 1 )
		{
			/* e2 from current position to outer edge of shell */
			realnum tau = MAX2(SMALLFLOAT , opac.TauAbsTotal[i] - opac.TauAbsFace[i] );
			opac.E2TauAbsOut[i] = (realnum)e2( tau );
			ASSERT( opac.E2TauAbsOut[i]>=0. && opac.E2TauAbsOut[i]<=1. );
		}
		else
			opac.E2TauAbsOut[i] = 1.;

	}

	/* update continuum */
	RT_OTS_Update(&SumOTS);

	thermal.FreeFreeTotHeat = 0.;
	atoms.p2nit = p2nit;
	atoms.d5200r = d5200r;

	if( called.lgTalk )
	{
		fprintf( ioQQQ, "\f\n          Start Iteration Number %ld   %75.75s\n\n\n", 
			 iteration, input.chTitle.c_str() );
	}

	ASSERT(lgElemsConserved());
	return;
}

/* do some work with ending the iteration */
void IterEnd(void)
{

	DEBUG_ENTRY( "IterEnd()" );

	/* give indication of geometry */
	double fac = radius.depth/radius.rinner;
	if( fac < 0.1 )
	{
		geometry.lgGeoPP = true;
	}
	else
	{
		geometry.lgGeoPP = false;
	}

	if( ! dynamics.isInitialRelaxIteration( iteration ) && dynamics.lgTimeDependentStatic 
		&& strncmp(rfield.chCumuType,"NONE", sizeof(rfield.chCumuType)) != 0)
	{
		// report cumulative lines per unit mass rather than flux (per unit
		// area), so total is meaningful when density isn't constant

		double cumulative_factor;

		if (strncmp(rfield.chCumuType,"MASS", sizeof(rfield.chCumuType)) == 0)
		{
			cumulative_factor = (dynamics.timestep/ colden.TotMassColl);
		}
		else if (strncmp(rfield.chCumuType,"FLUX", sizeof(rfield.chCumuType)) == 0)
		{
			cumulative_factor = dynamics.timestep;
		}
		else
		{
			fprintf( ioQQQ, " PROBLEM IterEnd called with insane cumulative type, %4.4s\n", 
						rfield.chCumuType );		
			TotalInsanity();
		}

		// save cumulative lines
		for( long n=0; n<LineSave.nsum; ++n )
		{
			LineSave.lines[n].SumLineAccum(cumulative_factor);
		}

		// save cumulative continua
		rfield.flux.accumulate_flux( cumulative_factor, dynamics.timestep, dynamics.time_elapsed );
		rfield.ConEmitReflec.accumulate_flux( cumulative_factor, dynamics.timestep, dynamics.time_elapsed );
		rfield.ConEmitOut.accumulate_flux( cumulative_factor, dynamics.timestep, dynamics.time_elapsed );
		rfield.ConRefIncid.accumulate_flux( cumulative_factor, dynamics.timestep, dynamics.time_elapsed );
		rfield.flux_total_incident.accumulate_flux( cumulative_factor, dynamics.timestep, dynamics.time_elapsed );
		rfield.reflin.accumulate_flux( cumulative_factor, dynamics.timestep, dynamics.time_elapsed );
		rfield.outlin.accumulate_flux( cumulative_factor, dynamics.timestep, dynamics.time_elapsed );
	}
	
	
	/* save previous iteration's results */
	struc.nzonePreviousIteration = nzone;
	for( long i=0; i<struc.nzonePreviousIteration; ++i )
	{
		struc.depth_last[i] = struc.depth[i];
		struc.drad_last[i] = struc.drad[i];
	}

	/* all continue were attenuated in last zone in radius_increment to represent the intensity
	 * in the middle of the next zone - this was too much since we now want
	 * intensity emergent from outer edge of last zone */
	for( long i=0; i < rfield.nflux; i++ )
	{
		double tau = opac.opacity_abs[i]*radius.drad_x_fillfac/2.*geometry.DirectionalCosin;
		{
			enum{DEBUG_LOC=false};
			if( DEBUG_LOC)
			{
				fprintf(ioQQQ,"i=%li opac %.2e \n", i, tau );
			}
		}
		fac = sexp( tau );

		/* >>chng 02 dec 14, add first test to see whether product of ratio is within
		 * range of a float - ConInterOut can be finite and fac just above a small float,
		 * so ratio exceeds largest size of a float */
		/*if( fac > SMALLFLOAT )*/
		if( (realnum)(fac/SDIV(rfield.ConInterOut[i]))>SMALLFLOAT && fac > SMALLFLOAT )
		{
			rfield.ConInterOut[i] /= (realnum)fac;
			rfield.outlin[0][i] /= (realnum)fac;
			rfield.outlin_noplot[i] /= (realnum)fac;
		}
	}

	/* remember thickness of previous iteration */
	iterations.StopThickness[iteration-1] = radius.depth;
	return;
}
