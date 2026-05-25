/* This file is part of Cloudy and is copyright (C)1978-2025 by Gary J. Ferland and
 * others.  For conditions of distribution and use see copyright notice in license.txt */
/*SaveDo produce save output during calculation,
 * chTime is 'MIDL' during calculation, 'LAST' at the end */
/*SaveLineStuff save optical depths or source functions for all transferred lines */
/*Save1Line called by SaveLineStuff to produce output for one line */
/*SaveLineIntensity produce the 'save lines intensity' output */
/* save h emission, for AGN3 chapter 4, routine is below */
/*SaveResults1Line do single line of output for the save results and save line intensity commands */
/* the number of emission lines across one line of printout */
/*SaveSpecial generate output for the save special command */
/*SaveResults save results from save results command */
/*SaveResults1Line do single line of output for the save results and save line intensity commands */
/*SaveGaunts called by save gaunts command to output gaunt factors */
/*FindStrongestLineLabels find strongest lines contributing to point in continuum energy mesh, output in some save commands */
#include "cddefines.h"
#include "cddrive.h"
#include "mean.h"
#include "taulines.h"
#include "struc.h"
#include "iso.h"
#include "hyperfine.h"
#include "rt.h"
#include "magnetic.h"
#include "hydrogenic.h"
#include "secondaries.h"
#include "grains.h"
#include "grainvar.h"
#include "lines.h"
#include "dynamics.h"
#include "colden.h"
#include "ionbal.h"
#include "yield.h"
#include "prt.h"
#include "iterations.h"
#include "heavy.h"
#include "conv.h"
#include "geometry.h"
#include "called.h"
#include "helike.h"
#include "opacity.h"
#include "phycon.h"
#include "timesc.h"
#include "radius.h"
#include "monitor_results.h"
#include "thermal.h"
#include "wind.h"
#include "hmi.h"
#include "pressure.h"
#include "elementnames.h"
#include "ipoint.h"
#include "hcmap.h"
#include "input.h"
#include "save.h"
#include "warnings.h"
#include "grid.h"
#include "atmdat.h"
#include "h2.h"
#include "gammas.h"
#include "mole.h"
#include "rfield.h"
#include "doppvel.h"
#include "freebound.h"
#include "dense.h"
#include "atmdat_gaunt.h"
#include "generic_state.h"

enum cont_type { CT_INCI,       // incident continuum, not attenuated
		 CT_OUTW_INCI,  // incident continuum, attenuated through cloud
		 CT_REFL_INCI,  // incident continuum, reflected towards observer
		 CT_OUTW_DIFF,  // diffuse emission, attenuated through cloud
		 CT_OUTW_DIFF_UNATT, // diffuse emission, summed over all zones, NO attenuation
		 CT_REFL_DIFF,  // diffuse emission, reflected towards observer
		 CT_OUTW_LIN,   // line emission, attenuated through cloud
		 CT_REFL_LIN,   // line emission, reflected towards observer
		 CT_GRN_SIL,    // diffuse emission from silicate grains
		 CT_GRN_GRA,    // diffuse emission from graphite grains
		 CT_GRN_TOT };  // diffuse emission from all grains
		 

// implements the absorption option on the
// set save line width command
inline double PrettyTransmission(long j, double transmission)
{
	if( save.ResolutionAbs < 0_r )
		// option to conserve energy
		return transmission;
	else
	{
		double corr = double(save.ResolutionAbs)*rfield.widflx(j)/rfield.anu(j);
		return max(0., 1. - (1.-transmission)*corr);
	}
}

// flxCell: calculate flux of specified continuum type at frequency anu(j)
inline double flxCell(long j, long nEmType, cont_type ct, bool lgForceConserve = false,
		      bool lgPrtIsotropicCont = true, const realnum *trans_coef_total = NULL)
{
	double val, factor = rfield.anu2(j)*EN1RYD/rfield.widflx(j);
	// a value < 0. indicates that energy should be conserved
	double resfactor = ( save.Resolution < 0_r || lgForceConserve ) ? factor :
		rfield.anu(j)*double(save.Resolution)*EN1RYD;
	switch( ct )
	{
	case CT_INCI:
		val = rfield.flux_total_incident.get_flux(nEmType, j) * factor*radius.PI4_rinner_sq;
		break;
	case CT_OUTW_INCI:
		ASSERT( trans_coef_total != NULL );
		val = flux_correct_isotropic( lgPrtIsotropicCont, nEmType, j ) * factor*radius.PI4_Radius_sq;
		if( lgForceConserve )
			val *= double(trans_coef_total[j]);
		else
			val *= PrettyTransmission( j, double(trans_coef_total[j]) );
		break;
	case CT_REFL_INCI:
		val = rfield.ConRefIncid.get_flux(nEmType, j) * double(geometry.covgeo) *
			factor*radius.PI4_rinner_sq;
		break;
	case CT_OUTW_DIFF:
		val = rfield.ConEmitOut.get_flux(nEmType, j) * double(geometry.covgeo) * factor*radius.PI4_Radius_sq;
		break;
	case CT_OUTW_DIFF_UNATT:
		val = rfield.ConEmitOutUnatt.get_flux(nEmType, j) * double(geometry.covgeo) * factor*radius.PI4_Radius_sq;
		break;
	case CT_REFL_DIFF:
		val = rfield.ConEmitReflec.get_flux(nEmType, j) * double(geometry.covgeo) * factor*radius.PI4_rinner_sq;
		break;
	case CT_OUTW_LIN:
		val = rfield.outlin.get_flux(nEmType, j) * double(geometry.covgeo) * resfactor*radius.PI4_Radius_sq;
		break;
	case CT_REFL_LIN:
		val = rfield.reflin.get_flux(nEmType, j) * double(geometry.covgeo) * resfactor*radius.PI4_rinner_sq;
		break;
	case CT_GRN_SIL:
		val = double(gv.SilicateEmission[j]*geometry.covgeo)*factor*radius.PI4_Radius_sq;
		break;
	case CT_GRN_GRA:
		val = double(gv.GraphiteEmission[j]*geometry.covgeo)*factor*radius.PI4_Radius_sq;
		break;
	case CT_GRN_TOT:
		val = double(gv.GrainEmission[j]*geometry.covgeo)*factor*radius.PI4_Radius_sq;
		break;
	default:
		TotalInsanity();
	}
	return val;
}

/* This routine returns the spectrum needed for Keith Arnaud's XSPEC X-Ray
 * analysis code. It should be called after cdDrive has successfully computed a
 * model. The calling routine must ensure that the vectors have enough space to
 * store the resulting spectrum, given the bounds and energy resolution */
void cdSPEC2( 
	/* option - the type of spectrum to be returned (in photons/cm^2/s/bin)
	 *
	 * 0	the total continuum, all components outward and reflected
	 *
	 * 1	the incident continuum
	 *
	 * 2	the attenuated incident continuum
	 * 3	the reflected incident continuum
	 *
	 * 4	diffuse emission, lines + continuum, outward
	 * 5	diffuse emission, lines + continuum, reflected
	 *
	 * 6	diffuse continuous emission, outward
	 * 7	diffuse continuous emission, reflected
	 *
	 * 8	total transmitted, incident + lines and continuum
	 * 9	total reflected, incident + lines and continuum
	 *
	 *10	exp(-tau) to the illuminated face */
	int nOption,

	/* the returned spectrum, should have rfield.nflux elements */
	realnum ReturnedSpectrum[] )
{
	DEBUG_ENTRY( "cdSPEC2()" );

	const realnum *trans_coef_total = ( nOption == 0 || nOption == 2 || nOption == 8 || nOption == 10 ) ?
		rfield.getCoarseTransCoef() : NULL;

	for( long j = 0; j < rfield.nflux; j++ )
	{
		double returnval;
		if( nOption == 0 )
		{
			/* the attenuated incident continuum */
			double flxatt = flxCell(j, 0, CT_OUTW_INCI, true, true, trans_coef_total);

			/* the outward emitted continuum */
			double conem = flxCell(j, 0, CT_OUTW_DIFF, true) +
				flxCell(j, 0, CT_OUTW_LIN, true);

			/* the reflected continuum */
			double flxref = flxCell(j, 0, CT_REFL_INCI, true) +
				flxCell(j, 0, CT_REFL_DIFF, true) +
				flxCell(j, 0, CT_REFL_LIN, true);

			returnval = flxatt + conem + flxref;
		}
		else if( nOption == 1 )
		{
			/* this is the incident continuum, col 2 of save continuum command */
			returnval = flxCell(j, 0, CT_INCI, true);
		}
		else if( nOption == 2 )
		{
			/* the attenuated transmitted continuum, no diffuse emission,
			 * col 3 of save continuum command */
			returnval = flxCell(j, 0, CT_OUTW_INCI, true, true, trans_coef_total);
		}
		else if( nOption == 3 )
		{
			/* reflected incident continuum, col 6 of save continuum command */
			returnval = flxCell(j, 0, CT_REFL_INCI, true);
		}
		else if( nOption == 4 )
		{
			/* all outward diffuse emission */
			returnval = flxCell(j, 0, CT_OUTW_DIFF, true) +
				flxCell(j, 0, CT_OUTW_LIN, true);
		}
		else if( nOption == 5 )
		{
			/* all reflected diffuse emission */
			returnval = flxCell(j, 0, CT_REFL_DIFF, true) +
				flxCell(j, 0, CT_REFL_LIN, true);
		}
		else if( nOption == 6 )
		{
			/* all outward line emission */
			returnval = flxCell(j, 0, CT_OUTW_LIN, true);
		}
		else if( nOption == 7 )
		{
			/* all reflected line emission */
			returnval = flxCell(j, 0, CT_REFL_LIN, true);
		}
		else if( nOption == 8 )
		{
			/* total transmitted continuum */
			returnval = flxCell(j, 0, CT_OUTW_INCI, true, true, trans_coef_total) +
				flxCell(j, 0, CT_OUTW_DIFF, true) +
				flxCell(j, 0, CT_OUTW_LIN, true);
		}
		else if( nOption == 9 )
		{
			/* total reflected continuum */
			returnval = flxCell(j, 0, CT_REFL_INCI, true) +
				flxCell(j, 0, CT_REFL_DIFF, true) +
				flxCell(j, 0, CT_REFL_LIN, true);
		}
		else if( nOption == 10 )
		{
			/* this is exp(-tau) */
			/* This is the TOTAL attenuation in both the continuum and lines.  
			 * Jon Miller discovered that the line attenuation was missing in 07.02 */
			ASSERT( trans_coef_total != NULL );
			returnval = double(opac.ExpmTau[j]*trans_coef_total[j]);
		}
		else
		{
			fprintf(ioQQQ," cdSPEC called with impossible nOption (%i)\n", nOption);
			cdEXIT(EXIT_FAILURE);
		}

		// convert to photons/cm^2/s/bin and normalize on inner radius
		if( nOption != 10 )
			returnval /= rfield.anu2(j)*EN1RYD*radius.PI4_rinner_sq/rfield.widflx(j);

		ReturnedSpectrum[j] = realnum(returnval);
		ASSERT( ReturnedSpectrum[j] >= 0.f );
	}
}

// find strongest lines contributing to point in continuum energy mesh, output in some save commands
STATIC void FindStrongestLineLabels( void )
{
	long low_index=0;
	long high_index=0;
	long j_min = 0;
	double MaxFlux = 0.;
	long ipMaxFlux = 0;
	long j = 0;

	ASSERT( LineSave.ipass==1 );

	while( rfield.anumax(j_min) < RYDLAM/LineSave.lines[LineSave.SortWL[0]].wavlVac() )
		j_min++;

	for( j=0; j<rfield.nflux; j++ )
	{
		if( j < j_min )
		{
			rfield.chLineLabel[j] = "    ";
			continue;
		}

		ASSERT( LineSave.lines[LineSave.SortWL[low_index]].wavlVac() != 0. );

		while( RYDLAM/LineSave.lines[LineSave.SortWL[low_index]].wavlVac() < rfield.anumin(j) && low_index < LineSave.nsum-1 )
		{
			low_index++;
			if( LineSave.lines[LineSave.SortWL[low_index]].wavlVac() == 0. )
			{
				// hit the end of real wavelengths.  Pad rest of labels with spaces
				for( long j1=j; j1<rfield.nflux; j1++ )
					rfield.chLineLabel[j1] = "    ";
				return;
			}
		}

		high_index = low_index;
		ASSERT( LineSave.lines[LineSave.SortWL[high_index]].wavlVac() != 0. );

		while( RYDLAM/LineSave.lines[LineSave.SortWL[high_index]].wavlVac() < rfield.anumax(j) && high_index < LineSave.nsum-1 )
		{
			high_index++;
			if( LineSave.lines[LineSave.SortWL[high_index]].wavlVac() == 0. )
			{
				high_index--;
				break;
			}
		}
		// while loop found first one greater than j bin, decrement again to get back into j bin
		high_index--;

		ASSERT( LineSave.lines[LineSave.SortWL[low_index]].wavlVac() > 0. );
		ASSERT( LineSave.lines[LineSave.SortWL[high_index]].wavlVac() > 0. );
		ASSERT( RYDLAM/LineSave.lines[LineSave.SortWL[low_index]].wavlVac() >= rfield.anumin(j) );
		ASSERT( RYDLAM/LineSave.lines[LineSave.SortWL[high_index]].wavlVac() <= rfield.anumax(j) );

		MaxFlux = 0.;
		ipMaxFlux = 0;

		for( long k = low_index; k <= high_index; k++ )
		{
			size_t ipLine = LineSave.SortWL[k];
			if( LineSave.lines[ipLine].isCollisional() ||
				 LineSave.lines[ipLine].isHeat() ||
				 LineSave.lines[ipLine].isPump() ||
				 LineSave.lines[ipLine].isNInu() ||
				 LineSave.lines[ipLine].isNFnu() ||
				 LineSave.lines[ipLine].isInwardTotal() ||
				 LineSave.lines[ipLine].isInwardContinuum() ||
				 LineSave.lines[ipLine].isInward() ||
				 LineSave.lines[ipLine].isCaseA() ||
				 LineSave.lines[ipLine].isCaseB() ||
				 LineSave.lines[ipLine].isPhoPlus() ||
				 LineSave.lines[ipLine].isPcon() ||
				 LineSave.lines[ipLine].isQH() ||
				 LineSave.lines[ipLine].isUnit() )
				continue;

			if( LineSave.lines[ipLine].SumLine(0) > MaxFlux )
			{
				MaxFlux = LineSave.lines[ipLine].SumLine(0);
				ipMaxFlux = k;
			}
		}

		/* line	label */
		if( ipMaxFlux > 0 )
			rfield.chLineLabel[j] = LineSave.lines[LineSave.SortWL[ipMaxFlux]].chALab();
	}

	return;
}

// index for loop over series of SAVE commands, available across file
STATIC long int ipPun;

/** SAVE command has option LOG to print log quantities as in <= C13 */
double PrtLogLin( double value )
{
	if( save.lgPrtOldStyleLogs[ipPun] )
		return log10( SDIV(value) );
	else
		return value;
}

/*SaveLineResults do single line of output for the save results and save line intensity commands */
/* the number of emission lines across one line of printout */
namespace
{

	static const int LINEWIDTH = 6;
	class SaveLineResults
	{
		long ipLine;
		const LinSv *m_lines[LINEWIDTH];
		FILE *m_ioPUN;
		int m_typ;
	public:
		void save(const LinSv *line);
		SaveLineResults(FILE *ioPUN, int typ)
			{
				m_ioPUN = ioPUN;
				ipLine = 0;
				m_typ = typ;
			}
		void flush()
			{
				if( ipLine > 0 )
				{
					/* this is an option to print many emission lines across an output line,
					 * the array option, or a single column of numbers, the column option
					 * that appears on the "save results" and "save intensity" commands
					 */
					/* usual array 6 wide */
					for( long i=0; i < ipLine; i++ )
					{
						fprintf( m_ioPUN, " ");
						m_lines[i]->prt(m_ioPUN);
						fprintf( m_ioPUN,"\t%.3e", m_lines[i]->SumLine(m_typ) );
						/* >>chng 02 apr 24, do not print type */
						/* single column for input into data base */
						if( strcmp(::save.chPunRltType,"column") == 0 )				
							fprintf( m_ioPUN, "\n" );
					}
					if( strcmp(::save.chPunRltType,"array ") == 0 )
						fprintf( m_ioPUN, " \n" );
				}
				ipLine = 0;
			}
	};	
	
	int getEmType(int ipPun)
	{
		DEBUG_ENTRY( "getEmType()" );
		int nEmType = (int)save.punarg[ipPun][0];
		ASSERT( nEmType==0 || nEmType==1 );
		
		if (nEmType == 1 && strncmp(rfield.chCumuType,"NONE",4) == 0)
		{
			fprintf(ioQQQ," Must type 'set cumulative' before using 'save cumulative' output\n");
			cdEXIT(EXIT_FAILURE);
		}

		return nEmType;
	}
}

/*SaveGaunts called by save gaunts command to output gaunt factors */
STATIC void SaveGaunts(FILE* ioPUN);

/*SaveResults save results from save results command */
/*SaveResults1Line do single line of output for the save results and save line intensity commands */
STATIC void SaveResults(FILE* ioPUN);

STATIC void SaveLineStuff(
  FILE * ioPUN,
  const char *chJob , 
  realnum xLimit);

/* save h emission, for chapter 4, routine is below */
STATIC void AGN_Hemis(FILE *ioPUN );

/*SaveLineIntensity produce the 'save lines intensity' output */
STATIC void SaveLineIntensity(FILE * ioPUN , long int ipPun, realnum Threshold);

char *chDummy;

void SaveDo(
	/* chTime is null terminated 4 char string, either "MIDL" or "LAST" */
	const char *chTime) 
{
	long int
	  i,
	  j;

	DEBUG_ENTRY( "SaveDo()" );

	/* 
	 * the "last" option on save command, to save on last iteration,
	 * is parsed at the top of the loop in only one place.  
	 * no further action is needed at all for save last to work
	 * ok throughout this routine 
	 */

	/* 
	 * each branch can have a test whether chTime is or is not "LAST"
	 *
	 * if( lgLastOnly )  <== print after iteration is complete 
	 *
	 * if "LAST" then this is last call to routine after iteration complete
	 * save only if "LAST" when results at end of iteration are needed
	 *
	 * if( ! lgLastOnly )  <== print for every zone 
	 *
	 * test for .not."LAST" is for every zone result, where you do not
	 * want to save last zone twice
	 */

	/* return if no save to do */
	if( save.nsave < 1 )
	{ 
		return;
	}

	/** lgLastOnly true, print after iteration is complete, false, evey zone */
	bool lgLastOnly = (strcmp(chTime,"LAST") == 0);

	// sort line labels if this is last call, this avoids multiple calls if several
	// output options need sorted labels and is safer since labels will be sorted in
	// case new code is added that reports the strong lines.  The disadvantage is that
	// we sort even if the labels are not used
	if( lgLastOnly )
	{
		// sort emission line intensities so strongest lines are reported
		FindStrongestLineLabels();
	}

	for( ipPun=0; ipPun < save.nsave; ipPun++ )
	{
		/* this global variable to remember where in the save stack we are */
		save.ipConPun = ipPun;

		/* used to identify case where no key found */
		bool lgNoHitFirstBranch = false;

		/* iterations.lgLastIt is true if this is last iteration
		 * lgPunLstIter set true if 'last' key occurred on save command
		 * normally is false.  This will skip saving if last set and
		 * this is not last iteration */
		/* IMPORTANT: there is a second, identical if-statement halfway
		 * down this routine. Any changes here should be copied there! */
		const bool lgActive = ( iterations.lgLastIt || !save.lgPunLstIter[ipPun] ||
				( dynamics.lgTimeDependentStatic && dynamics.lgStatic_completed ) );

		if (lgActive)
		{

			if( strcmp(save.chSave[ipPun],"ABUN") == 0 )
			{
				/* save abundances vs depth */
				if( ! lgLastOnly )
				{
					fprintf( save.params[ipPun].ipPnunit, "%.2f", 
						log10(MAX2(SMALLFLOAT,dense.gas_phase[ipHYDROGEN])) );
					for( long nelem=ipHELIUM; nelem < LIMELM; nelem++ )
					{
						/* >>chng 05 feb 03, protect against non-positive abundances,
						 * bug caught by Marcelo Castellanos */
						fprintf( save.params[ipPun].ipPnunit, "\t%.2f", 
						  log10(MAX2(SMALLFLOAT,dense.gas_phase[nelem])) );
					}
					fprintf( save.params[ipPun].ipPnunit, "\n" );
				}
			}

			else if( strcmp(save.chSave[ipPun],"21CM") == 0 )
			{
				/* save information about 21 cm line */
				if( ! lgLastOnly )
				{
					fprintf( save.params[ipPun].ipPnunit, 
					  "%.5e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\n",
					  /* depth, cm */
					  radius.depth_mid_zone,
					  hyperfine.Tspin21cm ,
					  phycon.te ,
					  /* temperature from Lya - 21 cm optical depth ratio */
					  3.84e-7* iso_sp[ipH_LIKE][ipHYDROGEN].trans(ipH2p,ipH1s).Emis().TauCon() /
					  SDIV( HFLines[0].Emis().TauCon() ),
					  /*TexcLine( &iso_sp[ipH_LIKE][ipHYDROGEN].trans(ipH2p,ipH1s) ),*/
					  (*HFLines[0].Lo()).Pop() ,
					  (*HFLines[0].Hi()).Pop() ,
					  OccupationNumberLine( iso_sp[ipH_LIKE][ipHYDROGEN].trans(ipH2p,ipH1s) ),
					  HFLines[0].Emis().TauCon() , 
					  iso_sp[ipH_LIKE][ipHYDROGEN].trans(ipH2p,ipH1s).Emis().TauCon(),
					  HFLines[0].Emis().PopOpc(),
					  /* term in () is density (cm-3) of 1s, so this is n(1s) / Ts */
					  (iso_sp[ipH_LIKE][ipHYDROGEN].st[ipH1s].Pop())/SDIV( hyperfine.Tspin21cm),
					  /* why was above multiplied by this following term? */
					  /* *HFLines[0].EnergyErg/BOLTZMANN/4.,*/
					  HFLines[0].Emis().TauIn(),
					  TexcLine( iso_sp[ipH_LIKE][ipHYDROGEN].trans(ipH2p,ipH1s) ) ,
					  colden.H0_ov_Tspin,
					  /*>>chng 27 mar, GS, integrated 21cm spin temperature*/
					  colden.H0_21cm_lower,
					  colden.H0_21cm_upper,
					  -HFLines[0].EnergyK() / log((colden.H0_21cm_upper/3.)/colden.H0_21cm_lower),
					  1./(iso_sp[ipH_LIKE][ipHYDROGEN].trans(ipH2p,ipH1s).Emis().Pesc()+
						  iso_sp[ipH_LIKE][ipHYDROGEN].trans(ipH2p,ipH1s).Emis().Pdest()) );
				}
			}

			else if( strcmp(save.chSave[ipPun],"AGES") == 0 )
			{
				/* save timescales vs depth */
				if( ! lgLastOnly )
				{
					int ipCO, ipOH;
					ipCO = findspecies("CO")->index;
					ipOH = findspecies("OH")->index;
					fprintf( save.params[ipPun].ipPnunit, "%.5e\t%.2e\t%.2e\t%.2e\t%.2e\t%.2e\n", 
					  /* depth, cm */
					  radius.depth_mid_zone,
					  /* cooling timescale */
					  dense.pden*BOLTZMANN*1.5*phycon.te/ thermal.htot, 
					  /* H2 destruction timescale */
					  timesc.time_H2_Dest_here, 
					  /* CO destruction timescale */
					  1./SDIV((ipCO != -1) ? mole.species[ipCO].snk : 0.), 
					  /* OH destruction timescale */
					  1./SDIV((ipOH != -1) ? mole.species[ipOH].snk : 0.), 
					  /* H recombination timescale */
					  1./(dense.eden*2.90e-10/(phycon.te70*phycon.te10/phycon.te03)) );
				}
			}

			else if( strcmp(save.chSave[ipPun]," AGN") == 0 )
			{
				if( lgLastOnly )
				{
					if( strcmp( save.chSaveArgs[ipPun], "HECS" ) == 0 )
					{
						/* this routine is in helike.c */
						AGN_He1_CS(save.params[ipPun].ipPnunit);
					}
					if( strcmp( save.chSaveArgs[ipPun], "HEMI" ) == 0 )
					{
						/* save h emiss, for chapter 4, routine is below */
						AGN_Hemis(save.params[ipPun].ipPnunit);
					}
					else
					{
						fprintf( ioQQQ, " SaveDo does not recognize flag %4.4s set for AGN save.  This is impossible.\n", 
						  save.chSave[ipPun] );
						ShowMe();
						cdEXIT(EXIT_FAILURE);
					}
				}
			}

			else if( strcmp(save.chSave[ipPun],"IMAG") == 0 )
			{
				// NB NB
				//
				// This command generates images for debugging purposes,
				// and it is executed near the call to the linear algebra
				// function that solves the rate equations for level
				// populations
				//
				// Here we do nothing
				//
				continue;
			}

			else if( strcmp(save.chSave[ipPun],"MONI") == 0 )
			{
				if( lgLastOnly )
				{
					/* save the monitor output */
					lgCheckMonitors(save.params[ipPun].ipPnunit);
				}
			}

			else if( strcmp(save.chSave[ipPun],"AVER") == 0 )
			{
				if( lgLastOnly )
				{
					/* save the averages output */
					save_average( ipPun );
				}
			}

			else if( strncmp(save.chSave[ipPun],"CHA",3) == 0 )
			{
				if( lgLastOnly )
				{
					/* one of the charge transfer options, all in chargtran.c */
					ChargTranPun( save.params[ipPun].ipPnunit , save.chSave[ipPun] );
				}
			}

			else if( strcmp( save.chSave[ipPun],"CHIA") == 0)
			{
				static bool lgRunOnce = true;
				if( lgRunOnce )
				{
					lgRunOnce = false;
					// save chianti collision data in physical units
					int ipLo = 0;
					int ipHi = 0;
					double fupsilon = 0.;
					double initTemp = 3.0;
					double finalTemp = 9.1;
					double stepTemp = 0.2;
					for (int ipSpecies=0; ipSpecies < nSpecies; ++ipSpecies)
					{
						if( dBaseSpecies[ipSpecies].database == "Chianti" )
						{
							fprintf(save.params[ipPun].ipPnunit,"Species\tLo\tHi\tWlAng\tAul\n");
							for( EmissionList::iterator tr=dBaseTrans[ipSpecies].Emis().begin();
								  tr != dBaseTrans[ipSpecies].Emis().end(); ++tr)
							{
								ipLo = tr->Tran().ipLo();
								ipHi = tr->Tran().ipHi();
								fprintf( save.params[ipPun].ipPnunit,"%s\t%i\t%i\t",
										dBaseSpecies[ipSpecies].chLabel,ipLo+1,ipHi+1);
								fprintf( save.params[ipPun].ipPnunit,"%s\t%.5e",tr->Tran().twav().sprt_wl().c_str(),
										 tr->Tran().Emis().Aul() );
								fprintf( save.params[ipPun].ipPnunit,"\n");
							}
							// temperature scale
							fprintf(save.params[ipPun].ipPnunit,"Species\tLo\tHi\t");
							for(double logtemp = initTemp;logtemp < finalTemp;logtemp = logtemp + stepTemp )
							{
								fprintf( save.params[ipPun].ipPnunit,"\t%2.1f",logtemp);
							}
							fprintf( save.params[ipPun].ipPnunit,"\n");
							// and the collision strengths
							for( ipHi = 1; ipHi <dBaseSpecies[ipSpecies].numLevels_max; ++ipHi )
							{
								for( ipLo =0; ipLo < ipHi; ++ipLo )
								{
									fprintf( save.params[ipPun].ipPnunit,"%s\t%i\t%i\t",
											dBaseSpecies[ipSpecies].chLabel,ipLo+1,ipHi+1);
									for(double logtemp = initTemp;logtemp < finalTemp;logtemp = logtemp + stepTemp )
									{
										fupsilon = CHIANTI_Upsilon(ipSpecies, ipELECTRON, ipHi, ipLo, exp10(logtemp));
										fprintf( save.params[ipPun].ipPnunit,"\t%.3e",fupsilon);
									}
									fprintf( save.params[ipPun].ipPnunit,"\n");
								}
							}
						}
					}
				}
			}

			else if( strcmp(save.chSave[ipPun],"COOL") == 0 ||
					  strcmp(save.chSave[ipPun],"EACH") == 0 )
			{
				/* save cooling, routine in file of same name */
				if( ! lgLastOnly )
					CoolSave(save.params[ipPun].ipPnunit, save.chSave[ipPun]);
			}

			else if( strcmp(save.chSave[ipPun],"DOMI") == 0 )
			{
				/* save dominant rates */
				if( ! lgLastOnly )
				{
					molecule *debug_species = findspecies( save.chSpeciesDominantRates[ipPun].c_str() );
					if (debug_species == null_mole)
					{
						fprintf( ioQQQ,"Error in SAVE DOMINANT RATES, species %s not found\n",
									save.chSpeciesDominantRates[ipPun].c_str());
					}
					else
					{
						fprintf( save.params[ipPun].ipPnunit,
									"%e\t%e\t", radius.depth_mid_zone, mole.species[ debug_species->index ].column );
						vector<const molecule*> debug_list;
						debug_list.push_back(debug_species);
						mole_dominant_rates( debug_list, save.params[ipPun].ipPnunit,
													true, save.nLineList[ipPun], 0.0);
					}
				}
			}

			else if( strcmp( save.chSave[ipPun],"CHRT") == 0 ||
				strcmp( save.chSave[ipPun],"CHRC") == 0 )
			{
				bool lgCoef = false;
				if( strcmp( save.chSave[ipPun],"CHRC") == 0 )
					lgCoef = true;

				/* save chemistry rates command */
				if( ! lgLastOnly )
				{
					bool lgHeader, lgData;
					if( save.lgSaveHeader(ipPun) )
					{
						lgHeader = true;
						lgData = false;
						mole_save(save.params[ipPun].ipPnunit,save.optname[ipPun].c_str(),save.chSaveArgs[ipPun],lgHeader,lgData,lgCoef,radius.depth_mid_zone);
						save.SaveHeaderDone(ipPun);
					}
					lgHeader = false;
					lgData = true;
					mole_save(save.params[ipPun].ipPnunit,save.optname[ipPun].c_str(),save.chSaveArgs[ipPun],lgHeader,lgData,lgCoef,radius.depth_mid_zone);
				}
			}

			else if( strncmp(save.chSave[ipPun],"DYN" , 3) == 0 )
			{
				/* save dynamics xxx, information about dynamical solutions */
				if( ! lgLastOnly )
					DynaSave(save.params[ipPun].ipPnunit ,save.chSave[ipPun][3] );
			}

			else if( strcmp(save.chSave[ipPun],"ENTH") == 0 )
			{
				if( ! lgLastOnly )
					fprintf( save.params[ipPun].ipPnunit,
						"%.5e\t%.2e\t%.2e\t%.2e\t%.2e\t%.2e\t%.2e\t%.2e\n",
						radius.depth_mid_zone,
						phycon.EnthalpyDensity,
						phycon.EnergyExcitation,
						phycon.EnergyIonization,
						phycon.EnergyBinding ,
						0.5*POW2(wind.windv)*dense.xMassDensity ,	/* KE */
						5./2.*pressure.PresGasCurr ,				/* thermal plus PdV work */
						magnetic.EnthalpyDensity);						/* magnetic terms */
			}

			else if( strcmp(save.chSave[ipPun],"COMP") == 0 )
			{
				/* Compton energy exchange coefficients */
				if( ! lgLastOnly )
				{
					for( long jj=0; jj<rfield.nflux; jj = jj + save.ncSaveSkip)
					{
						fprintf( save.params[ipPun].ipPnunit, "%10.2e%10.2e%10.2e\n", 
						  rfield.anu(jj), rfield.comup[jj]/rfield.widflx(jj), 
						  rfield.comdn[jj]/rfield.widflx(jj) );
					}
				}
			}

			/* save continuum commands */
			else if( strcmp(save.chSave[ipPun],"CON ") == 0 )
			{
				/* this is the usual "save continuum" case */
				/* >>chng 06 apr 03, add every option to do every zone */
				/* if lgSaveEveryZone is true then nSaveEveryZone must be positive
				 * was init to -1 */
				bool lgPrintThis =false;
				if( save.lgSaveEveryZone[ipPun] )
				{
					/* this branch, every option is on line so want to print every n zone */
					if( ! lgLastOnly )
					{
						/* not last zone - print first and intermediate cases */
						if( nzone==1 )
						{
							lgPrintThis = true;
						}
						else if( nzone%save.nSaveEveryZone[ipPun]==0 )
						{
							lgPrintThis = true;
						}
					}
					else
					{
						/* this is last zone, print only if did not trip on above */
						if( nzone!=1 && nzone%save.nSaveEveryZone[ipPun]!=0 )
						{
							lgPrintThis = true;
						}
					}
				}
				else
				{
					/* this branch, not "every", so only print the last zone */
					if( lgLastOnly )
						lgPrintThis = true;
				}
				ASSERT( !save.lgSaveEveryZone[ipPun] || save.nSaveEveryZone[ipPun]>0 );
				if( lgPrintThis )
				{
					if( save.lgSaveEveryZone[ipPun] && nzone!=1)
						fprintf( save.params[ipPun].ipPnunit, "%s\n",
							 save.chHashString.c_str() );

					/* option to also print same arrays but for cumulative arrays */
					int nEmType = getEmType(ipPun);

					const realnum *trans_coef_total=rfield.getCoarseTransCoef();
					for( j=0; j<rfield.nflux; j = j+save.ncSaveSkip)
					{
						/* four continua predicted here;
						 * incident, attenuated incident, emitted,
						 * then attenuated incident + emitted, last reflected
						 * reflected continuum is stored relative to inner radius
						 * others are stored for this radius */

						/* NB this code also used in save emitted,
						 * transmitted continuum commands */

						/* the incident continuum, flux_total_incident evaluated at illuminated face */
						double flxin = flxCell(j, nEmType, CT_INCI);

						/* the reflected line emission */
						double flxreflin = flxCell(j, nEmType, CT_REFL_LIN);

						/* the total reflected continuum, evaluated at each zone so at outer radius */
						double flxref = flxCell(j, nEmType, CT_REFL_INCI) +
							flxCell(j, nEmType, CT_REFL_DIFF) + flxreflin;

						/* the attenuated incident continuum, no covering factor since prediction is for view through cloud */
						double flxatt = flxCell(j, nEmType, CT_OUTW_INCI, false,
							save.lgPrtIsotropicCont[ipPun], trans_coef_total);

						/* outward line emission */
						double flxlinem = flxCell(j, nEmType, CT_OUTW_LIN);

						/* the total outward emitted continuum */
						double conem = flxCell(j, nEmType, CT_OUTW_DIFF) + flxlinem;

						/* sum of emitted and transmitted continua */
						double flxtrn = conem + flxatt;

						/* photon energy in appropriate energy or wavelength units */
						fprintf( save.params[ipPun].ipPnunit,"%.5e\t", AnuUnit(rfield.anu(j)) );
						/* incident continuum */
						fprintf( save.params[ipPun].ipPnunit,"%.3e\t", flxin ); 
						/* trans cont */
						fprintf( save.params[ipPun].ipPnunit,"%.3e\t", flxatt ); 
						/* DiffOut cont */
						fprintf( save.params[ipPun].ipPnunit,"%.3e\t", conem ); 
						/* net trans cont */
						fprintf( save.params[ipPun].ipPnunit,"%.3e\t", flxtrn ); 
						/* reflected cont */
						fprintf( save.params[ipPun].ipPnunit,"%.3e\t", flxref ); 
						/* total cont */
						fprintf( save.params[ipPun].ipPnunit,"%.3e\t", flxref + flxtrn );
						/* reflected lines */
						fprintf( save.params[ipPun].ipPnunit,"%.3e\t", flxreflin );
						/* outward lines */
						fprintf( save.params[ipPun].ipPnunit,"%.3e\t", flxlinem );

						fprintf( save.params[ipPun].ipPnunit, "%s\t%s\t", 
						/* line	label */
						  rfield.chLineLabel[j].c_str() ,
						/* cont label*/
						  rfield.chContLabel[j].c_str() );
						/* number of lines within that cell over cell width
						 * save raw continuum has number by itself */
						fprintf( save.params[ipPun].ipPnunit, "%.2f\n", rfield.line_count[j]/rfield.widflx(j)*rfield.anu(j) );
					}
				}
			}

			else if( strcmp(save.chSave[ipPun],"CONC") == 0 )
			{
				/* save incident continuum */
				/* set pointer for possible change in units of energy in continuum
				 * AnuUnit will give anu in whatever units were set with save units */
				if( lgLastOnly )
				{
					/* incident continuum */
					for( j=0; j<rfield.nflux; j = j + save.ncSaveSkip)
					{
						double flxin = flxCell(j, 0, CT_INCI);
						/* >>chng 96 oct 22, format of anu to .5 to resolve energy mesh near 1 Ryd */
						fprintf( save.params[ipPun].ipPnunit, "%.5e\t%.4e\t%.4e\n",
						  AnuUnit(rfield.anu(j)), flxin, rfield.OccNumbIncidCont[j]);
					}
				}
			}

			else if( strcmp(save.chSave[ipPun],"CONG") == 0 )
			{
				/* save emitted grain continuum in optically thin limit */
				if( lgLastOnly )
				{
					/* GrainMakeDiffuse broke out emission into types 
					 * according to matType */
					for( j=0; j<rfield.nflux; j = j + save.ncSaveSkip)
					{
						double fgra = flxCell(j, 0, CT_GRN_GRA);
						double fsil = flxCell(j, 0, CT_GRN_SIL);
						double ftot = flxCell(j, 0, CT_GRN_TOT);
						/* anu is .5e format to resolve energy mesh near 1 Ryd 
						 * AnuUnit gives anu in whatever units were set with units option */
						fprintf( save.params[ipPun].ipPnunit, "%.5e\t%.3e\t%.3e\t%.3e\n", 
							 AnuUnit(rfield.anu(j)) , fgra, fsil, ftot );
					}
				}
			}

			else if( strcmp(save.chSave[ipPun],"CONR") == 0 )
			{
				/* save reflected continuum */
				/* set pointer for possible change in units of energy in continuum
				 * AnuUnit will give anu in whatever units were set with save units */
				if( lgLastOnly )
				{
					if( geometry.lgSphere )
					{
						fprintf( save.params[ipPun].ipPnunit, " Reflected continuum not predicted when SPHERE is set.\n" );
						fprintf( ioQQQ , 
							"\n\n>>>>>>>>>>>>>\n Reflected continuum not predicted when SPHERE is set.\n" );
						cdEXIT(EXIT_FAILURE);
					}

					for( j=0; j<rfield.nflux; j = j + save.ncSaveSkip)
					{
						// a value < 0. indicates that energy should be conserved
						realnum resolution = ( save.Resolution < 0_r ) ?
							rfield.anu(j)/rfield.widflx(j) : save.Resolution;

						/* reflected continuum */
						double flxref = rfield.anu2(j)*((double)rfield.ConRefIncid[0][j]+rfield.ConEmitReflec[0][j])/
						  rfield.widflx(j)*EN1RYD;
						/* reflected lines */
						double fref = rfield.anu(j)*resolution*rfield.reflin[0][j]*EN1RYD;
						double av;
						/* ratio of reflected to incident continuum, the albedo */
						if( rfield.flux_total_incident[0][j] > 1e-25 )
						{
							av = rfield.ConRefIncid[0][j]/rfield.flux_total_incident[0][j];
						}
						else
						{
							av = 0.;
						}
						/* >>chng 96 oct 22, format of anu to .5 to resolve energy mesh near 1 Ryd */
						fprintf( save.params[ipPun].ipPnunit, "%.5e\t%.4e\t%.4e\t%.4e\t%.4e\t%.4s\n", 
						  AnuUnit(rfield.anu(j)), flxref, fref, flxref + fref, 
						  av, rfield.chContLabel[j].c_str() );
					}
				}
			}

			else if( strcmp(save.chSave[ipPun],"CNVE") == 0 )
			{
				/* the save convergence error command */
				if( ! lgLastOnly )
				{
					fprintf( save.params[ipPun].ipPnunit, 
						"%.5e\t%li\t%.4e\t%.4f\t%.4e\t%.4e\t%.3f\t%.4e\t%.4e\t%.4f\n", 
						radius.depth_mid_zone, 
						conv.nPres2Ioniz,
						pressure.PresTotlCurr, 
						pressure.PresTotlError*100., 
						dense.EdenTrue,
						dense.eden,
						(dense.EdenTrue - dense.eden)*100./dense.EdenTrue,
						thermal.htot,
						thermal.ctot,
						(thermal.htot - thermal.ctot)*100./thermal.htot );
				}
			}

			else if( strcmp(save.chSave[ipPun],"CONB") == 0 )
			{
				/* save continuum bins binning */
				/* set pointer for possible change in units of energy in continuum
				 * AnuUnit will give anu in whatever units were set with save units */
				if( ! lgLastOnly )
				{
					for( j=0; j<rfield.nflux_with_check; j = j + save.ncSaveSkip)
					{
						fprintf( save.params[ipPun].ipPnunit, "%14.5e\t%14.5e\t%14.5e\n",
						  AnuUnit(rfield.anu(j)), rfield.anu(j), rfield.widflx(j) );
					}
				}
			}

			else if( strcmp(save.chSave[ipPun],"COND") == 0 )
			{
				ASSERT( fp_equal( save.punarg[ipPun][0] , (realnum)2. ) ||
					fp_equal( save.punarg[ipPun][0] , (realnum)1.) );

				/* save diffuse continuum the local line and continuous emission */
				if( lgLastOnly && 
					fp_equal(save.punarg[ipPun][0] , (realnum)1.) )
				{
					/* this option to save diffuse emission for last zone
					 * as series of columns */
					for( j=0; j<rfield.nflux; j = j+save.ncSaveSkip)
					{
						// a value < 0. indicates that energy should be conserved
						realnum resolution = ( save.Resolution < 0_r ) ?
							rfield.anu(j)/rfield.widflx(j) : save.Resolution;

						double EmisLin = resolution*EN1RYD*
							rfield.DiffuseLineEmission[j]*rfield.anu(j);
						double EmisCon = rfield.ConEmitLocal[nzone][j]*
							rfield.anu2(j)*EN1RYD/rfield.widflx(j); 
						fprintf( save.params[ipPun].ipPnunit, "%.5e\t%.5e\t%.5e\t%.5e\n", 
						  AnuUnit(rfield.anu(j)), 
						  EmisCon ,
						  EmisLin , 
						  EmisCon+EmisLin);
					}
				}
				else if( ! lgLastOnly && 
					fp_equal(save.punarg[ipPun][0] , (realnum)2.) )
				{
					/* diffuse emission per zone, with each a long row */
					static bool lgMustPrintHeader=true;
					if( lgMustPrintHeader )
					{
						lgMustPrintHeader = false;
						fprintf( save.params[ipPun].ipPnunit, "%.5e", 
							AnuUnit(rfield.anu(0)) );
						for( j=1; j<rfield.nflux; j = j+save.ncSaveSkip)
						{
							fprintf( save.params[ipPun].ipPnunit, "\t%.5e", 
								AnuUnit(rfield.anu(j)) );
						}
						fprintf( save.params[ipPun].ipPnunit, "\n" );
					}
					// a value < 0. indicates that energy should be conserved
					realnum resolution = ( save.Resolution < 0_r ) ?
						rfield.anu(0)/rfield.widflx(0) : save.Resolution;
					double EmisLin = resolution*EN1RYD*
						rfield.DiffuseLineEmission[0]*rfield.anu(0);
					double EmisCon = rfield.ConEmitLocal[nzone][0]*
						rfield.anu2(0)*EN1RYD/rfield.widflx(0); 
					fprintf( save.params[ipPun].ipPnunit, "%.5e", 
						EmisCon+EmisLin);
					for( j=1; j<rfield.nflux; j = j+save.ncSaveSkip)
					{
						// a value < 0. indicates that energy should be conserved
						resolution = ( save.Resolution < 0_r ) ?
							rfield.anu(j)/rfield.widflx(j) : save.Resolution;
						double EmisLin = resolution*EN1RYD*
							rfield.DiffuseLineEmission[j]*rfield.anu(j);
						double EmisCon = rfield.ConEmitLocal[nzone][j]*
							rfield.anu2(j)*EN1RYD/rfield.widflx(j); 
						fprintf( save.params[ipPun].ipPnunit, "\t%.5e", 
							EmisCon+EmisLin);
					}
					fprintf( save.params[ipPun].ipPnunit, "\n" );
				}
			}

			else if( strcmp(save.chSave[ipPun],"CODU") == 0 )
			{
				/* save diffuse continuum unattenuated -
				 * cloud-integrated gas-phase nebular continuum emission
				 * (free-bound, free-free, two-photon) summed over all
				 * zones WITHOUT self-absorption (no opac.tmn factor).
				 * Does NOT include line emission or grain thermal
				 * emission. */
				if( lgLastOnly )
				{
					for( j=0; j<rfield.nflux; j = j+save.ncSaveSkip)
					{
						double diffcont_unatt = flxCell(j, 0, CT_OUTW_DIFF_UNATT);
						double diffcont_att = flxCell(j, 0, CT_OUTW_DIFF);
						fprintf( save.params[ipPun].ipPnunit, "%.5e\t%.3e\t%.3e\n",
						  AnuUnit(rfield.anu(j)),
						  diffcont_unatt,
						  diffcont_att );
					}
				}
			}

			else if( strcmp(save.chSave[ipPun],"CONE") == 0 )
			{
				/* save emitted continuum */
				/* set pointer for possible change in units of energy in continuum
				 * AnuUnit will give anu in whatever units were set with save units */
				if( lgLastOnly )
				{
					/* save emitted continuum */
					for( j=0; j<rfield.nflux; j+= save.ncSaveSkip)
					{
						/* this is the reflected component */
						double flxref = flxCell(j, 0, CT_REFL_INCI) + flxCell(j, 0, CT_REFL_DIFF) + flxCell(j, 0, CT_REFL_LIN);

						/* this is the total emission in the outward direction */
						double conem = flxCell(j, 0, CT_OUTW_DIFF) + flxCell(j, 0, CT_OUTW_LIN);

						/* output: photon energy, reflected, outward, total emission
						 *  >>chng 96 oct 22, format of anu to .5e to resolve energy mesh near 1 Ryd */
						fprintf( save.params[ipPun].ipPnunit, "%.5e\t%.3e\t%.3e\t%.3e\t%-*.*s\t%-*.*s\n", 
						  AnuUnit(rfield.anu(j)), 
						  flxref, 
						  conem, 
						  flxref + conem, 
						  NCHLAB-1, NCHLAB-1, 
						  rfield.chLineLabel[j].c_str(), 
						  NCHLAB-1, NCHLAB-1, 
						  rfield.chContLabel[j].c_str()
						   );
					}
				}
			}

			/* save fine continuum command */
			else if( strcmp(save.chSave[ipPun],"CONf") == 0 )
			{
				if( lgLastOnly )
				{
					long nu_hi , nskip;
					if( save.punarg[ipPun][0] > 0. )
						/* possible lower bounds to energy range - 
						 * 0 if not set with range option*/
						j = ipFineCont( save.punarg[ipPun][0] );
					else
						j = 0;

					/* upper limit set with range option */
					if( save.punarg[ipPun][1]> 0. )
						nu_hi = ipFineCont( save.punarg[ipPun][1] );
					else
						nu_hi = rfield.nfine;

					/* number of cells to bring together, default is 10 */
					nskip = (long)save.punarg[ipPun][2];
					nskip = MAX2( 1, nskip );

					do
					{
						vector<int> all_stack_lines;

						// when winds are present, line center in opacity
						// vector is offset from its original position
						// -- rfield.fine_lstack works with original position
						//
						long fine_index_no_wind = j - rfield.ipFineConVelShift;

						realnum sum1 = 0.,
							xnu = 0.;

						for( long jj=0; jj<nskip; ++jj, ++j, ++fine_index_no_wind )
						{
							xnu += rfield.fine_anu[j];
							sum1 += rfield.fine_opt_depth[j];

							if( fine_index_no_wind > 0 )
							{
								auto got = rfield.fine_lstack.find( fine_index_no_wind );
								if( got != rfield.fine_lstack.end() )
								{
									for( auto &lst_ind : got->second )
									{
										all_stack_lines.emplace_back( lst_ind );
									}
								}
							}
						}

						// sort lines in decreasing opacity
						//
						sort( all_stack_lines.begin(),
							all_stack_lines.end(),
							[](int i1, int i2)
							{
								return LineSave.lines[i1].getTransition().Emis().TauInSpecific()
								     > LineSave.lines[i2].getTransition().Emis().TauInSpecific();
							} );

						double transm = sexp(sum1/nskip);
						fprintf( save.params[ipPun].ipPnunit, 
							"%.6e\t%.3e", 
							AnuUnit(xnu/nskip),
							transm );

						static const realnum odep_limit = 0.01;

						for( auto &ind: all_stack_lines )
						{
							TransitionProxy tr = LineSave.lines[ind].getTransition();

							// Report mean optical depths, rather than line center
							// Keeps optical depths consistent with main output
							//
							realnum odep = tr.Emis().TauInSpecific() * SQRTPI;
							if( odep < odep_limit )
								break;

							fprintf( save.params[ipPun].ipPnunit,
								"\t\"%s\"\t%.3e",
								tr.chLabel().c_str(), odep );
						}
						fprintf( save.params[ipPun].ipPnunit, "\n" );
					} while( j < nu_hi );
				}
			}

			else if( strcmp(save.chSave[ipPun],"CONi") == 0 )
			{
				/* save continuum interactions */
				/* set pointer for possible change in units of energy in continuum
				 * AnuUnit will give anu in whatever units were set with save units */

				/* continuum interactions */
				if( ! lgLastOnly )
				{
					long i1;
					/* this is option to set lowest energy */
					if( save.punarg[ipPun][0] <= 0. )
					{
						i1 = 1;
					}
					else if( save.punarg[ipPun][0] < 100. )
					{
						i1 = ipoint(save.punarg[ipPun][0]);
					}
					else
					{
						i1 = (long int)save.punarg[ipPun][0];
					}

					double fref = 0.;
					double fout = 0.;
					double fsum = 0.;
					double sum = 0.;
					double flxin = 0.;

					for( j=i1-1; j < rfield.nflux; j++ )
					{
						fref += rfield.flux[0][j]*opac.opacity_abs[j];
						fout += rfield.otslin[j]*opac.opacity_abs[j];
						fsum += rfield.otscon[j]*opac.opacity_abs[j];
						sum += rfield.ConInterOut[j]*opac.opacity_abs[j];
						flxin += (rfield.outlin[0][j] + rfield.outlin_noplot[j])*opac.opacity_abs[j];
					}
					fprintf( save.params[ipPun].ipPnunit, "%10.2e\t%10.2e\t%10.2e\t%10.2e\t%10.2e\n",
					  fref, fout, fsum, sum, flxin );
				}
			}

			else if( strcmp(save.chSave[ipPun],"CONI") == 0 )
			{
				/* save ionizing continuum */
				/* set pointer for possible change in units of energy in continuum
				 * AnuUnit will give anu in whatever units were set with save units */

				if( save.lgSaveEveryZone[ipPun] || (lgLastOnly) )
				{
					/* this flag will remember whether we have ever printed anything */
					bool lgPrt=false;
					if( save.lgSaveEveryZone[ipPun] )
						fprintf(save.params[ipPun].ipPnunit,"#save every zone %li\n", nzone);

					/* save ionizing continuum command
					 * this is option to set lowest energy,
					 * if no number was entered then this was zero */
					long i1;
					if( save.punarg[ipPun][0] <= 0. )
						i1 = 1;
					else if( save.punarg[ipPun][0] < 100. )
						i1 = ipoint(save.punarg[ipPun][0]);
					else
						i1 = (long int)save.punarg[ipPun][0];

					double sum = 0.;
					for( j=i1-1; j < rfield.nflux; j++ )
					{
						double flxcor = rfield.flux[0][j] + 
						  rfield.otslin[j] + 
						  rfield.otscon[j] + 
						  rfield.ConInterOut[j] +
						  rfield.outlin[0][j] + rfield.outlin_noplot[j];

						sum += flxcor*opac.opacity_abs[j];
					}

					if( sum > 0. )
						sum = 1./sum;
					else
						sum = 1.;

					double fsum = 0.;

					for( j=i1-1; j<rfield.nflux; ++j)
					{
						double flxcor = rfield.flux[0][j] + 
						  rfield.otslin[j] + 
						  rfield.otscon[j] + 
						  rfield.ConInterOut[j]+
						  rfield.outlin[0][j] + rfield.outlin_noplot[j];

						fsum += flxcor*opac.opacity_abs[j];

						/* punched quantities are freq, flux, flux*cross sec,
						 * fraction of total, integral fraction of total */
						double RateInter = flxcor*opac.opacity_abs[j]*sum;

						/* punage(ipPun,2) is lowest interaction rate to consider, def=0.01 (1 percent) */
						/* >>chng 01 nov 22, format to c-friendly */
						if( (RateInter >= save.punarg[ipPun][1]) && (flxcor > SMALLFLOAT) )
						{
							lgPrt = true;
							/* >>chng 96 oct 22, format of anu to 11.5 to resolve energy mesh near 1 Ryd */
							fprintf( save.params[ipPun].ipPnunit, 
								"%li\t%.5e\t%.2e\t%.2e\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2e\t%.2e\t%.4s\t%.4s\n", 
							  j,
							  AnuUnit(rfield.anu(j)), 
							  flxcor, 
							  flxcor*opac.opacity_abs[j] / rfield.widflx(j),
							  rfield.flux[0][j]/flxcor, 
							  rfield.otslin[j]/flxcor, 
							  rfield.otscon[j]/flxcor, 
							  (rfield.outlin[0][j] + rfield.outlin_noplot[j])/flxcor, 
							  rfield.ConInterOut[j]/flxcor, 
							  RateInter, 
							  fsum*sum, 
							  rfield.chLineLabel[j].c_str(), 
							  rfield.chContLabel[j].c_str() );
						}
					}
					if( !lgPrt )
					{
						/* entered logical block but did not print anything */
						fprintf(save.params[ipPun].ipPnunit,
							" SaveDo, the SAVE IONIZING CONTINUUM command "
							"did not find a strongly interacting energy, sum and fsum were %.2e %.2e\n",
							sum,fsum);
						fprintf(save.params[ipPun].ipPnunit,
							" SaveDo, the low-frequency energy was %.5e Ryd\n",
							rfield.anu(i1-1));
						fprintf(save.params[ipPun].ipPnunit,
							" You can reset the threshold for the lowest fractional "
							"interaction to print with the second number of the save command\n"
							" The fraction was %.3f and this was too large.\n",
							save.punarg[ipPun][1]);
					}
				}
			}

			else if( strcmp(save.chSave[ipPun],"CONS") == 0 )
			{
				if( ! lgLastOnly )
				{
					// continuum volume emissivity and opacity as a function of radius
					// command was "save continuum emissivity" */
					if( save.ipEmisFreq[ipPun] < 0 )
						save.ipEmisFreq[ipPun] = ipoint(save.emisfreq[ipPun].Ryd());
					j = save.ipEmisFreq[ipPun]-1;

					fprintf( save.params[ipPun].ipPnunit, 
						 "%.14e\t%.14e\t%.5e\t%.5e\t%.5e\n", 
						 radius.Radius_mid_zone,
						 radius.depth_mid_zone,
						 rfield.anu2(j)*rfield.ConEmitLocal[nzone][j]/rfield.widflx(j)*EN1RYD, 
						 opac.opacity_abs[j], 
						 opac.opacity_sct[j] );
				}
			}

			else if( strcmp(save.chSave[ipPun],"CORA") == 0 )
			{
				/* save raw continuum */
				/* set pointer for possible change in units of energy in continuum
				 * AnuUnit will give anu in whatever units were set with save units */

				if( lgLastOnly )
				{
					/* this option to save all raw ionizing continuum */
					for( j=0;j<rfield.nflux;j = j + save.ncSaveSkip)
					{
						fprintf( save.params[ipPun].ipPnunit, 
							"%.5e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%4.4s\t%4.4s\t",
						  AnuUnit(rfield.anu(j)), 
						  rfield.flux[0][j], 
						  rfield.otslin[j], 
						  rfield.otscon[j], 
						  rfield.reflin[0][j],
						  rfield.ConRefIncid[0][j],
						  rfield.ConEmitReflec[0][j], 
						  rfield.ConInterOut[j],
						  rfield.outlin[0][j],
						  rfield.outlin_noplot[j],
						  rfield.ConEmitOut[0][j],
						  rfield.chLineLabel[j].c_str(), 
						  rfield.chContLabel[j].c_str()
						  );
						/* number of lines within that cell */
						fprintf( save.params[ipPun].ipPnunit, "%li\n", rfield.line_count[j] );
					}
				}
			}

			else if( strcmp(save.chSave[ipPun],"CONT") == 0 )
			{
				/* save transmitted continuum - this is not the main "save continuum"
				 * command - search on "CON " above 
				 * set pointer for possible change in units of energy in continuum
				 * AnuUnit will give anu in whatever units were set with save units */

				if( lgLastOnly )
				{
					fprintf( save.params[ipPun].ipPnunit, "#\n" );
					fprintf( save.params[ipPun].ipPnunit, "%32ld # file format version number\n",
							 VERSION_TRNCON );
					fprintf( save.params[ipPun].ipPnunit, "%s # check 1\n",
							 rfield.mesh_cksum().c_str() );
					union {
						double x;
						uint32 i[2];
					} u;
					u.x = rfield.emm();
					if( cpu.i().big_endian() )
						fprintf( save.params[ipPun].ipPnunit, "%23.8x %8.8x # check 2\n",
								 u.i[0], u.i[1] );
					else
						fprintf( save.params[ipPun].ipPnunit, "%23.8x %8.8x # check 2\n",
								 u.i[1], u.i[0] );
					u.x = rfield.egamry();
					if( cpu.i().big_endian() )
						fprintf( save.params[ipPun].ipPnunit, "%23.8x %8.8x # check 3\n",
								 u.i[0], u.i[1] );
					else
						fprintf( save.params[ipPun].ipPnunit, "%23.8x %8.8x # check 3\n",
								 u.i[1], u.i[0] );
					u.x = rfield.getResolutionScaleFactor();
					if( cpu.i().big_endian() )
						fprintf( save.params[ipPun].ipPnunit, "%23.8x %8.8x # check 4\n",
								 u.i[0], u.i[1] );
					else
						fprintf( save.params[ipPun].ipPnunit, "%23.8x %8.8x # check 4\n",
								 u.i[1], u.i[0] );
					fprintf( save.params[ipPun].ipPnunit, "%32.16e # radius, -1 if not set\n",
							 radius.lgRadiusKnown ? radius.Radius : -1. );
					fprintf( save.params[ipPun].ipPnunit, "%32ld # nflux\n",
							 (rfield.nflux+save.ncSaveSkip-1)/save.ncSaveSkip );
					fprintf( save.params[ipPun].ipPnunit, "#\n" );

					const realnum *trans_coef_total = rfield.getCoarseTransCoef();

					/* this option to save transmitted continuum */
					for( j=0; j < rfield.nflux; j += save.ncSaveSkip )
					{
						/* attenuated incident continuum
						 * >>chng 97 jul 10, remove SaveLWidth from this one only since
						 * we must conserve energy even in lines 
						 * >>chng 07 apr 26 include transmission coefficient */
						double flxatt = flxCell(j, 0, CT_OUTW_INCI, true, save.lgPrtIsotropicCont[ipPun],
												trans_coef_total);

						/* >>chng 00 jan 03, above did not include all contributors.  
						 * Pasted in below from usual
						 * save continuum command */
						/* >>chng 04 jul 15, removed factor of save.SaveLWidth -
						 * this should not be there to conserve energy, as explained in hazy
						 * where command was documented, and in comment above.  caught by PvH */
						/* >>chng 04 jul 23, incorrect use of outlin - before multiplied by an2,
						 * quantity should be photons per Ryd, since init quantity is
						 * photons per cell.  Must div by widflx.  caught by PvH  */
						double conem = flxCell(j, 0, CT_OUTW_DIFF, true) + flxCell(j, 0, CT_OUTW_LIN, true);

						/* always save in intensity units, even when running in luminosity mode */
						double intentrn = (conem + flxatt)/radius.PI4_Radius_sq;

						fprintf( save.params[ipPun].ipPnunit, "%.5e\t%.3e\t%.3e\n",
								 rfield.anu(j), intentrn, trans_coef_total[j] );
					}
				}
			}

			else if( strcmp(save.chSave[ipPun],"CON2") == 0 )
			{
				/* save total two-photon continuum  */
				if( lgLastOnly )
				{
					/* this option to save diffuse continuum */
					for( j=0; j<rfield.nflux; j = j+save.ncSaveSkip)
					{
						fprintf( save.params[ipPun].ipPnunit, "%.5e\t%.5e\t%.5e\n", 
						  AnuUnit(rfield.anu(j)), 
						  rfield.TotDiff2Pht[j]/rfield.widflx(j) , 
						  rfield.TotDiff2Pht[j]*rfield.anu2(j)*EN1RYD/rfield.widflx(j));
					}
				}
			}

			else if( strcmp(save.chSave[ipPun],"DUSE") == 0 )
			{
				/* save grain extinction - includes only grain opacity, not total */
				if( ! lgLastOnly )
				{
					fprintf( save.params[ipPun].ipPnunit, " %.5e\t", 
						radius.depth_mid_zone );

					/* visual extinction of an extended source (like a PDR)*/
					fprintf( save.params[ipPun].ipPnunit, "%.2e\t" , rfield.extin_mag_V_extended);

					/* visual extinction of point source (star)*/
					fprintf( save.params[ipPun].ipPnunit, "%.2e\n" , rfield.extin_mag_V_point);
				}
			}

			else if( strcmp(save.chSave[ipPun],"DUSO") == 0 )
			{
				/* save grain cross sections per hydrogen, cm^2/H */
				if( lgLastOnly )
				{
					for( j=0; j < gv.nflux; j++ )
					{
						double scat;
						fprintf( save.params[ipPun].ipPnunit, 
						  "%.5e\t%.2e\t%.2e\t%.2e\t", 
						  /* photon energy or wavelength */
						  AnuUnit(gv.anu(j)), 
						  /* total cross section per hydrogen cm^2/H, discount forward scattering */
						  gv.dstab0[j] + gv.dstsc0[j], 
						  /* absorption cross section per H */
						  gv.dstab0[j], 
						  /* scatter, with forward discounted */
						  gv.dstsc0[j] );
						/* add together total scattering, discounting 1-g */
						scat = 0.;
						/* sum over all grain species */
						for( size_t nd=0; nd < gv.bin.size(); nd++ )
						{
							scat += gv.bin[nd].pure_sc1[j]*gv.bin[nd].dstAbund;
						}
						/* finally, scattering including effects of forward scattering */
						fprintf( save.params[ipPun].ipPnunit, 
						  "%.2e\t", scat );
						fprintf( save.params[ipPun].ipPnunit, 
						  "%.2e\n", gv.dstsc[j]/SDIV(gv.dstab[j] + gv.dstsc[j]) );
					}
				}
			}

			/* save grain abundance and save grain D/G ratio commands */
			else if( strcmp(save.chSave[ipPun],"DUSA") == 0 ||
				 strcmp(save.chSave[ipPun],"DUSD") == 0 )
			{
				bool lgDGRatio = ( strcmp(save.chSave[ipPun],"DUSD") == 0 );

				/* grain abundance */
				if( ! lgLastOnly )
				{
					/* print grain header first if this has not yet been done */
					if( save.lgSaveHeader(ipPun) )
					{
						fprintf( save.params[ipPun].ipPnunit, "#Depth" );
						for( size_t nd=0; nd < gv.bin.size(); ++nd ) 
							fprintf( save.params[ipPun].ipPnunit, "\t%s", gv.bin[nd].chDstLab );
						fprintf( save.params[ipPun].ipPnunit, "\ttotal\n" );
						save.SaveHeaderDone(ipPun);
					}
					fprintf( save.params[ipPun].ipPnunit, " %.5e", 
						radius.depth_mid_zone );
					/* grain abundance per bin in g/cm^3 */
					double total = 0.;
					for( size_t nd=0; nd < gv.bin.size(); ++nd ) 
					{
						double abund = gv.bin[nd].IntVol*gv.bin[nd].dustp[0]*
							gv.bin[nd].cnv_H_pCM3;
						if( lgDGRatio )
							abund /= dense.xMassDensity;
						fprintf( save.params[ipPun].ipPnunit, "\t%.3e", abund );
						total += abund;
					}
					fprintf( save.params[ipPun].ipPnunit, "\t%.3e\n", total );
				}
			}

			else if( strcmp(save.chSave[ipPun],"DUSP") == 0 )
			{
				/* grain potential */
				if( ! lgLastOnly )
				{
					/* do labels first if this is first zone */
					if( save.lgSaveHeader(ipPun) )
					{
						/* first print string giving grain id */
						fprintf( save.params[ipPun].ipPnunit, "#Depth" );
						for( size_t nd=0; nd < gv.bin.size(); ++nd ) 
							fprintf( save.params[ipPun].ipPnunit, "\t%s", gv.bin[nd].chDstLab );
						fprintf( save.params[ipPun].ipPnunit, "\n" );
						save.SaveHeaderDone(ipPun);
					}
					fprintf( save.params[ipPun].ipPnunit, " %.5e", 
						radius.depth_mid_zone );
					/* grain potential in eV */
					for( size_t nd=0; nd < gv.bin.size(); ++nd ) 
						fprintf( save.params[ipPun].ipPnunit, "\t%.3e", gv.bin[nd].dstpot*EVRYD );
					fprintf( save.params[ipPun].ipPnunit, "\n" );
				}
			}

			else if( strcmp(save.chSave[ipPun],"DUSR") == 0 )
			{
				/* grain H2 formation rates */
				if( ! lgLastOnly )
				{
					if( save.lgSaveHeader(ipPun) )
					{
						/* first print string giving grain id */
						fprintf( save.params[ipPun].ipPnunit, "#Depth" );
						for( size_t nd=0; nd < gv.bin.size(); ++nd ) 
							fprintf( save.params[ipPun].ipPnunit, "\t%s", gv.bin[nd].chDstLab );
						fprintf( save.params[ipPun].ipPnunit, "\n" );
						save.SaveHeaderDone(ipPun);
					}
					fprintf( save.params[ipPun].ipPnunit, " %.5e", 
						radius.depth_mid_zone );
					/* grain formation rate for H2 */
					for( size_t nd=0; nd < gv.bin.size(); ++nd ) 
						fprintf( save.params[ipPun].ipPnunit, "\t%.3e", gv.bin[nd].rate_h2_form_grains_used );
					fprintf( save.params[ipPun].ipPnunit, "\n" );
				}
			}

			else if( strcmp(save.chSave[ipPun],"DUST") == 0 )
			{
				/* grain temperatures - K*/
				if( ! lgLastOnly )
				{
					/* do labels first if this is first zone */
					if( save.lgSaveHeader(ipPun) )
					{
						/* first print string giving grain id */
						fprintf( save.params[ipPun].ipPnunit, "#Depth" );
						for( size_t nd=0; nd < gv.bin.size(); ++nd ) 
							fprintf( save.params[ipPun].ipPnunit, "\t%s", gv.bin[nd].chDstLab );
						fprintf( save.params[ipPun].ipPnunit, "\n" );
						save.SaveHeaderDone(ipPun);
					}
					fprintf( save.params[ipPun].ipPnunit, " %.5e", 
						radius.depth_mid_zone );
					for( size_t nd=0; nd < gv.bin.size(); ++nd ) 
						fprintf( save.params[ipPun].ipPnunit, "\t%.3e", gv.bin[nd].tedust );
					fprintf( save.params[ipPun].ipPnunit, "\n" );
				}
			}

			else if( strcmp(save.chSave[ipPun],"DUSC") == 0 )
			{
				/* save grain charge - eden from grains and 
				 * charge per grain in electrons / grain */
				if( ! lgLastOnly )
				{
					/* do labels first if this is first zone */
					if( save.lgSaveHeader(ipPun) )
					{
						/* first print string giving grain id */
						fprintf( save.params[ipPun].ipPnunit, "#Depth\tne(grn)" );
						for( size_t nd=0; nd < gv.bin.size(); ++nd ) 
							fprintf( save.params[ipPun].ipPnunit, "\t%s", gv.bin[nd].chDstLab );
						fprintf( save.params[ipPun].ipPnunit, "\n" );
						save.SaveHeaderDone(ipPun);
					}

					fprintf( save.params[ipPun].ipPnunit, " %.5e\t%.4e", 
						radius.depth_mid_zone ,
						/* electron density contributed by grains, in e/cm^3, 
						 * positive number means grain supplied free electrons */
						gv.TotalEden );

					/* average charge per grain in electrons */
					for( size_t nd=0; nd < gv.bin.size(); ++nd )
					{
						fprintf( save.params[ipPun].ipPnunit, "\t%.3e", gv.bin[nd].AveDustZ );
					}
					fprintf( save.params[ipPun].ipPnunit, "\n" );
				}
			}

			else if( strcmp(save.chSave[ipPun],"DUSH") == 0 )
			{
				/* grain heating */
				if( ! lgLastOnly )
				{
					/* save grain charge, but do labels first if this is first zone */
					if( save.lgSaveHeader(ipPun) )
					{
						/* first print string giving grain id */
						fprintf( save.params[ipPun].ipPnunit, "#Depth" );
						for( size_t nd=0; nd < gv.bin.size(); ++nd ) 
							fprintf( save.params[ipPun].ipPnunit, "\t%s", gv.bin[nd].chDstLab );
						fprintf( save.params[ipPun].ipPnunit, "\n" );
						save.SaveHeaderDone(ipPun);
					}
					fprintf( save.params[ipPun].ipPnunit, " %.5e", 
						radius.depth_mid_zone );
					/* grain heating */
					for( size_t nd=0; nd < gv.bin.size(); ++nd ) 
						fprintf( save.params[ipPun].ipPnunit, "\t%.3e", gv.bin[nd].GasHeatPhotoEl );
					fprintf( save.params[ipPun].ipPnunit, "\n" );
				}
			}

			else if( strcmp(save.chSave[ipPun],"DUSV") == 0 )
			{
				/* grain drift velocities */
				if( ! lgLastOnly )
				{
					/* save grain velocity, but do labels first if this is first zone */
					if( save.lgSaveHeader(ipPun) )
					{
						/* first print string giving grain id */
						fprintf( save.params[ipPun].ipPnunit, "#Depth" );
						for( size_t nd=0; nd < gv.bin.size(); ++nd ) 
							fprintf( save.params[ipPun].ipPnunit, "\t%s", gv.bin[nd].chDstLab );
						fprintf( save.params[ipPun].ipPnunit, "\n" );
						save.SaveHeaderDone(ipPun);
					}
					fprintf( save.params[ipPun].ipPnunit, " %.5e", 
						radius.depth_mid_zone );
					/* grain drift velocity in km/s */
					for( size_t nd=0; nd < gv.bin.size(); ++nd ) 
						fprintf( save.params[ipPun].ipPnunit, "\t%.3e", gv.bin[nd].DustDftVel*1e-5 );
					fprintf( save.params[ipPun].ipPnunit, "\n" );
				}
			}

			/* >>chng 02 dec 30, separated scattering cross section and asymmetry factor, PvH */
			else if( strcmp(save.chSave[ipPun],"DUSQ") == 0 )
			{
				/* save grain Qs */
				if( lgLastOnly )
				{
					if( save.lgSaveHeader(ipPun) )
					{
						/* first print string giving grain id */
						fprintf( save.params[ipPun].ipPnunit, "#grain nu/Ryd" );
						for( size_t nd=0; nd < gv.bin.size(); ++nd ) 
							fprintf( save.params[ipPun].ipPnunit, "\tQ_abs%s\tQ_scat*(1-g)%s",
								 gv.bin[nd].chDstLab, gv.bin[nd].chDstLab );
						fprintf( save.params[ipPun].ipPnunit, "\n" );
						save.SaveHeaderDone(ipPun);
					}
					for( j=0; j < gv.nflux; j++ )
					{
						fprintf( save.params[ipPun].ipPnunit, " %.5e", 
								 gv.anu(j) );
						for( size_t nd=0; nd < gv.bin.size(); nd++ )
						{
							fprintf( save.params[ipPun].ipPnunit, "\t%.3e\t%.3e", 
									 gv.bin[nd].dstab1[j]*4./gv.bin[nd].IntArea,
									 gv.bin[nd].pure_sc1[j]*gv.bin[nd].asym[j]*4./gv.bin[nd].IntArea );
						}
						fprintf( save.params[ipPun].ipPnunit, "\n" );
					}
				}
			}

			else if( strcmp(save.chSave[ipPun],"ELEM") == 0 )
			{
				if( ! lgLastOnly )
				{
					realnum renorm = 1.f;

					/* this is the index for the atomic number on the physical scale */
					/* >>chng 04 nov 23, use c scale throughout */
					long nelem = (long int)save.punarg[ipPun][0];
					ASSERT( nelem >= ipHYDROGEN );

					/* don't do this if element is not turned on */
					if( dense.lgElmtOn[nelem] )
					{
						/* >>chng 04 nov 23, add density option, leave as cm-3 
						* default is still norm to total of that element */
						if( save.punarg[ipPun][1] == 0 )
							renorm = dense.gas_phase[nelem];

						fprintf( save.params[ipPun].ipPnunit, " %.5e", radius.depth_mid_zone );

						if( nelem==ipHYDROGEN )
						{
							for( j=0; j <= (nelem + 1); ++j)
							{
								fprintf( save.params[ipPun].ipPnunit, "\t%.2e", 
											dense.xIonDense[nelem][j]/renorm );
							}
							/* H2 */
							fprintf( save.params[ipPun].ipPnunit, "\t%.2e", 
								hmi.H2_total/renorm );
						}
						/* >>chng 04 nov 23 add C and O fine structure pops */
						else
						{
							vector<string>& chList = save.chSaveSpecies[ipPun];
							for ( size_t ic = 0; ic < chList.size(); ++ic )
							{
								vector<genericState> v = matchGeneric( chList[ic], false );
								double dens = 0;
								for (size_t j=0; j<v.size(); ++j)
									dens += density(v[j]);
								fprintf( save.params[ipPun].ipPnunit, "\t%.2e",dens/renorm);
							}
						}
						fprintf( save.params[ipPun].ipPnunit, "\n" );
					}
				}
			}

			else if( strcmp(save.chSave[ipPun],"FRED") == 0 )
			{
				/* set with save Fred command, this punches some stuff from
				 * Fred Hamann's dynamics project */
				if( ! lgLastOnly )
				{
					/* Fred's list */
					fprintf( save.params[ipPun].ipPnunit, "%.5e\t%.5e\t%.3e\t%.3e\t%.3e"
						"\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e"
						"\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e"
						"\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e"
						"\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\n",
						//"\t%.3e\t%.3e\n",
						radius.Radius, radius.depth ,wind.windv/1e5,
						wind.dvdr,
						dense.gas_phase[ipHYDROGEN], dense.eden , phycon.te,
						wind.AccelLine , wind.AccelCont ,
						wind.fmul , 
						// acceleration in this zone due to electron scattering,
						// if incident SED was not attenuated
						pressure.pinzon_PresIntegElecThin/dense.xMassDensity/radius.drad_x_fillfac ,
						mean.xIonMean[0][ipHYDROGEN][0][0] , mean.xIonMean[0][ipHYDROGEN][1][0] ,
						mean.xIonMean[0][ipHELIUM][0][0] , mean.xIonMean[0][ipHELIUM][1][0] ,
						mean.xIonMean[0][ipHELIUM][2][0] ,
						mean.xIonMean[0][ipCARBON][1][0] , mean.xIonMean[0][ipCARBON][2][0] ,
						mean.xIonMean[0][ipCARBON][3][0] ,
						mean.xIonMean[0][ipOXYGEN][0][0] , mean.xIonMean[0][ipOXYGEN][1][0] ,
						mean.xIonMean[0][ipOXYGEN][2][0] , mean.xIonMean[0][ipOXYGEN][3][0] ,
						mean.xIonMean[0][ipOXYGEN][4][0] , mean.xIonMean[0][ipOXYGEN][5][0] ,
						mean.xIonMean[0][ipOXYGEN][6][0] , mean.xIonMean[0][ipOXYGEN][7][0] ,
						dense.xIonDense[ipHYDROGEN][0] , dense.xIonDense[ipHYDROGEN][1] ,
						dense.xIonDense[ipHELIUM][0] , dense.xIonDense[ipHELIUM][1] ,
						dense.xIonDense[ipHELIUM][2] ,
						dense.xIonDense[ipCARBON][1] , dense.xIonDense[ipCARBON][2] ,
						dense.xIonDense[ipCARBON][3] ,
						dense.xIonDense[ipOXYGEN][0] , dense.xIonDense[ipOXYGEN][1] ,
						dense.xIonDense[ipOXYGEN][2] , dense.xIonDense[ipOXYGEN][3] ,
						dense.xIonDense[ipOXYGEN][4] , dense.xIonDense[ipOXYGEN][5] ,
						dense.xIonDense[ipOXYGEN][6] , dense.xIonDense[ipOXYGEN][7] ,
						mean.xIonMean[0][ipMAGNESIUM][1][0] , dense.xIonDense[ipMAGNESIUM][1]);
				}
			}

			/* save spectra in fits format */
			else if( strcmp(save.chSave[ipPun],"FITS") == 0 )
			{
				if( lgLastOnly )
					saveFITSfile( save.params[ipPun].ipPnunit, NUM_OUTPUT_TYPES );
			}
			/* save gammas (but without element) */
			else if( strcmp(save.chSave[ipPun],"GAMt") == 0 )
			{
				if( ! lgLastOnly )
				{
					long ns;
					/* save photoionization rates, with the SAVE GAMMAS command */
					for( long nelem=0; nelem < LIMELM; nelem++ )
					{
						if( !dense.lgElmtOn[nelem] )
							continue;

						for( long ion=0; ion <= nelem; ion++ )
						{
							for( ns=0; ns < Heavy.nsShells[nelem][ion]; ns++ )
							{
								fprintf( save.params[ipPun].ipPnunit, "%3ld%3ld%3ld%10.2e%10.2e%10.2e", 
									nelem+1, ion+1, ns+1, 
									ionbal.PhotoRate_Shell[nelem][ion][ns][0], 
									ionbal.PhotoRate_Shell[nelem][ion][ns][1] ,
									ionbal.PhotoRate_Shell[nelem][ion][ns][2] );

								for( j=0; j < t_yield::Inst().nelec_eject(nelem,ion,ns); j++ )
								{
									fprintf( save.params[ipPun].ipPnunit, "%5.2f",
										 t_yield::Inst().elec_eject_frac(nelem,ion,ns,j) );
								}
								fprintf( save.params[ipPun].ipPnunit, "\n" );
							}
						}
					}
				}
			}

			/* save gammas element, ion */
			else if( strcmp(save.chSave[ipPun],"GAMe") == 0 )
			{
				if( ! lgLastOnly )
				{
					int ns;
					long nelem = (long)save.punarg[ipPun][0];
					long ion = (long)save.punarg[ipPun][1];

					if( !dense.lgElmtOn[nelem] )
					{
						fprintf( ioQQQ, "SAVE GAMMAS ELEMENT: requested element is not active. Bailing out.\n" );
						cdEXIT(EXIT_FAILURE);
					}

					/* valence shell */
					ns = Heavy.nsShells[nelem][ion]-1;
					/* show what some of the ionization sources are */
					GammaPrt( 
						opac.ipElement[nelem][ion][ns][0] , 
						opac.ipElement[nelem][ion][ns][1] , 
						opac.ipElement[nelem][ion][ns][2] , 
						save.params[ipPun].ipPnunit, 
						ionbal.PhotoRate_Shell[nelem][ion][ns][0] , 
						ionbal.PhotoRate_Shell[nelem][ion][ns][0]*0.1 );
				}
			}

			else if( strcmp(save.chSave[ipPun],"GAUN") == 0 )
			{
				/* save gaunt factors */
				if( ! lgLastOnly )
					SaveGaunts(save.params[ipPun].ipPnunit);
			}

			else if( strcmp(save.chSave[ipPun],"GRID") == 0 )
			{
				// generating the SAVE GRID output has been moved to cdPrepareExit()
				// to make sure that the output always records any type of failure
			}
			else
			{
				//no hit this branch, key should be in next
				lgNoHitFirstBranch = true;
			}
		}

		// hack needed for code to compile with Visual Studio
		// keep this identical to the if-statement further up!!
		if( lgActive )
		{
			if( strcmp(save.chSave[ipPun],"HISp") == 0 )
			{
				/* save pressure history of current zone */
				if( ! lgLastOnly )
				{
					/* note if pressure convergence failure occurred in history that follows */
					if( !conv.lgConvPres )
					{
						fprintf( save.params[ipPun].ipPnunit, 
							"#PROBLEM  Pressure not converged iter %li zone %li density-pressure follows:\n",
							iteration , nzone );
					}
					/* note if temperature convergence failure occurred in history that follows */
					if( !conv.lgConvTemp )
					{
						fprintf( save.params[ipPun].ipPnunit, 
							"#PROBLEM  Temperature not converged iter %li zone %li density-pressure follows:\n",
							iteration , nzone );
					}
					for( unsigned long k=0; k < conv.hist_pres_density.size(); ++k )
					{
						/* save history of density - pressure, with correct pressure */
						fprintf( save.params[ipPun].ipPnunit , "%2li %4li\t%.5e\t%.5e\t%.5e\n",
							iteration,
							nzone,
							conv.hist_pres_density[k],
							conv.hist_pres_current[k],
							conv.hist_pres_error[k]);
					}
				}
			}

			else if( strcmp(save.chSave[ipPun],"HISt") == 0 )
			{
				/* save temperature history of current zone */
				if( ! lgLastOnly )
				{
					/* note if pressure convergence failure occurred in history that follows */
					if( !conv.lgConvPres )
					{
						fprintf( save.params[ipPun].ipPnunit, 
							"#PROBLEM  Pressure not converged iter %li zone %li temp heat cool follows:\n",
							iteration , nzone );
					}
					/* note if temperature convergence failure occurred in history that follows */
					if( !conv.lgConvTemp )
					{
						fprintf( save.params[ipPun].ipPnunit, 
							"#PROBLEM  Temperature not converged iter %li zone %li temp heat cool follows:\n",
							iteration , nzone );
					}
					for( unsigned long k=0; k < conv.hist_temp_temp.size(); ++k )
					{
						/* save history of density - pressure, with correct pressure */
						fprintf( save.params[ipPun].ipPnunit , "%2li %4li\t%.5e\t%.5e\t%.5e\n",
							iteration,
							nzone,
							conv.hist_temp_temp[k],
							conv.hist_temp_heat[k],
							conv.hist_temp_cool[k]);
					}
				}
			}

			else if( strncmp(save.chSave[ipPun],"H2",2) == 0 )
			{
				/* all save info on large H2 molecule include H2 PDR pdr */
				save.whichDiatomToPrint[ipPun]->H2_PunchDo( save.params[ipPun].ipPnunit , save.chSave[ipPun] , chTime, ipPun );
			}

			else if( strcmp(save.chSave[ipPun],"HEAT") == 0 )
			{
				/* save heating, routine in file of same name */
				if( ! lgLastOnly )
					SaveHeat(save.params[ipPun].ipPnunit);
			}

			else if( strncmp(save.chSave[ipPun],"HE",2) == 0 )
			{
				/* various save helium commands */
				/* save helium line wavelengths */
				if( strcmp(save.chSave[ipPun] , "HELW") == 0 )
				{
					if( lgLastOnly )
					{
						/* save helium & he-like wavelengths, first header */
						fprintf( save.params[ipPun].ipPnunit, 
							"Z\tElem\t2 1P->1 1S\t2 3P1->1 1S\t2 3P2->1 1S"
							"\t2 3S->1 1S\t2 3P2->2 3S\t2 3P1->2 3S\t2 3P0->2 3S" );
						fprintf( save.params[ipPun].ipPnunit, "\n" );
						for( long nelem=ipHELIUM; nelem<LIMELM; ++nelem )
						{
							/* print element name, nuclear charge */
							fprintf( save.params[ipPun].ipPnunit, "%li\t%s", 
								nelem+1 , elementnames.chElementSym[nelem] );
							/*prt_wl print wavelength in output format */
							fprintf( save.params[ipPun].ipPnunit, "\t" );
							iso_sp[ipHE_LIKE][nelem].trans(ipHe2p1P,ipHe1s1S).twav().prt_wl(save.params[ipPun].ipPnunit);
							fprintf( save.params[ipPun].ipPnunit, "\t" );
							iso_sp[ipHE_LIKE][nelem].trans(ipHe2p3P1,ipHe1s1S).twav().prt_wl(save.params[ipPun].ipPnunit);
							fprintf( save.params[ipPun].ipPnunit, "\t" );
							iso_sp[ipHE_LIKE][nelem].trans(ipHe2p3P2,ipHe1s1S).twav().prt_wl(save.params[ipPun].ipPnunit);
							fprintf( save.params[ipPun].ipPnunit, "\t" );
							iso_sp[ipHE_LIKE][nelem].trans(ipHe2s3S,ipHe1s1S).twav().prt_wl(save.params[ipPun].ipPnunit);
							fprintf( save.params[ipPun].ipPnunit, "\t" );
							iso_sp[ipHE_LIKE][nelem].trans(ipHe2p3P2,ipHe2s3S).twav().prt_wl(save.params[ipPun].ipPnunit);
							fprintf( save.params[ipPun].ipPnunit, "\t" );
							iso_sp[ipHE_LIKE][nelem].trans(ipHe2p3P1,ipHe2s3S).twav().prt_wl(save.params[ipPun].ipPnunit);
							fprintf( save.params[ipPun].ipPnunit, "\t" );
							iso_sp[ipHE_LIKE][nelem].trans(ipHe2p3P0,ipHe2s3S).twav().prt_wl(save.params[ipPun].ipPnunit);
							fprintf( save.params[ipPun].ipPnunit, "\n"); 
						}
					}
				}
				else
					TotalInsanity();
			}

			/* save hummer, results needed for Lya transport, to feed into David's routine */
			else if( strcmp(save.chSave[ipPun],"HUMM") == 0 )
			{
				double eps = iso_sp[ipH_LIKE][ipHYDROGEN].trans(ipH2p,ipH1s).Emis().Aul()/
					iso_sp[ipH_LIKE][ipHYDROGEN].trans(ipH2p,ipH1s).Coll().ColUL( colliders );
				fprintf( save.params[ipPun].ipPnunit, 
					" %.5e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e\n", 
				  radius.depth_mid_zone, 
				  iso_sp[ipH_LIKE][ipHYDROGEN].trans(ipH2p,ipH1s).Emis().TauIn(), 
				  iso_sp[ipH_LIKE][ipHYDROGEN].st[ipH1s].Pop(), 
				  iso_sp[ipH_LIKE][ipHYDROGEN].st[ipH2p].Pop(), 
				  phycon.te, iso_sp[ipH_LIKE][ipHYDROGEN].trans(ipH2p,ipH1s).Emis().damp(), eps );
			}

			else if( strncmp( save.chSave[ipPun] , "HYD", 3 ) == 0 )
			{
				/* various save hydrogen commands */
				if( strcmp(save.chSave[ipPun],"HYDc") == 0 )
				{
					if( ! lgLastOnly )
					{
						/* save hydrogen physical conditions */
						fprintf( save.params[ipPun].ipPnunit, 
					    " %.5e\t%.2e\t%.2e\t%.2e\t%.2e\t%.2e\t%.2e\t%.2e\t%.2e\t%.2e\n", 
					    radius.depth_mid_zone, phycon.te, dense.gas_phase[ipHYDROGEN], dense.eden, 
					    dense.xIonDense[ipHYDROGEN][0]/dense.gas_phase[ipHYDROGEN], 
					    dense.xIonDense[ipHYDROGEN][1]/dense.gas_phase[ipHYDROGEN], 
					    hmi.H2_total/dense.gas_phase[ipHYDROGEN], 
					    findspecieslocal("H2+")->den/dense.gas_phase[ipHYDROGEN], 
					    findspecieslocal("H3+")->den/dense.gas_phase[ipHYDROGEN], 
					    findspecieslocal("H-")->den/dense.gas_phase[ipHYDROGEN] );
					}
				}

				else if( strcmp(save.chSave[ipPun],"HYDi") == 0 )
				{
					if( ! lgLastOnly )
					{
						/* save hydrogen ionization
						 * this will be total decays to ground */
						double RateInter = 0.;
						double stage = iso_sp[ipH_LIKE][ipHYDROGEN].fb[0].ColIoniz*dense.eden*iso_sp[ipH_LIKE][ipHYDROGEN].st[ipH1s].Pop();
						double fref = iso_sp[ipH_LIKE][ipHYDROGEN].fb[ipH1s].gamnc*iso_sp[ipH_LIKE][ipHYDROGEN].st[ipH1s].Pop();
						double fout = iso_sp[ipH_LIKE][ipHYDROGEN].st[ipH1s].Pop();
						/* 06 aug 28, from numLevels_max to _local. */
						for( long ion=ipH2s; ion < iso_sp[ipH_LIKE][ipHYDROGEN].numLevels_local; ion++ )
						{
							/* this is total decays to ground */
							RateInter += 
								iso_sp[ipH_LIKE][ipHYDROGEN].trans(ion,ipH1s).Emis().Aul()*
								(iso_sp[ipH_LIKE][ipHYDROGEN].trans(ion,ipH1s).Emis().Ploss());
							/* total photo from all levels */
							fref += iso_sp[ipH_LIKE][ipHYDROGEN].fb[ion].gamnc*iso_sp[ipH_LIKE][ipHYDROGEN].st[ion].Pop();
							/* total col ion from all levels */
							stage += iso_sp[ipH_LIKE][ipHYDROGEN].fb[ion].ColIoniz*dense.eden*
								iso_sp[ipH_LIKE][ipHYDROGEN].st[ion].Pop();
							fout += iso_sp[ipH_LIKE][ipHYDROGEN].st[ion].Pop();
						}
						
						/* make these relative to parent ion */
						stage /= dense.xIonDense[ipHYDROGEN][1];
						fref /= dense.xIonDense[ipHYDROGEN][1];
						fout /= dense.xIonDense[ipHYDROGEN][1];

						fprintf( save.params[ipPun].ipPnunit, "hion\t%4ld\t%.2e\t%.2e\t%.2e", 
						  nzone, 
						  /* photo and collision ion rates have units s-1 */
						  iso_sp[ipH_LIKE][ipHYDROGEN].fb[ipH1s].gamnc, 
						  iso_sp[ipH_LIKE][ipHYDROGEN].fb[ipH1s].ColIoniz* dense.EdenHCorr,
						  ionbal.RateRecomTot[ipHYDROGEN][0] );

						fprintf( save.params[ipPun].ipPnunit, "\t%.2e", 
							iso_sp[ipH_LIKE][ipHYDROGEN].RadRec_caseB );

						fprintf( save.params[ipPun].ipPnunit, 
							"\t%.2e\t%.2e\t%.2e\t%.2e\t%.2e\t%.2e\t%.2e\t%.2e\n", 
						  dense.xIonDense[ipHYDROGEN][1]/dense.xIonDense[ipHYDROGEN][0], 
						  iso_sp[ipH_LIKE][ipHYDROGEN].fb[ipH1s].gamnc/(ionbal.RateRecomTot[ipHYDROGEN][0]), 
						  iso_sp[ipH_LIKE][ipHYDROGEN].fb[1].RadRecomb[ipRecEsc], 
						  RateInter, 
						  fref/MAX2(1e-37,fout), 
						  stage/MAX2(1e-37,fout), 
						  /* simple H+ */
						  safe_div( iso_sp[ipH_LIKE][ipHYDROGEN].fb[ipH1s].gamnc*dense.xIonDense[ipHYDROGEN][0], dense.eden*dense.xIonDense[ipHYDROGEN][1] ),
						  secondaries.csupra[ipHYDROGEN][0]);

						GammaPrt(iso_sp[ipH_LIKE][ipHYDROGEN].fb[0].ipIsoLevNIonCon,rfield.nflux,iso_sp[ipH_LIKE][ipHYDROGEN].fb[ipH1s].ipOpac,
						  save.params[ipPun].ipPnunit,iso_sp[ipH_LIKE][ipHYDROGEN].fb[ipH1s].gamnc,iso_sp[ipH_LIKE][ipHYDROGEN].fb[ipH1s].gamnc*
						  0.05);
					}
				}

				else if( strcmp(save.chSave[ipPun],"HYDp") == 0 )
				{
					if( ! lgLastOnly )
					{
						/* save hydrogen populations 
						 * first give total atom and ion density [cm-3]*/
						fprintf( save.params[ipPun].ipPnunit, "%.5e\t%.2e\t%.2e", 
						  radius.depth_mid_zone, 
						  dense.xIonDense[ipHYDROGEN][0], 
						  dense.xIonDense[ipHYDROGEN][1] );

						/* next give state-specific densities [cm-3] */
						for( j=ipH1s; j < iso_sp[ipH_LIKE][ipHYDROGEN].numLevels_local-1; j++ )
						{
							fprintf( save.params[ipPun].ipPnunit, "\t%.2e", 
								iso_sp[ipH_LIKE][ipHYDROGEN].st[j].Pop() );
						}
						fprintf( save.params[ipPun].ipPnunit, "\n" );
					}
				}

				else if( strcmp(save.chSave[ipPun],"HYDl") == 0 || strcmp(save.chSave[ipPun],"HYDa") == 0 )
				{
					if( lgLastOnly )
					{
						/* save hydrogen line 
						 * gives intensities and optical depths */

						double flin = 1.;

						/* get the number of levels we want to avoid edge effects */
						long int nLoop  = iso_Max_Emitting_Level(ipHYDROGEN, ipH_LIKE, prt.lgPrnIsoCollapsed);

						for( long ipHi=1; ipHi<nLoop ;++ipHi )
						{
							for( long ipLo=0; ipLo<ipHi; ++ipLo )
							{

								if( iso_sp[ipH_LIKE][ipHYDROGEN].trans(ipHi,ipLo).ipCont() < 0 )
									continue;

								if( geometry.lgSphere && geometry.lgStatic)
									flin = 2.;

								if (strcmp(save.chSave[ipPun],"HYDa") == 0 &&
										iso_sp[ipH_LIKE][ipHYDROGEN].st[ipHi].n() != iso_sp[ipH_LIKE][ipHYDROGEN].st[ipLo].n()+1)
									continue;

								double relI,absI,PrtQuantity;
								t_wavl WV = iso_sp[ipH_LIKE][ipHYDROGEN].trans(ipHi,ipLo).twav();

								if (cdLine("H  1",WV,&relI,&absI) == 0)
									continue;

								if( save.punarg[ipPun][0] > 0 )
									PrtQuantity = absI;
								else
									PrtQuantity = relI;

								string wavlStr = iso_sp[ipH_LIKE][ipHYDROGEN].trans(ipHi,ipLo).twav().sprt_wl();

								if (ipHi< iso_sp[ipH_LIKE][ipHYDROGEN].numLevels_local - iso_sp[ipH_LIKE][ipHYDROGEN].nCollapsed_local )
									/* print resolved levels */
									fprintf(save.params[ipPun].ipPnunit, "%li\t%li\t%li\t%li\t%s\t%.2e\t%.4e\t\n",
											iso_sp[ipH_LIKE][ipHYDROGEN].st[ipHi].n(),
											iso_sp[ipH_LIKE][ipHYDROGEN].st[ipHi].l(),
											iso_sp[ipH_LIKE][ipHYDROGEN].st[ipLo].n(),
											iso_sp[ipH_LIKE][ipHYDROGEN].st[ipLo].l(),
											wavlStr.c_str(),
											iso_sp[ipH_LIKE][ipHYDROGEN].trans(ipHi,ipLo).Emis().TauTot()*SQRTPI/flin,
											PrtQuantity);
								else if (ipLo<iso_sp[ipH_LIKE][ipHYDROGEN].numLevels_local- iso_sp[ipH_LIKE][ipHYDROGEN].nCollapsed_local)
										/* print collapsed to resolved */
									fprintf(save.params[ipPun].ipPnunit, "%li\t%i\t%li\t%li\t%s\t%.2e\t%.4e\t\n",
											iso_sp[ipH_LIKE][ipHYDROGEN].st[ipHi].n(),
											-1,
											iso_sp[ipH_LIKE][ipHYDROGEN].st[ipLo].n(),
											iso_sp[ipH_LIKE][ipHYDROGEN].st[ipLo].l(),
											wavlStr.c_str(),
											iso_sp[ipH_LIKE][ipHYDROGEN].trans(ipHi,ipLo).Emis().TauTot()*SQRTPI/flin,
											PrtQuantity);
								else
									/* print collapsed to collapsed */
									fprintf(save.params[ipPun].ipPnunit, "%li\t%i\t%li\t%i\t%s\t%.2e\t%.4e\t\n",
											iso_sp[ipH_LIKE][ipHYDROGEN].st[ipHi].n(),
											-1,
											iso_sp[ipH_LIKE][ipHYDROGEN].st[ipLo].n(),
											-1,
											wavlStr.c_str(),
											iso_sp[ipH_LIKE][ipHYDROGEN].trans(ipHi,ipLo).Emis().TauTot()*SQRTPI/flin,
											PrtQuantity);

							}
						}
					}
				}

				/* save hydrogen Lya - some details about Lya */
				else if( strcmp(save.chSave[ipPun],"HYDL") == 0 )
				{
					if( ! lgLastOnly )
					{
						/* the population ratio for Lya */
						double popul = iso_sp[ipH_LIKE][ipHYDROGEN].st[ipH2p].Pop()/SDIV(iso_sp[ipH_LIKE][ipHYDROGEN].st[ipH1s].Pop());
						/* the excitation temperature of Lya */
						double texc = TexcLine( iso_sp[ipH_LIKE][ipHYDROGEN].trans(ipH2p,ipH1s) );
						fprintf( save.params[ipPun].ipPnunit, 
						  "%.5e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\n", 
						  radius.depth_mid_zone,
						  iso_sp[ipH_LIKE][ipHYDROGEN].trans(ipH2p,ipH1s).Emis().TauIn(), 
						  iso_sp[ipH_LIKE][ipHYDROGEN].trans(ipH2p,ipH1s).Emis().TauTot(), 
						  popul, 
						  texc, 
						  phycon.te, 
						  texc/phycon.te ,
						  iso_sp[ipH_LIKE][ipHYDROGEN].trans(ipH2p,ipH1s).Emis().Pesc(), 
						  iso_sp[ipH_LIKE][ipHYDROGEN].trans(ipH2p,ipH1s).Emis().Pdest(), 
						  iso_sp[ipH_LIKE][ipHYDROGEN].trans(ipH2p,ipH1s).Emis().pump(), 
						  opac.opacity_abs[iso_sp[ipH_LIKE][ipHYDROGEN].trans(ipH2p,ipH1s).ipCont()-1],
						  opac.albedo[iso_sp[ipH_LIKE][ipHYDROGEN].trans(ipH2p,ipH1s).ipCont()-1] );
					}
				}

				else if( strcmp(save.chSave[ipPun],"HYDr") == 0 )
				{
					/* save hydrogen recc - recombination cooling for AGN3 */
					vector<double> Temps { 5000., 10000., 20000. };
					for( auto &t: Temps )
					{
						TempChange( t, false );

						double rtot = iso_RRCoef_Te(0, 0, phycon.te,
										iso_sp[ipH_LIKE][0].numLevels_max -
										iso_sp[ipH_LIKE][0].nCollapsed_max);
						double rgrd = iso_RRCoef_Te(0, 0, phycon.te, 0);

						fprintf( save.params[ipPun].ipPnunit, " %3.1f\t", phycon.te);
						fprintf( save.params[ipPun].ipPnunit, " %10.3e\t", rtot );
						fprintf( save.params[ipPun].ipPnunit, " %10.3e\t", rgrd);
						fprintf( save.params[ipPun].ipPnunit, " %10.3e\n", rtot-rgrd);
					}
					/* must exit since we have disturbed the solution */
					fprintf(ioQQQ , "save agn now exits since solution is disturbed.\n");
					cdEXIT( EXIT_SUCCESS );
				}
				else
					TotalInsanity();
			}

			else if( strcmp(save.chSave[ipPun],"IONI") == 0 )
			{
				if( lgLastOnly )
				{
					/* save mean ionization distribution */
					PrtMeanIon( 'i', false , save.params[ipPun].ipPnunit );
				}
			}

			/* save ionization rates */
			else if( strcmp(save.chSave[ipPun],"IONR") == 0 )
			{
				if( ! lgLastOnly )
				{
					/* this is element number */
					long nelem = (long)save.punarg[ipPun][0];
					fprintf( save.params[ipPun].ipPnunit, 
						"%.5e\t%.4e\t%.4e", 
						radius.depth_mid_zone,
						dense.eden ,
						dynamics.Rate);
					/* >>chng 04 oct 15, from nelem+2 to nelem+1 - array over read -
					 * caught by PnH */
					for( long ion=0; ion<nelem+1; ++ion )
					{
						fprintf( save.params[ipPun].ipPnunit, 
							"\t%.4e\t%.4e\t%.4e\t%.4e", 
							dense.xIonDense[nelem][ion] ,
							ionbal.RateIonizTot(nelem,ion) ,
							ionbal.RateRecomTot[nelem][ion] ,
							dynamics.Source[nelem][ion] );
					}
					fprintf( save.params[ipPun].ipPnunit, "\n");
				}
			}

			else if( strcmp(save.chSave[ipPun]," IP ") == 0 )
			{
				if( lgLastOnly )
				{
					/* save valence shell ip's */
					for( long nelem=0; nelem < LIMELM; nelem++ )
					{
						int ion_big;
						double energy;

						/* this is the largest number of ion stages per line */
						const int NELEM_LINE = 10;
						/* this loop in case all ions do not fit across page */
						for( ion_big=0; ion_big<=nelem; ion_big += NELEM_LINE )
						{
							int ion_limit = MIN2(ion_big+NELEM_LINE-1,nelem);

							/* new line then element name */
							fprintf( save.params[ipPun].ipPnunit, 
								"\n%2.2s", elementnames.chElementSym[nelem]);

							/* print ion stages across line */
							for( long ion=ion_big; ion <= ion_limit; ++ion )
							{
								fprintf( save.params[ipPun].ipPnunit, "\t%4ld", ion+1 );
							}
							fprintf( save.params[ipPun].ipPnunit, "\n" );

							/* this loop is over all shells */
							ASSERT( ion_limit < LIMELM );
							/* upper limit is number of shells in atom */
							for( long ips=0; ips < Heavy.nsShells[nelem][ion_big]; ips++ )
							{

								/* print shell label */
								fprintf( save.params[ipPun].ipPnunit, "%2.2s", Heavy.chShell[ips]);

								/* loop over possible ions */
								for( long ion=ion_big; ion<=ion_limit; ++ion )
								{

									/* does this subshell exist for this ion? break if it does not*/
									/*if( Heavy.nsShells[nelem][ion]<Heavy.nsShells[nelem][0] )*/
									if( ips >= Heavy.nsShells[nelem][ion] )
										break;

									/* array elements are shell, numb of electrons, element, 0 */
									energy = t_ADfA::Inst().getEthresh(ips+1,nelem-ion+1,nelem+1);

									/* now print threshold with correct format */
									if( energy < 10. )
									{
										fprintf( save.params[ipPun].ipPnunit, "\t%6.3f", energy );
									}
									else if( energy < 100. )
									{
										fprintf( save.params[ipPun].ipPnunit, "\t%6.2f", energy );
									}
									else if( energy < 1000. )
									{
										fprintf( save.params[ipPun].ipPnunit, "\t%6.1f", energy );
									}
									else
									{
										fprintf( save.params[ipPun].ipPnunit, "\t%6ld",  (long)(energy) );
									}
								}

								/* put cs at end of long line */
								fprintf( save.params[ipPun].ipPnunit, "\n" );
							}
						}
					}
				}
			}

			else if( strcmp(save.chSave[ipPun],"LINC") == 0 )
			{
				/* save line cumulative */
				if( ! lgLastOnly )
				{
					save_line(save.params[ipPun].ipPnunit,"PUNC",
								 save.lgEmergent[ipPun],ipPun); 
				}
			}
			else if( strcmp(save.chSave[ipPun],"LINT") == 0 )
			{
				/* save line optical depth */
				if( ! lgLastOnly )
				{
					save_line(save.params[ipPun].ipPnunit,"PUNO",
								 save.lgEmergent[ipPun],ipPun); 
				}
			}

			else if( strcmp(save.chSave[ipPun],"LIND") == 0 )
			{
				/* save line data, then stop */
				SaveLineData(save.params[ipPun].ipPnunit);
			}

			else if( strcmp(save.chSave[ipPun],"LINL") == 0 )
			{
				/* save line labels, only run one time */
				static bool lgRunOnce=false;
				if( lgRunOnce )
					continue;
				lgRunOnce = true;

				bool lgPrintAll=false;
				/* LONG keyword on save line labels command sets this to 1 */
				if( save.punarg[ipPun][0]>0. )
					lgPrintAll = true;

				bool lgPrintIndex = true;
				if( save.punarg[ipPun][1] < 1. )
					lgPrintIndex = false;

				prt_LineLabels(save.params[ipPun].ipPnunit, lgPrintAll, lgPrintIndex );
			}

			else if( strcmp(save.chSave[ipPun],"LINO") == 0 )
			{
				if( lgLastOnly )
				{
					/* save line optical depths, routine is below, file static */
					SaveLineStuff(save.params[ipPun].ipPnunit,"optical" , save.punarg[ipPun][0]);
				}
			}

			else if( strcmp(save.chSave[ipPun],"LINP") == 0 )
			{
				if( ! lgLastOnly )
				{
					if( save.punarg[ipPun][0] )
					{
						fprintf( save.params[ipPun].ipPnunit,
							"%.5e", radius.depth_mid_zone );
						for( auto &specline : save.LineList[ipPun] )
						{
							long ipobs = LineSave.findline(specline);
							TransitionProxy tr = LineSave.lines[ipobs].getTransition();

							double quant = 0.;

							/* NB NB
							 *
							 * Recombination lines are not associated with an atomic
							 * model.  Weed them out to avoid segfault.
							 */
							if( LineSave.lines[ipobs].chSumTyp() == 't'
								&& tr.associated() )
							{
								if( save.punarg[ipPun][1] == 0 )
									quant = tr.Lo()->Pop();
								else if( save.punarg[ipPun][1] == 1 )
									quant = tr.Hi()->Pop();
								else if( save.punarg[ipPun][1] == 2 )
									quant = safe_div( tr.Hi()->Pop() * tr.Lo()->g(),
											tr.Lo()->Pop() * tr.Hi()->g(), 0. );
								else if( save.punarg[ipPun][1] == 3 )
									quant = TexcLine( tr );
								else
								{
									TotalInsanity();
								}
							}

							fprintf( save.params[ipPun].ipPnunit, "\t%.4e", quant );
						}
						fprintf( save.params[ipPun].ipPnunit, "\n" );
					}
					else
					{
						for( auto &specline : save.LineList[ipPun] )
						{
							fprintf( save.params[ipPun].ipPnunit,
								"%.5e", radius.depth_mid_zone );
							fprintf( save.params[ipPun].ipPnunit,
									 "\t%s ", specline.chLabel().c_str() );
							fprintf( save.params[ipPun].ipPnunit,
									 "%s", specline.twav().sprt_wl().c_str() );

							double lower = 0.,
							       upper = 0.,
							       ratio = 0.,
							       tspin = 0.;

							long ipobs = LineSave.findline(specline);
							TransitionProxy tr = LineSave.lines[ipobs].getTransition();

							/* NB NB
							 *
							 * Recombination lines are not associated with an atomic
							 * model.  Weed them out to avoid segfault.
							 */
							if( LineSave.lines[ipobs].chSumTyp() == 't'
								&& tr.associated() )
							{
								lower = tr.Lo()->Pop();
								upper = tr.Hi()->Pop();
								ratio = safe_div( tr.Hi()->Pop() * tr.Lo()->g(),
											tr.Lo()->Pop() * tr.Hi()->g(), 0. );
								tspin = TexcLine( tr );
							}

							fprintf( save.params[ipPun].ipPnunit,
								"\t%10.4e\t%10.4e\t%.4e\t%11.4e\n",
								lower, upper, ratio, tspin );
						}
					}
				}
			}

			else if( strcmp(save.chSave[ipPun],"LINS") == 0 )
			{
				/* save line emissivity */
				if( ! lgLastOnly )
				{
					save_line(save.params[ipPun].ipPnunit,"PUNS",
								 save.lgEmergent[ipPun],ipPun);
				}
			}

			else if( strcmp(save.chSave[ipPun],"LINR") == 0 )
			{
				/* save line RT */
				if( ! lgLastOnly )
					Save_Line_RT( save.params[ipPun].ipPnunit);
			}

			else if( strcmp(save.chSave[ipPun],"LINA") == 0 )
			{
				/* save line array */
				if( lgLastOnly )
				{
					/* save out all lines with energies */
					for( j=0; j < LineSave.nsum; j++ )
					{
						if( LineSave.lines[j].wavlVac() > 0. && 
							LineSave.lines[j].SumLine(0) > 0. )
						{
							/* line energy, in units set with units option */
							fprintf( save.params[ipPun].ipPnunit, "%12.5e", 
										AnuUnit((realnum)RYDLAM/LineSave.lines[j].wavlVac()) );
							/* line label */
							fprintf( save.params[ipPun].ipPnunit, "\t");
							LineSave.lines[j].prt(save.params[ipPun].ipPnunit);
							/* intrinsic intensity */
							fprintf( save.params[ipPun].ipPnunit, "\t%8.3f", 
								log10(SDIV(LineSave.lines[j].SumLine(0) * radius.Conv2PrtInten)) );
							/* emergent line intensity, r recombination  */
							fprintf( save.params[ipPun].ipPnunit, "\t%8.3f", 
								log10(SDIV(LineSave.lines[j].SumLine(1) * radius.Conv2PrtInten) ) );
							/* type of line, i for info, etc */
							fprintf( save.params[ipPun].ipPnunit, " \t%c\n", 
										LineSave.lines[j].chSumTyp());
						}
					}
				}
			}

			else if( strcmp(save.chSave[ipPun],"LINI") == 0 )
			{
				if( lgLastOnly && 
					(nzone/save.LinEvery)*save.LinEvery != nzone )
				{
					/* this is the last zone
					 * save line intensities - but do not do last zone twice */
					SaveLineIntensity(save.params[ipPun].ipPnunit , ipPun , save.punarg[ipPun][0] );
				}
				else if( ! lgLastOnly )
				{
					/* following so we only save first zone if LinEvery reset */
					if( (save.lgLinEvery && nzone == 1) || 
					  (nzone/save.LinEvery)*save.LinEvery == nzone )
					{
						/* this is middle of calculation
						 * save line intensities */
						SaveLineIntensity(save.params[ipPun].ipPnunit , ipPun , save.punarg[ipPun][0]);
					}
				}
			}

			else if( strcmp( save.chSave[ipPun],"LEIL") == 0)
			{
				/* some line intensities for the Leiden PDR,
				 * but only do this when calculation is complete */
				if( lgLastOnly )
				{
					double absval , rel;
					long int n;
					/* the lines we will find,
					 * for a sample list of PDR lines look at LineList_PDR_H2.dat
					 * in the cloudy data dir */
					/* the number of H2 lines */
					const int NLINE_H2 = 30; 
					/* the number of lines which are not H2 */
					const int NLINE_NOTH_H2 = 5; 
					/* the labels and air wavelengths in angstrom for the lines that are not H2 */
					char chLabel[NLINE_NOTH_H2][NCHLAB]=
					{ "C  2", "O  1", "O  1", "C  1", "C  1" };
					t_wavl Wl[NLINE_NOTH_H2]=
					{ 157.636e4_air, 63.1679e4_air, 145.495e4_air, 609.590e4_air, 370.269e4_air };
					/* these are air wavelengths in angstrom */
					/* >>chng 05 sep 06, many of following wavelengths updated to agree
					 * with output - apparently not updated when energies changed */
					t_wavl Wl_H2[NLINE_H2]=
					{2.12125e4_air,
					 28.2111e4_air, 17.0302e4_air, 12.2753e4_air, 9.66228e4_air, 8.02285e4_air, 6.90763e4_air,
					 6.10690e4_air, 5.50968e4_air, 5.05174e4_air, 4.69333e4_air, 4.40859e4_air, 4.17994e4_air,
					 3.99506e4_air, 3.84506e4_air, 3.72267e4_air, 3.62518e4_air, 3.54662e4_air, 3.48542e4_air,
					 3.43693e4_air, 3.40323e4_air, 3.38030e4_air, 3.36779e4_air, 3.36496e4_air, 3.37126e4_air,
					 3.38638e4_air, 3.41019e4_air, 3.44280e4_air, 3.54241e4_air, 3.60100e4_air};
					/* print a header for the lines */
					for( n=0; n<NLINE_NOTH_H2; ++n )
					{
						prt_line_inlist( save.params[ipPun].ipPnunit, chLabel[n], Wl[n] );
						/* get the line, non positive return says didn't find it */
						/* arguments are 4-char label, wavelength, return log total intensity, linear rel inten */
						if( cdLine( chLabel[n], Wl[n], &rel, &absval ) <= 0 )
						{
							fprintf(save.params[ipPun].ipPnunit, " did not find\n");
						}
						else
						{
							fprintf(save.params[ipPun].ipPnunit, "\t%.3e\t%.3e\n", absval, rel);
						}
					}
					fprintf(save.params[ipPun].ipPnunit, "\n\n\n");

					/* only print the H2 lines if the big molecule is turned on */
					if( h2.lgEnabled )
					{
						fprintf(save.params[ipPun].ipPnunit, 
							"Here are some of the H2 Intensities, The first one is the\n"
							"1-0 S(0) line and the following ones are the 0-0 S(X)\n"
							"lines where X goes from 0 to 29\n\n");
						for( n=0; n<NLINE_H2; ++n )
						{
							prt_line_inlist( save.params[ipPun].ipPnunit, "H2  ", Wl_H2[n] );
							/* get the line, non positive return says didn't find it */
							if( cdLine( "H2", Wl_H2[n], &rel, &absval ) <= 0 )
							{
								fprintf(save.params[ipPun].ipPnunit, " did not find\n");
							}
							else
							{
								fprintf(save.params[ipPun].ipPnunit, "\t%.3e\t%.3e\n", absval, rel);
							}
						}
					}
				}
			}

			else if( strcmp( save.chSave[ipPun],"LEIS") == 0)
			{
				if( ! lgLastOnly )
				{
					/* get some column densities we shall need */
					double col_ci , col_oi , col_cii, col_heii;
					if( cdColm("carb" , 1 , &col_ci ) )
						TotalInsanity();
					if( cdColm("carb" , 2 , &col_cii ) )
						TotalInsanity();
					if( cdColm("oxyg" , 1 , &col_oi ) )
						TotalInsanity();
					if( cdColm("heli" , 2 , &col_heii ) )
						TotalInsanity();
					/* save Leiden structure - some numbers for the Leiden PDR model comparisons */
					fprintf( save.params[ipPun].ipPnunit, 
					"%.5e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t"
					"%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t"
					"%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t" 
					"%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t"
					"%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\n", 
					/* depth of this point */
					radius.depth_mid_zone,
					/* A_V for an extended source */
					0.00,	
					/* A_V for a point source */
					rfield.extin_mag_V_point,
					/* temperature */
					phycon.te ,
					dense.xIonDense[ipHYDROGEN][0],
					hmi.H2_total,
					dense.xIonDense[ipCARBON][0],
					dense.xIonDense[ipCARBON][1],
					dense.xIonDense[ipOXYGEN][0],
					findspecieslocal("CO")->den,
					findspecieslocal("O2")->den,
					findspecieslocal("CH")->den,
					findspecieslocal("OH")->den,
					dense.eden,
					dense.xIonDense[ipHELIUM][1],
					dense.xIonDense[ipHYDROGEN][1],
					findspecieslocal("H3+")->den,
					findspecieslocal("H")->column,
					findspecieslocal("H2")->column+
					findspecieslocal("H2*")->column,
					col_ci,
					col_cii,
					col_oi,
					findspecieslocal("CO")->column,
					findspecieslocal("O2")->column,
					findspecieslocal("CH")->column,
					findspecieslocal("OH")->column,
					colden.colden[ipCOL_elec],
					col_heii,
					findspecieslocal("H+")->column,
					findspecieslocal("H3+")->column,
					hmi.H2_Solomon_dissoc_rate_used_H2g ,
					gv.rate_h2_form_grains_used_total,
					hmi.H2_photodissoc_used_H2g,
					hmi.UV_Cont_rel2_Draine_DB96_depth,
					/* CO and C dissociation rate */
					mole.findrk("PHOTON,CO=>C,O"),
					/* total CI ionization rate */
					ionbal.PhotoRate_Shell[ipCARBON][0][2][0],
					/* total heating, erg cm-3 s-1 */
					thermal.htot,
					/* total cooling, erg cm-3 s-1 */
					thermal.ctot,
					/* GrnP grain photo heating */
					thermal.heating(0,13),
					/* grain collisional cooling */
					MAX2(0.,gv.GasCoolColl),	
					/* grain collisional heating */
					-1.*MIN2(0.,gv.GasCoolColl),	
					/* COds - CO dissociation heating */
					thermal.heating(0,9),
					/* H2dH-Heating due to H2 dissociation */
					hmi.HeatH2Dish_used,
					/* H2vH-Heating due to collisions with H2 */
					hmi.HeatH2Dexc_used ,
					/* ChaT - charge transfer heating */
					thermal.heating(0,24) ,
					/* cosmic ray heating */
					thermal.heating(1,6) ,
					/* heating due to atoms of various heavy elements */
					thermal.heating(ipMAGNESIUM,0),
					thermal.heating(ipSULPHUR,0),
					thermal.heating(ipSILICON,0),
					thermal.heating(ipIRON,0),
					thermal.heating(ipSODIUM,0),
					thermal.heating(ipALUMINIUM,0),
					thermal.heating(ipCARBON,0),
					0.0,
					0.0,
					0.0,
					0.0,
					0.0);
				}
			}

			else if( strcmp( save.chSave[ipPun],"LLST") == 0)
			{
				/* save linelist command - do on last iteration */
				if( lgLastOnly )
				{
					fprintf( save.params[ipPun].ipPnunit, "iteration %li" , iteration );
					if( save.punarg[ipPun][1] )// column print
						fprintf( save.params[ipPun].ipPnunit, "\n" );

					/* -1 is flag saying that this save command was not set */
					if( save.nLineList[ipPun] < 0 )
						TotalInsanity();

					int LineType = 0;
					if( save.lgEmergent[ipPun] )
						LineType = 1;
					if( save.lgCumulative[ipPun] )
						LineType += 2;

					bool	lgBadLine = false;
					/* loop over all lines in the file we read */
					for( j=0; j<save.nLineList[ipPun]; ++j )
					{
						double relative , absolute, PrtQuantity;
						if( cdLine(save.LineList[ipPun][j], &relative , &absolute , LineType) <= 0 )
						{
							if( !h2.lgEnabled && save.LineList[ipPun][j].chLabel() == "H2" )
							{
								static bool lgMustPrintFirstTime = true;
								if( lgMustPrintFirstTime )
								{
									/* it's an H2 line and H2 is not being done - ignore it */
									fprintf( ioQQQ,"Did not find an H2 line, the large model is not "
										"included, so I will ignore it.  Log intensity set to -30.\n" );
									fprintf( ioQQQ,"I will totally ignore any future missed H2 lines\n");
									lgMustPrintFirstTime = false;
								}
								relative = -30.f;
								absolute = -30.f;
							}
							else
							{
								lgBadLine = true;
							}
						}

						/* options to do either relative or absolute intensity
						 * default is relative, is absolute keyword on line then
						 * punarg set to 1 */
						/* straight line intensities */
						if( save.punarg[ipPun][0] > 0 )
							PrtQuantity = absolute;
						else
							PrtQuantity = relative;

						// column mode, print label
						if( save.punarg[ipPun][1] )
						{
							/* if taking ratio then put div sign between pairs */
							if( save.lgLineListRatio[ipPun] && is_odd(j) )
								fprintf( save.params[ipPun].ipPnunit , "/" );

							fprintf(save.params[ipPun].ipPnunit, "%s ", save.LineList[ipPun][j].chLabel().c_str());
							fprintf(save.params[ipPun].ipPnunit, "%s ", save.LineList[ipPun][j].twav().sprt_wl().c_str());
						}

						/* if taking ratio print every other line as ratio
						 * with previous line */
						if( save.lgLineListRatio[ipPun] )
						{
							/* do line pair ratios */
							static double SaveQuantity = 0;
							if( is_odd(j) )
								fprintf( save.params[ipPun].ipPnunit, "\t%.4e" , 
									SaveQuantity / SDIV( PrtQuantity ) );
							else
								SaveQuantity = PrtQuantity;
						}
						else
						{
							fprintf( save.params[ipPun].ipPnunit, "\t%.4e" , PrtQuantity );
						}
						// column printout, but check if first of pair
						if( save.punarg[ipPun][1] )
						{
							if( !save.lgLineListRatio[ipPun] ||
									is_odd(j) )
								fprintf( save.params[ipPun].ipPnunit, "\n" );
						}
					}
					fprintf( save.params[ipPun].ipPnunit, "\n" );
					if( lgBadLine )
					{
						fprintf(ioQQQ,"DISASTER - did not find line(s) in the Line List table\n");
						cdEXIT(EXIT_FAILURE);
					}
				}
			}

			else if( strcmp(save.chSave[ipPun],"MAP ") == 0 )
			{
				/* do the map now if we are at the zone, or if this
				 * is the LAST call to this routine and map not done yet */
				if(  !hcmap.lgMapDone &&
					(nzone == hcmap.MapZone  ||  lgLastOnly ) )
				{
					bool lgTlkSav = called.lgTalk;
					called.lgTalk = cpu.i().lgMPI_talk();
					hcmap.lgMapBeingDone = true;
					map_do(save.params[ipPun].ipPnunit , " map");
					called.lgTalk = lgTlkSav;
				}
			}

			// save molecules
			else if( strcmp(save.chSave[ipPun],"MOLE") == 0 )
			{
				if( save.lgSaveHeader(ipPun) )
				{
					fprintf( save.params[ipPun].ipPnunit, 
									 "#depth\tAV(point)\tAV(extend)\tCO diss rate\tC recom rate");
					
					for(i=0; i<mole_global.num_calc; ++i )
					{
						if( mole_global.list[i]->n_react > 0 )
							fprintf( save.params[ipPun].ipPnunit, "\t%s",
								 mole_global.list[i]->label.c_str() );
					}
					fprintf ( save.params[ipPun].ipPnunit, "\n");
					save.SaveHeaderDone(ipPun);
				}
				if( ! lgLastOnly )
				{
					/* molecules, especially for PDR, first give radius */
					fprintf( save.params[ipPun].ipPnunit, "%.5e\t" , radius.depth_mid_zone );

					/* visual extinction of point source (star)*/
					fprintf( save.params[ipPun].ipPnunit, "%.5e\t" , rfield.extin_mag_V_point);

					/* visual extinction of an extended source (like a PDR)*/
					fprintf( save.params[ipPun].ipPnunit, "%.5e\t" , rfield.extin_mag_V_extended);

					/* carbon monoxide photodissociation rate */
					fprintf( save.params[ipPun].ipPnunit, "%.5e\t" , mole.findrk("PHOTON,CO=>C,O") );

					/* carbon recombination rate */
					fprintf( save.params[ipPun].ipPnunit, "%.5e" , ionbal.RateRecomTot[ipCARBON][0] );

					/* now do all the molecules */
					for(j=0; j<mole_global.num_calc; ++j )
					{
						if( mole_global.list[j]->n_react > 0 )
							fprintf( save.params[ipPun].ipPnunit, "\t%.5e",
								 mole.species[j].den );
					}

					fprintf(save.params[ipPun].ipPnunit,"\n");
				}
			}

			else if( strcmp(save.chSave[ipPun],"OPAC") == 0 )
			{
				/* save opacity- routine will parse which type of opacity save to do */
				if(  save.lgSaveEveryZone[ipPun] || lgLastOnly )
					save_opacity(save.params[ipPun].ipPnunit,ipPun);
			}

			/* save coarse optical depths command */
			else if( strcmp(save.chSave[ipPun],"OPTc") == 0 )
			{
				if( save.lgSaveEveryZone[ipPun] || lgLastOnly )
				{
					for( j=0; j < rfield.nflux; j++ )
					{
						fprintf( save.params[ipPun].ipPnunit, 
							"%13.5e\t%.3e\t%12.4e\t%.3e\n", 
						  AnuUnit(rfield.anu(j)), 
						  opac.TauAbsFace[j]+opac.TauScatFace[j], 
						  opac.TauAbsFace[j], 
						  opac.TauScatFace[j] );
					}
				}
			}

			/* save fine optical depths command */
			else if( strcmp(save.chSave[ipPun],"OPTf") == 0 )
			{
				if( save.lgSaveEveryZone[ipPun] || lgLastOnly )
				{
					long nu_hi , nskip;
					if( save.punarg[ipPun][0] > 0. )
						/* possible lower bounds to energy range - will be zero if not set */
						j = ipFineCont( save.punarg[ipPun][0] );
					else
						j = 0;

					/* upper limit */
					if( save.punarg[ipPun][1]> 0. )
						nu_hi = ipFineCont( save.punarg[ipPun][1]);
					else
						nu_hi = rfield.nfine;

					/* we will bring nskip cells together into one printed
					 * number to make output smaller - default is 10 */
					nskip = (long)abs(save.punarg[ipPun][2]);
					nskip = MAX2( 1, nskip );

					do
					{
						realnum sum1 = rfield.fine_opt_depth[j];
						realnum sum2 = rfield.fine_opac_zone[j];
						/* want to report the central wavelength of the cell */
						realnum xnu = rfield.fine_anu[j];
						for( long jj=1; jj<nskip; ++jj )
						{
							sum1 += rfield.fine_opt_depth[j+jj];
							sum2 += rfield.fine_opac_zone[j+jj];
							xnu += rfield.fine_anu[j+jj];
						}
						// report each point, even 0, if ALL keyword appears
						if( sum2>0.  ||  save.punarg[ipPun][2]<0)
							fprintf( save.params[ipPun].ipPnunit, 
							  "%12.6e\t%.3e\t%.3e\n", 
							  AnuUnit(xnu/nskip), 
							  sum1/nskip , 
							  sum2/nskip);
						j += nskip;
					}while( j < nu_hi );
				}
			}

			else if( strcmp(save.chSave[ipPun]," OTS") == 0 )
			{
				double ConMax = 0.;
				double xLinMax = 0.;
				double opConSum = 0.;
				double opLinSum = 0.;
				long ipLinMax = 1;
				long ipConMax = 1;

				for( j=0; j < rfield.nflux; j++ )
				{
					opConSum += rfield.otscon[j]*opac.opacity_abs[j];
					opLinSum += rfield.otslin[j]*opac.opacity_abs[j];
					if( rfield.otslin[j]*opac.opacity_abs[j] > xLinMax )
					{
						xLinMax = rfield.otslin[j]*opac.opacity_abs[j];
						ipLinMax = j+1;
					}
					if( rfield.otscon[j]*opac.opacity_abs[j] > ConMax )
					{
						ConMax = rfield.otscon[j]*opac.opacity_abs[j];
						ipConMax = j+1;
					}
				}
				fprintf( save.params[ipPun].ipPnunit, 
				  "tot con lin=%.2e%.2e lin=%.4s%.4e%.2e con=%.4s%.4e%.2e\n", 
				  opConSum, opLinSum, rfield.chLineLabel[ipLinMax-1].c_str()
				  , rfield.anu(ipLinMax-1), xLinMax, rfield.chContLabel[ipConMax-1].c_str()
				  , rfield.anu(ipConMax-1), ConMax );
			}

			else if( strcmp(save.chSave[ipPun],"OVER") == 0 )
			{
				/* save overview
				 * this is the floor for the smallest ionization fractions printed */
				double toosmall = SMALLFLOAT ,
					hold;

				/* overview of model results,
				 * depth, te, hden, eden, ion fractions H, He, c, O */
				if( ! lgLastOnly )
				{

					/* print the depth */
					fprintf( save.params[ipPun].ipPnunit, "%.5e\t", radius.depth_mid_zone );

					/* temperature, heating */
					if(dynamics.Cool() > dynamics.Heat()) 
					{
						fprintf( save.params[ipPun].ipPnunit, "%.4e\t%.3e",
							PrtLogLin(phycon.te ),
							PrtLogLin(thermal.htot-dynamics.Heat() ) );
					}
					else
					{
						double diff = fabs(thermal.htot-dynamics.Cool());
						fprintf( save.params[ipPun].ipPnunit, "%.4e\t%.3e",
							PrtLogLin(phycon.te),
							PrtLogLin( diff ) );
					}

					/* hydrogen and electron densities */
					fprintf( save.params[ipPun].ipPnunit, "\t%.4e\t%.4e",
						PrtLogLin(dense.gas_phase[ipHYDROGEN]),
						PrtLogLin(dense.eden ) );

					/* molecular fraction of hydrogen */
					fprintf( save.params[ipPun].ipPnunit, "\t%.4e",
						PrtLogLin(MAX2(toosmall,2.*hmi.H2_total/dense.gas_phase[ipHYDROGEN])) );

					/* ionization fractions of hydrogen */
					fprintf( save.params[ipPun].ipPnunit, "\t%.4e\t%.4e",
						PrtLogLin(MAX2(toosmall,dense.xIonDense[ipHYDROGEN][0]/dense.gas_phase[ipHYDROGEN])),
						PrtLogLin(MAX2(toosmall,dense.xIonDense[ipHYDROGEN][1]/dense.gas_phase[ipHYDROGEN])) );

					/* ionization fractions of helium */
					for( j=1; j <= 3; j++ )
					{
						double arg1 = SDIV(dense.gas_phase[ipHELIUM]);
						arg1 = MAX2(toosmall,dense.xIonDense[ipHELIUM][j-1]/arg1 );
						fprintf( save.params[ipPun].ipPnunit, "\t%.4e",
							PrtLogLin(arg1) );
					}

					/* carbon monoxide molecular fraction of CO */
					hold = SDIV(dense.gas_phase[ipCARBON]);
					hold = findspecieslocal("CO")->den/hold;
					hold = MAX2(toosmall, hold );
					fprintf( save.params[ipPun].ipPnunit, "\t%.4e", PrtLogLin(hold) );

					/* ionization fractions of carbon */
					for( j=1; j <= 4; j++ )
					{
						hold = SDIV(dense.gas_phase[ipCARBON]);
						hold = MAX2(toosmall,dense.xIonDense[ipCARBON][j-1]/hold);
						fprintf( save.params[ipPun].ipPnunit, "\t%.4e",
							PrtLogLin(hold) );
					}

					/* ionization fractions of oxygen */
					for( j=1; j <= 6; j++ )
					{
						hold = SDIV(dense.gas_phase[ipOXYGEN]);
						hold = MAX2(toosmall,dense.xIonDense[ipOXYGEN][j-1]/hold);
						fprintf( save.params[ipPun].ipPnunit, "\t%.4e",
							PrtLogLin(hold) );
					}

					// molecular fraction of H2O 
					hold = SDIV(dense.gas_phase[ipOXYGEN]);
					hold = findspecieslocal("H2O")->den/hold;
					hold = MAX2(toosmall, hold );
					fprintf( save.params[ipPun].ipPnunit, "\t%.4e", PrtLogLin(hold) );

					/* visual extinction of point source (star)*/
					fprintf( save.params[ipPun].ipPnunit, "\t%.2e" , rfield.extin_mag_V_point);

					/* visual extinction of an extended source (like a PDR)*/
					fprintf( save.params[ipPun].ipPnunit, "\t%.2e" , rfield.extin_mag_V_extended);

					/* >>chng 20 07 25 add Opt Depth @ 1 Ryd as per Vianney LeBouteiller post */
					/*Lyman continuum optical depth to illuminated face */
					fprintf( save.params[ipPun].ipPnunit, "\t%.4e\n", opac.TauAbsFace[iso_sp[ipH_LIKE][ipHYDROGEN].fb[ipH1s].ipIsoLevNIonCon]);
				}
			}

			else if( strcmp(save.chSave[ipPun]," PDR") == 0 )
			{
				/* this is the save PDR command */
				if( ! lgLastOnly )
				{
					/* convert optical depth at wavelength of V filter
					 * into magnitudes of extinction */
					/* >>chyng 03 feb 25, report extinction to illuminated face,
					 * rather than total extinction which included far side when
					 * sphere was set */

					fprintf( save.params[ipPun].ipPnunit, 
						"%.5e\t%.2e\t%.2e\t%.2e\t%.2e\t%.2e\t%.2e\t%.2e\t%.2e\t%.2e\t", 
					  radius.depth_mid_zone, 
					  /* total hydrogen column density, all forms */
					  colden.colden[ipCOL_HTOT], 
					  phycon.te, 
					  /* fraction of H that is atomic */
					  dense.xIonDense[ipHYDROGEN][0]/dense.gas_phase[ipHYDROGEN], 
					  /* ratio of n(H2) to total H, == 0.5 when fully molecular */
					  2.*findspecieslocal("H2")->den/dense.gas_phase[ipHYDROGEN], 
					  2.*findspecieslocal("H2*")->den/dense.gas_phase[ipHYDROGEN], 
					  /* atomic to total carbon */
					  dense.xIonDense[ipCARBON][0]/SDIV(dense.gas_phase[ipCARBON]), 
					  findspecieslocal("CO")->den/SDIV(dense.gas_phase[ipCARBON]), 
					  findspecieslocal("H2O")->den/SDIV(dense.gas_phase[ipOXYGEN]),
					  /* hmi.UV_Cont_rel2_Habing_TH85 is field relative to Habing background, dimensionless */
					  hmi.UV_Cont_rel2_Habing_TH85_depth);

					/* visual extinction due to dust alone, of point source (star)*/
					fprintf( save.params[ipPun].ipPnunit, "%.2e\t" , rfield.extin_mag_V_point);

					/* visual extinction due to dust alone,  of an extended source (like a PDR)*/
					fprintf( save.params[ipPun].ipPnunit, "%.2e\t" , rfield.extin_mag_V_extended);

					/* visual extinction (all sources) of a point source (like a PDR)*/
					long ip = rfield.ipointC( RYDLAM / WL_V_FILT );
					fprintf( save.params[ipPun].ipPnunit, "%.2e\n", opac.TauAbsGeo[0][ip] );
				}
			}

			/* performance characteristics per zone */
			else if( strcmp(save.chSave[ipPun],"PERF") == 0 )
			{
				if( save.lgSaveHeader(ipPun) )
				{
					fprintf( save.params[ipPun].ipPnunit,
						 "#zone\tdTime\tElapsed t\tnPres2Ioniz" );
					for( size_t i = 0; i < conv.ntypes(); i++ )
					{
						fprintf( save.params[ipPun].ipPnunit, "\t%s",
							 conv.getCounterName(i) );
					}
					fprintf( save.params[ipPun].ipPnunit, "\n" );
					save.SaveHeaderDone(ipPun);
				}
				if( ! lgLastOnly )
				{
					static double ElapsedTime , ZoneTime;
					if( nzone<=1 )
					{
						ElapsedTime = cdExecTime();
						ZoneTime = 0.;
					}
					else
					{
						double t = cdExecTime();
						ZoneTime = t - ElapsedTime;
						ElapsedTime = t;
					}

					/* zone, time for this zone, elapsed time */
					fprintf( save.params[ipPun].ipPnunit, " %ld\t%.3f\t%.2f\t%li",
						nzone,  ZoneTime , ElapsedTime, conv.nPres2Ioniz );
					// print various loop counters
					for( size_t i=0; i<conv.ntypes(); ++i )
						fprintf( save.params[ipPun].ipPnunit, "\t%li", conv.getCounterZone(i) );
					fprintf( save.params[ipPun].ipPnunit, "\n" );
				}
			}

			else if( strcmp(save.chSave[ipPun],"PHYS") == 0 )
			{
				if( ! lgLastOnly )
				{
					/* save physical conditions */
					fprintf( save.params[ipPun].ipPnunit, "%.5e\t%.4e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\n", 
					  radius.depth_mid_zone, phycon.te, dense.gas_phase[ipHYDROGEN], 
					  dense.eden, thermal.htot, wind.AccelTotalOutward, geometry.FillFac );
				}
			}

			else if( strcmp(save.chSave[ipPun],"PRES") == 0 )
			{
				/* the save pressure command */
				if( ! lgLastOnly )
				{
					fprintf( save.params[ipPun].ipPnunit, 
					  "%.5e\t%.5e\t%.5e\t%.5e\t%.5e\t%.5e\t%.5e\t%.5e\t%.5e\t%.5e\t%.5e\t%.5e\t%.5e\t%.5e\t%.5e\t%.5e\t%c\n", 
					  /*A 1 #P depth */
					  radius.depth_mid_zone, 
					  /*B 2 Perror */
					  pressure.PresTotlError*100., 
					  /*C 3 Pcurrent */
					  pressure.PresTotlCurr, 
					  /*D 4 Pln + pintg 
					   * >>chng 06 apr 19, subtract pinzon the acceleration added in this zone
					   * since is not total at outer edge of zone, above is at inner edge */
					  pressure.PresTotlInit + pressure.PresInteg - pressure.pinzon, 
					  /*E 5 pgas (0) */
					  pressure.PresTotlInit, 
					  /*F 6 Pgas */
					  pressure.PresGasCurr, 
					  /*G 7 Pram */
					  pressure.PresRamCurr, 
					  /*H 8 P rad in lines */
					  pressure.pres_radiation_lines_curr, 
					  /*I 9 Pinteg subtract continuum rad pres which has already been added on */
					  pressure.PresInteg - pressure.pinzon, 
					  /*J 10 V(wind km/s) wind speed in km/s */
					  wind.windv/1e5,
					  /*K cad(km/s) sound speed in km/s */
					  timesc.sound_speed_adiabatic/1e5,
					  /* the magnetic pressure */
					  magnetic.pressure ,
					  /* the local turbulent velocity in km/s */
					  DoppVel.TurbVel/1e5 ,
					  /* turbulent pressure */
					  pressure.PresTurbCurr*DoppVel.lgTurb_pressure ,
					  /* gravitational pressure */
					  pressure.IntegRhoGravity,
					  // the integral of electron scattering acceleration in
					  // the absence of any absorptio, minus acceleration in current
					  // zone, which has been added in - done this way, result is
					  // zero in first zonen
					  pressure.PresIntegElecThin-pressure.pinzon_PresIntegElecThin,
					  // is this converged?
					  TorF(conv.lgConvPres) );
				}
			}
			else if( strcmp(save.chSave[ipPun],"PREL") == 0 )
			{
				/* line pressure contributors */
				fprintf( save.params[ipPun].ipPnunit, 
					"%.5e\t%.3e\t%.3e\t", 
					/*A 1 #P depth */
					radius.depth_mid_zone ,
					pressure.PresTotlCurr,
					pressure.pres_radiation_lines_curr/SDIV(pressure.PresTotlCurr) );
				PrtLinePres(save.params[ipPun].ipPnunit);

			}

			else if( save.chSave[ipPun][0]=='R' )
			{
				/* work around internal limits to Microsoft vs compiler */
				if( strcmp(save.chSave[ipPun],"RADI") == 0 )
				{
					/* save radius information for all zones */
					if( ! lgLastOnly )
					{
						fprintf( save.params[ipPun].ipPnunit, "%ld\t%.5e\t%.4e\t%.4e\n", 
						  nzone, radius.Radius_mid_zone, radius.depth_mid_zone, 
						  radius.drad );
					}
				}

				else if( strcmp(save.chSave[ipPun],"RADO") == 0 )
				{
					/* save radius information for only the last zone */
					if( lgLastOnly )
					{
						fprintf( save.params[ipPun].ipPnunit, "%ld\t%.5e\t%.4e\t%.4e\n", 
							nzone, radius.Radius_mid_zone, radius.depth_mid_zone, 
							radius.drad );
					}
				}

				else if( strcmp(save.chSave[ipPun],"RESU") == 0 )
				{
					/*  save results of the calculation */
					if( lgLastOnly )
						SaveResults(save.params[ipPun].ipPnunit);
				}

				else if( strcmp(save.chSave[ipPun],"RECA") == 0 )
				{
					/* this will create table for AGN3 then exit */
					ion_recombAGN( save.params[ipPun].ipPnunit );
					cdEXIT(EXIT_SUCCESS);
				}

				else if( strcmp(save.chSave[ipPun],"RECE") == 0 )
				{
					/* save recombination efficiencies,
					 * option turned on with the  "save recombination efficiencies" command
					 * output for the save recombination coefficients command is actually
					 * produced by a series of routines, as they generate the recombination
					 * coefficients.  these include
					 * dielsupres, helium, hydrorecom, iibod, and makerecomb*/
					fprintf( save.params[ipPun].ipPnunit,
						"%12.4e %12.4e %12.4e %12.4e\n",
					  iso_sp[ipH_LIKE][ipHYDROGEN].fb[0].RadRecomb[ipRecRad],
					  iso_sp[ipH_LIKE][ipHYDROGEN].fb[0].RadRecomb[ipRecNetEsc] ,
					  iso_sp[ipH_LIKE][ipHYDROGEN].fb[2].RadRecomb[ipRecRad],
					  iso_sp[ipH_LIKE][ipHYDROGEN].fb[2].RadRecomb[ipRecNetEsc]);
				}

				else
				{
					/* this can't happen */
					TotalInsanity();
				}
			}

			else if( strcmp(save.chSave[ipPun],"SECO") == 0 )
			{
				/*  save secondary ionization */
				if( ! lgLastOnly )
					fprintf(save.params[ipPun].ipPnunit,
					"%.5e\t%.3e\t%.3e\t%.3e\n",
					radius.depth ,
					secondaries.csupra[ipHYDROGEN][0],
					secondaries.csupra[ipHYDROGEN][0]*2.02,
					secondaries.x12tot );
			}

			else if( strcmp(save.chSave[ipPun],"SOUS") == 0 )
			{
				/* full spectrum of continuum source function at 1 depth
				 *  command was "save source spectrum" */
				if( ! lgLastOnly )
				{
					long limit = MIN2(rfield.ipMaxBolt,rfield.nflux);
					for( j=0; j < limit; j++ )
					{
						fprintf( save.params[ipPun].ipPnunit, 
							"%.5e\t%.4e\t%.4e\t%.4e\t%.4e\t%.4e\n", 
						  AnuUnit(rfield.anu(j)),
						  rfield.ConEmitLocal[nzone][j]/rfield.widflx(j),
						  opac.opacity_abs[j],
						  rfield.ConSourceFcnLocal[nzone][j],
						  rfield.ConSourceFcnLocal[nzone][j]/plankf(j),
						  safe_div(rfield.ConSourceFcnLocal[nzone][j],rfield.flux[0][j]));
					}
				}
			}

			else if( strcmp(save.chSave[ipPun],"SOUD") == 0 )
			{
				/* parts of continuum source function vs depth
				 * command was save source function depth */
				j = iso_sp[ipH_LIKE][ipHYDROGEN].fb[ipH1s].ipIsoLevNIonCon + 2;
				fprintf( save.params[ipPun].ipPnunit, 
					"%.4e\t%.4e\t%.4e\t%.4e\n", 
				  opac.TauAbsFace[j-1], 
				  rfield.ConEmitLocal[nzone][j-1]/rfield.widflx(j-1)/MAX2(1e-35,opac.opacity_abs[j-1]), 
				  rfield.otscon[iso_sp[ipH_LIKE][ipHYDROGEN].fb[ipH1s].ipIsoLevNIonCon-1], 
				  rfield.otscon[iso_sp[ipH_LIKE][ipHYDROGEN].fb[0].ipIsoLevNIonCon-1]/opac.opacity_abs[iso_sp[ipH_LIKE][ipHYDROGEN].fb[ipH1s].ipIsoLevNIonCon-1] );
			}

			/* this is save special option */
			else if( strcmp(save.chSave[ipPun],"SPEC") == 0 )
			{
				SaveSpecial(save.params[ipPun].ipPnunit,chTime);
			}

			/* this is save species option */
			else if( strcmp(save.chSave[ipPun],"SPCS") == 0 )
			{
				if( strncmp( save.chSaveArgs[ipPun], "CON", 3 ) == 0 )
				{
					if( lgLastOnly )
						SaveSpeciesPseudoCont( ipPun,
							save.chSaveSpecies[ipPun][0] );
				}
				else if( strcmp( save.chSaveArgs[ipPun], "BAND" ) == 0 )
				{
					if( lgLastOnly )
						SaveSpeciesBands( ipPun,
							save.chSaveSpecies[ipPun][0],
							save.SpeciesBandFile[ipPun],
							save.lgEmergent[ipPun] );
				}
				else if( strcmp( save.chSaveArgs[ipPun], "OPTD" ) == 0 )
				{
					if( lgLastOnly )
						SaveSpeciesOptDep( ipPun, save.chSaveSpecies[ipPun][0] );
				}
				else if( ( ! lgLastOnly && strcmp(save.chSaveArgs[ipPun],"COLU") != 0 ) ||
					( lgLastOnly && strcmp(save.chSaveArgs[ipPun],"COLU") == 0 ) )
						SaveSpecies(save.params[ipPun].ipPnunit , ipPun);
			}

			else if( strcmp(save.chSave[ipPun],"TEMP") == 0 )
			{
				static double deriv_old=-1;
				double deriv=-1. , deriv_sec;
				/* temperature and its derivatives */
				fprintf( save.params[ipPun].ipPnunit, "%.5e\t%.4e\t%.2e", 
					radius.depth_mid_zone, 
					phycon.te, 
					thermal.dCooldT );
				/* if second zone then have one deriv */
				if( nzone >1 )
				{
					deriv = (phycon.te - struc.testr[nzone-2])/ radius.drad;
					fprintf( save.params[ipPun].ipPnunit, "\t%.2e", deriv );
					/* if third zone then have second deriv */
					if( nzone > 2 )
					{
						deriv_sec = (deriv-deriv_old)/ radius.drad;
						fprintf( save.params[ipPun].ipPnunit, "\t%.2e", 
						  deriv_sec );
					}
					deriv_old = deriv;
				}
				fprintf( save.params[ipPun].ipPnunit, "\n");
			}

			/* time dependent model */
			else if( strcmp(save.chSave[ipPun],"TIMD") == 0 )
			{
				if( lgLastOnly )
					DynaPunchTimeDep( save.params[ipPun].ipPnunit , "END" );
			}

			/* execution time per zone */
			else if( strcmp(save.chSave[ipPun],"XTIM") == 0 )
			{
				static double ElapsedTime , ZoneTime;
				if( nzone<=1 )
				{
					ElapsedTime = cdExecTime();
					ZoneTime = 0.;
				}
				else
				{
					double t = cdExecTime();
					ZoneTime = t - ElapsedTime;
					ElapsedTime = t;
				}

				/* zone, time for this zone, elapsed time */
				fprintf( save.params[ipPun].ipPnunit, " %ld\t%.3f\t%.2f\n", 
				  nzone,  ZoneTime , ElapsedTime );
			}

			else if( strcmp(save.chSave[ipPun],"TPRE") == 0 )
			{
				/* temperature and its predictors, turned on with save tprid */
				fprintf( save.params[ipPun].ipPnunit, "%5ld %11.4e %11.4e %11.4e %g\n", 
				  nzone, phycon.TeInit, phycon.TeProp, phycon.te, 
				  (phycon.TeProp- phycon.te)/phycon.te );
			}

			else if( strcmp(save.chSave[ipPun],"WIND") == 0 )
			{
				/* wind velocity, radiative acceleration, and ratio total
				 * to electron scattering acceleration */
				/* first test only save last zone */
				if( (save.punarg[ipPun][0] == 0 && lgLastOnly)
					||
					/* this test save all zones */
					(save.punarg[ipPun][0] == 1  && ! lgLastOnly ) )
				{
					fprintf( save.params[ipPun].ipPnunit, 
						"%.5e\t%.5e\t%.4e\t%.4e\t%.4e\t%.4e\t%.4e\t%.4e\n", 
						radius.Radius_mid_zone, 
						radius.depth_mid_zone, 
						wind.windv, 
						wind.AccelTotalOutward, 
						wind.AccelLine,
						wind.AccelCont ,
						wind.fmul ,
						wind.AccelGravity );
				}
			}

			else if( strcmp(save.chSave[ipPun],"XATT") == 0 )
			{
				/* attenuated incident continuum */
				ASSERT( grid.lgOutputTypeOn[2] );

				if( lgLastOnly )
					saveFITSfile( save.params[ipPun].ipPnunit, 2 );
			}
			else if( strcmp(save.chSave[ipPun],"XRFI") == 0 )
			{
				/* reflected incident continuum */
				ASSERT( grid.lgOutputTypeOn[3] );

				if( lgLastOnly )
					saveFITSfile( save.params[ipPun].ipPnunit, 3 );
			}
			else if( strcmp(save.chSave[ipPun],"XINC") == 0 )
			{
				/* incident continuum */
				ASSERT( grid.lgOutputTypeOn[1] );

				if( lgLastOnly )
					saveFITSfile( save.params[ipPun].ipPnunit, 1 );
			}
			else if( strcmp(save.chSave[ipPun],"XDFR") == 0 )
			{
				/* reflected diffuse continuous emission */
				ASSERT( grid.lgOutputTypeOn[5] );

				if( lgLastOnly )
					saveFITSfile( save.params[ipPun].ipPnunit, 5 );
			}
			else if( strcmp(save.chSave[ipPun],"XDFO") == 0 )
			{
				/* diffuse continuous emission outward */
				ASSERT( grid.lgOutputTypeOn[4] );

				if( lgLastOnly )
					saveFITSfile( save.params[ipPun].ipPnunit, 4 );
			}
			else if( strcmp(save.chSave[ipPun],"XLNR") == 0 )
			{
				/* reflected lines */
				ASSERT( grid.lgOutputTypeOn[7] );

				if( lgLastOnly )
					saveFITSfile( save.params[ipPun].ipPnunit, 7 );
			}
			else if( strcmp(save.chSave[ipPun],"XLNO") == 0 )
			{
				/* outward lines */
				ASSERT( grid.lgOutputTypeOn[6] );

				if( lgLastOnly )
					saveFITSfile( save.params[ipPun].ipPnunit, 6 );
			}
			else if( strcmp(save.chSave[ipPun],"XREF") == 0 )
			{
				/* total reflected, lines and continuum */
				ASSERT( grid.lgOutputTypeOn[9] );

				if( lgLastOnly )
					saveFITSfile( save.params[ipPun].ipPnunit, 9 );
			}
			else if( strcmp(save.chSave[ipPun],"XTOT") == 0 )
			{
				/* total spectrum, reflected plus transmitted */
				ASSERT( grid.lgOutputTypeOn[0] );

				if( lgLastOnly )
					saveFITSfile( save.params[ipPun].ipPnunit, 0 );
			}
			else if( strcmp(save.chSave[ipPun],"XTRN") == 0 )
			{
				/* total outward, lines and continuum */
				ASSERT( grid.lgOutputTypeOn[8] );

				if( lgLastOnly )
					saveFITSfile( save.params[ipPun].ipPnunit, 8 );
			}
			else if( strcmp(save.chSave[ipPun],"XSPM") == 0 )
			{
				/* exp(-tau) to the illuminated face */
				ASSERT( grid.lgOutputTypeOn[10] );

				if( lgLastOnly )
					saveFITSfile( save.params[ipPun].ipPnunit, 10 );
			}
			// termination of second set of nested if's
			// error if we have not matched key
			/* there are a few "save" commands that are handled elsewhere
			 * save dr is an example.  These will have lgRealSave set false */
			// lgNoHitFirstBranch says did not find in previous nest of if's
			else if( save.lgRealSave[ipPun] && lgNoHitFirstBranch )
			{
				/* this is insanity, internal flag set in ParseSave not seen here */
				fprintf( ioQQQ, " PROBLEM DISASTER SaveDo does not recognize flag %4.4s set by ParseSave.  This is impossible.\n", 
				  save.chSave[ipPun] );
				TotalInsanity();
			}

			/* print special hash string to separate out various iterations
			 * chTime is LAST on last iteration
			 * save.lgHashEndIter flag is true by default, set false
			 * with "no hash" keyword on save command
			 * save.lg_separate_iterations is true by default, set false
			 * when save time dependent calcs since do not want special
			 * character between time steps
			 * grid.lgGrid is only true when doing a grid of calculations */
			if( lgLastOnly &&
				!(iterations.lgLastIt && !grid.lgGrid ) &&
				save.lgHashEndIter[ipPun] &&
				save.lg_separate_iterations[ipPun] &&
				!save.lgFITS[ipPun] )
			{
				if( dynamics.lgTimeDependentStatic && save.chHashString == "TIME_DEP" )
				{
					fprintf( save.params[ipPun].ipPnunit, "\"time=%f\n",
						dynamics.time_elapsed );
				}
				else if( save.chHashString == "\n" )
				{
					fprintf( save.params[ipPun].ipPnunit, "%s\n",
						 save.chHashString.c_str() );
				}
				else
				{
					if( grid.lgGrid && iterations.lgLastIt )
						fprintf( save.params[ipPun].ipPnunit, "%s\n",
							 save.chGridDelimeter(optimize.nOptimiz).c_str() );
					else
						fprintf( save.params[ipPun].ipPnunit, "%s\n",
							 save.chHashString.c_str() );
				}
			}
			if( save.lgFLUSH )
				fflush( save.params[ipPun].ipPnunit );
		}
	}
	return;
}

/*SaveLineIntensity produce the 'save lines intensity' output */
STATIC void SaveLineIntensity(FILE * ioPUN, long int ipPun , realnum Threshold )
{
	long int i;

	DEBUG_ENTRY( "SaveLineIntensity()" );

	/* used to save out all the emission line intensities
	 * first initialize the line image reader */

	fprintf( ioPUN, "**********************************************************************************************************************************\n" );
	input.echo(ioPUN);

	/* now print any cautions or warnings */
	cdWarnings( ioPUN);
	cdCautions( ioPUN);
	fprintf( ioPUN, "zone=%5ld\n", nzone );
	fprintf( ioPUN, "**********************************************************************************************************************************\n" );
	fprintf( ioPUN, "begin emission lines\n" );


	// check whether intrinsic or emergent line emissivity
	bool lgEmergent = false;
	if( save.punarg[ipPun][0] > 0 )
		lgEmergent = true;

	/* only save non-zero intensities */
	fixit("value of lgEmergent isn't consistent with lgEmergent");
	SaveLineResults slr(ioPUN,save.lgEmergent[ipPun]);

	for( i=0; i < LineSave.nsum; i++ )
	{
		// Threshold is zero by default on save line intensity,
		// all option sets to negative number so that we report all lines
		if( LineSave.lines[i].SumLine(lgEmergent) > Threshold )
		{
			slr.save(&LineSave.lines[i]);
		}
	}

	slr.flush();

	fprintf( ioPUN, "     \n" );
	fprintf( ioPUN, "**********************************************************************************************************************************\n" );

	return;
}

/* lgSaveOpticalDepths true says save optical depths */
static bool lgPopsFirstCall , lgSaveOpticalDepths;

/*SaveLineStuff save optical depths or source functions for all transferred lines */
STATIC void SaveLineStuff(
  FILE * ioPUN,
  const char *chJob , 
  realnum xLimit )
{
	DEBUG_ENTRY( "SaveLineStuff()" );

	/*find out which job this is and set a flag to use later */
	if( strcmp( &*chJob , "optical" ) == 0 )
	{
		/* save line optical depths */
		lgSaveOpticalDepths = true;
		lgPopsFirstCall = false;
	}
	else if( strcmp( &*chJob , "populat" ) == 0 )
	{
		static bool lgFirst=true;
		lgSaveOpticalDepths = false;
		/* level population information */
		if( lgFirst )
		{
			lgPopsFirstCall = true;
			fprintf(ioPUN,"index\tAn.ion\tgLo\tgUp\tE(wn)\tgf\n");
			lgFirst = false;
		}
		else
		{
			lgPopsFirstCall = false;
		}
	}
	else
	{
		fprintf( ioQQQ, " insane job in SaveLineStuff =%s\n", 
		  &*chJob );
		cdEXIT(EXIT_FAILURE);
	}

	long index = 0;
	/* loop over all lines, calling put1Line to create info (routine located below) */
	/* hydrogen like lines */
	/* >>chng 02 may 16, had been explicit H and He-like loops */
	for( long ipISO=ipH_LIKE; ipISO<NISO; ++ipISO )
	{
		for( long nelem=ipISO; nelem < LIMELM; nelem++ )
		{
			if( dense.lgElmtOn[nelem]  )
			{
				/* 06 aug 28, from numLevels_max to _local. */
				for( long ipHi=1; ipHi < iso_sp[ipISO][nelem].numLevels_local; ipHi++ )
				{
					for( long ipLo=0; ipLo <ipHi; ipLo++ )
					{
						if( iso_sp[ipISO][nelem].trans(ipHi,ipLo).Emis().Aul() <= iso_ctrl.SmallA )
							continue;

						/* j-resolved lyman lines */
						if( ipISO == ipH_LIKE && lgIsLymanLineResolved(iso_sp[ipISO][nelem].trans(ipHi,ipLo),
									ExtraLymanLinesJ05[nelem][N_(ipHi)], ExtraLymanLinesJ15[nelem][N_(ipHi)]) )
						{
							++index;

							Save1Line( ExtraLymanLinesJ05[nelem][N_(ipHi)], ioPUN,
									xLimit, index, GetDopplerWidth(dense.AtomicWeight[nelem]) );

							++index;

							Save1Line( ExtraLymanLinesJ15[nelem][N_(ipHi)], ioPUN,
									xLimit, index, GetDopplerWidth(dense.AtomicWeight[nelem]) );
						}
						else
						{
							++index;
							Save1Line( iso_sp[ipISO][nelem].trans(ipHi,ipLo), ioPUN,
									xLimit, index, GetDopplerWidth(dense.AtomicWeight[nelem]) );
						}
					}
				}
				/* also do extra Lyman lines if optical depths are to be done,
				 * these are line that are included only for absorption, not in the
				 * model atoms */
				if( lgSaveOpticalDepths )
				{
					if (ipISO == ipH_LIKE)
					{
						for( long nHi=iso_sp[ipISO][nelem].st[iso_sp[ipISO][nelem].numLevels_local-1].n()+1; nHi < iso_ctrl.nLymanHLike[nelem]; nHi++ )
						{
							++index;

							Save1Line( ExtraLymanLinesJ05[nelem][nHi], ioPUN,
									xLimit, index, GetDopplerWidth(dense.AtomicWeight[nelem]) );

							++index;

							Save1Line( ExtraLymanLinesJ15[nelem][nHi], ioPUN,
									xLimit, index, GetDopplerWidth(dense.AtomicWeight[nelem]) );
						}
					}
					/* >>chng 02 aug 23, for he-like, had starting on much too high a level since
					 * index was number of levels - caught by Adrian Turner */
					/* now output extra line lines, starting one above those already done above */
					/* 06 aug 28, from numLevels_max to _local. */
					else if( ipISO == ipHE_LIKE )
					{
						for( long ipHi=iso_sp[ipISO][nelem].st[iso_sp[ipISO][nelem].numLevels_local-1].n()+1; ipHi < iso_ctrl.nLyman[ipISO]; ipHi++ )
						{
							++index;
							Save1Line( ExtraLymanLinesHeLike[nelem][ipExtraLymanLinesHeLike[nelem][ipHi]], ioPUN, xLimit, index, GetDopplerWidth(dense.AtomicWeight[nelem]) );
						}
					}
					else
						TotalInsanity();
				}
			}
		}
	}

	for( long i=0; i < nWindLine; i++ )
	{
		if( (*TauLine2[i].Hi()).IonStg() < (*TauLine2[i].Hi()).nelem()+1-NISO )
		{
			++index;
			Save1Line( TauLine2[i], ioPUN, xLimit, index, GetDopplerWidth(dense.AtomicWeight[(*TauLine2[i].Hi()).nelem()-1]) );
		}
	}

	for( size_t i=0; i < UTALines.size(); i++ )
	{
		++index;
		Save1Line( UTALines[i], ioPUN, xLimit, index, GetDopplerWidth(dense.AtomicWeight[(*UTALines[i].Hi()).nelem()-1]) );
	}

	/* save optical depths of H2 lines */
	h2.H2_PunchLineStuff( ioPUN , xLimit  , index);

	/* data base lines */
	for (int ipSpecies=0; ipSpecies < nSpecies; ++ipSpecies)
	{
		realnum DopplerWidth = GetDopplerWidth( dBaseSpecies[ipSpecies].fmolweight );
		for( EmissionList::iterator em=dBaseTrans[ipSpecies].Emis().begin();
			em != dBaseTrans[ipSpecies].Emis().end(); ++em)
		{
			++index;
			Save1Line( (*em).Tran(), ioPUN, xLimit, index, DopplerWidth );
		}
	}

	fprintf( ioPUN , "%s\n",save.chHashString.c_str() );
	return;
}

/*Save1Line called by SaveLineStuff to produce output for one line */
void Save1Line( const TransitionProxy& t , FILE * ioPUN , realnum xLimit  , long index, realnum DopplerWidth )
{

	if( lgSaveOpticalDepths )
	{
		/* optical depths, no special first time, only print them */
		if( t.Emis().TauIn() >= xLimit )
		{
			/* label like "C  4" or "H  1" */
			fprintf( ioPUN, "%-*.*s\t",CHARS_SPECIES, CHARS_SPECIES, chIonLbl(t).c_str());

			/* print wavelengths, either line in main printout labels, 
			 * or in various units in exponential notation */
			if( strcmp( save.chConSavEnr[save.ipConPun], "labl" ) == 0 )
			{
				t.twav().prt_wl(ioPUN);
			}
			else
			{
				/* this converts energy in Rydbergs into any of the other units */
				fprintf( ioPUN , "%.7e", AnuUnit((realnum)(t.EnergyRyd())) );
			}
			/* print the optical depth */
			fprintf( ioPUN , "\t%.3f", t.Emis().TauIn()*SQRTPI );
			/* print the line-specific optical depth */
			fprintf( ioPUN , "\t%.3f", t.Emis().TauInSpecific()*SQRTPI );
			/* damping constant */
			fprintf(ioPUN, "\t%.3e", 
				t.Emis().dampXvel() / DopplerWidth );
			fprintf(ioPUN, "\n");
		}
	}
	else if( lgPopsFirstCall )
	{
		/* first call to line populations, print atomic parameters and indices */
		fprintf(ioPUN, "%li\t%s" , index , chLineLbl(t).c_str());
		/* stat weights */
		fprintf(ioPUN, "\t%.0f\t%.0f", 
			(*t.Lo()).g() ,(*t.Hi()).g());
		/* energy difference, gf */
		fprintf(ioPUN, "\t%.2f\t%.3e", 
			t.EnergyWN() ,t.Emis().gf());
		fprintf(ioPUN, "\n");
	}
	else
	{
		/* not first call, so do level populations and indices defined above */
		if( (*t.Hi()).Pop() > xLimit )
		{
			/* >>chng 05 may 08, add abundances, which for iso-seq species is
			 * the density of the parent ion, for other lines, is unity.
			 * had not been included so pops for iso seq were rel to parent ion.
			 * caught by John Everett */
			/* multiplication by abundance no longer necessary since iso pops now denormalized */
			fprintf(ioPUN,"%li\t%.2e\t%.2e\n", index, (*t.Lo()).Pop(), (*t.Hi()).Pop() );
		}
	}
}

/* save AGN3 hemiss, for Chapter 4, routine is below */
STATIC void AGN_Hemis(FILE *ioPUN )
{
	const int NTE = 4;
	realnum te[NTE] = {5000., 10000., 15000., 20000.};
	vector<realnum> agn_continuum[NTE];
	double TempSave = phycon.te;
	long i , j;

	DEBUG_ENTRY( "AGN_Hemis()" );

	/* make table of continuous emission at various temperatuers */
	/* first allocate space */
	for( i=0; i < NTE; ++i )
	{
		agn_continuum[i].resize(rfield.nflux);

		/* set the next temperature */
		/* recompute everything at this new temp */
		TempChange(te[i] , true);
		/* converge the pressure-temperature-ionization solution for this zone */
		ConvPresTempEdenIoniz();

		/* now get the thermal emission */
		RT_diffuse();
		for(j=0;j<rfield.nflux; ++j )
		{
			agn_continuum[i][j] = rfield.ConEmitLocal[nzone][j]/(realnum)dense.eden/
				(dense.xIonDense[ipHYDROGEN][1] + dense.xIonDense[ipHELIUM][1] + dense.xIonDense[ipHELIUM][2] );
		}
	}

	/* print title for line */
	fprintf(ioPUN,"wl");
	for( i=0;i<NTE; ++i)
	{
		fprintf(ioPUN,"\tT=%.0f",te[i]);
	}
	fprintf( ioPUN , "\tcont\n"); 

	/* not print all n temperatures across a line */
	for(j=0;j<rfield.nflux; ++j )
	{
		fprintf( ioPUN , "%12.5e", 
		  AnuUnit(rfield.anu(j)) );
		/* loop over the temperatures, and for each, calculate a continuum */
		for( i=0;i<NTE; ++i)
		{
			fprintf(ioPUN,"\t%.3e",agn_continuum[i][j]*rfield.anu2(j)*EN1RYD/rfield.widflx(j));
		}
		/* cont label and end of line*/
		fprintf( ioPUN , "\t%s\n" , rfield.chContLabel[j].c_str()); 
	}

	/* Restore temperature stored before this routine was called	*/
	/* and force update */
	TempChange(TempSave , true);

	fprintf( ioQQQ, "AGN_Hemis - result of save AGN3 hemis - I have left the code in a disturbed state, and will now exit.\n");
	cdEXIT(EXIT_SUCCESS);
}

/*SaveResults save results from save results command */
/*SaveResults1Line do single line of output for the save results and save line intensity commands */
STATIC void SaveResults(FILE* ioPUN)
{
	long int i , nelem , ion;

	DEBUG_ENTRY( "SaveResults()" );

	/* used to save out line intensities, optical depths,
	 * and column densities */

	fprintf( ioPUN, "**********************************************************************************************************************************\n" );
	input.echo(ioPUN);

	/* first print any cautions or warnings */
	cdWarnings(ioPUN);
	cdCautions(ioPUN);
	fprintf( ioPUN, "**********************************************************************************************************************************\n" );

	fprintf( ioPUN, "C*OPTICAL DEPTHS ELECTRON=%10.3e\n", opac.telec );

	fprintf( ioPUN, "BEGIN EMISSION LINES\n" );
	SaveLineResults slr(ioPUN,0);

	for( i=0; i < LineSave.nsum; i++ )
	{
		if( LineSave.lines[i].SumLine(0) > 0. )
		{
			slr.save(&LineSave.lines[i]);
		}
	}

	slr.flush();

	fprintf( ioPUN, "     \n" );

	fprintf( ioPUN, "BEGIN COLUMN DENSITIES\n" );

	/* this dumps out the whole array,*/
	/* following loop relies on LIMELM being 30, assert it here in case
	 * this is ever changed */
	ASSERT( LIMELM == 30 );
	/* this order of indices is to keep 30 as the fastest variable,
	 * and the 32 (LIMELM+1) as the slower one */
	for( nelem=0; nelem<LIMELM; nelem++ )
	{
		for(ion=0; ion < nelem+1; ion++)
		{
			fprintf( ioPUN, " %10.3e", mean.xIonMean[0][nelem][ion][0] );
			/* throw line feed every 10 numbers */
			if( nelem==9|| nelem==19 || nelem==29 )
			{
				fprintf( ioPUN, "\n" );
			}
		}
	}

	fprintf( ioPUN, "END OF RESULTS\n" );
	fprintf( ioPUN, "**********************************************************************************************************************************\n" );
	return;
}

namespace
{
	/*SaveResults1Line do single line of output for the save results and save line intensity commands */
	/* the number of emission lines across one line of printout */
	void SaveLineResults::save(
		/* 4 char + null string */
		const LinSv *line) 
	{
		
		DEBUG_ENTRY( "SaveLineResults::save()" );
		
		/* if LineWidth is changed then change format in write too */
		
		/* save results in array so that they can be printed when done */
		m_lines[ipLine] = line;
		
		/* now increment the counter and then check if we have filled the line, 
		 * and so should print it */
		++ipLine;
		/* do print now if we are in column mode (one line per line) or if we have filled up
		 * the line */
		if( ( strcmp(::save.chPunRltType,"column") == 0 ) || ipLine == LINEWIDTH )
		{
			/* "array " is usual array 6 wide, "column" is one line per line */
			flush();
		}
	}
}

/*SaveGaunts called by save gaunts command to output Gaunt factors */
STATIC void SaveGaunts(FILE* ioPUN)
{
	DEBUG_ENTRY( "SaveGaunts()" );

	static const int NENR_GAUNT = 33;
	static const int NTE_GAUNT = 20;

	double ener[NENR_GAUNT], ste[NTE_GAUNT], g[NENR_GAUNT][NTE_GAUNT];

	/* this routine is called from the PUNCH GAUNTS command
	 * it drives the Gaunt factor routine to save gaunts over full range */

	for( int i=0; i < NTE_GAUNT; i++ )
		ste[i] = 0.5f*(i+1);

	for( int i=0; i < NENR_GAUNT; i++ )
		ener[i] = 0.5f*i - 9.f;

	for( int charge=1; charge <= LIMELM; charge++ )
	{
		/* energy is log of energy */
		for( int ite=0; ite < NTE_GAUNT; ite++ )
		{
			for( int j=0; j < NENR_GAUNT; j++ )
			{
				g[j][ite] = t_gaunt::Inst().gauntff( charge, exp10(ste[ite]), exp10(ener[j]) );
			}
		}

		/* now save out the results */
		fprintf( ioPUN, "\tlg(nu)\\lg(Te)" );
		for( int i=1; i <= NTE_GAUNT; i++ )
			fprintf( ioPUN, "\t%.3e", ste[i-1] );
		fprintf( ioPUN, "\n" );

		for( int j=0; j < NENR_GAUNT; j++ )
		{
			fprintf( ioPUN, "\t%10.3e", ener[j] );
			for( int ite=0; ite < NTE_GAUNT; ite++ )
				fprintf( ioPUN, "\t%.3e", g[j][ite] );
			fprintf( ioPUN, "\n" );
		}

		fprintf( ioPUN, "\tlg(nu)/lg(Te)" );
		for( int i=0; i < NTE_GAUNT; i++ )
			fprintf( ioPUN, "\t%.3e", ste[i] );
		fprintf( ioPUN, "\n\n" );

		fprintf( ioPUN, "Below is log(gamma^2), log(u), gff\n" );
		/* print log(gamma2), log(u) instead of temp and energy. */

		double z = log10((double)charge);

		for( int i=0; i < NTE_GAUNT; i++ )
		{
			for( int j=0; j < NENR_GAUNT; j++ )
			{
				fprintf( ioPUN, "\t%10.3e\t%10.3e\t%10.3e\n",
					 2.*z + log10(TE1RYD) - ste[i],
					 log10(TE1RYD) + ener[j] - ste[i], 
					 g[j][i] );
			}
		}
		fprintf( ioPUN, "end of charge = %i\n", charge );
		fprintf( ioPUN, "****************************\n" );
	}
}

void SaveGrid(FILE* pnunit, exit_type status)
{
	DEBUG_ENTRY( "SaveGrid()" );

	bool lgCreate = ( pnunit == NULL );

	if( lgCreate )
	{
		// normally the save file will already be open, but it is possible that
		// there was an early exit during parsing before the file was opened,
		// in which case we open the file here so we can report the error.
		string fnam = GridPointPrefix( optimize.nOptimiz ) + save.chFileName[save.ipSaveGrid];
		pnunit = open_data( fnam, "w" );
	}

	if( optimize.nOptimiz == 0 )
	{
		/* start of line gives abort and warning summary */	
		fprintf( pnunit, "#Index\tFailure?\tWarnings?\tExit code\t#rank\t#seq" );
		/* print start of each variable command line */
		for( int i=0; i < grid.nintparm; i++ )
		{
			string chStr( optimize.chVarFmt[i] );
			fprintf( pnunit, "\t%s", chStr.substr(0,9).c_str() );
		}
		fprintf( pnunit, "\tgrid parameter string\n" );
	}
	/* abort / warning summary for this sim */
	bool lgNoFailure = ( status == ES_SUCCESS || status == ES_WARNINGS );
	fprintf( pnunit, "%9.9ld\t%c\t%c\t%20s\t%ld\t%ld",
		 optimize.nOptimiz,
		 TorF(!lgNoFailure),
		 TorF(warnings.lgWarngs),
		 cpu.i().chExitStatus(status).c_str(),
		 cpu.i().nRANK(),
		 grid.seqNum );
	/* the grid parameters */
	ostringstream chGridParam;
	for( int j=0; j < grid.nintparm; j++ )
	{
		if( j > 0 )
			chGridParam << ", ";
		chGridParam << fixed << grid.interpParameters[optimize.nOptimiz][j];
		fprintf( pnunit, "\t%f", grid.interpParameters[optimize.nOptimiz][j] );
	}
	fprintf( pnunit, "\t%s\n", chGridParam.str().c_str() );

	if( lgCreate )
		fclose( pnunit );
}
