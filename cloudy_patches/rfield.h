/* This file is part of Cloudy and is copyright (C)1978-2025 by Gary J. Ferland and
 * others.  For conditions of distribution and use see copyright notice in license.txt */

#ifndef RFIELD_H_
#define RFIELD_H_

/* rfield.h */
#include "energy.h"
#include "module.h"
#include "vectorize.h"
#include "mesh.h"
#include "container_classes.h"

/** wavelength of V filter in Angstroms */
const double WL_V_FILT = 5500.;

/** wavelength of B filter in Angstroms */
const double WL_B_FILT = 4400.;

/** parameters to do with incident continuum */
/** limit to number of spectra that can be entered */
const int LIMSPC = 100;

/** zero out rfield arrays between certain limits, code in zero.c */
void rfield_opac_zero( long lo , long ihi );

/** set true when allocated, init to false */
extern bool lgRfieldAllocated;

namespace Illumination {
	typedef enum { FORWARD, REVERSE, SYMMETRIC } IlluminationType;
}

namespace Accumulate {
	typedef enum { INSTANTANEOUS = 0, CUMULATIVE = 1 } AccumulationType;
}


class Spectrum
{
private:
	/**
	 * flux[0] holds the steady-state continuum flux emitted by a cloud;
	 * 	   in a dynamical run, this should be understood as the
	 * 	   instantaneous continuum of the cloud
	 * flux[1] holds the total flux accumulated over a dynamical run;
	 * 	   the cumulative continuum may easily overflow single-precision
	 * 	   storage; for this reason, the total flux is scaled by the
	 * 	   normalization factor below;
	 * 	   in a steady-state run it is all zero
	 */
	vector<realnum> flux[2];

	/** cumulative spectrum normalization factor */
	double norm;

	/** time-averaged flux */
	double time_avg_flux;

	/** vector lengths */
	long int nflux;


public:
	// constructor
	Spectrum()
	{
		norm = 1.;
		time_avg_flux = 0.;
	}

	void resize( long int nbins );

	void accumulate_flux( const double this_cumulfac, double timestep, double total_time )
	{
		DEBUG_ENTRY( "accumulate_flux()" );

		vector<realnum> &flux_inst = flux[0];
		vector<realnum> &flux_cumul = flux[1];

		double mean = flux_cumul[ 0 ] * norm + flux_inst[ 0 ] * this_cumulfac;
	
		for( long ip = 1; ip < nflux; ip++ )
		{
			/* >>refer	running mean	Welford, 1962, Technometrics, 4-419 */
			double tmp_fl = flux_cumul[ ip ] * norm +
					flux_inst[ ip ] * this_cumulfac;
			mean += (tmp_fl - mean) / double( ip+1 );
		}
	
		if( fp_equal( mean, 0. ) )
		       return;
	
		double mean_inv = 1. / mean;
	
		for( long ip = 0; ip < nflux; ip++ )
		{
			double tmp_flux = flux_cumul[ ip ] * norm +
					flux_inst[ ip ] * this_cumulfac;
			flux_cumul[ ip ] = realnum( tmp_flux * mean_inv );
		}
	
		norm = mean;

		if( fp_equal( time_avg_flux, 0.0 ) )
		{
			time_avg_flux = mean;
		}
		else
		{
			time_avg_flux += ( mean - time_avg_flux ) *
					timestep / ( timestep + total_time );
		}
	
		return;
	}

	double get_flux( long int accType, long int ip ) const
	{
		DEBUG_ENTRY( "get_flux" );

		ASSERT( accType == Accumulate::AccumulationType::INSTANTANEOUS ||
			accType == Accumulate::AccumulationType::CUMULATIVE );

		ASSERT( ip >= 0 && ip < nflux );

		if( accType == Accumulate::AccumulationType::INSTANTANEOUS )
		{
			return double( flux[0][ip] );
		}
		else
		{
			return double( flux[1][ip] ) * norm;
		}
	}

	double get_time_avg_flux() const
	{
		return time_avg_flux;
	}

	vector<realnum>& operator[]( int accType )
	{
		ASSERT( accType == Accumulate::AccumulationType::INSTANTANEOUS ||
			accType == Accumulate::AccumulationType::CUMULATIVE );
		return flux[ accType ];
	}
};

struct t_rfield : public module, public t_mesh {
	const char *chName() const
	{
		return "rfield";
	}
	
	void zero();
	void comment(t_warnings&) {}
	/** ================================================================================= */
	/** the following define the continuum energy scale and its limits */

	/** nflux is number of continuum points needed to get to high energy
	 * end of continuum.  this excludes the unity test cell. */
	long int nflux;

	/** number of frequency cells including unity test cell */
	long int nflux_with_check;

	/** number of cells to include highest continuum cell with non-zero photon flux,
	 * loops over continuum should go to <nPositive to include positive cells  */
	long int nPositive;

	/**faintest high energy flux to consider, set with set flxfnt command */
	double FluxFaint;

	/** used to keep track of number of lines per freq interval */
	vector<long> line_count;

	/** outward emitted continuum */
	vector<realnum> OccNumbContEmitOut;

	/** ================================================================================= */
	/** the following are the arrays containing the local radiation field */

	/**flux is photons per cell N.B. width of cells vary with energy, given by widflx */
	Spectrum flux;

	/** this is the isotropic part of the constant continuum */
	vector<realnum> flux_isotropic;

	/** this is the variable and constant parts of the above */
	vector<realnum> flux_beam_time, flux_beam_const;

	/** the accumulated flux, sum from this energy to infinity */
	vector<realnum> flux_accum;

	/** extinction factor set with extinguish command */
	vector<realnum> ExtinguishFactor;
	realnum ExtinguishLeakage, 
	 ExtinguishColumnDensity, 
	 ExtinguishLowEnergyLimit,
	 /** the constant that multiplies the column density to get optical depth */
	 ExtinguishConvertColDen2OptDepth ,
	 /** the power on the energy */
	 ExtinguishEnergyPowerLow;

	/** this is set true is one of incident continua is expected to have
	 * all H-ionizing radiation blocked.  That is done separately with
	 * the extinguish command.  */
	bool lgMustBlockHIon;

	/* this is set true if H-ionizing radiation is blocked with extinguish
	 * command */
	bool lgBlockHIon;

	/** option to not do line transfer, set false with no line transfer command */
	bool lgDoLineTrans;

	/** says whether to constantly reevaluate opacities, normally true,
	 * set false with no opacity reevaluate command */
	bool lgOpacityReevaluate;

	/** this flag says that CMB has been set */
	bool lgCMB_set;

	/** says whether to constantly reevaluate ionization, normally true,
	 * set false with no ionization reevaluate command */
	bool lgIonizReevaluate;

	/** convoc is the conversion factor from rfield to OccNumbIncidCont */
	vector<realnum> convoc;

	/** OccNumbIncidCont is the continuum occupation number for the 
	 * attenuated incident ONLY */
	vector<realnum> OccNumbIncidCont;

	/** OccNumbDiffCont is the continuum occupation number, for local diffuse continuum */
	vector<realnum> OccNumbDiffCont;

	/** array of Boltzmann factors for the continuum energy grid and current 
	 * temperature */
	vector_avx<double> ContBoltz;

	/** helper arrays for calculating Boltzmann factors averaged over the width of the cell */
	vector_avx<double> ContBoltzHelp1;
	vector_avx<double> ContBoltzHelp2;

	/** helper array for vexp() calls */
	vector_avx<double> vexp_arg;

	/** array of Boltzmann factors for the continuum energy grid and current 
	 * temperature properly averaged over the width of the cell */
	vector_avx<double> ContBoltzAvg;

	/** ConEmitLocal is local diffuse continuum, units photons cm-3 s-1 cell-1, 
	 * for each zone, evaluated in RT_diffuse */
	multi_arr<realnum,2> ConEmitLocal/* [depth][energy]*/;

	/** the local source function - diffuse emission, photons cell-1 cm-2 s-1 */
	multi_arr<realnum,2> ConSourceFcnLocal/* [depth][energy]*/;

	/** reflected diffuse emission continuum */
	Spectrum ConEmitReflec;

	/** outward diffuse emission continuum (not the interactive one),
	 * this is incremented in radinc because of interplay between absorption
	 * and emission - get the outward bremsstrahlung right
	 * photons cell-1 cm-2 s-1 */
	Spectrum ConEmitOut;

	/** outward diffuse emission continuum, summed over all zones WITHOUT
	 * attenuation by intervening opacity (opac.tmn). This gives the
	 * intrinsic gas-phase nebular emission (free-bound, free-free,
	 * two-photon) integrated over the entire cloud, as if the nebula
	 * were optically thin to its own radiation. Grain thermal emission
	 * is excluded. Same units as ConEmitOut: photons cell-1 cm-2 s-1 */
	Spectrum ConEmitOutUnatt;

	/** this is set in RT_diffuse and carries interactive continua */
	vector<realnum> ConInterOut;

	/** ConRefIncid is reflected portion of incident continuum */
	Spectrum ConRefIncid;

	/** these are energy-by-energy sums of various arrays, used to save time in
	 * evaluating rate integrals */
	vector<double> SummedCon;
	vector<realnum> SummedDif;
	vector<realnum> SummedOcc;
	vector<realnum> SummedDifSave;

	/** this will control array of locally destroyed continuum photons,
	 * zeroed and evaluated in RT_diffuse, currently only two-photon */
	vector<realnum> ConOTS_local_photons;
		/** the local photoionization rate corresponding to above photons */
	vector<realnum> ConOTS_local_OTS_rate;

	/** computed in rt_diffuse.cpp, escaping continuum emission 
	 * added to beam in radius_increment */
	vector<realnum> DiffuseEscape;

	/** saves total two photon continuum for debugging, set in RT_diffuse */
	vector<realnum> TotDiff2Pht;

	/** otsline and otscon - local ots fields for line and continua
	 * outlin outward line fields */
	/** the local ots line rates */
	vector<realnum> otslin;
	/** the local ots continuum rates */
	vector<realnum> otscon;
	multi_arr<realnum,2> otssav;

	/** outward directed line emission photons cm-2 s-1 */
	Spectrum outlin;
	vector<realnum> outlin_noplot;

	/** local diffuse line emission, photons cm-3 s-1 */
	vector<realnum> DiffuseLineEmission;

	/** reflected line */
	Spectrum reflin;

	/** save incident continuum for later iterations */
	Spectrum flux_total_incident;
	vector<realnum> flux_beam_const_save, flux_time_beam_save, flux_isotropic_save;

	/** ==1 for time steady, when continuum varies with time, is scale factor */
	realnum time_continuum_scale;

	/** this is zero or one depending whether pumping by diffuse fields
	 * is turned on (1) or turned off (0) */
	realnum DiffPumpOn;

	/** string identifying the first line that occurred at this energy */
	vector<string> chLineLabel/*[rfield.nflux_with_check][5]*/;

	/** string identifying the first continuum edge that occurred at this energy */
	vector<string> chContLabel/*[rfield.nflux_with_check][5]*/;

	/** method for transferring diffuse continuum
	 * either 'ots' or 'oux' where x is n for which */
	char chDffTrns[4];

	/** another flag, true if outward only - 
	 * used to multiply the ConInterOut continuum for creating the interactive
	 * continuum - when false, no outward only, this is not added */
	bool lgOutOnly;

	/** ipEnergyBremsThin is index for lowest energy thin to ff abs and plasma frequency
	 * EnergyBremsThin is energy there, Ryd */
	long int ipEnergyBremsThin;
	realnum EnergyBremsThin;

	/** index of highest cell with positive Boltzmann factor */
	long int ipMaxBolt;

	/** turn off continuum pumping, set with 'no induced processes' command */
	bool lgInducProcess;

	/** saves for the upward and downward Compton shifts */
	vector<double> comup, comdn;

	/** array indices for centers of B and V filters */
	long int ipB_filter , ipV_filter;

	/** these are the lower and upper bounds for the G0 radiation field
	 * used by Tielens & Hollenbach in their PDR work */
	long int ipG0_TH85_lo , ipG0_TH85_hi;

	/** these are the lower and upper bounds for the G0 radiation field
	 * used by Tielens & Hollenbach in their PDR work */
	long int ipG0_DB96_lo , ipG0_DB96_hi;

	/** these are the lower and upper bounds for the special G0 radiation field */
	long int ipG0_spec_lo , ipG0_spec_hi;

	/** this is the wavelength where Bertoldi & Draine estimate the Habing field */
	long int ip1000A;

	/** extinction in magnitudes at B and V filters for a point source,
	 * this does not discount forward scattering */
	double extin_mag_B_point , extin_mag_V_point;

	/** extinction in magnitudes at B and V filters for a resolved source,
	 * this does discount forward scattering, so is appropriate for a resolved source */
	double extin_mag_B_extended , extin_mag_V_extended;

	/** these are total opacities at these wavelengths, used to stop at exact Av */
	double opac_mag_B_point, opac_mag_V_point, opac_mag_B_extended , opac_mag_V_extended;

	/** coefficients for fitting Tarter expressions for Compton 
	 * heating and cooling, over full energy range of code */
	vector<realnum> csigh, csigc;

	/** ee brems emission, still needs to be multiplied with pow2(dense.eden) */
	vector<realnum> eeBremsDif;

	double comtot, 
	  cmheat, 
	  cmcool, 
	  cinrat;
	bool lgComptonOn;

	/** set true if Compton cooling underflows */
	bool lgComUndr;

	double totpow[LIMSPC], 
	  slope[LIMSPC], 
	  cutoff[LIMSPC][3], 
	  spfac[LIMSPC];

	/** option to have continuum intensity be time dependent */
	bool lgTimeVary[LIMSPC];

	/* beamed or isotropic continuum?  if isotropic then does not vary
	 * with time */
	bool lgBeamed[LIMSPC];

	/** 1 / cos( illumination angle, angle measured from normal,
	 * default is angle=zero, normal illumination, DirectCos = 1 */
	realnum OpticalDepthScaleFactor[LIMSPC];

	Illumination::IlluminationType Illumination[LIMSPC];

	/** nShape is SED shape index number, this must equal the number
	 * of field intensities that are specified
	 * ipSpec is radiation field source number 
	 */
	long int nShape, 
	  ipSpec;

	/** these are used for interpolate and table commands, 
	 * all have two indices, continuum and frequency
	 * tNuRyd is the linear energy Rydberg of continuum point */
	/** must be explicit arrays again so that
	 * table commands will work before continuum defined.*/
	vector<Energy> tNu[LIMSPC];
	vector<realnum> tslop[LIMSPC];
	/** this is the log of f_nu of continuum point */
	vector<realnum> tFluxLog[LIMSPC];
	/** TABLE READ input is normalized on this radius */
	double TableRadius[LIMSPC];
	/** does TABLE READ SED need additional spherical dilution? */
	bool lgSphericalDilution[LIMSPC];
	
	long ncont[LIMSPC];

	/** used to store a check on the continuum mesh resolution scale factor
	 * this is used for the output of the SAVE TRANSMITTED CONTINUUM command */
	double RSFCheck[LIMSPC];

	/** energy range over which the intensity is integrated for normalizing
	 * each continuum source that contributes to the total source */
	double range[LIMSPC][2];

	/** chSpNorm says how spectrum was normalized - intensity or luminosity case
	 * chRSpec says whether per unit area or tot 4 pi  */
	char chSpNorm[LIMSPC][5], 
		chRSpec[LIMSPC][5], 
		chSpType[LIMSPC][6],
		chCumuType[5];

	/** these are total numbers of photons over various energy ranges */
	realnum qhtot, 
	  qhe, 
	  qheii, 
	  qbal, 
	  qrad, 
	  qtot;

	/** hydrogen ionization parameter */
	realnum uh;

	/** helium ion ionization parameter */
	realnum uheii;

	/** lgUSphON flag set when we hit Stromgren radius in spherical geometry	*/
	bool lgUSphON;
	/** the Stromgren radius var set to get u spherical */
	realnum rstrom;

	/** flag set if incident radiation field less than
	 * 10x the Habing ISM field */
	bool lgHabing;

	/** heaviest element to be considered - default is iron - the fine opacity
	 * array resolution depends on heaviest element and lowest temperature */
	long int fine_opac_nelem;

	/** number of resolution elements over width of this element, default is 4 */
	long int fine_opac_nresolv;

	/** number of cm/s of each cell in fine mesh */
	realnum fine_opac_velocity_width;

private:
	/** these are TOTAL transmission coefficients for fine opacity degraded to coarse continuum */
	vector<realnum> trans_coef_total;

public:
	bool trans_coef_total_stale;

	/** and array indices for lower and upper bounds of each coarse continuum mapped onto the
	 * fine continuum 0 (false) if fine continuum does not extend to these cells */
	vector<long> ipnt_coarse_2_fine;

	/** low and high bounds of fine continuum - set by need to include all resonance lines */
	realnum fine_ener_lo, fine_ener_hi;
	/** the number of fine continuum cells - will be several million */
	long nfine;
	/** the dimensionless resolution of the fine continuum - dE/E */
	double fine_resol;
	/** the fine continuum opacity array  */
	vector<realnum> fine_opac_zone;
	/** total optical depth array for fine continuum */
	vector<realnum> fine_opt_depth;
	/** energies at center of each bin for fine continuum */
	vector<realnum> fine_anu;

	/** map of list of line stack indices (value)
	 *  centered on fine opacity vector index (key) */
	unordered_map<long, vector<int> > fine_lstack;

	/** shift in fine continuum rest frame scale due to velocity gradient, 
	 * rest frame is velocity of first zone, positive means that first zone 
	 * is blue shifted relative to current zone.  This is a decelerating
	 * flow.  Evaluated in RT_line_all */
	long int ipFineConVelShift;

	/** says that fine optical depths will be saved, so save them */
	bool lgSaveOpacityFine;

	/** flag saying whether to include H1 Lya ots in the radiation field - 
	 * usually true, set false with no Lya ots command */
	bool lgLyaOTS;

	/** flag saying whether to include HeII Lya and rec cont ots in the 
	 * radiation field - - usually true, set false with no HeII ots command */
	bool lgHeIIOTS;

	/** flag saying whether to kill outward only lines */
	bool lgKillOutLine;

	/** flag saying whether to kill outward only continuum */
	bool lgKillOutCont;

	/** flag saying whether to kill ots lines */
	bool lgKillOTSLine;

	/** following deal with plasma frequency, which enters the continuum
	 * array for even moderate densities due to very low frequencies considered */
	/** set true if plasma freq enters energy array */
	bool lgPlasNu;
	/** plasma frequency for current position in slab */
	realnum plsfrq, 
		/** store highest energy plasma frequency encountered*/
		plsfrqmax;
	// the zone where the plasma frequency is evaluated 
	long int nZonePlsFrqEval;
	/** pointer to current plasma frequency */
	long int ipPlasma, 
		/** pointer to largest plasma freq encountered */
	  ipPlasmax;

	/** these are series of flags that say whether different parts of the
	 * continuum where entered ok */
	bool lgMMok, 
	  lgHPhtOK, 
	  lgXRayOK, 
	  lgGamrOK;

	/** lowest energy counted as gamma rays, 100 keV	*/
	realnum EnerGammaRay;
	long int ipEnerGammaRay;

	/** lgHionRad set to .true. if no hydrogen ionizing radiation */
	bool lgHionRad;

	/** these store photon occupation numbers at various energies in the continuum */
	realnum occmax, 
	  occmnu, 
	  tbrmax, 
	  tbrmnu, 
	  tbr4nu, 
	  occ1nu;

	/** flag saying that occupation number at 1 Ryd > 1 */
	bool lgOcc1Hi;

	/** intensity [erg cm-2 s-1] in incident and diffuse continua in
	 * current zone */
	realnum EnergyIncidCont ,
		EnergyDiffCont;

	// constructor
	t_rfield( )
	{
		nZonePlsFrqEval = -1;

		// the constant that multiplies the column density to get optical depth at 1 Ryd
		ExtinguishConvertColDen2OptDepth = (realnum)6.22e-18;
		// the power on the energy for the extinction
		ExtinguishEnergyPowerLow = (realnum)-2.43;
	}
	
	const realnum *getCoarseTransCoef();

	void setCoarseTransCoefPtr(size_t size)
	{
		trans_coef_total.resize(size);
	}
	void resetCoarseTransCoef()
	{
		for (long i=0; i<nflux_with_check; ++i)
			trans_coef_total[i] = 1.0;
		trans_coef_total_stale = true;
	}
	void setTrimming();
};
extern t_rfield rfield;

/**flux_correct_isotropic - Do not include isotropic continua in the reported
 *			    continua, or band fluxes, if explicitly requested.
 *			    This version of the function is intended for 'save'
 *			    commands.
 *
 * \param lgSaveIsotrp	isotropic option for save file
 * \param nEmType	type of output continuum (instantaneous or time-integrated)
 * \param iflux		index in the flux arrays
 *
 * \return		total flux, by default including isotropic continua
 */
double flux_correct_isotropic( const bool lgSaveIsotrp, const int nEmType, const int j );

/**flux_correct_isotropic - Do not include isotropic continua in the reported
 *			    continua, or band fluxes, if explicitly requested.
 *
 * \param nEmType	type of output continuum (instantaneous or time-integrated)
 * \param iflux		index in the flux arrays
 *
 * \return		total flux, by default including isotropic continua
 */
double flux_correct_isotropic( const int nEmType, const int j );

#endif /* RFIELD_H_ */
