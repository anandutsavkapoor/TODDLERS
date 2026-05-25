/* This file is part of Cloudy and is copyright (C)1978-2025 by Gary J. Ferland and
 * others.  For conditions of distribution and use see copyright notice in license.txt */
/*ParseSave parse the save command */
/*SaveFilesInit initialize save file pointers, called from cdInit */
/*CloseSaveFiles close all save files */
/*ChkUnits check for keyword UNITS on line, then scan wavelength or energy units if present */
#include "cddefines.h"
#include "parse.h"
#include "cddrive.h"
#include "elementnames.h"
#include "input.h"
#include "iterations.h"
#include "prt.h"
#include "rfield.h"
#include "hcmap.h"
#include "h2.h"
#include "version.h"
#include "grainvar.h"
#include "grid.h"
#include "save.h"
#include "parser.h"
#include "service.h"
#include "species.h"
#include "dense.h"
#include "continuum.h"

/* check for keyword UNITS on line, then scan wavelength or energy units if present */
STATIC const char* ChkUnits(Parser &p);

inline void saveXSPEC(unsigned int option)
{
	static const char* labelXSPEC[NUM_OUTPUT_TYPES] = {
		"XTOT", "XINC", "XATT", "XRFI", "XDFO", "XDFR", "XLNO", "XLNR", "XTRN", "XREF", "XSPM"
	};

	ASSERT( option < NUM_OUTPUT_TYPES );
	strcpy( save.chSave[save.nsave], labelXSPEC[option] );
	grid.lgOutputTypeOn[option] = true;
	save.FITStype[save.nsave] = option;
}

/* NB NB NB NB NB NB NB NB NB NB
 *
 * if any "special" save commands are added (commands that copy the file pointer
 * into a separate variable, e.g. like SAVE _DR_), be sure to add that file pointer
 * to SaveFilesInit and CloseSaveFiles !!!
 *
 * SAVE FILE POINTERS SHOULD NEVER BE ALTERED BY ROUTINES OUTSIDE THIS MODULE !!!
 *
 * hence initializations of save file pointers should never be included in zero() !!
 * the pointer might be lost without the file being closed
 * 
 * NB NB NB NB NB NB NB NB NB NB */

/* SAVING HEADERS OF SAVE FILES
 *
 * save.lgSaveHeader() determines whether header should be saved
 * save.SaveHeaderDone() is called when saving the header is done
 *
 * inside ParseSave():
 *
 * save the header into the string chHeader when the command
 * is parsed. Use the command sncatf() to do this. The header
 * is then automatically printed at the end of the routine
 *
 * outside ParseSave():
 *
 * sometimes saving the header cannot be done in ParseSave()
 * because the necessary information is not yet available.
 * In that case save directly to the file in SaveDo() before
 * doing any other I/O as follows:
 *
 *   if( save.lgSaveHeader(ipPun) )
 *   {
 *       fprintf( save.params[ipPun].ipPnunit, "#some header..." );
 *       save.SaveHeaderDone(ipPun);
 *   }
 *
 * Note that you can only save a header in one location, so
 * saving part of the header in ParseSave() and part in SaveDo()
 * will not work as expected! Only the first part will be printed.
 */


void ParseSave(Parser& p)
{
	/* pointer to column across line image for free format number reader*/
	long int i, nelem;

	DEBUG_ENTRY( "ParseSave()" );

	/* check that limit not exceeded */
	if( save.nsave >= LIMPUN )
	{
		fprintf( ioQQQ, 
			"The limit to the number of SAVE options is %ld.  Increase "
			"LIMPUN in save.h if more are needed.\nSorry.\n", 
		  LIMPUN );
		cdEXIT(EXIT_FAILURE);
	}

	/* initialize this flag, forced true for special cases below (e.g. for FITS files) */
	save.lgSaveToSeparateFiles[save.nsave] = p.nMatch("SEPA");

	/* LAST keyword is an option to save only on last iteration */
	save.lgPunLstIter[save.nsave] = p.nMatch("LAST");

	/* get file name for this save output.
	 * GetQuote does the following -
	 * first copy original version of file name into chLabel,
	 * string does include null termination.
	 * set filename in OrgCard and second parameter to spaces so 
	 * that not picked up below as keyword
	 * last parameter says whether to abort if no quote found 	 */
	string chLabel;
	if( p.GetQuote( chLabel ) )
		p.StringError();

	if( !grid.lgInsideGrid )
		save.chFileName.push_back(chLabel);

	/* check that name is not same as opacity.opc, a special file */
	if( strcmp(chLabel.c_str() , "opacity.opc") == 0 )
	{
		fprintf( ioQQQ, "ParseSave will not allow save file name %s, please choose another.\nSorry.\n",
					chLabel.c_str());
		cdEXIT(EXIT_FAILURE);
	}
	else if( chLabel=="" )
	{
		fprintf( ioQQQ, "ParseSave found a null file name between double quotes, please check command line.\nSorry.\n");
		cdEXIT(EXIT_FAILURE);
	}

	/* now copy to chFilename, with optional grid prefix first (set with set save prefix command) */
	string chFilename = save.chGridPrefix + save.chFilenamePrefix + chLabel;

	/* there may be a second file name, and we need to get it off the line
	 * before we parse options, last false parameter says not to abort if
	 * missing - this is not a problem at this stage */
	bool lgSecondFilename;
	string chSecondFilename;
	if( p.GetQuote( chSecondFilename ) )
		lgSecondFilename = false;
	else
	{
		lgSecondFilename = true;
		trimTrailingWhiteSpace( chSecondFilename );
	}

	/* CLOBBER clobber keyword is an option to overwrite rather than
	 * append to a given file */
	if( p.nMatch("CLOB") )
	{
		if( p.nMatch(" NO ") )
		{
			/* do not clobber files */
			save.lgNoClobber[save.nsave] = true;
		}
		else
		{
			/* clobber files */
			save.lgNoClobber[save.nsave] = false;
		}
	}

	/** code printed log quantities, historically.  Maintain backwards compatibility */
	save.lgPrtOldStyleLogs[save.nsave] = p.nMatch(" LOG");

	string chTitle;

	/* put version number and title of model on output file, but only if
	 * this is requested with a "title" on the line*/
	/* >>chng 02 may 10, invert logic from before - default had been title */
	/* put version number and title of model on output file, but only if
	 * there is not a "no title" on the line*/
	if( !p.nMatch("NO TI") && p.nMatch("TITL") )
		chTitle = "# " + string(t_version::Inst().chVersion) + " " + input.chTitle + "\n";

	/* usually results for each iteration are followed by a series of
	 * hash marks, ####, which fool excel.  This is an option to not print
	 * the line.  If the keyword NO HASH no hash appears the hash marks 
	 * will not occur */
	if( p.nMatch("NO HA") )
		save.lgHashEndIter[save.nsave] = false;

	ostringstream chHeader;
	// initialize to empty string so that we can concatenate further down

	/* save opacity must come first since elements appear as sub-keywords */
	if( p.nMatch("OPAC") )
	{
		/* check for keyword UNITS on line, then scan wavelength or energy units if present,
		 * units are copied into save.chConSavEnr */
		save.chConSavEnr[save.nsave] = ChkUnits(p);

		strcpy( save.chSave[save.nsave], "OPAC" );

		/* "every" option to save this on every zone -
		 * not present then only last zone is saved */
		if( p.nMatch("EVER" ) )
		{
			/* save every zone */
			save.lgSaveEveryZone[save.nsave] = true;
			save.nSaveEveryZone[save.nsave] = 1;
		}
		else
		{
			/* only save last zone */
			save.lgSaveEveryZone[save.nsave] = false;
			save.nSaveEveryZone[save.nsave] = 1;
		}

		if( p.nMatch("TOTA") )
		{
			/* DoPunch will call save_opacity to parse the subcommands
			 * save total opacity */
			strcpy( save.chOpcTyp[save.nsave], "TOTL" );
			sncatf( chHeader, 
				"#nu/%s\tTot opac\tAbs opac\tScat opac\tAlbedo\telem\n",
				save.chConSavEnr[save.nsave]);
		}

		else if( p.nMatch("FIGU") )
		{
			/* do figure for hazy */
			strcpy( save.chOpcTyp[save.nsave], "FIGU" );
			sncatf( chHeader, 
				"#nu/%s\tH\tHe\ttot opac\n",
				save.chConSavEnr[save.nsave] );
		}

		else if( p.nMatch("FINE") )
		{
			/* save the fine opacity array */
			rfield.lgSaveOpacityFine = true;
			strcpy( save.chOpcTyp[save.nsave], "FINE" );
			/* check for keyword UNITS on line, then scan wavelength or energy units if present,
			 * units are copied into save.chConSavEnr */
			save.chConSavEnr[save.nsave] = ChkUnits(p);

			sncatf( chHeader, 
				"#nu/%s\topac\n",
				save.chConSavEnr[save.nsave] );

			/* range option - important since so much data - usually want to
			 * only give portion of the continuum */
			if( p.nMatch("RANGE") ) 
			{
				/* get lower and upper range, eventually must be in Ryd */
				double Energy1 = p.FFmtRead();
				double Energy2 = p.FFmtRead();
				if( p.lgEOL() )
				{
					fprintf(ioQQQ,"There must be two numbers, the lower and upper energy range in Ryd.\nSorry.\n");
					cdEXIT(EXIT_FAILURE);
				}
				if( p.nMatch("UNIT" ) )
				{
					// apply units to range option
					const char *energyUnits = p.StandardEnergyUnit();
					Energy unitChange;
					unitChange.set(Energy1, energyUnits );
					Energy1 = unitChange.Ryd();
					unitChange.set(Energy2, energyUnits );
					Energy2 = unitChange.Ryd();
				}
				/* get lower and upper rang in Ryd */
				save.punarg[save.nsave][0] = (realnum)MIN2( Energy1 , Energy2 );
				save.punarg[save.nsave][1] = (realnum)MAX2( Energy1 , Energy2 );
				//fprintf(ioQQQ , "DEBUG units change fine %.3e %.3e\n" , save.punarg[save.nsave][0] ,
				//		save.punarg[save.nsave][1] );
				//cdEXIT(EXIT_FAILURE);
			}
			else
			{
				/* these mean full energy range */
				save.punarg[save.nsave][0] = 0.;
				save.punarg[save.nsave][1] = 0.;
			}
			/* optional last parameter - how many points to bring together */
			save.punarg[save.nsave][2] = (realnum)p.FFmtRead();

			if( !p.lgEOL()  && save.punarg[save.nsave][2] < 1 )
			{
				fprintf(ioQQQ,"The number of fine continuum points to skip must be > 0 \nSorry.\n");
				cdEXIT(EXIT_FAILURE);
			}

			/* default is to average together ten */
			if( p.lgEOL() )
				save.punarg[save.nsave][2] = 10;

			/* ALL means to report all cells, even those with zero, to maintain uniform output */
			if( p.nMatch(" ALL" ) )
				save.punarg[save.nsave][2] *= -1;
		}

		else if( p.nMatch("GRAI") )
		{
			/* check for keyword UNITS on line, then scan wavelength or energy units, 
			 * sets save.chConSavEnr*/
			save.chConSavEnr[save.nsave] = ChkUnits(p);

			/* save grain opacity command, give optical properties of grains in calculation */
			strcpy( save.chSave[save.nsave], "DUSO" );
			/* save grain opacity command in twice, here and above in opacity */
			sncatf( chHeader, 
				"#grain\tnu/%s\tabs+scat*(1-g)\tabs\tscat*(1-g)\tscat\tscat*(1-g)/[abs+scat*(1-g)]\n",
				save.chConSavEnr[save.nsave] );
		}

		else if( p.nMatch("BREM") )
		{
			/* save bremsstrahlung opacity */
			strcpy( save.chOpcTyp[save.nsave], "BREM" );
			sncatf( chHeader, 
				"#nu\tbrems opac\te-e brems opac\n" );
		}

		else if( p.nMatch("SHEL") )
		{
			/* save shells, a form of the save opacity command for showing subshell crossections*/
			strcpy( save.chSave[save.nsave], "OPAC" );

			/* save subshell cross sections */
			strcpy( save.chOpcTyp[save.nsave], "SHEL" );

			/* this is element */
			save.punarg[save.nsave][0] = (realnum)p.FFmtRead();

			/* this is ion */
			save.punarg[save.nsave][1] = (realnum)p.FFmtRead();

			/* this is shell */
			save.punarg[save.nsave][2] = (realnum)p.FFmtRead();

			if( p.lgEOL() )
			{
				fprintf( ioQQQ, "There must be atom number, ion, shell\nSorry.\n" );
				cdEXIT(EXIT_FAILURE);
			}
			sncatf( chHeader, 
				"#sub shell cross section\n" );
		}

		else if( p.nMatch("ELEM") )
		{
			/* save element opacity, produces n name.n files, one for each stage of 
			 * ionization.  the name is the 4-char version of the element's name, and
			 * n is the stage of ionization.  the file name on the card is ignored.
			 * The code stops in save_opacity after these files are produced. */

			/* this will be used as check that we did find an element on the command lines */
			/* nelem is -1 if an element was not found */
			if( (nelem = p.GetElem() ) < 0 )
			{
				fprintf( ioQQQ, "I did not find an element name on the opacity element command.  Sorry.\n" );
				cdEXIT(EXIT_FAILURE);
			}

			/* copy string over */
			strcpy( save.chOpcTyp[save.nsave], elementnames.chElementNameShort[nelem] );
		}
		else
		{
			fprintf( ioQQQ, " I did not recognize a keyword on this save opacity command.\n" );
			fprintf( ioQQQ, " Sorry.\n" );
			cdEXIT(EXIT_FAILURE);
		}
	}

	/* save H2 has to come early since it has many suboptions */
	else if( p.nMatchErase(" H2 ") )
	{
		/* this is in mole_h2_io.c */
		h2.H2_ParseSave( p, chHeader );
	}

	/* save HD has to come early since it has many suboptions */
	else if( p.nMatchErase(" HD ") )
	{
		/* this is in mole_h2_io.c */
		hd.H2_ParseSave( p, chHeader );
	}

	/* save grain abundance will be handled later */
	else if( p.nMatch("ABUN") && !p.nMatch("GRAI") )
	{
		/* save abundances */
		strcpy( save.chSave[save.nsave], "ABUN" );
		sncatf( chHeader, 
			"#abund H" );
		for(nelem=ipHELIUM;nelem<LIMELM; ++nelem )
		{
			sncatf( chHeader,
				"\t%s",
				elementnames.chElementNameShort[nelem] );
		}
		sncatf( chHeader, "\n");
	}

	else if( p.nMatch(" AGE") )
	{
		/* save ages */
		strcpy( save.chSave[save.nsave], "AGES" );
		sncatf( chHeader, 
			"#ages depth\tt(cool)\tt(H2 dest)\tt(CO dest)\tt(OH dest)\tt(H rec)\n" );
	}

	else if( p.nMatch(" AGN") )
	{
		/* save tables needed for AGN3 */
		strcpy( save.chSave[save.nsave], " AGN" );
		/* this is the AGN option, to produce a table for AGN */

		/* charge exchange rate coefficients */
		if( p.nMatch("CHAR") )
		{
			strcpy( save.chSave[save.nsave], "CHAG" );
			sncatf( chHeader, 
				"#charge exchange rate coefficnt\n" );
		}

		else if( p.nMatch("RECO") )
		{
			/* save recombination rates for AGN3 table */
			strcpy( save.chSave[save.nsave], "RECA" );
			sncatf( chHeader, 
				"#Recom rates for AGN3 table\n" );
		}

		else if( p.nMatch("OPAC") )
		{
			/* create table for appendix in AGN */
			strcpy( save.chOpcTyp[save.nsave], " AGN" );
			strcpy( save.chSave[save.nsave], "OPAC" );
		}

		else if( p.nMatch("HECS") )
		{
			/* create table for appendix in AGN */
			strcpy( save.chSaveArgs[save.nsave], "HECS" );
			sncatf( chHeader, 
				"#AGN3 he cs \n" );
		}

		else if( p.nMatch("HEMI") )
		{
			/* HEMIS - continuum emission needed for chap 4 of AGN3 */
			strcpy( save.chSaveArgs[save.nsave], "HEMI" );

			/* check for keyword UNITS on line, then scan wavelength or energy units if present,
			 * units are copied into save.chConSavEnr */
			save.chConSavEnr[save.nsave] = ChkUnits(p);
		}
		else if( p.nMatch("RECC") )
		{
			/* recombination cooling, for AGN */
			strcpy( save.chSave[save.nsave], "HYDr" );
			sncatf( chHeader, 
				"#T\tbAS\tb1\tbB\n" );
		}
		else
		{
			fprintf( ioQQQ, " I did not recognize this option on the SAVE AGN command.\n" );
			fprintf( ioQQQ, " Sorry.\n" );
			cdEXIT(EXIT_FAILURE);
		}
	}

	else if( p.nMatch("ARRA") && p.nMatch( "LEVE" ) )
	{
		strcpy( save.chSave[save.nsave], "IMAG" );

		string species = chLabel;

		if( species == "" )
		{
			fprintf( ioQQQ, "Empty species label: ''\n" );
			cdEXIT( EXIT_FAILURE );
		}

		save.img_matrix.lgImgRates = true;
		save.img_matrix.setSpecies( species );
		if( p.nMatch( "FITS" ) )
			save.img_matrix.lgFITS = true;
		else if( p.nMatch( " PPM" ) )
			save.img_matrix.lgFITS = false;
		else
		{
			fprintf( ioQQQ, "Please use either the FITS or PPM keywords\n" );
			cdEXIT( EXIT_FAILURE );
		}

		if( p.nMatch( "ITER" ) )
			save.img_matrix.iteration = p.getNumberCheck( "iter" );
		if( p.nMatch( "ZONE" ) )
			save.img_matrix.zone = p.getNumberCheck( "zone" );

		// NB NB
		//
		// This command generates images for debugging purposes,
		// and it is executed near the call to the linear algebra
		// function that solves the rate equations for level
		// populations
		//
		// Because there is no output file for save_do.cpp to
		// write in, we have to leave the function immediately,
		// or a file will be created with the given species as
		// its name
		//
		return;
	}

	else if( p.nMatch("AVER") )
	{
		/* save averages */
		strcpy( save.chSave[save.nsave], "AVER" );
		/* no need to print this standard line of explanation*/
		/*sncatf( chHeader, " asserts\n" );*/

		/* actually get the averages from the input stream, and allocate the 
		 * space in the arrays 
		 * save io unit not used in read */
		parse_save_average( p, save.nsave, chHeader );
	}

	/* save charge transfer */
	else if( p.nMatch("CHAR") && p.nMatch("TRAN") )
	{
		/* NB in SaveDo only the first three characters are compared to find this option,
		 * search for "CHA" */
		/* save charge transfer */
		strcpy( save.chSave[save.nsave], "CHAR" );
		sncatf( chHeader, 
			"#charge exchange rate coefficient\n" );
	}

	// save chianti collision strengths in physical units
	else if( p.nMatch("CHIA"))
	{
		strcpy( save.chSave[save.nsave], "CHIA" );
	}

	else if( p.nMatch("CHEM") )
	{
		if( p.nMatch( "RATE" ) )
		{
			/* >>chng 06 May 30, NPA.  Save reaction rates for selected species */
			if( lgSecondFilename )
			{
				if( p.nMatch( "DEST" ) )
					strcpy( save.chSaveArgs[save.nsave], "DEST" );
				else if( p.nMatch( "CREA" ) )
					strcpy( save.chSaveArgs[save.nsave], "CREA" );
				else if( p.nMatch( "CATA" ) )	
					strcpy( save.chSaveArgs[save.nsave], "CATA" );
				else if( p.nMatch( "ALL" ) )
					strcpy( save.chSaveArgs[save.nsave], "ALL " );
				else
					strcpy( save.chSaveArgs[save.nsave], "DFLT" );
		
				if( p.nMatch("COEF") )			
					strcpy( save.chSave[save.nsave], "CHRC" );
				else
					strcpy( save.chSave[save.nsave], "CHRT" );

				save.optname[save.nsave] = chSecondFilename;
				// Haven't read chemistry database yet, so put off setting up header
				//sncatf( chHeader, "#");  
			} 

			else
			{
				fprintf(ioQQQ," A species label must appear within a second set of quotes (following the output filename).\n" );
				fprintf( ioQQQ, " Sorry.\n" );
				cdEXIT(EXIT_FAILURE);
			}

		}
		else
		{
			fprintf( ioQQQ, " I did not recognize a sub keyword on this SAVE CHEMISTRY command.\n" );
			fprintf( ioQQQ, " Sorry.\n" );
			cdEXIT(EXIT_FAILURE);
		}
	}

	else if( p.nMatch("COMP") )
	{
		/* save Compton, details of the energy exchange problem */
		strcpy( save.chSave[save.nsave], "COMP" );
		sncatf( chHeader, 
			"#nu, comup, comdn\n" );
	}

	else if( p.nMatch("COOL") )
	{
		/* save cooling, actually done by routine cool_save */
		if( p.nMatch("EACH") )
		{
			strcpy( save.chSave[save.nsave], "EACH");
			sncatf( chHeader, 
					 "#depth(cm)\tTemp(K)\tCtot(erg/cm3/s)\t" );
			for( int i = 0 ; i < LIMELM ; i++ )
			{
				sncatf(chHeader,
				       "%s\t", elementnames.chElementSym[i] );
			}
			sncatf(chHeader,
			       "%s", "molecule\tdust\tH2cX\tCT C\tH-fb\tH2ln\tHD  \tH2+ \tFF_H\tFF_M\teeff\tadve\tComp\tExtr\tExpn\tCycl\tdima\n" );
		}
		else
		{
			strcpy( save.chSave[save.nsave], "COOL");
			/*>>chng 06 jun 06, revise to be same as save cooling */
			sncatf( chHeader,
				 "#depth cm\tTemp K\tHtot erg/cm3/s\tCtot erg/cm3/s\tAdve Ctot erg/cm3/s\tcool fracs\n" );
		}
	}

	// punch the dominant rates for a given species
	else if( p.nMatch("DOMI") && p.nMatch("RATE"))
	{			
		if( !lgSecondFilename )
		{
			fprintf( ioQQQ,"This command requires two items in quotes (a filename and a species label).  Only one set of quotes was found.\nSorry.\n");
			cdEXIT(EXIT_FAILURE);
		}
		/* in this case the "second filename" is really the species label. */
		save.chSpeciesDominantRates[save.nsave] = chSecondFilename;

		/* save dominant rates "species" */
		strcpy( save.chSave[save.nsave], "DOMI" );
		sncatf( chHeader, 
			"#depth cm\t%s col cm-2\tsrc s-1\tsnk s-1\n", 
				  save.chSpeciesDominantRates[save.nsave].c_str() );

		save.nLineList[save.nsave] = 1;
		int nreact = (realnum)p.FFmtRead();
		if ( !  p.lgEOL() )
			save.nLineList[save.nsave] = nreact;
	}

	else if( p.nMatch("DYNA") )
	{
		/* save something dealing with dynamics 
		 * in SaveDo the DYN part of key is used to call DynaPunch,
		 * with the 4th char as its second argument.  DynaSave uses that
		 * 4th letter to decide the job */
		if( p.nMatch( "ADVE" ) )
		{
			/* save information relating to advection */
			strcpy( save.chSave[save.nsave], "DYNa");
			sncatf( chHeader, 
				"#advection depth\tHtot\tadCool\tadHeat\tdCoolHeatdT\t"
				"Source[hyd][hyd]\tRate\tEnthalph\tadSpecEnthal\n" );
		}
		else
		{
			fprintf( ioQQQ, " I did not recognize a sub keyword on this SAVE DYNAMICS command.\n" );
			fprintf( ioQQQ, " Sorry.\n" );
			cdEXIT(EXIT_FAILURE);
		}
	}

	else if( p.nMatch("ENTH") )
	{
		/* contributors to the total enthalpy */
		strcpy( save.chSave[save.nsave], "ENTH" );
		sncatf( chHeader, 
			"#depth\tTotal\tExcit\tIoniz\tBind\tKE\tther+PdV\tmag \n" );
	}

	else if( p.nMatch("EXEC") && p.nMatch("TIME") )
	{
		/* output the execution time per zone */
		strcpy( save.chSave[save.nsave], "XTIM" );
		sncatf( chHeader, 
			"#zone\tdTime\tElapsed t\n" );
	}

	else if( p.nMatch("FEII") || p.nMatch("FE II") )
	{
		fprintf(ioQQQ,"Error: The 'save feii' commands are obsolete. "
				" They have been replaced by the 'save species' commands.\n");
		fprintf(ioQQQ,"Please update your input.\nSorry.\n");
		cdEXIT( EXIT_FAILURE );
	}

	/* the save continuum command, with many options,
	 * the first 3 char of the chSave flag will always be "CON" 
	 * with the last indicating which one */
	else if( p.nMatch("CONT") && !p.nMatch("XSPE") && !p.nMatch("SPEC") )
	{
		/* this flag is checked in PrtComment to generate a caution
		 * if continuum is saved but iterations not performed */
		save.lgPunContinuum = true;

		save.lgPrtIsotropicCont[save.nsave] = true;
		if( p.nMatch( " NO " ) && p.nMatch( "ISOT" ) )
			save.lgPrtIsotropicCont[save.nsave] = false;

		/* check for keyword UNITS on line, then scan wavelength or energy units if present,
		 * units are copied into save.chConSavEnr */
		save.chConSavEnr[save.nsave] = ChkUnits(p);

		if( p.nMatch("BINS") )
		{
			/* continuum binning */
			strcpy( save.chSave[save.nsave], "CONB" );

			sncatf( chHeader, 
				"#Cont bin Anu/%s\tAnu/Ryd\td(anu)/Ryd\n",
				save.chConSavEnr[save.nsave] );
		}

		else if( p.nMatch("DIFF") )
		{
			if( p.nMatch("UNAT") )
			{
				/* save diffuse continuum unattenuated -
				 * cloud-integrated gas-phase nebular continuum emission
				 * (free-bound, free-free, two-photon) without
				 * self-absorption, summed over all zones. Does NOT
				 * include line emission or grain thermal emission.
				 * The attenuated version (which does include grains)
				 * is given alongside for comparison. */
				strcpy( save.chSave[save.nsave], "CODU" );

				sncatf( chHeader,
					"#energy/%s\tDiffContUnatt\tDiffContAtt\n",
					save.chConSavEnr[save.nsave] );
			}
			else
			{
				/* diffuse continuum, the locally emitted lines and continuum */
				strcpy( save.chSave[save.nsave], "COND" );

				/* by default gives lines and continuum separately only for
				 * last zone.  The keyword ZONE says to give the total for every
				 * zone in one very low row */
				if( p.nMatch("ZONE") )
				{
					sncatf( chHeader,
						"#energy/%s then emission per zone\n",
						save.chConSavEnr[save.nsave] );
					save.punarg[save.nsave][0] = 2.;

				}
				else
				{
					sncatf( chHeader,
						"#energy/%s\tConEmitLocal\tDiffuseLineEmission\tTotal\n",
						save.chConSavEnr[save.nsave] );
					save.punarg[save.nsave][0] = 1.;
				}
			}
		}

		else if( p.nMatch("EMIS") )
		{
			/* continuum volume emissivity and opacity as a function of radius */
			strcpy( save.chSave[save.nsave], "CONS" );

			double num = p.FFmtRead();
			if( p.lgEOL() )
				p.NoNumb( "continuum emissivity frequency" );
			save.emisfreq[save.nsave].set( num, save.chConSavEnr[save.nsave] );
			if( save.emisfreq[save.nsave].Ryd() < rfield.emm() ||
			    save.emisfreq[save.nsave].Ryd() > rfield.egamry() )
			{
				fprintf( ioQQQ, " The frequency is outside the Cloudy range\n Sorry.\n" );
				cdEXIT(EXIT_FAILURE);
			}

			sncatf( chHeader, 
				 "#Radius\tdepth\tnujnu\tkappa_abs\tkappa_sct @ %e Ryd\n",
				 save.emisfreq[save.nsave].Ryd() );
		}

		else if( p.nMatch("EMIT") )
		{
			/* continuum emitted by cloud */
			strcpy( save.chSave[save.nsave], "CONE" );

			sncatf( chHeader, 
				"#Energy/%s\treflec\toutward\ttotal\tline\tcont\n",
				save.chConSavEnr[save.nsave] );
		}

		else if( p.nMatch("FINE" ) )
		{
			rfield.lgSaveOpacityFine = true;
			/* fine transmitted continuum cloud */
			strcpy( save.chSave[save.nsave], "CONf" );

			sncatf( chHeader, 
				"#Energy/%s\tTransmitted\tSpecLine\tSingle-Line Opt Depth\n",
				save.chConSavEnr[save.nsave] );

			/* range option - important since so much data */
			if( p.nMatch("RANGE") )
			{
				/* get lower and upper range, eventually must be in Ryd */
				double Energy1 = p.FFmtRead();
				double Energy2 = p.FFmtRead();
				if( p.lgEOL() )
				{
					fprintf(ioQQQ,"There must be two numbers, the lower and upper energies in Ryd.\nSorry.\n");
					cdEXIT(EXIT_FAILURE);
				}
				if( p.nMatch("UNIT" ) )
				{
					// apply units to range option
					const char *energyUnits = p.StandardEnergyUnit();
					Energy unitChange;
					unitChange.set(Energy1, energyUnits );
					Energy1 = unitChange.Ryd();
					unitChange.set(Energy2, energyUnits );
					Energy2 = unitChange.Ryd();
				}
				/* get lower and upper rang in Ryd */
				save.punarg[save.nsave][0] = (realnum)MIN2( Energy1 , Energy2 );
				save.punarg[save.nsave][1] = (realnum)MAX2( Energy1 , Energy2 );
				//fprintf(ioQQQ , "DEBUG units change fine %.3e %.3e\n" , save.punarg[save.nsave][0] ,
				//		save.punarg[save.nsave][1] );
				//cdEXIT(EXIT_FAILURE);
			}
			else
			{
				/* these mean full energy range */
				save.punarg[save.nsave][0] = 0.;
				save.punarg[save.nsave][1] = 0.;
			}
			/* optional last parameter - how many points to bring together */
			save.punarg[save.nsave][2] = (realnum)p.FFmtRead();

			if( !p.lgEOL()  && save.punarg[save.nsave][2] < 1 )
			{
				fprintf(ioQQQ,"The number of fine continuum points to skip must be > 0 \nSorry.\n");
				cdEXIT(EXIT_FAILURE);
			}

			/* default is to bring together ten */
			if( p.lgEOL() )
				save.punarg[save.nsave][2] = 10;

		}

		else if( p.nMatch("GRAI") )
		{
			/* save grain continuum in optically thin limit */
			strcpy( save.chSave[save.nsave], "CONG" );

			sncatf( chHeader, 
				"#energy\tgraphite\trest\ttotal\n" );
		}

		else if( p.nMatch("INCI") )
		{
			/* incident continuum */
			strcpy( save.chSave[save.nsave], "CONC" );

			sncatf( chHeader, 
				"#Incident Continuum, Enr\tnFn\tOcc Num\n" );
		}

		else if( p.nMatch("INTE") )
		{
			/* continuum interactions */
			strcpy( save.chSave[save.nsave], "CONi" );

			sncatf( chHeader, 
				"#Continuum interactions inc \totslin \totscon \tConInterOut \toutlin\n" );
			/* this is option for lowest energy, if nothing then zero */
			save.punarg[save.nsave][0] = (realnum)p.FFmtRead();
		}

		else if( p.nMatch("IONI") )
		{
			/* save ionizing continuum*/
			strcpy( save.chSave[save.nsave], "CONI" );

			/* this is option for lowest energy, if nothing then zero */
			save.punarg[save.nsave][0] = (realnum)p.FFmtRead();

			/* this is option for smallest interaction to save, def is 1 percent */
			save.punarg[save.nsave][1] = (realnum)p.FFmtRead();
			if( p.lgEOL() )
				save.punarg[save.nsave][1] = 0.01f;

			/* "every" option to save this on every zone -
			 * not present then only last zone is saved */
			if( p.nMatch("EVER" ) )
			{
				/* save every zone */
				save.lgSaveEveryZone[save.nsave] = true;
				save.nSaveEveryZone[save.nsave] = 1;
			}
			else
			{
				/* only save last zone */
				save.lgSaveEveryZone[save.nsave] = false;
				save.nSaveEveryZone[save.nsave] = 1;
			}

			/* put the header at the top of the file */
			sncatf( chHeader, 
				"#cell(on C scale)\tnu\tflux\tflx*cs\tFinc\totsl\totsc\toutlin\toutcon\trate/tot\tintegral\tline\tcont\n" );
		}

		else if( p.nMatch("TRAN") )
		{
			/* transmitted continuum */
			strcpy( save.chSave[save.nsave], "CONT" );
			save.lgPunLstIter[save.nsave] = true;
			/* Cloudy cannot read concatenated output back in... */
			save.lgSaveToSeparateFiles[save.nsave] = true;

			sncatf( chHeader, 
				"#ener\tTran Contin\ttrn coef\n" );
		}

		else if( p.nMatch(" TWO") )
		{
			/* total two photon continua rfield.TotDiff2Pht */
			strcpy( save.chSave[save.nsave], "CON2" );

			sncatf( chHeader, 
				"#energy\t n_nu\tnuF_nu \n" );
		}

		else if( p.nMatch(" RAW") )
		{
			/* "raw" continua */
			strcpy( save.chSave[save.nsave], "CORA" );

			sncatf( chHeader, 
				"#Raw Con anu\tflux\totslin\totscon\treflin\tConRefIncid\tConEmitReflec\tConInterOut\toutlin\toutlin_noplot\tConEmitOut\tline\tcont\tnLines\n" );
		}

		else if( p.nMatch("REFL") )
		{
			/* reflected continuum */
			strcpy( save.chSave[save.nsave], "CONR" );

			sncatf( chHeader, 
				"#Reflected\tcont\tline\ttotal\talbedo\tConID\n" );
		}

		else
		{
			/* this is the usual save continuum command,
			 * ipType is index for continuum array to set either
			 * iteration or cumulative output */
			int ipType = 0;
			if( p.nMatch( "CUMU" ) )
				ipType = 1;
			if( ipType == 1 && ! save.lgPrtIsotropicCont[save.nsave] )
			{
				fprintf( ioQQQ, "ERROR: Illegal request of isotropic continuum removal "
						"for time integrations\n" );
				cdEXIT( EXIT_FAILURE );
			}
			save.punarg[save.nsave][0] = (realnum)ipType;
			strcpy( save.chSave[save.nsave], "CON " );
			string chHold;
			chHold = "#Cont ";
			if( ipType > 0 )
				chHold = "#Cumul ";
			sncatf( chHeader, 
				"%s nu\tincident\ttrans\tDiffOut\tnet trans\treflc\ttotal\treflin\toutlin\tlineID\tcont\tnLine\n" ,
				chHold.c_str() );

			/* >>chng 06 apr 03, add "every" option to save this on every zone -
			 * if every is not present then only last zone is saved */
			if( p.nMatch("EVER" ) )
			{
				/* save every zone */
				save.lgSaveEveryZone[save.nsave] = true;
				/* option to say how many to skip */
				save.nSaveEveryZone[save.nsave] = (long)p.FFmtRead();
				if( p.lgEOL() )
					save.nSaveEveryZone[save.nsave] = 1;
			}
			else
			{
				/* only save last zone */
				save.lgSaveEveryZone[save.nsave] = false;
				save.nSaveEveryZone[save.nsave] = 1;
			}
		}
	}

	/* save information about convergence of this model 
	 * reason - why it did not converge an iteration
	 * error - zone by zone display of various convergence errors */
	else if( p.nMatch("CONV") )
	{
		if( p.nMatch("REAS") )
		{
			// this is a special save option handled below
			(void)0;
		}
		else if( p.nMatch("ERRO") )
		{
			/* save zone by zone errors in pressure, electron density, and heating-cooling */
			/* convergence error */
			strcpy( save.chSave[save.nsave], "CNVE" );
			sncatf( chHeader, 
				"#depth\tnPres2Ioniz\tP(cur)\tP%%error\tNE(cor)\tNE(cur)\tNE%%error\tHeat\tCool\tHC%%error\n" );
		}
		else if( p.nMatch("BASE") )
		{
			// this is a special save option handled below
			(void)0;
		}
		else
		{
			fprintf( ioQQQ, "There must be a second keyword on this command.\n" );
			fprintf( ioQQQ, "The ones I know about are REASON, ERROR, and BASE.\n" );
			fprintf( ioQQQ, "Sorry.\n" );
			cdEXIT(EXIT_FAILURE);
		}
	}

	else if( p.nMatch("SPECIES") && p.nMatch("DATA") && p.nMatch("SOURCE") )
	{
		// this is a special save option handled below
		(void)0;
	}

	else if( p.nMatch(" DR ") )
	{
		// this is a special save option handled below
		(void)0;
	}

        else if( p.nMatch(" DT ") )
	{
		// this is a special save option handled below
		(void)0;
	}

	else if( p.nMatch("ELEM") && !p.nMatch("GAMMA") && !p.nMatch("COOL") ) // do not trip on SAVE COOLING EACH ELEMENT
	{
		/* option to save ionization structure of some element
		 * will give each stage of ionization, vs depth */
		strcpy( save.chSave[save.nsave], "ELEM" );

		/* this returns element number on c scale */
		/* >>chng 04 nov 23, had converted to f scale, leave on c */
		nelem = p.GetElem();
		if( nelem < 0 || nelem >= LIMELM )
		{
			fprintf( ioQQQ, " I could not recognize a valid element name on this line.\n" );
			fprintf( ioQQQ, " Please check your input script. Bailing out...\n" );
			cdEXIT(EXIT_FAILURE);
		}

		/* this is the atomic number on the c scale */
		save.punarg[save.nsave][0] = (realnum)nelem;
		vector<string>& chList = save.chSaveSpecies[save.nsave];

		chList.clear();

		/* >>chng 04 nov 24, add DENSE option to print density rather than fraction */
		save.punarg[save.nsave][1] = 0;
		if( p.nMatch("DENS")  )
			save.punarg[save.nsave][1] = 1.;

		/* start printing header line - first will be the depth in cm */
		sncatf( chHeader, "#depth");

		/* next come the nelem+1 ion stages */
		for(i=0; i<=nelem+1;++i )
		{
			chList.push_back( makeChemical( nelem, i ) );
		}

		/* finally some fine structure or molecular populations */
		/* >>chng 04 nov 23, add fs pops of C, O 
		 * >>chng 04 nov 25, add molecules */
		
		if( nelem==ipHYDROGEN )
		{
			chList.push_back("H2");
		}
		else if( nelem==ipCARBON )
		{
			chList.push_back("C[1]");
			chList.push_back("C[2]");
			chList.push_back("C[3]");
			chList.push_back("C+[1]");
			chList.push_back("C+[2]");
			chList.push_back("CO");
		}
		else if( nelem==ipOXYGEN )
		{
			chList.push_back("O[1]");
			chList.push_back("O[2]");
			chList.push_back("O[3]");
		}

		for (size_t ic=0; ic != chList.size(); ++ic)
			sncatf( chHeader, "\t%s",chList[ic].c_str());

		/* finally the new line */
		sncatf( chHeader, "\n");
	}

	else if( p.nMatch("FITS") )
	{

#ifdef FLT_IS_DBL
		fprintf( ioQQQ, "Saving FITS files is not currently supported in double precision.\n" );
		fprintf( ioQQQ, "Please recompile without the FLT_IS_DBL option.\n" );
		fprintf( ioQQQ, "Sorry.\n" );
		cdEXIT(EXIT_FAILURE);
#else
		/* say that this is a FITS file output */
		save.lgFITS[save.nsave] = true;
		/* concatenating files in a grid run would be illegal FITS */
		save.lgSaveToSeparateFiles[save.nsave] = true;
		save.lgPunLstIter[save.nsave] = true;
		save.FITStype[save.nsave] = NUM_OUTPUT_TYPES;

		strcpy( save.chSave[save.nsave], "FITS" );
#endif

	}

	else if( p.nMatch("FRED") )
	{
		/* save out some stuff for Fred's dynamics project */
		sncatf( chHeader, 
			"#Radius\tDepth\tVelocity(km/s)\tdvdr(cm/s)\thden\teden\tTemperature\tRadAccel line\tRadAccel con\t"
			"Force multiplier\ta(e thin)\t"
			"HI\tHII\tHeI\tHeII\tHeIII\tC2\tC3\tC4\tO1\t"
			"O2\tO3\tO4\tO5\tO6\tO7\tO8\t" 
			"HI\tHII\tHeI\tHeII\tHeIII\tC2\tC3\tC4\tO1\t"
			"O2\tO3\tO4\tO5\tO6\tO7\tO8\tMg2\tMg2\tOVI(1034) TauIn\tTauCon\n");

		strcpy( save.chSave[save.nsave], "FRED" );
	}

	else if( p.nMatch("GAMM") )
	{
		/* save all photoionization rates for all subshells */
		sncatf( chHeader, 
			"#Photoionization rates \n" );
		if( p.nMatch("ELEMENT") )
		{
			/* element keyword, find element name and stage of ionization, 
			 * will print photoionization rates for valence of that element */
			strcpy( save.chSave[save.nsave], "GAMe" );

			/* this returns element number on c scale */
			nelem = p.GetElem();
			/* this is the atomic number on the C scale */
			save.punarg[save.nsave][0] = (realnum)nelem;

			/* this will become the ionization stage on C scale */
			save.punarg[save.nsave][1] = (realnum)p.FFmtRead() - 1;
			if( p.lgEOL() )
				p.NoNumb("element ionization stage" );
			if( save.punarg[save.nsave][1]<0 || save.punarg[save.nsave][1]> nelem+1 )
			{
				fprintf(ioQQQ,"Bad ionization stage - please check Hazy.\nSorry.\n");
				cdEXIT(EXIT_FAILURE);
			}
		}
		else
		{
			/* no element - so make table of all rates */
			strcpy( save.chSave[save.nsave], "GAMt" );
		}

	}
	else if( p.nMatch("GRAI") )
	{
		/* save grain ... options */
		if( p.nMatch("OPAC") )
		{
			// the save grain opacity command was already handled above (key "DUSO")
			(void)0;
		}
		else if( p.nMatch("ABUN") )
		{
			/* save grain abundance */
			strcpy( save.chSave[save.nsave], "DUSA" );
		}
		else if( p.nMatch("D/G ") )
		{
			/* save grain dust/gas mass ratio */
			strcpy( save.chSave[save.nsave], "DUSD" );
		}
		else if( p.nMatch("PHYS") )
		{
			/* save grain physical conditions */
			strcpy( save.chSave[save.nsave], "DUSP" );
		}
		else if( p.nMatch(" QS ") )
		{
			/* save absorption and scattering efficiency */
			strcpy( save.chSave[save.nsave], "DUSQ" );
		}
		else if( p.nMatch("TEMP") )
		{
			/* save temperatures of each grain species */
			strcpy( save.chSave[save.nsave], "DUST" );
		}
		else if( p.nMatch("DRIF") )
		{
			/* save drift velocity of each grain species */
			strcpy( save.chSave[save.nsave], "DUSV" );
		}
		else if( p.nMatch("EXTI") )
		{
			/* save grain extinction */
			strcpy( save.chSave[save.nsave], "DUSE" );
			sncatf( chHeader, 
				"#depth\tA_V(extended)\tA_V(point)\n" );
		}
		else if( p.nMatch("CHAR") )
		{
			/* save charge per grain (# elec/grain) for each grain species */
			strcpy( save.chSave[save.nsave], "DUSC" );
		}
		else if( p.nMatch("HEAT") )
		{
			/* save heating due to each grain species */
			strcpy( save.chSave[save.nsave], "DUSH" );
		}
		else if( p.nMatch("POTE") )
		{
			/* save floating potential of each grain species */
			strcpy( save.chSave[save.nsave], "DUSP" );
		}
		else if( p.nMatch("H2RA") )
		{
			/* save grain H2rate - H2 formation rate for each type of grains */
			strcpy( save.chSave[save.nsave], "DUSR" );
		}
		else
		{
			fprintf( ioQQQ, "There must be a second key on this GRAIN command; The options I know about follow (required key in CAPS):\n");
			fprintf( ioQQQ, "OPACity, ABUNdance, D/G mass ratio, PHYSical conditions,  QS , TEMPerature, DRIFt velocity, EXTInction, CHARge, HEATing, POTEntial, H2RAtes\nSorry.\n" );
			cdEXIT(EXIT_FAILURE);
		}
	}

	else if( p.nMatch("GAUN") )
	{
		strcpy( save.chSave[save.nsave], "GAUN" );
		sncatf( chHeader, 
			"#Gaunt factors.\n" );
	}
	else if( p.nMatch("GRID") )
	{
		strcpy( save.chSave[save.nsave], "GRID" );
		/* automatically generate no hash option */
		save.lgHashEndIter[save.nsave] = false;
		save.ipSaveGrid = save.nsave;
	}
	else if( p.nMatch( "HIST" ) )
	{
		/* save pressure history of current zone */
		if( p.nMatch( "PRES") )
		{
			/* save pressure history - density - pressure for this zone */
			strcpy( save.chSave[save.nsave], "HISp" );
			sncatf( chHeader, 
				"#iter zon\tdensity\tpres cur\tpres error\n" );
		}
		/* save temperature history of current zone */
		else if( p.nMatch( "TEMP" ) )
		{
			/* save pressure history - density - pressure for this zone */
			strcpy( save.chSave[save.nsave], "HISt" );
			sncatf( chHeader, 
				"#iter zon\ttemperature\theating\tcooling\n" );
		}
	}

	else if( p.nMatch("HTWO") )
	{
		fprintf(ioQQQ," Sorry, this command has been replaced with the "
			"SAVE H2 CREATION and SAVE H2 DESTRUCTION commands.\n");
		cdEXIT(EXIT_FAILURE);
	}

	/* QHEAT has to come before HEAT... */
	else if( p.nMatch("QHEA") )
	{
		/* this is just a dummy clause, do the work below after parsing is over. 
		 * this is a no-nothing, picked up to stop optimizer */
		((void)0);
	}

	else if( p.nMatch("HEAT") )
	{
		/* save heating */
		strcpy( save.chSave[save.nsave], "HEAT" );
		/*>>chng 06 jun 06, revise to be same as save cooling */
		sncatf( chHeader, 
			"#depth cm\tTemp K\tHtot erg/cm3/s\tCtot erg/cm3/s\tAdve Htot erg/cm3/s\theat fracs\n" );
	}

	else if( p.nMatch("HELI") && !p.nMatch("IONI") )
	{
		/* save helium & helium-like iso sequence, but not save helium ionization rate
		 * save helium line wavelengths */
		if( p.nMatch("LINE") && p.nMatch("WAVE") )
		{
			strcpy( save.chSave[save.nsave], "HELW" );
			sncatf( chHeader, 
				"#wavelengths of lines from He-like ions\n" );
		}
		else
		{
			fprintf( ioQQQ, "save helium has options: LINE WAVElength.\nSorry.\n" );
			cdEXIT(EXIT_FAILURE);
			/* no key */
		}
	}

	else if( p.nMatch("HUMM") )
	{
		strcpy( save.chSave[save.nsave], "HUMM" );
		sncatf( chHeader, 
			"#input to DHs routine.\n" );
	}

	else if( p.nMatch("HYDR") )
	{
		/* save hydrogen physical conditions */
		if( p.nMatch("COND") )
		{
			strcpy( save.chSave[save.nsave], "HYDc" );
			sncatf( chHeader, 
				"#depth\tTe\tHDEN\tEDEN\tHI/H\tHII/H\tH2/H\tH2+/H\tH3+/H\tH-/H\n" );
			/* save hydrogen ionization */
		}

		/* save information on 21 cm excitation processes - accept either keyword 21cm or 21 cm */
		else if( p.nMatch("21 CM") || p.nMatch("21CM") )
		{
			/* save information about 21 cm line */
			strcpy( save.chSave[save.nsave], "21CM" );
			sncatf( chHeader, 
				"#depth\tT(spin)\tT(kin)\tT(Lya/21cm)\tnLo\tnHi\tOccLya\ttau(21cm)"
				"\ttau(Lya)\topac(21 cm)\tn/Ts\ttau(21)\tTex(Lya)\tN(H0)/Tspin"
				"\tSum_F0\tSum_F1\tSum_T21\tn scat(Lya)\n" );
		}

		else if( p.nMatch("IONI") )
		{
			/* save hydrogen ionization */
			strcpy( save.chSave[save.nsave], "HYDi" );
			sncatf( chHeader, 
				"#hion\tzn\tgam1\tcoll ion1\tRecTot\tHRecCaB\thii/hi\tSim hii/hi"
				"\time_Hrecom_long(esc)\tdec2grd\texc pht\texc col\trec eff\tsec ion\n" );
		}
		else if( p.nMatch("POPU") )
		{
			/* save hydrogen populations */
			strcpy( save.chSave[save.nsave], "HYDp" );
			sncatf( chHeader, 
				"#depth\tn(H0)\tn(H+)\tn(1s)\tn(2s)\tn(2p)\tetc\n" );
		}
		else if( p.nMatch("LINE") )
		{
			/* save hydrogen lines
			 * hydrogen line intensities and optical depths  */
			strcpy( save.chSave[save.nsave], "HYDl" );
			if (p.nMatch("ALPH"))
				strcpy( save.chSave[save.nsave], "HYDa" );

			if( p.nMatch("ABSO") )
			{
				save.punarg[save.nsave][0] = 1;
			}
			else
			{
				save.punarg[save.nsave][0] = 0;
			}

			sncatf( chHeader, 
				"#nHi\tlHi\tnLo\tlLo\tWV(A)\ttau\tIntensity\n" );
		}
		else if( p.nMatch(" LYA") )
		{
			/* save hydrogen Lya some details about Lyman alpha  */
			strcpy( save.chSave[save.nsave], "HYDL" );
			sncatf( chHeader, 
				"#depth\tTauIn\tTauTot\tn(2p)/n(1s)\tTexc\tTe\tTex/T\tPesc\tPdes\tpump\topacity\talbedo\n" );
		}
		else
		{
			fprintf( ioQQQ, "Save hydrogen has options: CONDitions, 21 CM, LINE, POPUlations, and IONIzation.\nSorry.\n" );
			cdEXIT(EXIT_FAILURE);
		}


	}

	else if( p.nMatch("IONI") )
	{
		if( p.nMatch("RATE") )
		{
			/* save ionization rates, search for the name of an element */
			if( (nelem = p.GetElem() ) < 0 )
			{
				fprintf( ioQQQ, "There must be an element name on the ionization rates command.  Sorry.\n" );
				cdEXIT(EXIT_FAILURE);
			}
			save.punarg[save.nsave][0] = (realnum)nelem;
			strcpy( save.chSave[save.nsave], "IONR" );
			sncatf( chHeader, 
				"#%s depth\teden\tdynamics.Rate\tabund\tTotIonize\tTotRecom\tSource\t ... \n",
				elementnames.chElementSym[nelem]);
		}
		else
		{
			/* save table giving ionization means */
			strcpy( save.chSave[save.nsave], "IONI" );
			sncatf( chHeader, 
				"#Mean ionization distribution\n" );
		}
	}

	else if( p.nMatch(" IP ") )
	{
		strcpy( save.chSave[save.nsave], " IP " );
		sncatf( chHeader, 
			"#ionization potentials, valence shell\n" );
	}

	else if( p.nMatch("LEID") )
	{
		if( p.nMatch( "LINE" ) )
		{
			/* save Leiden lines
			 * final intensities of the Leiden PDR models */
			strcpy( save.chSave[save.nsave], "LEIL" );
			sncatf( chHeader, "#ion\twl\tInt\trel int\n");
		}
		else
		{
			/* save Leiden structure
			* structure of the Leiden PDR models */
			strcpy( save.chSave[save.nsave], "LEIS" );
			sncatf( chHeader, 
				/* 1-17 */
				"#Leid  depth\tA_V(extentd)\tA_V(point)\tTe\tH0\tH2\tCo\tC+\tOo\tCO\tO2\tCH\tOH\te\tHe+\tH+\tH3+\t"
				/* 18 - 30 */
				"N(H0)\tN(H2)\tN(Co)\tN(C+)\tN(Oo)\tN(CO)\tN(O2)\tN(CH)\tN(OH)\tN(e)\tN(He+)\tN(H+)\tN(H3+)\t"
				/* 31 - 32 */
				"H2(Sol)\tH2(FrmGrn)\tH2(photodiss)\t"
				/* 33 - 46*/
				"G0(DB96)\trate(CO)\trate(C)\theat\tcool\tGrnP\tGr-Gas-Cool\tGr-Gas-Heat\tCOds\tH2dH\tH2vH\tChaT\tCR H\tMgI\tSI\t"
				"Si\tFe\tNa\tAl\tC\tC610\tC370\tC157\tC63\tC146\n" );
		}
	}

	else if( (p.nMatch("LINE") && p.nMatch("LIST")) || p.nMatch("LINELIST") )
	{
		/* save line list "output file" "Line List file" */
		strcpy( save.chSave[save.nsave], "LLST" );

		/* we parsed off the second file name at start of this routine
		 * check if file was found, use it if it was, else abort */
		if( !lgSecondFilename )
		{
			fprintf(ioQQQ , "There must be a second file name between "
				"double quotes on the SAVE LINE LIST command.  This second"
				" file contains the input line list.  I did not find it.\nSorry.\n");
			cdEXIT(EXIT_FAILURE);
		}

		/* actually get the lines, and allocate the space in the arrays 
		 * cdGetLineList will look on path, only do one time in grid */
		if( save.params[save.nsave].ipPnunit == NULL )
		{
			/* make sure we free any allocated space from a previous call */
			save.SaveLineListFree(save.nsave);

			save.nLineList[save.nsave] = cdGetLineList(chSecondFilename, save.LineList[save.nsave]);

			if( save.nLineList[save.nsave] < 0 )
			{
				fprintf(ioQQQ,"DISASTER could not open SAVE LINE LIST file %s \n",
						  chSecondFilename.c_str() );
				cdEXIT(EXIT_FAILURE);
			}
		}

		// check whether intrinsic or emergent line emissivity
		save.lgEmergent[save.nsave] = false;
		if( p.nMatch("EMER") )
			save.lgEmergent[save.nsave] = true;

		// check whether cumulative or specific line emission
		save.lgCumulative[save.nsave] = false;
		if( p.nMatch("CUMU") )
			save.lgCumulative[save.nsave] = true;

		/* ratio option, in which pairs of lines form ratios, first over
		 * second */
		if( p.nMatch("RATI") )
		{
			save.lgLineListRatio[save.nsave] = true;
			if( save.nLineList[save.nsave]%2 )
			{
				/* odd number of lines - cannot take ratio */
				fprintf(ioQQQ , "There must be an even number of lines to"
					" take ratios of lines.  There were %li, an odd number."
					"\nSorry.\n", save.nLineList[save.nsave]);
				cdEXIT(EXIT_FAILURE);
			}
		}
		else
		{
			/* no ratio */
			save.lgLineListRatio[save.nsave] = false;
		}

		/* keyword absolute says to do absolute rather than relative intensities 
		 * relative intensities are the default */
		if( p.nMatch("ABSO") )
		{
			save.punarg[save.nsave][0] = 1;
		}
		else
		{
			save.punarg[save.nsave][0] = 0;
		}

		// check whether column or row (default)
		if( p.nMatch("COLUMN") )
		{
			save.punarg[save.nsave][1] = 1;
		}
		else
		{
			save.punarg[save.nsave][1] = 0;
		}

		/* give header line */
		sncatf( chHeader, "#lineslist" );
		// do header now if reporting rows of lines
		if( !save.punarg[save.nsave][1] )
		{
			for( long int j=0; j<save.nLineList[save.nsave]; ++j )
			{
				/* if taking ratio then put div sign between pairs */
				if( save.lgLineListRatio[save.nsave] && is_odd(j) )
					sncatf( chHeader, "/" );
				else
					sncatf( chHeader, "\t" );
				sncatf( chHeader,
						"%s ", save.LineList[save.nsave][j].chLabel().c_str() );
				sncatf( chHeader, "%s", save.LineList[save.nsave][j].twav().sprt_wl().c_str() );
			}
		}
		sncatf( chHeader, "\n" );
	}

	else if( p.nMatch("LINE") && !p.nMatch("XSPE") && !p.nMatch("NEAR") && !p.nMatch("SPECIES"))
	{
		/* save line options -
		 * this is not save xspec lines and not linear option
		 * check for keyword UNITS on line, then scan wavelength or energy units, 
		 * sets save.chConSavEnr*/
		save.chConSavEnr[save.nsave] = ChkUnits(p);

		/* save line emissivity, line intensity, line array,
		 * and line data */
		if( p.nMatch("STRU") )
		{
			fprintf(ioQQQ," The	SAVE LINES STRUCTURE command is now SAVE LINES "
				"EMISSIVITY.\n Sorry.\n\n");
			cdEXIT(EXIT_FAILURE);
		}

		else if( p.nMatch("PRES") )
		{
			/* save contributors to line pressure */
			strcpy( save.chSave[save.nsave], "PREL" );
			sncatf( chHeader, 
				"#P depth\tPtot\tPline/Ptot\tcontributors to line pressure\n" );
		}

		else if( p.nMatch("EMIS") )
		{
			/* this used to be the save lines structure command, is now
			 * the save lines emissivity command 
			 * give line emissivity vs depth */
			// check whether intrinsic or emergent line emissivity
			save.lgEmergent[save.nsave] = false;
			if( p.nMatch("EMER") )
				save.lgEmergent[save.nsave] = true;
			strcpy( save.chSave[save.nsave], "LINS" );
			/* read in the list of lines to examine */
			parse_save_line(p, false, chHeader, save.nsave);
		}

		else if( p.nMatch(" RT " ) )
		{
			/* save line RT */
			strcpy( save.chSave[save.nsave], "LINR" );
			/* save some details needed for line radiative transfer 
			 * routine in save_line.cpp */
			Parse_Save_Line_RT(p);
		}

		else if( p.nMatch("ZONE") && p.nMatch("CUMU") )
		{
			bool lgEOL;
			/* save lines zone cumulative
			 * this will be integrated line intensity, function of depth */
			strcpy( save.chSave[save.nsave], "LINC" );
			// option for intrinsic (default) or emergent
			save.lgEmergent[save.nsave] = false;
			if( p.nMatch("EMER") )
				save.lgEmergent[save.nsave] = true;
			/* option for either relative intensity or abs luminosity */
			lgEOL = p.nMatch("RELA");
			/* read in the list of lines to examine */
			parse_save_line(p, lgEOL, chHeader, save.nsave);
		}

		else if( p.nMatch("DATA") )
		{
			/* save line data, done in SaveLineData */

			/* the default will be to make wavelengths like in the printout, called labels,
			 * if units appears then other units will be used instead */
			save.chConSavEnr[save.nsave] = "labl";

			/* check for keyword UNITS on line, then scan wavelength or energy units if present,
			 * units are copied into save.chConSavEnr */
			if( p.nMatch("UNIT") )
				save.chConSavEnr[save.nsave] = ChkUnits(p);
			strcpy( save.chSave[save.nsave], "LIND" );
			sncatf( chHeader, 
				"#Emission line data.\n" );
		}

		else if( p.nMatch("ARRA") )
		{
			/* save line array -
			 * output energies and luminosities of predicted lines */
			strcpy( save.chSave[save.nsave], "LINA" );
			sncatf( chHeader, 
				"#enr\tID\tI(intrinsic)\tI(emergent)\ttype\n" );
		}

		else if( p.nMatch("LABE") )
		{
			/* this controls whether we will print lots of redundant 
			 * info labels for transferred lines - if keyword LONG appears
			 * then do so, if does not appear then do not - this is default */
			if( p.nMatch("LONG") )
				save.punarg[save.nsave][0] = 1;
			else
				save.punarg[save.nsave][0] = 0;

			/* if 'no index' is given, the index in the line stack
			 * is not reported
			 * this is useful when comparing the line stack before
			 * and after a significant change in the atomic data
			 * -- added for switching from Chianti v7 to v10 */
			if( p.nMatch( " NO " ) && p.nMatch( "INDE" ) )
				save.punarg[save.nsave][1] = 0;
			else
				save.punarg[save.nsave][1] = 1;

			/* save line labels */
			strcpy( save.chSave[save.nsave], "LINL" );
			if( save.punarg[save.nsave][1] > 0. )
			{
				sncatf( chHeader, 
					"#index\tlabel\twavelength\tcomment\n" );
			}
			else
			{
				sncatf( chHeader, 
					"#label\twavelength\tcomment\n" );
			}
		}

		else if( p.nMatch("OPTI") && !p.nMatch("SPECIES") )
		{
			if( p.nMatch("SOME") )
			{
				/* save lines optical depth some
				 * this will be inward optical depth */
				strcpy( save.chSave[save.nsave], "LINT" );
				// option for intrinsic (default) or emergent
				save.lgEmergent[save.nsave] = false;
				/* read in the list of lines to examine */
				parse_save_line(p, false, chHeader, save.nsave);
			}
			else
			{
				/* save line optical depths, done in SaveLineStuff */
				strcpy( save.chSave[save.nsave], "LINO" );
				
				/* the default will be to make wavelengths line in the printout, called labels,
				 * if units appears then other units will be used instead */
				save.chConSavEnr[save.nsave] = "labl";
				
				/* check for keyword UNITS on line, then scan wavelength or energy units if present,
				 * units are copied into save.chConSavEnr */
				if( p.nMatch("UNIT") )
					save.chConSavEnr[save.nsave] = ChkUnits(p);
				
				sncatf( chHeader, 
						  "#species\tenergy/%s\tmean opt depth\tspec opt depth\tdamp\n",
						  save.chConSavEnr[save.nsave] );
				
				/* this is optional limit to smallest optical depths */
				save.punarg[save.nsave][0] = (realnum)exp10(p.FFmtRead());
				/* this is default of 0.1 napier */
				if( p.lgEOL() )
				{
					save.punarg[save.nsave][0] = 0.1f;
				}
			}
		}

		else if( p.nMatch("POPU") )
		{
			/* save line populations "output" "LineList" */
			strcpy( save.chSave[save.nsave], "LINP" );

			/* we parsed off the second file name at start of this routine
			 * check if file was found, use it if it was, else abort */
			if( !lgSecondFilename )
			{
				fprintf(ioQQQ ,
					"There must be a second file name between double"
					" quotes on the SAVE LINE POPULATIONS command\n."
					"This second file contains the input list of"
					" spectral lines.\n"
					"I did not find it.\nSorry.\n");
				cdEXIT(EXIT_FAILURE);
			}

			/* actually get the lines, and allocate the space in the arrays 
			 * cdGetLineList will look on path, only do one time in grid */
			if( save.params[save.nsave].ipPnunit == NULL )
			{
				/* make sure we free any allocated space from a previous call */
				save.SaveLineListFree(save.nsave);
	
				save.nLineList[save.nsave] = cdGetLineList(chSecondFilename, save.LineList[save.nsave]);
	
				if( save.nLineList[save.nsave] < 0 )
				{
					fprintf(ioQQQ,
						"DISASTER could not open"
						" SAVE LINE POPULATIONS file %s \n",
						chSecondFilename.c_str() );
					cdEXIT(EXIT_FAILURE);
				}
			}

			// by default give numbers in columns
			// ROW keyword says to write the numbers across as one long row
			// subsequent keyword tells which column to write
			if( p.nMatch(" ROW") )
			{
				save.punarg[save.nsave][0] = 1;
	
				if( p.nMatch( " LOW" ) )
					/* lower level population */
					save.punarg[save.nsave][1] = 0;
				else if( p.nMatch( " UPP" ) )
					/* upper level population */
					save.punarg[save.nsave][1] = 1;
				else if( p.nMatch( " TSP" ) )
					/* spin temperature */
					save.punarg[save.nsave][1] = 3;
				else
					/* DEFAULT: population ratio: (nu/gu)/(nl/gl) */
					save.punarg[save.nsave][1] = 2;
			}
			else
				// the default, one line per row, multiple columns
				save.punarg[save.nsave][0] = 0;

			sncatf( chHeader, "#depth\t" );
			if( save.punarg[save.nsave][0] )
			{
				for( long int j=0; j<save.nLineList[save.nsave]; ++j )
				{
					sncatf( chHeader, "%s ",
							save.LineList[save.nsave][j].chLabel().c_str() );
					sncatf( chHeader, "%s", save.LineList[save.nsave][j].twav().sprt_wl().c_str() );
					if( j != save.nLineList[save.nsave] )
					{
						sncatf( chHeader, "\t" );
					}
				}
				sncatf( chHeader, "\n" );
			}
			else
			{	
				sncatf( chHeader,
					"emline\tnl\tnu\t(nu/gu)/(nl/gl)\tTspin\n" );
			}
		}

		else if( p.nMatch("INTE") )
		{
			/* this will be full set of line intensities */
			strcpy( save.chSave[save.nsave], "LINI" );
			sncatf( chHeader, 
				"#Emission line intrinsic intensities per unit inner area\n" );
			if( p.nMatch("COLU") )
				/* column is key to save single column */
				strcpy( save.chPunRltType, "column" );
			else
				/* array is key to save large array */
				strcpy( save.chPunRltType, "array " );

			save.punarg[save.nsave][0] = 0.;
			// ALL option - all lines, even zero intensities
			if( p.nMatch( " ALL" ) )
				save.punarg[save.nsave][0] = -1.;

			// check whether intrinsic or emergent line emissivity
			save.lgEmergent[save.nsave] = false;
			if( p.nMatch("EMER") )
				save.lgEmergent[save.nsave] = true;

			if( p.nMatch("EVER") )
			{
				save.LinEvery = (long int)p.FFmtRead();
				save.lgLinEvery = true;
				if( p.lgEOL() )
				{
					fprintf( ioQQQ, 
						"There must be a second number, the number of zones to print.\nSorry.\n" );
					cdEXIT(EXIT_FAILURE);
				}
			}
			else
			{
				save.LinEvery = iterations.nend[0];
				save.lgLinEvery = false;
			}
		}
		else
		{
			fprintf( ioQQQ, 
				"This option for SAVE LINE is something that I do not understand.  Sorry.\n" );
			cdEXIT(EXIT_FAILURE);
		}
	}

	else if( p.nMatch(" MAP") )
	{
		strcpy( save.chSave[save.nsave], "MAP " );
		sncatf( chHeader, 
			"#te, heating, cooling.\n" );
		/* do cooling space map for specified zones
		 * if no number, or <0, do map and save out without doing first zone
		 * does map by calling punt(" map") 
		 */
		hcmap.MapZone = (long)p.FFmtRead();
		if( p.lgEOL() )
		{
			hcmap.MapZone = 1;
		}

		if( p.nMatch("RANG") )
		{
			bool lgLogOn;
			hcmap.RangeMap[0] = (realnum)p.FFmtRead();
			if( hcmap.RangeMap[0] <= 10. && !p.nMatch("LINE") )
			{
				hcmap.RangeMap[0] = exp10(hcmap.RangeMap[0]);
				lgLogOn = true;
			}
			else
			{
				lgLogOn = false;
			}

			hcmap.RangeMap[1] = (realnum)p.FFmtRead();
			if( lgLogOn )
				hcmap.RangeMap[1] = exp10(hcmap.RangeMap[1]);

			if( p.lgEOL() )
			{
				fprintf( ioQQQ, "There must be a zone number, followed by two temperatures, on this line.  Sorry.\n" );
				cdEXIT(EXIT_FAILURE);
			}
			if( hcmap.RangeMap[1] <= hcmap.RangeMap[0] )
			{
				fprintf( ioQQQ, " The upper temperature limit must be larger than the lower: "
					 "found lower=%g upper=%g.\n", hcmap.RangeMap[0], hcmap.RangeMap[1] );
				cdEXIT(EXIT_FAILURE);
			}
		}
	}

	else if( p.nMatch("MOLE") )
	{
		/* save molecules, especially for PDR calculations */
		strcpy( save.chSave[save.nsave], "MOLE" );
	}

	else if( p.nMatch("MONI") )
	{
		/* save monitors */
		strcpy( save.chSave[save.nsave], "MONI" );
	}

	else if( p.nMatch("OPTICAL") && p.nMatch("DEPTH") && !p.nMatch("SPECIES") )
	{
		/* check for keyword UNITS on line, then scan wavelength or energy units if present,
		 * units are copied into save.chConSavEnr */
		save.chConSavEnr[save.nsave] = ChkUnits(p);

		/* "every" option to save this on every zone -
		 * not present then only last zone is saved */
		if( p.nMatch("EVER" ) )
		{
			/* save every zone */
			save.lgSaveEveryZone[save.nsave] = true;
			save.nSaveEveryZone[save.nsave] = 1;
		}
		else
		{
			/* only save last zone */
			save.lgSaveEveryZone[save.nsave] = false;
			save.nSaveEveryZone[save.nsave] = 1;
		}
		
		if( p.nMatch("FINE") )
		{
			/* save fine continuum optical depths */
			rfield.lgSaveOpacityFine = true;
			strcpy( save.chSave[save.nsave], "OPTf" );
			sncatf( chHeader, "#energy/%s\tTau tot\topacity\n",
				save.chConSavEnr[save.nsave] );
			/* range option - important since so much data */
			if( p.nMatch("RANGE") ) 
			{
				/* get lower and upper range, eventually must be in Ryd */
				double Energy1 = p.FFmtRead();
				double Energy2 = p.FFmtRead();
				if( p.lgEOL() )
				{
					fprintf(ioQQQ,"There must be two numbers, the lower and upper energy range in Ryd.\nSorry.\n");
					cdEXIT(EXIT_FAILURE);
				}
				if( p.nMatch("UNIT" ) )
				{
					// apply units to range option
					const char *energyUnits = p.StandardEnergyUnit();
					Energy unitChange;
					unitChange.set(Energy1, energyUnits );
					Energy1 = unitChange.Ryd();
					unitChange.set(Energy2, energyUnits );
					Energy2 = unitChange.Ryd();
				}
				/* get lower and upper rang in Ryd */
				save.punarg[save.nsave][0] = (realnum)MIN2( Energy1 , Energy2 );
				save.punarg[save.nsave][1] = (realnum)MAX2( Energy1 , Energy2 );
				//fprintf(ioQQQ , "DEBUG units change fine %.3e %.3e\n" , save.punarg[save.nsave][0] ,
				//		save.punarg[save.nsave][1] );
				//cdEXIT(EXIT_FAILURE);
			}
			else
			{
				/* these mean full energy range */
				save.punarg[save.nsave][0] = 0.;
				save.punarg[save.nsave][1] = 0.;
			}
			/* optional last parameter - how many points to bring together */
			save.punarg[save.nsave][2] = (realnum)p.FFmtRead();

			if( !p.lgEOL()  && save.punarg[save.nsave][2] < 1 )
			{
				fprintf(ioQQQ,"The number of fine continuum points to skip must be > 0 \nSorry.\n");
				cdEXIT(EXIT_FAILURE);
			}

			/* default is to bring together ten */
			if( p.lgEOL() )
				save.punarg[save.nsave][2] = 10;

			/* ALL means to report all cells, even those with zero, to maintain uniform output */
			if( p.nMatch(" ALL" ) )
				save.punarg[save.nsave][2] *= -1;
		}
		else
		{
			/* save coarse continuum optical depths */
			strcpy( save.chSave[save.nsave], "OPTc" );
			sncatf( chHeader, 
				"#energy/%s\ttotal\tabsorp\tscat\n",
				save.chConSavEnr[save.nsave] );
		}

	}
	else if( p.nMatch(" OTS") )
	{
		strcpy( save.chSave[save.nsave], " OTS" );
		sncatf( chHeader, 
			"#otscon, lin, conOpac LinOpc\n" );
	}

	else if( p.nMatch(" OVER") )
	{
		/* save overview of model results */
		strcpy( save.chSave[save.nsave], "OVER" );
		/* >>chng 20 07 25 add Opt Depth to illuminated face @ 1 Ryd as per Vianney LeBouteiller post */
		sncatf( chHeader, 
			"#depth\tTe\tHtot\thden\teden\t2H_2/H\tHI\tHII\tHeI\tHeII\tHeIII\tCO/C\tC1\tC2\tC3\tC4\tO1\tO2\tO3\tO4\tO5\tO6\tH2O/O\tAV(point)\tAV(extend)\tTau912\n" );
	}

	else if( p.nMatch(" PDR") )
	{
		strcpy( save.chSave[save.nsave], " PDR" );
		sncatf( chHeader, 
			"#depth\tH colden\tTe\tHI/HDEN\tH2/HDEN\tH2*/HDEN\tCI/C\tCO/C\tH2O/O\tG0\tAV(point)\tAV(extend)\tTauV(point)\n" );
	}

	else if( p.nMatch("PERF") )
	{
		/* output performance characteristics per zone */
		strcpy( save.chSave[save.nsave], "PERF" );
	}
	else if( p.nMatch("PHYS") )
	{
		/* save physical conditions */
		strcpy( save.chSave[save.nsave], "PHYS" );
		sncatf( chHeader, 
			"#PhyC depth\tTe\tn(H)\tn(e)\tHtot\taccel\tfillfac\n" );
	}

	else if( p.nMatch("POIN") )
	{
		// this is a special save option handled below
		(void)0;
	}

	else if( p.nMatch("PRES") )
	{
		/* the save pressure command */
		strcpy( save.chSave[save.nsave], "PRES" );
		sncatf( chHeader, 
			"#P depth\tPerror%%\tPcurrent\tPIn+Pinteg\tPgas(r0)\tPgas\tPram"
			"\tPrad(line)\tPinteg\tV(wind km/s)\tcad(wind km/s)\tP(mag)\tV(turb km/s)"
			"\tP(turb)\tPgr_Int\tint thin elec\tconv?\n" );
	}

	else if( p.nMatch("RADI") )
	{
		/* the save radius command */
		sncatf( chHeader, "#NZONE\tradius\tdepth\tdr\n" );
		/* option to only save the outer radius */
		if( p.nMatch( "OUTE" ) )
		{
			/* only outer radius */
			strcpy( save.chSave[save.nsave], "RADO" );
		}
		else
		{
			/* all radii */
			strcpy( save.chSave[save.nsave], "RADI" );
		}
	}

	else if( p.nMatch("RECO") )
	{
		if( p.nMatch("COEF") )
		{
			// this is a special save option handled below
			(void)0;
		}

		else if( p.nMatch("EFFI") )
		{
			/* save recombination efficiency, rate and escaping fraction for 1s and 2p of H */
			strcpy( save.chSave[save.nsave], "RECE" );
			sncatf( chHeader, 
				"#Recom rate, esc frac, H 1S 2P\n" );
		}

		else
		{
			fprintf( ioQQQ, "No option recognized on this save recombination command\n" );
			fprintf( ioQQQ, "Valid options are COEFFICIENTS, AGN, and EFFICIENCY\nSorry.\n" );
			cdEXIT(EXIT_FAILURE);
		}
	}

	/* save results command, either as single column or wide array */
	else if( p.nMatch("RESU") )
	{
		strcpy( save.chSave[save.nsave], "RESU" );
		if( p.nMatch("COLU") )
		{
			/* column is key to save single column */
			strcpy( save.chPunRltType, "column" );
		}
		else
		{
			/* array is key to save large array */
			strcpy( save.chPunRltType, "array " );
		}

		/* do not change following, is used as flag in getlines */
		if( prt.lgPrintTime ) sncatf( chHeader,
			"#results of calculation\n" );
	}

	else if( p.nMatch("SECO") )
	{
		/* save secondary ionization rate */
		strcpy( save.chSave[save.nsave], "SECO" );
		sncatf( chHeader, 
			"#depth\tIon(H^0)\tDiss(H_2)\tExcit(Lya)\n" );
	}

	else if( p.nMatch("SOURCE") && !p.nMatch("SPECIES") )
	{

		/* check for keyword UNITS on line, then scan wavelength or energy units if present,
		* units are copied into save.chConSavEnr */
		save.chConSavEnr[save.nsave] = ChkUnits(p);

		if( p.nMatch("DEPT") )
		{
			/* print continuum source function as function of depth */
			strcpy( save.chSave[save.nsave], "SOUD" );
			sncatf( chHeader, 
				"#continuum source function vs depth\n" );
		}
		else if( p.nMatch("SPECTRUM") )
		{
			/* print spectrum continuum source function at 1 depth */
			strcpy( save.chSave[save.nsave], "SOUS" );
			sncatf( chHeader, 
				"#continuum source function nu/%s\tConEmitLocal/widflx"
				"\tabs opac\tConSourceFcnLocal\tConSourceFcnLocal/plankf\tConSourceFcnLocal/flux\n",
				save.chConSavEnr[save.nsave] );
		}
		else
		{
			fprintf( ioQQQ, "A second keyword must appear on this line.\n" );
			fprintf( ioQQQ, "They are DEPTH and SPECTRUM.\n" );
			fprintf( ioQQQ, "Sorry.\n" );
			cdEXIT(EXIT_FAILURE);
		}
	}

	else if( p.nMatch("SPECIAL") )
	{
		/* save special, will call routine SaveSpecial */
		strcpy( save.chSave[save.nsave], "SPEC" );
		sncatf( chHeader, "#Special.\n" );
	}

	else if( p.nMatch("SPECIES") )
	{
		strcpy( save.chSave[save.nsave], "SPCS" );

		// option to save information about a particular species,
		// the "second filename" may really be the species label.  Rename here for clarity
		chLabel = chSecondFilename;
		save.chSaveSpecies[save.nsave].clear();
		bool readlist = true;
		if( lgSecondFilename )
		{
			save.chSaveSpecies[save.nsave].push_back(chLabel);
			readlist = false;
		}

		if (p.nMatch( "ALL" ) )
		{
			if ( readlist == false )
			{
				fprintf( ioQQQ, "ParseSave SAVE SPECIES command must have just one of a) a single species matcher, b) a list, or c) the keyword ALL.  Sorry.\n" );
				cdEXIT(EXIT_FAILURE);
			}
			readlist = false;
		}

		if( p.nMatch( "BAND" ) )
		{
			strcpy( save.chSaveArgs[save.nsave], "BAND" );

			if( ! lgSecondFilename )
			{
				fprintf( ioQQQ, "Error: File containing bands not specified\n" );
				cdEXIT( EXIT_FAILURE );
			}

			string speciesLabel;
			if( p.GetQuote( speciesLabel ) )
			{
				fprintf( ioQQQ, "Error: Species not specified\n" );
				cdEXIT( EXIT_FAILURE );
			}

			save.chSaveSpecies[save.nsave].back() = speciesLabel;
			save.SpeciesBandFile[save.nsave] = chSecondFilename;
			//	printf("fname = '%s'\t spec = '%s'\n",
			//		chSecondFilename.c_str(), speciesLabel.c_str());

			addUniqueSpeciesBand( chSecondFilename, speciesLabel );

			// check whether intrinsic or emergent line emissivity
			save.lgEmergent[save.nsave] = false;
			if( p.nMatch("EMER") )
				save.lgEmergent[save.nsave] = true;
		}

		/* second keyword appears as both 'density' and 'densities'
		 * in the test suite */
		else if (p.nMatch( "COLUMN" ) && p.nMatch( "DENSIT" ) )
		{
			/* column densities*/
			strcpy( save.chSaveArgs[save.nsave], "COLU" );
		}

		else if( p.nMatch( "CONT" ) )
		{
			// Add species to vector only when not present
			string speciesLabel = string( chSecondFilename );
			if( speciesLabel == "" )
			{
				fprintf( ioQQQ, "Error: Species not specified\n" );
				cdEXIT( EXIT_FAILURE );
			}

			if( find( save.contSaveSpeciesLabel.begin(),
				save.contSaveSpeciesLabel.end(),
				speciesLabel ) == save.contSaveSpeciesLabel.end() )
			{
				save.contSaveSpeciesLabel.push_back( speciesLabel );
			}
			save.chSaveSpecies[save.nsave].push_back( speciesLabel );

			// save species continuum, options are total (default), inward,
			// and outward
			if( p.nMatch("INWA") )
			{
				// inward continuum
				strcpy( save.chSaveArgs[save.nsave], "CONi" );
			}
			else if( p.nMatch(" OUT") )
			{
				// outward continuum
				strcpy( save.chSaveArgs[save.nsave], "CONo" );
			}
			else
			{
				// total continuum
				strcpy( save.chSaveArgs[save.nsave], "CONt" );
			}

			// default units of spectral energy are Ryd, this can change to
			// many other units
			save.chConSavEnr[save.nsave] = ChkUnits(p);

			// by default give numbers in two columns, row keyword says to
			// write the numbers across as one long row 
			if( p.nMatch(" ROW") )
				save.punarg[save.nsave][0] = 1;
			else
				// the default, two columns
				save.punarg[save.nsave][0] = 2;

			{
				enum {DEBUG_IN=false};
				if( DEBUG_IN )
				{
					fprintf( ioQQQ, "\t species    :\t %s\n", speciesLabel.c_str() );
					fprintf( ioQQQ, "\t chSave     :\t %s\n", save.chSave[save.nsave] );
					fprintf( ioQQQ, "\t chSaveArgs :\t %s\n", save.chSaveArgs[save.nsave] );
					fprintf( ioQQQ, "\t chConSavEnr:\t %s\n", save.chConSavEnr[save.nsave] );
					fprintf( ioQQQ, "\t punarg     :\t %g\n", save.punarg[save.nsave][0] );
				}
			}
			readlist = false;
		}
		else if (p.nMatch( "DEPAR" ) )
		{
			/* save species departure coefficients for all levels */
			strcpy( save.chSaveArgs[save.nsave], "DEPA" );
		}
		else if (p.nMatch( "ENERG" ) )
		{
			/* energy levels, default Rydbergs but option to change units */
			save.chConSavEnr[save.nsave] = ChkUnits(p);
			strcpy( save.chSaveArgs[save.nsave], "ENER" );
		}
		else if( p.nMatch("LABELS") )
		{
			strcpy( save.chSaveArgs[save.nsave], "LABE" );
		}
		else if (p.nMatch( "LEVELS" ) )
		{
			/* the number of levels in this zone */
			strcpy( save.chSaveArgs[save.nsave], "LEVL" );
		}

		// save species line - data for lines
		else if( p.nMatch("LINE" ) )
		{
			strcpy( save.chSaveArgs[save.nsave], "DATA" );
			if( p.nMatch( " WN " ) )
			{
				// save wavenumbers rather than wavelength
				save.lgSaveDataWn = true;
			}

			if( p.nMatch( "GF " ) )
			{
				// save gf rather than Aup
				save.lgSaveDataGf = true;
			}

			if( p.nMatch( "RATE" ) )
			{
				// save deexcitation rate rather than collision strength
				save.lgSaveDataRates = true;
			}
			// we will report all lines
			readlist = false;
		}

		else if (p.nMatch( "POPUL" ) )
		{
			/* save species populations */
			fprintf(ioQQQ,"Warning, 'save species populations' has changed to 'save species densities'.\n");
			fprintf(ioQQQ,"'save species populations' is deprecated, please update your input.\n");
			strcpy( save.chSaveArgs[save.nsave], "DENS" );
		}
		else if (p.nMatch( "OPTICAL" ) )
		{
			/* save species optical depths for all transitions */
			strcpy( save.chSaveArgs[save.nsave], "OPTD" );
		}
		else if (p.nMatch( "DENS" ) )
		{
			/* save species densities */
			strcpy( save.chSaveArgs[save.nsave], "DENS" );
		}
		else if (p.nMatch( "OPTICAL" ) && p.nMatch("DEPTH") )
		{
			/* save species densities */
			strcpy( save.chSaveArgs[save.nsave], "OPTD" );
		}
		else
		{
			fprintf( ioQQQ, "ParseSave cannot find a recognized keyword on this SAVE SPECIES command line.\n" );
			fprintf( ioQQQ, "I know about the keywords COLUMN DENSITIES, DENSITIES, DEPARTURE, CONTINUUM, BANDS, "
					"OPTICAL DEPTH, ENERGIES, LABELS, LEVELS, and DATA SOURCES.\nSorry.\n" );
			cdEXIT(EXIT_FAILURE);
		}
		if (readlist)
		{
			p.readList(save.chSaveSpecies[save.nsave],"species");
		}
	}

	else if( p.nMatch("TEMP") )
	{
		/* save temperature command */
		strcpy( save.chSave[save.nsave], "TEMP" );
		sncatf( chHeader, 
			"#depth\tTe\tcC/dT\tdt/dr\td^2T/dr^2\n" );
	}

	else if( p.nMatch("TIME") && p.nMatch("DEPE") )
	{
		/* information about time dependent solutions */
		strcpy( save.chSave[save.nsave], "TIMD" );
		/* do not want to separate iterations with special character */
		save.lg_separate_iterations[save.nsave] = false;
		/* write header */
		sncatf( chHeader,
			"#elapsed time\ttime step \tscale cont\tscalingDen\t<T>\t<H+/H rad>\t<H0/H rad>\t<2*H2/H rad>\t<He+/He rad>\t<CO/H>\t<redshift>\t<ne/nH>\n" );
	}

	else if( p.nMatch("TPRE") )
	{
		/* debug output from the temperature predictor in zonestart, 
		 * set with save tpred command */
		strcpy( save.chSave[save.nsave], "TPRE" );
		sncatf( chHeader, 
			"#zone  old temp,  guess Tnew, new temp    delta \n" );
	}

	else if( p.nMatch("WIND") )
	{
		strcpy( save.chSave[save.nsave], "WIND" );
		sncatf( chHeader, 
			"#radius\tdepth\tvel [cm/s]\tTot accel [cm s-2]\tLin accel [cm s-2]"
			"\tCon accel [cm s-2]\tforce multiplier\ta_gravity\n" );
		if( p.nMatch( "TERM" ) )
		{
			/* only save for last zone, the terminal velocity, for grids */
			save.punarg[save.nsave][0] = 0.;
		}
		else
		{
			/* one means save every zone */
			save.punarg[save.nsave][0] = 1.;
		}
	}

	else if( p.nMatch("XSPE") )
	{
		/* say that this is a FITS file output */
		save.lgFITS[save.nsave] = true;
		save.lgXSPEC[save.nsave] = true;

		/* trying to produce separate files is not supported */
		save.lgSaveToSeparateFiles[save.nsave] = false;

		/* the save xspec commands */
		save.lgPunLstIter[save.nsave] = true;

		/* remember that a save xspec command was entered */
		grid.lgSaveXspec = true;

		double gridLo = rfield.anumin(0);
		double gridHi = rfield.anumax(rfield.nflux-1);

		/* range option - important since so much data */
		if( p.nMatch("RANGE") ) 
		{
			/* get lower and upper range, must be in keV */
			double Elo_keV = p.FFmtRead();
			double Ehi_keV = p.FFmtRead();
			if( p.lgEOL() )
				p.NoNumb("lower and upper energy range");
			if( Elo_keV >= Ehi_keV )
			{
				fprintf(ioQQQ,"The two energies for the range must be in increasing order.\nSorry.\n");
				cdEXIT(EXIT_FAILURE);
			}
			// convert to Ryd
			save.punarg[save.nsave][0] = realnum(Elo_keV*1000./EVRYD);
			save.punarg[save.nsave][1] = realnum(Ehi_keV*1000./EVRYD);
			if( save.punarg[save.nsave][0] <= gridLo || save.punarg[save.nsave][0] >= gridHi ||
			    save.punarg[save.nsave][1] <= gridLo || save.punarg[save.nsave][1] >= gridHi )
			{
				fprintf(ioQQQ, "Energy range is out of bounds.\nSorry.\n");
				cdEXIT(EXIT_FAILURE);
			}
		}
		else
		{
			/* these mean full energy range */
			save.punarg[save.nsave][0] = 0.;
			save.punarg[save.nsave][1] = 0.;
		}

		if( p.nMatch("ATAB") )
		{
			/* save xspec atable command */
			
			if( p.nMatch("TOTA") )
			{
				saveXSPEC(0); /* total spectrum */
			}
			else if( p.nMatch("INCI") )
			{
				if( p.nMatch("ATTE") )
					saveXSPEC(2); /* attenuated incident continuum */
				else if( p.nMatch("REFL") )
					saveXSPEC(3); /* reflected incident continuum */
				else
					saveXSPEC(1); /* incident continuum */
			}
			else if( p.nMatch("DIFF") )
			{
				if( p.nMatch("REFL") )
					saveXSPEC(5); /* reflected diffuse continuous emission */
				else
					saveXSPEC(4); /* diffuse continuous emission outward */
			}
			else if( p.nMatch("LINE") )
			{
				if( p.nMatch("REFL") )
					saveXSPEC(7); /* reflected lines */
				else
					saveXSPEC(6); /* outward lines */
			}
			else if( p.nMatch("SPEC") )
			{
				if( p.nMatch("REFL") )
					saveXSPEC(9); /* reflected spectrum */
				else
					saveXSPEC(8); /* transmitted spectrum */
			}
			else
			{
				saveXSPEC(8); /* transmitted spectrum */
			}

			if( p.nMatch("NORM") )
			{
				// option to normalize the spectrum at a given frequency
				// the number entered is in keV, the default is 1 keV
				double Enorm_keV = p.FFmtRead();
				if( p.lgEOL() )
					Enorm_keV = 1.;
				save.punarg[save.nsave][2] = realnum(Enorm_keV*1000./EVRYD);
				if( save.punarg[save.nsave][2] <= gridLo || save.punarg[save.nsave][2] >= gridHi )
				{
					fprintf(ioQQQ, "Normalization energy is out of bounds.\nSorry.\n");
					cdEXIT(EXIT_FAILURE);
				}
			}
			else
			{
				save.punarg[save.nsave][2] = 0.;
			}
		}
		else if( p.nMatch("MTAB") )
		{
			saveXSPEC(10); /* save xspec mtable */
		}
		else
		{
			fprintf( ioQQQ, "Support only for xspec atable and xspec mtable.\n" );
			cdEXIT( EXIT_FAILURE );
		}
	}

	/* save column density has to come last so do not trigger specific column 
	 * densities, H2, FeII, etc.
	 * Need both keywords since column is also the keyword for one line per line */
	else if( p.nMatch("COLU") && p.nMatch("DENS") )
	{
		fprintf(ioQQQ,"The SAVE COLUMN DENSITIES command is now an option to SAVE SPECIES, please use that command.\nSorry.\n");
		cdEXIT(EXIT_FAILURE);
	}
	else
	{
		fprintf( ioQQQ, 
			"ParseSave cannot find a recognized keyword on this SAVE command line.\nSorry.\n" );
		cdEXIT(EXIT_FAILURE);
	}

	/* only open if file has not already been opened during a previous call */
	if( save.params[save.nsave].ipPnunit == NULL )
	{
		string mode = "w";
		if( save.lgFITS[save.nsave] )
			mode += "b";

		/* open the file with the name and mode generated above */
		save.params[save.nsave].ipPnunit = open_data( chFilename, mode );
	
		/* option to set no buffering for this file.  The setbuf command may 
		 * ONLY be issued right after the open of the file.  Giving it after 
		 * i/o has been done may result in loss of the contents of the buffer, PvH */
		if( p.nMatch("NO BUFFER") )
			setbuf( save.params[save.nsave].ipPnunit , NULL );
	}

	/***************************************************************/
	/*                                                             */
	/*  The following are special save options and must be done   */
	/*  after the parsing and file opening above.                  */
	/*                                                             */
	/*  NB:  these are ALSO parsed above.  Here we DO something.   */
	/*                                                             */
	/***************************************************************/

	if( p.nMatch("CONV") && p.nMatch("REAS") )
	{
		/* save reason model declared not converged
		 * not a true save command, since done elsewhere */
		save.ipPunConv = save.params[save.nsave].ipPnunit;
		save.lgPunConv_noclobber = save.lgNoClobber[save.nsave];
		save.lgPunConv = true;
		sncatf( chHeader,
			"# reason for continued iterations\n" );
		strcpy( save.chSave[save.nsave], "" );
		save.lgRealSave[save.nsave] = false;
	}

	else if( p.nMatch("CONV") && p.nMatch("BASE") )
	{
		/* save some quantities we are converging */
		save.lgTraceConvergeBase = true;
		/* the second save occurrence - file has been opened,
		* copy handle, also pass on special no hash option */
		save.lgTraceConvergeBaseHash = save.lgHashEndIter[save.nsave];
		save.ipTraceConvergeBase = save.params[save.nsave].ipPnunit;
		/* set save last flag to whatever it was above */
		save.lgTraceConvergeBase_noclobber = save.lgNoClobber[save.nsave];
		sncatf( chHeader,
			"#zone\theat\tcool\teden\n" );
		strcpy( save.chSave[save.nsave], "" );
		save.lgRealSave[save.nsave] = false;
	}

	else if( p.nMatch(" DR ") )
	{
		/* the second save dr occurrence - file has been opened,
		 * copy handle to ipDRout, also pass on special no hash option */
		save.lgDROn = true;
		save.lgDRHash = save.lgHashEndIter[save.nsave];
		save.ipDRout = save.params[save.nsave].ipPnunit;
		/* set save last flag to whatever it was above */
		save.lgDRPLst = save.lgPunLstIter[save.nsave];
		save.lgDROn_noclobber = save.lgNoClobber[save.nsave];
		sncatf( chHeader,
			"#zone\tdepth\tdr\tdr 2 go\treason\n" );
		strcpy( save.chSave[save.nsave], "" );
		save.lgRealSave[save.nsave] = false;
	}

	else if( p.nMatch(" DT ") )
	{
		/* the second save dt occurrence - file has been opened,
		 * copy handle to ipDTout, also pass on special no hash option */
		save.lgDTOn = true;
		save.lgDTHash = save.lgHashEndIter[save.nsave];
		save.ipDTout = save.params[save.nsave].ipPnunit;
		/* set save last flag to whatever it was above */
		save.lgDTPLst = save.lgPunLstIter[save.nsave];
		save.lgDTOn_noclobber = save.lgNoClobber[save.nsave];
		sncatf( chHeader,
			"#time\tTe\ttimestep\tfactor\treason\n" );
		strcpy( save.chSave[save.nsave], "" );
		save.lgRealSave[save.nsave] = false;
	}

	else if( p.nMatch("SPECIES") && p.nMatch("DATA") && p.nMatch("SOURCE") )
	{
		/* save database sources
		 * print a table of data sources (chianti,stout, etc) for each species*/
		save.lgSDSOn = true;
		save.ipSDSFile = save.params[save.nsave].ipPnunit;
		strcpy( save.chSave[save.nsave], "" );
		save.lgRealSave[save.nsave] = false;
	}

	else if( p.nMatch("QHEA") )
	{
		gv.QHSaveFile = save.params[save.nsave].ipPnunit;
		gv.lgQHPunLast = save.lgPunLstIter[save.nsave];
		save.lgQHSaveFile_noclobber = save.lgNoClobber[save.nsave];
		sncatf( chHeader,
			"#Probability distributions from quantum heating routine\n" );
		save.lgRealSave[save.nsave] = false;
	}

	else if( p.nMatch("POIN") )
	{
		/* save out the pointers */
		save.ipPoint = save.params[save.nsave].ipPnunit;
		save.lgPunPoint_noclobber = save.lgNoClobber[save.nsave];
		save.lgPunPoint = true;
		sncatf( chHeader,
			"#pointers\n" );
		strcpy( save.chSave[save.nsave], "" );
		save.lgRealSave[save.nsave] = false;
	}

	else if( p.nMatch("RECO") && p.nMatch("COEF") )
	{
		/* recombination coefficients for everything
		 * save.lgioRecom set to false in routine zero, non-zero value
		 * is flag to save recombination coefficients. the output is actually
		 * produced by a series of routines, as they generate the recombination
		 * coefficients.  these include 
		 * diel supres, helium, hydrorecom, iibod, and makerecomb*/
		save.ioRecom = save.params[save.nsave].ipPnunit;
		save.lgioRecom_noclobber = save.lgNoClobber[save.nsave];
		/* this is logical flag used in routine ion_recom to create the save output */
		save.lgioRecom = true;
		sncatf( chHeader,
			"#recombination coefficients cm3 s-1 for current density and temperature\n" );
		strcpy( save.chSave[save.nsave], "" );
		save.lgRealSave[save.nsave] = false;
	}

	else if( p.nMatch("GRID") )
	{
		/* this enables saving GRID output outside the main SaveDo() loop */
		grid.pnunit = save.params[save.nsave].ipPnunit;
		save.lgSaveGrid_noclobber = save.lgNoClobber[save.nsave];
	}

	else if( p.nMatch(" MAP") )
	{
		/* say output goes to special save */
		ioMAP = save.params[save.nsave].ipPnunit;
	}

	/* if not done already and chTitle has been set to a string then print title
	 * logic to prevent more than one title in grid calculation */
	if( save.lgSaveTitle(save.nsave) && chTitle.length() > 0 )
	{
		fprintf( save.params[save.nsave].ipPnunit, "%s", chTitle.c_str() );
		save.SaveTitleDone(save.nsave);
	}

	/* same logic for the regular save file header */
	if( save.lgSaveHeader(save.nsave) && chHeader.str().length() > 0 )
	{
		fprintf( save.params[save.nsave].ipPnunit, "%s", chHeader.str().c_str() );
		save.SaveHeaderDone(save.nsave);
	}

	/* increment total number of save commands, */
	++save.nsave;
	return;
}

/*SaveFilesInit initialize save file pointers, called from InitCoreload
 * called one time per core load 
 * NB KEEP THIS ROUTINE SYNCHED UP WITH THE NEXT ONE, CloseSaveFiles */
void SaveFilesInit()
{
	long int i;
	static bool lgFIRST = true;

	DEBUG_ENTRY( "SaveFilesInit()" );

	if( !lgFIRST )
		TotalInsanity();
	lgFIRST = false;

	/* set lgNoClobber to not overwrite files, reset with clobber on save line
	 * if we are running a grid (grid command entered in cdRead) grid.lgGrid 
	 * true, is false if single sim.  For grid we want to not clobber files 
	 * by default, do clobber for optimizer since this was behavior before */
	bool lgNoClobberDefault = false;
	if( grid.lgGrid )
	{
		/* cdRead encountered grid command - do not want to clobber files */
		lgNoClobberDefault = true;
	}

	for( i=0; i < LIMPUN; i++ )
	{
		save.lgNoClobber[i] = lgNoClobberDefault;
	}
	save.lgPunConv_noclobber = lgNoClobberDefault;
	save.lgDROn_noclobber = lgNoClobberDefault;
	save.lgDTOn_noclobber = lgNoClobberDefault;
	save.lgTraceConvergeBase_noclobber = lgNoClobberDefault;
	save.lgPunPoint_noclobber = lgNoClobberDefault;
	save.lgioRecom_noclobber = lgNoClobberDefault;
	save.lgQHSaveFile_noclobber = lgNoClobberDefault;
	save.lgSaveGrid_noclobber = lgNoClobberDefault;

	for( i=0; i < LIMPUN; i++ )
	{
		save.params[i].ipPnunit = NULL;

		// is this a real save command?  set false with the dummy 
		// save commands like save dr
		save.lgRealSave[i] = true;
	}

	save.lgTraceConvergeBase = false;

	save.ipDRout = NULL;
	save.lgDROn = false;

	save.ipDTout = NULL;
	save.lgDTOn = false;

	save.ipTraceConvergeBase = NULL;
	save.lgTraceConvergeBase = false;

	save.ipPunConv = NULL;
	save.lgPunConv = false;

	save.ipPoint = NULL;
	save.lgPunPoint = false;

	gv.QHSaveFile = NULL;

	save.ioRecom = NULL;
	save.lgioRecom = false;

	grid.pnunit = NULL;

	ioMAP = NULL;

	return;
}

/*CloseSaveFiles close save files called from cdEXIT upon termination,
 * from cloudy before returning 
 * NB - KEEP THIS ROUTINE SYNCHED UP WITH THE PREVIOUS ONE, SaveFilesInit */
void CloseSaveFiles( bool lgFinal )
{
	long int i;

	DEBUG_ENTRY( "CloseSaveFiles()" );

	/* close all save units cloudy opened with save command,
	 * lgNoClobber is set false with CLOBBER option on save, says to
	 * overwrite the files */
	for( i=0; i < save.nsave; i++ )
	{
		/* if lgFinal is true, we close everything, no matter what. 
		 * this means ignoring "no clobber" options */
		if( save.params[i].ipPnunit != NULL && ( !save.lgNoClobber[i] || lgFinal ) )
		{
			fclose( save.params[i].ipPnunit );

			/* Test that any FITS files are the right size! */ 
			if( save.lgFITS[i] && !save.lgXSPEC[i] )
			{
				auto file_size = FileSize(save.chFileName[i]);
				if( file_size == FS_UNKNOWN || file_size%2880 != 0 )
					fprintf( ioQQQ, " PROBLEM FITS file %s has the wrong size!\n", save.chFileName[i].c_str() );
			}

			save.params[i].ipPnunit = NULL;
		}
	}

	/* following file handles are aliased to ipPnunit which was already closed above */
	if( save.ipDRout != NULL && ( !save.lgDROn_noclobber || lgFinal ) )
	{
		save.ipDRout = NULL;
		save.lgDROn = false;
	}

	/* following file handles are aliased to ipPnunit which was already closed above */
	if( save.ipDTout != NULL && ( !save.lgDTOn_noclobber || lgFinal ) )
	{
		save.ipDTout = NULL;
		save.lgDTOn = false;
	}

	if( save.ipTraceConvergeBase != NULL && ( !save.lgTraceConvergeBase_noclobber || lgFinal ) )
	{
		save.ipTraceConvergeBase = NULL;
		save.lgTraceConvergeBase = false;
	}

	if( save.ipPunConv != NULL && ( !save.lgPunConv_noclobber || lgFinal ) )
	{
		save.ipPunConv = NULL;
		save.lgPunConv = false;
	}
	if( save.ipPoint != NULL && ( !save.lgPunPoint_noclobber || lgFinal ) )
	{
		save.ipPoint = NULL;
		save.lgPunPoint = false;
	}
	if( gv.QHSaveFile != NULL  && ( !save.lgQHSaveFile_noclobber || lgFinal ) )
	{
		gv.QHSaveFile = NULL;
	}
	if( save.ioRecom != NULL  && ( !save.lgioRecom_noclobber || lgFinal ) )
	{
		save.ioRecom = NULL;
		save.lgioRecom = false;
	}
	if( grid.pnunit != NULL && ( !save.lgSaveGrid_noclobber || lgFinal ) )
	{
		grid.pnunit = NULL;
	}
	ioMAP = NULL;

	return;
}

/*ChkUnits check for keyword UNITS on line, then scan wavelength or energy units if present.
 * When doing output, the routine call
 * AnuUnit( energy ) will automatically return the energy in the right units,
 * when called to do save output */
STATIC const char* ChkUnits( Parser &p )
{
	DEBUG_ENTRY( "ChkUnits()" );

	const char* val="";
	/* option to set units for continuum energy in save output */
	if( p.nMatch("UNITS") )
	{
		// p.StandardEnergyUnit() will terminate if no unit was recognized
		val = p.StandardEnergyUnit();
	}
	else
	{
		val = StandardEnergyUnit(" RYD ");
	}
	return val;
}
