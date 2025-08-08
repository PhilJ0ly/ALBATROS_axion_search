#include <stdio.h>
#include <stdlib.h>
#include "mathlib.h"
#include "sph_cap.h"
#include "calcChar.h"
#include "ACHAIMDB.h"
#include "ACHAIM_Chars.h"
#include "BACKGROUND_Chars.h"

// Philippe Joly 2025-06-02

// This script extract the maximal F2 electron density predicted by A-CHAIM and E-CHAIM given a CHAIM database file.

// Usage: ./CHAIM_extract.o <Filename>
// Associated Makefile: Makefile

int main(int argc, char *argv[])
{
	// ALBATROS Uapishka deployment Coords
	double lat[1] = {51.4646065}, lon[1] = {-68.2352594};
	int nLat = 1;
	char *	dbFile = argv[1];
	double * output_ac, * output_ec;
		
	output_ac = AC_Char("nmf2", lat, lon, nLat, dbFile); // A-CHAIM
	printf("%f ",output_ac[0]);
	
	output_ec = BG_Char("nmf2", lat, lon, nLat, dbFile); // E-CHAIM
	printf("%f",output_ec[0]);
}
