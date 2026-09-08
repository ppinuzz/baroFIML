/*******************************************************************************
			UDF FOR BAROTROPIC PROPERTIES (ANSYS Fluent 2023 R2)
********************************************************************************/

/* General rules for Fluent UDF (as of 2023 R2): 
	- Use only C99 or earlier [UDF23, pag. 4]
	- Header must always use *.h extension [UDF23, pag. 4]
	- Always include the "udf.h" header [UDF23, pag. 4]
	- (Unclear) all UDFs should be written for parallel usage [UDF23, pag. 5]

	- Name arguments must be in LOWERCASE [UDF23, pag. 5]
	- Place all functions parameters on the SAME LINE as the DEFINE macro [UDF23, pag. 6]
	- NO SPACES between the macro and the opening '(' [UDF23, pag. 6]
	- Do NOT include a DEFINE macro statement in a comment [UDF23, pag. 6]
Sources:
	[UDF23] Ansys Fluent Customization Manual, release 2023 R2
*/

#include "udf.h"
#include <math.h>

/* THINGS YOU CAN CHANGE*/
// number of points used to discretise the pressure in the LUT
#define DISCR_PROPR_P 9001

// if defined, speed of sound is interpolated from the speed of sound LUT,
// otherwise it is computed from the derivative of the density LUT
#define SOS_FROM_LUT
#define VISCOUS_FLOW


/*******************************************************************************
				DO NOT TOUCH ANYTHING BELOW THIS LINE
********************************************************************************/

// numerical tolerance used to check whether p1 == p2 in interpolation function
#define NUM_TOL 1E-6
#define IS_EQUAL(a, b) (fabs((a) - (b)) < NUM_TOL)

/* anonymous enums can be used in C to generate *actual* integer constants 
https://stackoverflow.com/a/10157273/17220538
*/
enum {
	PROP_DENSITY = 0,
	#ifdef SOS_FROM_LUT
		PROP_SOUND_SPEED,
	#endif
	#ifdef VISCOUS_FLOW
		PROP_VISCOSITY,
	#endif
	/* automatically assigned the last enum value + 1, if you start from 0, this
	is automatically equal to the total number of values */
	NUMPROP_LIQ
};

// static -> make them "private" to this file
static double p_LUT[DISCR_PROPR_P];
static double prop_LUT[NUMPROP_LIQ][DISCR_PROPR_P];
static double dp, p_min, p_max;
// const -> make sure they're read-only (cannot be modified)
static const char* namefile[NUMPROP_LIQ];
static const char* Pname;

/* (1st parameter is the on-loading UDF name, 2nd is automatically set by Fluent) */
DEFINE_EXECUTE_ON_LOADING(load_lookup_tables, libname)
{
	namefile[PROP_DENSITY] = "./barotropic_properties/density.txt";
	#ifdef SOS_FROM_LUT
		namefile[PROP_SOUND_SPEED] = "./barotropic_properties/soundspeed.txt";
	#endif
	#ifdef VISCOUS_FLOW
		namefile[PROP_VISCOSITY] = "./barotropic_properties/viscosity.txt";
	#endif
	Pname = "./barotropic_properties/pressure.txt";

	Message0("Operating pressure: %lf Pa\n", op_pres);
	Message0("Reading lookup tables...\n");

	int outcode;
	FILE* Ppoint = fopen(Pname, "r");
	if (Ppoint == NULL){
		Error("Could not read pressure file %s\n", Pname);
	}
	for (size_t i = 0; i < DISCR_PROPR_P; i++){
		outcode = fscanf(Ppoint, "%lf", &p_LUT[i]);
		// fscanf() returns the number of successfully read items
		// (should be 1, as there's only 1 number per line)
		if (outcode != 1){
			Error("Error reading pressure file %s\n at entry %zu", Pname, i);
		}
	}
	fclose(Ppoint);
	Message0("Pressure array loaded!\n");

	dp = p_LUT[1] - p_LUT[0];
	p_min = p_LUT[0];
	p_max = p_LUT[DISCR_PROPR_P-1];
	Message0("Pressure array:\n"
			"\tMinimum pressure : %lf Pa\n"
			"\tMaximum pressure : %lf Pa\n"
			"\tPressure step    : %lf Pa\n"
			"\tNumber of points : %d\n", 
			p_min, p_max, dp, DISCR_PROPR_P);

	
	// read actual property LUTs
	for (size_t k = 0; k < NUMPROP_LIQ; k++){
		FILE* Fpoint = fopen(namefile[k], "r");
		if (Fpoint == NULL){
			Error("Could not read LUT %zu!\n", k);
		}
		for (size_t i = 0; i < DISCR_PROPR_P; i++){
			outcode = fscanf(Fpoint, "%lf", &prop_LUT[k][i]);
			if (outcode != 1){
				Error("Error reading LUT %zu\n at entry %zu", k, i);
			}
		}
		fclose(Fpoint);
		Message0("LUT %zu loaded!\n", k);
	}

	Message0("Property LUT loading ended! Loaded %d LUTs\n", NUMPROP_LIQ);
}


static double linear_interpolation(double P, int prop_idx)
{
	int indPlow, indPhigh;

	/* Handling out of bounds T, P and effectively clipping them and the relative index */

	if (P < p_min){
		indPlow = 0;
		indPhigh = 0;
	}
	else if (P > p_max){
		indPlow = DISCR_PROPR_P - 1;
		indPhigh = DISCR_PROPR_P - 1;
	}
	else {
		indPlow = floor((P - p_min) / dp);
		indPhigh = ceil((P - p_min) / dp);
	}

	double phi_low, phi_high;
	double Plow, Phigh;
	double phi;

	Plow = p_LUT[indPlow];
	Phigh = p_LUT[indPhigh];

	phi_low = prop_LUT[prop_idx][indPlow];
	phi_high = prop_LUT[prop_idx][indPhigh];

	if (indPlow == indPhigh){
		phi = phi_low;
	}
	else {
		phi = ((Phigh - P) / dp) * phi_low + ((P - Plow) / dp) * phi_high;
	}

	return phi;
}


/* CUSTOM PROPERTIES */

DEFINE_PROPERTY(barotropic_density, c, t)
{
	// operating pressure is set to 0 Pa for compressible simulations, but Fluent's
	// C_P(c,t) is the releative pressure by default! [UDF23, pag. 128]
	double P = C_P(c, t) + op_pres;
	double prop = linear_interpolation(P, PROP_DENSITY);
	return prop;
}

#ifdef VISCOUS_FLOW
	DEFINE_PROPERTY(barotropic_viscosity, c, t)
	{
		double P = C_P(c, t) + op_pres;
		double prop = linear_interpolation(P, PROP_VISCOSITY);
		return prop;
	}
#endif

// required for compressible liquids (such as the barotropic) [UDF23, pag. 123]
#ifdef SOS_FROM_LUT
	DEFINE_PROPERTY(barotropic_sound_speed, c, t)
	{
		double P = C_P(c, t) + op_pres;
		double prop = linear_interpolation(P, PROP_SOUND_SPEED);
		return prop;
	}
#else
	DEFINE_PROPERTY(barotropic_sound_speed, c, t)
	{
		double P = C_P(c, t) + op_pres;
		double p_plus = P + 0.5*dp;
		double p_minus = P - 0.5*dp;
		double rho_plus = linear_interpolation(p_plus, PROP_DENSITY);
		double rho_minus = linear_interpolation(p_minus, PROP_DENSITY);
		// if both pressure values have been clipped, then the density values 
		// will be equal and the speed of sound will be 0/0 => force the "last
		// known value" of the speed of sound
		if (IS_EQUAL(rho_plus, rho_minus)){
			if (p_minus > p_max){
				p_plus = p_max;
				p_minus = p_plus - dp;
				rho_plus = linear_interpolation(p_plus, PROP_DENSITY);
				rho_minus = linear_interpolation(p_minus, PROP_DENSITY);
			}
			else if (p_plus < p_min){
				p_minus = p_min;
				p_plus = p_minus + dp;
				rho_plus = linear_interpolation(p_plus, PROP_DENSITY);
				rho_minus = linear_interpolation(p_minus, PROP_DENSITY);
			}
			else{
				Error("Unexpected error in speed of sound computation!\n");
			}
		}
		double dpdrho = (p_plus - p_minus) / (rho_plus - rho_minus);
		double prop = sqrt(dpdrho);
		return prop;
	}
#endif
