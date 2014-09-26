#include <stdio.h>
#include <string.h>

struct ProgramSettings {
	int     values_set;
	int     image_size;
	int     number_ballbearings;
	int     number_electrodes;
	float   viscosity;
	float   damping;
	int     equipotentials;
	int     restitution;
	int     save_pos;

	float  *ball_positions;
	int    *ball_sizes_radius;
	float  *ball_voltage;
	float  *elec_positions;
	int    *elec_sizes_radius;
	float  *elec_voltage;

	ProgramSettings (void) {
		values_set = 0;
		save_pos   = 0;
	}
};

ProgramSettings get_settings(char *filename) {
	ProgramSettings settings;
	FILE *fl;
	int counter_balls = 0;
	int counter_elect = 0;
	char name[15];
	float value1, value4;
	int value2, value3;

	fl = fopen(filename, "r");

	// if the file can't be opened return an empty ProgramSettings 
	if (fl == NULL) return settings;

	// Loop through the file
	while(EOF != fscanf(fl, "%[^;]; %f %i %i %f", name, &value1, &value2, &value3, &value4) ) {
		
		if ( strstr(name, "image_size") ) {
			// the value of image_size should be divisible by 16
			settings.image_size = (int)value1;

		} else if ( strstr(name, "num_electrodes") ) {
			settings.number_electrodes = (int)value1;

			// If the number electrodes has been set to greater than zero then allocate the
			// memory for the electrode positions and sizes
			if ( settings.number_electrodes > 0 ) {
				settings.elec_positions = (float*) malloc ( 2*settings.number_electrodes*sizeof(float) );
				settings.elec_sizes_radius = (int*) malloc ( settings.number_electrodes*sizeof(int) );
				settings.elec_voltage = (float*)malloc (settings.number_electrodes*sizeof(float) );
			}

		} else if ( strstr(name, "num_balls") ){
			settings.number_ballbearings = (int)value1;

			// If the number balls has been set to greater than zero then allocate the
			// memory for the ball positions and sizes
			if ( settings.number_ballbearings > 0 ) {
				settings.ball_positions = (float*) malloc ( 2*settings.number_ballbearings*sizeof(float) );
				settings.ball_sizes_radius = (int*) malloc ( settings.number_ballbearings*sizeof(int) );
				settings.ball_voltage = (float*) malloc (settings.number_ballbearings*sizeof(float) );
			}

		} else if ( strstr(name, "viscosity") ){
			settings.viscosity = (int)value1;

		} else if ( strstr(name, "damping") ){
			settings.damping = (float)value1;

		} else if ( strstr(name, "equipotentials") ) {
			settings.equipotentials = (int)value1;
		
		} else if ( strstr(name, "restitution") ) {
			settings.restitution = (int)value1;
		
		} else if ( strstr(name, "save_pos") ) {
			settings.save_pos = (int)value1;
		
		} else if ( strstr(name, "pos_ball") ){
			// pos_ball holds the x & y coordinates
			// as well as the radius and voltage of the ball
			settings.ball_positions[2*counter_balls]   = (float)value1;
			settings.ball_positions[2*counter_balls+1] = (float)value2;
			settings.ball_sizes_radius[counter_balls]  = (int)value3;
			settings.ball_voltage[counter_balls] = (float)value4;

			counter_balls++;

		} else if ( strstr(name, "pos_electrode") ) {
			// pos_electrode holds the x & y coordinates
			// as well as the radius and voltage of the electrode
			settings.elec_positions[2*counter_elect]   = (float)value1;
			settings.elec_positions[2*counter_elect+1] = (float)value2;
			settings.elec_sizes_radius[counter_elect]  = (int)value3;
			settings.elec_voltage[counter_elect] = (float)value4;

			counter_elect++;

		}
	}
	
	fclose(fl);

	// Show that all of the values have been set
	settings.values_set = 1;

	return settings;
}

void save_ball_positions( char *filename, int *ballpos, int *ballradius, int num_electrodes, int num_balls ) {
	FILE *fl;
	float neutral_voltage = 0.5f;

	fl = fopen(filename, "w");

	for( int i = num_electrodes; i < num_electrodes+num_balls; i++ ) {
		fprintf(fl, "pos_ball; %i %i %i %f\n", ballpos[2*i], ballpos[2*i+1], ballradius[i], neutral_voltage);
	}

	fclose(fl);
}
