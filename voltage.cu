#define WIN32_LEAN_AND_MEAN
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "math_functions.h"
#include "common/book.h"
#include "common/cpu_anim.h"
#include <stdio.h>
#include "settings.h"

#define MAX_TEMP 12.0f
#define MIN_TEMP 0.0f

#define DEFAULT_DIM 512
#define DEFAULT_NUMBALLS 25
#define DEFAULT_NUMELECTRODES 2
#define DEFAULT_EQUIPOTENTIALS 1
#define DEFAULT_VISCOSITY 1000
#define DEFAULT_PI 3.1415926535897932f
#define DEFAULT_DAMPING 1.0f // This was 0.95. "1" = no inertia, 0=all inertia preserved. 
#define DEFAULT_RESTITUTION 1
#define DEFAULT_RESISTANCE 5000.0
#define DEFAULT_NEGRESISTANCE 1.0

// globals needed by the update routine
struct DataBlock {
    unsigned char   *output_bitmap;
    float           *dev_inSrc;
    float           *dev_outSrc;
    float           *dev_constSrc;
	//tagz
	float			*dev_current;
	FILE			*fileptr;
    CPUAnimBitmap	*bitmap;
	int             *dev_ballradius;
	int				*dev_ballpos;
	float			*dev_ballposfloat;
	float			*dev_ballvel;
	float			*dev_forces;
	float			*dev_ballvoltage;
	float			*dev_ballvoltage_prev;
	int             *dev_output_parms;

    cudaEvent_t     start, stop;
    float           totalTime;
    float           frames;

	// program settings
	int             image_dim;
	int             num_balls;
	int             num_electrodes;
	int             viscosity;
	float           damping;
	int             equipotentials;
	int             restitution;

	// debug
	int             *debug_parms;
};

// GPU functions
__global__ void blend_kernel( float *in, float *out, const float *c, int image_dim );
__global__ void float_to_color( unsigned char *optr, const float *outSrc, const int *ballpos, int *ballradius, int equipotentials, int num_electrodes, int num_balls, int image_dim );
__global__ void calc_forces ( float *voltage, float *forces, int *ballpos, int *ballradius, int num_electrodes, int num_balls, int viscosity, int image_size );
__global__ void move_balls ( float *forces, int *ballpos, float *ballposfloat, float *ballvolt, float *ballvel, int *ballradius, int num_electrodes, int num_balls, float damping, int image_dim );
__global__ void recalc_const ( float *constvoltage, int *ballpos, float *ballvolt, int *ballradius, int num_electrodes, int num_balls, float *ballvolt_initial );
//tagz
__global__ void calc_current (float *current, float *ballposfloat, float *ballvel, float *ballvolt, int *ballradius, int num_electrodes, int num_balls );


__global__ void debug( float *pos, float *vel, float *force, int *found_neg, int number_electrodes, int number_balls, int frame_number, char func );
__global__ void debug_print_values( float *list, int *parm, int number_electrodes, int number_balls, char list_type );

// CPU functions
void anim_gpu( DataBlock *d, int ticks );
void anim_exit( DataBlock *d );

int main( void ) {
    DataBlock data;

	// get the initial settings from the external file
	printf("Loading settings...\n");
	ProgramSettings settings = get_settings("settings.txt");

	if ( settings.values_set==1 ) {
		data.image_dim      = settings.image_size;
		data.num_balls      = settings.number_ballbearings;
		data.num_electrodes = settings.number_electrodes;
		data.viscosity      = settings.viscosity;
		data.damping        = settings.damping;
		data.equipotentials = settings.equipotentials;
		data.restitution    = settings.restitution;
	} else {
		printf("Using default values...\n");
		data.image_dim      = DEFAULT_DIM;
		data.num_balls      = DEFAULT_NUMBALLS;
		data.num_electrodes = DEFAULT_NUMELECTRODES;
		data.viscosity      = DEFAULT_VISCOSITY;
		data.damping        = DEFAULT_DAMPING;
		data.equipotentials = DEFAULT_EQUIPOTENTIALS;
		data.restitution    = DEFAULT_RESTITUTION;
	}

	if (data.damping > 1 || data.damping < 0) 
	{
		data.damping=1;
		printf("You tried to set damping > 1. Don't do that!\n");
	}  
	printf("Settings loaded!\n");

	FILE *f = fopen("current.csv", "w"); 
    fprintf(f, "0.0,\n"); 
    fclose(f);
	data.fileptr= fopen("current.csv", "a");  

    CPUAnimBitmap bitmap( data.image_dim, data.image_dim, &data );
    data.bitmap    = &bitmap;
    data.totalTime = 0;
    data.frames    = 0;

	int total_balls    = data.num_balls+data.num_electrodes;
    long imageSize     = bitmap.image_size();
	int ballSize       = total_balls*sizeof(int);
	int ballSize_float = total_balls*sizeof(float);

    HANDLE_ERROR( cudaMalloc( (void**)&data.output_bitmap, imageSize ) );
    HANDLE_ERROR( cudaMalloc( (void**)&data.dev_inSrc, imageSize ) );
    HANDLE_ERROR( cudaMalloc( (void**)&data.dev_outSrc, imageSize ) );
    HANDLE_ERROR( cudaMalloc( (void**)&data.dev_constSrc, imageSize ) );
	HANDLE_ERROR( cudaMalloc( (void**)&data.dev_ballpos, 2*ballSize ) );
	HANDLE_ERROR( cudaMalloc( (void**)&data.dev_ballposfloat, 2*ballSize_float ) );
	HANDLE_ERROR( cudaMalloc( (void**)&data.dev_ballvel, 2*ballSize_float ) );
	HANDLE_ERROR( cudaMalloc( (void**)&data.dev_forces, 2*ballSize_float ) );
	HANDLE_ERROR( cudaMalloc( (void**)&data.dev_ballvoltage, ballSize_float ) );
	HANDLE_ERROR( cudaMalloc( (void**)&data.dev_ballvoltage_prev, ballSize_float ) );
	HANDLE_ERROR( cudaMalloc( (void**)&data.dev_ballradius, ballSize ) );
	HANDLE_ERROR( cudaMalloc( (void**)&data.dev_output_parms, 2*sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&data.dev_current, sizeof(float) ) );

	HANDLE_ERROR( cudaMalloc( (void**)&data.debug_parms, 2*sizeof(int) ) );

	HANDLE_ERROR( cudaMemset( data.dev_forces, 0.0f, 2*ballSize_float ) );
	HANDLE_ERROR( cudaMemset( data.dev_ballvel, 0.0f, 2*ballSize_float ) );
	HANDLE_ERROR( cudaMemset( data.dev_output_parms, 0, 2*sizeof(int) ) );
	//tagz
	HANDLE_ERROR( cudaMemset( data.dev_current, 0.0f, sizeof(float) ) );

	HANDLE_ERROR( cudaMemset( data.debug_parms, 0, 2*sizeof(int) ) );

    // intialize the constant data
    float *temp         = (float*)malloc( imageSize );
	int   *ballpos      = (int*)malloc( 2*ballSize );
	float *ballposfloat = (float*)malloc( 2*ballSize_float );
	float *ballvolt     = (float*)malloc( ballSize_float );
	int   *ballradius   = (int*)malloc( ballSize );

	if ( settings.values_set == 1 ) {
		// set the electrode positions and sizes from the settings file values

		for ( int i = 0; i < data.num_electrodes; i++ ) {
			ballposfloat[2*i]   = settings.elec_positions[2*i];
			ballposfloat[2*i+1] = settings.elec_positions[2*i+1];

			ballpos[2*i]        = (int)settings.elec_positions[2*i];
			ballpos[2*i+1]      = (int)settings.elec_positions[2*i+1];

			ballradius[i]       = settings.elec_sizes_radius[i];
			ballvolt[i]         = settings.elec_voltage[i];
			
		}

		// set the ball positions and sizesfrom the settings file values
		for ( int k = data.num_electrodes; k < total_balls; k++) {
			ballposfloat[2*k]   = settings.ball_positions[2*(k-data.num_electrodes)];
			ballposfloat[2*k+1] = settings.ball_positions[2*(k-data.num_electrodes)+1];

			ballpos[2*k]        = (int)ballposfloat[2*k];
			ballpos[2*k+1]      = (int)ballposfloat[2*k+1];

			ballradius[k]       = settings.ball_sizes_radius[k-data.num_electrodes];
			ballvolt[k]         = settings.ball_voltage[k-data.num_electrodes];

		}
	} else {
		// Set the electrode positions using default program values

		// The first two positions (ballpos 0 & 1) are for the top electrode
		// and the next two (ballpos 2 & 3) are for the bottom electrode
		ballpos[0]=data.image_dim/2;ballpos[1]=data.image_dim/32;
		ballpos[2]=data.image_dim/2;ballpos[3]=31*data.image_dim/32;
	
		// Set the ball bearing positions
		// Start above the number of electrodes
		srand( (unsigned)time( NULL ) );
		int j=data.num_electrodes;
		while ( j<total_balls ) {
			int redo=0;
			ballpos[2*j] = rand() % (data.image_dim-data.image_dim/16) + data.image_dim/32;
			ballpos[2*j+1] = rand() % (data.image_dim-data.image_dim/16) + data.image_dim/32;
			for (int k=0; k<j; k++) {
				int	dist = sqrtf(powf(ballpos[2*j]-ballpos[2*k],2) + powf(ballpos[2*j+1]-ballpos[2*k+1],2));
				if ( dist < (data.image_dim/16) ) {
					redo=1;
				}
			}
			if (redo==0) {
				j++;
			}
		}	

		for ( int j=0; j<total_balls; j++ ) {
			ballposfloat[2*j]=ballpos[2*j];
			ballposfloat[2*j+1]=ballpos[2*j+1];
		}

		// Set the first electrode to be positive
		// and the rest to be negative
		for ( int i= 0; i < data.num_electrodes ; i++ ) {
			if ( i==0 ) {
				ballvolt[i]=MAX_TEMP;
			} else {
				ballvolt[i]=MIN_TEMP;
			}
		}
		// set all of the ball bearings to a neutral voltage
		for ( int j=data.num_electrodes; j<total_balls; j++ ) {
			ballvolt[j]=(MAX_TEMP + MIN_TEMP)/2;
		}
	}
		
	// loop through each pixel element
	// if the pixel is within the radius of a ball or electrode
	// set the pixel's voltage to the ball/electrode voltage.
    for ( int i=0; i< data.image_dim*data.image_dim; i++) {
        int x = i % data.image_dim;
        int y = i / data.image_dim;
		for ( int j=0; j<total_balls; j++ ) {
			float dist = sqrtf(powf(x-ballpos[2*j],2) + powf(y-ballpos[2*j+1],2));

			if ( settings.values_set==1 ) {
				if ( dist < settings.ball_sizes_radius[j] ) {
					temp[i] = ballvolt[j];
				} else {
					temp[i] = -1.0f;
				}
			} else {
				if ( dist < data.image_dim/32 ) {
					temp[i]=ballvolt[j];
				} else {
					temp[i] = -1.0f;
				}
			}
		}
    } 
	
	HANDLE_ERROR( cudaMemcpy( data.dev_constSrc, temp, imageSize, cudaMemcpyHostToDevice ) );    
	HANDLE_ERROR( cudaMemcpy( data.dev_ballpos, ballpos, 2*ballSize, cudaMemcpyHostToDevice ) );  
	HANDLE_ERROR( cudaMemcpy( data.dev_ballposfloat, ballposfloat, 2*ballSize_float, cudaMemcpyHostToDevice ) );  
	HANDLE_ERROR( cudaMemcpy( data.dev_ballvoltage, ballvolt, ballSize_float, cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( data.dev_ballvoltage_prev, ballvolt, ballSize_float, cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( data.dev_ballradius, ballradius, ballSize, cudaMemcpyHostToDevice ) );

    // initialize the input data
    for (int i=0; i<data.image_dim*data.image_dim; i++) {
            temp[i] = (MAX_TEMP+MIN_TEMP)/2;
    }

    HANDLE_ERROR( cudaMemcpy( data.dev_inSrc, temp, imageSize, cudaMemcpyHostToDevice ) );   
    HANDLE_ERROR( cudaMemcpy( data.dev_outSrc, temp, imageSize, cudaMemcpyHostToDevice ) );   

	free( temp );
	free( ballpos );
	free( ballposfloat );
	free( ballvolt );
	free( ballradius );
	free( settings.ball_positions );
	free( settings.elec_positions );
	free( settings.ball_sizes_radius );
	free( settings.elec_sizes_radius );
	free( settings.ball_voltage );
	free( settings.elec_voltage );

    bitmap.anim_and_exit( (void (*)(void*,int))anim_gpu, (void (*)(void*))anim_exit );

}

void anim_gpu( DataBlock *d, int ticks ) {
	int     dim = d->image_dim;
    dim3    blocks( dim/16, dim/16 );
    dim3    threads(16,16);
    CPUAnimBitmap  *bitmap = d->bitmap;
	//tagz
	float cpucurrent;

	//debug_print_values<<<1,1>>>( d->dev_ballvel, d->debug_parms, d->num_electrodes, d->num_balls, 'v' );
	//debug_print_values<<<1,1>>>( d->dev_forces, d->debug_parms, d->num_electrodes, d->num_balls, 'f' );

// The electrodes will charge a little each time step until they reach their max charge


    blend_kernel<<<blocks,threads>>>( d->dev_inSrc, d->dev_outSrc, d->dev_constSrc, d->image_dim );
	float_to_color<<<blocks,threads>>>( d->output_bitmap, d->dev_inSrc, d->dev_ballpos, d->dev_ballradius, d->equipotentials, d->num_electrodes, d->num_balls, d->image_dim );
	calc_forces<<<blocks,threads>>> (d->dev_inSrc, d->dev_forces, d->dev_ballpos, d->dev_ballradius, d->num_electrodes, d->num_balls, d->viscosity, d->image_dim );
//	debug_print_values<<<1,1>>>( d->dev_forces, d->debug_parms, d->num_electrodes, d->num_balls, 'f' );
//	debug<<<1,1>>>( d->dev_ballposfloat, d->dev_ballvel, d->dev_forces, d->debug_parms, d->num_electrodes, d->num_balls, d->frames, 'c' );
	move_balls<<<blocks,threads>>> ( d->dev_forces, d->dev_ballpos, d->dev_ballposfloat, d->dev_ballvoltage, d->dev_ballvel, d->dev_ballradius, d->num_electrodes, d->num_balls, d->damping, d->image_dim );
//	debug<<<1,1>>>( d->dev_ballposfloat, d->dev_ballvel, d->dev_forces, d->debug_parms, d->num_electrodes, d->num_balls, d->frames, 'm' );
	recalc_const<<<blocks,threads>>> (d->dev_constSrc, d->dev_ballpos, d->dev_ballvoltage, d->dev_ballradius, d->num_electrodes, d->num_balls , d->dev_ballvoltage_prev);
	//tagz
	calc_current<<<blocks,threads>>> (d->dev_current, d->dev_ballposfloat, d->dev_ballvel, d->dev_ballvoltage, d->dev_ballradius, d->num_electrodes, d->num_balls );
    
    HANDLE_ERROR( cudaMemcpy( bitmap->get_ptr(), d->output_bitmap, bitmap->image_size(), cudaMemcpyDeviceToHost ) );
	//tagz
	HANDLE_ERROR( cudaMemcpy( &cpucurrent, d->dev_current, sizeof(float), cudaMemcpyDeviceToHost ) );

    //fprintf(d->fileptr, "%f,\n", cpucurrent);  
}

// clean up memory allocated on the GPU
// add save the final positions of the ball bearings
void anim_exit( DataBlock *d ) {
	int ballSize            = (d->num_balls+d->num_electrodes)*sizeof(int);
	int *ballpos_temp       = (int*)malloc( 2*ballSize );
	int *ballradius_temp    = (int*)malloc( ballSize );

	HANDLE_ERROR( cudaMemcpy( ballradius_temp, d->dev_ballradius, ballSize, cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( ballpos_temp, d->dev_ballpos, 2*ballSize, cudaMemcpyDeviceToHost ) );

	save_ball_positions("final_ball_positions.txt", ballpos_temp, ballradius_temp, d->num_electrodes, d->num_balls );

	HANDLE_ERROR( cudaFree( d->output_bitmap ) );
	HANDLE_ERROR( cudaFree( d->dev_inSrc ) );
	HANDLE_ERROR( cudaFree( d->dev_outSrc ) );
	HANDLE_ERROR( cudaFree( d->dev_constSrc ) );
	HANDLE_ERROR( cudaFree( d->dev_ballpos ) );
	HANDLE_ERROR( cudaFree( d->dev_ballposfloat ) );
	HANDLE_ERROR( cudaFree( d->dev_ballvel ) );
	HANDLE_ERROR( cudaFree( d->dev_forces ) );
	//tagz
	HANDLE_ERROR( cudaFree( d->dev_current ) );
	HANDLE_ERROR( cudaFree( d->dev_ballvoltage ) );
	HANDLE_ERROR( cudaFree( d->dev_ballvoltage_prev ) );
	HANDLE_ERROR( cudaFree( d->dev_ballradius ) );
	HANDLE_ERROR( cudaFree( d->dev_output_parms ) );

	free( ballpos_temp );
	free( ballradius_temp );
	fclose(d->fileptr);
}

__global__ void debug( float *pos, float *vel, float *force, int *found_neg, int number_electrodes, int number_balls, int frame_number, char func ) {
	if( found_neg[0]==0 ) {
		for( int i=number_electrodes; i < number_electrodes+number_balls; i++ ) {
			if( force[2*i] < 0 || force[2*i+1] < 0 ) {
				printf("negative force found in frame %i on ball %i in function %c\n", frame_number, i, func );

				found_neg[0]=1;
			}
			if( vel[2*i] < 0 || vel[2*i+1] < 0 ) {
				printf("negative velocity found in frame %i on ball %i in function %c\n", frame_number, i, func );

				found_neg[0]=1;
			}
			if( pos[2*i] < 0 || pos[2*i+1] < 0 ) {
				printf("negative pos found in frame %i on ball %i in function %c\n", frame_number, i, func);

				found_neg[0]=1;
			}
		}
	}
}

__global__ void debug_print_values( float *list, int *parm, int number_electrodes, int number_balls, char list_type ) {
	if( parm[1]==0 ) {
		for( int i=number_electrodes; i < number_electrodes+number_balls; i++ ) {
			printf("%c values ball %03i: %+11.10f, %+11.10f\n", list_type, i, list[2*i], list[2*i+1]);
		}
		parm[1]=1;
	}
}

// this kernel takes in a 2-d array of floats
// it updates the value-of-interest by a scaled value based
// on itself and its nearest neighbors
__global__ void blend_kernel( float *in, float *out, const float *c, int image_dim ) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    int left = offset - 1;
    int right = offset + 1;
    if (x == 0)   left++;
    if (x == image_dim-1) right--;

    int top = offset - image_dim;
    int bottom = offset + image_dim;
    if (y == 0)   top += image_dim;
    if (y == image_dim-1) bottom -= image_dim;

    float t,b,l,r;
	for (int k=0;k<=2*blockDim.x;k++) {
		if (k % 2 == 0) {
			if (c[top] == -1.0f) t=in[top];
			else t=c[top];
			if (c[bottom] == -1.0f) b=in[bottom];
			else b=c[bottom];
			if (c[left] == -1.0f) l=in[left];
			else l=c[left];
			if (c[right] == -1.0f) r=in[right];
			else r=c[right];
			out[offset]=(l+r+t+b+in[offset])/5;
		} else {
			if (c[top] == -1.0f) t=out[top];
			else t=c[top];
			if (c[bottom] == -1.0f) b=out[bottom];
			else b=c[bottom];
			if (c[left] == -1.0f) l=out[left];
			else l=c[left];
			if (c[right] == -1.0f) r=out[right];
			else r=c[right];
			in[offset]=(l+r+t+b+out[offset])/5;
		}
	}

}

__global__ void float_to_color( unsigned char *optr, const float *outSrc, const int *ballpos, int *ballradius, int equipotentials, int num_electrodes, int num_balls, int image_dim ) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float l = outSrc[offset];
	int q = (int) equipotentials*l;
	float r = equipotentials*l - q;
	
	optr[offset*4 + 3] = 255;

	for (int j=0; j<num_electrodes+num_balls;j++) {
		float dist = sqrtf(powf(x-ballpos[2*j],2) + powf(y-ballpos[2*j+1],2));
		if (dist < ballradius[j] ) {
			optr[offset*4 + 0] = 255*outSrc[ballpos[2*j]+ballpos[2*j+1]*image_dim]/MAX_TEMP;
			optr[offset*4 + 1] = 255*outSrc[ballpos[2*j]+ballpos[2*j+1]*image_dim]/MAX_TEMP;
			optr[offset*4 + 2] = 255*outSrc[ballpos[2*j]+ballpos[2*j+1]*image_dim]/MAX_TEMP;
			return;
		}
	}

	switch (q%3)	{
		case 0: {
			optr[offset*4 + 0] = 255-255*r;
			optr[offset*4 + 1] = 255*r;
			optr[offset*4 + 2] = 0;
			break;
		}
		case 1: {
			optr[offset*4 + 0] = 0;
			optr[offset*4 + 1] = 255-255*r;
			optr[offset*4 + 2] = 255*r;
			break;
		}
		case 2:	{
			optr[offset*4 + 0] = 255*r;
			optr[offset*4 + 1] = 0;
			optr[offset*4 + 2] = 255-255*r;
			break;
		}
	}

}

__global__ void calc_forces ( float *voltage, float *forces, int *ballpos, int *ballradius, int num_electrodes, int num_balls, int viscosity, int image_dim ) {
    //  map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
	int displacementx;
	int displacementy;
	float dist;

	// loop through each of the balls, not including the electrodes
	for (int i=num_electrodes;i<num_balls+num_electrodes;i++){
		displacementx=x-ballpos[2*i];
		displacementy=y-ballpos[2*i+1];
		// get the distance of the pixel to the ball
		dist = sqrtf(powf(displacementx,2) + powf(displacementy,2));
		// if the distance is within 1.4 pixels of the ball's diameter
		// calculate and add the forces 
		if ( dist < (ballradius[i] + 1.414) && dist > ballradius[i] ) {
				atomicAdd(&forces[2*i], displacementx*powf(voltage[offset]-voltage[ballpos[2*i]+ballpos[2*i+1]*image_dim], 2)/(viscosity * ballradius[i]));
				atomicAdd(&forces[2*i+1], displacementy*powf(voltage[offset]-voltage[ballpos[2*i]+ballpos[2*i+1]*image_dim],2)/(viscosity * ballradius[i]));
		}
	}
//	if (offset < num_balls) printf("The forces on ball %i are %f and %f\n",offset,forces[2*offset],forces[2*offset+1]);
}

__global__ void move_balls ( float *forces, int *ballpos, float *ballposfloat, float *ballvolt, float *ballvel, int *ballradius, int num_electrodes, int num_balls, float damping, int image_dim ) {
    int a = threadIdx.x + blockIdx.x * blockDim.x;
    int b = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = a + b * blockDim.x * gridDim.x;


	if ( ( (num_electrodes - 1) < offset) && (offset < (num_balls + num_electrodes)) ) {
		ballposfloat[offset*2]+=ballvel[offset*2];
		ballposfloat[offset*2+1]+=ballvel[offset*2+1];
		ballvel[offset*2]+=forces[offset*2]-ballvel[offset*2]*damping;
		ballvel[offset*2+1]+=forces[offset*2+1]-ballvel[offset*2+1]*damping;
		forces[offset*2]=0;
		forces[offset*2+1]=0;
		// x-position boundary checks
		// if the ball's position is less than its radius then set its position to the edge
		// and reverse its x velocity
		if (ballposfloat[offset*2]<=ballradius[offset*2]) {
			ballposfloat[offset*2]=ballradius[offset*2];
			if (ballvel[offset*2]<0) ballvel[offset*2]= -ballvel[offset*2];
		}
		// if the ball's position is greater than the image's length minus the ball's radius then set its position to the edge
		// and reverse its x velocity
		if (ballposfloat[offset*2]>=image_dim-ballradius[offset*2]) {
			ballposfloat[offset*2]=image_dim-ballradius[offset*2];
			if (ballvel[offset*2]>0) ballvel[offset*2]= -ballvel[offset*2];
		}
		// y-position boundary checks
		// if the ball's position is less than its radius then set its position to the edge
		// and reverse its y velocity
		if (ballposfloat[offset*2+1]<=ballradius[offset*2]) {
			ballposfloat[offset*2+1]=ballradius[offset*2];
			if (ballvel[offset*2+1]<0) ballvel[offset*2+1]= -ballvel[offset*2+1];
		}
		// if the ball's position is greater than the image's length minus the ball's radius then set its position to the edge
		// and reverse its y velocity
		if (ballposfloat[offset*2+1]>=image_dim-ballradius[offset*2]) {
			ballposfloat[offset*2+1]=image_dim-ballradius[offset*2];
			if (ballvel[offset*2+1]>0) ballvel[offset*2+1]= -ballvel[offset*2+1];
		}
		if (ballvel[offset*2] > 100) printf("Ball velocity %i is %f\n",offset-num_electrodes+1,ballvel[offset*2]);
	}

	
	if ( (a < b) && (b < (num_balls + num_electrodes) )) {
		float displacex=ballposfloat[2*b]-ballposfloat[2*a];
		float displacey=ballposfloat[2*b+1]-ballposfloat[2*a+1];
		float dist = sqrtf(powf(displacex,2) + powf(displacey,2));
		float unitx;
		float unity;
		if ( dist == 0 ) {
			unitx=0;
			unity=0;

		} else {
			unitx=displacex/dist;
			unity=displacey/dist;

		}
		// Check if the balls hit each other
		if ( dist <= (ballradius[a]+ballradius[b]) ) {
			// The balls did hit so now we calculate the new velocities of the balls
			float collisionvelocityax = ballvel[2*a]*powf(unitx,2)+ballvel[2*a+1]*unitx*unity;
			float collisionvelocityay = ballvel[2*a]*unitx*unity+ballvel[2*a+1]*powf(unity,2);
			float collisionvelocitybx = ballvel[2*b]*powf(unitx,2)+ballvel[2*b+1]*unitx*unity;
			float collisionvelocityby = ballvel[2*b]*unitx*unity+ballvel[2*b+1]*powf(unity,2);
			
			if ( a < num_electrodes ) {
				// the ball hit another ball
				// set the voltage of the balls as an average of their original voltages
				//ballvolt[a]=(ballvolt[a]+ballvolt[b])/2;
				//let the positive electrodes dischrge but not the negative ones
				//if(ballvolt[a]!=0) ballvolt[a]=(ballvolt[a]+ballvolt[b])/2;
				ballvolt[a]=(ballvolt[a]+ballvolt[b])/2;
				ballvolt[b]=ballvolt[a];

				ballposfloat[2*b]+=((ballradius[a]+ballradius[b])-dist)*unitx;
				ballposfloat[2*b+1]+=((ballradius[a]+ballradius[b])-dist)*unity;
				
				ballvel[2*b]=ballvel[2*b]-2*collisionvelocitybx;
				ballvel[2*b+1]=ballvel[2*b+1]-2*collisionvelocityby;

			} else {

				// the ball hit another ball
				// set the voltage of the balls as an average of their original voltages
				ballvolt[a]=(ballvolt[a]+ballvolt[b])/2;
				ballvolt[b]=ballvolt[a];

				ballposfloat[2*a]-=((ballradius[a]+ballradius[b])-dist)*unitx/2;
				ballposfloat[2*a+1]-=((ballradius[a]+ballradius[b])-dist)*unity/2;
				ballposfloat[2*b]+=((ballradius[a]+ballradius[b])-dist)*unitx/2;
				ballposfloat[2*b+1]+=((ballradius[a]+ballradius[b])-dist)*unity/2;

				ballvel[2*a]=ballvel[2*a]+collisionvelocitybx-collisionvelocityax;
				ballvel[2*a+1]=ballvel[2*a+1]+collisionvelocityby-collisionvelocityay;
				ballvel[2*b]=ballvel[2*b]+collisionvelocityax-collisionvelocitybx;
				ballvel[2*b+1]=ballvel[2*b+1]+collisionvelocityay-collisionvelocityby;
				
			}
		}
	}
	
	// Set the integer ball position to the rounded float ball position
	if ( ( (num_electrodes - 1) < offset) && (offset < (num_balls + num_electrodes)) ) {
		ballpos[offset*2]  =(int)rintf(ballposfloat[offset*2]);
		ballpos[offset*2+1]=(int)rintf(ballposfloat[offset*2+1]);
	}
}

__global__ void recalc_const (float *constvoltage, int *ballpos, float *ballvolt, int *ballradius, int num_electrodes, int num_balls, float *ballvolt_initial ) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    for (int j=0; j<num_electrodes+num_balls;j++) {
		float dist = sqrtf(powf(x-ballpos[2*j],2) + powf(y-ballpos[2*j+1],2));
		if ( dist < ballradius[j] ) {
				constvoltage[offset]=ballvolt[j];
				return;
		}
	}
	constvoltage[offset]=-1;
	//give the electrodes a finite charging rate, but make the negative electrode charge at a different rate
	if (offset < num_electrodes&&ballvolt_initial[offset]!=0) ballvolt[offset]+=(ballvolt_initial[offset]-ballvolt[offset])/DEFAULT_RESISTANCE;
	if (offset < num_electrodes&&ballvolt_initial[offset]==0) ballvolt[offset]+=(ballvolt_initial[offset]-ballvolt[offset])/DEFAULT_NEGRESISTANCE;

	
}
//tagz
__global__ void calc_current (float *current, float *ballposfloat, float *ballvel, float *ballvolt, int *ballradius, int num_electrodes, int num_balls ) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

	if (offset==0){
		
		float charge = 0;
		float dist = 0;
		for (int j=0; j<num_electrodes;j++) {
		//tagz
			float currenttmp = 0;
			for (int k=j+1; k<num_electrodes;k++) {
				for (int i=num_electrodes; i<(num_balls + num_electrodes); i++){
					dist = sqrtf(powf(ballposfloat[2*j]-ballposfloat[2*k],2) + powf(ballposfloat[2*j+1]-ballposfloat[2*k+1],2));	
					charge = ballvolt[i]-(ballvolt[j]-ballvolt[k])/2;
					float unitx=(ballposfloat[2*j]-ballposfloat[2*k])/dist*(ballvolt[k]-ballvolt[j]);
					float unity=(ballposfloat[2*j+1]-ballposfloat[2*k+1])/dist*(ballvolt[k]-ballvolt[j]);
					//tagz
					currenttmp =currenttmp+ charge * (ballvel[2*i]*unitx + ballvel[2*i+1]*unity) * ballradius[i] * ballradius[i];
				}
				//tagz
				//printf("The current between electrode %i and electrode %i is %f\n",j,k,currenttmp);
				*current=currenttmp;
			}
		}
	}
}
