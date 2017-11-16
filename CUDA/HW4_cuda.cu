#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <time.h>
#include <sys/time.h>
#include <algorithm>

#define INF 1000000

double t1, t2;
int n, m;	// Number of vertices, edges
int *Dist;
int *dev_dist;

double wallclock(void)
{	struct timeval tv;
	struct timezone tz;
	double t;

	gettimeofday(&tv, &tz);

	t = (double)tv.tv_sec*1000;
	t += ((double)tv.tv_usec)/1000.0;

	return t;
}// millisecond

void input(char *inFileName){	
	FILE *infile = fopen(inFileName, "r");
	fscanf(infile, "%d %d", &n, &m);

	//Dist = (int*)malloc(sizeof(int)*n*n);
	cudaMallocHost((void**) &Dist, sizeof(int)*n*n); //Pinned Memory
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			(i==j)?Dist[i*n+j]=0:Dist[i*n+j]=INF;

	int a, b, v;
	while (m--) {
		fscanf(infile, "%d %d %d", &a, &b, &v);
		Dist[(a-1)*n+(b-1)] = v;
	}
}

void output(char *outFileName){	
	FILE *outfile = fopen(outFileName, "w");
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			if (Dist[i*n+j] >= INF)	fprintf(outfile, "INF ");
			else					fprintf(outfile, "%d ", Dist[i*n+j]);
		}
		fprintf(outfile, "\n");
	}		
}

int ceil(int a, int b){	
	return (a + b -1)/b;
}

__global__ void cal(int n, int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height, int* dev_dist){	
	int block_end_x = block_start_x + block_height;
	int block_end_y = block_start_y + block_width;

	for (int b_i = block_start_x+blockIdx.x; b_i < block_end_x; b_i+=gridDim.x) {
		for (int b_j = block_start_y+blockIdx.y; b_j < block_end_y; b_j+=gridDim.y) {
			// To calculate B*B elements in the block (b_i, b_j)
			// For each block, it need to compute B times
			for (int k = Round * B; k < (Round +1) * B && k < n; k++) {
				// To calculate original index of elements in the block (b_i, b_j)
				// For instance, original index of (0,0) in block (1,2) is (2,5) for V=6,B=2D
				int block_internal_start_x = b_i * B;
				int block_internal_end_x   = (b_i +1) * B;
				int block_internal_start_y = b_j * B; 
				int block_internal_end_y   = (b_j +1) * B;

				if (block_internal_end_x > n)	block_internal_end_x = n;
				if (block_internal_end_y > n)	block_internal_end_y = n;

				for (int i = block_internal_start_x+threadIdx.x; i < block_internal_end_x; i+=blockDim.x) {
					for (int j = block_internal_start_y+threadIdx.y; j < block_internal_end_y; j+=blockDim.y) {
						if (dev_dist[i*n+k] + dev_dist[k*n+j] < dev_dist[i*n+j])
							dev_dist[i*n+j] = dev_dist[i*n+k] + dev_dist[k*n+j];
					}
				}
				__syncthreads();
			}
		}
	}
}

int init_device(void){	
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	
	if (deviceCount == 0) {
		printf("No CUDA device found.\n\n");
		return 0;
	} else {
		cudaSetDevice(0);
		return 0;
	}
}

void block_APSP(int B){
	init_device();	
	int round = ceil(n, B);

	cudaMalloc((void**) &dev_dist, sizeof(int)*n*n);
	cudaMemcpy(dev_dist, Dist,  sizeof(int)*n*n, cudaMemcpyHostToDevice);

	dim3 block(round, round), thread(min(B,32), min(B,32));
	//dim3 block(10, 10), thread(10, 10);

	for (int r = 0; r < round; r++) {
		/* Phase 1*/
		cal<<<block, thread>>>(n, B, r, r, r, 1, 1, dev_dist);

		/* Phase 2*/
		cal<<<block, thread>>>(n, B, r, r, 0, r, 1, dev_dist);
		cal<<<block, thread>>>(n, B, r, r, r+1, round-r-1, 1, dev_dist);
		cal<<<block, thread>>>(n, B, r, 0, r, 1, r, dev_dist);
		cal<<<block, thread>>>(n, B, r, r+1, r, 1, round-r-1, dev_dist);

		/* Phase 3*/
		cal<<<block, thread>>>(n, B, r, 0, 0, r, r, dev_dist);
		cal<<<block, thread>>>(n, B, r, 0, r+1, round-r-1, r, dev_dist);
		cal<<<block, thread>>>(n, B, r, r+1, 0, r, round-r-1, dev_dist);
		cal<<<block, thread>>>(n, B, r, r+1, r+1, round-r-1, round-r-1, dev_dist);
	}
	cudaMemcpy(Dist, dev_dist, sizeof(int)*n*n, cudaMemcpyDeviceToHost);
}

int main(int argc, char* argv[]){
	input(argv[1]);

	int B = 64;

	t1 = wallclock();
	block_APSP(B);
	t2 = wallclock();
	//printf("total  time %10.3lf\n", t2-t1);

	output(argv[2]);
	cudaFree(dev_dist);
	return 0;
}

