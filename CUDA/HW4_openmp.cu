#include <cuda_runtime.h>
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <time.h>
#include <sys/time.h>
#include <algorithm>

#define INF 1000000

double t1, t2;
int n, m;	// Number of vertices, edges
int *Dist;
int *dev_dist[2];

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
			for (int k = Round * B; k < (Round +1) * B && k < n; k++) {
			
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

void block_APSP(int B){
	omp_set_num_threads(2);

	#pragma omp parallel
	{
		int gpu_id = omp_get_thread_num();
		cudaSetDevice(gpu_id);
		cudaMalloc((void**) &dev_dist[gpu_id], sizeof(int)*n*n);
		cudaMemcpy(dev_dist[gpu_id], Dist,  sizeof(int)*n*n, cudaMemcpyHostToDevice);
	}

	int round = ceil(n, B);
	dim3 block(round, round), thread(min(B,32), min(B,32));
	//dim3 block(10, 10), thread(10, 10);

	for (int r = 0; r < round; r++) {

		#pragma omp parallel
		{
			int gpu_id = omp_get_thread_num();
			cudaSetDevice(gpu_id);
			/* Phase 1*/
			cal<<<block, thread>>>(n, B, r, r, r, 1, 1, dev_dist[gpu_id]);
		
			if(gpu_id==0){
				/* Phase 2*/
				cal<<<block, thread>>>(n, B, r, r, 0, r, 1, dev_dist[gpu_id]);			//left
				cal<<<block, thread>>>(n, B, r, r, r+1, round-r-1, 1, dev_dist[gpu_id]);//right
				cal<<<block, thread>>>(n, B, r, 0, r, 1, r, dev_dist[gpu_id]);			//top
				/*Phase 3*/
				cal<<<block, thread>>>(n, B, r, 0, 0, r, r, dev_dist[gpu_id]);						//left-top
				cal<<<block, thread>>>(n, B, r, 0, r+1, round-r-1, r, dev_dist[gpu_id]);			//right-top

				cudaMemcpy(&Dist[0], &dev_dist[0][0], sizeof(int)*r*B*n, cudaMemcpyDeviceToHost);
				#pragma omp barrier
				cudaMemcpy(&dev_dist[0][r*B*n], &Dist[r*B*n],  sizeof(int)*(n-r*B)*n, cudaMemcpyHostToDevice);
			}else{
				/*Phase 2*/
				cal<<<block, thread>>>(n, B, r, r, 0, r, 1, dev_dist[gpu_id]);			//left
				cal<<<block, thread>>>(n, B, r, r, r+1, round-r-1, 1, dev_dist[gpu_id]);//right
				cal<<<block, thread>>>(n, B, r, r+1, r, 1, round-r-1, dev_dist[gpu_id]);//bottom
				/*Phase 3*/
				cal<<<block, thread>>>(n, B, r, r+1, 0, r, round-r-1, dev_dist[gpu_id]);			//left-bottom
				cal<<<block, thread>>>(n, B, r, r+1, r+1, round-r-1, round-r-1, dev_dist[gpu_id]);	//right-bottom

				cudaMemcpy(&Dist[r*B*n], &dev_dist[1][r*B*n], sizeof(int)*(n-r*B)*n, cudaMemcpyDeviceToHost);
				#pragma omp barrier
				cudaMemcpy(&dev_dist[1][0], Dist,  sizeof(int)*r*B*n, cudaMemcpyHostToDevice);
			}
		}
	}
	cudaSetDevice(0);
	cudaMemcpy(Dist, &dev_dist[0][0], sizeof(int)*n*n, cudaMemcpyDeviceToHost);
}

int main(int argc, char* argv[]){
	input(argv[1]);

	int B = 64;
	t1 = wallclock();
	block_APSP(B);
	t2 = wallclock();
	//printf("total     time %10.3lf\n", t2-t1);

	output(argv[2]);
	cudaFree(dev_dist);
	return 0;
}

