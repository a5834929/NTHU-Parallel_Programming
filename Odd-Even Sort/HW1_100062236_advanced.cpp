#include<cstdio>
#include<cstdlib>
#include<algorithm>
#include "mpi.h"
#define ROOT 0
#define INF 2147483647
using namespace std;

double iostart, ioend, IO = 0;
double comstart, comend, COM = 0;
double cpustart, cpuend, CPU = 0;

int cmp(const void*p, const void*q){
	return (*(int*)p-*(int*)q);
}

int SortPhase(int n, int myRank, int nProc, int*seg, int phase){	
	int sorted = 1;
	int *tmp, *merge;
	MPI_Status status;
	
	if(myRank%2==(1-phase) && myRank>(phase-1)){  // send
		tmp = new int[n];

		comstart = MPI_Wtime();
		MPI_Send(seg, n, MPI_INT, myRank-1, 0, MPI_COMM_WORLD);
		comend = MPI_Wtime();
		COM+=comend-comstart;
		
		comstart = MPI_Wtime();
		MPI_Recv(tmp, n, MPI_INT, myRank-1, 0, MPI_COMM_WORLD, &status);
		comend = MPI_Wtime();
		COM+=comend-comstart;
		for(int i=0;i<n;i++){
			if(seg[i]!=tmp[i]) sorted = 0;
			seg[i] = tmp[i];
		}
	}
	
	if(myRank%2==phase && myRank!=nProc-1){ // receive
		tmp = new int[n];
		merge = new int[2*n];
	
		comstart = MPI_Wtime();
		MPI_Recv(tmp, n, MPI_INT, myRank+1, 0, MPI_COMM_WORLD, &status);
		comend = MPI_Wtime();
		COM+=comend-comstart;

		for(int i=0, j=0, k=0;k<2*n;){
			if(i<n && j<n){
				if(seg[i]<tmp[j]){
					merge[k] = seg[i];
					i++;
				}
				else{
					merge[k] = tmp[j];
					j++;
				}
				k++;
			}
			else if(i==n){
				for(;k<2*n;){
					merge[k] = tmp[j];
					k++, j++;
				}
			}
			else{
				for(;k<2*n;){
					merge[k] = seg[i];
					k++, i++;
				}
			}	
		}

		for(int i=0;i<n;i++){
			if(seg[i]!=merge[i]) sorted = 0;
			seg[i] = merge[i];
		}
		comstart = MPI_Wtime();
		MPI_Send(merge+n, n, MPI_INT, myRank+1, 0, MPI_COMM_WORLD);
		comend = MPI_Wtime();
		COM+=comend-comstart;
	}
	return sorted;
}

int main(int argc, char*argv[]){
	int myRank, nProc;
	int n, pn, N, offset, cnt;	
	int sortE, sortO;
	int res, result;
	int *arr, *seg;

	MPI_Init(&argc, &argv);
	cpustart = MPI_Wtime();
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	MPI_Comm_size(MPI_COMM_WORLD, &nProc);
	MPI_File fin, fout;
	MPI_Status status;
	res = result = 0;
	
	N = atoi(argv[1]);
	pn = N;
	if(N%nProc) pn += (nProc-pn%nProc);
	n = pn/nProc;
	offset = myRank*n;
	cnt = n;
	if(offset>N) cnt = 0;
	if(N-offset<n) cnt = N-offset;
	
	seg = new int[n];
	for(int i=0;i<n;i++) seg[i] = INF;

	iostart = MPI_Wtime();
        MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &fin);
        MPI_File_seek(fin, offset*4, MPI_SEEK_CUR);
        MPI_File_read(fin, seg, cnt, MPI_INT, &status);
	ioend = MPI_Wtime();
	IO+=ioend-iostart;
	
	while(!result){
		qsort(seg, n, sizeof(int), cmp);
		sortE = SortPhase(n, myRank, nProc, seg, 0);
		sortO = SortPhase(n, myRank, nProc, seg, 1);
		res = sortE*sortO;
		
		comstart = MPI_Wtime();
		MPI_Allreduce(&res, &result, 1, MPI_INT, MPI_PROD, MPI_COMM_WORLD);
		comend = MPI_Wtime();
		COM+=comend-comstart;
	}

	iostart = MPI_Wtime();
	MPI_File_open(MPI_COMM_WORLD, argv[3], MPI_MODE_WRONLY|MPI_MODE_CREATE, MPI_INFO_NULL, &fout);
        MPI_File_seek(fout, offset*4, MPI_SEEK_CUR);
        MPI_File_write(fout, seg, cnt, MPI_INT, &status);
	ioend = MPI_Wtime();
	IO+=ioend-iostart;

	cpuend = MPI_Wtime();
	CPU+=cpuend-cpustart;
	printf("IO %lf\n", IO);
	printf("COM %lf\n", COM);
	printf("CPU %lf\n", CPU);
	MPI_Finalize();	
	return 0;
}
