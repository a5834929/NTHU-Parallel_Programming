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

int SortPhase(int n, int myRank, int nProc, int*seg, int phase){
	int first = n*myRank;
	int last = n*myRank+n;
	int sorted = 1, s = 0;
	int flag1=0, flag2=0, recv;
	MPI_Request req1, req2;
	MPI_Status status;
	
	if(first%2==(1-phase)){ // need to recv
		s = 1;
		if(myRank!=0){
			flag1 = 1;
			comstart = MPI_Wtime();
			MPI_Irecv(&recv, 1, MPI_INT, myRank-1, myRank, MPI_COMM_WORLD, &req1);	
			comend = MPI_Wtime();
			COM+=comend-comstart;
			//printf("COM %lf\n", comend-comstart);
		}
	}

	if((last-1)%2==phase && myRank!=nProc-1){ // need to send
		flag2 = 1;
		comstart = MPI_Wtime();
		MPI_Isend(&seg[n-1], 1, MPI_INT, myRank+1, myRank+1, MPI_COMM_WORLD, &req2);
		comend = MPI_Wtime();
		COM+=comend-comstart;
		//printf("COM %lf\n", comend-comstart);
	}
	
	for(int i=s;i<n-1;i+=2){ // need to swap
		if(seg[i]>seg[i+1]){
			swap(seg[i], seg[i+1]);
			sorted = 0;
		}		
	}

	if(flag1==1){
		comstart = MPI_Wtime();
		MPI_Wait(&req1, &status);
		comend = MPI_Wtime();
		COM+=comend-comstart;
		//printf("COM %lf\n", comend-comstart);
		if(recv>seg[0]){
			swap(recv, seg[0]);
			sorted = 0;	
		}
		comstart = MPI_Wtime();
		MPI_Isend(&recv, 1, MPI_INT, myRank-1, myRank-1, MPI_COMM_WORLD, &req2);
		comend = MPI_Wtime();
		COM+=comend-comstart;
		//printf("COM %lf\n", comend-comstart);
	}
	if(flag2==1){
		comstart = MPI_Wtime();
		MPI_Irecv(&seg[n-1], 1, MPI_INT, myRank+1, myRank, MPI_COMM_WORLD, &req1);
		MPI_Wait(&req1, &status);
		comend = MPI_Wtime();
		COM+=comend-comstart;
		//printf("COM %lf\n", comend-comstart);
	}
	return sorted;
}

int main(int argc, char*argv[]){
	int myRank, nProc, active;
	int N, pn, n, offset, cnt;
	int *arr, *seg;
	int sortE, sortO;
	int res, result;
	
	MPI_Init(&argc, &argv);
	cpustart = MPI_Wtime();
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	MPI_Comm_size(MPI_COMM_WORLD, &nProc);
	MPI_File fin, fout;
	MPI_Status status;
	res = result = 0;
	
	N = atoi(argv[1]);
	pn = N;
	if(N%nProc) pn += (nProc-N%nProc);
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
	//printf("IO %lf\n", ioend-iostart);
	
	while(!result){
		sortE = SortPhase(n, myRank, nProc, seg, 0);
		sortO = SortPhase(n, myRank, nProc, seg, 1);
		res = sortE*sortO;
	
		comstart = MPI_Wtime();
		MPI_Allreduce(&res, &result, 1, MPI_INT, MPI_PROD, MPI_COMM_WORLD);
		comend = MPI_Wtime();
		COM+=comend-comstart;
		//printf("COM %lf\n", comend-comstart);
	}
	iostart = MPI_Wtime();
	MPI_File_open(MPI_COMM_WORLD, argv[3], MPI_MODE_WRONLY|MPI_MODE_CREATE, MPI_INFO_NULL, &fout);
	MPI_File_seek(fout, offset*4, MPI_SEEK_CUR);
	MPI_File_write(fout, seg, cnt, MPI_INT, &status);
	ioend = MPI_Wtime();
	IO+=ioend-iostart;
	//printf("IO %lf\n", ioend-iostart);	
	
	cpuend = MPI_Wtime();
	CPU+=cpuend-cpustart;
	printf("IO %lf\n", IO);
	printf("COM %lf\n", COM);
	printf("CPU %lf\n", CPU);
	//printf("CPU %lf\n", cpuend-cpustart);	
	MPI_Finalize();
	return 0;
}

