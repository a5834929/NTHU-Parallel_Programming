#include "mpi.h"
#include<omp.h>
#include<X11/Xlib.h>
#include<unistd.h>
#include<cstdio>
#include<cstdlib>
#include<cstring>
#define ROOT 0
using namespace std;

double start=0, stop=0;

void draw(int i, int j, int repeats);
void initGraph(int x, int y, int width,int height);
GC gc;
Display *display;
Window window;
int screen;

int threadN, width, height;
double x1, x2, y1, y2;
int enable = 0;

struct Compl{
	double real, imag;
};

int main(int argc, char* argv[]){	
	int rank, size;
	int *reps;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Request req;
	MPI_Status status;

	threadN = atoi(argv[1]);
	sscanf(argv[2], "%lf", &x1);
	sscanf(argv[3], "%lf", &x2);
	sscanf(argv[4], "%lf", &y1);
	sscanf(argv[5], "%lf", &y2);
	width = atoi(argv[6]);
	height = atoi(argv[7]);
	if(strcmp("enable", argv[8])==0) enable = 1;

	start = MPI_Wtime();
	if(rank==ROOT){
		reps = new int[height];
		if(enable)
			initGraph((int)(x1+x2)/2, (int)(y1+y2)/2, width, height);
			
		int pcount = 0, col = 0, recvCol;
		for(int i=1;i<size;i++){
			MPI_Isend(&col, 1, MPI_INT, i, 1, MPI_COMM_WORLD, &req);
			pcount++;
			col++;
		}
		while(pcount){
			MPI_Recv(&recvCol, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
			MPI_Recv(reps, height, MPI_INT, status.MPI_SOURCE, 1, MPI_COMM_WORLD, &status);
			pcount--;
			if(col<width){
				MPI_Isend(&col, 1, MPI_INT, status.MPI_SOURCE, 1, MPI_COMM_WORLD, &req);
				pcount++;
				col++;
			}else
				MPI_Isend(&col, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD, &req);

			if(enable)
				for(int i=0;i<height;i++) draw(recvCol, i, reps[i]);
		}
		if(enable){
			XFlush(display);
			sleep(10);
		}		
	}else{
		reps = new int[height];
		int col;

		MPI_Recv(&col, 1, MPI_INT, ROOT, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		
		while(status.MPI_TAG!=0){
			#pragma omp parallel for num_threads(threadN) schedule(dynamic)
			for(int i=0;i<height;i++){
				start = MPI_Wtime();
				Compl z, c;
				int repeats;
				double temp, lengthsq;
				
				z.real = 0.0;
				z.imag = 0.0;
				c.real = x1+col*(x2-x1)/width;  
				c.imag = y1+i*(y2-y1)/height; 
				repeats = 0;
				lengthsq = 0.0;

				while(repeats < 100000 && lengthsq < 4.0) { 
					temp = z.real*z.real - z.imag*z.imag + c.real;
					z.imag = 2*z.real*z.imag + c.imag;
					z.real = temp;
					lengthsq = z.real*z.real + z.imag*z.imag; 
					repeats++;
				} 
				reps[i] = repeats;
				//reps[i] = rank*threadN+omp_get_thread_num()-1;
				//printf("%d\n", rank*threadN+omp_get_thread_num());
			}
			MPI_Send(&col, 1, MPI_INT, ROOT, 1, MPI_COMM_WORLD);
			MPI_Send(reps, height, MPI_INT, ROOT, 1, MPI_COMM_WORLD);
			MPI_Recv(&col, 1, MPI_INT, ROOT, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		}
	}

	stop = MPI_Wtime();
	//printf("CPU %lf\n", stop-start);
	MPI_Finalize();	
	return 0;
}

void draw(int i, int j, int repeats){
	XSetForeground(display, gc,  1024 * 1024 * ((repeats) % 256));		
	XDrawPoint(display, window, gc, i, j);	
}

void initGraph(int x, int y, int width, int height){
	display = XOpenDisplay(NULL);
	if(display == NULL){
		fprintf(stderr, "cannot open display\n");
		exit(1);
	}
	screen = DefaultScreen(display);

	int border_width = 0;
	window = XCreateSimpleWindow(display, RootWindow(display, screen), x, y, width, height, border_width,
					BlackPixel(display, screen), WhitePixel(display, screen));	

	XGCValues values;
	long valuemask = 0;
	
	gc = XCreateGC(display, window, valuemask, &values);
	XSetForeground(display, gc, BlackPixel (display, screen));
	XSetBackground(display, gc, 0X0000FF00);
	XSetLineAttributes (display, gc, 1, LineSolid, CapRound, JoinRound);
	
	XMapWindow(display, window);
	XSync(display, 0);

	XSetForeground(display,gc,BlackPixel(display,screen));
	XFillRectangle(display,window,gc,0,0,width,height);
	XFlush(display);
}
