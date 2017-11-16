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
	int *I, *reps, *result;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

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
		I = new int[size];
		if(enable)
			initGraph((int)(x1+x2)/2, (int)(y1+y2)/2, width, height);
	}
	
	int chunk, cnt=0;
	(width%size==0)?(chunk=width/size):(chunk=(width+(size-width%size))/size);
	reps = new int[height*chunk];
	if(rank==ROOT) result = new int[chunk*size*height];
	
	#pragma omp parallel for num_threads(threadN)
	for(int i=rank*chunk;i<rank*chunk+chunk;i++){
		start = MPI_Wtime();
		for(int j=0;j<height && i<width;j++){
			Compl z, c;
			int repeats;
			double temp, lengthsq;
			
			z.real = 0.0;
			z.imag = 0.0;
			c.real = x1+i*(x2-x1)/width;  
			c.imag = y1+j*(y2-y1)/height; 
			repeats = 0;
			lengthsq = 0.0;

			while(i<width && repeats < 100000 && lengthsq < 4.0) { 
				temp = z.real*z.real - z.imag*z.imag + c.real;
				z.imag = 2*z.real*z.imag + c.imag;
				z.real = temp;
				lengthsq = z.real*z.real + z.imag*z.imag; 
				repeats++;
			} 
			reps[height*(i-rank*chunk)+j] = repeats;
			//reps[height*(i-rank*chunk)+j] = rank*threadN+omp_get_thread_num();
			
		}
		
	}
	MPI_Gather(reps, height*chunk, MPI_INT, result, height*chunk, MPI_INT, ROOT, MPI_COMM_WORLD);
	
	if(enable && rank==ROOT){
		int cnt = 0;
		for(int i=0;i<width;i++)
			for(int j=0;j<height;j++)
				draw(i, j, result[cnt++]);
		XFlush(display);
		sleep(5);
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
