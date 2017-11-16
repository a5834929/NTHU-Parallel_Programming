#include<X11/Xlib.h>
#include<unistd.h>
#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<omp.h>
#include<sys/time.h>
using namespace std;

double curTime(struct timeval start){
	struct timeval now;
	double mtime, seconds, useconds;
	gettimeofday(&now, NULL);

	seconds = now.tv_sec - start.tv_sec;
	useconds = now.tv_usec - start.tv_usec;
	mtime = ((seconds) * 1000 + useconds/1000.0);
	return mtime/1000;
}

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
	

	threadN = atoi(argv[1]);
	sscanf(argv[2], "%lf", &x1);
	sscanf(argv[3], "%lf", &x2);
	sscanf(argv[4], "%lf", &y1);
	sscanf(argv[5], "%lf", &y2);
	width = atoi(argv[6]);
	height = atoi(argv[7]);
	if(strcmp("enable", argv[8])==0) enable = 1;

	if(enable)initGraph((int)(x1+x2)/2, (int)(y1+y2)/2, width, height);

	#pragma omp parallel for num_threads(threadN)
	for(int i=0; i<width; i++) {
		struct timeval start;
		gettimeofday(&start, NULL);

		for(int j=0; j<height; j++) {
			Compl z, c;
			int repeats;
			double temp, lengthsq;
			
			z.real = 0.0;
			z.imag = 0.0;
			c.real = x1+i*(x2-x1)/width;  
			c.imag = y1+j*(y2-y1)/height; 
			repeats = 0;
			lengthsq = 0.0;

			while(repeats < 100000 && lengthsq < 4.0) { 
				temp = z.real*z.real - z.imag*z.imag + c.real;
				z.imag = 2*z.real*z.imag + c.imag;
				z.real = temp;
				lengthsq = z.real*z.real + z.imag*z.imag; 
				repeats++;
			} 
			if(enable){
				#pragma omp critical
				draw(i, j, repeats);
				//draw(i, j, omp_get_thread_num());
			}
			//printf("%d\n", omp_get_thread_num());
		}
		
	}
	if(enable){
		XFlush(display);
		sleep(5);
	}

	//printf("CPU %lf\n", curTime());
	return 0;
}

void draw(int i, int j, int repeats){
	XSetForeground(display, gc,  1024 * 1024 * ((repeats) % 256));		
	XDrawPoint(display, window, gc, i, j);	
}

void initGraph(int x, int y, int width, int height){
	/* open connection with the server */ 
	display = XOpenDisplay(NULL);
	if(display == NULL) {
		fprintf(stderr, "cannot open display\n");
		exit(1);
	}
	screen = DefaultScreen(display);

	/* border width in pixels */
	int border_width = 0;

	/* create window */
	window = XCreateSimpleWindow(display, RootWindow(display, screen), x, y, width, height, border_width,
					BlackPixel(display, screen), WhitePixel(display, screen));	

	/* create graph */
	XGCValues values;
	long valuemask = 0;
	
	gc = XCreateGC(display, window, valuemask, &values);
	XSetForeground(display, gc, BlackPixel (display, screen));
	XSetBackground(display, gc, 0X0000FF00);
	XSetLineAttributes (display, gc, 1, LineSolid, CapRound, JoinRound);
	
	/* map(show) the window */
	XMapWindow(display, window);
	XSync(display, 0);

	/* draw rectangle */
	XSetForeground(display,gc,BlackPixel(display,screen));
	XFillRectangle(display,window,gc,0,0,width,height);
	XFlush(display);
}
