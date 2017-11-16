#include<X11/Xlib.h>
#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<cmath>
#include<unistd.h>
#include<sys/time.h>
#include<pthread.h>
#define G 6.673*(1e-11)
using namespace std;

pthread_mutex_t mutexComp;

void draw(double x,double y);
void clear(double x,double y);
void initGraph(int width,int height);
GC gc;
Display *display;
Window window;
int screen;
int enable = 0;

int threadN, stepT, dataN, LEN;
double massM, deltaT, theta, xmin, ymin, len;
char xwindow[10];

int allDone = 0, cnt = 0;

struct timeval start;
double curTime(void){
	struct timeval now;
	double mtime, seconds, useconds;
	gettimeofday(&now, NULL);

	seconds = now.tv_sec - start.tv_sec;
	useconds = now.tv_usec - start.tv_usec;
	mtime = ((seconds) * 1000 + useconds/1000.0);
	return mtime/1000;
}

struct XY{
	double x, y;
};
XY* P, *newP, *V, *newV, *force;

struct Segment{
	int first, last, id;
};
Segment* seg;

double computeDist(int p1, int p2){
	return (P[p1].x-P[p2].x)*(P[p1].x-P[p2].x)+(P[p1].y-P[p2].y)*(P[p1].y-P[p2].y);
}

XY computeForce(int p){
	double F, D;
	XY tmp;
	tmp.x = tmp.y = 0;
	for(int i=0;i<dataN;i++){
		if(i!=p){
			D = computeDist(p, i);
			if(D>0){
				F = (G*massM*massM)/D;
				tmp.x += F*(P[i].x-P[p].x)/sqrt(D);
				tmp.y += F*(P[i].y-P[p].y)/sqrt(D);
			}
		}
	}
	return tmp;
}

void Initialization(void){
	P = new XY[dataN];
	newP = new XY[dataN];
	V = new XY[dataN];
	newV = new XY[dataN];
	force = new XY[dataN];
	seg = new Segment[threadN];
	pthread_mutex_init(&mutexComp, NULL);
}

void* Simulation(void* segment){
	int tid = (*(Segment*)segment).id;
	
	while(cnt<stepT){
		for(int i=seg[tid].first;i<seg[tid].last;i++){
			force[i] = computeForce(i);
			newV[i].x = V[i].x+force[i].x*deltaT/massM;
			newP[i].x = P[i].x+newV[i].x*deltaT;
			newV[i].y = V[i].y+force[i].y*deltaT/massM;
			newP[i].y = P[i].y+newV[i].y*deltaT;
		}
		
		pthread_mutex_lock(&mutexComp);
			allDone++;
		pthread_mutex_unlock(&mutexComp);
		
		while(allDone<threadN);
		if(tid==0){
			for(int i=0;i<dataN;i++){
				if(enable){
					clear(P[i].x, P[i].y);
					draw(newP[i].x, newP[i].y);
				}
				V[i] = newV[i];
				P[i] = newP[i];
				if(enable) XFlush(display);
			}
			cnt++;
			allDone = 0;
		}
		while(allDone);
	}
	pthread_exit(NULL);
}

int main(int argc, char*argv[]){
	gettimeofday(&start, NULL);
	threadN = atoi(argv[1]);
	sscanf(argv[2], "%lf", &massM);
	stepT = atoi(argv[3]);
	sscanf(argv[4], "%lf", &deltaT);
	FILE* fp = fopen(argv[5], "r");
	sscanf(argv[6], "%lf", &theta);
	strcpy(xwindow, argv[7]);
	if(xwindow[0]=='e'){
		enable = 1;
		sscanf(argv[8], "%lf", &xmin);
		sscanf(argv[9], "%lf", &ymin);
		sscanf(argv[10], "%lf", &len);
		LEN = atoi(argv[11]);
	}
	
	if(enable) initGraph(LEN, LEN);
	fscanf(fp, "%d", &dataN);
	Initialization();
	
	for(int i=0;i<dataN;i++){
		fscanf(fp, "%lf%lf", &P[i].x, &P[i].y);
		fscanf(fp, "%lf%lf", &V[i].x, &V[i].y);
		newP[i].x = newP[i].y = newV[i].x = newV[i].y = 0;
		force[i].x = force[i].y = 0;
	}
	
	int divide = dataN/threadN;
	int remain = dataN%threadN;
	int index = 0;
	
	pthread_t threads[threadN];
	for(int i=0;i<threadN;i++){
		seg[i].id = i;
		seg[i].first = index;
		seg[i].last = index+divide;
		index+=divide;
		if(i<remain){
			seg[i].last++;
			index++;
		}
		pthread_create(&threads[i], NULL, Simulation, (void*)&seg[i]);
	}
	
	for(int i=0;i<threadN;i++) pthread_join(threads[i], NULL);
	pthread_mutex_destroy(&mutexComp);
	pthread_exit(NULL);
	return 0;
}

void draw(double x,double y){
	double X=(x-xmin)*LEN/len;
	double Y=(y-ymin)*LEN/len;
	XSetForeground(display,gc,WhitePixel(display,screen));
	XDrawPoint(display, window, gc, X, Y);
}

void clear(double x,double y){
	double X=(x-xmin)*LEN/len;
	double Y=(y-ymin)*LEN/len;
	XSetForeground(display,gc,BlackPixel(display,screen));
	XDrawPoint(display, window, gc, X, Y);
}

void initGraph(int width,int height){
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
	window = XCreateSimpleWindow(display, RootWindow(display, screen), 0, 0, width, height, border_width, BlackPixel(display, screen), WhitePixel(display, screen));
	
	/* create graph */
	XGCValues values;
	long valuemask = 0;
	
	gc = XCreateGC(display, window, valuemask, &values);
	//XSetBackground (display, gc, WhitePixel (display, screen));
	XSetForeground (display, gc, BlackPixel (display, screen));
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



