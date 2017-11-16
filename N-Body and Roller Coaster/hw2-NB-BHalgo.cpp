#include<X11/Xlib.h>
#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<cmath>
#include<unistd.h>
#include<sys/time.h>
#include<pthread.h>
#include<vector>
#include<deque>
#include<algorithm>
#define G 6.673*(1e-11)
#define INF 2147000000
using namespace std;

void draw(double x,double y);
void clear(double x,double y);
void initGraph(int width,int height);
GC gc;
Display *display;
Window window;
int screen;
int enable = 0;

struct XY{
	double x, y;
};
XY* P, *newP, *V, *newV, force;

struct DATA{
	double x1, x2, y1, y2;
	int index;
	vector<XY> point;
};

struct NODE{
	XY center; 
	double d;
	int child[4];
	int childN;
};
NODE* nodes;
deque<DATA> Q;
int nodeIndex, leaf;

int threadN, stepT, dataN, LEN;
double massM, deltaT, theta, xmin, ymin, len;
char xwindow[10];
int allDone = 0;

pthread_barrier_t treeBar, compBar;
pthread_mutex_t treeMutex, leafMutex, compMutex;

struct Segment{
	int first, last, id;
};
Segment* seg;

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

void Initialization(void){
	nodes = new NODE[3*dataN];
	P = new XY[dataN];
	newP = new XY[dataN];
	V = new XY[dataN];
	newV = new XY[dataN];
	seg = new Segment[threadN];
	Q.clear();
	memset(nodes, 0, sizeof(nodes));
	nodeIndex = 1, leaf = 0;
	allDone = 0;
	pthread_barrier_init(&treeBar, NULL, threadN+1);
	pthread_barrier_init(&compBar, NULL, threadN+1);
	pthread_mutex_init(&treeMutex, NULL);
	pthread_mutex_init(&leafMutex, NULL);
	pthread_mutex_init(&compMutex, NULL);
}

double computeDist(XY p1, XY p2){
	return (p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y);
}

XY computeForce(int p, int n){
	double F, D;
	XY tmp;
	tmp.x = tmp.y = 0;
	D = computeDist(P[p], nodes[n].center);
	if(D>0){
		if((nodes[n].d/sqrt(D))<theta || nodes[n].childN==0){	
			F = (G*massM*massM)/D;
			tmp.x = F*(nodes[n].center.x-P[p].x)/sqrt(D);
			tmp.y = F*(nodes[n].center.y-P[p].y)/sqrt(D);
			return tmp;
		}
		else{
			XY sum;
			sum.x = sum.y = 0;
			for(int i=0;i<nodes[n].childN;i++){
				tmp = computeForce(p, nodes[n].child[i]);
				sum.x += tmp.x;
				sum.y += tmp.y;
			}
			return sum;
		}
	}
	return tmp;
}

DATA* createDATA(double xmin, double xmid, double xmax, 
			   double ymin, double ymid, double ymax, vector<XY> &p){
	DATA* d = new DATA[4];
	d[0].x1 = xmin, d[0].x2 = xmid, d[0].y1 = ymid, d[0].y2 = ymax;
	d[1].x1 = xmid, d[1].x2 = xmax, d[1].y1 = ymid, d[1].y2 = ymax;
	d[2].x1 = xmin, d[2].x2 = xmid, d[2].y1 = ymin, d[2].y2 = ymid;
	d[3].x1 = xmid, d[3].x2 = xmax, d[3].y1 = ymin, d[3].y2 = ymid;
	for(int i=0;i<p.size();i++){
		if(p[i].x>=d[0].x1 && p[i].x<=d[0].x2 && p[i].y>=d[0].y1 && p[i].y<=d[0].y2)
			d[0].point.push_back(p[i]);
		else if(p[i].x>=d[1].x1 && p[i].x<=d[1].x2 && p[i].y>=d[1].y1 && p[i].y<=d[1].y2)
			d[1].point.push_back(p[i]);
		else if(p[i].x>=d[2].x1 && p[i].x<=d[2].x2 && p[i].y>=d[2].y1 && p[i].y<=d[2].y2)
			d[2].point.push_back(p[i]);
		else if(p[i].x>=d[3].x1 && p[i].x<=d[3].x2 && p[i].y>=d[3].y1 && p[i].y<=d[3].y2)
			d[3].point.push_back(p[i]);
	}
	for(int i=0;i<4;i++){
		if(d[i].point.size()>0){
			pthread_mutex_lock(&treeMutex);
				d[i].index = nodeIndex;
				nodeIndex++;
				Q.push_back(d[i]);
			pthread_mutex_unlock(&treeMutex);
		}
	}
	return d;
}

NODE createNode(DATA &data){
	NODE tmpN;
	DATA* d = new DATA[4];
	int childN;
	double cx = 0, cy = 0;
	double xmin = data.x1, xmax = data.x2;
	double ymin = data.y1, ymax = data.y2;
	double xmid = (xmax+xmin)/2, ymid = (ymax+ymin)/2;
	for(int i=0;i<data.point.size();i++){
		cx += data.point[i].x;
		cy += data.point[i].y;
	}
	tmpN.center.x = cx/data.point.size();
	tmpN.center.y = cy/data.point.size();
	tmpN.childN = 0;
	if(data.point.size()>1){
		d = createDATA(xmin, xmid, xmax, ymin, ymid, ymax, data.point);
		for(int i=0;i<4;i++){
			if(d[i].point.size()){
				tmpN.child[tmpN.childN++] = d[i].index;
				tmpN.d = d[i].x2-d[i].x1;
			}
		}
	}
	else{
		pthread_mutex_lock(&leafMutex);
			leaf++;
		pthread_mutex_unlock(&leafMutex);
	}
	return tmpN;
}

void buildTree(int id){
	DATA d;
	while(leaf<dataN){
		while(leaf<dataN){
			int get=0;
			pthread_mutex_lock(&treeMutex);
				if(!Q.empty()){
					d = Q.front(), Q.pop_front();
					get=1;
				}
			pthread_mutex_unlock(&treeMutex);			
			if(get) break;
		}
		if(leaf==dataN) break;
		nodes[d.index] = createNode(d);
	}
}

void* Simulation(void* segment){
	int tid = (*(Segment*)segment).id;
	
	while(stepT){
		buildTree(tid);
		pthread_barrier_wait(&treeBar);
		
		for(int i=seg[tid].first;i<seg[tid].last;i++){
			force = computeForce(i, 0);
			newV[i].x = V[i].x+force.x*deltaT/massM;
			newP[i].x = P[i].x+newV[i].x*deltaT;
			newV[i].y = V[i].y+force.y*deltaT/massM;
			newP[i].y = P[i].y+newV[i].y*deltaT;
		}
		
		pthread_mutex_lock(&compMutex);
			if(allDone!=threadN) allDone++;
		pthread_mutex_unlock(&compMutex);
		pthread_barrier_wait(&compBar);
	}
	pthread_exit(NULL);
}

int main(int argc, char* argv[]){
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
	
	DATA root;
	double xmin, xmax, ymin, ymax, diff;
	while(stepT){
		/****build root node start****/
		xmin = INF, xmax = -INF;
		ymin = INF, ymax = -INF;
		for(int i=0;i<dataN;i++){
			xmin = min(P[i].x, xmin);
			xmax = max(P[i].x, xmax);
			ymin = min(P[i].y, ymin);
			ymax = max(P[i].y, ymax);
		}
		diff = abs(((xmax-xmin)-(ymax-ymin))/2);
		if(xmax-xmin > ymax-ymin){
			ymax += diff;
			ymin -= diff;
		}else if (xmax-xmin < ymax-ymin){
			xmax += diff;
			xmin -= diff;
		}
		root.x1 = xmin, root.x2 = xmax;
		root.y1 = ymin, root.y2 = ymax;
		root.index = 0;
		root.point.clear();
		for(int i=0;i<dataN;i++) root.point.push_back(P[i]);		
		nodes[0] = createNode(root);
		/****build root node end****/
		pthread_barrier_wait(&treeBar);

		while(allDone<threadN);		
		for(int i=0;i<dataN;i++){
			if(enable){
				clear(P[i].x, P[i].y);
				draw(newP[i].x, newP[i].y);
			}
			V[i] = newV[i];
			P[i] = newP[i];
			if(enable) XFlush(display);
		}
		allDone = 0;
		Q.clear();
		nodeIndex = 1, leaf = 0;
		stepT--;
		pthread_barrier_wait(&compBar);
	}
	for(int i=0;i<threadN;i++) pthread_join(threads[i], NULL);
	pthread_barrier_destroy(&treeBar);
	pthread_barrier_destroy(&compBar);
	pthread_mutex_destroy(&treeMutex);
	pthread_mutex_destroy(&leafMutex);
	pthread_mutex_destroy(&compMutex);
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



