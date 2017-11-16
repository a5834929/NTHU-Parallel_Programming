#include<cstdio>
#include<cstdlib>
#include<unistd.h>
#include<sys/time.h>
#include<pthread.h>
#include<deque>
using namespace std;

deque<int> Q;
int passN, rideC, rideT, stepN;
int cnt, *getOff;

struct Passenger{
	int id, inQ;
};
Passenger* passenger;

pthread_mutex_t mutexRun, mutexPrint;
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

void* RollerCoaster(void* thread){
	Passenger* pass = (Passenger*)thread;
	gettimeofday(&start, NULL);
	
	while(cnt<stepN){
		if((*pass).id==0){ // roller coaster
			while(Q.size()<rideC);
			pthread_mutex_lock(&mutexRun);
			for(int i=0;i<rideC;i++){
				int j = Q.front();
				getOff[i] = j;
				Q.pop_front();
			}
			pthread_mutex_unlock(&mutexRun);
			
			pthread_mutex_lock(&mutexPrint);
				printf("Car departure at %lf seconds. Passenger ", curTime());
				for(int i=0;i<rideC;i++) printf("%d ", getOff[i]);
				printf("are in the car.\n");
			pthread_mutex_unlock(&mutexPrint);
			
			usleep(rideT*1000);
			
			pthread_mutex_lock(&mutexPrint);
				printf("Car arrives at %lf seconds. Passenger ", curTime());
				for(int i=0;i<rideC;i++){
					printf("%d ", getOff[i]);
					passenger[getOff[i]].inQ = 0;
				}
				printf("get off the car.\n");
			pthread_mutex_unlock(&mutexPrint);
			
			cnt++;
		}
		else{
			pthread_mutex_lock(&mutexPrint);
				printf("Passenger %d wanders around the park.\n", (*pass).id);
			pthread_mutex_unlock(&mutexPrint);
			
			//usleep(rideT*500);
			usleep((rand()%100+100)*1000);
			
			pthread_mutex_lock(&mutexPrint);
				printf("Passenger %d returns for a ride.\n", (*pass).id);
			pthread_mutex_unlock(&mutexPrint);
			
			pthread_mutex_lock(&mutexRun);
				Q.push_back((*pass).id);
			pthread_mutex_unlock(&mutexRun);
			
			/*double st, et;
			st=curTime();*/
			if(cnt!=stepN)(*pass).inQ = 1;
			while((*pass).inQ);
			/*et=curTime();
			printf("%lf\n", et-st);*/
		}
	}
	if((*pass).id==0){
		for(int i=0;i<passN+1;i++) passenger[i].inQ = 0;
		pthread_mutex_lock(&mutexRun);
			Q.clear();
		pthread_mutex_unlock(&mutexRun);
	}
	pthread_exit(NULL);
}

int main(int argc, char*argv[]){
	//gettimeofday(&start, NULL);
	Q.clear();
	cnt = 0;
	passN = atoi(argv[1]);
	rideC = atoi(argv[2]);
	rideT = atoi(argv[3]);
	stepN = atoi(argv[4]);
	pthread_t threads[passN+1];
	pthread_mutex_init(&mutexRun, NULL);
	pthread_mutex_init(&mutexPrint, NULL);
	getOff = new int[passN];
	passenger = new Passenger[passN+1];
	srand(time(NULL));
	
	for(int i=0;i<passN+1;i++){
		passenger[i].id = i;
		passenger[i].inQ = 0;
		pthread_create(&threads[i], NULL, RollerCoaster, (void*)&passenger[i]);
	}
	for(int i=0;i<passN+1;i++) pthread_join(threads[i], NULL);	
	pthread_mutex_destroy(&mutexRun);
	pthread_mutex_destroy(&mutexPrint);
	pthread_exit(NULL);
	return 0;
}
