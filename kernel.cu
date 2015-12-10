#include <cuda.h>
#include <ctime>
#include <vector>
#include <time.h>
#include <Windows.h>
#include "cuda_runtime.h"
#include <device_functions.h>
#include <iostream>
#include <cstdlib>
#include <cstdio>
using namespace std;


#define FLOATINF FLT_MAX
#define BLOCK_SIZE 16
#define RANGE 996
#define GRAPH_SIZE 1024*2
#define INF (1<<22) 
#define TOTAL_WIDTH GRAPH_SIZE			
#define GEMM GRAPH_SIZE



typedef pair<pair<int, int>, int> SINGLE_PATH;

void CPU_FW(int *G, int *Gpath, int N);
void print_on_screen(int start, int end, const vector<SINGLE_PATH> &path, const int *D, const int N);
bool get_patial_path(int cur, int next, vector<SINGLE_PATH> &path, const int *D, const int *Dpath, const int N);
void get_all_path(const int *D, const int *Dpath, const int N);
void generate_graph(int *G, int N, int range, int density);
void APSP(int * G,int * Gpath,int width, int start);


__global__ void GPU_FW(int * G, int * Gpath,int width, int start);
__global__ void matrixMINPLUS(int * C, int wA, int startCx, int sCy, int sAx, int sAy, int sBx, int sBy, int overwrite, int * C_PATH);
int main(){
		//this is the definitions of parameters
		const int Size = GRAPH_SIZE*GRAPH_SIZE*sizeof(int);
		clock_t start = clock(), diff;

		int *OrigGraph = (int *)malloc(Size);
		int *CPU_G = (int *)malloc(Size);
		int * CPU_PATH= (int *)malloc(Size);


		int * GPU_G= (int *)malloc(Size);
		int *GPU_PATH = (int *)malloc(Size);


		int * device_GPU_G;
		int *device_GPU_PATH;

		cudaMalloc((int **)&device_GPU_G, Size);
		cudaMalloc((int **)&device_GPU_PATH, Size);

		start = clock();

		generate_graph(OrigGraph, GRAPH_SIZE, RANGE, 25);
		for (int i = 0; i<GRAPH_SIZE*GRAPH_SIZE; i++){
			CPU_G[i] = GPU_G[i] = OrigGraph[i];
			CPU_PATH[i] = GPU_PATH[i] = -1;
		}

		diff = clock() - start;
		int msec = diff * 1000 / CLOCKS_PER_SEC;
		printf("for vertices of %d\r\n", GRAPH_SIZE);
		printf("CPU_FW taken %d seconds %d nanoseconds\r\n", msec / 1000, msec % 1000);


		cudaError_t err = cudaMemcpy(device_GPU_G, GPU_G, Size, cudaMemcpyHostToDevice);
		if (err != cudaSuccess){ printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); }
		err = cudaMemcpy(device_GPU_PATH, GPU_PATH, Size, cudaMemcpyHostToDevice);
		if (err != cudaSuccess){ printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); }

		start = clock();


		APSP(device_GPU_G, device_GPU_PATH, 0, TOTAL_WIDTH);

		diff = clock() - start;
		msec = diff * 1000 / CLOCKS_PER_SEC;
		printf("for vertices of %d\r\n", GRAPH_SIZE);
		printf("GPU_FW taken %d seconds %d nanoseconds\r\n", msec / 1000, msec % 1000);

		err = cudaMemcpy(GPU_G, device_GPU_G, Size, cudaMemcpyDeviceToHost);
		if (err != cudaSuccess){ printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); }
		err = cudaMemcpy(GPU_PATH, device_GPU_PATH, Size, cudaMemcpyDeviceToHost);
		if (err != cudaSuccess){ printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); }
		


		get_all_path(GPU_G, GPU_PATH, GRAPH_SIZE);
		free(OrigGraph);
		free(CPU_G);
		free(CPU_PATH);
		free(GPU_G);
		free(GPU_PATH);
	


}

void APSP(int * G, int * Gpath, int start, int width)
{
	int nw = width / 2;
	if (width <= BLOCK_SIZE)
	{
		dim3 blockDim(width, width);
		dim3 gridDim(1, 1);

		GPU_FW << <gridDim, blockDim >> >(G, Gpath, width, start);

	}
	else if (width <=GEMM)
	{
		APSP(G, Gpath, start, nw);
		dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
		dim3 gridD_dim(nw / BLOCK_SIZE, nw / BLOCK_SIZE);
		//  B = AB
		matrixMINPLUS <<< gridD_dim, block_dim >>>(G, nw, start + nw, start, start, start, start + nw, start, 0,Gpath);
		// C = CA			  
		matrixMINPLUS <<< gridD_dim, block_dim >>>(G, nw, start, start + nw, start, start + nw, start, start, 0, Gpath);
		//  D += CB      		  
		matrixMINPLUS <<< gridD_dim, block_dim >>>(G, nw, start + nw, start + nw, start, start + nw, start + nw, start, 1, Gpath);
		// D							  
		APSP(G, Gpath, start + nw, nw);
		// BD			  
		matrixMINPLUS <<< gridD_dim, block_dim >>>(G, nw, start + nw, start, start + nw, start, start + nw, start + nw, 0, Gpath);
		//  C = DC			  
		matrixMINPLUS <<< gridD_dim, block_dim >>>(G, nw, start, start + nw, start + nw, start + nw, start, start + nw, 0, Gpath);
		//  A += BC			  
		matrixMINPLUS <<< gridD_dim, block_dim >>>(G, nw, start, start, start + nw, start, start, start + nw, 1, Gpath);
	}



}


void CPU_FW(int *G, int *Gpath, int N){
	for (int k = 0; k<N; ++k)for (int i = 0; i<N; ++i)for (int j = 0; j<N; ++j){
		int curloc = i*N + j, from = i*N + k, to = k*N + j;
		if (G[curloc]>(G[from] + G[to])){
			G[curloc] = (G[from] + G[to]);
			Gpath[curloc] = k;
		}
	}
}

void print_on_screen(int start, int end, const vector<SINGLE_PATH> &path, const int *in, const int NUMBER){
	cout << "\n The shortest cost path from " << start << " to " << end << ", costs " << in[start*NUMBER + end] << ".\n";
	for (int i = path.size() - 1; i >= 0; --i){
		cout << "From " << path[i].first.first << " to " << path[i].first.second << " costs of " << path[i].second << '\n';
	}
	cout << '\n';
}

bool get_patial_path(int cur, int next, vector<SINGLE_PATH> &path, const int *D, const int *Dpath, const int N){
	int curIdx = cur*N + next;
	if (D[curIdx] >= INF)return false;
	if (Dpath[curIdx] == -1){
		path.push_back(make_pair(make_pair(cur, next), D[curIdx]));
		return true;
	}
	else{
		path.push_back(make_pair(make_pair(Dpath[curIdx], next), D[Dpath[curIdx] * N + next]));
		return get_patial_path(cur, Dpath[curIdx], path, D, Dpath, N);
	}
}

void get_all_path(const int *D, const int *Dpath, const int N){
	int starts = -1, ends = -1;
	vector<SINGLE_PATH> path;
	do{
		path.clear();
		cout << "Enter starting vertex #:";
		cin >> starts;
		cout << "Enter ending vertex(-1=exit) #:";
		cin >> ends;
		if (starts<0 || starts >= N || ends<0 || ends >= N)return;

		if (get_patial_path(starts, ends, path, D, Dpath, N)){
			print_on_screen(starts, ends, path, D, N);

		}
		else{
			cout << "\n No valid path between " << starts << " , and " << ends << '\n';

		}
	} while (1);
}

void generate_graph(int *G, int N, int range, int density){
	int set_range = (100 / density);
	for (int i = 0; i<N; i++){
		for (int j = 0; j<N; j++){
			if (i == j){
				G[i*N + j] = 0;
				continue;
			}
			int pr = rand() % set_range;
			G[i*N + j] = pr == 0 ? ((rand() % range) + 1) : INF;
		}
	}
}
__global__ void GPU_FW(int * G, int * Gpath,int width, int start){
	//G will be the adjacency matrix, P will be path matrix
	int tx = threadIdx.x;
	int ty = threadIdx.y;


	int offset = start * TOTAL_WIDTH + start;	// memory offset
	for (int k = 0; k < width; ++k)
	{
		int MUL_A = G[offset + ty*TOTAL_WIDTH + k];		// kth column
		int MUL_B = G[offset + k*TOTAL_WIDTH + tx];		// kth row

		G[offset + ty*TOTAL_WIDTH + tx] = fminf(MUL_A + MUL_B, G[offset + ty*TOTAL_WIDTH + tx]);
		if ((G[offset + ty*TOTAL_WIDTH + tx] == MUL_A + MUL_B)&&(k!=ty&&k!=tx))
		{
			Gpath[offset + ty*TOTAL_WIDTH + tx] = k;

		}
		__syncthreads();
	}



}
__global__ void matrixMINPLUS(int * C, int wA, int startCx, int sCy, int sAx, int sAy, int sBx, int sBy, int overwrite,int * C_PATH)
{
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int sa = sAy * TOTAL_WIDTH + sAx;
	int sb = sBy * TOTAL_WIDTH + sBx;
	int sc = sCy * TOTAL_WIDTH + startCx;

	int ba = TOTAL_WIDTH *  BLOCK_SIZE * by;
	int bb = BLOCK_SIZE * bx;		
	__shared__ int current_vertice[BLOCK_SIZE*BLOCK_SIZE];
	current_vertice[ty*BLOCK_SIZE + tx] = -1;
	int min = FLOATINF;

	int number = wA / BLOCK_SIZE;	// number of blocks in one dimension


	for (int m = 0; m < number; ++m)
	{
		__shared__ int in_A[BLOCK_SIZE * BLOCK_SIZE];
		__shared__ int in_B[BLOCK_SIZE * BLOCK_SIZE];
		in_A[ty*BLOCK_SIZE + tx] = C[sa + ba + m * BLOCK_SIZE  + tx + ty*TOTAL_WIDTH];
		in_B[ty*BLOCK_SIZE + tx] = C[sb + bb + m * BLOCK_SIZE* TOTAL_WIDTH + tx + ty*TOTAL_WIDTH];
		__syncthreads();


		for (int k = 0; k < BLOCK_SIZE; ++k)
		{
			int a = in_A[ty * BLOCK_SIZE + k];	//A is to k, so ty determines the start
			int b = in_B[k * BLOCK_SIZE + tx];	//B is to from, so tx determines the end
			min = fminf(a + b, min);
			if (min==a+b)
				current_vertice[ty*BLOCK_SIZE + tx] = sAx + m*BLOCK_SIZE + k;
		}
		__syncthreads();
	}


	if (overwrite)
	{

		C[sc + ba + bb + ty * TOTAL_WIDTH + tx] = fminf(C[sc + ba + bb + ty * TOTAL_WIDTH + tx], min);

	}
	else
	{
		C[sc + ba + bb + ty * TOTAL_WIDTH + tx] = min;

	}		
	if ((C[sc + ba + bb + ty * TOTAL_WIDTH + tx] == min) && (current_vertice[ty*BLOCK_SIZE + tx] !=sAx+ BLOCK_SIZE *bx + tx) && (current_vertice[ty*BLOCK_SIZE + tx] != sBy+BLOCK_SIZE*by + ty))
		C_PATH[sc + ba + bb + ty * TOTAL_WIDTH + tx] = current_vertice[ty*BLOCK_SIZE + tx];
}


