#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <chrono>
#include <math.h>
#include <cuda_runtime.h>

using namespace std;

//*******************************************

// Write down the kernels here

__device__ __constant__ int gxcoord[1000];
__device__ __constant__ int gycoord[1000];

__global__ void finding_opponents(int *gRound, int T, int *ghp, int *gOpponents){
    if (*gRound % T == 0){
        return;
    }
    int attacker =  blockIdx.x;
    int k = threadIdx.x;
    int x4, y4,my,mx;
    __shared__ int opponent;
    __shared__ int nearest;
    __shared__ int x1, y1, x2, y2, slopex, slopey, x3, y3;

    if (threadIdx.x == 0) {
        opponent = -1;
        nearest = -1;
    }
    __syncthreads();
    if (threadIdx.x == 0 && ghp[attacker] > 0){
        opponent = (attacker + *gRound) % T;

        x1 = gxcoord[attacker];
        y1 = gycoord[attacker];
        x2 = gxcoord[opponent];
        y2 = gycoord[opponent];
        slopey = (y2 - y1);
        slopex = (x2 - x1);

        if ((x2 >= x1 && y2 >= y1)) {
            atomicExch(&x3,INT_MAX);
            atomicExch(&y3,INT_MAX);
        }
        else if ((x2 <= x1 && y2 >= y1)){
            atomicExch(&x3,INT_MIN);
            atomicExch(&y3,INT_MAX);
        }
        else if ((x2 <= x1 && y2 <= y1)) {
            atomicExch(&x3,INT_MIN);
            atomicExch(&y3,INT_MIN);
        }
        else if ((x2 >= x1 && y2 <= y1)){
            atomicExch(&x3,INT_MAX);
            atomicExch(&y3,INT_MIN);
        }
    }

    __syncthreads();
    if (ghp[attacker] > 0){
        
        if (k != attacker && ghp[k] > 0){
            x4 = gxcoord[k];
            y4 = gycoord[k];
            
            my = (y4 - y1);
            mx = (x4 - x1);
            
            if (slopey * mx == slopex * my){
                if ((x2 >= x1 && y2 >= y1) && (x4 >= x1 && y4 >= y1)){
                    if (x1 != x4) atomicMin(&x3, x4);
                    else atomicMin(&y3, y4);
                }
                else if ((x2 <= x1 && y2 >= y1) && (x4 <= x1 && y4 >= y1)){
                    if (x1 != x4) atomicMax(&x3, x4);
                    else atomicMin(&y3, y4);
                }
                else if ((x2 <= x1 && y2 <= y1) && (x4 <= x1 && y4 <= y1)){
                    if (x1 != x4) atomicMax(&x3, x4);
                    else atomicMax(&y3, y4);
                }
                else if ((x2 >= x1 && y2 <= y1) && (x4 >= x1 && y4 <= y1)){
                    if (x1 != x4) atomicMin(&x3, x4);
                    else atomicMax(&y3, y4);
                }
            }
        }
    }
    __syncthreads();
    if (ghp[attacker] > 0){
        if (x1 != x4){
            if (x3 == x4 && slopey * mx == slopex * my) nearest = k;
        }
        else {
            if (y3 == y4 && slopey * mx == slopex * my) nearest = k;
        }
    }



    __syncthreads();
    
    if (threadIdx.x == 0 && ghp[attacker] > 0){
        if (nearest == -1 || ghp[nearest] <= 0) gOpponents[attacker] = -1;
        else gOpponents[attacker] = nearest;
    }
}

__global__ void eval_score(int *gRound, int *ghp, int *gOpponents, int *gscore, int *gcount, int T){
    
    if (threadIdx.x == 0){
        (*gRound)++;
    }
    int attacker = threadIdx.x;
    if (ghp[attacker] <= 0) return;
    __syncthreads();
    int opponent = gOpponents[attacker];
    if ((*gRound - 1) % T != 0 && opponent != -1){
        if(atomicAdd(&ghp[opponent], -1) == 1) atomicAdd(gcount, 1);
        gscore[attacker]++;
    }
    
}

__global__ void setHP(int *ghp, int H){
    int id = threadIdx.x;
    ghp[id] = H;
}


//***********************************************


int main(int argc,char **argv)
{
    // Variable declarations
    int M,N,T,H,*xcoord,*ycoord,*score;

    FILE *inputfilepointer;
    //File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer    = fopen( inputfilename , "r");
    if ( inputfilepointer == NULL )  {
        printf("input.txt file failed to open.");
        return 0; 
    }
    fscanf( inputfilepointer, "%d", &M );
    fscanf( inputfilepointer, "%d", &N );
    fscanf( inputfilepointer, "%d", &T ); // T is number of Tanks
    fscanf( inputfilepointer, "%d", &H ); // H is the starting Health point of each Tank
	
    // Allocate memory on CPU
    xcoord=(int*)malloc(T * sizeof (int));  // X coordinate of each tank
    ycoord=(int*)malloc(T * sizeof (int));  // Y coordinate of each tank
    score=(int*)malloc(T * sizeof (int));  // Score of each tank (ensure that at the end you have copied back the score calculations on the GPU back to this allocation)

    // Get the Input of Tank coordinates
    for(int i=0;i<T;i++)
    {
      fscanf( inputfilepointer, "%d", &xcoord[i] );
      fscanf( inputfilepointer, "%d", &ycoord[i] );
    }
		

    auto start = chrono::high_resolution_clock::now();

    //*********************************
    // Your Code begins here (Do not change anything in main() above this comment)
    //********************************
    int *count = (int *) malloc(sizeof(int));
    *count = 0;

    int *gxcoord_cpy, *gycoord_cpy, *gscore, *ghp, *gOpponents, *gcount, *gRound;
    cudaMalloc(&gxcoord_cpy, T * sizeof(int));
    cudaMalloc(&gycoord_cpy, T * sizeof(int));
    cudaMalloc(&gscore, T * sizeof(int));
    cudaMemset(gscore, 0, T * sizeof(int));
    cudaMalloc(&ghp, T * sizeof(int));
    cudaMalloc(&gOpponents, T * sizeof(int));
    cudaMalloc(&gcount, sizeof(int));
    cudaMemset(gcount, 0, sizeof(int));
    cudaMalloc(&gRound, sizeof(int));
    cudaMemset(gRound, 0, sizeof(int));

    cudaMemcpy(gxcoord_cpy, xcoord, T * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gycoord_cpy, ycoord, T * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(gxcoord, gxcoord_cpy, T * sizeof(int));
    cudaMemcpyToSymbol(gycoord, gycoord_cpy, T * sizeof(int));

    setHP<<<1, T>>>(ghp, H);

    while(*count + 1 != T && *count != T){
        
        finding_opponents<<<T, T>>>(gRound, T, ghp, gOpponents);
        eval_score<<<1, T>>>(gRound, ghp, gOpponents, gscore, gcount, T);
        cudaMemcpy(count, gcount, sizeof(int), cudaMemcpyDeviceToHost);
   
    }
    cudaMemcpy(score, gscore, T * sizeof(int), cudaMemcpyDeviceToHost);


    //*********************************
    // Your Code ends here (Do not change anything in main() below this comment)
    //********************************

    auto end  = chrono::high_resolution_clock::now();

    chrono::duration<double, std::micro> timeTaken = end-start;

    printf("Execution time : %f\n", timeTaken.count());

    // Output
    char *outputfilename = argv[2];
    char *exectimefilename = argv[3]; 
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");

    for(int i=0;i<T;i++)
    {
        fprintf( outputfilepointer, "%d\n", score[i]);
    }
    fclose(inputfilepointer);
    fclose(outputfilepointer);

    outputfilepointer = fopen(exectimefilename,"w");
    fprintf(outputfilepointer,"%f", timeTaken.count());
    fclose(outputfilepointer);

    free(xcoord);
    free(ycoord);
    free(score);
    cudaDeviceSynchronize();
    return 0;
}