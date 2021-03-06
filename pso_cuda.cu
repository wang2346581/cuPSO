#include <stdio.h>
#include <time.h>
#include <math.h>
#include <curand_kernel.h>
#include "common.h"
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

// cuda prototype
__host__ __device__ double fit(double x);
__global__ void init(double *position_d, double *velocity_d, double *fitness_d, 
    double *pbest_pos_d, double *pbest_fit_d, volatile particle *gbest, int *lock);
__global__ void reductionInOneBlock(volatile particle *gbest, 
        double *aux, double *aux_pos, int block_size);
__global__ void move(double *position_d, double *velocity_d, double *fitness_d, 
    double *pbest_pos_d, double *pbest_fit_d, volatile particle *gbest, 
    int *lock, double *aux, double *aux_pos);

//cuda constant memory
__constant__ double w_d;
__constant__ double c1_d;
__constant__ double c2_d;
__constant__ double max_pos_d;
__constant__ double min_pos_d;
__constant__ double max_v_d;
__constant__ int max_iter_d;
__constant__ int particle_cnt_d;
__constant__ int tile_size;

//cuda function
__global__ void init(double *position_d, double *velocity_d, double *fitness_d, 
    double *pbest_pos_d, double *pbest_fit_d, volatile particle *gbest, int *lock){
    const double pos_range = max_pos_d - min_pos_d; // 解寬度
    int idx =  blockIdx.x * blockDim.x + threadIdx.x;
    int tidx = threadIdx.x;
    extern __shared__ double sharedMemory[];
    double *privateBest = (double *)sharedMemory;
    double *privateBestPos = (double *)&sharedMemory[tile_size];
    //printf("idx is %d \n", idx);
    curandState state1, state2;
    curand_init((unsigned long long)clock() + idx, 0, 0, &state1);
    curand_init((unsigned long long)clock() + idx, 0, 0, &state2);

    if(idx < particle_cnt_d){
        privateBestPos[tidx] = curand_uniform_double(&state1) * pos_range + min_pos_d;
        velocity_d[idx] = curand_uniform_double(&state2) * max_v_d;
        pbest_pos_d[idx] = position_d[idx] = privateBestPos[tidx];
        privateBest[tidx] = fit(position_d[idx]);
        pbest_fit_d[idx] = fitness_d[idx] = privateBest[tidx];
    }
    for(unsigned int stride = blockDim.x/2; stride >=1; stride >>=1){
        __syncthreads();
        if(tidx < stride){
            if(privateBest[tidx] < privateBest[tidx+stride]){
                privateBest[tidx]    = privateBest[tidx+stride];
                privateBestPos[tidx] = privateBestPos[tidx+stride];
            }
        }
    }
    __syncthreads();
    if(tidx == 0){
        while(atomicCAS(lock, 0, 1) != 0);
        if(privateBest[0] > gbest->fitness){
            gbest->position = privateBestPos[0];
            gbest->fitness  = privateBest[0];
        }
        atomicExch(lock, 0);
    }
}

__global__ void findBestInOneBlock(volatile particle *gbest, double *aux, double *aux_pos, int block_size){
    int idx =  blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ double privateBest[1024];
    __shared__ double privateBestPos[1024];
    while(idx < block_size){
        privateBestPos[idx] = aux_pos[idx];
        privateBest[idx] = aux[idx];
        for(unsigned int stride = blockDim.x/2; stride >= 1; stride >>= 1){
            __syncthreads();
            if(idx < stride && idx+stride < block_size){
                if(privateBest[idx] < privateBest[idx+stride]){
                    privateBestPos[idx] = privateBestPos[idx+stride];
                    privateBest[idx] = privateBest[idx+stride];
                }
            }
        }
        if(threadIdx.x == 0){
            if(privateBest[0] > gbest->fitness){
                gbest->position = privateBestPos[0];
                gbest->fitness  = privateBest[0];
                __threadfence_block();
            }
        }
        idx += blockDim.x;
    }
}

__global__ void move(double *position_d, double *velocity_d, double *fitness_d, 
    double *pbest_pos_d, double *pbest_fit_d, volatile particle *gbest, 
    int *lock, double *aux, double *aux_pos){
    int idx =  blockIdx.x * blockDim.x + threadIdx.x;
    int tidx = threadIdx.x;
    extern __shared__ double sharedMemory[];
    double *privateBest    = (double *)sharedMemory;
    double *privateBestPos = (double *)&sharedMemory[tile_size];
    double v    = velocity_d[idx];
    double pos  = position_d[idx];
    double ppos = pbest_pos_d[idx];
    double fitness  = fitness_d[idx];
    double pfitness = pbest_fit_d[idx];
    curandState state1, state2;
    curand_init((unsigned long long)clock() + idx, 0, 0, &state1);
    curand_init((unsigned long long)clock() + idx, 0, 0, &state2);

    if(idx < particle_cnt_d){
        v = w_d * v + c1_d * curand_uniform_double(&state1) * (ppos - pos) 
            + c2_d * curand_uniform_double(&state2) * (gbest->position - pos);
        if(v < -max_v_d)
            v = -max_v_d;
        else if(v > max_v_d)
            v = max_v_d;
        pos = pos + v;
        if(pos > max_pos_d)
            pos = max_pos_d; // 限制最大位置
        else if(pos < min_pos_d)
            pos = min_pos_d; // 限制最小位置
        fitness = fit(pos);
        privateBestPos[tidx] = pos;
        privateBest[tidx]    = fitness;
        if(fitness > pfitness){
            //ppos = pos;
            //pfitness = fitness;
            pbest_pos_d[idx] = pos;
            pbest_fit_d[idx] = fitness;
        }
    }
    for(unsigned int stride = blockDim.x/2; stride >= 1; stride >>= 1){
        __syncthreads();
        if(tidx < stride){
            if(privateBest[tidx] < privateBest[tidx+stride]){
                privateBestPos[tidx] = privateBestPos[tidx+stride];
                privateBest[tidx]    = privateBest[tidx+stride];
            }
        }
    }
    __syncthreads();
    // if(tidx == 0){
    //     while(atomicCAS(lock, 0, 1) != 0);
    //     if(privateBest[0] > gbest->fitness){
    //         gbest->position = privateBestPos[0];
    //         gbest->fitness  = privateBest[0];
    //         __threadfence();
    //     }
    //     atomicExch(lock, 0);
    // }
    if(idx < particle_cnt_d){
        if(tidx==0){
            aux[blockIdx.x] = privateBest[tidx];
            aux_pos[blockIdx.x] = privateBestPos[tidx];
        }
        position_d[idx] = pos;
        velocity_d[idx] = v;
        fitness_d[idx]  = fitness;
    }
}

__host__ __device__ double fit(double x){
    // x**3 - 0.8x**2 - 1000x + 8000
    return fabs(8000.0 + x*(-10000.0+x*(-0.8+x)));
}

int main(int argc, char **argv){
    arguments args = {100000, 4096, 1024, 4, 0};
    int retError = pargeArgs(&args, argc, argv);
    cudaEvent_t start, stop, start_init, stop_init;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventCreate(&start_init));
    HANDLE_ERROR(cudaEventCreate(&stop_init));
    float exe_time; //double init_time;
    // 變數宣告
    clock_t begin_app  = clock();
    clock_t begin_init = begin_app;
    particle_Coal *p; // p : 粒子群

    double *position_d;
    double *velocity_d;
    double *fitness_d;
    double *pbest_pos_d;
    double *pbest_fit_d;
    particle *gbest_d;
    double *aux, *aux_pos;
    int *lock_d; // block level lock for gbest
    int block_size = min(1024, args.blocks_per_grid);

    // 設定參數
    min_pos = -100.0 , max_pos = +100.0;  // 位置限制, 即解空間限制
    w = 1, c1 = 2.0, c2 = 2.0;            // 慣性權重與加速常數設定
    particle_cnt = args.particle_cnt;     // 設粒子個數
    max_v = (max_pos-min_pos) * 1.0;      // 設最大速限

    p = (particle_Coal*) malloc(sizeof(particle_Coal));
    //HANDLE_ERROR(cudaMallocManaged((void **)&gbest_u, sizeof(particle)));

    //p->position  = (double *) malloc(sizeof(double)* particle_cnt);
    //p->velocity  = (double *) malloc(sizeof(double)* particle_cnt); 
    //p->fitness   = (double *) malloc(sizeof(double)* particle_cnt); 
    //p->pbest_pos = (double *) malloc(sizeof(double)* particle_cnt); 
    //p->pbest_fit = (double *) malloc(sizeof(double)* particle_cnt);
    ParticleInitCoal(p); // 粒子初始化

    printf("Allocating device memory\n");
    //HANDLE_ERROR(cudaMalloc((void **)&p_d, sizeof(particle_Coal)));
    HANDLE_ERROR(cudaMalloc((void **)&position_d, sizeof(double)* particle_cnt));
    HANDLE_ERROR(cudaMalloc((void **)&velocity_d, sizeof(double)* particle_cnt));
    HANDLE_ERROR(cudaMalloc((void **)&fitness_d, sizeof(double)* particle_cnt));
    HANDLE_ERROR(cudaMalloc((void **)&pbest_pos_d, sizeof(double)* particle_cnt));
    HANDLE_ERROR(cudaMalloc((void **)&pbest_fit_d, sizeof(double)* particle_cnt));
    HANDLE_ERROR(cudaMalloc((void **)&gbest_d, sizeof(particle)));
    HANDLE_ERROR(cudaMalloc((void**)&lock_d, sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void **)&aux, sizeof(double)* args.blocks_per_grid));
    HANDLE_ERROR(cudaMalloc((void **)&aux_pos, sizeof(double)* args.blocks_per_grid));
    printf("Copying to device\n");
    HANDLE_ERROR(cudaMemcpy(position_d, p->position, sizeof(double)* particle_cnt, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(velocity_d, p->velocity, sizeof(double)* particle_cnt, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(fitness_d, p->fitness, sizeof(double)* particle_cnt, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(pbest_pos_d, p->pbest_pos, sizeof(double)* particle_cnt, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(pbest_fit_d, p->pbest_fit, sizeof(double)* particle_cnt, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(gbest_d, &gbest, sizeof(particle), cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMemcpyToSymbol(w_d, &w, sizeof(double)));
    HANDLE_ERROR(cudaMemcpyToSymbol(c1_d, &c1, sizeof(double)));
    HANDLE_ERROR(cudaMemcpyToSymbol(c2_d, &c2, sizeof(double)));
    HANDLE_ERROR(cudaMemcpyToSymbol(max_pos_d, &max_pos, sizeof(double)));
    HANDLE_ERROR(cudaMemcpyToSymbol(min_pos_d, &min_pos, sizeof(double)));
    HANDLE_ERROR(cudaMemcpyToSymbol(max_v_d, &max_v, sizeof(double)));
    HANDLE_ERROR(cudaMemcpyToSymbol(max_iter_d, &args.max_iter, sizeof(int)));
    HANDLE_ERROR(cudaMemcpyToSymbol(particle_cnt_d, &args.particle_cnt, sizeof(int)));
    HANDLE_ERROR(cudaMemcpyToSymbol(tile_size, &args.threads_per_block, sizeof(int)));
    HANDLE_ERROR(cudaMemset(lock_d, 0, sizeof(int)));
    clock_t end_init = clock();
    //clock_t begin_exe  = end_init;
    // HANDLE_ERROR(cudaEventRecord(start_init));
    // init<<<args.blocks_per_grid, args.threads_per_block, 2 * sizeof(double) * args.threads_per_block>>>
    //         (position_d, velocity_d, fitness_d, pbest_pos_d, pbest_fit_d, gbest_d, lock_d);
    // HANDLE_ERROR(cudaEventRecord(stop_init));
    // HANDLE_ERROR(cudaMemset(lock_d, 0, sizeof(int)));    
    HANDLE_ERROR(cudaEventRecord(start));
    for(unsigned int i = 0; i < args.max_iter; i++){
        move<<<args.blocks_per_grid, args.threads_per_block, 2 * sizeof(double) * args.threads_per_block>>>
            (position_d, velocity_d, fitness_d, pbest_pos_d, pbest_fit_d, 
                gbest_d, lock_d, aux, aux_pos);
        findBestInOneBlock<<<1, block_size>>>(gbest_d, aux, aux_pos, block_size);
    }
    HANDLE_ERROR(cudaEventRecord(stop));
    HANDLE_ERROR(cudaMemcpy(p->position, position_d, sizeof(double)* particle_cnt, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(p->velocity, velocity_d, sizeof(double)* particle_cnt, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(p->fitness, fitness_d, sizeof(double)* particle_cnt, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(p->pbest_pos, pbest_pos_d, sizeof(double)* particle_cnt, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(p->pbest_fit, pbest_fit_d, sizeof(double)* particle_cnt, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(&gbest, gbest_d, sizeof(particle), cudaMemcpyDeviceToHost));
    clock_t end_exe  = clock();
    //HANDLE_ERROR(cudaEventSynchronize(stop_init));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    //HANDLE_ERROR(cudaEventElapsedTime(&init_time, start_init, stop_init));
    HANDLE_ERROR(cudaEventElapsedTime(&exe_time, start, stop));

    //for(int i=0; i<particle_cnt; i++)
    //    printf("#%d : %lf , %lf . %lf\n", i+1, p->position[i], p->fitness[i], p->velocity[i]);
    free(p);
    cudaFree(position_d);
    cudaFree(velocity_d);
    cudaFree(fitness_d);
    cudaFree(pbest_pos_d);
    cudaFree(pbest_fit_d);
    cudaFree(gbest_d);
    //cudaFree(gbest_u);
    cudaFree(lock_d);
    printf("the answer : %10.6lf, %lf\n", -57.469, fit(-57.469));
    printf("best result: %10.6lf, %lf\n", gbest.position, gbest.fitness);
    printf("[Initial   time]: %lf (sec)\n", (double)(end_init - begin_init) / CLOCKS_PER_SEC);
    //printf("[Execution time]: %lf (sec)\n", (double)(end_exe - begin_exe) / CLOCKS_PER_SEC);
    //printf("[Cuda Init time]: %f (sec)\n", init_time / 1000);
    printf("[Cuda Exec time]: %f (sec)\n", exe_time / 1000);
    printf("[Elapsed   time]: %lf (sec)\n", (double)(clock() - begin_app) / CLOCKS_PER_SEC);
    return 0;
}

void ParticleInitCoal(particle_Coal *p){
	unsigned int i;
	const double pos_range = max_pos - min_pos; // 解寬度
    srand((unsigned)time(NULL));
    p->position  = (double *) malloc(sizeof(double)* particle_cnt);
    p->velocity  = (double *) malloc(sizeof(double)* particle_cnt); 
    p->fitness   = (double *) malloc(sizeof(double)* particle_cnt); 
    p->pbest_pos = (double *) malloc(sizeof(double)* particle_cnt); 
    p->pbest_fit = (double *) malloc(sizeof(double)* particle_cnt);
    // 以下程式碼效率不佳, 但較易懂一點
    //clock_t start_init  = clock();
	for(i=0; i<particle_cnt; i++) {
		// 隨機取得粒子位置, 並設為該粒子目前最佳適應值
		p->pbest_pos[i] = p->position[i] = RND() * pos_range + min_pos; 
		// 隨機取得粒子速度
		p->velocity[i] = RND() * max_v;
		// 計算該粒子適應值, 並設為該粒子目前最佳適應值
		p->pbest_fit[i] = p->fitness[i] = fit(p->position[i]);
		// 全域最佳設定
		if(i==0 || p->pbest_fit[i] > gbest.fitness){
			gbest.position = p->position[i];      // 目前位置, 即x value    
			gbest.velocity = p->velocity[i];      // 目前粒子速度           
			gbest.fitness = p->fitness[i];       // 適應函式值            
			gbest.pbest_pos = p->pbest_pos[i];    // particle 目前最好位置
			gbest.pbest_fit = p->pbest_fit[i];   // particle 目前最佳適應值
		} 
    }
    //printf("[cpu Init time]: %f (sec)\n",  (double)(clock() - start_init) / CLOCKS_PER_SEC);
}
