#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>  /* getopt */
#include <limits.h>
#include <time.h>

#define MAX_ITERA 100000
#define BlocksPerGrid ((COUNT-1)/ThreadsPerBlock+1)
#define ThreadsPerBlock 1024
#define RND() ((double)rand()/RAND_MAX) // 產生[0,1] 亂數
#define min(x,y) (x)<(y)?(x):(y)
#define COUNT 4096

//typedef struct tag_particle_Coal{
//    double position[COUNT];  // 目前位置, 即x value
//    double velocity[COUNT];  // 目前粒子速度
//    double fitness[COUNT] ;  // 適應函式值
//    double pbest_pos[COUNT]; // particle 目前最好位置
//    double pbest_fit[COUNT]; // particle 目前最佳適應值
//} particle_Coal;

typedef struct tag_particle_Coal{
    double *position;  // 目前位置, 即x value
    double *velocity;  // 目前粒子速度
    double *fitness ;  // 適應函式值
    double *pbest_pos; // particle 目前最好位置
    double *pbest_fit; // particle 目前最佳適應值
} particle_Coal;

typedef struct tag_particle{
    double position;  // 目前位置, 即x value
    double velocity;  // 目前粒子速度
    double fitness ;  // 適應函式值
    double pbest_pos; // particle 目前最好位置
    double pbest_fit; // particle 目前最佳適應值
} particle;

typedef struct tag_arguments{
    int max_iter;
    int particle_cnt;
    int threads_per_block;
    int blocks_per_grid;
    int verbose;
} arguments;

double w, c1, c2;                    // 相關權重參數
double max_v;                        // 最大速度限制
double max_pos, min_pos;             // 最大,小位置限制
unsigned int particle_cnt;               // 粒子數量
particle gbest = { INT_MIN, INT_MIN, 
    INT_MIN, INT_MIN, INT_MIN };     // 全域最佳值

//void ParticleInitCoal(particle_Coal *p); // 粒子初始化
void ParticleInitCoal(particle_Coal *p);
void ParticleInit(particle *p);

static void HandleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {        
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);        
        exit(EXIT_FAILURE);    
    }
}

int pargeArgs(arguments *args, int argc, char **argv){
    int cmd_opt = 0;
    //fprintf(stderr, "argc:%d\n", argc);
    while(1) {
        //fprintf(stderr, "proces index:%d\n", optind);
        cmd_opt = getopt(argc, argv, "v:m:c:t:b::");
        /* End condition always first */
        if (cmd_opt == -1) {
            break;
        }
        /* Print option when it is valid */
        //if (cmd_opt != '?') {
        //    fprintf(stderr, "option:-%c\n", cmd_opt);
        //}
        /* Lets parse */
        switch (cmd_opt) {
            case 'm':
                args->max_iter = atoi(optarg);
                break;
            case 'c':
                args->particle_cnt = atoi(optarg);
                break;
            case 't':
                args->threads_per_block = atoi(optarg);
                break;
            case 'b':
                args->blocks_per_grid = atoi(optarg);
                break;
            case 'v':
                args->verbose = atoi(optarg);
                break;
            /* Error handle: Mainly missing arg or illegal option */
            case '?':
                fprintf(stderr, "Illegal option %c \n", (char)optopt);
                break;
            default:
                fprintf(stderr, "Not supported option\n");
                break;
        }
    }
    // Do we have args?
    //if (argc > optind) {
    //    int i = 0;
    //    for (i = optind; i < argc; i++) {
    //        fprintf(stderr, "argv[%d] = %s\n", i, argv[i]);
    //    }
    //}

    // TODO: check there, when threads_per_block is bigger than particle_cnt
    // the block level atomic lock will be fail
    // [Result]: save value to shared memory should check that value is valid
    //           or just set threads_per_block is bigger than particle_cnt
    //if(args->threads_per_block > args->particle_cnt)
    //    args->threads_per_block = args->particle_cnt;
    if(args->threads_per_block && args->particle_cnt)
        args->blocks_per_grid = ((args->particle_cnt-1)/args->threads_per_block+1);
    if(args->verbose){
        printf("max_iter is %d \n", args->max_iter);
        printf("particle_cnt is %d \n", args->particle_cnt);
        printf("threads_per_block is %d \n", args->threads_per_block);
        printf("blocks_per_grid is %d \n", args->blocks_per_grid);
        printf("=============================\n");
    }
    return 1;
}