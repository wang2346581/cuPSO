/*******************************************************************/
/*                                                                 */
/*     filename : simple_pso.c                                     */
/*     author   : edison.shih/edisonx                              */
/*     compiler : Visual C++ 2008                                  */
/*     date     : 2011.03.07                                       */
/*                                                                 */
/*         A.L.L.  R.I.G.H.T.S.  R.E.S.E.R.V.E.                    */
/*                                                                 */
/*                                                                 */
/*******************************************************************/


/************************************************************************/
/* 重要變數與參數說明                                                   */
/*                                                                      */
/* 1.particle_cnt: 粒子數, 通常於-40,複雜設-200                         */
/*                                                                      */
/* 2.粒子長度: 即解空間維度,或使用變數個數,由於此問題之適應函式只用到   */
/*             x 變數, 故粒子長度為，將再補程式碼做為考慮粒子長度為n    */
/*             之情形                                                   */
/*                                                                      */
/* 3.max_pos, min_pos: 粒子範圍,即解空間中,各維度(變數)之範圍限制,      */
/*             普遍性應考慮n 維之情形                                   */
/*                                                                      */
/* 4.max_v : 即最大速度限制, 通常設定成粒子之寬度,即解空間範圍,普遍性考 */
/*           慮n 維情形,即每維之解空間範圍                              */
/* 5.c1,c2 : 學習常數, c1,c2多設, 一般c1=c2=[0,4]                       */
/*                                                                      */
/* 6.w     : 慣性權重, 此常數為後來由Yuhui Shi and Russell Eberhart 提  */
/*           出,多設在[0,1.5] 之間, 設1.0 為保守值                      */
/*                                                                      */
/************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <time.h>
#include <math.h>
#include <unistd.h>  /* getopt */

typedef struct tag_arguments{
    int max_iter;
    int particle_cnt;
    int threads_per_block;
    int blocks_per_grid;
    int block_queue_size;
    int verbose;
} arguments;

int pargeArgs(arguments *args, int argc, char **argv){
    int cmd_opt = 0;
    //fprintf(stderr, "argc:%d\n", argc);
    while(1) {
        //fprintf(stderr, "proces index:%d\n", optind);
        cmd_opt = getopt(argc, argv, "v:m:c:t:b:q::");
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
            case 'q':
                args->block_queue_size = atoi(optarg);
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
    if(args->threads_per_block && args->particle_cnt)
        args->blocks_per_grid = ((args->particle_cnt-1)/args->threads_per_block+1);
    if(args->verbose){
        printf("max_iter is %d \n", args->max_iter);
        printf("particle_cnt is %d \n", args->particle_cnt);
        printf("threads_per_block is %d \n", args->threads_per_block);
        printf("blocks_per_grid is %d \n", args->blocks_per_grid);
        printf("block_queue_size is %d \n", args->block_queue_size);
        printf("=============================\n");
    }
    return 1;
}
//////////////////////////////////////////////////////////////////////////
// -----------------------------------------------------------------------
// fitness value
// best value on [-100, 100] , aboue fit(-57.469) = 390245.791738

double fit(double x)
{
    // x**3 - 0.8x**2 - 1000x + 8000
    return fabs(8000.0 + x*(-10000.0+x*(-0.8+x)));
}


//////////////////////////////////////////////////////////////////////////
// -----------------------------------------------------------------------
// * 使用pso 求x**3 - 0.8x**2 - 10000x + 8000 於[-100,100] 間之最大值    *
// -----------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////////

/* 定義結構體particle */
typedef struct tag_particle{
    double position;  /* 目前位置, 即x value    */
    double velocity;  /* 目前粒子速度           */
    double fitness ;  /* 適應函式值             */
    double pbest_pos; /* particle 目前最好位置  */
    double pbest_fit; /* particle 目前最佳適應值*/
}particle;

// -----------------------------------------------------------------------
// * 全域變數宣告, 較好佈局為拆pos.h / pos.c , 放在pos.c 宣告成static    *
// * 再做set_param function 為入口, 設定所有static param variable        *
// * 或直接用#define 方式, 也需get_gbest 做入口取得全域最佳值            *
// -----------------------------------------------------------------------

double w, c1, c2;                    /* 相關權重參數   */
double max_v;                        /* 最大速度限制   */
double max_pos, min_pos;             /* 最大,小位置限制*/
unsigned particle_cnt;               /* 粒子數量       */
particle gbest;                      /* 全域最佳值     */

// -----------------------------------------------------------------------
// * pso 相關函式宣告                                                    *
// -----------------------------------------------------------------------

#define RND() ((double)rand()/RAND_MAX) /* 產生[0,1] 亂數         */
particle* AllocateParticle();           /* 配置particle_cnt 個粒子*/
void ParticleInit(particle *p);         /* 粒子初始化             */
int ParticleMove(particle *p);         /* 開始移動               */
void ParticleRelease(particle* p);      /* 釋放粒子記憶體         */
void ParticleDisplay(particle* p);      /* 顯示所有粒子資訊       */
double random_v(int *next){
    return (double) 
        (((*next = *next * 1103515245U + 12345U)>>16) & 32767 )/ 32767; 
}

//int counter[MAX_ITERA] = {0};

//////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv){
    arguments args = {100000, 4096, 1024, 4, 10, 0};
    int retError = pargeArgs(&args, argc, argv);
    int *counter = (int *)malloc(sizeof(int) * args.max_iter);
    /* 變數宣告*/
    clock_t begin_app  = clock();
    clock_t begin_init = begin_app;
    unsigned i, j;
    unsigned max_itera = args.max_iter;           /* max_itera : 最大演化代數*/
    particle* p;                         /* p         : 粒子群      */
    /* 設定參數*/
    min_pos = -100.0 , max_pos = +100.0;  /* 位置限制, 即解空間限制   */
    w = 1, c1 = 2.0, c2 = 2.0;          /* 慣性權重與加速常數設定   */
    particle_cnt = args.particle_cnt;             /* 設粒子個數               */
    max_v = (max_pos-min_pos) * 1.0;    /* 設最大速限               */
    /* 開始進行*/
    p = AllocateParticle();      // 配置記憶體
    ParticleInit(p);             // 粒子初始化
    clock_t end_init = clock();
    clock_t begin_exe  = end_init;
    for(i=0; i<max_itera; i++)   // 進行迭代
        counter[i] = ParticleMove(p);// 粒子移動
    clock_t end_exe  = clock();
    ParticleDisplay(p);          // 顯示最後結果
    ParticleRelease(p);          // 釋放記憶體
    // 暴力取得之較佳值
    printf("know2: %10.6lf , %lf\n", -57.469, fit(-57.469));
    printf("[Initial   time]: %lf (sec)\n", (double)(end_init - begin_init) / CLOCKS_PER_SEC);
    printf("[Execution time]: %lf (sec)\n", (double)(end_exe - begin_exe) / CLOCKS_PER_SEC);
    printf("[Elapsed   time]: %lf (sec)\n", (double)(clock() - begin_app) / CLOCKS_PER_SEC);
    
    //for(i = 0; i < max_itera; i++){
    //    if(counter[i] > 0){
    //        printf("index is %d, value is %d\n", i, counter[i]);
    //    }
    //}
    return 0;
}

//////////////////////////////////////////////////////////////////////////

// 配置particle_cnt 個粒子
particle* AllocateParticle()
{
     return (particle*)malloc(sizeof(particle)*particle_cnt);
}

// 粒子初始化
void ParticleInit(particle *p)
{
    unsigned i;
    const double pos_range = max_pos - min_pos; // 解寬度
    srand((unsigned)time(NULL));

    // 以下程式碼效率不佳, 但較易懂一點
    for(i=0; i<particle_cnt; i++) {
         // 隨機取得粒子位置, 並設為該粒子目前最佳適應值
         p[i].pbest_pos = p[i].position = RND() * pos_range + min_pos; 
         // 隨機取得粒子速度
         p[i].velocity = RND() * max_v;
         // 計算該粒子適應值, 並設為該粒子目前最佳適應值
         p[i].pbest_fit = p[i].fitness = fit(p[i].position);

         // 全域最佳設定
         if(i==0 || p[i].pbest_fit > gbest.fitness) 
              memcpy((void*)&gbest, (void*)&p[i], sizeof(particle));
    }
}

// 開始移動
int ParticleMove(particle *p)
{
    unsigned i;
    double v, pos;     // 暫存每個粒子之速度, 位置用
    double ppos, gpos; // 暫存區域及全域最佳位置用
    gpos = gbest.position;
    int flags = 0;
    //static int index = 0;
    int next = 1;
    // 更新速度與位置
    for(i=0; i<particle_cnt; i++){
         v = p[i].velocity; // 粒子目前速度
         pos=p[i].position; // 粒子目前位置
         ppos=p[i].pbest_pos; // 粒子目前曾到到最好位置
         
         v = w*v + c1*RND()*(ppos-pos) + c2*RND()*(gpos-pos); // 更新速度
         //v = w*v + c1*random_v(&next)*(ppos-pos) + c2*random_v(&next)*(gpos-pos);
         if(v<-max_v) v=-max_v;    // 限制最大速度
         else if(v>max_v) v=max_v; // 限制最大速度
         
         pos = pos + v;               // 更新位置
         if(pos>max_pos) pos=max_pos; // 限制最大位置
         else if(pos<min_pos) pos=min_pos; // 限制最小位置
         p[i].velocity= v;        // 更新粒子速度      
         p[i].position=pos;       // 更新粒子位置
         p[i].fitness = fit(pos); // 更新粒子適應值
         // 更新該粒子目前找過之最佳值
         if(p[i].fitness > p[i].pbest_fit) {
              p[i].pbest_fit = p[i].fitness ;
              p[i].pbest_pos = p[i].position;
              //printf("[iter %d] idx: %d, pbest is %lf\n", index, i, p[i].pbest_fit);
         }
         // 更新全域最佳值
         if(p[i].fitness > gbest.fitness){
              memcpy((void*)&gbest, (void*)&p[i], sizeof(particle));
              flags += 1;
         }
    }
    //index ++;
    return flags;
}

// 釋放粒子記憶體
void ParticleRelease(particle* p)
{
    free(p);
}

// 顯示所有粒子資訊
void ParticleDisplay(particle* p)
{
    unsigned i;
    /* 若想看完整的粒子資料，可把下面三行註解拿掉，這裡只顯示最佳解。*/
     
//  for(i=0; i<particle_cnt; i++)
//         printf("#%d : %lf , %lf . %lf\n", i+1, p[i].position, p[i].fitness, p[i].velocity);
//   puts("------------------------------\n");  
    printf("best : %10.6lf , %lf\n", gbest.position, gbest.fitness);     
}

