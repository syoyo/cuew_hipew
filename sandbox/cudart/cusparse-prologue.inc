struct csrsv2Info;
typedef struct csrsv2Info* csrsv2Info_t;

struct csrsm2Info;
typedef struct csrsm2Info* csrsm2Info_t;

struct bsrsv2Info;
typedef struct bsrsv2Info* bsrsv2Info_t;

struct bsrsm2Info;
typedef struct bsrsm2Info* bsrsm2Info_t;

struct csric02Info;
typedef struct csric02Info* csric02Info_t;

struct bsric02Info;
typedef struct bsric02Info* bsric02Info_t;

struct csrilu02Info;
typedef struct csrilu02Info* csrilu02Info_t;

struct bsrilu02Info;
typedef struct bsrilu02Info* bsrilu02Info_t;

struct csrgemm2Info;
typedef struct csrgemm2Info* csrgemm2Info_t;

struct csru2csrInfo;
typedef struct csru2csrInfo* csru2csrInfo_t;

struct pruneInfo;
typedef struct pruneInfo* pruneInfo_t;


typedef void (*cusparseLoggerCallback_t)(int         logLevel,
                                         const char* functionName,
                                         const char* message);
