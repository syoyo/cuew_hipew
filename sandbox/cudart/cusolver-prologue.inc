struct syevjInfo;
typedef struct syevjInfo *syevjInfo_t;

struct gesvdjInfo;
typedef struct gesvdjInfo *gesvdjInfo_t;

struct csrqrInfo;
typedef struct csrqrInfo *csrqrInfo_t;

struct csrqrInfoHost;
typedef struct csrqrInfoHost *csrqrInfoHost_t;

typedef void (*cusolverDnLoggerCallback_t)(
      int         logLevel,
      const char *functionName,
      const char *message);

