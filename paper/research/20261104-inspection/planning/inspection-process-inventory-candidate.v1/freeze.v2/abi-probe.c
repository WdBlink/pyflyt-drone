#include <stdio.h>
#include <stddef.h>
#include <sys/proc_info.h>
int main(void) { printf("{\"size\":%zu,\"comm\":%zu,\"name\":%zu,\"nice\":%zu,\"start_seconds\":%zu,\"start_microseconds\":%zu}\n", sizeof(struct proc_bsdinfo), offsetof(struct proc_bsdinfo,pbi_comm), offsetof(struct proc_bsdinfo,pbi_name), offsetof(struct proc_bsdinfo,pbi_nice), offsetof(struct proc_bsdinfo,pbi_start_tvsec), offsetof(struct proc_bsdinfo,pbi_start_tvusec)); return 0; }
