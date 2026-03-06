#define _POSIX_C_SOURCE 200112L

#include "keylog.h"

#include <fcntl.h>
#include <time.h>
#include <stdio.h>

/* 全局日志文件描述符，-1 表示未初始化 */
int g_log_fd = -1;

/* 初始化日志文件：按启动时间生成文件名并打开 */
int log_init(void)
{
    time_t t = time(NULL);
    struct tm *tm_info = localtime(&t);
    char log_path[64];
    strftime(log_path, sizeof(log_path), "search_%Y%m%d_%H:%M:%S.log", tm_info);
    g_log_fd = open(log_path, O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (g_log_fd < 0) {
        fprintf(stderr, "错误：无法创建日志文件 %s\n", log_path);
        return -1;
    }
    return 0;
}

/* 关闭日志文件 */
void log_close(void)
{
    if (g_log_fd >= 0) {
        close(g_log_fd);
        g_log_fd = -1;
    }
}
