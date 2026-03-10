#define _POSIX_C_SOURCE 200112L

#include "keylog.h"

#include <fcntl.h>
#include <time.h>
#include <stdio.h>

/* Global log file descriptor, -1 means uninitialized */
int g_log_fd = -1;

/* Initialize log file: generate filename based on startup time and open */
int log_init(void)
{
    time_t t = time(NULL);
    struct tm *tm_info = localtime(&t);
    char log_path[64];
    strftime(log_path, sizeof(log_path), "search_%Y%m%d_%H:%M:%S.log", tm_info);
    g_log_fd = open(log_path, O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (g_log_fd < 0) {
        fprintf(stderr, "Error: cannot create log file %s\n", log_path);
        return -1;
    }
    return 0;
}

/* Close log file */
void log_close(void)
{
    if (g_log_fd >= 0) {
        close(g_log_fd);
        g_log_fd = -1;
    }
}
