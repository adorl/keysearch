#ifndef KEYLOG_H
#define KEYLOG_H

/*
 * keylog.h — logging infrastructure
 *
 * Provides three log level macros:
 *   keylog_info(fmt, ...)   — info level, prefix [info]
 *   keylog_warn(fmt, ...)   — warning level, prefix [warn]
 *   keylog_error(fmt, ...)  — error level, prefix [error]
 *
 * Must call log_init() before use, call log_close() before program exit.
 * Log file name format: search_YYYYMMDD_HH:MM:SS.log
 * Uses open/write/close internally, single write per log entry, thread-safe.
 */

#include <unistd.h>
#include <stdio.h>

/* Initialize log file, returns 0 on success, -1 on failure */
int  log_init(void);

/* Close log file */
void log_close(void);

/* Global log file descriptor (defined in keylog.c, used internally by macros) */
extern int g_log_fd;

/* Internal helper macro: assemble prefix+formatted content into a single write, ensuring log lines don't interleave in multi-threaded context */
#define _keylog_write(prefix, fmt, ...) \
    do { \
        char _buf[1024]; \
        int _len = snprintf(_buf, sizeof(_buf), prefix fmt "\n", ##__VA_ARGS__); \
        if (_len > 0 && g_log_fd >= 0) { \
            if (_len >= (int)sizeof(_buf)) _len = (int)sizeof(_buf) - 1; \
            (void)write(g_log_fd, _buf, (size_t)_len); \
        } \
    } while (0)

#define keylog_info(fmt, ...)  _keylog_write("[info] ",  fmt, ##__VA_ARGS__)
#define keylog_warn(fmt, ...)  _keylog_write("[warn] ",  fmt, ##__VA_ARGS__)
#define keylog_error(fmt, ...) _keylog_write("[error] ", fmt, ##__VA_ARGS__)

#endif /* KEYLOG_H */
