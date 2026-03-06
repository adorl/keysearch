#ifndef KEYLOG_H
#define KEYLOG_H

/*
 * keylog.h — 日志基础设施
 *
 * 提供三个级别的日志宏：
 *   keylog_info(fmt, ...)   — 信息级别，前缀 [info]
 *   keylog_warn(fmt, ...)   — 警告级别，前缀 [warn]
 *   keylog_error(fmt, ...)  — 错误级别，前缀 [error]
 *
 * 使用前须调用 log_init()，程序退出前调用 log_close()。
 * 日志文件名格式：search_YYYYMMDD_HH:MM:SS.log
 * 底层使用 open/write/close，每条日志单次 write，多线程安全。
 */

#include <unistd.h>
#include <stdio.h>

/* 初始化日志文件，成功返回 0，失败返回 -1 */
int  log_init(void);

/* 关闭日志文件 */
void log_close(void);

/* 全局日志文件描述符（由 keylog.c 定义，宏内部使用） */
extern int g_log_fd;

/* 内部辅助宏：将前缀+格式化内容拼装为单次 write，保证多线程下日志行不交错 */
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
