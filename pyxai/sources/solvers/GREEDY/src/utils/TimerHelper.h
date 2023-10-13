#ifndef miniMOLS_TimerHelper_h
#define miniMOLS_TimerHelper_h


//It is __GLIBC__ instead of __linux__ 
#if defined(__GLIBC__)
#include <fpu_control.h>
#endif

#if defined(_MSC_VER) || defined(__MINGW32__)
#include <time.h>
#else
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>
#include <iostream>
#endif

#if defined(_WIN32)
#include <Windows.h>
#include <stdint.h> // portable: uint64_t   MSVC: __int64 
#include <winsock2.h>
// MSVC defines this in winsock2.h!?
typedef struct timeval {
    long tv_sec;
    long tv_usec;
} timeval;

int gettimeofday(struct timeval * tp, struct timezone * tzp)
{
    // Note: some broken versions only have 8 trailing zero's, the correct epoch has 9 trailing zero's
    // This magic number is the number of 100 nanosecond intervals since January 1, 1601 (UTC)
    // until 00:00:00 January 1, 1970 
    static const uint64_t EPOCH = ((uint64_t) 116444736000000000ULL);

    SYSTEMTIME  system_time;
    FILETIME    file_time;
    uint64_t    time;

    GetSystemTime( &system_time );
    SystemTimeToFileTime( &system_time, &file_time );
    time =  ((uint64_t)file_time.dwLowDateTime )      ;
    time += ((uint64_t)file_time.dwHighDateTime) << 32;

    tp->tv_sec  = (long) ((time - EPOCH) / 10000000L);
    tp->tv_usec = (long) (system_time.wMilliseconds * 1000);
    return 0;
}
#endif // _WIN32


namespace pyxai {

    static double initRealTime = 0;
    static double initCpuTime = 0;
    static bool isInitialized = false;

    namespace TimerHelper {
      
        inline double cpuTime() {
          if(!isInitialized){
            std::cout << "Warning: initializeTime() has not been called before !" << std::endl;
          }
#if defined(_MSC_VER) || defined(__MINGW32__)
          return ((double)clock() / CLOCKS_PER_SEC) - initCpuTime;
#else
          struct rusage ru;
          getrusage(RUSAGE_SELF, &ru);
          if (initCpuTime != 0)
            return ((double)ru.ru_utime.tv_sec + (double)ru.ru_utime.tv_usec / 1000000) - initCpuTime;

          return (double)ru.ru_utime.tv_sec + (double)ru.ru_utime.tv_usec / 1000000; 
#endif
        }

        inline double realTime() {
          if(!isInitialized){
            std::cout << "Warning: initializeTime() has not been called before !" << std::endl;
          }
          struct timeval tv;
          gettimeofday(&tv, NULL);
          if (initRealTime != 0)
            return ((double)tv.tv_sec + (double) tv.tv_usec / 1000000) - initRealTime;
          
          return (double)tv.tv_sec + (double) tv.tv_usec / 1000000; 
        }

        inline void initializeTime(){
          initRealTime = 0;
          initCpuTime = 0;
          isInitialized = true;
          initRealTime = realTime();
          initCpuTime = cpuTime();
        }

        

    }
}
#endif