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
#include <chrono>

int gettimeofday(struct timeval* tp, struct timezone* tzp) {
  namespace sc = std::chrono;
  sc::system_clock::duration d = sc::system_clock::now().time_since_epoch();
  sc::seconds s = sc::duration_cast<sc::seconds>(d);
  tp->tv_sec = s.count();
  tp->tv_usec = sc::duration_cast<sc::microseconds>(d - s).count();

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