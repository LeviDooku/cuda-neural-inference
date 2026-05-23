//Pedro Velasco Santana
//File: timer.cpp

//Purpose: measure times in CPU

#include "timer.h"

void Timer::start(){
    start_time = std::chrono::steady_clock::now();
}

double Timer::stop_milliseconds(){
    std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end_time - start_time;
    return elapsed.count();
}



