//Pedro Velasco Santana
//File: timer.h

//Purpose: measure times in CPU

#ifndef TIMER_H
#define TIMER_H

#include <chrono>

class Timer{
private:
    std::chrono::steady_clock::time_point start_time;

public:
    void start();
    double stop_milliseconds();
};

#endif