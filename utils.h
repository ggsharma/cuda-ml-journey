#include <iostream>
#include <chrono>
#include <string>

class Timer{
    private:
    decltype(std::chrono::high_resolution_clock::now()) _start;
    decltype(std::chrono::high_resolution_clock::now()) _stop;
    std::string param = "";
    public:
        Timer(const std::string& s):param(s) {};
        void start(){
               _start = std::chrono::high_resolution_clock::now();
        }

        void stop(){
            _stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(_stop - _start);
            
            std::cout << "Time taken for " << param << " : " << duration.count() << " microseconds" << std::endl;
            std::cout << "Time taken for " << param << " : " << duration.count() / 1000.0 << " milliseconds" << std::endl;
        }
};