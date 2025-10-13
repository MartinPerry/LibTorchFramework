#ifndef PROGRESS_BAR_H
#define PROGRESS_BAR_H

#include <chrono>
#include <string>
#include <unordered_map>

class ProgressBar 
{
public:
    ProgressBar(int width = 50);

    void ClearParams();
    void SetParam(const std::string& name, const std::string& val);

    void Start(int total);
    void NextStep();
    void Update(int current);

    void Finish();

private:
    int barWidth;
    int total;
    int lastProgress;
    int lastCurrent;
    
    std::unordered_map<std::string, std::string> params;

    std::chrono::time_point<std::chrono::steady_clock> start_time;

};

#endif
