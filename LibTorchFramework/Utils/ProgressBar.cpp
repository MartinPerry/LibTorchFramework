#include "./ProgressBar.h"

#include <iostream>
#include <string>

ProgressBar::ProgressBar(int width) : 
    barWidth(width), 
    total(0), 
    lastProgress(0),
    lastCurrent(0)
{    
}

void ProgressBar::ClearParams()
{
    this->params.clear();
}

void ProgressBar::SetParam(const std::string& name, const std::string& val)
{
    auto it = this->params.try_emplace(name, val);
    if (it.second)
    {
        it.first->second = val;
    }
}

void ProgressBar::Start(int total)
{
    this->total = total;
    this->lastCurrent = 0;
    this->lastProgress = -1;

    start_time = std::chrono::steady_clock::now();
}

void ProgressBar::NextStep()
{
    this->Update(lastCurrent + 1);
}

void ProgressBar::Update(int current) 
{
    lastCurrent = current;

    int progress = static_cast<int>(100.0 * current / total);
    if (progress == lastProgress)
    {
        return;  // avoid flicker
    }
    lastProgress = progress;

    double ratio = static_cast<double>(current) / total;
    int filled = static_cast<int>(barWidth * ratio);

    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();

    std::string bar(filled, '#');
    bar.resize(barWidth, '-');

    std::string paramsStr = "";
    for (const auto& [p, v] : this->params)    
    {
        paramsStr += "[";
        paramsStr += p;
        paramsStr += "=";
        paramsStr += v;
        paramsStr += "]";
    }

    std::cout << "\r[" << bar << "] "
        << progress << "% (" << current << "/" << total << ") "
        << elapsed << "s " << paramsStr << std::flush;
}

void ProgressBar::Finish() 
{
    this->Update(total);
    std::cout << std::endl;
}