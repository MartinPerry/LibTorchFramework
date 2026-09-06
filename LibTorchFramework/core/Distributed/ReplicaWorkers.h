#ifndef REPLICA_WORKERS_H
#define REPLICA_WORKERS_H

#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

// One persistent host thread per replica. Run is called by one coordinator;
// every submitted task finishes before Run returns, including on exceptions.
class ReplicaWorkers
{
public:
    explicit ReplicaWorkers(size_t count)
    {
        if (count == 0)
        {
            throw std::invalid_argument("At least one replica worker is required");
        }
        workers.reserve(count);
        for (size_t i = 0; i < count; ++i)
        {
            workers.push_back(std::make_unique<Worker>());
        }
    }

    void Run(const std::function<void(size_t)>& fn)
    {
        std::vector<std::future<void>> completions;
        completions.reserve(workers.size());
        std::exception_ptr failure;
        try
        {
            for (size_t i = 0; i < workers.size(); ++i)
            {
                completions.push_back(workers[i]->Submit([&fn, i] { fn(i); }));
            }
        }
        catch (...)
        {
            failure = std::current_exception();
        }
        for (auto& completion : completions)
        {
            try
            {
                completion.get();
            }
            catch (...)
            {
                if (!failure)
                {
                    failure = std::current_exception();
                }
            }
        }
        if (failure)
        {
            std::rethrow_exception(failure);
        }
    }

private:
    class Worker
    {
    public:
        Worker() : thread([this] { Loop(); }) {}

        ~Worker()
        {
            {
                std::lock_guard<std::mutex> lock(mutex);
                stopping = true;
            }
            ready.notify_one();
            thread.join();
        }

        std::future<void> Submit(std::function<void()> fn)
        {
            std::packaged_task<void()> task(std::move(fn));
            auto completion = task.get_future();
            {
                std::lock_guard<std::mutex> lock(mutex);
                pending = std::move(task);
            }
            ready.notify_one();
            return completion;
        }

    private:
        void Loop()
        {
            for (;;)
            {
                std::packaged_task<void()> task;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    ready.wait(lock, [this] { return stopping || pending.valid(); });
                    if (!pending.valid())
                    {
                        return;
                    }
                    task = std::move(pending);
                }
                task(); // packaged_task transfers exceptions to the coordinator.
            }
        }

        std::mutex mutex;
        std::condition_variable ready;
        std::packaged_task<void()> pending;
        bool stopping = false;
        std::thread thread;
    };

    std::vector<std::unique_ptr<Worker>> workers;
};

#endif
