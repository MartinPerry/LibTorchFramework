#include "./AbstractModel.h"

#include "./Snapshot/FreezeInfo.h"

AbstractModel::AbstractModel() : 
	optimizer(nullptr)
{
}

AbstractModel::~AbstractModel()
{
}

void AbstractModel::RemoveOptimizer()
{	
	this->optimizer = nullptr;
}

void AbstractModel::SetFrozen(std::shared_ptr<FreezeInfo> freezeInfo) const
{    
    torch::OrderedDict<std::string, at::Tensor> params = this->named_parameters();

    if (freezeInfo == nullptr)
    {
        //unfreeze
        for (auto& kv : params)
        {
            kv.value().set_requires_grad(true);            
        }

        return;
    }

    if (freezeInfo->IsFreezeAllEnabled())
    {
        for (auto& kv : params)
        {
            kv.value().set_requires_grad(false);
        }

        return;
    }

    std::vector<std::string> freezedParts;
    for (auto& kv : params)
    {
        if (freezeInfo->enabled && freezeInfo->CanFreeze(kv.key()))
        {
            freezedParts.push_back(kv.key());
        }
    }

    if (freezedParts.empty())
    {
        return;
    }

    for (auto& kv : params)
    {
        if (std::find(freezedParts.begin(), freezedParts.end(), kv.key()) != freezedParts.end())
        {
            //MY_LOG_INFO("Freezing: %s", kv.key().c_str());
            kv.value().set_requires_grad(false);
        }
        else 
        {
            kv.value().set_requires_grad(true);
        }
    }
}

void AbstractModel::OnBatchStart()
{
}

void AbstractModel::OnBatchEnd()
{
}

void AbstractModel::OnEpochStart()
{
}

void AbstractModel::OnEpochEnd()
{
}