#include "./AbstractModel.h"


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