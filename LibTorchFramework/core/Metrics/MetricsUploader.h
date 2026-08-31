#ifndef METRICS_UPLOADER_H
#define METRICS_UPLOADER_H


#include <vector>
#include <unordered_map>
#include <string>

#include "./MetricsDefault.h"

class MetricsUploader
{
public:
	static std::string API_URL;
	static std::string UPLOAD_TOKEN;

	MetricsUploader();
	~MetricsUploader();

	void SetRunId(const std::string& id);
	void SetImageUploadEnabled(bool val);

	void UploadMetrics(const std::unordered_map<std::string, float>& metrics, 
		const MetricsDefault::SaveInfo& si);

	void UploadImage(int imageIndex, const std::string& imagePath, 
		const MetricsDefault::SaveInfo& si);
	

protected:
	std::string runId;
	bool imgUploadEnabled;

	std::vector<char> LoadImageData(const std::string& filePath) const;

	std::string BuildRequestBody(const std::string& runId, const std::unordered_map<std::string, float>& metrics) const;
};

#endif