#ifndef METRICS_UPLOADER_H
#define METRICS_UPLOADER_H


#include <vector>
#include <unordered_map>
#include <string>


class MetricsUploader
{
public:
	static std::string API_URL;
	static std::string UPLOAD_TOKEN;

	MetricsUploader();
	~MetricsUploader();

	void SetRunId(const std::string& id);

	void UploadMetrics(const std::unordered_map<std::string, float>& metrics);

	

protected:
	std::string runId;

	std::string BuildRequestBody(const std::string& runId, const std::unordered_map<std::string, float>& metrics) const;
};

#endif