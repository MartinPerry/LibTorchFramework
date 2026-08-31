#include "./MetricsUploader.h"

#include <sstream>

#include <Utils/Internet/DownloadManager.h>

#ifdef _WIN32
#   pragma comment(lib, "iphlpapi.lib")
#   pragma comment(lib, "secur32.lib")
#   pragma comment(lib, "crypt32.lib")
#   pragma comment(lib, "libcurl.lib")
#   pragma comment(lib, "libcrypto.lib")
#   pragma comment(lib, "libssl.lib")
#   pragma comment(lib, "nghttp2.lib")
#   pragma comment(lib, "ngtcp2.lib")
#   pragma comment(lib, "ngtcp2_crypto_ossl.lib")
#endif

std::string MetricsUploader::API_URL = "";
std::string MetricsUploader::UPLOAD_TOKEN = "";

MetricsUploader::MetricsUploader() : 
    runId(std::to_string(time(0)))
{
    DownloadManager::Init();
}

MetricsUploader::~MetricsUploader()
{
    //DownloadManager::Destroy();
}

void MetricsUploader::SetRunId(const std::string& id)
{
    this->runId = id;
}


static std::string escapeJson(const std::string& input)
{
    std::ostringstream output;
    const char hex[] = "0123456789abcdef";
    for (char c : input) 
    {        
        switch (c) 
        {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (c < 0x20) 
                {
                    output << "\\u00" << hex[(c >> 4) & 0x0f] << hex[c & 0x0f];
                }
                else 
                {
                    output << c;
                }
                break;
        }
    }
    return output.str();
}

std::string MetricsUploader::BuildRequestBody(const std::string& runId, const std::unordered_map<std::string, float>& metrics) const
{
    std::ostringstream body;
    body << "{\"run_id\":\"" << escapeJson(runId) << "\",\"data\":{";

    bool first = true;
    for (auto [key, value] : metrics) 
    {
        if (!first) 
        {
            body << ',';
        }
        first = false;
        if (std::isnan(value))
        {
            value = std::numeric_limits<float>::max();
        }

        body << '"' << escapeJson(key) << "\":\"" << escapeJson(std::to_string(value)) << '"';
    }
    body << "}}";
    return body.str();
}

static size_t receiveResponse(char* contents, size_t size, size_t count, void* userData)
{
    const size_t byteCount = size * count;
    std::string* response = static_cast<std::string*>(userData);
    response->append(contents, byteCount);
    return byteCount;
}

void MetricsUploader::UploadMetrics(const std::unordered_map<std::string, float>& metrics, 
    const MetricsDefault::SaveInfo& si)
{        
    auto dl = DownloadManager::GetInstance();
    if (dl == nullptr)
    {
        MY_LOG_ERROR("Web manager not inited");
        return;
    }

    const std::string requestBody = this->BuildRequestBody(this->runId, metrics);

    const std::string tokenHeader = "X-Upload-Token: " + MetricsUploader::UPLOAD_TOKEN;


    //dl->SetVerbose(true);
    DownloadJobSettings s;
    s.additionalHttpHeaders.push_back("Content-Type: application/json");
    s.additionalHttpHeaders.push_back("Accept: application/json");
    s.additionalHttpHeaders.push_back(tokenHeader);

    s.url = MetricsUploader::API_URL;
    s.dataType = DownloadJobSettings::DATA_TYPE::TEXT;
    s.expertSettings.SetRawPostData(requestBody);

    s.onFinish = [](std::shared_ptr<DownloadJob> job) {
        auto& v = job->GetData();

        std::string buf(v.data(), v.size());
        if (buf.find("true") == std::string::npos)
        {
            MY_LOG_ERROR("MetricsUploader: %s", buf.c_str());
        }
    };

    auto job = dl->AddDownload(s, false);


    //job->WaitToFinish();

    //printf("x");
}

std::vector<char> MetricsUploader::LoadImageData(const std::string& filePath) const
{
    FILE* imageFile = fopen(filePath.c_str(), "rb");
    if (imageFile == nullptr)
    {
        return {};
    }

    fseek(imageFile, 0, SEEK_END);
    const long fileSize = ftell(imageFile);
    fseek(imageFile, 0, SEEK_SET);

    if (fileSize <= 0)
    {
        fclose(imageFile);
        return {};
    }

    std::vector<char> buf(fileSize);
    
    const size_t bytesRead = fread(buf.data(), 1, buf.size(), imageFile);
    fclose(imageFile);

    if (bytesRead != buf.size())
    {
        return {};
    }

    return buf;
}

void MetricsUploader::UploadImage(int imageIndex, const std::string& imagePath, 
    const MetricsDefault::SaveInfo& si)
{
    auto dl = DownloadManager::GetInstance();
    if (dl == nullptr)
    {
        MY_LOG_ERROR("Web manager not inited");
        return;
    }
    
    auto imgData = this->LoadImageData(imagePath);
    if (imgData.size() == 0)
    {
        MY_LOG_ERROR("Failed to load image %s", imagePath.c_str());
        return;
    }
   
    std::string tmp(imgData.data(), imgData.size());
    
    std::string url = MetricsUploader::API_URL;
    url += "?action=image&run_id=" + this->runId;
    url += "&img_index=" + std::to_string(imageIndex);
    url += "&run_type=";
    if (si.runMode == RunMode::TRAIN)
    {
        url += "train";
    }
    else if (si.runMode == RunMode::TEST)
    {
        url += "test";
    }
    else if (si.runMode == RunMode::VALID)
    {
        url += "valid";
    }
    url += "&run_index=" + std::to_string(si.epochId);

    const std::string tokenHeader = "X-Upload-Token: " + MetricsUploader::UPLOAD_TOKEN;

    DownloadJobSettings s;
    s.additionalHttpHeaders.push_back("Content-Type: application/octet-stream");
    s.additionalHttpHeaders.push_back("Accept: application/json");
    s.additionalHttpHeaders.push_back(tokenHeader);
    
    s.dataType = DownloadJobSettings::DATA_TYPE::TEXT;
    s.expertSettings.SetRawPostData(tmp);
    
    s.url = url;
    s.dataType = DownloadJobSettings::DATA_TYPE::TEXT;    

    s.onFinish = [](std::shared_ptr<DownloadJob> job) {
        auto& v = job->GetData();

        std::string buf(v.data(), v.size());
        if (buf.find("true") == std::string::npos)
        {
            MY_LOG_ERROR("MetricsUploader: %s", buf.c_str());
        }
        };

    auto job = dl->AddDownload(s, false);

}


/*
const std::string requestBody = this->BuildRequestBody(runId, metrics);

    const std::string tokenHeader = "X-Upload-Token: " + MetricsUploader::UPLOAD_TOKEN;

    auto dl = DownloadManager::GetInstance();

    DownloadJobSettings s;
    s.additionalHttpHeaders.push_back("Content-Type: application/json");
    s.additionalHttpHeaders.push_back("Accept: application/json");
    s.additionalHttpHeaders.push_back(tokenHeader);

    s.url = MetricsUploader::API_URL;
    s.dataType = DownloadJobSettings::DATA_TYPE::TEXT;
    s.expertSettings.postFields.try_emplace("", requestBody);

    s.onFinish = [](std::shared_ptr<DownloadJob> job) {
        auto& v = job->GetData();

        std::string buf(v.data(), v.size());

        MY_LOG_INFO("%s", buf.c_str());
        };

    dl->AddDownload(s, false);
*/