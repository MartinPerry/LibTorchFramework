#include "./safetensors.h"

#include <fstream>
#include <cstring>
#include <algorithm>
#include <climits>
#include <fcntl.h>
#include <FileUtils/MemMapFile.h>

using namespace safetensors;


SimpleJSONParser::SimpleJSONParser(const char* json_str) : 
	json(json_str), 
	pos(0) 
{
}

void SimpleJSONParser::skipWhitespace()
{
	while (json[pos] == ' ' || json[pos] == '\n' || json[pos] == '\r' || json[pos] == '\t')
		pos++;
}

std::string SimpleJSONParser::parseString()
{
	std::string result;
	pos++; // Skip opening quote
	while (json[pos] != '"')
	{
		if (json[pos] == '\\')
		{
			pos++;
			if (json[pos] == 'u')
			{
				// Handle Unicode escape (simplified)
				pos += 4;
			}
		}
		result += json[pos++];
	}
	pos++; // Skip closing quote
	return result;
}

std::vector<int64_t> SimpleJSONParser::parseArray()
{
	std::vector<int64_t> result;
	pos++; // Skip opening bracket
	while (json[pos] != ']')
	{
		skipWhitespace();
		size_t num_start = pos;
		while (std::isdigit(json[pos]))
			pos++;
		result.push_back(std::stoll(std::string(json + num_start, pos - num_start)));
		skipWhitespace();
		if (json[pos] == ',')
			pos++;
	}
	pos++; // Skip closing bracket
	return result;
}

std::array<size_t, 2> SimpleJSONParser::parseDataOffsets()
{
	std::array<size_t, 2> result;
	pos++; // Skip opening bracket
	skipWhitespace();
	size_t num_start = pos;
	while (std::isdigit(json[pos]))
		pos++;
	result[0] = std::stoull(std::string(json + num_start, pos - num_start));
	skipWhitespace();
	pos++; // Skip comma
	skipWhitespace();
	num_start = pos;
	while (std::isdigit(json[pos]))
		pos++;
	result[1] = std::stoull(std::string(json + num_start, pos - num_start));
	skipWhitespace();
	pos++; // Skip closing bracket
	return result;
}

void SimpleJSONParser::skipValue() {
	skipWhitespace();
	if (json[pos] == '"') {
		// Skip string
		parseString();
	}
	else if (json[pos] == '[') {
		// Skip array
		pos++;
		while (json[pos] != ']') {
			skipValue();
			skipWhitespace();
			if (json[pos] == ',')
				pos++;
		}
		pos++; // Skip closing bracket
	}
	else if (json[pos] == '{') {
		// Skip object
		pos++;
		while (json[pos] != '}') {
			skipWhitespace();
			if (json[pos] == '"') {
				parseString(); // Skip key
				skipWhitespace();
				if (json[pos] == ':')
					pos++;
				skipValue(); // Skip value
				skipWhitespace();
				if (json[pos] == ',')
					pos++;
			}
		}
		pos++; // Skip closing brace
	}
	else {
		// Skip number, boolean, or null
		while (json[pos] != ',' && json[pos] != '}' && json[pos] != ']' &&
			json[pos] != ' ' && json[pos] != '\n' && json[pos] != '\r' && json[pos] != '\t')
			pos++;
	}
}


TensorInfo SimpleJSONParser::parseTensorInfo()
{
	TensorInfo info;
	pos++; // Skip opening brace
	while (json[pos] != '}')
	{
		skipWhitespace();
		std::string key = parseString();
		skipWhitespace();
		pos++; // Skip colon
		skipWhitespace();
		if (key == "dtype")
		{
			info.dtype = parseString();
		}
		else if (key == "shape")
		{
			info.shape = parseArray();
		}
		else if (key == "data_offsets")
		{
			info.data_offsets = parseDataOffsets();
		}
		else
		{
			// Skip unknown fields
			//while (json[pos] != ',' && json[pos] != '}')
			//    pos++;
			skipValue();
		}
		skipWhitespace();
		if (json[pos] == ',')
			pos++;
	}
	pos++; // Skip closing brace
	return info;
}

std::unordered_map<std::string, TensorInfo> SimpleJSONParser::parse()
{
	size_t startPos = pos;
	std::unordered_map<std::string, TensorInfo> result;
	skipWhitespace();
	if (json[pos++] != '{')
		throw SafetensorsException("Expected object");
	while (json[pos] != '}')
	{
		skipWhitespace();
		std::string key = parseString();
		skipWhitespace();
		pos++; // Skip colon
		skipWhitespace();
		if (key != "__metadata__")
		{
			result[key] = parseTensorInfo();
		}
		else
		{
			// Skip metadata
			skipValue();
			//while (json[pos] != ',')
			//{
			//    pos++;
			//}
		}
		skipWhitespace();
		if (json[pos] == ',')
		{
			pos++;
		}
	}
	pos = startPos;
	return result;
}


//===============================================================================


static const std::unordered_map<std::string_view, torch::ScalarType> str_to_dtype = {
	{"BOOL", torch::kBool},
	{"U8", torch::kUInt8},
	{"I8", torch::kInt8},
	{"U16", torch::kUInt16},
	{"I16", torch::kInt16},
	{"U32", torch::kUInt32},
	{"I32", torch::kInt32},
	{"U64", torch::kUInt64},
	{"I64", torch::kInt64},
	{"F16", torch::kFloat16},
	{"BF16", torch::kBFloat16},
	{"F32", torch::kFloat32},
	{"F64", torch::kFloat64}
};

static const std::unordered_map<torch::ScalarType, std::string_view> dtype_to_str = [](){
	std::unordered_map<torch::ScalarType, std::string_view> result;
	for (const auto& [str, dtype] : str_to_dtype)
	{
		result.try_emplace(dtype, str);
	}
	return result;
}();

torch::ScalarType SafeTensorManager::get_torch_dtype(const std::string& dtype_str)
{
	auto it = str_to_dtype.find(dtype_str);
	if (it != str_to_dtype.end())
	{
		return it->second;
	}
	throw SafetensorsException("Unknown dtype: " + dtype_str);
}

std::string_view SafeTensorManager::get_safetensors_dtype(torch::ScalarType dtype)
{
	auto it = dtype_to_str.find(dtype);
	if (it != dtype_to_str.end())
	{
		return it->second;
	}
	throw SafetensorsException("Unsupported dtype");
}

bool SafeTensorManager::is_big_endian()
{
	union
	{
		uint32_t i;
		char c[4];
	} bint = { 0x01020304 };

	return bint.c[0] == 1;
}

void SafeTensorManager::validate_string_length(const std::string& str, const std::string& context)
{
	if (str.length() > SAFETENSORS_MAX_STRING_SIZE)
	{
		throw SafetensorsException(context + " exceeds maximum allowed length");
	}
}

std::unordered_map<std::string, TensorInfo> SafeTensorManager::ParseHeaderInfo(const char* data, size_t size)
{
	if (size < 8)
		throw SafetensorsException("Invalid file size");

	uint64_t header_size;
	std::memcpy(&header_size, data, sizeof(uint64_t));

	if (8 + header_size > size)
		throw SafetensorsException("Invalid header size");

	SimpleJSONParser parser(data + 8);
	return parser.parse();
}

SafeTensorManager::TensorMap SafeTensorManager::Load(const std::string& filename)
{
	return this->Load(filename, nullptr, nullptr);
}

SafeTensorManager::TensorMap SafeTensorManager::Load(const std::string& filename,
	std::function<std::string(const std::string&)> remapName)
{
	return this->Load(filename, remapName, nullptr);
}

void SafeTensorManager::Load(const std::string& filename,
	std::function<void(const std::string&, const torch::Tensor&)> fill)
{
	this->Load(filename, nullptr, fill);
}

SafeTensorManager::TensorMap SafeTensorManager::Load(const std::string& filename,
	std::function<std::string(const std::string&)> remapName,
	std::function<void(const std::string&, const torch::Tensor&)> fill)
{

	auto memFile = MemMapFile(filename.c_str(), O_RDONLY);

	if (memFile.IsOpened() == false)
	{
		throw SafetensorsException("Failed to open file: " + filename);
	}

	
	size_t file_size = memFile.GetSize();

	if (file_size > SAFETENSORS_MAX_FILE_SIZE)
	{
		memFile.Close();
		throw SafetensorsException("File size exceeds maximum allowed size");
	}

	void* mapped_file = memFile.Map(PROT_READ, MAP_PRIVATE);

	if (mapped_file == MAP_FAILED)
	{
		memFile.Close();
		throw SafetensorsException("Failed to memory map file");
	}

	try
	{
		uint64_t header_size;
		std::memcpy(&header_size, mapped_file, sizeof(uint64_t));
		if (is_big_endian())
		{
			header_size = swap_endian(header_size);
		}

		if (8 + header_size > file_size)
		{
			throw SafetensorsException("Invalid header size");
		}

		auto tensor_infos = ParseHeaderInfo(static_cast<char*>(mapped_file), file_size);

		if (tensor_infos.size() > SAFETENSORS_MAX_TENSORS)
		{
			throw SafetensorsException("Number of tensors exceeds maximum allowed");
		}

		TensorMap tensors;
		char* data_start = static_cast<char*>(mapped_file) + 8 + header_size;

		for (const auto& [name, info] : tensor_infos)
		{
			validate_string_length(name, "Tensor name");

			if (info.shape.size() > SAFETENSORS_MAX_DIM)
			{
				throw SafetensorsException("Tensor dimension exceeds maximum allowed");
			}

			torch::ScalarType dtype = get_torch_dtype(info.dtype);

			auto options = torch::TensorOptions()
				.dtype(dtype)
				.device(torch::kCPU);

			torch::Tensor cpu_tensor = torch::from_blob(
				data_start + info.data_offsets[0],
				info.shape,
				options);
			
			cpu_tensor = cpu_tensor.pin_memory();

			//.clone(); // Clone to own the data

			if (is_big_endian() && 
				(dtype == torch::kFloat16 || dtype == torch::kFloat32 || dtype == torch::kFloat64))
			{
				cpu_tensor = cpu_tensor.clone();

				auto data_ptr = static_cast<char*>(cpu_tensor.data_ptr());
				for (int64_t i = 0; i < cpu_tensor.numel() * cpu_tensor.element_size(); i += cpu_tensor.element_size())
				{
					std::reverse(data_ptr + i, data_ptr + i + cpu_tensor.element_size());
				}
			}

			if (fill)
			{
				//may not be cloned
				fill(name, cpu_tensor);				
			}
			else if (remapName)
			{
				tensors.try_emplace(remapName(name), cpu_tensor.clone());
			}
			else
			{
				tensors.try_emplace(name, cpu_tensor.clone());
			}
		}

		memFile.Close();

		return tensors;
	}
	catch (...)
	{
		memFile.Close();
		throw;
	}
}



inline void SafeTensorManager::Save(const SafeTensorManager::TensorMap& tensors, const std::string& filename, 
	const std::unordered_map<std::string, std::string>& metadata)
{
	if (tensors.size() > SAFETENSORS_MAX_TENSORS)
	{
		throw SafetensorsException("Number of tensors exceeds maximum allowed");
	}

	std::string header_json = "{";
	std::vector<char> data_buffer;
	size_t current_offset = 0;

	if (!metadata.empty())
	{
		header_json += "\"__metadata__\":{";
		bool first_meta = true;
		for (const auto& [key, value] : metadata)
		{
			validate_string_length(key, "Metadata key");
			validate_string_length(value, "Metadata value");

			if (!first_meta)
				header_json += ",";
			header_json += "\"" + key + "\":\"" + value + "\"";
			first_meta = false;
		}
		header_json += "},";
	}

	for (const auto& [name, tensor] : tensors)
	{
		validate_string_length(name, "Tensor name");

		auto cpu_tensor = tensor.to(torch::kCPU).contiguous();

		if (is_big_endian() && (cpu_tensor.dtype() == torch::kFloat16 || cpu_tensor.dtype() == torch::kFloat32 || cpu_tensor.dtype() == torch::kFloat64))
		{
			cpu_tensor = cpu_tensor.to(torch::kCPU, cpu_tensor.dtype(), /*non_blocking=*/false, /*copy=*/true);
			auto data_ptr = static_cast<char*>(cpu_tensor.data_ptr());
			for (int64_t i = 0; i < cpu_tensor.numel() * cpu_tensor.element_size(); i += cpu_tensor.element_size())
			{
				std::reverse(data_ptr + i, data_ptr + i + cpu_tensor.element_size());
			}
		}

		if (cpu_tensor.dim() > SAFETENSORS_MAX_DIM)
		{
			throw SafetensorsException("Tensor dimension exceeds maximum allowed");
		}

		auto dtype = get_safetensors_dtype(cpu_tensor.scalar_type());
		auto shape = cpu_tensor.sizes().vec();
		size_t tensor_size = cpu_tensor.numel() * cpu_tensor.element_size();

		if (header_json.length() > 1)
			header_json += ",";
		header_json += "\"" + name + "\":{";
		header_json += "\"dtype\":\"" + std::string(dtype) + "\",";
		header_json += "\"shape\":[";
		for (size_t i = 0; i < shape.size(); ++i)
		{
			if (i > 0)
				header_json += ",";
			header_json += std::to_string(shape[i]);
		}
		header_json += "],";
		header_json += "\"data_offsets\":[" + std::to_string(current_offset) + "," + std::to_string(current_offset + tensor_size) + "]";
		header_json += "}";

		const char* tensor_data = static_cast<const char*>(cpu_tensor.data_ptr());
		data_buffer.insert(data_buffer.end(), tensor_data, tensor_data + tensor_size);

		current_offset += tensor_size;
	}

	header_json += "}";
	uint64_t header_size = header_json.size();

	if (header_size > SAFETENSORS_MAX_METADATA_SIZE)
	{
		throw SafetensorsException("Metadata size exceeds maximum allowed size");
	}

	if (8 + header_size + data_buffer.size() > SAFETENSORS_MAX_FILE_SIZE)
	{
		throw SafetensorsException("Total file size exceeds maximum allowed size");
	}

	std::ofstream file(filename, std::ios::binary);
	if (!file)
	{
		throw SafetensorsException("Failed to open file for writing: " + filename);
	}

	uint64_t little_endian_header_size = header_size;
	if (is_big_endian())
	{
		little_endian_header_size = swap_endian(header_size);
	}
	file.write(reinterpret_cast<const char*>(&little_endian_header_size), sizeof(uint64_t));

	file.write(header_json.data(), header_json.size());

	file.write(data_buffer.data(), data_buffer.size());

	if (!file)
	{
		throw SafetensorsException("Failed to write to file: " + filename);
	}
}


