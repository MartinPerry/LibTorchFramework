# PyTorch training metrics viewer (PHP)

A standalone browser dashboard for metric JSON files named like:

```text
exPreCastModel_2026_08_27_20_08_train_3.json
```

## Server layout

```text
dashboard/
├── index.php
├── api.php
├── upload.php
├── upload_config.php
└── data/
    ├── run_001/
    │   ├── model_2026_08_27_12_00_train_0.json
    │   └── model_2026_08_27_12_10_train_1.json
    └── run_002/
        └── model_2026_08_28_09_00_train_0.json
```

Each immediate subfolder of `data` is treated as a separate experiment run and appears in the **Runs on this server** list. JSON files placed directly in `data` are also supported under the `(data root)` entry.

## Deploy

Copy `index.php`, `api.php`, `upload.php`, and the `data` directory to a PHP-enabled webserver. No database, Composer package, or JavaScript dependency is required. The PHP/webserver user must have write permission for `data`.

For local testing with PHP's built-in server:

```powershell
php -S 127.0.0.1:8000
```

Visit `http://127.0.0.1:8000`. The page first requests `api.php?action=runs` to show the available runs. It requests `api.php?action=metrics&run=...` only after the user selects a run.

The folder picker and drag-and-drop area remain available for inspecting files that are not on the server. Browser-selected files are processed locally and are not uploaded.

Within the selected server run folder, the page groups files by model name and run type, then orders checkpoints by numeric run ID. If several files have the same run ID, the newest checkpoint is shown.

## Configure metric uploads

Copy the example configuration and replace its placeholder with a long random secret:

```powershell
Copy-Item upload_config.example.php upload_config.php
```

Alternatively, configure the `DASHBOARD_UPLOAD_TOKEN` environment variable on the webserver. The environment variable takes precedence over `upload_config.php`.

Do not publish or commit `upload_config.php`. It is included in `.gitignore`.

The upload endpoint accepts an authenticated `POST` request at `upload.php`:

```json
{
  "run_id": "run_001",
  "data": {
    "loss": "0.4275924861",
    "psnr": "23.4453"
  }
}
```

Send the secret in the `X-Upload-Token` header. `loss` is required; every supplied metric must be numeric and finite. Optional `model` and `run_type` request fields control parts of the generated filename and default to `metrics` and `train`.

The endpoint validates the request, creates `data/<run_id>/` when needed, assigns the next checkpoint ID under a file lock, and atomically writes a filename such as:

```text
data/run_001/metrics_2026_08_28_17_07_train_42.json
```

## C++ uploader

[`metrics_uploader.cpp`](metrics_uploader.cpp) contains the reusable `uploadMetrics` function and a command-line example. It requires libcurl and C++11. The client explicitly forces HTTP/1.1 to avoid HTTP/3/QUIC connection-reset errors.

Linux:

```bash
g++ -std=c++11 -O2 metrics_uploader.cpp -lcurl -o metrics_uploader
```

Windows MinGW with libcurl installed:

```powershell
g++ -std=c++11 -O2 metrics_uploader.cpp -lcurl -o metrics_uploader.exe
```

Example:

```bash
./metrics_uploader \
  https://example.com/dashboard/upload.php \
  YOUR_UPLOAD_TOKEN \
  run_001 \
  loss=0.4275924861 psnr=23.4453 rmse=0.06725
```

To call it from training code, pass the existing map directly:

```cpp
std::unordered_map<std::string, std::string> metrics;
metrics["loss"] = "0.4275924861";
metrics["psnr"] = "23.4453";

std::string response;
long httpStatus = 0;
bool uploaded = uploadMetrics(
    "https://example.com/dashboard/upload.php",
    "YOUR_UPLOAD_TOKEN",
    "run_001",
    metrics,
    response,
    httpStatus
);
```
