<?php

/* Authenticated JSON upload endpoint. Compatible with PHP 5.3+. */

ini_set('display_errors', '0');
header('Content-Type: application/json; charset=utf-8');
header('Cache-Control: no-store, no-cache, must-revalidate');

//==================================================================================================
// Methods
//==================================================================================================

function uploadResponse($payload, $statusCode)
{
    if ($statusCode === 201) {
        header('HTTP/1.1 201 Created');
    } elseif ($statusCode === 400) {
        header('HTTP/1.1 400 Bad Request');
    } elseif ($statusCode === 401) {
        header('HTTP/1.1 401 Unauthorized');
    } elseif ($statusCode === 405) {
        header('HTTP/1.1 405 Method Not Allowed');
    } elseif ($statusCode === 409) {
        header('HTTP/1.1 409 Conflict');
    } elseif ($statusCode === 413) {
        header('HTTP/1.1 413 Payload Too Large');
    } elseif ($statusCode === 507) {
        header('HTTP/1.1 507 Insufficient Storage');
    } elseif ($statusCode !== 200) {
        header('HTTP/1.1 500 Internal Server Error');
    }

    $json = json_encode($payload);
    if ($json === false) {
        header('HTTP/1.1 500 Internal Server Error');
        echo '{"ok":false,"error":"Could not encode response"}';
    } else {
        echo $json;
    }
    exit;
}

function secureTokenEquals($expected, $provided)
{
    if (!is_string($expected) || !is_string($provided)) {
        return false;
    }
    $expectedLength = strlen($expected);
    if ($expectedLength !== strlen($provided)) {
        return false;
    }
    $difference = 0;
    for ($index = 0; $index < $expectedLength; $index++) {
        $difference |= ord($expected[$index]) ^ ord($provided[$index]);
    }
    return $difference === 0;
}

function configuredUploadToken()
{
    $environmentToken = getenv('DASHBOARD_UPLOAD_TOKEN');
    if (is_string($environmentToken) && $environmentToken !== '') {
        return $environmentToken;
    }

    $configPath = dirname(__FILE__) . DIRECTORY_SEPARATOR . 'upload_config.php';
    if (!is_file($configPath)) {
        return '';
    }
    $config = include $configPath;
    if (!is_array($config) || !isset($config['upload_token'])) {
        return '';
    }
    return (string) $config['upload_token'];
}

function requestUploadToken()
{
    if (isset($_SERVER['HTTP_X_UPLOAD_TOKEN'])) {
        return (string) $_SERVER['HTTP_X_UPLOAD_TOKEN'];
    }
    if (isset($_SERVER['HTTP_AUTHORIZATION'])) {
        $authorization = (string) $_SERVER['HTTP_AUTHORIZATION'];
        if (stripos($authorization, 'Bearer ') === 0) {
            return substr($authorization, 7);
        }
    }
    return '';
}

function safeFilePart($value, $fallback)
{
    $cleaned = preg_replace('/[^A-Za-z0-9.-]+/', '_', (string) $value);
    $cleaned = trim($cleaned, '._-');
    if ($cleaned === '') {
        return $fallback;
    }
    return substr($cleaned, 0, 80);
}

function nextCheckpointId($runDirectory)
{
    $maximum = -1;
    $files = @glob($runDirectory . DIRECTORY_SEPARATOR . '*.json');
    if ($files === false) {
        return 0;
    }
    foreach ($files as $file) {
        $matches = array();
        if (preg_match('/_(\d+)\.json$/i', basename($file), $matches)) {
            $value = (int) $matches[1];
            if ($value > $maximum) {
                $maximum = $value;
            }
        }
    }
    return $maximum + 1;
}

function validRunId($runId)
{
    return is_string($runId) && preg_match('/^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/', $runId);
}

function ensureRunDirectory($runId)
{
    $dataRoot = dirname(__FILE__) . DIRECTORY_SEPARATOR . 'data';
    $runDirectory = $dataRoot . DIRECTORY_SEPARATOR . $runId;
    if (!is_dir($dataRoot) && !@mkdir($dataRoot, 0775, true)) {
        return false;
    }
    if (!is_dir($runDirectory) && !@mkdir($runDirectory, 0775, true)) {
        return false;
    }
    return $runDirectory;
}

function imageExtension($bytes)
{
    if (strlen($bytes) >= 3 && substr($bytes, 0, 3) === "\xFF\xD8\xFF") {
        return 'jpg';
    }
    $signature = substr($bytes, 0, 6);
    if ($signature === 'GIF87a' || $signature === 'GIF89a') {
        return 'gif';
    }
    return '';
}

function reserveImageUploadSpace($dataRoot, $imageDirectory, $imageSize)
{
    // Leave 5 MiB available after saving the incoming image.
    $requiredSpace = 5 * 1024 * 1024 + $imageSize;
    $freeSpace = @disk_free_space($imageDirectory);
    if ($freeSpace === false) {
        return false;
    }
    if ($freeSpace >= $requiredSpace) {
        return true;
    }

    $imageDirectories = array($dataRoot . DIRECTORY_SEPARATOR . 'img');
    $runs = @glob($dataRoot . DIRECTORY_SEPARATOR . '*', GLOB_ONLYDIR);
    if ($runs !== false) {
        foreach ($runs as $run) {
            if (!is_link($run)) {
                $imageDirectories[] = $run . DIRECTORY_SEPARATOR . 'img';
            }
        }
    }
    $images = array();
    foreach ($imageDirectories as $directory) {
        if (is_link($directory) || !is_dir($directory)) {
            continue;
        }
        $files = @glob($directory . DIRECTORY_SEPARATOR . '*');
        if ($files === false) {
            continue;
        }
        foreach ($files as $file) {
            // Only prune dashboard images; exclude links, metrics and temporary files.
            if (is_link($file) || !is_file($file) ||
                !preg_match('/^\d+_(train|test|valid)_\d+\.(jpe?g|gif)$/i', basename($file))) {
                continue;
            }
            $modified = @filemtime($file);
            if ($modified !== false) {
                $images[] = array('path' => $file, 'modified' => $modified);
            }
        }
    }
    usort($images, function ($left, $right) {
        if ($left['modified'] === $right['modified']) {
            return strcmp($left['path'], $right['path']);
        }
        return $left['modified'] < $right['modified'] ? -1 : 1;
    });

    foreach ($images as $image) {
        if (!@unlink($image['path'])) {
            continue;
        }
        clearstatcache();
        $freeSpace = @disk_free_space($imageDirectory);
        if ($freeSpace === false) {
            return false;
        }
        if ($freeSpace >= $requiredSpace) {
            return true;
        }
    }
    return false;
}


//==================================================================================================
// Script main part
//==================================================================================================

if (!isset($_SERVER['REQUEST_METHOD']) || $_SERVER['REQUEST_METHOD'] !== 'POST') {
    header('Allow: POST');
    uploadResponse(array('ok' => false, 'error' => 'Use POST with a JSON request body'), 405);
}

$expectedToken = configuredUploadToken();
if ($expectedToken === '' || $expectedToken === 'CHANGE_THIS_TO_A_LONG_RANDOM_SECRET') {
    uploadResponse(array('ok' => false, 'error' => 'Upload token is not configured on the server'), 500);
}
if (!secureTokenEquals($expectedToken, requestUploadToken())) {
    uploadResponse(array('ok' => false, 'error' => 'Invalid upload token'), 401);
}

$action = isset($_GET['action']) ? (string) $_GET['action'] : 'metrics';
$runId = isset($_GET['run_id']) ? (string) $_GET['run_id'] : '';
$runType = isset($_GET['run_type']) ? strtolower((string) $_GET['run_type']) : '';
$runIndexText = isset($_GET['run_index']) ? (string) $_GET['run_index'] : '';

//==================================================================================================
// Image
//==================================================================================================
if ($action === 'image') {
    $imageContentLength = isset($_SERVER['CONTENT_LENGTH']) ? (int) $_SERVER['CONTENT_LENGTH'] : 0;
    if ($imageContentLength <= 0) {
        uploadResponse(array('ok' => false, 'error' => 'Image request body is empty'), 400);
    }
    if ($imageContentLength > 26214400) {
        uploadResponse(array('ok' => false, 'error' => 'Image exceeds 25 MiB'), 413);
    }

    $imageRunId = isset($_GET['run_id']) ? (string) $_GET['run_id'] : '';    
    $imageIndexText = isset($_GET['img_index']) ? (string) $_GET['img_index'] : '';    
    if (!validRunId($runId)) {
        uploadResponse(array('ok' => false, 'error' => 'run_id contains invalid characters'), 400);
    }
    if ($runType !== 'train' && $runType !== 'test' && $runType !== 'valid') {
        uploadResponse(array('ok' => false, 'error' => 'run_type must be train, test, or valid'), 400);
    }
    if (!preg_match('/^\d{1,10}$/', $imageIndexText) || !preg_match('/^\d{1,10}$/', $runIndexText)) {
        uploadResponse(array('ok' => false, 'error' => 'img_index and run_index must be non-negative integers'), 400);
    }

    $imageBytes = file_get_contents('php://input');
    if ($imageBytes === false || $imageBytes === '') {
        uploadResponse(array('ok' => false, 'error' => 'Could not read image body'), 400);
    }
    $imageSize = strlen($imageBytes);
    if ($imageSize > 26214400) {
        uploadResponse(array('ok' => false, 'error' => 'Image exceeds 25 MiB'), 413);
    }
    $extension = imageExtension($imageBytes);
    if ($extension === '') {
        uploadResponse(array('ok' => false, 'error' => 'Only valid JPG and GIF image data is accepted'), 400);
    }
    if (function_exists('getimagesizefromstring') && @getimagesizefromstring($imageBytes) === false) {
        uploadResponse(array('ok' => false, 'error' => 'Image data is corrupt'), 400);
    }

    $imageRunDirectory = ensureRunDirectory($runId);
    if ($imageRunDirectory === false) {
        uploadResponse(array('ok' => false, 'error' => 'Could not create run directory'), 500);
    }
    $imageDirectory = $imageRunDirectory . DIRECTORY_SEPARATOR . 'img';
    if (!is_dir($imageDirectory) && !@mkdir($imageDirectory, 0775, true)) {
        uploadResponse(array('ok' => false, 'error' => 'Could not create image directory'), 500);
    }
    $imageFileName = (int) $imageIndexText . '_' . $runType . '_' . (int) $runIndexText . '.' . $extension;
    $imageFinalPath = $imageDirectory . DIRECTORY_SEPARATOR . $imageFileName;
    // Keep duplicate checks, global pruning and writes in one critical section.
    // PHP releases this lock when uploadResponse exits, including on errors.
    $dataRoot = dirname(__FILE__) . DIRECTORY_SEPARATOR . 'data';
    $imageLock = @fopen($dataRoot . DIRECTORY_SEPARATOR . '.image-upload.lock', 'c');
    if ($imageLock === false || !flock($imageLock, LOCK_EX)) {
        uploadResponse(array('ok' => false, 'error' => 'Could not lock image uploads'), 500);
    }
    clearstatcache();
    if (is_file($imageFinalPath)) {
        uploadResponse(array('ok' => false, 'error' => 'Image already exists'), 409);
    }
    if (!reserveImageUploadSpace($dataRoot, $imageDirectory, $imageSize)) {
        uploadResponse(array('ok' => false, 'error' => 'Could not free enough disk space for image and 5 MiB reserve'), 507);
    }
    $imageTemporaryPath = @tempnam($imageDirectory, '.image-');
    if ($imageTemporaryPath === false || @file_put_contents($imageTemporaryPath, $imageBytes, LOCK_EX) !== $imageSize) {
        if ($imageTemporaryPath !== false) {
            @unlink($imageTemporaryPath);
        }
        uploadResponse(array('ok' => false, 'error' => 'Could not write image file'), 500);
    }
    if (!@rename($imageTemporaryPath, $imageFinalPath)) {
        @unlink($imageTemporaryPath);
        uploadResponse(array('ok' => false, 'error' => 'Could not finalize image file'), 500);
    }
    @chmod($imageFinalPath, 0664);
    uploadResponse(array(
        'ok' => true,
        'run_id' => $runId,
        'image_index' => (int) $imageIndexText,
        'run_type' => $runType,
        'run_index' => (int) $runIndexText,
        'file' => 'data/' . $runId . '/img/' . $imageFileName
    ), 201);
}

//==================================================================================================

if ($action !== 'metrics') {
    uploadResponse(array('ok' => false, 'error' => 'Unknown upload action'), 400);
}

$contentLength = isset($_SERVER['CONTENT_LENGTH']) ? (int) $_SERVER['CONTENT_LENGTH'] : 0;
if ($contentLength > 1048576) {
    uploadResponse(array('ok' => false, 'error' => 'Request body exceeds 1 MiB'), 413);
}

$rawBody = file_get_contents('php://input');
if ($rawBody === false || $rawBody === '') {
    uploadResponse(array('ok' => false, 'error' => 'Request body is empty'), 400);
}
$request = json_decode($rawBody);
if (json_last_error() !== JSON_ERROR_NONE || !is_object($request)) {
    uploadResponse(array('ok' => false, 'error' => 'Request body must be a JSON object'), 400);
}
if (!isset($request->run_id) || !is_string($request->run_id)) {
    uploadResponse(array('ok' => false, 'error' => 'run_id must be a string'), 400);
}

$runId = isset($request->run_id) ? $request->run_id : $runId;
if (!validRunId($runId)) {
    uploadResponse(array('ok' => false, 'error' => 'run_id contains invalid characters'), 400);
}
if (!isset($request->data) || !is_object($request->data)) {
    uploadResponse(array('ok' => false, 'error' => 'data must be a JSON object of metric key/value pairs'), 400);
}

$inputMetrics = get_object_vars($request->data);
$metrics = array();
foreach ($inputMetrics as $key => $value) {
    if (!is_string($key) || !preg_match('/^[A-Za-z0-9_.-]{1,128}$/', $key)) {
        uploadResponse(array('ok' => false, 'error' => 'Metric name contains invalid characters'), 400);
    }
    if (!(is_int($value) || is_float($value) || (is_string($value) && is_numeric($value)))) {
        uploadResponse(array('ok' => false, 'error' => 'Metric values must be numeric'), 400);
    }
    $numericValue = (float) $value;
    if (is_nan($numericValue) || is_infinite($numericValue)) {
        uploadResponse(array('ok' => false, 'error' => 'Metric values must be finite'), 400);
    }
    $metrics[$key] = $numericValue;
}
if (!isset($metrics['loss'])) {
    uploadResponse(array('ok' => false, 'error' => 'The required loss metric is missing'), 400);
}

$dataRoot = dirname(__FILE__) . DIRECTORY_SEPARATOR . 'data';
$runDirectory = $dataRoot . DIRECTORY_SEPARATOR . $runId;
if (!is_dir($dataRoot) && !@mkdir($dataRoot, 0775, true)) {
    uploadResponse(array('ok' => false, 'error' => 'Could not create data directory'), 500);
}
if (!is_dir($runDirectory) && !@mkdir($runDirectory, 0775, true)) {
    uploadResponse(array('ok' => false, 'error' => 'Could not create run directory'), 500);
}

$lockPath = $runDirectory . DIRECTORY_SEPARATOR . '.upload.lock';
$lock = @fopen($lockPath, 'c');
if ($lock === false || !flock($lock, LOCK_EX)) {
    if ($lock !== false) {
        fclose($lock);
    }
    uploadResponse(array('ok' => false, 'error' => 'Could not lock run directory'), 500);
}

$checkpointId = nextCheckpointId($runDirectory);
$model = isset($request->model) ? safeFilePart($request->model, 'metrics') : 'metrics';
$timestamp = date('Y_m_d_H_i');
$fileName = $model . '_' . $timestamp . '_' . $runType . '_' . $checkpointId . '.json';
$finalPath = $runDirectory . DIRECTORY_SEPARATOR . $fileName;
$temporaryPath = @tempnam($runDirectory, '.upload-');
$encodedMetrics = json_encode($metrics);

if ($temporaryPath === false || $encodedMetrics === false || @file_put_contents($temporaryPath, $encodedMetrics, LOCK_EX) === false) {
    if ($temporaryPath !== false && is_file($temporaryPath)) {
        @unlink($temporaryPath);
    }
    flock($lock, LOCK_UN);
    fclose($lock);
    uploadResponse(array('ok' => false, 'error' => 'Could not write metric file'), 500);
}
if (!@rename($temporaryPath, $finalPath)) {
    @unlink($temporaryPath);
    flock($lock, LOCK_UN);
    fclose($lock);
    uploadResponse(array('ok' => false, 'error' => 'Could not finalize metric file'), 500);
}

@chmod($finalPath, 0664);
flock($lock, LOCK_UN);
fclose($lock);

uploadResponse(array(
    'ok' => true,
    'run_id' => $runId,
    'checkpoint_id' => $checkpointId,
    'file' => 'data/' . $runId . '/' . $fileName,
    'metric_count' => count($metrics)
), 201);
