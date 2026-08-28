<?php

/* JSON API for the training dashboard. Compatible with PHP 5.3+. */

ini_set('display_errors', '0');
header('Content-Type: application/json; charset=utf-8');
header('Cache-Control: no-store, no-cache, must-revalidate');

function sendJson($payload, $statusCode)
{
    if ($statusCode !== 200) {
        if ($statusCode === 400) {
            header('HTTP/1.1 400 Bad Request');
        } elseif ($statusCode === 404) {
            header('HTTP/1.1 404 Not Found');
        } else {
            header('HTTP/1.1 500 Internal Server Error');
        }
    }

    $json = json_encode($payload);
    if ($json === false) {
        header('HTTP/1.1 500 Internal Server Error');
        echo '{"error":"Could not encode API response as JSON"}';
    } else {
        echo $json;
    }
    exit;
}

function jsonErrorText()
{
    $error = json_last_error();
    if ($error === JSON_ERROR_DEPTH) {
        return 'Maximum JSON depth exceeded';
    }
    if ($error === JSON_ERROR_STATE_MISMATCH) {
        return 'Invalid or malformed JSON';
    }
    if ($error === JSON_ERROR_CTRL_CHAR) {
        return 'Unexpected control character in JSON';
    }
    if ($error === JSON_ERROR_SYNTAX) {
        return 'JSON syntax error';
    }
    if (defined('JSON_ERROR_UTF8') && $error === JSON_ERROR_UTF8) {
        return 'Invalid UTF-8 in JSON';
    }
    return 'Unknown JSON error';
}

function findRunSources($dataRoot)
{
    $sources = array();
    if (!is_dir($dataRoot)) {
        return $sources;
    }

    $directories = @glob($dataRoot . DIRECTORY_SEPARATOR . '*', GLOB_ONLYDIR);
    if ($directories === false) {
        $directories = array();
    }
    natcasesort($directories);
    foreach ($directories as $directory) {
        $sources[basename($directory)] = $directory;
    }

    $rootFiles = @glob($dataRoot . DIRECTORY_SEPARATOR . '*.json');
    if ($rootFiles !== false && count($rootFiles) > 0) {
        $sources['__data_root__'] = $dataRoot;
    }
    return $sources;
}

function jsonFilesIn($directory)
{
    $files = @glob($directory . DIRECTORY_SEPARATOR . '*.json');
    if ($files === false) {
        return array();
    }
    natcasesort($files);
    return array_values($files);
}

function listRuns($sources)
{
    $runs = array();
    foreach ($sources as $id => $directory) {
        $jsonFiles = jsonFilesIn($directory);
        $latestModified = 0;
        foreach ($jsonFiles as $jsonFile) {
            $modified = @filemtime($jsonFile);
            if ($modified !== false && $modified > $latestModified) {
                $latestModified = $modified;
            }
        }
        $runs[] = array(
            'id' => $id,
            'label' => $id === '__data_root__' ? '(data root)' : $id,
            'fileCount' => count($jsonFiles),
            'modifiedAt' => $latestModified > 0 ? date('c', $latestModified) : null
        );
    }
    return $runs;
}

function loadMetricFile($path, $runLabel, &$files, &$errors)
{
    $raw = @file_get_contents($path);
    if ($raw === false) {
        $errors[] = array('name' => basename($path), 'error' => 'Could not read file');
        return;
    }

    $decoded = json_decode($raw);
    if (json_last_error() !== JSON_ERROR_NONE) {
        $errors[] = array('name' => basename($path), 'error' => jsonErrorText());
        return;
    }
    if (!is_object($decoded)) {
        $errors[] = array('name' => basename($path), 'error' => 'Root must be a JSON object');
        return;
    }

    $data = get_object_vars($decoded);
    $hasNumericMetric = false;
    foreach ($data as $value) {
        if (is_int($value) || is_float($value) || (is_string($value) && is_numeric($value))) {
            $hasNumericMetric = true;
            break;
        }
    }
    if (!$hasNumericMetric) {
        $errors[] = array('name' => basename($path), 'error' => 'No numeric metrics found');
        return;
    }

    $relativePath = substr($path, strlen(dirname(__FILE__)) + 1);
    $files[] = array(
        'name' => basename($path),
        'path' => str_replace(DIRECTORY_SEPARATOR, '/', $relativePath),
        'runGroup' => $runLabel,
        'data' => $data
    );
}

$dataRoot = dirname(__FILE__) . DIRECTORY_SEPARATOR . 'data';
$sources = findRunSources($dataRoot);
$action = isset($_GET['action']) ? (string) $_GET['action'] : '';

if ($action === 'runs') {
    sendJson(array('runs' => listRuns($sources)), 200);
}

if ($action === 'metrics') {
    $requestedRun = isset($_GET['run']) ? (string) $_GET['run'] : '';
    if ($requestedRun === '' || !array_key_exists($requestedRun, $sources)) {
        sendJson(array('files' => array(), 'errors' => array(array('error' => 'Unknown or missing run folder'))), 400);
    }

    $files = array();
    $errors = array();
    $runLabel = $requestedRun === '__data_root__' ? '(data root)' : $requestedRun;
    $jsonFiles = jsonFilesIn($sources[$requestedRun]);
    foreach ($jsonFiles as $jsonFile) {
        loadMetricFile($jsonFile, $runLabel, $files, $errors);
    }
    sendJson(array('files' => $files, 'errors' => $errors), 200);
}

sendJson(array('error' => 'Unknown API action'), 400);
