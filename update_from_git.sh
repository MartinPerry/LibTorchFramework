#!/usr/bin/env bash
set -e

REPO="https://github.com/MartinPerry/LibTorchFramework.git"
BRANCH="master"
SUBDIR="LibTorchFramework"
DEST="./LibTorchFramework"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

git clone --depth 1 --filter=blob:none --sparse --branch "$BRANCH" "$REPO" "$TMP/repo"

cd "$TMP/repo"
git sparse-checkout set "$SUBDIR"
cd - >/dev/null

mkdir -p "$DEST"

find "$TMP/repo/$SUBDIR" -mindepth 1 -maxdepth 1 ! -name 'build_debian.sh' -exec cp -rf {} "$DEST/" \;