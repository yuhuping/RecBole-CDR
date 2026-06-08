#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_NAME="AmazonClothUniCDRSource"
TARGET_NAME="AmazonSportUniCDRTarget"
SOURCE_DIR="${ROOT_DIR}/dataset/${SOURCE_NAME}"
TARGET_DIR="${ROOT_DIR}/dataset/${TARGET_NAME}"

CLOTH_TRAIN="${ROOT_DIR}/dataset/AmazonClothUniCDR/AmazonClothUniCDR.train.inter"
SPORT_TRAIN="${ROOT_DIR}/dataset/AmazonSportUniCDR/AmazonSportUniCDR.inter"
SPORT_TEST="${ROOT_DIR}/../DisenCDR/dataset/sport_cloth/test.txt"

mkdir -p "${SOURCE_DIR}" "${TARGET_DIR}"

cp "${CLOTH_TRAIN}" "${SOURCE_DIR}/${SOURCE_NAME}.inter"
cp "${SPORT_TRAIN}" "${TARGET_DIR}/${TARGET_NAME}.train.inter"

{
    printf 'user_id:token\titem_id:token\trating:float\ttimestamp:float\n'
    awk -F '\t' -v OFS='\t' -v offset=92612 \
        '{printf "%s\tsport::%s\t%.1f\t%.1f\n", $1, $2, $3 + 0, offset + NR - 1}' \
        "${SPORT_TEST}"
} > "${TARGET_DIR}/${TARGET_NAME}.valid.inter"

cp "${TARGET_DIR}/${TARGET_NAME}.valid.inter" \
   "${TARGET_DIR}/${TARGET_NAME}.test.inter"

printf 'Built Cloth -> Sport datasets:\n'
wc -l "${SOURCE_DIR}/${SOURCE_NAME}.inter" \
      "${TARGET_DIR}/${TARGET_NAME}.train.inter" \
      "${TARGET_DIR}/${TARGET_NAME}.valid.inter" \
      "${TARGET_DIR}/${TARGET_NAME}.test.inter"
