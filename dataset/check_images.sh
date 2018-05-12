#!/usr/bin/env bash

find . -type f -iname "*.jpg" -o -iname "*.jpeg" \
	| xargs jpeginfo -c \
	| grep -E "WARNING|ERROR" \
	| cut -d " " -f 1
