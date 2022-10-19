#!/usr/bin/env bash

cd "${0%/*}/../../" || exit 1

if [[ ! -e "data/gutenberg_poetry" ]] ; then
	mkdir "data/cmu_dict"
fi