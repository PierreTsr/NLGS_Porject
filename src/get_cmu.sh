#!/usr/bin/env bash

cd "${0%/*}/../" || exit 1

if [[ ! -e "data/cmu_dict" ]] ; then
	mkdir "data/cmu_dict"
fi

cd "data/cmu_dict" || exit 1

wget -O "LICENSE" "https://raw.githubusercontent.com/Alexir/CMUdict/master/LICENSE"
wget -O "cmudict-0.7b" "https://raw.githubusercontent.com/Alexir/CMUdict/master/cmudict-0.7b"