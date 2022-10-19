#!/usr/bin/env bash

cd "${0%/*}/../../" || exit 1

if [[ ! -e "data/gutenberg_poetry" ]] ; then
	mkdir "data/gutenberg_poetry"
fi

cd "data/gutenberg_poetry" || exit 1

wget -O gutenberg_poetry_corpus.gz http://static.decontextualize.com/gutenberg-poetry-v001.ndjson.gz
gzip -d ./gutenberg_poetry_corpus.gz
wget -O gutenber_dammit.zip http://static.decontextualize.com/gutenberg-dammit-files-v002.zip
touch gutenberg_metadata.json
unzip -p gutenber_dammit.zip gutenberg-dammit-files/gutenberg-metadata.json > gutenberg_metadata.json
rm gutenber_dammit.zip
touch LICENSE
echo "Copyright 2018 Allison Parrish

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the \"Software\"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE." > LICENSE