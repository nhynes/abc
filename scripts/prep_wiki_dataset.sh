#!/usr/bin/env bash

CACHE_DIR="../data/_cache"
WIKI="simplewiki"
WIKI_DUMP_DATE="20171103"
WIKI_NAME="$WIKI-$WIKI_DUMP_DATE"
WIKI_DUMP_URL="https://dumps.wikimedia.org/$WIKI/$WIKI_DUMP_DATE/$WIKI_NAME-pages-articles.xml.bz2"
WIKI_DUMP_PATH="$CACHE_DIR/$WIKI_NAME.xml.bz2"

WIKI_EXTRACTOR="$HOME/tools/wikiextractor/WikiExtractor.py"

curl -L "$WIKI_DUMP_URL" > "$WIKI_DUMP_PATH"

$WIKI_EXTRACTOR --filter_disambig_pages --no-templates --min_text_length 100 "$WIKI_DUMP_PATH" -o - --json -q > "$CACHE_DIR/$WIKI_NAME.json"
