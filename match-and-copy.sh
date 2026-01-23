#!/bin/bash

for file in assets/data/2025.12.17.01/annotations/*.json; do if [ -f "assets/data/sets-of-rules/$(basename $file | rev | cut -d '.' -f2- | rev).docx" ] && [ ! -f "assets/data/2026.01.22.01/source/$(basename $file | rev | cut -d '.' -f2- | rev).docx" ]; then cp $file assets/data/2026.01.24.01/annotations; fi; done
