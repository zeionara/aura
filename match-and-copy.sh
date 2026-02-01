#!/bin/bash

for file in assets/data/2025.12.17.01/annotations/*.json; do if [ -f "assets/data/sets-of-rules/$(basename $file | rev | cut -d '.' -f2- | rev).docx" ] && [ ! -f "assets/data/2026.01.22.01/source/$(basename $file | rev | cut -d '.' -f2- | rev).docx" ]; then cp $file assets/data/2026.01.24.01/annotations; fi; done

# for file in assets/data/2025.12.17.01/annotations/*.json; do
#   source_file="assets/data/sets-of-rules/$(basename "$file" | rev | cut -d '.' -f2- | rev).docx"
#   if [ ! -f "$source_file" ]; then
#     echo missing source file $source_file in sets-of-rules folder
#   fi
# done

# for file in assets/data/2025.12.17.01/annotations/*.json; do
#   set_of_rules_source_file="assets/data/sets-of-rules/$(basename "$file" | rev | cut -d '.' -f2- | rev).docx"
#   large_tables_source_file="assets/data/2026.01.22.01/source/$(basename "$file" | rev | cut -d '.' -f2- | rev).docx"
#   if [ ! -f "$set_of_rules_source_file" ] && [ ! -f "$large_tables_source_file" ]; then
#     echo missing source for $file
#   fi
# done
