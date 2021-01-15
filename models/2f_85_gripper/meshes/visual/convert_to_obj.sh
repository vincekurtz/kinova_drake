#!/usr/bin/env bash

##
#
# A quick script for converting other mesh filetypes to .obj 
# (since Drake can only handle obj meshes) using meshlab. 
#
##

for file in ./*.dae; do
    echo "$file ===> ${file%.*}.obj "
    meshlabserver -i "$file" -o "${file%.*}.obj"
done
