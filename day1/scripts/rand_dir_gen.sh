#!/bin/bash

# Decimal ASCII codes (see man ascii)
ARR=( {48..57} {65..90} {97..122} )

# Array count
arrcount=${#ARR[@]}

# return a random string
get_rand_dir(){
    for ((i=1; i<$((RANDOM%30)); i++)) {
        printf \\$(printf '%03o' ${ARR[RANDOM%arrcount]});
    }
}

dir=/tmp/

# appending random characters to make a hierarchy
for ((i=0; i<$((RANDOM%15)); i++)) {
    dir+="$(get_rand_dir)/"
}

echo $dir
mkdir -p "$dir"

oldir=$(echo "$dir" | cut -d '/' -f1-3)

while [[ $dir ]]; do
    dir=${dir%/*}
    cd $dir
    for ((i=0; i<$((RANDOM%15)); i++)) {
        mkdir &>/dev/null -p $(get_rand_dir)
    }
done

tree "$oldir"
