#!/bin/bash

for i in {1..19}
do
        cp -r find_rocks fcu$i
        chown -R fcu$i:fcu$i fcu$i/find_rocks
done    
