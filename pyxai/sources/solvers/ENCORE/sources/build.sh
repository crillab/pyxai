#!/bin/bash

set -e
set -u
set -o pipefail

opt=0

while getopts 'ds' OPTION
do
    case "$OPTION" in
        d)
            opt=1
            ;;
        s)
            opt=2
            ;;

    esac
done

curRep=$PWD
if [ $opt -eq 1 ]
then
   if ! [ -f 3rdParty/glucose-3.0/core/lib_debug.a ]
   then
       cd 3rdParty/glucose-3.0/core/
       make libd       
   fi
elif [ $opt -eq 2 ]
then
    if ! [ -f 3rdParty/glucose-3.0/core/lib_static.a ]
    then
        cd 3rdParty/glucose-3.0/core/
        make libr       
    fi
else
    if ! [ -f 3rdParty/glucose-3.0/core/lib_standard.a ]
    then
        cd 3rdParty/glucose-3.0/core/
        make libs
    fi
fi


cd $curRep
mkdir -p build
cd build
cmake .. -DBUILD_MODE=$opt
make -j
