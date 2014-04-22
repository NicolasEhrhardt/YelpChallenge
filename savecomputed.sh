#!/bin/bash

dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
newdir=$dir/saved/$1

mkdir -p $newdir
cp -v $dir/computed/* $newdir
