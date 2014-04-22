#!/bin/bash

dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
savedir=$dir/saved/$1

cp -v $savedir/* $dir/computed
