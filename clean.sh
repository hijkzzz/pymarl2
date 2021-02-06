#!/bin/bash

pkill -u $(whoami) python
pkill Main_Thread

# kill -HUP $( ps -A -ostat,ppid | grep -e '^[Zz]' | awk '{print $2}')

if [ "$1" = "all" ]; then
    rm -rf results/sacred
fi
