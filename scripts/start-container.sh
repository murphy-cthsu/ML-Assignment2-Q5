#!/bin/bash
set -e
./minizero/scripts/start-container.sh -v /mnt/nfs/work/zero/strength-detection:/strength-detection --image kds285/strength-detection $@
