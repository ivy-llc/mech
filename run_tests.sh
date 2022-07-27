#!/bin/bash -e
docker run --rm -it -v "$(pwd)":/mech unifyai/mech:latest python3 -m pytest ivy_mech_tests/

