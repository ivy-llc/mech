#!/bin/bash -e
docker run --rm -it -v "$(pwd)":/mech unifyai/ivy-mech:latest python3 -m pytest ivy/ivy_mech_tests/
