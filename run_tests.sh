#!/bin/bash -e
docker run --rm -it -v "$(pwd)":/ivy_mech unifyai/ivy-mech:latest python3 -m pytest ivy_mech_tests/
