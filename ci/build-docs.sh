#!/bin/bash
doxygen Doxyfile
rsync -avz -e'ssh -v' --numeric-ids --delete doc/html/* tmdoel@storm:/cs/sys/www0/marine/html/cmic.cs.ucl.ac.uk/giftsurg/dybaorf  2>&1
