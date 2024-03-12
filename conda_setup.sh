#!/bin/bash

################
### Defaults ###
################
ENV_NAME="helicast"
PYTHON_VERSION=3.10

#################
### Functions ###
#################
show_help() {
cat << EOF
Usage: ${0##*/} [--env-name ENV_NAME] [--cpu-only] [--help]
This script installs the conda environment for the Helicast development repo.

    --env-name ENV_NAME     Specify a conda environment name name. Default is '$ENV_NAME'.
    --help                  Display this help and exit.
EOF
}


#####################
### Option parser ###
#####################
TEMP=$(getopt -o '' --long env-name:,help -- "$@")

# Exit if options are not properly provided
if [ $? != 0 ]; then
    echo ''
    show_help
    exit 1
fi

eval set -- "$TEMP"
while true; do
    case "$1" in
        --env-name)
            ENV_NAME="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Programming error"
            exit 3
            ;;
    esac
done

##############
### Script ###
##############
# Check that we are NOT within a conda environment
if [ ${CONDA_SHLVL-0} != "0" ];
then
    echo -ne "\033[1;31mERROR: "
    echo -ne " \033[1;31mYou are trying to install a conda environment but you're already in one!"
    echo -e " \033[1;31mPlease deactivate before executing the script."
    exit 1
fi

conda create -n ${ENV_NAME} python=$PYTHON_VERSION pip --yes

conda run -n ${ENV_NAME} --live-stream pip install -e '.[dev]'