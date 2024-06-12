#! /bin/bash

QUEUE=qa-interactive
COMPUTE_GROUP="compute-$USER"
OUTPUT_PATH="/scratch1/fs1/$USER/outputs"
CONFIG_PATH="/home/$USER/configs"
MEMORY="64GB"
SLOTS=4
IMAGE_TAG="docker.io/melonistic/rosetta-fold:1.0.0-beta"

DOCKER_VOLUMES="$OUTPUT_PATH:/app/RoseTTAFold/RoseTTAFold-All-Atom/outputs \
$CONFIG_PATH:/app/RoseTTAFold/RoseTTAFold-All-Atom/rf2aa/config/inference \
/scratch1/fs1/ris/references/RoseTTAFold/bfd:/app/RoseTTAFold/RoseTTAFold-All-Atom/bfd \
/scratch1/fs1/ris/references/RoseTTAFold/pdb100_2021Mar03:/app/RoseTTAFold/RoseTTAFold-All-Atom/pdb100_2021Mar03 \
/scratch1/fs1/ris/references/RoseTTAFold/UniRef30_2020_06:/app/RoseTTAFold/RoseTTAFold-All-Atom/UniRef30_2020_06 \
$LSF_DOCKER_VOLUMES"

PATH="$PATH:/app/RoseTTAFold/mambaforge/condabin:/app/RoseTTAFold/mambaforge/bin" \
LSF_DOCKER_VOLUMES="$DOCKER_VOLUMES" \
bsub \
-n $SLOTS \
-M $MEMORY \
-R "rusage[mem=$MEMORY]" \
-R "gpuhost" \
-G $COMPUTE_GROUP \
-q $QUEUE \
-gpu 'num=1' \
-Is -a "docker($IMAGE_TAG)" /bin/bash
