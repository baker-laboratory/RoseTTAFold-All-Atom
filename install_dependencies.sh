#!/bin/bash
# From: https://github.com/RosettaCommons/RoseTTAFold

# install external program not supported by conda installation
case "$(uname -s)" in
    Linux*)     platform=linux;;
    Darwin*)    platform=macosx;;
    *)          echo "unsupported OS type. exiting"; exit 1
esac
echo "Installing dependencies for ${platform}..."

# the cs-blast platform descriptoin includes the width of memory addresses
# we expect a 64-bit operating system
if [[ ${platform} == "linux" ]]; then
    platform=${platform}64
fi

# download cs-blast
echo "Downloading cs-blast ..."
wget http://wwwuser.gwdg.de/~compbiol/data/csblast/releases/csblast-2.2.3_${platform}.tar.gz -O csblast-2.2.3.tar.gz
mkdir -p csblast-2.2.3
tar xf csblast-2.2.3.tar.gz -C csblast-2.2.3 --strip-components=1

mv csblast-2.2.3 $CONDA_PREFIX/share/

rm -f csblast-2.2.3.tar.gz

# https://github.com/baker-laboratory/RoseTTAFold-All-Atom/issues/5#issuecomment-1990991606
# https://github.com/RosettaCommons/RoseTTAFold/issues/13#issuecomment-1405850297
echo "Downloading blast ..."
wget https://ftp.ncbi.nlm.nih.gov/blast/executables/legacy.NOTSUPPORTED/2.2.26/blast-2.2.26-x64-linux.tar.gz
mkdir -p blast-2.2.26
tar -xf blast-2.2.26-x64-linux.tar.gz -C blast-2.2.26

mv blast-2.2.26 $CONDA_PREFIX/share/
rm -f blast-2.2.26-x64-linux.tar.gz