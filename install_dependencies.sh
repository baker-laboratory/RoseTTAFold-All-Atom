#!/bin/bash
# From: https://github.com/RosettaCommons/RoseTTAFold

# install external program not supported by conda installation
case "$(uname -s)" in
    Linux*)     platform=linux64;; #We might need to update this to get the 32 bit version when necessary
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
wget https://wwwuser.gwdg.de/~compbiol/data/csblast/releases/csblast-2.2.3_${platform}.tar.gz -O csblast-2.2.3.tar.gz
mkdir -p csblast-2.2.3
tar xf csblast-2.2.3.tar.gz -C csblast-2.2.3 --strip-components=1
