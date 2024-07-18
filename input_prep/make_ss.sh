#!/bin/bash
# From: https://github.com/RosettaCommons/RoseTTAFold

if [ -z  "${BLASTMAT}" ]; then
    export BLASTMAT=$PIPE_DIR/blast-2.2.26/data/
fi

DATADIR="$CONDA_PREFIX/share/psipred_4.01/data"
echo $DATADIR

i_a3m="$1"
o_ss="$2"

ID=$(basename $i_a3m .a3m).tmp

$PIPE_DIR/csblast-2.2.3/bin/csbuild -i $i_a3m -I a3m -D $PIPE_DIR/csblast-2.2.3/data/K4000.crf -o $ID.chk -O chk

head -n 2 $i_a3m > $ID.fasta
echo $ID.chk > $ID.pn
echo $ID.fasta > $ID.sn

makemat -P $ID
psipred $ID.mtx $DATADIR/weights.dat $DATADIR/weights.dat2 $DATADIR/weights.dat3 > $ID.ss
psipass2 $DATADIR/weights_p2.dat 1 1.0 1.0 $i_a3m.csb.hhblits.ss2 $ID.ss > $ID.horiz

(
echo ">ss_pred"
grep "^Pred" $ID.horiz | awk '{print $2}'
echo ">ss_conf"
grep "^Conf" $ID.horiz | awk '{print $2}'
) | awk '{if(substr($1,1,1)==">") {print "\n"$1} else {printf "%s", $1}} END {print ""}' | sed "1d" > $o_ss

rm ${i_a3m}.csb.hhblits.ss2
rm $ID.*
