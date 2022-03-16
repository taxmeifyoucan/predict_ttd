#!/bin/bash

#Step in seconds, data granuality 
GRAN=3600
RPC="http://127.0.0.1:8545/"

if [ -f "result.csv" ]; 
then
read ROW BNdec TSfirst < <(echo $(tail -n 1 result.csv | awk -F, '{ print $1 "\n" $2 "\n" $6 }'))
TSnow=`date +%s`
if [ $(($TSnow - $TSfirst)) -lt $GRAN ]; then
python3 polynom.py; exit; fi
BN=`python3 blockbytime.py $(($TSfirst+$GRAN)) $BNdec $(($BNdec+10))` 
BN=`printf '%X\n' $BN`
else
echo "Row,BlockNumber,TTD,Difficulty,Time,UnixTimestamp" > result.csv
BNdec=$1
BN=`printf '%X\n' $BNdec`
fi

read BNlatest TSlatest < <(echo $(curl -s -X POST $RPC -H "Content-Type: application/json" --data '{"jsonrpc":"2.0", "method":"eth_getBlockByNumber", "params":["latest",false], "id":1}' | jq -r '.result.number, .result.timestamp'))
BNlatest=`printf "%d" $BNlatest`
TSlatest=`printf "%d" $TSlatest`
let STEP="$GRAN / (($TSlatest - $TSfirst) / ($BNlatest-$BNdec))"
let NPOINTS="($TSlatest - $TSfirst) / $GRAN"

for (( c=$ROW+1; c<=$ROW+$NPOINTS; c++ ))
do

read TTD DIFF TIME < <(echo $(curl -s -X POST $RPC \
     -H "Content-Type: application/json" --data '{"jsonrpc":"2.0", "method":"eth_getBlockByNumber", "params":["'0x$BN'", false], "id":1}' \
     | jq -r '.result.totalDifficulty, .result.difficulty, .result.timestamp'))

BN=`echo "obase=10; ibase=16; $BN" | bc`
TIME=`echo $TIME |  python3 -c 'import sys; print(int(sys.stdin.read(), 16))'`

echo $c | tr '\n' ',' >> result.csv
echo $BN | tr '\n' ',' >> result.csv
echo $TTD | python3 -c 'import sys; print(int(sys.stdin.read(), 16))' | rev | cut -c5- | rev | tr '\n' ',' >> result.csv
echo $DIFF |  python3 -c 'import sys; print(int(sys.stdin.read(), 16))' | tr '\n' ',' >> result.csv
echo $TIME | gawk '{print strftime("%m/%d/%Y %H:%M:%S", $0)}' | tr '\n' ','  >> result.csv
echo $TIME  >> result.csv

if [ $c -lt $(($ROW+$NPOINTS)) ]; then
NEXT_TS=`printf "%d\n" $(($TIME + $GRAN))`
NEXT_BN=`printf "%d\n" $(($STEP+$BN))`
BN=`python3 blockbytime.py $NEXT_TS $NEXT_BN $(($NEXT_BN+10))` 
BN=`printf '%X\n' $BN`
fi

done

python3 polynom.py 
