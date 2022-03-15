#!/bin/bash

#Step in seconds, data granuality 
GRAN=3600
RPC="https://bordel.xyz/"

if [ -f "result.csv" ]; 
then
BNdec=`tail -n 1 result.csv | awk -F, '{ print $2 }'`
BN=`printf '%X\n' $BNdec`
ROW=`tail -n 1 result.csv | awk -F, '{ print $1 }'`
else
echo "Row,BlockNumber,TTD,Difficulty,Time,UnixTimestamp" > result.csv
BNdec=$1
BN=`printf '%X\n' $BNdec`
fi


read BNlatest TSlatest < <(echo $(curl -s -X POST $RPC -H "Content-Type: application/json" --data '{"jsonrpc":"2.0", "method":"eth_getBlockByNumber", "params":["latest",false], "id":1}' | jq -r '.result.number, .result.timestamp'))
TSfirst=`curl -s -X POST $RPC -H "Content-Type: application/json" --data '{"jsonrpc":"2.0", "method":"eth_getBlockByNumber", "params":["'0x$BN'",false], "id":1}' | jq '.result.timestamp' | tr -d '"' | python3 -c 'import sys; print(int(sys.stdin.read(), 16))'`  

BNlatest=`printf "%d" $BNlatest`
TSlatest=`printf "%d" $TSlatest`

let STEP="$GRAN / (($TSlatest - $TSfirst) / ($BNlatest-$BNdec))"
let NPOINTS="($BNlatest - $BNdec) / $STEP"

for (( c=$ROW+1; c<=$ROW+$NPOINTS; c++ ))
do

read TTD DIFF TIME < <(echo $(curl -s -X POST $RPC \
     -H "Content-Type: application/json" --data '{"jsonrpc":"2.0", "method":"eth_getBlockByNumber", "params":["'0x$BN'", false], "id":1}' \
     | jq -r '.result.totalDifficulty, .result.difficulty, .result.timestamp'))

BN=`echo "obase=10; ibase=16; $BN" | bc`
TIME=`echo $TIME |  python3 -c 'import sys; print(int(sys.stdin.read(), 16))'`

echo $c | tr '\n' ',' >> result.csv
echo $BN | tr '\n' ',' >> result.csv
echo $TTD | python3 -c 'import sys; print(int(sys.stdin.read(), 16))' | rev | cut -c10- | rev | tr '\n' ',' >> result.csv
echo $DIFF |  python3 -c 'import sys; print(int(sys.stdin.read(), 16))' | tr '\n' ',' >> result.csv
echo $TIME | gawk '{print strftime("%m/%d/%Y %H:%M:%S", $0)}' | tr '\n' ','  >> result.csv
echo $TIME  >> result.csv

NEXT_TS=`printf "%d\n" $(($TIME + $GRAN))`
NEXT_BN=`printf "%d\n" $(($STEP+$BN))`
BN=`python3 blockbytime.py $NEXT_TS $NEXT_BN $(($NEXT_BN+10))` 
BN=`printf '%X\n' $BN`

done

python3 polynom.py
