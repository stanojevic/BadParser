#!/bin/sh


vsizes=(32 64 128 256 512)
wsizes=(8 16)
useCBOWs=(0 1)

for vsize in "${vsizes[@]}"; do
	for wsize in "${wsizes[@]}"; do
		for useCBOW in "${useCBOWs[@]}"; do

			X=vsize_${vsize}_wsize_${wsize}_cbow_${useCBOW}
			echo "Running: $X";

			sbatch -c 16 -J w2v$X -e log.${X}.err -o log.${X}.out --wrap="./word2vec -train ../../../de/full.txt -output ${X}.bin -cbow $useCBOW -size $vsize -window $wsize -negative 25 -hs 0 -sample 1e-4 -threads 15 -binary 1 -iter 15"


			#./distance vectors.bin
	
			sleep 1s

done;
done;
done;
