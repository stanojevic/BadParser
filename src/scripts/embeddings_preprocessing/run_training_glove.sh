vsizes=(32 64 128 256 512)

for vsize in "${vsizes[@]}"; do
	 echo $vsize;
	  cat mother_trainer.sh | sed -e "s/VECTOR_SIZE=.*$/VECTOR_SIZE=$vsize/"  -e "/^VOCAB_FILE/ s/$/_$vsize/"  -e "/^COOCCURRENCE_FILE/ s/$/_$vsize/" 	  -e "/^COOCCURRENCE_SHUF_FILE/ s/$/_$vsize/" 	   -e "/^SAVE_FILE/ s/$/_$vsize/"   >train_$vsize.sh;

	   sbatch  -c 9 -e log.$vsize.err -o log.$vsize.out train_$vsize.sh
	   sleep 1s;
   done
