for rate in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 
do
    for algo in Patch 
    do
        sh scripts/train-on-objaverse_inference.sh $algo $rate 
    done
done
