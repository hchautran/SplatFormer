for rate in 0.8 
do
    for algo in fps 
    do
        sh scripts/train-on-objaverse_inference.sh $algo $rate 
    done
done
