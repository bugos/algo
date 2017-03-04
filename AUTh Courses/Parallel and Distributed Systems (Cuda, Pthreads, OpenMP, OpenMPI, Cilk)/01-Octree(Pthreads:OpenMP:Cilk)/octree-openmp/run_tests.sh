
# A simple bash script that for running experiments
# Note: To run the script make sure it you have execution rights 
# (use: chmod u+x run_tests.sh to give execution rights) 
#!/bin/bash

NAME=$(hostname)
DATE=$(date "+%Y-%m-%d-%H:%M:%S")
FILE_PREF=$NAME-$DATE-test-tree

DIST_NAME=( 'cube' 'Plummer' )
R=4        #repeat

#Set number of threads
#export OMP_NUM_THREADS=<number of threads to use>

echo $NAME
echo $DATE

make clean; make

for D in 1 ; do \
    for T in 2 4 8 16 32 64 256 512; do \
        echo $T threads;
        for N in  1048576 2097152 4194304 8388608 16777216   ; do \
            for P in 128 ; do \
                for L in 18 ; do \
                    echo ${DIST_NAME[$D]} N=$N && ./test_octree $N $D $P $R $L $T >> results/$FILE_PREF-${DIST_NAME[$D]}.txt || exit;  
                    #-$D-$T-$N-$P-$L    33554432 
                done ;
            done ;
        done ;
    done ;
done ;

