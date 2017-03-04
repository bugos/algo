# A simple bash script that for running experiments
# Note: To run the script make sure it you have execution rights 
# (use: chmod u+x run_tests.sh to give execution rights) 
#!/bin/bash

NAME=$(hostname)
DATE=$(date "+%Y-%m-%d-%H:%M:%S")
FILE_PREF=$NAME-$DATE-test-tree

DIST_NAME=( 'cube' 'Plummer' )
R=4        #repeat


echo $NAME
echo $DATE

make clean; make

for D in 0 1 ; do \
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


exit;
DIST=1      #distribution code (0-cube, 1-sphere)
L=18        #max heigh L=18
P=128       #population S=128 max
# run cube experiments
for T in 2 4 8 16 32 64 128 256 512 1024 2048 ; do \
    for N in 1048576 2097152 4194304 8388608 16777216 33554432 ; do \
        echo cube N=$N && ./test_octree $N 0 $P $R $L $T >> results/$FILE_PREF-$T-$N-0.txt ; \
    done ;
done ;

# run octant experiments
for T in 2 4 8 16 32 64 128 256 512 1024 2048 ; do \
    for N in 1000000 2000000 ; do \
        echo Plummer N=$N && ./test_octree $N 1 $P $R $L $T >> results/$FILE_PREF-$T-$N-1.txt ; \
    done ; 
done ; 
