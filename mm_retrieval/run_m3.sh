#!/bin/bash

VEC=$1

/opt/python/bin/python2.7 m3_tm.py $VEC 0 1000 1000 &
/opt/python/bin/python2.7 m3_tm.py $VEC 1 1000 1000 &

/opt/python/bin/python2.7 m3_tm.py $VEC 0 1000 10000 &
/opt/python/bin/python2.7 m3_tm.py $VEC 1 1000 10000 &

/opt/python/bin/python2.7 m3_tm.py $VEC 0 1000 100000 &
/opt/python/bin/python2.7 m3_tm.py $VEC 1 1000 100000 &



/opt/python/bin/python2.7 m3_tm.py $VEC 0 10000 1000 &
/opt/python/bin/python2.7 m3_tm.py $VEC 1 10000 1000 &

/opt/python/bin/python2.7 m3_tm.py $VEC 0 10000 10000 &
/opt/python/bin/python2.7 m3_tm.py $VEC 1 10000 10000 &

/opt/python/bin/python2.7 m3_tm.py $VEC 0 10000 100000 &
/opt/python/bin/python2.7 m3_tm.py $VEC 1 10000 100000 &



/opt/python/bin/python2.7 m3_tm.py $VEC 0 100000 1000 &
/opt/python/bin/python2.7 m3_tm.py $VEC 1 100000 1000 &

/opt/python/bin/python2.7 m3_tm.py $VEC 0 100000 10000 &
/opt/python/bin/python2.7 m3_tm.py $VEC 1 100000 10000 &

/opt/python/bin/python2.7 m3_tm.py $VEC 0 100000 100000 &
/opt/python/bin/python2.7 m3_tm.py $VEC 1 100000 100000 &

wait

echo "stai senza pensieri"


