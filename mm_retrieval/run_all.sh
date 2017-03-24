#!/bin/bash

python m1.py mm_test 2
python m2.py mm_test 2
python m3_tm.py mm_test 2
python m4.py mm_test 2
python m1.py mm_train 2
python m2.py mm_train 2
python m3_tm.py mm_train 2
python m4.py mm_train 2

python m1.py raw_test 2
python m2.py raw_test 2
python m3_tm.py raw_test 2
python m4.py raw_test 2
python m1.py raw_train 2
python m2.py raw_train 2
python m3_tm.py raw_train 2
python m4.py raw_train 2

python m1.py mm_test 1
python m2.py mm_test 1
python m3_tm.py mm_test 1
python m4.py mm_test 1
python m1.py mm_train 1
python m2.py mm_train 1
python m3_tm.py mm_train 1
python m4.py mm_train 1

python m1.py raw_test 1
python m2.py raw_test 1
python m3_tm.py raw_test 1
python m4.py raw_test 1
python m1.py raw_train 1
python m2.py raw_train 1
python m3_tm.py raw_train 1
python m4.py raw_train 1

echo "stai senza pensieri"


