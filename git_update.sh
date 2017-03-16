#!/bin/bash
#pass comment and update git rep
#git pull origin master
git pull origin master

git add *
git add -u
git add -A
git commit -m $1
git push origin master

