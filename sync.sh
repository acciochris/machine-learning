#!/usr/bin/env bash

git subtree pull --prefix ml-book git@github.com:acciochris/ml-book.git main --squash \
&& git subtree pull --prefix kaggle git@github.com:acciochris/kaggle.git main --squash \
&& git push
