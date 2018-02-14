#!/usr/bin/env python
# coding: utf-8

from arff2pandas import a2p


with open('db/coup_de_soleil.arff') as f:
    df = a2p.load(f)
    print(df)

