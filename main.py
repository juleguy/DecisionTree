#!/usr/bin/env python
# coding: utf-8

from arff2pandas import a2p
import pandas


with open('db/coup_de_soleil.arff') as f:
    df = a2p.load(f)


possibleValues = {}
colNames = []

# Extracting the columns names and the possible values for each column
for col in df.columns:
    posAt = col.find('@')
    colName = col[:posAt]

    values = col[posAt+1:]
    values = values.replace("{", "")
    values = values.replace("}", "")

    values = values.split(",")

    df[colName] = df[col]
    df = df.drop(col, axis=1)
    possibleValues[colName] = values
    colNames.append(colName)

# Supposing the target column is the last one
targetIndex = len(colNames)-1
targetColName = colNames[targetIndex]

# Initializing the positive and negative arrays
dfPos = pandas.DataFrame()
dfNeg = pandas.DataFrame()

# Creating the columns in dfPos and dfNeg
for colName in colNames:

    if colName != targetColName:
        dfPos[colName] = None
        dfNeg[colName] = None

# Creating the Neg and Pos dataframes
for indexRow, row in df.iterrows():

    newRow = {}

    for indexCol, colName in enumerate(df.columns):

        if indexCol != targetIndex:
            newRow[colName] = row[colName]

    # Adding the row to the right dataframe
    if row[targetIndex] == possibleValues[targetColName][0]:
        dfPos = dfPos.append(newRow, ignore_index=True)
    else:
        dfNeg = dfNeg.append(newRow, ignore_index=True)

print(dfPos)
print(dfNeg)


