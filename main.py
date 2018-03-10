#!/usr/bin/env python
# coding: utf-8

import pandas

from foilprop import FoilProp

foilProp = FoilProp()
foilProp.fit("db/weather_nominal.arff")

foilProp.prediction_mode = True

print(foilProp.training_set_size)



