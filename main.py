## 
#   1)  Create Database
#       Download news headlines     >   save to database
#       Download timeseries data    >   save to database
#
#   2)  Create dataframe from headlines and timeseries
#           -   use datetime as uniqueID (round ms to next second)
#           -   remove headlines outside of trading hours
#           -   remove ts values 30M before close & 30M after open
#
#   3)  Create Model to predict if news headline will result in 
#           stock price increasing or decreasing
#
#   4)  Consider other stuff
 
# exec(open('main.py').read())
from Company import Company
from Model import Model

cba = Company("CBA")
model = Model(cba)