from fastai.tabular.all import *
import pandas as pd
from pathlib import Path

df = pd.read_csv("adult.csv")


dls = TabularDataLoaders.from_df(
    df,
    y_names="income", 
    cat_names=['workclass', 'education', 'marital-status', 'occupation',
               'relationship', 'race', 'sex', 'native-country'],
    cont_names=['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'],
    procs=[Categorify, FillMissing, Normalize]
)

learn = tabular_learner(dls, metrics=accuracy)
learn.fit_one_cycle(5)
learn.export("adult_model.pkl")