from model import InputForm
from flask import Flask, render_template, request
from da_machine_updated import give_me_the_number
import sys
import os

import pandas as pd
import numpy as np

import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from flask_sqlalchemy import SQLAlchemy
app = Flask(__name__)

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///db/baseball.sqlite"
db = SQLAlchemy(app)
Base = automap_base()
# reflect the tables
Base.prepare(db.engine, reflect=True)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    form = InputForm(request.form)
    if request.method == 'POST' and form.validate():
        result = give_me_the_number(form.Team1.data, form.Team1_League.data,
                         form.Team2.data, form.Team2_League.data)
    else:
        result = None
    return render_template('view_plain.html',
                           form=form, result=result)

if __name__ == '__main__':
    app.run()