from wtforms import Form, FloatField, StringField, validators


class InputForm(Form):
    Team1 = StringField(
        label="What is the team's abbreviation?", default="LAA",
        validators=[validators.InputRequired()])
    Team1_League = StringField(
        label="What league is the first team in?", default="American",
        validators=[validators.InputRequired()])
    Team2 = StringField(
        label="Use the team's abbreviation", default="HOU",
        validators=[validators.InputRequired()])
    Team2_League = StringField(
        label="What league is the second team in?", default="American",
        validators=[validators.InputRequired()])