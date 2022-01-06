# import necessary libraries
from flask_sqlalchemy import SQLAlchemy
from models import create_classes
import os
from flask import (
    Flask,
    render_template,
    jsonify,
    request,
    redirect)


# 1. create a new clean env
# python -m venv <name>
# 2. add this name to gitignore
# 3. use git bash > source <name>/Scripts/activate
# 4. conda deactivate
# 5. pip freeze > make sure it's clean
# 6. pip install dependencies > make sure you will have gunicorn
# 7. make sure you create a db on heroku > make sure 'posgresql'
# go to heroku website > go to app > Setting > create a new key for your correct DB link
# 8. put this new var to your code
# 9. python app.py on localhost


#################################################
# Flask Setup
#################################################
app = Flask(__name__)

#################################################
# Database Setup
#################################################

urlz = os.getenv('DATABASE_URLZ')  # used for Heroku deployment
# urlz = 'postgresql://njfgtyroqyycuk:2b4d75c46d7bb5075e0146ac8e835985246903b252cc2423ad4779e9a7bb74ac@ec2-3-231-40-72.compute-1.amazonaws.com:5432/d15jf7fk1ra3kr'  # local machine testing

print(f'urlz before try and except {urlz}', flush=True)

try:
    if urlz:
        print(f'try has worked, urlz is: {urlz}', flush=True)
except Exception as error:
    urlz = "postgresql://njfgtyroqyycuk:2b4d75c46d7bb5075e0146ac8e835985246903b252cc2423ad4779e9a7bb74ac@ec2-3-231-40-72.compute-1.amazonaws.com:5432/d15jf7fk1ra3kr"
    print('except has activated', flush=True)

print(f'urlz after try and except = {urlz}')

app.config['SQLALCHEMY_DATABASE_URI'] = urlz

# 1. create Postgres database on Heroku
# 2. create Database URL on Heroku with new PostgreSQL syntax
# 3. input Database URL + NEW
# 4. Replace sqlite with PostgreSQL Alchemy log in information

# Remove tracking modifications
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


Pet = create_classes(db)

# create route that renders index.html template


@app.route("/")
def home():
    return render_template("index.html")


# Query the database and send the jsonified results
@app.route("/send", methods=["GET", "POST"])
def send():
    if request.method == "POST":
        name = request.form["petName"]
        lat = request.form["petLat"]
        lon = request.form["petLon"]

        pet = Pet(name=name, lat=lat, lon=lon)
        db.session.add(pet)
        db.session.commit()
        return redirect("/", code=302)

    return render_template("form.html")


@app.route("/api/pals")
def pals():
    results = db.session.query(Pet.name, Pet.lat, Pet.lon).all()

    hover_text = [result[0] for result in results]
    lat = [result[1] for result in results]
    lon = [result[2] for result in results]

    pet_data = [{
        "type": "scattergeo",
        "locationmode": "USA-states",
        "lat": lat,
        "lon": lon,
        "text": hover_text,
        "hoverinfo": "text",
        "marker": {
            "size": 50,
            "line": {
                "color": "rgb(8,8,8)",
                "width": 1
            },
        }
    }]

    return jsonify(pet_data)


if __name__ == "__main__":
    db.create_all()
    app.run(debug=True)
