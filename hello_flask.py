import flask



app = flask.Flask(__name__)

#Make a route. A simple controller,  a way to move around the website (ex. /home, /route)
@app.route("/home")
def index():
    return flask.render_template("index.html",title = "HomePage")

if __name__ == "__main__":
    app.debug=True
    app.run()
