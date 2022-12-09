from flask import Flask


app = Flask(__name__, template_folder='C:\\Users\\aaron\PycharmProjects\\regression\\template')
app.config['SECRET_KEY'] = 'Talley'
from routes import *

if __name__ == '__main__':
    app.run(debug=True)