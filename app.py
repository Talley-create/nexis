from flask import Flask


app = Flask(__name__, template_folder='####################################')
app.config['SECRET_KEY'] = '###############'
from routes import *

if __name__ == '__main__':
    app.run(debug=True)
