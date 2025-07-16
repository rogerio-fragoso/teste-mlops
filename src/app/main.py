import os
import pickle

from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth
from textblob import TextBlob


# Carrega o modelo criado no colab
pickle_in = open('../../models/model.pkl', 'rb')
lr = pickle.load(pickle_in)
colunas = ['tamanho', 'ano', 'garagem']

app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = os.environ.get('BASIC_AUTH_USERNAME')
app.config['BASIC_AUTH_PASSWORD'] = os.environ.get('BASIC_AUTH_PASSWORD')

basic_auth = BasicAuth(app)

@app.route('/')
def home():
    return 'Minha primeira API!'

@app.route('/sentimento/<frase>')
@basic_auth.required
def sentimento(frase):
    tb = TextBlob(frase)
    polaridade = tb.sentiment.polarity
    return f'Polaridade: "{polaridade}"'

@app.route('/cotacao/', methods=['POST'])
@basic_auth.required
def cotacao():
    dados = request.get_json()
    dados_input = [dados[col] for col in colunas]
    preco = lr.predict([dados_input])[0]
    return jsonify({'preco': preco})

app.run(debug=True, host='0.0.0.0')
