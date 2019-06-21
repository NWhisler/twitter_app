import os
import shutil
import pandas as pd 
import numpy as np 
from twitter import twitter_model
from flask import Flask, render_template, request, abort

app = Flask(__name__)

@app.route('/')
def index():
	
	return render_template('index.html')

@app.route('/predictions',methods=['POST'])
def generate_predictions():

	file = request.files.get('data')
	
	if file:

		os.mkdir('upload')

		file_name = file.filename
		file_path = 'upload/'+file_name
		file.save(file_path)

		if file_name.endswith('.csv') or file_name.endswith('.json'):

			model = twitter_model(file_path)
			model.feature_generation()
			predictions = model.evaluate()
			tickers = model.tickers
			rows = range(len(predictions))
			row_ticker = 'row_ticker'
			row_prediction = 'row_prediction'
			data = {}
			for row in rows:
				data[row] = {'row_ticker':tickers[row],'row_prediction':predictions[row]}
			df_company = pd.DataFrame(np.array([model.tickers,predictions]).T,columns=['ticker','prediction'])
			ticker_symbols = sorted(set(tickers))
			down = 0
			up = 1
			company_data = {}
			for ticker in ticker_symbols:
				down_predictions = len(df_company.prediction[(df_company.ticker==ticker) & (df_company.prediction == 0.0)])
				up_predictions = len(df_company.prediction[(df_company.ticker==ticker) & (df_company.prediction == 1.0)])
				company_data[ticker] = {down:down_predictions,up:up_predictions}
			shutil.rmtree('upload')
			return render_template('result.html',data=data,rows=rows,row_ticker=row_ticker,row_prediction=row_prediction \
												,company_data=company_data,ticker_symbols=ticker_symbols,up=up,down=down)

	else:

		abort(501)

if __name__ == '__main__':

	app.run(port=5000)