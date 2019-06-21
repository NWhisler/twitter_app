From command line:

	git clone https://github.com/NWhisler/twitter_app.git

	cd twitter_app

		pip install -r requirements.txt

		python app_data.py

		cd App

			Either pip install -r requirements.txt then python app.py

			Or open app.py add host='0.0.0.0' as an argument to app.run() followed by docker build . then docker run -p 5000:5000