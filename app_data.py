import tarfile
import wget

def get_data():

	wget.download('https://storage.googleapis.com/springboard_project_data/app_data.tar')

def extract_data():

	fh = tarfile.open('app_data.tar','r')

	fh.extractall('App')

if __name__ == '__main__':

	get_data()

	extract_data()