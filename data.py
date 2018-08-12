import pandas as pd
import numpy as np
import os
import json

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def random_truncate(coordinates):
	if len(coordinates) <= 1:
   		return coordinates
	n = np.random.randint(len(coordinates) - 1)
	if n > 0:
		return coordinates[0:n]
	else:
		return coordinates

def convert_coordinates(string):
	coordinates = json.loads(string)
	coordinates_object = []
	#print("***", len(coordinates))
	for (x, y) in coordinates:
		t = []
		t.append(y)
		t.append(x)
		coordinates_object.append(t)
	#print(coordinates_object[0][1])
	return coordinates_object


def encode_feature(feature, train, test):
	encoder = LabelEncoder()
	train_values = train[feature].copy()
	test_values = test[feature].copy()
	train_values[np.isnan(train_values)] = 0
	test_values[np.isnan(test_values)] = 0
	#print(train_values.head())
	#encoder.fit(pd.concat([train_values, test_values]))
	train[feature + '_ENCODED'] = encoder.fit_transform(train_values)
	test[feature + '_ENCODED'] = encoder.fit_transform(test_values)
	return encoder

def extract_info(df):
	df['POLYLINE'] = df['POLYLINE'].apply(convert_coordinates)
	df['START_LAT'] = df['POLYLINE'].apply(lambda x: x[0][0])
	df['START_LONG'] = df['POLYLINE'].apply(lambda x: x[0][1])
	datetime_index = pd.DatetimeIndex(df['TIMESTAMP'])
	df['QUARTER_HOUR'] = datetime_index.hour * 4 + datetime_index.minute / 15
	df['DAY_OF_WEEK'] = datetime_index.dayofweek
	df['WEEK_OF_YEAR'] = datetime_index.weekofyear - 1
	df['DURATION'] = df['POLYLINE'].apply(lambda x: 15 * len(x))

def load_data():
	train_cache = 'cache/train.pickle'
	train_labels_cache = 'cache/train-labels.npy'
	validation_cache = 'cache/validation.pickle'
	validation_labels_cache = 'cache/validation-labels.npy'
	test_cache = 'cache/test.pickle'
	test_labels_cache = 'cache/test-labels.npy'
	competition_test_cache = 'cache/competition-test.pickle'
	metadata_cache = 'cache/metadata.pickle'

	if os.path.isfile(train_cache):
		# Load from cached files if they already exist
		train = pd.read_pickle(train_cache)
		validation = pd.read_pickle(validation_cache)
		test = pd.read_pickle(test_cache)
		train_labels = np.load(train_labels_cache)
		validation_labels = np.load(validation_labels_cache)
		test_labels = np.load(test_labels_cache)
	else:
		print("hellooooooo")
		data = pd.read_csv('./datasets/train.csv')
		test = pd.read_csv('./datasets/test.csv')
		data = data[:100000]				#Remove this later

		combine = [data, test]
		datasets = []
		#cdprint(data.POLYLINE)
		#print(data[['MISSING_DATA', 'TRIP_ID']].groupby(['MISSING_DATA']).count())
		#print(data.head())

		for dataset in combine:
			dataset = dataset[dataset['MISSING_DATA'] == False]
			dataset = dataset[dataset['POLYLINE'] != '[]']
			dataset.drop('MISSING_DATA', axis=1, inplace=True)
			dataset.drop('DAY_TYPE', axis=1, inplace=True)
			dataset['TIMESTAMP'] = dataset['TIMESTAMP'].astype('datetime64[s]')
			extract_info(dataset)
			datasets.append(dataset)

		#print(datasets[0].columns)
		'''
		data.drop('MISSING_DATA', axis=1, inplace=True)
		data.drop('DAY_TYPE', axis=1, inplace=True)
		test.drop('MISSING_DATA', axis=1, inplace=True)
		test.drop('DAY_TYPE', axis=1, inplace=True)
		#data['TIMESTAMP'] = data['TIMESTAMP'].astype('datetime64[s]')
		combine = [data, test]

		print(combine[0]['TIMESTAMP'])
		'''
		#print(datasets[0][['ORIGIN_CALL', 'TRIP_ID']].groupby(['ORIGIN_CALL']).count().sort_values(by=['TRIP_ID'], ascending=False))
		train, competition_test = datasets

		client_encoder = encode_feature('ORIGIN_CALL', train, competition_test)
		taxi_encoder = encode_feature('TAXI_ID', train, competition_test)
		stand_encoder = encode_feature('ORIGIN_STAND', train, competition_test)
		#print(train[['ORIGIN_CALL_ENCODED', 'TRIP_ID']].groupby(['ORIGIN_CALL_ENCODED']).count().sort_values(by=['TRIP_ID'], ascending=False))

		train['FULL_POLYLINE'] = train['POLYLINE'].copy()
		train['POLYLINE'] = train['POLYLINE'].apply(random_truncate)

		train_labels = train['FULL_POLYLINE'].apply(lambda x: x[-1])

		train, validation, train_labels, validation_labels = train_test_split(train, train_labels, test_size=0.02)
		validation, test, validation_labels, test_labels  = train_test_split(validation, validation_labels, test_size=0.5)

		metadata = {
            'n_quarter_hours': 96,  
            'n_days_per_week': 7,
            'n_weeks_per_year': 52,
            'n_client_ids': len(client_encoder.classes_),
            'n_taxi_ids': len(taxi_encoder.classes_),
            'n_stand_ids': len(stand_encoder.classes_),
        }

		train.to_pickle(train_cache)
		validation.to_pickle(validation_cache)
		test.to_pickle(test_cache)
		np.save(train_labels_cache, train_labels)
		np.save(validation_labels_cache, validation_labels)
		np.save(test_labels_cache, test_labels)
		competition_test.to_pickle(competition_test_cache)
		with open(metadata_cache, 'wb') as handle:
            pickle.dump(metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)

	data = Data()
	data.__dict__.update({
		'train': train,
		'train_labels': train_labels,
		'validation': validation,
		'validation_labels': validation_labels,
		'test': test,
		'test_labels': test_labels,
		'competition_test': competition_test,
		'metadata': metadata,
	})
	return data
