import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(fn='../data/data.csv'):
	'''
	load all data from fn

	Returns:
	- data: list[list[strings]], N*1 dims
	- labels: list, N dims
	'''
	all_data = pd.read_csv(fn, sep='\t', header=None, encoding='utf-8')

	data = all_data[0].values.reshape(-1, 1)
	labels = all_data[1].values
	return data, labels

def split_data(data, labels, test_size=0.2, shuffle=True, stratify=None):
	data_train, data_test, label_train, label_test = train_test_split(data, labels, test_size=0.2, shuffle=True,stratify=labels)
	return data_train, data_test, label_train, label_test

def save_data(data, label, fn='../data/train.csv'):
	all = ['%s\t%s\n' % (X[0], y) for X, y in zip(data, label)]
	with open(fn, 'w', encoding='utf-8') as f:
		for line in all:
			f.write(line)

if __name__ == '__main__':
	data, labels = load_data()
	data_train, data_test, label_train, label_test = split_data(data, labels, test_size=0.2, shuffle=True, stratify=None)
	save_data(data_train, label_train, '../data/train.csv')
	save_data(data_test, label_test, '../data/test.csv')


	print(data.shape)