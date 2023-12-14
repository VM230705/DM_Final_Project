from optparse import OptionParser
import pandas as pd
import os


def split_data(path, test_size=0.1, seed=42):
    train_name = os.path.join(path, 'train.csv')
    test_name = os.path.join(path, 'test.csv')
    salary_notnull = data['salary'].notna().to_numpy()
    result = data.loc[salary_notnull, :].sample(frac=1, replace=False, axis=0, random_state=seed)

    split_idx = round(result.shape[0] * (1 - test_size))
    train_data = result.iloc[:split_idx, :]
    test_data = result.iloc[split_idx:, :]

    train_data.to_csv(train_name, index=False, encoding='latin-1')
    test_data.to_csv(test_name, index=False, encoding='latin-1')
    print(f'Write training dataset to {train_name}')
    print(f'Write testing dataset to {test_name}')


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option(
        '-f', dest='fin', type=str, help='csv file path', default='data.csv'
    )
    options, args = parser.parse_args()
    data = pd.read_csv(options.fin, encoding='latin-1')

    basename = os.path.splitext(str(options.fin))[0]
    head, _ = os.path.split(str(basename))
    split_data(head, test_size=0.1, seed=42)
