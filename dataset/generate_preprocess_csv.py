from optparse import OptionParser
import pandas as pd
import numpy as np
import math
import os


def corr(x, y):
    def is_num(s):
        try:
            float(s)
        except ValueError:
            return False
        else:
            return not math.isnan(float(s))

    x_num, y_num = [], []
    for item in zip(list(x), list(y)):
        if is_num(item[0]) and is_num(item[1]):
            x_num.append(float(item[0]))
            y_num.append(float(item[1]))

    if len(x_num) <= 1 or sum(x_num) == 0 or sum(y_num) == 0:
        return 0
    matrix = np.corrcoef(np.array([x_num, y_num]))
    return matrix[0][1]


def data_cleansing():
    # salary to number
    min_salary_str = [txt.split()[0] for txt in data['salary'].to_numpy()]
    min_salary = [int(txt[1:].replace(',', '')) for txt in min_salary_str]
    max_salary_str = [txt.split()[2] for txt in data['salary'].to_numpy()]
    max_salary = [int(txt[1:].replace(',', '')) for txt in max_salary_str]
    data.drop(['salary'], inplace=True, axis=1)
    data['salary_min'] = min_salary
    data['salary_max'] = max_salary

    # snippet remove strange character
    for i, snippet in enumerate(data['snippet']):
        if not str(snippet[-2]).isascii():
            data.at[i, 'snippet'] = snippet[:-2]

    # drop useless column
    useless_column = [
        'link',  # unique url
        'job_link',  # unique url
        'hiring_event_job',  # all false
        'remote_location',  # all false
        'indeed_apply_enabled',  # corr=1 with indeed applicable
        'ad_id'  # drop id
    ]
    data.drop(useless_column, inplace=True, axis=1)
    """
    print('Correlation between minimum salary and source_id: {}'.format(corr(data['salary'], data['source_id'])))
    print('Correlation between minimum salary and postal: {}'.format(corr(data['salary'], data['job_location_postal'])))
    """


def data_transform_impute():
    see_category = (lambda f: [*{*list(data[f].to_numpy())}])

    # company link
    company_has_link = data['company_overview_link'].notna().to_numpy()
    data.drop(['company_overview_link'], inplace=True, axis=1)
    data['company_has_link'] = company_has_link

    # types
    job_types = ['Full-time', 'Part-time', 'Temporary', 'Contract', 'Internship', 'N/A']
    expand_features = []
    for txt in data['types']:
        flag = [False, False, False, False, False, False]
        if not isinstance(txt, str):
            flag[-1] = True
        else:
            for d in txt.split(','):
                flag[job_types.index(d.strip())] = True
        expand_features.append(flag)
    for i, typename in enumerate(job_types):
        data[f'job_type_{typename}'] = [d[i] for d in expand_features]
    data.drop(['types'], inplace=True, axis=1)

    # relative time
    relative_time = [txt.split()[0] for txt in data['relative_time']]
    for i, txt in enumerate(relative_time):
        if txt == '30+':
            relative_time[i] = 31
        elif txt.isdigit():
            relative_time[i] = int(txt)
        else:
            relative_time[i] = 0  # Today & Just posted
    data.drop(['relative_time'], inplace=True, axis=1)
    data['relative_time'] = relative_time

    # activity date
    activity_date = []
    for txt in data['activity_date']:
        if not isinstance(txt, str):
            activity_date.append(-1)  # nan
        elif txt.split()[1].isdigit():
            activity_date.append(int(txt.split()[1]))
        else:
            activity_date.append(31)  # 30+
    data.drop(['activity_date'], inplace=True, axis=1)
    data['activity_date'] = activity_date
    data['activity_date_na'] = (data['activity_date'] == -1)

    # features related to location
    data['location_remote'] = (data['location'] == 'Remote')
    data['no_postal'] = data['job_location_postal'].isna()
    work_model = []
    for txt in data['remote_work_model'].to_numpy():
        if not isinstance(txt, str):
            work_model.append(0)  # nan
        elif txt == 'REMOTE_COVID_TEMPORARY':
            work_model.append(1)
        elif txt == 'REMOTE_ALWAYS':
            work_model.append(2)
        else:
            work_model.append(0)
    data.drop(['remote_work_model', 'location', 'location_extras'], inplace=True, axis=1)
    data['remote_work_model'] = work_model

    # hires needed
    data['hires_needed_na'] = data['hires_needed'].isna()
    data['hires_needed_exact_na'] = data['hires_needed_exact'].isna()
    hires_needed_exact = []
    for txt in data['hires_needed_exact'].to_numpy():
        if not isinstance(txt, str):
            hires_needed_exact.append(0)  # nan
        elif txt == 'TEN_PLUS':
            hires_needed_exact.append(11)
        elif txt.isdigit():
            hires_needed_exact.append(int(txt))
        else:
            hires_needed_exact.append(0)
    data.drop(['hires_needed', 'hires_needed_exact'], inplace=True, axis=1)
    data['hires_needed_exact'] = hires_needed_exact

    # impute postal
    data['job_location_postal'].fillna(value=data['job_location_postal'].median(), inplace=True)


def generate_csv(filename):
    data_cleansing()
    data_transform_impute()
    data.to_csv(filename, index=False, encoding='latin-1')
    print(f'Write preprocessed file to {filename}')


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option(
        '-f', dest='fin', type=str, help='csv file path', default='train.csv'
    )
    options, args = parser.parse_args()
    data = pd.read_csv(options.fin, encoding='latin-1')

    output_filename = f'{os.path.splitext(str(options.fin))[0]}_preprocess.csv'
    generate_csv(output_filename)
