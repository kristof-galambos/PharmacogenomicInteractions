import pandas as pd

if __name__ == '__main__':
    print('reading')
    gdsc_data = pd.read_excel('/Users/kristof/Downloads/GDSC2_fitted_dose_response_27Oct23.xlsx')
    print('reading done')

    print(gdsc_data['CANCER_TYPE'].unique())
