import pandas as pd
import numpy as np


def get_emp_length(emp_length):
    """Takes our emp_length data (string) and returns numeric value
    in number of years.
    """
    unique_values = {         #keys taken from loans.emp_length.unique()
        '10+ years' : 10, 
        '< 1 year' : .5,      #between 0 and 1
        '3 years' : 3, 
        '9 years' : 9, 
        '4 years' : 4, 
        '5 years' : 5,
        '1 year' : 1, 
        '6 years' : 6, 
        '2 years' : 2, 
        '7 years' : 7, 
        '8 years' : 8, 
        'n/a': 0
    }
    return unique_values[emp_length]


def convert_date(input_string, current_year=2011):
    """changes date from format 'Mon-YYYY' to
    an integer number of months before the end of [current_year]
    (when the input data was published)
    """
    months_num = {
        'Jan' : 1,
        'Feb' : 2,
        'Mar' : 3,
        'Apr' : 4,
        'May' : 5,
        'Jun' : 6,
        'Jul' : 7,
        'Aug' : 8,
        'Sep' : 9,
        'Oct' : 10,
        'Nov' : 11,
        'Dec' : 12
    }
    mon = input_string[:3]
    year = int(input_string[-4:])
    if mon in months_num:
        num = months_num[mon]
    else:
        raise ValueError('{} not found in dictionary'.format(mon))
    months_passed = (12 - num) + ((current_year - year) * 12)
    if not str(months_passed).isnumeric():
        raise Exception('Error: return object not numeric: {}'.format(months_passed))
    return months_passed


def convert_home(home_type):
    """Convert housing status to numeric value.
    Tries to rank numbers roughly by wealth-association with each
    homeownership status.
    """
    unique_values = {   #keys taken from loans.home_status.unique() 
        'OTHER' : 0, 
        'NONE' : 0,     #These three vague categories all assigned to zero
        'ANY' : 0,
        'RENT' : 1,
        'MORTGAGE' : 2, 
        'OWN' : 3
    }
    return unique_values[home_type]


def convert_grade(grade):
    """Converts borrower grade to numeric value"""
    unique_grades = { #keys taken from loans.grade.unique()
        'A1' : 35,
        'A2' : 34,
        'A3' : 33,
        'A4' : 32,
        'A5' : 31,
        'B1' : 30,
        'B2' : 29,
        'B3' : 28,
        'B4' : 27,
        'B5' : 26,
        'C1' : 25,
        'C2' : 24,
        'C3' : 23,
        'C4' : 22,
        'C5' : 21,
        'D1' : 20,
        'D2' : 19,
        'D3' : 18,
        'D4' : 17,
        'D5' : 16,
        'E1' : 15,
        'E2' : 14,
        'E3' : 13,
        'E4' : 12,
        'E5' : 11,
        'F1' : 10,
        'F2' : 9,
        'F3' : 8,
        'F4' : 7,
        'F5' : 6,
        'G1' : 5,
        'G2' : 4,
        'G3' : 3,
        'G4' : 2,
        'G5' : 1
    }
    return unique_grades[grade]


def handle_nulls(series):  #Use this because high values 
    """Takes pandas series with null values and replaces them with value significantly higher than
    maximum value.
    """
    new_val = series.max() + 5 * series.std()  #Set values way higher than mean
    nulls = pd.isnull(series)
    return np.where(nulls, new_val, series)


def set_verification_status(status):
    """Enumerates varification statuses.
    """
    statuses = {
        'Not Verified' : 0,
        'Verified' : 1,
        'Source Verified' : 2
    }
    return statuses[status]


def treat_loan_data(loans, year=2011):
    """Takes Lending Club data and applies a series of actions to clean up data.
    Inputs data frame and returns dataframe. Must be specifically formatted as 
    downloaded from https://www.lendingclub.com/info/download-data.action
    """

    #Convert dollar amounts: floating point to int in cents
    loans.loan_amnt = loans.loan_amnt.astype('int')*100
    loans.installment = loans.installment.astype('int')*100
    loans.annual_inc = loans.annual_inc.astype('int')*100
    loans['log_inc'] = np.where(loans.annual_inc == 0, 0, np.log(loans.annual_inc))
    loans.annual_inc_joint = np.where((pd.isnull(loans.annual_inc_joint)), loans.annual_inc, loans.annual_inc_joint)
    loans.annual_inc_joint = loans.annual_inc_joint.astype('int')*100
    loans['log_inc_joint'] = np.where(loans.annual_inc_joint == 0, 0, np.log(loans.annual_inc_joint))
    loans.revol_bal = loans.revol_bal.astype('int')*100
    loans.delinq_amnt = loans.delinq_amnt.astype('int')*100


    #Convert strings to numeric types
    loans.int_rate = pd.to_numeric(loans.int_rate.str.slice(0, -1)) #Convert percentages (strings) to numeric values
    loans.revol_util = pd.to_numeric(loans.revol_util.str.slice(0, -1))

    loans.term = pd.to_numeric(loans.term.str.slice(0, 3)).astype('int') #Convert loan term to number
    loans['term_type'] = np.where((loans.term == 60), 1, 0) #Bool for loan term, as only two types exist.

    loans.dti_joint = np.where((pd.isnull(loans.dti_joint)), loans.dti, loans.dti_joint) 


    #Convert employee length strings to numeric values

    loans.emp_length = loans.emp_length.apply(get_emp_length)


    #Convert dates into numeric format.

    loans.issue_d = loans.issue_d.apply(lambda x: (convert_date(x, year)))
    loans.earliest_cr_line = loans.earliest_cr_line.apply(lambda x: (convert_date(x, year)))


    #Calculate number of months first credit line before loan issued

    loans['fcl_before_loan'] = loans.earliest_cr_line - loans.issue_d


    #Convert home status to a numeric value

    loans.home_ownership = loans.home_ownership.apply(convert_home)


    # Convert grade to numeric value

    loans.sub_grade = loans.sub_grade.apply(convert_grade)


    #Replace null values with high values; 5 stds above mean value

    loans.mths_since_last_delinq = handle_nulls(loans.mths_since_last_delinq)
    loans.mths_since_last_record = handle_nulls(loans.mths_since_last_record)


    #enumerate employee titles by how commonly they occur

    loans.emp_title = np.where((pd.isnull(loans.emp_title)), 'none', loans.emp_title)

    titles = {}
    for i, title in enumerate(loans.emp_title.value_counts().index):
        titles[title] = i

    for row in loans.index: #Set titles by rank of frequency
        loans.set_value(row, 'emp_title', titles[loans.emp_title.loc[row]])


    #Handle nulls in revolving credit utilization                                                      

    loans.revol_util = np.where((pd.isnull(loans.revol_util)), 0, loans.revol_util)


    #Calculate utilized credit balance
    loans['used_cred_bal'] = loans.revol_util * loans.revol_bal
    loans['log_used_cred'] = np.where(loans.used_cred_bal < 1, 0, np.log(loans.used_cred_bal)) #log of used cred bal


    #Map verification statuses and application types to numeric values

    loans.verification_status = loans.verification_status.apply(set_verification_status)
    loans.application_type = np.where((loans.application_type == 'INDIVIDUAL'), 0, 1)                                                 


    #Very roughly calculated monthly cash flow
    loans['cash_flow'] = loans.annual_inc_joint/12 - ((loans.dti_joint * loans.annual_inc_joint * .004) + loans.installment)
    loans['log_cf'] = np.log((loans.cash_flow - loans.cash_flow.min() + 1)) #normalize a bit with log transform

    return loans
