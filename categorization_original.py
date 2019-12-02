#!/usr/bin/env python
# coding: utf-8

import re
import json
from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import datetime
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

application = Flask(__name__)

@application.route("/categorization", methods=['GET','POST'])
def bank_category():
    # importing data into system and saving it to df
    if request.method == 'POST':

        # reading in the csv file from a http post method
        df = pd.read_csv(request.files.get('file'))

        # creating the list of possible columns names in a bank statement

        date = ['transaction.date', 'date','trans. date', 'trans date', 'transaction date', 'posted date', 'txn date', 'posting date']
        description = ['description', 'remarks', 'narration', 'transaction details', 'transaction']
        withdrawal = ['withdrawls', 'debit', 'withdrawal(dr)', 'debits', 'payments', 'withdrawals', 'withdrawal']
        deposit = ['credit', 'deposits', 'lodgements', 'deposit(cr)', 'credits', 'deposit']
        balance = ['balance', 'balances', 'closing balance']

        for column in df:
            if column.lower() in date:
                df.rename({column:'Date'}, axis = 1, inplace=True)
        for column in df:
            if column.lower() in description:
                df.rename({column:'Description'}, axis = 1, inplace=True)
        for column in df:
            if column.lower() in withdrawal:
                df.rename({column:'Withdrawls'}, axis = 1, inplace=True)
        for column in df:
            if column.lower() in balance:
                df.rename({column:'Balance'}, axis = 1, inplace=True)
        for column in df:
            if column.lower() in deposit:
                df.rename({column:'Deposits'}, axis = 1, inplace=True)

        df = df[pd.notnull(df['Balance'])]

        new = df['Date'].astype(str).str.split("\s+", n=1, expand=True)
        # new = df['Date'].astype(str).str.split("\n", n=1, expand=True)
        # making separate first name column from new data frame
        df['Date'] = new[0]

        if(1 in new.columns):
            # making separate last name column from new data frame
            df["Wrong Date"] = new[1].copy()
            # Dropping old Name columns
            df.drop(columns=["Wrong Date"], inplace=True)
            # converting the Date string to a Date utility
            df['Date'] = pd.to_datetime(df['Date'])
        else:
            # converting the Date string to a Date utility
            df['Date'] = pd.to_datetime(df['Date'])

        df["Comment"] = df["Description"].str.replace(r"[^a-zA-Z ]+", " ").str.strip()
        df['Comment'] = df['Comment'].str.lower()

        withdrawl_keys = ['wd', 'wth', 'wdr', 'wdl', 'wthl', 'cshw', 'csh', 'withdrawal']
        transfer_keys = ['tr', 'trans', 'trtr', 'trf', 'trsf', 'tnf', 'transfer']
        bill_keys = ['bills', 'bill', 'phcn', 'ikedc', 'ekedc', 'dstv', 'gotv']
        channel_keys = ['nip', 'nibss', 'atm', 'quickteller', 'ussd', 'charges']
        charge_keys = ['tax', 'charge', 'vat', 'fee', 'commission', 'fees', 'maint', 'maintainance', 'nipfee', 'nipvat']
        purchase_keys = ['purchase', 'buy', 'payment', 'web', 'pos']
        airtime_keys = ['vtu', 'airtime purchase', 'airtime', 'mtn', 'etisalat', 'glo', 'airtel', 'topup', 'top up']
        salary_keys = ['salary', 'sal', 'renumeration', 'renum']

        df['Comment'] = df['Comment'].fillna('others')
        df['Comment'] = df.apply(lambda row: word_tokenize(row['Comment']), axis=1)

        def replaceMultiple(mainString, tobeReplaced, newstring):
            for elem in tobeReplaced:
                if elem in mainString:
                    if newstring not in mainString:
                        mainString = [string.replace(elem, newstring) for string in mainString]
            return mainString

        df['Comment'] = df.apply(lambda row: replaceMultiple(row['Comment'], withdrawl_keys, 'withdrawal'), axis=1)
        df['Comment'] = df.apply(lambda row: replaceMultiple(row['Comment'], transfer_keys, 'transfer'), axis=1)
        df['Comment'] = df.apply(lambda row: replaceMultiple(row['Comment'], purchase_keys, 'purchase'), axis=1)
        df['Comment'] = df.apply(lambda row: replaceMultiple(row['Comment'], channel_keys, 'channels'), axis=1)
        df['Comment'] = df.apply(lambda row: replaceMultiple(row['Comment'], charge_keys, 'charges'), axis=1)
        df['Comment'] = df.apply(lambda row: replaceMultiple(row['Comment'], airtime_keys, 'airtime'), axis=1)
        df['Comment'] = df.apply(lambda row: replaceMultiple(row['Comment'], bill_keys, 'bills'), axis=1)
        df['Comment'] = df.apply(lambda row: TreebankWordDetokenizer().detokenize(row['Comment']), axis=1)
        df['Comment'] = df['Comment'].str.replace('.*'+'charges'+'.*', 'charges', regex=True)
        df['Comment'] = df['Comment'].str.replace('.*'+'airtime'+'.*', 'airtime', regex=True)
        df['Comment'] = df['Comment'].str.replace('.*'+'withdrawal'+'.*', 'withdrawal', regex=True)
        df['Comment'] = df['Comment'].str.replace('.*'+'transfer'+'.*', 'transfer', regex=True)
        df['Comment'] = df['Comment'].str.replace('.*'+'purchase'+'.*', 'purchase', regex=True)
        df['Comment'] = df['Comment'].str.replace('.*'+'channel'+'.*', 'channel', regex=True)
        df['Comment'] = df['Comment'].str.replace('.*'+'bills'+'.*', 'bills', regex=True)

        if df['Balance'].dtype != np.number:
            df['Balance'] = df['Balance'].str.replace(',', '')
            df['Balance'] = df['Balance'].str.extract('(\-?\d*\.?\d*)', expand=False).astype(float)
        if df['Deposits'].dtype != np.number:
            df['Deposits'] = df['Deposits'].str.replace(',', '')
            df['Deposits'] = df['Deposits'].str.extract('(\d*\.?\d*)', expand=False)
            df['Deposits'] = df['Deposits'].replace(r'\s+',np.nan,regex=True).replace('',np.nan)
            df['Deposits'] = df['Deposits'].fillna(0.00).astype(float)
        if df['Withdrawls'].dtype != np.number:
            df['Withdrawls'] = df['Withdrawls'].str.replace(',', '')
            df['Withdrawls'] = df['Withdrawls'].str.extract('(\d*\.?\d*)', expand=False)
            df['Withdrawls'] = df['Withdrawls'].replace(r'\s+',np.nan,regex=True).replace('',np.nan)
            df['Withdrawls'] = df['Withdrawls'].fillna(0.00).astype(float)

        df.loc[(df['Comment'] == 'withdrawal') & (df['Withdrawls'] > 0), 'Category'] = 'Withdrawals'
        df.loc[(df['Comment'] == 'withdrawal') & (df['Deposits'] > 0), 'Category'] = 'Deposit'
        df.loc[(df['Comment'] == 'transfer') & (df['Withdrawls'] > 0), 'Category'] = 'Transfer'
        df.loc[(df['Comment'] == 'transfer') & (df['Deposits'] > 0), 'Category'] = 'Deposit'
        df.loc[(df['Comment'] == 'charges'), 'Category'] = 'Charges'
        df.loc[(df['Comment'] == 'charges') & (df['Withdrawls'] > 2000), 'Category'] = 'Transfer'
        df.loc[(df['Comment'] == 'charges') & (df['Withdrawls'] <= 500), 'Category'] = 'Charges'
        df.loc[(df['Comment'] == 'channel') & (df['Deposits'] > 0) & (df['Withdrawls'] == 0), 'Category'] = 'Deposit'
        df.loc[(df['Comment'] == 'channel') & (df['Withdrawls'] <= 200) & (df['Deposits'] == 0), 'Category'] = 'Charges'
        df.loc[(df['Comment'] == 'channel') & (df['Withdrawls'] > 200) & (df['Deposits'] == 0), 'Category'] = 'Transfer'
        df.loc[(df['Comment'] == 'airtime'), 'Category'] = 'Airtime'
        df.loc[(df['Comment'] == 'bills'), 'Category'] = 'Bills'
        df.loc[(df['Comment'] == 'purchase') & (df['Withdrawls'] > 0), 'Category'] = 'Purchase'
        df.loc[(df['Comment'] == 'purchase') & (df['Deposits'] > 0), 'Category'] = 'Deposit'

        other_cat = 'airtime|channel|transfer|purchase|charges|cash|bills'
        df.loc[~df['Comment'].str.contains(other_cat),'Category'] = 'Others'
        df['Category'] = df['Category'].fillna('Others')

        df.loc[(df['Category'] == 'Others') & (df['Deposits'] > 0) & (df['Deposits'] <= 500), 'Category'] = 'Others'
        df.loc[(df['Category'] == 'Others') & (df['Deposits'] > 500), 'Category'] = 'Deposit'
        df.loc[(df['Category'] == 'Others') & (df['Withdrawls'] > 500), 'Category'] = 'Withdrawal'
        df.loc[(df['Category'] == 'Others') & (df['Withdrawls'] > 0) & (df['Withdrawls'] <= 500), 'Category'] = 'Others'

        df['Category'] = df['Category'].str.upper()
        # converting data to dectionary
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        df = df.to_json(orient = 'records')

    return df

@application.route("/character", methods=['GET','POST'])
def risk_character():
    # importing data into system and saving it to df
    if request.method == 'POST':

        # reading in the csv file from a http post method
        df = pd.read_csv(request.files.get('file'))

        # creating the list of possible columns names in a bank statement

        date = ['transaction.date', 'date','trans. date', 'trans date', 'transaction date', 'posted date', 'txn date', 'posting date']
        description = ['description', 'remarks', 'narration', 'transaction details', 'transaction']
        withdrawal = ['withdrawls', 'debit', 'withdrawal(dr)', 'debits', 'payments', 'withdrawals', 'withdrawal']
        deposit = ['credit', 'deposits', 'lodgements', 'deposit(cr)', 'credits', 'deposit']
        balance = ['balance', 'balances', 'closing balance']

        for column in df:
            if column.lower() in date:
                df.rename({column:'Date'}, axis = 1, inplace=True)
        for column in df:
            if column.lower() in description:
                df.rename({column:'Description'}, axis = 1, inplace=True)
        for column in df:
            if column.lower() in withdrawal:
                df.rename({column:'Withdrawls'}, axis = 1, inplace=True)
        for column in df:
            if column.lower() in balance:
                df.rename({column:'Balance'}, axis = 1, inplace=True)
        for column in df:
            if column.lower() in deposit:
                df.rename({column:'Deposits'}, axis = 1, inplace=True)

        df = df[pd.notnull(df['Balance'])]

        new = df['Date'].astype(str).str.split("\s+", n=1, expand=True)
        # new = df['Date'].astype(str).str.split("\n", n=1, expand=True)
        # making separate first name column from new data frame
        df['Date'] = new[0]

        if(1 in new.columns):
            # making separate last name column from new data frame
            df["Wrong Date"] = new[1].copy()
            # Dropping old Name columns
            df.drop(columns=["Wrong Date"], inplace=True)
            # converting the Date string to a Date utility
            df['Date'] = pd.to_datetime(df['Date'])
        else:
            # converting the Date string to a Date utility
            df['Date'] = pd.to_datetime(df['Date'])

        df["Comment"] = df["Description"].str.replace(r"[^a-zA-Z ]+", " ").str.strip()
        df['Comment'] = df['Comment'].str.lower()
        df['Comment'] = df['Comment'].fillna('others')
        df['month'] = df['Date'].dt.strftime('%b')
        df['Transaction.Day'] = df['Date'].dt.day
        months = df['Date'].dt.strftime('%b')
        months = list(set(months))

        if df['Balance'].dtype != np.number:
            df['Balance'] = df['Balance'].str.replace(',', '')
            df['Balance'] = df['Balance'].str.extract('(\-?\d*\.?\d*)', expand=False).astype(float)
        if df['Deposits'].dtype != np.number:
            df['Deposits'] = df['Deposits'].str.replace(',', '')
            df['Deposits'] = df['Deposits'].str.extract('(\d*\.?\d*)', expand=False)
            df['Deposits'] = df['Deposits'].replace(r'\s+',np.nan,regex=True).replace('',np.nan)
            df['Deposits'] = df['Deposits'].fillna(0.00).astype(float)
        if df['Withdrawls'].dtype != np.number:
            df['Withdrawls'] = df['Withdrawls'].str.replace(',', '')
            df['Withdrawls'] = df['Withdrawls'].str.extract('(\d*\.?\d*)', expand=False)
            df['Withdrawls'] = df['Withdrawls'].replace(r'\s+',np.nan,regex=True).replace('',np.nan)
            df['Withdrawls'] = df['Withdrawls'].fillna(0.00).astype(float)

        df['Comment'] = df.apply(lambda row: word_tokenize(row['Comment']), axis=1)

        def replaceMultiple(mainString, tobeReplaced, newstring):
            for elem in tobeReplaced:
                if elem in mainString:
                    if newstring not in mainString:
                        mainString = [string.replace(elem, newstring) for string in mainString]
            return mainString


        purchase_keys = ['purchase', 'buy', 'web', 'pos','vtu', 'airtime purchase', 'airtime', 'mtn', 'etisalat',
                         'glo', 'airtel', 'topup', 'top up', 'vtup', 'vtop']
        # df['Comment'] = df.apply(lambda row: word_tokenize(row['Description']), axis=1)
        df['Comment'] = df.apply(lambda row: replaceMultiple(row['Comment'], purchase_keys, 'purchase'), axis=1)
        df['Comment'] = df.apply(lambda row: TreebankWordDetokenizer().detokenize(row['Comment']), axis=1)
        df['Comment'] = df['Comment'].str.replace('.*' + 'loan' + '.*', 'loan', regex=True)
        df['Comment'] = df['Comment'].str.replace('.*'+'purchase'+'.*', 'purchase', regex=True)

        df.loc[(df['Comment'] == 'purchase') & (df['Withdrawls'] > 0), 'Category'] = 'Purchase'

        def textSimilarity(mainString, checkString):
            num = 0
            for elem in mainString:
                if elem in checkString:
                    num = num+1
            similarity = num/len(checkString)
            return similarity

        salary_string = ['salary', 'sal', 'renumeration', 'remuneration', 'renum', 'jan', 'january','feb', 'february', 'march', 'mar', 'april', 'apr', 'may', 'june', 'jun', 'jul',
                         'july', 'aug',' august', 'sep', 'sept', 'september','oct', 'october', 'nov', 'november', 'dec', 'december']

        df['Comment'] = df.apply(lambda row: word_tokenize(row['Comment']), axis=1)
        df['TextSimilarity'] = df.apply(lambda row: textSimilarity(row['Comment'], salary_string), axis=1)
        df.loc[(df['TextSimilarity'] > 0) & (df['Deposits'] > 0), 'Category'] = 'Salary'
        df = df.reset_index()

        def sweeper_func(data, income):
            expense_data = []
            deposit_data = []
            try:
                for i in range(len(data)):
                    if data.Deposits[i] == income:
                        j = i+1
                        if data.Withdrawls[j] > 0:
                            value = data['Withdrawls'][j] + data['Withdrawls'][j+1] + data['Withdrawls'][j+2] + data['Withdrawls'][j+3] + data['Withdrawls'][j+4] + data['Withdrawls'][j+5] + data['Withdrawls'][j+6]
        #                     deposit_data.append(data['Deposits'][i])
                            expense_data.append(value)
                            break
            except KeyError as error:
                pass
            spent_avg = value/income
            if spent_avg > 0.80:
                sweeper = 'yes'
            else:
                sweeper = 'no'

            return sweeper, spent_avg, value

        def lavish_func(data, salary):
            withdraw_data = data.loc[data['Category'] == 'Purchase'].copy()
            spending = withdraw_data.loc[withdraw_data['month'] == month]['Withdrawls'].sum()
            spent_percentage = spending/salary
            if spent_percentage > 0.3:
                lavish = 'yes'
            else:
                lavish = 'no'
            return lavish, spent_percentage

        def prime_func(data, salary):
            daily_balance=df.loc[df['month'] == 'Apr'].groupby(df['Date'].dt.strftime('%d')).agg({'Balance':'last'})
            daily_balance=daily_balance.dropna()
            balance_list = [item for item in daily_balance['Balance'] if item >= 0.2*salary]
            if len(balance_list) >= 0.3*len(daily_balance):
                prime = 'yes'
            else:
                prime = 'no'
            return prime

        salary_data = df.loc[df['Category'] == 'Salary']

        count = 0
        salary_days = [1, 2, 3, 4, 5, 25, 26, 27, 28, 29, 30, 31]
        consistency_count = 0
        salary_day = []
        salary_list = []
        sweeper_profile = []
        lavish_profile = []
        lavish_rating = []
        sweeper_rating = []
        salary_month = []
        value_list = []
        if len(salary_data) > 0:
            for month in months:
                for row in range(len(df)):
                    salary = 0
                    if df['month'][row] == month:
                        if df['Category'][row] == 'Salary':
                            salary = df['Deposits'][row]
                            salary_list.append(salary)
                            salary_day.append(df['Transaction.Day'][row])
                            salary_month.append(month)
                            consistency_count = consistency_count + 1
                            sweeper = sweeper_func(df, salary)
                            lavish = lavish_func(df, salary)
                            sweeper_profile.append(sweeper[0])
                            sweeper_rating.append(sweeper[1])
                            lavish_rating.append(lavish[1])
                            lavish_profile.append(lavish[0])
            consistency_prop = consistency_count/len(df['month'].value_counts())
        else:
            salary = 'Not Available'
            salary_list.append(salary)
            salary_day.append('Not Available')
            salary_month.append('Not Available')
            consistency_count = 'Not Available'
            sweeper_profile.append('Not Available')
            sweeper_rating.append('Not Available')
            lavish_rating.append('Not Available')
            lavish_profile.append('Not Available')

        result = zip(salary_month, salary_day, salary_list, sweeper_profile, sweeper_rating, lavish_profile, lavish_rating)
        data_frame = pd.DataFrame(list(result), columns = ['salary_month', 'salary_day', 'salary_list', 'sweeper_profile', 'sweeper_rating', 'lavish_profile', 'lavish_rating'])
        month_dict = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}
        data_frame['month_index'] = data_frame['salary_month'].map(month_dict)

        if len(salary_data) > 0:
            character_list = []
        for i in range(len(data_frame)):
            if data_frame['sweeper_profile'][i] == 'yes':
                character = 'sweeper'
                character_list.append(character)
            elif data_frame['sweeper_profile'][i] == 'no':
                if data_frame['lavish_profile'][i] == 'yes':
                    character = 'lavish'
                    character_list.append(character)
                elif data_frame['lavish_profile'][i] == 'no':
                    character = 'prime'
                    character_list.append(character)

        data_frame['character'] = character_list
        data_frame = data_frame.sort_values(by='month_index')

        character_result = data_frame.to_json(orient='records')


    return character_result


@application.route("/capacity", methods=['GET','POST'])
def risk_capacity():
    # importing data into system and saving it to df
    if request.method == 'POST':

        # reading in the csv file from a http post method
        df = pd.read_csv(request.files.get('file'))

        # creating the list of possible columns names in a bank statement

        date = ['transaction.date', 'date','trans. date', 'trans date', 'transaction date', 'posted date', 'txn date', 'posting date']
        description = ['description', 'remarks', 'narration', 'transaction details', 'transaction']
        withdrawal = ['withdrawls', 'debit', 'withdrawal(dr)', 'debits', 'payments', 'withdrawals', 'withdrawal']
        deposit = ['credit', 'deposits', 'lodgements', 'deposit(cr)', 'credits', 'deposit']
        balance = ['balance', 'balances', 'closing balance']

        for column in df:
            if column.lower() in date:
                df.rename({column:'Date'}, axis = 1, inplace=True)
        for column in df:
            if column.lower() in description:
                df.rename({column:'Description'}, axis = 1, inplace=True)
        for column in df:
            if column.lower() in withdrawal:
                df.rename({column:'Withdrawls'}, axis = 1, inplace=True)
        for column in df:
            if column.lower() in balance:
                df.rename({column:'Balance'}, axis = 1, inplace=True)
        for column in df:
            if column.lower() in deposit:
                df.rename({column:'Deposits'}, axis = 1, inplace=True)

        df = df[pd.notnull(df['Balance'])]

        new = df['Date'].astype(str).str.split("\s+", n=1, expand=True)
        # new = df['Date'].astype(str).str.split("\n", n=1, expand=True)
        # making separate first name column from new data frame
        df['Date'] = new[0]

        if(1 in new.columns):
            # making separate last name column from new data frame
            df["Wrong Date"] = new[1].copy()
            # Dropping old Name columns
            df.drop(columns=["Wrong Date"], inplace=True)
            # converting the Date string to a Date utility
            df['Date'] = pd.to_datetime(df['Date'])
        else:
            # converting the Date string to a Date utility
            df['Date'] = pd.to_datetime(df['Date'])

        df["Comment"] = df["Description"].str.replace(r"[^a-zA-Z ]+", " ").str.strip()
        df['Comment'] = df['Comment'].str.lower()
        df['Comment'] = df['Comment'].fillna('others')
        df['month'] = df['Date'].dt.strftime('%b')
        df['Transaction.Day'] = df['Date'].dt.day
        months = df['Date'].dt.strftime('%b')
        months = list(set(months))

        if df['Balance'].dtype != np.number:
            df['Balance'] = df['Balance'].str.replace(',', '')
            df['Balance'] = df['Balance'].str.extract('(\-?\d*\.?\d*)', expand=False).astype(float)
        if df['Deposits'].dtype != np.number:
            df['Deposits'] = df['Deposits'].str.replace(',', '')
            df['Deposits'] = df['Deposits'].str.extract('(\d*\.?\d*)', expand=False)
            df['Deposits'] = df['Deposits'].replace(r'\s+',np.nan,regex=True).replace('',np.nan)
            df['Deposits'] = df['Deposits'].fillna(0.00).astype(float)
        if df['Withdrawls'].dtype != np.number:
            df['Withdrawls'] = df['Withdrawls'].str.replace(',', '')
            df['Withdrawls'] = df['Withdrawls'].str.extract('(\d*\.?\d*)', expand=False)
            df['Withdrawls'] = df['Withdrawls'].replace(r'\s+',np.nan,regex=True).replace('',np.nan)
            df['Withdrawls'] = df['Withdrawls'].fillna(0.00).astype(float)

        df['Comment'] = df.apply(lambda row: word_tokenize(row['Comment']), axis=1)

        def replaceMultiple(mainString, tobeReplaced, newstring):
            for elem in tobeReplaced:
                if elem in mainString:
                    if newstring not in mainString:
                        mainString = [string.replace(elem, newstring) for string in mainString]
            return mainString

        loan_keys = ['loan', 'repayment', 'loan repayment', 'loans', 'adims credit and investment limited',
                     'aso savings & loans plc', 'city-code mortgage bank ltd', 'coop savings & loans limited',
                     'crc credit bureau', 'credcentral', 'direct bridge nigeria limited', 'one credit', 'questmoney',
                     'resort savings and loans plc', 'startcredit', 'tiagocredit', 'union homes savings and loans plc',
                     'cashbridge global & leasing co.', 'centage savings & loans', 'eurobank savings & loans limited',
                     'grofin', 'smedan', 'lidya', 'aella credit', 'zedvance', 'paylater', 'kiakia', 'onefi',
                     'c24 limited', 'quickcheck', 'specta', 'fastcredit', 'renmoney', 'kwik cash',
                     'rosabon finance quick loan', 'page financials', 'fint loan', 'payconnect', 'fairmoney',
                     'snapcredit']
        purchase_keys = ['purchase', 'buy', 'web', 'pos','vtu', 'airtime purchase', 'airtime', 'mtn', 'etisalat',
                         'glo', 'airtel', 'topup', 'top up', 'vtup', 'vtop']
        # df['Comment'] = df.apply(lambda row: word_tokenize(row['Description']), axis=1)
        df['Comment'] = df.apply(lambda row: replaceMultiple(row['Comment'], purchase_keys, 'purchase'), axis=1)
        df['Comment'] = df.apply(lambda row: replaceMultiple(row['Comment'], loan_keys, 'loan'), axis=1)
        df['Comment'] = df.apply(lambda row: TreebankWordDetokenizer().detokenize(row['Comment']), axis=1)
        df['Comment'] = df['Comment'].str.replace('.*' + 'loan' + '.*', 'loan', regex=True)
        df['Comment'] = df['Comment'].str.replace('.*'+'purchase'+'.*', 'purchase', regex=True)

        df.loc[(df['Comment'] == 'purchase') & (df['Withdrawls'] > 0), 'Category'] = 'Purchase'
        df.loc[(df['Comment'] == 'loan') & (df['Withdrawls'] > 0) & (df['Deposits'] == 0), 'Category'] = 'Loan Repayment'

        def textSimilarity(mainString, checkString):
            num = 0
            for elem in mainString:
                if elem in checkString:
                    num = num+1
            similarity = num/len(checkString)
            return similarity

        salary_string = ['salary', 'sal', 'renumeration', 'remuneration', 'renum', 'jan', 'january','feb', 'february', 'march', 'mar', 'april', 'apr', 'may', 'june', 'jun', 'jul',
                         'july', 'aug',' august', 'sep', 'sept', 'september','oct', 'october', 'nov', 'november', 'dec', 'december']

        df['Comment'] = df.apply(lambda row: word_tokenize(row['Comment']), axis=1)
        df['TextSimilarity'] = df.apply(lambda row: textSimilarity(row['Comment'], salary_string), axis=1)
        df.loc[(df['TextSimilarity'] > 0) & (df['Deposits'] > 0), 'Category'] = 'Salary'
        df = df.reset_index()

        def sweeper_func(data, income):
            expense_data = []
            deposit_data = []
            try:
                for i in range(len(data)):
                    if data.Deposits[i] == income:
                        j = i+1
                        if data.Withdrawls[j] > 0:
                            value = data['Withdrawls'][j] + data['Withdrawls'][j+1] + data['Withdrawls'][j+2] + data['Withdrawls'][j+3] + data['Withdrawls'][j+4] + data['Withdrawls'][j+5] + data['Withdrawls'][j+6]
        #                     deposit_data.append(data['Deposits'][i])
                            expense_data.append(value)
                            break
            except KeyError as error:
                pass
            spent_avg = value/income
            if spent_avg > 0.80:
                sweeper = 'yes'
            else:
                sweeper = 'no'

            return sweeper, spent_avg, value

        def lavish_func(data, salary):
            withdraw_data = data.loc[data['Category'] == 'Purchase'].copy()
            spending = withdraw_data.loc[withdraw_data['month'] == month]['Withdrawls'].sum()
            spent_percentage = spending/salary
            if spent_percentage > 0.3:
                lavish = 'yes'
            else:
                lavish = 'no'
            return lavish, spent_percentage

        def prime_func(data, salary):
            daily_balance=df.loc[df['month'] == 'Apr'].groupby(df['Date'].dt.strftime('%d')).agg({'Balance':'last'})
            daily_balance=daily_balance.dropna()
            balance_list = [item for item in daily_balance['Balance'] if item >= 0.2*salary]
            if len(balance_list) >= 0.3*len(daily_balance):
                prime = 'yes'
            else:
                prime = 'no'
            return prime

        salary_data = df.loc[df['Category'] == 'Salary']

        count = 0
        salary_days = [1, 2, 3, 4, 5, 25, 26, 27, 28, 29, 30, 31]
        consistency_count = 0
        salary_day = []
        salary_list = []
        sweeper_profile = []
        lavish_profile = []
        lavish_rating = []
        sweeper_rating = []
        salary_month = []
        value_list = []
        if len(salary_data) > 0:
            for month in months:
                for row in range(len(df)):
                    salary = 0
                    if df['month'][row] == month:
                        if df['Category'][row] == 'Salary':
                            salary = df['Deposits'][row]
                            salary_list.append(salary)
                            salary_day.append(df['Transaction.Day'][row])
                            salary_month.append(month)
                            consistency_count = consistency_count + 1
                            sweeper = sweeper_func(df, salary)
                            lavish = lavish_func(df, salary)
                            sweeper_profile.append(sweeper[0])
                            sweeper_rating.append(sweeper[1])
                            lavish_rating.append(lavish[1])
                            lavish_profile.append(lavish[0])
            consistency_prop = consistency_count/len(df['month'].value_counts())
        else:
            salary = 'Not Available'
            salary_list.append(salary)
            salary_day.append('Not Available')
            salary_month.append('Not Available')
            consistency_count = 'Not Available'
            sweeper_profile.append('Not Available')
            sweeper_rating.append('Not Available')
            lavish_rating.append('Not Available')
            lavish_profile.append('Not Available')

        loan_repayment_data = df.loc[df['Category'] == 'Loan Repayment']
        result = zip(salary_month, salary_day, salary_list, sweeper_profile, sweeper_rating, lavish_profile, lavish_rating)
        data_frame = pd.DataFrame(list(result), columns = ['salary_month', 'salary_day', 'salary_list', 'sweeper_profile', 'sweeper_rating', 'lavish_profile', 'lavish_rating'])
        month_dict = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}
        data_frame['month_index'] = data_frame['salary_month'].map(month_dict)

        if len(salary_data) > 0:
            character_list = []
        for i in range(len(data_frame)):
            if data_frame['sweeper_profile'][i] == 'yes':
                character = 'sweeper'
                character_list.append(character)
            elif data_frame['sweeper_profile'][i] == 'no':
                if data_frame['lavish_profile'][i] == 'yes':
                    character = 'lavish'
                    character_list.append(character)
                elif data_frame['lavish_profile'][i] == 'no':
                    character = 'prime'
                    character_list.append(character)

        data_frame['character'] = character_list
        data_frame = data_frame.sort_values(by='month_index')

        def loan_check(loan_repayment, salary):
            if loan_repayment == 0:
                capacity = 0.85
                capacity_factor = 'safe'
            elif loan_repayment > 0:
                loan_char = loan_repayment/salary
                if loan_char <= 0.1:
                    capacity = 0.9
                    capacity_factor = 'prudent'
                elif loan_char > 0.1 and loan_char <= 0.2:
                    capacity = 0.7
                    capacity_factor = 'exposed'
                elif loan_char > 0.2:
                    capacity = 0.6
                    capacity_factor = 'geared'
            return capacity, capacity_factor

        loan_repayment_day = []
        loan_repayment_list = []
        loan_repayment_month = []
        capacity_rate_list = []
        capacity_factor_list = []
        if len(loan_repayment_data) > 0:
            for month in salary_month:
                for row in range(len(df)):
                    if df['month'][row] == month:
                        if df['Category'][row] == 'Loan Repayment':
                            loan_repayment = df['Withdrawls'][row]
                            loan_repayment_list.append(loan_repayment)
                            loan_repayment_day.append(df['Transaction.Day'][row])
                            loan_repayment_month.append(month)
                            if df['Balance'][row] >= 0:
                                salary = float(data_frame.loc[data_frame['salary_month'] == month ]['salary_list'])
                                capacity = loan_check(loan_repayment, salary)
                                capacity_rate = capacity[0]
                                capacity_rate_list.append(capacity_rate)
                                capacity_factor = capacity[1]
                                capacity_factor_list.append(capacity_factor)
                            elif df['Balance'][row] < 0:
                                capacity_rate = 0.1
                                capacity_rate_list.append(capacity_rate)
                                capacity_factor = 'runner'
                                capacity_factor_list.append(capacity_factor)

        else:
            loan_repayment = 'Not Available'
            loan_repayment_list.append(loan_repayment)
            loan_repayment_day.append('Not Available')
            loan_repayment_month.append('Not Available')
            capacity_rate = 'Not Available'
            capacity_rate_list.append('Not Available')
            capacity_factor = 'Not Available'
            capacity_factor_list.append('Not Available')

        loan_result = (zip(loan_repayment_month, loan_repayment_day, loan_repayment_list, capacity_rate_list, capacity_factor_list))
        loan_data = pd.DataFrame(list(loan_result), columns = ['month', 'day', 'loan_repayment', 'capacity_rate', 'capacity_factor'])

        character_result = data_frame.to_json(orient='records')
        capacity_result = loan_data.to_json(orient='records')


    return capacity_result

if __name__ == "__main__":
    application.run(debug=True, use_reloader=False)
