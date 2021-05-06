import pandas as pd
import sys
import time

import pandas # For reading from CSV file.
import numpy  # Slicing individual rows of data.
from numpy import array
from numpy import reshape

# Scikit-Learn data preparation.
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


#Skikit-Learn prediction algorithms.
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF # For GaussianProcessClassifier.
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# Didn't use histogram-based gradient boosting.
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

# Scikit-Learn measures of prediction accuracy.
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

pd.options.mode.chained_assignment = None  # default='warn'

algorithm_names=['GaussianNB','Gradient boosting','LDA','QDA','ADA BOOST','DecisionTreeClassifier','Gaussian process classifier','Random forest classifier'];
column_names=['ID-number','Age','Gender','SNA','SGLU','Glucose corrected sodium','SK','SCA','SALB','Albumin corrected calcium','SPO','SMG','Free calcium','Department','Death nr days','Main condition','Other conditions','Readmission_nr_days']
def get_choice(options):
    user_choice = input()
    while user_choice not in options:
        print("I’m sorry, I cannot understand.\n Please enter the corresponding number of the column  that need to be "
              "split\n "
              " (1) Main condition\n (2) Other conditions\n (3) Gender\n (4) Age\n (5) Quit")
        user_choice = input()
    return user_choice


def data_clean(file_path):  # function to clean the data in the dataset

    df_read_data = pd.read_excel(file_path)  # reading the data from file in file-path to data-frame
    df_read_data.fillna(0, inplace=True)
    df_read_data['Gender'] = df_read_data['Gender'].replace(['M', 'K'], [0, 1])
    df_read_data['Department'] = df_read_data['Department'].replace(['KIR', 'MED'], [1, 2])
    return df_read_data

def algorithm(my_dataset,algorithm_number,file_name):
    column_array=[]
    prediction_columns = ['Thresh', 'TP', 'FP', 'FN', 'TN', 'Precision_P', 'Recall_P']
    df_algo=pd.DataFrame(columns=prediction_columns)
    number_columns = len(my_dataset.columns)
    for i in range(0, number_columns):
        column_array.append(i)
    Y_list = column_array.pop()
    num_positives = 0.001
    if list(my_dataset.Readmission_nr_days).count(1)>0:
        num_positives=list(my_dataset.Readmission_nr_days).count(1)
    num_lines = len(my_dataset.index)  # One row of headers and 510 rows of data.

    X=my_dataset.iloc[:,column_array].values # input columns number
    Y=my_dataset.iloc[:,Y_list].values # prediction column
    #print(X)
    #print(Y)

    # Store results in a big array.
    all_results = [[0 for m in range(num_lines)] for n in range(5000)]

    # Do predictions for increasingly strict probabilities.
    # Precision equals TP / (TP + FP)
    # Recall = TP Rate TP / (TP + FN)
    print("#Starting on " + str(num_lines) + " leave-one-out trainings.", end="")
    sys.stdout.flush()  # Flush the buffer, so we can see progress.
    for line in range(0, num_lines):
        # Leave-one-out cross-validation, test data is a single row.
        X_test = array(X[line])
        X_test = X_test.reshape((1, X_test.shape[0]))
        y_test = array(Y[line])
        y_test = y_test.reshape((1))

        # Training data is everything except that row of test data.
        X_train = numpy.delete(X, (line), axis=0)
        y_train = numpy.delete(Y, (line), axis=0)
        if algorithm_number == 0:
            # Gaussian naive Bayesian.
            clf = GaussianNB()
        elif algorithm_number == 1:
            # Gradient boosting.
            clf = GradientBoostingClassifier()
        elif algorithm_number == 2:
            # Linear discriminant analysis (LDA).
            clf = LinearDiscriminantAnalysis()
        elif algorithm_number == 3:
            # Quadratic discriminant analysis (QDA).
            clf = QuadraticDiscriminantAnalysis()
        elif algorithm_number==4:
            # Ada boosting.
            clf = AdaBoostClassifier(n_estimators=100, random_state=17)
        elif algorithm_number==5:
            #Decision tree classifier.
            clf = DecisionTreeClassifier(random_state=17)
        elif algorithm_number==6:
             #Gaussian process classifier.
            my_kernel = 1.0 * RBF(1.0) # Radial basis function.
            clf = GaussianProcessClassifier(kernel=my_kernel, random_state=17)
        else:
            #Random forest classifier.
            clf = RandomForestClassifier(max_depth=10, random_state=17)

        # Use the training data to create a model.
        clf.fit(X_train, y_train)

        # Find the model's predicted probability of each output.
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)

        # Uncomment to show that 'Churn' is the first probability.
        # print(line, "Prediction", y_pred, "  Actual", y_test, " ",
        #            y_prob, (y_prob[0][0]+y_prob[0][1]))

        # Eye candy.
        if (0 == (line % 10)):
            print(".", end="")
            sys.stdout.flush()  # Flush the buffer, so we can see progress.

        # Is it sufficiently confident to venture a prediction?
        for threshint in range(5000, 10000):
            thresh = threshint / 10000.0  # Threshold goes from 0.5 to 0.999
            my_TP = 0
            my_FP = 0
            my_FN = 0
            my_TN = 0
            if (y_prob[0][0] >= thresh):  # First probability is P = churn, not stay.
                if (y_pred == y_test):
                    my_TP = 1
                    my_FP = 0
                else:
                    my_FP = 1
                    my_TP = 0
            elif (y_prob[0][1] >= thresh):  # Second probability is N = stay, not churn.
                if (y_pred == y_test):
                    my_TN = 1
                    my_FN = 0
                else:
                    my_FN = 1
                    my_TN = 0

            # Should be only one or zero predictions.
            if ((my_TP + my_FP + my_FN + my_TN) > 1):
                print("Error, should be only one prediction, or none, not ",
                      (my_TP + my_FP + my_FN + my_TN))

            # Store that set of results, as an immutable list.
            # print(thresh, line, y_prob[0][0], y_prob[0][1], my_TP, my_FP, my_FN, my_TN)
            single_result = (thresh, line, my_TP, my_FP, my_FN, my_TN)
            all_results[threshint - 5000][line] = single_result
    print(". done.")

    # So how did it go on all those leave-one-out data sets?
    print("#Thresh TP FP FN TN Precision_P    Recall_P")
    for threshint in range(5000, 10000):
        thresh = threshint / 10000.0  # Threshold goes from 0.5 to 0.999
        prev_TP = my_TP
        prev_FP = my_FP
        prev_FN = my_FN
        prev_TN = my_TN
        my_TP = 0
        my_FP = 0
        my_FN = 0
        my_TN = 0
        for line in range(0, num_lines):
            single_result = all_results[threshint - 5000][line]
            # print(single_result)
            if (1 == single_result[2]):
                my_TP += 1
            elif (1 == single_result[3]):
                my_FP += 1
            elif (1 == single_result[4]):
                my_FN += 1
            elif (1 == single_result[5]):
                my_TN += 1

        # Any different from the previous threshold?
        if ((threshint > 5000) and
                (prev_TP == my_TP) and
                (prev_FP == my_FP) and
                (prev_FN == my_FN) and
                (prev_TN == my_TN)):
            pass  # Don't print anything, same as previous line.
        else:
            # If there were any predictions, calculate precision and TP rate.
            if ((my_TP + my_FP) > 0):
                my_precis = my_TP / (my_TP + my_FP)
            else:
                my_precis = 0

            if ((my_TP + my_FN) > 0):
                my_recall = my_TP / (num_positives)  # Out of all positive cases.
            else:
                my_recall = 0

            print(str(thresh), "  ", my_TP, my_FP, my_FN, my_TN,
                  "  ", my_precis, my_recall)
            prediction_values=[str(thresh), my_TP, my_FP, my_FN, my_TN, my_precis, my_recall]
            dict={}


            for i in range(len(df_algo.columns)):
                dict[df_algo.columns[i]]=prediction_values[i]
            df_algo=df_algo.append(dict,ignore_index=True)
            print(df_algo)
            df_algo.to_excel(file_name + 'algorithm.xlsx')


    # All finished, repeat the row of headers.
    print("#Thresh TP FP FN TN Precision_P    Recall_P")
    #print('test')
    for i in range(8):
        df_algo.to_excel(file_name + algorithm_names[i]+ '.xlsx')


def mapping(df_read_data, column):  # function to replace disease code string values with numeric values
    condition_codes = []
    df_read_data[column] = df_read_data[column].str.strip()  # to remove whitespaces
    df_read_data[column] = df_read_data[column].str.replace(" ", ",")  # replacing the spaces between values with comma
    df_read_data[column] = df_read_data[column].str.split(",")  # splitting the dataframe based on comma
    df_read_data = df_read_data.apply(pd.Series.explode)  # split the values into multiple rows
    condition_values = df_read_data[column].unique()
    for value in range(len(condition_values)):
        condition_codes.append(str(value))
    df_read_data[column] = df_read_data[column].replace(condition_values, condition_codes)
    df_map_other_data = {'Disease_code': condition_values, 'Map_code': condition_codes}
    df = pd.DataFrame(df_map_other_data, columns=['Disease_code', 'Map_code'])
    # print(df)
    df.to_excel(column + "_code_map.xlsx", index=False)
    return df_read_data



def split_function(df_read_data,
                   column):  # function to split the input excel file into multiple excel files based on column name

    condition_values = df_read_data[column].unique()
    # print(df_read_data)

    for value in condition_values:
        # print(value)
        df_split_data = df_read_data[df_read_data[column] == value]
        #print(df_split_data)
        # df_split_data.drop(column, inplace=True, axis=1) # to drop the column
        output_excel = column + str(value)
        df_split_data.to_excel(output_excel+ ".xlsx", index=False)
        algorithm_number=[i for i in range(8)]
        #for i in algorithm_number:
         #   print(algorithm_names[i])
            #algorithm(df_split_data, algorithm_number)

        if len(df_split_data) > 20:
            for i in algorithm_number:
                print(algorithm_names[i])
                algorithm(df_split_data,algorithm_number,output_excel)






def split_age_function(df_read_data,
                       column):  # function to split the input excel file into multiple excel files based on age
    # column name
    # df_read_data = pd.read_excel(file_path)
    df_read_data.sort_values(by=[column], ascending=True)
    # df_read_data[column].fillna(0, inplace=True)
    # print(df_read_data)
    age_values = df_read_data[column]
    algorithm_number = [i for i in range(8)]
    # print(age_values)
    for value in age_values:
        # print(value)
        if (value <= 20) & (value > 0):
            age_1 = (df_read_data[column] <= 20) & (df_read_data[column] > 0)
            # print(df_read_data[age_1])
            output_file = "Age_0-20" + ".xlsx"
            df_read_data[age_1].to_excel(output_file, index=False)
            if len(df_read_data[age_1]) > 20:
                for i in algorithm_number:
                    print(algorithm_names[i])
                    algorithm(df_read_data[age_1], algorithm_number,output_file)

        elif (value <= 50) & (value > 20):
            age_2 = (df_read_data[column] <= 50) & (df_read_data[column] > 20)
            # print(df_read_data[age_2])
            output_file = "Age_20-50" + ".xlsx"
            df_read_data[age_2].to_excel(output_file, index=False)
            if len(df_read_data[age_2]) > 20:
                for i in algorithm_number:
                    print(algorithm_names[i])
                    algorithm(df_read_data[age_2], algorithm_number,output_file)
        elif (value <= 80) & (value > 50):
            age_3 = (df_read_data[column] <= 80) & (df_read_data[column] > 50)
            # print(df_read_data[age_3])
            output_file = "Age_50-80" + ".xlsx"
            df_read_data[age_3].to_excel(output_file, index=False)
            if len(df_read_data[age_3]) > 20:
                for i in algorithm_number:
                    print(algorithm_names[i])
                    algorithm(df_read_data[age_3], algorithm_number,output_file)
        else:
            age_4 = (df_read_data[column] <= 100) & (df_read_data[column] > 80)
            # print(df_read_data[age_4])
            output_file = "Age_80-100" + ".xlsx"
            df_read_data[age_4].to_excel(output_file, index=False)
            if len(df_read_data[age_4]) > 20:
                for i in algorithm_number:
                    print(algorithm_names[i])
                    algorithm(df_read_data[age_4], algorithm_number,output_file)


def show_menu():
    print(
        "Please enter the corresponding number of the column  that need to be split\n"
        " (1) Main condition\n (2) Other conditions\n (3) Gender\n (4) Age\n (5) Quit")


def main():
    file_path = input("please enter excel file path: ")
    options = ['1', '2', '3', '4', '5']
    show_menu()
    final_data = data_clean(file_path)
    split_data = mapping(final_data, 'Main condition')
    final_split_data = mapping(split_data, 'Other conditions')

    # print(
    #     "Please enter the corresponding number of the column  that need to be split \n"
    #     "(1) Main condition\n (2) Other conditions\n (3) Gender\n (4) Age\n  (5) Quit")

    column_selection = get_choice(options)
    while column_selection != '5':
        if column_selection == '1':
            column = "Main condition"
            split_function(final_split_data, column)
            print("File split completed for column", column)
            show_menu()

        elif column_selection == '2':
            column = "Other conditions"
            split_function(final_split_data, column)
            print("File split completed for column", column)
            show_menu()

        elif column_selection == '3':
            column = "Gender"
            split_function(final_split_data, column)
            print("File split completed for column", column)
            show_menu()

        elif column_selection == '4':
            column = "Age"
            split_age_function(final_split_data, column)
            print("File split completed for column", column)
            show_menu()

        column_selection = get_choice(options)

    print("Thank you.Goodbye!!")


main()
