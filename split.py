import pandas as pd


def split_function(file_path, column):  # function to split the input excel file into multiple excel files based on column name

    df_read_data = pd.read_excel(file_path)  # reading the data from file in file-path to data-frame
    # df_read_data[column].fillna(0, inplace = True)  # to fill empty cells with some value (newly added)
    # print(df_read_data)
    df_read_data[column] = df_read_data[column].str.strip()  # to remove whitespaces
    df_read_data[column] = df_read_data[column].str.replace(" ", ",")  # replacing the spaces between values with comma
    df_read_data[column] = df_read_data[column].str.split(",")  # splitting the data frame based on comma
    df_read_data = df_read_data.apply(pd.Series.explode)   # split the values into multiple rows
    condition_values = df_read_data[column].unique()
    for value in condition_values:
        df_split_data = df_read_data[df_read_data[column] == value]
        output_excel = column + str(value) + ".xlsx"
        df_split_data.to_excel(output_excel, index=False)


def split_age_function(file_path, column):
    df_read_data = pd.read_excel(file_path)
    df_read_data[column].fillna(0, inplace=True)
    age_values = df_read_data[column]
    # print(age_values)
    for value in age_values:
        # print(value)
        if (value <= 20) & (value > 0):
            age_1 = (df_read_data[column] <= 20) & (df_read_data[column] > 0)
            # print(df_read_data[age_1])
            output_file = "Age_0-20" + ".xlsx"
            df_read_data[age_1].to_excel(output_file, index=False)
        elif (value <= 50) & (value > 20):
            age_2 = (df_read_data[column] <= 50) & (df_read_data[column] > 20)
            # print(df_read_data[age_2])
            output_file = "Age_20-50" + ".xlsx"
            df_read_data[age_2].to_excel(output_file, index=False)
        elif (value <= 80) & (value > 50):
            age_3 = (df_read_data[column] <= 80) & (df_read_data[column] > 50)
            # print(df_read_data[age_3])
            output_file = "Age_50-80" + ".xlsx"
            df_read_data[age_3].to_excel(output_file, index=False)
        else:
            age_4 = (df_read_data[column] <= 100) & (df_read_data[column] > 80)
            # print(df_read_data[age_4])
            output_file = "Age_80-100" + ".xlsx"
            df_read_data[age_4].to_excel(output_file, index=False)


def main():

    file_path = input("please enter excel file path: ")
    for i in range(4):
        column = input("please enter column name that need to be split: ")
        if (column == 'Age'):
            split_age_function(file_path, column)
        else:
            split_function(file_path, column)


main()
