import pandas as pd


def split_function(file_path, column):
    df_read_data = pd.read_excel(file_path)  # reading the data from file in filepath to dataframe
    condition_values = df_read_data[column].unique()
    for value in condition_values:
        df_split_data = df_read_data[df_read_data[column] == value]
        output_excel = column + str(value) + ".xlsx"
        df_split_data.to_excel(output_excel, index=False)


def main():
    for i in range(0, 1):
        file_path = input("please enter excel file path: ")
        column = input("please enter column name that need to be split: ")
        split_function(file_path, column)


main()