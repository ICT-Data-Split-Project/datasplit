import pandas as pd


def split_function(file_path, column):
    df_read_data = pd.read_excel(file_path)  # reading the data from file in file-path to data-frame
    df_read_data[column] = df_read_data[column].str.strip()
    df_read_data[column] = df_read_data[column].str.replace(" ", ",")
    df_read_data[column] = df_read_data[column].str.split(",")
    df_read_data = df_read_data.apply(pd.Series.explode)
    condition_values = df_read_data[column].unique()
    for value in condition_values:
        df_split_data = df_read_data[df_read_data[column] == value]
        output_excel = column + str(value) + ".xlsx"
        df_split_data.to_excel(output_excel, index=False)


def main():
    file_path = input("please enter excel file path: ")
    for i in range(2):
        column = input("please enter column name that need to be split: ")
        split_function(file_path, column)


main()
