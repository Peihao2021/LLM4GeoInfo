from gpt4 import query, createCitiesList, zero_shot
import pandas as pd

def main():
    in_file_loc = 'worldcities100k.pkl'
    df = pd.read_pickle(in_file_loc)
    df_zero_shot = zero_shot(df)
    df_zero_shot.head()
    df_zero_shot.to_csv('geo-o-shot.csv', index=False)


if __name__ == "__main__":
    main()