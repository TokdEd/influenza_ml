import pandas as pd
file = "stage-2/data/Asia_FluNet_1_clean.csv"
df=pd.read_csv(file)
target =df['COUNTRY_AREA_TERRITORY'].unique() 
print(target)
