import pandas as pd

df = pd.DataFrame({'ColA': [1, 2], 'ColB': [3, 4], 'ColC': [5, 6]})
with pd.ExcelWriter('test_columns.xlsx') as writer:
    df.to_excel(writer, sheet_name='Spectra', index=False)
    df.to_excel(writer, sheet_name='Concentration', index=False)
print("Created test_columns.xlsx")
