import pandas as pd

test_data_dir = './data/Second_TestDataSet.csv'
sub_example_dir = './data/submit_example.csv'
test_data = pd.read_csv(test_data_dir)
df_examples = pd.DataFrame(pd.read_csv(sub_example_dir))
df_test = pd.DataFrame(test_data.values, index=test_data['id'].values, columns=['id', 'title', 'content'])
print(df_test)

examples_id = df_examples['id'].values
# print(examples_id)
df_test['tmp'] = df_test.index
df_test['tmp'] = df_test['tmp'].astype('category')
df_test['tmp'].cat.reorder_categories(examples_id, inplace=True)
print(df_test)
df_test.sort_values('tmp', inplace=True)
print(df_test)
df_test[['id', 'title', 'content']].to_csv('./data/test_v2_right_order.csv',index=False)
