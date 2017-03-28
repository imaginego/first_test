import pandas as pd 

def read_data():
    data_path = "../input/"
    train_file = data_path + "train.json"
    test_file = data_path + "test.json"
    train_df = pd.read_json(train_file)
    test_df = pd.read_json(test_file)
    interest_map = {'low':0,'medium':1,'high':2}
    train_df['interest_level'] = train_df['interest_level'].map(interest_map)
    fmt = lambda s: s.replace("\u00a0", "").strip().lower()
    train_df["street_address"] = train_df['street_address'].apply(fmt)
    train_df["display_address"] = train_df["display_address"].apply(fmt)
    #original_col = train_df.columns
    test_df["street_address"] = test_df['street_address'].apply(fmt)
    test_df["display_address"] = test_df["display_address"].apply(fmt)
    
    return train_df,test_df

def write_output(preds,test_df,prefix=''):
    out_df = pd.DataFrame(preds)
    out_df.columns = ["low", "medium", "high"]
    out_df["listing_id"] = test_df.listing_id.values
    
    import time
    filename = prefix + time.strftime("%Y.%m.%d.") + str(np.random.randint(0,100000))+'.res.csv'
    out_df.to_csv(filename, index=False)