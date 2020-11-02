

def load_data():
    print("loading data...")
    data_path = "./data/zeh.txt"
    h = open(data_path, encoding='utf-8')
    h_data = h.readlines()
    h_data
    #h_data[0:3]
    print("data type is:", type(h_data))
    data = str(h_data)
    print("data type changed to:", type(data))
    clean_data = data.replace("\n", " ")
    clean_data.load()
    print("data stripped from newlines")
        return print("data loaded, named 'clean_data' ")

load_data()
clean_data
#train, test = train_test_split(h_data,test_size=0.15)
# with open('train.txt', 'w') as f:
#     for item in train:
#         f.write("%s\n" % item)
# with open('test.txt', 'w') as f:
#     for item in test:
#         f.write("%s\n" % item)
# print("Train dataset length: "+str(len(train)))
# print("Test dataset length: "+ str(len(test)))