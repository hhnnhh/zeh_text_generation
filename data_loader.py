
h = open("./data/zeh.txt", encoding='utf-8')
h_data = h.readlines()
h_data[0:3]  # 显示前3封正常邮件
type(h_data)
data = str(h_data)
type(data)
train, test = train_test_split(h_data,test_size=0.15)
with open('train.txt', 'w') as f:
    for item in train:
        f.write("%s\n" % item)
with open('test.txt', 'w') as f:
    for item in test:
        f.write("%s\n" % item)
print("Train dataset length: "+str(len(train)))
print("Test dataset length: "+ str(len(test)))