import json

test_dict = {'bigberg': [7600, {1: [['iPhone', 6300], ['Bike', 800], ['shirt', 300]]}]}
print(test_dict)
with open("./record_test.json","w") as file:
    json.dump(test_dict,file)
    print("write ok")
