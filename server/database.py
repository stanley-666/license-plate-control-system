import json

class Record():
    def __init__(self,filename):
        with open(filename,"r") as file:
            data = json.load(file)
        file.close()
        self.data = data
    def dump(self):
        print(self.data)
        return self.data
    def IS_LEGAL(self, plate_number):
        data = self.data["car_plate"]
        for car in data:
            if car["plate_number"] == plate_number and car["registered"] == "YES" :
                return True
        return False
                
def search_plate(records,str):
    record = records
    #record.dump()
    #data = record.dump()
    plate_number = str
    if record.IS_LEGAL(plate_number):
        print(plate_number + " is in the list.")
        return True
    print(plate_number + " is not in the list.")
    return False
        
    