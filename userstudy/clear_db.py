from pymongo import MongoClient

client = MongoClient('0.0.0.0', 7771)
db = client.userstudy
collection = db.score
userdb = db.user

collection.drop()
userdb.drop()

