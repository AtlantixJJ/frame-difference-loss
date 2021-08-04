from pymongo import MongoClient
client = MongoClient('0.0.0.0', 27018)
db = client.video_20180708
db.drop_collection("score")