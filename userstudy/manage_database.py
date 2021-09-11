"""
Directly managing the database.
"""
import sys
from pymongo import MongoClient
from filelock import Timeout, FileLock
import util

client = MongoClient('0.0.0.0', 27027)
db = client.userstudy
collection = db.score
userdb = db.user
lock = FileLock("sync.lock")

# access the score collection
if sys.argv[1] == "score":
    cursor = collection.find()
    for _ in range(cursor.count()):
        obj = cursor.next()
        print(obj)

# clear the database
if sys.argv[1] == "clear":
    collection.drop()
    userdb.drop()


# access user database
if sys.argv[1] == "user":
    cursor = userdb.find()
    for _ in range(cursor.count()):
        obj = cursor.next()
        print(obj)

# check the sync
if sys.argv[1] == "check":
    print("=> Check the sync file")
    cursor = userdb.find()
    occupied_inds = []
    ind_name = {}
    for _ in range(cursor.count()):
        obj = cursor.next()
        if int(obj["id"]) > 1000: #admin
            continue
        
        if obj["id"] not in ind_name.keys():
            ind_name[obj["id"]] = []
        ind_name[obj["id"]].append(obj["name"])

        inds = []
        inds.extend(obj["used_id"])
        if obj["id"] not in inds:
            inds.append(obj["id"])
        for ind in inds:
            if ind in occupied_inds:
                print("!> ID=%d is occupied many times!" % ind)
                print("!> Used by %s" % " , ".join(ind_name[ind]))
            else:
                occupied_inds.append(ind)
    occupied_inds.sort()
    print(occupied_inds)

    # search for missing
    maxi = max(occupied_inds)
    missing = []
    for i in range(maxi):
        if i not in occupied_inds:
            missing.append(i)
    print("=> Missing: %s" % str(missing))


if sys.argv[1] == "complete":
    cursor = userdb.find()
    for _ in range(cursor.count()):
        obj = cursor.next()
        id = obj["id"]
        status = obj["status"]
        group_id = util.group_from_id(id)
        expr_ids = [group_id, group_id + util.EXPR_SEP]
        for ind in expr_ids:
            if status[ind] != 80:
                print("!> User %s id=%d does not complete experiment %d" %
                    (obj["name"], id, ind))


# [DANEROUS] delete a record of an id
if sys.argv[1] == "delrecord":
    cursor = collection.find({"id": int(sys.argv[2])})
    for _ in range(cursor.count()):
        obj = cursor.next()
        print(obj)
    result = collection.delete_one({"id": int(sys.argv[2])})
    

# [DANGEROUS] delete a user
if sys.argv[1] == "deluser":
    result = userdb.delete_one({'qid': sys.argv[2]})
    print(result)

# [DANGEROUS] mark a user's current job as complete
if sys.argv[1] == "compuser":
    qid = sys.argv[2]
    cursor = userdb.find({"qid": qid})
    print("=> find user")
    for _ in range(cursor.count()):
        obj = cursor.next()
        print(obj)
    
    status = obj["status"]
    id = obj["id"]
    used_id = obj["used_id"]
    used_id.append(id)
    expr_id = util.group_from_id(id)
    status[expr_id] = status[expr_id + util.EXPR_SEP] = 80
    userdb.update(
        {"qid": qid},
        {"$set" : 
            {
                "used_id" : used_id,
                "status" : status
            }
        })

# [DANGEROUS] cancel a user's current job
if sys.argv[1] == "cancel":
    qid = sys.argv[2]
    cursor = userdb.find({"qid": qid})
    print("=> find user")
    for _ in range(cursor.count()):
        obj = cursor.next()
        print(obj)
    
    status = obj["status"]
    id = obj["id"]
    expr_id = util.group_from_id(id)
    status[expr_id] = status[expr_id + util.EXPR_SEP] = 80
    id = obj["used_id"][-1] # the user need to have some completed jobs

    userdb.update(
        {"qid": qid},
        {"$set" : 
            {
                "id" : id,
                "status" : status
            }
        })

# [DANGEROUS] Reassign a job to a user
if sys.argv[1] == "fixuser":
    qid = sys.argv[2]
    cursor = userdb.find({"qid": qid})
    print("=> find user")
    for _ in range(cursor.count()):
        obj = cursor.next()
        print(obj)
    print("=> Try to fix its id")

    def func(x):
        flag = True
        for old_id in obj["used_id"]:
            if old_id % util.EXPR_SEP == x % util.EXPR_SEP:
                flag = False
                break
        return flag

    with lock:
        new_id = util.get_next_id(util.read_sync(), func)
    print("=> Fix from ID=%s to ID=%s" % (obj["id"], new_id))
    userdb.update({"qid": qid}, {"$set" : {"id" : new_id}})

# [DANGEROUS] Synchronize the sync file with user records
if sys.argv[1] == "sync":
    print("=> Synchronize sync with user")
    cursor = userdb.find()
    occupied_inds = []
    for _ in range(cursor.count()):
        obj = cursor.next()
        if int(obj["id"]) > 1000: #admin
            continue

        inds = []
        inds.extend(obj["used_id"])
        if obj["id"] not in inds:
            inds.append(obj["id"])
        for ind in inds:
            if ind in occupied_inds:
                print("!> ID=%d is occupied many times!" % ind)
            else:
                occupied_inds.append(ind)
    occupied_inds.sort()
    print(occupied_inds)

    s = list("0" * util.EXPR_NUM * util.EXPR_SIZE)
    for ind in occupied_inds:
        s[ind] = "1"
    with lock:
        util.write_sync("".join(s))
        print("Next to occupy: %s" % util.get_next_id(util.read_sync()))