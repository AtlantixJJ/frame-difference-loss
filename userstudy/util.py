import random
from pymongo import MongoClient
import numpy as np
import csv
from filelock import Timeout, FileLock
import os
import flask_login

ADMIN_ID = 1948202859
random.seed(7752)

client = MongoClient("0.0.0.0", 27027)
db = client.userstudy
collection = db.score
userdb = db.user
per = np.random.permutation(np.arange(45))

EXPR_SEP = 3
EXPR_NUM = 6
EXPR_SIZE = 20

class User(flask_login.UserMixin):
    def __init__(self):
        self.empty = True
        self.attributes = ["password","id","name","age","gender","vision","qid","degree","status","used_id"]

    # to user db
    def to_dic(self):
        if self.empty:
            print("=> Empty user profile")
            return {}
        return {k : getattr(self, k) for k in self.attributes}
    
    # from profile form
    def from_dic(self, dic):
        self.empty = False
        for k, v in dic.items():
            setattr(self, k, v)
        return self


def read(dic, key):
    if key in dic.keys():
        return dic[key]


# if user not exists, return empty User
def get_user(query):
    user = User()
    res = userdb.find(query)
    if res.count() == 1:
        n = res.next()
        if "qid" in n.keys():
            # non-empty user profile
            user.from_dic(n)
    elif res.count() > 1:
        print("!> Error. Multiple users corresponding to query %s" % str(query))
    else:
        print("=> User of query %s not found" % str(query))
    return user


def write_user_to_db(user):
    query = {"qid" : user.qid}
    old_user = get_user(query)
    if not old_user.empty:
        print("=> Updating old profile")
        print(old_user.to_dic())
        print("=> To new profile")
        print(user.to_dic())
        userdb.update(
            {"qid" : user.qid},
            {"$set" : user.to_dic()})
    else:
        print("=> Add new user profile")
        print(user.to_dic())
        userdb.insert(user.to_dic())


def read_sync():
    if not os.path.exists("sync"):
        write_sync("0" * EXPR_NUM * EXPR_SIZE)
    with open("sync", "r") as f:
        s = f.read().strip()
    return s


def write_sync(s):
    with open("sync", "w") as f:
        f.write(s)

def get_next_id(s, func=None):
    i = s.find("0")

    # no spare at all!
    if i < 0:
        return -1

    # has filter
    if func is not None:
        flag = False
        for _ in range(EXPR_SEP):
            # exceed
            if i >= len(s):
                i = -1
                flag = False
                break
            # can allocate and pass filter
            if s[i] == "0" and func(i):
                flag = True
                break
            # go to next
            i += 1
        if flag:
            return i
        else:
            return -1
    
    # no filter
    return i

def fetch_id(func=None):
    lock = FileLock("sync.lock")
    with lock:
        s = read_sync()
        i = get_next_id(s, func)
        if i < 0:
            return -1
        
        # no filter
        l = list(s)
        l[i] = "1"
        write_sync("".join(l))
    return i


def group_from_id(id_):
    return id_ % EXPR_SEP


def get_password(qid):
    user = userdb.find({"qid" : qid})
    if user.count() == 0:
        print("?> User %s not found" % qid)
        return -1
    user = user.next()

    if "password" not in user.keys():
        print("?> User %s have empty profile" % qid)
        return -2
    
    return user["password"]


def has_qid(qid):
    user = userdb.find({"qid" : qid})
    if user.count() > 1:
        print("!> Error. Multiple users corresponding to qid=%s" % qid)
    return user.count() == 1


def has_user(user):
    if not hasattr(user, "empty"):
        return False
    if user.empty:
        return False
    return has_qid(user.qid)
        

def read_csv(csv_file):
    csv_file = open(csv_file, "r")
    csv_reader = csv.reader(csv_file, dialect="excel")
    header = next(csv_reader)#.next()
    lists = [[] for _ in header]
    while True:
        try:
            l = next(csv_reader)#.next()
        except StopIteration:
            break
        for i in range(len(l)):
            lists[i].append(l[i])
    csv_file.close()
    return header, lists


def inc_user_status(user, expr_id, index):
    id = user.id
    if user.status[expr_id] < index:
        user.status[expr_id] = index
    else:
        print("!> Error. %d >= %d" % (user.status[expr_id], index))

    userdb.update(
        {"id" : id},
        {"$set" : {f"status.{expr_id}" : index}})


def store(optionA, optionB, expr_id, index, choice, time, ip, id):
    c = collection.find({"id" : id, "expr_id" : expr_id})
    prev_index = 0
    if c.count() == 0:
        # not in database
        collection.insert({
            "id"        : id        ,
            "expr_id"   : expr_id   , 
            "optionA"   : []        ,
            "optionB"   : []        ,
            "index"     : []        ,
            "choice"    : []        ,
            "time"      : []        ,
            "ip"        : []
        })
    else:
        n = c.next()
        if len(n["index"]) > 0:
            prev_index = n["index"][-1]

    if prev_index + 1 != index:
        print("!> Ordering Error. Previous index is %d, current index is %d" % (prev_index, index))
        return False

    collection.update(
        {"id" : id, "expr_id" : expr_id},
        {"$push" : {
                "optionA": optionA,
                "optionB": optionB,
                "index": index,
                "choice": choice,
                "time" : time,
                "ip": ip
                }
        })
    return True


##
## AppCache
##

def appcache(list, name):
    s = ""
    with open(name, "w") as f:
        s += "CACHE MANIFEST\n"
        s += "\n".join(list)
        s += "\nNETWORK\n*"
    return s

##
## Admin functions
##

def get_registered_users():
    cursor = userdb.find()
    attrs = ["name","qid","id","status","used_id","age","gender","vision","degree"]
    res = [attrs]
    for _ in range(cursor.count()):
        obj = cursor.next()
        s = []
        for k in attrs:
            data = obj[k]
            if "status" == k:
                data = " | ".join([f"{i}: {d}/80"
                    for i, d in enumerate(data) if d > 0])
            if "used_id" == k:
                data = " | ".join([str(id) for id in data])
            s.append(str(data))
        res.append(s)
    return res

def format_to_csv(strs):
    return "\n".join([",".join(s) for s in strs])

from parsedb import get_exprs, get_full_record

if __name__ == "__main__":
    fetch_id()


"""
class AccessManager(object):
    def __init__(self):
        with open("access_code") as f:
            lines = f.readlines()
        self.access_codes = []
        self.code_groups = []
        for i, l in enumerate(lines):
            items = l.strip().split(",")
            self.access_codes.extend(items)
            self.code_groups.extend([i] * len(items))
        self.access_codes.extend(["abc", "efg"])
        self.code_groups .extend([8, 8])

    def get_info_by_code(self, code):
        if code not in self.access_codes:
            return -1, -1, -1

        ind = self.access_codes.index(code)
        group = self.code_groups[ind]
        return ind, code, group

    def get_info_by_id(self, id):
        code = self.access_codes[id]
        group = self.code_groups[id]
        return id, code, group
    
    def access(self, code):
        if code not in self.access_codes:
            return False
        return True

AM = AccessManager()
"""