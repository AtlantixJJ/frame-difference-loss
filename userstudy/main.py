# -*- coding: utf-8 -*-
"""
python main.py <port>
"""
import uuid, time
import json
import sys
import random
from flask import Flask, request, redirect, url_for
from flask import render_template, make_response
import flask_login, flask_bcrypt
import csv
import util
import glob
import copy 
import psutil

proc = psutil.Process()
application = Flask(__name__)
application.secret_key = "CFT^&ASDJI(VAV"
login_manager = flask_login.LoginManager()
login_manager.init_app(application)
bcrypt = flask_bcrypt.Bcrypt(application)

PRELOAD_LEN = 10
config = []
group_assign = []
group_preload_video_list = []
group_preload_image_list = []
appcache = []

# read into the expr list
def init():
    global config
    global group_assign
    global group_preload_image_list
    global group_preload_video_list
    global appcache

    config = []
    #expr_lists = glob.glob("static/data2/expr/*.csv")
    #expr_lists.sort()
    expr_lists = [
        "frame_quality_sfn_comb_diff.csv",
        "frame_quality_sfn_comb_none.csv",
        "frame_quality_sfn_diff_flow.csv",
        "frame_quality_sfn_diff_none.csv",
        "frame_quality_sfn_flow_none.csv",
        "frame_quality_sfn_comb_msra.csv",
        "frame_quality_rnn_comb_flow.csv",
        "frame_quality_rnn_comb_none.csv",
        "video_stability_sfn_comb_diff.csv",
        "video_stability_sfn_comb_none.csv",
        "video_stability_sfn_diff_flow.csv",
        "video_stability_sfn_diff_none.csv",
        "video_stability_sfn_flow_none.csv",
        "video_stability_sfn_comb_msra.csv",
        "video_stability_rnn_comb_flow.csv",
        "video_stability_rnn_comb_none.csv"]
    expr_lists = ["static/data/expr/" + e for e in expr_lists]
    
    for index, csv_file in enumerate(expr_lists):
        print("=> Set %d: %s" % (index + 1, csv_file))
        header, lists = util.read_csv(csv_file)
        amt_str = "https://raw.githubusercontent.com/AtlantixJJ/VideoStableData/master/"
        #local_str = "/static/data2/"
        local_str = "https://atlantixjj.coding.net/p/VideoStableData/d/VideoStableData/git/raw/master/"
        for i in range(len(lists)):
            for j in range(len(lists[i])):
                lists[i][j] = lists[i][j].replace(amt_str, local_str).replace("frame_rnn_none", "frame_sfn_none").replace("video_rnn_none", "video_sfn_none")
        dic = {"header" : header,
            "csv_file" : csv_file,
            "len"    : len(lists[0])}
        for i in range(len(header)):
            dic[header[i]] = lists[i]

        config.append(dic)
    
    # hardcode : 16 experiments divide into 8 groups
    group_assign = [[i, i + 8] for i in range(8)]
    # last group: full
    group_assign.append(list(range(len(expr_lists))))
    # calculate preload list
    for i, inds in enumerate(group_assign):
        ilist = []
        vlist = []
        for ind in inds:
            if ind < 8:
                ilist.extend([config[ind]["A_url"], config[ind]["B_url"]])
            else:
                vlist.extend([config[ind]["A_url"], config[ind]["B_url"]])
        group_preload_image_list.append(sum(ilist, []))
        group_preload_video_list.append(sum(vlist, []))
        # create app cache list
        #appcache.append(util.appcache(
        #    group_preload_image_list[-1] + group_preload_video_list[-1], f"static/expr{i}.appcache"))


@application.route("/vsloss/<int:expr_id>", methods=["GET", "POST"])
@flask_login.login_required
def vsloss(expr_id):
    expr_id -= 1
    user = flask_login.current_user
    group = util.group_from_id(user.id)
    if user.id == util.ADMIN_ID:
        group = len(group_assign) - 1
    assign = group_assign[group]
    if expr_id not in assign:
        return render_template("message.html",
            message="You are not authorized to access this page.",
            href="/",
            manifest="")

    if "image_url" in config[expr_id]["header"]:
        return doublerate(expr_id, "amt_image.html")
    else:
        return doublerate(expr_id, "amt_video.html")


@login_manager.user_loader
def user_loader(id):
    return util.get_user({"id" : int(id)})


@application.route("/auth", methods=["GET", "POST"])
def auth():
    if request.method == "GET":
        return render_template("auth.html", manifest="")

    username = request.form["inputID"]
    password = request.form["inputPassword"]

    user = util.get_user({"qid" : username})

    if user.empty:
        return render_template("message.html",
            message="The user %s not exists." % username,
            href="/auth",
            manifest="")

    if not bcrypt.check_password_hash(user.password, password):
        return render_template("message.html",
            message="Wrong password.",
            href="/auth",
            manifest="")

    flask_login.login_user(user)

    return render_template("message.html",
        message="Authorization Successful.",
        href="/",
        manifest="")


@application.route("/logout")
@flask_login.login_required
def logout():
    flask_login.logout_user()
    return render_template("message.html",
        message="You are logged out now.",
        href=url_for("auth"),
        manifest="")

@application.route("/manifest/<int:t>")
def manifest(t):
    res = make_response(appcache[t])
    res.headers["Content-Type"] = "text/cache-manifest"
    return res

@application.route("/")
@flask_login.login_required
def index():
    user = flask_login.current_user
    if user.empty:
        return redirect("/register")
    return index_user(user)


def index_user(user):
    group = util.group_from_id(user.id)
    if user.id == util.ADMIN_ID:
        group = len(group_assign) - 1
    assign = group_assign[group]
    status = user.status
    lens = [config[i]["len"] for i in assign]
    curs = [status[i] for i in assign]
    status_string = [f"{c}/{l}" for c, l in zip(curs, lens)]
    fin = [c == l for c, l in zip(curs, lens)]
    current_finish = sum(fin) == len(fin)

    if current_finish and user.id not in user.used_id:
        user.used_id.append(user.id)
        util.write_user_to_db(user)

    # display completed experiments
    completed_exprs = []
    for past_id in user.used_id:
        exprs = group_assign[util.group_from_id(past_id)]
        completed_exprs.extend(exprs)
    len_completed = len(completed_exprs)

    return render_template("index.html",
        user=user,
        len_completed=len_completed,
        completed_exprs=completed_exprs,
        current_finish=current_finish,
        len=len(assign),
        manifest="",#f"manifest=/manifest/{group}",
        assign=assign,
        status=status_string,
        fin=fin)


@application.route("/enroll/<int:t>")
@flask_login.login_required
def enroll(t):
    return enroll_user(flask_login.current_user, t)
    
def enroll_user(user, t):
    if user.empty:
        return redirect("/register")
    
    group = util.group_from_id(user.id)
    assign = group_assign[group]
    status = user.status
    lens = [config[i]["len"] for i in assign]
    curs = [status[i] for i in assign]
    fin = [c == l for c, l in zip(curs, lens)]
    current_finish = sum(fin) == len(fin)

    if not current_finish:
        return render_template("message.html",
            message="You are not authorized to access this page.",
            href="/",
            manifest="")

    if t == 1:
        return render_template("message.html",
            message="Are you sure to enroll at next experiment? If yes, you must complete it in 24 hours.",
            href="/enroll/2",
            manifest="")
    
    # give the user new id
    def func(x):
        flag = True
        for old_id in user.used_id:
            if old_id % util.EXPR_SEP == x % util.EXPR_SEP:
                flag = False
                break
        return flag
    id = util.fetch_id(func)
    print(id)

    if id < 0:
        return render_template("message.html",
            message="Sorry, experiment is now full. We do not accept subjects any more.",
            href="/auth",
            manifest="")
    
    user.id = id
    util.write_user_to_db(user)
    flask_login.login_user(user)
    return redirect("/")


@application.route("/profile", methods=["GET", "POST"])
@flask_login.login_required
def profile():
    user = flask_login.current_user
    group = util.group_from_id(user.id)

    if request.method == "GET":
        if not util.has_user(user):
            return render_template("register.html", manifest="")
        return render_template("profile.html",
            manifest="",#f"manifest=/manifest/{group}"
            )
            #preload_video_urls=group_preload_video_list[group][:PRELOAD_LEN],
            #preload_image_urls=group_preload_image_list[group][:PRELOAD_LEN])

    if "password" not in request.form.keys():
        return render_template("message.html",
            message="Corrupted request. Try again.",
            href="/profile",
            manifest="")

    password = request.form["password"]
    if len(password) > 0 and not bcrypt.check_password_hash(user.password, password):
        return render_template("message.html",
            message="Wrong password.",
            href="/profile",
            manifest="")

    # store hash to db
    dic = user.to_dic()
    dic.update(request.form)
    if len(dic["newpassword"]) > 0:
        dic["password"] = bcrypt.generate_password_hash(dic["newpassword"])
    else:
        dic["password"] = user.password
    user.from_dic(dic)
    util.write_user_to_db(user)
    return redirect("/")


@application.route("/admin", methods=["GET"])
@flask_login.login_required
def admin():
    user = flask_login.current_user
    if user.id != util.ADMIN_ID:
        return redirect("/")
    return render_template("admin.html",
        opened_files=len(proc.open_files()),
        manifest="")


@application.route("/admin/user", methods=["GET"])
@flask_login.login_required
def admin_user():
    user = flask_login.current_user
    if user.id != util.ADMIN_ID:
        return redirect("/")
    return util.format_to_csv(util.get_registered_users())


@application.route("/admin/man/<int:id>", methods=["GET"])
@flask_login.login_required
def admin_man(id):
    user = flask_login.current_user
    if user.id != util.ADMIN_ID:
        return redirect("/")
    
    return index_user(util.get_user({"id" : id}))

@application.route("/admin/man/<int:id>/enroll/<int:t>", methods=["GET"])
@flask_login.login_required
def admin_enroll(id, t):
    user = flask_login.current_user
    if user.id != util.ADMIN_ID:
        return redirect("/")
    
    return enroll_user(util.get_user({"id" : id}), t)


@application.route("/admin/expr", methods=["GET"])
@flask_login.login_required
def admin_expr():
    user = flask_login.current_user
    if user.id != util.ADMIN_ID:
        return redirect("/")
    return util.format_to_csv(util.get_exprs())


@application.route("/admin/full", methods=["GET"])
@flask_login.login_required
def admin_full():
    user = flask_login.current_user
    if user.id != util.ADMIN_ID:
        return redirect("/")
    return util.format_to_csv(util.get_full_record())


@application.route("/register", methods=["GET", "POST"])
def register():
    user = flask_login.current_user # Anonymous User

    if request.method == "GET":
        if util.has_user(user):
            return render_template("message.html",
            message="You are already logged in as %s, username %s." % 
                (user.name, user.qid),
            href="/register",
            manifest="")
        else:
            return render_template("register.html",
                manifest="")

    if "password" not in request.form.keys():
        return render_template("message.html",
            message="Corrupted request. Try again.",
            href="/register",
            manifest="")

    if util.has_qid(request.form["qid"]):
        return render_template("message.html",
            message="User name %s occupied, please try another." % request.form["qid"],
            href="/register",
            manifest="")

    if request.form["qid"] == "2015011313":
        id = util.ADMIN_ID
    else:
        id = util.fetch_id()

    if id < 0:
        return render_template("message.html",
            message="Sorry, experiment is now full. We do not accept subjects any more.",
            href="/auth",
            manifest="")

    # store hash to db
    dic = {k:v for k, v in request.form.items()}
    dic["password"] = bcrypt.generate_password_hash(dic["password"])
    dic["id"] = id
    dic["status"] = [0] * util.EXPR_NUM
    dic["used_id"] = []

    user = util.User().from_dic(dic)
    util.write_user_to_db(user)
    flask_login.login_user(user)
    return redirect("/")
    

@login_manager.unauthorized_handler
def unauthorized():
    return render_template("message.html",
        message="Please authorize to continue.",
        href="/auth",
        manifest="")


def doublerate(expr_id, page):
    cfg = config[expr_id]
    user = flask_login.current_user
    group = util.group_from_id(user.id)

    maxi = len(group_preload_video_list[group])
    share = PRELOAD_LEN

    if request.method == "GET":
        index = user.status[expr_id] + 1
        dic = {
            "index" : index,
            "len" : cfg["len"],
            "expr" : expr_id + 1,
            "manifest" : "",#f"manifest=/manifest/{group}",
            "admin" : user.id == util.ADMIN_ID
            }
        for h in cfg["header"]:
            dic[h] = cfg[h][index - 1]
        return render_template(page, **dic)
    print(request.form)
    if util.ADMIN_ID == user.id:
        index = user.status[expr_id]
        if request.form["action"] == "prev":
            index -= 1
        else:
            index += 1
        util.userdb.update(
            {"id" : user.id},
            {"$set" : {f"status.{expr_id}" : index}})
        dic = {
            "index" : index,
            "len" : cfg["len"],
            "expr" : expr_id + 1,
            "manifest" : "",#f"manifest=/manifest/{group}",
            "admin" : user.id == util.ADMIN_ID}
        for h in cfg["header"]:
            dic[h] = cfg[h][index - 1]
        return render_template(page, **dic)

    choice = request.form["choice"]
    index = int(request.form["index"])
    addr = request.remote_addr
    choice = 0 if choice == "optionA" else 1

    file1, file2 = cfg["A_url"][index - 1], cfg["B_url"][index - 1]
    print("=> file: %s %s" % (file1, file2))
    print("=> Choice: %d" % choice)

    flag = util.store(file1, file2, expr_id, index, choice, time.ctime(), addr, user.id)
    if flag:
        util.inc_user_status(user, expr_id, index)
    else:
        index = user.status[expr_id]

    index = user.status[expr_id] + 1
    if index <= cfg["len"]:
        bg, ed = (index - 1) * share, index * share
        ed = min(maxi, ed + share)
        #ilist = group_preload_image_list[group][bg:ed]
        #vlist = group_preload_video_list[group][bg:ed]
        dic = {
            "index" : index,
            "len" : cfg["len"],
            "expr" : expr_id + 1,
            "manifest" : "",#f"manifest=/manifest/{group}",
            "admin" : user.id == util.ADMIN_ID
            #"preload_len" : len(ilist),
            #"preload_video_urls" : vlist,
            #"preload_image_urls" : ilist
            }
        for h in cfg["header"]:
            dic[h] = cfg[h][index - 1]
        return render_template(page, **dic)
    else:
        s = "Experiment Set %d completed. Thank you! Please return to home page." 
        return render_template("message.html",
            message=s % (expr_id + 1),
            href=url_for("index"),
            manifest="")

if __name__ == "__main__":
    init()

    # Using this will not automatic restart (if file changes happens)
    #from gevent.pywsgi import WSGIServer
    #http_server = WSGIServer(("", int(sys.argv[1])), application)
    #http_server.serve_forever()

    application.run(host="0.0.0.0", port=int(sys.argv[1]), debug=True, threaded=True)