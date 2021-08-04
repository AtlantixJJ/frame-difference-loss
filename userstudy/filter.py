"""
Filter the select time.
"""
import os
import csv

csvfile = open("document/videostability_sfn_flow_none_4_Batch_3869707_batch_results.csv", 'r')
csv_reader = csv.reader(csvfile, dialect='excel')
newcsv = open("document/videostability_sfn_flow_none_5_Batch_3869707_batch_results.csv", "w")
csv_writer = csv.writer(newcsv, dialect='excel')

headers = csv_reader.next()
ind = headers.index("LifetimeApprovalRate")
csv_writer.writerow(headers)

while True:
    try:
        items = csv_reader.next()
    except StopIteration:
        break

    #print(items)
    time = int(items[ind].split("%")[0])
    if time >= 100:
        csv_writer.writerow(items)

    {% if len_completed > 0 %}

    <h4 class="form-signin-heading">You have completed experiment <strong>{{completed_exprs[0] + 1}}</strong>{% for i in range(1, len_completed) %},<strong>{{completed_exprs[i] + 1}}</strong>{% endfor %}.</h4>

    {% endif %}