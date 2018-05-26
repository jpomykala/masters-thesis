import json
import time
import urllib.parse
import urllib.request

from src.utils import save_text

user = "jakub.pomykala@gmail.com"
# lpmn = "any2txt|tagger({\"lang\":\"polish\"})"
lpmn = "any2txt|wcrft2"
url = "http://ws.clarin-pl.eu/nlprest2/base"


def upload(file_path):
    with open(file_path, "rb") as file:
        file_bytes = file.read()
    req = urllib.request.Request(url + '/upload/', file_bytes, {'Content-Type': 'binary/octet-stream'})
    return urllib.request.urlopen(req).read().decode("utf-8")


def process(data):
    json_data = json.dumps(data).encode('utf-8')
    start_task_req = urllib.request.Request(url=url + '/startTask/')
    start_task_req.add_header('Content-Type', 'application/json; charset=utf-8')
    start_task_req.add_header('Content-Length', len(json_data))
    task_id = urllib.request.urlopen(start_task_req, json_data).read().decode("utf-8")

    time.sleep(0.1)

    get_status_req = urllib.request.Request(url + '/getStatus/' + task_id)
    status_response = urllib.request.urlopen(get_status_req)
    data = json.load(status_response)
    while data["status"] == "QUEUE" or data["status"] == "PROCESSING":
        time.sleep(0.1)

        get_status_req = urllib.request.Request(url + '/getStatus/' + task_id)
        status_response = urllib.request.urlopen(get_status_req)
        data = json.load(status_response)
    if data["status"] == "ERROR":
        print("Error " + data["value"])
        return None
    return data["value"]

def process_text(source_file, output_file):
    print("[T] processing: " + source_file)
    file = open(source_file, "r", encoding="utf-8")
    text = file.read()
    data = {'lpmn': lpmn, 'user': user, 'text': text}
    data = process(data)
    data = data[0]["fileID"]
    download_req = urllib.request.Request(url + '/download' + data)
    content = urllib.request.urlopen(download_req).read().decode("utf-8")

    output_file = output_file + ".ccl.xml"
    save_text(output_file, content)


def process_file(source_file, output_file):
    file_id = upload(source_file)
    print("[F] Processing: " + source_file)
    data = {'lpmn': lpmn, 'user': user, 'file': file_id}
    data = process(data)
    data = data[0]["fileID"]
    download_req = urllib.request.Request(url + '/download' + data)
    content = urllib.request.urlopen(download_req).read().decode("utf-8")

    output_file = output_file + ".ccl.xml"
    save_text(output_file, content)
