from websocket_server import WebsocketServer
from model import ClassificationModel
import os
import gc
import datetime
import json
import logging
from end2end_process import EndToEndProcessor

# Set Logging
logger = logging.getLogger('server')
hdlr = logging.FileHandler('./server.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)


# Get last updated time
lastm = int(os.path.getmtime("."))
time_str = datetime.datetime.fromtimestamp(lastm).strftime('%Y-%m-%d %H:%M')
logger.info("Starting server, last modified at " + time_str)

# Load Model
replacements = ["model/clever_replacements", "model/misc_replacements", "model/radlex_replacements"]
e2e = EndToEndProcessor(replacements)


state = {"ftModel" : None, "num_clients" : 0, "model_loaded" : False}
model_path = "model/MODEL"

# Returns (process_report_text, ground_truth, predicted_label)
def output_prob(text, end_to_end=e2e, state=state):
    if not state["model_loaded"]:
        logger.info("Loading model")
        state["ftModel"] = ClassificationModel(path=model_path)
        state["model_loaded"] = True
    report_text = "IMPRESSION: " + text + "\nEND OF IMPRESSION"
    processed_report_text, ground_truth = e2e.transform([report_text])[0]
    logger.info(processed_report_text)
    processed_report_text = " ".join(processed_report_text)
    prediction = state["ftModel"].predict(processed_report_text)
    logger.info(prediction)
    return (processed_report_text, ground_truth, prediction)

# Server methods
def new_client(client, server, state=state):
    state["num_clients"] += 1
    if not state["model_loaded"]:
        logger.info("Loading model")
        state["ftModel"] = ClassificationModel(path=model_path)
        state["model_loaded"] = True

    logger.info("New client connected and was given id %d" % client['id'])
    server.send_message(client, time_str)

def client_left(client, server, state=state):
    state["num_clients"] -= 1
    if state["num_clients"] == 0:
        logger.info("Taking model down")
        state["ftModel"] = None
        state["model_loaded"] = False
        gc.collect()
        logger.info("Garbage: " + str(gc.garbage))

def message_received(client, server, message):
    logger.info("Client(%d) sent: %s" % (client['id'], message))
    obj = json.loads(message)
    if obj["type"] == "impression":
        impression = obj["payload"]
        processed_report_text, ground_truth, predicted_label = output_prob(impression)
        predicted_label = min(predicted_label, 1)
        predicted_label = max(predicted_label, 0)
        server.send_message(client, "%0.3f" % predicted_label)
    else:
        raise NotImplementedError

# Run server
PORT=443
server = WebsocketServer(PORT, host="0.0.0.0")
server.set_fn_new_client(new_client)
server.set_fn_client_left(client_left)
server.set_fn_message_received(message_received)
server.run_forever()
