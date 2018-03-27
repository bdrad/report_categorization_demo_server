from websocket_server import WebsocketServer
import fastText
import os
import datetime
from end2end_process import EndToEndProcessor


# Get last updated time
lastm = int(os.path.getmtime("."))
time_str = datetime.datetime.fromtimestamp(lastm).strftime('%Y-%m-%d %H:%M')
print(time_str)

# Load Model
radlex_path = "model/radlex_replacements"
clever_path = "model/clever_replacements"
e2e = EndToEndProcessor(clever_path, radlex=radlex_path)

model_path = "model/model.bin"
ftModel = fastText.load_model(model_path)

# Returns (process_report_text, ground_truth, predicted_label)
def output_prob(text, end_to_end=e2e, model=ftModel):
    report_text = "IMPRESSION: " + text + "\nEND OF IMPRESSION"
    processed_report_text, ground_truth = e2e.transform([report_text])[0]
    print(processed_report_text)
    processed_report_text = " ".join(processed_report_text)
    prediction = model.predict(processed_report_text)
    conf = 1 - prediction[1][0] if prediction[0][0] == '__label__0' else prediction[1][0]
    return (processed_report_text, ground_truth, conf)

# Server methods
def new_client(client, server):
    print("New client connected and was given id %d" % client['id'])
    server.send_message(client, time_str)

def client_left(client, server):
    print("Client(%d) disconnected" % client['id'])

def message_received(client, server, message):
    print("Client(%d) sent report: %s" % (client['id'], message))
    processed_report_text, ground_truth, predicted_label = output_prob(message)
    server.send_message(client, "%0.3f" % predicted_label)
    # TODO: add logging

# Run server
PORT=443
server = WebsocketServer(PORT, host="0.0.0.0")
server.set_fn_new_client(new_client)
server.set_fn_client_left(client_left)
server.set_fn_message_received(message_received)
server.run_forever()
