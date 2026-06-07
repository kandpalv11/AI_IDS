from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("ids_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    result = ""

    if request.method == "POST":

        protocol = request.form["protocol"]
        flow_duration = float(request.form["flow_duration"])
        total_forward_packets = float(request.form["total_forward_packets"])
        total_backward_packets = float(request.form["total_backward_packets"])

        # Protocol Encoding
        protocol_map = {
            "TCP": 0,
            "UDP": 1,
            "ICMP": 2
        }

        protocol = protocol_map.get(protocol.upper(), 0)

        sample = pd.DataFrame({
            "protocol": [protocol],
            "flow_duration": [flow_duration],
            "total_forward_packets": [total_forward_packets],
            "total_backward_packets": [total_backward_packets]
        })

        prediction = model.predict(sample)

        try:
            result = label_encoder.inverse_transform(prediction)[0]
        except:
            result = str(prediction[0])

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)