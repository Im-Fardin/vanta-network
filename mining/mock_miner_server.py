from flask import Flask, request, jsonify
import waitress

app = Flask(__name__)

@app.route("/api/submit-order", methods=["POST"])
def submit_order():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.json
    print("\n[MOCK MINER] Received order payload:")
    print(data)
    
    # بررسی فیلدهای پایه
    if "trade_pair" not in data or "order_type" not in data:
        return jsonify({"error": "Missing trade_pair or order_type"}), 400
        
    return jsonify({
        "status": "success",
        "message": "Order successfully simulated in local sandbox.",
        "order": data
    }), 200

if __name__ == "__main__":
    print("=" * 60)
    print("Mock Miner Server listening on http://127.0.0.1:8088")
    print("Ready to accept orders from run_ml_strategy.py")
    print("=" * 60)
    waitress.serve(app, host="127.0.0.1", port=8088)
