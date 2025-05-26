from flask import Flask, request, jsonify
from reranker import compute_scores

app = Flask(__name__)

@app.route('/score', methods=['POST'])
def score():
    data = request.get_json()
    sentence_pairs = data.get('sentence_pairs')
    # Ensure sentence_pairs is a list of lists
    if not isinstance(sentence_pairs, list) or not all(isinstance(pair, list) and len(pair) == 2 for pair in sentence_pairs):
        return jsonify({'error': 'Invalid input format. Expected a list of sentence pairs.'}), 400

    normalize = data.get('normalize', False)
    result = compute_scores(sentence_pairs, normalize)

    return jsonify({'scores': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=12456, debug=True)