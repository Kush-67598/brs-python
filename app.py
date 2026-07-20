from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

popular_df = pickle.load(open('models/popular.pkl', 'rb'))
books = pickle.load(open('models/books.pkl', 'rb'))
pt = pickle.load(open('models/pt.pkl', 'rb'))
similarity_scores = pickle.load(open('models/similarity_scores.pkl', 'rb'))

app = Flask(__name__)


@app.route("/")
def hello_world():
    return render_template(
        'index.html',
        book_name=list(popular_df['Book-Title'].values),
        author=list(popular_df['Book-Author'].values),
        image=list(popular_df['Image-URL-M'].values),
        votes=list(popular_df['num_ratings'].values),
        rating=list(popular_df['avg_rating'].values),
    )


@app.route("/recommend", methods=['POST'])
def recommend_route():
    payload = request.get_json(silent=True) or {}
    user_input = payload.get('book_name', '').strip()

    if not user_input:
        return jsonify({"error": "No book title provided."}), 400

    # case-insensitive match — pt.index is exact-cased, users won't type exact case
    matches = [title for title in pt.index if title.lower() == user_input.lower()]

    if not matches:
        return jsonify({
            "error": f"'{user_input}' isn't in our catalog. Check spelling or try another title."
        }), 404

    exact_title = matches[0]

    try:
        data = recommend(exact_title)
    except IndexError:
        return jsonify({"error": "Could not generate recommendations for that title."}), 500

    return jsonify({"results": data})


def recommend(book_name):
    index = np.where(pt.index == book_name)[0][0]

    similar_items = sorted(
        list(enumerate(similarity_scores[index])),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    data = []
    for i in similar_items:
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
        temp_df = temp_df.drop_duplicates('Book-Title')

        data.append({
            "title": temp_df['Book-Title'].values[0],
            "image": temp_df['Image-URL-M'].values[0],
            "author": temp_df['Book-Author'].values[0]
        })

    return data


if __name__ == '__main__':
    app.run(debug=True)
