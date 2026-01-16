import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np

from sklearn.neighbors import NearestNeighbors
from tqdm.notebook import tqdm


def plot_movie_data(df, movie_x1, movie_x2, movie_y, movie_names):
    # clean nan's from data
    subset = df[[movie_x1, movie_x2]].dropna()
    has_label = df[movie_y].notna()
    subset[movie_y] = df[movie_y]
    
    # start plot
    plt.figure(figsize=(8, 8))

    # set shape and limits
    ax = plt.gca()
    ax.set_aspect('equal')
    limit = max(df.max().max(), -1 * df.min().min()) + 0.2
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    plt.axhline(y=0, color='black', linewidth=1)
    plt.axvline(x=0, color='black', linewidth=1)
    
    # separate green and red points
    with_label = subset[subset[movie_y].notna()]
    without_label = subset[subset[movie_y].isna()]
    
    # plot green points
    plt.scatter(with_label[movie_x1], with_label[movie_x2], color='green', label=f'did see {movie_names[movie_y]}')
    for user_id, row in with_label.iterrows():
        plt.text(row[movie_x1] + 0.03, row[movie_x2] + 0.03, f"{user_id}: {row[movie_y]:.1f}", fontsize=8, color='green')
    
    # plot red pointsl)
    plt.scatter(without_label[movie_x1], without_label[movie_x2], color='red', label=f'did not see {movie_names[movie_y]}')
    for user_id, row in without_label.iterrows():
        plt.text(row[movie_x1] + 0.03, row[movie_x2] + 0.03, f"{user_id}: ?", fontsize=8, color='red')

    # plot labels, etc.
    plt.xlabel(movie_names[movie_x1])
    plt.ylabel(movie_names[movie_x2])
    plt.title(f"{movie_names[movie_x2]} vs {movie_names[movie_x1]}\n(labels = {movie_names[movie_y]})")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_movie_data_cosine(df, movie_x1, movie_x2, movie_y, movie_names, highlight_ids):
    """
    Cosine-similarity style plot.

    highlight_ids: iterable of two index labels from df (e.g. ['U025', 'U027'])
    """

    # clean NaNs for the 2D projection
    subset = df[[movie_x1, movie_x2]].dropna()
    subset[movie_y] = df[movie_y]

    # start figure
    plt.figure(figsize=(5, 5))
    ax = plt.gca()
    ax.set_aspect('equal')

    # symmetric limits around 0
    limit = max(df.max().max(), -1 * df.min().min()) + 0.2
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)

    # axes
    plt.axhline(y=0, color='black', linewidth=1)
    plt.axvline(x=0, color='black', linewidth=1)

    # split in labeled / unlabeled for coloring
    with_label = subset[subset[movie_y].notna()]
    without_label = subset[subset[movie_y].isna()]

    # 1) plot ALL points faintly in background
    plt.scatter(with_label[movie_x1], with_label[movie_x2],
                color='green', alpha=0.1)
    plt.scatter(without_label[movie_x1], without_label[movie_x2],
                color='red', alpha=0.1)

    # 2) highlight the two selected points + vectors
    highlighted_vectors = []
    for uid in highlight_ids:
        if uid not in subset.index:
            continue  # silently skip if missing from subset

        row = subset.loc[uid]
        x, y = row[movie_x1], row[movie_x2]
        color = 'green' if pd.notna(row[movie_y]) else 'red'

        # bright point on top
        plt.scatter([x], [y], color=color, s=80, edgecolors='black', zorder=3, label=f'did {"not " if color == "red" else ""}see {movie_names[movie_y]}')

        # label near point
        label_text = f"{uid}"
        if pd.notna(row[movie_y]):
            label_text += f": {row[movie_y]:.1f}"
        else:
            label_text += ": ?"
        plt.text(x + 0.03, y + 0.03, label_text, fontsize=10, color=color)

        # vector from origin (no arrows)
        plt.plot([0, x], [0, y], color=color, linewidth=2.5)

        highlighted_vectors.append((x, y))

    # 3) angle arc between the two vectors, labeled α
    if len(highlighted_vectors) == 2:
        (x1, y1), (x2, y2) = highlighted_vectors

        # angles in degrees
        angle1 = np.degrees(np.arctan2(y1, x1))
        angle2 = np.degrees(np.arctan2(y2, x2))

        # order angles so arc is the smaller one
        theta1, theta2 = sorted([angle1, angle2])
        # if the gap is > 180, flip so we draw the smaller angle
        if theta2 - theta1 > 180:
            theta1, theta2 = theta2, theta1 + 360

        # choose a radius for the arc
        radius = 0.3 * limit

        # arc centered at origin
        arc = patches.Arc((0, 0), 2*radius, 2*radius,
                          angle=0, theta1=theta1, theta2=theta2,
                          linewidth=2)
        ax.add_patch(arc)

        # place the α label at middle of the arc
        mid_angle = np.radians((theta1 + theta2) / 2.0)
        text_r = radius * 1.2
        tx = text_r * np.cos(mid_angle)
        ty = text_r * np.sin(mid_angle)
        plt.text(tx, ty, r'$\alpha$', fontsize=14, ha='center', va='center')

    # axes labels, title, grid, legend
    plt.xlabel(movie_names[movie_x1])
    plt.ylabel(movie_names[movie_x2])
    plt.title(f"{movie_names[movie_x2]} vs {movie_names[movie_x1]}\n(labels = {movie_names[movie_y]})")
    plt.grid(True)
    plt.legend()
    
    plt.savefig("cosine.png", dpi=300)
    plt.show()


def build_utility_matrices(train_df):
    """
    From the training ratings, build a utility rating matrix.
    """
    # User–item matrix with NaN for missing ratings
    utility = train_df.pivot(index="userId", columns="movieId", values="rating")

    # Per-user mean rating (skip NaNs)
    user_means = utility.mean(axis=1)

    # Mean-centered ratings (subtract user mean from each observed rating)
    utility_centered = utility.sub(user_means, axis=0)

    return utility, utility_centered, user_means
    
def fit_user_knn(utility_centered, k = 20):
    """
    Fit a NearestNeighbors model on the user-centered rating matrix.
    """
    # Fill NaNs with 0 for similarity computations
    X_users = utility_centered.fillna(0.0).values

    knn_model = NearestNeighbors(n_neighbors = k + 1, metric="cosine", algorithm="brute", n_jobs=-1)
    knn_model.fit(X_users)

    return knn_model, X_users

def predict_user_based_knn(train_matrix, utility_centered, user_means, test_df, k = 20, min_neighbors = 1):
    """
    Predict ratings for all (userId, movieId) pairs in test_df using
    user-based kNN with similarity weighting and mean-centering.
    """

    # get knn_model (using sklearn NearestNeighors)
    knn_model, X_users = fit_user_knn(utility_centered, k = k)

    # get all users and items from training data
    users = utility_centered.index
    items = utility_centered.columns

    # get indices of users and items (for .iloc)
    user_to_row = {u: idx for idx, u in enumerate(users)}
    item_to_col = {i: idx for idx, i in enumerate(items)}

    # no good way to do everything in pure Pandas/Numpy, so collect predictions in list
    preds = []

    # Group test by user to avoid calling knn for each row
    for user_id, group in tqdm(test_df.groupby("userId")):
        
        # If user not in training set, use global mean
        if user_id not in user_to_row:
            global_mean = user_means.mean()
            preds.extend([global_mean] * len(group))

            # skip ehead to next user
            continue

        # get iloc of user
        u_idx = user_to_row[user_id]

        # Find neighbors of this user
        distances, indices = knn_model.kneighbors(
            X_users[u_idx].reshape(1, -1),
            n_neighbors=k + 1
        )
        distances = distances[0]
        neighbor_idxs = indices[0]

        # Convert cosine distances to similarities
        sims = 1.0 - distances

        # Drop self (usually the first neighbor)
        mask_not_self = neighbor_idxs != u_idx
        neighbor_idxs = neighbor_idxs[mask_not_self]
        sims = sims[mask_not_self]

        # For each movie in this user's test rows, predict rating
        for _, row in group.iterrows():
            movie_id = row["movieId"]

            # If movie is not present in training, use user mean
            if movie_id not in item_to_col:
                preds.append(user_means.loc[user_id])
                continue

            col_idx = item_to_col[movie_id]

            # Centered ratings of neighbors for this movie
            neighbor_ratings_centered = utility_centered.iloc[neighbor_idxs, col_idx].values

            # Keep only neighbors who rated this movie (non-NaN)
            mask_rated = ~np.isnan(neighbor_ratings_centered)
            neighbor_ratings_centered = neighbor_ratings_centered[mask_rated]
            neighbor_sims = sims[mask_rated]

            # If not enough users rated the movie for prediction, use user mean instead 
            if len(neighbor_ratings_centered) < min_neighbors:
                pred = user_means.loc[user_id]
            else:
                # Similarity-weighted combination
                denom = np.sum(np.abs(neighbor_sims))
                if denom == 0:
                    # if neighborhood similarities happen to add up to zero, use user mean 
                    pred = user_means.loc[user_id]
                else:
                    weighted_sum = np.dot(neighbor_sims, neighbor_ratings_centered)
                    # add user mean back to get (real non centered) predcitions
                    pred = user_means.loc[user_id] + weighted_sum / denom

            preds.append(pred)

    predictions = test_df.drop(columns=["rating"]).copy()
    predictions["pred_rating"] = preds
    
    return predictions


def get_user_mean(user_id, users, user_means):
    if user_id not in users:
        return user_means.mean()

    return user_means.loc[user_id]

                  
def predict_user_mean(train_matrix, utility_centered, user_means, test_df):
    """
    Predict ratings for all (userId, movieId) pairs in test_df using
    user-based kNN with similarity weighting and mean-centering.
    """
    # get all users and items from training data
    users = utility_centered.index

    # get indices of users and items (for .iloc)
    user_to_row = {u: idx for idx, u in enumerate(users)}

    # no good way to do everything in pure Pandas/Numpy, so collect predictions in list
    preds = []

    # Group test by user to avoid calling knn for each row
    for user_id, group in tqdm(test_df.groupby("userId")):
        # If user not in training set, use global mean
  
        preds.extend([get_user_mean(user_id, user_to_row, user_means)] * len(group))

    predictions = test_df.drop(columns=["rating"]).copy()
    predictions["pred_rating"] = preds
    
    return predictions