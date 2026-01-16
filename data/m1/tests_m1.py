import pandas as pd
import numpy as np
from html import escape

from IPython.display import display, HTML

def knn_01(pivot_ratings):
    print("Testing...")
    mock = pd.DataFrame({"userID": [1, 1, 2], "movieID": [10, 20, 10], "rating": [4.0, 5.0, 3.0]})
    expected = mock.pivot(index="userID", columns="movieID", values="rating")
    actual = pivot_ratings(mock)
    
    assert_frame_equal_pretty(expected, actual, mock_input=mock)
    print("Success!")


def knn_02(mean_center):
    print("Testing...")
    mock = pd.DataFrame({"M1": [4.0, 2.0, 3.0], "M2": [6.0, np.nan, 5.0]}, index=["U1", "U2", "U3"])
    expected = pd.DataFrame({"M1": [-1.0,  0.0, -1.0], "M2": [ 1.0,  np.nan,  1.0]}, index=["U1", "U2", "U3"])
    centered = mean_center(mock)

    assert_frame_equal_pretty(expected, centered, mock_input=mock)
    print("Success!")


def knn_03(transform_data_for_knn):
    print("Testing...")
    df = pd.DataFrame({"M1": [1.0, np.nan, 3.0], "M2": [2.0, 4.0, np.nan], "M3": [5.0, np.nan, 6.0]}, index=["U1", "U2", "U3"])
    X_cols, y_col = ["M1", "M2"], "M3"

    X_known, y_known, X_unknown, y_unknown = transform_data_for_knn(df, X_cols, y_col)

    expected_X_known = pd.DataFrame({"M1": [1.0, 3.0], "M2": [2.0, 3.0]}, index=["U1", "U3"])
    expected_X_unknown = pd.DataFrame({"M1": [2.0], "M2": [4.0]}, index=["U2"])
    expected_y_known = pd.Series([5.0, 6.0], index=["U1", "U3"], name="M3")
    expected_y_unknown = pd.Series([np.nan], index=["U2"], name="M3")

    assert_frame_equal_pretty(expected_X_known, X_known, mock_input=df)
    assert_frame_equal_pretty(expected_X_unknown, X_unknown, mock_input=df)
    assert_series_equal_pretty(expected_y_known, y_known, mock_input=df)
    assert_series_equal_pretty(expected_y_unknown, y_unknown, mock_input=df)
    print("Success!")

def knn_04(cosine_similarity_matrix):
    print("Testing...")
    X1 = pd.DataFrame({"M1": [3.0, 0.0], "M2": [4.0, 5.0]}, index=["U1", "U2"])
    X2 = pd.DataFrame({"M1": [0.0], "M2": [5.0]}, index=["U3"])
    expected = pd.DataFrame({"U3": [0.8, 1.0]}, index=["U1", "U2"])

    sim = cosine_similarity_matrix(X1, X2)

    assert_frame_equal_pretty(expected, sim, mock_input = [X1, X2])
    print("Success!")


def knn_05(predictions):
    print("Testing...")
    expected = pd.Series([-0.565795, 2.797070], index=["U032", "U758"], name="M4096")

    assert isinstance(predictions, pd.Series)
    assert predictions.name == expected.name
    assert predictions.index.tolist() == expected.index.tolist()

    np.testing.assert_allclose(expected.values, predictions.values, atol=0.2)
    print("Success!")

def knn_06(recommend):
    print("Testing...")
    predictions = pd.Series([1.5, 2.0, 3.2], index=["U1", "U2", "U3"], name="M1")
    threshold = 2.0
    expected = pd.Series([False, True, True], index=["U1", "U2", "U3"], name="M1")

    result = recommend(predictions, threshold)
    assert_series_equal_pretty(expected, result, mock_input = [predictions, threshold])
    print("Success!")


def evaluation_01(train_test_split):
    print("Testing...")
    X = pd.DataFrame({"M1": [1.0, 2.0, 3.0, 4.0], "M2": [5.0, 6.0, 7.0, 8.0]}, index=["U1", "U2", "U3", "U4"])
    y = pd.Series([10.0, 20.0, 30.0, 40.0], index=["U1", "U2", "U3", "U4"], name="target")

    test_size = 0.25

    splits = []

    # run multiple times to test randomness
    for _ in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size)
        assert len(X_train) == len(y_train),         "X_train and y_train are not aligned"
        assert len(X_test) == len(y_test),           "X_test and y_test are not aligned"
        assert len(X_train) + len(X_test) == len(X), "X_train and X_test are do not represent full data"
        assert X_train.shape[1] == X.shape[1],       "X_train does not have correct number of columns"
        assert X_test.shape[1] == X.shape[1],        "X_test does not have correct number of columns"
        assert X_train.index.equals(y_train.index),  "X_train and y_train are not aligned"
        assert X_test.index.equals(y_test.index),    "X_test and y_test are not aligned"
        splits.append(tuple(sorted(X_test.index)))

    assert len(set(splits)) > 1, "train_test_split not random"
    print("Success!")



def evaluation_02(mse):
    print("Testing...")
    y_true = pd.Series([1.0, -1.0, 3.0], index=["U1", "U2", "U3"])
    y_pred = pd.Series([3.0, 0.0, 4.0], index=["U1", "U2", "U3"])

    expected = 2.0

    result = mse(y_true, y_pred)

    assert isinstance(result, float), "mse() does not return float"

    assert result == expected, f"expected output: {expected}, your solution: {result} (with inputs {list(y_true)} and {list(y_pred)})"
    print("Success!")


import pandas as pd

def evaluation_03(confusion):
    print("Testing...")
    y_true = pd.Series([True, True, True, True, True, False, False, False, False, False], index=[f"U{i}" for i in range(10)])
    y_pred = pd.Series([True, True, False, False, False, True, False, False, False, False], index=y_true.index)

    expected = pd.DataFrame(
        {"actual pos": [2, 3], "actual neg": [1, 4]},
        index=["predicted pos", "predicted neg"],
    )

    result = confusion(y_true, y_pred)
    assert_frame_equal_pretty(expected, result, mock_input = [y_true, y_pred])
    print("Success!")
    

def evaluation_04(precision):
    print("Testing...")
    y_true = pd.Series([True, False, True, False], index=["U1", "U2", "U3", "U4"])
    y_pred = pd.Series([True, True, False, False], index=y_true.index)
    expected = 0.5

    result = precision(y_true, y_pred)

    assert isinstance(result, float), "function does not return float"
    assert result == expected, f"expected: {expected}, got: {result} (with inputs {list(y_true)} and {list(y_pred)})"
    print("Success!")


def evaluation_05(recall):
    print("Testing...")
    y_true = pd.Series([True, False, True, False], index=["U1", "U2", "U3", "U4"])
    y_pred = pd.Series([True, True, False, False], index=y_true.index)
    expected = 0.5

    result = recall(y_true, y_pred)

    assert isinstance(result, float), "function does not return float"
    assert result == expected, f"expected: {expected}, got: {result} (with inputs {list(y_true)} and {list(y_pred)})"
    print("Success!")



def real_data_01(precision_knn, precision_mean, recall_knn, recall_mean):
    print("Testing...")
    expected = {
        "precision_knn": approx(0.93, abs = 0.01), 
        "precision_mean": approx(0.84, abs = 0.01),
        "recall_knn": approx(0.50, abs = 0.01), 
        "recall_mean": approx(0.13, abs = 0.01)
    }
    errors = [f"expected {var_name}: {val}" for var_name, val in expected.items() if eval(var_name) != val]
    assert len(errors) == 0, "\n".join(errors)
    print("Success!")

def real_data_02(f1_knn, f1_mean):
    print("Testing...")
    expected = {
        "f1_knn": approx(0.65, abs = 0.01), 
        "f1_mean": approx(0.23, abs = 0.01)
    }
    errors = [f"expected {var_name}: {val}" for var_name, val in expected.items() if eval(var_name) != val]
    assert len(errors) == 0, "\n".join(errors)
    print("Success!")


############# Helper stuff #############

def display_side_by_side(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    header_left: str,
    header_right: str):
    html = f"""
    <div style="display: flex; gap: 60px; align-items: flex-start;">
        <div>
            <h4 style="margin-bottom: 8px;">{header_left}</h4>
            {df_left.style.to_html()}
        </div>
        <div>
            <h4 style="margin-bottom: 8px;">{header_right}</h4>
            {df_right.style.to_html()}
        </div>
    </div>
    """
    display(HTML(html))

def _render_object(obj) -> str:
    """Return HTML representation for various object types."""
    if isinstance(obj, pd.DataFrame):
        return obj.style.to_html()
    if isinstance(obj, pd.Series):
        return obj.to_frame().style.to_html()
    if isinstance(obj, (int, float, str)):
        return f"<pre>{escape(str(obj))}</pre>"
    return f"<pre>{escape(repr(obj))}</pre>"

def display_mock_inputs(
    mock_inputs,
    headers: list[str] | None = None,
    title: str = "input used for testing the function",
):
    if not isinstance(mock_inputs, (list, tuple)):
        mock_inputs = [mock_inputs]

    if headers is None:
        headers = [f"input argument {i+1}" for i in range(len(mock_inputs))]

    blocks = []
    for obj, header in zip(mock_inputs, headers):
        blocks.append(
            f"""
            <div>
                <h4 style="margin-bottom: 6px;">{escape(header)}</h4>
                {_render_object(obj)}
            </div>
            """
        )

    html = f"""
    <h4>{escape(title)}</h4>
    <div style="display: flex; gap: 30px; align-items: flex-start;">
        {''.join(blocks)}
    </div>
    """
    display(HTML(html))



def assert_is_series(obj, *, name: str = "result"):
    if not isinstance(obj, pd.Series):
        raise AssertionError(
            f"{name} must be a pandas Series, got {type(obj).__name__}."
        )
        
def assert_series_equal_pretty(
    expected: pd.Series,
    actual,
    *,
    mock_input=None,
    mock_headers: list[str] | None = None,
    name_expected: str = "expected output",
    name_actual: str = "output of your solution",
):
    assert_is_series(actual, name=name_actual)

    try:
        pd.testing.assert_series_equal(expected, actual)
    except AssertionError as e:
        if mock_input is not None:
            display_mock_inputs(mock_input, mock_headers)

        display_side_by_side(
            expected.to_frame(),
            actual.to_frame(),
            name_expected,
            name_actual,
        )

        raise AssertionError(str(e)) from None


def assert_is_dataframe(obj, *, name: str = "result"):
    """
    Assert obj is a pandas DataFrame, otherwise raise AssertionError.
    """
    if not isinstance(obj, pd.DataFrame):
        raise AssertionError(
            f"{name} must be a pandas DataFrame, got {type(obj).__name__}."
        )

def assert_frame_equal_pretty(
    expected: pd.DataFrame,
    actual,
    *,
    mock_input=None,
    mock_headers: list[str] | None = None,
    check_dtype: bool = False,
    check_like: bool = False,
):
    assert_is_dataframe(actual, name="your solution")

    try:
        pd.testing.assert_frame_equal(
            expected,
            actual,
            check_dtype=check_dtype,
            check_like=check_like,
        )
    except AssertionError as e:
        if mock_input is not None:
            display_mock_inputs(mock_input, mock_headers)

        display_side_by_side(
            expected,
            actual,
            "expected output",
            "output of your solution",
        )

        raise AssertionError(str(e)) from None




class approx:
    def __init__(self, expected, *, rel=1e-6, abs=1e-12):
        self.expected = float(expected)
        self.rel = rel
        self.abs = abs
        self._compute_d_()
        
    def _compute_d_(self):
        self.d = max(self.abs, self.rel * abs(self.expected))

    def __eq__(self, actual):
        actual = float(actual)
        return abs(actual - self.expected) <= self.d

    def __repr__(self):
        return f"{self.expected} ± {self.d}"


        
    

