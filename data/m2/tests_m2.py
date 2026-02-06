import pandas as pd
import numpy as np
from html import escape

from IPython.display import display, HTML

def test_01(news_df):
    print("Testing...")
    expected = MOCK_DATA
    expected.index = expected.index.set_names("NewsID")
    
    assert_frame_equal_pretty(expected, news_df)
    print("Success!")



def test_02(tokenize_and_lemmatize):
    print("Testing...")
    texts = pd.Series(["Cats are running!", "The mice ate cheese."], index=["A", "B"])
    expected = pd.Series([["cat", "run"], ["mouse", "eat", "cheese"]], index=["A", "B"])
    result = tokenize_and_lemmatize(texts)

    assert_series_equal_pretty(expected, result)
    print("Success!")

def test_03(compute_term_document_counts):
    print("Testing...")
    lemmas = pd.Series([["cat", "dog"], ["dog", "mouse"], ["cat", "mouse"], ["cat"]], index=["D1", "D2", "D3", "D4"])
    expected = pd.Series({"dog": 2, "mouse": 2}, dtype=int)
    result = compute_term_document_counts(lemmas, min_df=0.25, max_df=0.5)

    assert_series_equal_pretty(result.sort_index(), expected.sort_index(), mock_input = [lemmas, 0.25, 0.5])
    print("Success!")

def test_04(compute_tf):
    print("Testing...")
    lemmas = pd.Series([["cat", "dog", "cat"], ["dog", "bird"], []], index=["D1", "D2", "D3"])
    vocab = ["cat", "dog"]
    expected = pd.DataFrame({"cat": [2/3, 0.0, 0.0], "dog": [1/3, 1.0, 0.0]}, index=["D1", "D2", "D3"])
    result = compute_tf(lemmas, vocab)

    assert_frame_equal_pretty(expected, result, mock_input = [lemmas, vocab])
    print("Success!")

def test_05(compute_tfidf):
    print("Testing...")
    tf = pd.DataFrame({"cat": [0.5, 0.0], "dog": [0.5, 1.0]}, index=["D1", "D2"])
    doc_counts = pd.Series({"cat": 1, "dog": 2})
    expected = pd.DataFrame({"cat": [0.5 * np.log(2), 0.0], "dog": [0.0, 0.0]}, index=["D1", "D2"])

    result = compute_tfidf(tf, doc_counts)
    assert_frame_equal_pretty(expected, result, mock_input=[tf, doc_counts])
    print("Success!")


def test_06(transform):
    print("Testing...")
    impressions = pd.DataFrame({"UserID": ["U1", "U2"], "Impressions": ["N0001-1 N0002-0", "N0010-0"]})
    expected_X = pd.DataFrame({"UserID": ["U1", "U1", "U2"], "ArticleID": ["N0001", "N0002", "N0010"]})
    expected_y = pd.Series([True, False, False], name="Clicked")
    X, y = transform(impressions)

    assert_frame_equal_pretty(expected_X, X.reset_index(drop=True), mock_input=[impressions])
    assert_series_equal_pretty(expected_y, y.reset_index(drop=True), mock_input=[impressions])
    print("Success!")

def test_07(knn_recommend):
    print("Testing...")
    X_train = pd.DataFrame(
        [['U0019', 'N0005'], ['U0019', 'N0008'], ['U0019', 'N0011'], ['U0020', 'N0008'], ['U0020', 'N0011'], ['U0020', 'N0005'],
         ['U0019', 'N0019'], ['U0019', 'N0012'], ['U0019', 'N0013'], ['U0019', 'N0015'], ['U0019', 'N0017'], ['U0020', 'N0013'],
         ['U0020', 'N0017'], ['U0020', 'N0019'], ['U0020', 'N0020']],
        index=[0, 3, 6, 51, 54, 67, 68, 196, 197, 199, 201, 294, 298, 300, 399],
        columns=['UserID', 'ArticleID'])
    
    y_train = pd.Series(
        [False, False, False, True, False, True, False, False, False, True, False, False, True, True, True],
        index=[0, 3, 6, 51, 54, 67, 68, 196, 197, 199, 201, 294, 298, 300, 399],
        name='Clicked')
    
    X_test = pd.DataFrame([['U0019', 'N0020'], ['U0020', 'N0015']], index=[69, 296], columns=['UserID', 'ArticleID'])

    y_hat_expected = pd.Series([False, True], index=[69, 296], name=None)
    
    y_hat = knn_recommend(X_train, y_train, X_test, SIM, 3)
    assert_series_equal_pretty(y_hat_expected, y_hat, mock_input = [X_train, y_train, X_test, SIM, 3])
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
    atol: float = 0.0,
):
    assert_is_dataframe(actual, name="your solution")

    try:
        pd.testing.assert_frame_equal(
            expected,
            actual,
            check_dtype=check_dtype,
            check_like=check_like,
            atol = atol
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







SIM = pd.DataFrame(
    [[0.9999999999999997, -0.41572898898655286, -0.2692806904391772, -0.2163158916169383, -0.21464011810797173, -0.2678506982412996, -0.3216605626703846, -0.3216605626703846, -0.2678506982412996], [-0.41572898898655286, 1.0000000000000004, 0.29041270361460086, 0.5927730560850033, 0.4061420103166114, -0.29522975194224654, -0.3545399312016382, -0.3545399312016382, -0.29522975194224654], [-0.2692806904391772, 0.29041270361460086, 0.9999999999999999, -0.15443675503346926, 0.872000461182872, 0.2118678775337788, 0.13423166697326627, 0.13423166697326627, 0.2118678775337788], [-0.2163158916169383, 0.5927730560850033, -0.15443675503346926, 1.0000000000000002, -0.12309951852297032, -0.1536166317843673, -0.18447744411244557, -0.18447744411244557, -0.1536166317843673], [-0.21464011810797173, 0.4061420103166114, 0.872000461182872, -0.12309951852297032, 1.0000000000000002, -0.1524265819911844, -0.18304831927314374, -0.18304831927314374, -0.1524265819911844], [-0.2678506982412996, -0.29522975194224654, 0.2118678775337788, -0.1536166317843673, -0.1524265819911844, 1.0, 0.8459857103818813, 0.8459857103818813, 1.0], [-0.3216605626703846, -0.3545399312016382, 0.13423166697326627, -0.18447744411244557, -0.18304831927314374, 0.8459857103818813, 1.0000000000000002, 1.0000000000000002, 0.8459857103818813], [-0.3216605626703846, -0.3545399312016382, 0.13423166697326627, -0.18447744411244557, -0.18304831927314374, 0.8459857103818813, 1.0000000000000002, 1.0000000000000002, 0.8459857103818813], [-0.2678506982412996, -0.29522975194224654, 0.2118678775337788, -0.1536166317843673, -0.1524265819911844, 1.0, 0.8459857103818813, 0.8459857103818813, 1.0]],
    index=['N0005', 'N0008', 'N0011', 'N0012', 'N0013', 'N0015', 'N0017', 'N0019', 'N0020'],
    columns=['N0005', 'N0008', 'N0011', 'N0012', 'N0013', 'N0015', 'N0017', 'N0019', 'N0020']
)




MOCK_DATA = pd.DataFrame.from_dict({'Title': {'N0001': 'Summit treaty talks', 'N0002': 'Parliament debates sanctions', 'N0003': 'Ceasefire corridor monitored', 'N0004': 'Ministers expand sanctions', 'N0005': 'Coalition vote after summit', 'N0006': 'Border talks pause', 'N0007': 'Regional summit statement', 'N0008': 'Model update hits benchmarks', 'N0009': 'Chipset cores and power', 'N0010': 'Kernel patch improves stability', 'N0011': 'Cloud outage and routing logs', 'N0012': 'Battery charging cycle gains', 'N0013': 'Firmware exploit mitigation', 'N0014': 'Quantum toolchain demo', 'N0015': 'Sequel review: pacing wins', 'N0016': 'Drama review: acting and script', 'N0017': 'Reboot review: plot problems', 'N0018': 'Indie romance review surprise', 'N0019': 'Thriller review: suspense lands', 'N0020': 'Comedy sequel review: chemistry'}, 'Abstract': {'N0001': 'Diplomats meet at summit; treaty draft targets border security and sanctions relief.', 'N0002': 'Parliament votes on sanctions package; coalition ministers argue for diplomacy at the summit.', 'N0003': 'Diplomats announce ceasefire corridor; monitors report border compliance and security checks.', 'N0004': 'Finance ministers expand sanctions; treaty terms link trade access to ceasefire monitoring.', 'N0005': 'Coalition leaders face confidence vote; parliament weighs treaty, sanctions, and border guarantees.', 'N0006': 'Negotiators pause border talks; diplomats cite treaty language, security guarantees, and sanctions disputes.', 'N0007': 'Summit statement urges diplomacy, treaty progress, ceasefire monitoring, and election oversight by parliament.', 'N0008': 'New model update improves benchmark accuracy; lower latency reported after patch and toolchain tuning.', 'N0009': 'Chipset launch touts faster cores and lower power; firmware update targets stability on benchmarks.', 'N0010': 'Kernel patch fixes memory bug; update reduces latency and improves security across toolchain builds.', 'N0011': 'Cloud outage traced to routing error; engineers review logs and ship patch update to restore service.', 'N0012': 'Battery chemistry boosts charging speed and cycle life; benchmarks show lower heat and power draw.', 'N0013': 'Researchers disclose firmware exploit; vendors push update patch and mitigation guidance with monitoring logs.', 'N0014': 'Quantum demo accelerates optimization; developers integrate toolchain update and compare benchmarks and latency.', 'N0015': 'Review praises pacing and soundtrack; director delivers action and strong box office returns.', 'N0016': 'Review highlights acting and screenplay; director earns awards as audience ratings climb.', 'N0017': 'Review criticizes plot and dialogue; visuals strong but box office and audience scores fall.', 'N0018': 'Review notes cinematography and acting; screenplay feels warm and audience buzz lifts box office.', 'N0019': 'Review applauds suspense and editing; finale twist boosts audience ratings and box office legs.', 'N0020': 'Review enjoys jokes and cast chemistry; pacing steady and soundtrack supports the sequel’s box office.'}, 'text': {'N0001': 'Summit treaty talks Diplomats meet at summit; treaty draft targets border security and sanctions relief.', 'N0002': 'Parliament debates sanctions Parliament votes on sanctions package; coalition ministers argue for diplomacy at the summit.', 'N0003': 'Ceasefire corridor monitored Diplomats announce ceasefire corridor; monitors report border compliance and security checks.', 'N0004': 'Ministers expand sanctions Finance ministers expand sanctions; treaty terms link trade access to ceasefire monitoring.', 'N0005': 'Coalition vote after summit Coalition leaders face confidence vote; parliament weighs treaty, sanctions, and border guarantees.', 'N0006': 'Border talks pause Negotiators pause border talks; diplomats cite treaty language, security guarantees, and sanctions disputes.', 'N0007': 'Regional summit statement Summit statement urges diplomacy, treaty progress, ceasefire monitoring, and election oversight by parliament.', 'N0008': 'Model update hits benchmarks New model update improves benchmark accuracy; lower latency reported after patch and toolchain tuning.', 'N0009': 'Chipset cores and power Chipset launch touts faster cores and lower power; firmware update targets stability on benchmarks.', 'N0010': 'Kernel patch improves stability Kernel patch fixes memory bug; update reduces latency and improves security across toolchain builds.', 'N0011': 'Cloud outage and routing logs Cloud outage traced to routing error; engineers review logs and ship patch update to restore service.', 'N0012': 'Battery charging cycle gains Battery chemistry boosts charging speed and cycle life; benchmarks show lower heat and power draw.', 'N0013': 'Firmware exploit mitigation Researchers disclose firmware exploit; vendors push update patch and mitigation guidance with monitoring logs.', 'N0014': 'Quantum toolchain demo Quantum demo accelerates optimization; developers integrate toolchain update and compare benchmarks and latency.', 'N0015': 'Sequel review: pacing wins Review praises pacing and soundtrack; director delivers action and strong box office returns.', 'N0016': 'Drama review: acting and script Review highlights acting and screenplay; director earns awards as audience ratings climb.', 'N0017': 'Reboot review: plot problems Review criticizes plot and dialogue; visuals strong but box office and audience scores fall.', 'N0018': 'Indie romance review surprise Review notes cinematography and acting; screenplay feels warm and audience buzz lifts box office.', 'N0019': 'Thriller review: suspense lands Review applauds suspense and editing; finale twist boosts audience ratings and box office legs.', 'N0020': 'Comedy sequel review: chemistry Review enjoys jokes and cast chemistry; pacing steady and soundtrack supports the sequel’s box office.'}})

