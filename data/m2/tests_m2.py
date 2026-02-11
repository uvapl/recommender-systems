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
    

def w2v_test_03(sim):
    print("Testing...")
    SIMW2V.index.name = "NewsID"
    SIMW2V.columns.name = "NewsID"
    assert_frame_equal_pretty(SIMW2V, sim, atol = 0.01)
    print("Success!")


def jaccard_test_01(one_hot):
    print("Testing...")
    data = {
        "EU":        [1, 1, 1, 0, 0],
        "US":        [0, 1, 0, 0, 0],
        "analysis":  [0, 1, 0, 0, 1],
        "climate":   [0, 0, 0, 0, 1],
        "economy":   [0, 1, 0, 0, 1],
        "europe":    [1, 0, 0, 0, 0],
        "football":  [0, 0, 1, 0, 0],
        "global":    [0, 0, 0, 1, 1],
        "interview": [0, 0, 0, 1, 0],
        "news":      [1, 0, 1, 0, 0],
        "politics":  [1, 1, 0, 0, 0],
        "sports":    [0, 0, 1, 1, 0],
        "tennis":    [0, 0, 0, 1, 0],
    }
    index = ["N001", "N002", "N003", "N004", "N005"]
    solution_df = pd.DataFrame(data, index=index)
    solution_df.index.name = "article_id"
    solution_df.columns.name = "features"
    assert_frame_equal_pretty(solution_df, one_hot)
    print("Success!")



def jaccard_test_02(sim):
    print("Testing...")
    assert_frame_equal_pretty(SIM_JACC, sim, atol = 0.01)
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



SIMW2V = pd.DataFrame(
    [[1.0, 0.826, 0.826, 0.827, 0.861, 0.918, 0.882, 0.642, 0.71, 0.565, 0.754, 0.69, 0.726, 0.514, 0.712, 0.667, 0.736, 0.707, 0.693, 0.669], 
     [0.826, 1.0, 0.744, 0.825, 0.889, 0.839, 0.87, 0.645, 0.695, 0.523, 0.697, 0.707, 0.673, 0.529, 0.702, 0.667, 0.72, 0.701, 0.693, 0.657], 
     [0.826, 0.744, 1.0, 0.811, 0.778, 0.827, 0.824, 0.697, 0.722, 0.588, 0.734, 0.69, 0.718, 0.57, 0.683, 0.649, 0.683, 0.671, 0.659, 0.634], 
     [0.827, 0.825, 0.811, 1.0, 0.799, 0.823, 0.828, 0.739, 0.755, 0.637, 0.811, 0.775, 0.764, 0.642, 0.744, 0.697, 0.759, 0.711, 0.709, 0.699], 
     [0.861, 0.889, 0.778, 0.799, 1.0, 0.87, 0.9, 0.64, 0.695, 0.547, 0.733, 0.739, 0.706, 0.52, 0.739, 0.71, 0.769, 0.755, 0.75, 0.701], 
     [0.918, 0.839, 0.827, 0.823, 0.87, 1.0, 0.877, 0.663, 0.717, 0.575, 0.754, 0.674, 0.721, 0.533, 0.737, 0.714, 0.761, 0.759, 0.721, 0.677], 
     [0.882, 0.87, 0.824, 0.828, 0.9, 0.877, 1.0, 0.727, 0.755, 0.583, 0.801, 0.747, 0.789, 0.602, 0.771, 0.758, 0.804, 0.785, 0.766, 0.744], 
     [0.642, 0.645, 0.697, 0.739, 0.64, 0.663, 0.727, 1.0, 0.88, 0.795, 0.848, 0.806, 0.817, 0.859, 0.762, 0.751, 0.768, 0.702, 0.726, 0.741], 
     [0.71, 0.695, 0.722, 0.755, 0.695, 0.717, 0.755, 0.88, 1.0, 0.801, 0.838, 0.815, 0.819, 0.797, 0.74, 0.696, 0.748, 0.699, 0.706, 0.708], 
     [0.565, 0.523, 0.588, 0.637, 0.547, 0.575, 0.583, 0.795, 0.801, 1.0, 0.786, 0.674, 0.75, 0.768, 0.626, 0.581, 0.667, 0.609, 0.605, 0.623], 
     [0.754, 0.697, 0.734, 0.811, 0.733, 0.754, 0.801, 0.848, 0.838, 0.786, 1.0, 0.824, 0.871, 0.777, 0.759, 0.739, 0.786, 0.754, 0.718, 0.755], 
     [0.69, 0.707, 0.69, 0.775, 0.739, 0.674, 0.747, 0.806, 0.815, 0.674, 0.824, 1.0, 0.77, 0.717, 0.746, 0.704, 0.741, 0.689, 0.707, 0.755], 
     [0.726, 0.673, 0.718, 0.764, 0.706, 0.721, 0.789, 0.817, 0.819, 0.75, 0.871, 0.77, 1.0, 0.759, 0.705, 0.692, 0.717, 0.672, 0.662, 0.697], 
     [0.514, 0.529, 0.57, 0.642, 0.52, 0.533, 0.602, 0.859, 0.797, 0.768, 0.777, 0.717, 0.759, 1.0, 0.636, 0.622, 0.661, 0.59, 0.612, 0.642], 
     [0.712, 0.702, 0.683, 0.744, 0.739, 0.737, 0.771, 0.762, 0.74, 0.626, 0.759, 0.746, 0.705, 0.636, 1.0, 0.872, 0.922, 0.891, 0.895, 0.927], 
     [0.667, 0.667, 0.649, 0.697, 0.71, 0.714, 0.758, 0.751, 0.696, 0.581, 0.739, 0.704, 0.692, 0.622, 0.872, 1.0, 0.904, 0.895, 0.872, 0.86], 
     [0.736, 0.72, 0.683, 0.759, 0.769, 0.761, 0.804, 0.768, 0.748, 0.667, 0.786, 0.741, 0.717, 0.661, 0.922, 0.904, 1.0, 0.928, 0.944, 0.923], 
     [0.707, 0.701, 0.671, 0.711, 0.755, 0.759, 0.785, 0.702, 0.699, 0.609, 0.754, 0.689, 0.672, 0.59, 0.891, 0.895, 0.928, 1.0, 0.924, 0.895], 
     [0.693, 0.693, 0.659, 0.709, 0.75, 0.721, 0.766, 0.726, 0.706, 0.605, 0.718, 0.707, 0.662, 0.612, 0.895, 0.872, 0.944, 0.924, 1.0, 0.904], 
     [0.669, 0.657, 0.634, 0.699, 0.701, 0.677, 0.744, 0.741, 0.708, 0.623, 0.755, 0.755, 0.697, 0.642, 0.927, 0.86, 0.923, 0.895, 0.904, 1.0]],
    columns = ['N0001', 'N0002', 'N0003', 'N0004', 'N0005', 'N0006', 'N0007', 'N0008', 'N0009', 'N0010', 'N0011', 'N0012', 'N0013', 'N0014', 'N0015', 'N0016', 'N0017', 'N0018', 'N0019', 'N0020'],
    index = ['N0001', 'N0002', 'N0003', 'N0004', 'N0005', 'N0006', 'N0007', 'N0008', 'N0009', 'N0010', 'N0011', 'N0012', 'N0013', 'N0014', 'N0015', 'N0016', 'N0017', 'N0018', 'N0019', 'N0020'])


SIM_JACC = pd.DataFrame(
    [[1.0, 0.286, 0.333, 0.0, 0.0],
     [0.286, 1.0, 0.125, 0.0, 0.286],
     [0.333, 0.125, 1.0, 0.143, 0.0],
     [0.0, 0.0, 0.143, 1.0, 0.143],
     [0.0, 0.286, 0.0, 0.143, 1.0]],
    index = ['N001', 'N002', 'N003', 'N004', 'N005'],
    columns =['N001', 'N002', 'N003', 'N004', 'N005'])
SIM_JACC.index.name = 'article_id'
SIM_JACC.columns.name = 'article_id'
SIM_JACC