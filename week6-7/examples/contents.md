## Example projects

### Example 1: Polarization in MIND

**Perspective**
We work for the Microsoft on their news recommender system and we want to understand if the current algorithm could potentially amplify political polarization.

**Context**
In the final assignment on news diversity, we did not find lower diversity in the recommendations produced from the Microsoft News (MIND) dataset. However, the used diversity metric might not focus on the right dimension

Filter-bubble effects are often less pronounced than you might intuitively expect, but none the less we want to make absolutely sure that our recommender system does not contribute to increased political polarization.

Detecting political affiliation (relevant to polarization) is not something that is captured by the similarity-based diversity measures we have used so far. As a result, these effects would never emerge give the limitation of the previous assignment. For example, we will not be able to see if specific US-based users get recommended more Republican or Democratic news content.

**Goal**
Develop a metric that uses linguistic cues in data that suggests partisanship and using those to see if the Microsoft Recommender system gives politically biased recommendations.

Discuss your findings. Are the recommendations politically biases. Is this good or bad for the user? And for Microsoft? And for the news platform? And for society? Why?

### Example 2: Fairness (gender) in MovieLens
As a consultant for a movie streaming service you want to make suggestions for improvement of their algorithm, specifically to fairness. As a first step you are analyzing existing data. We use the data of MovieLens as a rerefence. Are the recommendations using this data fair or biased? If they are biased, what impact would this have? Why would it matter? What alterations could you make to the algorithm? What (unintended) consequences could this have?

### Example 3: Credibility of a NRS
Design a recommender system that can takes the credibility of news articles into account. How can you determine trustworthiness? Why would that matter? What impact would it have for small grassroots/citizen journalism platforms?