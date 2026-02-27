# Final Project: Recommender Systems in Society

Volgende week begint het eindproject.Daar moet je deze week al twe belangrijke dingen voor doen.

- Vorm een groepje van vier studenten. (Ja het moeten er precies vier zijn ;) )
- Lees de opzet voor het project hieronder.
- Bedenk een onderwerp.

**Maandag 9 maart tussen 9:00 en 11:00** moet je (met je hele groep) het idee aan ons (Dina, Simon en Rein) voorleggen! Dan krijg je van ons een og/no-go. 

## Opzet

For this project, you will work in groups of four to design, analyze, or evaluate a recommender system in a real-world domain, such as: media, streaming services, news, social media, labor market, e commerce.
Your project must combine a technical component and a theoretical justification component.

**Technical component.** You choose one or more of the following technical routes:

- Analyze the working of an existing recommender algorithm
- Analyze data from an existing recommender system (e.g. MIND).
- Design and prototype your own recommender algorithm 
- Implement a new or adapted evaluation metric (e.g. for diversity, fairness, exposure, polarization, credibility).
- Compare existing algorithms or configurations

**Theoretical justifcation.** All technical choices and conclusions must be grounded in the theories from Dina’s lectures and tutorials. You should explicitly connect to any of the themes as discussed in those classes. For example:

-	Who is advantaged/disadvantaged by the RS?
-	What effects does the RS have on the distribution of information? 
-	Who is in charge of the data? Where does it come from? 
-	What biases can you identify (and avoid)? 
-	What outcomes are you going to prioritize and what trade-offs do you have to make?
-	What about: Accuracy, transparency, bias, explainability, fairness, user wants vs. user needs? 
-	How do you reconcile economic interest with public interest?
-	What negative implications can you anticipate – how can you prevent them?  

The main objective is not just to build something that works, but to show that you can ground your design decisions in theoretical, scientific and ethical concepts and that you can use to justify your design choices.

You are allowed to specialize within the group (for instance, two students focusing more on programming and two on conceptual/theoretical work), but the final product must clearly integrate both aspects into a single coherent project.

The preferred language for the project is English, but you are allowed to use Dutch.

To give you an idea of something what we’re looking for as a project, have a look at the examples below.


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


## Output

The project is treated as group work, and all group members receive the same grade.

The output for this assignment therefore consists of three parts, all due: **March 20th at 11:00h**:

1. A short presentation/pitch delivered during the final tutorial meeting (5 to 7 minutes max)
2. A written report that elaborates on and substantiates the key findings highlighted in the presentation (2000 - 3000 words max)
3. Code base and data, preferably in the form of a github repository. All code should be well documented, straightforward to run by the staff, and produce the results as described in the report.  

## Grading

| Component                                    | Weight |
|----------------------------------------------|--------|
| Technical effort	                           | 30%    |
| Theoretical justification & understanding	   | 30%    |
| Integration of technical & societal aspects  | 20%    |
| Presentation (pitch)	                       | 20%    |

You can use one of the projects proposed by us, but if you do so, your final project **grade is capped at 8.0** (no point deductions; only the maximum is limited).

### Timeline

- Week 5: Form groups of four, choose a recommender‑system domain, and draft a project idea.
- Week 6:
    - Monday: Discuss your idea to staff and get approval/refinement. 
    - Thursday: Mid‑week progress check; each student reports their concrete contributions.  
- Week 7:
    - Monday: Show a working MVP and discuss remaining work with staff.  
    - Friday: Deadline final report + final presentation (10‑minute pitch).
