# Tentamenstof

### Opzet

Het tentamen is een schriftelijk (pen en papier) tentamen en bestaat uit voornamelijk open vragen. Je hoeft op het tentamen geen code te produceren, maar je wordt wel geacht de algoritmes die in het vak behandeld zijn te kunnen reproduceren. Als je iets uit moet rekenen zullen de waardes zo gekozen worden dat je geen rekenmachine nodig hebt.

### Stof

De tentamenstof bevat:

- de begrippen uit alle notebooks, bestaande uit:
    - de conceptuele uitleg
    - de behandelde algoritmes
    - de gebruikte similarity- en evaluatiematen
    - de open vragen
- de hoorcolleges:
    - De technische hoorcolleges (Simon)
    - De theoretische hoorcolleges (Dina)
- het leesmateriaal:
    - hoofdstuk 1 en 2 van Aggarwal: [link](/week2/book-1-2)
    - hoofdstuk 7 van Aggarwal: [link](/week5/book-7)

## Oefenvragen

De vragen hieronder geven je een beeld van het *soort* vragen dat je kan verwachten. Het echte tentamen zal wel andere specifieke onderwerpen bevragen en meer vragen bevatten. Beantwoord vragen kort maar krachtig. Betrek geen onnodige details in je antwoorden. In veel gevallen zullen een paar zinnen volstaan. 

### Vraag 1

**Deel a.**

We hebben een verzameling documenten die ieder uit een enkele zin bestaan.

* **Zin 1**: Vlijtige graven maken hun eigen huisjes.
* **Zin 2**: Gekke graven graven hun eigen huisjes.
* **Zin 3**: Morbide graven graven hun eigen graven.

Vul in de onderstaande tabel de *TF-score* voor alle termen in alle zinnen:

|       | vlijtige | gekke | morbide | maken | graven | hun | eigen | huisjes  |
|-------|----------|-------|---------|-------|--------|-----|-------|----------|
| Zin 1 |          |       |         |       |        |     |       |          |
| Zin 2 |          |       |         |       |        |     |       |          |
| Zin 2 |          |       |         |       |        |     |       |          |


**Deel b.**

We hebben de volgende *IDF-scores*:

|     | vlijtige | gekke | morbide | maken | graven | hun | eigen | huisjes |
|-----|----------|-------|---------|-------|--------|-----|-------|----------|
| IDF | 6        | 3     | 9       | 0     | 6      | 0   | 3     | 3        |

Geef de *TF-IDF-vectorisatie* van de drie zinnen.


**Deel c.**

Leg in woorden kort uit waarom word2vec vaak een betere vecorisatie geeft dan TF-IDF.

### Vraag 2

**Deel a.**

Dit zijn onze matrices:

$$
A =
\begin{bmatrix}
1 & 0 & 0.5 \\
1 & 1 & 0
\end{bmatrix}
, B = 
\begin{bmatrix}
2 & 1\\
0 & 1\\
2 & 0
\end{bmatrix}
$$

We hebben de volgende vergelijking:

$$
A\cdot B = C
$$

Wat is de matrix $$C$$?


**Deel b.**

* Voor matrixfactorisatie hebben we de matrices $$A$$, $$B$$, en $$\hat{Y}$$. 
* We hebben de vergelijking, $$A \cdot B = \hat{Y}$$. 
* We weten dat de matrix $$\hat{Y}$$, 6 rijen en 8 kolommen heeft. 
* Verder weten we dat de matrix $$A$$, 6 rijen en 5 kolommen heeft. 

Hoeveel rijen en kolommen heeft $$B$$?

**Deel c.**

$$
A =
\begin{bmatrix}
0.0 & 1.0 \\
2.0 & 0.5 \\
? & 2.0 \\
\end{bmatrix},
B = 
\begin{bmatrix}
 0.0 & 0.0 & 2.0 \\
 0.5 & 3.0 & 2.0 \\
\end{bmatrix},
Y =
\begin{bmatrix}
0.61 & 3.01 & 1.99 \\
0.25 & 1.48 & 5.11 \\
1.00 & 5.91 & 6.00 \\
\end{bmatrix}
$$

We hebben de matrices $$A$$ en $$B$$ verkregen door $$Y$$ te factoriseren. Wat is de beste waarde die op de plek van het vraagteken (?) kan staan? Beargumenteer.

### Vraag 3

Hieronder zie je een bekende functie die een diversiteits *metric* berekent op basis van gebruikersdata uit een nieuwsrecommendersysteem. Van elke gebruiker weten we welke artikelen aan hen zijn aanbevolen, en van die artikelen op welke zij hebben geklikt.

    def compute_mean_distance_data(selected_data, column = "Clicked"):
        means = []

        for idx, row in tqdm(list(selected_data.iterrows())):
            articles = row[column]
            genome_selected = genome.loc[articles]

            if len(genome_selected) > 20: 
                genome_selected = genome_selected.sample(20)

            mean = compute_mean_distance(genome_selected)
            means.append(mean)

        return sum(means)/len(means)

Het eindresultaat van deze functie is een getal dat **diversiteit** weergeeft.

**Deel a.** 

Leg uit hoe diversiteit wordt gedefinieerd, op basis van hoe deze in de code wordt berekend.

**Deel b.** 

Leg uit wat diversiteit betekent. Wat zegt een hoge of lage diversiteit over een recommender system?

**Deel c.**

De resultaten van bovenstaande functie zijn als volgt:

    Diversiteit van aangeklikte “gelikete” artikelen: 1.150
    Diversiteit van aanbevolen artikelen: 1.206

Naast deze twee resultaten hebben we ook de gemiddelde afstand berekend voor een groep gebruikers in combinatie met willekeurige artikelen. Dit leidde tot een baseline-diversiteit van 1.67.

Wat betekenen deze diversiteitswaarden?

### Vraag 4
Is matrixvermenigvuldiging commutatief? Met andere woorden: is voor twee matrices $$A$$ en $$B$$ het product $$A \cdot B$$ hetzelfde als $$B \cdot A$$? Waarom wel of waarom niet?

### Vraag 5
We hebben een *KNN-regressiealgoritme* dat beoordelingen voorspelt op een schaal van 1 tot 10. Voor aanbevelingen gebruiken we een drempelwaarde van 7,5. Het algoritme heeft een lage *precision*, maar een hoge *recall* voor onze aanbevelingen. Wat zouden we kunnen doen om de *precision* te verbeteren? En welk effect zou dat hebben op de *recall*? Leg uit waarom.


### Vraag 6
Strava is een fitness-app voor hardlopen, fietsen of wandelen. Wanneer je sport, registreert de app je route op een kaart, samen met optionele gegevens zoals je hartslag. Daarna kun je je sportsessie delen met andere mensen die de Strava-app hebben, samen met foto’s en opmerkingen.

**Deel a.** 
Noem drie verschillende soorten data die Strava direct kan registreren.

**Deel b.** 
Voortbouwend op je antwoorden op de vorige vraag: noem nog drie ander soorteen datat die (alleen) via *proxies* kunnen worden afgeleid. Leg voor elke proxy uit hoe die inferentie kan worden gemaakt.


### Vraag 7
Wat is gatekeeping? En waarin verschilt algoritmische gatekeeping van traditionele gatekeeping?

### Vraag 8
Mejias & Couldry (2019) beschrijven dataficatie als een circulair proces in twee stappen.

1. Noem de twee stappen van dit proces. 
2. Leg kort uit wat elke stap inhoudt.


### Vraag 9

Automatisering maakt inmiddels deel uit van ons dagelijks leven, of we ons daarvan bewust zijn of niet. We hebben twee domeinen besproken waarin automatisering duidelijk zichtbaar is in het dagelijks leven: de samenleving (*society*) en de media. Kies één van deze twee domeinen en bespreek dit aan de hand van de volgende onderdelen:

1. Noem één concreet voorbeeld waarin algoritmen beslissingen nemen binnen dit domein en leg kort uit hoe algoritmen bij deze besluitvorming betrokken zijn (3–5 zinnen).

2. Bespreek kort één onbedoeld gevolg van automatisering binnen dit domein. Tip: infinite scroll is hier geen voorbeeld van, omdat dat een onbedoeld gevolg is van platformontwerp, niet van automatisering. (2–3 zinnen)

3. Beschouw Gillespie’s (2014) zes dimensies van publiek relevante algoritmen met politieke betekenis:

   * *Patterns of inclusion* (Patronen van inclusie)
   * *Cycles of anticipation* (Cycli van anticipatie)
   * *Evaluation of relevance* (Evaluatie van relevantie)
   * *Entanglement with practice* (Verstrengeling met praktijken)
   * *Promise of algorithmic objectivity* (Belofte van algoritmische objectiviteit)
   * *Production of cultivated publics* (Productie van gecultiveerde publieken)

Kies drie van deze dimensies en bespreek hoe ze relevant zijn voor het voorbeeld dat je hierboven hebt gekozen. Je mag deze alleen bespreken in relatie tot jouw eigen voorbeeld, niet aan de hand van een andere casus.



## Beoordeling

De beoordeling is gebaseerd op de [SOLO-taxonomie](https://en.wikipedia.org/wiki/Structure_of_observed_learning_outcome). Voor elke vraag kijken we naar je inzicht gegeven de behandelede theorie. Per vraag kan je inzicht op volgende niveau's laten zien:

1. Je antwoord maakt geen gebruik van relevante concepten uit de theorie of bevat onwaarheden.
2. Je antwoord is gelinkt aan een relevant concept uit de theorie.
3. Je antwoord behandelt een aantal losse concepten die elk gelinkt zijn aan een relevant concept uit de theorie.
4. Je antwoord legt een duidelijk verband tussen meerdere verschillende relevante concepten.
5. Je antwoord laat zien hoe relevante concepten toegepast worden op andere dan de bekende voorbeelden.

Elke vraag valt tot op niveau 3 te beantwoorden. Bij soommige vragen wordt je verwacht om op niveau 4 of 5 te antwoorden.