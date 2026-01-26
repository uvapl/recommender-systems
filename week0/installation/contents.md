# Installatie

### 1. Installeer de conda-omgeving

* Download het environment-bestand:
  * Gebruik dit commando:

        curl -L -o environment.yml https://github.com/uvapl/recommender-systems/raw/refs/heads/2025/data/m0/environment.yml
        
  * Of download handmatige: [download](https://github.com/uvapl/recommender-systems/raw/refs/heads/2025/data/m0/environment.yml)
* Installeer de omgeving:
  `conda env create -f environment.yml`
* Ruim het bestand op:
  `rm environment.yml`
* Activeer de omgeving:
  `conda activate RecSys`

### 2. Installeer spaCy-modellen

* Voer de volgende commando’s uit om de spaCy-taalmodellen te downloaden:
    ```
    python -m spacy download en_core_web_sm
    python -m spacy download en_core_web_md
    ```

### 3. Start Jupyter

* Maak een map aan waarin je wilt werken. Het is het eenvoudigst om alle notebooks in dezelfde directory te plaatsen, omdat ze soms dezelfde databestanden gebruiken.
* Vanaf dit moment kun je alle notebooks vanuit deze map uitvoeren.
* Ga naar de aangemaakte directory:
  `cd ~/[jouw folder]`
* Controleer of je de juiste conda-omgeving gebruikt: `(RecSys)`
* Start Jupyter:

  * JupyterLab: `jupyter lab`
  * of klassiek: `jupyter notebook`

