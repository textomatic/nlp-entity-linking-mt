
# NLP Entity Linking for Medical Transcripts

nlp-entity-linking-mt

**About this project**:
- Dataset of raw medical transcriptions from Kaggle was used
- Pandas library was used to explore, prepare, and process the data (including handling missing values)
- NLP analysis on the data was done by making calls to the IBM Project Debater API to leverage the following:
	- Argument Quality service - to assess the relevance and quality of a sample of the medical transcription text
	- Key Points Analysis service - to identify key points (i.e. medical diagnoses) as supporting elements within the medical transcription text
	- Term Wikifier service - to link the key points (i.e. medical diagnoses) to closely related terms from Wikipedia as a knowledge base; this helps to disambiguate the results of analysis and uncover more commonly used terminology than is found in doctors' phrasing
- Bar chart visualizations of the NLP analysis result were created using Matplotlib and Seaborn
- Finally, an interactive data visualization dashboard of the analysis result was created using Streamlit; this displays result in a form more readily used by stakeholders

**Possible next steps**:
- Based on the NLP analysis result, construct links and annotations to build a knowledge graph

## Demo

The Streamlit app is hosted on Streamlit Cloud. Visit link to view the data visualization of the analysis result.
  
## How to run this project

### Run from the command line

1. Set up the Python environment:
```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -e .
```

2. Launch JupyterLab:
```bash
jupyter-lab
```

3. Your browser will open JupyterLab. Run the Jupyter notebooks under `./notebooks` to go through the data processing and analysis steps.

4.  Launch Streamlit:
```
streamlit run streamlit_app.py
```

### Run from a container:

1. Alternatively, build a Docker container:
```bash
docker build --pull --rm -f "Dockerfile" -t nlp-entity-linking-mt:latest .
```

2. Then run it:
```bash
docker run --rm -p 10000:10000 -p 8501:8501 -it nlp-entity-linking-mt
```

3. Open `http://localhost:8888` in your browser to launch JupyterLab. Run the Jupyter notebooks under `./notebooks` to go through the data processing and analysis steps.

4. Execute command in Docker container to run Streamlit:
```
docker exec -ti <container_name> streamlit run streamlit_app.py
```

5. Open `http://localhost:8501` in your browser to launch the Streamlit app.
