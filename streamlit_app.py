import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pathlib
from collections import OrderedDict, Counter
from debater_python_api.api.debater_api import DebaterApi


def get_dataframe():
    df = pd.read_csv('./data/mtsamples_descriptions_clean.csv')
    df = df.dropna()
    df['id'] = df['id'].astype('str')
    df['id_description'] = df['id_description'].astype('str')
    return df


@st.cache
def get_arg_quality(client, topic, sentences, top_k):
    # Create list of dictionaries for Argument Quality API call
    sentences_topic = [{ "sentence": sentence["text"], "topic": topic} for sentence in sentences]
    # Store results in scores
    scores = client.run(sentences_topic)
    # Sort sentences in descending order based on score
    sentences_sorted = [s for s, _ in sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)]
    # Store top_k sentences in separate variable
    sentences_top_k = sentences_sorted[:top_k]
    return sentences_top_k


@st.cache
def get_keypoints(client, domain, sentences_top_k, mapping_threshold, n_top_kps):
    # Configure parameters
    run_params = {"mapping_threshold": mapping_threshold, "n_top_kps": n_top_kps}
    # Store all text in a list
    sentences_texts = [sentence["text"] for sentence in sentences_top_k]
    # Store all id in a list
    sentences_ids = [sentence["id"] for sentence in sentences_top_k]
    # Clear domain on KPA service
    try:
        client.delete_domain_cannot_be_undone(domain)
    except Exception:
        pass
    # Upload data to KPA service
    client.upload_comments(domain=domain, comments_ids=sentences_ids, comments_texts=sentences_texts, dont_split=True)
    client.wait_till_all_comments_are_processed(domain=domain)
    future = client.start_kp_analysis_job(domain=domain, comments_ids=sentences_ids, run_params=run_params)
    kpa_result = future.get_result(high_verbosity=False, polling_timout_secs=5)
    future.get_job_id()
    return kpa_result


@st.cache
def save_kp_as_df(kpa_result):
    # Initialize a list for storing the keypoints
    matchings_rows = []
    # Loop through all keypoint matchings and store in the list
    for keypoint_matching in kpa_result["keypoint_matchings"]:
        kp = keypoint_matching["keypoint"]
        for match in keypoint_matching["matching"]:
            match_row = [
                kp,
                match["sentence_text"],
                match["score"],
                match["comment_id"],
                match["sentence_id"],
                match["sents_in_comment"],
                match["span_start"],
                match["span_end"],
                match["num_tokens"],
                match["argument_quality"],
                match["kp_quality"]
            ]
            matchings_rows.append(match_row)
    # Define column headers for dataframe
    cols = [
        "kp",
        "sentence_text",
        "match_score",
        "comment_id",
        "sentence_id",
        "sents_in_comment",
        "span_start",
        "span_end",
        "num_tokens",
        "argument_quality",
        "keypoint_quality"
    ]
    # Store the list as dataframe
    df_match = pd.DataFrame(matchings_rows, columns=cols)
    return df_match


@st.cache
def get_sentence_to_mentions(client, sentences_texts):
    mentions_list = client.run(sentences_texts)
    sentence_to_mentions = {}

    for sentence_text, mentions in zip(sentences_texts, mentions_list):
        sentence_to_mentions[sentence_text] = set([mention["concept"]["title"] for mention in mentions])
    
    return sentence_to_mentions


@st.cache
def get_all_mentions(client, df):
    terms = {}

    for kp in set(df["kp"].values):
        sentence_to_mentions = get_sentence_to_mentions(client, df["sentence_text"][df["kp"] == kp].values)
        all_mentions = [mention for sentence in sentence_to_mentions for mention in sentence_to_mentions[sentence]]
        term_count = dict(Counter(all_mentions))
        # Remove mentions of "history"
        if "History" in term_count.keys():
            term_count.pop("History")
        terms[kp] = term_count
    
    return terms


def plot_kp_cluster(df):
    # Plot keypoint cluster size
    fig = plt.figure(figsize=(4, 3))
    sns.countplot(y="kp", data=df, palette="Blues_d")
    plt.xlabel('Keypoint')
    plt.ylabel('Size of cluster')
    st.text('Comparison of the size of each keypoint cluster')
    st.pyplot(fig)


def plot_freq_by_medspec(df):
    # Plot relative frequency grouped by medical specialty
    fig = plt.figure(figsize=(8, 6)) #(12, 8)
    sns.countplot(y="medical_specialty_new", data=df, order=df["medical_specialty_new"].value_counts().index)
    plt.xlabel('Medical Specialty')
    plt.ylabel('Count')
    st.text('Frequency of keypoints grouped by medical specialty')
    st.pyplot(fig)


def plot_medspec_for_kp(df, kp):
    st.text(f"Distribution of medical specialities for: \"{kp}\"")
    fig = plt.figure(figsize=(6, 4)) #(8, 6)
    data = df[df["kp"] == kp]["medical_specialty_new"].value_counts(normalize=True)
    data.plot(kind='barh')
    # plt.title(f"Keypoint: {kp}")
    plt.ylabel('Medical Specialty')
    plt.xlabel('Distribution')
    st.pyplot(fig)


def show_top_wikiterms(wikiterms, top_k, kp):
    st.text(f"Top-10 Related Wikipedia Terms for: \"{kp}\"")
    fig = plt.figure(figsize=(6, 4)) #(8, 6)
    data = pd.DataFrame(list(wikiterms[kp].items()), columns=["Term", "Count"]).sort_values(by="Count", ascending=False)#.iloc[:10, :]
    # data.plot(x='Term', kind="bar")
    ax = sns.barplot(x=data["Count"].head(top_k), y=data["Term"].head(top_k), palette = "Blues_d")
    plt.xlabel('Number of Mentions')
    plt.ylabel('Wikipedia Term')
    st.pyplot(fig)


if __name__ == '__main__':
    # Set page config of Streamlit
    st.set_page_config(page_title='NLP Entity Linking', layout='centered')

    # Load data
    dataframe = get_dataframe()
    sentences = dataframe.to_dict(orient="records", into=OrderedDict)

    # Debater API setup
    api_key = st.secrets["debater_api_key"]
    debater_api = DebaterApi(apikey=api_key)
    arg_quality_client = debater_api.get_argument_quality_client()
    keypoints_client = debater_api.get_keypoints_client()

    # Set topic and top k
    topic = " Left heart catheterization, left ventriculography, coronary angiography, and successful stenting of tight lesion in the distal circumflex and moderately tight lesion in the mid-right coronary artery."
    top_k = 500
    sentences_top_k = get_arg_quality(arg_quality_client, topic, sentences, top_k)

    # Set domain and KeyPoint analysis parameters
    domain = "medical_demo"
    mapping_threshold = 0.98
    n_top_kps = 5
    kpa_result = get_keypoints(keypoints_client, domain, sentences_top_k, mapping_threshold, n_top_kps)

    # Save results as dataframe
    df_match = save_kp_as_df(kpa_result)
    df_merge = pd.merge(left=df_match, right=dataframe[["id", "id_description", "medical_specialty_new"]], left_on = "comment_id", right_on = "id", validate = "one_to_one")

    # Remove "none" keypoint
    df_clean = df_merge[df_merge["kp"] != "none"]

    # Get related mentions on Wikipedia
    term_wikifier_client = debater_api.get_term_wikifier_client()
    terms = get_all_mentions(term_wikifier_client, df_clean)

    # Set title of Streamlit dashboard
    st.title('NLP Entity Linking for Medical Transcripts')
    st.markdown("""---""")
    
    # Show topic at top of page
    st.subheader(f"Topic: {topic}")
    st.markdown("""---""")

    # Plot size of each keypoint cluster
    plot_kp_cluster(df_clean)

    # Plot relative frequency grouped by medical specialty
    plot_freq_by_medspec(df_clean)

    # List containing all keypoints
    kp_list = df_clean["kp"].value_counts().index

    # Selectbox widget for selecting a keypoint
    selected_kp = st.selectbox('Pick one', kp_list)

    # Plot distribution of medical specialties for selected keypoint
    plot_medspec_for_kp(df_clean, selected_kp)

    # Show top-k Wikipedia terms for selected keypoint
    top_k_terms = 10
    show_top_wikiterms(terms, top_k_terms, selected_kp)
