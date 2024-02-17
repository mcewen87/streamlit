import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import altair as alt
import numpy as np
# Data
songs = pd.read_csv('country_songs_by_decade.csv', index_col=[0])

# SELECT MEANINGFUL FEATURES
# - these features are auidble

features = songs[['artist', 'name', 'decade', 'danceability',
                  'loudness', 'valence', 'energy', 'tempo']].copy()
feature_values = songs[['danceability', 'loudness',
                        'valence', 'energy', 'tempo']].copy()

# NORMALIZE VALUES
# - loudness and tempo are not on a [0,1] scale
min_max_scaler = MinMaxScaler()
x = features[['loudness', 'tempo']]
x_scaled = min_max_scaler.fit_transform(x)
features[['loudness', 'tempo']] = x_scaled

# ADD DECADE FIELD
# - use .map to map decade values
decade_mapping = {
    '70s': 1970,
    '80s': 1980,
    '90s': 1990,
    '2000s': 2000,
    '2010s': 2010,
}

features['decade_year'] = songs['decade'].map(decade_mapping)

correlation = feature_values.corr()


def summary_logic(df):
    data = {}
    # danceability
    data['avg_danceability'] = df['danceability'].mean()
    data['std_danceability'] = df['danceability'].std()
    # loudness
    data['avg_loudness'] = df['loudness'].mean()
    data['std_loudness'] = df['loudness'].std()
    # valence
    data['avg_valence'] = df['valence'].mean()
    data['std_valence'] = df['valence'].std()
    # energy
    data['avg_energy'] = df['energy'].mean()
    data['std_energy'] = df['energy'].std()
    # tempo
    data['avg_tempo'] = df['tempo'].mean()
    data['std_tempo'] = df['tempo'].std()

    return pd.Series(data)


summary = features.groupby(by=['decade_year']).apply(
    summary_logic).reset_index()


# Custom CSS
custom_css = """
<style>
.big-font {
    font-size:20px !important;
}

.green-font {
    font-size:20px !important;
    color: #228B22;
}

.green {
    color: #228B22;
    font-size:20px !important;
}
</style>
"""

# Inject custom CSS with st.markdown
st.markdown(custom_css, unsafe_allow_html=True)


intro = """Has country music gotten more cheerful and optimistic over the decades?

This question randomly occurred to me, and I quickly thought of using Spotify's audio elements to find the answer!

I used the Spotify API to pull down the audio elements of country songs from five decades:  the 70s, 80s, 90s, 2000s, and 2010s. 

One of the audio elements Spotify's API provides is valence, which measures how uplifting or happy a song sounds. Classic country has a reputation for being somber. Let's see if things became more uplifting over the years."""

st.title("Spotify API: Country Music By The Decade")
st.divider()
st.header("Intro")


st.markdown(f'<p class="big-font">{intro}</p>', unsafe_allow_html=True)

st.header("Audio elements:")
st.markdown(f'<p class="green-font"> Descriptions from Spotify website. </p>',
            unsafe_allow_html=True)
st.markdown("""
            Loudness - The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typically range between -60 and 0 db.
         
            Valence - A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).
         
            Energy - Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.
         
            Tempo - The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.

            Danceability - Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.
            """)
st.header("Summary table:")
st.markdown(f'<p class="big-font"> Before looking at visuals, here’s a summary table for our analysis. </p>',
            unsafe_allow_html=True)
st.dataframe(summary)
st.markdown(f'<p class="big-font">We can already see some easy takeaways. The 90s were the happiest decade, and country music has gotten progressively louder. </p>', unsafe_allow_html=True)
st.header("Line Graphs:")
st.divider()
st.markdown(
    f'<p class="big-font">These line charts track the average value of each feature across the decades. Averages are on a [0,1] scale.</p>', unsafe_allow_html=True)

# Dance
st.subheader("Danceability")


c_danceability = (
    alt.Chart(summary)
    .mark_line()
    .encode(x=alt.X("decade_year:O", axis=alt.Axis(labelAngle=-45), title="Decade"), y=alt.Y('avg_danceability:Q', scale=alt.Scale(domain=[.3, 1]), title="Average Danceability"))
    .configure_axis(
        labelFontSize=16,  # Adjust font size for axis labels
        titleFontSize=18   # Adjust font size for axis titles
    )
    .configure_legend(
        labelFontSize=16,  # Adjust font size for legend labels
        titleFontSize=18   # Adjust font size for legend title
    )
    .configure_view(
        strokeWidth=0  # Removes the border around the chart
    )
)

st.altair_chart(c_danceability, use_container_width=True)
st.markdown(f'<p class="green"> Not much change here! </p>',
            unsafe_allow_html=True)

# Loudness
st.subheader("Loudness")
c_loudness = (
    alt.Chart(summary)
    .mark_line()
    .encode(x=alt.X("decade_year:O", axis=alt.Axis(labelAngle=-45), title="Decade"), y=alt.Y('avg_loudness:Q', scale=alt.Scale(domain=[.3, 1]), title="Average Loudness"))
    .configure_axis(
        labelFontSize=16,  # Adjust font size for axis labels
        titleFontSize=18   # Adjust font size for axis titles
    )
    .configure_legend(
        labelFontSize=16,  # Adjust font size for legend labels
        titleFontSize=18   # Adjust font size for legend title
    )
    .configure_view(
        strokeWidth=0  # Removes the border around the chart
    )
)

st.altair_chart(c_loudness, use_container_width=True)


st.markdown(f'<p class="green"> Things got louder, as you can see. </p>',
            unsafe_allow_html=True)

# Valence
st.subheader("Valence")

c_valence = (
    alt.Chart(summary)
    .mark_line()
    .encode(x=alt.X("decade_year:O", axis=alt.Axis(labelAngle=-45), title="Decade"), y=alt.Y('avg_valence:Q', scale=alt.Scale(domain=[.3, 1]), title="Average Valence"))
    .configure_axis(
        labelFontSize=16,  # Adjust font size for axis labels
        titleFontSize=18   # Adjust font size for axis titles
    )
    .configure_legend(
        labelFontSize=16,  # Adjust font size for legend labels
        titleFontSize=18   # Adjust font size for legend title
    )
    .configure_view(
        strokeWidth=0  # Removes the border around the chart
    )
)

st.altair_chart(c_valence, use_container_width=True)


st.markdown(f'<p class="green"> Contemporary country is slightly sadder than classic country.</p>',
            unsafe_allow_html=True)
# Tempo
st.subheader("Tempo")

c_tempo = (
    alt.Chart(summary)
    .mark_line()
    .encode(x=alt.X("decade_year:O", axis=alt.Axis(labelAngle=-45), title="Decade"), y=alt.Y('avg_tempo:Q', scale=alt.Scale(domain=[.3, 1]), title="Average Tempo"))
    .configure_axis(
        labelFontSize=16,  # Adjust font size for axis labels
        titleFontSize=18   # Adjust font size for axis titles
    )
    .configure_legend(
        labelFontSize=16,  # Adjust font size for legend labels
        titleFontSize=18   # Adjust font size for legend title
    )
    .configure_view(
        strokeWidth=0  # Removes the border around the chart
    )
)

st.altair_chart(c_tempo, use_container_width=True)


st.markdown(f'<p class="green"> A slight uptick in tempo over the years...</p>',
            unsafe_allow_html=True)
# Energy
st.subheader("Energy")

c_energy = (
    alt.Chart(summary)
    .mark_line()
    .encode(x=alt.X("decade_year:O", axis=alt.Axis(labelAngle=-45), title="Decade"), y=alt.Y('avg_energy:Q', scale=alt.Scale(domain=[.3, 1]), title="Average Energy"))
    .configure_axis(
        labelFontSize=16,  # Adjust font size for axis labels
        titleFontSize=18   # Adjust font size for axis titles
    )
    .configure_legend(
        labelFontSize=16,  # Adjust font size for legend labels
        titleFontSize=18   # Adjust font size for legend title
    )
    .configure_view(
        strokeWidth=0  # Removes the border around the chart
    )
)

st.altair_chart(c_energy, use_container_width=True)


st.markdown(f'<p class="green"> More energy over the years! </p>',
            unsafe_allow_html=True)

st.header("Correlation:")
st.markdown(f'<p class="big-font">From the line charts, what features do you think are the most tightly correlated? Here is what the data shows. </p>', unsafe_allow_html=True)
st.dataframe(correlation)
st.markdown(f'<p class="green"> No surprises here, loudness and energy have a strong correlation. Here is a scatter plot of the two features. </p>', unsafe_allow_html=True)
st.write("--")
c_cor = (
    alt.Chart(features)
    .mark_point()
    .encode(x=alt.X('loudness', title='Loudness'),  y=alt.Y('energy', title="Energy"))
    .configure_axis(
        labelFontSize=16,  # Adjust font size for axis labels
        titleFontSize=18   # Adjust font size for axis titles
    )
    .configure_legend(
        labelFontSize=16,  # Adjust font size for legend labels
        titleFontSize=18   # Adjust font size for legend title
    )
    .configure_view(
        strokeWidth=0  # Removes the border around the chart
    )
)

st.altair_chart(c_cor, use_container_width=True)

st.header("Classification with Support Vector Machines:")
st.markdown(f'<p class="big-font">Is it possible to train an algorithm to predict a song’s decade? Probably not. There is too much overlap in feature values between each decade. But I will try an SVM grid search just for fun. </p>', unsafe_allow_html=True)
code = '''feature_values = df[['danceability', 'loudness', 'valence', 'energy',  'tempo', 'decade']].copy()

X = feature_values.iloc[0:, 0:5]
y = feature_values['decade']

# Define parameter grid
param_grid = {
    'svc__kernel': ['linear', 'poly', 'rbf'],
    'svc__C': [0.1, 1, 10],
    'svc__degree': [2, 3],  # Only used for 'poly' kernel
    'svc__gamma': ['scale', 'auto']  # Only used for 'rbf' and 'poly' kernels
}


# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC())
])

# Initialize GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')

# Fit the model
grid_search.fit(X, y)

# Print results
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)'''

st.code(code, language='python')
st.markdown(f"<p>Best parameters: 'svc__C': 1, 'svc__degree': 2, 'svc__gamma': 'scale', 'svc__kernel': 'linear'</p>", unsafe_allow_html=True)

st.markdown(f'<p>Best cross-validation score: 0.4819418676561534</p>',
            unsafe_allow_html=True)


st.markdown(f'<p class="big-font">As you might expect, the SVM had trouble predicting the decade. It was only accurate about 50% of the time.</p>', unsafe_allow_html=True)
