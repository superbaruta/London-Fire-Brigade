import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load
import sklearn.metrics
import geopandas as gp
from shapely.wkt import loads
from shapely.ops import unary_union
from shapely.geometry import Point, LineString
from sklearn.neighbors import KNeighborsRegressor
import os

custom_css = """
    <style>
    body {
        background-color: #FFFFFF; /* Blanco */
        font-family: Arial, sans-serif; /* Tipo de fuente */
        color: #333333; /* Color del texto */
    }
    .sidebar .sidebar-content {
        background-color: #EF3E42; /* Rojo del Departamento de Bomberos de Londres */
        color: white; /* Texto en la barra lateral */
    }
    .css-1aumxhk {
        color: #EF3E42 !important; /* Color de acento */
    }
    </style>
"""


st.markdown(custom_css, unsafe_allow_html=True)

st.set_option('deprecation.showPyplotGlobalUse', False)


def parseLineArray(s):
    s = s.replace('[[', '')
    s = s.replace(']]', '')
    s = s.split('], [')
    s1 = np.fromstring(s[0],sep=',')
    s2 = np.fromstring(s[1],sep=',')
    return [s1, s2]

@st.cache_resource
def get_dataset(filename):
        return pd.read_csv(filename)

chart_data = pd.read_csv('kde_plots.csv', converters={'line': parseLineArray})
areas = get_dataset('area_shapes.csv')
london = unary_union(areas['geometry'].apply(loads).values)
london_df = gp.GeoDataFrame.from_dict({'geometry': [london]})
time_res_analysis = pd.read_csv('time_prediction_results_analysis.csv')
distance_res_analysis = pd.read_csv('distance_prediction_results_analysis.csv')
hour_res_analysis = pd.read_csv('hour_prediction_results_analysis.csv')
prediction_res = pd.read_csv('prediction_results.csv')
simp_time_res_analysis = pd.read_csv('simp_time_prediction_results_analysis.csv')
simp_distance_res_analysis = pd.read_csv('simp_distance_prediction_results_analysis.csv')
simp_hour_res_analysis = pd.read_csv('simp_hour_prediction_results_analysis.csv')
simp_prediction_res = pd.read_csv('simp_prediction_results.csv')
delays = pd.read_csv('delay_barplot.csv')
times = get_dataset('time_barplots.csv')
incident_sample = pd.read_csv('incidents_sample.csv')
mobilisation_sample = pd.read_csv('mobilisation_sample.csv')
stations = pd.read_csv('Londons_fire_stations.csv')





st.sidebar.image('lfb.png')
st.sidebar.title("London FB: A ML Project")

pages = ["The London FD", 'Datasets', 'Data Exploration',  "Data Visualisation", 'Methodology', "Modelling", 'Results', 'Conclusion']
page = st.sidebar.radio("Go to", pages)

if page == pages[0]:
    st.image('website_tile-grenfell_tower_fire_scene_002.jpg')
    st.title("London Fire Brigade: Predicting Travel Times")
    'Public services have been struggling to stay afloat under the pressure of increasing budget cuts. As any firefighter will tell you, they feel the squeeze; having to do maintenance and fundraising themselves.'
    'Our aim is to try accurately predict the travel times of a fire engine. This can be used for the simple purpose of knowing how long the journey will take when a caller calls in, or something more complex.'
    'We want to try find the optimal place to build a new firestation and predict how much time can be saved through this new fire station. The financial savings this would provide are hard to predict as there is a cost to fire damage and human injury.'
    'Improving the response times to an incident would objectively reduce these, but by how much?'

if page == pages[1]:
    st.image('1343_case_study_webpage_headers_desktop_1920x850_batch_3_london_fire.jpg')
    st.title("Presentation of Our Data")
    st.header('Datasets Used')
    # Preview of incidents, mobilisation, stations df
    st.subheader('Incident dataset')
    st.dataframe(incident_sample)
    st.subheader('Mobilisation dataset')
    st.dataframe(mobilisation_sample)
    st.subheader('Stations dataset')
    st.dataframe(stations.head())
    st.divider()

    # Explanation of key variables and how they were modified
    'The key variables we used were:'
    '- Incident coordinates'
    '- Local neighborhood name'
    '- Station coordinates'
    '- Station name'
    '- Travel time in seconds'
    '- Hour of call'
    'The coordinates are used to calculate the euclidean distance between the two points.'
    'The time and newly calculated distance are turned into an average speed over that distance.'
    'This average speed will be the target variable for our model.'
    'We also use the station name and local neighborhood as categorical variables in our standard approach'
    'In the methodology section we will explain what we use as our predictor variable more in depth.'
    st.divider()

    st.header('Data cleaning')
    'Some incidents did not have coordinates, these were removed.'
    'Some travel distances/times were impossibly low and were also removed'
    'Outliers outside of 1.5 x IQR were removed, we used the average speed calculated for this.'
    'We used speed as this considers both the time taken and over what distance.'
    'Obviously some imformation is lost at this step as not all "outliers" are true outliers due to the simplifactions we made.'

if page == pages[2]:
    st.image('fire-fighter-header.jpg')
    st.title("Data Exploration")

    # Plot 1
    f, ax = plt.subplots(figsize=(10,10))
    df = times[['IncGeo_BoroughName', 'TravelTimeSeconds']]
    mean = df['TravelTimeSeconds'].mean()
    sns.boxplot(ax=ax, data=df,x='IncGeo_BoroughName', y='TravelTimeSeconds')
    ax.hlines(mean, -0.5, len(df['IncGeo_BoroughName'].unique()) - 0.5, 'red', '--', label='Travel time mean')
    ax.set_xticklabels(df['IncGeo_BoroughName'].unique(), rotation=90)
    ax.set_title('Mean travel time per london borough', pad=10)
    ax.set_xlabel('London Boroughs')
    ax.set_ylabel('Travel Time')
    plt.legend()
    st.pyplot(f)
    st.divider()
    
    # Plot 2
    f, ax = plt.subplots(figsize=(10,10))
    df = times[['IncGeo_BoroughName', 'AttendanceTimeSeconds']]
    sns.boxplot(ax=ax, data=df,x='IncGeo_BoroughName', y='AttendanceTimeSeconds')
    ax.hlines(mean, -0.5, len(df['IncGeo_BoroughName'].unique()) - 0.5, 'red', '--', label='Travel time mean')
    ax.set_xticklabels(df['IncGeo_BoroughName'].unique(), rotation=90)
    ax.set_title('Mean attendance time per london borough', pad=10)
    ax.set_xlabel('London Boroughs')
    ax.set_ylabel('Attendance Time')
    plt.legend()
    st.pyplot(f)
    st.divider()

    # Plot 3
    # f, ax = plt.subplots(figsize=(10,10))
    # df = times[['TimeOfCall', 'AttendanceTimeSeconds']]
    # sns.boxplot(ax=ax, data=df,x='TimeOfCall', y='AttendanceTimeSeconds')
    # # ax.set_xticklabels(df['TimeOfCall'].unique(), rotation=90)
    # ax.set_title('Mean attendance time per hour of call', pad=10)
    # ax.set_xlabel('Hour of Call')
    # ax.set_ylabel('Attendance Time')
    # plt.legend()
    # st.pyplot(f)
    # st.divider()

    # Plot 4
    f, ax = plt.subplots(figsize=(10,10))
    df = times[['TimeOfCall', 'IncidentGroup']]
    df['TimeOfCall'] = [int(s[:2]) for s in df['TimeOfCall']]

    dic = {}

    for i in df['IncidentGroup'].unique():
        dic[i] = df[df['IncidentGroup'] == i].value_counts('TimeOfCall').sort_index().values
    pd.DataFrame(dic).plot(ax=ax, kind='bar', stacked=True)
    ax.set_title('Incident distribution throughout the day', pad=10)
    ax.set_ylabel('Incidents (over 3yr span)')
    ax.set_xlabel('Hour of Call')
    st.pyplot(f)
    st.divider()

    # Plot 5
    f, ax = plt.subplots(figsize=(10,10))
    df = times[['DateOfCall', 'IncidentGroup']]
    df['DateOfCall'] = [date[3:-3] for date in times['DateOfCall']]

    dic = {}
    for i in df['IncidentGroup'].unique():
        dic[i] = df[df['IncidentGroup'] == i].value_counts('DateOfCall').sort_index().reindex(df['DateOfCall'].unique()).values
    pd.DataFrame(dic).plot(ax=ax, kind='bar', stacked=True)
    ax.set_xticks(range(12), df['DateOfCall'].unique())
    ax.set_title('Incident distribution throughout the year', pad=10)
    ax.set_ylabel('Incidents (over 3yr span)')
    ax.set_xlabel('Month')
    st.pyplot(f)
    st.divider()

    # Plot 6
    f, ax = plt.subplots(figsize=(10,10))
    months = pd.Series([date[:-3] for date in pd.Series([date[3:] for date in times['DateOfCall']]).unique()]).value_counts()
    months.plot(ax=ax, kind='bar')
    ax.set_title('Distribution of months throughout the dataset', pad=10)
    ax.set_ylabel('Number of times in the dataset')
    ax.set_xlabel('Month')
    ax.set_yticks(range(5))
    st.pyplot(f)
    st.divider()

    # Plot 7
    f, ax = plt.subplots(figsize=(10,10))
    df = times[['DateOfCall', 'IncidentGroup']]
    df['DateOfCall'] = [date[3:-3] for date in times['DateOfCall']]

    dic = {}
    for i in df['IncidentGroup'].unique():
        dic[i] = (df[df['IncidentGroup'] == i].value_counts('DateOfCall').sort_index().reindex(df['DateOfCall'].unique()) / months).values
    pd.DataFrame(dic).plot(ax=ax, kind='bar', stacked=True)
    ax.set_xticks(range(12), df['DateOfCall'].unique())
    ax.set_title('Incident distribution throughout the year (adjusted)', pad=10)
    ax.set_ylabel('Incidents (over 3yr span)')
    ax.set_xlabel('Month')
    st.pyplot(f)
    st.divider()

    # Plot 8
    f, ax = plt.subplots(figsize=(10,10))
    if st.checkbox('Show all non delayed incidents for context'):
        delay_barplot = delays.copy()
        ax.text(-1.5, 0, 'Not delayed')
        ax.text(0.75, -0.75, 'Delayed')
        ax.text(-0.5, 0.05, '91%')
        ax.pie(x=delay_barplot['count'], explode=(delay_barplot['count'] / (delay_barplot['count'].max() * 6)))
    else:
        delay_barplot = delays.drop(0)
        ax.pie(x=delay_barplot['count'], autopct='%1i%%', labels=list(delay_barplot['DelayCode_Description'].values), explode=(delay_barplot['count'] / (delay_barplot['count'].max() * 6)))
    ax.set_title('Pie Chart of proportion of delay codes', pad=20)
    st.pyplot(f)
    st.divider()

    
    # Plot 9
    mean_line = chart_data['line'][len(chart_data) - 1]
    chart_data.drop(len(chart_data) - 1)
    fire_stations = st.select_slider('Fire Stations to include', options=range(1, len(chart_data) + 1))
    'Using {} fire stations with the most different distributions'.format(fire_stations)
    df = chart_data.head(fire_stations)
    f, ax = plt.subplots(figsize=(10,10))
    ax.plot(mean_line[0], mean_line[1], label='Overall KDE', lw=5)
    for i in range(len(df)):
        ax.plot(df['line'][i][0], df['line'][i][1], alpha=df['alphas'][i])
    ax.set_title('Travel time KDEs of each fire station as compared to the KDE for all incidents', pad=10)
    ax.set_xlabel('Travel Time')
    ax.set_yticks([])
    ax.legend()
    st.pyplot(f)
    if st.button('Show Chart Data'):
        st.dataframe(chart_data)

if page == pages[3]:
    st.image('27-November-2019-credit-Ian-Marlow.jpg')
    st.title("Data Visualisation")
    'Here we can visualize the comprehensiveness of the model improve as the number of incidents grows.'
    incidents = st.select_slider('Incidents to include', options=[10, 100, 1000, 10000, 100000, 584529])
    'Now using {} incidents at random'.format(incidents)
    # Getting the specific df that matches the chosen number of incidents
    df = pd.read_csv('vector_feild_viz_{}.csv'.format(incidents))

    df['geometry'] = df['geometry'].apply(loads)
    df = gp.GeoDataFrame.from_dict({'geometry': df['geometry'], 'Speed': df['Speed'], 'Color': df['Color']})

    ax = london_df.boundary.plot(edgecolor='black', linewidth=0.3)
    df.plot(ax=ax, column='Color', legend=True)
    plt.axis('off')
    st.pyplot()

if page == pages[4]:
    @st.cache_resource
    def show_method(number_of_incidents):
        # Creating fake test incident
        inc_x = np.random.randint(-20,-10,1)
        inc_y = np.random.randint(-20,20,1)
        station_x = np.random.randint(10,20,1)
        station_y = np.random.randint(-20,20,1)

        # Generating fake incidents
        X = np.random.randint(-20, -10, int(number_of_incidents / 3))
        Y = np.random.randint(-20, 20, int(number_of_incidents / 3))
        X = np.append(X, np.random.randint(10, 20, int(number_of_incidents / 3)))
        Y = np.append(Y, np.random.randint(-20, 20, int(number_of_incidents / 3)))
        X = np.append(X, np.random.randint(-10, 10, int(number_of_incidents / 6)))
        Y = np.append(Y, np.random.randint(10, 20, int(number_of_incidents / 6)))
        X = np.append(X, np.random.randint(-10, 10, int(number_of_incidents / 6)))
        Y = np.append(Y, np.random.randint(-20, -10, int(number_of_incidents / 6)))
        stat_x = np.zeros(number_of_incidents)
        stat_y = np.zeros(number_of_incidents)
        speed = np.append(np.array([5, 30]), np.random.randint(5,30,number_of_incidents - 2))
        fake_df = pd.DataFrame({'x_inc': X,
                                'y_inc': Y,
                                'x_station': stat_x, 
                                'y_station': stat_y,
                                'Speed': speed})
        fake_gdf = gp.GeoDataFrame.from_dict({'geometry': [LineString([[x1, y1], [x2, y2]]) for x1, y1, x2, y2, in zip(X, Y, stat_x, stat_y) ], 'Speed': speed})

        # Plotting the fake incidents
        f1, ax1 = plt.subplots(figsize=(10,10))
        fake_gdf.plot('Speed', lw=3, ax=ax1, legend=True)
        ax1.scatter([0], [0], c='red', s=100, zorder=5)
        ax1.axis('off')
        ax1.set_title('Journeys to incidents (speed in km/h)')

        X = []
        Y = []
        point_speed = []
        size = []
        alphas = []

        # Segmenting each journey every 1m, creating a new point at those coordinates with the Speed of journey as attribute
        for index in range(len(fake_df)):
            row = fake_df.iloc[index, :]
            line = LineString([Point(row['x_station'], row['y_station']), Point(row['x_inc'], row['y_inc'])])
            new_p = line.segmentize(max_segment_length=1)
            for point in new_p.coords:
                X.append(point[0])
                Y.append(point[1])
                point_speed.append(row['Speed'])
                size.append(50)
                alphas.append(0.3)
        
        # Plotting newly created points
        f2, ax2 = plt.subplots(figsize=(10,10))
        gp.GeoDataFrame({'geometry': [Point(x, y) for x, y in zip(X, Y)], 'Speed': point_speed, 'Size': size}).plot('Speed', ax=ax2, markersize='Size', legend=True)
        ax2.scatter([0], [0], c='red', s=100)
        ax2.axis('off')
        ax2.set_title('Journey points (speed in km/h)')

        # Training KNN to predict new points
        knn = KNeighborsRegressor()
        knn.fit(pd.DataFrame({'x': X, 'y': Y}), point_speed)

        # Split into line and predict
        line = LineString([[inc_x, inc_y], [station_x, station_y]])
        new_points = line.segmentize(max_segment_length=1)
        pred_speeds = knn.predict(np.array(new_points.coords).round())
        pred_speed = pred_speeds.mean()
        points = list(zip(*list(new_points.coords)))
        X1 = X + list(points[0])
        Y1 = Y + list(points[1])
        point_speed1 = point_speed + list(pred_speeds)
        size1 = size + [100 for i in range(len(pred_speeds))]
        alphas1 = alphas + [1 for i in range(len(pred_speeds))]

        f3, ax3 = plt.subplots(figsize=(10,10))
        df = gp.GeoDataFrame({'geometry': [Point(x, y) for x, y in zip(X1, Y1)], 'Speed': point_speed1, 'Size': size1})
        df.plot('Speed', ax=ax3, markersize='Size', alpha=alphas1, legend=True)
        ax3.scatter([0], [0], c='red', s=100)
        ax3.axis('off')
        ax3.set_title('New journey points prediction (speed in km/h)')

        # Plotting full predicted line with average speed
        points = [Point(x, y) for x, y in zip(X, Y)]
        new_points = LineString([[station_x, station_y], [inc_x, inc_y]]).segmentize(max_segment_length=0.1).coords
        points += [Point(x, y) for x, y in new_points]
        # points.append(Point([station_x, station_y]))
        point_speed += [pred_speed for i in range(len(new_points))]
        size += [100 for i in range(len(new_points))]
        alpha = [0.3 for i in range(len(alphas))]
        alpha += [1 for i in range(len(new_points))]

        f4, ax4 = plt.subplots(figsize=(10,10))
        df = gp.GeoDataFrame({'geometry': points, 'Speed': point_speed, 'Size': size, 'Alphas': alpha})
        df.plot('Speed', ax=ax4, markersize='Size', alpha=df['Alphas'], legend=True)
        ax4.scatter([0], [0], c='red', s=100)
        ax4.axis('off')
        ax4.set_title('New journey prediction (speed in km/h)')

        return [f1, f2, f3, f4, pred_speed]

    st.image('fire-service-1800.jpg')
    st.title("Methodology")
    st.header('Feature Engineering')
    'Due to the type of data we have access to, we needed to enhance it for it to be usable by our KNN regressor.'
    'We tackled the problem from two seperate directions:'
    st.divider()
    st.subheader('1. Standard Approach')
    '\n'
    'We Converted the station name and local neighborhood categorical variables into numerical codes.'
    'We used the station name, local neigborhood area, time of call, and incident coordinates to predict the travel time.'
    st.divider()
    st.subheader('2. Complex Approach')
    'We aimed to create a density feild that models how fast a fire engine would travel through London.'
    'We made a few simplifications for the sake of expediancy:'
    '1 - We plot the journey is in a straight line from station to incident'
    '2 - We assume constant speed throughout the Journey'
    incidents = st.selectbox('Incidents to simulate', (30, 300, 3000), key='incidents')
    plots = show_method(incidents)
    plot_state = st.select_slider(label='Follow along', options=['Step 1', 'Step 2', 'Step 3', 'Step 4'])
    
    if plot_state == 'Step 1':
        st.pyplot(plots[0])
        st.write('We create line objects from the station (in :red[RED]), to each incident.')
        'Each journey has the average speed of the fire engine as attribute.'

    if plot_state == 'Step 2':
        st.pyplot(plots[1])
        'We break each line into points for every 10m.'
        'Each point has the same speed as its parent line as attribute.'
        'We train a model using the (x, y) coordinates of each point to predict its speed attribute.'

    if plot_state == 'Step 3':
        st.pyplot(plots[2])
        'We plot a new line representing a new journey to estimate from point A (new station) to point B (actual incident).'
        'Following the same process we split the line into points.'
        'We then predict the speed at each of these points using our model.'


    if plot_state == 'Step 4':
        st.pyplot(plots[3])
        'Averaging the speed at each point we get the predicted speed for the journey.'
        st.write('The predicted speed for the journey is {} km/h.'.format(round(plots[4], 1)))
        'We can use this and the distance of the journey to calculate the predicted travel time.'
        'And this is how the model predicts travel time!'

if page == pages[5]:
    st.image('london-fire-brigade-hompage-firefighter-ladder-new-ppe.jpg')
    st.title("Modelling")
    'We decided to use the travel time as our target variable as this is the only part of total attendance time that can be explained.'
    'Our final prediction of total attendance time is the predicted travel time and the mean time to mobilise (73.3s with MAE of 25.7s).'

    st.header('1. Standard Approach')

    st.subheader('Training')
    'The unedited variables were used to train these models: incident coordinates, station coordinates, and hour of call were used to predict the travel time.'
    'We seprated incidents into training and testing sets'
    'Various models were trained on this data.'

    st.subheader('Testing')
    'Can you find the best result?'
    options = []
    for name in os.listdir('Model_res'):
        if 'simp_' in name:
            options.append(name.replace('simp_', '').replace('_results.csv', ''))
    option = st.selectbox('Choose model:', options=options)
    model = pd.read_csv('Model_res/simp_{}_results.csv'.format(option))
    params = model.drop('mean_test_score', axis=1).columns
    columns = st.columns(len(params))
    search_params = {}
    for col, param in zip(columns, params):
        with col:
            search_params[param] = st.selectbox(param.replace('param_', ''), options=model[param].unique())

    st.divider()
    model_search = model.copy()
    for param in params:
        model_search = model_search[model_search[param] == search_params[param]]
    score = model_search.iloc[0,0]
    st.write('The MAE for these parameters is {}s'.format(round(score, 1)))
    st.divider()

    if st.checkbox('Reveal the best parameters and score'):
        # reveal best params and score
        'The bagging model has the best score with an MAE of 64.6s with n_estimators=200.'
    'As was expected the models perform quite poorly as they cannot effectively interpret the data as is.'
    'For reference, if we just guessed the mean travel time every time, the MAE would be 106.4s, this is a benchmark MAE to determine if the model is out performing pure chance.'
    'These results demonstrate the necessity for some clever feature engineering to drastically improve model performance'
    st.divider()
    st.header('2. Complex Approach')

    st.subheader('Training')
    'Using the same training and testing sets as the 1st approach, the sets were split into points as per the methodology section.'
    'We trained several models on the data. With a total of over 11 million new data points, it was import to reduce the dimension of the dataset.'
    'Hence why we trained 1 model for each hour of the day, these 24 seperate models help account for the local variation in traffic throughout the day.'
    # 'We then trained a KNN regression model for each hour of the day to account for local variance throughout the day.'

    st.subheader('Testing')
    'As before, can you find the best result?'
    options = []
    for name in os.listdir('Model_res'):
        if 'feat_eng_' in name:
            options.append(name.replace('feat_eng_', '').replace('_results.csv', ''))
    option = st.selectbox('Choose model:', options=options)
    model = pd.read_csv('Model_res/feat_eng_{}_results.csv'.format(option))
    st.divider()
    st.write('The average MAE for predicted speed across hours is {} km/h'.format(round(model['MAE'].mean(), 1)))
    st.dataframe(model, height=200)
    'For reference, if we just guessed the mean travel time every time, the MAE would be 7.6 km/h, this is a benchmark MAE to determine if the model is out performing pure chance.'
    st.divider()
    # st.selectbox()
    if st.checkbox('Reveal best model results'):
        model_rank = prediction_res.drop(['IncidentNumber', 'Station', 'Distance'], axis=1).groupby('Time').mean()
        'The KNN model performs the best:'
        st.write('The models\' average MAE is: {}s'.format(round(prediction_res['MAE'].mean(), 1)))
        st.write('The model with the best MAE is: hour {} with MAE = {}s'.format(model_rank['MAE'].idxmin(), round(model_rank['MAE'].min(), 1)))
        st.write('The model with the worst MAE is: hour {} with MAE = {}s'.format(model_rank['MAE'].idxmax(), round(model_rank['MAE'].max(), 1)))
        st.write('If we recall back to earlier, the benchmark MAE score was 106.4s'.format())
        st.write('And for context the real mean travel time is {}s'.format(round(prediction_res['Real'].mean(), )))
        if st.checkbox('Show individual model results'):
            st.dataframe(model_rank)

if page == pages[6]:
    st.image('DNZDgVXWAAA1USS.jpg')
    st.title('Modelling Results')

    option = st.selectbox('Assess models along what metric:', options=['Travel Time', 'Distance', 'Time of Day'])
    if option == 'Travel Time':
        
        distance = st.select_slider(label='Choose journey time (s)', value=time_res_analysis['Distance'].min(), options=time_res_analysis['Distance'])
        st.write('The expected error for a {}s journey is {}s for the 1st approach'.format(distance, round(simp_time_res_analysis[simp_time_res_analysis['Distance'] == distance].iloc[0, 1], 1)))
        st.write('The expected error for a {}s journey is {}s for the 2nd approach'.format(distance, round(time_res_analysis[time_res_analysis['Distance'] == distance].iloc[0, 1], 1)))

        f, ax1 = plt.subplots()
        
        ax1.hist(prediction_res['Real'].astype(int), bins=120, color='blue', alpha=0.3, label='Amount of incidents')
        ax1.set_title('Model accuracy based on journey time')
        ax1.set_xlabel('Journey time')
        ax1.set_ylabel('Number of incidents', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        ax2 = ax1.twinx()
        ax2.plot(simp_time_res_analysis['Distance'], simp_time_res_analysis['DiffPercent'], color='red', ls=':', label='Non-linear regression line of prediction error (1st Approach)')
        ax2.plot(time_res_analysis['Distance'], time_res_analysis['DiffPercent'], color='red', label='Non-linear regression line of prediction error (2nd Approach)')
        ax2.hlines(0, time_res_analysis['Distance'].min(), time_res_analysis['Distance'].max(), 'black', '--', label='Mean Error = 0')
        ax2.set_ylabel('Prediction error (s)', color='red')
        
        ax2.legend()
        st.pyplot(f)

        'The 2nd approach is clearly much closer to 0 error around the highest volume of incidents, although it is noticeably less accurate on the quickest and longest journeys.'
        'This could either be because of some simplifications we made by estimating a lot of values, or a lack of data in the less common journey times.'

    if option == 'Distance':
        # 'The models performance starts to deteriorate under distances of 1000m'
        
        distance = st.select_slider(label='Choose journey distance (m)', value=distance_res_analysis['Distance'].min(), options=distance_res_analysis['Distance'])
        st.write('The expected error for a {}m journey is {}s for the 1st approach'.format(distance, round(simp_distance_res_analysis[simp_distance_res_analysis['Distance'] == distance].iloc[0, 1], 1)))
        st.write('The expected error for a {}m journey is {}s for the 2nd approach'.format(distance, round(distance_res_analysis[distance_res_analysis['Distance'] == distance].iloc[0, 1], 1)))

        f, ax1 = plt.subplots()
        
        ax1.hist(prediction_res['Distance'].astype(int), bins=120, color='blue', alpha=0.3, label='Amount of incidents')
        ax1.set_title('Model accuracy based on journey time')
        ax1.set_xlabel('Journey time')
        ax1.set_ylabel('Number of incidents', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        ax2 = ax1.twinx()
        ax2.plot(simp_distance_res_analysis['Distance'], simp_distance_res_analysis['DiffPercent'], color='red', ls=':', label='Non-linear regression line of prediction error (1st Approach)')
        ax2.plot(distance_res_analysis['Distance'], distance_res_analysis['DiffPercent'], color='red', label='Non-linear regression line of prediction error (2nd Approach)')
        ax2.hlines(0, distance_res_analysis['Distance'].min(), distance_res_analysis['Distance'].max(), 'black', '--', label='Mean Error = 0')
        ax2.set_ylabel('Prediction error (s)', color='red')
        
        ax2.legend()
        st.pyplot(f)

        'When looking at how the accuracy of the models is impacted by the travel distance, we notice that both models have similar weaknesses.'
        'It would appear that both models\' is not impacted so differently my the travel distance.'

    if option == 'Time of Day':
        
        distance = st.select_slider(label='Choose journey time (s)', value=hour_res_analysis['Distance'].min(), options=hour_res_analysis['Distance'])
        st.write('The expected error for a hour {} journey is {}s for the 1st approach'.format(distance, round(simp_hour_res_analysis[simp_hour_res_analysis['Distance'] == distance].iloc[0, 1], 1)))
        st.write('The expected error for a hour {} journey is {}s for the 2nd approach'.format(distance, round(hour_res_analysis[hour_res_analysis['Distance'] == distance].iloc[0, 1], 1)))

        f, ax1 = plt.subplots()
        
        ax1.hist(prediction_res['Time'].astype(int), bins=24, color='blue', alpha=0.3, label='Amount of incidents')
        ax1.set_title('Model accuracy based on journey time')
        ax1.set_xlabel('Journey time')
        ax1.set_ylabel('Number of incidents', color='blue')
        ax1.set_yticklabels(range(24))
        ax1.tick_params(axis='y', labelcolor='blue')

        ax2 = ax1.twinx()
        ax2.plot(simp_hour_res_analysis['Distance'], simp_hour_res_analysis['DiffPercent'], color='red', ls=':', label='Non-linear regression line of prediction error (1st Approach)')
        ax2.plot(hour_res_analysis['Distance'], hour_res_analysis['DiffPercent'], color='red', label='Non-linear regression line of prediction error (2nd Approach)')
        ax2.hlines(0, hour_res_analysis['Distance'].min(), hour_res_analysis['Distance'].max(), 'black', '--', label='Mean Error = 0')
        ax2.set_ylabel('Prediction error (s)', color='red')
        
        ax2.legend()
        st.pyplot(f)

        'Although the 2nd approach performs better at all times of day, both models seems to vary little based on the time of day.'
        'We can however note a slightly higher variance in the first model.'

if page == pages[7]:
    st.title('Conclusion')
    st.subheader('Outcomes')
    'To conclude, we are able to predict travel time for most journeys quite accurately.'
    'This could have multiple uses, for instance:'
    'If the LFB recieved budget to build a new fire station, predicting the optimal location to build this is vital.'
    'Using our model complex model we can better predict travel times from between an incident and a location that has never been seen before.'
    st.subheader('Further work')
    'To improve the weaknessess of the model (short journeys and more quantity to train from), we could better predict the length and journey of and the actual path the fire engine would follow.'
    'We could get more data from the LFB, however this data would be older and therefor less accurate. A possible option could be to use data from other emergency vehicles (ambulance or police), as these would be similarly affected by traffic.'
    'We could also try to develop a secondary model to adjust the prediction based on the weaknesses highlighted prior. Based on a very superficial trial, we improved our model by 0.65%.'
